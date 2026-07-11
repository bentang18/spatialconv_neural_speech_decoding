"""v14_converged_v3 — ragged (varlen) block-diagonal attention primitive (#24).

The L1 block attends block-diagonally: within a ``(clip, shaft)`` block the packed
tokens attend to each other, never across blocks. The padded path (``attention.py``
``L1Block``) realises this by gathering to ``(n_shafts, max_c)``, padding short
shafts, and masking the pad — paying the ``max_c`` zero-pad on every linear and
every attention score. This primitive realises the SAME block-diagonal attention
on the FLAT packed layout the pack plan produces (``packing.py``): the selected
tokens laid end to end, ``cu_seqlens`` marking each block, no padding.

Three backends, ONE numerical contract:

  * ``flash``     — ``flash_attn_varlen_func`` (GPU, FA2/FA3/FA4 stable API).
                    Physically ragged; the kernel never touches cross-block pairs.
                    Preferred when the wheel is present.
  * ``flex``      — ``torch.nn.attention.flex_attention`` with a block-diagonal
                    ``BlockMask`` built from ``cu_seqlens`` (GPU, no external wheel).
                    The compiled kernel is block-sparse: off-diagonal 128-tiles are
                    skipped, so memory/compute is ~O(Σ blockᵢ²), never O(total²).
                    The GPU path on a box without ``flash_attn`` (e.g. aarch64/GH200,
                    where no prebuilt flash wheel exists).
  * ``reference`` — a block-diagonal additive-mask SDPA (CPU/testing). It MATERIALISES
                    the dense ``(total, total)`` mask, so it is CPU-only (tiny N):
                    consumes the SAME ``cu_seqlens`` to test the exact flat layout +
                    RoPE-in-flat-layout the GPU kernels run; validated against the
                    padded ``L1Block`` (B1c) and against ``flex`` (F2).

``auto`` picks ``flash`` when q is on CUDA and ``flash_attn`` imports; else ``flex``
on CUDA; else ``reference`` on CPU. flash/flex are imported LAZILY so a CPU box never
needs either.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

NEG_INF_MASK = -1e4  # v14 convention: finite (not -inf) → all-masked rows go uniform


def _flash_available(q: Tensor) -> bool:
    if not q.is_cuda:
        return False
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


_FLEX_FN = None


def _flex_fn():
    """Lazily import + ``torch.compile`` ``flex_attention`` (once).

    The compiled lowering is what makes FlexAttention block-sparse: eager
    ``flex_attention`` materialises the full score grid (would OOM like
    ``reference``); the compiled kernel consults the ``BlockMask`` and skips
    fully-masked tiles. Compiled once, module-global (compile is expensive)."""
    global _FLEX_FN
    if _FLEX_FN is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX_FN = torch.compile(flex_attention)
    return _FLEX_FN


def _segment_ids(cu_seqlens: Tensor, total: int) -> Tensor:
    """(n_seg+1,) cumulative bounds → (total,) block id per token.

    ``repeat_interleave`` of the per-segment lengths. Zero-length segments (whole-
    masked shafts) contribute nothing, exactly as they occupy no packed tokens.
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)  # (n_seg,)
    seg = torch.arange(lengths.shape[0], device=cu_seqlens.device)
    return seg.repeat_interleave(lengths)  # (total,)


def _reference_block_diag(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor
) -> Tensor:
    # q,k,v: (total, H, hd). Block-diagonal additive-mask SDPA (test/CPU backend).
    total = q.shape[0]
    seg = _segment_ids(cu_seqlens, total)  # (total,)
    same = seg[:, None] == seg[None, :]  # (total, total) — same block ⇒ attend
    bias = torch.where(same, 0.0, NEG_INF_MASK).to(q.dtype)  # (total, total)
    qh = q.transpose(0, 1)[None]  # (1, H, total, hd)
    kh = k.transpose(0, 1)[None]
    vh = v.transpose(0, 1)[None]
    ctx = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias[None, None])
    return ctx[0].transpose(0, 1).contiguous()  # (total, H, hd)


def _flex_block_diag(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, *, compiled: bool = True
) -> Tensor:
    """Block-diagonal attention via FlexAttention (GPU, no external wheel).

    Same contract as the other backends: q,k,v ``(total, H, hd)``, blocks delimited
    by ``cu_seqlens``. A token attends iff it shares a block. Softmax scale
    ``1/sqrt(hd)`` (flex default). ``compiled=False`` runs eager ``flex_attention``
    (tiny-N numerical test only — eager materialises the score grid)."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    total = q.shape[0]
    seg = _segment_ids(cu_seqlens, total)  # (total,) block id per token

    def _mask_mod(b, h, q_idx, kv_idx):
        return seg[q_idx] == seg[kv_idx]

    # ``_compile`` builds the BlockMask per 128-tile (mask_mod evaluated block-wise);
    # WITHOUT it, create_block_mask materialises the dense ``(total, total)`` grid and
    # OOMs at real token counts (the F2b GH200 failure). ``total`` is constant per
    # session (constant masked-count invariant) ⇒ compiles once per session, reused.
    block_mask = create_block_mask(
        _mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=q.device,
        _compile=compiled,
    )
    qh = q.transpose(0, 1)[None]  # (1, H, total, hd)
    kh = k.transpose(0, 1)[None]
    vh = v.transpose(0, 1)[None]
    fn = _flex_fn() if compiled else flex_attention
    ctx = fn(qh, kh, vh, block_mask=block_mask)  # scale=None ⇒ 1/sqrt(hd)
    return ctx[0].transpose(0, 1).contiguous()  # (total, H, hd)


def _flash_varlen(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int
) -> Tensor:
    from flash_attn import flash_attn_varlen_func

    cu = cu_seqlens.to(torch.int32)
    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        causal=False,
    )
    return out


def varlen_block_diag_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    *,
    backend: str = "auto",
) -> Tensor:
    """Block-diagonal attention on the flat packed layout.

    q, k, v: ``(total, H, head_dim)`` — the packed tokens, blocks contiguous and
    delimited by ``cu_seqlens`` ``(n_seg+1,)``. ``max_seqlen`` = the longest block
    length (static upper bound = ``max_c * T``). Returns ``(total, H, head_dim)``.
    Softmax scale = ``1/sqrt(head_dim)`` on both backends (SDPA's default matches
    the explicit flash scale).
    """
    if backend == "auto":
        if _flash_available(q):
            backend = "flash"
        elif q.is_cuda:
            backend = "flex"  # GPU without a flash wheel (aarch64/GH200)
        else:
            backend = "reference"
    if backend == "flash":
        return _flash_varlen(q, k, v, cu_seqlens, max_seqlen)
    if backend == "flex":
        return _flex_block_diag(q, k, v, cu_seqlens)
    if backend == "reference":
        return _reference_block_diag(q, k, v, cu_seqlens)
    raise ValueError(f"unknown backend {backend!r}")
