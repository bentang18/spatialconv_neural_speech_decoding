"""v14_converged_v3 — ragged (varlen) block-diagonal attention primitive (#24).

The L1 block attends block-diagonally: within a ``(clip, shaft)`` block the packed
tokens attend to each other, never across blocks. The padded path (``attention.py``
``L1Block``) realises this by gathering to ``(n_shafts, max_c)``, padding short
shafts, and masking the pad — paying the ``max_c`` zero-pad on every linear and
every attention score. This primitive realises the SAME block-diagonal attention
on the FLAT packed layout the pack plan produces (``packing.py``): the selected
tokens laid end to end, ``cu_seqlens`` marking each block, no padding.

Two backends, ONE numerical contract:

  * ``flash``     — ``flash_attn_varlen_func`` (GPU/production, FA2/FA3/FA4 stable API).
                    Physically ragged; the kernel never touches cross-block pairs.
  * ``reference`` — a block-diagonal additive-mask SDPA (CPU/testing). Consumes the
                    SAME ``cu_seqlens`` so it tests the exact flat layout + RoPE-in-
                    flat-layout that flash will run; validated against the padded
                    ``L1Block`` (B1c) and, on GPU, against ``flash`` (F2).

``auto`` picks ``flash`` when q is on CUDA and ``flash_attn`` imports, else
``reference``. flash is imported LAZILY so a CPU box never needs the wheel.
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
        backend = "flash" if _flash_available(q) else "reference"
    if backend == "flash":
        return _flash_varlen(q, k, v, cu_seqlens, max_seqlen)
    if backend == "reference":
        return _reference_block_diag(q, k, v, cu_seqlens)
    raise ValueError(f"unknown backend {backend!r}")
