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
import os

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
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    *,
    n_clips: int | None = None,
    compiled: bool = True,
) -> Tensor:
    """Block-diagonal attention via FlexAttention (GPU, no external wheel).

    Same contract as the other backends: q,k,v ``(total, H, hd)``, blocks delimited
    by ``cu_seqlens``. A token attends iff it shares a block. Softmax scale
    ``1/sqrt(hd)`` (flex default). ``compiled=False`` runs eager ``flex_attention``
    (tiny-N numerical test only — eager materialises the score grid).

    ``n_clips`` (the packed batch B) switches to the BATCHED-per-clip layout: the
    flat ``total = B·P·T`` tokens are reshaped to ``(B, P·T)`` and flex is given a
    per-clip block-diagonal mask over the ``S`` shafts. This is IDENTICAL to the flat
    global mask — a token attends iff it shares (clip, shaft): same-clip is enforced
    by the batch dim, same-shaft by the mask — but it drops flex's tile-walk from
    ``(B·P·T/128)²`` to ``B·(P·T/128)²`` (~B× less), the G4 320k-token O(seq²) cost.
    ``None`` keeps the flat single-sequence layout (CPU test / tiny-N)."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    total = q.shape[0]
    H, hd = q.shape[1], q.shape[2]
    fn = _flex_fn() if compiled else flex_attention

    if n_clips is not None:
        # BATCHED per-clip: local shaft id (0..S-1) within each clip. The flat block
        # id (0..B·S-1) minus clip·S; clip = token // (P·T). Blocks are clip-major so
        # the reshape (total,)→(B, P·T) lands each clip's tokens contiguously.
        seg_flat = _segment_ids(cu_seqlens, total)  # (total,) 0..B*S-1
        s_per_clip = (cu_seqlens.shape[0] - 1) // n_clips  # S
        length = total // n_clips  # P*T
        clip_of = torch.arange(total, device=q.device) // length
        seg_bl = (seg_flat - clip_of * s_per_clip).reshape(n_clips, length)  # (B, P*T)

        def _mask_mod(b, h, q_idx, kv_idx):
            return seg_bl[b, q_idx] == seg_bl[b, kv_idx]

        block_mask = create_block_mask(
            _mask_mod, B=n_clips, H=None, Q_LEN=length, KV_LEN=length,
            device=q.device, _compile=compiled,
        )
        qb = q.reshape(n_clips, length, H, hd).transpose(1, 2)  # (B, H, P*T, hd)
        kb = k.reshape(n_clips, length, H, hd).transpose(1, 2)
        vb = v.reshape(n_clips, length, H, hd).transpose(1, 2)
        ctx = fn(qb, kb, vb, block_mask=block_mask)  # (B, H, P*T, hd)
        return ctx.transpose(1, 2).reshape(total, H, hd).contiguous()

    seg = _segment_ids(cu_seqlens, total)  # (total,) block id per token

    def _mask_mod_flat(b, h, q_idx, kv_idx):
        return seg[q_idx] == seg[kv_idx]

    # ``_compile`` builds the BlockMask per 128-tile (mask_mod evaluated block-wise);
    # WITHOUT it, create_block_mask materialises the dense ``(total, total)`` grid and
    # OOMs at real token counts (the F2b GH200 failure). ``total`` is constant per
    # session (constant masked-count invariant) ⇒ compiles once per session, reused.
    block_mask = create_block_mask(
        _mask_mod_flat, B=None, H=None, Q_LEN=total, KV_LEN=total, device=q.device,
        _compile=compiled,
    )
    qh = q.transpose(0, 1)[None]  # (1, H, total, hd)
    kh = k.transpose(0, 1)[None]
    vh = v.transpose(0, 1)[None]
    ctx = fn(qh, kh, vh, block_mask=block_mask)  # scale=None ⇒ 1/sqrt(hd)
    return ctx[0].transpose(0, 1).contiguous()  # (total, H, hd)


def sdpa_block_diag_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grid_pos: Tensor,
    n_shafts: int,
    max_c: int,
    n_time: int,
) -> Tensor:
    """Block-diagonal L1 attention via dense SDPA-per-block — the PRODUCTION path.

    q,k,v: ``(total = B*P*T, H, hd)`` — the packed tokens (linears already run on the
    packed P, keeping #24's visible-gather saving). ``grid_pos`` ``(B, P)`` is each
    packed slot's flat cell in the padded ``(S, max_c)`` grid (``PackPlan.grid_pos``).

    The packed q/k/v are SCATTERED into the padded ``(B, S*max_c, T)`` grid, reshaped
    to ``(B*S, max_c*T)`` so each ``(clip, shaft)`` block is one batch row, and a SINGLE
    dense ``scaled_dot_product_attention`` runs per block with the pad cells masked out
    (finite ``NEG_INF_MASK`` so a whole-masked row softmaxes to uniform, never NaN — the
    v14 convention). The context is gathered back to the packed layout. This is EXACTLY
    the padded ``L1Block._attn`` oracle's SDPA (same grid, same key set, same RoPE'd
    q/k) — per-contact-identical — and runs on every device (SDPA picks the math kernel
    on CPU, cuDNN/flash on GPU). No FlexAttention, no external flash wheel."""
    total, H, hd = q.shape
    B, P = grid_pos.shape
    T = n_time
    S = n_shafts
    G = S * max_c  # grid rows (padded contacts per clip)

    idx = grid_pos[:, :, None, None, None].expand(B, P, T, H, hd)  # scatter/gather index
    qg = q.new_zeros(B, G, T, H, hd).scatter_(1, idx, q.reshape(B, P, T, H, hd))
    kg = k.new_zeros(B, G, T, H, hd).scatter_(1, idx, k.reshape(B, P, T, H, hd))
    vg = v.new_zeros(B, G, T, H, hd).scatter_(1, idx, v.reshape(B, P, T, H, hd))

    # key mask: a grid cell is a valid key iff a packed slot scattered into it. (B, G) →
    # (B*S, 1, 1, max_c*T), additive NEG (not -inf) so all-pad query rows go uniform.
    filled = q.new_zeros(B, G, dtype=torch.bool).scatter_(
        1, grid_pos, torch.ones_like(grid_pos, dtype=torch.bool)
    )
    key_ok = filled.reshape(B * S, max_c)[:, :, None].expand(B * S, max_c, T).reshape(B * S, 1, 1, max_c * T)
    attn_bias = torch.where(key_ok, 0.0, NEG_INF_MASK).to(v.dtype)

    # (B, S, max_c, T, H, hd) → (B*S, H, max_c*T, hd). Token = slot*T + t (contact-major),
    # matching the oracle's c*T + t ordering.
    def _to_blocks(g: Tensor) -> Tensor:
        return g.reshape(B * S, max_c * T, H, hd).transpose(1, 2)

    ctx = F.scaled_dot_product_attention(
        _to_blocks(qg), _to_blocks(kg), _to_blocks(vg), attn_mask=attn_bias
    )  # (B*S, H, max_c*T, hd)
    ctx = ctx.transpose(1, 2).reshape(B, G, T, H, hd)
    return ctx.gather(1, idx).reshape(total, H, hd)  # back to packed (total, H, hd)


@torch._dynamo.disable
def njt_block_diag(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens_drop: Tensor, max_seqlen: int
) -> Tensor:
    """Block-diagonal L1 attention via jagged NestedTensor — the GPU PRODUCTION path.

    Same block-diagonal attention as ``sdpa_block_diag_packed`` (the CPU/test oracle)
    but with NO pad and NO mask: ``nested_tensor_from_jagged`` VIEWS the flat packed
    ``(total, H, hd)`` values as a jagged ``(n_block, jᵢ, H, hd)`` delimited by the
    offsets, so SDPA dispatches to the ragged flash-varlen kernel (fast fwd AND bwd)
    instead of the mask-forced mem-efficient kernel. ~3× on attention fwd+bwd at real
    shapes. Grad-preserving: ``_from_jagged`` views (unlike the ``nested_tensor``
    constructor, which detaches).

    ``cu_seqlens_drop`` (``PackPlan.cu_seqlens_drop``, int64) is the COMPACTED offset
    table — whole-masked shafts already dropped (an empty jagged row corrupts the varlen
    backward: NaN grad). The drop is a data-dependent ``masked_select`` that CPU-syncs;
    it is done ONCE per plan in ``build_pack_plan`` and reused across every L1 block, so
    a ~9-block encoder / ~8-block predictor tower pays 1 sync, not ~12 per block. Passing
    ``max_seqlen`` (the static ``max_c*T`` bound) explicitly likewise skips the ragged
    SDPA's own offset-diff reduction (a second per-call sync). Net: ≈206→4 DtoH syncs on
    a 17-call step, measured. Empty blocks carry no tokens, so dropping their boundary
    leaves the packed values + order unchanged — the varlen-native "drop" of an absent
    shaft.

    Runs EAGER (``torch._dynamo.disable``): the NestedTensor subclass tangent breaks
    torch-2.10 compiled autograd (AOTAutograd ``process_runtime_tangent`` asserts), so
    the SDPA is a graph break — projections/RoPE around it stay compiled, and the
    per-step-varying offsets never touch the compiled graph (no recompile). Default
    SDPA scale ``1/sqrt(hd)`` matches the oracle. Returns packed ``(total, H, hd)``."""

    def _nt(x: Tensor) -> Tensor:
        return torch.nested.nested_tensor_from_jagged(
            x, offsets=cu_seqlens_drop, min_seqlen=1, max_seqlen=max_seqlen
        ).transpose(1, 2)

    ctx = F.scaled_dot_product_attention(_nt(q), _nt(k), _nt(v))
    return ctx.transpose(1, 2).values()


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


# ── the GPU L1 dispatcher (what attention.py actually calls) ──────────────────────
# ``varlen_block_diag_attention`` below is the ORIGINAL backend selector and is reached
# only from tests; the towers call ``gpu_block_diag``. Keeping the split deliberate: the
# selector picks flex on a CUDA box without flash, which is ~10x SLOWER than SDPA-per-block
# and must never reach production (47ec65f replaced it).

_FLASH_ENV = os.environ.get("V3_FLASH_VARLEN", "0") == "1"
_FLASH_FN = None
_FLASH_LOOKED = False


def _flash_varlen_fn():
    """Cached module-level import of ``flash_attn_varlen_func`` (None when absent).

    HOISTED OUT OF THE CALL ON PURPOSE. An ``import`` inside a compiled function is
    itself a graph break, which would cancel the one thing routing L1 to flash buys.
    """
    global _FLASH_FN, _FLASH_LOOKED
    if not _FLASH_LOOKED:
        _FLASH_LOOKED = True
        try:
            from flash_attn import flash_attn_varlen_func
        except ImportError:
            _FLASH_FN = None
        else:
            _FLASH_FN = flash_attn_varlen_func
    return _FLASH_FN


def flash_varlen_available(q: Tensor) -> bool:
    """Whether flash can serve THIS tensor: opted in, CUDA, a wheel, and a half dtype.

    FA2 rejects fp32, so the dtype test is a correctness guard, not a preference.

    OFF BY DEFAULT — set ``V3_FLASH_VARLEN=1`` to enable. flash and njt are different
    kernels, so a default-on flag would silently change numerics on every future run and
    resume, breaking arm-vs-arm comparability against an njt-trained baseline.
    """
    return (
        _FLASH_ENV
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and _flash_varlen_fn() is not None
    )


def flash_varlen_block_diag(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens_drop: Tensor, max_seqlen: int
) -> Tensor:
    """Block-diagonal L1 attention via ``flash_attn_varlen_func`` — njt's exact contract.

    Consumes the SAME compacted offsets as ``njt_block_diag`` (whole-masked shafts already
    dropped, so no empty jagged row reaches the kernel) and the same explicit
    ``1/sqrt(head_dim)`` scale that SDPA applies by default. Unlike ``njt_block_diag`` this
    carries NO ``torch._dynamo.disable``: staying inside the compiled graph is the point.
    Returns packed ``(total, H, head_dim)``.
    """
    fn = _flash_varlen_fn()
    assert fn is not None, "flash_varlen_block_diag called without a flash_attn build"
    cu = cu_seqlens_drop.to(torch.int32)
    return fn(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(q.shape[-1]),
        causal=False,
    )


def gpu_block_diag(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens_drop: Tensor, max_seqlen: int
) -> Tensor:
    """GPU L1 attention: flash when available and opted in, else njt (the default)."""
    if flash_varlen_available(q):
        return flash_varlen_block_diag(q, k, v, cu_seqlens_drop, max_seqlen)
    return njt_block_diag(q, k, v, cu_seqlens_drop, max_seqlen)


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
