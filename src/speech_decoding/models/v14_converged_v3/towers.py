"""v14_converged_v3 — encoder / predictor towers (Phase 4c).

Two pre-norm block stacks over the (B, N, T, d) token grid, dispatching each
block to its geometry (L1 ← shaft gather; L2 ← parcel identity). Memo
project-v14-converged-v3-sensor-architecture (ENCODER/PREDICTOR OFFLOAD, the
MAE/V-JEPA local-heavy-encoder asymmetry):

  Encoder  12 WIDE, d_model 256, 4 heads (head_dim 64). ``L1×3 · [L2 L1 L1]×3``
           = 9 L1 : 3 L2. A short 3-block local front-load, then three ``L2 L1 L1``
           digests — each cross-sensor L2 mix is followed by 2 local blocks that
           fold it in. Global mixing starts at block 4 (Ben 2026-07-10). The
           encoder RETAINS the local, overfit-safe capacity; its final block
           output is the SINGLE tap.
  Predictor 12 NARROW, d_model 128 (0.5×), 4 heads (head_dim 32). ``[L2L1L1]×4``
           = 8 L1 : 4 L2, L2-first. Carries the cross-sensor capacity that is
           DISCARDED at inference.

The width→width projection (256→128) and mask-query insertion that connect the
encoder tap to the predictor input belong to the JEPA objective assembly
(Phase 6), not these bare towers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from speech_decoding.models.v14_converged_v3.attention import LN_EPS, L1Block, L2Block
from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.packing import PackPlan
from speech_decoding.models.v14_converged_v3.pe import (
    ParcelIdentityEmbed,
    init_transformer_weights,
)

ENC_LAYOUT: tuple[str, ...] = ("L1",) * 3 + ("L2", "L1", "L1") * 3
PRED_LAYOUT: tuple[str, ...] = ("L2", "L1", "L1") * 4

ENC_D_MODEL, ENC_N_HEADS = 256, 4  # head_dim 64
PRED_D_MODEL, PRED_N_HEADS = 128, 4  # head_dim 32

# Deep-supervision taps (V-JEPA 2.1, #61): the depth-12 encoder is tapped at the
# equally-spaced quartiles — upstream `hierarchical_layers=[2,5,8,11]` 0-indexed =
# {3,6,9,12} 1-indexed (`app/vjepa_2_1/models/vision_transformer.py:149`). Each tap
# gets its own affine `norms_block` LN; the outputs are concatenated → [.,4·d].
ENC_SUP_TAPS: tuple[int, ...] = (3, 6, 9, 12)
N_LEVELS = len(ENC_SUP_TAPS)


class V3Tower(nn.Module):
    """Pre-norm L1/L2 block stack; forward dispatches per block kind."""

    def __init__(
        self,
        layout: Sequence[str],
        *,
        d_model: int,
        n_heads: int,
        n_parcels: int,
        deep_sup: bool = False,
        sup_taps: Sequence[int] = (),
    ) -> None:
        super().__init__()
        # Parcel/DKT identity is added ONCE here at the tower input (V-JEPA 2.1
        # modality-embedding style: the learned categorical embed is added to the
        # encoder AND predictor inputs alongside RoPE) — it then rides the residual
        # into every block, so L2 needs no per-block identity injection.
        self.parcel_embed = ParcelIdentityEmbed(n_parcels, d_model)
        blocks: list[nn.Module] = []
        for kind in layout:
            if kind == "L1":
                blocks.append(L1Block(d_model, n_heads))
            elif kind == "L2":
                blocks.append(L2Block(d_model, n_heads))
            else:
                raise ValueError(f"unknown block kind {kind!r}")
        self.blocks = nn.ModuleList(blocks)
        self.deep_sup = bool(deep_sup)
        self.sup_taps = tuple(int(b) for b in sup_taps) if self.deep_sup else ()
        if self.deep_sup:
            # DEEP SUPERVISION (#61, V-JEPA 2.1): tap ``sup_taps`` blocks, each through
            # its OWN affine LayerNorm (`norms_block`, one per level), and CONCATENATE
            # → [.,n_levels·d]. There is NO separate terminal ``norm_out``: upstream
            # reuses the deepest level's norm as the terminal norm
            # (`vision_transformer.py:176-178, 328-340` — training path returns
            # `cat(hier)` with no extra `self.norm`).
            if not self.sup_taps:
                raise ValueError("deep_sup=True requires a non-empty sup_taps")
            self.norms_block = nn.ModuleList(
                [nn.LayerNorm(d_model, eps=LN_EPS) for _ in self.sup_taps]
            )
            self.norm_out = None
        else:
            # Terminal affine LayerNorm (single-tap path) — V-JEPA 2 encoder
            # `self.norm` / predictor `predictor_norm`, also v2 (pred_norm / ln_out).
            # Keeps the predictor output unit-scale against the affine-free-LN'd target
            # (kills the scale-runaway incentive) and makes the target the exact
            # upstream `affinefree_LN(affine_LN(h))` form. The predictor ALWAYS uses
            # this (deep_sup=False) — it emits from one wide proj on its final block.
            self.norms_block = None
            self.norm_out = nn.LayerNorm(d_model, eps=LN_EPS)
        # V-JEPA 2 init (vision_transformer.py:130-159 @204698b4): Linear
        # trunc_normal(0.02)+zero-bias / LN weight 1 bias 0, then depth-scaled
        # residual rescale div_(sqrt(2·layer_id)), 1-indexed, on attn out-proj +
        # mlp.fc2 — verified byte-for-byte against 2.0 AND 2.1. The parcel embed is
        # an nn.Embedding: init_transformer_weights matches only Linear/LayerNorm, so
        # it KEEPS its own near-zero init (1e-6 default; 0.02 is the A/B arm).
        self.apply(init_transformer_weights)
        self._rescale_blocks()

    def _rope_cos_sin(
        self, x: Tensor, plan: PackPlan
    ) -> tuple[Tensor, Tensor] | None:
        # Build the shared L1 rotary table once for this tower forward (B5 #46).
        # Uses the first L1 block's rope; every L1RoPE here is identical so the table
        # serves all L1 blocks. None when the tower has no L1 block.
        first_l1 = next((b for b in self.blocks if isinstance(b, L1Block)), None)
        if first_l1 is None:
            return None
        B, P, T, _ = x.shape
        total = B * P * T
        idx = plan.depth[:, :, None].expand(B, P, T).reshape(total)  # (total,)
        tt = torch.arange(T, device=x.device)[None, None, :].expand(B, P, T).reshape(total)
        return first_l1.rope.cos_sin(idx, tt)

    def _rescale_blocks(self) -> None:
        # V-JEPA 2 vision_transformer.py:231-237 / predictor.py:206-212 — divide the
        # attn out-proj and mlp.fc2 by sqrt(2·layer_id) (1-indexed) so residual-branch
        # variance stays flat with depth.
        for layer_id, block in enumerate(self.blocks):
            scale = math.sqrt(2.0 * (layer_id + 1))
            block.out.weight.data.div_(scale)
            block.mlp.fc2.weight.data.div_(scale)

    def forward(
        self,
        x: Tensor,
        geom: L1Geometry,
        parcel_id: Tensor,
        visible: Tensor | None = None,
        *,
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        # ``tap_blocks`` (monitor only): 1-based block indices whose RAW output to
        # capture (e.g. (3, 12) for the rankme/feat_std depth comparison). Empty ⇒
        # the default single-return path (no tuple), so non-monitor callers and the
        # compiled hot path are unchanged.
        # (B,N,T,d) + (1,N,1,d). ``.to(x.dtype)``: nn.Embedding output is not autocast-
        # downcast (stays fp32), so under bf16-mixed the bare add would promote the whole
        # residual stream back to fp32; cast keeps it bf16. No-op under fp32.
        x = x + self.parcel_embed(parcel_id).to(x.dtype)[None, :, None, :]
        taps: dict[int, Tensor] = {}
        levels: list[Tensor] = []
        for i, b in enumerate(self.blocks):
            x = b(x, geom, visible) if isinstance(b, L1Block) else b(x, visible)
            if (i + 1) in tap_blocks:
                taps[i + 1] = x
            if self.deep_sup and (i + 1) in self.sup_taps:
                levels.append(self.norms_block[self.sup_taps.index(i + 1)](x))
        out = torch.cat(levels, dim=-1) if self.deep_sup else self.norm_out(x)
        if tap_blocks:
            return out, taps
        return out

    # ── packed (varlen) path (#24) — the PRODUCTION forward ──────────────────
    # ``forward`` above is the padded ORACLE (test_towers). The packed forward runs
    # the tower over the SELECTED contacts only (M_vis for the online encoder, all-N
    # for teacher/predictor), shaft-grouped per ``plan.order``. L1 dispatches to the
    # varlen block-diagonal attention; L2 to the dense-over-P mixer. Parcel identity
    # is added ONCE here (packed slots ← ``parcel_id[plan.order]``, supplied as
    # ``parcel_packed``), exactly as the padded path adds it once over N.
    def forward_packed(
        self,
        x: Tensor,
        plan: PackPlan,
        parcel_packed: Tensor,
        *,
        backend: str = "auto",
        tap_blocks: tuple[int, ...] = (),
    ) -> Tensor | tuple[Tensor, dict[int, Tensor]]:
        # x: (B, P, T, d); parcel_packed: (B, P) long — parcel id per packed slot.
        x = x + self.parcel_embed(parcel_packed).to(x.dtype)[:, :, None, :]
        # B5 hoist (#46): the L1 rotary cos/sin table is identical across every L1
        # block in this tower (same head_dim/bases, same plan coords) — build it ONCE
        # from the plan and share, instead of recomputing it in each of the n_L1
        # blocks. Bit-identical (same fp32 table, same rotate math).
        rope_cs = self._rope_cos_sin(x, plan)
        taps: dict[int, Tensor] = {}
        levels: list[Tensor] = []
        for i, b in enumerate(self.blocks):
            if isinstance(b, L1Block):
                x = b.forward_packed(x, plan, backend=backend, rope_cs=rope_cs)
            else:
                x = b.forward_packed(x)
            if (i + 1) in tap_blocks:
                taps[i + 1] = x
            if self.deep_sup and (i + 1) in self.sup_taps:
                levels.append(self.norms_block[self.sup_taps.index(i + 1)](x))
        out = torch.cat(levels, dim=-1) if self.deep_sup else self.norm_out(x)
        if tap_blocks:
            return out, taps
        return out


def build_encoder(*, n_parcels: int, deep_sup: bool = True) -> V3Tower:
    # deep_sup default ON (#61, Ben-greenlit "copy exactly"): tap {3,6,9,12} → 4
    # affine-normed levels concatenated. deep_sup=False = the single-tap ablation arm.
    return V3Tower(
        ENC_LAYOUT, d_model=ENC_D_MODEL, n_heads=ENC_N_HEADS, n_parcels=n_parcels,
        deep_sup=deep_sup, sup_taps=ENC_SUP_TAPS if deep_sup else (),
    )


def build_predictor(*, n_parcels: int) -> V3Tower:
    # The predictor is NEVER deep-supervised: it emits all levels from one wide proj
    # on its final block (objective.pred_to_target), with the terminal norm_out =
    # upstream `predictor_norm`. It does not tap its own intermediate blocks.
    return V3Tower(
        PRED_LAYOUT, d_model=PRED_D_MODEL, n_heads=PRED_N_HEADS, n_parcels=n_parcels
    )
