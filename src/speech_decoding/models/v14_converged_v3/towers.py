"""v14_converged_v3 — encoder / predictor towers (Phase 4c).

Two pre-norm block stacks over the (B, N, T, d) token grid, dispatching each
block to its geometry (L1 ← shaft gather; L2 ← parcel identity). Memo
project-v14-converged-v3-sensor-architecture (ENCODER/PREDICTOR OFFLOAD, the
MAE/V-JEPA local-heavy-encoder asymmetry):

  Encoder  12 WIDE, d_model 256, 4 heads (head_dim 64). ``L1×6 · L2L1L1 · L2L1L1``
           = 10 L1 : 2 L2. The encoder RETAINS the local, overfit-safe capacity;
           its final block output is the SINGLE tap.
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

from torch import Tensor, nn

from speech_decoding.models.v14_converged_v3.attention import LN_EPS, L1Block, L2Block
from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.pe import (
    ParcelIdentityEmbed,
    init_transformer_weights,
)

ENC_LAYOUT: tuple[str, ...] = ("L1",) * 6 + ("L2", "L1", "L1") * 2
PRED_LAYOUT: tuple[str, ...] = ("L2", "L1", "L1") * 4

ENC_D_MODEL, ENC_N_HEADS = 256, 4  # head_dim 64
PRED_D_MODEL, PRED_N_HEADS = 128, 4  # head_dim 32


class V3Tower(nn.Module):
    """Pre-norm L1/L2 block stack; forward dispatches per block kind."""

    def __init__(
        self,
        layout: Sequence[str],
        *,
        d_model: int,
        n_heads: int,
        n_parcels: int,
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
        # Terminal affine LayerNorm, applied to BOTH towers before the downstream
        # projection — V-JEPA 2 encoder `self.norm` (vision_transformer.py:210) and
        # predictor `predictor_norm` (predictor.py:241), also present in v2
        # (pred_norm / ln_out). Keeps the predictor output unit-scale against the
        # affine-free-LN'd target (kills the scale-runaway incentive: the target is
        # already unit, so the affine gamma has nothing to inflate toward), and makes
        # the teacher target `affinefree_LN(affine_LN(h))` — the exact upstream form.
        self.norm_out = nn.LayerNorm(d_model, eps=LN_EPS)
        # V-JEPA 2 init (vision_transformer.py:115-116): trunc_normal(0.02)+zero-bias
        # on Linears / LN 1,0, then depth-scaled residual rescale. The parcel embed
        # self-inits (0.02) and is skipped by init_transformer_weights.
        self.apply(init_transformer_weights)
        self._rescale_blocks()

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
        # capture (e.g. (6, 12) for the rankme/feat_std depth comparison). Empty ⇒
        # the default single-return path (no tuple), so non-monitor callers and the
        # compiled hot path are unchanged.
        x = x + self.parcel_embed(parcel_id)[None, :, None, :]  # (B,N,T,d) + (1,N,1,d)
        taps: dict[int, Tensor] = {}
        for i, b in enumerate(self.blocks):
            x = b(x, geom, visible) if isinstance(b, L1Block) else b(x, visible)
            if (i + 1) in tap_blocks:
                taps[i + 1] = x
        out = self.norm_out(x)
        if tap_blocks:
            return out, taps
        return out


def build_encoder(*, n_parcels: int) -> V3Tower:
    return V3Tower(
        ENC_LAYOUT, d_model=ENC_D_MODEL, n_heads=ENC_N_HEADS, n_parcels=n_parcels
    )


def build_predictor(*, n_parcels: int) -> V3Tower:
    return V3Tower(
        PRED_LAYOUT, d_model=PRED_D_MODEL, n_heads=PRED_N_HEADS, n_parcels=n_parcels
    )
