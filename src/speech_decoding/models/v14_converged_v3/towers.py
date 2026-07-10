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

from collections.abc import Sequence

from torch import Tensor, nn

from speech_decoding.models.v14_converged_v3.attention import L1Block, L2Block
from speech_decoding.models.v14_converged_v3.geometry import L1Geometry

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
        blocks: list[nn.Module] = []
        for kind in layout:
            if kind == "L1":
                blocks.append(L1Block(d_model, n_heads))
            elif kind == "L2":
                blocks.append(L2Block(d_model, n_heads, n_parcels=n_parcels))
            else:
                raise ValueError(f"unknown block kind {kind!r}")
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: Tensor,
        geom: L1Geometry,
        parcel_id: Tensor,
        visible: Tensor | None = None,
    ) -> Tensor:
        for b in self.blocks:
            x = (
                b(x, geom, visible)
                if isinstance(b, L1Block)
                else b(x, parcel_id, visible)
            )
        return x


def build_encoder(*, n_parcels: int) -> V3Tower:
    return V3Tower(
        ENC_LAYOUT, d_model=ENC_D_MODEL, n_heads=ENC_N_HEADS, n_parcels=n_parcels
    )


def build_predictor(*, n_parcels: int) -> V3Tower:
    return V3Tower(
        PRED_LAYOUT, d_model=PRED_D_MODEL, n_heads=PRED_N_HEADS, n_parcels=n_parcels
    )
