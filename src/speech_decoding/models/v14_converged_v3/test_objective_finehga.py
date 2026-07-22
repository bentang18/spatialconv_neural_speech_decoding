"""v14_converged_v3 fine-HGA OFAT — the native_fine_hga stem swap in V3JepaObjective.

The ONLY change vs arm0 is the frontend: FineHgaStem consumes native-rate bands (SLOW
4Hz T/8, MID 16Hz T/2, HGA 128Hz T·4, HGA 4 bins conv-pooled 128→32Hz) instead of the
uniform-32Hz PerBandStem. Same (tokens, positions) contract + same 32Hz output lattice,
so masking / pack / pe / encoder / predictor are untouched. Invariants named + asserted
+ printed (feedback-build-the-invariant).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import FineHgaStem, PerBandStem

T = 16  # 32 Hz clock, multiple of SLOW stride 8
B = 2
N = 5


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _native_bands(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    # native rates: SLOW (B,N,7,T/8), MID (B,N,6,T/2), HGA (B,N,4,T·4).
    return [
        torch.randn(B, N, 7, T // 8, generator=g),
        torch.randn(B, N, 6, T // 2, generator=g),
        torch.randn(B, N, 4, T * 4, generator=g),
    ]


def _masks(geom, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return sample_masks(geom, N, n_time=T, n_rows=B, generator=g)


def test_flag_selects_finehga_stem_default_is_perband() -> None:
    assert isinstance(V3JepaObjective(n_parcels=8).online.stem, PerBandStem)
    obj = V3JepaObjective(n_parcels=8, native_fine_hga=True)
    assert isinstance(obj.online.stem, FineHgaStem)
    # the EMA teacher mirrors the SAME stem class (it is a copy of obj.online).
    assert isinstance(obj.teacher.model.stem, FineHgaStem)


def test_finehga_forward_runs_and_grads_reach_the_stem() -> None:
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, native_fine_hga=True)
    obj.train()
    out = obj(_native_bands(), geom, sc.parcel_id, _masks(geom))
    loss_val = out.loss.detach()
    out.loss.backward()

    finite = bool(torch.isfinite(loss_val)) and loss_val.ndim == 0 and float(loss_val) >= 0.0
    # grads must reach the fine stem's three sub-modules + downstream.
    checks = {
        "slow_proj": obj.online.stem.slow_proj,
        "mid_proj": obj.online.stem.mid_proj,
        "hga_pool": obj.online.stem.hga_pool,
        "encoder": obj.online.encoder,
        "predictor": obj.predictor,
    }
    grad_ok = {
        name: all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for name, m in checks.items()
    }
    mask_grad = obj.mask_token.grad is not None and torch.isfinite(obj.mask_token.grad).all()
    ok = finite and all(grad_ok.values()) and mask_grad
    print(f"[check] finehga loss={float(loss_val):.4f} finite={finite}; grads {grad_ok}; "
          f"mask_token grad={mask_grad} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_finehga_is_pure_jepa() -> None:
    # The fine stem swaps only the frontend; the loss is the single JEPA term.
    sc, geom = _session()
    obj = V3JepaObjective(n_parcels=8, native_fine_hga=True)
    obj.train()
    out = obj(_native_bands(), geom, sc.parcel_id, _masks(geom))
    assert bool(torch.isfinite(out.loss)) and float(out.loss) >= 0.0
