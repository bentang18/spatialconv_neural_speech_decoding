"""v3r6 (native-perband) model wiring — end-to-end + session_plan.

The top-level V3ConvergedModel must thread native_perband into the NativePerBandStem frontend,
the per-shaft 3-band margin-gated masks (sample_masks_r6), and the r4 grid / token_flags_r6, and
its session_plan must precompute the same shape constants the forward derives.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import (
    NativePerBandStem,
    PER_BAND_SPECS,
)

N_PARCELS = 8
T32 = 32  # 32 Hz clock; native bands SLOW T/8=4, MID T/2=16, HGA T=32.
BINS = tuple(nb for nb, _ in PER_BAND_SPECS)  # (7, 6, 7)
STRIDES = tuple(st for _, st in PER_BAND_SPECS)  # (8, 2, 1)


def _session(shaft_sizes=(5, 4, 4)):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s % N_PARCELS)
    sc = build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))
    return sc, build_l1_geometry(sc)


def _bands(n, B=1):
    """(SLOW (B,n,7,T/8), MID (B,n,6,T/2), HGA (B,n,7,T)) native-rate |STFT| frames."""
    return [torch.randn(B, n, BINS[b], T32 // STRIDES[b]) for b in range(3)]


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_r6_end_to_end_mae() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS, native_perband=True, mae=True)
    assert model.native_perband is True and model.objective.native_perband is True
    assert isinstance(model.objective.online.stem, NativePerBandStem)
    out = model(_bands(n, B=2), geom, sc.parcel_id, generator=_gen())
    assert out.loss.ndim == 0 and torch.isfinite(out.loss) and out.loss.requires_grad
    out.loss.backward()
    stem = model.objective.online.stem
    # grads flow through all three per-band projections + the band-type embedding.
    for proj in stem.projs:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in proj.parameters())
    assert stem.band_type_emb.grad is not None


def test_r6_session_plan_matches_forward_shapes() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS, native_perband=True, mae=True)
    gms, m_vis, pms = model.session_plan(geom, sc.parcel_id, T32)
    out = model(
        _bands(n, B=2), geom, sc.parcel_id, generator=_gen(),
        grid_max_seqlen=gms, m_vis=m_vis, pack_max_seqlen=pms,
    )
    assert torch.isfinite(out.loss)
    assert gms > 0 and m_vis > 0 and pms > 0
