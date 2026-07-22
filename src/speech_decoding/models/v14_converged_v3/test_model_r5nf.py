"""v3r5nf (no-fusion) model wiring — end-to-end + session_plan.

The top-level V3ConvergedModel must thread no_fusion into the NoFusionStem frontend, the
independent per-stream masks (sample_masks_r5nf), and the two-band grid/flags, and its
session_plan must precompute the same shape constants the forward derives.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

N_PARCELS = 8
T32 = 16


def _session(shaft_sizes=(5, 4, 4)):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s % N_PARCELS)
    sc = build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))
    return sc, build_l1_geometry(sc)


def _streams(n, B=1):
    """(HGA (B,n,4,2T), LFS (B,n,1,2T)) — 64 Hz frames for the NoFusionStem."""
    return [torch.randn(B, n, 4, 2 * T32), torch.randn(B, n, 1, 2 * T32)]


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_nf_end_to_end_mae() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS, no_fusion=True, mae=True)
    assert model.no_fusion is True and model.objective.no_fusion is True
    out = model(_streams(n, B=2), geom, sc.parcel_id, generator=_gen())
    assert out.loss.ndim == 0 and torch.isfinite(out.loss) and out.loss.requires_grad
    out.loss.backward()
    online = model.objective.online
    # grads flow through BOTH stems.
    for stem in (online.stem.hga_stem, online.stem.lfs_stem):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stem.parameters())


def test_nf_session_plan_matches_forward_shapes() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS, no_fusion=True, mae=True)
    gms, m_vis, pms = model.session_plan(geom, sc.parcel_id, T32)
    out = model(
        _streams(n, B=2), geom, sc.parcel_id, generator=_gen(),
        grid_max_seqlen=gms, m_vis=m_vis, pack_max_seqlen=pms,
    )
    assert torch.isfinite(out.loss)
    assert gms > 0 and m_vis > 0 and pms > 0
