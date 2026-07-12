"""v14_converged_v3 Phase 7 — top-level assembly (TDD).

End-to-end wiring of the settled components: 3-band |STFT| inputs → SpectralStem
(Phase 2) → sample the per-session electrode-tube mask (Phase 5) → plain-JEPA
objective (Phase 6) over the L1/L2 towers (Phase 4) with the sidecar geometry
(Phase 1/4a). One forward turns raw band frames into the masked-JEPA loss.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

N_PARCELS = 8
T32 = 16  # small clock for tests; uniform hop=64 → all bands at 32 Hz (T32 each)


def _session(shaft_sizes=(5, 4, 4)):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s % N_PARCELS)
    sc = build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))
    return sc, build_l1_geometry(sc)


def _bands(n, B=1):
    slow = torch.randn(B, n, 7, T32)
    mid = torch.randn(B, n, 6, T32)
    hga = torch.randn(B, n, 7, T32)
    return [slow, mid, hga]


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_end_to_end_bands_to_loss() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    out = model(_bands(n), geom, sc.parcel_id, generator=_gen())
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.loss.requires_grad


def test_masked_count_is_constant_and_matches_frac() -> None:
    # Dual-axis: n_masked = total masked CELLS = N·T − m_vis·t_kept, a per-session
    # CONSTANT (independent of WHICH cells are masked) — verify it holds across seeds.
    sc, geom = _session()
    n = len(sc.labels)  # 13
    cfg = V3MaskConfig(space_frac=0.5, time_frac=0.5)
    model = V3ConvergedModel(n_parcels=N_PARCELS, mask_cfg=cfg)
    a = model(_bands(n), geom, sc.parcel_id, generator=_gen(1))
    b = model(_bands(n), geom, sc.parcel_id, generator=_gen(2))  # different mask
    m_vis = n - round(0.5 * n)
    t_kept = T32 - round(0.5 * T32)
    assert a.n_masked == b.n_masked == n * T32 - m_vis * t_kept


def test_backward_reaches_stem_and_online_towers() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    model(_bands(n), geom, sc.parcel_id, generator=_gen()).loss.backward()
    # the stem now lives inside the EMA-mirrored online tower
    online = model.objective.online
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in online.stem.parameters())
    assert all(p.grad is None for p in model.objective.teacher.parameters())


def test_update_teacher_advances() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    model(_bands(n), geom, sc.parcel_id, generator=_gen()).loss.backward()
    coeff = model.update_teacher()
    assert 0.0 < coeff < 1.0


def test_deterministic_given_generator() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    torch.manual_seed(0)
    m1 = V3ConvergedModel(n_parcels=N_PARCELS)
    bands = _bands(n)
    a = m1(bands, geom, sc.parcel_id, generator=_gen(3)).loss
    b = m1(bands, geom, sc.parcel_id, generator=_gen(3)).loss
    assert torch.allclose(a, b)


def test_batched_clips_get_independent_masks() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS)
    out = model(_bands(n, B=4), geom, sc.parcel_id, generator=_gen())
    assert torch.isfinite(out.loss)
    m_vis = n - round(V3MaskConfig().space_frac * n)
    t_kept = T32 - round(V3MaskConfig().time_frac * T32)
    assert out.n_masked == 4 * (n * T32 - m_vis * t_kept)
