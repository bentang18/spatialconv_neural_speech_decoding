"""v14_converged_v3 Phase 7 — top-level assembly (TDD).

End-to-end wiring of the settled components: 3-band |STFT| inputs → SpectralStem
(Phase 2) → sample the per-session electrode-tube mask (Phase 5) → plain-JEPA
objective (Phase 6) over the L1/L2 towers (Phase 4) with the sidecar geometry
(Phase 1/4a). One forward turns raw band frames into the masked-JEPA loss.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig, sample_masks
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    build_r4_grid,
    build_visible_pack,
    token_flags,
)
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


def test_masked_query_count_is_constant_across_seeds() -> None:
    # The compile-once-per-session contract: mask REALIZATION must not change any compiled
    # shape. The predictor-query set (token_flags `masked`) has a per-session-CONSTANT size —
    # masking.py fixes the per-shaft spatial d_s and per-band temporal counts and the time mask
    # is global, so masked.sum() = (#masked contacts)·ΣT_b + (#visible contacts)·Σ(masked
    # band-tokens) is independent of WHICH tokens are hidden ⇒ the visible-pack m_vis (the
    # online encoder's compiled length) is seed-invariant. The margin-gated SCORED count
    # (`n_masked`/in_loss) IS realization-dependent — it is the loss denominator, not a shape.
    sc, geom = _session()
    n = len(sc.labels)  # 13
    grid = build_r4_grid(geom, n_time=T32)
    cfg = V3MaskConfig(space_frac=0.5)

    def shapes(seed):
        m = sample_masks(geom, n, n_time=T32, n_rows=1, generator=_gen(seed), cfg=cfg)
        masked, _ = token_flags(grid, m)
        pack = build_visible_pack(grid, masked, sc.parcel_id[grid.contact])
        return int(masked.sum()), pack.m_vis

    (ma, va), (mb, vb) = shapes(1), shapes(2)  # two different mask realizations
    assert ma == mb > 0  # session-constant query count
    assert va == vb  # ⇒ seed-invariant compiled online length


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
    grid = build_r4_grid(geom, n_time=T32)
    m = sample_masks(
        geom, n, n_time=T32, n_rows=4, generator=_gen(), cfg=V3MaskConfig(space_frac=0.5)
    )
    masked, _ = token_flags(grid, m)  # (4, total)
    per_row = masked.sum(1)
    # per-clip query count is a session constant ⇒ every row hides the SAME number of tokens…
    assert (per_row == per_row[0]).all() and int(per_row[0]) > 0
    # …but the masks are INDEPENDENT: not every row hides the identical token set.
    assert not bool((masked == masked[0]).all(1).all())
    # and the model forward runs finite on the B=4 batch.
    out = V3ConvergedModel(n_parcels=N_PARCELS)(
        _bands(n, B=4), geom, sc.parcel_id, generator=_gen()
    )
    assert torch.isfinite(out.loss)
