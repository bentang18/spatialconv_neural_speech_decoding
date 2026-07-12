"""v14_converged_v3 Phase D1 — per-session setup (TDD).

``build_session_setup`` is the pure core of the per-session build (memo
project-v3-pipeline-build-contract-2026-07-10): from the FULL session voltage
electrode order + hard parcel tags + the guard-1 bad-electrode set, produce the
survivor ordering the clip loader reads (``keep_idx``), the sidecar (shaft/depth/
parcel), the L1 geometry, and a fail-loud feasibility check (#36).

GUARD-1 (bad ELECTRODES) is voltage-domain + hop-independent, so it lands HERE at
sidecar/geometry build, NOT as a cache recompute: drop ``extra_bad`` (manual,
travels in the spec .json ``key.timeline.extra_bad``) ∪ LOF/``drop_bads`` from the
valid set. Both sources name ELECTRODES (label strings), the stable identity
across the two sources, so the core drops by label.

The depth-gap invariant (Ben's RoPE catch) is the sharpest test: dropping a
mid-shaft contact must LEAVE THE GAP in ``depth`` — the survivors keep their raw
clinical index, never re-densified.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig
from speech_decoding.models.v14_converged_v3.session_setup import build_session_setup


def _labels(shaft_sizes):
    labels = []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
    return labels


def _parcels(labels):
    # one parcel per shaft prefix, deterministic
    prefixes = {}
    pid = []
    for lab in labels:
        pre = lab.rstrip("0123456789")
        prefixes.setdefault(pre, len(prefixes))
        pid.append(prefixes[pre])
    return torch.tensor(pid, dtype=torch.long)


def test_no_drops_keeps_every_contact_in_order() -> None:
    labels = _labels((4, 3, 3))
    parcel = _parcels(labels)
    setup = build_session_setup(labels, parcel, drop_labels=set())
    assert setup.keep_idx.tolist() == list(range(10))
    assert setup.sidecar.labels == tuple(labels)
    assert torch.equal(setup.parcel_id, parcel)
    assert setup.geom.n_shafts == 3


def test_drop_removes_named_electrodes_order_preserved() -> None:
    labels = _labels((4, 3, 3))  # LA1..4 LB1..3 LC1..3
    parcel = _parcels(labels)
    setup = build_session_setup(labels, parcel, drop_labels={"LA2", "LB1"})
    kept = [labels[i] for i in setup.keep_idx.tolist()]
    assert kept == ["LA1", "LA3", "LA4", "LB2", "LB3", "LC1", "LC2", "LC3"]
    assert setup.keep_idx.tolist() == [0, 2, 3, 5, 6, 7, 8, 9]


# These four exercise electrode BOOKKEEPING (keep_idx / depth gaps / parcel realign),
# not masking. On these toy 2-shaft montages the dominant shaft exceeds D=round(0.5·N),
# which trips the whole-shaft feasibility assert — irrelevant here, so they opt out with
# ``whole_shaft_frac=0`` (intra-only masking; real montages keep the default).
_NO_WHOLE = V3MaskConfig(whole_shaft_frac=0.0)


def test_depth_gap_preserved_after_mid_shaft_drop() -> None:
    # LA1..LA5 → drop LA3 → survivors keep clinical depths [1,2,4,5], NOT re-densified.
    labels = ["LA1", "LA2", "LA3", "LA4", "LA5", "LB1", "LB2"]
    parcel = _parcels(labels)
    setup = build_session_setup(labels, parcel, drop_labels={"LA3"}, mask_cfg=_NO_WHOLE)
    la = setup.sidecar.shaft_id == 0
    assert setup.sidecar.depth[la].tolist() == [1, 2, 4, 5]


def test_parcel_id_realigns_to_survivors() -> None:
    labels = _labels((3, 3))
    parcel = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.long)
    setup = build_session_setup(labels, parcel, drop_labels={"LA2"}, mask_cfg=_NO_WHOLE)
    # LA2 (parcel 11) removed; survivors carry [10,12,20,21,22]
    assert setup.parcel_id.tolist() == [10, 12, 20, 21, 22]


def test_keep_idx_indexes_full_order_for_memmap_read() -> None:
    labels = _labels((4, 3))
    parcel = _parcels(labels)
    setup = build_session_setup(
        labels, parcel, drop_labels={"LA1", "LB3"}, mask_cfg=_NO_WHOLE
    )
    # keep_idx must be a strictly-ascending gather into the full voltage order
    ki = setup.keep_idx
    assert ki.dtype == torch.long
    assert torch.equal(ki, ki.sort().values)
    assert (ki[1:] > ki[:-1]).all()


def test_drop_label_not_in_montage_is_a_noop() -> None:
    # extra_bad may name a channel already excluded upstream; intersect, don't fail.
    labels = _labels((3, 3))
    parcel = _parcels(labels)
    setup = build_session_setup(
        labels, parcel, drop_labels={"LZ9", "LA2"}, mask_cfg=_NO_WHOLE
    )
    assert "LA2" not in setup.sidecar.labels
    assert len(setup.sidecar.labels) == 5


def test_feasibility_assert_fires_on_oversubscribed_montage() -> None:
    # Dual-axis (Ben 2026-07-12): whole-shaft dropping is feasible only if SOME shaft
    # fits under D=round(space_frac·N). One dominant shaft > D ⇒ no whole shaft fits ⇒
    # assert_mask_feasible must raise at setup (fail loud, wires #36).
    # S=4, sizes 40,3,3,3 (N=49). D=round(0.5·49)=24; largest shaft 40 > 24.
    labels = _labels((40, 3, 3, 3))
    parcel = _parcels(labels)
    with pytest.raises(ValueError, match="whole-shaft infeasible"):
        build_session_setup(labels, parcel, drop_labels=set())


def test_feasibility_passes_for_uniform_seeg() -> None:
    # Uniform sEEG (largest shaft ≪ D) is always feasible — no raise.
    labels = _labels((8, 8, 8, 8, 8, 8))
    parcel = _parcels(labels)
    setup = build_session_setup(labels, parcel, drop_labels=set())  # must not raise
    assert setup.geom.n_shafts == 6


def test_custom_mask_cfg_threads_to_feasibility() -> None:
    # The dominant-shaft montage (25,5,5,5): D=round(0.5·40)=20, largest 25 > 20 ⇒ the
    # DEFAULT whole tier is infeasible and setup must raise. Setting whole_shaft_frac=0
    # opts OUT of the whole tier (intra-only) ⇒ same montage now passes. Proves mask_cfg
    # threads through to the feasibility gate.
    labels = _labels((25, 5, 5, 5))  # N=40
    parcel = _parcels(labels)
    with pytest.raises(ValueError, match="whole-shaft infeasible"):
        build_session_setup(labels, parcel, drop_labels=set())  # default 0.15 whole
    build_session_setup(
        labels, parcel, drop_labels=set(),
        mask_cfg=V3MaskConfig(whole_shaft_frac=0.0),  # opt out → feasible
    )
