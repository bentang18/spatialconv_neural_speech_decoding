"""L3 — Golden tracer-marble end-to-end electrode-row alignment integrity test.

The single most load-bearing invariant in the v14 data line: for every event and
every electrode-axis row ``c`` the three tensors the encoder joins by bare
positional index must describe the SAME physical electrode —
``electrode_tokens[c]`` (front-end |STFT| of the voltage), ``support[c]`` (DK
one-hot parcel), ``valid_mask[c]`` (real-vs-pad gate) — all equal to
``voltage_electrode_order(subject)[c]`` for ``c < n_real``. Violating it routes a
voltage into the wrong parcel SILENTLY (the model has no identity to cross-check).
See reports/bt_alignment/electrode_desync_damage_2026_06_09.md.

This is the *tracer-marble* test: drop a uniquely-coloured marble in each
electrode slot at the front-end input, run it through the REAL pooled scatter
(``MultiStftView._scatter_spec_to_global``, the exact call the spec-cache and the
recompute path use), and confirm the marble that comes out at row ``c`` is the
same physical electrode the (independently built) DK support / valid-mask
extractors put at row ``c``. A heterogeneous multi-subject cohort is used because
the desync only manifests when more than one subject shares the pooled
``MneRaw._channels`` dict — single-subject is always aligned.

Regression direction: if the per-subject-voltage-order fix in
``MultiStftView._get_channels`` is reverted to the shared last-writer-wins global
index, the tracer at row ``c`` decodes to some OTHER subject's electrode while
support/valid stay in voltage order → the three-way tie below fails loud.

Scope note: this exercises the scatter lane up to ``C_global``; the front-end's
own ``C_global -> c_max`` zero-pad lives in ``_finalize_clip`` and is covered by
the view tests. Real-voltage clips are DCC-only, so the front-end lane is driven
by a tracer rather than |STFT| of recorded voltage — the scatter mapping (the
thing that desynced) is identical either way.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
from speech_decoding.extractors.valid_mask import ElectrodeValidMask
from speech_decoding.extractors.view import MultiStftView
from speech_decoding.studies.braintreebank.anatomy import (
    aligned_voltage_support,
    voltage_electrode_order,
)

# The single most load-bearing data-line invariant — gate the dispatch on it.
# Real-cohort tracer tests skip cleanly when vendored fixtures are absent (CI);
# the synthetic fuzz still runs everywhere. Run by `scripts/dcc/dispatch`.
pytestmark = pytest.mark.must_pass_before_dispatch

_BT_CACHE = pathlib.Path(__file__).resolve().parents[3] / ".cache" / "braintreebank"

# Heterogeneous cohort (electrode counts 238 / 183 / 96). Deliberately ordered so
# the LAST writer (sub_9, 96) is the SHORTEST and an earlier writer (sub_7, 238)
# is the longest — exercising the row-count guard (``C_global`` must stay 238, not
# shrink to 96). sub_4 & sub_7 share 64 electrode names, the heaviest collision in
# the vendored fixtures, so the buggy shared-dict scatter would lose data here.
_COHORT = (7, 4, 9)
_C_MAX = 384  # production lock (dispatch_v14.py), > longest cohort session


def _require_cache() -> str:
    if not (_BT_CACHE / "electrode_labels" / "sub_7").exists():
        pytest.skip("vendored braintreebank fixtures not present")
    if not (_BT_CACHE / "localization" / "sub_7" / "depth-wm.csv").exists():
        pytest.skip("vendored braintreebank localization not present")
    return str(_BT_CACHE)


def _pooled_view(orders: dict[int, list[str]]) -> MultiStftView:
    """A MultiStftView whose ``_channels`` is populated exactly as the neuralset
    pooled ``prepare()`` loop populates it — one ``_update_channels`` per session,
    in cohort order, last writer wins. ``channel_order='original'`` matches the
    production dispatch (dispatch_v14.py), the config under which the desync
    occurred."""
    view = MultiStftView(
        event_types="Ieeg",
        car="shaft",
        notch_filter=60.0,
        apply_log=False,
        channel_order="original",
    )
    for sid in _COHORT:
        view._update_channels(list(orders[sid]))
    return view


def test_tracer_marble_three_way_alignment_real_cohort() -> None:
    """For each subject in a pooled heterogeneous cohort and each real row ``c``:
    the front-end tracer, the DK support, and the valid mask ALL describe
    ``voltage_electrode_order(subject)[c]``."""
    cache = _require_cache()
    orders = {s: list(voltage_electrode_order(cache, s)) for s in _COHORT}
    view = _pooled_view(orders)

    support_ext = V14DKHardSupportExtractor(
        bt_root=cache, c_max=_C_MAX, unmapped_policy="zero"
    )
    valid_ext = ElectrodeValidMask(
        bt_root=cache, c_max=_C_MAX, unmapped_policy="zero"
    )

    c_global_expected = max(len(orders[s]) for s in _COHORT)  # 238 = sub_7

    for sid in _COHORT:
        order = orders[sid]
        n_real = len(order)

        # --- front-end lane: tracer through the REAL pooled scatter -----------
        # Row i carries the marble (i + 1); +1 so a real row is never confusable
        # with the zero background/pad row (that confusion is exactly DP2's
        # zero-fed-as-valid). F=2, T=3 are arbitrary — the scatter is per-row.
        tracer = (
            (torch.arange(n_real, dtype=torch.float32) + 1.0)
            .reshape(n_real, 1, 1)
            .expand(n_real, 2, 3)
            .contiguous()
        )
        out = view._scatter_spec_to_global(tracer, list(order))

        # The global row count must cover the LONGEST session, not collapse to the
        # last (shortest) writer (out-of-bounds / lost-rows guard).
        assert out.shape[0] == c_global_expected, (
            f"sub_{sid}: C_global {out.shape[0]} != longest session "
            f"{c_global_expected} — row count tracks the wrong session"
        )

        # --- anatomy lanes: built independently, in voltage order ------------
        support = support_ext.get_static(SimpleNamespace(subject=str(sid)))
        valid = valid_ext.get_static(SimpleNamespace(subject=str(sid)))
        avs = aligned_voltage_support(cache, sid, unmapped_policy="zero")
        assert support.shape == (_C_MAX, len(avs.support[0]))
        assert valid.shape == (_C_MAX,)

        decoded = out[:n_real, 0, 0].round().long() - 1  # marble -> voltage pos

        # (1) Front-end lane keeps this subject's voltage order — no permutation
        #     across subjects (DP1) and no two electrodes on one row (DP2).
        np.testing.assert_array_equal(
            decoded.numpy(),
            np.arange(n_real),
            err_msg=f"sub_{sid}: electrode_tokens row != voltage position",
        )
        assert len(set(decoded.tolist())) == n_real, (
            f"sub_{sid}: within-subject row collision -> signal loss"
        )

        # (2) Anatomy lanes are in voltage order (extractor == aligned source,
        #     padded to c_max with zero/False).
        np.testing.assert_array_equal(
            support[:n_real].numpy(), avs.support, err_msg=f"sub_{sid}: support"
        )
        np.testing.assert_array_equal(
            valid[:n_real].numpy(), avs.valid, err_msg=f"sub_{sid}: valid"
        )

        # (3) Three-way tie + no zero-fed-as-valid: every electrode the front-end
        #     marks valid carries a real (non-background) marble, and where it is
        #     valid the support row is a live one-hot. All three then name the same
        #     physical electrode voltage_electrode_order(sid)[c].
        for c in range(n_real):
            if bool(valid[c]):
                assert out[c, 0, 0] != 0.0, (
                    f"sub_{sid} row {c}: valid but front-end row is background "
                    "zero (zero-fed-as-valid / lost electrode)"
                )
                assert support[c].sum() == 1.0, (
                    f"sub_{sid} row {c}: valid but support is not a live one-hot"
                )

        # --- padding rows are inert in every lane ---------------------------
        if n_real < _C_MAX:
            assert support[n_real:].sum() == 0.0, f"sub_{sid}: support pad nonzero"
            assert not valid[n_real:].any(), f"sub_{sid}: valid pad set"
        if n_real < c_global_expected:
            assert out[n_real:].sum() == 0.0, f"sub_{sid}: front-end pad nonzero"


def test_tracer_marble_detects_a_simulated_scatter_regression() -> None:
    """Meta-test: prove the tracer is actually sensitive. Hand-scatter the FIRST
    writer's rows with the BUGGY shared-dict global index (the pre-fix behaviour)
    and confirm the three-way check would have caught it — guards against a future
    refactor that silently makes the marble test a no-op.

    The victim under the bug is an EARLIER writer whose electrode names a later
    writer overwrites. ``_COHORT[0]`` (sub_7) shares 64 names with the next writer
    (sub_4), so its rows get mis-placed; sub_4 itself is clean here because the
    last writer (sub_9) shares no names with it."""
    cache = _require_cache()
    orders = {s: list(voltage_electrode_order(cache, s)) for s in _COHORT}

    # Reconstruct the buggy shared name->global-index dict (last writer wins).
    shared: dict[str, int] = {}
    for sid in _COHORT:
        for i, name in enumerate(orders[sid]):
            shared[name] = i  # neuro.py 'original': name-keyed, last writer wins
    c_global = max(shared.values()) + 1

    sid = _COHORT[0]  # first writer -> overwritten by later sessions
    order = orders[sid]
    n_real = len(order)
    buggy_idx = [shared[name] for name in order]  # where the OLD scatter sent rows

    # Under the bug, at least one of sub_4's rows lands on a global index owned by
    # sub_7/sub_9 (collision) — so decoded != voltage position for those rows.
    buggy_out = torch.zeros(c_global)
    marble = torch.arange(n_real, dtype=torch.float32) + 1.0
    buggy_out[torch.as_tensor(buggy_idx)] = marble  # last writer overwrites

    valid_ext = ElectrodeValidMask(
        bt_root=cache, c_max=_C_MAX, unmapped_policy="zero"
    )
    valid = valid_ext.get_static(SimpleNamespace(subject=str(sid)))

    # The contract check the e2e test runs: read row c, decode marble, expect c.
    mismatches = sum(
        1
        for c in range(n_real)
        if int(buggy_out[c].round()) - 1 != c
    )
    assert mismatches > 0, (
        "buggy shared-dict scatter should mis-place sub_4 rows but didn't — "
        "the tracer test would be a no-op"
    )
    # And the buggy scatter loses data: fewer distinct occupied rows than reals.
    assert len(set(buggy_idx)) < n_real, "expected collisions to drop rows"
    # A dropped row leaves a valid electrode reading the zero background.
    occupied = set(buggy_idx)
    assert any(
        bool(valid[c]) and c not in occupied for c in range(n_real)
    ), "expected at least one valid electrode left reading background zero"


# --------------------------------------------------------------------------- #
# L5 — property-based fuzz of the scatter alignment invariant.                 #
# Dependency-free (seeded RNG, many iterations) so it adds no `hypothesis` dep #
# and is fully deterministic. For EVERY random pooled cohort — random subject  #
# count, random per-session electrode counts, heavy cross-session name reuse,  #
# random write order — the per-subject scatter must place session row i at     #
# output row i (its own voltage order), never collide two of a session's own   #
# rows, and size the global row count to the LONGEST session.                  #
# --------------------------------------------------------------------------- #


def _random_cohort(rng, name_pool):
    """A random pooled cohort: 1–6 sessions, each a unique-within-session subset
    of a SHARED small name pool (so cross-session collisions are frequent — the
    exact condition that broke the old shared-dict scatter)."""
    n_sessions = int(rng.integers(1, 7))
    sessions = []
    for _ in range(n_sessions):
        size = int(rng.integers(1, min(40, len(name_pool)) + 1))
        names = list(rng.choice(name_pool, size=size, replace=False))
        sessions.append(names)
    return sessions


def test_fuzz_scatter_alignment_invariant() -> None:
    """Property: across many random pooled cohorts the fixed scatter keeps every
    session in its own per-session order with no row loss. Quantified: runs
    ITERS independent cohorts and asserts the invariant on each session of each."""
    name_pool = [f"E{i}" for i in range(60)]  # small pool -> frequent collisions
    ITERS = 300
    total_sessions = 0
    collisions_the_bug_would_cause = 0

    for seed in range(ITERS):
        rng = np.random.default_rng(seed)
        sessions = _random_cohort(rng, name_pool)

        view = MultiStftView(
            event_types="Ieeg",
            car="shaft",
            notch_filter=60.0,
            apply_log=False,
            channel_order="original",
        )
        for names in sessions:
            view._update_channels(list(names))

        c_global = max(view._channels.values()) + 1
        longest = max(len(s) for s in sessions)
        assert c_global >= longest, (
            f"seed {seed}: C_global {c_global} < longest session {longest} "
            "-> out-of-bounds scatter"
        )

        # Quantify how much damage the OLD shared-dict scatter would have done on
        # this cohort, so the fuzz also proves it is exercising the failure mode.
        shared: dict[str, int] = {}
        for names in sessions:
            for i, nm in enumerate(names):
                shared[nm] = i

        for names in sessions:
            total_sessions += 1
            n = len(names)
            # FIXED scatter: per-session voltage order, no permutation, no loss.
            idx = view._get_channels(list(names))
            assert idx == list(range(n)), f"seed {seed}: not per-session order"
            assert len(set(idx)) == n, f"seed {seed}: within-session row collision"

            tracer = (
                (torch.arange(n, dtype=torch.float32) + 1.0)
                .reshape(n, 1, 1)
                .expand(n, 2, 2)
                .contiguous()
            )
            out = view._scatter_spec_to_global(tracer, list(names))
            assert out.shape[0] == c_global
            decoded = out[:n, 0, 0].round().long() - 1
            np.testing.assert_array_equal(decoded.numpy(), np.arange(n))
            # padding rows inert
            if n < c_global:
                assert out[n:].sum() == 0.0

            # Would the buggy shared-dict scatter have mis-placed this session?
            buggy = [shared[nm] for nm in names]
            if buggy != list(range(n)) or len(set(buggy)) < n:
                collisions_the_bug_would_cause += 1

    assert total_sessions > ITERS, "fuzz did not generate multi-session cohorts"
    # Sensitivity floor: the fuzz must actually hit the failure mode on a real
    # fraction of sessions, else it is vacuously green.
    assert collisions_the_bug_would_cause > ITERS // 5, (
        f"fuzz exercised the desync on only {collisions_the_bug_would_cause} "
        f"sessions of {total_sessions} — not stressing the bug enough"
    )
