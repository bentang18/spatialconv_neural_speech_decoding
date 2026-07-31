"""v3_board_readout — the board protocol's contract (TDD).

The failure this guards against is invisible in the output: a readout that selects on the test
half, or reports a val number as test, prints numbers that look exactly like correct ones. So
the tests pin the contract itself — which rows are fit on, which are selected on, which are
reported — on synthetic caches where the right answer is known by construction.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.neuroprobe.v3_board_readout import (
    BOARD_TASKS,
    CS_TEST_CELLS,
    CS_TRAIN_ANCHOR,
    LAM_MULTS,
    LITE_SESSIONS,
    _absorb,
    _blank,
    auroc,
    _cs_cell,
    _finalize,
    _lam_grid,
    _select_lam,
    _ws_cell,
)


def _rec(n=64, n_parcels=4, feat=8, *, parcels=None, signal=True, seed=0):
    """A synthetic session cache in the encode's payload format.

    Labels alternate 0/1; when `signal` the feature's first column IS the label, so a correct
    readout must score AUROC 1.0 on the test half and a broken one cannot.
    """
    rng = np.random.default_rng(seed)
    y = np.array([float(i % 2) for i in range(n)])
    x = rng.normal(size=(n, n_parcels, feat)).astype(np.float32)
    if signal:
        x[:, 0, 0] = y * 10.0
    half = n // 2
    ws_split = {
        t: {
            0: {"train": np.arange(half), "val": np.arange(half, half + half // 2),
                "test": np.arange(half + half // 2, n)},
            1: {"train": np.arange(half, n), "val": np.arange(0, half // 2),
                "test": np.arange(half // 2, half)},
        }
        for t in BOARD_TASKS
    }
    cs_split = {t: {"val": np.arange(0, half), "test": np.arange(half, n)} for t in BOARD_TASKS}
    return {
        "feats": {"enc12": {"raw": torch.from_numpy(x).to(torch.float16)},
                  "enc0": {"raw": torch.from_numpy(x).to(torch.float16)}},
        "present_parcels": np.asarray(
            parcels if parcels is not None else np.arange(n_parcels), dtype=np.int64),
        "labels": {t: y.copy() for t in BOARD_TASKS},
        "ws_split": ws_split,
        "cs_split": cs_split,
    }


def test_board_constants_match_upstream_neuroprobe() -> None:
    """The eval universe IS the claim — pin it against the installed upstream package."""
    import importlib.util
    import os

    os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp")
    path = ".venv/lib/python3.12/site-packages/neuroprobe/config.py"
    spec = importlib.util.spec_from_file_location("npcfg", path)
    if spec is None or spec.loader is None:
        pytest.skip("upstream neuroprobe not installed")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    assert BOARD_TASKS == tuple(m.NEUROPROBE_TASKS)
    assert LITE_SESSIONS == tuple(tuple(x) for x in m.NEUROPROBE_LITE_SUBJECT_TRIALS)
    assert CS_TRAIN_ANCHOR == (m.DS_DM_TRAIN_SUBJECT_ID, m.DS_DM_TRAIN_TRIAL_ID)
    # Upstream asserts test_subject != anchor subject; the 10 cells are every other Lite cell.
    assert CS_TEST_CELLS == tuple(c for c in LITE_SESSIONS if c[0] != CS_TRAIN_ANCHOR[0])


def test_lam_grid_scores_every_lambda_on_every_eval_set() -> None:
    rng = np.random.default_rng(0)
    y = np.array([float(i % 2) for i in range(40)])
    z = rng.normal(size=(40, 6))
    z[:, 0] = y * 5.0
    out = _lam_grid(z[:20], y[:20], {"val": (z[20:30], y[20:30]), "test": (z[30:], y[30:])})
    assert set(out) == {"val", "test"}
    from scripts.neuroprobe.v3_board_readout import LAM_MULTS

    assert set(out["val"]) == set(LAM_MULTS)
    # A feature that IS the label separates perfectly at some lambda.
    assert max(out["test"].values()) == pytest.approx(1.0)


def test_select_lam_takes_argmax_val_and_reports_that_lambdas_test() -> None:
    """The core contract: the winner is chosen by VAL, but the number returned is its TEST."""
    got = _select_lam({"val": {1.0: 0.60, 10.0: 0.90}, "test": {1.0: 0.99, 10.0: 0.55}})
    assert got["lam_mult"] == 10.0          # even though lam=1 has the better TEST
    assert got["test"] == 0.55
    assert got["val"] == 0.90


def test_select_lam_never_peeks_at_test_to_break_ties() -> None:
    """Equal val ⇒ first-seen wins; the better TEST must not be able to pull the choice."""
    got = _select_lam({"val": {1.0: 0.80, 10.0: 0.80}, "test": {1.0: 0.10, 10.0: 0.99}})
    assert got["test"] == 0.10


def test_select_lam_all_nan_val_is_nan_not_a_default_lambda() -> None:
    got = _select_lam({"val": {1.0: float("nan")}, "test": {1.0: 0.99}})
    assert np.isnan(got["test"]) and np.isnan(got["lam_mult"])


def test_select_lam_flags_only_the_LO_boundary_as_truncation() -> None:
    """The two boundaries are different failures, and only LO invalidates a fit.

    HI is benign: AUROC saturates as λ→∞ (the smoother w/(w+λ)→0 uniformly, so the ranking — and
    hence the AUROC — converges; see test_auroc_saturates_at_high_lambda). "Maximal shrinkage
    won" is a faithful answer, not an artifact, and widening the grid cannot change it.
    LO has no such limit: λ→0 keeps moving, so a LO argmax really is truncated.
    """
    hi = _select_lam({"val": {LAM_MULTS[-1]: 0.9}, "test": {LAM_MULTS[-1]: 0.8}})
    assert hi["lam_pin"] == "hi" and hi["lam_pinned"] is False

    lo = _select_lam({"val": {LAM_MULTS[0]: 0.9}, "test": {LAM_MULTS[0]: 0.8}})
    assert lo["lam_pin"] == "lo" and lo["lam_pinned"] is True

    mid = LAM_MULTS[len(LAM_MULTS) // 2]
    got = _select_lam({"val": {mid: 0.9}, "test": {mid: 0.8}})
    assert got["lam_pin"] == "" and got["lam_pinned"] is False


def test_auroc_saturates_at_high_lambda() -> None:
    """The measurement the LO/HI split rests on: past some λ the AUROC stops moving entirely, so
    a HI pin cannot be hiding a better score further out.

    As λ→∞, α = V diag(1/(w+λ)) Vᵀ y → (1/λ)·y, so the scores → (1/λ)·K@y — a POSITIVE rescale of
    K@y, which AUROC is invariant to. So the top of the grid already IS the limit.
    """
    rng = np.random.default_rng(0)
    z_tr = rng.normal(size=(60, 8))
    y_tr = (z_tr[:, 0] + 0.3 * rng.normal(size=60) > 0).astype(float)
    z_te = rng.normal(size=(40, 8))
    y_te = (z_te[:, 0] > 0).astype(float)

    g = _lam_grid(z_tr, y_tr, {"test": (z_te, y_te)})
    limit = auroc(np.asarray(z_te @ z_tr.T) @ y_tr, y_te)
    assert g["test"][LAM_MULTS[-1]] == limit


def test_lambda_grid_brackets_the_diagnostics_pinned_lam_mult() -> None:
    """lam_mult=1.0 (the r4 diagnostic's fixed value) must be an interior grid point, so the
    board number is never worse than the diagnostic's for want of a λ."""
    assert min(LAM_MULTS) < 1.0 < max(LAM_MULTS)
    assert any(abs(m - 1.0) < 1e-9 for m in LAM_MULTS)


def test_ws_cell_reports_every_tap_x_norm_and_selects_only_lambda(monkeypatch) -> None:
    """Ben 2026-07-17: λ is the ONLY val-selected axis. Every (tap, norm) must come back with
    its own complete number — no argmax over taps, no argmax over norms, nothing dropped. (Opt in
    both norm columns via PROBE_NORMS; that norm coverage is exactly what this test guards.)"""
    import scripts.neuroprobe.v3_board_readout as mod
    monkeypatch.setattr(mod, "NORMS", ("std", "raw"))
    rec = _rec()
    rec["feats"]["enc12_elec"] = rec["feats"]["enc12"]
    got = mod._ws_cell(rec, "onset", ("enc12_elec", "enc12", "enc0"))
    assert set(got["cells"]) == {f"{t}|{n}" for t in ("enc12_elec", "enc12", "enc0")
                                 for n in ("std", "raw")}
    assert got["cells"]["enc12|std"]["test"] == pytest.approx(1.0)
    assert got["cells"]["enc12_elec|raw"]["test"] == pytest.approx(1.0)


def test_ws_cell_skips_taps_absent_from_the_cache() -> None:
    """A cache encoded without --elec-taps must not NaN the whole session. (Default norm set is
    std-only as of 2026-07-18; the point here is the absent tap is skipped, not the norm axis.)"""
    got = _ws_cell(_rec(), "onset", ("enc12_elec", "enc12"))
    assert set(got["cells"]) == {"enc12|std"}
    assert got["cells"]["enc12|std"]["test"] == pytest.approx(1.0)


def test_ws_cell_on_pure_noise_is_chance_not_one() -> None:
    """Guards the mirror failure: a readout that leaks labels scores ~1 on noise."""
    got = _ws_cell(_rec(signal=False, seed=7), "onset", ("enc12",))
    assert 0.15 < got["cells"]["enc12|std"]["test"] < 0.85


def test_cs_cell_transfers_over_the_parcel_intersection() -> None:
    anchor = _rec(parcels=[0, 1, 2, 3], seed=1)
    test = _rec(parcels=[2, 3, 4, 5], seed=2)
    got = _cs_cell(anchor, test, "onset", ("enc12",))
    # Intersection {2,3} — the signal lives in parcel 0, which is NOT shared, so the transfer
    # must NOT be perfect. The point is that it scored at all, over 2 aligned parcels.
    assert got["n_parcels"] == 2
    assert not np.isnan(got["cells"]["enc12|std"]["test"])


def test_cs_cell_aligns_parcel_columns_by_atlas_id_not_position() -> None:
    """The trap: same |P| on both sides, different atlas ids ⇒ positional alignment silently
    pairs unrelated regions. Put the signal in a shared parcel sitting at DIFFERENT positions."""
    anchor = _rec(parcels=[7, 1, 2, 3], seed=1)   # atlas 7 at position 0
    test = _rec(parcels=[1, 2, 3, 7], seed=1)     # atlas 7 at position 3
    # Move the signal into atlas-7's column on each side: anchor pos 0, test pos 3.
    for rec, pos in ((anchor, 0), (test, 3)):
        x = rec["feats"]["enc12"]["raw"].to(torch.float32).numpy()
        x[:, :, 0] = 0.0
        x[:, pos, 0] = rec["labels"]["onset"] * 10.0
        rec["feats"]["enc12"]["raw"] = torch.from_numpy(x).to(torch.float16)
    got = _cs_cell(anchor, test, "onset", ("enc12",))
    assert got["n_parcels"] == 4
    # Aligned by atlas id, the shared signal transfers; positionally it would be noise.
    assert got["cells"]["enc12|std"]["test"] == pytest.approx(1.0)


def test_cs_reports_all_three_norm_columns_when_opted_in(monkeypatch) -> None:
    """Ben 2026-07-17: per-domain std (AdaBN-style) is CS-only and a THIRD reported norm, on the
    same footing as std/raw. It is a different claim ("transfer GIVEN target statistics"), which
    is exactly why it must be a column of its own and never fused into the others by an argmax.
    Monkeypatch the module norm globals on — when opted in, all three columns are still separate."""
    import scripts.neuroprobe.v3_board_readout as mod
    monkeypatch.setattr(mod, "NORMS", ("std", "raw"))
    monkeypatch.setattr(mod, "REPORT_STD_TARGET", True)
    anchor, test = _rec(seed=1), _rec(seed=2)
    got = mod._cs_cell(anchor, test, "onset", ("enc12",))
    assert set(got["cells"]) == {"enc12|std", "enc12|raw", "enc12|std_target"}


def test_cs_defaults_to_std_only_column(monkeypatch) -> None:
    """Default as of 2026-07-20 (Ben): std-only, so the readout is ~2x faster (std and raw are
    separate ridge solves per tap and raw is retired). std_target is gated by REPORT_STD_TARGET."""
    import scripts.neuroprobe.v3_board_readout as mod
    monkeypatch.setattr(mod, "NORMS", ("std",))
    monkeypatch.setattr(mod, "REPORT_STD_TARGET", False)
    anchor, test = _rec(seed=1), _rec(seed=2)
    got = mod._cs_cell(anchor, test, "onset", ("enc12",))
    assert set(got["cells"]) == {"enc12|std"}


def test_norms_default_is_std_only() -> None:
    """Ben 2026-07-20: raw is retired, so the module norm default is std-only — the single column
    the leaderboard reports. std_target stays available as a CS-only extra, gated by
    REPORT_STD_TARGET / --no-std-target, and is never fused into std by an argmax."""
    import scripts.neuroprobe.v3_board_readout as mod
    assert mod.NORMS == ("std",)
    assert "raw" not in mod.NORMS
    assert not hasattr(mod, "_norm_config")  # env-var opt-in mechanism removed


def test_per_domain_standardize_uses_val_stats_only_never_test() -> None:
    """The legality condition: the target scaler is fit on the VAL half only. If test rows
    could move mu/sd, the ablation would be transductive on the reported half."""
    from scripts.neuroprobe.v3_board_readout import _standardize_per_domain

    rng = np.random.default_rng(0)
    z_tr = rng.normal(size=(20, 4))
    z_va = rng.normal(loc=5.0, scale=2.0, size=(20, 4))
    z_te = rng.normal(loc=5.0, scale=2.0, size=(20, 4))
    _, (b1, c1) = _standardize_per_domain(z_tr, z_va, z_te)
    # Perturbing TEST must not change how VAL was scaled, nor the scaler applied to test.
    _, (b2, c2) = _standardize_per_domain(z_tr, z_va, z_te * 100.0 + 7.0)
    assert np.allclose(b1, b2)
    assert np.allclose(c2, (z_te * 100.0 + 7.0 - z_va.mean(0)) / z_va.std(0))
    # The val half IS centred by its own stats — that is the point of the ablation.
    assert np.allclose(b1.mean(axis=0), 0.0, atol=1e-9)


def test_cs_cell_no_shared_parcels_is_empty_not_a_number() -> None:
    anchor = _rec(parcels=[0, 1], n_parcels=2, seed=1)
    test = _rec(parcels=[8, 9], n_parcels=2, seed=2)
    assert _cs_cell(anchor, test, "onset", ("enc12",))["cells"] == {}


def test_merge_keeps_every_grid_entry_over_every_cell() -> None:
    """The merge must not collapse the grid either: each "tap|norm" carries its own per-cell
    dict, and the cohort mean is taken WITHIN an entry — never across taps or norms."""
    res = _blank(["tg"])
    for name, v in (("S1T1", 0.6), ("S3T0", 0.8)):
        _absorb(res, {"kind": "cs", "name": name, "cells": {
            "tg|onset": {"cells": {"enc12|std": {"test": v, "lam_pinned": False},
                                   "enc12|raw": {"test": v - 0.1, "lam_pinned": False},
                                   "enc0|std": {"test": v - 0.2, "lam_pinned": False}},
                         "n_parcels": 5}}})
    _finalize(res)
    c = res["tg|onset"]
    assert set(c["cs_mean"]) == {"enc12|std", "enc12|raw", "enc0|std"}
    assert c["cs_mean"]["enc12|std"] == pytest.approx(0.7)      # (0.6+0.8)/2, within the entry
    assert c["cs_mean"]["enc0|std"] == pytest.approx(0.5)
    assert set(c["cs"]["enc12|std"]) == {"S1T1", "S3T0"}        # per-cell detail survives


def test_merge_carries_the_lambda_pin_flag_to_the_report() -> None:
    res = _blank(["tg"])
    _absorb(res, {"kind": "ws", "name": "S2T0", "cells": {
        "tg|onset": {"cells": {"enc12|std": {"test": 0.9, "lam_pinned": True},
                               "enc12|raw": {"test": 0.9, "lam_pinned": False}}}}})
    assert res["tg|onset"]["pinned"] == {"ws:enc12|std": ["S2T0"]}


def test_map_tasks_forked_gives_identical_results_to_serial() -> None:
    """Ben 2026-07-17: fork-over-tasks is a THROUGHPUT change and must be a NUMERICAL no-op.
    The failure it guards against is silent — a forked worker that mis-shares state returns
    plausible numbers, not an error. So: same input, both paths, byte-identical output."""
    from scripts.neuroprobe.v3_board_readout import _map_tasks

    rec = _rec(seed=3)
    fn = lambda task, tp: _ws_cell(rec, task, tp)      # a lambda: fork inherits, never pickles
    serial = _map_tasks(fn, ("enc12",), workers=1)
    forked = _map_tasks(fn, ("enc12",), workers=2)
    assert set(serial) == set(forked) == set(BOARD_TASKS)
    for t in BOARD_TASKS:
        for gk in serial[t]["cells"]:
            a, b = serial[t]["cells"][gk]["test"], forked[t]["cells"][gk]["test"]
            assert a == b or (np.isnan(a) and np.isnan(b)), f"{t}|{gk}: {a} != {b}"


def test_load_mmap_flag_reaches_torch_load() -> None:
    """Pin that the flag is threaded through to torch.load rather than silently defaulting."""
    import scripts.neuroprobe.v3_board_readout as B

    seen = {}
    orig = B.torch.load
    B.torch.load = lambda path, **kw: seen.update(kw) or {"ok": True}
    try:
        B._load("/cache", (2, 4), "tg", mmap=True)
        assert seen["mmap"] is True
        B._load("/cache", (2, 4), "tg")
        assert seen["mmap"] is False          # eager is the safe default; modes opt in
    finally:
        B.torch.load = orig


def test_mmap_is_never_the_default() -> None:
    """mmap DEFERS the read into the gathers at ~1/4 sequential bandwidth (cold 24 MB/s vs eager
    ~86 MB/s), so it is only ever worth it to save MEMORY.

    Memory is not scarce here, and believing it was is what made this default wrong: the shards
    that stalled were starved by NUMA node-0 pinning, not by size, and `numactl --interleave=all`
    is the fix. Under mmap a shard measured 15 MB/s with 3.7M major faults and produced nothing in
    43 minutes. Eager for both modes; the --mmap/--no-mmap knob stays for A/B only."""
    from scripts.neuroprobe.v3_board_readout import MMAP_DEFAULT

    assert MMAP_DEFAULT == {"ws": False, "cs": False, "csession": False}


# ── R8: standardize column-blocking + primal collapse ──────────────────────────────
# Both are pure SPEED changes to phases the shard timer said dominate (standardize was 81% of an
# eager WS shard). Neither may move a board number, so each is pinned against the path it replaces.


@pytest.mark.parametrize("blk", [2, 3, 7, 64, 1024, 10_000])
def test_standardize_column_blocking_is_bit_identical_at_every_block_size(blk) -> None:
    """The reduction is per COLUMN, so the block size is a memory-layout choice and NOTHING else.

    Pinned bitwise, not approximately: this function feeds every reported AUROC, and the whole
    argument for blocking is that it cannot change an answer. Spans blk larger than d (one block
    = the old whole-array path) down to the blk=2 floor.

    blk=1 is EXCLUDED and that is why _STD_BLOCK has a floor: a width-1 column slice makes
    np.mean/np.std reduce a contiguous 1-D vector, which numpy sums PAIRWISE, instead of
    accumulating a SIMD row-vector across columns. Measured drift 1.2e-7 on mu — harmless in size,
    but it is a different summation order, so the bitwise guarantee would become a claim about
    numpy's dispatch rather than about arithmetic. Every width >= 2 sums the same values in the
    same order (verified 2..4096 on d=4096, and 512..8192 on d=120000 on Delta).
    """
    import scripts.neuroprobe.v3_board_readout as B

    assert B._STD_BLOCK >= 2
    rng = np.random.default_rng(4)
    tr = rng.normal(size=(29, 53)).astype(np.float32)
    va = rng.normal(size=(11, 53)).astype(np.float32)
    te = rng.normal(size=(13, 53)).astype(np.float32)
    tr[:, 5] = 2.5                                     # a constant column: sd == 0 -> 1.0 branch
    ref = B._standardize(tr.copy(), [va.copy(), te.copy()])
    got_tr, (got_va, got_te) = B._standardize_inplace(tr.copy(), [va.copy(), te.copy()], blk=blk)
    assert np.array_equal(got_tr, ref[0])
    assert np.array_equal(got_va, ref[1][0])
    assert np.array_equal(got_te, ref[1][1])


def test_primal_matches_dual_on_the_same_fit() -> None:
    """d < n takes the primal; zero-padding d past n forces the dual on the IDENTICAL problem.

    Zero columns contribute nothing to ZZᵀ, to ZᵀZ's trace, or to any eval kernel, so the padded
    fit is the same ridge — only the factorization differs. Agreement to <1e-4 AUROC is the whole
    licence for the branch.
    """
    rng = np.random.default_rng(5)
    n, d = 90, 20
    y = np.array([float(i % 2) for i in range(n)])
    z = rng.normal(size=(n, d)).astype(np.float32)
    z[:, 0] += y * 1.5
    ev = {"val": (z[50:70], y[50:70]), "test": (z[70:], y[70:])}
    primal = _lam_grid(z[:50], y[:50], ev)
    pad = lambda a: np.hstack([a, np.zeros((a.shape[0], 60), np.float32)])   # noqa: E731
    dual = _lam_grid(pad(z[:50]), y[:50],
                     {"val": (pad(z[50:70]), y[50:70]), "test": (pad(z[70:]), y[70:])})
    for split in ("val", "test"):
        for m in LAM_MULTS:
            assert primal[split][m] == pytest.approx(dual[split][m], abs=1e-4)


def test_primal_branch_is_taken_only_when_d_is_below_n() -> None:
    """The dual builds eval kernels; the primal has none. Phase keys are the observable."""
    import scripts.neuroprobe.v3_board_readout as B

    rng = np.random.default_rng(6)
    y = np.array([float(i % 2) for i in range(40)])
    z = rng.normal(size=(40, 6)).astype(np.float32)
    ev = {"val": (z[20:30], y[20:30]), "test": (z[30:], y[30:])}

    B._PH.clear()
    _lam_grid(z[:20], y[:20], ev)                       # d=6 < n=20 -> primal
    assert "eval_kernels" not in B._PH

    wide = np.hstack([z, rng.normal(size=(40, 60))]).astype(np.float32)
    B._PH.clear()
    _lam_grid(wide[:20], y[:20],
              {"val": (wide[20:30], y[20:30]), "test": (wide[30:], y[30:])})   # d=66 > n=20
    assert "eval_kernels" in B._PH
    B._PH.clear()


def test_lam_grid_is_the_published_grid_and_any_widening_must_keep_its_spacing() -> None:
    """The grid that produced every published board number, plus the contract for changing it.

    LO-pinning is asymmetric across taps (ws enc12 39 vs enc0 16), which is why widening keeps
    coming up. Two invariants make a widening safe when it happens, so pin them now:

      spacing — extend at the SAME 1/3-decade step and the old points survive exactly, so a cell
                that did not pin re-selects the same λ and reproduces its old test AUROC. Drift
                the spacing and every board number moves silently.
      HI end  — must stay at 1e4. AUROC is constant in λ past it (next test), so widening up
                cannot change a number and only buys a 34-shard board re-run.
    """
    from scripts.neuroprobe.v3_board_readout import LAM_MULTS

    steps = np.diff(np.log10(np.asarray(LAM_MULTS)))
    print(f"[check] grid: {len(LAM_MULTS)} points, "
          f"{np.log10(min(LAM_MULTS)):.0f}..{np.log10(max(LAM_MULTS)):.0f} decades, "
          f"step={steps.mean():.4f} dec (uniform={np.ptp(steps) < 1e-9}) OK")
    assert np.ptp(steps) < 1e-9, "spacing is not uniform"
    assert abs(steps.mean() - 1.0 / 3.0) < 1e-9, \
        "spacing left 1/3 decade — a widening at this step no longer contains the old grid"
    assert max(LAM_MULTS) == pytest.approx(1e4), "the HI end must not move — it provably saturates"


def test_widening_only_extends_downward_because_the_hi_end_saturates(monkeypatch) -> None:
    """Why the fix is one-sided: AUROC is constant in λ past the HI end, so widening up is a no-op.

    _select_lam's docstring asserts AUROC(1e4) == AUROC(1e16). Pin it so nobody 'symmetrically'
    widens the top and pays for a 34-shard board that cannot change a number.
    """
    import scripts.neuroprobe.v3_board_readout as mod

    rng = np.random.default_rng(0)
    y = np.array([float(i % 2) for i in range(40)])
    z = rng.normal(size=(40, 6))
    z[:, 0] = y * 5.0
    monkeypatch.setattr(mod, "LAM_MULTS", (1e4, 1e8, 1e16))
    out = mod._lam_grid(z[:20], y[:20], {"test": (z[20:], y[20:])})
    vals = [out["test"][m] for m in (1e4, 1e8, 1e16)]
    print(f"[check] test AUROC at lam_mult 1e4/1e8/1e16 = {vals} "
          f"(want all equal — the HI end saturates, widening UP cannot move a number) OK")
    assert vals[0] == vals[1] == vals[2], vals


def test_val_ties_make_the_selected_lambda_depend_on_the_GRID_not_the_DATA() -> None:
    """🚨 The confound in the LO-pin audit: a 'lo pin' is not necessarily a truncated optimum.

    _select_lam takes argmax with a STRICT `>` while iterating LAM_MULTS in ascending order, so on
    a val TIE it keeps the SMALLEST λ. The val half is coarse (AUROC over a few dozen rows), so
    ties are the common case, not the exception — on this fixture most of the grid ties at val=1.0
    and the tied cells' TEST AUROCs differ materially. Two consequences:

      1. A cell flagged ``lam_pinned`` may simply be a TIE resolved to the grid floor, NOT an
         optimum that sits below the grid. Only the second is truncation.
      2. Widening the grid downward moves the tie-break, so it changes numbers in cells where the
         data expressed no preference at all — in either direction.

    So the pin counts alone cannot license "our depth gains are lower bounds". This test pins the
    mechanism so the inference is not made from the flag again.
    """
    from scripts.neuroprobe.v3_board_readout import LAM_MULTS, _select_lam

    tied = {m: 1.0 for m in LAM_MULTS}                     # val: dead flat, no preference
    test = {m: 0.90 + 0.05 * (i / len(LAM_MULTS)) for i, m in enumerate(LAM_MULTS)}
    got = _select_lam({"val": tied, "test": test})
    print(f"[check] val tied across all {len(LAM_MULTS)} λ -> selected λ={got['lam_mult']:.3e} "
          f"(== grid min {min(LAM_MULTS):.3e}), flagged lam_pinned={got['lam_pinned']} "
          f"while the data preferred NOTHING OK")
    assert got["lam_mult"] == min(LAM_MULTS), "tie-break is not smallest-λ; audit logic changed"
    assert got["lam_pinned"] is True, "a pure tie is reported as a LO pin — that is the confound"


def test_widening_the_grid_can_LOWER_a_pinned_cell_not_raise_it(monkeypatch) -> None:
    """🚨 THE COUNTEREXAMPLE. Resolving a LO pin does NOT mean the AUROC goes up.

    The tempting inference from the pin audit (ws enc12 pins lo 39× vs enc0 16×) is that pinned
    cells are truncated downward, so widening the grid raises them and the reported depth gains
    are lower bounds. On this repo's OWN ws fixture that is false, and it fails in the unlucky
    direction. Widening logspace(-4,4,25) → logspace(-8,4,37) on the enc12|std cell:

        grid       fold λ            lam_pinned   test (mean of the 2 folds)
        old        1e-4,   1e-4      True          1.0000000
        new        1e-6, 4.64e-7     False         0.9765625   ← pin RESOLVED, number FELL

    Why: val is dead flat at 1.0 across 14/25 (fold 0) and 25/25 (fold 1) of the old grid, while
    test over that same tied plateau spans 0.906..1.000. Val cannot discriminate, so _select_lam's
    smallest-λ tie-break lets the GRID FLOOR pick the operating point. The old floor happened to
    land on a good point; the new floor lands on a worse one. Nothing was truncated — the cell was
    under-determined, and both floors are equally justified by the val half.

    So a LO pin is not evidence of truncation, and widening is not a free improvement. Keep this
    test as the standing refutation before anyone spends a 34-shard board re-run on that premise.
    """
    import scripts.neuroprobe.v3_board_readout as mod

    monkeypatch.setattr(mod, "NORMS", ("std",))
    out = {}
    for name, grid in (("old", tuple(np.logspace(-4.0, 4.0, 25))),
                       ("new", tuple(np.logspace(-8.0, 4.0, 37)))):
        monkeypatch.setattr(mod, "LAM_MULTS", grid)
        out[name] = mod._ws_cell(_rec(), "onset", ("enc12",))["cells"]["enc12|std"]
        print(f"[check] {name} grid floor={min(grid):.0e} -> test={out[name]['test']:.7f} "
              f"lam/fold={[f'{m:.2e}' for m in out[name]['lam_mult']]} "
              f"lam_pinned={out[name]['lam_pinned']}")

    assert out["old"]["lam_pinned"] is True, "fixture no longer pins lo on the published grid"
    assert out["new"]["lam_pinned"] is False, "widening should have resolved the pin"
    assert out["new"]["test"] < out["old"]["test"], \
        "widening no longer LOWERS this cell — re-derive the claim before trusting a re-run"
    print(f"[check] pin resolved True->False while test FELL "
          f"{out['old']['test']:.7f} -> {out['new']['test']:.7f} "
          f"=> a LO pin is NOT evidence of downward truncation OK")


# ── depth concatenation (`cat:` virtual taps) ──────────────────────────────────────────────
# Layer combination is the one readout axis the board grid never explored: every reported number
# comes from ONE tap. A ridge over concatenated depths is the unconstrained form of SUPERB's
# learned layer-weighted sum, and it stays a linear probe on frozen features fit per task, so it
# spends no protocol parity. These pin that the concatenation is what it claims to be — the parts
# side by side, in order, indexed consistently — and that the unit-mixing footgun is refused.


def test_cat_feat_is_exactly_the_parts_hstacked_in_order() -> None:
    """The whole mechanism: `cat:a+b` must be `hstack([a, b])`, same rows, same order. If the
    parts were gathered with different row sets the ridge would silently regress mismatched
    trials against each other and still return a plausible AUROC."""
    import scripts.neuroprobe.v3_board_readout as mod
    rec = _rec(seed=3)
    rows = np.array([1, 5, 9, 2])
    a = mod._feat(rec, "enc0", rows)
    b = mod._feat(rec, "enc12", rows)
    cat = mod._feat(rec, "cat:enc0+enc12", rows)
    assert cat.shape == (len(rows), a.shape[1] + b.shape[1])
    np.testing.assert_array_equal(cat[:, :a.shape[1]], a)
    np.testing.assert_array_equal(cat[:, a.shape[1]:], b)


def test_cat_feat_applies_the_same_col_idx_to_every_part() -> None:
    """`col_idx` is the parcel (cs) or electrode (csession) intersection. It must select the SAME
    columns from every part -- a part indexed differently would align one depth's parcel 7 with
    another's parcel 3."""
    import scripts.neuroprobe.v3_board_readout as mod
    rec = _rec(seed=4)
    rows, cols = np.array([0, 3, 6]), np.array([1, 3])
    a = mod._feat(rec, "enc0", rows, cols)
    b = mod._feat(rec, "enc12", rows, cols)
    cat = mod._feat(rec, "cat:enc0+enc12", rows, cols)
    np.testing.assert_array_equal(cat, np.hstack([a, b]))


def test_cat_tap_scores_and_lands_under_its_own_grid_key() -> None:
    """A cat tap must be a first-class reported cell, not a fusion of existing ones: its own
    `tap|norm` key, alongside the singles, so the grid still reports one complete protocol per
    entry and nothing is picked by an argmax across taps."""
    rec = _rec()
    got = _ws_cell(rec, "onset", ("enc12", "cat:enc0+enc12"))
    assert set(got["cells"]) == {"enc12|std", "cat:enc0+enc12|std"}
    assert got["cells"]["cat:enc0+enc12|std"]["test"] == pytest.approx(1.0)


def test_cat_tap_is_skipped_when_any_part_is_missing() -> None:
    """Partial availability must skip the whole cat, not silently score a narrower feature block
    under the same name -- that would make one grid key mean two different feature sets across
    cells, which is the partial-cell defect in a new disguise."""
    import scripts.neuroprobe.v3_board_readout as mod
    rec = _rec()
    assert mod._have(rec, "cat:enc0+enc12")
    assert not mod._have(rec, "cat:enc0+enc6")
    got = _ws_cell(rec, "onset", ("cat:enc0+enc6", "enc12"))
    assert set(got["cells"]) == {"enc12|std"}


def test_cat_cs_cell_transfers_over_the_parcel_intersection() -> None:
    """The cs path indexes by the anchor-test parcel intersection; a cat must survive it."""
    anchor = _rec(parcels=[0, 1, 2, 3], seed=1)
    test = _rec(parcels=[2, 3, 4, 5], seed=2)
    got = _cs_cell(anchor, test, "onset", ("cat:enc0+enc12",))
    assert got["n_parcels"] == 2
    assert not np.isnan(got["cells"]["cat:enc0+enc12|std"]["test"])


@pytest.mark.parametrize("bad", ["cat:enc12_elec+enc6", "cat:enc12", "cat:enc0+enc0",
                                 "cat:enc0+nope", "bogus"])
def test_validate_taps_refuses_malformed_and_unit_mixing_cats(bad) -> None:
    """Refused at PARSE time. The unit-mixing case is the dangerous one: it would hstack an
    electrode block onto a parcel block and index both with one col_idx, gathering wrong columns
    from one of them -- a wrong number, not a crash."""
    import scripts.neuroprobe.v3_board_readout as mod
    with pytest.raises(SystemExit):
        mod._validate_taps((bad,))


def test_validate_taps_accepts_singles_and_well_formed_cats() -> None:
    import scripts.neuroprobe.v3_board_readout as mod
    mod._validate_taps(("enc12", "cat:enc6+enc9+enc12", "cat:enc0_elec+enc12_elec"))


def test_is_elec_routes_cats_by_their_parts_not_vacuously() -> None:
    """`all()` over an ordinary tap's empty parts tuple is vacuously True, which would route
    every parcel tap down the electrode branch of _csession_cell."""
    import scripts.neuroprobe.v3_board_readout as mod
    assert mod._is_elec("enc12_elec") and not mod._is_elec("enc12")
    assert mod._is_elec("cat:enc0_elec+enc12_elec")
    assert not mod._is_elec("cat:enc6+enc12")


# ── --lam-rule: the val-TIE convention ────────────────────────────────────────────────────────
# The published board resolves a val tie to the SMALLEST λ purely because _select_lam iterated an
# ascending tuple with a strict `>`. That is an artifact of tuple order, not a decision. `tiemax`
# makes the opposite, a-priori-defensible choice. Both read val ONLY, so the switch cannot be a
# selection-on-test move -- these tests pin exactly that, and that the DEFAULT never moves.

def test_lam_rule_defaults_to_the_published_argmax() -> None:
    """Every board number to date is argmax. If this default ever flips, past runs stop being
    reproducible from the same command line -- which is the failure that made the K>1 driver
    silently change meaning underneath a command that looked identical."""
    from scripts.neuroprobe.v3_board_readout import LAM_RULE
    assert LAM_RULE == "argmax"


def test_tiemax_takes_the_largest_tied_lambda_argmax_the_smallest() -> None:
    d = {"val": {1.0: 0.80, 10.0: 0.80, 100.0: 0.80}, "test": {1.0: 0.10, 10.0: 0.50, 100.0: 0.99}}
    assert _select_lam(d, rule="argmax")["lam_mult"] == 1.0
    assert _select_lam(d, rule="tiemax")["lam_mult"] == 100.0


def test_tiemax_does_not_override_a_genuine_val_winner() -> None:
    """tiemax is a TIE-break, not a preference for large λ. A strictly better val must still win
    even when a larger λ is available."""
    d = {"val": {1.0: 0.90, 10.0: 0.80, 100.0: 0.80}, "test": {1.0: 0.1, 10.0: 0.5, 100.0: 0.9}}
    for rule in ("argmax", "tiemax"):
        assert _select_lam(d, rule=rule)["lam_mult"] == 1.0


def test_both_rules_agree_when_there_are_no_ties() -> None:
    d = {"val": {1.0: 0.70, 10.0: 0.90, 100.0: 0.60}, "test": {1.0: 0.1, 10.0: 0.5, 100.0: 0.9}}
    a, t = _select_lam(d, rule="argmax"), _select_lam(d, rule="tiemax")
    assert a["lam_mult"] == t["lam_mult"] == 10.0 and a["test"] == t["test"]


def test_neither_rule_can_see_test() -> None:
    """THE LOAD-BEARING INVARIANT. Hold val fixed, vary test arbitrarily: the selected λ must not
    move under EITHER rule. If it did, --lam-rule would be a test-selection knob, not a
    convention, and no number produced under it would be submittable."""
    val = {1.0: 0.80, 10.0: 0.80, 100.0: 0.75}
    for rule in ("argmax", "tiemax"):
        picks = {_select_lam({"val": val, "test": t}, rule=rule)["lam_mult"]
                 for t in ({1.0: 0.9, 10.0: 0.1, 100.0: 0.2},
                           {1.0: 0.1, 10.0: 0.9, 100.0: 0.2},
                           {1.0: 0.5, 10.0: 0.5, 100.0: 0.5})}
        assert len(picks) == 1, f"{rule} moved with test: {picks}"


def test_tiemax_resolves_the_pure_tie_AWAY_from_the_lo_pin() -> None:
    """The confound in test_val_ties_make_the_selected_lambda_depend_on_the_GRID: a dead-flat val
    is reported as lam_pinned='lo' under argmax, which reads as truncation when nothing was
    truncated. Under tiemax the same grid pins HI, which the docstring calls BENIGN (AUROC
    saturates as λ→∞). Same data, opposite flag -- so the flag describes the CONVENTION, not the
    cell, and neither can be cited as evidence about truncation."""
    from scripts.neuroprobe.v3_board_readout import LAM_MULTS
    d = {"val": {m: 1.0 for m in LAM_MULTS}, "test": {m: 0.9 for m in LAM_MULTS}}
    assert _select_lam(d, rule="argmax")["lam_pinned"] is True
    hi = _select_lam(d, rule="tiemax")
    assert hi["lam_pinned"] is False and hi["lam_pin"] == "hi"
    assert hi["n_tied"] == len(LAM_MULTS)


def test_n_tied_counts_only_the_val_maximisers() -> None:
    d = {"val": {1.0: 0.80, 10.0: 0.80, 100.0: 0.70}, "test": {1.0: 0.1, 10.0: 0.2, 100.0: 0.3}}
    assert _select_lam(d)["n_tied"] == 2


def test_nan_val_still_returns_the_degenerate_record_with_n_tied_zero() -> None:
    got = _select_lam({"val": {1.0: float("nan")}, "test": {1.0: 0.99}}, rule="tiemax")
    assert np.isnan(got["test"]) and got["n_tied"] == 0


# ── gpool:/bpool: time pooling ────────────────────────────────────────────────────────────────
# A unit's cached block is k_full time tokens of width d, flattened. WS therefore fits 119*13312
# = 1.58M features on ~1750 rows. These prefixes collapse the TIME axis (the MAE convention) --
# the risk is entirely in the reshape, since a mean over the wrong axis still yields a
# well-shaped array and a plausible wrong number.

def _pool_mod():
    import scripts.neuroprobe.v3_board_readout as B
    return B


def test_pool_spec_splits_prefix_from_base_tap() -> None:
    from scripts.neuroprobe.v3_board_readout import _pool_spec
    assert _pool_spec("gpool:enc12_elec") == ("gpool:", "enc12_elec")
    assert _pool_spec("bpool:enc12") == ("bpool:", "enc12")
    assert _pool_spec("enc12") == ("", "enc12")


def test_pooled_tap_keeps_its_base_taps_unit() -> None:
    """Pooling collapses the FEATURE axis, never the unit axis, so routing must not change."""
    from scripts.neuroprobe.v3_board_readout import _is_elec
    assert _is_elec("gpool:enc12_elec") is True
    assert _is_elec("bpool:enc12") is False


def test_gpool_is_the_mean_over_every_token() -> None:
    B = _pool_mod()
    B.POOL_D, B.POOL_BANDS, B._POOL_ANNOUNCED = 2, (1, 1, 2), set()
    x = np.arange(1 * 2 * 8, dtype=np.float32).reshape(1, 2, 8)     # 8 = 4 tokens x d 2
    got = B._pool_feats(x, "gpool:", "t")
    assert got.shape == (1, 2, 2)
    exp = x.reshape(1, 2, 4, 2).mean(axis=2)
    assert np.allclose(got, exp)


def test_bpool_means_WITHIN_bands_and_keeps_them_separate() -> None:
    """The load-bearing difference from gpool. Tokens are [SLOW;MID;HGA] with 8:1 HGA:SLOW counts
    at 1 s, so a global mean is dominated by HGA; bpool must keep one mean PER BAND, in order."""
    B = _pool_mod()
    B.POOL_D, B.POOL_BANDS, B._POOL_ANNOUNCED = 1, (1, 2, 4), set()
    x = np.arange(7, dtype=np.float32).reshape(1, 1, 7)             # 7 tokens x d 1
    got = B._pool_feats(x, "bpool:", "t")
    assert got.shape == (1, 1, 3)
    assert np.allclose(got[0, 0], [0.0, (1 + 2) / 2, (3 + 4 + 5 + 6) / 4])


def test_bpool_is_NOT_gpool_when_bands_are_unbalanced() -> None:
    """If these ever coincide the band split is not doing anything and bpool is dead weight."""
    B = _pool_mod()
    B.POOL_D, B.POOL_BANDS, B._POOL_ANNOUNCED = 1, (1, 2, 4), set()
    x = np.arange(7, dtype=np.float32).reshape(1, 1, 7)
    assert not np.allclose(B._pool_feats(x, "bpool:", "t").mean(),
                           B._pool_feats(x, "gpool:", "t")[0, 0, 0])


def test_wrong_pool_d_is_refused_not_silently_reshaped() -> None:
    B = _pool_mod()
    B.POOL_D, B.POOL_BANDS, B._POOL_ANNOUNCED = 5, (1, 1, 1), set()
    with pytest.raises(SystemExit, match="does not divide"):
        B._pool_feats(np.zeros((1, 1, 8), dtype=np.float32), "gpool:", "t")


def test_band_counts_that_do_not_sum_to_k_are_refused() -> None:
    """A band split describing a DIFFERENT cache (e.g. a 2 s clip) must fail loudly -- it would
    otherwise average across band boundaries and report a plausible wrong number."""
    B = _pool_mod()
    B.POOL_D, B.POOL_BANDS, B._POOL_ANNOUNCED = 2, (1, 1, 1), set()
    with pytest.raises(SystemExit, match="does not describe this cache"):
        B._pool_feats(np.zeros((1, 1, 8), dtype=np.float32), "bpool:", "t")   # k=4, bands sum 3


def test_pool_cannot_be_combined_with_cat() -> None:
    from scripts.neuroprobe.v3_board_readout import _validate_taps
    with pytest.raises(SystemExit, match="both rewrite the feature axis"):
        _validate_taps(["gpool:cat:enc9+enc12"])


def test_validator_accepts_pooled_taps_and_still_refuses_unknown_bases() -> None:
    from scripts.neuroprobe.v3_board_readout import _validate_taps
    _validate_taps(["gpool:enc12_elec", "bpool:enc12", "enc12"])
    with pytest.raises(SystemExit, match="unknown tap"):
        _validate_taps(["bpool:nope"])


def test_pool_defaults_match_the_lite_board_layout() -> None:
    """52 = 4+16+32 tokens at d 256 == the 13312 width the board cache actually stores, so the
    shipped defaults describe the Lite board and a run that forgets the flags is still correct."""
    assert sum((4, 16, 32)) * 256 == 13312


# ── --dump-lam-grid: observation only ───────────────────────────────────────────────────────

def test_lam_grid_is_absent_by_default() -> None:
    """The published shard shape must not grow a key just because the code learned to dump one."""
    import scripts.neuroprobe.v3_board_readout as B
    assert B.LAM_GRID_DUMP is False
    got = _select_lam({"val": {1.0: 0.60, 10.0: 0.90}, "test": {1.0: 0.99, 10.0: 0.55}})
    assert "lam_grid" not in got


def test_dump_lam_grid_does_not_move_the_selected_point(monkeypatch) -> None:
    """THE INVARIANT. Dumping is a MEASUREMENT of how much val-argmax-λ leaves on the table; the
    moment it perturbs the selection, the arm stops being the one we published. Every reported
    field must be identical with the dump on and off -- and the test curve it records is exactly
    the one the selection was NOT allowed to look at."""
    import scripts.neuroprobe.v3_board_readout as B
    d = {"val": {m: v for m, v in zip(LAM_MULTS, np.linspace(0.5, 0.9, len(LAM_MULTS)))},
         "test": {m: v for m, v in zip(LAM_MULTS, np.linspace(0.9, 0.4, len(LAM_MULTS)))}}
    off = _select_lam(d)
    monkeypatch.setattr(B, "LAM_GRID_DUMP", True)
    on = _select_lam(d)
    for k in off:
        assert on[k] == off[k], f"{k} moved when the grid dump was enabled"


def test_dumped_grid_is_the_whole_ascending_grid_and_contains_the_selected_point(monkeypatch) -> None:
    import scripts.neuroprobe.v3_board_readout as B
    monkeypatch.setattr(B, "LAM_GRID_DUMP", True)
    d = {"val": {m: v for m, v in zip(LAM_MULTS, np.linspace(0.5, 0.9, len(LAM_MULTS)))},
         "test": {m: v for m, v in zip(LAM_MULTS, np.linspace(0.9, 0.4, len(LAM_MULTS)))}}
    got = _select_lam(d)
    grid = got["lam_grid"]
    assert len(grid) == len(LAM_MULTS)
    assert [r[0] for r in grid] == sorted(r[0] for r in grid), "grid must be ascending in λ"
    hit = [r for r in grid if r[0] == got["lam_mult"]]
    assert len(hit) == 1 and hit[0][1] == got["val"] and hit[0][2] == got["test"]


def test_the_dumped_test_curve_can_beat_the_selected_point_which_is_the_whole_point(monkeypatch) -> None:
    """A ceiling only exists if the grid can hold a better test value than the val pick. Pinning
    this stops the dump from being quietly useless (e.g. if it recorded val twice)."""
    import scripts.neuroprobe.v3_board_readout as B
    monkeypatch.setattr(B, "LAM_GRID_DUMP", True)
    d = {"val": {1.0: 0.90, 10.0: 0.80}, "test": {1.0: 0.55, 10.0: 0.99}}
    got = _select_lam(d)
    assert got["test"] == 0.55
    assert max(r[2] for r in got["lam_grid"]) == 0.99
