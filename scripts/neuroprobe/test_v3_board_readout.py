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


def test_select_lam_flags_lambda_pinned_to_a_grid_boundary() -> None:
    """A boundary argmax means the optimum is OUTSIDE the grid: the AUROC is a truncation
    artifact that looks exactly like a fit. It must reach the report."""
    assert _select_lam({"val": {LAM_MULTS[-1]: 0.9},
                        "test": {LAM_MULTS[-1]: 0.8}})["lam_pinned"] is True
    mid = LAM_MULTS[len(LAM_MULTS) // 2]
    assert _select_lam({"val": {mid: 0.9}, "test": {mid: 0.8}})["lam_pinned"] is False


def test_lambda_grid_brackets_the_diagnostics_pinned_lam_mult() -> None:
    """lam_mult=1.0 (the r4 diagnostic's fixed value) must be an interior grid point, so the
    board number is never worse than the diagnostic's for want of a λ."""
    assert min(LAM_MULTS) < 1.0 < max(LAM_MULTS)
    assert any(abs(m - 1.0) < 1e-9 for m in LAM_MULTS)


def test_ws_cell_reports_every_tap_x_norm_and_selects_only_lambda() -> None:
    """Ben 2026-07-17: λ is the ONLY val-selected axis. Every (tap, norm) must come back with
    its own complete number — no argmax over taps, no argmax over norms, nothing dropped."""
    rec = _rec()
    rec["feats"]["enc12_elec"] = rec["feats"]["enc12"]
    got = _ws_cell(rec, "onset", ("enc12_elec", "enc12", "enc0"))
    assert set(got["cells"]) == {f"{t}|{n}" for t in ("enc12_elec", "enc12", "enc0")
                                 for n in ("std", "raw")}
    assert got["cells"]["enc12|std"]["test"] == pytest.approx(1.0)
    assert got["cells"]["enc12_elec|raw"]["test"] == pytest.approx(1.0)


def test_ws_cell_skips_taps_absent_from_the_cache() -> None:
    """A cache encoded without --elec-taps must not NaN the whole session."""
    got = _ws_cell(_rec(), "onset", ("enc12_elec", "enc12"))
    assert set(got["cells"]) == {"enc12|std", "enc12|raw"}
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


def test_cs_reports_std_raw_and_std_target_as_three_separate_columns() -> None:
    """Ben 2026-07-17: per-domain std (AdaBN-style) is CS-only and a THIRD reported norm, on the
    same footing as std/raw. It is a different claim ("transfer GIVEN target statistics"), which
    is exactly why it must be a column of its own and never fused into the others by an argmax."""
    anchor, test = _rec(seed=1), _rec(seed=2)
    got = _cs_cell(anchor, test, "onset", ("enc12",))
    assert set(got["cells"]) == {"enc12|std", "enc12|raw", "enc12|std_target"}


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
