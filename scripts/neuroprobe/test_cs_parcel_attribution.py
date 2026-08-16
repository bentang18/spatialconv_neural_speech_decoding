"""Tests for the CS parcel-attribution fitter.

The load-bearing one is ``test_full_fit_reproduces_the_board``: the LOPO deltas and the weight
mass are both differences against the full-intersection fit, so if that fit is not the board's
fit, every number this script produces is measured against the wrong baseline.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import cs_parcel_attribution as A  # noqa: E402
import v3_board_readout as B  # noqa: E402

TASK = "onset"
# d = |P| * FEAT must exceed n so the fit takes the DUAL branch, as the real CS enc12 tap does
# (d = 7 parcels * 52 tokens * 384 = 139,776 against n_train <= 3,500). A fixture with d < n would
# exercise `_lam_grid_primal` and test a code path the board never runs here.
FEAT = 40
NARROW = 4          # enc0 fixture width -> d = |P| * 4 < n, the primal branch
CONTACTS = 3        # contacts per parcel in the fixture's parcel_canon


def _rec(seed, n, parcels, signal_parcel=None, shift=0.0, with_split=False):
    """A synthetic cache record. `signal_parcel` is a POSITION into `parcels`, and it is the only
    block whose features carry the label, so attribution has a known right answer."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(parcels), FEAT)).astype(np.float32) + shift
    y = np.asarray(rng.integers(0, 2, size=n), dtype=np.float64)
    if signal_parcel is not None:
        x[:, signal_parcel, :] += 3.0 * y[:, None]
    # enc0 is deliberately NARROW and enc12 WIDE, so the two taps land in opposite solver
    # branches exactly as they do on the real board (CS enc0 primal, CS enc12 dual).
    x0 = x[:, :, :NARROW].copy()
    rec = {
        "labels": {TASK: y},
        "present_parcels": np.asarray(parcels, dtype=np.int64),
        "parcel_canon": np.repeat(np.asarray(parcels, dtype=np.int64), CONTACTS),
        "feats": {"enc12": {"raw": torch.from_numpy(x).to(torch.float16)},
                  "enc0": {"raw": torch.from_numpy(x0).to(torch.float16)}},
    }
    if with_split:
        half = n // 2
        rec["cs_split"] = {TASK: {"val": np.arange(half), "test": np.arange(half, n)}}
    return rec


@pytest.fixture
def pair():
    # shared parcels are [5, 9, 12]; the signal sits on atlas id 9 in BOTH subjects, which is
    # position 2 in the anchor and position 1 in the test -- so a positional (rather than
    # atlas-id) alignment would put the mass on the wrong parcel and the test would catch it.
    anchor = _rec(0, 60, [3, 5, 9, 12], signal_parcel=2)
    test = _rec(1, 70, [5, 9, 12, 20], signal_parcel=1, shift=2.0, with_split=True)
    return anchor, test


def _pieces(anchor, test):
    a_idx, t_idx, common = A._cols(anchor, test)
    y_a = np.asarray(anchor["labels"][TASK], dtype=np.float64)
    y_t = np.asarray(test["labels"][TASK], dtype=np.float64)
    tr = B._finite(y_a, np.arange(len(y_a)))
    va = B._finite(y_t, test["cs_split"][TASK]["val"])
    te = B._finite(y_t, test["cs_split"][TASK]["test"])
    return a_idx, t_idx, common, tr, va, te, y_a, y_t


def test_intersection_is_atlas_aligned(pair):
    a_idx, t_idx, common = A._cols(*pair)
    assert common.tolist() == [5, 9, 12]
    assert a_idx == [1, 2, 3] and t_idx == [0, 1, 2]


def test_full_fit_reproduces_the_board(pair):
    anchor, test = pair
    ref = B._cs_cell(anchor, test, TASK, ("enc12",))["cells"]["enc12|std"]
    a_idx, t_idx, common, tr, va, te, y_a, y_t = _pieces(anchor, test)
    got, _, _, _ = A._fit(anchor, test, "enc12", tr, va, te, y_a, y_t, a_idx, t_idx, False)
    assert got["test"] == ref["test"]
    assert got["lam_mult"] == ref["lam_mult"]


def test_beta_reproduces_the_scores_in_both_branches(pair):
    """beta must score the eval set identically to the branch's own scoring path.

    This is the identity the whole weight readout rests on: if it does not hold, the vector being
    reshaped into parcels is not the decoder. Checked at BOTH taps because they take different
    branches -- enc12 recovers beta as Z_tr^T alpha from the dual, enc0 gets it straight out of
    the primal solve, and only one of those is a rearrangement that could be wrong.
    """
    anchor, test = pair
    a_idx, t_idx, common, tr, va, te, y_a, y_t = _pieces(anchor, test)
    for tap, want in (("enc12", "dual"), ("enc0", "primal")):
        z_tr = B._feat(anchor, tap, tr, a_idx)
        z_va = B._feat(test, tap, va, t_idx)
        z_te = B._feat(test, tap, te, t_idx)
        z_tr, (z_va, z_te) = B._standardize_inplace(z_tr, [z_va, z_te])
        evals = {"val": (z_va, y_t[va]), "test": (z_te, y_t[te])}
        grid, beta_at, branch = A._fit_grid(z_tr, y_a[tr], evals)
        assert branch == want, f"{tap} took the {branch} branch, expected {want}"
        m = B._select_lam(grid)["lam_mult"]
        s = np.asarray(z_te, dtype=np.float64) @ beta_at(m)
        assert B.auroc(s, y_t[te]) == pytest.approx(grid["test"][m], abs=1e-12)


def test_both_taps_reproduce_the_board(pair):
    """The gate, at both taps. enc0 exercises the primal branch, which is the one the board takes
    for CS enc0 and the one a naive dual would silently get almost-but-not-quite right."""
    anchor, test = pair
    a_idx, t_idx, common, tr, va, te, y_a, y_t = _pieces(anchor, test)
    board = B._cs_cell(anchor, test, TASK, ("enc0", "enc12"))["cells"]
    for tap in ("enc0", "enc12"):
        ref = board[f"{tap}|std"]
        got, _, branch, _ = A._fit(anchor, test, tap, tr, va, te, y_a, y_t,
                                   a_idx, t_idx, False)
        assert got["test"] == ref["test"], f"{tap} ({branch}) missed the board"
        assert got["lam_mult"] == ref["lam_mult"], f"{tap} ({branch}) selected a different lambda"


def test_weight_mass_and_lopo_both_find_the_signal_parcel(pair):
    """The planted parcel must win on BOTH readings. They are different statements -- mass is
    descriptive and LOPO is causal -- so agreeing on a case with a known answer is the check that
    neither the reshape nor the column-dropping is off by a block."""
    anchor, test = pair
    a_idx, t_idx, common, tr, va, te, y_a, y_t = _pieces(anchor, test)
    full, beta, _, _ = A._fit(anchor, test, "enc12", tr, va, te, y_a, y_t, a_idx, t_idx, True)
    assert beta.shape == (len(common),)
    assert common[int(np.argmax(beta))] == 9

    drops = {}
    for j, p in enumerate(common):
        keep = [k for k in range(len(common)) if k != j]
        s, _, _, _ = A._fit(anchor, test, "enc12", tr, va, te, y_a, y_t,
                            [a_idx[k] for k in keep], [t_idx[k] for k in keep], False)
        drops[int(p)] = full["test"] - s["test"]
    assert max(drops, key=lambda p: drops[p]) == 9
    assert drops[9] > 0.05, f"dropping the signal parcel barely moved AUROC: {drops}"


def test_contacts_per_parcel_is_the_size_control(pair):
    """Every LOPO row carries contacts_removed; if that count is wrong the size control is too."""
    anchor, test = pair
    _, _, common = A._cols(anchor, test)
    assert A._contacts(test, common) == {5: CONTACTS, 9: CONTACTS, 12: CONTACTS}
