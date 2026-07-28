"""Parity + erasure tests for the CS identity-erasure arm.

The load-bearing one is ``test_std_arm_is_bit_identical_to_the_board``: if our std column does not
reproduce ``v3_board_readout._cs_cell`` exactly, the leace/std_target deltas are measured against
a baseline that is not the leaderboard's, and nothing downstream means anything.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import v3_board_readout as B
import v3_cs_leace as L

TASK = "onset"
TAPS = ("enc0", "enc12")


def _rec(seed, n, parcels, feat_dim=4, shift=0.0, with_split=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(parcels), feat_dim)).astype(np.float32) + shift
    y = np.asarray(rng.integers(0, 2, size=n), dtype=np.float64)
    x[:, 0, 0] += 1.5 * y
    rec = {
        "labels": {TASK: y},
        "present_parcels": np.asarray(parcels, dtype=np.int64),
        "feats": {t: {"raw": torch.from_numpy(x).to(torch.float16)} for t in TAPS},
    }
    if with_split:
        half = n // 2
        rec["cs_split"] = {TASK: {"val": np.arange(half), "test": np.arange(half, n)}}
    return rec


@pytest.fixture
def pair():
    anchor = _rec(0, 160, [3, 5, 9, 12])
    test = _rec(1, 120, [5, 9, 12, 20], shift=2.0, with_split=True)
    return anchor, test


def test_std_arm_is_bit_identical_to_the_board(pair):
    anchor, test = pair
    board = B._cs_cell(anchor, test, TASK, TAPS)["cells"]
    mine = L._cs_cell_arms(anchor, test, TASK, TAPS, None)["cells"]
    std_keys = [k for k in board if k.endswith("|std")]
    assert std_keys, "board produced no std column"
    for k in std_keys:
        assert k in mine, f"{k} missing from the erasure driver"
        assert mine[k]["test"] == board[k]["test"], f"{k} diverged from the leaderboard baseline"
        assert mine[k]["lam_mult"] == board[k]["lam_mult"], f"{k} selected a different lambda"


def test_std_target_arm_also_matches_the_board(pair):
    anchor, test = pair
    board = B._cs_cell(anchor, test, TASK, TAPS)["cells"]
    mine = L._cs_cell_arms(anchor, test, TASK, TAPS, None)["cells"]
    for k in [k for k in board if k.endswith("|std_target")]:
        assert mine[k]["test"] == board[k]["test"]


def test_every_arm_is_reported_for_every_tap(pair):
    anchor, test = pair
    cells = L._cs_cell_arms(anchor, test, TASK, TAPS, None)["cells"]
    for tap in TAPS:
        for arm in L.ARMS:
            assert f"{tap}|{arm}" in cells, f"missing {tap}|{arm}"


def test_erasure_drives_held_out_identity_to_chance(pair):
    anchor, test = pair
    checks = L._cs_cell_arms(anchor, test, TASK, TAPS, None)["checks"]
    for tap, c in checks.items():
        assert c["id_auc_before"] > 0.9, f"{tap}: domains must be separable pre-erasure"
        assert abs(c["id_auc_after"] - 0.5) < 0.10, f"{tap}: erasure did not transfer"


def test_exact_erasure_leaves_no_residual_covariance(pair):
    anchor, test = pair
    checks = L._cs_cell_arms(anchor, test, TASK, TAPS, None)["checks"]
    for c in checks.values():
        assert c["residual_cov"] < 1e-10


def test_var_removed_is_reported_against_its_floor(pair):
    anchor, test = pair
    checks = L._cs_cell_arms(anchor, test, TASK, TAPS, None)["checks"]
    for c in checks.values():
        assert c["var_removed"] > 0
        assert c["var_removed_floor"] > 0


def test_anchor_is_rejected_as_a_test_cell():
    assert B.CS_TRAIN_ANCHOR == (2, 4)
    assert B.CS_TRAIN_ANCHOR not in [c for c in B.LITE_SESSIONS if c[0] != 2]


def test_ten_cross_subject_cells_exist():
    """Partial cells lie -- .6279 at 4 cells became .5991 at 10. The shard array must cover all."""
    assert len([c for c in B.LITE_SESSIONS if c != B.CS_TRAIN_ANCHOR and c[0] != 2]) == 10


def test_main_loads_both_caches_lazily(pair, tmp_path, monkeypatch):
    """A board session cache is 40-64 GB and this driver opens TWO. The eager path OOM-killed all
    ten cells in the loader at 64G (job 20540001). Assert the mmap kwarg actually reaches _load --
    a constant alone would not catch someone dropping the kwarg at the call site."""
    anchor, test = pair
    seen = []

    def fake_load(cache_dir, session, tag, mmap=False):
        seen.append(mmap)
        return anchor if session == B.CS_TRAIN_ANCHOR else test

    monkeypatch.setattr(B, "_load", fake_load)
    out = tmp_path / "leace_S1T1.json"
    monkeypatch.setattr(sys, "argv", [
        "v3_cs_leace.py", "--cache", "/nonexistent", "--tag", "t",
        "--cell", "1,1", "--taps", ",".join(TAPS), "--out", str(out),
    ])
    L.main()

    assert seen == [True, True], f"_load called with mmap={seen}; both must be lazy"
    assert out.exists()


def test_partial_results_survive_a_kill_mid_run(pair, tmp_path, monkeypatch):
    """A cell takes hours at enc12 d=93184. Writing the JSON only after the task loop meant a
    walltime kill lost every completed task. Simulate dying on task 2 and require task 1 durable."""
    anchor, test = pair
    for rec in (anchor, test):
        rec["labels"] = {"onset": rec["labels"][TASK], "volume": rec["labels"][TASK]}
    test["cs_split"] = {k: test["cs_split"][TASK] for k in ("onset", "volume")}

    monkeypatch.setattr(B, "_load",
                        lambda c, s, t, mmap=False: anchor if s == B.CS_TRAIN_ANCHOR else test)
    real = L._cs_cell_arms
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) > 1:
            raise KeyboardInterrupt("simulated walltime kill")
        return real(*a, **k)

    monkeypatch.setattr(L, "_cs_cell_arms", flaky)
    out = tmp_path / "leace_S1T1.json"
    monkeypatch.setattr(sys, "argv", [
        "v3_cs_leace.py", "--cache", "/nonexistent", "--tag", "t",
        "--cell", "1,1", "--taps", ",".join(TAPS), "--out", str(out),
    ])
    with pytest.raises(KeyboardInterrupt):
        L.main()

    assert out.exists(), "first task was lost -- writes are still deferred to the end"
    assert len(json.loads(out.read_text())) == 1
    assert not out.with_suffix(".partial").exists(), "temp file left behind"


def test_sbatch_cell_list_matches_the_code():
    """The launcher hardcodes the cells for $SLURM_ARRAY_TASK_ID; drift would silently score the
    wrong subject against the wrong shard filename."""
    text = (Path(__file__).resolve().parent / "cs_leace.sbatch").read_text()
    line = next(ln for ln in text.splitlines() if ln.startswith("CELLS="))
    listed = [tuple(int(v) for v in c.split(",")) for c in line.split("(", 1)[1].rstrip(")").split()]
    assert listed == [c for c in B.LITE_SESSIONS if c[0] != B.CS_TRAIN_ANCHOR[0]]
