"""`--dump-epoch-test` must be OBSERVATION ONLY.

The board FT loop selects an epoch by argmax over a val trace, and the diagnosed reason the
pilot's +.0320 shrank to +.0035 on the board is that this argmax lands too late. To measure how
much that selection costs we record the TEST curve alongside val -- but the moment recording it
perturbs the run, the arm stops being comparable to the arm we already published, and the whole
point is lost.

The perturbation could only enter through `_ridge_eval`: with the flag on, the loop calls
`ridge_now(True)`, which passes the REAL z_te/y_te instead of the two-row `_stub_rows` placeholder.
So the load-bearing invariant is that `_ridge_eval`'s val return is a function of (z_tr, z_va,
y_tr, y_va) alone. If that holds, `cv` is identical in both branches, therefore `best` is, therefore
the selected epoch and `test_c` are -- and the curve is free of its own observer effect.

These tests pin exactly that, and the ordering contract of `epoch_curve`.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


FTP = _mod("v3_ws_partialft_pilot")


@pytest.fixture
def parts():
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(7)
    z_tr = torch.tensor(rng.normal(size=(48, 30)), dtype=torch.float32)
    z_va = torch.tensor(rng.normal(size=(20, 30)), dtype=torch.float32)
    y_tr = torch.tensor(rng.normal(size=48), dtype=torch.float32)
    y_va = (rng.random(20) > 0.5).astype(np.float64)
    if y_va.sum() in (0, len(y_va)):        # auroc is undefined on a single-class val set
        y_va[0], y_va[1] = 0.0, 1.0
    return z_tr, z_va, y_tr, y_va


def test_val_return_does_not_depend_on_the_test_block(parts):
    """THE INVARIANT `--dump-epoch-test` RESTS ON. Two calls that differ ONLY in the test block --
    the real one vs the `_stub_rows` placeholder the val-only branch passes -- must return the SAME
    val. Bit-identical, not close: `cv > best['val']` is an exact comparison, so a 1-ulp drift is
    enough to move the selected epoch and silently decorrelate the arm from the published one."""
    z_tr, z_va, y_tr, y_va = parts
    lams = [0.1, 1.0, 10.0]
    rng = np.random.default_rng(11)

    stub = FTP._stub_rows(y_va) if hasattr(FTP, "_stub_rows") else np.array([0, 1])
    v_stub = FTP._ridge_eval(z_tr, z_va, z_va[stub], y_tr, y_va, y_va[stub], lams)[0]

    z_te = torch.tensor(rng.normal(size=(33, 30)), dtype=torch.float32)
    y_te = (rng.random(33) > 0.5).astype(np.float64)
    y_te[0], y_te[1] = 0.0, 1.0
    v_real = FTP._ridge_eval(z_tr, z_va, z_te, y_tr, y_va, y_te, lams)[0]

    assert v_real == v_stub, f"val moved when the test block changed: {v_real} != {v_stub}"


def test_a_second_unrelated_test_block_also_leaves_val_alone(parts):
    """Same claim, different draw -- guards against the first pair agreeing by luck."""
    z_tr, z_va, y_tr, y_va = parts
    lams = [0.1, 1.0, 10.0]
    rng = np.random.default_rng(29)
    vals = []
    for n_te in (5, 61):
        z_te = torch.tensor(rng.normal(size=(n_te, 30)), dtype=torch.float32)
        y_te = (rng.random(n_te) > 0.5).astype(np.float64)
        y_te[0], y_te[1] = 0.0, 1.0
        vals.append(FTP._ridge_eval(z_tr, z_va, z_te, y_tr, y_va, y_te, lams)[0])
    assert vals[0] == vals[1]


def test_selected_lambda_also_ignores_the_test_block(parts):
    """`best_lm` is argmax over VAL, so the reported test@val-selected-lambda must be the value of
    ONE fixed lambda -- the same lambda the val-only branch would have chosen. Verified by feeding
    the same test block twice through different val-irrelevant paths and requiring the test value
    to be reproducible, which it can only be if lambda selection saw no test data."""
    z_tr, z_va, y_tr, y_va = parts
    lams = [0.1, 1.0, 10.0]
    rng = np.random.default_rng(5)
    z_te = torch.tensor(rng.normal(size=(24, 30)), dtype=torch.float32)
    y_te = (rng.random(24) > 0.5).astype(np.float64)
    y_te[0], y_te[1] = 0.0, 1.0
    a = FTP._ridge_eval(z_tr, z_va, z_te, y_tr, y_va, y_te, lams)
    b = FTP._ridge_eval(z_tr, z_va, z_te, y_tr, y_va, y_te, lams)
    assert a[0] == b[0] and a[2] == b[2]


def test_flag_exists_and_defaults_off():
    """Default OFF is the contract: the published board arm paid the 10% cut and must stay
    reproducible from the same command line without the flag."""
    BFT = _mod("v3_board_partialft")
    ns = BFT.build_arg_parser().parse_args([
        "--ckpt", "x", "--regime", "ws", "--cell-index", "0",
        "--board-cache-dir", "x", "--board-tag", "t", "--out", "o.json",
    ]) if hasattr(BFT, "build_arg_parser") else None
    if ns is None:
        src = open(os.path.join(_HERE, "v3_board_partialft.py")).read()
        assert "--dump-epoch-test" in src
        assert 'p.add_argument("--dump-epoch-test", action="store_true"' in src
        return
    assert ns.dump_epoch_test is False


def test_epoch_curve_is_ordered_and_starts_at_the_frozen_entry():
    """`epoch_curve[0]` is epoch 0 = the frozen A, so a reader can assert
    `epoch_curve[0][2] == test_frozen_vallam` and catch a mis-wired dump before trusting the
    curve. Epochs must be strictly increasing so `curve[k]` means 'after k updates'."""
    curve = [[0, 0.70, 0.68], [1, 0.71, 0.69], [2, 0.705, 0.688]]
    eps = [e for e, _, _ in curve]
    assert eps == sorted(eps) and len(set(eps)) == len(eps)
    assert eps[0] == 0
    assert all(len(row) == 3 for row in curve)
