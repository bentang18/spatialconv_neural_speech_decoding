"""The WS partial-FT pilot re-implements three readout primitives on GPU tensors so the
per-epoch val selection is affordable. If any of them drifts from the frozen readout's own
numpy version, the FT arm and the frozen arm stop being comparable and the whole pilot is
meaningless. These tests pin them to the originals.

The fourth is the parcel pool: the pilot needs a differentiable twin of ``_pool_parcels``
(which casts to fp16 and so is useless for a backward), and it must agree once the cast is
reapplied — that cast is what makes the reported features numerically identical to the cache.
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


FT = _mod("v3_ws_partialft_pilot")
RDO = FT.RDO


@pytest.fixture
def data():
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(0)
    z_tr = rng.normal(size=(60, 40)).astype(np.float32)
    z_te = rng.normal(size=(25, 40)).astype(np.float32)
    y_tr = rng.normal(size=60)
    return z_tr, z_te, y_tr


def test_gpu_ridge_matches_readout_numpy(data):
    """The constant-lambda path must reproduce ``dual_ridge_scores`` — same G, same trace rule,
    same fp32-GEMM/fp64-solve split."""
    z_tr, z_te, y_tr = data
    want = RDO.dual_ridge_scores(z_tr, y_tr, z_te, lam_mult=RDO.CONST_LAM_MULT)
    got = FT._gpu_ridge(torch.from_numpy(z_tr), torch.from_numpy(y_tr), torch.from_numpy(z_te))
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_eigh_lambda_sweep_agrees_with_direct_solve(data):
    """One eigendecomposition serving many lambdas must equal solving each separately —
    otherwise the val-lambda control (the thing FT has to beat) is measuring nothing real."""
    z_tr, z_te, y_tr = data
    lams = [0.1, 1.0, 10.0]
    swept = FT._gpu_ridge(torch.from_numpy(z_tr), torch.from_numpy(y_tr),
                          torch.from_numpy(z_te), lams=lams)
    for lm in lams:
        direct = RDO.dual_ridge_scores(z_tr, y_tr, z_te, lam_mult=lm)
        assert np.allclose(swept[lm], direct, rtol=1e-6, atol=1e-8), lm


def test_ridge_eval_const_lambda_matches_readout(data):
    """``_ridge_eval`` is the path that produces EVERY reported number (val selection, the FT
    test score, and the frozen epoch-0 control). Its constant-lambda output must equal the
    readout's ``_ridge_test`` — one Gram + eigh instead of two solves must change nothing."""
    z_tr, z_te, y_tr = data
    rng = np.random.default_rng(7)
    z_va = rng.normal(size=(20, 40)).astype(np.float32)
    y_va = np.sign(rng.normal(size=20))
    y_te = np.sign(rng.normal(size=25))
    val, test, test_vl = FT._ridge_eval(
        torch.from_numpy(z_tr), torch.from_numpy(z_va), torch.from_numpy(z_te),
        torch.from_numpy(y_tr), y_va, y_te, [0.1, 1.0, 10.0])
    assert val == pytest.approx(RDO._ridge_test(z_tr, y_tr, z_va, y_va, "std"), abs=1e-9)
    assert test == pytest.approx(RDO._ridge_test(z_tr, y_tr, z_te, y_te, "std"), abs=1e-9)
    assert np.isfinite(test_vl)


def test_standardize_matches_readout(data):
    """Train-only mean/std, sigma=0 -> 1. numpy ``.std`` is population (ddof=0), so the torch
    twin must pass unbiased=False or every feature is rescaled by sqrt(n/(n-1))."""
    z_tr, z_te, _ = data
    z_tr = z_tr.copy()
    z_tr[:, 3] = 2.5                                    # a constant column: the sigma=0 branch
    a, b = RDO._standardize(z_tr.astype(np.float64), z_te.astype(np.float64))
    ga, gb = FT._std_gpu(torch.from_numpy(z_tr).double(), torch.from_numpy(z_te).double())
    assert np.allclose(ga.numpy(), a, atol=1e-12)
    assert np.allclose(gb.numpy(), b, atol=1e-12)
    assert np.allclose(ga.numpy()[:, 3], 0.0)


def test_pool_t_matches_pool_parcels_after_fp16_cast():
    """``_pool_t`` is the differentiable twin; with the cache's fp16 cast reapplied it must be
    bit-identical to ``_pool_parcels``, including parcel ORDER."""
    enc = _mod("v3_probe_encode_r4")
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.normal(size=(5, 9, 4, 6)).astype(np.float32))
    parcel_canon = np.array([7, 7, 3, 3, 3, 11, 7, 11, 3])
    present = np.unique(parcel_canon)
    want = enc._pool_parcels(x, parcel_canon, present)
    cols = [torch.as_tensor(np.where(parcel_canon == q)[0]) for q in present]
    got = FT._pool_t(x, cols).to(torch.float16)
    assert got.shape == want.shape == (5, 3, 24)
    assert torch.equal(got, want)


def test_pool_t_is_differentiable():
    x = torch.randn(2, 4, 3, 5, requires_grad=True)
    cols = [torch.as_tensor(np.array([0, 1])), torch.as_tensor(np.array([2, 3]))]
    FT._pool_t(x, cols).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_flat16_reproduces_cache_read_path():
    """The reported features must go fp16 -> fp32, exactly as ``RDO._feat`` reads the cache."""
    z = torch.randn(7, 3, 8)
    got = FT._flat16(z)
    want = z.reshape(7, -1).to(torch.float16).to(torch.float32)
    assert got.shape == (7, 24) and torch.equal(got, want)


@pytest.mark.parametrize("k,n,p", [(20, 28, 0.0357), (21, 28, 0.0126), (14, 28, 1.0)])
def test_sign_test_null_is_the_one_preregistered(k, n, p):
    """The decision map quotes 20/28 -> p=.036. If this drifts, the gate moves silently."""
    assert FT._sign_p(k, n) == pytest.approx(p, abs=5e-4)


def test_arm_params_are_nested_and_ordered_by_size():
    """norm-only must be a strict subset of the full block, and strictly smaller — the ladder's
    whole point is parameter count against ~3.6k training windows."""
    towers = __import__("speech_decoding.models.v14_converged_v3.towers",
                        fromlist=["build_encoder"])
    enc = towers.build_encoder(n_parcels=64)
    n_norm = sum(p.numel() for p in FT._arm_params(enc, "norm"))
    n_mlp = sum(p.numel() for p in FT._arm_params(enc, "mlp"))
    n_full = sum(p.numel() for p in FT._arm_params(enc, "block12"))
    assert 0 < n_norm < n_mlp < n_full
    ids_norm = {id(p) for p in FT._arm_params(enc, "norm")}
    assert ids_norm <= {id(p) for p in FT._arm_params(enc, "block12")}
    print(f"\narm params: norm={n_norm} mlp={n_mlp} block12={n_full}")


def test_every_ENC_symbol_the_pilot_uses_actually_exists():
    """2778967 died on the GPU with ``module 'v3_probe_encode_r4' has no attribute
    'make_bt_parcel_fn'`` — the encode SCRIPT imports that name inside its own main() (:526-532),
    so it is not a module attribute. Nothing in the suite touched main(), so a name error reached
    the queue. This walks the source for every ``ENC.<attr>`` and resolves it."""
    import re
    src = open(os.path.join(_HERE, "v3_ws_partialft_pilot.py")).read()
    used = sorted(set(re.findall(r"(?<![\w.])ENC\.([A-Za-z_]\w*)", src)))
    assert used, "regex found no ENC.<attr> — the guard would pass vacuously"
    missing = [a for a in used if not hasattr(FT.ENC, a)]
    assert not missing, f"v3_probe_encode_r4 has no {missing}"


def test_package_symbols_the_pilot_imports_in_main_resolve():
    """The three names that replaced the bad ENC lookups. They are imported lazily inside main()
    (heavy), so only a test reaches them before a GPU node does."""
    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn  # noqa: F401
    from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid  # noqa: F401
    from speech_decoding.models.v14_converged_v3.session_loader import (  # noqa: F401
        load_v3_sessions,
    )


def test_head_is_trainable_and_scores_like_a_readout():
    """The head must produce a per-window score AUROC can consume, and gradient must reach it —
    it IS cell D, the reported readout, not just a gradient source for block 12."""
    torch.manual_seed(0)
    head = FT._Head(12)
    z = torch.randn(20, 12)
    out = head(z)
    assert out.shape == (20,)
    out.sum().backward()
    assert head.fc.weight.grad is not None
    y = np.sign(np.random.default_rng(0).normal(size=20))
    assert 0.0 <= RDO.auroc(out.detach().numpy(), y) <= 1.0


def test_run_tail_equals_the_towers_own_forward():
    """THE LOAD-BEARING ONE. Running only blocks 12..end on a cached tap-11 must reproduce the
    tower's tap-12 exactly. This is the same invariant the pilot's parity-L0 gate enforces on
    real data; here it is pinned on a tiny random tower so a refactor of ``_run_flat`` breaks a
    test instead of silently invalidating a pilot."""
    towers = __import__("speech_decoding.models.v14_converged_v3.towers",
                        fromlist=["build_encoder"])
    torch.manual_seed(0)
    enc = towers.build_encoder(n_parcels=8).eval()
    n_blocks = len(enc.blocks)
    split, tap = n_blocks - 1, n_blocks

    class _G:
        pass

    B, n_contacts, T = 2, 6, 3
    M = n_contacts * T
    g = _G()
    g.total = M
    g.depth = torch.arange(n_contacts).repeat_interleave(T)
    g.time_pos = torch.arange(T).repeat(n_contacts)
    g.cu_seqlens = torch.tensor([0, M], dtype=torch.int32)
    g.max_seqlen = M
    d = enc.blocks[0].mlp.fc2.out_features
    x = torch.randn(B, M, d)
    parcel_packed = torch.randint(0, 8, (M,))

    with torch.no_grad():
        _out, taps = enc.forward_flat(x, g, parcel_packed, tap_blocks=(split, tap))
        ctx = FT._flat_ctx(enc, g, B, torch.device("cpu"))
        FT.SPLIT_AT, FT.TAP = split, tap
        got = FT._run_tail(enc, taps[split], ctx)
    assert torch.allclose(got, taps[tap], atol=1e-6), \
        f"max|d|={(got - taps[tap]).abs().max():.3e} — the block split is NOT equivalent"


def test_the_tap_cache_is_never_downcast():
    """Job 2779110_0 failed parity-L0 at rel 5.371e-3 because the tap-11 cache was cast to
    bf16 on the false belief that bf16 is the autocast residual dtype. It is not — LayerNorm
    autocasts to fp32 and the pre-norm residual add promotes the branch back up
    (attention.py:132-136), so the tap is fp32 and the cast cost exactly one ulp. The cache
    must carry ``taps[SPLIT_AT].dtype`` verbatim."""
    import re
    src = open(os.path.join(_HERE, "v3_ws_partialft_pilot.py")).read()
    body = src[src.index("cache tap 11 once"):src.index("def feats(")]
    assert "taps[SPLIT_AT]" in body
    bad = re.findall(r"taps\[SPLIT_AT\]\s*\.to\(", body)
    assert not bad, f"the tap is being cast before caching: {bad}"
    assert "dtype=t.dtype" in body, "the cache buffer must be allocated in the tap's own dtype"


def test_bf16_rounding_of_the_tap_really_does_break_the_split():
    """The counterfactual, so the guard above is not cargo cult: rounding tap-11 to bf16 and
    re-running the tail must move the output by ~a bf16 ulp, far outside L0's 1e-3 gate."""
    towers = __import__("speech_decoding.models.v14_converged_v3.towers",
                        fromlist=["build_encoder"])
    torch.manual_seed(0)
    enc = towers.build_encoder(n_parcels=8).eval()
    n_blocks = len(enc.blocks)
    split, tap = n_blocks - 1, n_blocks

    class _G:
        pass

    B, n_contacts, T = 2, 6, 3
    M = n_contacts * T
    g = _G()
    g.total, g.max_seqlen = M, M
    g.depth = torch.arange(n_contacts).repeat_interleave(T)
    g.time_pos = torch.arange(T).repeat(n_contacts)
    g.cu_seqlens = torch.tensor([0, M], dtype=torch.int32)
    d = enc.blocks[0].mlp.fc2.out_features
    x = torch.randn(B, M, d)
    parcel_packed = torch.randint(0, 8, (M,))
    with torch.no_grad():
        _out, taps = enc.forward_flat(x, g, parcel_packed, tap_blocks=(split, tap))
        ctx = FT._flat_ctx(enc, g, B, torch.device("cpu"))
        FT.SPLIT_AT, FT.TAP = split, tap
        exact = FT._run_tail(enc, taps[split], ctx)
        rounded = FT._run_tail(enc, taps[split].to(torch.bfloat16).float(), ctx)
    rel = float((exact - rounded).abs().max() / exact.abs().max())
    assert rel > 1e-3, f"bf16 rounding moved the tail by only rel={rel:.2e} — L0's gate is loose"
