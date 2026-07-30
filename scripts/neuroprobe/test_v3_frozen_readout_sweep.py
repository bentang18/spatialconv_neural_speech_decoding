"""Tests for the frozen-feature readout sweep. Synthetic data only — no cache, no GPU."""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SW = _load("v3_frozen_readout_sweep")
RDO = SW.RDO


def _blob(n=60, p=24, seed=0):
    g = np.random.default_rng(seed)
    y = (g.random(n) > 0.5).astype(np.float64)
    z = g.normal(size=(n, p)) + y[:, None] * 0.8
    return torch.from_numpy(z.astype(np.float32)), y


# ── the load-bearing parity test ──────────────────────────────────────────────────────
def test_ridge_from_normalized_kernels_reproduces_the_readouts_own_number():
    """A must BE the quoted column, not a re-implementation of it. If this drifts, every R0 is
    measured against a bar nobody reports and the whole sweep is meaningless. This also pins the
    basel normalization: dividing G by basel and dropping the `lam*basel` factor must cancel."""
    z_tr, y_tr = _blob(80, 32, seed=1)
    z_te, y_te = _blob(40, 32, seed=2)
    z_va, y_va = _blob(30, 32, seed=3)
    s_tr, s_va, s_te = SW._standardize_t(z_tr, z_va, z_te)
    g, k_va, k_te, _ = SW._kernels(s_tr, s_va, s_te)
    _, mine, _, _ = SW._ridge_from_kernels(g, k_va, k_te, y_tr, y_va, y_te, SW.RIDGE_LAMS)
    theirs = RDO._ridge_test(z_tr.numpy(), y_tr, z_te.numpy(), y_te, "std")
    assert mine == pytest.approx(theirs, abs=1e-9), (mine, theirs)


def test_ridge_effective_dof_is_bounded_by_n():
    """df <= n is the structural fact that keeps a 212,992-feature ridge from being crippled by
    1,279 rows. The MEASURED value on real data was 539/1279 — bounded, but not tiny."""
    z_tr, y_tr = _blob(40, 5000, seed=21)
    z_va, y_va = _blob(20, 5000, seed=22)
    z_te, y_te = _blob(20, 5000, seed=23)
    s = SW._standardize_t(z_tr, z_va, z_te)
    g, k_va, k_te, _ = SW._kernels(*s)
    *_, df = SW._ridge_from_kernels(g, k_va, k_te, y_tr, y_va, y_te, SW.RIDGE_LAMS)
    assert 0.0 < df <= 40.0, df


# ── the dual fit must BE the primal fit ───────────────────────────────────────────────
def test_dual_logistic_matches_an_explicit_primal_fit():
    """THE load-bearing test for the rewrite. The dual is only legitimate because the L2-penalized
    logistic optimum lies in the row span (w = Z^T alpha). Fit the SAME objective both ways and
    require the same predictions — if this fails, every R0 is measured with the wrong estimator."""
    torch.manual_seed(0)
    z_tr, y_tr = _blob(40, 60, seed=31)
    z_te, y_te = _blob(25, 60, seed=32)
    s_tr, s_te = SW._standardize_t(z_tr, z_te)
    g, _, k_te, basel = SW._kernels(s_tr, s_te, s_te)
    lam = 0.05
    yb = torch.as_tensor((y_tr > 0).astype(np.float64))
    lossf = torch.nn.BCEWithLogitsLoss()

    # dual: logits = G alpha + b, penalty 0.5*lam*alpha^T G alpha
    a = torch.zeros(len(y_tr), dtype=torch.float64, requires_grad=True)
    bd = torch.zeros((), dtype=torch.float64, requires_grad=True)
    od = torch.optim.Adam([a, bd], lr=0.05)
    for _ in range(4000):
        ga = g @ a
        ld = lossf(ga + bd, yb) + 0.5 * lam * (a @ ga)
        od.zero_grad(set_to_none=True); ld.backward(); od.step()

    # primal on the SAME normalized features: logits = (S/sqrt(basel)) w + b, penalty 0.5*lam*|w|^2
    sn = (s_tr.double() / basel ** 0.5)
    w = torch.zeros(60, dtype=torch.float64, requires_grad=True)
    bp = torch.zeros((), dtype=torch.float64, requires_grad=True)
    op = torch.optim.Adam([w, bp], lr=0.05)
    for _ in range(4000):
        lp = lossf(sn @ w + bp, yb) + 0.5 * lam * (w @ w)
        op.zero_grad(set_to_none=True); lp.backward(); op.step()

    with torch.no_grad():
        pd_ = (k_te @ a + bd).numpy()
        pp = ((s_te.double() / basel ** 0.5) @ w + bp).numpy()
    assert RDO.auroc(pd_, y_te) == pytest.approx(RDO.auroc(pp, y_te), abs=1e-6)
    assert np.corrcoef(pd_, pp)[0, 1] > 0.9999


def test_dual_logistic_separates_a_separable_problem():
    """Guards against reporting "the head loses" when the head simply never converged."""
    g_ = np.random.default_rng(11)
    y = np.r_[np.zeros(40), np.ones(40)]
    z = torch.from_numpy((g_.normal(size=(80, 12)) * 0.1 + y[:, None] * 3.0).astype(np.float32))
    s, = SW._standardize_t(z)
    g, k_va, k_te, _ = SW._kernels(s, s, s)
    r = SW._fit_logit_dual(g, k_va, k_te, y, y, y, lam=1e-4, lr=0.1, steps=1500, seed=0)
    assert r["val"] > 0.99, r


def test_hit_wall_flags_a_truncated_fit_and_clears_on_a_converged_one():
    """`hit_wall` is the whole defence against the underfit artifact that the first primal smoke
    run walked into (median_step == steps). It must fire when steps are too few..."""
    z, y = _blob(50, 20, seed=41)
    s, = SW._standardize_t(z)
    g, k_va, k_te, _ = SW._kernels(s, s, s)
    trunc = SW._fit_logit_dual(g, k_va, k_te, y, y, y, lam=1e-4, lr=0.3, steps=25, seed=0)
    assert trunc["hit_wall"] is True
    long = SW._fit_logit_dual(g, k_va, k_te, y, y, y, lam=10.0, lr=0.3, steps=3000, seed=0)
    assert long["hit_wall"] is False


def test_layernorm_variant_is_per_row_and_std_variant_is_per_feature():
    """The two variants must differ on the AXIS they normalize — that is the hypothesis."""
    z_tr = torch.randn(50, 8) * torch.tensor([50.0, 1, 1, 1, 1, 1, 1, 0.2])
    v = SW._variants(z_tr, z_tr, z_tr, ["std", "ln"])
    assert torch.allclose(v["std"][0].std(0, unbiased=False), torch.ones(8), atol=1e-5)
    assert torch.allclose(v["ln"][0].mean(1), torch.zeros(50), atol=1e-5)
    sd = v["ln"][0].std(0, unbiased=False)
    assert float(sd.max() / sd.min()) > 3.0        # ln leaves per-feature scales unequal


def test_variants_selects_only_what_was_asked_for():
    z = torch.randn(10, 6)
    assert set(SW._variants(z, z, z, ["std"])) == {"std"}
    assert set(SW._variants(z, z, z, ["std", "ln"])) == {"std", "ln"}


def test_standardize_matches_the_readouts_standardizer():
    z_tr, _ = _blob(50, 16, seed=4)
    z_te, _ = _blob(20, 16, seed=5)
    a, b = SW._standardize_t(z_tr, z_te)
    ra, rb = RDO._standardize(z_tr.numpy().copy(), z_te.numpy().copy())
    assert np.allclose(a.numpy(), ra, atol=1e-6)
    assert np.allclose(b.numpy(), rb, atol=1e-6)


def test_standardize_is_fit_on_train_only():
    """A test-fitted z-score would leak. Shifting test must NOT change the train mapping."""
    z_tr, _ = _blob(40, 8, seed=6)
    z_te, _ = _blob(20, 8, seed=7)
    a1, _ = SW._standardize_t(z_tr, z_te)
    a2, b2 = SW._standardize_t(z_tr, z_te + 100.0)
    assert torch.equal(a1, a2)
    assert float(b2.mean()) > 10.0            # the shift lands on test, unabsorbed


def test_constant_columns_do_not_blow_up():
    z_tr, _ = _blob(30, 6, seed=8)
    z_tr[:, 2] = 4.0
    out, = SW._standardize_t(z_tr)
    assert torch.isfinite(out).all()
    assert float(out[:, 2].abs().max()) == 0.0








# ── the report must not overclaim ──────────────────────────────────────────────────────
def _rows(delta, n=28):
    return [{"session": f"S{i // 4}", "task": f"t{i % 4}", "fold": 0, "A_test": 0.70,
             "A_val": 0.70, "A_test_vallam": 0.705, "ridge_df": 30.0, "n_train": 1279,
             "n_val": 300, "n_test": 300, "dim": 212992, "n_parcels": 16,
             "std_test": 0.70 + delta, "std_val": 0.70, "std_step": 100, "std_lr": 0.1,
             "std_lam": 0.01, "std_grad_norm": 1e-3, "std_hit_wall": False, "basel": 1.0}
            for i in range(n)]


def test_report_flags_a_frozen_head_that_beats_the_ridge():
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SW._report(_rows(+0.02), ["std"])
    out = buf.getvalue()
    assert "A FROZEN head BEATS" in out
    assert "R0=+0.0200" in out


def test_report_states_that_R0_does_not_predict_the_finetuned_cost():
    """Ben 07-30: "a ridge regression might be optimal on frozen features - but that might not be
    true on fine tuned features? don't be black and white here". The report must carry that caveat
    in its own output, so a reader of the log cannot take R0 as a verdict on the FT arms."""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SW._report(_rows(-0.01), ["std"])
    out = buf.getvalue()
    assert "does NOT predict the cost at fine-tuned features" in out.replace("\n  ", " ")
    assert "A FROZEN head BEATS" not in out


def test_sign_test_matches_known_binomial_tails():
    assert SW._sign_p(20, 28) == pytest.approx(0.03574, abs=1e-4)
    # 2 * sum_{i>=21} C(28,i) / 2^28 = 2 * 1683218 / 268435456. Computed, not remembered.
    assert SW._sign_p(21, 28) == pytest.approx(2 * 1683218 / 2 ** 28, abs=1e-9)
    assert SW._sign_p(21, 28) == pytest.approx(0.01254, abs=1e-5)
    assert SW._sign_p(14, 28) == pytest.approx(1.0)
    assert np.isnan(SW._sign_p(0, 0))


def test_report_prints_the_matched_bar_against_the_val_selected_ridge():
    """Every head number is val-selected over (lam, lr); `A_test` is the ridge at CONST lambda. So
    R0 alone hands the head a model-selection advantage. The report must ALSO test against
    `A_test_vallam`, the ridge selected on the same val split."""
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        SW._report(_rows(+0.02), ["std"])
    out = buf.getvalue()
    assert "R0_matched[std" in out
    assert "vs val-selected ridge" in out


def test_a_win_that_dies_on_the_matched_bar_is_called_model_selection():
    """_rows sets A_test=0.70 and A_test_vallam=0.705. A head at +0.02 beats the quoted column but
    +0.002 does NOT beat the val-selected ridge — the report must say so rather than bank the win."""
    import io, contextlib

    def run(delta):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            SW._report(_rows(delta), ["std"])
        return buf.getvalue()

    lost = run(+0.002)                       # 0.702 > 0.700 but < 0.705
    assert "A FROZEN head BEATS" in lost
    assert "MODEL SELECTION, not the readout family" in lost
    won = run(+0.02)                         # 0.720 > 0.705
    assert "survives the matched bar" in won
    assert "MODEL SELECTION, not the readout family" not in won


def test_merge_refuses_duplicated_cells():
    """REGRESSION: the merge glob `s*.json` also matched `smoke.json` -- a deliberately underfit
    60-step run over cells the real shards cover -- so a 56-cell design silently reported 58 cells
    with two duplicates dragging the mean. Pooling arms is this project's #1 defect class, so the
    merge must REFUSE, and must name the two files so the stale one is identifiable."""
    good = [{"session": "S1", "task": "onset", "fold": 0, "_src": "s0.json"},
            {"session": "S1", "task": "onset", "fold": 1, "_src": "s0.json"}]
    assert SW._assert_one_row_per_cell(good) == 2
    dupe = good + [{"session": "S1", "task": "onset", "fold": 0, "_src": "smoke.json"}]
    with pytest.raises(SystemExit) as e:
        SW._assert_one_row_per_cell(dupe)
    msg = str(e.value)
    assert "arms are being pooled" in msg
    assert "s0.json" in msg and "smoke.json" in msg
    assert "S1 onset f0" in msg


def test_merge_guard_accepts_the_full_56_cell_design():
    """7 sessions x 4 tasks x 2 folds = 56 distinct cells must pass untouched."""
    rows = [{"session": f"S{s}", "task": t, "fold": f, "_src": f"s{s}.json"}
            for s in range(7) for t in ("onset", "delta_volume", "word_index", "gpt2_surprisal")
            for f in (0, 1)]
    assert len(rows) == 56
    assert SW._assert_one_row_per_cell(rows) == 56
