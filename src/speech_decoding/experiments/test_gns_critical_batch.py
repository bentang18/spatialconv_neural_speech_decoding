"""Pure-math tests for the gradient-noise-scale estimator (laptop, no GPU)."""

from __future__ import annotations

import math

import torch

from speech_decoding.experiments.gns_critical_batch import (
    fit_gns_curve,
    flatten_grads,
    gns_param_groups,
    gradient_noise_scale,
)


def _synthetic_e_of_b(g_sq: float, tr_sigma: float, sizes):
    # E‖g_B‖² = ‖G‖² + tr(Σ)/B exactly (the model the estimator inverts).
    return [g_sq + tr_sigma / b for b in sizes]


def test_two_point_recovers_closed_form():
    g_sq, tr_sigma = 0.09, 31.36  # => B_crit = 348.4
    e = _synthetic_e_of_b(g_sq, tr_sigma, [32, 128])
    out = gradient_noise_scale(e[0], 32, e[1], 128)
    assert math.isclose(out["g_sq"], g_sq, rel_tol=1e-9)
    assert math.isclose(out["tr_sigma"], tr_sigma, rel_tol=1e-9)
    assert math.isclose(out["b_crit"], tr_sigma / g_sq, rel_tol=1e-9)


def test_curve_fit_recovers_closed_form_exactly_noise_free():
    g_sq, tr_sigma = 0.5, 100.0  # B_crit = 200
    sizes = [16, 32, 64, 128, 256]
    e = _synthetic_e_of_b(g_sq, tr_sigma, sizes)
    fit = fit_gns_curve(sizes, e)
    assert math.isclose(fit["g_sq"], g_sq, rel_tol=1e-6)
    assert math.isclose(fit["tr_sigma"], tr_sigma, rel_tol=1e-6)
    assert math.isclose(fit["b_crit"], 200.0, rel_tol=1e-6)
    assert fit["r2"] > 0.999999


def test_curve_fit_robust_when_averaged_over_rounds():
    # B_crit's intercept (‖G‖²) is a small-difference estimate, so a SINGLE noisy
    # draw is ~15-20% off — which is precisely why run_gns_probe averages E‖g_B‖²
    # over many rounds before fitting. This test mirrors that: average the noisy
    # per-round E, then fit, and the estimate tightens to <5%.
    torch.manual_seed(0)
    g_sq, tr_sigma = 1.0, 500.0  # B_crit = 500
    sizes = [16, 32, 64, 128, 256, 512]
    clean = _synthetic_e_of_b(g_sq, tr_sigma, sizes)
    rounds = 200
    acc = [0.0] * len(sizes)
    for _ in range(rounds):
        for i, v in enumerate(clean):
            acc[i] += v * (1.0 + 0.01 * float(torch.randn(1)))  # 1% per-round noise
    e_avg = [a / rounds for a in acc]
    fit = fit_gns_curve(sizes, e_avg)
    assert abs(fit["b_crit"] - 500.0) / 500.0 < 0.05  # within 5% after averaging
    assert fit["r2"] > 0.999


def test_two_point_requires_ordered_sizes():
    try:
        gradient_noise_scale(1.0, 128, 2.0, 32)
    except ValueError:
        return
    raise AssertionError("expected ValueError for b_small >= b_big")


def test_flatten_grads_concats_populated_only():
    a = torch.nn.Parameter(torch.zeros(3))
    b = torch.nn.Parameter(torch.zeros(2))
    c = torch.nn.Parameter(torch.zeros(4))  # left with grad=None
    a.grad = torch.tensor([1.0, 2.0, 3.0])
    b.grad = torch.tensor([4.0, 5.0])
    flat = flatten_grads([a, b, c])
    assert flat.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert flat.dtype == torch.float32


def test_flatten_grads_raises_when_empty():
    p = torch.nn.Parameter(torch.zeros(2))  # no grad
    try:
        flatten_grads([p])
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when no grads populated")


class _FakeConverged(torch.nn.Module):
    # mirrors the V14ConvergedSSL top-level submodule names the grouping keys on
    def __init__(self):
        super().__init__()
        self.student_frontend = torch.nn.Linear(2, 2)
        self.teacher_frontend = torch.nn.Linear(2, 2)  # frozen EMA shadow
        self.teacher_frontend.requires_grad_(False)
        self.latent = torch.nn.Linear(2, 2)
        self.m2_predictor = torch.nn.Linear(2, 2)
        self.m4_predictor = torch.nn.Linear(2, 2)
        self.lambda_m2 = torch.nn.Parameter(torch.ones(()))  # stray top-level → "other"


def test_param_groups_partition_components_and_exclude_frozen_teacher():
    groups = gns_param_groups(_FakeConverged())
    assert set(groups) == {"frontend", "latent", "m2_pred", "m4_pred", "other", "whole"}
    # frozen teacher contributes nothing to any group
    teacher_ids = {id(p) for p in _FakeConverged().teacher_frontend.parameters()}
    assert all(id(p) not in teacher_ids for p in groups["whole"])
    # components partition `whole` exactly (each param in exactly one component)
    comp_ids = [
        id(p) for g in ("frontend", "latent", "m2_pred", "m4_pred", "other")
        for p in groups[g]
    ]
    assert sorted(comp_ids) == sorted(id(p) for p in groups["whole"])
    assert len(comp_ids) == len(set(comp_ids))  # no param double-counted
    # frontend = weight+bias of student_frontend (2), stray scalar lands in other
    assert len(groups["frontend"]) == 2
    assert len(groups["other"]) == 1


def test_param_groups_strips_lightning_model_prefix():
    # the Lightning wrapper exposes params as "model.student_frontend.weight" etc.
    class _Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _FakeConverged()

    groups = gns_param_groups(_Wrapped())
    assert len(groups["frontend"]) == 2  # still keyed correctly after stripping
    assert len(groups["latent"]) == 2
