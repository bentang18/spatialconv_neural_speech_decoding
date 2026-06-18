"""TDD for the M4 target trial-variance guard accumulator.

Quantitative: the streaming Welford must recover the SAME per-(parcel, freq-time)
trial-variance a dense ``torch.var(..., unbiased=True)`` gives on the full stack,
a zero-variance (constant-across-trials) cell must read as ~0 / "easy", ragged
per-parcel counts must be honest, and a parcel seen < 2 trials must be undefined.
"""

from __future__ import annotations

import math

import pytest
import torch

from types import SimpleNamespace

from speech_decoding.models.m4_target_variance import (
    M4TargetVarianceAccumulator,
    M4VarianceSummary,
    accumulate_m4_target_variance,
)


def _one_hot_support(electrode_to_parcel: list[int], n_parcels: int) -> torch.Tensor:
    C = len(electrode_to_parcel)
    s = torch.zeros(C, n_parcels)
    for e, p in enumerate(electrode_to_parcel):
        s[e, p] = 1.0
    return s


def _fake_loader(feats: torch.Tensor, support: torch.Tensor, valid=None):
    """One clip per leading-dim slice; the slow band carries pre-baked teacher
    features so a pass-through fake frontend returns them as ``(B,C,38,d)``."""
    T, C, S, d = feats.shape
    K = support.shape[-1]
    for t in range(T):
        data = {
            "electrode_tokens_slow": feats[t][None],          # (1,C,S,d)
            "electrode_tokens_beta": torch.zeros(1, C, 1),    # unused by fake FE
            "electrode_tokens_hg": torch.zeros(1, C, 1),
            "support": support[None],                          # (1,C,K)
        }
        if valid is not None:
            data["valid_mask"] = valid[None]
        yield SimpleNamespace(data=data)


def _passthrough_frontend(slow, beta, hg):  # noqa: ARG001
    return slow  # already (B,C,38,d)


def _stream_all_parcels(acc, stack: torch.Tensor) -> None:
    """Feed a dense (T, K, S, d) stack one 'clip' at a time, every parcel present
    in every clip (the non-ragged case)."""
    T, K = stack.shape[0], stack.shape[1]
    ids = torch.arange(K)
    for t in range(T):
        acc.update(ids, stack[t])


def test_recovers_dense_unbiased_variance() -> None:
    torch.manual_seed(0)
    T, K, S, d = 40, 5, 38, 8
    stack = torch.randn(T, K, S, d) * 3.0 + 1.0  # arbitrary scale/offset
    acc = M4TargetVarianceAccumulator(n_parcels=K, n_tokens=S, d_model=d)
    _stream_all_parcels(acc, stack)
    summ = acc.finalize()

    # dense reference: sample variance over the T trials, mean over channels
    ref_per_channel = stack.to(torch.float64).var(dim=0, unbiased=True)  # (K,S,d)
    ref_per_cell = ref_per_channel.mean(dim=-1)                          # (K,S)
    got = summ.var_per_cell.to(torch.float64)
    assert torch.allclose(got, ref_per_cell, atol=1e-6, rtol=1e-5)
    assert summ.n_cells_measured == K * S


def test_constant_across_trials_is_zero_variance_and_easy() -> None:
    T, K, S, d = 20, 3, 38, 4
    # every trial identical → trial-variance exactly 0 everywhere
    one = torch.randn(K, S, d)
    stack = one[None].expand(T, K, S, d).contiguous()
    acc = M4TargetVarianceAccumulator(n_parcels=K, n_tokens=S, d_model=d)
    _stream_all_parcels(acc, stack)
    summ = acc.finalize()
    assert torch.allclose(
        summ.var_per_cell, torch.zeros_like(summ.var_per_cell), atol=1e-10
    )
    # pooled is degenerate (0) → guarded to 1.0; every cell reads as below every cut
    assert summ.frac_easy_at[0.001] == pytest.approx(1.0)


def test_high_vs_low_variance_cells_separate_on_easiness() -> None:
    torch.manual_seed(1)
    T, K, S, d = 60, 1, 2, 4
    stack = torch.empty(T, K, S, d)
    stack[:, 0, 0] = torch.randn(T, d) * 5.0   # cell (0,0): high trial-variance
    stack[:, 0, 1] = torch.randn(T, d) * 0.01  # cell (0,1): tiny trial-variance
    acc = M4TargetVarianceAccumulator(n_parcels=K, n_tokens=S, d_model=d)
    _stream_all_parcels(acc, stack)
    summ = acc.finalize()
    assert summ.var_per_cell[0, 0] > 100 * summ.var_per_cell[0, 1]


def test_ragged_counts_are_honest() -> None:
    # parcel 0 appears in every clip; parcel 1 only in half
    torch.manual_seed(2)
    S, d = 38, 4
    acc = M4TargetVarianceAccumulator(n_parcels=2, n_tokens=S, d_model=d)
    for t in range(10):
        if t % 2 == 0:
            acc.update(torch.tensor([0, 1]), torch.randn(2, S, d))
        else:
            acc.update(torch.tensor([0]), torch.randn(1, S, d))
    summ = acc.finalize()
    assert int(summ.count[0]) == 10
    assert int(summ.count[1]) == 5


def test_single_trial_parcel_is_undefined() -> None:
    S, d = 38, 4
    acc = M4TargetVarianceAccumulator(n_parcels=2, n_tokens=S, d_model=d)
    acc.update(torch.tensor([0]), torch.randn(1, S, d))      # parcel 0: 1 trial
    acc.update(torch.tensor([1]), torch.randn(1, S, d))
    acc.update(torch.tensor([1]), torch.randn(1, S, d))      # parcel 1: 2 trials
    summ = acc.finalize()
    assert torch.isnan(summ.var_per_cell[0]).all()           # parcel 0 undefined
    assert torch.isfinite(summ.var_per_cell[1]).all()        # parcel 1 defined
    assert summ.n_cells_measured == S                        # only parcel 1's cells


def test_empty_accumulator_finalizes_without_measured_cells() -> None:
    acc = M4TargetVarianceAccumulator(n_parcels=3, n_tokens=38, d_model=4)
    summ = acc.finalize()
    assert summ.n_cells_measured == 0
    assert math.isnan(summ.median_var)


def test_distinct_parcel_guard() -> None:
    acc = M4TargetVarianceAccumulator(n_parcels=4, n_tokens=38, d_model=4)
    with pytest.raises(ValueError, match="DISTINCT"):
        acc.update(torch.tensor([1, 1]), torch.randn(2, 38, 4))


def test_out_of_range_and_shape_guards() -> None:
    acc = M4TargetVarianceAccumulator(n_parcels=2, n_tokens=38, d_model=4)
    with pytest.raises(ValueError, match="out of range"):
        acc.update(torch.tensor([5]), torch.randn(1, 38, 4))
    with pytest.raises(ValueError, match="must be"):
        acc.update(torch.tensor([0]), torch.randn(1, 10, 4))  # wrong S


def test_summary_as_dict_is_jsonable() -> None:
    import json

    torch.manual_seed(3)
    acc = M4TargetVarianceAccumulator(n_parcels=2, n_tokens=4, d_model=4)
    for _ in range(5):
        acc.update(torch.tensor([0, 1]), torch.randn(2, 4, 4))
    summ = acc.finalize()
    assert isinstance(summ, M4VarianceSummary)
    json.dumps(summ.as_dict())  # must not raise
    assert summ.as_dict()["n_cells_measured"] == 2 * 4


def test_construction_guards() -> None:
    with pytest.raises(ValueError):
        M4TargetVarianceAccumulator(n_parcels=0, n_tokens=38, d_model=4)


# ----------------------------------------------------- loader-streaming helper
def test_helper_recovers_variance_end_to_end() -> None:
    torch.manual_seed(0)
    K = C = 2
    S, d, T = 38, 4, 30
    support = _one_hot_support([0, 1], K)          # electrode e → parcel e
    feats = torch.randn(T, C, S, d) * 2.0
    valid = torch.ones(C, dtype=torch.bool)
    summ = accumulate_m4_target_variance(
        _passthrough_frontend, _fake_loader(feats, support, valid),
        n_parcels=K, n_tokens=S, d_model=d,
    )
    # one electrode per parcel ⇒ electrode-mean == that electrode's feature ⇒
    # parcel p's trial-variance == var over clips of feats[:, p]
    ref = feats.to(torch.float64).var(dim=0, unbiased=True).mean(dim=-1)  # (C=K, S)
    assert torch.allclose(summ.var_per_cell.to(torch.float64), ref, atol=1e-6, rtol=1e-5)
    assert int(summ.count[0]) == T and int(summ.count[1]) == T


def test_helper_electrode_mean_over_two_in_parcel() -> None:
    torch.manual_seed(1)
    K, C = 1, 2
    S, d, T = 38, 3, 20
    support = _one_hot_support([0, 0], K)          # both electrodes → parcel 0
    feats = torch.randn(T, C, S, d)
    summ = accumulate_m4_target_variance(
        _passthrough_frontend, _fake_loader(feats, support),
        n_parcels=K, n_tokens=S, d_model=d,
    )
    mean_feat = feats.mean(dim=1)                  # (T, S, d) electrode-mean
    ref = mean_feat.to(torch.float64).var(dim=0, unbiased=True).mean(dim=-1)  # (S,)
    assert torch.allclose(summ.var_per_cell[0].to(torch.float64), ref, atol=1e-6, rtol=1e-5)


def test_helper_max_clips_caps_accumulation() -> None:
    K = C = 2
    S, d, T = 38, 4, 50
    support = _one_hot_support([0, 1], K)
    feats = torch.randn(T, C, S, d)
    summ = accumulate_m4_target_variance(
        _passthrough_frontend, _fake_loader(feats, support),
        n_parcels=K, n_tokens=S, d_model=d, max_clips=8,
    )
    # 8 clips accumulated total (each clip contributes both parcels)
    assert int(summ.count.sum()) <= 8 * K
    assert int(summ.count[0]) <= 8


def test_helper_respects_valid_mask_dropping_an_electrode() -> None:
    K, C = 2, 2
    S, d, T = 38, 3, 12
    support = _one_hot_support([0, 1], K)
    feats = torch.randn(T, C, S, d)
    valid = torch.tensor([True, False])            # electrode 1 (parcel 1) dropped
    summ = accumulate_m4_target_variance(
        _passthrough_frontend, _fake_loader(feats, support, valid),
        n_parcels=K, n_tokens=S, d_model=d,
    )
    assert int(summ.count[0]) == T                 # parcel 0 always present
    assert int(summ.count[1]) == 0                 # parcel 1 never present
