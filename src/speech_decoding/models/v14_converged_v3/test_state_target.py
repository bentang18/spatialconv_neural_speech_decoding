"""v14_converged_v3 r4 — model-free state TARGET invariants (B3).

Load-bearing properties, each asserted with a printed ``[check]``: (1) raw parcel mean/std
match hand computation; (2) a SINGLETON parcel has undefined std (raw std = 0) and its std
dims are present=False; (3) cm-removal zeroes the cross-parcel mean of every PRESENT dim at
every (clip, slot) — the defining property; (4) the target is stop-grad (model-free); (5)
raw_state_stats recovers per-(parcel,dim) mean/std over the (clip, slot) sample axes.
"""

from __future__ import annotations

import math

import torch

from speech_decoding.models.v14_converged_v3.state_target import (
    N_BANDS,
    STATE_DIM,
    StateStatsAccumulator,
    build_state_target,
    raw_state_stats,
    raw_state_vectors,
)

B, N, F, T = 4, 6, 5, 16
PARCEL_ID = torch.tensor([0, 0, 0, 1, 1, 2])  # n_elec = [3, 2, 1] — parcel 2 is a singleton
S = T // 8  # 2 slots


def _const_bands() -> list[torch.Tensor]:
    # each band = constant per electrode (value = electrode index), broadcast over (F, T);
    # then env=const, 4 Hz slots=const, so parcel mean/std are just over the electrode set.
    e = torch.arange(N, dtype=torch.float32)
    return [e[None, :, None, None].expand(B, N, F, T).clone() for _ in range(N_BANDS)]


def test_raw_parcel_mean_and_std_match_hand_computation() -> None:
    raw, parcels, n_elec = raw_state_vectors(_const_bands(), PARCEL_ID)
    assert raw.shape == (B, S, STATE_DIM) or raw.shape == (B, 3, S, STATE_DIM)
    # parcel 0 = electrodes {0,1,2}: mean 1, population std sqrt(2/3).
    p0_mu = raw[:, 0, :, 0]
    p0_sd = raw[:, 0, :, N_BANDS]
    ok_mu = torch.allclose(p0_mu, torch.ones_like(p0_mu))
    ok_sd = torch.allclose(p0_sd, torch.full_like(p0_sd, math.sqrt(2 / 3)), atol=1e-5)
    assert torch.equal(n_elec, torch.tensor([3, 2, 1]))
    print(f"[check] parcel0 mean=1 ({ok_mu}), pop-std=√(2/3) ({ok_sd}), "
          f"n_elec={n_elec.tolist()} {'OK' if ok_mu and ok_sd else 'VIOLATED'}")
    assert ok_mu and ok_sd


def test_singleton_parcel_std_is_undefined_and_masked() -> None:
    target, present, parcels = build_state_target(
        _const_bands(), PARCEL_ID,
        stat_mean=torch.zeros(3, STATE_DIM), stat_std=torch.ones(3, STATE_DIM),
    )
    # parcel index 2 is the singleton (id 2): std dims present=False, mean dims present.
    p_singleton = int((parcels == 2).nonzero().item())
    std_absent = not present[p_singleton, N_BANDS:].any()
    mean_present = present[p_singleton, :N_BANDS].all()
    others_full = present[[i for i in range(3) if i != p_singleton]].all()
    ok = std_absent and mean_present and others_full
    print(f"[check] singleton parcel std masked (present row "
          f"{present[p_singleton].tolist()}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_common_mode_removed_zeroes_cross_parcel_mean() -> None:
    # Identity z-score isolates the cm step: after removal the cross-parcel mean of each
    # PRESENT dim must be ~0 at every (clip, slot). Non-constant input so it's a real test.
    g = torch.Generator().manual_seed(3)
    bands = [torch.rand(B, N, F, T, generator=g) + 0.1 for _ in range(N_BANDS)]
    target, present, parcels = build_state_target(
        bands, PARCEL_ID,
        stat_mean=torch.zeros(3, STATE_DIM), stat_std=torch.ones(3, STATE_DIM),
    )
    # mean dims: all 3 parcels present → straight cross-parcel mean is ~0.
    mean_cm = target[:, :, :, :N_BANDS].mean(dim=1).abs().max().item()
    # std dims: only present (n>=2) parcels contribute → masked cross-parcel mean ~0.
    stdp = present[:, N_BANDS:]  # (P, 3) which parcels are present per std dim
    max_std_cm = 0.0
    for d in range(N_BANDS):
        sel = stdp[:, d]  # (P,)
        vals = target[:, sel, :, N_BANDS + d]  # (B, n_present, S)
        max_std_cm = max(max_std_cm, vals.mean(dim=1).abs().max().item())
    ok = mean_cm < 1e-5 and max_std_cm < 1e-5
    print(f"[check] cm removed: cross-parcel mean-dim {mean_cm:.2e}, std-dim {max_std_cm:.2e} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_target_is_stop_grad() -> None:
    bands = [torch.rand(B, N, F, T, requires_grad=True) for _ in range(N_BANDS)]
    target, _, _ = build_state_target(
        bands, PARCEL_ID,
        stat_mean=torch.zeros(3, STATE_DIM), stat_std=torch.ones(3, STATE_DIM),
    )
    ok = not target.requires_grad
    print(f"[check] target is stop-grad (requires_grad={target.requires_grad}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_raw_state_stats_recovers_per_parcel_dim_moments() -> None:
    g = torch.Generator().manual_seed(7)
    bands = [torch.rand(9, N, F, T, generator=g) for _ in range(N_BANDS)]
    raw, parcels, _ = raw_state_vectors(bands, PARCEL_ID)  # (9, P, S, 6)
    mean, std = raw_state_stats(raw)
    # brute-force reference for parcel 0, dim 0 over (clip, slot).
    ref = raw[:, 0, :, 0].reshape(-1)
    ok = (torch.allclose(mean[0, 0], ref.mean(), atol=1e-6)
          and torch.allclose(std[0, 0], ref.std(unbiased=False), atol=1e-6)
          and mean.shape == (3, STATE_DIM))
    print(f"[check] raw_state_stats moments match brute force "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_accumulator_single_session_matches_raw_state_stats() -> None:
    # The producer's cross-session accumulator, run on ONE session, must reproduce
    # raw_state_stats exactly (population moments over the clip·slot samples), placed at
    # the parcel-id VALUE (not position). Absent values stay 0.
    g = torch.Generator().manual_seed(11)
    bands = [torch.rand(9, N, F, T, generator=g) for _ in range(N_BANDS)]
    raw, parcels, _ = raw_state_vectors(bands, PARCEL_ID)  # (9, P, S, 6), values {0,1,2}
    ref_mean, ref_std = raw_state_stats(raw)
    acc = StateStatsAccumulator()
    acc.add(raw, parcels)
    mean, std = acc.finalize(n_parcels=5)
    per_value = all(
        torch.allclose(mean[int(v)], ref_mean[pi], atol=1e-5)
        and torch.allclose(std[int(v)], ref_std[pi], atol=1e-5)
        for pi, v in enumerate(parcels.tolist())
    )
    absent_zero = bool(
        torch.all(mean[3] == 0) and torch.all(mean[4] == 0)
        and torch.all(std[3] == 0)
    )
    ok = per_value and absent_zero and mean.shape == (5, STATE_DIM)
    print(f"[check] accumulator single-session == raw_state_stats by value, "
          f"absent→0 {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_accumulator_pools_across_sessions_by_value() -> None:
    # A parcel value present in TWO sessions (with different data + different parcel
    # sets) must pool to the moments of the CONCATENATED sample set — the defining
    # property of value-keyed cross-session accumulation.
    g = torch.Generator().manual_seed(13)
    raw_a = torch.rand(3, 2, S, STATE_DIM, generator=g)
    raw_b = torch.rand(2, 2, S, STATE_DIM, generator=g)
    par_a = torch.tensor([5, 8])  # value 5 at position 0
    par_b = torch.tensor([1, 5])  # value 5 at position 1
    acc = StateStatsAccumulator()
    acc.add(raw_a, par_a)
    acc.add(raw_b, par_b)
    mean, std = acc.finalize(n_parcels=9)
    pooled = torch.cat(
        [raw_a[:, 0].reshape(-1, STATE_DIM), raw_b[:, 1].reshape(-1, STATE_DIM)], dim=0
    )
    ok = (
        torch.allclose(mean[5], pooled.mean(0), atol=1e-5)
        and torch.allclose(std[5], pooled.std(0, unbiased=False), atol=1e-5)
        and torch.allclose(mean[8], raw_a[:, 1].reshape(-1, STATE_DIM).mean(0), atol=1e-5)
    )
    print(f"[check] value 5 pooled over 2 sessions == concat moments "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok
