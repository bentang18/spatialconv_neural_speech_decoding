"""v14_converged_v3 r5-mod — model-free state TARGET invariants (B3).

Load-bearing properties, each asserted with a printed ``[check]``: (1) ``_relmod`` puts a
known-frequency envelope oscillation in the right band (the FFT bin math); (2) raw parcel
means match hand computation and a constant envelope has ZERO relative modulation; (3) the
scatter/index_add vectorization equals the per-parcel reference loop on ragged geometry;
(4) a parcel with < MIN_MOD_ELEC electrodes has its modulation dims present=False; (5)
cm-removal zeroes the cross-parcel mean of every PRESENT dim at every (clip, slot); (6) the
target is stop-grad (model-free); (7) raw_state_stats / the cross-session accumulator recover
per-(parcel,dim) population moments.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.state_target import (
    HGA_BAND,
    MIN_MOD_ELEC,
    N_MEAN,
    N_MOD,
    STATE_DIM,
    StateStatsAccumulator,
    _relmod,
    build_state_target,
    dim_presence,
    raw_state_stats,
    raw_state_vectors,
)

B, N, F, T = 4, 6, 5, 16
PARCEL_ID = torch.tensor([0, 0, 0, 1, 1, 2])  # n_elec = [3, 2, 1] — parcel 2 is a singleton
S = T // 8  # 2 slots


def test_relmod_localizes_known_frequency() -> None:
    """The physics the target rests on: a pure envelope oscillation at f Hz must land in the
    band containing f and (mostly) nowhere else. 64 frames @32 Hz → 0.5 Hz bins; a 6 Hz tone
    → relmod48 (4-8 Hz) ≫ relmod816 (8-16 Hz); a 10 Hz tone → the reverse. Pins the rFFT bin
    math so a fixed clip length can never silently mis-map the Hz bands."""
    fs, Tc = 32.0, 64
    t = torch.arange(Tc, dtype=torch.float64) / fs
    env6 = (1.0 + 0.5 * torch.sin(2 * torch.pi * 6.0 * t))[None, None, :]   # (1,1,64)
    env10 = (1.0 + 0.5 * torch.sin(2 * torch.pi * 10.0 * t))[None, None, :]
    m6 = _relmod(env6, fs=fs).squeeze()   # (2,)
    m10 = _relmod(env10, fs=fs).squeeze()
    ok = (m6[0] > 0.5 and m6[0] > 5 * m6[1] and m10[1] > 0.5 and m10[1] > 5 * m10[0])
    print(f"[check] _relmod localizes: 6Hz→[{m6[0]:.2f},{m6[1]:.2f}] (48 wins), "
          f"10Hz→[{m10[0]:.2f},{m10[1]:.2f}] (816 wins) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_dim_presence_layout_wellformed() -> None:
    """present_masked_diag_nll TRUSTS this layout instead of re-checking per step: mean dims
    [0:3] always present; the 2 modulation dims [3:5] all-or-none per parcel, on iff
    n_elec >= MIN_MOD_ELEC. Pinned at the constructor so the consumer-side guard can be
    removed safely (feedback-build-the-invariant-into-the-probe)."""
    n_elec = torch.arange(0, 40)  # 0,1 (mod off) .. large (mod on); covers both patterns
    present = dim_presence(n_elec)  # (P, 5)
    mean_always = bool(present[:, :N_MEAN].all())
    mod_block = present[:, N_MEAN:]
    mod_all_or_none = bool((mod_block.all(-1) | (~mod_block).all(-1)).all())
    mod_matches_count = bool((mod_block.all(-1) == (n_elec >= MIN_MOD_ELEC)).all())
    ok = mean_always and mod_all_or_none and mod_matches_count
    print(f"[check] dim_presence: mean-always-on={mean_always} mod-all-or-none={mod_all_or_none} "
          f"mod⇔(n≥{MIN_MOD_ELEC})={mod_matches_count} {'OK' if ok else 'VIOLATED'}")
    assert ok


def _const_bands() -> list[torch.Tensor]:
    # each band = constant per electrode (value = electrode index), broadcast over (F, T);
    # env is constant in time, so parcel means = mean over the electrode set and relmod = 0.
    e = torch.arange(N, dtype=torch.float32)
    return [e[None, :, None, None].expand(B, N, F, T).clone() for _ in range(N_MEAN)]


def test_raw_parcel_mean_and_constant_has_zero_modulation() -> None:
    raw, parcels, n_elec = raw_state_vectors(_const_bands(), PARCEL_ID)
    assert raw.shape == (B, 3, S, STATE_DIM)
    # parcel 0 = electrodes {0,1,2}: mean 1 on every mean dim.
    p0_mu = raw[:, 0, :, :N_MEAN]
    ok_mu = torch.allclose(p0_mu, torch.ones_like(p0_mu))
    # a constant-in-time envelope has NO fluctuation ⇒ relative modulation power = 0.
    mod = raw[:, :, :, N_MEAN:]
    ok_mod = torch.allclose(mod, torch.zeros_like(mod), atol=1e-6)
    ok = ok_mu and ok_mod and torch.equal(n_elec, torch.tensor([3, 2, 1]))
    print(f"[check] parcel0 mean=1 ({ok_mu}), constant→relmod=0 ({ok_mod}), "
          f"n_elec={n_elec.tolist()} {'OK' if ok else 'VIOLATED'}")
    assert ok


def _raw_state_vectors_loop(bands, parcel_id, *, slot_stride=8):
    """Reference per-parcel loop (the pre-vectorization implementation) — the fixture the
    vectorized ``raw_state_vectors`` must reproduce bit-for-bit on arbitrary geometry."""
    B, N, _, T = bands[0].shape
    S = T // slot_stride
    parcels = torch.unique(parcel_id)
    P = int(parcels.shape[0])
    slots = [b.mean(dim=2).reshape(B, N, S, slot_stride).mean(dim=-1) for b in bands]  # (B,N,S)
    mod_e = _relmod(bands[HGA_BAND].mean(dim=2))  # (B, N, N_MOD) per electrode
    n_elec = torch.empty(P, dtype=torch.long)
    raw = bands[0].new_zeros(B, P, S, STATE_DIM)
    for pi in range(P):
        idx = torch.nonzero(parcel_id == parcels[pi], as_tuple=False).squeeze(1)
        n_elec[pi] = idx.shape[0]
        for bi in range(N_MEAN):
            raw[:, pi, :, bi] = slots[bi][:, idx].mean(dim=1)
        m = mod_e[:, idx].mean(dim=1)                       # (B, N_MOD) parcel-pooled
        raw[:, pi, :, N_MEAN:] = m[:, None, :].expand(B, S, N_MOD)  # broadcast across slots
    return raw, parcels, n_elec


def test_vectorized_matches_reference_loop() -> None:
    # The scatter/index_add vectorization must equal the per-parcel loop it replaces, on a
    # ragged parcel layout (singleton, pair, and larger parcels; parcel ids non-contiguous).
    g = torch.Generator().manual_seed(17)
    Bv, Nv, Fv, Tv = 5, 9, 4, 24
    parcel_id = torch.tensor([7, 7, 7, 7, 2, 2, 5, 5, 9])  # counts {7:4, 2:2, 5:2, 9:1}
    bands = [torch.rand(Bv, Nv, Fv, Tv, generator=g) * 3.0 for _ in range(N_MEAN)]
    raw, parcels, n_elec = raw_state_vectors(bands, parcel_id)
    ref_raw, ref_parcels, ref_n = _raw_state_vectors_loop(bands, parcel_id)
    ok = (
        torch.equal(parcels, ref_parcels)
        and torch.equal(n_elec, ref_n)
        and torch.allclose(raw, ref_raw, atol=1e-5)
    )
    print(f"[check] vectorized raw_state_vectors == reference loop "
          f"(parcels {parcels.tolist()}, n_elec {n_elec.tolist()}) "
          f"max|Δ|={ (raw - ref_raw).abs().max().item():.2e} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_singleton_parcel_modulation_is_masked() -> None:
    target, present, parcels = build_state_target(
        _const_bands(), PARCEL_ID,
        stat_mean=torch.zeros(3, STATE_DIM), stat_std=torch.ones(3, STATE_DIM),
    )
    # parcel index 2 is the singleton (id 2): modulation dims present=False, mean dims present.
    p_singleton = int((parcels == 2).nonzero().item())
    mod_absent = not present[p_singleton, N_MEAN:].any()
    mean_present = present[p_singleton, :N_MEAN].all()
    others_full = present[[i for i in range(3) if i != p_singleton]].all()
    ok = mod_absent and mean_present and others_full
    print(f"[check] singleton parcel modulation masked (present row "
          f"{present[p_singleton].tolist()}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_common_mode_removed_zeroes_cross_parcel_mean() -> None:
    # Identity z-score isolates the cm step: after removal the cross-parcel mean of each
    # PRESENT dim must be ~0 at every (clip, slot). Non-constant input so it's a real test.
    g = torch.Generator().manual_seed(3)
    bands = [torch.rand(B, N, F, T, generator=g) + 0.1 for _ in range(N_MEAN)]
    target, present, parcels = build_state_target(
        bands, PARCEL_ID,
        stat_mean=torch.zeros(3, STATE_DIM), stat_std=torch.ones(3, STATE_DIM),
    )
    # mean dims: all 3 parcels present → straight cross-parcel mean is ~0.
    mean_cm = target[:, :, :, :N_MEAN].mean(dim=1).abs().max().item()
    # modulation dims: only present (n>=2) parcels contribute → masked cross-parcel mean ~0.
    modp = present[:, N_MEAN:]  # (P, N_MOD) which parcels are present per modulation dim
    max_mod_cm = 0.0
    for d in range(N_MOD):
        sel = modp[:, d]  # (P,)
        vals = target[:, sel, :, N_MEAN + d]  # (B, n_present, S)
        max_mod_cm = max(max_mod_cm, vals.mean(dim=1).abs().max().item())
    ok = mean_cm < 1e-5 and max_mod_cm < 1e-5
    print(f"[check] cm removed: cross-parcel mean-dim {mean_cm:.2e}, mod-dim {max_mod_cm:.2e} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_target_is_stop_grad() -> None:
    bands = [torch.rand(B, N, F, T, requires_grad=True) for _ in range(N_MEAN)]
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
    bands = [torch.rand(9, N, F, T, generator=g) for _ in range(N_MEAN)]
    raw, parcels, _ = raw_state_vectors(bands, PARCEL_ID)  # (9, P, S, 5)
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
    bands = [torch.rand(9, N, F, T, generator=g) for _ in range(N_MEAN)]
    raw, parcels, _ = raw_state_vectors(bands, PARCEL_ID)  # (9, P, S, 5), values {0,1,2}
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
