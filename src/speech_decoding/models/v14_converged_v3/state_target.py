"""v14_converged_v3 r5-mod — model-free secondary-head TARGET construction (B3).

The perceiver's secondary loss predicts, per present (parcel, 4 Hz slot), the joint
5-vector parcel state

    x = [slow_mu, mid_mu, hga_mu, relmod48, relmod816]

= the 3 per-band parcel MEANS (mean band-envelope over the parcel's electrodes, per 4 Hz
slot) + 2 HGA MODULATION dims (phase-free relative modulation power of the HGA envelope in
4-8 Hz and 8-16 Hz, per parcel per CLIP, broadcast across the clip's slots).

WHY MODULATION replaces the 3 within-parcel STDs (2026-07-17):
- The stds were low-reliability and the r4 secondary flatlined on the mean+std target (the
  means are richly reachable from JEPA already — envelope within-R² 0.85 — so the head
  learned them fast then stopped; the stds are noisy).
- Modulation is a normalized SECOND moment of the HGA envelope (band power / total power),
  QUADRATIC-in-envelope and NOT linearly reachable from the latent (within-R² 0.02), so the
  encoder must BUILD it — the non-redundant, above-MAE work. Validity probe
  (project-modulation-target-validity-2026-07-17): parcel-specific per-clip reliability
  r≈0.18 (4-8 Hz) / 0.24 (8-16 Hz), positive in all 3 board subjects — weak but real. Frozen
  σ²=(1−r)·Var(y) auto-downweights it to ~¼ the gradient of a reliable dim.
- Per-electrode (rather than parcel) was rejected: with MNI banned there is no absolute
  within-parcel PE, so a (parcel,slot) query cannot say WHICH electrode ⇒ it collapses to the
  parcel mean anyway; the only per-electrode readout would read each electrode's own token
  (the MAE trap). So parcel grain is FORCED, not lazy (Ben 2026-07-17).

MODEL-FREE, STOP-GRAD (collapse-proof; MaskFeat/BEST-RQ/HuBERT precedent) — computed
ON-THE-FLY in the objective from the SAME |STFT| bands the stem reads, so it stays in
lockstep with the input; only the frozen per-(subject,parcel,dim) z-score stats are
precomputed offline.

Pipeline (Ben-locked order): raw parcel means (4 Hz) + clip modulation (broadcast) → z-score
per (subject,parcel,dim) with FROZEN train stats → remove the per-slot cross-parcel common
mode (in the z-scored coordinate — keeps the measured NOISE_VAR floor valid) → per-(parcel,
dim) PRESENCE mask (modulation is averaged over the parcel's electrodes, undefined/degenerate
at n_elec=1, so present iff n_elec >= 2; the mean dims are always present). The head's diagonal
NLL marginalizes to the present dims.
"""

from __future__ import annotations

import torch
from torch import Tensor

SLOT_STRIDE: int = 8   # 32 Hz env → 4 Hz parcel-state grid (the head's slot rate)
N_BANDS: int = 3       # SLOW, MID, HGA (the mean dims)
N_MEAN: int = 3        # 3 per-band parcel means (dims 0:3)
N_MOD: int = 2         # 2 HGA modulation dims (dims 3:5)
STATE_DIM: int = N_MEAN + N_MOD  # 5
MIN_MOD_ELEC: int = 2  # modulation averages over the parcel's electrodes; needs >= 2 to denoise
HGA_BAND: int = 2      # HGA is the 3rd band (SLOW, MID, HGA order)
HGA_FS: float = 32.0   # HGA |STFT| envelope sample rate (Hz) — the modulation FFT's rate
MOD_BANDS: tuple[tuple[float, float], ...] = ((4.0, 8.0), (8.0, 16.0))  # relmod48, relmod816 (Hz)
_STD_EPS: float = 1e-6  # floor on the frozen z-score std (masked dims are ignored anyway)


def _relmod(hga_env: Tensor, *, fs: float = HGA_FS,
            bands: tuple[tuple[float, float], ...] = MOD_BANDS) -> Tensor:
    """Phase-free relative modulation power of the per-electrode HGA envelope, per band.

    ``hga_env`` (..., T) the HGA |STFT| envelope at ``fs`` Hz over a clip of T frames.
    Per band (f_lo, f_hi): demean → Hann → rFFT → band power / total AC power ∈ [0, 1] —
    the fraction of the envelope's fluctuation in that band, LEVEL-INVARIANT (the division
    by total power removes how loud the envelope is). Returns (..., len(bands)). FFT bins
    are 0.5·(fs·2/T) apart; band k-ranges are derived from T so the Hz bands are fixed for
    any clip length (r4 clip-len is an adjustable HP)."""
    T = hga_env.shape[-1]
    x = hga_env - hga_env.mean(dim=-1, keepdim=True)
    win = torch.hann_window(T, periodic=False, dtype=x.dtype, device=x.device)
    spec = torch.fft.rfft(x * win, dim=-1)
    power = spec.real ** 2 + spec.imag ** 2                 # (..., T//2 + 1)
    tot = power[..., 1:].sum(dim=-1) + 1e-12                # total AC power (drop DC bin 0)
    outs = []
    for f_lo, f_hi in bands:
        k_lo = int(round(f_lo * T / fs))
        k_hi = min(int(round(f_hi * T / fs)), power.shape[-1])
        k_lo = min(k_lo, k_hi)
        band_p = power[..., k_lo:k_hi].sum(dim=-1)
        outs.append(band_p / tot)
    return torch.stack(outs, dim=-1)                        # (..., len(bands))


def raw_state_vectors(
    band_inputs: list[Tensor] | tuple[Tensor, ...],
    parcel_id: Tensor,
    *,
    slot_stride: int = SLOT_STRIDE,
) -> tuple[Tensor, Tensor, Tensor]:
    """Model-free RAW 5-vector (pre-normalization), per (clip, parcel, 4 Hz slot).

    ``band_inputs``: 3 |STFT| magnitude tensors (B, N, F_b, T) in SLOW,MID,HGA order.
    ``parcel_id``: (N,) long, the DKT parcel per electrode (this session's geometry).

    Returns ``(raw (B, P, S, 5), parcels (P,) sorted unique parcel ids, n_elec (P,))``.
    Slot count S = T // slot_stride. Dim order [slow_mu, mid_mu, hga_mu, relmod48, relmod816]
    (3 per-slot band means, then 2 clip-level HGA modulation dims broadcast across the S
    slots), matching secondary_head.NOISE_VAR. STOP-GRAD: detached (it is a target)."""
    if len(band_inputs) != N_BANDS:
        raise ValueError(f"expected {N_BANDS} bands, got {len(band_inputs)}")
    with torch.no_grad():
        B, N, _, T = band_inputs[0].shape
        S = T // slot_stride
        if S * slot_stride != T:
            raise ValueError(f"T={T} not divisible by slot_stride={slot_stride}")
        parcels, inv = torch.unique(parcel_id, return_inverse=True)  # sorted; inv (N,) parcel/elec
        P = int(parcels.shape[0])
        n_elec = torch.bincount(inv, minlength=P)  # (P,) electrodes per parcel

        # --- 3 per-band 4 Hz means (dims 0:3) ---
        envs = []
        for b in band_inputs:
            env = b.mean(dim=2)                                   # (B, N, T) mean over freq bins
            env = env.reshape(B, N, S, slot_stride).mean(dim=-1)  # (B, N, S) 4 Hz
            envs.append(env)
        env = torch.stack(envs, dim=-1)                           # (B, N, S, N_BANDS)
        cnt = n_elec.clamp_min(1).to(env.dtype)[None, :, None, None]  # (1, P, 1, 1)
        sum_e = env.new_zeros(B, P, S, N_BANDS).index_add_(1, inv, env)
        mean = sum_e / cnt                                        # (B, P, S, N_MEAN) parcel MEAN

        # --- 2 HGA modulation dims (dims 3:5), clip-level, broadcast across slots ---
        hga_env = band_inputs[HGA_BAND].mean(dim=2)               # (B, N, T) 32 Hz HGA envelope
        mod_e = _relmod(hga_env)                                  # (B, N, N_MOD) per electrode
        cnt_m = n_elec.clamp_min(1).to(mod_e.dtype)[None, :, None]  # (1, P, 1)
        sum_m = mod_e.new_zeros(B, P, N_MOD).index_add_(1, inv, mod_e)
        mod = (sum_m / cnt_m)[:, :, None, :].expand(B, P, S, N_MOD)  # (B, P, S, N_MOD)

        raw = torch.cat([mean, mod], dim=-1)                      # (B, P, S, 5)
        return raw.detach(), parcels, n_elec


def dim_presence(n_elec: Tensor, *, min_mod_elec: int = MIN_MOD_ELEC) -> Tensor:
    """(P,) electrode counts → (P, 5) bool presence: mean dims [0:3] always present; the 2
    modulation dims [3:5] present all-or-none iff n_elec >= min_mod_elec (a parcel-averaged
    modulation is degenerate below that)."""
    P = int(n_elec.shape[0])
    present = torch.ones(P, STATE_DIM, dtype=torch.bool, device=n_elec.device)
    mod_ok = n_elec >= min_mod_elec  # (P,)
    present[:, N_MEAN:] = mod_ok[:, None]
    return present


def normalize_target(
    raw: Tensor,
    stat_mean: Tensor,
    stat_std: Tensor,
    present: Tensor,
) -> Tensor:
    """z-score (frozen stats) THEN remove the per-slot cross-parcel common mode.

    ``raw`` (B, P, S, 5); ``stat_mean``/``stat_std`` (P, 5) FROZEN per-(parcel,dim) train
    stats; ``present`` (P, 5) bool. Returns the target (B, P, S, 5). Order is Ben-locked:
    z-score first (comparable units), then cm-remove in those units. The common mode of a
    dim at each (clip, slot) is the mean over the parcels PRESENT for that dim. Values at
    absent dims are set 0 (the NLL marginalizes them out — only finiteness matters)."""
    sm = stat_mean[:, None, :]                     # (P, 1, 5)
    ss = stat_std[:, None, :].clamp_min(_STD_EPS)  # (P, 1, 5)
    z = (raw - sm) / ss                            # (B, P, S, 5)
    pmask = present[None, :, None, :].to(z.dtype)  # (1, P, 1, 5)
    present_count = pmask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (1, 1, 1, 5)
    cm = (z * pmask).sum(dim=1, keepdim=True) / present_count      # (B, 1, S, 5)
    target = (z - cm) * pmask                       # absent dims → 0
    return target


def build_state_target(
    band_inputs: list[Tensor] | tuple[Tensor, ...],
    parcel_id: Tensor,
    stat_mean: Tensor,
    stat_std: Tensor,
    *,
    slot_stride: int = SLOT_STRIDE,
    min_mod_elec: int = MIN_MOD_ELEC,
) -> tuple[Tensor, Tensor, Tensor]:
    """End-to-end model-free target. Returns ``(target (B, P, S, 5), present (P, 5) bool,
    parcels (P,))``. ``stat_mean``/``stat_std`` are this session's frozen per-(parcel,dim)
    train stats (see :func:`raw_state_stats`)."""
    raw, parcels, n_elec = raw_state_vectors(
        band_inputs, parcel_id, slot_stride=slot_stride
    )
    present = dim_presence(n_elec, min_mod_elec=min_mod_elec)
    target = normalize_target(raw, stat_mean, stat_std, present)
    return target, present, parcels


def raw_state_stats(raw: Tensor) -> tuple[Tensor, Tensor]:
    """Per-(parcel,dim) mean/std over a stack of RAW 5-vectors (the offline frozen-stats
    pass; a producer accumulates ``raw`` from :func:`raw_state_vectors` over the TRAIN
    clips of one session). ``raw`` (M, P, S, 5) → ``(mean (P, 5), std (P, 5))`` over the
    (M, S) sample axes. Population std. Modulation dims are slot-constant, so their std is
    over clips only (the intended per-clip statistic)."""
    flat = raw.movedim(1, 0).reshape(raw.shape[1], -1, STATE_DIM)  # (P, M*S, 5)
    return flat.mean(dim=1), flat.std(dim=1, unbiased=False)


class StateStatsAccumulator:
    """Pooled per-(parcel VALUE, dim) mean/std for the frozen state-norm table.

    The offline producer of the per-subject ``sub-<id>.npz`` (``stat_mean``/``stat_std``,
    each (n_parcels, 5) indexed by parcel-id VALUE) the secondary head z-scores against
    (see :func:`normalize_target`, ``session_loader._load_state_stats``). Generalizes
    :func:`raw_state_stats` across sessions: a subject's sessions have DIFFERENT parcel
    sets, so we key running sufficient statistics by parcel-id value, not position, and
    pool a value that recurs across sessions. Population std (ddof=0). Sums are fp64.
    Common-mode removal is NOT applied here — it happens at consumption, in z-scored
    coords (see the module docstring)."""

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        self.dim = int(state_dim)
        self._sum: dict[int, Tensor] = {}
        self._sumsq: dict[int, Tensor] = {}
        self._count: dict[int, int] = {}

    def add(self, raw: Tensor, parcels: Tensor) -> None:
        """Accumulate one ``raw`` block. ``raw`` (B, P, S, 5) from
        :func:`raw_state_vectors`; ``parcels`` (P,) its sorted unique parcel-id VALUES.
        Adds B·S samples per parcel value."""
        if raw.shape[-1] != self.dim:
            raise ValueError(f"raw last dim {raw.shape[-1]} != state_dim {self.dim}")
        if raw.shape[1] != parcels.shape[0]:
            raise ValueError(f"raw P={raw.shape[1]} != parcels {parcels.shape[0]}")
        r = raw.detach().to(torch.float64)
        s = r.sum(dim=(0, 2))    # (P, 5)
        sq = (r * r).sum(dim=(0, 2))  # (P, 5)
        cnt = int(raw.shape[0] * raw.shape[2])  # B·S samples/parcel
        for pi in range(int(parcels.shape[0])):
            v = int(parcels[pi])
            if v not in self._sum:
                self._sum[v] = s[pi].clone()
                self._sumsq[v] = sq[pi].clone()
                self._count[v] = cnt
            else:
                self._sum[v] += s[pi]
                self._sumsq[v] += sq[pi]
                self._count[v] += cnt

    def finalize(self, n_parcels: int | None = None) -> tuple[Tensor, Tensor]:
        """``(stat_mean (V, 5), stat_std (V, 5))`` value-indexed, float32. ``V`` =
        ``n_parcels`` (must cover ``max value + 1``) or the observed ``max value + 1``.
        Absent values → 0 (masked/floored downstream). Population std."""
        max_v = max(self._sum) if self._sum else -1
        v_out = int(n_parcels) if n_parcels is not None else max_v + 1
        if v_out <= max_v:
            raise ValueError(f"n_parcels={v_out} <= max parcel value {max_v}")
        mean = torch.zeros(v_out, self.dim, dtype=torch.float64)
        std = torch.zeros(v_out, self.dim, dtype=torch.float64)
        for v, c in self._count.items():
            m = self._sum[v] / c
            var = (self._sumsq[v] / c) - m * m
            mean[v] = m
            std[v] = var.clamp_min(0.0).sqrt()
        return mean.float(), std.float()
