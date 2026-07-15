"""v14_converged_v3 r4 — model-free secondary-head TARGET construction (B3).

The perceiver's secondary loss predicts, per present (parcel, 4 Hz slot), the joint
6-vector parcel state (contract project-r4-contract-2026-07-15 §7)

    x = [slow_mu, mid_mu, hga_mu, slow_sd, mid_sd, hga_sd]

= the 3 per-band parcel MEANS (mean band-envelope over the parcel's electrodes) + the 3
per-band within-parcel STDs (SD across the parcel's electrodes), on the 4 Hz slot grid.

MODEL-FREE, STOP-GRAD (collapse-proof; MaskFeat/BEST-RQ/HuBERT precedent) — computed
ON-THE-FLY in the objective from the SAME |STFT| bands the stem reads (Ben MCQ 2026-07-15),
so it stays in lockstep with the input; only the frozen per-(subject,parcel,dim) z-score
stats are precomputed offline.

Pipeline (Ben-locked order): raw parcel mean/std → window-average to 4 Hz → z-score per
(subject,parcel,dim) with FROZEN train stats → remove the per-slot cross-parcel common
mode (in the z-scored coordinate — keeps the measured NOISE_VAR floor valid) → per-(parcel,
dim) PRESENCE mask (the 3 std dims are UNDEFINED at n_elec=1, so present iff n_elec >= 2;
the mean dims are always present). The head's NLL marginalizes to the present dims.
"""

from __future__ import annotations

import torch
from torch import Tensor

SLOT_STRIDE: int = 8  # 32 Hz env → 4 Hz parcel-state grid (the head's slot rate)
N_BANDS: int = 3      # SLOW, MID, HGA
STATE_DIM: int = 6    # 3 means + 3 stds
MIN_STD_ELEC: int = 2  # std is undefined at 1 electrode (Ben edge-case 2026-07-15)
_STD_EPS: float = 1e-6  # floor on the frozen z-score std (masked dims are ignored anyway)


def raw_state_vectors(
    band_inputs: list[Tensor] | tuple[Tensor, ...],
    parcel_id: Tensor,
    *,
    slot_stride: int = SLOT_STRIDE,
) -> tuple[Tensor, Tensor, Tensor]:
    """Model-free RAW 6-vector (pre-normalization), per (clip, parcel, 4 Hz slot).

    ``band_inputs``: 3 |STFT| magnitude tensors (B, N, F_b, T) in SLOW,MID,HGA order.
    ``parcel_id``: (N,) long, the DKT parcel per electrode (this session's geometry).

    Returns ``(raw (B, P, S, 6), parcels (P,) the sorted unique parcel ids,
    n_elec (P,) electrodes per parcel)``. Slot count S = T // slot_stride. The 6-vector
    dim order is [b0_mu, b1_mu, b2_mu, b0_sd, b1_sd, b2_sd] (band-major mean then std),
    matching secondary_head.NOISE_VAR. STOP-GRAD: detached (it is a target)."""
    if len(band_inputs) != N_BANDS:
        raise ValueError(f"expected {N_BANDS} bands, got {len(band_inputs)}")
    with torch.no_grad():
        B, N, _, T = band_inputs[0].shape
        S = T // slot_stride
        if S * slot_stride != T:
            raise ValueError(f"T={T} not divisible by slot_stride={slot_stride}")
        parcels = torch.unique(parcel_id)  # sorted ascending
        P = int(parcels.shape[0])
        # per-band per-electrode 4 Hz envelope: mean over freq bins → window-average.
        slots = []
        for b in band_inputs:
            env = b.mean(dim=2)  # (B, N, T)  mean over freq bins
            env = env.reshape(B, N, S, slot_stride).mean(dim=-1)  # (B, N, S) 4 Hz
            slots.append(env)
        n_elec = torch.empty(P, dtype=torch.long, device=parcel_id.device)
        raw = band_inputs[0].new_zeros(B, P, S, STATE_DIM)
        for pi in range(P):
            idx = torch.nonzero(parcel_id == parcels[pi], as_tuple=False).squeeze(1)
            n_elec[pi] = idx.shape[0]
            for bi in range(N_BANDS):
                e = slots[bi][:, idx]  # (B, n_p, S)
                raw[:, pi, :, bi] = e.mean(dim=1)  # parcel MEAN
                # population SD (ddof=0) — matches the M11/M17 reliability measurement;
                # =0 at n=1 (masked out downstream). std over the electrode axis.
                raw[:, pi, :, N_BANDS + bi] = e.std(dim=1, unbiased=False)
        return raw.detach(), parcels, n_elec


def dim_presence(n_elec: Tensor, *, min_std_elec: int = MIN_STD_ELEC) -> Tensor:
    """(P,) electrode counts → (P, 6) bool presence: mean dims always present; std dims
    present iff n_elec >= min_std_elec (std is undefined/degenerate below that)."""
    P = int(n_elec.shape[0])
    present = torch.ones(P, STATE_DIM, dtype=torch.bool, device=n_elec.device)
    std_ok = n_elec >= min_std_elec  # (P,)
    present[:, N_BANDS:] = std_ok[:, None]
    return present


def normalize_target(
    raw: Tensor,
    stat_mean: Tensor,
    stat_std: Tensor,
    present: Tensor,
) -> Tensor:
    """z-score (frozen stats) THEN remove the per-slot cross-parcel common mode.

    ``raw`` (B, P, S, 6); ``stat_mean``/``stat_std`` (P, 6) FROZEN per-(parcel,dim) train
    stats; ``present`` (P, 6) bool. Returns the target (B, P, S, 6). Order is Ben-locked:
    z-score first (comparable units), then cm-remove in those units. The common mode of a
    dim at each (clip, slot) is the mean over the parcels PRESENT for that dim (a 1-elec
    parcel contributes nothing to the std common mode). Values at absent dims are set 0
    (the NLL marginalizes them out — only finiteness matters)."""
    # stats are per-(parcel,dim) (P, 6); insert the slot axis so they broadcast over
    # (B, P, S, 6) at the parcel and dim axes.
    sm = stat_mean[:, None, :]  # (P, 1, 6)
    ss = stat_std[:, None, :].clamp_min(_STD_EPS)  # (P, 1, 6)
    z = (raw - sm) / ss  # (B, P, S, 6)
    pmask = present[None, :, None, :].to(z.dtype)  # (1, P, 1, 6)
    # per-dim cross-parcel common mode over PRESENT parcels, per (clip, slot).
    present_count = pmask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (1, 1, 1, 6) present-per-dim
    cm = (z * pmask).sum(dim=1, keepdim=True) / present_count  # (B, 1, S, 6)
    target = (z - cm) * pmask  # absent dims → 0
    return target


def build_state_target(
    band_inputs: list[Tensor] | tuple[Tensor, ...],
    parcel_id: Tensor,
    stat_mean: Tensor,
    stat_std: Tensor,
    *,
    slot_stride: int = SLOT_STRIDE,
    min_std_elec: int = MIN_STD_ELEC,
) -> tuple[Tensor, Tensor, Tensor]:
    """End-to-end model-free target. Returns ``(target (B, P, S, 6), present (P, 6) bool,
    parcels (P,))``. ``stat_mean``/``stat_std`` are this session's frozen per-(parcel,dim)
    train stats (see :func:`raw_state_stats`)."""
    raw, parcels, n_elec = raw_state_vectors(
        band_inputs, parcel_id, slot_stride=slot_stride
    )
    present = dim_presence(n_elec, min_std_elec=min_std_elec)
    target = normalize_target(raw, stat_mean, stat_std, present)
    return target, present, parcels


def raw_state_stats(raw: Tensor) -> tuple[Tensor, Tensor]:
    """Per-(parcel,dim) mean/std over a stack of RAW 6-vectors (the offline frozen-stats
    pass; a producer accumulates ``raw`` from :func:`raw_state_vectors` over the TRAIN
    clips of one session). ``raw`` (M, P, S, 6) → ``(mean (P, 6), std (P, 6))`` over the
    (M, S) sample axes. Population std; degenerate std dims (n_elec=1, all-zero) yield 0,
    floored downstream and masked out anyway."""
    flat = raw.movedim(1, 0).reshape(raw.shape[1], -1, STATE_DIM)  # (P, M*S, 6)
    return flat.mean(dim=1), flat.std(dim=1, unbiased=False)
