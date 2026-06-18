"""M4 target trial-variance guard (converged-arch memo, M4 "Guard" bullet).

The one real M4 risk is the electrode-mean teacher target being too **easy**: if
``Var_trial(mean teacher feature | parcel, freq-time)`` is near zero, the M4
predictor can memorise a constant per (parcel, freq-time) cell and the gradient
is weak. This module is the measurement the memo asks to run "on a teacher ckpt
before committing M4".

The core is a streaming Welford accumulator over the M4 target — the per-(parcel,
freq-time-cell) **trial-variance** of the electrode-mean teacher-frontend feature
(:func:`speech_decoding.models.v14_converged.parcel_electrode_mean`). Streaming so
the DCC driver never holds every trial's targets in memory; ragged because the
parcels present vary per subject/clip.

The summary is deliberately threshold-FREE on the decision: it reports the
trial-variance distribution and a dimensionless "easiness" ratio (per-cell
trial-variance ÷ the pooled across-trial feature variance) at several readable
cut points, so the human reads "too easy?" off the numbers rather than the probe
auto-deciding. Per the memo, any escalation if too-easy is on the
subject-invariant axes ONLY (finer freq-time grid, higher parcel-mask ratio,
deeper teacher target) — NEVER the electrode axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class M4VarianceSummary:
    """Finalised diagnostic. ``var_per_cell`` / ``count`` are the raw per-(parcel,
    freq-time) trial-variance (mean over feature channels) and per-parcel trial
    count; the scalar fields are the threshold-free read-off the memo wants."""

    var_per_cell: Tensor      # (K, S) sample trial-variance, NaN where count < 2
    count: Tensor             # (K,) trials seen per parcel
    n_cells_measured: int     # cells with count >= 2 (a defined variance)
    median_var: float         # median per-cell trial-variance (measured cells)
    mean_var: float
    pooled_feature_var: float  # across-trial feature variance pooled over all cells
    median_easiness: float     # median (per-cell var / pooled_feature_var)
    frac_easy_at: dict[float, float]  # easiness threshold -> fraction below it

    def as_dict(self) -> dict[str, object]:
        """JSON-able (tensors → lists) for the DCC driver's report file."""
        return {
            "n_cells_measured": self.n_cells_measured,
            "median_var": self.median_var,
            "mean_var": self.mean_var,
            "pooled_feature_var": self.pooled_feature_var,
            "median_easiness": self.median_easiness,
            "frac_easy_at": {str(k): v for k, v in self.frac_easy_at.items()},
            "count_per_parcel": self.count.tolist(),
        }


# Easiness cut points reported by default (per-cell trial-variance as a fraction
# of the pooled across-trial feature variance). Spread over orders of magnitude so
# the human can read where the mass sits; NOT a committed decision threshold.
_DEFAULT_EASINESS_CUTS: tuple[float, ...] = (0.001, 0.01, 0.05, 0.1)


class M4TargetVarianceAccumulator:
    """Streaming per-(parcel, freq-time-cell, channel) Welford accumulator for the
    M4 target trial-variance.

    One :meth:`update` per clip ingests that clip's electrode-mean teacher feature
    for the parcels PRESENT in it (one "trial" per present parcel). Parcels absent
    from a clip are simply not updated, so the per-parcel trial count is honest for
    a ragged cohort. Finalises to the per-(parcel, freq-time) trial-variance (mean
    over the ``d`` channels) and the memo's easiness summary."""

    def __init__(self, n_parcels: int, n_tokens: int, d_model: int) -> None:
        if n_parcels < 1 or n_tokens < 1 or d_model < 1:
            raise ValueError(
                f"n_parcels/n_tokens/d_model must be >=1, got "
                f"{n_parcels}/{n_tokens}/{d_model}."
            )
        self.n_parcels = n_parcels
        self.n_tokens = n_tokens
        self.d_model = d_model
        # float64 accumulators: trial counts are modest but the squared-deviation
        # sum (M2) is variance-sensitive, so keep the running stats in double.
        self.count = torch.zeros(n_parcels, dtype=torch.long)
        self._mean = torch.zeros(n_parcels, n_tokens, d_model, dtype=torch.float64)
        self._m2 = torch.zeros(n_parcels, n_tokens, d_model, dtype=torch.float64)

    @torch.no_grad()
    def update(self, parcel_ids: Tensor, parcel_mean_feats: Tensor) -> None:
        """Ingest one clip. ``parcel_ids`` ``(P,)`` long = the DISTINCT parcel ids
        present in this clip (0..n_parcels-1); ``parcel_mean_feats`` ``(P, S, d)`` =
        the electrode-mean teacher-frontend feature per present parcel."""
        pid = torch.as_tensor(parcel_ids).reshape(-1).long().cpu()
        if pid.numel() == 0:
            return
        if torch.unique(pid).numel() != pid.numel():
            raise ValueError(
                "parcel_ids for one clip must be DISTINCT (one electrode-mean "
                "trial per present parcel); got duplicates."
            )
        if int(pid.min()) < 0 or int(pid.max()) >= self.n_parcels:
            raise ValueError(
                f"parcel_ids out of range [0,{self.n_parcels}); got "
                f"[{int(pid.min())},{int(pid.max())}]."
            )
        feats = torch.as_tensor(parcel_mean_feats).to(torch.float64).cpu()
        if feats.shape != (pid.numel(), self.n_tokens, self.d_model):
            raise ValueError(
                f"parcel_mean_feats must be (P={pid.numel()}, S={self.n_tokens}, "
                f"d={self.d_model}); got {tuple(feats.shape)}."
            )
        # Welford on the DISTINCT parcels of this clip (no in-clip aliasing → the
        # advanced-index updates don't collide).
        self.count[pid] += 1
        c = self.count[pid].to(torch.float64)[:, None, None]   # (P,1,1)
        delta = feats - self._mean[pid]                         # (P,S,d)
        self._mean[pid] += delta / c
        delta2 = feats - self._mean[pid]
        self._m2[pid] += delta * delta2

    @torch.no_grad()
    def finalize(
        self, easiness_cuts: tuple[float, ...] = _DEFAULT_EASINESS_CUTS,
    ) -> M4VarianceSummary:
        """Per-(parcel, freq-time) sample trial-variance (mean over channels) + the
        threshold-free easiness read-off."""
        measured = self.count >= 2                              # (K,)
        denom = (self.count.to(torch.float64) - 1.0).clamp_min(1.0)[:, None, None]
        var_per_channel = self._m2 / denom                     # (K,S,d) sample var
        var_per_cell = var_per_channel.mean(dim=-1)            # (K,S) over channels
        # Undefined where a parcel was seen < 2 trials.
        var_per_cell = torch.where(
            measured[:, None], var_per_cell,
            torch.full_like(var_per_cell, float("nan")),
        )

        cell_vals = var_per_cell[measured].reshape(-1)         # measured cells only
        n_cells = int(cell_vals.numel())
        if n_cells == 0:
            # No parcel reached 2 trials — nothing measurable.
            return M4VarianceSummary(
                var_per_cell=var_per_cell.to(torch.float32),
                count=self.count.clone(),
                n_cells_measured=0,
                median_var=float("nan"), mean_var=float("nan"),
                pooled_feature_var=float("nan"), median_easiness=float("nan"),
                frac_easy_at={c: float("nan") for c in easiness_cuts},
            )

        # Pooled across-trial feature scale: the mean per-channel variance over ALL
        # measured cells (the typical magnitude the cell-level variance is "easy"
        # relative to). Guarded against a degenerate zero.
        pooled = float(var_per_channel[measured].mean().item())
        pooled_safe = pooled if pooled > 0 else 1.0
        easiness = cell_vals / pooled_safe                     # dimensionless
        frac_easy = {
            float(cut): float((easiness < cut).to(torch.float64).mean().item())
            for cut in easiness_cuts
        }
        return M4VarianceSummary(
            var_per_cell=var_per_cell.to(torch.float32),
            count=self.count.clone(),
            n_cells_measured=n_cells,
            median_var=float(cell_vals.median().item()),
            mean_var=float(cell_vals.mean().item()),
            pooled_feature_var=pooled,
            median_easiness=float(easiness.median().item()),
            frac_easy_at=frac_easy,
        )


@torch.no_grad()
def accumulate_m4_target_variance(
    teacher_frontend,
    loader,
    *,
    n_parcels: int,
    n_tokens: int,
    d_model: int,
    max_clips: int | None = None,
    band_keys: tuple[str, str, str] = (
        "electrode_tokens_slow", "electrode_tokens_beta", "electrode_tokens_hg",
    ),
) -> M4VarianceSummary:
    """Stream a converged dataloader through a teacher frontend and accumulate the
    M4 target trial-variance — the DCC driver's core, factored out so it is
    testable with a fake frontend + loader (no BT data, no ckpt).

    ``teacher_frontend(slow, beta, hg) -> (B, C, 38, d)`` is the frozen EMA-teacher
    frontend (``model.teacher_frontend`` or the eval ``encode_frontend`` tap).
    ``loader`` yields batches with a ``.data`` dict carrying the three band keys +
    ``support`` (one-hot DK → argmax parcel id) + optional ``valid_mask``. Each
    sample contributes ONE electrode-mean "trial" per parcel present in it.
    ``max_clips`` caps the number of accumulated clips (None = the whole loader)."""
    from speech_decoding.models.v14_converged import parcel_electrode_mean

    acc = M4TargetVarianceAccumulator(n_parcels, n_tokens, d_model)
    seen = 0
    for batch in loader:
        data = batch.data
        slow, beta, hg = (data[k] for k in band_keys)
        support = data["support"]
        B, C = support.shape[0], support.shape[1]
        valid = data.get("valid_mask")
        emask = (
            valid.to(torch.bool) if valid is not None
            else torch.ones(B, C, dtype=torch.bool)
        )
        pe = support.argmax(dim=-1)                              # (B, C)
        t_f = teacher_frontend(slow, beta, hg).detach().cpu()   # (B, C, 38, d)
        pe = pe.cpu()
        emask = emask.cpu()
        for b in range(B):
            real = emask[b]
            present = pe[b][real].unique()
            if present.numel() == 0:
                continue
            pe_b = torch.where(real, pe[b], torch.full_like(pe[b], -1))
            means = parcel_electrode_mean(t_f[b], pe_b, present)  # (P, 38, d)
            acc.update(present, means)
            seen += 1
            if max_clips is not None and seen >= max_clips:
                return acc.finalize()
    return acc.finalize()


__all__ = [
    "M4TargetVarianceAccumulator",
    "M4VarianceSummary",
    "accumulate_m4_target_variance",
]
