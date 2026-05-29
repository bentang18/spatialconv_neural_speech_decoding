"""B02 v14 cross-corpus batch composition: vb-eh-weighted WRS sampler.

Drift-table row B02 (closed-as-spec 2026-05-23, code scaffold below):

  α=0.5 **hierarchical** over EXACT-PRECOMPUTED valid-bin-electrode-hours
  (vb-eh). Two-group composition: SWEC vs broadband (AJILE12 + D + BT). The
  macro split is **50 / 50** (a hard cap on SWEC's share so the cross-domain
  sEEG/ECoG corpora are not drowned by SWEC's 11 kHz volume). Within the
  broadband group, the share goes as ``vb-eh ** α=0.5``.

Per-row weight (verbatim from §B02 "Verifiable as")::

    w[row] = group_share[corpus] × within_group_share[corpus] / count_rows[corpus]

The downstream sampler stack is::

    sampler = WeightedRandomSampler(weights, num_samples=..., replacement=True)
    loader  = StatefulDataLoader(dataset, sampler=sampler, ...)

The ``StatefulDataLoader`` wrapper from ``torchdata.stateful_dataloader`` is
imported lazily inside :func:`build_stateful_dataloader` so a v14 development
checkout without ``torchdata`` installed can still build the sampler + run the
weight-computation tests. The factory raises with a clear hint when the import
is missing.

What this module owns:

* :func:`compute_vb_eh_per_corpus` — aggregate vb-eh totals from manifest rows.
* :func:`compute_per_row_weights` — turn the per-corpus vb-eh totals + macro /
  within-group shares into per-row sampling weights (numpy float64).
* :func:`build_wrs_sampler` — construct
  :class:`torch.utils.data.WeightedRandomSampler` from per-row weights.
* :func:`build_stateful_dataloader` — lazy-import wrapper around
  ``torchdata.stateful_dataloader.StatefulDataLoader`` that consumes the WRS.

What stays out of scope here (handled by the multi-corpus loader build):

* Manifest discovery + chain construction across corpora.
* Per-rank locality sharding for SWEC subjects (the spec calls for
  ``subj_to_rank[s] = hash(s) % world_size`` static across epochs — the rank-
  aware split is applied by the loader layer, NOT by this primitive).
* Mid-epoch resume orchestration via ``state_dict`` / ``load_state_dict``.

Sister cells are reached by overriding the ``alpha`` parameter (or by passing
a non-default ``group_macro_split``) — see B02 §Sister cells.
"""

from __future__ import annotations

import dataclasses
import typing as tp

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


SWEC_CORPUS = "swec_ieeg"
BROADBAND_CORPORA: tuple[str, ...] = ("ajile12", "d_cohort", "braintreebank")
ALL_CORPORA: tuple[str, ...] = (SWEC_CORPUS,) + BROADBAND_CORPORA

DEFAULT_ALPHA: float = 0.5
"""B02 lock 2026-05-23: α=0.5 within-broadband; macro 50/50 SWEC vs broadband
(XLS-R §4.1 verbatim "α=0.5 in all cases" precedent)."""

DEFAULT_MACRO_SPLIT: tp.Mapping[str, float] = {
    "swec": 0.5, "broadband": 0.5,
}


@dataclasses.dataclass(frozen=True, slots=True)
class ManifestRow:
    """Minimal manifest row consumed by the vb-eh weight computer.

    Fields are the exact subset needed for B02's vb-eh formula — keeping the
    surface tiny so multi-corpus manifests can be projected onto this row type
    without dragging their full schema.

    ``valid_bins`` is the *count* of valid frequency bins for the row's
    corpus (e.g. SWEC k0–k21 → 22; D-cohort + BT k0–k29 → 30; AJILE12 k0–k20
    → 21). It is a per-corpus property keyed at extractor init.
    """

    corpus: str
    n_electrodes: int
    hours: float
    valid_bins: int

    def vb_eh(self) -> float:
        """Per-row vb-eh contribution = ``hours × n_electrodes × valid_bins``."""
        return float(self.hours) * float(self.n_electrodes) * float(self.valid_bins)


def compute_vb_eh_per_corpus(
    rows: tp.Sequence[ManifestRow],
) -> dict[str, float]:
    """Aggregate the vb-eh totals per corpus across the manifest.

    Returns a dict of ``{corpus_name: total_vb_eh}``. Corpora with no rows
    are absent from the dict (the WRS factory builds weights only over the
    corpora that have rows).
    """
    totals: dict[str, float] = {}
    for row in rows:
        totals[row.corpus] = totals.get(row.corpus, 0.0) + row.vb_eh()
    return totals


def _within_broadband_share(
    vb_eh_per_corpus: tp.Mapping[str, float],
    *,
    alpha: float,
) -> dict[str, float]:
    """Within-broadband share ∝ ``vb_eh ** alpha``. Normalised to 1.0."""
    pres = [c for c in BROADBAND_CORPORA if c in vb_eh_per_corpus]
    if not pres:
        return {}
    raw = {c: float(vb_eh_per_corpus[c]) ** float(alpha) for c in pres}
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError(
            "within-broadband total is non-positive; check vb_eh inputs"
        )
    return {c: raw[c] / total for c in pres}


def compute_per_row_weights(
    rows: tp.Sequence[ManifestRow],
    *,
    alpha: float = DEFAULT_ALPHA,
    macro_split: tp.Mapping[str, float] = DEFAULT_MACRO_SPLIT,
) -> np.ndarray:
    """Per-row WRS weights `w[row] = group_share × within_group_share / count_rows[corpus]`.

    Parameters
    ----------
    rows
        Sequence of :class:`ManifestRow` over the cohort being sampled.
    alpha
        Within-broadband temperature. Default ``0.5`` per B02 lock. Sisters
        reach `R-sampler-alpha03` (``α=0.3``), `R-sampler-sqrth` (``α=0.5``,
        same default — kept as an explicit sister entry per `ablations.md
        §6j`), `R-sampler-pure-h` (``α=1.0``), and `R-sampler-uniform`
        (``α=0.0`` — uniform across broadband corpora).
    macro_split
        Macro SWEC / broadband share. Default ``{"swec": 0.5,
        "broadband": 0.5}`` (B02 hard cap). Sisters `R-sampler-40-60` and
        `R-sampler-60-40` tilt the macro.

    Returns
    -------
    ndarray, shape ``(len(rows),)``, dtype ``float64``. Sums to 1.0 over rows
    that contributed a positive within-group share. The
    :class:`WeightedRandomSampler` consumer normalises internally; the
    explicit normalisation here keeps the per-row weights interpretable.
    """
    if not rows:
        raise ValueError("rows must be non-empty")
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0; got {alpha}")
    mac_swec = float(macro_split.get("swec", 0.0))
    mac_broad = float(macro_split.get("broadband", 0.0))
    if mac_swec < 0.0 or mac_broad < 0.0:
        raise ValueError(
            f"macro_split must have non-negative shares; got swec={mac_swec}, "
            f"broadband={mac_broad}"
        )
    if not (mac_swec > 0.0 or mac_broad > 0.0):
        raise ValueError(
            "macro_split must have at least one positive share"
        )

    vb_eh_per_corpus = compute_vb_eh_per_corpus(rows)
    counts_per_corpus: dict[str, int] = {}
    for row in rows:
        counts_per_corpus[row.corpus] = counts_per_corpus.get(row.corpus, 0) + 1

    # If SWEC is absent, its macro share collapses into the broadband side
    # (and vice-versa) — keeps the sampler well-defined on partial cohorts.
    has_swec = vb_eh_per_corpus.get(SWEC_CORPUS, 0.0) > 0.0
    has_broad = any(
        vb_eh_per_corpus.get(c, 0.0) > 0.0 for c in BROADBAND_CORPORA
    )
    if not has_swec:
        mac_broad += mac_swec
        mac_swec = 0.0
    if not has_broad:
        mac_swec += mac_broad
        mac_broad = 0.0
    macro_total = mac_swec + mac_broad
    if macro_total <= 0.0:
        raise ValueError(
            "macro_split has no positive share after collapsing absent "
            "groups; the cohort has neither SWEC nor broadband rows"
        )
    mac_swec /= macro_total
    mac_broad /= macro_total

    within_broad = _within_broadband_share(vb_eh_per_corpus, alpha=alpha)

    # ``corpus_share[c]`` is the target sampling probability for one DRAW
    # falling into corpus ``c`` — group_share × within_group_share.
    corpus_share: dict[str, float] = {}
    if has_swec:
        corpus_share[SWEC_CORPUS] = mac_swec
    for c, w in within_broad.items():
        corpus_share[c] = mac_broad * w

    weights = np.zeros(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        share = corpus_share.get(row.corpus, 0.0)
        n_rows = counts_per_corpus.get(row.corpus, 0)
        if share <= 0.0 or n_rows == 0:
            continue
        weights[i] = share / float(n_rows)

    return weights


def build_wrs_sampler(
    weights: np.ndarray,
    *,
    num_samples: int,
    generator: tp.Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    """Construct :class:`WeightedRandomSampler` with ``replacement=True``.

    B02 spec is explicit on ``replacement=True`` — the macro 50/50 cap +
    within-broadband ``α=0.5`` deliver the desired per-step mixture in
    expectation, and replacement keeps the per-row draw probability stable
    across the epoch (without-replacement would shift the SWEC share as the
    epoch consumes SWEC rows).
    """
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1-d; got shape {weights.shape}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0; got {num_samples}")
    weights_t = torch.as_tensor(weights, dtype=torch.float64)
    if torch.any(weights_t < 0.0):
        raise ValueError("weights must be non-negative")
    if float(weights_t.sum().item()) <= 0.0:
        raise ValueError("weights sum to zero — no row has positive share")
    return WeightedRandomSampler(
        weights=weights_t,
        num_samples=int(num_samples),
        replacement=True,
        generator=generator,
    )


def build_stateful_dataloader(
    dataset: tp.Any,
    *,
    sampler: WeightedRandomSampler,
    batch_size: int,
    num_workers: int = 0,
    persistent_workers: bool = True,
    **kwargs: tp.Any,
) -> tp.Any:
    """Lazy-import wrapper around ``torchdata.stateful_dataloader.StatefulDataLoader``.

    Imports the dependency at call time so this module stays importable on
    development checkouts without ``torchdata``. Raises a clear hint when
    the dependency is missing — the multi-corpus loader build (separate
    blocker bundle) is responsible for installing ``torchdata`` before the
    full B02 stack runs.
    """
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
    except ImportError as exc:  # pragma: no cover — env-dependent
        raise RuntimeError(
            "torchdata.stateful_dataloader is required for the B02 sampler "
            "stack; install with `uv pip install torchdata`. The "
            "WeightedRandomSampler primitive in this module works without "
            "torchdata — only the StatefulDataLoader wrapper needs it."
        ) from exc
    return StatefulDataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        persistent_workers=bool(persistent_workers) and num_workers > 0,
        **kwargs,
    )


__all__ = [
    "ALL_CORPORA",
    "BROADBAND_CORPORA",
    "DEFAULT_ALPHA",
    "DEFAULT_MACRO_SPLIT",
    "ManifestRow",
    "SWEC_CORPUS",
    "build_stateful_dataloader",
    "build_wrs_sampler",
    "compute_per_row_weights",
    "compute_vb_eh_per_corpus",
]
