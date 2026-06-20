"""Raw-feature logistic baseline on the Neuroprobe-Lite EVAL cells (piece 4).

Pieces 2/3 (:mod:`online_probe_raw_baseline`, :mod:`offline_probe_bench`) measure
the raw 3STFT ``|STFT|`` logistic floor on the PRETRAIN cohort. This measures the
SAME floor on the official Neuroprobe-Lite EVAL cells (``BT_LITE_SESSIONS``), across
the three leaderboard eval modes, so the encoder's eventual lite numbers land on an
upstream-parity raw baseline — same StandardScaler + L2 ``LogisticRegression``
(C=1.0) as the "logistic-on-Multi-STFT 0.663" line, same raw tokens (30 / electrode,
d=1), same contiguous-fold protocol:

  - **WithinSession**: each lite cell scored alone, per-electrode (``C·30`` features),
    contiguous 2-fold (``online_probe.ws_auroc_2fold`` protocol).
  - **CrossSession**: per subject, fit one trial / score the other (BOTH directions,
    averaged), per-electrode on the electrodes valid in BOTH trials. Guard-1 STATIC
    is per-session, so two trials of one subject can drop different electrodes — the
    per-electrode fit must use the ``mask_a & mask_b`` intersection (same montage,
    same column order) or the feature dims disagree.
  - **CrossSubject**: fit the ``(2, 4)`` anchor / score every other-subject lite cell,
    electrode→DKT-parcel pooled on the shared parcel set (the only cross-montage
    coordinate). Upstream leaderboard default train cell = ``(2, 4)``.

The scoring/feature/pooling/intersection primitives are imported unchanged from
:mod:`linear_probe_logistic` / :mod:`online_probe` / :mod:`online_probe_raw_baseline`;
this module adds only the three eval-mode loops (pure numpy/torch, laptop-TDD'd in
``test_lite_eval_raw_baseline``) and a DCC-only per-cell materializer that reuses the
probe segmenter. Lite cells are session-DISJOINT from pretrain (the §6 firewall), so
this path needs its OWN 3STFT lite spec cache (built CPU-side, like the 2STFT one).
"""

from __future__ import annotations

import os
import typing as tp
from collections import defaultdict

import numpy as np

from speech_decoding.experiments.linear_probe_logistic import (
    cs_auroc_logistic,
    ws_auroc_2fold_logistic,
)
from speech_decoding.experiments.online_probe import (
    SubjectProbeData,
    _finite_rows,
    feature_matrix,
    parcel_intersection,
)
from speech_decoding.experiments.online_probe_raw_baseline import (
    feature_matrix_per_electrode,
    pool_electrodes_to_parcels,
    raw_tokens_from_bands,
)
from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS

if tp.TYPE_CHECKING:  # avoid importing the heavy Data/ns chain on the laptop test path
    from speech_decoding.experiments.data import Data

# Default upstream leaderboard CrossSubject train cell (DS_DM_TRAIN_SUBJECT_ID=2 /
# DS_DM_TRAIN_TRIAL_ID=4) — see word_events._assign_cross_subject_split.
DEFAULT_TRAIN_CELL: tuple[int, int] = (2, 4)

Cell = SubjectProbeData
Cells = dict[tuple[int, int], SubjectProbeData]

__all__ = [
    "DEFAULT_TRAIN_CELL",
    "build_lite_eval_cells",
    "n_parcels_of",
    "run_cross_session",
    "run_cross_subject",
    "run_lite_eval_raw_baseline",
    "run_within_session",
]


def n_parcels_of(cells: Cells) -> int:
    """Parcel-vocabulary width = max DKT parcel id over every cell + 1. The CS pool
    indexes a fixed table, so it must cover every cell's ids (``_support_width``)."""
    return max(int(c.parcel_per_electrode.max().item()) for c in cells.values()) + 1


def _raw(cell: SubjectProbeData):
    return raw_tokens_from_bands(cell.slow, cell.beta, cell.hg)


def run_within_session(
    cells: Cells, *, tasks: tp.Sequence[str], max_iter: int = 10000
) -> tuple[dict[str, float], dict[tuple[int, int, str], float]]:
    """Per lite cell, per-electrode contiguous-2-fold logistic AUROC, mean over cells.

    Returns ``(summary, per_cell)`` where ``summary`` is
    ``{val_lite/raw/within_session/<task>: nanmean}`` and ``per_cell`` keys are
    ``(subject, trial, task)``."""
    summary: dict[str, float] = {}
    per_cell: dict[tuple[int, int, str], float] = {}
    for task in tasks:
        vals: list[float] = []
        for (s, t), cell in sorted(cells.items()):
            z = feature_matrix_per_electrode(_raw(cell), cell.electrode_mask).numpy()
            zf, yf = _finite_rows(z, cell.labels[task])
            v = ws_auroc_2fold_logistic(zf, yf, max_iter=max_iter) if len(yf) >= 4 else float("nan")
            vals.append(v)
            per_cell[(s, t, task)] = v
        summary[f"val_lite/raw/within_session/{task}"] = (
            float(np.nanmean(vals)) if vals else float("nan")
        )
    return summary, per_cell


def run_cross_session(
    cells: Cells, *, tasks: tp.Sequence[str], max_iter: int = 10000
) -> tuple[dict[str, float], dict[tuple[int, int, int, str], float]]:
    """Per subject with ≥2 lite trials, fit one trial / score the other (both
    directions), per-electrode on the BOTH-trials-valid montage; mean over
    (subject, direction). ``per_cell`` keys are ``(subject, train_trial, test_trial,
    task)``."""
    import torch

    by_subj: dict[int, list[int]] = defaultdict(list)
    for s, t in cells:
        by_subj[s].append(t)

    summary: dict[str, float] = {}
    per_cell: dict[tuple[int, int, int, str], float] = {}
    for task in tasks:
        vals: list[float] = []
        for s, trials in sorted(by_subj.items()):
            trials = sorted(trials)
            if len(trials) < 2:
                continue
            for ta in trials:
                for tb in trials:
                    if ta == tb:
                        continue
                    A, B = cells[(s, ta)], cells[(s, tb)]
                    shared = A.electrode_mask.bool() & B.electrode_mask.bool()
                    if not bool(torch.any(shared)):
                        per_cell[(s, ta, tb, task)] = float("nan")
                        vals.append(float("nan"))
                        continue
                    za = feature_matrix_per_electrode(_raw(A), shared).numpy()
                    zb = feature_matrix_per_electrode(_raw(B), shared).numpy()
                    zaf, yaf = _finite_rows(za, A.labels[task])
                    zbf, ybf = _finite_rows(zb, B.labels[task])
                    if len(yaf) < 2 or len(ybf) < 1:
                        v = float("nan")
                    else:
                        v = cs_auroc_logistic(zaf, yaf, zbf, ybf, max_iter=max_iter)
                    per_cell[(s, ta, tb, task)] = v
                    vals.append(v)
        summary[f"val_lite/raw/cross_session/{task}"] = (
            float(np.nanmean(vals)) if vals else float("nan")
        )
    return summary, per_cell


def run_cross_subject(
    cells: Cells,
    *,
    tasks: tp.Sequence[str],
    n_parcels: int,
    train_cell: tuple[int, int] = DEFAULT_TRAIN_CELL,
    max_iter: int = 10000,
) -> tuple[dict[str, float], dict[tuple[int, int, str], float]]:
    """Fit the ``train_cell`` anchor / score every lite cell whose subject differs,
    parcel-pooled on the anchor∩test parcel intersection; mean over test cells.
    ``per_cell`` keys are ``(test_subject, test_trial, task)``."""
    if train_cell not in cells:
        raise KeyError(
            f"cross-subject train cell {train_cell} absent from lite cells "
            f"{sorted(cells)}"
        )
    a = cells[train_cell]
    pa, pres_a = pool_electrodes_to_parcels(
        _raw(a), a.parcel_per_electrode, a.electrode_mask, n_parcels
    )
    train_subj = train_cell[0]

    pooled: dict[tuple[int, int], tuple] = {}
    for (s, t), cell in cells.items():
        if s == train_subj:
            continue
        pooled[(s, t)] = pool_electrodes_to_parcels(
            _raw(cell), cell.parcel_per_electrode, cell.electrode_mask, n_parcels
        )

    summary: dict[str, float] = {}
    per_cell: dict[tuple[int, int, str], float] = {}
    for task in tasks:
        vals: list[float] = []
        for (s, t), (pt, pres_t) in sorted(pooled.items()):
            inter = parcel_intersection(pres_a, pres_t)
            if inter.numel() == 0:
                v = float("nan")
            else:
                za, ya = _finite_rows(feature_matrix(pa, inter).numpy(), a.labels[task])
                zt, yt = _finite_rows(feature_matrix(pt, inter).numpy(), cells[(s, t)].labels[task])
                if len(ya) < 2 or len(yt) < 1:
                    v = float("nan")
                else:
                    v = cs_auroc_logistic(za, ya, zt, yt, max_iter=max_iter)
            per_cell[(s, t, task)] = v
            vals.append(v)
        summary[f"val_lite/raw/cross_subject/{task}"] = (
            float(np.nanmean(vals)) if vals else float("nan")
        )
    return summary, per_cell


def run_lite_eval_raw_baseline(
    cells: Cells,
    *,
    tasks: tp.Sequence[str],
    n_parcels: int | None = None,
    train_cell: tuple[int, int] = DEFAULT_TRAIN_CELL,
    max_iter: int = 10000,
) -> dict[str, tp.Any]:
    """All three eval modes over the lite cells. Returns
    ``{metrics, per_cell:{within_session,cross_session,cross_subject}}`` — ``metrics``
    is the flat ``val_lite/raw/<mode>/<task>`` dict, ``per_cell`` keeps the cell-level
    AUROCs so nothing is hidden behind the means."""
    if n_parcels is None:
        n_parcels = n_parcels_of(cells)
    ws, ws_pc = run_within_session(cells, tasks=tasks, max_iter=max_iter)
    cs, cs_pc = run_cross_session(cells, tasks=tasks, max_iter=max_iter)
    csub, csub_pc = run_cross_subject(
        cells, tasks=tasks, n_parcels=n_parcels, train_cell=train_cell, max_iter=max_iter
    )
    return {
        "metrics": {**ws, **cs, **csub},
        "per_cell": {
            "within_session": {f"{s},{t},{task}": v for (s, t, task), v in ws_pc.items()},
            "cross_session": {
                f"{s},{ta}->{tb},{task}": v for (s, ta, tb, task), v in cs_pc.items()
            },
            "cross_subject": {f"{s},{t},{task}": v for (s, t, task), v in csub_pc.items()},
        },
        "n_parcels": int(n_parcels),
        "train_cell": list(train_cell),
        "n_cells": len(cells),
    }


def build_lite_eval_cells(
    lite_data: "Data",
    *,
    n_cap: int,
    seed: int = 0,
    tasks: tp.Sequence[str],
    batch_size: int = 256,
) -> Cells:  # pragma: no cover - DCC-only (needs BT voltage + the lite spec cache)
    """Materialize each Neuroprobe-Lite cell's 1 s word-onset band tensors + anatomy
    + ±1 labels from a LITE :class:`Data` (``Wang2024Treebank(mode="lite")`` study +
    the 3STFT lite spec cache).

    Mirrors :func:`online_probe_dataset.build_probe_dataset` but PER CELL (each
    ``(subject, trial)`` is its own train/test unit in the eval modes) and with the
    §6 firewall INVERTED — every present cell MUST be a ``BT_LITE_SESSIONS`` member
    (the lite cells are the point here, not a leak). Reuses the probe segmenter
    (1 s / zero-lag), the CLIP bad-window filter, the enriched-words est_idx join,
    and ``_materialize_subject`` unchanged, so the band tensors are byte-identical to
    the online probe's — only the corpus (lite vs pretrain) and the per-cell grouping
    differ. DCC-only."""
    import numpy as _np

    from speech_decoding.experiments.online_probe_dataset import (
        _load_enriched_words,
        _materialize_subject,
        _probe_segmenter,
        _subject_id_int,
        filter_probe_events,
        pm1_labels,
        select_window_indices,
    )

    segmenter = _probe_segmenter(lite_data)
    events = lite_data.study.run()
    events = filter_probe_events(events, lite_data.bad_window_dir)
    dataset = segmenter.apply(events)
    dataset.prepare()
    triggers = dataset.triggers

    sid = _subject_id_int(triggers["subject_id"])
    tid = _np.asarray(triggers["trial_id"]).astype(int)
    present_cells = sorted({(int(s), int(t)) for s, t in zip(sid, tid)})
    lite = {tuple(c) for c in BT_LITE_SESSIONS}
    leaked = [c for c in present_cells if c not in lite]
    if leaked:
        raise AssertionError(
            f"lite-eval baseline saw non-lite cells {leaked}; the lite Data must emit "
            f"only BT_LITE_SESSIONS"
        )

    enriched_words = _load_enriched_words(
        present_cells, bt_root=os.environ.get("ROOT_DIR_BRAINTREEBANK")
    )

    cells: Cells = {}
    for s, t in present_cells:
        pos = _np.flatnonzero((sid == s) & (tid == t))
        positions = pos[select_window_indices(len(pos), n_cap, seed)]
        if len(positions) == 0:
            continue
        rec = _materialize_subject(
            dataset, triggers, positions, batch_size=batch_size,
            enriched_words=enriched_words,
        )
        cells[(s, t)] = SubjectProbeData(
            subject_id=s,
            slow=rec["slow"], beta=rec["beta"], hg=rec["hg"],
            parcel_per_electrode=rec["parcel_per_electrode"],
            electrode_mask=rec["electrode_mask"],
            labels={task: pm1_labels(rec["words_df"], task) for task in tasks},
            sessions=[(s, t)],
        )
    if not cells:
        raise RuntimeError("lite-eval baseline materialized no cells from the lite Data")
    return cells
