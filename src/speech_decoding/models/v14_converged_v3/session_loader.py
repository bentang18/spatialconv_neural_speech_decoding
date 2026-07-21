"""v14_converged_v3 — load_v3_sessions orchestrator (Phase D1/D2 adapter).

Composes the tested parsers (``cache_index``) + the tested pure cores
(``build_session_setup`` / ``build_session_spec``) + the braintreebank parcel
lookup into the ``list[V3SessionSpec]`` the ``V3DataModule`` consumes.

For each requested ``(subject_id, trial_id)``:
  1. match its ``.npy``/``.stats.npz`` in each of the 3 band cache dirs (by the
     spec sidecar ``key``); the band ``ch_names`` is the FULL voltage order;
  2. resolve the DKT hard ``parcel_id`` over those labels (``parcel_fn`` — the real
     BT ``aligned_voltage_support`` call, injectable so the orchestration is unit-
     testable with a stub);
  3. guard-1 drop = LOF bad set (``extra_bad`` is already gone from ``ch_names`` —
     a harmless no-op if re-passed) → ``build_session_setup`` (sidecar + geom +
     feasibility assert, #36);
  4. load each band's frozen robust-z stats and the guard-2 spans →
     ``build_session_spec`` (survivor-sliced normalizers).

The only F2-validated seam is the default ``parcel_fn`` (one BT-anatomy call);
everything else is exercised locally with synthetic caches + a stub parcel_fn.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch import Tensor

from speech_decoding.models.v14_converged_v3.cache_index import (
    BandCacheEntry,
    index_band_cache,
    index_bad_windows,
    parse_key_session,
    parse_lof_report,
)
from speech_decoding.models.v14_converged_v3.dataset import (
    UNIFORM_BAND_RATES,
    V3SessionSpec,
    build_session_spec,
    reference_n_frames,
)
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig
from speech_decoding.models.v14_converged_v3.session_setup import build_session_setup

ParcelFn = Callable[[int, int, Sequence[str]], Tensor]
KeepLabelsFn = Callable[[int, int, Sequence[str]], set[str]]


def _entry_for(index: dict[str, BandCacheEntry], subject_id: int, trial_id: int) -> BandCacheEntry:
    """The unique cache entry for a session, matched on the sidecar ``key``.

    ``key`` is the real producer uid (nested JSON with ``"subject_id":S`` /
    ``"trial_id":T`` inside ``timeline``); ``parse_key_session`` extracts the two
    ids (full-integer, so S=1 never aliases 12). Exactly one entry per session per
    band leaf (one continuous whole-movie cache), so >1 or 0 is a fail-loud."""
    hits = [e for k, e in index.items() if parse_key_session(k) == (subject_id, trial_id)]
    if len(hits) != 1:
        raise ValueError(
            f"expected exactly one cache entry for subject {subject_id} trial "
            f"{trial_id}, found {len(hits)}"
        )
    return hits[0]


def _load_stats(stats_path: str) -> tuple[Tensor, Tensor]:
    z = np.load(stats_path)
    return torch.from_numpy(z["median"]).float(), torch.from_numpy(z["sigma"]).float()


def _load_state_stats(
    state_stats_dir: str | None, subject_id: int
) -> tuple[Tensor | None, Tensor | None]:
    """Per-SUBJECT frozen state-norm table for the secondary Gaussian-NLL.

    ``<state_stats_dir>/sub-<subject_id>.npz`` holds ``stat_mean``/``stat_std``, each
    (n_parcels, 6) indexed by parcel id VALUE — the same index space as the perceiver's
    parcel embedding and ``raw_state_vectors``. ``build_session_setup`` gathers the rows
    for this session's survivor parcels and fails loud on shape/coverage. Stats are frozen
    per SUBJECT (over the subject's train clips), so all of a subject's sessions share one
    table. ``None`` dir ⇒ secondary OFF (JEPA-only)."""
    if state_stats_dir is None:
        return None, None
    path = os.path.join(state_stats_dir, f"sub-{subject_id}.npz")
    z = np.load(path)
    return (
        torch.from_numpy(z["stat_mean"]).float(),
        torch.from_numpy(z["stat_std"]).float(),
    )


def load_v3_sessions(
    *,
    sessions: Sequence[tuple[int, int]],
    band_cache_dirs: Sequence[str],  # 3 dirs in v3 concat order (slow, mid, hga)
    span_dir: str,
    parcel_fn: ParcelFn,
    lof_report_path: str | None = None,
    sigma_floor: float = 1e-6,
    winsor: float | Sequence[float] | None = None,
    mask_cfg: V3MaskConfig = V3MaskConfig(),
    state_stats_dir: str | None = None,
    keep_labels_fn: KeepLabelsFn | None = None,
    band_rates: Sequence[tuple[int, int]] = UNIFORM_BAND_RATES,
) -> list[V3SessionSpec]:
    """Assemble the per-session ``V3SessionSpec`` list for the datamodule.

    ``band_rates`` (opt): per-band read rate (num, den) vs the 32 Hz clip clock, in the
    (slow, mid, hga) cache order. Default uniform ((1,1)×3) = arm0 (all bands cached at
    32 Hz). Native fine-HGA passes ((1,8),(1,2),(4,1)); each band cache then carries its
    native frame count and the spec's ``n_frames`` is the derived 32 Hz reference. The
    dataset MUST be built with the same ``band_rates`` (memo
    project-fine-hga-bt-rebake-tasklist-2026-07-21).

    ``state_stats_dir`` (opt): dir of per-subject frozen state-norm tables that turn ON
    the secondary Gaussian-NLL; omit ⇒ JEPA-only.

    ``keep_labels_fn`` (opt): ``(subject_id, trial_id, labels) -> set[str]`` naming the
    electrodes to KEEP; everything else joins the LOF drop set. Injected like ``parcel_fn``
    so study-specific montages (e.g. the Neuroprobe-Lite electrode list) stay out of this
    loader. Restricting here rather than subsetting the built spec is what makes it safe:
    ``keep_idx``/``parcel_id``/``sidecar``/``geom``/``band_stats`` are then all constructed
    on the restricted axis by the same code path as a normal run, so none of them can
    desync. (``geom`` in particular CANNOT be masked post-hoc — ``gather_idx`` stores
    indices INTO the survivor axis, so dropping survivors invalidates every stored index.)
    ``None`` ⇒ keep everything (the training path; byte-identical to no argument)."""
    if len(band_cache_dirs) != 3:
        raise ValueError(f"expected 3 band cache dirs, got {len(band_cache_dirs)}")
    if len(band_rates) != 3:
        raise ValueError(f"expected 3 band_rates, got {len(band_rates)}")
    band_indexes = [index_band_cache(d) for d in band_cache_dirs]
    bad_idx = index_bad_windows(span_dir)
    lof = parse_lof_report(lof_report_path)

    specs: list[V3SessionSpec] = []
    for subject_id, trial_id in sessions:
        entries = [_entry_for(bi, subject_id, trial_id) for bi in band_indexes]
        ch0 = entries[0].ch_names
        for e in entries[1:]:
            if e.ch_names != ch0:
                raise ValueError(
                    f"band caches disagree on channel order for session "
                    f"{subject_id}/{trial_id}: {e.ch_names[:3]}... vs {ch0[:3]}..."
                )
        labels = list(ch0)
        parcel_id = parcel_fn(subject_id, trial_id, labels).long()
        drop = set(lof.get((subject_id, trial_id), set()))
        if keep_labels_fn is not None:
            keep = keep_labels_fn(subject_id, trial_id, labels)
            restricted = {lab for lab in labels if lab not in keep}
            if len(labels) - len(restricted | drop) == 0:
                raise ValueError(
                    f"session {subject_id}/{trial_id}: keep_labels_fn kept 0 of "
                    f"{len(labels)} electrodes — montage does not match this cache"
                )
            drop |= restricted
        stat_mean, stat_std = _load_state_stats(state_stats_dir, subject_id)
        setup = build_session_setup(
            labels, parcel_id, drop_labels=drop, mask_cfg=mask_cfg,
            stat_mean=stat_mean, stat_std=stat_std,
        )
        band_stats = [_load_stats(e.stats_path) for e in entries]
        n_frames_32 = reference_n_frames(
            [e.total_frames for e in entries], band_rates
        )
        specs.append(
            build_session_spec(
                session_key=(subject_id, trial_id),
                band_paths=tuple(e.npy_path for e in entries),
                band_stats=tuple(band_stats),
                setup=setup,
                n_frames=n_frames_32,
                bad_spans_s=bad_idx.get((subject_id, trial_id), []),
                sigma_floor=sigma_floor,
                winsor=winsor,
            )
        )
    return specs
