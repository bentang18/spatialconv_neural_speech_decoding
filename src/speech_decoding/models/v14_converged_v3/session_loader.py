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
    assert_band_rates_match_cache,
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

    ``keep_labels_fn`` (opt): ``(subject_id, trial_id, labels) -> set[str]`` naming the
    electrodes to KEEP; everything else joins the LOF drop set. Injected like ``parcel_fn``
    so study-specific montages (e.g. the Neuroprobe-Lite electrode list) stay out of this
    loader. Restricting here rather than subsetting the built spec is what makes it safe:
    ``keep_idx``/``parcel_id``/``sidecar``/``geom``/``band_stats`` are then all constructed
    on the restricted axis by the same code path as a normal run, so none of them can
    desync. (``geom`` in particular CANNOT be masked post-hoc — ``gather_idx`` stores
    indices INTO the survivor axis, so dropping survivors invalidates every stored index.)
    ``None`` ⇒ keep everything (the training path; byte-identical to no argument)."""
    # band-count agnostic: r4/arm0 pass 3 (slow, mid, hga); r5 passes 2 (v3hga, v3lfs).
    # dirs must align to rates; everything below zips over band_indexes.
    if len(band_cache_dirs) != len(band_rates):
        raise ValueError(
            f"band_cache_dirs ({len(band_cache_dirs)}) must align with band_rates "
            f"({len(band_rates)}); pass 3 for r4/arm0, 2 for r5"
        )
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
        setup = build_session_setup(
            labels, parcel_id, drop_labels=drop, mask_cfg=mask_cfg,
        )
        band_stats = [_load_stats(e.stats_path) for e in entries]
        # the declared read rates must match what was actually baked, else every per-band
        # window is a compressed, time-shifted slice at the RIGHT shape (r6 2026-07-23).
        assert_band_rates_match_cache(
            [e.frame_rate_hz for e in entries],
            band_rates,
            where=f"session {subject_id}/{trial_id}",
        )
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
