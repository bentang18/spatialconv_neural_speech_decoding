"""Offline producer of the per-subject FROZEN state-norm tables for the r4 secondary
Gaussian-NLL head (contract project-r4-contract-2026-07-15 §7).

Writes ``<out-dir>/sub-<subject_id>.npz`` (arrays ``stat_mean``/``stat_std``, each
(n_parcels, 6) indexed by parcel-id VALUE) — the exact schema
``session_loader._load_state_stats`` reads and the r4 launch turns on via
``--state-stats-dir``. Without these files the secondary NLL cannot run.

It draws the SAME train clips the datamodule draws (reuses ``load_v3_sessions`` +
``V3SessionDataset`` + ``make_bt_parcel_fn`` from dispatch_v3, so the robust-z bands are
byte-identical to training), runs the model-free ``raw_state_vectors`` reduction, and
pools per-(parcel VALUE, dim) mean/std across ALL of a subject's sessions via
``StateStatsAccumulator``. Common-mode removal is NOT applied here (it happens at
consumption, in z-scored coords). Model-free, CPU — run on a login node / CPU job where
the band caches live, never a GPU allocation.

Example (one subject, its sessions listed together)::

    python -m scripts.neuroprobe.build_state_stats \
        --band-cache-dir CACHE_SLOW --band-cache-dir CACHE_MID --band-cache-dir CACHE_HGA \
        --span-dir SPANS --lof-report-path LOF.json --bt-root BT_ROOT \
        --session 6:0 --session 6:1 --clips-per-session 2000 --clip-len 3.0 \
        --out-dir STATE_STATS_DIR
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.dataset import V3SessionDataset, V3SessionSpec
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from speech_decoding.models.v14_converged_v3.state_target import (
    StateStatsAccumulator,
    raw_state_vectors,
)

_FPS = 32.0  # v3 uniform frame clock (must match dispatch_v3._FPS)


def _parse_sessions(raw: list[str]) -> list[tuple[int, int]]:
    out = []
    for item in raw:
        s, _, t = item.partition(":")
        if not t:
            raise ValueError(f"--session must be S:T (subject:trial), got {item!r}")
        out.append((int(s), int(t)))
    return out


def _accumulate_subject(
    specs: list[V3SessionSpec],
    *,
    clips_per_session: int,
    clip_frames: int,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pool one subject's sessions → (stat_mean (V,6), stat_std (V,6), n_samples)."""
    acc = StateStatsAccumulator()
    max_parcel = 0
    n_clips = 0
    for spec in specs:
        max_parcel = max(max_parcel, int(spec.setup.parcel_id.max()))
        ds = V3SessionDataset(
            [spec],
            clips_per_session=clips_per_session,
            clip_frames=clip_frames,
            fps=_FPS,
            seed=seed,
        )
        for epoch in range(epochs):
            ds.set_epoch(epoch)
            for index in range(len(ds)):
                sample = ds[index]
                band_inputs = [b.unsqueeze(0) for b in sample.bands]  # (1, N, F_b, T)
                raw, parcels, _ = raw_state_vectors(
                    band_inputs, sample.setup.parcel_id
                )
                acc.add(raw, parcels)
                n_clips += 1
    mean, std = acc.finalize(n_parcels=max_parcel + 1)
    return mean.numpy(), std.numpy(), n_clips


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append",
                   required=True, metavar="DIR",
                   help="one per band in v3 concat order (slow, mid, hga); pass 3×")
    p.add_argument("--span-dir", required=True, help="guard-2 bad-window span JSONs")
    p.add_argument("--lof-report-path", default=None, help="guard-1 LOF report (opt)")
    p.add_argument("--bt-root", required=True, help="braintreebank anatomy root")
    p.add_argument("--session", dest="sessions", action="append", required=True,
                   metavar="S:T", help="subject:trial, repeatable (e.g. 6:0)")
    p.add_argument("--clips-per-session", type=int, default=2000,
                   help="clips drawn per session for the estimate (match training epoch)")
    p.add_argument("--clip-len", type=float, default=3.0, help="clip seconds (match launch)")
    p.add_argument("--epochs", type=int, default=1,
                   help="re-draw windows this many times for a larger sample (default 1)")
    p.add_argument("--session-z-winsor-slow", dest="winsor_slow", type=float, default=15.0)
    p.add_argument("--session-z-winsor-mid", dest="winsor_mid", type=float, default=15.0)
    p.add_argument("--session-z-winsor-hga", dest="winsor_hga", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out-dir", required=True, help="dir for sub-<id>.npz tables")
    args = p.parse_args(argv)

    if len(args.band_cache_dirs) != 3:
        raise ValueError(
            f"need 3 --band-cache-dir (slow, mid, hga), got {len(args.band_cache_dirs)}"
        )
    clip_frames = round(args.clip_len * _FPS)
    os.makedirs(args.out_dir, exist_ok=True)

    specs = load_v3_sessions(
        sessions=_parse_sessions(args.sessions),
        band_cache_dirs=args.band_cache_dirs,
        span_dir=args.span_dir,
        parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=args.lof_report_path,
        winsor=(args.winsor_slow, args.winsor_mid, args.winsor_hga),
        state_stats_dir=None,  # producing them — do NOT try to load
    )

    by_subject: dict[int, list[V3SessionSpec]] = defaultdict(list)
    for spec in specs:
        by_subject[int(spec.session_key[0])].append(spec)

    for subject_id, subj_specs in sorted(by_subject.items()):
        mean, std, n_clips = _accumulate_subject(
            subj_specs,
            clips_per_session=args.clips_per_session,
            clip_frames=clip_frames,
            epochs=args.epochs,
            seed=args.seed,
        )
        out_path = os.path.join(args.out_dir, f"sub-{subject_id}.npz")
        np.savez(out_path, stat_mean=mean, stat_std=std)
        covered = int((np.abs(mean).sum(axis=1) > 0).sum())
        print(
            f"[sub-{subject_id}] sessions={len(subj_specs)} clips={n_clips} "
            f"table={mean.shape} parcels_covered={covered} "
            f"mean|range=[{mean.min():.3f},{mean.max():.3f}] "
            f"std|range=[{std.min():.3f},{std.max():.3f}] -> {out_path}"
        )


if __name__ == "__main__":
    main()
