#!/usr/bin/env python
"""Pretrain probe suite dispatch — raw-371 floor FIRST (model-free, CPU), then encoder taps.

The probe is a leaderboard-FAITHFUL proxy run ENTIRELY on the firewall-legal pretrain
cohort (never ``BT_LITE_SESSIONS``). This script is the BT-data dispatch around the pure,
unit-tested ``pretrain_probe_*`` modules.

PHASE 0 (this file's first job, Ben 2026-06-30) — the **raw-371 sanity gate**: per session,
materialize the run's 2-band |STFT| at the balanced word windows, flatten every bin
(D_raw=371 @1 s), fit the linear ridge for all 15 tasks (WithinSession + CrossSubject) and
write the ledger. NO checkpoint, NO GPU. If the per-task values land in the Neuroprobe
leaderboard linear-baseline neighborhood, the materialize→label→split→score chain is proven
end-to-end before any encoder forward.

PHASE 1 (later, GPU) — load ``ladder-60000`` and add the M2/M3/M4 encoder-tap readouts +
the attentive sweep, off the SAME cached windows.

Faithful ``xp`` build: Run A's exact eval-relevant argv is parsed by ``dispatch_v14._parser``
→ ``_common_build_kwargs`` → ``build_v14_experiment`` so the 2-band segmenter (hops, freq
bins, robust-z winsor, spec cache) is byte-identical to the run's. Materialization mirrors
``v2_probe_dataset._materialize_subject_v2`` but triggers on a nonverbal-INCLUSIVE query
(onset/speech/word_gap need nonverbal anchors → the full 15 tasks, not the bench's ≤12) and
forwards EXACTLY the union word windows (est_idx-aligned, fail-loud on any mismatch).

DeltaAI: login node for Phase 0 (model-free), GPU only for Phase 1.

    ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank \
    EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai \
    .venv/bin/python scripts/neuroprobe/run_pretrain_probe_suite.py \
        --preflight                      # one session, a few tasks, print shapes + AUROC
        # --full --out reports/neuroprobe_probe_results.csv
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import numpy as np
import pandas as pd
import torch

# Run A's encoder/data arch (from launch_runA_60k_fixed.sbatch). Only the data-relevant
# flags matter for the raw floor; the predictor/mask/LR/tube flags are SSL-only but are
# kept verbatim so the parser + _common_build_kwargs see exactly the run's namespace.
RUN_A_ARGV: list[str] = [
    "--phase", "1", "--mode", "full", "--frontend", "2band", "--atlas", "dkt",
    "--d-model", "384", "--n-heads", "6",
    "--converged-frontend-layers", "6", "--converged-latent-layers", "12",
    "--converged-v2-pred-dim", "192",
    "--converged-v2-m2-pred-layers", "6", "--converged-v2-m4-pred-layers", "12",
    "--converged-v2-m3-pred-layers", "6", "--converged-v2-qk-norm",
    "--converged-v2-w-m2", "1.0", "--converged-v2-w-m3", "1.0", "--converged-v2-w-m4", "1.0",
    "--converged-v2-support-weight", "--converged-v2-k", "2",
    "--converged-tube-ratio", "0.25", "--clip-len", "5.0",
    "--lr", "6e-3", "--weight-decay", "0.04", "--grad-clip", "3.0",
    "--lr-schedule", "warmup_cosine", "--min-lr-ratio", "1.0", "--warmup-steps", "3000",
    "--ema-tau", "0.99925", "--adam-beta2", "0.95", "--seed", "33",
    "--batch-size", "16", "--accumulate-grad-batches", "4",
    "--session-z-winsor-lfs", "15", "--session-z-winsor-hga", "20",
    "--bad-window-dir", "/projects/bhqk/htang13/v14_bad_windows_2band",
    "--spec-only", "--spec-cache-dir", "/work/nvme/bhqk/htang13/v14_2band_v2_spec_pretrain",
    "--trial-durations", "/projects/bhqk/htang13/v14_trial_durations.json",
]

PROBE_CLIP_DUR_S = 1.0


def _build_xp():
    """Build the run's experiment so ``xp.data`` carries Run A's exact 2-band segmenter."""
    from speech_decoding.experiments.dispatch_v14 import (
        _common_build_kwargs,
        _parser,
        _resolve_static_forward_cohesion,
        build_v14_experiment,
    )

    args = _parser().parse_args(RUN_A_ARGV)
    _resolve_static_forward_cohesion(args)
    common = _common_build_kwargs(args)
    # The eval probe forwards 1 s clips; the SSL phase-1 data chain carries the 2-band
    # segmenter + pretrain corpus we need (we only ever read xp.data, never train).
    xp = build_v14_experiment(
        **common, joint_phase=True, jepa_phase="p1", clip_len=1.0,
    )
    return xp


def _ieeg_index(xp) -> dict[tuple[int, int], pd.Series]:
    """Map (subject,trial) -> its Ieeg row (recording source + timeline) from the corpus."""
    events = xp.data.study.run()
    ieeg = events.loc[events["type"] == "Ieeg"]
    out: dict[tuple[int, int], pd.Series] = {}
    for _, row in ieeg.iterrows():
        out[(int(row["subject_id"]), int(row["trial_id"]))] = row
    return out


def _label_events(subject_id: int, trial_id: int, timeline: str, tasks, bt_root) -> pd.DataFrame:
    """Balanced, all-task word-event rows for ONE pretrain session (full electrodes)."""
    from speech_decoding.studies.braintreebank.word_events import (
        _load_neural_to_movie_map,
        _load_pitch_volume_features,
        _load_words_and_nonverbal,
        _tasks_need_pitch_volume,
        _word_event_rows,
    )

    words_df, nonverbal_df = _load_words_and_nonverbal(
        subject_id, trial_id, bt_root=bt_root, enrich=True
    )
    neural_to_movie = _load_neural_to_movie_map(subject_id, trial_id, bt_root)
    pvf = (
        _load_pitch_volume_features(subject_id, trial_id)
        if _tasks_need_pitch_volume(tasks) else None
    )
    return _word_event_rows(
        subject_id=subject_id, trial_id=trial_id, timeline=timeline,
        words_df=words_df, nonverbal_df=nonverbal_df, tasks=tuple(tasks),
        binary_tasks=True, lite=False, nano=False, random_seed=42,
        duration=PROBE_CLIP_DUR_S, balance=True,
        pitch_volume_features=pvf, neural_to_movie=neural_to_movie,
    )


def _materialize_bands(xp, ieeg_row: pd.Series, union_starts: np.ndarray, *, batch_size=256):
    """Forward EXACTLY the union word windows through the run's 2-band segmenter.

    Triggers on a nonverbal-INCLUSIVE query (full 15 tasks) and fails loud if the
    materialized window starts don't match the requested union axis (clock/alignment guard).
    Returns (bands=[lfs,hga], parcel_per_electrode (C,), electrode_mask (C,), n_parcels)."""
    import neuralset as ns
    from torch.utils.data import DataLoader

    from speech_decoding.experiments.v2_probe_dataset import _PROBE_SEGMENTER_KEYS_V2

    subject_id = int(ieeg_row["subject_id"])
    trial_id = int(ieeg_row["trial_id"])
    timeline = str(ieeg_row["timeline"])
    union_words = pd.DataFrame({
        "type": "Word",
        "subject_id": str(subject_id),
        "trial_id": str(trial_id),
        "timeline": timeline,
        "text": "<union>",
        "start": np.asarray(union_starts, dtype=float),
        "duration": PROBE_CLIP_DUR_S,
    })
    events = pd.concat([ieeg_row.to_frame().T, union_words], ignore_index=True)

    src = xp.data.segmenter.extractors
    seg = ns.dataloader.Segmenter(
        extractors={k: src[k] for k in _PROBE_SEGMENTER_KEYS_V2},
        trigger_query="type == 'Word'",          # nonverbal-INCLUSIVE → all 15 tasks
        start=0.0, duration=PROBE_CLIP_DUR_S,
    )
    dataset = seg.apply(events)
    dataset.prepare()
    got = np.sort(dataset.triggers["start"].to_numpy(dtype=float))
    want = np.sort(np.asarray(union_starts, dtype=float))
    if got.shape != want.shape or not np.allclose(got, want, atol=1e-6):
        raise RuntimeError(
            f"materialized windows != union axis for ({subject_id},{trial_id}): "
            f"got {got.shape} want {want.shape}; the segmenter clock must match "
            f"_word_event_rows start=est_idx/SR (no silent misalignment)."
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=dataset.collate_fn)
    lfs, hga = [], []
    support0 = valid0 = None
    starts: list[float] = []
    for batch in loader:
        lfs.append(batch.data["electrode_tokens_lfs"])
        hga.append(batch.data["electrode_tokens_hga"])
        starts.extend(float(s) for s in batch.triggers["start"].to_numpy())
        if support0 is None:
            support0 = batch.data["support"][0]      # (C, K) one-hot
            valid0 = batch.data["valid_mask"][0]     # (C,) bool
    if support0 is None:
        raise RuntimeError(f"({subject_id},{trial_id}) yielded no windows")
    # The loader preserves trigger order; reorder bands to the sorted union axis.
    order = np.argsort(np.asarray(starts))
    bands = [torch.cat(lfs, 0)[order], torch.cat(hga, 0)[order]]
    return (
        bands,
        support0.argmax(dim=-1).long(),
        valid0.bool(),
        int(support0.shape[1]),
    )


def _raw_grid(bands) -> torch.Tensor:
    """Raw |STFT| floor grid as an electrode-space tap: (N,C,D_raw,1) -> (N,C,1,D_raw)."""
    from speech_decoding.experiments.v2_raw_probe import raw_bins_from_bands

    raw = raw_bins_from_bands(bands)                  # (N,C,D_raw,1)
    n, c, d_raw, _ = raw.shape
    return raw.reshape(n, c, d_raw).unsqueeze(2)      # (N,C,1,D_raw)


def _session_cache(xp, ieeg, session, tasks, bt_root):
    """Materialize one session into a SessionTapCache holding the raw-371 grid."""
    from speech_decoding.experiments.pretrain_probe_labels import build_session_targets
    from speech_decoding.experiments.pretrain_probe_stage1 import (
        SessionMeta,
        cache_from_targets,
    )

    subject_id, trial_id = session
    row = ieeg[session]
    events = _label_events(subject_id, trial_id, str(row["timeline"]), tasks, bt_root)
    targets = build_session_targets(events, subject_id=subject_id, trial_id=trial_id)
    bands, ppe, em, n_parcels = _materialize_bands(xp, row, targets.clip_starts)
    grid = _raw_grid(bands)
    meta = SessionMeta(parcel_per_electrode=ppe, electrode_mask=em, n_parcels=n_parcels)
    cache = cache_from_targets(targets, {"raw": grid}, meta)
    return cache


def _stamp() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def run_raw_floor(sessions, tasks, *, anchor, out_path, lam_grid=None):
    """Phase 0: raw-371 linear-ridge floor, all tasks, WS + CS, → ledger CSV."""
    from speech_decoding.experiments.pretrain_probe_csv import ResultRow, append_results
    from speech_decoding.experiments.pretrain_probe_readout import (
        DEFAULT_LAM_GRID,
        linear_cs_cell_auroc,
        linear_ws_cell_auroc,
    )

    lam_grid = lam_grid or DEFAULT_LAM_GRID
    bt_root = os.environ.get("ROOT_DIR_BRAINTREEBANK")
    xp = _build_xp()
    ieeg = _ieeg_index(xp)
    stamp = _stamp()

    caches = {s: _session_cache(xp, ieeg, s, tasks, bt_root) for s in sessions}
    n_parcels = max(c.n_parcels for c in caches.values())
    anchor_cache = caches[anchor]
    rows: list[ResultRow] = []

    for s, cache in caches.items():
        for task in tasks:
            y = cache.labels.get(task)
            if y is None or np.sum(np.isfinite(y)) < 4:
                continue
            # WithinSession (fold 0; the held-out half is val+test).
            ws = cache.ws_split[task][0]
            auc = linear_ws_cell_auroc(
                cache.grids["raw"], y,
                train_rows=ws["train"], val_rows=ws["val"], test_rows=ws["test"],
                tap_space="electrode", parcel_per_electrode=cache.parcel_per_electrode,
                electrode_mask=cache.electrode_mask, n_parcels=cache.n_parcels,
                lam_grid=lam_grid,
            )
            if np.isfinite(auc):
                rows.append(ResultRow(
                    stamp=stamp, ckpt="raw_371", readout="ridge", tap="raw_371",
                    eval_mode="WithinSession", task=task, split="test",
                    auroc=round(float(auc), 4), n=int(len(ws["test"])),
                    notes=f"sess={s}",
                ))
            # CrossSubject: anchor=train, this session=test (skip the anchor as test).
            if s == anchor:
                continue
            cs = cache.cs_split[task]
            ya = anchor_cache.labels.get(task)
            if ya is None or np.sum(np.isfinite(ya)) < 4:
                continue
            auc_cs = linear_cs_cell_auroc(
                anchor_cache.grids["raw"], ya, cache.grids["raw"], y,
                val_rows=cs["val"], test_rows=cs["test"], tap_space="electrode",
                pe_anchor=anchor_cache.parcel_per_electrode, em_anchor=anchor_cache.electrode_mask,
                pe_test=cache.parcel_per_electrode, em_test=cache.electrode_mask,
                n_parcels=n_parcels, lam_grid=lam_grid,
            )
            if np.isfinite(auc_cs):
                rows.append(ResultRow(
                    stamp=stamp, ckpt="raw_371", readout="ridge", tap="raw_371",
                    eval_mode="CrossSubject", task=task, split="test",
                    auroc=round(float(auc_cs), 4), n=int(len(cs["test"])),
                    notes=f"anchor={anchor} test={s}",
                ))

    append_results(rows, out_path)
    _print_summary(rows)
    print(f"[pretrain-probe] wrote {len(rows)} rows -> {out_path}")
    return rows


def _print_summary(rows) -> None:
    by = {}
    for r in rows:
        by.setdefault((r.eval_mode, r.task), []).append(r.auroc)
    print(f"\n{'mode':<14}{'task':<14}{'mean_auroc':>11}{'n':>5}")
    for (mode, task), vals in sorted(by.items()):
        print(f"{mode:<14}{task:<14}{np.mean(vals):>11.4f}{len(vals):>5}")


def main(argv: list[str] | None = None) -> int:
    from speech_decoding.experiments.pretrain_probe_suite import (
        DEFAULT_CS_TRAIN_ANCHOR,
        PRETRAIN_UNIVERSE,
        NEUROPROBE_TASKS,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", action="store_true",
                   help="one session, a few tasks: print shapes + WS AUROC, no CSV write")
    p.add_argument("--full", action="store_true",
                   help="all firewall-legal sessions, all 15 tasks, WS+CS → CSV")
    p.add_argument("--out", default="reports/neuroprobe_probe_results.csv")
    p.add_argument("--anchor", default=None,
                   help="CS train anchor 'subj,trial' (default the contract anchor)")
    args = p.parse_args(argv)

    anchor = DEFAULT_CS_TRAIN_ANCHOR
    if args.anchor:
        a, b = args.anchor.split(",")
        anchor = (int(a), int(b))

    if args.preflight:
        bt_root = os.environ.get("ROOT_DIR_BRAINTREEBANK")
        xp = _build_xp()
        ieeg = _ieeg_index(xp)
        tasks = ("onset", "speech", "volume")
        cache = _session_cache(xp, ieeg, anchor, tasks, bt_root)
        print(f"[preflight] session {anchor} raw grid {tuple(cache.grids['raw'].shape)} "
              f"n_parcels={cache.n_parcels} valid_elec={int(cache.electrode_mask.sum())}")
        from speech_decoding.experiments.pretrain_probe_readout import linear_ws_cell_auroc
        for task in tasks:
            y = cache.labels.get(task)
            if y is None or np.sum(np.isfinite(y)) < 4:
                print(f"[preflight] {task}: too few labels"); continue
            ws = cache.ws_split[task][0]
            auc = linear_ws_cell_auroc(
                cache.grids["raw"], y, train_rows=ws["train"], val_rows=ws["val"],
                test_rows=ws["test"], tap_space="electrode",
                parcel_per_electrode=cache.parcel_per_electrode,
                electrode_mask=cache.electrode_mask, n_parcels=cache.n_parcels,
            )
            print(f"[preflight] WS {task}: AUROC={auc:.4f} (n_test={len(ws['test'])})")
        return 0

    if args.full:
        run_raw_floor(tuple(PRETRAIN_UNIVERSE), tuple(NEUROPROBE_TASKS),
                      anchor=anchor, out_path=args.out)
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
