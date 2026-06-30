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
    if getattr(args, "exca_mode", None) is None:
        args.exca_mode = "cached"          # main() resolves this default; we skip main()
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


def _label_events(subject_id: int, trial_id: int, timeline: str, tasks, bt_root,
                  *, lite_cap: bool = True) -> pd.DataFrame:
    """Balanced, all-task word-event rows for ONE pretrain session (full electrodes).

    ``lite_cap`` (default True) matches the Neuroprobe-Lite leaderboard's per-task sample
    budget: ``derive_label_indices(lite=True)`` caps each class at ``LITE_MAX_SAMPLES//n_classes``
    (3500 total/task) via the seeded ``rng.choice`` upstream uses — the SAME subsample the real
    Lite eval scores, so the proxy isn't optimistic from extra samples. The ``lite`` flag is
    label-only here (it never selects lite electrodes/sessions — those stay full/pretrain).
    Set False for the uncapped Neuroprobe-Full reference."""
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
        binary_tasks=True, lite=lite_cap, nano=False, random_seed=42,
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

    # Stream one session at a time: hold only the anchor cache + the current
    # session (<=2 caches), and append rows to the CSV per session. Building all
    # caches up front OOM-killed the login-node cgroup; an end-only write also
    # lost everything on a mid-run death. Same numbers, bounded RAM.
    anchor_cache = _session_cache(xp, ieeg, anchor, tasks, bt_root)
    all_rows: list[ResultRow] = []

    def _eval_session(s, cache) -> list[ResultRow]:
        rows: list[ResultRow] = []
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
            # Globally-fixed DK parcel grid; per-pair max is identical to a global
            # max here (extra parcels are unsupported in both -> dropped on intersect).
            n_parcels = max(anchor_cache.n_parcels, cache.n_parcels)
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
        return rows

    for s in sessions:
        cache = anchor_cache if s == anchor else _session_cache(xp, ieeg, s, tasks, bt_root)
        rows = _eval_session(s, cache)
        append_results(rows, out_path)  # incremental: survives a later death
        all_rows.extend(rows)
        print(f"[pretrain-probe] session {s}: +{len(rows)} rows -> {out_path}", flush=True)
        if cache is not anchor_cache:
            del cache

    _print_summary(all_rows)
    print(f"[pretrain-probe] wrote {len(all_rows)} rows -> {out_path}")
    return all_rows


def _apply_lite_montage(bands, ppe, em, *, bt_root, subject_id, trial_id):
    """Subset the electrode axis of (bands, parcel_per_electrode, electrode_mask) to the
    Neuroprobe-Lite montage — the SAME ~93–120 electrodes the real Lite leaderboard evals
    on (parity), and the lever that takes M2's all-electrode keep-S grid from ~256 → Lite
    width (the host-RAM ceiling that was OOM-ing the encode).

    Uses the vendored ``lite_voltage_mask`` (a boolean mask over ``voltage_electrode_order``,
    the canonical voltage row order the front-end and the support/valid_mask extractors all
    key off). Applying the one mask in lockstep to bands AND ppe/em keeps ``support[c]`` ↔
    ``electrode_tokens[c]`` aligned. Fails loud if the mask length != the electrode axis —
    a voltage-order desync that would silently mis-route electrodes into parcels."""
    import torch

    from speech_decoding.studies.braintreebank.anatomy import lite_voltage_mask

    if bt_root is None:
        raise ValueError("electrode_set='lite' needs ROOT_DIR_BRAINTREEBANK (lite montage)")
    mask = lite_voltage_mask(bt_root, subject_id, trial_id)        # (C,) bool over voltage order
    c_axis = int(ppe.shape[0])
    if mask.shape[0] != c_axis:
        raise ValueError(
            f"({subject_id},{trial_id}) Lite mask length {mask.shape[0]} != electrode axis "
            f"{c_axis}: voltage-order desync. Refusing to silently mis-route electrodes."
        )
    m = torch.from_numpy(mask)
    return [b[:, m] for b in bands], ppe[m], em[m]


def run_encode(sessions, tasks, *, ckpt_path, out_dir, cache_untagged_m3=True,
               electrode_set="lite", batch_size=64):
    """Stage 1: forward the trained encoder over each session's union clips → tap caches.

    One GPU forward per session over EXACTLY the union word windows (est_idx-aligned,
    fail-loud), keeping the full token grids the Stage-2 readouts consume:
      ``M2`` frontend ``(N,C,S,d)`` electrode-space; ``M3`` pool_tagged ``(N,P,k,S,d)``;
      ``M4`` latent ``(N,P,k,S,d)``; ``M3_untagged`` bare pool (the ridge A/B surface,
    skipped if ``cache_untagged_m3`` is False). Each session's :class:`SessionTapCache`
    (grids + per-task labels + WS/CS split rows + electrode anatomy) is ``torch.save``-d
    to ``out_dir`` so the CPU Stage-2 sweep reruns without re-forwarding. Pure reuse of
    ``encode_subject_tokens`` (the unreduced full-grid encode) — no new model logic.

    ``electrode_set='lite'`` (default) subsets each session to its Neuroprobe-Lite montage
    BEFORE the forward (leaderboard parity + the M2 host-RAM lever); ``'all'`` keeps every
    electrode. Sessions are processed one at a time and freed, so peak host RAM is one
    session's grids, not the whole cohort's."""
    import torch

    from speech_decoding.experiments.pretrain_probe_driver import save_cache
    from speech_decoding.experiments.pretrain_probe_labels import build_session_targets
    from speech_decoding.experiments.pretrain_probe_stage1 import (
        SessionMeta,
        cache_from_targets,
    )
    from speech_decoding.experiments.v2_probe_bench import (
        encode_subject_tokens,
        load_v2_converged_model,
    )

    bt_root = os.environ.get("ROOT_DIR_BRAINTREEBANK")
    xp = _build_xp()
    ieeg = _ieeg_index(xp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_v2_converged_model(xp, ckpt_path, device=device)
    # Global DK parcel atlas — the model's own parcel-embed table: constant across every
    # session and ≥ each subject's max DKT id (the model was trained on this cohort). The
    # parcel-space M3/M4 grids stay COMPACTED (~16 present parcels); CS intersection aligns
    # via the per-session ``parcel_labels`` (DKT ids of the compacted P axis), mapped to
    # grid positions lazily in the readouts. ``n_parcels`` is the shared atlas-id space.
    n_parcels_atlas = int(model.latent.parcel_embed.num_embeddings)
    os.makedirs(out_dir, exist_ok=True)

    surfaces = ["frontend", "m3", "m4"] + (["m3_untagged"] if cache_untagged_m3 else [])
    tap_of = {"frontend": "M2", "m3": "M3", "m4": "M4", "m3_untagged": "M3_untagged"}

    for session in sessions:
        subject_id, trial_id = session
        row = ieeg[session]
        events = _label_events(subject_id, trial_id, str(row["timeline"]), tasks, bt_root)
        targets = build_session_targets(events, subject_id=subject_id, trial_id=trial_id)
        bands, ppe, em, _ = _materialize_bands(xp, row, targets.clip_starts)
        if electrode_set == "lite":
            bands, ppe, em = _apply_lite_montage(
                bands, ppe, em, bt_root=bt_root, subject_id=subject_id, trial_id=trial_id
            )
        grids_sf, labels = encode_subject_tokens(
            model, bands, ppe, clip_len_s=PROBE_CLIP_DUR_S, device=device,
            surfaces=tuple(surfaces), batch_size=batch_size,
        )
        grids = {tap_of[sf]: g for sf, g in grids_sf.items()}
        meta = SessionMeta(
            parcel_per_electrode=ppe, electrode_mask=em, n_parcels=n_parcels_atlas,
            parcel_labels=labels,
        )
        cache = cache_from_targets(targets, grids, meta)
        path = os.path.join(out_dir, f"taps_s{subject_id}_t{trial_id}.pt")
        save_cache(cache, path)
        shp = {k: tuple(v.shape) for k, v in grids.items()}
        print(f"[encode] {session}: {shp} -> {path}", flush=True)
        del bands, grids_sf, grids, cache


def _load_tap_caches(cache_dir, *, maxsize=2):
    """Index the per-session `--encode` tap caches → a :class:`LazyCacheMap` that loads each
    session on access and keeps at most ``maxsize`` resident.

    The keep-S grids are large (Lite-120 M2 ≈ 50 GB/session); eager-loading all 7 cohort
    sessions would OOM the readout node. The lazy map holds only what the current cell needs
    (test + anchor = 2), so peak readout RAM is ~2 sessions, not the whole cohort."""
    import glob
    import re

    from speech_decoding.experiments.pretrain_probe_driver import LazyCacheMap

    paths = {}
    pat = re.compile(r"taps_s(\d+)_t(\d+)\.pt$")
    for path in sorted(glob.glob(os.path.join(cache_dir, "taps_s*_t*.pt"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        paths[(int(m.group(1)), int(m.group(2)))] = path
    if not paths:
        raise FileNotFoundError(f"no taps_s*_t*.pt caches under {cache_dir}")
    return LazyCacheMap(paths, maxsize=maxsize)


def run_score(cache_dir, *, ckpt_tag, anchor, out_path, base_cfg=None, hp_grid=None,
              modes=("WithinSession", "CrossSubject"), tasks=None, taps=None,
              device=None):
    """Stage 2 (CPU): load the encoder-tap caches → sweep attentive HP once → eval every
    cell × tap (ridge + attentive) at the fixed HP → aggregate → ledger CSV + summary.

    Consumes ONLY the `--encode` caches (no GPU, no BT reads); the cross-subject mean is
    the headline number the pooled encoder taps must push above the 0.666 raw_tok floor.
    ``base_cfg``/``hp_grid`` default to the contract config + Ben's 3×2×2 grid; overrides
    exist only so the wiring is unit-testable on a tiny config."""
    from speech_decoding.experiments.pretrain_probe_collect import RIDGE_FLOOR_DELTA_VOLUME
    from speech_decoding.experiments.pretrain_probe_csv import ResultRow, append_results
    from speech_decoding.experiments.pretrain_probe_run import select_best_checkpoint
    from speech_decoding.experiments.pretrain_probe_suite import NEUROPROBE_TASKS
    from speech_decoding.experiments.pretrain_probe_sweep import TAPS
    from speech_decoding.experiments.v2_attentive_train import HeadTrainConfig

    # Train the attentive heads on the GPU when one is present (the score job runs on a
    # GH200): minibatches move to device while the large keep-S grids stay in host RAM, so
    # the ~2000-step heads finish fast without inflating GPU memory. Ridge stays on CPU.
    if device is None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    caches = _load_tap_caches(cache_dir)
    # Read d_model off ONE session (loads + LRU-caches it) — never materialize all caches.
    d_model = int(caches[next(iter(caches.keys()))].grids["M2"].shape[-1])
    cfg = base_cfg if base_cfg is not None else HeadTrainConfig(d_model=d_model)
    caches_by_ckpt = {ckpt_tag: caches}
    # Enumerate cells over EXACTLY the cached sessions — a cell can never request a cache
    # the --encode step didn't write (e.g. the 13→7 cohort cut, or a partial encode).
    cohort = tuple(sorted(caches.keys()))
    sel = select_best_checkpoint(
        caches_by_ckpt, cfg, representative_ckpt=ckpt_tag,
        modes=tuple(modes), tasks=tuple(tasks) if tasks else NEUROPROBE_TASKS,
        taps=tuple(taps) if taps else TAPS,
        cs_train_anchor=anchor, cohort=cohort, hp_grid=hp_grid, device=device,
    )
    stamp = _stamp()
    hp = sel.sweep.best_hp
    hp_note = f"wd={hp.weight_decay},pd={hp.parcel_dropout},dr={hp.dropout}"
    summary = sel.summary_by_ckpt[ckpt_tag]

    rows: list[ResultRow] = []
    for s in sel.scores_by_ckpt[ckpt_tag]:
        for readout, auc in (("ridge", s.linear), ("attentive", s.attentive)):
            if not np.isfinite(auc):
                continue
            rows.append(ResultRow(
                stamp=stamp, ckpt=ckpt_tag, readout=readout, tap=s.tap,
                eval_mode=s.eval_mode, task=s.task, split="test",
                auroc=round(float(auc), 4), n=0,
                lam="" if readout == "attentive" else "val",
                notes=f"{s.label};{hp_note}" if readout == "attentive" else s.label,
            ))
    # Aggregate MEAN rows (the selection signal): per (mode, tap, readout). aggregate_scores
    # labels the linear readout 'linear'; the ledger schema uses 'ridge' — normalize.
    for (mode, tap, readout), m in sorted(summary.mean_by_mode_tap.items()):
        if not np.isfinite(m):
            continue
        rows.append(ResultRow(
            stamp=stamp, ckpt=ckpt_tag, readout="ridge" if readout == "linear" else readout,
            tap=tap, eval_mode=mode, task="MEAN", split="test",
            auroc=round(float(m), 4), n=summary.n_cells,
            lam="", notes=f"agg;{hp_note}",
        ))
    append_results(rows, out_path)

    best_tap, best_readout, best_m = summary.best_cross_subject
    print(f"\n[score] ckpt={ckpt_tag} d_model={d_model} cells={summary.n_cells} "
          f"sweep_best_hp=({hp_note})")
    print(f"{'mode':<14}{'tap':<6}{'readout':<11}{'mean_auroc':>11}")
    for (mode, tap, readout), m in sorted(summary.mean_by_mode_tap.items()):
        if np.isfinite(m):
            print(f"{mode:<14}{tap:<6}{readout:<11}{m:>11.4f}")
    delta = best_m - RIDGE_FLOOR_DELTA_VOLUME
    print(f"\n[score] best CrossSubject = {best_tap}/{best_readout} {best_m:.4f} "
          f"(floor {RIDGE_FLOOR_DELTA_VOLUME:.3f}, delta {delta:+.4f}) -> {out_path}")
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
        PROBE_COHORT_7,
        NEUROPROBE_TASKS,
    )

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", action="store_true",
                   help="one session, a few tasks: print shapes + WS AUROC, no CSV write")
    p.add_argument("--full", action="store_true",
                   help="all firewall-legal sessions, all 15 tasks, WS+CS → CSV")
    p.add_argument("--encode", action="store_true",
                   help="Stage 1: GPU forward a checkpoint → per-session tap caches")
    p.add_argument("--ckpt", default=None, help="checkpoint to forward (required for --encode)")
    p.add_argument("--cache-dir", default=None,
                   help="output dir for --encode tap caches (required for --encode)")
    p.add_argument("--no-untagged-m3", action="store_true",
                   help="skip the untagged-pool M3 A/B surface (cache M2/M3/M4 only)")
    p.add_argument("--electrode-set", choices=("lite", "all"), default="lite",
                   help="--encode montage: 'lite' (Neuroprobe-Lite ~120, leaderboard parity, "
                        "default) or 'all' electrodes")
    p.add_argument("--score", action="store_true",
                   help="Stage 2 (CPU): load --cache-dir tap caches → sweep + eval → CSV")
    p.add_argument("--ckpt-tag", default="ladder-60000",
                   help="checkpoint label for the --score ledger rows")
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
        run_raw_floor(tuple(PROBE_COHORT_7), tuple(NEUROPROBE_TASKS),
                      anchor=anchor, out_path=args.out)
        return 0

    if args.encode:
        if not args.ckpt or not args.cache_dir:
            p.error("--encode requires --ckpt and --cache-dir")
        run_encode(tuple(PROBE_COHORT_7), tuple(NEUROPROBE_TASKS),
                   ckpt_path=args.ckpt, out_dir=args.cache_dir,
                   cache_untagged_m3=not args.no_untagged_m3,
                   electrode_set=args.electrode_set)
        return 0

    if args.score:
        if not args.cache_dir:
            p.error("--score requires --cache-dir (the --encode tap caches)")
        run_score(args.cache_dir, ckpt_tag=args.ckpt_tag, anchor=anchor, out_path=args.out)
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
