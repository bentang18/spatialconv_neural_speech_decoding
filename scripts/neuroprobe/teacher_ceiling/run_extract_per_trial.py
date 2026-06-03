"""DCC entry point: extract Whisper-large-v3 features for one (subject, trial).

Usage:
    .venv/bin/python scripts/neuroprobe/teacher_ceiling/run_extract_per_trial.py \\
        --subject-id 3 --trial-id 0 \\
        --wav-path /work/ht203/data/braintreebank_wavs/cars-2.wav \\
        --words-df-dir /work/ht203/repo/neuroprobe_upstream/neuroprobe/braintreebank_features_time_alignment \\
        --transcripts-root /work/ht203/data/braintreebank/transcripts \\
        --out-dir /hpc/group/coganlab/ht203/cache_neuroai/whisper_ceiling/v3 \\
        --model openai/whisper-large-v3

Writes <out_dir>/sub_<id>_trial<id>.npz containing per-layer (W, 1280) fp16
features keyed `layer_<idx>` plus word_index, movie_start_s, layers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from speech_decoding.whisper_ceiling.extract import (
    DEFAULT_LAYERS,
    WHISPER_SR,
    WhisperMultiLayerExtractor,
    extract_trial_features,
    load_trial_labels,
)

# Mirror of upstream BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING entries we need.
# Hard-coded here so the script imports without ROOT_DIR_BRAINTREEBANK.
# VERIFIED 2026-06-02 against .cache/neuroprobe_upstream/neuroprobe/config.py
# (authoritative). The prior hand-typed dict had 6 wrong values — (2,4)/(2,5),
# (6,1), (8,0)/(9,0), (10,1) — which mispaired wrong-movie audio+labels.
SUBJECT_TRIAL_MOVIE = {
    (1, 0): "fantastic-mr-fox", (1, 1): "the-martian", (1, 2): "thor-ragnarok",
    (2, 0): "venom", (2, 1): "spider-man-3-homecoming",
    (2, 2): "guardians-of-the-galaxy", (2, 3): "guardians-of-the-galaxy-2",
    (2, 4): "avengers-infinity-war", (2, 5): "black-panther", (2, 6): "aquaman",
    (3, 0): "cars-2", (3, 1): "lotr-1", (3, 2): "lotr-2",
    (4, 0): "shrek-the-third", (4, 1): "megamind", (4, 2): "incredibles",
    (5, 0): "fantastic-mr-fox",
    (6, 0): "megamind", (6, 1): "toy-story", (6, 4): "coraline",
    (7, 0): "cars-2", (7, 1): "megamind",
    (8, 0): "sesame-street-episode-3990", (9, 0): "ant-man",
    (10, 0): "cars-2", (10, 1): "spider-man-far-from-home",
}


def main() -> None:
    args = parse_args()
    movie_name = SUBJECT_TRIAL_MOVIE.get((args.subject_id, args.trial_id))
    if movie_name is None:
        sys.exit(f"unknown (subject, trial) = ({args.subject_id}, {args.trial_id})")
    print(f"[extract] sub_{args.subject_id} trial{args.trial_id} -> {movie_name}", flush=True)

    print(f"[extract] loading WAV {args.wav_path}", flush=True)
    wav, sr = sf.read(str(args.wav_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.float32)
    if sr != WHISPER_SR:
        sys.exit(f"need {WHISPER_SR}Hz audio, got {sr}; pre-resample before this script")
    print(f"[extract] WAV length {len(wav)/sr:.1f}s @ {sr}Hz", flush=True)

    print(f"[extract] loading labels", flush=True)
    words_df = load_trial_labels(
        subject_id=args.subject_id,
        trial_id=args.trial_id,
        movie_name=movie_name,
        words_df_dir=args.words_df_dir,
        transcripts_root=args.transcripts_root,
        pitch_volume_dir=args.pitch_volume_dir,
    )
    print(f"[extract] {len(words_df)} words; cols={list(words_df.columns)[:8]}...", flush=True)

    print(f"[extract] loading model {args.model}", flush=True)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    processor = WhisperProcessor.from_pretrained(args.model)
    model.eval()
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"[extract] using CUDA", flush=True)
    else:
        print(f"[extract] using CPU", flush=True)

    layers = tuple(args.layers) if args.layers else DEFAULT_LAYERS
    extractor = WhisperMultiLayerExtractor(model, processor, layers=layers)

    def _progress(ci, total):
        if ci % 20 == 0 or ci == total - 1:
            print(f"[extract]   chunk {ci+1}/{total}", flush=True)

    payload = extract_trial_features(
        extractor=extractor,
        wav=wav,
        sample_rate=sr,
        words_df=words_df,
        before_s=args.before_s,
        after_s=args.after_s,
        progress_cb=_progress,
    )
    extractor.close()

    # Stash per-task label arrays aligned to the kept word_index.
    word_idx = payload["word_index"]
    labels_out = {}
    for col in words_df.columns:
        vals = words_df.iloc[word_idx][col].to_numpy()
        if vals.dtype.kind in ("i", "f"):
            labels_out[col] = vals
        elif vals.dtype.kind in ("O", "U", "S"):
            # Categorical → keep as object array (numpy will save as Unicode)
            labels_out[col] = vals.astype(str)
    label_columns = list(labels_out.keys())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"sub_{args.subject_id}_trial{args.trial_id}.npz"
    if out_path.exists() and not args.force:
        print(f"[extract] {out_path} already exists -- skip (use --force to overwrite)", flush=True)
        return
    print(f"[extract] writing {out_path}", flush=True)
    np.savez_compressed(
        out_path,
        **{k: v for k, v in payload.items() if not k == "layers"},
        layers=np.array(layers, dtype=np.int32),
        label_columns=np.array(label_columns, dtype=str),
        **{f"label_{c}": labels_out[c] for c in label_columns},
    )

    summary = {
        "subject_id": args.subject_id,
        "trial_id": args.trial_id,
        "movie": movie_name,
        "wav_path": str(args.wav_path),
        "model": args.model,
        "layers": list(layers),
        "before_s": args.before_s,
        "after_s": args.after_s,
        "n_words_total": int(len(words_df)),
        "n_words_kept": int(len(word_idx)),
        "out_path": str(out_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    summary_path = args.out_dir / f"sub_{args.subject_id}_trial{args.trial_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[extract] done", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subject-id", type=int, required=True)
    p.add_argument("--trial-id", type=int, required=True)
    p.add_argument("--wav-path", type=Path, required=True)
    p.add_argument(
        "--words-df-dir", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream/neuroprobe/braintreebank_features_time_alignment"),
    )
    p.add_argument(
        "--transcripts-root", type=Path,
        default=Path("/work/ht203/data/braintreebank/transcripts"),
    )
    p.add_argument(
        "--pitch-volume-dir", type=Path,
        default=Path("/work/ht203/repo/neuroprobe_upstream/neuroprobe/pitch_volume_features"),
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/hpc/group/coganlab/ht203/cache_neuroai/whisper_ceiling/v3"),
    )
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--before-s", type=float, default=0.0)
    p.add_argument("--after-s", type=float, default=1.0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--force", action="store_true",
                   help="re-extract even if the per-trial NPZ already exists")
    return p.parse_args()


if __name__ == "__main__":
    main()
