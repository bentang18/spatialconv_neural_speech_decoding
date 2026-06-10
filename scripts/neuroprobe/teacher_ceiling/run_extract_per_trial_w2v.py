"""DCC entry point: extract wav2vec2-large-xlsr-53 features for one (subject, trial).

Head-to-head sibling of run_extract_per_trial.py (Whisper). Writes NPZs in the
SAME key layout the probe reads, into a separate features-dir, so the existing
run_probe_ceiling.py produces a direct Whisper-vs-wav2vec2 comparison.

Usage:
    .venv/bin/python scripts/neuroprobe/teacher_ceiling/run_extract_per_trial_w2v.py \\
        --subject-id 3 --trial-id 0 \\
        --wav-path /work/ht203/data/braintreebank_wavs/cars-2.wav \\
        --out-dir /hpc/group/coganlab/ht203/cache_neuroai/wav2vec2_ceiling/xlsr53 \\
        --model facebook/wav2vec2-large-xlsr-53

Writes <out_dir>/sub_<id>_trial<id>.npz containing per-layer (W, 1024) fp16
features keyed `layer_<idx>` (idx 0..24) plus word_index, movie_start_s,
layers, label_columns, label_<col>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from speech_decoding.wav2vec2_ceiling.extract import (
    W2V_DEFAULT_MODEL,
    W2V_SR,
    Wav2Vec2MultiLayerExtractor,
    extract_trial_features_w2v,
)
from speech_decoding.whisper_ceiling.extract import load_trial_labels

# Reuse the verified (subject, trial) -> movie mapping from the Whisper entry.
from run_extract_per_trial import SUBJECT_TRIAL_MOVIE


def main() -> None:
    args = parse_args()
    movie_name = SUBJECT_TRIAL_MOVIE.get((args.subject_id, args.trial_id))
    if movie_name is None:
        sys.exit(f"unknown (subject, trial) = ({args.subject_id}, {args.trial_id})")
    print(f"[extract-w2v] sub_{args.subject_id} trial{args.trial_id} -> {movie_name}", flush=True)

    print(f"[extract-w2v] loading WAV {args.wav_path}", flush=True)
    wav, sr = sf.read(str(args.wav_path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1).astype(np.float32)
    if sr != W2V_SR:
        sys.exit(f"need {W2V_SR}Hz audio, got {sr}; pre-resample before this script")
    print(f"[extract-w2v] WAV length {len(wav)/sr:.1f}s @ {sr}Hz", flush=True)

    print(f"[extract-w2v] loading labels", flush=True)
    words_df = load_trial_labels(
        subject_id=args.subject_id,
        trial_id=args.trial_id,
        movie_name=movie_name,
        words_df_dir=args.words_df_dir,
        transcripts_root=args.transcripts_root,
        pitch_volume_dir=args.pitch_volume_dir,
    )
    print(f"[extract-w2v] {len(words_df)} words; cols={list(words_df.columns)[:8]}...", flush=True)

    print(f"[extract-w2v] loading model {args.model}", flush=True)
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    import torch
    model = Wav2Vec2Model.from_pretrained(args.model)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model.eval()
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print(f"[extract-w2v] using CUDA", flush=True)
    else:
        print(f"[extract-w2v] using CPU", flush=True)

    layers = tuple(args.layers) if args.layers else None
    extractor = Wav2Vec2MultiLayerExtractor(model, processor, layers=layers)
    print(f"[extract-w2v] capturing hidden states {extractor.layers}", flush=True)

    def _progress(ci, total):
        if ci % 20 == 0 or ci == total - 1:
            print(f"[extract-w2v]   chunk {ci+1}/{total}", flush=True)

    payload = extract_trial_features_w2v(
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
            labels_out[col] = vals.astype(str)
    label_columns = list(labels_out.keys())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"sub_{args.subject_id}_trial{args.trial_id}.npz"
    if out_path.exists() and not args.force:
        print(f"[extract-w2v] {out_path} already exists -- skip (use --force)", flush=True)
        return
    layers_arr = payload["layers"]
    print(f"[extract-w2v] writing {out_path}", flush=True)
    np.savez_compressed(
        out_path,
        **{k: v for k, v in payload.items() if k != "layers"},
        layers=layers_arr,
        label_columns=np.array(label_columns, dtype=str),
        **{f"label_{c}": labels_out[c] for c in label_columns},
    )

    summary = {
        "subject_id": args.subject_id,
        "trial_id": args.trial_id,
        "movie": movie_name,
        "wav_path": str(args.wav_path),
        "model": args.model,
        "layers": [int(x) for x in layers_arr],
        "before_s": args.before_s,
        "after_s": args.after_s,
        "n_words_total": int(len(words_df)),
        "n_words_kept": int(len(word_idx)),
        "out_path": str(out_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    summary_path = args.out_dir / f"sub_{args.subject_id}_trial{args.trial_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[extract-w2v] done", flush=True)


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
        default=Path("/hpc/group/coganlab/ht203/cache_neuroai/wav2vec2_ceiling/xlsr53"),
    )
    p.add_argument("--model", default=W2V_DEFAULT_MODEL)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--before-s", type=float, default=0.0)
    p.add_argument("--after-s", type=float, default=1.0)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
