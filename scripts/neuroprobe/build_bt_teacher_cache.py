"""Build the v14 P3 Whisper teacher cache over the BrainTreebank movies (T14).

Runs Whisper-large-v3 ONCE per movie over 30 s grid chunks and stores one dense
``(T, 1280)`` fp16 feature stream per movie at 50 Hz (the encoder's native
rate). Every training clip is then a free slice
``dense[round(t0_movie_s × 50) : +250]`` of this stream — see
``speech_decoding.bt_alignment.teacher_cache.write_movie_cache`` for the
chunking rationale (Whisper's encoder is fixed at a 30 s window, so a whole
movie must be chunked or it silently truncates to its first 30 s).

The teacher target is the plain mean over all 32 encoder layers
(``--layer-merge mean_all``, the locked default); model + layer-merge fold into
the cache path so two target-defining configs never share a file.

DCC GPU job. Preflight on the login node with ``--dry-run`` (lists movies +
output paths, loads no model). CPU smoke with ``--model openai/whisper-tiny
--device cpu --limit 1``.

Example (DCC GPU):
    .venv/bin/python scripts/neuroprobe/build_bt_teacher_cache.py \
        --wav-dir /hpc/group/coganlab/ht203/data/braintreebank_wavs \
        --out-dir /hpc/group/coganlab/ht203/cache_neuroai/bt_teacher_cache
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from speech_decoding.bt_alignment.teacher_cache import (
    DEFAULT_LAYER_MERGE,
    WHISPER_ENCODER_WINDOW_S,
    WHISPER_SR,
    MovieCacheEntry,
    WhisperFeatureExtractor,
    merge_slug,
    movie_cache_path,
    write_movie_cache,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("/hpc/group/coganlab/ht203/data/braintreebank_wavs"),
        help="Directory of <movie>.wav 16 kHz mono files (T13 port).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Teacher-cache root. Default $EXCA_CACHE_FOLDER/bt_teacher_cache.",
    )
    p.add_argument("--model", default="openai/whisper-large-v3",
                   help="HF Whisper id. Use openai/whisper-tiny for CPU smoke.")
    p.add_argument("--layer-merge", default=str(DEFAULT_LAYER_MERGE),
                   help="'mean_all' (locked default) or an int single layer.")
    p.add_argument("--device", default="cuda", help="cuda | cpu")
    p.add_argument("--chunk-s", type=float, default=WHISPER_ENCODER_WINDOW_S,
                   help="Grid chunk seconds (default 30 = encoder window).")
    p.add_argument("--movies", nargs="*", default=None,
                   help="Optional subset of movie slugs (default: all *.wav).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N movies (smoke).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-encode movies whose cache file already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="List movies + output paths; load no model.")
    return p.parse_args(argv)


def _resolve_layer_merge(raw: str) -> int | str:
    return raw if raw == "mean_all" else int(raw)


def _resolve_out_dir(arg: Path | None) -> Path:
    if arg is not None:
        return arg
    env = os.environ.get("EXCA_CACHE_FOLDER")
    if not env:
        raise SystemExit(
            "--out-dir not given and EXCA_CACHE_FOLDER unset; pass --out-dir."
        )
    return Path(env) / "bt_teacher_cache"


def _movie_slugs(wav_dir: Path, subset: list[str] | None) -> list[str]:
    if not wav_dir.is_dir():
        raise SystemExit(f"--wav-dir not a directory: {wav_dir}")
    available = sorted(p.stem for p in wav_dir.glob("*.wav"))
    if subset is None:
        return available
    missing = [m for m in subset if m not in available]
    if missing:
        raise SystemExit(f"requested movies absent from {wav_dir}: {missing}")
    return [m for m in available if m in subset]


def _load_mono_16k(path: Path) -> np.ndarray:
    """Read a wav as float32 mono in [-1, 1] at 16 kHz; downmix stereo, fail loud
    on rate.

    Uses ``scipy.io.wavfile`` (the audio reader present in the DCC venv —
    ``soundfile`` is NOT a declared dependency). Unlike soundfile's
    ``dtype="float32"`` read, ``wavfile.read`` returns the WAV's native dtype
    WITHOUT normalizing, so an int PCM stream must be divided by its dtype max to
    land in [-1, 1] — Whisper's mel front-end expects that range, and raw ±32768
    int16 would corrupt the log-mel. The BT movies are int16/16 kHz/mono."""
    sr, data = wavfile.read(str(path))
    if sr != WHISPER_SR:
        raise SystemExit(
            f"{path.name}: sample rate {sr} != {WHISPER_SR}; re-extract at 16 kHz."
        )
    orig_dtype = data.dtype
    data = data.astype(np.float32)
    if data.ndim == 2:  # stereo → mono (float-domain mean; int mean would clip)
        data = data.mean(axis=1).astype(np.float32)
    if np.issubdtype(orig_dtype, np.signedinteger):
        # PCM int → [-1, 1) by 2^(bits-1), the soundfile/librosa/torchaudio
        # convention (int16 ÷ 32768, NOT ÷ iinfo.max 32767) so this matches the
        # float read the Whisper ceiling probe validated against.
        data = data / float(2 ** (8 * orig_dtype.itemsize - 1))
    return np.ascontiguousarray(data, dtype=np.float32)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _expected_path(out_dir: Path, model: str, merge: int | str, movie: str) -> Path:
    # Single source of truth for cache layout (writer/reader/skip-check share it).
    return movie_cache_path(out_dir, model, merge, movie)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = _resolve_out_dir(args.out_dir)
    merge = _resolve_layer_merge(args.layer_merge)
    slugs = _movie_slugs(args.wav_dir, args.movies)
    if args.limit is not None:
        slugs = slugs[: args.limit]

    print(f"BT teacher-cache build — {len(slugs)} movie(s)")
    print(f"  wav_dir={args.wav_dir}")
    print(f"  out_dir={out_dir}")
    print(f"  model={args.model} layer_merge={merge!r} chunk_s={args.chunk_s} "
          f"device={args.device}")
    print(f"  movies={slugs}")

    if args.dry_run:
        for movie in slugs:
            print(f"  [dry] {movie} -> {_expected_path(out_dir, args.model, merge, movie)}")
        return 0

    # Load Whisper only for a real build (keeps --dry-run a login-node preflight).
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  WARNING: cuda requested but unavailable; falling back to cpu.")
        device = "cpu"
    # Force fp32. transformers v5 loads a checkpoint in its NATIVE dtype, and
    # large-v3's HF weights are fp16 — but the teacher target is a mean over all
    # 32 encoder layers, and accumulating that in fp16 drifts from the fp32
    # ceiling-probe that validated `mean_all`. fp32 large-v3 ≈6 GB, well within
    # the 32 GB RTX 5000 Ada; the dense stream is still stored fp16 downstream.
    model = (
        WhisperForConditionalGeneration.from_pretrained(args.model)
        .eval().float().to(device)
    )
    proc = WhisperProcessor.from_pretrained(args.model)
    fe = WhisperFeatureExtractor(model, proc, layer_merge=merge)

    entries: list[MovieCacheEntry] = []
    try:
        for i, movie in enumerate(slugs, 1):
            dest = _expected_path(out_dir, args.model, merge, movie)
            if dest.exists() and not args.overwrite:
                print(f"  [{i}/{len(slugs)}] {movie}: cache exists, skip "
                      f"(--overwrite to rebuild)")
                continue
            wav = _load_mono_16k(args.wav_dir / f"{movie}.wav")
            entry = write_movie_cache(
                fe, wav, sample_rate=WHISPER_SR, movie=movie,
                out_dir=out_dir, chunk_s=args.chunk_s,
            )
            entries.append(entry)
            print(f"  [{i}/{len(slugs)}] {movie}: {entry.duration_s:.1f}s "
                  f"-> {entry.n_frames} frames @ {entry.rate_hz}Hz, d={entry.d_model} "
                  f"({entry.out_path})")
            sys.stdout.flush()
    finally:
        fe.close()

    # Manifest lives in the SAME <model>/<merge_slug>/ dir as the .pt files it
    # describes — keying by model alone let a `mean_all` and an `L8` build clobber
    # each other's manifest while their caches sat side by side.
    cache_dir = out_dir / args.model.replace("/", "_") / merge_slug(merge)
    manifest_path = cache_dir / "build_manifest.json"
    if not entries:
        # A pure-skip resume built nothing — do NOT clobber a good manifest with
        # an empty one (the bug a skip-only re-run would otherwise cause).
        print(f"Built 0 new movie(s); manifest unchanged -> {manifest_path}")
        return 0
    cached_total = sorted(p.stem for p in cache_dir.glob("*.pt"))
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "model": args.model,
        "layer_merge": merge,
        "chunk_s": args.chunk_s,
        "rate_hz": 50,
        "wav_dir": str(args.wav_dir),
        "movies_built": [e.movie for e in entries],         # this run only
        "movies_cached_total": cached_total,                # full cache state (filenames)
        "entries": [vars(e) for e in entries],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Built {len(entries)} movie(s); cached total {len(cached_total)}; "
          f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
