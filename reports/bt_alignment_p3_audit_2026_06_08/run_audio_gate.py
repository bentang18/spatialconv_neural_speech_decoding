"""Empirical per-word AUDIO alignment gate for the v14 P3 distillation chain.

WHY THIS EXISTS (P3 alignment audit, 2026-06-08)
------------------------------------------------
P3 distillation pairs the student encoder against a *whole-movie Whisper-encoder
cache* sliced at ``round(movie_onset_s * 50)`` frames (see
``speech_decoding.extractors.whisper_target.WhisperTargetExtractor`` and
``scripts/neuroprobe/build_bt_teacher_cache.py``). That cache is built from OUR
rip audio ``audio/bt_16k/<movie>.wav``. ``movie_onset_s`` is BT's transcript
``features.csv['start']`` (the MOVIE clock). So the teacher target is only valid
if, at every BT word time ``[start, end]``, OUR wav's audio content matches the
audio BT transcribed. A constant offset or a slow drift between BT's movie clock
and our rip's clock silently feeds the student Whisper features of the WRONG
audio -> P3 cannot learn.

WHAT THIS CHECKS
----------------
For each movie:
  1. Read BT ``features.csv``: per-word ``start, end, rms, pitch, magnitude``.
  2. From OUR ``audio/bt_16k/<movie>.wav``, recompute a comparable per-word
     loudness (RMS) and an F0 / voicing proxy over the SAME ``[start, end]``
     window.
  3. Pearson-correlate ours-vs-BT across ALL words (whole film length).
  4. Sliding-window correlation across the film timeline to localize any region
     where alignment breaks (drift / wrong reel / missing scene).

The absolute feature SCALES differ (BT used its own extractor at its own sample
rate; ``pitch``/``magnitude`` are not in our units). Pearson r is invariant to
per-feature affine scaling, so a high whole-film r and uniformly high
sliding-window r prove TEMPORAL alignment regardless of unit mismatch. RMS is
the primary, most-trustworthy channel (real loudness, monotone in our recompute);
pitch is corroborating.

DEPENDENCIES: numpy + scipy + soundfile only (librosa NOT required; the DCC
training venv lacks it). If soundfile is absent, falls back to
``scipy.io.wavfile``.

USAGE
-----
  # default audit set (the 3 films the audit specified)
  .venv/bin/python reports/bt_alignment_p3_audit_today/run_audio_gate.py

  # whole corpus (all 21 films present in audio/bt_16k + transcripts.zip)
  .venv/bin/python reports/bt_alignment_p3_audit_today/run_audio_gate.py --all

  # explicit subset
  .venv/bin/python reports/bt_alignment_p3_audit_today/run_audio_gate.py \
      --movies megamind lotr-1 the-martian

Writes ``audio_gate_results.json`` next to this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy import signal

# --- repo-anchored paths (script lives in reports/bt_alignment_p3_audit_today/) -
REPO = Path(__file__).resolve().parents[2]
WAV_DIR = REPO / "audio" / "bt_16k"
TRANSCRIPTS_ZIP = REPO / ".cache" / "braintreebank" / "transcripts.zip"
OUT_JSON = Path(__file__).resolve().parent / "audio_gate_results.json"

DEFAULT_MOVIES = ["megamind", "lotr-1", "the-martian"]

# Pitch search range for the autocorrelation F0 proxy (human speech F0 band).
F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0
# A word window shorter than this (s) can't host a reliable F0 estimate.
MIN_PITCH_WIN_S = 0.04
# Sliding-window correlation: words per bin and bins to slide.
SLIDE_WORDS = 200
SLIDE_STEP = 100
# A sliding bin with rms-r below this is flagged as a misaligned region.
SLIDE_R_FLOOR = 0.45


# --------------------------------------------------------------------------- #
# audio IO
# --------------------------------------------------------------------------- #
def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Return (float32 mono in ~[-1,1], sample_rate). Prefer soundfile; fall
    back to scipy.io.wavfile (normalizing int PCM by 2^(bits-1))."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1).astype(np.float32)
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)
    except ImportError:
        from scipy.io import wavfile

        sr, data = wavfile.read(str(path))
        orig = data.dtype
        data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1).astype(np.float32)
        if np.issubdtype(orig, np.signedinteger):
            data = data / float(2 ** (8 * orig.itemsize - 1))
        return np.ascontiguousarray(data, dtype=np.float32), int(sr)


# --------------------------------------------------------------------------- #
# BT transcript loading (straight from the zip, no extraction to disk needed)
# --------------------------------------------------------------------------- #
def load_bt_words(movie: str) -> dict[str, np.ndarray]:
    """Read per-word ``start,end,rms,pitch,magnitude`` from the transcript zip.

    Keeps only rows with a finite ``start`` AND a finite ``rms`` (BT leaves
    non-onset word-pieces' timing/feature cells blank). Returns column arrays
    sorted by ``start``."""
    import csv
    import io

    inner = f"transcripts/{movie}/features.csv"
    want = ["start", "end", "rms", "pitch", "magnitude"]
    cols: dict[str, list[float]] = {c: [] for c in want}
    with zipfile.ZipFile(TRANSCRIPTS_ZIP) as zf:
        with zf.open(inner) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
            hdr = next(reader)
            idx = {h: i for i, h in enumerate(hdr)}
            for c in want:
                if c not in idx:
                    raise KeyError(f"{movie}: features.csv missing column {c!r}")
            for row in reader:
                s = row[idx["start"]]
                rms = row[idx["rms"]]
                if not s or not rms:
                    continue
                try:
                    vals = {c: float(row[idx[c]]) for c in want}
                except ValueError:
                    continue
                if not np.isfinite(vals["start"]) or not np.isfinite(vals["rms"]):
                    continue
                for c in want:
                    cols[c].append(vals[c])
    arr = {c: np.asarray(cols[c], dtype=np.float64) for c in want}
    order = np.argsort(arr["start"])
    return {c: arr[c][order] for c in want}


# --------------------------------------------------------------------------- #
# our-side per-word feature recompute
# --------------------------------------------------------------------------- #
def word_rms_ours(
    wav: np.ndarray, sr: int, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """RMS of OUR wav over each [start, end] window. NaN if window invalid/empty."""
    n = len(wav)
    out = np.full(len(starts), np.nan, dtype=np.float64)
    lo = np.clip(np.round(starts * sr).astype(np.int64), 0, n)
    hi = np.clip(np.round(ends * sr).astype(np.int64), 0, n)
    for i in range(len(starts)):
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        seg = wav[a:b].astype(np.float64)
        out[i] = np.sqrt(np.mean(seg * seg))
    return out


def word_pitch_ours(
    wav: np.ndarray, sr: int, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """Autocorrelation F0 proxy (Hz) over each [start, end] window.

    Used only as corroboration. We do not match BT's pitch units; Pearson r is
    affine-invariant, and even our absolute Hz vs BT's relative pitch should
    co-vary if the audio is aligned. NaN where the window is too short or
    unvoiced (flat autocorrelation)."""
    n = len(wav)
    out = np.full(len(starts), np.nan, dtype=np.float64)
    lo = np.clip(np.round(starts * sr).astype(np.int64), 0, n)
    hi = np.clip(np.round(ends * sr).astype(np.int64), 0, n)
    lag_min = int(sr / F0_MAX_HZ)
    lag_max = int(sr / F0_MIN_HZ)
    min_len = int(MIN_PITCH_WIN_S * sr)
    for i in range(len(starts)):
        a, b = lo[i], hi[i]
        if b - a < max(min_len, lag_max + 1):
            continue
        seg = wav[a:b].astype(np.float64)
        seg = seg - seg.mean()
        e = np.dot(seg, seg)
        if e <= 1e-12:
            continue
        ac = signal.correlate(seg, seg, mode="full")
        ac = ac[len(seg) - 1:]  # nonneg lags
        if lag_max >= len(ac):
            continue
        window = ac[lag_min:lag_max + 1]
        if window.size == 0:
            continue
        peak = int(np.argmax(window)) + lag_min
        # Voicing: peak autocorr must be a real fraction of zero-lag energy.
        if ac[peak] / ac[0] < 0.3:
            continue
        out[i] = sr / float(peak)
    return out


# --------------------------------------------------------------------------- #
# correlation helpers
# --------------------------------------------------------------------------- #
def pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Pearson r over finite, positive-variance pairs. Returns (r, n_used)."""
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m], b[m]
    if x.size < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan"), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def best_shift_per_quarter(
    wav: np.ndarray,
    sr: int,
    starts: np.ndarray,
    ends: np.ndarray,
    bt_rms: np.ndarray,
    grid_s: np.ndarray | None = None,
) -> list[dict]:
    """Strongest drift probe: recompute OUR log-RMS at a grid of time shifts and
    report, per film-quarter, the shift that maximizes correlation with BT.

    Perfect alignment with no drift => the argmax shift is the SAME (≈0) in every
    quarter. A walking argmax across quarters is the unambiguous drift signature
    (the rip clock ticks at a different rate than BT's movie clock). This is more
    sensitive than whole-film r, which a slow drift only mildly depresses."""
    if grid_s is None:
        grid_s = np.round(np.arange(-1.0, 1.01, 0.05), 2)

    def logsafe(x):
        out = np.full_like(x, np.nan)
        m = np.isfinite(x) & (x > 0)
        out[m] = np.log(x[m])
        return out

    bt_log = logsafe(bt_rms)
    # Precompute our shifted log-RMS once per grid point (reused across quarters).
    by_shift = {
        float(sh): logsafe(word_rms_ours(wav, sr, starts + sh, ends + sh))
        for sh in grid_s
    }
    n = len(starts)
    out = []
    for q in range(4):
        sl = slice(q * n // 4, (q + 1) * n // 4)
        best_r, best_sh = float("-inf"), float("nan")
        for sh, ours_log in by_shift.items():
            r, used = pearson(ours_log[sl], bt_log[sl])
            if np.isfinite(r) and r > best_r:
                best_r, best_sh = r, sh
        out.append(
            {
                "quarter": q + 1,
                "t_lo_min": round(float(starts[q * n // 4]) / 60, 2),
                "t_hi_min": round(float(starts[min((q + 1) * n // 4, n - 1)]) / 60, 2),
                "best_shift_s": best_sh,
                "r_at_best": None if best_r == float("-inf") else round(best_r, 4),
            }
        )
    return out


def sliding_corr(
    starts: np.ndarray,
    ours: np.ndarray,
    bt: np.ndarray,
    win: int = SLIDE_WORDS,
    step: int = SLIDE_STEP,
) -> list[dict]:
    """Per-bin Pearson r across the film timeline (words already start-sorted)."""
    bins = []
    n = len(starts)
    for s0 in range(0, max(1, n - win + 1), step):
        s1 = min(s0 + win, n)
        r, used = pearson(ours[s0:s1], bt[s0:s1])
        bins.append(
            {
                "word_lo": int(s0),
                "word_hi": int(s1),
                "t_lo_s": float(starts[s0]),
                "t_hi_s": float(starts[s1 - 1]),
                "r": r,
                "n_used": used,
            }
        )
    return bins


# --------------------------------------------------------------------------- #
# per-film driver
# --------------------------------------------------------------------------- #
def audit_film(movie: str) -> dict:
    t0 = time.time()
    wav_path = WAV_DIR / f"{movie}.wav"
    result: dict = {"film": movie}
    if not wav_path.is_file():
        result["error"] = f"wav missing: {wav_path}"
        return result

    bt = load_bt_words(movie)
    wav, sr = load_wav_mono(wav_path)
    wav_dur = len(wav) / sr

    starts, ends = bt["start"], bt["end"]
    bt_last = float(ends.max())
    result.update(
        {
            "n_words": int(len(starts)),
            "wav_dur_s": float(wav_dur),
            "bt_last_word_end_s": bt_last,
            "wav_minus_bt_margin_s": float(wav_dur - bt_last),
            "bt_start_range_s": [float(starts.min()), float(starts.max())],
            "sample_rate": int(sr),
        }
    )
    # Out-of-bounds words: BT word starts that fall beyond our wav are a hard
    # alignment failure (our rip is too short / cut wrong).
    oob = int(np.sum(starts >= wav_dur))
    result["words_past_wav_end"] = oob

    ours_rms = word_rms_ours(wav, sr, starts, ends)
    ours_pitch = word_pitch_ours(wav, sr, starts, ends)

    # log-RMS correlation (loudness is heavy-tailed; log linearizes it).
    def logsafe(x):
        out = np.full_like(x, np.nan)
        m = np.isfinite(x) & (x > 0)
        out[m] = np.log(x[m])
        return out

    r_rms, n_rms = pearson(logsafe(ours_rms), logsafe(bt["rms"]))
    r_rms_lin, _ = pearson(ours_rms, bt["rms"])
    r_pitch, n_pitch = pearson(ours_pitch, bt["pitch"])
    # magnitude: BT spectral magnitude vs our RMS (both track speech energy).
    r_mag, n_mag = pearson(logsafe(ours_rms), logsafe(bt["magnitude"]))

    result["pearson"] = {
        "rms_log": {"r": r_rms, "n": n_rms},
        "rms_linear": {"r": r_rms_lin, "n": n_rms},
        "pitch_autocorr": {"r": r_pitch, "n": n_pitch},
        "magnitude_vs_ours_rms": {"r": r_mag, "n": n_mag},
    }

    result["best_shift_per_quarter"] = best_shift_per_quarter(
        wav, sr, starts, ends, bt["rms"]
    )

    slide = sliding_corr(starts, logsafe(ours_rms), logsafe(bt["rms"]))
    result["sliding_rms_log"] = slide
    bad = [
        b
        for b in slide
        if np.isfinite(b["r"]) and b["r"] < SLIDE_R_FLOOR
    ]
    result["misaligned_regions"] = [
        {
            "t_lo_s": b["t_lo_s"],
            "t_hi_s": b["t_hi_s"],
            "t_lo_min": round(b["t_lo_s"] / 60, 2),
            "t_hi_min": round(b["t_hi_s"] / 60, 2),
            "r": b["r"],
        }
        for b in bad
    ]
    finite_slide = [b["r"] for b in slide if np.isfinite(b["r"])]
    result["sliding_summary"] = {
        "n_bins": len(slide),
        "n_bad_bins": len(bad),
        "min_r": float(min(finite_slide)) if finite_slide else float("nan"),
        "median_r": float(np.median(finite_slide)) if finite_slide else float("nan"),
    }
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Audit every film present.")
    g.add_argument("--movies", nargs="+", help="Explicit movie slugs.")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    if args.all:
        avail = sorted(p.stem for p in WAV_DIR.glob("*.wav"))
        with zipfile.ZipFile(TRANSCRIPTS_ZIP) as zf:
            have_tr = {
                n.split("/")[1]
                for n in zf.namelist()
                if n.startswith("transcripts/") and n.endswith("features.csv")
            }
        movies = [m for m in avail if m in have_tr]
        skipped = [m for m in avail if m not in have_tr]
        if skipped:
            print(f"  (no transcript, skipping: {skipped})")
    elif args.movies:
        movies = args.movies
    else:
        movies = DEFAULT_MOVIES

    print(f"Audio alignment gate — {len(movies)} film(s): {movies}")
    results = []
    for mv in movies:
        print(f"  [{mv}] ...", flush=True)
        r = audit_film(mv)
        results.append(r)
        if "error" in r:
            print(f"    ERROR: {r['error']}")
            continue
        p = r["pearson"]
        ss = r["sliding_summary"]
        shifts = [q["best_shift_s"] for q in r["best_shift_per_quarter"]]
        print(
            f"    n_words={r['n_words']} oob={r['words_past_wav_end']} "
            f"| rms_log r={p['rms_log']['r']:.4f} (n={p['rms_log']['n']}) "
            f"pitch r={p['pitch_autocorr']['r']:.4f} "
            f"| slide median_r={ss['median_r']:.3f} min_r={ss['min_r']:.3f} "
            f"bad_bins={ss['n_bad_bins']}/{ss['n_bins']} "
            f"| best_shift/quarter={shifts}s "
            f"({r['elapsed_s']}s)"
        )

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wav_dir": str(WAV_DIR),
        "transcripts_zip": str(TRANSCRIPTS_ZIP),
        "params": {
            "f0_min_hz": F0_MIN_HZ,
            "f0_max_hz": F0_MAX_HZ,
            "slide_words": SLIDE_WORDS,
            "slide_step": SLIDE_STEP,
            "slide_r_floor": SLIDE_R_FLOOR,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
