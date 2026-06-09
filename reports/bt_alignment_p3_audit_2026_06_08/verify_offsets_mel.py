"""RIGOROUS mel-based verification that the proposed per-film clock offsets
actually realign fox (+1.75 s) and lotr-2 (+0.1 s -> +1.0 s @ ~100 min).

WHY (P3 alignment audit, 2026-06-08; Ben: "rigorously review that the offsets
will have fixed the problem")
---------------------------------------------------------------------------
The scalar-RMS gate (run_audio_gate.py / run_scale_gate.py) is a 1-D loudness
proxy. This uses BT's 128-d per-word ``mel`` vector (features.csv['mel'], a JSON
128-vector) — the richest per-word acoustic feature BT ships. A misaligned window
scrambles all 128 bins; the aligned window does not.

METHOD
------
For OUR rip we build our own 128-mel filterbank (HTK mel, scipy STFT, NO librosa)
and compute a per-word 128-d mel over [start+offset, end+offset]. Score(offset) =
mean over the 128 bins of Pearson r( our_bin_trajectory_across_words ,
bt_bin_trajectory_across_words ), after per-bin log + z-score. This is invariant
to any filterbank-definition mismatch between us and BT: the mismatch is constant
across offsets, so the offset that MAXIMIZES the score and the SHARPNESS of that
peak are the rigorous alignment signal. We sweep offset and report:
  * peak offset + peak score (should match the RMS-derived offset),
  * sharpness (score drop at peak +- 0.25 s and +- 0.5 s),
  * clean-film controls at offset 0 to set the achievable bar.

A proposed offset is VALIDATED iff: peak is at (or within ~0.1 s of) the proposed
offset, peak score is within the clean-film control band, and the peak is sharp
(score falls clearly within +-0.5 s). Otherwise -> re-rip.

scipy/numpy/soundfile only.
"""

from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

REPO = Path(__file__).resolve().parents[2]
WAV_DIR = REPO / "audio" / "bt_16k"
TRANSCRIPTS_ZIP = REPO / ".cache" / "braintreebank" / "transcripts.zip"
OUT_JSON = Path(__file__).resolve().parent / "verify_offsets_mel_results.json"

SR = 16000
N_FFT = 400          # 25 ms
HOP = 160            # 10 ms
N_MELS = 128
FMIN, FMAX = 0.0, 8000.0


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """(N_MELS, N_FFT//2+1) HTK triangular mel filterbank."""
    n_freqs = N_FFT // 2 + 1
    fft_freqs = np.linspace(0, SR / 2, n_freqs)
    m_pts = np.linspace(hz_to_mel(FMIN), hz_to_mel(FMAX), N_MELS + 2)
    f_pts = mel_to_hz(m_pts)
    fb = np.zeros((N_MELS, n_freqs), dtype=np.float32)
    for i in range(N_MELS):
        lo, ctr, hi = f_pts[i], f_pts[i + 1], f_pts[i + 2]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    return fb


def film_mel(wav: np.ndarray) -> np.ndarray:
    """(n_frames, N_MELS) power-mel of the whole film, memory-frugal chunking."""
    fb = mel_filterbank()
    win = np.hanning(N_FFT).astype(np.float32)
    frames = sliding_window_view(wav, N_FFT)[::HOP]  # view, no copy
    n = frames.shape[0]
    mel = np.empty((n, N_MELS), dtype=np.float32)
    CH = 20000
    for i in range(0, n, CH):
        fr = frames[i:i + CH].astype(np.float32) * win
        spec = np.abs(np.fft.rfft(fr, axis=1)).astype(np.float32) ** 2
        mel[i:i + CH] = spec @ fb.T
    return mel


def load_bt(movie: str):
    """per-word start, end, and 128-d mel (rows with finite start + parseable mel)."""
    import csv
    inner = f"transcripts/{movie}/features.csv"
    starts, ends, mels = [], [], []
    with zipfile.ZipFile(TRANSCRIPTS_ZIP) as zf, zf.open(inner) as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
        hdr = next(reader)
        ix = {h: i for i, h in enumerate(hdr)}
        for row in reader:
            s = row[ix["start"]]
            mcell = row[ix["mel"]]
            if not s or not mcell:
                continue
            try:
                st = float(s); en = float(row[ix["end"]]); mv = json.loads(mcell)
            except (ValueError, json.JSONDecodeError):
                continue
            if not (np.isfinite(st) and len(mv) == N_MELS):
                continue
            starts.append(st); ends.append(en); mels.append(mv)
    order = np.argsort(starts)
    return (np.asarray(starts)[order], np.asarray(ends)[order],
            np.asarray(mels, dtype=np.float64)[order])


def word_mel_ours(mel: np.ndarray, starts, ends, offset: float) -> np.ndarray:
    """(n_words, N_MELS) mean power-mel over each [start+off, end+off] window."""
    n_fr = mel.shape[0]
    lo = np.clip(np.round((starts + offset) * SR / HOP).astype(np.int64), 0, n_fr)
    hi = np.clip(np.round((ends + offset) * SR / HOP).astype(np.int64), 0, n_fr)
    out = np.full((len(starts), N_MELS), np.nan, dtype=np.float64)
    for i in range(len(starts)):
        a, b = lo[i], hi[i]
        if b <= a:
            continue
        out[i] = mel[a:b].mean(axis=0)
    return out


def zlog(x: np.ndarray) -> np.ndarray:
    """log then per-column z-score across words (nan-safe)."""
    out = np.log(np.where(x > 0, x, np.nan))
    mu = np.nanmean(out, axis=0, keepdims=True)
    sd = np.nanstd(out, axis=0, keepdims=True)
    return (out - mu) / np.where(sd > 1e-9, sd, np.nan)


def score(our_mel: np.ndarray, bt_mel_z: np.ndarray) -> float:
    """mean per-bin Pearson r between our and BT z-log mel trajectories."""
    ours_z = zlog(our_mel)
    rs = []
    for b in range(N_MELS):
        x, y = ours_z[:, b], bt_mel_z[:, b]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            continue
        rs.append(np.corrcoef(x[m], y[m])[0, 1])
    return float(np.mean(rs)) if rs else float("nan")


def sweep(mel, starts, ends, bt_mel_z, offsets, sl=None):
    if sl is not None:
        starts, ends, bt_mel_z = starts[sl], ends[sl], bt_mel_z[sl]
    out = {}
    for off in offsets:
        out[round(float(off), 3)] = round(score(
            word_mel_ours(mel, starts, ends, off), bt_mel_z), 4)
    return out


def peak(d: dict):
    k = max(d, key=lambda o: (d[o] if np.isfinite(d[o]) else -9))
    return k, d[k]


def main() -> int:
    fb_built = mel_filterbank()
    print(f"mel filterbank {fb_built.shape}, sr={SR} n_fft={N_FFT} hop={HOP}")
    fine = np.round(np.arange(-1.0, 1.01, 0.1), 3)
    results = {}

    # ---- clean-film controls: peak should be ~0, sets the bar ----
    for ctrl in ["megamind", "the-martian", "lotr-1"]:
        t0 = time.time()
        wav, _ = _load(ctrl)
        mel = film_mel(wav)
        s, e, m = load_bt(ctrl)
        bz = zlog(m)
        d = sweep(mel, s, e, bz, fine)
        po, ps = peak(d)
        results[ctrl] = {"role": "control", "n_words": len(s), "peak_offset": po,
                         "peak_score": ps, "score_at_0": d.get(0.0),
                         "sweep": d, "elapsed_s": round(time.time() - t0, 1)}
        print(f"[CONTROL {ctrl}] peak@{po}s score={ps} (score@0={d.get(0.0)}) "
              f"n={len(s)} {results[ctrl]['elapsed_s']}s")

    # ---- fox: expect sharp peak near +1.75 ----
    t0 = time.time()
    wav, _ = _load("fantastic-mr-fox")
    mel = film_mel(wav)
    s, e, m = load_bt("fantastic-mr-fox")
    bz = zlog(m)
    foff = np.round(np.arange(0.5, 3.01, 0.1), 3)
    d = sweep(mel, s, e, bz, foff)
    po, ps = peak(d)
    results["fantastic-mr-fox"] = {
        "role": "fix", "proposed_offset": 1.75, "n_words": len(s),
        "peak_offset": po, "peak_score": ps,
        "score_at_proposed": d.get(1.75),
        "score_at_proposed_minus_0.5": d.get(round(1.75 - 0.5, 3)),
        "score_at_proposed_plus_0.5": d.get(round(1.75 + 0.5, 3)),
        "score_at_0_for_ref": round(score(word_mel_ours(mel, s, e, 0.0), bz), 4),
        "sweep": d, "elapsed_s": round(time.time() - t0, 1)}
    print(f"[FIX fox] peak@{po}s score={ps} | @1.75={d.get(1.75)} "
          f"@1.25={d.get(1.25)} @2.25={d.get(2.25)} @0={results['fantastic-mr-fox']['score_at_0_for_ref']} "
          f"{results['fantastic-mr-fox']['elapsed_s']}s")

    # ---- lotr-2: front half ~+0.1, back half ~+1.0; localize step ----
    t0 = time.time()
    wav, _ = _load("lotr-2")
    mel = film_mel(wav)
    s, e, m = load_bt("lotr-2")
    bz = zlog(m)
    n = len(s)
    # split at ~100 min by word time
    step_t = 100 * 60
    front = s < step_t
    back = ~front
    loff = np.round(np.arange(-0.5, 1.81, 0.1), 3)
    d_front = sweep(mel, s, e, bz, loff, sl=front)
    d_back = sweep(mel, s, e, bz, loff, sl=back)
    pf, sf = peak(d_front); pb, sb = peak(d_back)
    # localize step: per-eighth peak offset
    per8 = []
    for q in range(8):
        sl = np.zeros(n, bool); sl[q * n // 8:(q + 1) * n // 8] = True
        dq = sweep(mel, s, e, bz, loff, sl=sl)
        pq, sq = peak(dq)
        per8.append({"bin": q + 1, "t_lo_min": round(float(s[q * n // 8]) / 60, 1),
                     "peak_offset": pq, "peak_score": sq})
    results["lotr-2"] = {
        "role": "fix", "proposed": "front +0.1, back(>=100min) +1.0", "n_words": n,
        "front_peak_offset": pf, "front_peak_score": sf,
        "back_peak_offset": pb, "back_peak_score": sb,
        "per_eighth": per8,
        "sweep_front": d_front, "sweep_back": d_back,
        "elapsed_s": round(time.time() - t0, 1)}
    print(f"[FIX lotr-2] front peak@{pf}s score={sf} | back peak@{pb}s score={sb} "
          f"{results['lotr-2']['elapsed_s']}s")
    print("  per-eighth peak offsets:",
          [(b["t_lo_min"], b["peak_offset"], b["peak_score"]) for b in per8])

    OUT_JSON.write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "params": {"sr": SR, "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS,
                    "fmin": FMIN, "fmax": FMAX},
         "results": results}, indent=2))
    print(f"\nWrote {OUT_JSON}")
    return 0


def _load(movie: str):
    import soundfile as sf
    data, sr = sf.read(str(WAV_DIR / f"{movie}.wav"), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)
    if sr != SR:
        raise SystemExit(f"{movie}: sr={sr} != {SR}")
    return np.ascontiguousarray(data, dtype=np.float32), sr


if __name__ == "__main__":
    sys.exit(main())
