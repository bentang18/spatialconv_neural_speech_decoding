#!/usr/bin/env python3
"""P2 — INDEPENDENT audio<->transcript clock-offset probe (fully local).

WHY THIS EXISTS (the blind spot it pierces)
-------------------------------------------
The per-film rip<->BT clock offsets in
``speech_decoding.extractors.whisper_target._MOVIE_CLOCK_OFFSET_S``
(fantastic-mr-fox +1.75 s; lotr-2 +0.1 then +1.0 s at ~100 min) gate the P3
Whisper teacher slice: ``onset_s = movie_onset_s + offset``. Those values were
derived ONCE in ``reports/bt_alignment_p3_audit_2026_06_08/`` (RMS + mel gates).
That derivation and the production code share the same tooling and the same
hand-picked anchor windows, so a re-rip, a wrong film->WAV mapping, or a hidden
reel-join in a film currently assumed constant-0 would slip past both.

This probe is an INDEPENDENT third source. It never touches the derivation. For
each (subject, trial, film) it:

  1. Reads the transcript word table (``*_words_df.csv``): each word's BT
     movie-clock span ``[start, end]`` (seconds).
  2. Reads OUR rip WAV (``audio/bt_16k/<film>.wav``) and computes its RMS energy
     envelope from the raw audio bytes.
  3. Cross-correlates the transcript SPEECH MASK (1 during ``[start,end]``)
     against the energy envelope. The lag that maximises correlation IS the
     rip<->transcript offset: the WAV plays each word at ``start + offset``, so
     the mask, shifted forward by ``offset``, lands on speech energy.
  4. Compares the MEASURED offset to the value the production code would apply,
     ``_movie_clock_offset(film, t)``.

It also slides the window across each film (``--window-s`` / ``--window-hop-s``)
and measures the offset PER WINDOW. A film whose offset is constant in the table
but drifts/steps across windows is the alignment analogue of the S9 bug: a
silent per-region metadata inconsistency that yields plausible-but-wrong teacher
targets. Such a film is reported as a STEP finding even if its global offset
"passes".

Exit 0 iff every conclusive window agrees with the table within tolerance and no
unexpected step is found; exit 1 otherwise. Windows with weak speech signal
(corr below ``--min-corr``) are INCONCLUSIVE, never a hard FAIL.

Fully local: needs only ``.cache/braintreebank/subject_timings`` (for the
film map via neuroprobe), the vendored ``*_words_df.csv`` tables, and
``audio/bt_16k/*.wav``. No DCC, no h5, no model.

Run:  .venv/bin/python scripts/neuroprobe/independent_audio_alignment.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

# neuroprobe.config hard-requires this at import; we only read a static dict +
# the vendored word tables, so a dummy root is fine (no h5 is opened).
os.environ.setdefault("ROOT_DIR_BRAINTREEBANK", "/tmp/bt_dummy_alignment_probe")

REPO = Path(__file__).resolve().parents[2]
WAV_DIR = REPO / "audio" / "bt_16k"
# words_df tables ship inside the neuroprobe wheel; prefer the installed copy,
# fall back to the .cache mirror.
_WORDS_DIR_CANDIDATES = [
    REPO / ".venv/lib/python3.12/site-packages/neuroprobe/braintreebank_features_time_alignment",
    REPO / ".cache/neuroprobe_2026/neuroprobe/braintreebank_features_time_alignment",
]


def _words_dir() -> Path:
    for d in _WORDS_DIR_CANDIDATES:
        if d.is_dir():
            return d
    raise FileNotFoundError(
        "no braintreebank_features_time_alignment dir found; looked in:\n  "
        + "\n  ".join(str(d) for d in _WORDS_DIR_CANDIDATES)
    )


def _onset_envelope(wav_path: Path, fps: int, band_hz: tuple[float, float]) -> np.ndarray:
    """Speech-band spectral-flux onset envelope of the WAV.

    Broadband RMS energy is dominated by the orchestral score / SFX in action and
    animated films (it goes NEGATIVE-correlated with dialogue), so it cannot
    locate speech. Spectral flux — the sum of positive frame-to-frame magnitude
    increases within the speech formant band — fires on speech onsets but stays
    low under sustained music, giving a feature that tracks WHEN words are
    spoken. Audio is decimated to 8 kHz first (speech < 4 kHz) to halve memory.
    """
    import soundfile as sf
    from scipy.signal import resample_poly, stft

    x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr > 8000:
        x = resample_poly(x, 8000, sr)
        sr = 8000
    nperseg = int(round(sr * 0.025))  # 25 ms
    hop = max(1, int(round(sr / fps)))  # 1/fps-second hop
    noverlap = nperseg - hop
    f, _, Z = stft(
        x, fs=sr, nperseg=nperseg, noverlap=noverlap,
        boundary=None,  # type: ignore[arg-type]  # scipy accepts None (no padding); stub says str
        padded=False,
    )
    mag = np.abs(Z)  # (n_freq, n_frames)
    lo, hi = band_hz
    band = (f >= lo) & (f <= hi)
    mag = mag[band, :]
    flux = np.diff(mag, axis=1)
    flux[flux < 0] = 0.0  # half-wave rectify: onsets only
    onset = flux.sum(axis=0)  # (n_frames-1,)
    return onset


def _onset_train(start: np.ndarray, n_frames: int, fps: int, sigma_s: float) -> np.ndarray:
    """Gaussian-smoothed impulse train at transcript word-onset times."""
    train = np.zeros(n_frames, dtype=np.float64)
    idx = np.round(start * fps).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    np.add.at(train, idx, 1.0)
    sigma_f = max(1.0, sigma_s * fps)
    radius = int(round(3 * sigma_f))
    k = np.arange(-radius, radius + 1)
    kern = np.exp(-0.5 * (k / sigma_f) ** 2)
    kern /= kern.sum()
    return np.convolve(train, kern, mode="same")


def _zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def _best_offset(
    e: np.ndarray, m: np.ndarray, fps: int, lo_s: float, hi_s: float, step_s: float
) -> tuple[float, float]:
    """Lag (s) maximising corr(energy[k], mask[k - lag]); returns (offset_s, corr).

    mask shifted forward by +lag aligns transcript speech with WAV energy, so a
    positive offset means the WAV plays each word LATER than its transcript time.
    """
    ez = _zscore(e)
    if ez.std() == 0 or m.std() == 0:
        return (0.0, 0.0)
    step_f = max(1, int(round(step_s * fps)))
    best_off, best_c = 0.0, -2.0
    for df in range(int(round(lo_s * fps)), int(round(hi_s * fps)) + 1, step_f):
        ms = np.zeros_like(m)
        if df > 0:
            ms[df:] = m[: len(m) - df]
        elif df < 0:
            ms[:df] = m[-df:]
        else:
            ms = m
        msz = _zscore(ms)
        if msz.std() == 0:
            continue
        c = float((ez * msz).mean())
        if c > best_c:
            best_c, best_off = c, df / fps
    return (best_off, best_c)


def _iter_sessions() -> list[tuple[int, int, str]]:
    from neuroprobe.config import BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING as M

    out: list[tuple[int, int, str]] = []
    for key, film in sorted(M.items()):
        mt = re.match(r"btbank(\d+)_(\d+)$", key)
        if not mt:
            continue
        out.append((int(mt.group(1)), int(mt.group(2)), film))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=int, default=100, help="onset-envelope frame rate (Hz)")
    ap.add_argument("--band-lo", type=float, default=200.0, help="speech-band low edge (Hz)")
    ap.add_argument("--band-hi", type=float, default=3000.0, help="speech-band high edge (Hz)")
    ap.add_argument("--onset-sigma-s", type=float, default=0.05, help="word-onset smoothing sigma (s)")
    ap.add_argument("--search-s", type=float, default=3.0, help="+/- offset search range (s)")
    ap.add_argument("--step-s", type=float, default=0.02, help="offset search resolution (s)")
    ap.add_argument("--tol-s", type=float, default=0.25, help="measured-vs-table tolerance (s)")
    ap.add_argument("--min-corr", type=float, default=0.06, help="min corr for a conclusive window")
    ap.add_argument("--window-s", type=float, default=900.0, help="sliding window length (s)")
    ap.add_argument("--window-hop-s", type=float, default=450.0, help="sliding window hop (s)")
    ap.add_argument("--step-tol-s", type=float, default=0.30,
                    help="max spread across a film's windows before flagging a hidden step")
    ap.add_argument("--only-film", default=None, help="restrict to one film slug")
    ap.add_argument("--verbose-windows", action="store_true",
                    help="print every sliding-window (mid_t, measured, table, corr)")
    ap.add_argument("--list", action="store_true", help="list sessions and exit")
    args = ap.parse_args()

    from speech_decoding.extractors.whisper_target import (
        _MOVIE_CLOCK_OFFSET_S,
        _movie_clock_offset,
    )

    sessions = _iter_sessions()
    if args.list:
        for s, t, f in sessions:
            print(f"sub{s} trial{t} -> {f}")
        return 0

    words_dir = _words_dir()
    # Group by film so each WAV envelope is computed once and we can cross-check
    # that different subjects watching the same film report the same offset.
    envelope_cache: dict[str, np.ndarray] = {}
    per_film_global: dict[str, list[tuple[str, float, float]]] = {}

    failures: list[str] = []
    step_findings: list[str] = []
    skipped: list[str] = []
    rows: list[str] = []

    for sub, trial, film in sessions:
        if args.only_film and film != args.only_film:
            continue
        wav = WAV_DIR / f"{film}.wav"
        wdf_path = words_dir / f"subject{sub}_trial{trial}_words_df.csv"
        if not wav.exists():
            skipped.append(f"sub{sub} trial{trial} {film}: WAV missing ({wav.name})")
            continue
        if not wdf_path.exists():
            skipped.append(f"sub{sub} trial{trial} {film}: words_df missing ({wdf_path.name})")
            continue

        import pandas as pd

        wdf = pd.read_csv(wdf_path)
        start = wdf["start"].to_numpy(dtype=float)
        end = wdf["end"].to_numpy(dtype=float)
        ok = np.isfinite(start) & np.isfinite(end) & (end > start)
        start, end = start[ok], end[ok]
        if len(start) < 50:
            skipped.append(f"sub{sub} trial{trial} {film}: too few words ({len(start)})")
            continue

        if film not in envelope_cache:
            envelope_cache[film] = _onset_envelope(wav, args.fps, (args.band_lo, args.band_hi))
        e = envelope_cache[film]
        n_frames = len(e)
        if n_frames == 0:
            skipped.append(f"sub{sub} trial{trial} {film}: empty WAV envelope")
            continue
        m = _onset_train(start, n_frames, args.fps, args.onset_sigma_s)

        g_off, g_corr = _best_offset(e, m, args.fps, -args.search_s, args.search_s, args.step_s)
        per_film_global.setdefault(film, []).append((f"sub{sub}t{trial}", g_off, g_corr))

        # table offset, evaluated at the transcript midpoint (constant films are
        # insensitive to where we sample; piecewise films use their schedule).
        mid_t = float(np.median(start))
        expect = _movie_clock_offset(film, mid_t)
        g_status = (
            "INCONCLUSIVE" if g_corr < args.min_corr
            else "PASS" if abs(g_off - expect) <= args.tol_s
            else "FAIL"
        )
        rows.append(
            f"  sub{sub:>2} t{trial} {film:28s} global meas={g_off:+.3f}s "
            f"table={expect:+.3f}s corr={g_corr:.3f} -> {g_status}"
        )
        if g_status == "FAIL":
            failures.append(
                f"sub{sub} trial{trial} {film}: measured offset {g_off:+.3f}s != "
                f"table {expect:+.3f}s (corr {g_corr:.3f}, tol {args.tol_s}s)"
            )

        # sliding-window measurement: detect a hidden step/drift the table misses.
        win_f = int(round(args.window_s * args.fps))
        hop_f = int(round(args.window_hop_s * args.fps))
        win_offsets: list[tuple[float, float, float]] = []  # (mid_s, off, corr)
        for w0 in range(0, max(1, n_frames - win_f // 2), hop_f):
            w1 = min(w0 + win_f, n_frames)
            if w1 - w0 < win_f // 2:
                break
            ew, mw = e[w0:w1], m[w0:w1]
            if mw.sum() < 5 * args.fps:  # need some speech in the window
                continue
            off, corr = _best_offset(ew, mw, args.fps, -args.search_s, args.search_s, args.step_s)
            if corr >= args.min_corr:
                win_offsets.append(((w0 + w1) / 2 / args.fps, off, corr))

        if args.verbose_windows and win_offsets:
            print(f"  -- windows sub{sub} t{trial} {film}:")
            for t, o, corr in win_offsets:
                print(f"       t={t:7.0f}s meas={o:+.3f}s "
                      f"table={_movie_clock_offset(film, t):+.3f}s corr={corr:.3f}")

        if len(win_offsets) >= 2:
            offs = np.array([o for _, o, _ in win_offsets])
            # expected per-window offsets from the (possibly piecewise) table
            exp = np.array([_movie_clock_offset(film, t) for t, _, _ in win_offsets])
            resid = offs - exp  # remove the KNOWN schedule; flag leftover structure
            spread = float(resid.max() - resid.min())
            if spread > args.step_tol_s:
                detail = ", ".join(
                    f"t={t:.0f}s meas={o:+.2f} exp={x:+.2f}"
                    for (t, o, _), x in zip(win_offsets, exp)
                )
                step_findings.append(
                    f"sub{sub} trial{trial} {film}: residual offset spread {spread:.2f}s "
                    f"after removing the table schedule (>{args.step_tol_s}s) — possible "
                    f"unmodelled reel-join/drift. windows: [{detail}]"
                )

    print("=" * 78)
    print("P2 INDEPENDENT AUDIO<->TRANSCRIPT OFFSET PROBE")
    print(f"table under test (_MOVIE_CLOCK_OFFSET_S): {dict(_MOVIE_CLOCK_OFFSET_S)}")
    print("=" * 78)
    for r in rows:
        print(r)

    # cross-subject consistency: same film, multiple subjects -> same offset.
    print("\n--- same-film cross-subject consistency (global) ---")
    for film, measurements in sorted(per_film_global.items()):
        if len(measurements) < 2:
            continue
        offs = np.array([o for _, o, c in measurements if c >= args.min_corr])
        if len(offs) < 2:
            continue
        spread = float(offs.max() - offs.min())
        tag = "OK" if spread <= args.tol_s else "INCONSISTENT"
        who = ", ".join(f"{w}:{o:+.2f}" for w, o, _ in measurements)
        print(f"  {film:28s} spread={spread:.3f}s [{who}] -> {tag}")
        if spread > args.tol_s:
            failures.append(
                f"{film}: same film, subjects disagree on offset by {spread:.3f}s "
                f"(> tol {args.tol_s}s) — [{who}]"
            )

    if skipped:
        print("\n--- skipped ---")
        for s in skipped:
            print(f"  {s}")

    if step_findings:
        print("\n!!! HIDDEN-STEP / DRIFT FINDINGS (alignment S9-class) !!!")
        for s in step_findings:
            print(f"  {s}")

    print("\n" + "=" * 78)
    if failures or step_findings:
        print(f"RESULT: FAIL — {len(failures)} offset mismatch(es), "
              f"{len(step_findings)} hidden-step finding(s)")
        for f in failures:
            print(f"  FAIL: {f}")
        return 1
    print(f"RESULT: PASS — every conclusive window matches the table "
          f"(tol {args.tol_s}s); no hidden steps. {len(rows)} sessions checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
