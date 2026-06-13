"""Regression guard — the P3 rip<->BT clock offsets must still match the AUDIO.

``whisper_target._MOVIE_CLOCK_OFFSET_S`` shifts the Whisper teacher slice per
film (fantastic-mr-fox +1.75 s; lotr-2 +0.1 then +1.0 s at ~100 min). Its own
docstring warns: "tied to the CURRENT rip files — re-rip => re-measure." Nothing
enforced that. A silent re-rip (or a wrong film->WAV mapping) would leave the
table pointing at the wrong lead-in and mis-align every P3 teacher target for
that film — a plausible-but-wrong (S9-class) failure with no crash.

This guard re-derives the offset straight from the WAV bytes, by a method wholly
independent of the original 2026-06-08 derivation: cross-correlate the
transcript word-onset train against the WAV's speech-band spectral-flux onset
envelope (see ``scripts/neuroprobe/independent_audio_alignment.py``). It asserts
the two non-trivial table entries still hold:

  * fox global offset is large (~+1.7 s), clearly distinct from the 0-films;
  * lotr-2 is small (~0) before the 6000 s reel-join and large (~+1 s) after it.

The onset feature carries a small (~0.05-0.07 s) constant group-delay bias, so
the bands are wide (±0.35 s); they still separate "corrected" from "uncorrected"
unambiguously. Skips when the gitignored ``audio/bt_16k`` rips are absent (CI).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
_PROBE = _REPO / "scripts" / "neuroprobe" / "independent_audio_alignment.py"
_FPS = 100
_BAND = (200.0, 3000.0)
_SIGMA_S = 0.05
_SEARCH_S = 3.0
_STEP_S = 0.02
_WIN_S = 600.0  # the offset is constant within a window; one big block dilutes corr to noise
_HOP_S = 300.0
_MIN_CORR = 0.04  # per-window conclusiveness floor (matches the probe)


def _load_probe():
    spec = importlib.util.spec_from_file_location("_iaa_probe", _PROBE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _words_csv(p, sub: int, trial: int) -> Path:
    return p._words_dir() / f"subject{sub}_trial{trial}_words_df.csv"


def _measure(p, film: str, sub: int, trial: int, lo_s: float | None = None,
             hi_s: float | None = None) -> tuple[float, int]:
    """Windowed-median offset over the [lo_s, hi_s) movie-time slice.

    The rip<->BT offset is piecewise-constant, so within a window it is one value
    but a single block-wide cross-correlation averages over too much non-speech
    and washes the peak out (corr -> ~0). So we slide ``_WIN_S`` windows (the same
    method the probe uses for step detection), keep windows whose peak clears
    ``_MIN_CORR``, and return ``(median_offset, n_conclusive_windows)``.
    """
    import pandas as pd

    wav = p.WAV_DIR / f"{film}.wav"
    wdf_path = _words_csv(p, sub, trial)
    if not wav.exists() or not wdf_path.exists():
        pytest.skip(f"rip or words_df absent for {film} (sub{sub} t{trial})")
    e = p._onset_envelope(wav, _FPS, _BAND)
    wdf = pd.read_csv(wdf_path)
    start = wdf["start"].to_numpy(dtype=float)
    start = start[np.isfinite(start)]
    m = p._onset_train(start, len(e), _FPS, _SIGMA_S)
    a = int((lo_s or 0.0) * _FPS)
    b = int(hi_s * _FPS) if hi_s is not None else len(e)
    a, b = max(a, 0), min(b, len(e))

    win_f, hop_f = int(_WIN_S * _FPS), int(_HOP_S * _FPS)
    offs: list[float] = []
    for w0 in range(a, max(a + 1, b - win_f // 2), hop_f):
        w1 = min(w0 + win_f, b)
        if w1 - w0 < win_f // 2:
            break
        ew, mw = e[w0:w1], m[w0:w1]
        if mw.sum() < 30:  # need enough word onsets in the window
            continue
        off, corr = p._best_offset(ew, mw, _FPS, -_SEARCH_S, _SEARCH_S, _STEP_S)
        if corr >= _MIN_CORR:
            offs.append(off)
    if not offs:
        return (float("nan"), 0)
    return (float(np.median(offs)), len(offs))


@pytest.fixture(scope="module")
def probe():
    if not _PROBE.exists():
        pytest.skip("independent_audio_alignment.py probe absent")
    return _load_probe()


@pytest.mark.slow
def test_fox_offset_still_large(probe) -> None:
    """fox must still measure a large positive lead-in (~+1.75 s), not ~0."""
    off, n = _measure(probe, "fantastic-mr-fox", 1, 0)
    assert n >= 2, f"fox alignment signal too weak ({n} conclusive windows) — investigate"
    assert 1.40 <= off <= 2.10, (
        f"fox rip<->BT offset measured {off:+.3f}s from audio; table says +1.75s. "
        "A re-rip or wrong film->WAV mapping would land here — re-measure with "
        "scripts/neuroprobe/independent_audio_alignment.py and update "
        "_MOVIE_CLOCK_OFFSET_S."
    )


@pytest.mark.slow
def test_fox_offset_consistent_across_subjects(probe) -> None:
    """Both subjects who watched fox (sub1, sub5) must measure the same offset."""
    o1, n1 = _measure(probe, "fantastic-mr-fox", 1, 0)
    o5, n5 = _measure(probe, "fantastic-mr-fox", 5, 0)
    assert n1 >= 2 and n5 >= 2, f"fox signal too weak (sub1 {n1}, sub5 {n5} windows)"
    assert abs(o1 - o5) <= 0.20, (
        f"fox offset disagrees across subjects (sub1 {o1:+.3f}s vs sub5 {o5:+.3f}s) — "
        "same film+rip must give one offset; a per-subject divergence is the S9 class."
    )


@pytest.mark.slow
def test_lotr2_reel_join_step(probe) -> None:
    """lotr-2 must be ~0 before the 6000 s reel-join and ~+1 s after it."""
    pre, n_pre = _measure(probe, "lotr-2", 3, 2, lo_s=0.0, hi_s=5700.0)
    post, n_post = _measure(probe, "lotr-2", 3, 2, lo_s=7800.0, hi_s=None)
    assert n_pre >= 2 and n_post >= 2, (
        f"lotr-2 alignment signal too weak (pre {n_pre}, post {n_post} windows)"
    )
    assert -0.30 <= pre <= 0.45, (
        f"lotr-2 pre-reel-join offset {pre:+.3f}s; table says +0.1s (expect ~0)."
    )
    assert 0.65 <= post <= 1.35, (
        f"lotr-2 post-reel-join offset {post:+.3f}s; table says +1.0s. The reel-join "
        "step at ~6000s must survive — re-measure if this fails (ledger LG13)."
    )
    assert (post - pre) >= 0.55, (
        f"lotr-2 reel-join STEP collapsed (pre {pre:+.3f} -> post {post:+.3f}); "
        "the +1.0s second-half correction is the bug fixed in the 2026-06-08 audit."
    )
