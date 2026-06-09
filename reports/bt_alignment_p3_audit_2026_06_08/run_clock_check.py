"""EMPIRICAL CLOCK GATE for the v14 P3 alignment audit (local data only).

For each of 5 diverse (subject, trial) sessions:
  (1) recompute movie_time -> neural_index mapping from subject_timings,
      accounting for pauses (movie clock freezes, neural index keeps running);
  (2) check whether words_df.est_idx is consistent with mapping(words_df.start);
  (3) decisively compare against the NAIVE round(start*2048) (ignores trigger
      offset = the critical bug);
  (4) quantify offset + drift per session.

Output: clock_check.json next to this script.
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

REPO = "/Users/bentang/Documents/Code/speech"
TIMINGS = os.path.join(REPO, ".cache/braintreebank/subject_timings")
WORDS = os.path.join(
    REPO,
    ".cache/neuroprobe_upstream/neuroprobe/braintreebank_features_time_alignment",
)
SR = 2048

# (subject_id, trial_id, timings_stub, words_stub, movie)
SESSIONS = [
    (1, 1, "sub_1_trial001", "subject1_trial1", "the-martian"),
    (2, 4, "sub_2_trial004", "subject2_trial4", "avengers-infinity-war"),
    (3, 1, "sub_3_trial001", "subject3_trial1", "lotr-1"),
    (4, 1, "sub_4_trial001", "subject4_trial1", "megamind"),
    (10, 0, "sub_10_trial000", "subject10_trial0", "cars-2"),
]


def build_anchors(t: pd.DataFrame):
    """Return monotonic (movie_time, index) anchors + pause spans.

    During a pause the movie clock freezes while the neural index keeps
    running, so the timings file has a pause row and an unpause row sharing the
    same movie_time but with different (larger) index. For a movie-time ->
    neural-index map we keep, at each movie_time, the LARGEST index (post-pause
    anchor) so playback after the pause maps correctly. Words whose movie_time
    lands inside a frozen pause-span are physically impossible (movie not
    playing); we flag them as pause-edge artifacts.
    """
    mt = t["movie_time"].to_numpy(dtype=float)
    idx = t["index"].to_numpy(dtype=float)
    order = np.argsort(mt, kind="stable")
    mt_s, idx_s = mt[order], idx[order]
    uniq_mt, inv = np.unique(mt_s, return_inverse=True)
    uniq_idx = np.array([idx_s[inv == i].max() for i in range(len(uniq_mt))])
    # pause spans = movie_time values where index jumps far more than playback
    # would predict (slope ~ SR samples/s); record for artifact flagging.
    pause_rows = t[t["type"] == "pause"][["movie_time", "index"]].to_numpy()
    unpause_rows = t[t["type"] == "unpause"][["movie_time", "index"]].to_numpy()
    return uniq_mt, uniq_idx, pause_rows, unpause_rows


def segment_slopes(uniq_mt, uniq_idx):
    """Local samples-per-second slope between consecutive anchors (playback rate)."""
    dmt = np.diff(uniq_mt)
    didx = np.diff(uniq_idx)
    ok = dmt > 1e-6
    return (didx[ok] / dmt[ok])


def jsonify(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


results = {}
for subj, trial, tstub, wstub, movie in SESSIONS:
    t = pd.read_csv(os.path.join(TIMINGS, f"{tstub}_timings.csv"))
    w = pd.read_csv(os.path.join(WORDS, f"{wstub}_words_df.csv"))
    uniq_mt, uniq_idx, pause_rows, unpause_rows = build_anchors(t)

    start = w["start"].to_numpy(dtype=float)
    est_idx = w["est_idx"].to_numpy(dtype=float)

    # (1)+(2) trigger-offset mapping residual
    mapped = np.interp(start, uniq_mt, uniq_idx)
    resid = est_idx - mapped

    # flag words inside a frozen pause span (movie_time between a pause and its
    # unpause share movie_time, so any word at exactly that movie_time is an
    # edge artifact). Identify by nearest-anchor playback-slope sanity: a word
    # whose mapped index lands inside a [pause_index, unpause_index] gap.
    pause_artifact = np.zeros(len(start), dtype=bool)
    for (pmt, pidx), (umt, uidx) in zip(pause_rows, unpause_rows):
        # word's own est_idx falling inside the dead neural span => it sits at a
        # pause edge; interpolation across the frozen movie_time is degenerate.
        inside = (est_idx > pidx) & (est_idx < uidx)
        pause_artifact |= inside

    clean = resid[~pause_artifact]
    abs_clean = np.abs(clean)

    # (3) NAIVE comparison
    naive = est_idx - np.round(start * SR)

    # (4) offset + drift: fit est_idx ~ a*start + b on clean words; the residual
    # drift across the trial = (neural_seconds - movie_seconds) at first vs last.
    neural_s = est_idx / SR
    drift_s = neural_s - start  # per-word offset between the two clocks
    first_off = float(drift_s[0])
    last_off = float(drift_s[-1])

    # classification verdict. The trigger-mapping floor = inter-trigger spacing
    # (~85 ms ≈ 170 samples), so a correctly-mapped est_idx sits within ~one
    # trigger interval of interp(start); the residual median-abs is ~5 samples
    # (2.5 ms, pure quantization, corr-with-movie-time ≈ 0 => no drift). The
    # NAIVE round(start*2048) is off by 10^5–10^6 samples (100–2500 s). A
    # generous 50-sample threshold cleanly separates the two regimes.
    median_abs_clean = float(np.median(abs_clean))
    # correlation of clean residual with movie time: ~0 => no systematic
    # drift, confirming the residual is quantization noise not misalignment.
    drift_corr = float(
        np.corrcoef(start[~pause_artifact], clean)[0, 1]
    ) if (~pause_artifact).sum() > 2 else float("nan")
    is_trigger_mapped = median_abs_clean < 50.0  # within ~1 trigger interval
    is_naive = bool(np.median(np.abs(naive)) < 50.0)  # would be true ONLY if naive

    results[f"subject{subj}_trial{trial}"] = {
        "movie": movie,
        "n_words": int(len(w)),
        "n_timing_triggers": int((t["type"] == "trigger").sum()),
        "n_pauses": int((t["type"] == "pause").sum()),
        "n_pause_artifact_words": int(pause_artifact.sum()),
        "trigger_mapping_residual_samples": {
            "median": float(np.median(clean)),
            "mean": float(clean.mean()),
            "std": float(clean.std()),
            "median_abs": median_abs_clean,
            "p95_abs": float(np.percentile(abs_clean, 95)),
            "p99_abs": float(np.percentile(abs_clean, 99)),
            "max_abs": float(abs_clean.max()),
            "frac_within_1_sample": float((abs_clean < 1).mean()),
            "frac_within_20_samples": float((abs_clean < 20).mean()),
            "corr_residual_vs_movie_time": drift_corr,
        },
        "trigger_mapping_residual_seconds": {
            "median_abs": median_abs_clean / SR,
            "p99_abs": float(np.percentile(abs_clean, 99)) / SR,
        },
        "naive_round_start_x2048_residual_samples": {
            "median": float(np.median(naive)),
            "min": float(naive.min()),
            "max": float(naive.max()),
            "median_abs_seconds": float(np.median(np.abs(naive))) / SR,
        },
        "clock_offset_seconds": {
            "first_word_neural_minus_movie": first_off,
            "last_word_neural_minus_movie": last_off,
            "drift_across_trial": last_off - first_off,
        },
        "playback_slope_samples_per_s": {
            "median": float(np.median(segment_slopes(uniq_mt, uniq_idx))),
            "expected": SR,
        },
        "verdict": {
            "est_idx_is_trigger_offset_mapped": bool(is_trigger_mapped),
            "est_idx_is_naive_round_start_x2048": is_naive,
        },
    }

summary = {
    "audit": "v14 P3 clock gate (movie<->neural alignment)",
    "sample_rate_hz": SR,
    "method": (
        "Recompute movie_time->neural_index from subject_timings (monotonic "
        "movie_time anchors, keep max index per movie_time to absorb pauses); "
        "compare words_df.est_idx to interp(start) and to round(start*2048)."
    ),
    "overall_verdict": (
        "est_idx is TRIGGER-OFFSET-MAPPED (matches interp(start) within a few "
        "samples) and is NOT naive round(start*2048) in every session"
        if all(
            r["verdict"]["est_idx_is_trigger_offset_mapped"]
            and not r["verdict"]["est_idx_is_naive_round_start_x2048"]
            for r in results.values()
        )
        else "MIXED / FAILURE — inspect per-session"
    ),
    "sessions": results,
}

out = os.path.join(REPO, "reports/bt_alignment_p3_audit_today/clock_check.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2, default=jsonify)
print("WROTE", out)
print(json.dumps(summary, indent=2, default=jsonify))
