"""Guard-2 bad-window precompute for the Cogan D-cohort — CACHE-READING slider.

Layer-2 of the bad-electrode defense for the Cogan v3 SSL corpus (guard-1 = the
static contact drop, already baked into the spec cache PRE-CAR via
``cogan_extra_bad`` → ``cogan_load_raw``). Unlike the BT producer
(``precompute_bad_windows.py``), which re-derives the whole front-end from raw
voltage, this reads the ALREADY-BAKED 32 Hz spec cache and slides the detector
over its frames.

Why cache-reading is correct here (and better): in the Cogan DAG the cache is
strictly downstream of guard-1 (static drop applied pre-CAR), so the cached
``|STFT|`` robust-z domain IS the training domain, exactly. Reading it is:
  * domain-exact by construction — measures the identical frames training
    samples; no parallel band-geometry to drift out of lockstep with the cache;
  * fast — no STFT, no shaft-CAR, no 595 GB raw re-read; just the compact cache;
  * one guard with BT — same v3slow/v3mid/hga geometry, same 32 Hz grid, same 1 s
    detection window, same imported detector as the BT ``--frontend v3`` scan.

Per session ``(subject_id, trial_id)``:
    band_{v3slow,v3mid,hga}.npy (raw |STFT| mag, C×F×T @ 32 Hz)
      + band .stats.npz (per-(C,F) session median/σ, the frozen normalizer stats)
    → z = (mag − median) / max(σ, floor)      [PRE-winsor: guard-2 hunts exactly
                                               what the read-time winsor clamp
                                               CANNOT repair, so no clamp here]
    → per-(elec, band) per-1 s-window |z|-max   (the detector's ``ewm`` grid)
    → slow-band per-(elec, window) z-std         (the dropout sentinel ``n_flat``)
  → ``_decide_bad_windows`` (imported verbatim): per-band self-calibrating
    q = P99, hot / cat / abs / flat rules → bad window indices
  → ``_merge_bad_windows`` → [lo, hi] spans → ``cogan{sid}_t{tid}.json``
    (schema = ``bad_windows.load_bad_windows`` / ``cache_index.index_bad_windows``
    — integer ``subject_id`` / ``trial_id`` fields so the v3 consumer keys it).

Per-band multiplier finals (HOT_MULT / CAT_MULT_BY_BAND / ABS_FLOOR_MAD / …) are
Ben-locked; they carry over from BT because the granularity is UNCHANGED at 1 s
(q recalibrates only if the window changes). A knee sanity-check on Cogan's own
cache is prudent before locking, but not a re-derivation.

Runs AFTER the spec cache-bake in the Cogan build DAG; as a SLURM array it takes
one session per task (``--manifest`` + ``SLURM_ARRAY_TASK_ID``) reading the
``/work``-staged cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

from speech_decoding.models.v14_converged_v3.cache_index import (
    index_band_cache,
    parse_key_session,
)

# The LOCKED detector + its constants, shared VERBATIM with the BT scan so BT and
# Cogan are one guard (importing is laptop-safe: precompute_bad_windows' only
# top-level BT import is bt_load_raw; BrainTreebankSubject/neuroprobe is lazy).
from precompute_bad_windows import (  # noqa: E402  (sibling script on sys.path)
    CAT_MULT_BY_BAND,
    FLAT_STD,
    HOT_MULT_BY_BAND,
    SIGMA_FLOOR,
    _decide_bad_windows,
    _merge_bad_windows,
)

# v3 sensor frontend, in the cache's concat order. Lowest band = the dropout
# sentinel (its z-std drives the flat rule); hga inherits its cat_mult=6 override
# through CAT_MULT_BY_BAND by name.
BAND_NAMES: tuple[str, ...] = ("v3slow", "v3mid", "hga")
FLAT_BAND: str = "v3slow"
FPS: float = 32.0  # the v3 uniform frame clock (hop=64 @ 2048 Hz)
DETECT_WINDOW_S: float = 1.0  # v3 detection window (< the 3 s clip → tight spans)


def compute_ewm_from_cache(
    band_mags: list[np.ndarray],
    band_stats: list[tuple[np.ndarray, np.ndarray]],
    band_names: tuple[str, ...],
    *,
    flat_band: str,
    clip_s: float,
    fps: float,
    sigma_floor: float = SIGMA_FLOOR,
) -> tuple[dict[str, np.ndarray], np.ndarray, int, float, int]:
    """Cached |STFT| magnitudes → the detector's per-band ``ewm`` grid + ``n_flat``.

    ``band_mags[b]`` is ``(C, F_b, T)`` raw magnitude (the ``.npy``); ``band_stats[b]``
    is the band's frozen ``(median, sigma)`` (each ``(C, F_b)`` or ``(C, F_b, 1)``).
    Applies the SAME robust-z the encoder reads — median/σ per (channel, freq-bin)
    — but WITHOUT the read-time winsor clamp (guard-2 detects pre-winsor). Returns
    ``(ewm_by_band, n_flat, n_elec, total_s, n_windows)`` in the exact shapes
    ``_decide_bad_windows`` consumes, so the detector is reused untouched."""
    T = int(band_mags[0].shape[2])
    for name, mag in zip(band_names, band_mags):
        if int(mag.shape[2]) != T:
            raise ValueError(
                f"band '{name}' has {mag.shape[2]} frames, expected {T} "
                "(band caches must share the 32 Hz time axis)"
            )
    n_elec = int(band_mags[0].shape[0])
    total_s = T / float(fps)
    n_windows = int(np.ceil(total_s / clip_s)) if T else 0

    # frame → detection-window index (tiled, non-overlapping; last window caps at T)
    t_start = np.arange(T) / float(fps)
    win = np.clip((t_start / clip_s).astype(np.int64), 0, max(n_windows - 1, 0))
    ii = np.broadcast_to(np.arange(n_elec)[:, None], (n_elec, T))
    jj = np.broadcast_to(win[None, :], (n_elec, T))

    ewm_by_band: dict[str, np.ndarray] = {}
    n_flat = np.zeros(n_windows, np.int32)
    for name, mag, (median, sigma) in zip(band_names, band_mags, band_stats):
        mag = np.asarray(mag, dtype=np.float32)
        med = np.asarray(median, dtype=np.float32)
        sig = np.asarray(sigma, dtype=np.float32)
        if med.ndim == 2:
            med = med[:, :, None]
        if sig.ndim == 2:
            sig = sig[:, :, None]
        z = (mag - med) / np.maximum(sig, sigma_floor)  # (C, F, T) signed, pre-winsor
        per_frame = np.abs(z).max(axis=1)  # (C, T): the worst bin per frame

        ewm = np.zeros((n_elec, n_windows), np.float32)
        np.maximum.at(ewm, (ii, jj), per_frame)  # max-pool frames → 1 s windows
        ewm_by_band[name] = ewm

        if name == flat_band:
            # per-(electrode, window) z-std over the window's (freq-bin × frame)
            # cells — a dropout is near-constant → std → 0. Mirrors the BT flat
            # accumulation (F cells per frame), vectorized.
            n_bins = z.shape[1]
            fsum = z.sum(axis=1)  # (C, T)
            fsq = (z * z).sum(axis=1)  # (C, T)
            wsum = np.zeros((n_elec, n_windows), np.float64)
            wsq = np.zeros((n_elec, n_windows), np.float64)
            wcnt = np.zeros(n_windows, np.float64)
            np.add.at(wsum, (ii, jj), fsum)
            np.add.at(wsq, (ii, jj), fsq)
            np.add.at(wcnt, win, float(n_bins))
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = wsum / wcnt
                var = wsq / wcnt - mean * mean
            elec_std = np.sqrt(np.clip(var, 0.0, None))
            elec_std[:, wcnt <= 0] = np.inf  # empty window → never flat
            n_flat = (elec_std < FLAT_STD).sum(axis=0).astype(np.int32)

    return ewm_by_band, n_flat, n_elec, total_s, n_windows


def _entry_for(index: dict, subject_id: int, trial_id: int):
    """The unique cache entry for a session, matched on the sidecar ``key`` (the
    same match ``load_v3_sessions`` uses). >1 or 0 is a fail-loud."""
    hits = [e for k, e in index.items() if parse_key_session(k) == (subject_id, trial_id)]
    if len(hits) != 1:
        raise ValueError(
            f"expected exactly one cache entry for subject {subject_id} trial "
            f"{trial_id}, found {len(hits)}"
        )
    return hits[0]


def scan_session_cache(
    subject_id: int,
    trial_id: int,
    band_cache_dirs: list[str],
    *,
    clip_s: float = DETECT_WINDOW_S,
    fps: float = FPS,
) -> dict:
    """Scan one session's baked spec cache → its bad-window sidecar payload.

    ``band_cache_dirs`` are the 3 band roots in v3 concat order (slow, mid, hga);
    each is resolved to its ``band_hop=64`` leaf by ``index_band_cache``."""
    indexes = [index_band_cache(d) for d in band_cache_dirs]
    entries = [_entry_for(bi, subject_id, trial_id) for bi in indexes]

    ch0 = entries[0].ch_names
    for e in entries[1:]:
        if e.ch_names != ch0:
            raise ValueError(
                f"band caches disagree on channel order for session "
                f"{subject_id}/{trial_id}: {e.ch_names[:3]}... vs {ch0[:3]}..."
            )
    total_frames = entries[0].total_frames
    for e in entries[1:]:
        if e.total_frames != total_frames:
            raise ValueError(
                f"band caches disagree on total_frames for session "
                f"{subject_id}/{trial_id}: {e.total_frames} vs {total_frames}"
            )
    for e in entries:
        if e.sample_rate != int(round(fps)):
            raise ValueError(
                f"cache sample_rate {e.sample_rate} != expected fps {fps} for "
                f"session {subject_id}/{trial_id}"
            )

    band_mags: list[np.ndarray] = []
    band_stats: list[tuple[np.ndarray, np.ndarray]] = []
    for e in entries:
        mm = np.load(e.npy_path, mmap_mode="r")
        band_mags.append(np.asarray(mm, dtype=np.float32))
        del mm
        st = np.load(e.stats_path)
        band_stats.append((st["median"], st["sigma"]))

    ewm_by_band, n_flat, n_elec, total_s, n_windows = compute_ewm_from_cache(
        band_mags, band_stats, BAND_NAMES,
        flat_band=FLAT_BAND, clip_s=clip_s, fps=fps,
    )
    bad_idx, decision = _decide_bad_windows(
        ewm_by_band, n_flat, n_elec,
        hot_mult_by_band=HOT_MULT_BY_BAND, cat_mult_by_band=CAT_MULT_BY_BAND,
    )
    bad_windows_s = _merge_bad_windows(bad_idx, clip_s, total_s)

    tag = f"cogan{int(subject_id)}_t{int(trial_id)}"
    return {
        "session": tag,
        "subject_id": int(subject_id),
        "trial_id": int(trial_id),
        "bad_windows_s": [[float(lo), float(hi)] for lo, hi in bad_windows_s],
        "rule": {"clip_s": clip_s, **decision},
        "n_elec": n_elec,
        "duration_s": float(total_s),
        "n_windows": int(n_windows),
        "n_bad_windows": int(len(bad_idx)),
        "frac_bad": float(len(bad_idx) / n_windows) if n_windows else 0.0,
        "max_n_flat": int(n_flat.max()) if n_windows else 0,
        "n_hot_windows": decision["n_hot_windows"],
        "n_cat_windows": decision["n_cat_windows"],
        "n_abs_windows": decision["n_abs_windows"],
        "n_flat_windows": decision["n_flat_windows"],
    }


def _band_dirs(spec_cache_dir: str) -> list[str]:
    """The 3 band roots under a v3 spec-cache dir (dispatch_v14 ``band_{name}``)."""
    return [os.path.join(spec_cache_dir, f"band_{b}") for b in BAND_NAMES]


def _manifest_sessions(manifest_path: str) -> list[tuple[int, int]]:
    """Manifest CSV → ordered ``(global_subject_id, trial_id)`` list (array index)."""
    with open(manifest_path, newline="") as fh:
        reader = csv.DictReader(fh)
        return [(int(r["global_subject_id"]), int(r["trial_id"])) for r in reader]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True,
                    help="sidecar output directory (= v3 bad_window_dir / span_dir)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--spec-cache-dir",
                     help="v3 spec-cache dir holding band_{v3slow,v3mid,hga}")
    src.add_argument("--band-dir", nargs=3, metavar=("SLOW", "MID", "HGA"),
                     help="the 3 band cache roots explicitly (v3 concat order)")
    ap.add_argument("--session", nargs=2, type=int, metavar=("SUBJECT", "TRIAL"),
                    help="scan exactly this one (global_subject_id, trial_id)")
    ap.add_argument("--manifest",
                    help="cogan manifest CSV; SLURM_ARRAY_TASK_ID indexes its rows")
    ap.add_argument("--detect-window", type=float, default=DETECT_WINDOW_S,
                    help=f"detection-window seconds (default {DETECT_WINDOW_S}; "
                         "tiling stride AND merged-span resolution)")
    ap.add_argument("--fps", type=float, default=FPS, help=f"frame rate (default {FPS})")
    args = ap.parse_args()

    band_dirs = _band_dirs(args.spec_cache_dir) if args.spec_cache_dir else list(args.band_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.session is not None:
        todo = [(args.session[0], args.session[1])]
    elif args.manifest is not None:
        sessions = _manifest_sessions(args.manifest)
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        todo = [sessions[int(task)]] if task is not None else sessions
    else:
        ap.error("provide --session S T or --manifest CSV")

    for subject_id, trial_id in todo:
        result = scan_session_cache(
            subject_id, trial_id, band_dirs,
            clip_s=args.detect_window, fps=args.fps,
        )
        out_path = os.path.join(args.out_dir, f"{result['session']}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        r = result["rule"]
        bands = "  ".join(
            f"{name}[q={pb['q']:.1f},cell_max={pb['max_cell_max']:.0f},"
            f"nhot={pb['max_n_hot']}]"
            for name, pb in r["per_band"].items()
        )
        print(
            f"[{result['session']}] {result['duration_s'] / 60:.1f}min  "
            f"{result['n_windows']} win  bad={result['n_bad_windows']} "
            f"({100 * result['frac_bad']:.2f}%)  "
            f"by_rule[hot={result['n_hot_windows']},cat={result['n_cat_windows']},"
            f"abs={result['n_abs_windows']},flat={result['n_flat_windows']}]  "
            f"{bands}  -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
