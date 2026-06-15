"""Precompute the SSL clip-level bad-time-window sidecars — Layer 2 of the v14
bad-electrode defense (``src/speech_decoding/experiments/bad_windows.py``).

Run on DCC AFTER the static-exclusion cache rebuild. The static drop (whole flaky
contacts, dropped pre-CAR in the loader) changes each shaft's CAR reference and
therefore the cached ``|STFT|`` robust-z, so the bad windows must be scanned on the
REBUILT domain — a scan of the old cache would point at the wrong frames.

For each pretrain session this reproduces the encoder's exact dual-band input:

    bt_load_raw (already post-static-drop + native→2048 resample)
      → notch(60) + HPF(0.5) → shaft-CAR
      → dual-band |STFT| cropped to the PRODUCTION freq bins (LOW k1..k14, HIGH k4..k12)
      → per-(electrode, freq-bin) SESSION robust-z  (== SessionRobustZNormalizer)

then tiles the session into ``CLIP_S``-second windows and flags a window BAD iff

    win_cell_max > CELL_MAX_THRESH   OR   n_electrodes_hot(|z| > HOT_Z) >= N_HOT_THRESH

(the Ben-locked rule: a single giant cell, OR a common-mode blow-up across many
contacts at once that the per-cell winsor cannot fix because it is a JOINT pattern).
Contiguous bad windows are merged into spans and written to a per-session sidecar
``btbank{s}_t{t}.json`` (schema = ``bad_windows.load_bad_windows``) in NEURAL seconds
from recording start — the same clock as a Word event's ``start``.

Detection is on the RAW (un-winsored) robust-z: the clip filter and the per-cell
winsor clamp are COMPLEMENTARY layers. Winsor caps what survives inside a kept clip;
this filter removes whole windows whose joint blow-up winsor cannot repair. Measuring
on raw z is what makes the locked thresholds (5000, 1000z) mean what they say.

CAR commutes with the per-channel linear filters (notch/HPF), so applying CAR after
the filter in float32 is numerically equivalent to the production CAR-then-filter.
Memory-frugal: one electrode's STFT at a time.

READ-ONLY w.r.t. the dataset/cache. DCC-only: needs ``ROOT_DIR_BRAINTREEBANK`` + BT
voltage. Array-parallel via ``SLURM_ARRAY_TASK_ID`` (one session per task).

Usage
-----
    # one session locally (laptop has no BT voltage; DCC only)
    .venv/bin/python -m scripts.neuroprobe.precompute_bad_windows \
        --out-dir /work/ht203/v14_bad_windows --session 2 1

    # full pretrain corpus, one Slurm array task per session
    #   sbatch --array=0-12 ... precompute_bad_windows.py --out-dir /work/ht203/v14_bad_windows
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import mne
import numpy as np
import torch
from scipy.signal import stft

mne.set_log_level("ERROR")

from speech_decoding.extractors.normalize import robust_z
from speech_decoding.extractors.reference import parse_shaft
from speech_decoding.extractors.view import (
    STFT_2BAND_HIGH,
    STFT_2BAND_LOW,
    _stft_band_k_range,
)
from speech_decoding.studies.braintreebank.loader import bt_load_raw
from speech_decoding.studies.braintreebank.manifest import V14_PRETRAIN_SESSIONS

# ---- Ben-locked detection rule (do not retune without a fresh per-window scan) ----
CLIP_S: float = 5.0  # SSL clip length (dispatch clip_len); the window grid stride
CELL_MAX_THRESH: float = 5000.0  # single-giant: |z| above this in a window -> drop
HOT_Z: float = 1000.0  # an electrode is "hot" in a window if any cell |z| exceeds this
N_HOT_THRESH: int = 8  # common-mode: >= this many hot electrodes in a window -> drop

NOTCH_HZ: float = 60.0
HPF_HZ: float = 0.5
SIGMA_FLOOR: float = 1e-6  # == SessionRobustZNormalizer / view.session_z_sigma_floor

# Production dual-band geometry (one source of truth = STFT_2BAND_LOW/HIGH in view.py).
# Each entry: (nperseg, hop, k0, k1) with [k0, k1] the INCLUSIVE rfft-bin slice the
# production cache stores (LOW 4-56 Hz -> k1..k14; HIGH 64-192 Hz -> k4..k12).
_BAND_SPECS: list[tuple[int, int, int, int]] = []
for _band in (STFT_2BAND_LOW, STFT_2BAND_HIGH):
    _nperseg, _hop = int(_band["band_nperseg"]), int(_band["band_hop"])
    _k0, _k1 = _stft_band_k_range(
        _band["band_f_lo_hz"], _band["band_f_hi_hz"], nperseg=_nperseg
    )
    _BAND_SPECS.append((_nperseg, _hop, _k0, _k1))


def _robust_z_perbin(mag: np.ndarray) -> np.ndarray:
    """Per-(freq-bin) robust z over the session time axis. Wraps the PRODUCTION
    ``extractors.normalize.robust_z`` directly (torch lower-median, matching
    ``SessionRobustZNormalizer.fit``) so the precompute is byte-faithful to the
    cache the encoder reads — no median-convention drift from a numpy reimpl.
    ``mag`` is ``(F, T_frames)`` for one electrode/band; z over the last axis."""
    z = robust_z(torch.from_numpy(mag), sigma_floor=SIGMA_FLOOR, dim=-1)
    return z.numpy()


def _merge_bad_windows(
    bad_idx: list[int], clip_s: float, total_s: float
) -> list[tuple[float, float]]:
    """Merge a sorted list of bad window INDICES into neural-second spans. Window
    ``i`` covers ``[i*clip_s, (i+1)*clip_s)``; the final partial window is capped at
    the session duration. Contiguous bad windows fuse into one span."""
    spans: list[tuple[float, float]] = []
    for i in sorted(bad_idx):
        lo = i * clip_s
        hi = min((i + 1) * clip_s, total_s)
        if spans and lo <= spans[-1][1] + 1e-9:
            spans[-1] = (spans[-1][0], hi)
        else:
            spans.append((lo, hi))
    return spans


def scan_session(subject_id: int, trial_id: int) -> dict:
    """Scan one (subject, trial) session and return its sidecar payload."""
    from neuroprobe.braintreebank_subject import BrainTreebankSubject

    tag = f"btbank{subject_id}_t{trial_id}"
    bt = BrainTreebankSubject(
        subject_id=subject_id, cache=False, coordinates_type="cortical"
    )
    # bt_load_raw already drops the static-excluded contacts pre-CAR and resamples
    # native -> 2048 Hz, so the measurement is on the rebuilt-cache domain.
    data, ch_names, sfreq = bt_load_raw(bt, trial_id=trial_id, subject_id=subject_id)
    ch_names = list(ch_names)
    sfreq = float(sfreq)
    data = np.asarray(data, dtype=np.float32)

    # notch + HPF (float64 internally), then shaft-CAR (production reference path).
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="seeg", verbose=False)
    raw = mne.io.RawArray(data.astype(np.float64), info, verbose=False)
    raw.load_data()
    raw.notch_filter(NOTCH_HZ, phase="zero", filter_length="auto", verbose=False)
    raw.filter(HPF_HZ, None, verbose=False)
    filt = np.asarray(raw.get_data(), dtype=np.float32)  # (C, N), pre-CAR
    del raw, data
    gc.collect()

    shafts = [parse_shaft(c)[0] for c in ch_names]
    for sh in set(shafts):
        rows = np.array([i for i, s in enumerate(shafts) if s == sh])
        if rows.size:
            filt[rows] -= filt[rows].mean(axis=0, keepdims=True)

    n_samples = filt.shape[-1]
    total_s = n_samples / sfreq
    n_windows = int(np.ceil(total_s / CLIP_S))
    win_cell_max = np.zeros(n_windows, np.float32)
    n_hot = np.zeros(n_windows, np.int32)

    for i in range(filt.shape[0]):
        x = filt[i]
        elec_win_max = np.zeros(n_windows, np.float32)  # max |z| over BOTH bands
        for nperseg, hop, k0, k1 in _BAND_SPECS:
            _, _, z_stft = stft(
                x, fs=sfreq, nperseg=nperseg, noverlap=nperseg - hop,
                boundary=None, padded=False,  # type: ignore[arg-type]  # scipy: None disables boundary ext
            )
            mag = np.abs(z_stft[k0 : k1 + 1]).astype(np.float32)  # crop to prod bins
            az = np.abs(_robust_z_perbin(mag))  # (F_band, T_frames)
            per_frame = az.max(axis=0)
            t_start = (np.arange(per_frame.shape[0]) * hop) / sfreq
            win = np.clip((t_start / CLIP_S).astype(np.int64), 0, n_windows - 1)
            np.maximum.at(elec_win_max, win, per_frame)
            del z_stft, mag, az, per_frame
        np.maximum(win_cell_max, elec_win_max, out=win_cell_max)
        n_hot += (elec_win_max > HOT_Z).astype(np.int32)
    del filt
    gc.collect()

    bad_idx = [
        i
        for i in range(n_windows)
        if win_cell_max[i] > CELL_MAX_THRESH or n_hot[i] >= N_HOT_THRESH
    ]
    bad_windows_s = _merge_bad_windows(bad_idx, CLIP_S, total_s)

    return {
        "session": tag,
        "subject_id": subject_id,
        "trial_id": trial_id,
        "bad_windows_s": [[float(lo), float(hi)] for lo, hi in bad_windows_s],
        # diagnostics (ignored by load_bad_windows)
        "rule": {
            "clip_s": CLIP_S,
            "cell_max_thresh": CELL_MAX_THRESH,
            "hot_z": HOT_Z,
            "n_hot_thresh": N_HOT_THRESH,
        },
        "n_elec": len(ch_names),
        "duration_s": float(total_s),
        "n_windows": int(n_windows),
        "n_bad_windows": int(len(bad_idx)),
        "frac_bad": float(len(bad_idx) / n_windows) if n_windows else 0.0,
        "max_cell_max": float(win_cell_max.max()) if n_windows else 0.0,
        "max_n_hot": int(n_hot.max()) if n_windows else 0,
    }


def _resolve_sessions(arg_sessions: str | None) -> list[tuple[int, int]]:
    """Sessions to scan: ``--sessions "2:1,2:2,..."`` override, else the
    leaderboard-legal v14 pretrain corpus (the corpus the clip filter protects)."""
    if arg_sessions:
        out = []
        for tok in arg_sessions.split(","):
            s, t = tok.split(":")
            out.append((int(s), int(t)))
        return out
    return list(V14_PRETRAIN_SESSIONS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="sidecar output directory (= bad_window_dir)")
    ap.add_argument("--session", nargs=2, type=int, metavar=("SUBJECT", "TRIAL"),
                    help="scan exactly this one session (overrides array indexing)")
    ap.add_argument("--sessions", default=None,
                    help='comma list "S:T,S:T,..." to scan (array indexes into it)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.session is not None:
        todo = [(args.session[0], args.session[1])]
    else:
        sessions = _resolve_sessions(args.sessions)
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        todo = [sessions[int(task)]] if task is not None else sessions

    for subject_id, trial_id in todo:
        result = scan_session(subject_id, trial_id)
        out_path = os.path.join(args.out_dir, f"{result['session']}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"[{result['session']}] {result['duration_s']/60:.1f}min  "
            f"{result['n_windows']} win  bad={result['n_bad_windows']} "
            f"({100*result['frac_bad']:.2f}%)  max_cell_max={result['max_cell_max']:.0f}  "
            f"max_n_hot={result['max_n_hot']}  -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
