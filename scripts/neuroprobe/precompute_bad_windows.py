"""Precompute the SSL clip-level bad-time-window sidecars — Layer 2 of the v14
bad-electrode defense (``src/speech_decoding/experiments/bad_windows.py``).

Run on DCC AFTER the static-exclusion cache rebuild. The static drop (whole flaky
contacts, dropped pre-CAR in the loader) changes each shaft's CAR reference and
therefore the cached ``|STFT|`` robust-z, so the bad windows must be scanned on the
REBUILT domain — a scan of the old cache would point at the wrong frames.

For each pretrain session this reproduces the encoder's exact 3-band input:

    bt_load_raw (already post-static-drop + native→2048 resample)
      → notch(60) + HPF(0.5) → shaft-CAR
      → 3-band |STFT| cropped to the PRODUCTION freq bins (slow / beta / HG, STFT_3BAND_*)
      → per-(electrode, freq-bin) SESSION robust-z  (== SessionRobustZNormalizer)

then tiles the session into ``CLIP_S``-second windows and flags a window BAD iff, for
ANY band b with its OWN self-calibrating scale ``q_b``,

    #hot electrodes(|z|-max > HOT_MULT*q_b) >= max(N_FLOOR, FRAC_HOT*C)   (common-mode)
    OR  win_cell_max_b > CAT_MULT*q_b                                     (catastrophic single cell)
    OR  #flat electrodes(slow-band z-std < FLAT_STD) >= max(N_FLOOR, FRAC_FLAT*C)  (dropout)

Every threshold is PORTABLE — a fraction of the electrode count C, or a multiple of a
per-(session, BAND) self-calibrating scale ``q_b`` (= the P99 of that band's per-window
|z|-max). PER-BAND (#231) because the slow/beta/HG |z| distributions differ: a single
pooled ``q`` is dominated by the loudest band, so a transient in a quieter band slips the
fence. One ``q_b`` per band lets each band's fence ride just above its OWN clean ceiling.
Badness is on |STFT| for ALL bands (incl. the Cartesian slow band — the magnitude
transient is the artifact signal regardless of how the band is stored for the model). The two HIGH-amplitude rules: a common-mode blow-up across many contacts
at once (a JOINT pattern the per-cell winsor cannot repair), OR a single cell so giant
it poisons the micro-batch alone. The third is the complementary LOW-energy rule: a
recording dropout flattens many contacts (low-band z-std -> 0); the loudest cell stays
tiny so it is invisible to amplitude rules and to the winsor cap, yet it collapses that
micro-batch's parcel rank (the once-per-epoch rankme needle, e.g. subj2 t2 ~3925-3965s,
~30/133 contacts flat). Contiguous bad windows are merged into spans and written to a
per-session sidecar ``btbank{s}_t{t}.json`` (schema = ``bad_windows.load_bad_windows``)
in NEURAL seconds from recording start — the same clock as a Word event's ``start``.

Detection is on the RAW (un-winsored) robust-z: the clip filter and the per-cell winsor
clamp are COMPLEMENTARY layers. Winsor caps what survives inside a kept clip; this
filter removes whole windows whose joint blow-up winsor cannot repair. The fences are
intentionally AGGRESSIVE (Ben 2026-06-16): the 2STFT 3-session knee sweep put the healthy
per-window ceiling at ~q*7 and the healthy single-cell ceiling at ~q*11, so HOT_MULT=5 /
CAT_MULT=12 drop hard right up to those ceilings (clean session ~0.1% dropped, all
genuine artifacts). Those multipliers are carried over as PER-BAND PRELIMINARY defaults
(applied to each band's own ``q_b``); Ben locks the per-band finals from the knee sweep
on the rebuilt 3STFT cache (override via HOT_MULT_BY_BAND / CAT_MULT_BY_BAND). Non-
aggressive cleaning leaves rank-collapsing clips in and taxes scaling — the opposite of
the goal.

CAR commutes with the per-channel linear filters (notch/HPF), so applying CAR after
the filter in float32 is numerically equivalent to the production CAR-then-filter.
Memory-frugal: one electrode's STFT at a time.

READ-ONLY w.r.t. the dataset/cache. DCC-only: needs ``ROOT_DIR_BRAINTREEBANK`` + BT
voltage. Array-parallel via ``SLURM_ARRAY_TASK_ID`` (one session per task).

Usage (DCC; scripts/ is not a package, so run by PATH not ``-m``)
-----
    # one session (DCC only — laptop has no BT voltage)
    ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
      .venv/bin/python scripts/neuroprobe/precompute_bad_windows.py \
        --out-dir /work/ht203/v14_bad_windows --session 2 1

    # full pretrain corpus, one Slurm array task per session (13 sessions -> 0-12)
    #   sbatch --array=0-12 ... \
    #     "ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
    #      .venv/bin/python scripts/neuroprobe/precompute_bad_windows.py \
    #        --out-dir /work/ht203/v14_bad_windows"
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
    STFT_3BAND_BETA,
    STFT_3BAND_HG,
    STFT_3BAND_SLOW,
    _stft_band_k_range,
)
from speech_decoding.studies.braintreebank.loader import bt_load_raw
from speech_decoding.studies.braintreebank.manifest import V14_PRETRAIN_SESSIONS

# ---- portable, self-calibrating, PER-BAND detection rule ----
# Every threshold below is a FRACTION of the electrode count or a MULTIPLE of a
# per-(session, BAND) scale ``q_band`` (= P99 of all per-(electrode, window) |z|-max
# WITHIN that band), so the rule transfers to any cohort / electrode count / noise
# level without retuning. Going per-band (#231) fixes the 2STFT pooled-``q`` flaw: the
# 3STFT bands (Cartesian slow, |STFT| beta, |STFT| HG) have very different |z|
# distributions, so a single pooled ``q`` is dominated by the loudest band and a
# transient in a quieter band slips under the fence. One ``q_band`` per band lets each
# band's fences ride just above its OWN clean ceiling.
#
# PRELIMINARY multipliers (Ben locks finals after profiling the rebuilt 3STFT cache):
# carried over from the 2STFT knee sweep (healthy per-window ceiling ~7q, single-cell
# ~11q → HOT_MULT 5 / CAT_MULT 12), applied per-band. ``HOT_MULT``/``CAT_MULT`` may be
# overridden per band via the dicts below once each band's knee is measured.
CLIP_S: float = 5.0  # SSL clip length (dispatch clip_len); the window grid stride
Q_PCT: float = 99.0  # per-(session,band) scale q = this pct of all (elec, window) |z|-max
FRAC_HOT: float = 0.05  # >= this fraction of electrodes hot together -> common-mode drop
FRAC_FLAT: float = 0.05  # >= this fraction of electrodes flat together -> dropout drop
N_FLOOR: int = 3  # ...but never fewer than this many electrodes (small-array guard)
HOT_MULT: float = 5.0  # an electrode is "hot" in a window if its |z|-max > HOT_MULT * q
CAT_MULT: float = 12.0  # any single cell > CAT_MULT * q -> drop (poisons the batch alone)
FLAT_STD: float = 0.05  # an electrode is "flat" in a window if its slow-band z-std < this

# Per-band multiplier OVERRIDES (PRELIMINARY: empty → every band uses the scalar
# HOT_MULT/CAT_MULT above). Ben fills these from the per-band knee sweep on the rebuilt
# cache, e.g. ``HOT_MULT_BY_BAND = {"slow": 6.0}``. Keys must be in {"slow","beta","hg"}.
HOT_MULT_BY_BAND: dict[str, float] = {}
CAT_MULT_BY_BAND: dict[str, float] = {}

NOTCH_HZ: float = 60.0
HPF_HZ: float = 0.5
SIGMA_FLOOR: float = 1e-6  # == SessionRobustZNormalizer / view.session_z_sigma_floor

# Production 3-band geometry (one source of truth = STFT_3BAND_* in view.py). The flat
# (dropout) rule runs on the FIRST band, so SLOW is listed first to keep the lowest band
# as the dropout sentinel (its low-frequency z-std collapses first on a recording freeze).
# Each entry: (name, nperseg, hop, k0, k1) with [k0, k1] the INCLUSIVE rfft-bin slice the
# production cache stores. Badness is detected on |STFT| for ALL bands (incl. the Cartesian
# slow band): the magnitude transient is the artifact signal regardless of how the band is
# stored for the model — so no cartesian special-case is needed in the scan.
_BAND_ORDER: tuple[str, ...] = ("slow", "beta", "hg")
_BAND_DICTS = {"slow": STFT_3BAND_SLOW, "beta": STFT_3BAND_BETA, "hg": STFT_3BAND_HG}
_BAND_SPECS: list[tuple[str, int, int, int, int]] = []
for _name in _BAND_ORDER:
    _band = _BAND_DICTS[_name]
    _nperseg, _hop = int(_band["band_nperseg"]), int(_band["band_hop"])
    _k0, _k1 = _stft_band_k_range(
        float(_band["band_f_lo_hz"]), float(_band["band_f_hi_hz"]), nperseg=_nperseg
    )
    _BAND_SPECS.append((_name, _nperseg, _hop, _k0, _k1))


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


def _decide_bad_windows(
    ewm_by_band: dict[str, np.ndarray],
    n_flat: np.ndarray,
    n_elec: int,
    *,
    q_pct: float = Q_PCT,
    frac_hot: float = FRAC_HOT,
    frac_flat: float = FRAC_FLAT,
    n_floor: int = N_FLOOR,
    hot_mult: float = HOT_MULT,
    cat_mult: float = CAT_MULT,
    hot_mult_by_band: dict[str, float] | None = None,
    cat_mult_by_band: dict[str, float] | None = None,
) -> tuple[list[int], dict]:
    """Pure PER-BAND bad-window decision (#231) — the testable core, no I/O.

    ``ewm_by_band[name]`` is ``(n_elec, n_windows)``: each electrode's per-window
    |z|-max WITHIN that band. ``n_flat`` is ``(n_windows,)``: the count of
    slow-band-flat electrodes per window. Each band self-calibrates its own scale
    ``q_band`` (P99 of its own (elec, window) |z|-max) and fences ride above it; a
    window is BAD iff ANY band fires hot (common-mode) or cat (single-cell), OR the
    slow band fires flat (dropout). Returns ``(sorted bad_idx, diagnostics)``."""
    hot_mult_by_band = hot_mult_by_band or {}
    cat_mult_by_band = cat_mult_by_band or {}
    n_windows = int(n_flat.shape[0])
    hot_count_thresh = max(n_floor, int(np.ceil(frac_hot * n_elec)))
    flat_count_thresh = max(n_floor, int(np.ceil(frac_flat * n_elec)))
    hot_bad = np.zeros(n_windows, bool)
    cat_bad = np.zeros(n_windows, bool)
    per_band: dict[str, dict] = {}
    for name, ewm in ewm_by_band.items():
        q = float(np.percentile(ewm, q_pct)) if ewm.size else 0.0
        hm = float(hot_mult_by_band.get(name, hot_mult))
        cm = float(cat_mult_by_band.get(name, cat_mult))
        hot_fence, cat_fence = hm * q, cm * q
        win_cell_max = (
            ewm.max(axis=0) if ewm.shape[0] else np.zeros(n_windows, np.float32)
        )
        n_hot = (ewm > hot_fence).sum(axis=0).astype(np.int32)
        hb = n_hot >= hot_count_thresh
        cb = win_cell_max > cat_fence
        hot_bad |= hb
        cat_bad |= cb
        per_band[name] = {
            "q": q, "hot_mult": hm, "cat_mult": cm,
            "hot_fence": float(hot_fence), "cat_fence": float(cat_fence),
            "n_hot_windows": int(hb.sum()), "n_cat_windows": int(cb.sum()),
            "max_n_hot": int(n_hot.max()) if n_windows else 0,
            "max_cell_max": float(win_cell_max.max()) if n_windows else 0.0,
        }
    flat_bad = n_flat >= flat_count_thresh
    bad = hot_bad | cat_bad | flat_bad
    bad_idx = [int(w) for w in np.nonzero(bad)[0]]
    diag = {
        "q_pct": q_pct, "frac_hot": frac_hot, "frac_flat": frac_flat,
        "n_floor": n_floor, "flat_std": FLAT_STD,
        "hot_count_thresh": int(hot_count_thresh),
        "flat_count_thresh": int(flat_count_thresh),
        "per_band": per_band,
        "n_hot_windows": int(hot_bad.sum()),
        "n_cat_windows": int(cat_bad.sum()),
        "n_flat_windows": int(flat_bad.sum()),
        "max_n_flat": int(n_flat.max()) if n_windows else 0,
    }
    return bad_idx, diag


def compute_ewm_by_band(
    subject_id: int, trial_id: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, int, float, int]:
    """Streaming pass — the EXPENSIVE part of the scan, factored out so the
    aggressiveness sweep can pay it once per session and re-decide cheaply over a
    fence grid (``_decide_bad_windows`` is pure numpy on these arrays).

    BT load (static contacts already dropped pre-CAR, native→2048 Hz) → notch+HPF
    → shaft-CAR → per-band STFT robust-z, accumulating each electrode's per-window
    |z|-max WITHIN each band plus the per-window slow-band flat-electrode count.
    Returns ``(ewm_by_band, n_flat, n_elec, total_s, n_windows)`` — byte-identical
    to what ``scan_session`` fed ``_decide_bad_windows`` before the extraction."""
    from neuroprobe.braintreebank_subject import BrainTreebankSubject

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
    n_elec = int(filt.shape[0])
    # per-(electrode, window) |z|-max kept PER BAND so each band self-calibrates its own
    # scale q_band (#231) — the slow/beta/HG |z| distributions differ, so one pooled q
    # would let a transient in a quieter band slip the fence.
    ewm_by_band = {
        name: np.zeros((n_elec, n_windows), np.float32) for name, *_ in _BAND_SPECS
    }
    n_flat = np.zeros(n_windows, np.int32)  # electrodes flat (slow-band z-std < FLAT_STD)
    flat_band = _BAND_SPECS[0][0]  # SLOW: lowest band is the dropout sentinel

    for i in range(n_elec):
        for name, nperseg, hop, k0, k1 in _BAND_SPECS:
            _, _, z_stft = stft(
                filt[i], fs=sfreq, nperseg=nperseg, noverlap=nperseg - hop,
                boundary=None, padded=False,  # type: ignore[arg-type]  # scipy: None disables boundary ext
            )
            mag = np.abs(z_stft[k0 : k1 + 1]).astype(np.float32)  # |STFT|, crop to prod bins
            z = _robust_z_perbin(mag)  # signed (F_band, T_frames)
            per_frame = np.abs(z).max(axis=0)
            t_start = (np.arange(per_frame.shape[0]) * hop) / sfreq
            win = np.clip((t_start / CLIP_S).astype(np.int64), 0, n_windows - 1)
            np.maximum.at(ewm_by_band[name][i], win, per_frame)
            if name == flat_band:  # slow band: per-window z-std -> flat/dropout detection
                fsum = z.sum(axis=0, dtype=np.float64)
                fsq = np.square(z, dtype=np.float64).sum(axis=0)
                w_sum = np.zeros(n_windows, np.float64)
                w_sq = np.zeros(n_windows, np.float64)
                w_cnt = np.zeros(n_windows, np.float64)
                np.add.at(w_sum, win, fsum)
                np.add.at(w_sq, win, fsq)
                np.add.at(w_cnt, win, float(z.shape[0]))  # F_slow cells per frame
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean = w_sum / w_cnt
                    var = w_sq / w_cnt - mean * mean
                elec_std = np.where(w_cnt > 0, np.sqrt(np.maximum(var, 0.0)), np.inf)
                n_flat += (elec_std < FLAT_STD).astype(np.int32)
            del z_stft, mag, z, per_frame
    del filt
    gc.collect()

    return ewm_by_band, n_flat, n_elec, total_s, n_windows


def scan_session(subject_id: int, trial_id: int) -> dict:
    """Scan one (subject, trial) session and return its sidecar payload."""
    tag = f"btbank{subject_id}_t{trial_id}"
    ewm_by_band, n_flat, n_elec, total_s, n_windows = compute_ewm_by_band(
        subject_id, trial_id,
    )
    bad_idx, decision = _decide_bad_windows(
        ewm_by_band, n_flat, n_elec,
        hot_mult_by_band=HOT_MULT_BY_BAND, cat_mult_by_band=CAT_MULT_BY_BAND,
    )
    bad_windows_s = _merge_bad_windows(bad_idx, CLIP_S, total_s)

    return {
        "session": tag,
        "subject_id": subject_id,
        "trial_id": trial_id,
        "bad_windows_s": [[float(lo), float(hi)] for lo, hi in bad_windows_s],
        # diagnostics (ignored by load_bad_windows) — now PER BAND (#231): every
        # band's self-calibrated q + fences live under rule["per_band"][name].
        "rule": {"clip_s": CLIP_S, **decision},
        "n_elec": n_elec,
        "duration_s": float(total_s),
        "n_windows": int(n_windows),
        "n_bad_windows": int(len(bad_idx)),
        "frac_bad": float(len(bad_idx) / n_windows) if n_windows else 0.0,
        "max_n_flat": int(n_flat.max()) if n_windows else 0,
        "n_hot_windows": decision["n_hot_windows"],
        "n_cat_windows": decision["n_cat_windows"],
        "n_flat_windows": decision["n_flat_windows"],
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
        r = result["rule"]
        bands = "  ".join(
            f"{name}[q={pb['q']:.1f},cell_max={pb['max_cell_max']:.0f},"
            f"nhot={pb['max_n_hot']}]"
            for name, pb in r["per_band"].items()
        )
        print(
            f"[{result['session']}] {result['duration_s']/60:.1f}min  "
            f"{result['n_windows']} win  bad={result['n_bad_windows']} "
            f"({100*result['frac_bad']:.2f}%)  "
            f"need[hot>={r['hot_count_thresh']},flat>={r['flat_count_thresh']}]/{result['n_elec']}elec  "
            f"by_rule[hot={result['n_hot_windows']},cat={result['n_cat_windows']},"
            f"flat={result['n_flat_windows']}]  max_n_flat={result['max_n_flat']}  "
            f"{bands}  -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
