"""Stage-0 L.5.P8 — 60 Hz residual power post-notch (and pre-notch reference).

For each BT Lite session, computes per-channel PSD via Welch over the full
session voltage with and without the L.3.F1 notch filter (60/120/180 Hz).
Reports the per-channel residual power at 60 Hz relative to the 50-70 Hz
broadband floor in dB. Flag if median residual > floor by > 6 dB; bad notch
or wrong harmonic set.

L.3 was frozen at no-op (F0 wins), so this probe is informative for whether a
future notch pass would be material. Single sbatch job; loops 12 sessions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal

from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS
from speech_decoding.views import apply_temporal_filter_inplace


def _band_power_db(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return np.full(psd.shape[0], np.nan)
    band_p = psd[:, mask].mean(axis=1)
    return 10.0 * np.log10(band_p + 1e-30)


def main() -> None:
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bt_root: Path = args.bt_root.resolve()
    repo_dir: Path = args.neuroprobe_repo.resolve()
    sys.path.insert(0, str(repo_dir))
    os.environ["ROOT_DIR_BRAINTREEBANK"] = str(bt_root)

    from neuroprobe.braintreebank_subject import BrainTreebankSubject  # noqa: E402
    from neuroprobe.train_test_splits import subset_electrodes  # noqa: E402
    import neuroprobe.config as nconfig  # noqa: E402

    sr = int(nconfig.SAMPLING_RATE)
    sessions = list(BT_LITE_SESSIONS)
    notch_freqs = (60.0, 120.0, 180.0)

    rows: list[dict[str, float | int | str]] = []
    for s_idx, (sub_id, trial_id) in enumerate(sessions):
        t_session = time.time()
        subject = BrainTreebankSubject(sub_id, cache=True, dtype=torch.float32)
        subset_electrodes(subject, lite=True, nano=False)
        subject.load_neural_data(trial_id)
        x = subject.neural_data_cache[trial_id]  # (C, T) torch tensor

        x_pre = x.detach().cpu().numpy().astype(np.float64)
        nperseg = min(int(sr * 4), x_pre.shape[1])
        freqs, psd_pre = signal.welch(x_pre, fs=sr, nperseg=nperseg, axis=-1)

        x_post_t = x.clone()
        apply_temporal_filter_inplace(
            x_post_t, sampling_rate=sr, notch_freqs=notch_freqs, hpf_hz=0.0,
        )
        x_post = x_post_t.detach().cpu().numpy().astype(np.float64)
        del x_post_t
        _, psd_post = signal.welch(x_post, fs=sr, nperseg=nperseg, axis=-1)

        for ref_label, psd in (("pre", psd_pre), ("post", psd_post)):
            p_60   = _band_power_db(psd, freqs, 59.0, 61.0)
            p_120  = _band_power_db(psd, freqs, 119.0, 121.0)
            p_180  = _band_power_db(psd, freqs, 179.0, 181.0)
            p_floor = _band_power_db(psd, freqs, 50.0, 58.0)
            rows.append({
                "subject_id": int(sub_id),
                "trial_id": int(trial_id),
                "n_channels": int(psd.shape[0]),
                "ref": ref_label,
                "median_60hz_minus_floor_db":  float(np.nanmedian(p_60   - p_floor)),
                "median_120hz_minus_floor_db": float(np.nanmedian(p_120  - p_floor)),
                "median_180hz_minus_floor_db": float(np.nanmedian(p_180  - p_floor)),
                "p95_60hz_minus_floor_db":     float(np.nanpercentile(p_60 - p_floor, 95)),
            })

        print(
            f"[{s_idx + 1}/{len(sessions)}] sub{sub_id} trial{trial_id}: "
            f"PRE 60Hz {rows[-2]['median_60hz_minus_floor_db']:+.2f}dB  "
            f"POST 60Hz {rows[-1]['median_60hz_minus_floor_db']:+.2f}dB  "
            f"{time.time() - t_session:.1f}s"
        )
        subject.clear_neural_data_cache(trial_id)
        del x_pre, x_post
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "p8_60hz_residual.csv", index=False)
    post = df[df["ref"] == "post"]
    summary = {
        "cell_id": "L.5.P8",
        "n_sessions": int(post["subject_id"].nunique() * post["trial_id"].nunique()),
        "post_median_60hz_minus_floor_db_median":
            float(post["median_60hz_minus_floor_db"].median()),
        "post_p95_60hz_minus_floor_db_max":
            float(post["p95_60hz_minus_floor_db"].max()),
        "kill_threshold_db": 6.0,
        "kill_any":
            bool((post["median_60hz_minus_floor_db"] > 6.0).any()),
        "notch_freqs_hz": list(notch_freqs),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    env_root = os.environ.get("ROOT_DIR_BRAINTREEBANK", "").strip()
    p.add_argument("--bt-root", type=Path, default=Path(env_root) if env_root else None)
    p.add_argument(
        "--neuroprobe-repo", type=Path,
        default=(
            Path("/work/ht203/repo/neuroprobe_upstream")
            if Path("/work/ht203").exists()
            else Path(".cache/neuroprobe_upstream")
        ),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.bt_root is None:
        p.error("--bt-root or ROOT_DIR_BRAINTREEBANK is required")
    return args


if __name__ == "__main__":
    main()
