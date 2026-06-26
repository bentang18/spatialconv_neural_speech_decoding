"""Per-band robust-z TAIL straight from the built spec cache — no raw reload, no
re-STFT, no Hilbert. The cache already holds everything the WINSOR cap W needs.

Each cached session entry is ``<hash>.npy`` (whole-movie magnitudes, (C, F, T)) +
``<hash>.stats.npz`` (``median`` + ``sigma``, each (C, F, 1)) + ``<hash>.json``
(the session key). Production robust-z is exactly ``z = (mag - median) / sigma``,
so the tail / survival / knee — and the µV reading of a high-|z| cell — come out of
the cache in seconds (pure numpy on a ~70 MB array per session).

For each band (band_lfs, band_hga) it reports, per session and pooled over the
cohort (via a fixed |z| histogram so we never hold all cells at once):
  * |z| tail percentiles P99 .. P99.9999 and max
  * log-survival P(|z|>t) on a grid -> the KNEE = the principled W
  * the CLEAN-SIGNAL Rayleigh baseline (stationary-Gaussian |STFT|) for reference
  * sigma/median per band so a |z|=W cap reads as a magnitude fold over typical

    .venv/bin/python scripts/neuroprobe/measure_band_tail_from_cache.py \
        --cache-dir /work/ht203/cache_neuroai/v14_2band_v2_spec_pretrain
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# Rayleigh (stationary-Gaussian |STFT|) robust-z baseline, derived analytically:
#   median = 1.1774 sigma ; 1.4826*MAD = 0.665 sigma ; R_p/sigma = sqrt(-2 ln(1-p))
_RAY_MED, _RAY_SCALE = 1.1774, 0.665


def _rayleigh_z(p: float) -> float:
    return (float(np.sqrt(-2.0 * np.log1p(-p))) - _RAY_MED) / _RAY_SCALE


# |z| histogram edges: fine below 20 (knee lives here), coarse out to 500.
_EDGES = np.concatenate([
    np.arange(0.0, 20.0, 0.05),
    np.arange(20.0, 60.0, 0.5),
    np.arange(60.0, 500.0, 5.0),
    [np.inf],
])
_CENT = _EDGES[:-1]
_PCTS = (99, 99.9, 99.99, 99.999, 99.9999)
_SURV_GRID = [3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100, 200]


def _pcts_from_hist(counts: np.ndarray) -> dict:
    total = counts.sum()
    cdf = np.cumsum(counts) / total
    out = {}
    for p in _PCTS:
        i = int(np.searchsorted(cdf, p / 100.0))
        out[p] = float(_CENT[min(i, _CENT.size - 1)])
    return out


def _survival_from_hist(counts: np.ndarray) -> dict:
    total = counts.sum()
    return {t: float(counts[_CENT > t].sum()) / total for t in _SURV_GRID}


def _session_label(meta_key: str) -> str:
    try:
        k = json.loads(meta_key)["key"] if meta_key.startswith("{") else meta_key
        inner = json.loads(k.split("}_")[0] + "}") if "}_" in k else json.loads(k)
        tl = inner["timeline"]
        return f"btbank{tl['subject_id']}_t{tl['trial_id']}"
    except Exception:
        return meta_key[:40]


def measure_band(cache_dir: str, band: str) -> dict:
    pattern = os.path.join(cache_dir, f"band_{band}",
                           "speech_decoding.extractors.view.MultiStftView._get_data,1",
                           "*", "*.npy")
    npys = sorted(glob.glob(pattern))
    if not npys:
        return {"band": band, "sessions": [], "note": "no entries"}

    pooled = np.zeros(_CENT.size, dtype=np.float64)
    sessions = []
    sig_over_med = []
    for npy in npys:
        h = npy[:-4]
        meta = json.load(open(h + ".json"))
        label = _session_label(meta.get("key", h))
        st = np.load(h + ".stats.npz")
        median, sigma = st["median"], st["sigma"]  # (C, F, 1)
        mag = np.load(npy, mmap_mode="r")           # (C, F, T)
        z = np.abs((np.asarray(mag) - median) / sigma)
        counts, _ = np.histogram(z, bins=_EDGES)
        pooled += counts
        # typical-band readout: sigma/median per (elec,bin), cohort-pooled median
        smr = float(np.median((sigma / np.maximum(median, 1e-6)).ravel()))
        sig_over_med.append(smr)
        sessions.append({
            "session": label, "shape": list(z.shape),
            "pct": _pcts_from_hist(counts), "max": float(z.max()),
            "sigma_over_median": smr,
        })
        del z

    return {
        "band": band,
        "n_sessions": len(sessions),
        "pooled_pct": _pcts_from_hist(pooled),
        "pooled_survival": _survival_from_hist(pooled),
        "sigma_over_median_cohort": float(np.median(sig_over_med)),
        "sessions": sessions,
    }


def _print(band_res: dict) -> None:
    b = band_res["band"]
    print(f"\n===== band {b.upper()}  ({band_res.get('n_sessions', 0)} sessions) =====")
    if not band_res.get("sessions"):
        print("  (no cache entries)")
        return
    pp = band_res["pooled_pct"]
    print(f"  POOLED |z| tail: P99={pp[99]:.2f}  P99.9={pp[99.9]:.2f}  "
          f"P99.99={pp[99.99]:.2f}  P99.999={pp[99.999]:.2f}  P99.9999={pp[99.9999]:.2f}")
    print(f"  Rayleigh clean baseline: P99.9={_rayleigh_z(0.999):.2f}  "
          f"P99.99={_rayleigh_z(0.9999):.2f}  P99.999={_rayleigh_z(0.99999):.2f}")
    smr = band_res["sigma_over_median_cohort"]
    print(f"  sigma/median (cohort) = {smr:.3f}  -> a |z|=W cell is "
          f"mag = (1 + W*{smr:.3f})x the per-bin median")
    print("  POOLED log-survival P(|z|>t):")
    s = band_res["pooled_survival"]
    print("    " + "  ".join(f"{t}:{s[t]:.1e}" for t in _SURV_GRID))
    print("  per-session P99.99 / max:")
    for se in band_res["sessions"]:
        print(f"    {se['session']:<14} P99.99={se['pct'][99.99]:6.2f}  max={se['max']:8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True,
                    help="2-band spec cache root (contains band_lfs/ and band_hga/).")
    ap.add_argument("--out", default=None, help="optional JSON dump")
    args = ap.parse_args()
    results = {b: measure_band(args.cache_dir, b) for b in ("lfs", "hga")}
    for b in ("lfs", "hga"):
        _print(results[b])
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
