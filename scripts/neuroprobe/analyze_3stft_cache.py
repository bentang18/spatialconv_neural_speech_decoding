#!/usr/bin/env python
"""Authenticity + completeness check for a built 3STFT per-band spec cache (DCC).

Pairs with submit_build_bt_3stft_cache.py. For each band (slow | beta | hg) it
re-constructs the band MultiStftView EXACTLY as dispatch_v14 does (shared
``STFT_3BAND_*`` + the same ``common_fe_kwargs``), derives the EXPECTED spec-cache
namespace, and asserts the built caches live there — this is the forward-HIT
guarantee: a future ``--frontend 3stft`` training run constructing the same view
will read these caches back (matching namespace), not silently rebuild.

Per (band, session) it then checks the on-disk triple (``.npy`` payload + ``.json``
geometry sidecar + ``.stats.npz``):
  - F bins: slow 12 (= 2x6 Re/Im Cartesian), beta 6, HG 9.
  - payload shape (C, F, T), finite; C = montage electrode count (STATIC-dropped).
  - stats (C, F, 1); slow: median == 0 and sigma[:F/2] == sigma[F/2:] (shared Re/Im
    |STFT| scale, scale-only); mag bands: standard robust-z (median may be != 0).

Usage (on DCC):
    .venv/bin/python scripts/neuroprobe/analyze_3stft_cache.py \\
        --spec-cache-dir /work/ht203/cache_neuroai/v14_3stft_spec_cache \\
        --corpus both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from speech_decoding.extractors.view import (
    STFT_3BAND_BETA,
    STFT_3BAND_HG,
    STFT_3BAND_SLOW,
    MultiStftView,
)

# Mirrors build_v14_experiment's common_fe_kwargs (the band-view preprocessing).
# c_max / notch are not part of the band STFT geometry that keys the namespace, but
# we build the view the same way for an exact namespace match. notch is the BT
# default (60 Hz + harmonics is applied inside the extractor); the namespace folds
# only front_end + band geometry + channelization, so these preproc values do not
# perturb it — kept faithful for clarity.
_BANDS = {
    "slow": (STFT_3BAND_SLOW, 12),
    "beta": (STFT_3BAND_BETA, 6),
    "hg": (STFT_3BAND_HG, 9),
}
_CORPUS_N = {"pretrain": 13, "eval": 12}


def _band_view(name: str, spec_cache_dir: Path) -> MultiStftView:
    const, _ = _BANDS[name]
    return MultiStftView(
        event_types="Ieeg", car="shaft", filter=(0.5, None), scaler=None,
        apply_log=False, channel_order="original", session_robust_z=True,
        front_end="band", **const, hop_length=int(const["band_hop"]),
        spec_cache_dir=str(spec_cache_dir / f"band_{name}"),
    )


def _check_session(npy: Path, exp_f: int, is_slow: bool, const: dict) -> dict:
    stem = npy.with_suffix("")
    meta = json.loads(Path(f"{stem}.json").read_text())
    z = np.load(f"{stem}.stats.npz")
    median, sigma = z["median"], z["sigma"]
    arr = np.load(npy, mmap_mode="r")
    C, F, T = arr.shape
    issues = []
    if F != exp_f:
        issues.append(f"F={F}!={exp_f}")
    # .json geometry sidecar must record the band STFT this cache was built for
    # (LG4 default-edit guard). Cross-check N/hop against the band constant.
    geom = meta.get("geometry") or meta
    for k in ("band_nperseg", "band_hop"):
        if k in geom and int(geom[k]) != int(const[k]):
            issues.append(f"{k}={geom[k]}!={const[k]}")
    if sigma.shape != (C, F, 1) or median.shape != (C, F, 1):
        issues.append(f"stats shape {median.shape}/{sigma.shape} != ({C},{F},1)")
    if is_slow:
        if not np.allclose(median, 0):
            issues.append(f"slow median!=0 (max|m|={np.abs(median).max():.3g})")
        h = F // 2
        if not np.allclose(sigma[:, :h], sigma[:, h:], atol=1e-5):
            issues.append("slow sigma halves differ (Re/Im scale not shared)")
    # finiteness on a bounded sample (full-finiteness is the LG14c read guard's job)
    if not np.isfinite(np.asarray(arr[:, :, : min(T, 256)])).all():
        issues.append("non-finite payload sample")
    return {"C": C, "F": F, "T": T, "issues": issues}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec-cache-dir", type=Path, required=True)
    ap.add_argument("--corpus", choices=("pretrain", "eval", "both"), default="both")
    ap.add_argument("--bands", default="slow,beta,hg")
    args = ap.parse_args()

    corpora = ["pretrain", "eval"] if args.corpus == "both" else [args.corpus]
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    total_expected = sum(_CORPUS_N[c] for c in corpora)

    all_ok = True
    for name in bands:
        const, exp_f = _BANDS[name]
        view = _band_view(name, args.spec_cache_dir)
        ns = view._spec_cache_namespace()
        cache_root = args.spec_cache_dir / f"band_{name}" / ns
        print(f"\n=== band {name} (exp F={exp_f}) ===")
        print(f"  expected namespace: band_{name}/{ns}")
        if not cache_root.is_dir():
            print(f"  MISSING namespace dir → {cache_root}")
            all_ok = False
            continue
        npys = sorted(cache_root.glob("*.npy"))
        print(f"  sessions cached: {len(npys)} (expect {total_expected} for "
              f"corpus={args.corpus})")
        c_counts, bad = [], 0
        for npy in npys:
            r = _check_session(npy, exp_f, is_slow=(name == "slow"), const=const)
            c_counts.append(r["C"])
            if r["issues"]:
                bad += 1
                print(f"    {npy.name[:12]}… C={r['C']} F={r['F']} T={r['T']} "
                      f"ISSUES: {r['issues']}")
        if c_counts:
            print(f"  electrode counts (C, STATIC-dropped): "
                  f"min={min(c_counts)} max={max(c_counts)}")
        ok = bad == 0 and len(npys) == total_expected
        print(f"  → {'OK' if ok else 'CHECK'} ({bad} sessions with issues, "
              f"{len(npys)}/{total_expected} present)")
        all_ok &= ok
    print(f"\n{'ALL BANDS OK' if all_ok else 'SOME CHECKS FAILED'}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
