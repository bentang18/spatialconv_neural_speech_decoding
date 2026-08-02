#!/usr/bin/env python
"""Does the new --band-rates code path still produce the PUBLISHED enc0?

The 64 Hz HGA arm is only interpretable if the same script, run with uniform rates on the same
caches, reproduces the enc0 the board numbers were computed from. enc0 touches no weights, so
this should be exact up to cross-architecture float reassociation (the published records were
encoded on DeltaAI aarch64; this runs on Delta CPU x86).

Compares the taps AND the metadata a consumer slices them with -- a matching tap under a
different band layout, row count or label vector would still be a different experiment.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

LITE_SESSIONS = ((1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                 (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))
TAPS = ("enc0", "enc0_elec")


def _cmp(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, bool]:
    """max|Δ|, max|Δ| relative to the reference's scale, and bit-equality."""
    exact = a.shape == b.shape and torch.equal(a, b)
    x, y = a.to(torch.float64), b.to(torch.float64)
    d = float((x - y).abs().max())
    return d, d / (float(x.abs().max()) + 1e-12), exact


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--published-dir", required=True)
    p.add_argument("--published-tag", required=True)
    p.add_argument("--new-dir", required=True)
    p.add_argument("--new-tag", required=True)
    p.add_argument("--rtol", type=float, default=1e-3,
                   help="max tolerated relative deviation; fp16 taps carry ~1e-3 of resolution "
                        "at full scale, so anything above this is a real path change, not "
                        "float reassociation")
    a = p.parse_args()

    worst, n_seen, n_exact, failures = 0.0, 0, 0, []
    for s, t in LITE_SESSIONS:
        pub = os.path.join(a.published_dir, f"enc_s{s}_t{t}_{a.published_tag}.pt")
        new = os.path.join(a.new_dir, f"enc_s{s}_t{t}_{a.new_tag}.pt")
        if not (os.path.exists(pub) and os.path.exists(new)):
            print(f"[skip] s{s}_t{t}: pub={os.path.exists(pub)} new={os.path.exists(new)}")
            continue
        P = torch.load(pub, map_location="cpu", weights_only=False)
        N = torch.load(new, map_location="cpu", weights_only=False)
        n_seen += 1

        # Metadata first: a matching tap under a different layout is a different experiment.
        for k in ("band_lengths", "band_fdims", "n_windows"):
            pv, nv = tuple(np.ravel(P[k]).tolist()), tuple(np.ravel(N[k]).tolist())
            if pv != nv:
                failures.append(f"s{s}_t{t} {k}: published {pv} != new {nv}")
        pl, nl = np.asarray(P["present_parcels"]), np.asarray(N["present_parcels"])
        if not np.array_equal(pl, nl):
            failures.append(f"s{s}_t{t} present_parcels differ ({len(pl)} vs {len(nl)})")
        if not np.allclose(np.asarray(P["clip_starts"]), np.asarray(N["clip_starts"])):
            failures.append(f"s{s}_t{t} clip_starts differ — different windows entirely")

        for tap in TAPS:
            if tap not in P["feats"] or tap not in N["feats"]:
                failures.append(f"s{s}_t{t} {tap}: missing (pub={tap in P['feats']} "
                                f"new={tap in N['feats']})")
                continue
            x, y = P["feats"][tap]["raw"], N["feats"][tap]["raw"]
            if x.shape != y.shape:
                failures.append(f"s{s}_t{t} {tap}: shape {tuple(x.shape)} != {tuple(y.shape)}")
                continue
            d, rel, exact = _cmp(x, y)
            n_exact += int(exact)
            worst = max(worst, rel)
            flag = "BIT-EXACT" if exact else ("ok" if rel <= a.rtol else "DIVERGENT")
            print(f"[parity] s{s}_t{t} {tap:10s} shape={tuple(x.shape)} max|d|={d:.4g} "
                  f"rel={rel:.3g} -> {flag}", flush=True)
            if rel > a.rtol:
                failures.append(f"s{s}_t{t} {tap}: rel {rel:.3g} > rtol {a.rtol}")
        del P, N

    print(f"\n[summary] sessions={n_seen}/{len(LITE_SESSIONS)} taps_bit_exact={n_exact}/"
          f"{n_seen * len(TAPS)} worst_rel={worst:.3g} rtol={a.rtol}")
    if n_seen != len(LITE_SESSIONS):
        failures.append(f"only {n_seen}/{len(LITE_SESSIONS)} sessions compared")
    if failures:
        print("\n[FAIL] the new code path does NOT reproduce the published enc0:")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("[PASS] new code path reproduces the published enc0 on every session and tap.")


if __name__ == "__main__":
    main()
