"""MEAN-FLOOR-VS-COUNT — measure r_μ(n), the count-dependent reliability of the cm-removed
parcel MEAN at 4 Hz, so the r4 secondary head's 3 MEAN dims get a count-dependent floor the
way the 3 STD dims already do (secondary_head.count_dependent_noise_var). As a byproduct it
measures the STD the same way, testing the shipped 1/(n−1) law in the small-n regime it is
currently EXTRAPOLATED into (n=2..5; #28 only scored n>=6, so that regime was never measured).

WHY THIS, AND WHAT IT CAN / CANNOT SEE.
  A size-n parcel mean is signal + sampling noise. Split the noise into (i) INDEPENDENT
  per-electrode noise, which a size-n mean averages down as 1/n, and (ii) a WITHIN-PARCEL
  SHARED component (biological or artifact), common to all electrodes, which does NOT average
  out. Any internal-consistency estimator (split-half / disjoint subsets) counts everything
  shared across the two halves — including (ii) — as TRUE SIGNAL. So this probe measures the
  count-dependent part (i) and CANNOT see (ii): a shared-noise "plateau below 1" is a
  CROSS-SUBJECT phenomenon (does a parcel mean transfer to the SAME parcel in another subject),
  which is r4's held-out eval, not measurable here. What we need for the floor IS (i): how much
  independent noise a finite montage fails to average. That is the exact SEM analog of the std
  floor's 1/(n−1) sampling law.

ESTIMATOR (model-free, direct — no Spearman-Brown extrapolation).
  For each parcel and each subset size n (with the parcel holding >= 2n contacts), draw two
  DISJOINT size-n electrode subsets, mean each band envelope over them at 4 Hz, project out the
  leak-free cross-parcel common mode (the head's target is cm-removed), and correlate the two
  residuals. That correlation IS the reliability of a size-n mean at that n (both sides carry the
  same size-n noise level), measured where the parcel actually lives — no extrapolation.

INVARIANT, BUILT IN AND PRINTED (feedback-build-the-invariant-into-the-probe).
  The noise fraction phi(n) = (1 − r(n)) / r(n) = k/n — LINEAR IN 1/n THROUGH THE ORIGIN. The
  intercept phi0 is ZERO BY CONSTRUCTION for any within-session internal-consistency estimator:
  a within-parcel SHARED component sits in BOTH disjoint subsets, so it is counted in the
  correlation's numerator (signal), lowering the slope k but NEVER creating an intercept. (This
  is the proof that the plateau is invisible here: it would need noise shared between the halves
  yet NOT counted as signal, which an internal split cannot produce — the plateau is a
  CROSS-SUBJECT quantity, r4's eval.) So we regress phi on 1/n and print (a) the R² of the line
  and (b) phi0. phi0 ≈ 0 confirms the pure sampling law the floor uses; a MATERIALLY nonzero phi0
  flags an ESTIMATOR problem (cm-removal leaking signal, non-stationarity), not a plateau. The
  load-bearing output is the per-band slope k, which sets the floor r_mu(n) = 1/(1 + k/n). The
  self-test plants a KNOWN shared component and checks phi0 STAYS ~0 while k drops as predicted.

Anchoring: reliabilities at N_REF (mean electrode count) reconcile with #28's per-dim mean/std
reliabilities; the fit's implied r at N_REF is printed next to #28's frozen values.

Model-FREE (no checkpoint). CPU. DeltaAI/Delta LOGIN node:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_mean_floor_vs_count \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/mean_floor_vs_count.json

  .venv/bin/python -m scripts.neuroprobe.probe_v3_mean_floor_vs_count --self-test   # run first
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

BANDS = ("slow", "mid", "hga")
IS_STD = (False, True)               # measure the mean (SEM) and the std (sample-var) laws
KIND = ("mu", "sd")
SLOT_FRAMES = 8                      # 32 Hz env -> 4 Hz parcel-state grid (#28 convention)
N_GRID = (1, 2, 3, 4, 5, 6, 8, 10)   # subset sizes; a parcel contributes to n iff it has >= 2n
N_DRAWS = 40                         # disjoint-pair draws per (parcel, n)
MIN_OUT = 4                          # need >= 4 out-of-parcel contacts for a leak-free cm

# #28 frozen per-dim reliabilities at N_REF, for the reconciliation print (order slow/mid/hga).
REF28_MEAN_REL = (0.881, 0.796, 0.828)
REF28_STD_REL = (0.652, 0.496, 0.546)
N_REF = 13.35


# ------------------------------------------------------------------- pure estimators (self-tested)
def _project_out(y: np.ndarray, cm: np.ndarray) -> np.ndarray:
    A = np.column_stack([cm, np.ones(len(cm))])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def _slots(env: np.ndarray, slot_frames: int) -> np.ndarray:
    """(n_clips, N, T) 32 Hz -> (n_clips, N, T//slot_frames) 4 Hz."""
    n, N, T = env.shape
    k = T // slot_frames
    return env[:, :, : k * slot_frames].reshape(n, N, k, slot_frames).mean(-1)


def _summary(e_sub: np.ndarray, is_std: bool) -> np.ndarray:
    """e_sub (n_clips, n, T_slots) -> flat (n_clips*T_slots,) parcel mean or within-parcel std."""
    s = e_sub.std(1) if is_std else e_sub.mean(1)
    return s.reshape(-1)


def _pair_reliability(
    e_band: np.ndarray, idx: np.ndarray, cm: np.ndarray, n: int, is_std: bool, rng
) -> float | None:
    """Correlation of two DISJOINT size-n subset summaries (cm-removed) — reliability at n.
    e_band (n_clips, N, T_slots); idx the parcel's electrode columns; cm the leak-free
    common-mode regressor (n_clips*T_slots,). Needs len(idx) >= 2n and, for std, n >= 2."""
    if len(idx) < 2 * n or (is_std and n < 2):
        return None
    rs = []
    for _ in range(N_DRAWS):
        perm = rng.permutation(idx)
        a, b = perm[:n], perm[n : 2 * n]
        c1 = _project_out(_summary(e_band[:, a], is_std), cm)
        c2 = _project_out(_summary(e_band[:, b], is_std), cm)
        if c1.std() < 1e-8 or c2.std() < 1e-8:
            continue
        rs.append(float(np.corrcoef(c1, c2)[0, 1]))
    return float(np.mean(rs)) if rs else None


def fit_noise_law(n_vals: np.ndarray, r_vals: np.ndarray) -> dict:
    """Regress the noise fraction phi(n) = (1−r)/r on 1/n. Under pure 1/n averaging phi = k/n
    (intercept 0). Returns slope k, intercept phi0, R², implied reliability at N_REF, and the
    plateau reliability 1/(1+phi0). Invariant: R² high AND phi0 ≈ 0 ⇒ the shipped 1/n-style law
    holds; phi0 > 0 ⇒ a within-session-visible shared component (plateau below 1)."""
    r = np.clip(r_vals, 1e-3, 0.999)
    phi = (1.0 - r) / r
    x = 1.0 / n_vals
    A = np.column_stack([x, np.ones_like(x)])
    (k, phi0), *_ = np.linalg.lstsq(A, phi, rcond=None)
    pred = A @ np.array([k, phi0])
    ss_res = float(((phi - pred) ** 2).sum())
    ss_tot = float(((phi - phi.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    phi_ref = k / N_REF + phi0
    return {
        "slope_k": round(float(k), 4),
        "intercept_phi0": round(float(phi0), 4),
        "r2_linear": round(float(r2), 4),
        "reliability_at_n_ref": round(float(1.0 / (1.0 + phi_ref)), 4),
        "plateau_reliability": round(float(1.0 / (1.0 + max(phi0, 0.0))), 4),
    }


# ------------------------------------------------------------------------------------- self-test
def _self_test() -> None:
    print("SELF-TEST — estimator + law fit on synthetic parcels with KNOWN variance components\n")
    rng = np.random.default_rng(0)
    n_clips, T = 200, 12
    N_elec = 24
    S = 1.0  # signal variance

    def make_parcel(v: float, rho: float, amp_signal: float = 0.0):
        """Return (n_clips, N_elec, T): x_i = s + shared + a(t)·z_i, per-electrode noise var v,
        shared fraction rho. mean over n electrodes has noise var v*rho + v(1-rho)/n. When
        amp_signal>0 the independent-noise amplitude a(t) fluctuates (common across electrodes,
        E[a²] held = v(1−rho)) so the WITHIN-parcel std carries signal — needed to exercise the
        std path (a pure-white parcel has no std signal, only sampling noise)."""
        s = rng.standard_normal((n_clips, 1, T)) * math.sqrt(S)
        shared = rng.standard_normal((n_clips, 1, T)) * math.sqrt(v * rho)
        a2 = v * (1.0 - rho)
        if amp_signal > 0:
            g = np.exp(rng.standard_normal((n_clips, 1, T)) * amp_signal)
            a = math.sqrt(a2) * g / math.sqrt(float(np.mean(g**2)))  # keep E[a²]=a2
        else:
            a = np.full((n_clips, 1, T), math.sqrt(a2))
        indep = a * rng.standard_normal((n_clips, N_elec, T))
        return s + shared + indep

    # (1) pure independent noise (rho=0): intercept phi0 must be ~0, and the size-n reliability
    #     must match S/(S + v/n) at each n.
    v = 1.0
    e = make_parcel(v, rho=0.0)
    idx = np.arange(N_elec)
    cm = np.zeros(n_clips * T)  # no common mode in the synthetic; projecting out a constant is a no-op
    ns, rs = [], []
    for n in (1, 2, 3, 4, 6, 8, 10):
        r = _pair_reliability(e, idx, cm, n, is_std=False, rng=rng)
        assert r is not None
        expect = S / (S + v / n)
        ns.append(n); rs.append(r)
        ok = abs(r - expect) < 0.03
        print(f"[check] rho=0  n={n:2d}  r={r:.3f}  expect S/(S+v/n)={expect:.3f} : "
              f"{'OK' if ok else 'VIOLATED'}")
        assert ok
    fit0 = fit_noise_law(np.array(ns, float), np.array(rs, float))
    ok_int = abs(fit0["intercept_phi0"]) < 0.03 and fit0["r2_linear"] > 0.98
    k0 = fit0["slope_k"]  # pure indep noise: k = v(1-rho)/(S+v*rho) = v/S = 1.0
    print(f"[check] rho=0 fit: phi0={fit0['intercept_phi0']:.3f}≈0, R²={fit0['r2_linear']:.3f}>0.98,"
          f" k={k0:.3f}≈1.0 : {'OK' if ok_int and abs(k0 - 1.0) < 0.05 else 'VIOLATED'}")
    assert ok_int and abs(k0 - 1.0) < 0.05

    # (2) shared component present (rho=0.3): phi0 must STAY ~0 (an internal split cannot see the
    #     shared component as noise — it is counted as signal), while the slope k DROPS to the
    #     predicted v(1−rho)/(S+v*rho). This is the proof the plateau is invisible within-session.
    rho = 0.3
    e = make_parcel(v, rho=rho)
    ns, rs = [], []
    for n in (1, 2, 3, 4, 6, 8, 10):
        r = _pair_reliability(e, idx, cm, n, is_std=False, rng=rng)
        assert r is not None
        rs.append(r); ns.append(n)
    fit1 = fit_noise_law(np.array(ns, float), np.array(rs, float))
    k_expect = v * (1.0 - rho) / (S + v * rho)  # = 0.7/1.3 = 0.538
    ok_shared = abs(fit1["intercept_phi0"]) < 0.05 and abs(fit1["slope_k"] - k_expect) < 0.06
    print(f"\n[check] rho=0.3 fit: phi0={fit1['intercept_phi0']:.3f}≈0 (plateau invisible), "
          f"k={fit1['slope_k']:.3f}≈{k_expect:.3f} (dropped from {k0:.2f}) : "
          f"{'OK' if ok_shared else 'VIOLATED'}")
    assert ok_shared

    # (3) std law: with a real within-parcel-std signal (amp_signal>0), the sample-std sampling
    #     noise ∝ 1/(n−1), so std reliability RISES with n. Assert the noise fraction decreases
    #     monotonically (allowing a small tolerance for the coarse large-n grid).
    e = make_parcel(v, rho=0.0, amp_signal=0.6)
    prev, mono = None, True
    for n in (2, 3, 4, 6, 8, 10):
        r = _pair_reliability(e, idx, cm, n, is_std=True, rng=rng)
        assert r is not None
        phi = (1 - r) / r
        if prev is not None and phi > prev + 0.02:
            mono = False
        prev = phi
    print(f"[check] std noise fraction decreases with n (real std signal) : "
          f"{'OK' if mono else 'VIOLATED'}")
    assert mono

    print("\nSELF-TEST PASSED — estimator sound; phi0 structurally ~0 (plateau is cross-subject).")


# -------------------------------------------------------------------------------- data-path driver
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--band-root")
    p.add_argument("--span-dir")
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--slot-frames", type=int, default=SLOT_FRAMES)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    a = p.parse_args()

    if a.self_test:
        _self_test()
        return
    if not (a.band_root and a.span_dir):
        p.error("--band-root and --span-dir required unless --self-test")

    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
    from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, V3_SESSIONS, WINSOR
    from scripts.neuroprobe.probe_v3_global_structure import _read_clips

    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(a.band_root, b) for b in BAND_DIRS],
        span_dir=a.span_dir,
        parcel_fn=make_bt_parcel_fn(a.bt_root),
        lof_report_path=None,
        winsor=WINSOR,
    )
    rate_hz = round(32.0 / a.slot_frames, 3)
    print(f"MEAN-FLOOR-VS-COUNT | {len(specs)} sessions | {rate_hz} Hz (slot={a.slot_frames}) | "
          f"disjoint size-n subsets, {N_DRAWS} draws | n-grid {N_GRID}\n", flush=True)

    # accumulators: per (kind, band, n) -> list of per-parcel reliabilities
    acc = {(k, b, n): [] for k in KIND for b in BANDS for n in N_GRID}
    n_parcels_at = {(b, n): 0 for b in BANDS for n in N_GRID}

    for spec in specs:
        sid, tid = spec.session_key
        parcel_id = spec.setup.parcel_id.cpu().numpy()
        parcels = [q for q in np.unique(parcel_id) if int((parcel_id == q).sum()) >= 2]
        if not parcels:
            continue
        bands, _ = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        env3 = [_slots(b.mean(2), a.slot_frames) for b in bands]  # 3 x (n_clips, N, T_slots)
        rng = np.random.default_rng(a.seed + sid * 100 + tid)

        for pc in parcels:
            idx = np.where(parcel_id == pc)[0]
            out_idx = np.where(parcel_id != pc)[0]
            if len(out_idx) < MIN_OUT:
                continue
            for bi, band in enumerate(BANDS):
                e = env3[bi]
                cm = e[:, out_idx].mean(1).reshape(-1)  # leak-free additive common mode
                for n in N_GRID:
                    if len(idx) < 2 * n:
                        continue
                    for ki, is_std in enumerate(IS_STD):
                        r = _pair_reliability(e, idx, cm, n, is_std, rng)
                        if r is not None:
                            acc[(KIND[ki], band, n)].append(r)
                    n_parcels_at[(band, n)] += 1
        print(f"[s{sid}t{tid}] parcels {len(parcels)}", flush=True)

    # -------------------------------------------------------------------------- report + fit
    def _m(v):
        return round(float(np.mean(v)), 4) if v else None

    out: dict = {"slot_frames": a.slot_frames, "rate_hz": rate_hz, "n_grid": list(N_GRID),
                 "n_draws": N_DRAWS, "curves": {}, "fits": {}}
    for kind, ref, floor_name in (("mu", REF28_MEAN_REL, "mean"), ("sd", REF28_STD_REL, "std")):
        print("\n" + "=" * 92)
        print(f"{floor_name.upper()} floor — r(n) per band, and the phi=k/n fit "
              f"(intercept phi0 = within-session shared component).")
        print("=" * 92)
        for bi, band in enumerate(BANDS):
            ns, rs = [], []
            for n in N_GRID:
                r = _m(acc[(kind, band, n)])
                if r is not None:
                    ns.append(n); rs.append(r)
            out["curves"][f"{kind}_{band}"] = {
                "n": ns, "r": rs, "n_parcels": [n_parcels_at[(band, n)] for n in ns]}
            if len(ns) < 3:
                print(f"  [{band}] too few n points ({len(ns)}) to fit")
                continue
            fit = fit_noise_law(np.array(ns, float), np.array(rs, float))
            out["fits"][f"{kind}_{band}"] = fit
            curve = "  ".join(f"n{n}:{r:.3f}" for n, r in zip(ns, rs))
            print(f"  [{band}] {curve}")
            print(f"        fit k={fit['slope_k']}  phi0={fit['intercept_phi0']}  "
                  f"R²(lin)={fit['r2_linear']}  r@N_REF={fit['reliability_at_n_ref']} "
                  f"(#28 {ref[bi]})  plateau_r={fit['plateau_reliability']}")
            intercept_ok = abs(fit["intercept_phi0"]) < 0.05
            print(f"        [check] phi0≈0 (pure sampling law, no within-session plateau): "
                  f"{'OK' if intercept_ok else 'NONZERO — shared component present'}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
