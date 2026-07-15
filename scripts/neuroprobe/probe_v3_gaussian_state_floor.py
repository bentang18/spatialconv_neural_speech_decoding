"""#28 — the GAUSSIAN-NLL floor for the r4 secondary head, on the REAL joint 6-vector @ 4 Hz.

The head is now a single full-covariance Gaussian NLL over the 6-vector
`x = [slow_μ, mid_μ, hga_μ, slow_σ, mid_σ, hga_σ]` per (parcel, 4 Hz slot) — NOT the soft-bin
KL M12 floored. So the floor must be re-expressed in Gaussian-entropy nats, on the actual
target at the actual rate. Two things this resolves that M12/M17 did NOT:

  PART 1 — THE COMMON-MODE DECISION (the open r4 knob). The target is common-mode-removed on
  every dim (contract §7 step 3). For the MEANS that is validated (M10/M12). For the STDs it is
  OPEN: the within-parcel-std common mode is plausibly a real GLOBAL-AMPLITUDE state — exactly
  what a write-only global perceiver should carry — so removing it could throw away the signal
  the head exists to model. DECIDE IT WITH TWO NUMBERS per dim:
    (a) CM variance FRACTION  = R²(z_dim ~ leak-free cm_dim). How much of the dim IS the cm.
    (b) CM split-half RELIABILITY = corr of two cm estimates from DISJOINT out-of-parcel electrode
        halves (Spearman-Brown). Is the common mode itself REAL SIGNAL or noise?
  Rule: remove cm on a dim only if it is a large, and/or UN-reliable, fraction. A cm that is a
  large fraction AND highly reliable (a real global signal) should be KEPT (esp. on the σ dims).

  PART 2 — THE JOINT GAUSSIAN FLOOR (cm-removed residual, the head's actual target):
    CE_marginal (collapsed)   = H(Σ_resid)                — the joint marginal entropy (the
                                collapsed head that ignores its input).
    CE_factorized             = Σ_d H(var_d)              — independent per-dim marginals; the
                                gap CE_factorized − CE_marginal = TOTAL CORRELATION (the joint's
                                worth, M17's ~0.200 nats — collapse-to-marginals is visible here).
    CE_ceiling (perfect head) = H(N), N = diag(1 − reliability_d @ 4 Hz) — a perfect head knows
                                the signal and is defeated only by measurement noise.
    headroom for the latents  = CE_marginal − CE_ceiling  — the joint 6-vector version of M12's
                                "nats left for the latents" (cm already removed ⇒ all latent-earnable).

  PART 3 — T_state CONFIRM: per-dim split-half reliability at the 4 Hz window-averaged target
  (μ and σ). If HGA/MID hold (M17: means denoise upward, σ .601/.464/.478 > 0.3), T_state=16 @ 4 Hz
  stands; a large drop ⇒ the state is faster than 4 Hz ⇒ bump T_state.

Every estimator is a pure function with a named invariant asserted + printed (feedback-build-the-
invariant-into-the-probe). ``--self-test`` runs them on synthetic data with KNOWN Σ / reliability /
cm-fraction — RUN THAT LOCALLY before spending remote CPU. Model-FREE, CPU, dtai login node:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_gaussian_state_floor \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/gaussian_state_floor.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

BANDS = ("slow", "mid", "hga")
DIMS = ("slow_mu", "mid_mu", "hga_mu", "slow_sd", "mid_sd", "hga_sd")
STATE_DIM = 6
SLOT_FRAMES = 8      # 32 Hz env -> 4 Hz parcel-state grid (the head's slot rate)
N_SPLITS = 40        # split-half draws (matches M11/M17)
MIN_CONTACTS = 6     # need >=3 per half for a defined std; parcels below are skipped
LOG2PIE = math.log(2.0 * math.pi * math.e)


# ----------------------------------------------------------------- pure estimators (self-testable)
def _spearman_brown(r: float) -> float:
    return 2.0 * r / (1.0 + r) if r > -1.0 else 0.0


def _gauss_entropy(cov: np.ndarray) -> float:
    """Differential entropy of a d-dim Gaussian, ½(d·log(2πe) + logdet cov), in nats."""
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("cov not PD")
    return 0.5 * (d * LOG2PIE + logdet)


def _project_out(y: np.ndarray, cm: np.ndarray) -> np.ndarray:
    A = np.column_stack([cm, np.ones(len(cm))])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def cm_fraction(z: np.ndarray, cm: np.ndarray) -> float:
    """Fraction of z's variance explained by the leak-free common mode = R²(z ~ [cm, 1])."""
    resid = _project_out(z, cm)
    tot = float(z.var())
    return float(np.clip(1.0 - resid.var() / max(tot, 1e-12), 0.0, 1.0))


def gaussian_floors(x: np.ndarray, noise_var: np.ndarray) -> dict:
    """x: (n, 6) cm-removed residual (unit-ish variance per dim). noise_var: (6,) = 1 − reliability.
    Returns the joint Gaussian floors in nats (see PART 2)."""
    cov = np.cov(x, rowvar=False)
    var = np.diag(cov)
    ce_marginal = _gauss_entropy(cov)
    ce_factorized = float(0.5 * (np.log(2.0 * np.pi * np.e * np.maximum(var, 1e-12))).sum())
    tc = ce_factorized - ce_marginal                      # ≥ 0 by Hadamard
    ce_ceiling = float(0.5 * (np.log(2.0 * np.pi * np.e * np.maximum(noise_var, 1e-12))).sum())
    return {
        "CE_marginal": round(ce_marginal, 4),
        "CE_factorized": round(ce_factorized, 4),
        "TC": round(tc, 4),
        "CE_ceiling": round(ce_ceiling, 4),
        "headroom_latents": round(ce_marginal - ce_ceiling, 4),
        "headroom_factorized": round(ce_factorized - ce_ceiling, 4),
    }


# ------------------------------------------------------------------------------------- self-test
def _self_test() -> None:
    print("SELF-TEST — estimators on synthetic data with KNOWN Σ / reliability / cm-fraction\n")
    rng = np.random.default_rng(0)
    n = 60000

    # (1) cm_fraction: z = a*cm + noise, known R².
    cm = rng.standard_normal(n)
    a = 0.8
    z = a * cm + rng.standard_normal(n) * math.sqrt(1 - a**2)  # unit-var z, true R² = a²
    frac = cm_fraction(z, cm)
    ok1 = abs(frac - a**2) < 0.02
    print(f"[check] cm_fraction {frac:.3f} ≈ true R² {a**2:.3f} : {'OK' if ok1 else 'VIOLATED'}")
    assert ok1

    # (2) cm reliability: two disjoint noisy estimates of a shared cm signal.
    sig = rng.standard_normal(n)
    r_intended = 0.5  # each half = sqrt(r)·sig + sqrt(1-r)·indep ⇒ half-corr r ⇒ SB(2r/(1+r))
    h1 = math.sqrt(r_intended) * sig + math.sqrt(1 - r_intended) * rng.standard_normal(n)
    h2 = math.sqrt(r_intended) * sig + math.sqrt(1 - r_intended) * rng.standard_normal(n)
    rel = _spearman_brown(float(np.corrcoef(h1, h2)[0, 1]))
    expect = _spearman_brown(r_intended)
    ok2 = abs(rel - expect) < 0.02
    print(f"[check] cm reliability SB {rel:.3f} ≈ {expect:.3f} : {'OK' if ok2 else 'VIOLATED'}")
    assert ok2

    # (3) gaussian_floors: known correlated 6-vector; TC≥0, ceiling<marginal, TC matches -½logdet R.
    base = rng.standard_normal((STATE_DIM, STATE_DIM)) * 0.5
    cov_true = np.eye(STATE_DIM) + base @ base.T
    L = np.linalg.cholesky(cov_true)
    x = rng.standard_normal((n, STATE_DIM)) @ L.T
    noise_var = np.array([0.208, 0.361, 0.439, 0.399, 0.536, 0.522])
    f = gaussian_floors(x, noise_var)
    R = np.corrcoef(x, rowvar=False)
    tc_closed = -0.5 * np.linalg.slogdet(R)[1]
    ok3 = (f["TC"] >= 0 and f["CE_ceiling"] < f["CE_marginal"]
           and abs(f["TC"] - tc_closed) < 0.02)
    print(f"[check] floors: TC {f['TC']:.3f} ≈ −½logdet R {tc_closed:.3f}; "
          f"ceiling {f['CE_ceiling']:.3f} < marginal {f['CE_marginal']:.3f} : "
          f"{'OK' if ok3 else 'VIOLATED'}")
    assert ok3
    print("\nSELF-TEST PASSED — estimators sound.")


# -------------------------------------------------------------------------------- data-path pieces
def _slots(env: np.ndarray, slot_frames: int) -> np.ndarray:
    """(n_clips, N, T) 32 Hz -> (n_clips, N, T//slot_frames) 4 Hz parcel-state grid."""
    n, N, T = env.shape
    k = T // slot_frames
    return env[:, :, : k * slot_frames].reshape(n, N, k, slot_frames).mean(-1)


def _dim_vectors(env3, idx, out_idx):
    """Per band, the parcel MEAN and STD over its electrodes, flattened over (clip, slot).
    Returns (X6 (n_samp,6) z-scored, cms (6,) list of leak-free cm arrays, out-electrode arrays)."""
    z_cols, cm_cols = [], []
    for bi in range(3):
        e = env3[bi]                             # (n_clips, N, T_slots)
        mu = e[:, idx].mean(1).reshape(-1)        # parcel mean
        sd = e[:, idx].std(1).reshape(-1)         # within-parcel std
        cm_mu = e[:, out_idx].mean(1).reshape(-1)  # additive common mode (means)
        cm_sd = e[:, out_idx].std(1).reshape(-1)   # dispersion common mode (stds, M17)
        z_cols.append(mu); z_cols.append(sd)       # placeholder order fixed below
        cm_cols.append(cm_mu); cm_cols.append(cm_sd)
    # reorder to [slow_mu,mid_mu,hga_mu, slow_sd,mid_sd,hga_sd]
    order = [0, 2, 4, 1, 3, 5]
    raw = [z_cols[o] for o in order]
    cms = [cm_cols[o] for o in order]
    z = []
    for col in raw:
        z.append((col - col.mean()) / max(col.std(), 1e-8))
    return z, cms


def _cm_reliability(env3, idx, out_idx, rng) -> list[float | None]:
    """Split-half reliability of each dim's leak-free common mode (disjoint out-electrode halves)."""
    rels: list[float | None] = []
    band_of = [0, 1, 2, 0, 1, 2]
    is_std = [False, False, False, True, True, True]
    for dim in range(STATE_DIM):
        bi = band_of[dim]
        e = env3[bi]
        rs = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(out_idx)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
            if len(a) < 2 or len(b) < 2:
                continue
            if is_std[dim]:
                c1, c2 = e[:, a].std(1).reshape(-1), e[:, b].std(1).reshape(-1)
            else:
                c1, c2 = e[:, a].mean(1).reshape(-1), e[:, b].mean(1).reshape(-1)
            if c1.std() < 1e-8 or c2.std() < 1e-8:
                continue
            rs.append(float(np.corrcoef(c1, c2)[0, 1]))
        rels.append(_spearman_brown(float(np.mean(rs))) if rs else None)
    return rels


def _dim_reliability(env3, idx, rng) -> list[float | None]:
    """Split-half reliability of each dim's TARGET (parcel μ / σ) — PART 3, T_state confirm."""
    rels: list[float | None] = []
    band_of = [0, 1, 2, 0, 1, 2]
    is_std = [False, False, False, True, True, True]
    for dim in range(STATE_DIM):
        e = env3[band_of[dim]]
        rs = []
        for _ in range(N_SPLITS):
            perm = rng.permutation(idx)
            a, b = perm[: len(perm) // 2], perm[len(perm) // 2 :]
            if len(a) < 2 or len(b) < 2:
                continue
            if is_std[dim]:
                c1, c2 = e[:, a].std(1).reshape(-1), e[:, b].std(1).reshape(-1)
            else:
                c1, c2 = e[:, a].mean(1).reshape(-1), e[:, b].mean(1).reshape(-1)
            if c1.std() < 1e-8 or c2.std() < 1e-8:
                continue
            rs.append(float(np.corrcoef(c1, c2)[0, 1]))
        rels.append(_spearman_brown(float(np.mean(rs))) if rs else None)
    return rels


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--band-root")
    p.add_argument("--span-dir")
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--slot-frames", type=int, default=SLOT_FRAMES,
                   help="32 Hz frames per state slot; 8 -> 4 Hz (default), 4 -> 8 Hz")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    a = p.parse_args()

    if a.self_test:
        _self_test()
        return
    if not (a.band_root and a.span_dir):
        p.error("--band-root and --span-dir required unless --self-test")

    # heavy imports deferred so --self-test runs anywhere (no data/deps).
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
    print(f"#28 GAUSSIAN state floor | {len(specs)} sessions | {rate_hz} Hz "
          f"(slot={a.slot_frames}) | {N_SPLITS} split-half draws\n", flush=True)

    per_dim_frac = {d: [] for d in DIMS}
    per_dim_cmrel = {d: [] for d in DIMS}
    per_dim_rel = {d: [] for d in DIMS}
    n_elec_scored: list[int] = []  # effective N_REF: mean electrode count over scored parcels
    floor_keys = ("CE_marginal", "CE_factorized", "TC", "CE_ceiling",
                  "headroom_latents", "headroom_factorized")
    floor_acc = {k: [] for k in floor_keys}
    rows = []

    for spec in specs:
        sid, tid = spec.session_key
        parcel_id = spec.setup.parcel_id.cpu().numpy()
        parcels = [q for q in np.unique(parcel_id)
                   if int((parcel_id == q).sum()) >= MIN_CONTACTS]
        if len(parcels) < 3:
            continue
        bands, _ = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        env3 = [_slots(b.mean(2), a.slot_frames) for b in bands]  # 3 x (n_clips, N, T_slots)
        rng = np.random.default_rng(a.seed + sid * 100 + tid)

        for pc in parcels:
            idx = np.where(parcel_id == pc)[0]
            out_idx = np.where(parcel_id != pc)[0]
            if len(idx) < MIN_CONTACTS or len(out_idx) < 4:
                continue
            n_elec_scored.append(int(len(idx)))
            z, cms = _dim_vectors(env3, idx, out_idx)
            noise = np.empty(STATE_DIM)
            dim_rel = _dim_reliability(env3, idx, rng)
            cm_rel = _cm_reliability(env3, idx, out_idx, rng)
            resid = []
            for di, d in enumerate(DIMS):
                frac = cm_fraction(z[di], cms[di])
                per_dim_frac[d].append(frac)
                if cm_rel[di] is not None:
                    per_dim_cmrel[d].append(cm_rel[di])
                if dim_rel[di] is not None:
                    per_dim_rel[d].append(dim_rel[di])
                r = float(np.clip(dim_rel[di] if dim_rel[di] is not None else 0.0, 1e-3, 0.999))
                noise[di] = 1.0 - r
                rr = _project_out(z[di], cms[di])
                resid.append((rr - rr.mean()) / max(rr.std(), 1e-8))
            X = np.column_stack(resid)
            f = gaussian_floors(X, noise)
            for k in floor_keys:
                floor_acc[k].append(f[k])

        rows.append({"subject_id": sid, "trial_id": tid, "n_parcels": len(parcels)})
        print(f"[s{sid}t{tid}] parcels {len(parcels)}", flush=True)

    def _m(v):
        return round(float(np.mean(v)), 4) if v else None

    print("\n" + "=" * 90)
    print("PART 1 — COMMON-MODE audit per dim (the cm-removal DECISION).")
    print("  FRACTION = share of the dim that IS the cm.  CM-REL = is that cm real signal?")
    print("  Remove cm iff it is a large fraction AND/OR un-reliable; KEEP a large+reliable cm.")
    print("=" * 90)
    pooled_p1 = {}
    for d in DIMS:
        frac, cmrel, rel = _m(per_dim_frac[d]), _m(per_dim_cmrel[d]), _m(per_dim_rel[d])
        pooled_p1[d] = {"cm_fraction": frac, "cm_reliability": cmrel, "target_reliability": rel}
        keep = (cmrel is not None and cmrel > 0.5 and frac is not None and frac > 0.15)
        verdict = "KEEP cm (large + reliable global signal)" if keep else "remove cm (small or noisy)"
        print(f"  {d:<8} cm_frac {frac}   cm_reliability {cmrel}   target_rel@4Hz {rel}   -> {verdict}")
    ok_frac = all(pooled_p1[d]["cm_fraction"] is None
                  or 0.0 <= pooled_p1[d]["cm_fraction"] <= 1.0 for d in DIMS)
    print(f"  [check] all cm_fraction in [0,1] : {'OK' if ok_frac else 'VIOLATED'}")

    print("\n" + "=" * 90)
    print(f"PART 2 — JOINT GAUSSIAN floor (cm-removed 6-vector residual @ {rate_hz} Hz), nats.")
    print("=" * 90)
    pooled_p2 = {k: _m(floor_acc[k]) for k in floor_keys}
    print(f"  CE_marginal (collapsed)     : {pooled_p2['CE_marginal']}")
    print(f"  CE_factorized (indep dims)  : {pooled_p2['CE_factorized']}")
    print(f"  TOTAL CORRELATION (joint's worth) : {pooled_p2['TC']}  "
          f"(compare M17 gauss-TC 0.200)")
    print(f"  CE_ceiling (perfect head)   : {pooled_p2['CE_ceiling']}")
    print(f"  HEADROOM for latents (marginal − ceiling) : {pooled_p2['headroom_latents']}  "
          f"(joint 6-vector version of M12 'nats left')")
    ok_order = (pooled_p2["CE_ceiling"] is not None
                and pooled_p2["CE_ceiling"] < pooled_p2["CE_marginal"])
    ok_tc = pooled_p2["TC"] is not None and pooled_p2["TC"] >= -1e-6
    print(f"  [check] ceiling < marginal (noise entropy < signal+noise) : "
          f"{'OK' if ok_order else '*** VIOLATED ***'}")
    print(f"  [check] TC >= 0 (Hadamard)                                : "
          f"{'OK' if ok_tc else '*** VIOLATED ***'}")

    print("\n" + "=" * 90)
    print(f"PART 3 — per-dim target reliability @ {rate_hz} Hz -> secondary_head.NOISE_VAR refresh.")
    print(f"  NOISE_VAR[d] = 1 - reliability[d]. STD_REF_RELIABILITY = the 3 *_sd reliabilities.")
    print("=" * 90)
    noise_var_refresh = {}
    for d in DIMS:
        rel = _m(per_dim_rel[d])
        nv = round(1.0 - rel, 4) if rel is not None else None
        noise_var_refresh[d] = {"reliability": rel, "noise_var": nv}
        print(f"  {d:<8} reliability {rel}   NOISE_VAR {nv}")
    n_ref = round(float(np.mean(n_elec_scored)), 3) if n_elec_scored else None
    print(f"\n  N_REF (mean electrode count over {len(n_elec_scored)} scored parcel-instances, "
          f"n>={MIN_CONTACTS}) = {n_ref}")
    print(f"  -> secondary_head.py: NOISE_VAR = ({', '.join(str(noise_var_refresh[d]['noise_var']) for d in DIMS)})")
    print(f"     STD_REF_RELIABILITY = ({', '.join(str(noise_var_refresh[d]['reliability']) for d in DIMS[3:])}); "
          f"N_REF = {n_ref}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"slot_frames": a.slot_frames, "rate_hz": rate_hz, "n_splits": N_SPLITS,
                       "min_contacts": MIN_CONTACTS, "n_ref": n_ref, "part1_cm": pooled_p1,
                       "part2_floor": pooled_p2, "noise_var_refresh": noise_var_refresh,
                       "per_session": rows}, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
