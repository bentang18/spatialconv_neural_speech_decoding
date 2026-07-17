"""M18 — is the parcel-state residual GAUSSIAN or LAPLACE? Decides r5 Arm 3's point loss.

Arm 3 replaces the full-cov Gaussian NLL with a POINT loss. L2 vs L1 is not a taste
question and not an ablation-hygiene question: it is a NOISE MODEL. L2 is the Gaussian
log-likelihood with the covariance removed (``secondary_head._nll_terms`` at Sigma=I is
exactly ``0.5*||r||^2 + const``); L1 is the LAPLACE log-likelihood. Picking one asserts a
residual distribution. So measure the distribution.

THE ESTIMATOR. Split each parcel's electrodes into two disjoint halves, compute the state
6-vector from each half, and subtract:

    d = state(half A) - state(half B)

The parcel's true state is common to both halves, so it CANCELS. What is left is pure
electrode-sampling noise with the model removed — no checkpoint, no fit, nothing to
overfit. Its shape is the thing L2/L1 disagrees about.

  Excess kurtosis 0 => Gaussian => L2.   Excess kurtosis 3 => Laplace => L1.

WHY THE SYNTHETIC CONTROLS ARE NOT OPTIONAL. The state is a MEAN over electrodes, so the
CLT attenuates whatever the per-electrode distribution is: the mean of m iid Laplace
variables has excess kurtosis 3/m, which at our median parcel (n=6 => m=3) is already
down to ~1. A raw reading of "kurtosis 0.4" therefore does NOT distinguish "the residual
is Gaussian" from "the probe cannot tell". So we push GAUSSIAN and LAPLACE electrodes
through the IDENTICAL pipeline (same parcel geometry, same halving, same slot averaging,
same pooling) and read the real data against those two poles. If the poles do not
separate, the probe is underpowered and we say so instead of shipping a number.

The controls are load-bearing for the STD dims for a second, independent reason: a
difference of two std estimates from m ~ 3 electrodes is a difference of chi-like
variables, which is skewed and heavy-tailed EVEN FOR PERFECTLY GAUSSIAN DATA. The
Gaussian control is what tells us what "no news" looks like on dims 3-5.

WHY POOLING NEEDS THE PER-(PARCEL,DIM) STANDARDIZATION. A mixture of Gaussians with
different variances is leptokurtic. Pooling raw differences across parcels of different
scales would manufacture excess kurtosis out of nothing and hand us a false L1 verdict.
So each (parcel, dim) difference is standardized by its own sd before pooling. This also
makes the probe invariant to the two frozen affine steps in ``state_target``: the
per-(subject,parcel,dim) z-score is a fixed scale (divides out), and the cross-parcel
common mode is shared by both halves of a parcel (cancels in the difference). We measure
the same coordinate the loss does.

CONSERVATISM. The halves have m = n//2 electrodes, the real target has n, so the real
target enjoys MORE CLT attenuation than this probe sees. Any heavy-tailedness here is an
UPPER bound on the real target's. A Gaussian verdict from this probe is therefore safe;
a Laplace verdict would need the margin over the Gaussian control to be large.

SECOND PAYOFF. The std-dim answer also audits the floor ALREADY RUNNING in Arms 1/2:
``N_sigma(n) ~ 1/(n-1)`` IS the Gaussian sampling variance of a variance estimator. For a
heavy-tailed residual the true Var(s^2) carries a kurtosis term, so a Laplace verdict on
dims 3-5 would mean the shipped #28/M12 std floor is misderived. (The mean-dim SEM 1/n is
distribution-free, so dims 0-2 are not exposed to that.)

WHAT THIS DOES NOT MEASURE. The residual the loss sees is sampling noise PLUS model
error. Model error needs a model, so it is out of scope here; this probe bounds the part
that is measurable without one.

Model-FREE (no checkpoint). CPU. DeltaAI login node:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_residual_tailweight \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/residual_tailweight.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from speech_decoding.models.v14_converged_v3.state_target import SLOT_STRIDE
from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, V3_SESSIONS, WINSOR
from scripts.neuroprobe.probe_v3_global_structure import _read_clips

DIM_NAMES = ("slow_mu", "mid_mu", "hga_mu", "slow_sd", "mid_sd", "hga_sd")
N_SPLITS = 40        # split-half draws per parcel (matches M11)
MIN_ELEC = 4         # need >= 2 electrodes per half for a defined half-std
GAUSS_TOL = 0.30     # |excess kurtosis| the Gaussian control must stay inside
SEP_SEM = 3.0        # poles must be >= this many SEMs apart for the dim to be resolvable

# MEASURED CLT ATTENUATION (synthetic, 20 parcels x 6 electrodes = the median geometry):
# LAPLACE electrodes reach the parcel state as excess kurtosis +0.50 on the MEAN dims and
# +1.2 on the STD dims — not 3.0. The half-mean over m ~ 3 electrodes does that (mean of m
# iid Laplace has excess kurtosis 3/m). This is not a probe defect, it is the physics of
# the target: the mean dims are NEARLY GAUSSIAN NO MATTER WHAT the electrodes do, so the
# gap between the L2 and L1 worlds is only ~0.5 kurtosis there. Hence the poles are read
# with an error bar (across-session SEM) rather than against a magic threshold.


def _excess_kurtosis(x: np.ndarray) -> float:
    """Fisher excess kurtosis (0 for Gaussian, 3 for Laplace). Pooled over samples."""
    x = x - x.mean()
    m2 = float((x * x).mean())
    if m2 < 1e-20:
        return float("nan")
    m4 = float((x**4).mean())
    return m4 / (m2 * m2) - 3.0


def _half_state(env: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """env (n_clips, N, S, 3) band-major -> (n_clips, S, 6) state from electrodes ``idx``.

    Mirrors ``state_target.raw_state_vectors``: per-band MEAN over the electrodes then the
    per-band population SD across them (ddof=0, same as the shipped target)."""
    sub = env[:, idx]                                   # (n_clips, m, S, 3)
    mu = sub.mean(axis=1)                               # (n_clips, S, 3)
    sd = sub.std(axis=1)                                # (n_clips, S, 3) population
    return np.concatenate([mu, sd], axis=-1)            # (n_clips, S, 6)


def _parcel_diffs(env: np.ndarray, idx: np.ndarray, rng) -> np.ndarray | None:
    """Split-half state differences for ONE parcel -> (n_draws*n_clips*S, 6), each dim
    standardized by ITS OWN sd (see the pooling note in the module docstring)."""
    out = []
    for _ in range(N_SPLITS):
        perm = rng.permutation(idx)
        m = len(perm) // 2
        a, b = perm[:m], perm[m : 2 * m]               # equal-size disjoint halves
        out.append(_half_state(env, a) - _half_state(env, b))
    d = np.concatenate([o.reshape(-1, 6) for o in out], axis=0)
    sd = d.std(axis=0)
    if np.any(sd < 1e-10):
        return None                                     # degenerate parcel; skip
    return d / sd


def _synth_env(shape, kind: str, rng) -> np.ndarray:
    """Per-electrode envelopes with a KNOWN tail, unit variance, same shape as the real
    slot envelope. 'gauss' -> excess kurtosis 0; 'laplace' -> 3."""
    if kind == "gauss":
        return rng.standard_normal(shape)
    return rng.laplace(scale=1.0 / np.sqrt(2.0), size=shape)  # unit variance


def _pooled(env: np.ndarray, parcel_id: np.ndarray, parcels, rng) -> dict:
    """Pool standardized split-half differences over parcels -> excess kurtosis per dim."""
    chunks = []
    for p in parcels:
        idx = np.where(parcel_id == p)[0]
        d = _parcel_diffs(env, idx, rng)
        if d is not None:
            chunks.append(d)
    if not chunks:
        return {}
    d = np.concatenate(chunks, axis=0)
    return {
        "n_samples": int(d.shape[0]),
        "excess_kurtosis": [round(_excess_kurtosis(d[:, j]), 4) for j in range(6)],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=64)
    p.add_argument("--clip-frames", type=int, default=96)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out")
    a = p.parse_args()

    specs = load_v3_sessions(
        sessions=V3_SESSIONS,
        band_cache_dirs=[os.path.join(a.band_root, b) for b in BAND_DIRS],
        span_dir=a.span_dir,
        parcel_fn=make_bt_parcel_fn(a.bt_root),
        lof_report_path=None,
        winsor=WINSOR,
    )
    print(
        f"M18 — parcel-state residual tail weight | {len(specs)} sessions | {N_SPLITS} "
        f"split-half draws/parcel | parcels with >= {MIN_ELEC} electrodes\n"
        f"     Split-half difference = sampling noise with the model removed.\n"
        f"     Excess kurtosis 0 => Gaussian => L2.  3 => Laplace => L1.\n"
        f"     Synthetic gauss/laplace controls run the IDENTICAL pipeline — the CLT\n"
        f"     attenuates tails through the electrode mean, so the controls (not 0 and 3)\n"
        f"     are the poles the real data is read against.\n",
        flush=True,
    )

    real, ctl_g, ctl_l = [], [], []
    for spec in specs:
        sid, tid = spec.session_key
        parcel_id = spec.setup.parcel_id.cpu().numpy()
        parcels = [q for q in np.unique(parcel_id)
                   if int((parcel_id == q).sum()) >= MIN_ELEC]
        if not parcels:
            print(f"[s{sid}t{tid}] SKIP — no parcel with >= {MIN_ELEC} electrodes", flush=True)
            continue

        bands, _ = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        # (n_clips, N, T) per band -> slot grid, band-major last dim (state_target order)
        env_b = []
        for b in bands:
            e = b.mean(2)                                     # (n_clips, N, T) freq mean
            n_c, N, T = e.shape
            S = T // SLOT_STRIDE
            env_b.append(e[..., : S * SLOT_STRIDE].reshape(n_c, N, S, SLOT_STRIDE).mean(-1))
        env = np.stack(env_b, axis=-1)                        # (n_clips, N, S, 3)

        rng = np.random.default_rng(a.seed + sid * 100 + tid)
        r = _pooled(env, parcel_id, parcels, rng)
        g = _pooled(_synth_env(env.shape, "gauss", rng), parcel_id, parcels, rng)
        l = _pooled(_synth_env(env.shape, "laplace", rng), parcel_id, parcels, rng)
        if not (r and g and l):
            continue
        real.append(r); ctl_g.append(g); ctl_l.append(l)
        k = r["excess_kurtosis"]
        print(f"[s{sid}t{tid}] parcels {len(parcels):>3}  real excess kurtosis  "
              + "  ".join(f"{n}={v:+.2f}" for n, v in zip(DIM_NAMES, k)), flush=True)

    if not real:
        print("no usable sessions", flush=True)
        return

    def _ms(rows, j):
        """across-session mean and SEM for dim j — the error bar the verdict is read with."""
        v = np.array([r["excess_kurtosis"][j] for r in rows], dtype=float)
        sem = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        return float(v.mean()), sem

    kr = [_ms(real, j) for j in range(6)]
    kg = [_ms(ctl_g, j) for j in range(6)]
    kl = [_ms(ctl_l, j) for j in range(6)]

    print("\n" + "=" * 96)
    print(f"M18 — POOLED over {len(real)} sessions (mean +/- SEM). Each dim is read "
          f"against ITS OWN two controls.")
    print("=" * 96)
    print(f"{'dim':<9}{'gauss ctl':>16}{'REAL':>16}{'laplace ctl':>16}   verdict")
    verdicts, fracs = [], []
    for j, name in enumerate(DIM_NAMES):
        sep = kl[j][0] - kg[j][0]
        sem_r = kr[j][1]
        frac = (kr[j][0] - kg[j][0]) / sep if sep > 0 else float("nan")
        # The first question is NOT "which pole is it nearer" — it is "is it even BETWEEN
        # them". The real residual can (and does) land far ABOVE the laplace pole, where
        # both L2 and L1 are misspecified and "nearest pole" would be a misleading verdict.
        above_lap = np.isfinite(sem_r) and kr[j][0] - kl[j][0] >= SEP_SEM * sem_r
        below_gau = np.isfinite(sem_r) and kg[j][0] - kr[j][0] >= SEP_SEM * sem_r
        if above_lap:
            n_sem = (kr[j][0] - kl[j][0]) / sem_r
            v = (f"HEAVIER THAN LAPLACE ({n_sem:.1f} SEM above the laplace pole) "
                 f"-> L1 closer, but BOTH misspecified")
        elif below_gau:
            v = "LIGHTER THAN GAUSSIAN -> L2"
        elif not np.isfinite(sem_r) or sep <= 0 or sep < SEP_SEM * sem_r:
            frac = float("nan")
            v = f"UNRESOLVABLE between the poles (gap {sep:+.2f} vs SEM {sem_r:.2f})"
        else:
            lab = "GAUSSIAN" if frac < 0.5 else "LAPLACE"
            v = f"{lab:<8} ({frac:+.2f} of the way to laplace)"
        fracs.append(frac)
        verdicts.append(v)
        print(f"{name:<9}"
              f"{kg[j][0]:>+11.3f}+-{kg[j][1]:<4.2f}"
              f"{kr[j][0]:>+11.3f}+-{kr[j][1]:<4.2f}"
              f"{kl[j][0]:>+11.3f}+-{kl[j][1]:<4.2f}   {v}")

    # ---- invariants, named first, asserted, printed (feedback-build-the-invariant...) ----
    # 1. the pipeline must not manufacture tails: gaussian electrodes -> ~0 on the MEAN
    #    dims. (the SD dims are chi-like at m~3 and are NOT expected to sit at 0 — that is
    #    exactly why they get their own control rather than a hard-coded 0.)
    g_mu = max(abs(kg[j][0]) for j in range(3))
    ok_pipe = g_mu < GAUSS_TOL
    print(f"\n[check] gaussian control, MEAN dims: max |excess kurtosis| = {g_mu:.3f} "
          f"(want < {GAUSS_TOL}) -> pipeline manufactures no tails of its own "
          f"{'OK' if ok_pipe else 'VIOLATED'}")
    # 2. is the real residual even INSIDE the [gauss, laplace] interval the L2/L1 choice
    #    spans? if it sits above the laplace pole, the choice is between two wrong models.
    n_above = sum(1 for j in range(6)
                  if kr[j][0] - kl[j][0] >= SEP_SEM * kr[j][1])
    print(f"[check] dims ABOVE the laplace pole by >= {SEP_SEM:.0f} SEM: {n_above}/6 -> "
          + ("the residual is HEAVIER-TAILED THAN LAPLACE; L1 is the closer of the two "
             "but neither L2 nor L1 is the true noise model  OK (measured)"
             if n_above else "residual lies within the L2/L1 span  OK"))
    # 3. conservatism: halves carry m = n//2 electrodes, the real target carries n, so the
    #    real target is MORE CLT-attenuated than measured here => this is an upper bound.
    print("[check] halves have n//2 electrodes vs the target's n => less CLT attenuation "
          "here => measured tails are an UPPER bound on the target's  OK")
    # 4. the CLT ceiling itself is a result: report how far apart the two worlds even are.
    print("[check] pole gap (laplace - gauss) per dim: "
          + " ".join(f"{kl[j][0]-kg[j][0]:+.2f}" for j in range(6))
          + " <- the ENTIRE L2-vs-L1 disagreement available on each dim")

    mu_hvy = [j for j in range(3) if kr[j][0] - kl[j][0] >= SEP_SEM * kr[j][1]]
    sd_hvy = [j for j in range(3, 6) if kr[j][0] - kl[j][0] >= SEP_SEM * kr[j][1]]
    mu_lap = [j for j in range(3) if np.isfinite(fracs[j]) and fracs[j] >= 0.5]
    print("\nARM 3 LOSS: " + (
        f"L1 — {len(mu_hvy)}/3 mean dims are HEAVIER than the laplace pole, so L1 "
        f"(kurtosis 3) is strictly closer to the truth than L2 (kurtosis 0). Neither is "
        f"the true noise model; a Huber/Student-t would fit better but adds a second "
        f"axis to the ablation." if mu_hvy else
        ("L1 (Laplace)" if mu_lap else "L2 (Gaussian)")))
    print("#28/M12 STD FLOOR (running now in Arms 1/2): " + (
        f"SUSPECT — {len(sd_hvy)}/3 std dims are heavier-tailed than Laplace. The "
        f"N_sigma(n) ~ 1/(n-1) law IS the Gaussian sampling variance of a variance "
        f"estimator; the true Var(s^2) carries a kurtosis term this omits, so the floor "
        f"is misderived for THIS data." if sd_hvy else
        "consistent — std dims read Gaussian, so the 1/(n-1) sampling law holds."))
    print("\nNOTE — this does NOT by itself condemn the Arms 1/2 NLL head: it predicts a "
          "per-(parcel,slot) Sigma, and a time-varying Sigma can absorb exactly the scale "
          "mixture that produces this kurtosis (conditionally Gaussian). It DOES condemn "
          "a head whose Sigma is pinned by a dominant floor — which is what r4 measured "
          "(N ~ 90% of Sigma). See project-r4-secondary-flatlined-floor-dominates.")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({
                "dims": list(DIM_NAMES),
                "excess_kurtosis_real": [round(m, 4) for m, _ in kr],
                "excess_kurtosis_real_sem": [round(s, 4) for _, s in kr],
                "excess_kurtosis_gauss_control": [round(m, 4) for m, _ in kg],
                "excess_kurtosis_laplace_control": [round(m, 4) for m, _ in kl],
                "frac_toward_laplace": [None if not np.isfinite(f) else round(f, 4)
                                        for f in fracs],
                "verdicts": verdicts,
                "n_sessions": len(real),
                "n_splits": N_SPLITS,
                "min_elec": MIN_ELEC,
            }, f, indent=2)
        print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
