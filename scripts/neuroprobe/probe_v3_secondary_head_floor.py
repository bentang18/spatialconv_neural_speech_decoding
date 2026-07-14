"""M12 — the model-free FLOOR for the B3 secondary head, in NATS.

WHY THIS EXISTS. M9 is the cautionary tale: we trained a 12-block encoder for 40k steps
believing it beat its input by +0.041, and an exact-parity floor said PARITY. The number
that would have caught it on day one was a floor computed from the data alone. B3's head
gets one BEFORE it is ever trained.

The B3 target (see [[project-parcel-state-kl-secondary-loss-2026-07-15]]) is, per time slot:
for every (DKT parcel p, band b) present in the montage, a SOFT distribution over K_b bins
of that parcel's z-scored activity, with kernel sigma_b = the measured noise sd (M11). The
head sees ONLY the M latents and must predict that distribution; loss is KL.

THREE FLOORS, all computable in closed form from the cache, no model:

  CE_marginal   The head predicts the MARGINAL bin distribution, ignoring its input entirely.
                This is the COLLAPSED head — the KL analogue of the smear. A regression head
                answering an unpredictable target emits the conditional mean; a KL head emits
                the marginal. The difference is that THIS ONE IS DETECTABLE, and this is the
                number that detects it.

  CE_commonmode The head predicts from the COMMON MODE ALONE (the montage-mean activity at t).
                This is THE BAR. The common mode is ALREADY FREE to the downstream readout —
                input-linear scores 0.6462 largely on it — so a head that learns only the
                common mode has taught the latents nothing the readout could not already get.
                Estimated NONPARAMETRICALLY: bin the common mode by quantile, and within each
                bin take the empirical mean target distribution. That is the best possible
                function of the common mode, so the bar is not understated by a weak model.

  CE_ceiling    The best ACHIEVABLE. A perfect head predicts the SIGNAL exactly but still
                cannot predict the MEASUREMENT NOISE, and that is what this costs.

                ⚠️ THE FIRST VERSION OF THIS WAS WRONG and it is instructive. It predicted
                HALF-B's target from HALF-A's summary — leak-free, but it scored a DIFFERENT,
                NOISIER TARGET (a half-parcel summary has ~sqrt(2) the noise of the full one).
                The result was CE_ceiling > CE_commonmode, i.e. a "lower bound" ABOVE the
                thing it was supposed to bound — impossible, and the tell that the estimator
                was measuring another problem.

                Correct version: a signal model with the MEASURED reliability r (M11).
                Write the z-scored parcel summary as z = s + e, with var(s) = r and
                var(e) = 1 - r. A perfect head knows s; its optimal predictive distribution
                over the bins is the true conditional of z given s, which has width
                sqrt(sigma_noise^2 + sigma_kernel^2). CE_ceiling is the expected cross-entropy
                of the soft target q(z) against that. Monte-Carlo'd on the same bin grid.
                ASSUMES s and e are Gaussian — stated, because it is the one modelling
                assumption in this script. H_target and CE_marginal below are model-free.

  The head's live cross-entropy must land in [CE_ceiling, CE_marginal], and its position in
  that interval IS the diagnostic. Beating CE_commonmode is the definition of "the latents
  learned a global state." Sitting at CE_marginal is the definition of collapse.

  Also reported: H_target — the target's OWN entropy, which is the IRREDUCIBLE AMBIGUITY the
  soft binning deliberately preserves (S-JEPA's point). CE cannot go below it.

EVERYTHING IS HELD OUT. Bin edges, quantile edges, and the conditional tables are estimated on
TRAIN clips and scored on TEST clips, with span-overlap pruning — otherwise the nonparametric
conditional memorizes and the "floor" comes back impossibly low.

Model-FREE. CPU. DeltaAI/Delta login node:

  ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
  .venv/bin/python -m scripts.neuroprobe.probe_v3_secondary_head_floor \
      --band-root $ROOT \
      --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
      --bt-root /projects/bhqk/htang13/braintreebank \
      --out /projects/bhqk/htang13/probe_out_v3/field_stats/secondary_head_floor.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from scripts.neuroprobe.probe_v3_field_stats import BAND_DIRS, V3_SESSIONS, WINSOR
from scripts.neuroprobe.probe_v3_global_structure import (
    MIN_PARCEL_CONTACTS,
    _read_clips,
    _split_clips,
)

BANDS = ("slow", "mid", "hga")

# FROZEN BY M11 (parcel_reliability_bands.json, 13 montages, pooled). K = empirical
# p1..p99 range / noise sd; sigma = sqrt(1 - reliability). Per-band, not shared — the
# kernel width IS the noise sd and it genuinely differs by band.
K_BINS = {"slow": 11, "mid": 8, "hga": 7}
KERNEL_SD = {"slow": 0.4397, "mid": 0.5904, "hga": 0.6484}
# split-half reliability, common mode REMOVED (M11 pooled). Sets the ceiling: var(signal) = r.
RELIABILITY = {"slow": 0.7916, "mid": 0.6389, "hga": 0.5612}

N_CM_BINS = 16      # quantile bins for the nonparametric common-mode conditional
N_SIG, N_NOISE = 512, 256   # Monte-Carlo draws for the ceiling
EPS = 1e-9


def _soft_target(z: np.ndarray, centers: np.ndarray, sigma: float) -> np.ndarray:
    """z (n,) -> (n, K) soft bin assignment. Gaussian kernel at the MEASURED noise sd, so
    the target's entropy IS the irreducible ambiguity of the measurement rather than a
    tunable softness. A cell that genuinely sits between two levels gets a bimodal target;
    an unambiguous one gets a sharp target. That is the property S-JEPA's soft posteriors
    exist to preserve — here it comes from a physical noise estimate, not a fitted GMM."""
    d = (z[:, None] - centers[None, :]) / max(sigma, 1e-6)
    logits = -0.5 * d**2
    logits -= logits.max(axis=1, keepdims=True)
    q = np.exp(logits)
    return q / np.maximum(q.sum(axis=1, keepdims=True), EPS)


def _entropy(q: np.ndarray) -> float:
    return float(-(q * np.log(np.maximum(q, EPS))).sum(axis=1).mean())


def _cross_entropy(q: np.ndarray, p: np.ndarray) -> float:
    """H(q, p) in nats, averaged over rows. p may be a single row (broadcast)."""
    p = np.maximum(p, EPS)
    return float(-(q * np.log(p)).sum(axis=1).mean())


def _ceiling(z_tr, centers, sigma, r, rng) -> tuple[float, float]:
    """CE of the BAYES-OPTIMAL head — one that knows the signal exactly and is defeated only
    by the measurement noise. Returns (CE_ceiling, H_sim) in nats.

    Model: z = s + e, var(s) = r, var(e) = 1 - r, e Gaussian and independent (r = M11's
    split-half reliability). The signal is NOT assumed Gaussian — s is resampled from the
    EMPIRICAL train z, rescaled to variance r, so its skew survives (HGA is not Gaussian and
    a Gaussian marginal would misplace the bin centers' occupancy). Only the NOISE is Gaussian.

    The optimal predictor given s is E[q(s + e) | s] — computed exactly by averaging the soft
    target over noise draws, so there is no closed-form-convolution approximation either.

    H_sim is the simulated target entropy; it must track the EMPIRICAL H_target. If it does
    not, the signal model is wrong and the ceiling is not to be trusted. Printed, not hidden."""
    sig_n = float(np.sqrt(max(1.0 - r, 1e-4)))
    zc = (z_tr - z_tr.mean()) / max(z_tr.std(), 1e-8)
    s = np.sqrt(max(r, 1e-4)) * rng.choice(zc, size=N_SIG, replace=True)
    z = s[:, None] + rng.normal(0.0, sig_n, size=(N_SIG, N_NOISE))
    q = _soft_target(z.reshape(-1), centers, sigma).reshape(N_SIG, N_NOISE, -1)

    p = q.mean(1)                                    # (N_SIG, K) — Bayes-optimal given s
    p /= np.maximum(p.sum(1, keepdims=True), EPS)
    ce = -(q * np.log(np.maximum(p[:, None, :], EPS))).sum(-1).mean()
    h = -(q * np.log(np.maximum(q, EPS))).sum(-1).mean()
    return float(ce), float(h)


def _cond_table(x_tr, q_tr, edges) -> np.ndarray:
    """Nonparametric E[q | bin(x)] on TRAIN. -> (n_bins, K). Empty bins fall back to the
    train marginal, so a test row landing in one is scored against the marginal, not zero."""
    idx = np.clip(np.searchsorted(edges, x_tr, side="right") - 1, 0, len(edges) - 2)
    n_bins, K = len(edges) - 1, q_tr.shape[1]
    tbl = np.tile(q_tr.mean(0), (n_bins, 1))
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 5:
            tbl[b] = q_tr[m].mean(0)
    return tbl / np.maximum(tbl.sum(1, keepdims=True), EPS)


def _apply_table(x_te, tbl, edges) -> np.ndarray:
    idx = np.clip(np.searchsorted(edges, x_te, side="right") - 1, 0, tbl.shape[0] - 1)
    return tbl[idx]


def _floors_for_band(env, parcel_id, parcels, tr, te, band: str, rng) -> dict | None:
    """All three floors for one band, pooled over the parcels of this session.

    ALL THREE ARE SCORED AGAINST THE SAME TARGET — the FULL-parcel soft distribution q. That
    is the whole point of the rewrite: the first version scored the ceiling against a
    half-parcel target and produced an ordering that cannot exist."""
    K, sigma, r = K_BINS[band], KERNEL_SD[band], RELIABILITY[band]
    # bin centers span the empirical p1..p99 range (TRAIN only), matching M11's derivation
    per = {"H_target": [], "CE_marginal": [], "CE_commonmode": [], "CE_ceiling": [],
           "H_sim_check": []}

    for p in parcels:
        idx = np.where(parcel_id == p)[0]
        out_idx = np.where(parcel_id != p)[0]
        if len(idx) < MIN_PARCEL_CONTACTS:
            continue

        def _sum(clips, rows):
            return env[clips][:, rows].mean(1).reshape(-1)

        s_tr, s_te = _sum(tr, idx), _sum(te, idx)
        mu, sd = float(s_tr.mean()), float(max(s_tr.std(), 1e-8))
        z_tr, z_te = (s_tr - mu) / sd, (s_te - mu) / sd

        lo, hi = np.percentile(z_tr, [1.0, 99.0])
        centers = np.linspace(lo, hi, K)
        q_tr, q_te = _soft_target(z_tr, centers, sigma), _soft_target(z_te, centers, sigma)

        # --- the target's own entropy: the irreducible ambiguity ---
        per["H_target"].append(_entropy(q_te))

        # --- CE_marginal: the COLLAPSED head (train marginal, applied to test) ---
        per["CE_marginal"].append(_cross_entropy(q_te, q_tr.mean(0, keepdims=True)))

        # --- CE_commonmode: THE BAR. common mode = mean of contacts OUTSIDE parcel p
        #     (leak-free — p is nowhere in its own predictor). ---
        cm_tr, cm_te = _sum(tr, out_idx), _sum(te, out_idx)
        edges = np.quantile(cm_tr, np.linspace(0, 1, N_CM_BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        tbl = _cond_table(cm_tr, q_tr, edges)
        per["CE_commonmode"].append(_cross_entropy(q_te, _apply_table(cm_te, tbl, edges)))

        # --- CE_ceiling: the BAYES-OPTIMAL head, defeated only by measurement noise.
        #     Same bin grid, same kernel, SAME TARGET. ---
        ce, h_sim = _ceiling(z_tr, centers, sigma, r, rng)
        per["CE_ceiling"].append(ce)
        per["H_sim_check"].append(h_sim)

    if not per["CE_marginal"]:
        return None
    return {k: round(float(np.mean(v)), 4) for k, v in per.items() if v} | {
        "K": K, "kernel_sd": sigma, "reliability": r,
        "n_parcels": len(per["CE_marginal"]),
        "H_uniform_logK": round(float(np.log(K)), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--band-root", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", default=os.environ.get("ROOT_DIR_BRAINTREEBANK", ""))
    p.add_argument("--n-clips", type=int, default=128)
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
    print(f"M12 — B3 secondary-head FLOOR (nats) | {len(specs)} sessions\n"
          f"     K {K_BINS}  kernel sd {KERNEL_SD}   (both FROZEN BY M11)\n"
          f"     reliability {RELIABILITY}   (M11 pooled; sets the ceiling)\n"
          f"     held out by clip, span-overlap pruned; conditional tables fit on TRAIN only\n",
          flush=True)

    rng = np.random.default_rng(a.seed)
    rows = []
    for spec in specs:
        sid, tid = spec.session_key
        parcel_id = spec.setup.parcel_id.cpu().numpy()
        parcels = [q for q in np.unique(parcel_id)
                   if int((parcel_id == q).sum()) >= MIN_PARCEL_CONTACTS]
        if len(parcels) < 3:
            continue

        bands, starts = _read_clips(spec, a.n_clips, a.clip_frames, a.seed)
        env_raw = [b.mean(2) for b in bands]
        tr, va, te = _split_clips(starts, a.clip_frames)
        tr = np.concatenate([tr, va]) if len(va) else tr      # no lambda to tune here
        if len(te) < 4:
            continue

        rec = {"subject_id": sid, "trial_id": tid, "n_parcels": len(parcels)}
        for bi, name in enumerate(BANDS):
            f = _floors_for_band(env_raw[bi], parcel_id, parcels, tr, te, name, rng)
            if f:
                rec[name] = f
        rows.append(rec)

        print(f"[s{sid}t{tid}] parcels {len(parcels)}", flush=True)
        for name in BANDS:
            if name not in rec:
                continue
            f = rec[name]
            print(f"    {name:<5} ceiling {f['CE_ceiling']:.4f} < "
                  f"BAR(common-mode) {f['CE_commonmode']:.4f} < "
                  f"collapsed {f['CE_marginal']:.4f}   (H_target {f['H_target']:.4f})",
                  flush=True)

    print("\n" + "=" * 84)
    print("M12 — POOLED. The B3 head's live cross-entropy must land in [ceiling, collapsed].")
    print("     BEATING 'BAR' IS THE DEFINITION OF 'the latents learned a global state'.")
    print("     SITTING AT 'collapsed' IS THE DEFINITION OF collapse. Log all three.")
    print("=" * 84)
    pooled: dict = {}
    for name in BANDS:
        vals = [r[name] for r in rows if name in r]
        if not vals:
            continue
        pooled[name] = {
            k: round(float(np.mean([v[k] for v in vals])), 4)
            for k in ("H_target", "H_sim_check", "CE_ceiling", "CE_commonmode",
                      "CE_marginal", "H_uniform_logK")
        } | {"K": K_BINS[name], "kernel_sd": KERNEL_SD[name],
             "reliability": RELIABILITY[name]}
        f = pooled[name]
        head = f["CE_marginal"] - f["CE_ceiling"]
        earn = f["CE_marginal"] - f["CE_commonmode"]
        left = head - earn
        print(f"\n[{name}]  K={f['K']}  kernel sd={f['kernel_sd']}  reliability="
              f"{f['reliability']}  (uniform would be log K = {f['H_uniform_logK']:.3f})")
        print(f"  H_target      (irreducible ambiguity)   : {f['H_target']:.4f} nats")
        print(f"  CE_ceiling    (a PERFECT head)          : {f['CE_ceiling']:.4f}")
        print(f"  CE_commonmode (THE BAR — the free part) : {f['CE_commonmode']:.4f}")
        print(f"  CE_marginal   (COLLAPSED head)          : {f['CE_marginal']:.4f}")
        print(f"  --> total headroom  collapsed - ceiling : {head:.4f} nats")
        print(f"  --> the common mode alone already earns : {earn:.4f} nats "
              f"({100 * earn / max(head, 1e-9):.0f}% of it)")
        print(f"  --> LEFT FOR THE LATENTS TO EARN        : {left:.4f} nats "
              f"({100 * left / max(head, 1e-9):.0f}%)")

        # --- the two checks that would have caught the FIRST version of this script ---
        ok_order = f["CE_ceiling"] < f["CE_commonmode"] < f["CE_marginal"]
        dh = abs(f["H_sim_check"] - f["H_target"])
        print(f"  [check] ordering ceiling < bar < collapsed : "
              f"{'OK' if ok_order else '*** VIOLATED — THE ESTIMATOR IS WRONG ***'}")
        print(f"  [check] simulated H {f['H_sim_check']:.4f} vs empirical H "
              f"{f['H_target']:.4f}  (|d| {dh:.4f}) : "
              f"{'OK' if dh < 0.08 else '*** signal model misfits — ceiling not trustworthy ***'}")
        pooled[name]["check_ordering_ok"] = bool(ok_order)
        pooled[name]["check_signal_model_ok"] = bool(dh < 0.08)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump({"K_bins": K_BINS, "kernel_sd": KERNEL_SD,
                       "n_cm_bins": N_CM_BINS, "per_session": rows, "pooled": pooled},
                      fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
