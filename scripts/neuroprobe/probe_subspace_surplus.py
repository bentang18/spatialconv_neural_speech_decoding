"""Is the low-LR arm's SURPLUS enc12 rank signal, or nuisance? (CPU-only, no GPU.)

The r6 LR sweep gave a clean contradiction: rankme orders 1e-3 > 3e-3 > 6e-3 (215/180/150
at d=256) while masked-token reconstruction EV orders 6e-3 > 3e-3 > 1e-3. Reconstruction
CANNOT adjudicate, for three structural reasons: (1) enc12 reaches the loss only through the
linear 256->128 projection into the d=128 predictor (towers.py:18,53), so encoder directions
past ~128 are in that map's null space and are invisible to the loss by construction;
(2) rankme is an entropy over SINGULAR VALUES (teacher_rank.py:140-143, p_k = s_k/sum s),
while reconstruction is an L2 (s^2) quantity, so rankme weights the small-s tail far more
heavily; (3) EV is floor-dominated once the top directions carry band power + slow envelope,
so it is near-blind to representation dimensionality. rankme is also taken on VISIBLE tokens
(objective.py:375) whereas EV is on MASKED ones.

So the arms differ mostly in a subspace the reconstruction loss cannot see. This probe asks
what lives there. For each arm it takes the enc12 spectrum, splits the d=256 basis into
consecutive rank BANDS ([0,64), [64,128), [128,192), [192,256) by default), and scores each
band on (a) the 4 downstream tasks, WS and CS, and (b) two NUISANCE targets. Bands are
dimension-MATCHED, so "more dims probe better" cannot explain a difference.

READ:
  surplus bands carry task info      -> the low-LR arm retains real structure; high LR is
                                        annealing away signal, and enc d=256 may be binding.
  surplus bands carry only nuisance  -> high LR is correctly compressing; rankme is misleading
                                        us and width is NOT the constraint.

FEATURE SPACE. Prefers the per-electrode tap ``enc12_elec`` (n, n_contacts, k*256) — the
native token space, exactly where the monitor's rankme is computed. That tap is OFF by
default; it needs ``--elec-taps 12`` on the encode (same forward, extra storage only). Falls
back to the standard parcel-pooled ``enc12`` cache, which lives in the SAME d=256 basis (each
(window, parcel, time) row is a contact-mean 256-vector) but attenuates directions that vary
within a parcel — a real caveat, printed loudly when it applies.

The downstream leg always uses the POOLED cache: parcel-mean is a mean over contacts, which
commutes with the linear projection onto a band, so pool(x @ V) == pool(x) @ V exactly. No
contact->parcel assignment is needed (the caches do not store one).

Ridge, splits, standardization and AUROC are imported from v3_probe_readout_r4 so the numbers
sit on the same footing as the real pretrain probe. Downstream uses the dual (Gram) solve on
n windows; the nuisance probe has many more rows than features, so it uses a primal solve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_probe_readout_r4 import (  # noqa: E402
    CS_TRAIN_ANCHOR,
    PROBE_COHORT_7,
    PROBE_TASKS,
    _finite,
    _load,
    _ridge_test,
    auroc,
)

from speech_decoding.experiments.monitors.teacher_rank import (  # noqa: E402
    _rankme_from_singular_values,
)
from speech_decoding.models.v14_converged_v3.towers import ENC_D_MODEL  # noqa: E402

TAP = "enc12"
ELEC_TAP = "enc12_elec"
FP16_EPS = 2.0**-11  # fp16 relative precision; caches are stored fp16


# ── feature access ──────────────────────────────────────────────────────────────────
def _pooled(rec) -> torch.Tensor:
    """(n, |P|, k, d) fp32 view of the parcel-pooled enc12 cache."""
    x = rec["feats"][TAP]["raw"]
    n, p, f = x.shape
    return x.to(torch.float32).reshape(n, p, f // ENC_D_MODEL, ENC_D_MODEL)


def _token_rows(rec, max_rows: int, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    """Rows in the native d=256 encoder basis. Prefers the unpooled per-electrode tap."""
    native = ELEC_TAP in rec["feats"]
    if native:
        x = rec["feats"][ELEC_TAP]["raw"]
        n, c, f = x.shape
        z = x.to(torch.float32).reshape(n * c * (f // ENC_D_MODEL), ENC_D_MODEL)
    else:
        z = _pooled(rec).reshape(-1, ENC_D_MODEL)
    z = z.numpy()
    if z.shape[0] > max_rows:
        z = z[rng.choice(z.shape[0], max_rows, replace=False)]
    return z, native


# ── spectrum ────────────────────────────────────────────────────────────────────────
def _spectrum(z: np.ndarray) -> dict:
    """Uncentered spectrum (what the monitor sees) + centered basis (correct for geometry).

    The monitor calls svdvals on raw features with no centering (teacher_rank.py:240-241), so
    its first component is dominated by the feature MEAN. Reported here for comparability; all
    subspace geometry and the band sweep use the CENTERED basis, which is the covariance
    eigenbasis and the meaningful object.
    """
    sv_raw = np.linalg.svd(z, compute_uv=False)
    zc = z - z.mean(0, keepdims=True)
    u_s, sv_c, vt = np.linalg.svd(zc, full_matrices=False)
    del u_s
    energy = sv_c**2
    cum = np.cumsum(energy) / energy.sum()
    # fp16 STORAGE FLOOR. Caches are written fp16 (v3_probe_encode_r4.py), whose relative
    # precision is ~2^-11. A direction whose singular value is below sv[0]*FP16_EPS is
    # quantization noise, not representation — a band living entirely under it reads as
    # "no information" no matter what the encoder actually learned. Reported so a dead tail
    # band is never mistaken for an informative negative result.
    floor = float(sv_c[0]) * FP16_EPS
    return {
        "rankme_uncentered": _rankme_from_singular_values(torch.from_numpy(sv_raw)),
        "rankme_centered": _rankme_from_singular_values(torch.from_numpy(sv_c)),
        "d": int(z.shape[1]),
        "n_rows": int(z.shape[0]),
        "dims_50pct_energy": int(np.searchsorted(cum, 0.50) + 1),
        "dims_90pct_energy": int(np.searchsorted(cum, 0.90) + 1),
        "dims_99pct_energy": int(np.searchsorted(cum, 0.99) + 1),
        "dims_above_fp16_floor": int((sv_c > floor).sum()),
        "V": vt.T,  # (d, d) centered right-singular vectors, columns ordered by singular value
        "sv_centered": sv_c,
    }


def _subspace_overlap(v_a: np.ndarray, v_b: np.ndarray, k: int) -> float:
    """Mean cos^2 of principal angles between the two top-k subspaces. 1 = identical, k/d = random."""
    m = v_a[:, :k].T @ v_b[:, :k]
    return float((np.linalg.svd(m, compute_uv=False) ** 2).mean())


def _cka(z_a: np.ndarray, z_b: np.ndarray) -> float:
    """Linear CKA on matched rows (feature-space similarity, rotation-invariant)."""
    a = z_a - z_a.mean(0, keepdims=True)
    b = z_b - z_b.mean(0, keepdims=True)
    c = np.linalg.norm(a.T @ b, "fro") ** 2
    return float(c / (np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")))


# ── band projection + downstream scoring ────────────────────────────────────────────
def _proj_feat(rec, rows, v_band: np.ndarray, col_idx=None) -> np.ndarray:
    """Pooled cache projected onto a rank band -> (r, |cols|*k*K) fp32.

    pool() is a mean over contacts and the projection is linear, so applying V to the pooled
    cache is exactly equal to projecting per-contact and then pooling.
    """
    x = _pooled(rec)[np.asarray(rows, dtype=np.int64)]
    if col_idx is not None:
        x = x[:, np.asarray(col_idx, dtype=np.int64)]
    x = torch.einsum("npkd,de->npke", x, torch.from_numpy(v_band.astype(np.float32)))
    return x.reshape(x.shape[0], -1).numpy()


def _ws_band(rec, task: str, v_band: np.ndarray) -> float:
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    folds = []
    for sp in rec["ws_split"][task].values():
        tr, te = _finite(y, sp["train"]), _finite(y, sp["test"])
        if len(tr) < 2 or len(te) < 2:
            folds.append(float("nan"))
            continue
        folds.append(_ridge_test(_proj_feat(rec, tr, v_band), y[tr],
                                 _proj_feat(rec, te, v_band), y[te], "std"))
    return float(np.nanmean(folds)) if folds else float("nan")


def _cs_band(anchor_rec, test_rec, task: str, v_band: np.ndarray) -> float:
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return float("nan")
    a_p = np.asarray(anchor_rec["present_parcels"], dtype=np.int64)
    t_p = np.asarray(test_rec["present_parcels"], dtype=np.int64)
    common = np.intersect1d(a_p, t_p)
    if common.size == 0:
        return float("nan")
    a_idx = [int(np.where(a_p == c)[0][0]) for c in common]
    t_idx = [int(np.where(t_p == c)[0][0]) for c in common]
    return _ridge_test(_proj_feat(anchor_rec, tr, v_band, a_idx), y_a[tr],
                       _proj_feat(test_rec, te, v_band, t_idx), y_t[te], "std")


# ── nuisance ────────────────────────────────────────────────────────────────────────
def _primal_ridge(x_tr, y_tr, x_te, lam_mult: float = 1.0) -> np.ndarray:
    """(X'X + lam I)^-1 X'y — rows >> features here, so the primal solve is the cheap side."""
    x_tr = np.asarray(x_tr, dtype=np.float64)
    x_te = np.asarray(x_te, dtype=np.float64)
    g = x_tr.T @ x_tr
    lam = lam_mult * float(np.trace(g) / max(g.shape[0], 1))
    w = np.linalg.solve(g + lam * np.eye(g.shape[0]), x_tr.T @ np.asarray(y_tr, dtype=np.float64))
    return x_te @ w


def _nuisance_band(rec, v_band: np.ndarray, max_classes: int) -> dict:
    """Can this band identify WHERE (parcel) and WHEN (time index) a token came from?

    Rows are (window, parcel, time). Train/test split is by WINDOW — the first WS fold — so
    window-level structure cannot leak across the split. One-vs-rest ridge, macro AUROC over
    the `max_classes` largest classes. Standardized on TRAIN stats only, as the task probe is.
    """
    sp = next(iter(rec["ws_split"][PROBE_TASKS[0]].values()))
    tr_w = np.asarray(sp["train"], dtype=np.int64)
    te_w = np.asarray(sp["test"], dtype=np.int64)
    if len(tr_w) < 2 or len(te_w) < 2:
        return {"parcel_id": float("nan"), "time_index": float("nan")}

    def _flat(w_idx):
        x = _pooled(rec)[w_idx]                                        # (w, P, k, d)
        x = torch.einsum("npkd,de->npke", x, torch.from_numpy(v_band.astype(np.float32)))
        w, p, k, e = x.shape
        lab_p = np.tile(np.repeat(np.arange(p), k), w)                 # row-major (w, p, k)
        lab_t = np.tile(np.arange(k), w * p)
        return x.reshape(w * p * k, e).numpy(), lab_p, lab_t

    x_tr, p_tr, t_tr = _flat(tr_w)
    x_te, p_te, t_te = _flat(te_w)
    mu, sd = x_tr.mean(0), x_tr.std(0)
    sd[sd == 0] = 1.0
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd

    out = {}
    for name, lab_tr, lab_te in (("parcel_id", p_tr, p_te), ("time_index", t_tr, t_te)):
        scores = []
        classes = sorted(np.unique(lab_tr), key=lambda c: -(lab_tr == c).sum())[:max_classes]
        for c in classes:
            if (lab_te == c).sum() < 2 or (lab_te != c).sum() < 2:
                continue
            s = _primal_ridge(x_tr, (lab_tr == c).astype(np.float64), x_te)
            scores.append(auroc(s, (lab_te == c).astype(np.float64)))
        out[name] = float(np.nanmean(scores)) if scores else float("nan")
    return out


# ── driver ──────────────────────────────────────────────────────────────────────────
def _bands(d: int, width: int) -> list[tuple[int, int]]:
    return [(lo, min(lo + width, d)) for lo in range(0, d, width)]


def _mean(vals) -> float:
    """nanmean that stays quiet on an all-nan column (e.g. the CS block under --skip-cs)."""
    v = [float(x) for x in vals if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=CACHE_DIR:TAG",
                    help="repeat per LR arm, e.g. --arm lr1e-3=/abs/cache_lr1e3:r6_lr1e3_10k")
    ap.add_argument("--band-width", type=int, default=64)
    ap.add_argument("--max-rows", type=int, default=200_000, help="rows subsampled for the SVD")
    ap.add_argument("--spectrum-session", default="2,1", help="session used for the SVD basis")
    ap.add_argument("--max-classes", type=int, default=8, help="nuisance one-vs-rest classes")
    ap.add_argument("--skip-cs", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=33)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        cache_dir, tag = rest.rsplit(":", 1)
        if not os.path.isabs(cache_dir):
            raise SystemExit(f"--arm {name}: cache dir must be ABSOLUTE, got {cache_dir}")
        arms[name] = (cache_dir, tag)

    sess_spec = tuple(int(v) for v in args.spectrum_session.split(","))
    results: dict = {"config": {"band_width": args.band_width, "spectrum_session": sess_spec,
                                "arms": {k: list(v) for k, v in arms.items()}}, "arms": {}}

    # ── Part A: spectrum per arm, on ONE shared session so rows are matched across arms ──
    print(f"\n[A] spectrum on session {sess_spec}", flush=True)
    spec_by_arm, rows_by_arm = {}, {}
    native_all = True
    for name, (cache_dir, tag) in arms.items():
        rec = _load(cache_dir, sess_spec, tag)
        z, native = _token_rows(rec, args.max_rows, np.random.default_rng(args.seed))
        native_all &= native
        sp = _spectrum(z)
        spec_by_arm[name], rows_by_arm[name] = sp, z
        results["arms"][name] = {"spectrum": {k: v for k, v in sp.items()
                                              if k not in ("V", "sv_centered")}}
        print(f"  {name:>10}  rankme(uncentered)={sp['rankme_uncentered']:7.2f}  "
              f"centered={sp['rankme_centered']:7.2f}  "
              f"dims@50/90/99%E={sp['dims_50pct_energy']}/{sp['dims_90pct_energy']}/"
              f"{sp['dims_99pct_energy']}  above-fp16-floor={sp['dims_above_fp16_floor']}"
              f"/{sp['d']}  rows={sp['n_rows']}", flush=True)
    if not native_all:
        print("  [warn] per-electrode tap enc12_elec absent -> using the parcel-POOLED cache.\n"
              "         Same d=256 basis, but contact-averaging attenuates within-parcel\n"
              "         directions, so the spectrum is a LOWER bound on token-space rank.\n"
              "         Re-encode with --elec-taps 12 for the monitor-exact space.", flush=True)
    print(f"  [check] native token space = {native_all} -> "
          f"{'OK' if native_all else 'POOLED FALLBACK'}", flush=True)

    # ── cross-arm geometry ──
    print("\n[A2] cross-arm subspace overlap (mean cos^2 of principal angles; random = k/d)",
          flush=True)
    names = list(arms)
    results["geometry"] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = {f"top{k}": _subspace_overlap(spec_by_arm[a]["V"], spec_by_arm[b]["V"], k)
                  for k in (32, 64, 128) if k <= spec_by_arm[a]["d"]}
            ck = _cka(rows_by_arm[a], rows_by_arm[b]) if (
                rows_by_arm[a].shape == rows_by_arm[b].shape) else float("nan")
            results["geometry"][f"{a}|{b}"] = {"overlap": ov, "cka": ck}
            rnd = {k: k / spec_by_arm[a]["d"] for k in (32, 64, 128)}
            print(f"  {a:>10} vs {b:<10} " + "  ".join(
                f"{k}={v:.3f}(rand {rnd[int(k[3:])]:.3f})" for k, v in ov.items())
                + f"  CKA={ck:.3f}", flush=True)

    # ── Part B: band sweep ──
    bands = _bands(spec_by_arm[names[0]]["d"], args.band_width)
    print(f"\n[B] band sweep, dimension-matched K={args.band_width}: "
          f"{[f'[{lo},{hi})' for lo, hi in bands]}", flush=True)
    anchor = {n: _load(arms[n][0], CS_TRAIN_ANCHOR, arms[n][1]) for n in names}

    for name in names:
        cache_dir, tag = arms[name]
        v_full = spec_by_arm[name]["V"]
        per_band = {}
        for lo, hi in bands:
            v_band = v_full[:, lo:hi]
            key = f"[{lo},{hi})"
            ws, cs = {t: [] for t in PROBE_TASKS}, {t: [] for t in PROBE_TASKS}
            nuis = []
            for sess in PROBE_COHORT_7:
                rec = _load(cache_dir, sess, tag)
                for task in PROBE_TASKS:
                    ws[task].append(_ws_band(rec, task, v_band))
                    if not args.skip_cs and tuple(sess) != tuple(CS_TRAIN_ANCHOR):
                        cs[task].append(_cs_band(anchor[name], rec, task, v_band))
                nuis.append(_nuisance_band(rec, v_band, args.max_classes))
                del rec
            per_band[key] = {
                "ws": {t: float(np.nanmean(v)) for t, v in ws.items()},
                "cs": {t: float(np.nanmean(v)) if v else float("nan") for t, v in cs.items()},
                "nuisance": {k: float(np.nanmean([d[k] for d in nuis])) for k in nuis[0]},
            }
            sv = spec_by_arm[name]["sv_centered"]
            live = bool((sv[lo:hi] > float(sv[0]) * FP16_EPS).any())
            per_band[key]["above_fp16_floor"] = live
            b = per_band[key]
            print(f"  {name:>10} {key:>12}  WS={_mean(b['ws'].values()):.4f}  "
                  f"CS={_mean(b['cs'].values()):.4f}  "
                  f"parcel={b['nuisance']['parcel_id']:.4f}  "
                  f"time={b['nuisance']['time_index']:.4f}"
                  f"{'' if live else '   [DEAD: band under fp16 floor, result uninformative]'}",
                  flush=True)
        results["arms"][name]["bands"] = per_band

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nwrote {args.out}", flush=True)

    print("\n[verdict inputs] per arm, tail-band task AUROC minus chance (0.5), vs its nuisance:",
          flush=True)
    for name in names:
        for key, b in results["arms"][name]["bands"].items():
            task = _mean(list(b["cs"].values()) + list(b["ws"].values()))
            print(f"  {name:>10} {key:>12}  task-0.5={task - 0.5:+.4f}  "
                  f"nuisance-0.5={_mean(b['nuisance'].values()) - 0.5:+.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
