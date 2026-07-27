"""Shared loading for the cross-subject figures: reduced .npz -> lobe-pooled condition means.

Every figure downstream is a view of the same object, so the decisions that could quietly
manufacture an effect live here once, in the open:

  * Lobe pooling is ELECTRODE-WEIGHTED. The cache stores a parcel MEAN, so pooling parcels
    into a lobe by a plain mean would weight a 1-electrode parcel like a 40-electrode one.
  * Standardization is per-channel, using each session's own mean/var over all trials,
    parcels and time. It removes a per-subject channel offset and gain -- exactly the
    nuisance that would otherwise dominate any cross-subject comparison -- and it cannot
    manufacture a shared pattern over time or anatomy. enc0 is the control that proves
    this: it gets the identical treatment, so anything enc12 shows that enc0 does not is
    the encoder's doing, not the standardizer's.
  * Per-subject centering is OPTIONAL and always reported. Subtracting a subject's own
    grand mean removes the identity offset. It is the difference between asking "are these
    subjects distinguishable" and "do they move the same way", and both are shown.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from speech_decoding.studies.braintreebank.anatomy import parcel_lobe_keys

UNKNOWN_LOBE = "unknown"


def lobe_of(parcel_ids: np.ndarray, *, pool_hemi: bool) -> list[str]:
    keys = parcel_lobe_keys()
    out = []
    for p in parcel_ids:
        k = keys[int(p)]
        out.append(k if (k == UNKNOWN_LOBE or not pool_hemi) else k.split("-", 1)[1])
    return out


@dataclass
class Session:
    subject_id: int
    trial_id: int
    parcels: np.ndarray        # (|P|,) atlas ids
    counts: np.ndarray         # (|P|,) electrodes per parcel
    lobes: list[str]           # (|P|,) lobe key per parcel
    cond: dict                 # (tap, task, cls, half) -> (|P|, T, C)
    chan_mu: dict              # tap -> (C,)
    chan_sd: dict              # tap -> (C,)
    shapes: dict               # tap -> (|P|, T, C)

    @property
    def key(self) -> str:
        return f"S{self.subject_id}T{self.trial_id}"


def load_session(path: str, *, pool_hemi: bool = True) -> Session:
    z = np.load(path, allow_pickle=False)
    parcels = z["present_parcels"]
    counts = z["parcel_counts"]
    cond, chan_mu, chan_sd, shapes = {}, {}, {}, {}
    for k in z.files:
        if k.endswith("/shape"):
            tap = k.split("/")[0]
            shapes[tap] = tuple(int(v) for v in z[k])
            n_p, t, c = shapes[tap]
            # pool the per-column moments over parcels and time -> exact per-channel moments
            m1 = z[f"{tap}/col_sum"].reshape(n_p, t, c)
            m2 = z[f"{tap}/col_sq"].reshape(n_p, t, c)
            mu = m1.mean(axis=(0, 1))
            var = m2.mean(axis=(0, 1)) - mu ** 2
            chan_mu[tap] = mu.astype(np.float64)
            chan_sd[tap] = np.sqrt(np.maximum(var, 0.0)).astype(np.float64) + 1e-8
    for k in z.files:
        parts = k.split("/")
        if len(parts) == 4 and parts[0] in shapes and parts[2].startswith("c"):
            tap, task, cls, half = parts[0], parts[1], int(parts[2][1:]), parts[3]
            cond[(tap, task, cls, half)] = z[k]
    return Session(int(z["subject_id"]), int(z["trial_id"]), parcels, counts,
                   lobe_of(parcels, pool_hemi=pool_hemi), cond, chan_mu, chan_sd, shapes)


def load_all(red_dir: str, *, pool_hemi: bool = True) -> list[Session]:
    paths = sorted(glob.glob(os.path.join(red_dir, "red_s*_t*_*.npz")))
    assert paths, f"no reduced sessions in {red_dir}"
    return [load_session(p, pool_hemi=pool_hemi) for p in paths]


def shared_lobes(sessions, *, min_elec: int = 2) -> list[str]:
    """Lobes every SUBJECT has with at least min_elec electrodes. Subjects, not sessions:
    two trials of one subject share a montage and would double-count as agreement."""
    by_subj: dict[int, dict[str, int]] = {}
    for s in sessions:
        d = by_subj.setdefault(s.subject_id, {})
        for lobe, n in zip(s.lobes, s.counts):
            d[lobe] = max(d.get(lobe, 0), int(n))
    subjects = sorted(by_subj)
    all_lobes = {lb for d in by_subj.values() for lb in d if lb != UNKNOWN_LOBE}
    return sorted(lb for lb in all_lobes
                  if all(by_subj[s].get(lb, 0) >= min_elec for s in subjects))


def lobe_mean(sess: Session, tap: str, task: str, cls: int, half: str, lobe: str,
              *, standardize: bool = True) -> np.ndarray | None:
    """Electrode-weighted mean over the parcels of one lobe -> (T, C), or None if absent."""
    key = (tap, task, cls, half)
    if key not in sess.cond:
        return None
    sel = [i for i, lb in enumerate(sess.lobes) if lb == lobe]
    if not sel:
        return None
    x = sess.cond[key][sel]                      # (|sel|, T, C)
    w = sess.counts[sel].astype(np.float64)
    out = np.tensordot(w, x, axes=(0, 0)) / w.sum()
    if standardize:
        out = (out - sess.chan_mu[tap]) / sess.chan_sd[tap]
    return out


def session_matrix(sess: Session, tap: str, task: str, cls: int, half: str, lobes,
                   *, standardize: bool = True) -> np.ndarray | None:
    """Stack the requested lobes -> (n_lobes, T, C). None if any lobe is missing."""
    rows = [lobe_mean(sess, tap, task, cls, half, lb, standardize=standardize) for lb in lobes]
    ok = [r for r in rows if r is not None]
    if len(ok) != len(rows):
        return None
    return np.stack(ok, axis=0)


def center_per_session(mats: list[np.ndarray]) -> list[np.ndarray]:
    """Remove each session's own grand mean over (rows, time). What is left is how that
    session MOVES, with its identity offset gone -- the content half of the split."""
    return [m - m.mean(axis=(0, 1), keepdims=True) for m in mats]


def pca_basis(stack: np.ndarray, k: int = 3):
    """Right singular vectors of tokens x channels. Returns (components (k, C), mean (C,),
    explained variance ratio (k,))."""
    mu = stack.mean(axis=0)
    x = stack - mu
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    ev = s ** 2 / max((x.shape[0] - 1), 1)
    return vt[:k], mu, (ev[:k] / ev.sum())


def to_rgb(proj: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Per-component robust percentile stretch to [0,1] -- the DINOv3 recipe. The stretch is
    computed on the POOLED projection so one shared colour scale covers every panel;
    stretching per panel would make different values look identical across subjects."""
    out = np.empty_like(proj, dtype=np.float64)
    for i in range(proj.shape[-1]):
        a, b = np.percentile(proj[..., i], [lo, hi])
        out[..., i] = np.clip((proj[..., i] - a) / max(b - a, 1e-12), 0, 1)
    return out
