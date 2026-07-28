"""Shared loading for the cross-subject figures: reduced .npz -> lobe-pooled condition means.

Every figure downstream is a view of the same object, so the decisions that could quietly
manufacture an effect live here once, in the open:

  * Lobe pooling is ELECTRODE-WEIGHTED. The cache stores a parcel MEAN, so pooling parcels
    into a lobe by a plain mean would weight a 1-electrode parcel like a 40-electrode one.
  * Standardization is per-channel, using each session's own mean/var over all trials,
    parcels and time. Be honest about its size: that is 2C numbers fit per session per tap
    (512 at the 256-d taps), not "one rescaling". Two things keep it from manufacturing the
    result. It is DIAGONAL, so it cannot rotate one subject's trajectory onto another's. And
    it is fit LABEL-FREE, over every window regardless of class, so it cannot invent a class
    contrast -- and in the contrast the mean term cancels outright, leaving only the gain.
    It is needed because PCA is not scale-invariant: unstandardized, a few big-variance
    channels own every component, and their scale differs by subject (electrode count,
    impedance, coverage), so the "shared" basis would be whichever session shouts loudest.
    enc0 is the control that proves the treatment is not the effect: it gets the identical
    pipeline, so anything enc12 shows that enc0 does not is the encoder's doing.
    NOT the same as the board readout's `std`, and the differences are worth stating because
    the resemblance is close enough to mislead. (1) The readout z-scores every column of the
    flattened (parcel x feature) vector separately (`v3_board_readout.py` `_standardize_inplace`),
    so it also removes each parcel's offset and each timepoint's own profile; here the estimate
    is pooled over parcels and time and is per-CHANNEL only, because the parcel and time
    structure is the trajectory. (2) The readout reduces over windows alone; this reduces over
    trials, parcels and time. (3) The readout fits on the TRAIN split because it has a held-out
    test half to protect -- nothing is held out here, which is what the LOSO and split-half
    bases in viz_figures are for. And in the CS regime plain `std` fits on the ANCHOR subject,
    mapping the target into the anchor's frame; the per-subject analogue of what this does is
    the readout's `std_target` (AdaBN-style), a separate reported column that is never selected.
  * Per-subject centering is OPTIONAL and always reported, with two possible references --
    see center_per_session. It is the difference between asking "are these subjects
    distinguishable" and "do they move the same way", and both are shown.

WHAT THIS PIPELINE DOES NOT SHOW, and no caption should imply. Lobe pooling averages every
electrode of a lobe into ONE vector per timepoint, so all within-lobe spatial structure is
gone; what remains is a lobe-mean time course. For the Lite cohort the shared-lobe
intersection is a SINGLE lobe, so "average over lobes" is a no-op and the claim is "within
temporal cortex, do subjects' lobe-averages move alike" -- not "do subjects share a spatial
pattern". And "the same lobe" is not the same tissue: a subject with 100 temporal contacts
and one with 10 average over very different samples of it, and nothing here corrects for it.
"""
from __future__ import annotations

import glob
import json
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


def center_per_session(mats: list[np.ndarray], *,
                       n_pre: int | None = None) -> list[np.ndarray]:
    """Put every session on a common origin. Two references, and the choice is visible.

    ``n_pre=None`` subtracts the grand mean over (rows, time) -- the window's own
    time-average. That is what shipped first, and it has a specific cost: the origin is the
    average of the response, so a contrast that RISES and stays up is drawn as leaving the
    origin and coming back to it. Sustained and transient look the same.

    ``n_pre=k`` subtracts the mean over the first k frames instead. Those frames are
    pre-stimulus (the 2 s window is [-0.5, +1.5] s, so k=16 at 32 Hz), which makes the origin
    "no class difference BEFORE the event" -- a reference you can state. The sustained part
    now survives as the trajectory failing to return, which is the whole distinction between
    `onset` (a word after silence: over by the end) and `speech` (a word inside ongoing talk:
    the difference persists). Standard ERP baselining, and it only exists at 2 s -- the 1 s
    window has no pre-stimulus frames, so it keeps the time-mean reference.

    Note what baselining re-references rather than removes: for `speech` the classes already
    differ before t=0 by construction, so that offset is real signal, and subtracting it means
    the figure reads "change relative to the state at word onset". The caption has to say so.
    """
    if n_pre is None:
        return [m - m.mean(axis=(0, 1), keepdims=True) for m in mats]
    assert n_pre > 0, f"n_pre must be positive or None, got {n_pre}"
    out = []
    for m in mats:
        assert n_pre < m.shape[1], f"n_pre={n_pre} needs < {m.shape[1]} frames"
        out.append(m - m[:, :n_pre].mean(axis=(0, 1), keepdims=True))
    return out


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


def board_cs_auroc(path: str, *, norm: str = "std") -> dict:
    """(task -> tap -> cross-subject AUROC) from a board results JSON.

    The figures answer "do subjects move alike"; this answers "and does that buy decoding
    accuracy". They are different questions and the page should show both, because a shared
    trajectory that decodes at chance would be a geometry curiosity, not a result.

    Board keys are "<run>|<task>", each carrying {"cs": {"<tap>|<norm>": {cell: auroc}}}.
    CS cells are PER-SUBJECT and the board number is their MEAN -- quoting one cell is the
    standard way to misreport this table, and partial cell sets read HIGH (a 4-cell subset
    of the decoder ablation showed .6279 where all 10 cells gave .5991).
    """
    with open(path) as fh:
        board = json.load(fh)
    out: dict[str, dict[str, float]] = {}
    for key, blob in board.items():
        task = key.split("|", 1)[1] if "|" in key else key
        cs = blob.get("cs") or {}
        for tapnorm, cells in cs.items():
            tap, _, nm = tapnorm.partition("|")
            if nm != norm or not cells:
                continue
            out.setdefault(task, {})[tap] = float(np.mean(list(cells.values())))
    return out
