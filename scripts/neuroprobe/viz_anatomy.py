"""DKT-parcel x time anatomical attribution for the cross-subject taps.

Greg's test: speech/onset had better be superior temporal, or something is wrong. This
answers it on the parcel features the CS decoder actually consumes, per subject rather
than pooled, with a split-half UNBIASED effect size rather than a contrast amplitude.

Why the split-half dot product is the statistic
-----------------------------------------------
For the two independent trial halves h0, h1 of one (parcel, time):

    v_h  = (c1_h - c0_h) / chan_sd                     # standardized class contrast, (C,)
    d2   = <v_h0, v_h1> / C

Because the halves are independent, their noise terms are uncorrelated and vanish in
expectation, so ``E[d2] = ||true standardized contrast||^2 / C``. That gives d2 a TRUE
ZERO: a parcel carrying no class difference sits at 0 and is negative half the time.

The single-half magnitude ``||v_all||^2`` does not have that property -- it is biased
upward everywhere by noise, in proportion to how few trials a cell has. That bias is
exactly how a coverage-driven map gets mistaken for an anatomical one, which is the
failure mode this figure exists to rule out. ``d_cv = sign(d2) * sqrt(|d2|)`` is reported
so the colour axis is in units of a standardized contrast per channel.

``rel`` is the cosine between the two halves' contrast VECTORS -- a stricter replication
check than magnitude agreement, since it asks whether the contrast points the same way.

What is deliberately NOT done
-----------------------------
* **No template warp, no borrowed surface mesh.** Native ``depth-wm.csv`` (L, I, P) per
  subject; labels from native volume, never a warp. MNI is banned in this project and a
  fsaverage mesh here would be drawing somebody else's brain behind these electrodes.
* **No occipital control**, because there is NO occipital coverage in these 6 subjects
  (``lateraloccipital`` / ``lingual`` / ``cuneus`` / ``pericalcarine`` are all 0). The
  visual-control version of this figure is unrunnable, and the script asserts the zero
  rather than quietly omitting the panel. The control used instead is **task contrast on
  a fixed electrode set**: identical electrodes, 15 tasks, so if every task yields the
  same anatomical map the map is coverage rather than anatomy.
* **No pooled-only claim.** ``superiortemporal`` is asserted per subject; the pooled map
  can be carried by S2 alone (42 superiortemporal contacts of 119).

Hemispheres are pooled into a DKT base name. Without pooling the all-6-subject parcel
intersection is empty purely because S1/S10 are right-lateralised and S3/S4/S7 left.

Time axis: the visualization encode windows -0.5 -> +1.5 s around word onset
(``v3_probe_encode_r4.py`` ``_shift_starts``), T=64 bins => 31.25 ms/bin, onset at bin 16.
"""
from __future__ import annotations

import argparse
import base64
import collections
import glob
import io
import json
import os

import numpy as np

TASKS = (
    "onset", "speech", "delta_volume", "word_index", "word_head_pos", "word_length",
    "gpt2_surprisal", "word_gap", "word_part_speech",
    "volume", "pitch", "local_flow", "global_flow", "face_num", "frame_brightness",
)
EVENT = TASKS[:9]
LEVEL = TASKS[9:]
TAPS = ("enc0", "enc3", "enc6", "enc12")

WIN_START_S = -0.5
WIN_END_S = 1.5
ONSET_BIN_OF = lambda T: int(round(-WIN_START_S / ((WIN_END_S - WIN_START_S) / T)))

# The regions Greg's test is about, and the ones that make it falsifiable.
TARGET = "superiortemporal"
NEIGHBOURS = ("middletemporal", "transversetemporal")
# Marked in every figure that has a region axis, in ONE colour scheme so a reader who learns it
# on the heatmap can carry it to the render. Heschl's is the hotter of the two and sits directly
# above STG by alphabet alone, so marking only STG gets read off by one row.
MARK = {"transversetemporal": "#00a000", TARGET: "#8e24aa"}
OCCIPITAL = ("lateraloccipital", "lingual", "cuneus", "pericalcarine")

LOBE_ORDER = ("temporal", "insula", "frontal", "parietal", "cingulate", "mtl", "unknown")

# The BrainTreebank's OWN template projection, transcribed from its quickstart.ipynb (cell 31)
# rather than derived. ``elec_coords_full.csv`` holds a hemisphere-COLLAPSED 2D lateral
# projection: the ``X`` column is anterior(+)/posterior(-), ``Y`` is superior(+)/inferior(-),
# ``Z`` is the unused out-of-plane depth, and ``Hemisphere`` selects WHICH image to draw on
# (1 = left, 0 = right, per the notebook's ``hem_index = 1 if hemisphere=="left" else 0``).
# Confirmed against the data: no column sign-separates Hemisphere, and Y's range is exactly
# +-72.98 = MATLAB_YLIM.
#
# This is precisely why the same file is BANNED for pitch or any distance measurement -- a 2D
# projection of collapsed hemispheres has no metric meaning -- and fine for a figure.
MATLAB_XLIM = (-108.0278, 108.0278)
MATLAB_YLIM = (-72.9774, 72.9774)


# --------------------------------------------------------------------------------------
# atlas plumbing
# --------------------------------------------------------------------------------------
def dkt_tables():
    """``(base_of_id, lobe_of_base)`` for DKT, with the reserved unknown id appended.

    ``atlas_spec`` is the only sanctioned way to get the (column, labels) pair; ids run
    0..K-1 with K itself reserved for 'no atlas row', which the reductions do emit.
    """
    from speech_decoding.studies.braintreebank.anatomy import atlas_spec, parcel_lobe_keys

    col, labels = atlas_spec("dkt")
    assert col == "DKT" and len(labels) == 74, (col, len(labels))
    keys = parcel_lobe_keys()
    assert len(keys) == len(labels) + 1, (len(keys), len(labels))

    base_of, lobe_of_base = {}, {}
    for i, lab in enumerate(labels):
        base = lab.split("-", 2)[2] if lab.startswith(("ctx-lh-", "ctx-rh-")) else lab
        base_of[i] = base
        lobe = keys[i]
        lobe_of_base[base] = lobe if lobe == "unknown" else lobe.split("-", 1)[1]
    base_of[len(labels)] = "unknown"
    lobe_of_base["unknown"] = "unknown"
    return base_of, lobe_of_base


# --------------------------------------------------------------------------------------
# the statistic
# --------------------------------------------------------------------------------------
N_PRE = 16   # pre-stimulus frames in the 2 s window at 32 Hz (see center_per_session)


def base_features(sess, tap: str, task: str, base_of, *,
                  baseline: bool = True, common_mode: bool = True) -> dict[str, dict]:
    """Electrode-weighted per-DKT-base contrast -> ``{base: {"v": {half: (T,C)}, ...}}``.

    Two nuisances are removed BEFORE the halves are combined, and both are linear and
    applied to each half separately, so the split-half dot product stays unbiased -- just
    unbiased for the residual rather than the raw contrast.

    ``baseline`` subtracts each parcel's own mean over the first ``N_PRE`` (pre-stimulus)
    frames. Necessary because for ``speech`` the two classes differ before t=0 *by
    construction* (a word inside ongoing talk vs silence), so the raw contrast carries a
    large offset that is not a response to anything. The map then reads "change relative to
    the state at word onset", which is standard ERP baselining and has to be said out loud.

    ``common_mode`` subtracts the contact-count-weighted mean contrast ACROSS PARCELS at
    each (time, channel). A contrast shared by every contact in the head localises nothing,
    and Greg's question is explicitly comparative -- is it superior temporal *rather than*
    elsewhere. Without this every parcel lights up for every task and the map is unreadable
    (``invariant_zero`` fails loudly: 0.0% of speech cells negative).

    Pooling happens on the FEATURES weighted by contact count, and the contrast is formed
    afterwards, so a base's value is the contrast of its mean signal -- the quantity the
    parcel-pooled decoder actually sees, not an average of per-parcel effect sizes.
    """
    need = [(task, c, h) for c in (0, 1) for h in ("h0", "h1")]
    if any((tap, t, c, h) not in sess.cond for t, c, h in need):
        return {}
    sd = sess.chan_sd[tap]
    w_all = sess.counts.astype(np.float64)

    per_half = {}
    for h in ("h0", "h1"):
        v = (sess.cond[(tap, task, 1, h)] - sess.cond[(tap, task, 0, h)]) / sd  # (P,T,C)
        if baseline:
            v = v - v[:, :N_PRE, :].mean(axis=1, keepdims=True)
        if common_mode:
            gm = np.tensordot(w_all, v, axes=(0, 0)) / w_all.sum()               # (T,C)
            v = v - gm[None]
        per_half[h] = v

    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, pid in enumerate(sess.parcels):
        groups[base_of[int(pid)]].append(i)

    out: dict[str, dict] = {}
    for base, idx in groups.items():
        if base == "unknown":
            continue
        w = w_all[idx]
        out[base] = {
            "v": {h: np.tensordot(w, per_half[h][idx], axes=(0, 0)) / w.sum()
                  for h in ("h0", "h1")},
            "n_elec": int(w.sum()),
        }
    return out


def dcv_rel(v: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(d_cv, rel)`` per time bin from the two halves' contrast vectors.

    ``d2`` is unbiased for the squared true contrast because the halves are independent;
    ``rel`` is the cosine, which tests that the contrast points the same way in both.
    """
    a, b = v["h0"], v["h1"]                       # (T, C)
    C = a.shape[1]
    d2 = (a * b).sum(axis=1) / C
    d_cv = np.sign(d2) * np.sqrt(np.abs(d2))
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    rel = (a * b).sum(axis=1) / np.maximum(na * nb, 1e-12)
    return d_cv, rel


def compute(red_dir: str, *, baseline: bool = True, common_mode: bool = True):
    """``(D, cov, T, base_of, lobe_of_base)`` where ``D[tap][task][subj][base] = (d_cv, rel)``.

    Sessions of one subject are averaged, not concatenated: two trials share a montage and
    would otherwise double-count as agreement.
    """
    # Imported HERE, not at module level, so this file's CONSTANTS and ``dkt_tables`` stay
    # importable in an environment without the study stack. ``viz_common`` pulls in
    # ``speech_decoding.studies.braintreebank``, whose package __init__ imports mne, which the
    # cluster's pytorch-conda module does not have -- and ``viz_elec_auroc`` needs the task
    # menu from this file while running there.
    from scripts.neuroprobe.viz_common import load_all

    base_of, lobe_of_base = dkt_tables()
    sessions = load_all(red_dir, pool_hemi=True)
    assert sessions, red_dir

    T: int | None = None
    acc: dict = {}
    cov: dict[int, dict[str, int]] = collections.defaultdict(dict)
    for sess in sessions:
        for tap in TAPS:
            if tap not in sess.shapes:
                continue
            if T is None:
                T = int(sess.shapes[tap][1])
            assert int(sess.shapes[tap][1]) == T, (tap, sess.key, sess.shapes[tap], T)
            for task in TASKS:
                bf = base_features(sess, tap, task, base_of,
                                   baseline=baseline, common_mode=common_mode)
                for base, rec in bf.items():
                    d_cv, rel = dcv_rel(rec["v"])
                    acc.setdefault(tap, {}).setdefault(task, {}).setdefault(
                        sess.subject_id, {}).setdefault(base, []).append((d_cv, rel))
                    cov[sess.subject_id][base] = max(
                        cov[sess.subject_id].get(base, 0), rec["n_elec"])

    D: dict = {}
    for tap, per_task in acc.items():
        for task, per_subj in per_task.items():
            for subj, per_base in per_subj.items():
                for base, runs in per_base.items():
                    d = np.mean([r[0] for r in runs], axis=0)
                    r = np.mean([r[1] for r in runs], axis=0)
                    D.setdefault(tap, {}).setdefault(task, {}).setdefault(
                        subj, {})[base] = (d, r)
    assert T is not None, f"no tap carried a shape in {red_dir}"
    return D, dict(cov), T, base_of, lobe_of_base


# --------------------------------------------------------------------------------------
# asserted invariants
# --------------------------------------------------------------------------------------
def gate(cov: dict, lobe_of_base: dict) -> list[str]:
    """Print the coverage gate and assert what the figures are allowed to claim."""
    subs = sorted(cov)
    print(f"[check] {len(subs)} subjects: {subs}")

    st = {s: cov[s].get(TARGET, 0) for s in subs}
    print(f"[check] {TARGET} contacts per subject: "
          + ", ".join(f"S{s}={st[s]}" for s in subs))
    assert all(v >= 5 for v in st.values()), f"{TARGET} too sparse to test per subject: {st}"
    print(f"[check] OK {TARGET} present in {len(subs)}/{len(subs)} subjects, min {min(st.values())}")

    occ = {o: {s: cov[s].get(o, 0) for s in subs} for o in OCCIPITAL}
    tot = sum(sum(v.values()) for v in occ.values())
    assert tot == 0, f"occipital coverage exists after all -- restore the visual control: {occ}"
    print(f"[check] OK occipital coverage is ZERO ({', '.join(OCCIPITAL)}) "
          f"=> the visual control is unrunnable, task-contrast control used instead")

    for m in (1, 2, 5):
        inter = sorted(set.intersection(*[
            {b for b, n in cov[s].items() if n >= m and b != "unknown"} for s in subs]))
        print(f"[check] DKT bases with >={m} contacts in EVERY subject ({len(inter)}): {inter}")

    lob = {s: collections.Counter() for s in subs}
    for s in subs:
        for b, n in cov[s].items():
            lob[s][lobe_of_base.get(b, "unknown")] += n
    shared_lobes = [L for L in LOBE_ORDER
                    if L != "unknown" and all(lob[s].get(L, 0) >= 2 for s in subs)]
    print(f"[check] lobes with >=2 contacts in every subject: {shared_lobes}")
    if "parietal" not in shared_lobes:
        miss = [s for s in subs if lob[s].get("parietal", 0) == 0]
        print(f"[check] NO cohort-wide parietal coverage (absent in S{miss}) "
              f"=> a 'superiortemporal -> parietal' propagation claim is NOT supportable")

    # rank bases by how many subjects have them, for the figure's row order
    n_by_base = collections.Counter()
    for s in subs:
        for b, n in cov[s].items():
            if b != "unknown" and n >= 1:
                n_by_base[b] += 1
    rows = [b for b, c in n_by_base.items() if c >= 3]
    rows.sort(key=lambda b: (LOBE_ORDER.index(lobe_of_base.get(b, "unknown"))
                             if lobe_of_base.get(b, "unknown") in LOBE_ORDER else 99, b))
    print(f"[check] {len(rows)} DKT bases present in >=3 subjects -> figure rows")
    return rows


# Regions a speech/onset response is allowed to peak in without that being a red flag.
# Reported, never asserted on -- the assert is on superiortemporal's RANK, because
# "which of 20 bases is the single argmax" is arbitrary at this noise level.
SPEECH_NET = (
    "superiortemporal", "middletemporal", "transversetemporal", "supramarginal",
    "postcentral", "precentral", "parsopercularis", "parstriangularis", "insula",
)
VISUAL_TASKS = ("face_num", "frame_brightness", "local_flow", "global_flow")


def invariant_st(D: dict, cov: dict, T: int, tap: str = "enc12", top_k: int = 3) -> dict:
    """Greg's test, in its falsifiable form, per subject rather than pooled.

    The claim is NOT "superiortemporal is the single argmax of 20 DKT bases" -- with this
    many candidates and this many trials that is arbitrary, and its failures are not even
    failures (S1's onset peak is ``postcentral``, S7's is ``parstriangularis``; both are
    speech-network). The claim asserted is:

      1. ``superiortemporal`` ranks in the top ``top_k`` bases for speech and onset, in a
         majority of subjects INDIVIDUALLY.
      2. It does NOT do that for the visual tasks. This is the falsifier that replaces the
         dead occipital control: with no occipital coverage we cannot show visual
         information IS occipital, but we can still show it is NOT superior temporal, and
         that is what makes the speech result mean something rather than being coverage.

    The peak is taken over POST-onset bins only; pre-onset is the baseline reference.
    """
    subs = sorted(cov)
    onset_bin = ONSET_BIN_OF(T)
    out = {}
    print(f"\n[check] --- {TARGET} rank per subject at {tap} "
          f"(peak d_cv over t>=0, among that subject's DKT bases) ---")
    for task in TASKS:
        per = D.get(tap, {}).get(task, {})
        argmax, rank, in_net = {}, {}, 0
        for s in subs:
            bases = {b: float(v[0][onset_bin:].max()) for b, v in per.get(s, {}).items()}
            if not bases:
                continue
            order = sorted(bases, key=lambda b: -bases[b])
            argmax[s] = order[0]
            if TARGET in order:
                rank[s] = order.index(TARGET) + 1
            if order[0] in SPEECH_NET:
                in_net += 1
        n_top = sum(1 for s in subs if rank.get(s, 99) <= top_k)
        out[task] = {"argmax": argmax, "rank": rank, "n_top": n_top,
                     "n_subj": len(argmax), "speech_net": in_net}
        flag = "  <<< Greg's test" if task in ("speech", "onset") else (
            "  <-- falsifier" if task in VISUAL_TASKS else "")
        print(f"  {task:<18} top{top_k} in {n_top}/{len(argmax)} | rank "
              + "/".join(f"{rank.get(s, 0):>2}" for s in subs)
              + f" | peak in speech-net {in_net}/{len(argmax)}" + flag)

    for task in ("speech", "onset"):
        r = out[task]
        assert r["n_top"] > r["n_subj"] / 2, (
            f"GREG'S TEST FAILS for {task}: {TARGET} is top-{top_k} in only "
            f"{r['n_top']}/{r['n_subj']} subjects. ranks = {r['rank']}, "
            f"argmax = {r['argmax']}")
        print(f"[check] OK Greg's test passes for {task}: {TARGET} is top-{top_k} in "
              f"{r['n_top']}/{r['n_subj']} subjects individually; peak lands in the speech "
              f"network in {r['speech_net']}/{r['n_subj']}")

    # The across-subject headline: which bases actually top the map, averaged over the
    # subjects that have them (n>=2, so a single thin parcel cannot carry a row).
    print(f"\n[check] --- top DKT bases by across-subject mean peak d_cv (t>=0, n>=2) ---")
    for task in TASKS:
        per = D.get(tap, {}).get(task, {})
        acc: dict[str, list[float]] = collections.defaultdict(list)
        for s in subs:
            for b_, (d, _) in per.get(s, {}).items():
                acc[b_].append(float(d[onset_bin:].max()))
        m = {b_: float(np.mean(v)) for b_, v in acc.items() if len(v) >= 2}
        top = sorted(m, key=lambda b_: -m[b_])[:4]
        print(f"  {task:<18} " + "  ".join(
            f"{b_}={m[b_]:.3f}(n={len(acc[b_])})" for b_ in top))

    worst_speech = min(out[t]["n_top"] for t in ("speech", "onset"))
    best_visual = max(out[t]["n_top"] for t in VISUAL_TASKS if t in out)
    assert best_visual < worst_speech, (
        f"DISSOCIATION FAILS: a visual task puts {TARGET} in the top-{top_k} for "
        f"{best_visual} subjects vs {worst_speech} for speech/onset. Without this the map "
        f"could be coverage rather than anatomy.")
    print(f"[check] OK dissociation holds: {TARGET} is top-{top_k} for speech/onset in "
          f">={worst_speech}/6 subjects but for the best VISUAL task in only "
          f"{best_visual}/6 => the temporal result is anatomy, not coverage")

    # POSITIVE control that survives the missing occipital coverage: face information should
    # still land in a face-selective region. fusiform (the fusiform face area) is covered in
    # 4 subjects, so this is checkable even though V1/V2 are not.
    def _mean_peak(task):
        per_ = D.get(tap, {}).get(task, {})
        acc_: dict[str, list[float]] = collections.defaultdict(list)
        for s in subs:
            for b_, (d, _) in per_.get(s, {}).items():
                acc_[b_].append(float(d[onset_bin:].max()))
        return {b_: float(np.mean(v)) for b_, v in acc_.items() if len(v) >= 2}

    fm = _mean_peak("face_num")
    if "fusiform" in fm and TARGET in fm:
        assert fm["fusiform"] > fm[TARGET], (
            f"face_num should not be led by {TARGET}: fusiform {fm['fusiform']:.4f} "
            f"vs {TARGET} {fm[TARGET]:.4f}")
        print(f"[check] OK positive control: face_num is led by FUSIFORM "
              f"({fm['fusiform']:.3f}) over {TARGET} ({fm[TARGET]:.3f}) -- the fusiform face "
              f"area, checkable even with zero occipital coverage")
    return out


def invariant_zero(D: dict, tap: str = "enc12") -> None:
    """d_cv has a true zero, so a chance task must be ~50% negative and speech must not be."""
    print(f"\n[check] --- the statistic's zero point at {tap} ---")
    frac = {}
    for task in TASKS:
        vals = np.concatenate([v[0] for per in D.get(tap, {}).get(task, {}).values()
                               for v in per.values()]) if D.get(tap, {}).get(task) else None
        if vals is None or vals.size == 0:
            continue
        frac[task] = float((vals < 0).mean())
    for t in sorted(frac, key=lambda t: -frac[t]):
        print(f"  {t:<18} {frac[t]*100:5.1f}% of (base,time) cells negative")
    assert frac["face_num"] > 0.25, (
        f"face_num sits at chance (enc0 .503) so its d_cv should straddle zero; "
        f"only {frac['face_num']*100:.1f}% negative => the statistic is biased upward")
    assert frac["speech"] < frac["face_num"], (
        f"speech ({frac['speech']*100:.1f}%) should be far less negative than "
        f"face_num ({frac['face_num']*100:.1f}%)")
    print(f"[check] OK zero point behaves: face_num {frac['face_num']*100:.1f}% negative "
          f"vs speech {frac['speech']*100:.1f}%")


def task_profiles(D: dict, rows: list[str], tap: str, T: int) -> tuple[np.ndarray, list[str]]:
    """``(P, tasks)`` with ``P[i]`` task i's flattened (base x time) across-subject profile."""
    prof, keep = [], []
    for task in TASKS:
        per = D.get(tap, {}).get(task, {})
        if not per:
            continue
        M = []
        for b in rows:
            vals = [per[s][b][0] for s in per if b in per[s]]
            M.append(np.mean(vals, axis=0) if vals else np.full(T, np.nan))
        prof.append(np.concatenate(M))
        keep.append(task)
    return np.asarray(prof), keep


def invariant_task_contrast(D: dict, rows: list[str], T: int,
                           tap: str = "enc12") -> np.ndarray:
    """The control: identical electrodes, 15 tasks. If every map agrees, it is coverage."""
    P, keep = task_profiles(D, rows, tap, T)
    ok = ~np.isnan(P).any(axis=0)
    X = P[:, ok]
    X = X - X.mean(axis=1, keepdims=True)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    R = X @ X.T
    off = R[~np.eye(len(keep), dtype=bool)]
    print(f"\n[check] --- task-contrast control at {tap} ({len(keep)} tasks, "
          f"{ok.sum()} base x time cells) ---")
    print(f"[check] off-diagonal profile r: mean {off.mean():+.3f} "
          f"median {np.median(off):+.3f} max {off.max():+.3f}")
    assert off.mean() < 0.9, (
        f"every task produces the same anatomical map (mean r {off.mean():.3f}) "
        f"=> the map is COVERAGE, not anatomy")
    print(f"[check] OK maps are task-specific (mean off-diagonal r {off.mean():+.3f} < 0.9) "
          f"=> not a coverage artifact")
    return R


# --------------------------------------------------------------------------------------
# coordinates, for the renders
# --------------------------------------------------------------------------------------
def load_template_coords(red_dir: str, bt_root: str) -> dict:
    """Contacts projected into the BrainTreebank's own two hemisphere renders.

    Returns ``{"img": {"left": arr, "right": arr},
               "pts": {(subj,trial): {"xy": (n,2) px, "side": (n,) 'left'/'right',
                                      "pid": (n,), "found": (n,) bool}}}``.

    The join key is the electrode LABEL from ``lite_voltage_order``, and alignment is TESTED,
    not assumed: the DKT tag re-derived for that order must equal the reduction's own
    ``parcel_canon``. A permuted electrode axis would paint real effects onto the wrong dots,
    which is worse than no render at all. Contacts absent from the CSV are marked
    ``found=False`` and reported rather than silently dropped.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from scripts.neuroprobe.viz_coords_dump import make_parcel_fn
    from speech_decoding.studies.braintreebank.anatomy import lite_voltage_order

    loc = os.path.join(bt_root, "localization")
    img = {"left": plt.imread(os.path.join(loc, "left_hem_clean.png")),
           "right": plt.imread(os.path.join(loc, "right_hem_clean.png"))}
    d = pd.read_csv(os.path.join(loc, "elec_coords_full.csv"))

    # the notebook's scaling, verbatim. Note it uses y_scale_r for the RIGHT panel's x axis;
    # that is the published convention (and only 1.2% off x_scale), so it is reproduced rather
    # than "fixed" -- the point is to land where the dataset's own figures land.
    xs = img["left"].shape[1] / (MATLAB_XLIM[1] - MATLAB_XLIM[0])
    ysl = img["left"].shape[0] / (MATLAB_YLIM[1] - MATLAB_YLIM[0])
    ysr = img["right"].shape[0] / (MATLAB_YLIM[1] - MATLAB_YLIM[0])

    lut = {(r.Subject, r.Electrode): (r.X, r.Y, int(r.Hemisphere)) for r in d.itertuples()}
    parcel_fn = make_parcel_fn(bt_root, atlas="dkt")
    pts, n_miss, n_tot = {}, 0, 0
    for p in sorted(glob.glob(os.path.join(red_dir, "red_s*_t*_*.npz"))):
        z = np.load(p, allow_pickle=False)
        subj, trial = int(z["subject_id"]), int(z["trial_id"])
        canon = z["parcel_canon"]
        order = list(lite_voltage_order(bt_root, subj, trial))
        if len(order) != len(canon):
            print(f"[tmpl] SKIP S{subj}T{trial}: lite order {len(order)} != canon {len(canon)}")
            continue
        if not np.array_equal(parcel_fn(subj, trial, order), canon):
            print(f"[tmpl] SKIP S{subj}T{trial}: DKT tags disagree with parcel_canon")
            continue
        xy = np.full((len(order), 2), np.nan)
        side = np.array(["left"] * len(order), dtype=object)
        found = np.zeros(len(order), dtype=bool)
        for i, e in enumerate(order):
            hit = lut.get((f"sub_{subj}", e))
            if hit is None:
                continue
            X, Y, hemi = hit
            found[i] = True
            if hemi == 1:                                   # left panel
                xy[i] = (-(X - MATLAB_XLIM[1]) * xs, -(Y - MATLAB_YLIM[1]) * ysl)
                side[i] = "left"
            else:                                           # right panel
                xy[i] = ((X + MATLAB_XLIM[1]) * ysr, -(Y - MATLAB_YLIM[1]) * ysr)
                side[i] = "right"
        n_miss += int((~found).sum()); n_tot += len(order)
        pts[(subj, trial)] = {"xy": xy, "side": side, "pid": canon, "found": found}
        print(f"[tmpl] OK S{subj}T{trial}: {int(found.sum())}/{len(order)} contacts projected")

    # Bounds are necessary but nowhere near sufficient -- a mirrored or transposed projection
    # stays in bounds. The anatomical checks are in invariant_projection().
    allxy = np.concatenate([v["xy"][v["found"]] for v in pts.values()])
    H = max(img["left"].shape[0], img["right"].shape[0])
    W = img["left"].shape[1]
    inb = ((allxy[:, 0] >= 0) & (allxy[:, 0] <= W)
           & (allxy[:, 1] >= 0) & (allxy[:, 1] <= H)).mean()
    print(f"[tmpl] {n_tot - n_miss}/{n_tot} contacts have coordinates "
          f"({n_miss} absent from the CSV, reported not dropped)")
    print(f"[tmpl] {100 * inb:.1f}% of projected contacts land inside the image bounds")
    assert inb > 0.99, f"projection puts {100 * (1 - inb):.1f}% of contacts off-image"
    return {"img": img, "pts": pts}


def invariant_projection(tmpl: dict, base_of: dict, lobe_of_base: dict) -> None:
    """Assert the projection is not mirrored or transposed, using ANATOMY, not a fit.

    Both expected directions follow from the transform itself, so they are predictions and not
    observations dressed up as ones. ``y_px = (YLIM_max - Y) * y_scale`` with ``Y`` superior(+),
    so INFERIOR structures get a LARGER y_px; temporal cortex is inferior to frontal, and sits
    below the lateral view's vertical midline. For x, the left panel uses
    ``(XLIM_max - X) * s`` and the right panel ``(X + XLIM_max) * s`` with ``X`` anterior(+), so
    ANTERIOR is a SMALLER x_px on the left panel and a LARGER x_px on the right -- opposite
    signs, which is exactly the asymmetry a copy-paste mirror bug would break.
    """
    H = max(tmpl["img"][s].shape[0] for s in ("left", "right"))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for v in tmpl["pts"].values():
        for i in np.where(v["found"])[0]:
            lobe = lobe_of_base.get(base_of.get(int(v["pid"][i]), "unknown"), "unknown")
            by[v["side"][i]][lobe].append(v["xy"][i])

    for sd in ("left", "right"):
        med = {lb: np.median(np.array(a), axis=0) for lb, a in by[sd].items() if len(a) >= 10}
        if not {"temporal", "frontal"} <= set(med):
            print(f"[tmpl] {sd}: too few contacts to check ({sorted(med)}) -- skipped")
            continue
        ty, fy = med["temporal"][1], med["frontal"][1]
        print(f"[tmpl] {sd}: median y_px temporal {ty:.0f} vs frontal {fy:.0f} "
              f"(midline {H / 2:.0f}) -- temporal must be larger and below midline")
        assert ty > fy, f"{sd}: temporal is ABOVE frontal ({ty:.0f} vs {fy:.0f}) -- y flipped?"
        assert ty > H / 2, f"{sd}: temporal sits above the midline ({ty:.0f} < {H / 2:.0f})"
        if "parietal" in med:
            fx, px = med["frontal"][0], med["parietal"][0]
            ant_is_small = sd == "left"
            print(f"[tmpl] {sd}: median x_px frontal {fx:.0f} vs parietal {px:.0f} -- frontal "
                  f"must be {'smaller' if ant_is_small else 'larger'}")
            assert (fx < px) == ant_is_small, (
                f"{sd}: anterior/posterior axis is reversed (frontal {fx:.0f}, "
                f"parietal {px:.0f}) -- the two panels' x transforms have opposite sign")
    print("[check] OK template projection is upright and correctly handed on both panels")


def fig_template_qc(tmpl: dict, base_of: dict, lobe_of_base: dict, out: str) -> None:
    """Colour every contact by LOBE on the two hemispheres. This is the check that the
    projection is anatomically right, and it is a figure because the eye is the only
    instrument that can confirm 'these dots are on the temporal lobe'."""
    import matplotlib.pyplot as plt

    colour = {"temporal": "#d62728", "frontal": "#1f77b4", "parietal": "#2ca02c",
              "insula": "#9467bd", "cingulate": "#8c564b", "mtl": "#e377c2",
              "unknown": "#999999"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150)
    for ax, sd in zip(axes, ("left", "right")):
        ax.imshow(tmpl["img"][sd]); ax.axis("off"); ax.set_title(f"{sd} hemisphere", fontsize=10)
        for lobe, c in colour.items():
            xs, ys = [], []
            for v in tmpl["pts"].values():
                m = v["found"] & (v["side"] == sd)
                for i in np.where(m)[0]:
                    if lobe_of_base.get(base_of.get(int(v["pid"][i]), "unknown"),
                                        "unknown") == lobe:
                        xs.append(v["xy"][i, 0]); ys.append(v["xy"][i, 1])
            if xs:
                ax.scatter(xs, ys, s=15, c=c, ec="k", lw=.3, label=f"{lobe} ({len(xs)})")
        ax.legend(fontsize=6, loc="lower left", framealpha=.85)
    fig.suptitle("PROJECTION QC: contacts by lobe, on the BrainTreebank's own hemisphere "
                 "renders.\nTemporal (red) must lie on the temporal lobe or the projection "
                 "is wrong.", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, "figAN5_template_qc.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------
def _t_ms(T: int) -> np.ndarray:
    step = (WIN_END_S - WIN_START_S) / T * 1000.0
    return WIN_START_S * 1000.0 + (np.arange(T) + 0.5) * step


def _mean_map(D, tap, task, rows, subs, T):
    M, n = [], []
    for b in rows:
        per = D.get(tap, {}).get(task, {})
        vals = [per[s][b][0] for s in subs if b in per.get(s, {})]
        M.append(np.mean(vals, axis=0) if vals else np.full(T, np.nan))
        n.append(len(vals))
    return np.asarray(M), n


def fig_coverage(cov, rows, out):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    subs = sorted(cov)
    M = np.asarray([[cov[s].get(b, 0) for b in rows] for s in subs], dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, 0.34 * len(rows)), 3.1), dpi=170)
    im = ax.imshow(np.where(M > 0, M, np.nan), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(rows, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(subs)))
    ax.set_yticklabels([f"S{s}" for s in subs], fontsize=7)
    for i in range(len(subs)):
        for j in range(len(rows)):
            if M[i, j] > 0:
                ax.text(j, i, str(int(M[i, j])), ha="center", va="center", fontsize=4.5,
                        color="w" if M[i, j] < M.max() * 0.6 else "k")
    if TARGET in rows:
        j = rows.index(TARGET)
        ax.add_patch(Rectangle((j - .5, -.5), 1, len(subs), fill=False,
                               ec="#d62728", lw=1.6))
    ax.set_title(f"DKT contact coverage (hemisphere-pooled). {TARGET} boxed: "
                 f"present in {len(subs)}/{len(subs)} subjects.\n"
                 f"No occipital column exists -- zero coverage in every subject.",
                 fontsize=7.5)
    fig.colorbar(im, ax=ax, label="contacts", pad=0.01)
    fig.tight_layout()
    p = os.path.join(out, "figAN0_coverage.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


def fig_dkt_time(D, rows, cov, T, out, tap):
    import matplotlib.pyplot as plt

    subs = sorted(cov)
    maps = {t: _mean_map(D, tap, t, rows, subs, T) for t in TASKS}
    finite = np.concatenate([m[np.isfinite(m)].ravel() for m, _ in maps.values()])
    lim = float(np.nanpercentile(np.abs(finite), 99))
    tms = _t_ms(T)

    im = None
    # Both auditory rows are marked, not just STG. transversetemporal (Heschl's) sits directly
    # ABOVE superiortemporal by alphabet alone, so one line next to two adjacent rows is read
    # off by one -- and the hotter of the two is Heschl's.
    fig, axes = plt.subplots(5, 3, figsize=(16.5, 15.0), dpi=165, sharex=True, sharey=True)
    for k, (ax, task) in enumerate(zip(axes.ravel(), TASKS)):
        M, n = maps[task]
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[tms[0], tms[-1], len(rows) - .5, -.5],
                       interpolation="nearest")
        ax.axvline(0, color="k", lw=.7, ls=":")
        # Only the leftmost column shows labels (the rows are identical in every panel), but
        # hiding them is `labelleft=False`, NOT set_yticklabels([]): these axes are sharey, so
        # an empty label list REPLACES THE SHARED FORMATTER and blanks the left column too.
        # That -- not the font size -- is why this figure previously had no readable y axis.
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{b} ({c})" for b, c in zip(rows, n)], fontsize=8.0)
        ax.tick_params(labelsize=8, labelleft=(k % 3 == 0))
        for lab in ax.get_yticklabels():
            if lab.get_text().split(" (")[0] in MARK:
                lab.set_color(MARK[lab.get_text().split(" (")[0]])
                lab.set_fontweight("bold")
        fam = "event" if task in EVENT else "level"
        ax.set_title(f"{task}  [{fam}]", fontsize=8,
                     color="#1f4e79" if fam == "event" else "#d98324")
        # marker OUTSIDE the left spine, not a line across the row: an axhline at the row
        # centre paints over the very cells the marker exists to point at.
        for base, c in MARK.items():
            if base in rows:
                ax.plot([tms[0]], [rows.index(base)], marker=">", ms=6, color=c,
                        clip_on=False, zorder=5)
    for ax in axes[-1]:
        ax.set_xlabel("time from word onset (ms)", fontsize=7)
    fig.suptitle(
        f"Split-half unbiased standardized class contrast (d_cv) by DKT base x time — {tap}\n"
        f"across-subject mean; (n) = subjects with that base; GREEN marker = transversetemporal "
        f"(Heschl's gyrus / A1) · PURPLE marker = superiortemporal; "
        f"zero is a TRUE zero, so blue is genuinely no-effect",
        fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes, label="d_cv (standardized contrast / channel)",
                     fraction=.02, pad=.01)
    p = os.path.join(out, f"figAN1_dkt_time_{tap}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


def fig_st_invariant(D, cov, T, out, tap="enc12"):
    import matplotlib.pyplot as plt

    subs = sorted(cov)
    tms = _t_ms(T)
    # NOT sharex: the top row's axis is time in ms, the bottom row's is d_cv. Sharing them
    # silently squashes every bar to invisibility against a -500..1500 range.
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), dpi=170)
    for col, task in enumerate(("onset", "speech", "pitch")):
        per = D.get(tap, {}).get(task, {})
        ax = axes[0, col]
        for s in subs:
            if TARGET in per.get(s, {}):
                ax.plot(tms, per[s][TARGET][0], lw=1.3, label=f"S{s}")
        ax.axhline(0, color="k", lw=.7)
        ax.axvline(0, color="k", lw=.7, ls=":")
        ax.set_title(f"{task} — {TARGET}, per subject ({tap})", fontsize=8)
        ax.legend(fontsize=5.5, ncol=3, frameon=False)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("time from word onset (ms)", fontsize=6.5)

        ax = axes[1, col]
        # MEAN across the subjects that have the base, not max: a max over subjects is a
        # selection statistic and would rank a 2-contact parcel in one subject above a
        # region measured in all six. n is annotated so a thin base cannot pass as robust.
        onset_bin = ONSET_BIN_OF(T)
        acc: dict[str, list[float]] = collections.defaultdict(list)
        for s in subs:
            for b_, (d, _) in per.get(s, {}).items():
                acc[b_].append(float(d[onset_bin:].max()))
        peak = {b_: float(np.mean(v)) for b_, v in acc.items() if len(v) >= 2}
        nsub = {b_: len(v) for b_, v in acc.items()}
        top = sorted(peak, key=lambda b_: -peak[b_])[:10]
        colr = ["#d62728" if b_ == TARGET else
                ("#ff9896" if b_ in NEIGHBOURS else "#7f7f7f") for b_ in top]
        vals = [peak[b_] for b_ in top]
        ax.barh(range(len(top)), vals, color=colr)
        for i, (v, b_) in enumerate(zip(vals, top)):
            ax.text(v, i, f" {v:.3f} (n={nsub[b_]})", va="center", fontsize=4.6, color="#333")
        ax.set_xlim(min(0, min(vals) * 1.15), max(vals) * 1.40)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top, fontsize=5.5)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=.7)
        ax.set_xlabel("mean peak d_cv over t>=0 (across subjects with the base)",
                      fontsize=6.5)
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Greg's test, per subject and not pooled — {tap}. Red = {TARGET}, "
                 f"pink = middle/transverse temporal.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, .95))
    p = os.path.join(out, f"figAN2_st_invariant_{tap}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


def fig_task_similarity(R, keep, out, tap="enc12"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=175)
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(keep, rotation=90, fontsize=6)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(keep, fontsize=6)
    for i, t in enumerate(keep):
        c = "#1f4e79" if t in EVENT else "#d98324"
        ax.get_xticklabels()[i].set_color(c)
        ax.get_yticklabels()[i].set_color(c)
    off = R[~np.eye(len(keep), dtype=bool)]
    ax.set_title(f"THE CONTROL: same electrodes, 15 tasks — {tap}\n"
                 f"anatomical-profile correlation. mean off-diagonal r = {off.mean():+.3f}.\n"
                 f"If this were all ~1 the map would be coverage, not anatomy.", fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.02, label="profile r")
    fig.tight_layout()
    p = os.path.join(out, f"figAN3_task_similarity_{tap}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


def fig_depth(D, rows, cov, T, out):
    import matplotlib.pyplot as plt

    subs = sorted(cov)
    tasks = ("onset", "speech", "pitch")
    fig, axes = plt.subplots(len(tasks), len(TAPS), figsize=(15, 3.1 * len(tasks)),
                             dpi=165, sharex=True)
    allv = np.concatenate([_mean_map(D, tp, t, rows, subs, T)[0].ravel()
                           for t in tasks for tp in TAPS])
    lim = float(np.nanpercentile(np.abs(allv[np.isfinite(allv)]), 99))
    tms = _t_ms(T)
    im = None
    for i, task in enumerate(tasks):
        for j, tap in enumerate(TAPS):
            M, _ = _mean_map(D, tap, task, rows, subs, T)
            ax = axes[i, j]
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                           extent=[tms[0], tms[-1], len(rows) - .5, -.5],
                           interpolation="nearest")
            ax.axvline(0, color="k", lw=.7, ls=":")
            if TARGET in rows:
                ax.axhline(rows.index(TARGET), color="#d62728", lw=.8, alpha=.65)
            if j == 0:
                ax.set_yticks(range(len(rows)))
                ax.set_yticklabels(rows, fontsize=4.2)
                ax.set_ylabel(task, fontsize=9)
            else:
                ax.set_yticks([])
            if i == 0:
                ax.set_title(tap, fontsize=9)
            ax.tick_params(labelsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("time from word onset (ms)", fontsize=7)
    fig.suptitle("Does the anatomy move with depth? Same rows, same colour scale, "
                 "enc0 -> enc12.", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=axes, label="d_cv", fraction=.015, pad=.01)
    p = os.path.join(out, "figAN4_depth.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


# --------------------------------------------------------------------------------------
# native-cloud renders + scrub demo
# --------------------------------------------------------------------------------------
def _pooled_contacts(tmpl: dict, base_of: dict) -> dict:
    """Contacts of ONE trial per subject, flattened, plus the (subject, base) group table.

    Two trials of the same subject are the same montage, so pooling both would double-weight
    that subject. Values live per (subject, DKT base), so the render only needs a group index
    per contact -- which is what keeps the demo payload small enough to be a single file.
    """
    pick = {}
    for key in sorted(tmpl["pts"]):
        pick.setdefault(key[0], key)

    groups, gmeta, pts = {}, [], []
    for s, key in sorted(pick.items()):
        v = tmpl["pts"][key]
        for i in np.where(v["found"])[0]:
            b = base_of.get(int(v["pid"][i]), "unknown")
            g = groups.setdefault((s, b), len(gmeta))
            if g == len(gmeta):
                gmeta.append((s, b))
            pts.append({"x": float(v["xy"][i, 0]), "y": float(v["xy"][i, 1]),
                        "side": v["side"][i], "subj": s, "base": b, "g": g})
    return {"pts": pts, "gmeta": gmeta}


def fig_template_render(D, tmpl, base_of, T, out, tap, task, times_ms=(-250, 0, 250, 500)):
    """Static paper version of the demo: hemispheres x a few time points, one tap/task."""
    import matplotlib.pyplot as plt

    per = D.get(tap, {}).get(task, {})
    if not per or not tmpl["pts"]:
        print(f"[fig] skip render {tap}|{task}: no data")
        return
    vals = np.concatenate([v[0] for s in per for v in per[s].values()])
    lim = float(np.nanpercentile(np.abs(vals), 98)) or 1.0
    tms = _t_ms(T)
    bins = [int(np.argmin(np.abs(tms - t))) for t in times_ms]
    P = _pooled_contacts(tmpl, base_of)["pts"]

    sc = None
    fig, axes = plt.subplots(2, len(bins), figsize=(3.6 * len(bins), 6.0), dpi=170)
    for r, sd in enumerate(("left", "right")):
        rowp = [q for q in P if q["side"] == sd]
        xs = np.array([q["x"] for q in rowp]); ys = np.array([q["y"] for q in rowp])
        # the mark is the dot's EDGE COLOUR, not a ring drawn over it. Contacts in STG sit
        # centimetres apart on this projection, so overlaid rings merge into one purple blob
        # and hide the values the marker exists to draw the eye to.
        ec = np.array([MARK.get(q["base"], "#000000") for q in rowp])
        lw = np.array([0.7 if q["base"] in MARK else 0.2 for q in rowp])
        for c, fi in enumerate(bins):
            ax = axes[r, c]
            ax.imshow(tmpl["img"][sd]); ax.axis("off")
            v = np.array([per.get(q["subj"], {}).get(q["base"], [np.full(T, np.nan)])[0][fi]
                          for q in rowp], dtype=float)
            miss = ~np.isfinite(v)
            if miss.any():
                ax.scatter(xs[miss], ys[miss], s=9, c="#c8c8c8",
                           ec=ec[miss], lw=lw[miss], zorder=3)
            if (~miss).any():
                sc = ax.scatter(xs[~miss], ys[~miss], c=v[~miss], cmap="RdBu_r",
                                vmin=-lim, vmax=lim, s=22,
                                ec=ec[~miss], lw=lw[~miss], zorder=4)
            if r == 0:
                ax.set_title(f"{tms[fi]:+.0f} ms", fontsize=9)
            if c == 0:
                ax.text(-.02, .5, f"{sd} hemi", rotation=90, va="center", ha="right",
                        transform=ax.transAxes, fontsize=8)
    fig.suptitle(f"d_cv on the BrainTreebank lateral projection — {task}, {tap}, 6 subjects "
                 f"pooled\ngreen outline = transversetemporal (Heschl's / A1) · "
                 f"purple outline = superiortemporal · grey fill = parcel absent in that subject",
                 fontsize=9)
    if sc is not None:
        fig.colorbar(sc, ax=axes, fraction=.018, pad=.01, label="d_cv")
    p = os.path.join(out, f"figAN6_render_{tap}_{task}.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {p}")


def build_demo(D, tmpl, base_of, T, out, tasks, taps):
    """Interactive canvas demo: the brain ships ONCE, the contacts are drawn live in JS.

    The earlier version pre-rendered a PNG per time bin -- 67 MB for 24 frames, and a picture
    cannot be interrogated. Sending coordinates + a per-(subject, base) value table instead is
    ~20x smaller, gives every one of the 64 bins instead of 24, and lets the reader hover a
    contact to be told which subject and which DKT parcel it is. Which was the actual
    complaint: not that the dots were wrong, but that a dot you cannot name is not evidence.
    """
    import matplotlib as mpl

    if not tmpl["pts"]:
        print("[demo] no projected contacts -- skipping")
        return
    P = _pooled_contacts(tmpl, base_of)
    pts, gmeta = P["pts"], P["gmeta"]

    series, lims = {}, {}
    for tap in taps:
        for task in tasks:
            per = D.get(tap, {}).get(task, {})
            if not per:
                continue
            k = f"{tap}|{task}"
            # scaled ints, not floats: d_cv lives in ~[-0.5, 0.5] so 1e-3 is well past the
            # resolution of a 26 px dot, and it roughly halves the payload.
            series[k] = [
                (None if b not in per.get(s, {}) else
                 [int(round(1000 * x)) if np.isfinite(x) else None for x in per[s][b][0]])
                for s, b in gmeta
            ]
            vals = np.concatenate([v[0] for s in per for v in per[s].values()])
            lims[k] = round(float(np.nanpercentile(np.abs(vals), 98)) or 1.0, 4)
    if not series:
        print("[demo] no (tap, task) had data -- skipping")
        return

    lut = (255 * mpl.colormaps["RdBu_r"](np.linspace(0, 1, 128))[:, :3]).round().astype(int)
    mark_of = {b: i + 1 for i, b in enumerate(MARK)}
    payload = {
        "img": {sd: base64.b64encode(_img_png(tmpl["img"][sd])).decode("ascii")
                for sd in ("left", "right")},
        "wh": {sd: [int(tmpl["img"][sd].shape[1]), int(tmpl["img"][sd].shape[0])]
               for sd in ("left", "right")},
        "x": [round(q["x"], 1) for q in pts],
        "y": [round(q["y"], 1) for q in pts],
        "side": [0 if q["side"] == "left" else 1 for q in pts],
        "g": [q["g"] for q in pts],
        "mark": [mark_of.get(q["base"], 0) for q in pts],
        "lab": [f"S{q['subj']} · {q['base']}" for q in pts],
        "series": series, "lim": lims, "t_ms": [round(float(v), 1) for v in _t_ms(T)],
        "lut": lut.tolist(), "ring": ["", *MARK.values()],
    }
    # menu order = the order asked for, NOT sorted: sorting opens the demo on
    # enc0|frame_brightness, which is the one view a reader has no reason to look at first.
    keys = [f"{tp}|{tk}" for tp in taps for tk in tasks if f"{tp}|{tk}" in series]
    html = _DEMO_HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))) \
                     .replace("__OPTIONS__", "".join(
                         # value stays the payload key; the label spells it out, because in a
                         # native <select> the pipe in "enc12|onset" reads as an l.
                         f'<option value="{k}">{k.replace("|", " — ")}</option>' for k in keys))
    p = os.path.join(out, "demo_anatomy.html")
    with open(p, "w") as f:
        f.write(html)
    print(f"[demo] {p}  ({os.path.getsize(p)/1e6:.1f} MB, {len(keys)} views, "
          f"{len(pts)} contacts, {T} bins, {len(gmeta)} (subject,parcel) groups)")


def _img_png(arr: np.ndarray) -> bytes:
    """PNG-encode the hemisphere image once, without a figure around it."""
    from PIL import Image

    a = arr
    if a.dtype != np.uint8:
        a = (255 * np.clip(a, 0, 1)).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


_DEMO_HTML = r"""<!doctype html><meta charset="utf-8">
<title>Anatomy of decodable information</title>
<style>
 body{font:14px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:22px;max-width:1400px;
      color:#222}
 h1{font-size:20px;margin:0 0 6px} p{color:#444;margin:5px 0 12px}
 .panes{display:flex;gap:10px} .pane{flex:1;position:relative}
 canvas{width:100%;border:1px solid #ddd;border-radius:6px;background:#fff;display:block}
 .cap{font-size:11.5px;color:#666;text-align:center;margin-top:2px}
 .row{display:flex;gap:14px;align-items:center;margin:12px 0;flex-wrap:wrap}
 select,input,button{font:inherit} button{padding:2px 12px}
 code{background:#f4f4f4;padding:1px 4px;border-radius:3px}
 .note{background:#fffbe6;border-left:3px solid #e8c000;padding:9px 13px;font-size:12.5px}
 #tip{position:fixed;pointer-events:none;background:#111;color:#fff;font-size:12px;
      padding:4px 8px;border-radius:4px;display:none;z-index:9}
 #bar{width:190px;height:12px;border:1px solid #bbb;border-radius:2px}
 .k{display:inline-flex;align-items:center;gap:5px;font-size:12px}
 .dot{width:12px;height:12px;border-radius:50%;border:2px solid}
</style>
<h1>Where is the decodable information? All 6 subjects on one brain, scrubbable in time</h1>
<p>Colour is the <b>split-half unbiased standardized class contrast</b> (<code>d_cv</code>).
Zero is a <i>true</i> zero — the two independent trial halves' contrast vectors are dotted, so
noise contributes zero in expectation and a no-effect contact is white, not faintly warm.
<b>Hover any contact</b> for its subject, DKT parcel and value.</p>
<div class="note"><b>Frame: the BrainTreebank's own 2D lateral projection</b> — not a warp of
mine, and not MNI. Positions are the <code>X</code>/<code>Y</code> columns of the dataset's
<code>elec_coords_full.csv</code> drawn on the hemisphere images it ships with, scaled exactly
as its own quickstart notebook does; <code>Hemisphere</code> picks the panel. Because it is a
flattened lateral view, depth along a shaft collapses and medial structures (cingulate, MTL)
land only approximately — read it for <i>which region</i>, never for millimetres. Colour is a
<b>parcel</b> value, so all of one subject's contacts in one parcel share it; grey means that
parcel has no value in that subject.</div>
<div class="row">
 <label>view <select id=k>__OPTIONS__</select></label>
 <label>time <input type=range id=t min=0 value=0 style="width:400px"></label>
 <b id=lab style="min-width:78px"></b>
 <button id=play>play</button>
 <span class=k><span class=dot style="border-color:#00a000"></span>Heschl's (A1)</span>
 <span class=k><span class=dot style="border-color:#8e24aa"></span>STG</span>
 <span class=k><span class=dot style="border-color:#666;background:#c8c8c8"></span>no value</span>
 <span class=k><span id=neg></span><canvas id=bar width=190 height=12></canvas><span id=pos></span></span>
</div>
<div class=panes>
 <div class=pane><canvas id=c0></canvas><div class=cap>left hemisphere</div></div>
 <div class=pane><canvas id=c1></canvas><div class=cap>right hemisphere</div></div>
</div>
<div id=tip></div>
<script>
const P=__PAYLOAD__;
const cv=[document.getElementById('c0'),document.getElementById('c1')];
const im=[new Image(),new Image()];
const t=document.getElementById('t'),k=document.getElementById('k'),
      lab=document.getElementById('lab'),pb=document.getElementById('play'),
      tip=document.getElementById('tip');
// open AT word onset, not at -484 ms: the first bin is pre-stimulus and shows nothing
t.max=P.t_ms.length-1; t.value=P.t_ms.findIndex(v=>v>=0);
['left','right'].forEach((sd,i)=>{cv[i].width=P.wh[sd][0];cv[i].height=P.wh[sd][1];
  im[i].src='data:image/png;base64,'+P.img[sd];});
// index contacts by panel once; only the values change as you scrub
const idx=[[],[]]; P.side.forEach((s,i)=>idx[s].push(i));
function col(v,lim){ // matplotlib RdBu_r, sent as a 128-entry LUT
  let u=Math.max(0,Math.min(1,(v/lim+1)/2)), c=P.lut[Math.round(u*(P.lut.length-1))];
  return 'rgb('+c[0]+','+c[1]+','+c[2]+')';}
function valAt(i,fi){const s=P.series[k.value][P.g[i]];
  if(s===null)return null; const v=s[fi]; return v===null?null:v/1000;}
function draw(){
  const fi=+t.value, lim=P.lim[k.value];
  lab.textContent=(P.t_ms[fi]>=0?'+':'')+P.t_ms[fi].toFixed(0)+' ms';
  document.getElementById('neg').textContent='-'+lim.toFixed(2)+' ';
  document.getElementById('pos').textContent=' +'+lim.toFixed(2);
  const bx=document.getElementById('bar').getContext('2d');
  for(let x=0;x<190;x++){bx.fillStyle=col((x/189*2-1)*lim,lim);bx.fillRect(x,0,1,12);}
  for(let s=0;s<2;s++){
    const g=cv[s].getContext('2d');
    g.clearRect(0,0,cv[s].width,cv[s].height);
    if(im[s].complete)g.drawImage(im[s],0,0,cv[s].width,cv[s].height);
    const R=Math.max(5,cv[s].width/210);
    for(const i of idx[s]){
      const v=valAt(i,fi);
      g.beginPath(); g.arc(P.x[i],P.y[i],P.mark[i]?R*1.05:R,0,6.2832);
      g.fillStyle=(v===null)?'#c8c8c8':col(v,lim); g.fill();
      g.lineWidth=P.mark[i]?R*.38:R*.22;  // a ring, not a filled disc: the value must stay visible
      g.strokeStyle=P.mark[i]?P.ring[P.mark[i]]:'rgba(0,0,0,.55)'; g.stroke();
    }
  }
}
im.forEach(x=>x.onload=draw);
k.onchange=draw; t.oninput=draw;
let timer=null;
pb.onclick=()=>{if(timer){clearInterval(timer);timer=null;pb.textContent='play';return;}
  pb.textContent='stop';
  timer=setInterval(()=>{t.value=(+t.value+1)%(+t.max+1);draw();},110);};
// hover: nearest contact in THIS panel, in canvas pixels
cv.forEach((c,s)=>{
  c.onmousemove=e=>{
    const r=c.getBoundingClientRect(), sc=c.width/r.width;
    const mx=(e.clientX-r.left)*sc, my=(e.clientY-r.top)*sc;
    let best=-1,bd=1e9;
    for(const i of idx[s]){const d=(P.x[i]-mx)**2+(P.y[i]-my)**2; if(d<bd){bd=d;best=i;}}
    if(best<0||bd>(14*sc)**2){tip.style.display='none';return;}
    const v=valAt(best,+t.value);
    tip.textContent=P.lab[best]+' · d_cv '+(v===null?'n/a':(v>=0?'+':'')+v.toFixed(3));
    tip.style.display='block';
    tip.style.left=(e.clientX+12)+'px'; tip.style.top=(e.clientY+12)+'px';};
  c.onmouseleave=()=>tip.style.display='none';});
draw();
</script>"""


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir",
                    default="results/viz_crosssubject/reductions/red_2s_15task_cdlin_45k")
    ap.add_argument("--bt-root", default=".cache/braintreebank")
    ap.add_argument("--out", default="results/showcase/paper")
    ap.add_argument("--tap", default="enc12", help="tap for the single-tap panels")
    ap.add_argument("--no-demo", action="store_true")
    ap.add_argument("--demo-tasks", default="onset,speech,pitch,frame_brightness")
    ap.add_argument("--demo-taps", default="enc0,enc12")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D, cov, T, base_of, lobe_of_base = compute(args.red_dir)
    print(f"[check] T={T} bins over [{WIN_START_S}, {WIN_END_S}] s "
          f"=> {(WIN_END_S-WIN_START_S)/T*1000:.2f} ms/bin, onset at bin {ONSET_BIN_OF(T)}")
    rows = gate(cov, lobe_of_base)
    invariant_st(D, cov, T, args.tap)
    invariant_zero(D, args.tap)
    R = invariant_task_contrast(D, rows, T, args.tap)
    _, keep = task_profiles(D, rows, args.tap, T)

    fig_coverage(cov, rows, args.out)
    for tap in TAPS:
        if tap in D:
            fig_dkt_time(D, rows, cov, T, args.out, tap)
    fig_st_invariant(D, cov, T, args.out, args.tap)
    fig_task_similarity(R, keep, args.out, args.tap)
    fig_depth(D, rows, cov, T, args.out)

    if not args.no_demo:
        tmpl = load_template_coords(args.red_dir, args.bt_root)
        invariant_projection(tmpl, base_of, lobe_of_base)
        fig_template_qc(tmpl, base_of, lobe_of_base, args.out)
        for task in ("onset", "speech"):
            fig_template_render(D, tmpl, base_of, T, args.out, args.tap, task)
        build_demo(D, tmpl, base_of, T, args.out,
                   tuple(t for t in args.demo_tasks.split(",") if t),
                   tuple(t for t in args.demo_taps.split(",") if t))


if __name__ == "__main__":
    main()
