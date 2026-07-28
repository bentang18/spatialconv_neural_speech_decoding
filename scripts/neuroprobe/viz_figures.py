"""Cross-subject encoder figures from the reduced condition means.

Figure A  trajectory of the trial-averaged response through a SHARED 3-PC space, one line
          per session. Three panels on purpose: without per-session centering (identity is
          the dominant structure and subjects sit apart), with it (do they MOVE the same
          way), and the class contrast (the content that distinguishes the labels).

Figure T  the same contrast trajectory, one panel per task in ONE shared basis. Within a
          decodable task subjects trace the same path; across tasks the paths differ.

Figure B  the DINOv3 panel. Tokens are (region x time), PCA runs over the CHANNEL axis, the
          first 3 components are painted as RGB. Panels are NOT row-matched across subjects
          -- the Lite cohort shares one lobe, so a matched grid does not exist -- and they
          do not need to be: DINOv3 paints different images through one basis and lets
          corresponding parts land on the same colour. Same construction here.

Quant     cross-subject similarity against the within-subject split-half ceiling, for every
          tap. The ceiling is the point: a raw correlation means nothing without knowing
          what the same subject's own two halves score.

enc0 is the control throughout. It is the untrained |STFT| frontend, gets byte-identical
treatment, and is constant across checkpoints -- so any enc12-minus-enc0 gap is the encoder
and not the pipeline.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.neuroprobe.viz_common import (  # noqa: E402
    center_per_session, load_all, pca_basis, session_matrix, shared_lobes, to_rgb,
)

SUBJ_COLORS = {1: "#e6194b", 2: "#3cb44b", 3: "#4363d8",
               4: "#f58231", 7: "#911eb4", 10: "#008080"}

# Sentinel class id meaning "class 1 minus class 0".
CONTRAST = "contrast"

# Below this split-half reliability the ceiling-normalized score is reported as nan.
CEILING_FLOOR = 0.10


def _flat(m: np.ndarray) -> np.ndarray:
    return m.reshape(-1)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


def _cond_matrix(sess, tap: str, task: str, cls, half: str, lobes):
    """One session's (lobes, T, C) for a class, or for the class-1 minus class-0 contrast."""
    if cls != CONTRAST:
        return session_matrix(sess, tap, task, cls, half, lobes)
    m1 = session_matrix(sess, tap, task, 1, half, lobes)
    m0 = session_matrix(sess, tap, task, 0, half, lobes)
    return None if (m1 is None or m0 is None) else m1 - m0


def collect(sessions, tap: str, task: str, cls, half: str, lobes, *, centered: bool):
    """(session, matrix) pairs for one condition, optionally identity-centered.

    cls=CONTRAST returns the class-1 minus class-0 difference. That is the quantity the
    figures should be scoring: a single condition mean is dominated by a large
    condition-INDEPENDENT response profile (the average evoked HGA shape), which is shared
    across subjects for every task and drives cross-subject r to ~0.95 whether or not the
    task is decodable at all. Only the contrast isolates what distinguishes the classes,
    which is also what the ridge actually reads.
    """
    out = []
    for s in sessions:
        m = _cond_matrix(s, tap, task, cls, half, lobes)
        if m is not None:
            out.append((s, m))
    if centered and out:
        mats = center_per_session([m for _, m in out])
        out = [(s, m) for (s, _), m in zip(out, mats)]
    return out


def unit_scale(pairs):
    """Rescale each session's matrix to unit norm, keeping its shape.

    Contrast AMPLITUDE varies several-fold across subjects (trial counts, electrode counts,
    SNR), so without this one subject sets the axis range and, worse, dominates the pooled
    PCA basis -- the shared space would be that subject's space. The reported score is a
    correlation, which is already scale-invariant, so leaving the figure scale-dependent
    would mean the picture and the number disagree about what is being compared.
    """
    out = []
    for s, m in pairs:
        n = float(np.linalg.norm(m))
        out.append((s, m / n if n > 0 else m))
    return out


def _traj(m: np.ndarray, comps: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """(lobes, T, C) -> (T, 3): project to the shared basis, then average over lobes."""
    p = (m.reshape(-1, m.shape[-1]) - mu) @ comps.T
    return p.reshape(m.shape[0], m.shape[1], 3).mean(axis=0)


def figure_a(sessions, lobes, tap: str, task: str, out_path: str) -> dict:
    """3-D trajectory in a shared PC space: raw, identity-centered, and the class contrast.

    Three panels because they answer three different questions, and only the third is about
    content. Panels 1-2 plot condition means, which are dominated by a large
    condition-INDEPENDENT response profile: the two classes trace nearly the same path
    whether or not the task is decodable. Panel 3 plots class1 - class0, which is the part
    that distinguishes the labels and the part a cross-subject decoder can actually use.
    """
    fig = plt.figure(figsize=(19, 6))
    info = {}
    pairs_c = unit_scale(collect(sessions, tap, task, CONTRAST, "all", lobes, centered=True))
    if pairs_c:
        stack_c = np.concatenate([m.reshape(-1, m.shape[-1]) for _, m in pairs_c], axis=0)
        comps_c, mu_c, evr_c = pca_basis(stack_c, k=3)
        ax = fig.add_subplot(1, 3, 3, projection="3d")
        for s, m in pairs_c:
            p = _traj(m, comps_c, mu_c)
            ax.plot(p[:, 0], p[:, 1], p[:, 2],
                    color=SUBJ_COLORS.get(s.subject_id, "#888"), lw=1.6, alpha=0.9)
            ax.scatter(*p[0], color=SUBJ_COLORS.get(s.subject_id, "#888"), s=22, marker="o")
        ax.set_title(f"{tap} · {task} · class contrast (content)\nEVR {evr_c.sum():.2f}",
                     fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        info["evr_contrast"] = [float(v) for v in evr_c]
    for col, centered in enumerate((False, True)):
        pairs = {c: collect(sessions, tap, task, c, "all", lobes, centered=centered)
                 for c in (0, 1)}
        stack = np.concatenate([m.reshape(-1, m.shape[-1])
                                for c in (0, 1) for _, m in pairs[c]], axis=0)
        comps, mu, evr = pca_basis(stack, k=3)
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        for c in (0, 1):
            for s, m in pairs[c]:
                p = _traj(m, comps, mu)
                ax.plot(p[:, 0], p[:, 1], p[:, 2],
                        color=SUBJ_COLORS.get(s.subject_id, "#888"),
                        ls="-" if c == 1 else ":", lw=1.6, alpha=0.9)
                ax.scatter(*p[0], color=SUBJ_COLORS.get(s.subject_id, "#888"),
                           s=22, marker="o")
        title = "identity-centered" if centered else "raw (identity included)"
        ax.set_title(f"{tap} · {task} · {title}\nEVR {evr.sum():.2f}", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        info["evr_centered" if centered else "evr_raw"] = [float(v) for v in evr]
    handles = [Line2D([], [], color=c, lw=2, label=f"S{s}")
               for s, c in SUBJ_COLORS.items()]
    handles += [Line2D([], [], color="k", ls="-", label=f"{task}=1"),
                Line2D([], [], color="k", ls=":", label=f"{task}=0")]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8)
    fig.suptitle(f"Cross-subject trajectory · lobes={lobes} · circle = window start", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return info


def figure_tasks(sessions, lobes, tap: str, tasks, out_path: str) -> dict:
    """One shared 3-PC space, one panel per task, contrast trajectories coloured by subject.

    The claim being shown is two-sided and one basis makes both visible at once: within a
    decodable task, subjects trace the SAME path; across tasks, the paths differ. The basis
    is fit on the pooled contrasts of every task so no task gets a basis flattering to it.
    """
    per_task = {t: collect(sessions, tap, t, CONTRAST, "all", lobes, centered=True)
                for t in tasks}
    per_task = {t: v for t, v in per_task.items() if v}
    # ONE scale per session, shared across tasks. Scaling each task separately would defeat
    # the figure: a task with no signal would be blown back up to the same size as onset,
    # and the point of the panel is that it stays small. Scaling per session removes the
    # several-fold amplitude spread BETWEEN subjects, which otherwise sets the axis range
    # and dominates the pooled basis, while keeping each subject's own task ordering intact.
    scale: dict[str, float] = {}
    for v in per_task.values():
        for s, m in v:
            scale[s.key] = scale.get(s.key, 0.0) + float((m ** 2).sum())
    scale = {k: np.sqrt(x) for k, x in scale.items()}
    per_task = {t: [(s, m / scale[s.key] if scale.get(s.key, 0) > 0 else m) for s, m in v]
                for t, v in per_task.items()}
    if not per_task:
        return {}
    stack = np.concatenate([m.reshape(-1, m.shape[-1])
                            for v in per_task.values() for _, m in v], axis=0)
    comps, mu, evr = pca_basis(stack, k=3)

    n = len(per_task)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(5.2 * ncol, 4.6 * nrow))
    align = {}
    # ONE axis range for every panel. Per-panel autoscaling is the lie here: a task with no
    # signal gets blown up to fill its box and reads as a trajectory, when the honest picture
    # is that it barely moves. With a shared range the dud collapses to a dot, which is what
    # a cross-subject r of ~0 actually looks like.
    lim = max(float(np.abs(_traj(m, comps, mu)).max())
              for v in per_task.values() for _, m in v)
    for i, (t, v) in enumerate(per_task.items()):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        trajs = []
        for s, m in v:
            p = _traj(m, comps, mu)
            trajs.append(p)
            ax.plot(p[:, 0], p[:, 1], p[:, 2],
                    color=SUBJ_COLORS.get(s.subject_id, "#888"), lw=1.5, alpha=0.9)
            ax.scatter(*p[0], color=SUBJ_COLORS.get(s.subject_id, "#888"), s=20)
        # agreement between subjects IN THIS 3-D VIEW, so the number matches what is drawn
        rs = [_corr(trajs[a].ravel(), trajs[b].ravel())
              for a in range(len(v)) for b in range(a + 1, len(v))
              if v[a][0].subject_id != v[b][0].subject_id]
        align[t] = float(np.nanmean(rs)) if rs else float("nan")
        ax.set_title(f"{t}\ncross-subject r (3-PC view) = {align[t]:+.2f}", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    handles = [Line2D([], [], color=c, lw=2, label=f"S{s}") for s, c in SUBJ_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.suptitle(f"Class-contrast trajectories · {tap} · ONE shared 3-PC basis across all "
                 f"tasks (EVR {evr.sum():.2f}) · circle = window start", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return {"evr": [float(v) for v in evr], "align_3pc": align}


def figure_b(sessions, tap: str, task: str, cls, out_path: str) -> dict:
    """Per-subject PC-RGB panels: rows are that subject's own lobes, columns are time."""
    per = []
    for s in sessions:
        lobes = sorted({lb for lb in s.lobes if lb != "unknown"})
        m = _cond_matrix(s, tap, task, cls, "all", lobes)
        if m is not None:
            per.append((s, lobes, m))
    assert per, "no session produced a panel"
    centered = center_per_session([m for _, _, m in per])
    stack = np.concatenate([m.reshape(-1, m.shape[-1]) for m in centered], axis=0)
    comps, mu, evr = pca_basis(stack, k=3)
    proj = [((m.reshape(-1, m.shape[-1]) - mu) @ comps.T).reshape(m.shape[0], m.shape[1], 3)
            for m in centered]
    rgb = to_rgb(np.concatenate([p.reshape(-1, 3) for p in proj], axis=0))
    off, rgbs = 0, []
    for p in proj:
        n = p.shape[0] * p.shape[1]
        rgbs.append(rgb[off:off + n].reshape(p.shape))
        off += n

    n = len(per)
    ncol = 4
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.5 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, ((s, lobes, _), img) in enumerate(zip(per, rgbs)):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(lobes)))
        ax.set_yticklabels(lobes, fontsize=6)
        ax.set_xlabel("time →", fontsize=7)
        ax.set_title(s.key, fontsize=8)
        ax.tick_params(labelsize=6)
    cond = f"{task} (class1 − class0)" if cls == CONTRAST else f"{task}={cls}"
    fig.suptitle(f"PC-RGB · {tap} · {cond} · one shared 3-PC channel basis "
                 f"(EVR {evr.sum():.2f}) · rows are each subject's own lobes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return {"evr": [float(v) for v in evr], "n_panels": n}


def retrieval(sessions, lobes, tap: str, task: str, out_path: str | None = None) -> dict:
    """Can a token from one subject find the SAME timepoint in another subject?

    Correlation says the trajectories look alike overall; retrieval asks something sharper
    and harder to fake: take subject A's contrast at time t, and among all of subject B's
    timepoints pick the nearest. Top-1 at chance 1/T means the shared structure carries no
    temporal identity. The whole cross-subject decoding claim rests on tokens being
    comparable ACROSS brains, and this is that claim in its most direct form.
    """
    pairs = collect(sessions, tap, task, CONTRAST, "all", lobes, centered=True)
    traj = []
    for s, m in pairs:
        x = m.mean(axis=0)                                    # (T, C), lobe-averaged
        x = x - x.mean(axis=0, keepdims=True)                 # per-channel over time
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        traj.append((s, x / np.maximum(nrm, 1e-12)))
    if len(traj) < 2:
        return {}
    t_len = traj[0][1].shape[0]
    hits, tot, sims, ranks = 0, 0, [], []
    for i, (si, a) in enumerate(traj):
        for sj, b in [(s, m) for s, m in traj[i + 1:]]:
            if si.subject_id == sj.subject_id:
                continue
            sim = a @ b.T                                     # (T, T) cosine
            sims.append(sim)
            for direction in (sim, sim.T):
                pred = direction.argmax(axis=1)
                hits += int((pred == np.arange(t_len)).sum())
                tot += t_len
                # rank of the true timepoint: top-1 is brittle at 64 frames, the rank is not
                order = np.argsort(-direction, axis=1)
                ranks.extend(int(np.where(order[k] == k)[0][0]) for k in range(t_len))
    if not tot:
        return {}
    out = {"tap": tap, "task": task, "top1": hits / tot, "chance": 1.0 / t_len,
           "median_rank": float(np.median(ranks)), "n_frames": t_len,
           "n_pairs": len(sims)}
    if out_path:
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(np.mean(sims, axis=0), cmap="magma", interpolation="nearest")
        ax.set_xlabel("subject B frame")
        ax.set_ylabel("subject A frame")
        ax.set_title(f"{tap} · {task} · mean cross-subject token similarity\n"
                     f"top-1 {out['top1']:.3f} (chance {out['chance']:.3f}), "
                     f"median rank {out['median_rank']:.0f}/{t_len}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_path, dpi=170)
        plt.close(fig)
    return out


def quantify(sessions, lobes, tap: str, task: str, cls) -> dict:
    """Cross-subject similarity vs the within-session split-half ceiling."""
    grand = collect(sessions, tap, task, cls, "all", lobes, centered=True)
    h0 = {s.key: m for s, m in collect(sessions, tap, task, cls, "h0", lobes, centered=True)}
    h1 = {s.key: m for s, m in collect(sessions, tap, task, cls, "h1", lobes, centered=True)}

    ceiling = [_corr(_flat(h0[s.key]), _flat(h1[s.key]))
               for s, _ in grand if s.key in h0 and s.key in h1]
    cross, within_subj = [], []
    for i in range(len(grand)):
        for j in range(i + 1, len(grand)):
            si, mi = grand[i]
            sj, mj = grand[j]
            r = _corr(_flat(mi), _flat(mj))
            (within_subj if si.subject_id == sj.subject_id else cross).append(r)

    def _m(v):
        return float(np.nanmean(v)) if v else float("nan")

    c, cl = _m(cross), _m(ceiling)
    # A ratio against a ceiling that is itself indistinguishable from zero is not a number
    # worth reporting: it means the subject's own two half-averages disagree, so there is no
    # reliable signal to normalize against and c/cl is dominated by noise in the denominator.
    # frame_brightness is exactly this case (ceiling ~.06), and left unguarded it prints a
    # confident-looking -0.62.
    usable = bool(np.isfinite(cl) and cl > CEILING_FLOOR)
    return {
        "tap": tap, "task": task, "class": cls, "n_sessions": len(grand),
        "cross_subject_r": c, "within_subject_diff_session_r": _m(within_subj),
        "split_half_ceiling_r": cl,
        "normalized": float(c / cl) if usable else float("nan"),
        "ceiling_usable": usable, "ceiling_floor": CEILING_FLOOR,
        "n_cross_pairs": len(cross),
    }


def identity_content(sessions, lobes, tap: str, task: str) -> dict:
    """Split the token cloud into a between-session (identity) and a within-session part.

    Separability, not invariance, is the claim being tested. Transfer does not require the
    identity directions to be absent -- it requires them not to be confounded with the
    content directions. So: build the identity subspace explicitly as the span of the
    per-session mean offsets, project it out, and ask whether cross-subject alignment
    SURVIVES. If it rises, the two are separable and removing identity is not removing
    content.
    """
    per, labels = [], []
    for cls in (0, 1):
        for s, m in collect(sessions, tap, task, cls, "all", lobes, centered=False):
            per.append(m.reshape(-1, m.shape[-1]))
            labels.append((s.key, cls))
    if len(per) < 4:
        return {}
    tokens = np.concatenate(per, axis=0)
    grand = tokens.mean(axis=0)

    keys = sorted({k for k, _ in labels})
    by_key = {k: np.concatenate([p for p, (kk, _) in zip(per, labels) if kk == k], axis=0)
              for k in keys}

    # Between/within variance decomposition -- the ANOVA identity, so the two shares sum to
    # the total exactly. NOT a subspace projection: projecting onto the rank-<=N span of the
    # session means captures ~rank/C of the variance by construction, which reads as a large
    # identity share even when the session means are pure noise.
    n_tot = sum(len(v) for v in by_key.values())
    between = sum(len(v) * float(((v.mean(axis=0) - grand) ** 2).sum())
                  for v in by_key.values()) / n_tot
    within = sum(float(((v - v.mean(axis=0)) ** 2).sum()) for v in by_key.values()) / n_tot
    total = between + within

    # Identity subspace for the projection test, rank-truncated: a direction with no energy
    # in the session means is not an identity direction, and QR of a degenerate matrix would
    # otherwise hand back arbitrary axes.
    ident = np.stack([by_key[k].mean(axis=0) - grand for k in keys])
    u_s = np.linalg.svd(ident, compute_uv=False)
    scale = np.sqrt(max(total, 0.0))
    keep = int((u_s > max(1e-6 * u_s.max(initial=0.0), 1e-8 * scale)).sum())
    if keep:
        q, _ = np.linalg.qr(ident[:keep].T)
    else:
        q = np.zeros((tokens.shape[1], 0))

    def _align(project_out: bool) -> float:
        # the CONTRAST, not one class: alignment of a single condition mean is carried by
        # the condition-independent response profile and says nothing about content.
        mats = []
        for s, m in collect(sessions, tap, task, CONTRAST, "all", lobes, centered=True):
            x = m.reshape(-1, m.shape[-1])
            if project_out:
                x = x - (x @ q) @ q.T
            mats.append((s, x))
        rs = [_corr(_flat(a), _flat(b))
              for i, (si, a) in enumerate(mats) for sj, b in mats[i + 1:]
              if si.subject_id != sj.subject_id]
        return float(np.nanmean(rs)) if rs else float("nan")

    return {
        "tap": tap, "task": task, "identity_rank": int(q.shape[1]),
        "n_sessions": len(keys),
        "identity_var_frac": float(between / total) if total > 0 else float("nan"),
        "within_session_var_frac": float(within / total) if total > 0 else float("nan"),
        "cross_subject_r": _align(False),
        "cross_subject_r_identity_removed": _align(True),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--out-dir", default="results/viz_crosssubject")
    ap.add_argument("--task", default="onset")
    ap.add_argument("--taps", default="enc0,enc3,enc6,enc12")
    ap.add_argument("--tasks-quant", default="onset,speech,delta_volume,word_index,"
                                             "word_part_speech,frame_brightness")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = load_all(args.red_dir)
    taps = [t for t in args.taps.split(",") if t]
    taps = [t for t in taps if any(t in s.shapes for s in sessions)]
    lobes = shared_lobes(sessions)
    print(f"[load] {len(sessions)} sessions, subjects "
          f"{sorted({s.subject_id for s in sessions})}, taps {taps}")
    print(f"[check] shared lobes across ALL subjects: {lobes}")
    assert lobes, "no lobe is shared by every subject — Figure A has no common axis"

    quant_tasks = [t for t in args.tasks_quant.split(",") if t]
    report: dict = {"sessions": [s.key for s in sessions], "shared_lobes": lobes,
                    "taps": taps, "figures": {}, "quant": []}

    for tap in taps:
        p = os.path.join(args.out_dir, f"figA_trajectory_{tap}_{args.task}.png")
        report["figures"][f"A/{tap}"] = figure_a(sessions, lobes, tap, args.task, p)
        print(f"[fig] {p}")
        p = os.path.join(args.out_dir, f"figB_pcrgb_{tap}_{args.task}.png")
        report["figures"][f"B/{tap}"] = figure_b(sessions, tap, args.task, 1, p)
        print(f"[fig] {p}")
        p = os.path.join(args.out_dir, f"figB_pcrgb_contrast_{tap}_{args.task}.png")
        report["figures"][f"Bc/{tap}"] = figure_b(sessions, tap, args.task, CONTRAST, p)
        print(f"[fig] {p}")
        p = os.path.join(args.out_dir, f"figT_tasks_{tap}.png")
        info = figure_tasks(sessions, lobes, tap, quant_tasks, p)
        report["figures"][f"T/{tap}"] = info
        print(f"[fig] {p}  align={ {k: round(v, 3) for k, v in info.get('align_3pc', {}).items()} }")

    for task in quant_tasks:
        for tap in taps:
            for cls in (CONTRAST, 0, 1):
                try:
                    q = quantify(sessions, lobes, tap, task, cls)
                except (KeyError, AssertionError):
                    continue
                report["quant"].append(q)
                lab = "diff" if cls == CONTRAST else f"c{cls}  "
                print(f"[quant] {tap:6s} {task:16s} {lab} cross={q['cross_subject_r']:+.4f} "
                      f"ceiling={q['split_half_ceiling_r']:+.4f} "
                      f"norm={q['normalized']:+.4f}")

    report["retrieval"] = []
    for tap in taps:
        for task in quant_tasks:
            p = os.path.join(args.out_dir, f"figR_retrieval_{tap}_{task}.png")
            r = retrieval(sessions, lobes, tap, task, p)
            if r:
                report["retrieval"].append(r)
                print(f"[retr]  {tap:6s} {task:16s} top1={r['top1']:.3f} "
                      f"(chance {r['chance']:.3f}) median_rank={r['median_rank']:.0f}"
                      f"/{r['n_frames']}")

    report["identity_content"] = []
    for tap in taps:
        ic = identity_content(sessions, lobes, tap, args.task)
        if ic:
            report["identity_content"].append(ic)
            print(f"[ident] {tap:6s} {args.task:16s} identity_var={ic['identity_var_frac']:.3f} "
                  f"(rank {ic['identity_rank']})  cross={ic['cross_subject_r']:+.4f} "
                  f"-> id-removed {ic['cross_subject_r_identity_removed']:+.4f}")

    dst = os.path.join(args.out_dir, "report.json")
    with open(dst, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[write] {dst}")


if __name__ == "__main__":
    main()
