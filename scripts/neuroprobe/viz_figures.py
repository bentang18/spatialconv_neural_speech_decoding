"""Cross-subject encoder figures from the reduced condition means.

Figure T  the class-contrast trajectory, one panel per task in ONE shared basis. Within a
          decodable task subjects trace the same path; across tasks the paths differ.

Figure R  cross-subject token retrieval, basis-free. `viz_retrieval_grid` draws these as a
          task x depth grid, which is the readable form.

Figure D  the depth ladder: every scalar this file produces, as a function of tap.

Quant     cross-subject similarity against the within-subject split-half ceiling, for every
          tap. The ceiling is the point: a raw correlation means nothing without knowing
          what the same subject's own two halves score.

Three figures were cut on 2026-07-29 rather than maintained. Figure A (3-D trajectory) is
strictly worse than the orbiting mp4 the same basis already produces. Figure B (PC-RGB
panels) is reproduced interactively, with a time cursor, in `viz_demo`. Figure I (identity
vs content) measured identity removal in a space small enough that the projector consumed
it -- the rank artifact retracted on 2026-07-28 -- and the LEACE run answers the same
question on CS decoding with a rank-1 eraser and a positive control.

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
    center_per_session, load_all, pca_basis, session_matrix, shared_lobes,
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


def collect(sessions, tap: str, task: str, cls, half: str, lobes, *, centered: bool,
            n_pre: int | None = None):
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
        mats = center_per_session([m for _, m in out], n_pre=n_pre)
        out = [(s, m) for (s, _), m in zip(out, mats)]
    return out


def _proj_origin(mu: np.ndarray, n_pre: int | None) -> np.ndarray:
    """Where the origin of the plotted space sits.

    ``pca_basis`` centers the token stack before the SVD, which is right for finding the
    DIRECTIONS but wrong for the origin once a pre-stimulus baseline is in play: projecting
    with that mean subtracted puts the origin back at the pooled average of the response --
    exactly the reference the baseline exists to escape. The symptom is a trajectory whose
    pre-stimulus frames sit well off centre even though they are identically zero in the
    data, which then reads as "it starts somewhere and comes back".

    So with a baseline, project from zero. The directions are unchanged; only the coordinate
    origin is, and it now means what the caption says it means. Without a baseline there is
    no meaningful zero in the data, so the pooled mean stays.
    """
    return np.zeros_like(mu) if n_pre else mu


def _traj(m: np.ndarray, comps: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """(lobes, T, C) -> (T, 3): project to the shared basis, then average over lobes."""
    p = (m.reshape(-1, m.shape[-1]) - mu) @ comps.T
    return p.reshape(m.shape[0], m.shape[1], 3).mean(axis=0)


def _scale_across_tasks(per_task: dict) -> dict:
    """ONE scale per session, shared across tasks; drops sessions with no signal at all."""
    scale: dict[str, float] = {}
    for v in per_task.values():
        for s, m in v:
            scale[s.key] = scale.get(s.key, 0.0) + float((m ** 2).sum())
    scale = {k: float(np.sqrt(x)) for k, x in scale.items()}
    return {t: [(s, m / scale[s.key]) for s, m in v if scale.get(s.key, 0.0) > 0]
            for t, v in per_task.items()}


def task_basis(sessions, lobes, tap: str, tasks, *, n_pre: int | None = None):
    """The shared 3-PC geometry behind both the task panel and the animation.

    Returned rather than recomputed in each renderer: a video drawn in a basis that is not
    quite the figure's basis is a bug nobody can see. (per_task, comps, mu, evr, lim).

    ``lim`` is ONE axis range for every panel and every frame. Per-panel autoscaling is the
    lie here: a task with no signal gets blown up to fill its box and reads as a trajectory,
    when the honest picture is that it barely moves. With a shared range the dud collapses
    to a dot, which is what a cross-subject r of ~0 actually looks like.
    """
    per_task = {t: collect(sessions, tap, t, CONTRAST, "all", lobes, centered=True,
                           n_pre=n_pre)
                for t in tasks}
    per_task = {t: v for t, v in per_task.items() if v}
    # ONE scale per session, shared across tasks. Scaling each task separately would defeat
    # the figure: a task with no signal would be blown back up to the same size as onset,
    # and the point of the panel is that it stays small. Scaling per session removes the
    # several-fold amplitude spread BETWEEN subjects, which otherwise sets the axis range
    # and dominates the pooled basis, while keeping each subject's own task ordering intact.
    per_task = _scale_across_tasks(per_task)
    if not per_task:
        return {}, np.zeros((3, 0)), np.zeros(0), np.zeros(3), 0.0
    stack = np.concatenate([m.reshape(-1, m.shape[-1])
                            for v in per_task.values() for _, m in v], axis=0)
    comps, mu, evr = pca_basis(stack, k=3)
    mu = _proj_origin(mu, n_pre)
    lim = max(float(np.abs(_traj(m, comps, mu)).max())
              for v in per_task.values() for _, m in v)
    return per_task, comps, mu, evr, lim


def figure_tasks(sessions, lobes, tap: str, tasks, out_path: str,
                 *, n_pre: int | None = None) -> dict:
    """One shared 3-PC space, one panel per task, contrast trajectories coloured by subject.

    The claim being shown is two-sided and one basis makes both visible at once: within a
    decodable task, subjects trace the SAME path; across tasks, the paths differ. The basis
    is fit on the pooled contrasts of every task so no task gets a basis flattering to it.
    """
    per_task, comps, mu, evr, lim = task_basis(sessions, lobes, tap, tasks, n_pre=n_pre)
    if not per_task:
        return {}

    n = len(per_task)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig = plt.figure(figsize=(5.2 * ncol, 4.6 * nrow))
    align = {}
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


def _cross_subject_r(v, comps, mu) -> float:
    """Mean pairwise r between DIFFERENT subjects' 3-PC trajectories in a given basis."""
    rs = [_corr(_traj(v[a][1], comps, mu).ravel(), _traj(v[b][1], comps, mu).ravel())
          for a in range(len(v)) for b in range(a + 1, len(v))
          if v[a][0].subject_id != v[b][0].subject_id]
    return float(np.nanmean(rs)) if rs else float("nan")


def align_loso(sessions, lobes, tap: str, tasks, *, n_pre: int | None = None) -> dict:
    """Cross-subject r with the 3-PC basis fit WITHOUT the subjects being scored.

    The pooled basis is fit on every session at once, so the first thing a reviewer asks is
    whether the agreement is the basis's doing -- three directions chosen to maximize
    variance over the very tokens that are then correlated. This refits per pair and excludes
    BOTH members, which is stricter than leave-one-subject-out and symmetric: a pair's score
    cannot depend on which of the two you nominate as held out.

    If the pooled number survives here, the concentration result is clean. If it collapses,
    the pooled basis was doing the work. (`retrieval` is already basis-free, so the pooled
    number was never the only evidence -- this makes the PCA number honest too.)
    """
    per_task, _, _, _, _ = task_basis(sessions, lobes, tap, tasks, n_pre=n_pre)
    if not per_task:
        return {}
    cache: dict[frozenset, tuple | None] = {}

    def basis_excluding(excl: frozenset):
        rows = [m.reshape(-1, m.shape[-1]) for v in per_task.values() for s, m in v
                if s.subject_id not in excl]
        if not rows:
            return None
        stack = np.concatenate(rows, axis=0)
        # a basis needs more tokens than components, else the SVD returns arbitrary directions
        if stack.shape[0] <= 3:
            return None
        comps, mu, evr = pca_basis(stack, k=3)
        return comps, _proj_origin(mu, n_pre), evr

    out = {}
    for t, v in per_task.items():
        rs = []
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                sa, ma = v[a]
                sb, mb = v[b]
                if sa.subject_id == sb.subject_id:
                    continue
                excl = frozenset((sa.subject_id, sb.subject_id))
                if excl not in cache:
                    cache[excl] = basis_excluding(excl)
                got = cache[excl]
                if got is None:
                    continue
                comps, mu, _ = got
                rs.append(_corr(_traj(ma, comps, mu).ravel(), _traj(mb, comps, mu).ravel()))
        out[t] = float(np.nanmean(rs)) if rs else float("nan")
    return out


def align_splithalf(sessions, lobes, tap: str, tasks, *, n_pre: int | None = None) -> dict:
    """Fit the basis on one interleaved trial half, score the other.

    A different question from LOSO: is the shared structure trial noise? h0 and h1 are
    interleaved halves of the same trials, so a basis fit on h0 that still lines subjects up
    on h1 is not fitting the noise of the tokens it scores. Returns the h1 score under the
    h0 basis, so nothing about the reported number was fit on the data it describes.
    """
    fit = {t: collect(sessions, tap, t, CONTRAST, "all", lobes, centered=True, n_pre=n_pre)
           for t in tasks}
    fit = _scale_across_tasks({t: v for t, v in fit.items() if v})
    score = {t: collect(sessions, tap, t, CONTRAST, "h1", lobes, centered=True, n_pre=n_pre)
             for t in tasks}
    score = _scale_across_tasks({t: v for t, v in score.items() if v})
    fit0 = {t: collect(sessions, tap, t, CONTRAST, "h0", lobes, centered=True, n_pre=n_pre)
            for t in tasks}
    fit0 = _scale_across_tasks({t: v for t, v in fit0.items() if v})
    if not fit0 or not score:
        return {}
    rows = [m.reshape(-1, m.shape[-1]) for v in fit0.values() for _, m in v]
    if not rows:
        return {}
    comps, mu, _ = pca_basis(np.concatenate(rows, axis=0), k=3)
    mu = _proj_origin(mu, n_pre)
    return {t: _cross_subject_r(v, comps, mu) for t, v in score.items()}


def peak_settle(sessions, lobes, tap: str, tasks, hz: float, offset: float,
                *, n_pre: int | None = None) -> dict:
    """When the contrast peaks, and whether it comes back. Makes "returns" vs "settles" readable.

    With a pre-stimulus baseline the origin means "no class difference before the event", so
    distance from the origin is directly interpretable: how far the response has moved from
    where it started. `onset` peaks and returns nearly all the way -- a word after silence is
    over. `speech` peaks and settles high -- the talking continues, so the difference from
    silence never goes away. Under the time-mean origin both looked like closed loops.

    `baseline_frac` is the built-in check, not a result: with n_pre set the pre-stimulus
    radius is zero by construction, so anything but ~0 means the baseline never got applied.
    """
    per_task, comps, mu, _, _ = task_basis(sessions, lobes, tap, tasks, n_pre=n_pre)
    out = {}
    for t, v in per_task.items():
        # mean radius curve over sessions; they are unit-scaled so the average is meaningful
        rad = np.mean([np.linalg.norm(_traj(m, comps, mu), axis=1) for _, m in v], axis=0)
        n_t = rad.shape[0]
        pk = int(np.argmax(rad))
        peak = float(rad[pk])
        tail = float(rad[max(n_t - n_t // 4, 1):].mean())
        base = float(rad[:n_pre].mean()) if n_pre else float("nan")
        out[t] = {
            "peak_s": round(offset + pk / hz, 4),
            "peak": round(peak, 6),
            "end_frac": round(float(rad[-1]) / peak, 4) if peak > 0 else float("nan"),
            "settle_frac": round(tail / peak, 4) if peak > 0 else float("nan"),
            "peak_over_settle": round(peak / tail, 4) if tail > 0 else float("nan"),
            "baseline_frac": round(base / peak, 4) if (n_pre and peak > 0) else None,
        }
    return out


def retrieval_sims(sessions, lobes, tap: str, task: str, *, n_pre: int | None = None):
    """The per-pair similarity matrices behind `retrieval`, plus its scalar summary.

    Split out so a figure can draw the matrices without recomputing them and without the
    scalars drifting from the ones already in report.json. Returns ([] , {}) when the task
    has fewer than two usable sessions, matching `retrieval`'s empty contract.
    """
    pairs = collect(sessions, tap, task, CONTRAST, "all", lobes, centered=True, n_pre=n_pre)
    traj = []
    for s, m in pairs:
        x = m.mean(axis=0)                                    # (T, C), lobe-averaged
        x = x - x.mean(axis=0, keepdims=True)                 # per-channel over time
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        traj.append((s, x / np.maximum(nrm, 1e-12)))
    if len(traj) < 2:
        return [], {}
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
        return [], {}
    out = {"tap": tap, "task": task, "top1": hits / tot, "chance": 1.0 / t_len,
           "median_rank": float(np.median(ranks)), "n_frames": t_len,
           "n_pairs": len(sims)}
    return sims, out


def retrieval(sessions, lobes, tap: str, task: str, out_path: str | None = None,
              *, n_pre: int | None = None) -> dict:
    """Can a token from one subject find the SAME timepoint in another subject?

    Correlation says the trajectories look alike overall; retrieval asks something sharper
    and harder to fake: take subject A's contrast at time t, and among all of subject B's
    timepoints pick the nearest. Top-1 at chance 1/T means the shared structure carries no
    temporal identity. The whole cross-subject decoding claim rests on tokens being
    comparable ACROSS brains, and this is that claim in its most direct form.
    """
    sims, out = retrieval_sims(sessions, lobes, tap, task, n_pre=n_pre)
    if not out:
        return {}
    t_len = out["n_frames"]
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


def quantify(sessions, lobes, tap: str, task: str, cls, *, n_pre: int | None = None) -> dict:
    """Cross-subject similarity vs the within-session split-half ceiling."""
    kw = {"centered": True, "n_pre": n_pre}
    grand = collect(sessions, tap, task, cls, "all", lobes, **kw)
    h0 = {s.key: m for s, m in collect(sessions, tap, task, cls, "h0", lobes, **kw)}
    h1 = {s.key: m for s, m in collect(sessions, tap, task, cls, "h1", lobes, **kw)}

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


def figure_depth(quant, retr, taps, tasks, out_path: str) -> dict:
    """Does cross-subject structure GROW with encoder depth? Two panels, same x axis.

    Left is retrieval top-1 above chance, right is the ceiling-normalized cross-subject
    correlation. Both are plotted for every task including the duds, because a depth ladder
    that only shows the tasks that work is not a ladder -- if the trend were an artefact of
    depth (more mixing -> smoother features -> higher correlation) it would lift
    frame_brightness too, and the figure has to be able to show that.

    Nothing is recomputed here: this draws the rows the quant and retrieval passes already
    produced, so the figure and the printed table cannot disagree.
    """
    order = {t: i for i, t in enumerate(taps)}
    q = {(r["tap"], r["task"]): r for r in quant if r.get("class") == CONTRAST}
    rr = {(r["tap"], r["task"]): r for r in retr}
    tasks = [t for t in tasks if any((tap, t) in rr for tap in taps)]
    if not tasks or len(taps) < 2:
        return {}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(taps))
    chance = next((v["chance"] for v in rr.values()), float("nan"))
    for t in tasks:
        y = [rr[(tap, t)]["top1"] if (tap, t) in rr else np.nan for tap in taps]
        axes[0].plot(x, y, marker="o", lw=1.6, label=t)
        y2 = [q[(tap, t)]["normalized"] if (tap, t) in q else np.nan for tap in taps]
        axes[1].plot(x, y2, marker="o", lw=1.6, label=t)
    axes[0].axhline(chance, color="k", ls="--", lw=1)
    axes[0].annotate(f"chance {chance:.3f}", (0, chance), fontsize=7,
                     xytext=(2, 3), textcoords="offset points")
    axes[0].set_ylabel("cross-subject retrieval, top-1")
    axes[1].axhline(0.0, color="k", ls="--", lw=1)
    axes[1].set_ylabel("cross-subject r / split-half ceiling")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(taps)
        ax.set_xlabel("encoder tap (depth →)")
        ax.grid(alpha=0.25)
    axes[1].set_title("gaps = split-half ceiling too low to normalize against", fontsize=8,
                      color="#666")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Depth ladder · class contrast · shared lobes", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    slope = {t: float(rr[(taps[-1], t)]["top1"] - rr[(taps[0], t)]["top1"])
             for t in tasks if (taps[0], t) in rr and (taps[-1], t) in rr}
    return {"taps": list(taps), "chance": float(chance),
            "top1_first_to_last": slope, "n_tasks": len(tasks), "tap_order": order}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--out-dir", default="results/viz_crosssubject")
    ap.add_argument("--taps", default="enc0,enc3,enc6,enc12")
    ap.add_argument("--tasks-quant", default="onset,speech,delta_volume,word_index,"
                                             "word_part_speech,frame_brightness")
    # Baseline reference. 0 keeps the window's time-average as the origin, which is what the
    # 1 s window has to use (it has no pre-stimulus frames). At 2 s pass 16 (= 0.5 s x 32 Hz)
    # so the origin becomes "no class difference before the event" and the sustained part of
    # the response survives instead of being centered away.
    ap.add_argument("--n-pre", type=int, default=0,
                    help="pre-stimulus frames to baseline against; 0 = time-mean origin")
    ap.add_argument("--hz", type=float, default=32.0)
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds of the first frame (negative if the window leads onset)")
    args = ap.parse_args()
    n_pre = args.n_pre or None
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = load_all(args.red_dir)
    taps = [t for t in args.taps.split(",") if t]
    taps = [t for t in taps if any(t in s.shapes for s in sessions)]
    lobes = shared_lobes(sessions)
    print(f"[load] {len(sessions)} sessions, subjects "
          f"{sorted({s.subject_id for s in sessions})}, taps {taps}")
    print(f"[check] shared lobes across ALL subjects: {lobes}")
    assert lobes, "no lobe is shared by every subject — there is no common axis to plot in"

    quant_tasks = [t for t in args.tasks_quant.split(",") if t]
    report: dict = {"sessions": [s.key for s in sessions], "shared_lobes": lobes,
                    "taps": taps, "figures": {}, "quant": [],
                    "n_pre": args.n_pre, "offset_s": args.offset, "hz": args.hz,
                    "centering": "baseline" if n_pre else "time-mean"}
    print(f"[check] origin = {report['centering']}"
          + (f" (first {args.n_pre} frames, up to t={args.offset + args.n_pre / args.hz:+.3f}s)"
             if n_pre else " (no pre-stimulus frames available)"))

    for tap in taps:
        p = os.path.join(args.out_dir, f"figT_tasks_{tap}.png")
        info = figure_tasks(sessions, lobes, tap, quant_tasks, p, n_pre=n_pre)
        report["figures"][f"T/{tap}"] = info
        print(f"[fig] {p}  align={ {k: round(v, 3) for k, v in info.get('align_3pc', {}).items()} }")

    # The honesty block: the pooled 3-PC number, the same number with the basis blind to the
    # pair being scored, and the same number with the basis fit on the other trial half.
    report["align_loso"], report["align_splithalf"], report["peak_settle"] = {}, {}, {}
    for tap in taps:
        pooled = report["figures"].get(f"T/{tap}", {}).get("align_3pc", {})
        loso = align_loso(sessions, lobes, tap, quant_tasks, n_pre=n_pre)
        half = align_splithalf(sessions, lobes, tap, quant_tasks, n_pre=n_pre)
        ps = peak_settle(sessions, lobes, tap, quant_tasks, args.hz, args.offset, n_pre=n_pre)
        report["align_loso"][tap] = loso
        report["align_splithalf"][tap] = half
        report["peak_settle"][tap] = ps
        for t in quant_tasks:
            if t not in loso:
                continue
            d = ps.get(t, {})
            bf = d.get("baseline_frac")
            flag = "" if bf is None else ("  [check] OK" if abs(bf) < 0.35
                                          else f"  [check] VIOLATED base_frac={bf:+.3f}")
            print(f"[basis] {tap:6s} {t:16s} pooled={pooled.get(t, float('nan')):+.4f} "
                  f"loso={loso[t]:+.4f} splithalf={half.get(t, float('nan')):+.4f} "
                  f"peak={d.get('peak_s', float('nan')):+.3f}s "
                  f"settle_frac={d.get('settle_frac', float('nan')):.3f}{flag}")

    for task in quant_tasks:
        for tap in taps:
            for cls in (CONTRAST, 0, 1):
                try:
                    q = quantify(sessions, lobes, tap, task, cls, n_pre=n_pre)
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
            r = retrieval(sessions, lobes, tap, task, p, n_pre=n_pre)
            if r:
                report["retrieval"].append(r)
                print(f"[retr]  {tap:6s} {task:16s} top1={r['top1']:.3f} "
                      f"(chance {r['chance']:.3f}) median_rank={r['median_rank']:.0f}"
                      f"/{r['n_frames']}")

    p = os.path.join(args.out_dir, "figD_depth_ladder.png")
    report["figures"]["D"] = figure_depth(report["quant"], report["retrieval"], taps,
                                          quant_tasks, p)
    print(f"[fig] {p}")

    dst = os.path.join(args.out_dir, "report.json")
    with open(dst, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"[write] {dst}")


if __name__ == "__main__":
    main()
