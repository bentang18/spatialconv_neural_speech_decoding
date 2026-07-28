"""Animate the cross-subject trajectories: draw them on in time, then orbit the camera.

A static 3-D scatter of six overlapping trajectories is close to unreadable -- the depth cue
is missing, and whether two paths coincide or merely cross in projection is exactly the
thing the figure is supposed to answer. Rotation supplies the parallax; the draw-on supplies
the time axis, which a finished 3-D curve throws away.

Two phases in one clip:

  1. DRAW  every subject advances one timepoint per frame, simultaneously. If the subjects
     are aligned, the heads move together; if they are not, they scatter immediately. This
     is the claim in the most direct form the medium allows.
  2. ORBIT the completed trajectories, one full turn, so the viewer can confirm the paths
     really do coincide in 3-D rather than only in one projection.

The geometry comes from ``task_basis``, the SAME call the static panel uses -- shared PCA
basis, shared per-session scaling, one shared axis range across every task and every frame.
A video in a slightly different basis than the figure is a discrepancy no reviewer could
catch, so there is only one copy of that computation.

The axis range is shared across tasks ON PURPOSE: a task with no cross-subject structure
must render as a jitter near the origin, not be blown up to fill the box.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from scripts.neuroprobe.viz_common import load_all, shared_lobes  # noqa: E402
from scripts.neuroprobe.viz_figures import (  # noqa: E402
    SUBJ_COLORS, _traj, task_basis,
)


def _writer(out_path: str, fps: int):
    """mp4 when ffmpeg is present, animated GIF otherwise. Never silently write nothing."""
    if out_path.endswith(".mp4") and animation.writers.is_available("ffmpeg"):
        return animation.FFMpegWriter(fps=fps, bitrate=2400), out_path
    if animation.writers.is_available("pillow"):
        return animation.PillowWriter(fps=fps), os.path.splitext(out_path)[0] + ".gif"
    raise SystemExit("no usable animation writer (need ffmpeg or pillow)")


def pool_subjects(trajs):
    """Average a subject's sessions into one curve. 12 overlaid paths is a tangle, and the
    claim is about SUBJECTS anyway -- two trials of one subject share a montage and would
    read as agreement that the cross-subject metric never counted."""
    by: dict[int, list] = {}
    for s, p in trajs:
        by.setdefault(s.subject_id, []).append(p)
    return [(sid, np.mean(v, axis=0)) for sid, v in sorted(by.items())]


def animate_task(sessions, lobes, tap: str, task: str, tasks, out_path: str, *,
                 fps: int = 12, orbit_frames: int = 90, hold: int = 6,
                 tail: int = 8, per_subject: bool = True) -> dict:
    """One task's contrast trajectories: draw on in time, then a full camera orbit.

    During the draw only the last ``tail`` points are shown. Twelve complete 3-D paths
    overlaid is unreadable regardless of how well they agree, so the full curve would hide
    the very thing the clip exists to show; a short tail makes co-movement of the heads the
    visible signal. The orbit phase then reveals the complete paths, which is when the whole
    shape is what matters.
    """
    per_task, comps, mu, _, lim = task_basis(sessions, lobes, tap, tasks)
    if task not in per_task:
        return {}
    pairs = per_task[task]
    trajs = [(s, _traj(m, comps, mu)) for s, m in pairs]
    n_sessions = len(trajs)
    if per_subject:
        trajs = [(sid, p) for sid, p in pool_subjects(trajs)]
    else:
        trajs = [(s.subject_id, p) for s, p in trajs]
    t_len = trajs[0][1].shape[0]

    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    lines, heads = [], []
    for sid, p in trajs:
        col = SUBJ_COLORS.get(sid, "#888")
        # a faint complete path behind everything: without it the draw phase is a near-empty
        # box whenever the trajectory happens to pass close to the origin, and the reader has
        # no idea whether that is the data or a rendering bug
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=col, lw=1.0, alpha=0.13)
        lines.append(ax.plot([], [], [], color=col, lw=1.7, alpha=0.9)[0])
        heads.append(ax.plot([], [], [], color=col, marker="o", ms=6, ls="none")[0])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    handles = [Line2D([], [], color=c, lw=2, label=f"S{s}") for s, c in SUBJ_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    title = ax.set_title("", fontsize=10)
    unit = f"{len(trajs)} subjects" if per_subject else f"{n_sessions} sessions"
    fig.suptitle(f"{tap} · {task} · class contrast · {unit}\n"
                 f"shared 3-PC basis and shared axis range across all tasks", fontsize=10)

    n_frames = t_len + hold + orbit_frames

    def draw(i: int):
        if i < t_len:
            k, phase = i + 1, "drawing"
            lo = max(0, k - tail)            # short tail: co-movement, not spaghetti
        else:
            k, phase = t_len, "orbit"
            lo = 0                           # the whole path, once the camera takes over
        for (_, p), ln, hd in zip(trajs, lines, heads):
            ln.set_data(p[lo:k, 0], p[lo:k, 1])
            ln.set_3d_properties(p[lo:k, 2])
            hd.set_data([p[k - 1, 0]], [p[k - 1, 1]])
            hd.set_3d_properties([p[k - 1, 2]])
        # rotate slowly during the draw, then a full turn: the reader sees the shape form
        # before the camera takes over, so neither cue is hidden behind the other
        spin = i * (25.0 / max(t_len, 1)) if i < t_len else \
            25.0 + (i - t_len - hold) * (360.0 / max(orbit_frames, 1))
        ax.view_init(elev=18, azim=-60 + spin)
        title.set_text(f"frame {min(k, t_len)}/{t_len} · {phase}")
        return lines + heads

    writer, dst = _writer(out_path, fps)
    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 // fps,
                                   blit=False)
    anim.save(dst, writer=writer)
    plt.close(fig)
    print(f"[write] {dst} ({n_frames} frames, {n_frames / fps:.1f}s)", flush=True)
    return {"path": dst, "tap": tap, "task": task, "n_frames": n_frames,
            "n_sessions": n_sessions, "n_curves": len(trajs), "t_len": t_len,
            "per_subject": per_subject, "tail": tail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--taps", default="enc12")
    ap.add_argument("--tasks", default="onset,speech,delta_volume,word_index,"
                                       "word_part_speech,frame_brightness")
    ap.add_argument("--animate", default="onset,speech,frame_brightness",
                    help="which tasks get a clip. The basis is always fit on --tasks, so a "
                         "dud stays small instead of being rescaled to look alive.")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    sessions = load_all(args.red_dir)
    lobes = shared_lobes(sessions)
    taps = [t for t in args.taps.split(",") if t and any(t in s.shapes for s in sessions)]
    tasks = [t for t in args.tasks.split(",") if t]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[load] {len(sessions)} sessions, taps {taps}, lobes {lobes}", flush=True)

    for tap in taps:
        for task in [t for t in args.animate.split(",") if t]:
            dst = os.path.join(args.out_dir, f"vid_trajectory_{tap}_{task}.mp4")
            animate_task(sessions, lobes, tap, task, tasks, dst, fps=args.fps)


if __name__ == "__main__":
    main()
