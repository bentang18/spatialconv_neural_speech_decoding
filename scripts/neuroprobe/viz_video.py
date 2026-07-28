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
                 tail: int = 8, per_subject: bool = True,
                 n_pre: int | None = None) -> dict:
    """One task's contrast trajectories: draw on in time, then a full camera orbit.

    During the draw only the last ``tail`` points are shown. Twelve complete 3-D paths
    overlaid is unreadable regardless of how well they agree, so the full curve would hide
    the very thing the clip exists to show; a short tail makes co-movement of the heads the
    visible signal. The orbit phase then reveals the complete paths, which is when the whole
    shape is what matters.
    """
    per_task, comps, mu, _, lim = task_basis(sessions, lobes, tap, tasks, n_pre=n_pre)
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


def _static_axis_chrome(ax, lim: float, *, n_grid: int = 6) -> None:
    """Strip every part of the 3-D box that RE-PICKS ITS EDGE, and replace it with a triad.

    Matplotlib decides which edge of the cube to hang each axis on from the current view
    angle, and re-decides every frame. Under rotation that is not motion, it is teleporting:
    the spine, its tick marks, its tick labels and the axis label all jump to a different
    edge as the camera crosses a threshold. Nothing here is wrong, it just cannot be watched.

    So all four go. What replaces them is a triad of three short arms drawn from ONE corner
    in data coordinates -- data coordinates rotate continuously with the box, so the labels
    travel with it and stay attached to an arm the viewer can see. Free-floating text at
    (lim, 0, 0) would stop popping too, but it is centred in the other two axes and so hangs
    in mid-air belonging to nothing; the arm is what makes it a label OF something.

    The corner is chosen over the origin because under a pre-stimulus baseline the trajectory
    STARTS at the origin, so a centred triad would sit on top of the data every replay.

    Tick labels are no loss: after per-session standardization the PC units are arbitrary.

    The panes go for the same reason. Matplotlib draws the three BACK walls of the cube, so
    which walls exist changes as the camera passes a corner -- side grids appear and vanish
    mid-rotation. Rather than fight that, the box keeps ONE surface: a floor grid drawn here
    as plain line segments at z = -lim. Being ordinary data-space lines, it is always the
    floor, from every angle, and there is nothing left in the frame that can swap.
    """
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        axis.line.set_visible(False)
        # Killing the tick DASHES needs the private _axinfo, and the two obvious public
        # routes are both wrong. set_ticks([]) takes the grid with it, because grid lines are
        # drawn AT the tick locations. tick_params(length=0) sets the tick's markersize,
        # which mplot3d ignores -- it computes tick endpoints in 3-D from these two factors.
        # Hiding the tick artists does not survive either: they are regenerated during draw,
        # so the dashes come back on frame 1. Asserted rather than assumed, so a matplotlib
        # upgrade that renames this fails loudly instead of silently restoring the flicker.
        tick_info = axis._axinfo["tick"]
        assert {"inward_factor", "outward_factor"} <= tick_info.keys(), \
            f"matplotlib {matplotlib.__version__} changed _axinfo['tick']: {tick_info}"
        tick_info["inward_factor"] = 0.0
        tick_info["outward_factor"] = 0.0
        axis.pane.set_visible(False)
    for v in np.linspace(-lim, lim, n_grid + 1):
        ax.plot([v, v], [-lim, lim], [-lim, -lim], color="#ccc", lw=0.8, zorder=0)
        ax.plot([-lim, lim], [v, v], [-lim, -lim], color="#ccc", lw=0.8, zorder=0)
    arm, c = lim * 0.42, -lim
    for dx, dy, dz, name in ((arm, 0, 0, "PC1"), (0, arm, 0, "PC2"), (0, 0, arm, "PC3")):
        ax.plot([c, c + dx], [c, c + dy], [c, c + dz], color="#555", lw=1.4, alpha=0.9)
        ax.text(c + dx * 1.22, c + dy * 1.22, c + dz * 1.22, name,
                fontsize=9, color="#333", ha="center", va="center")


def loop_schedule(t_len: int, repeats: int, hz: float, speed: float):
    """Frame budget for a looping replay. Kept separate so the arithmetic is testable
    without rendering a video: at ``speed`` x the data's own ``hz``, one pass lasts
    ``t_len / (hz*speed)`` seconds and the clip is ``repeats`` of those, seamlessly."""
    fps = max(1, round(hz * speed))
    n_frames = t_len * repeats
    return fps, n_frames, n_frames / fps


def animate_loop(sessions, lobes, tap: str, task: str, tasks, out_path: str, *,
                 hz: float = 32.0, speed: float = 0.5, repeats: int = 5,
                 deg_per_replay: float = 50.0, tail: int = 8, offset: float = 0.0,
                 per_subject: bool = True, n_pre: int | None = None) -> dict:
    """ONE continuous section: the trajectory replayed ``repeats`` times while the camera
    orbits slowly throughout.

    The two-phase clip (draw, then orbit) separates the two cues so neither hides the other,
    at the cost of showing the motion exactly once. Looping trades that back: the shape
    recurs often enough to be learned, and the parallax accumulates across passes instead of
    arriving all at once at the end.

    ``deg_per_replay`` is set against the two-phase clip's DRAW rate, which turns 25 degrees
    over one pass through the data. The default here is twice that: 25 reads as nearly static
    over five passes, while a full revolution spread across the clip is too fast to read --
    the shape is still being learned while the viewpoint has already moved on. 50 gives a
    visible orbit that never outruns the trajectory.

    Same ``task_basis`` call as the static figure and the two-phase video. There is one copy
    of that computation on purpose.
    """
    per_task, comps, mu, _, lim = task_basis(sessions, lobes, tap, tasks, n_pre=n_pre)
    if task not in per_task:
        return {}
    trajs = [(s, _traj(m, comps, mu)) for s, m in per_task[task]]
    n_sessions = len(trajs)
    trajs = pool_subjects(trajs) if per_subject else [(s.subject_id, p) for s, p in trajs]
    t_len = trajs[0][1].shape[0]
    fps, n_frames, dur = loop_schedule(t_len, repeats, hz, speed)

    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    lines, heads = [], []
    for sid, p in trajs:
        col = SUBJ_COLORS.get(sid, "#888")
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=col, lw=1.0, alpha=0.13)
        lines.append(ax.plot([], [], [], color=col, lw=1.7, alpha=0.9)[0])
        heads.append(ax.plot([], [], [], color=col, marker="o", ms=6, ls="none")[0])
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    _static_axis_chrome(ax, lim)
    handles = [Line2D([], [], color=c, lw=2, label=f"S{s}") for s, c in SUBJ_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)
    title = ax.set_title("", fontsize=10)
    origin = "pre-stimulus baseline" if n_pre else "window time-mean"
    unit = f"{len(trajs)} subjects" if per_subject else f"{n_sessions} sessions"
    fig.suptitle(f"{tap} · {task} · class contrast · {unit}\n"
                 f"shared 3-PC basis · origin = {origin} · {speed:g}x · "
                 f"{repeats} replays", fontsize=10)

    def draw(i: int):
        k = i % t_len + 1                     # seamless wrap: no pause between passes
        lo = max(0, k - tail)
        for (_, p), ln, hd in zip(trajs, lines, heads):
            ln.set_data(p[lo:k, 0], p[lo:k, 1])
            ln.set_3d_properties(p[lo:k, 2])
            hd.set_data([p[k - 1, 0]], [p[k - 1, 1]])
            hd.set_3d_properties([p[k - 1, 2]])
        ax.view_init(elev=18, azim=-60 + i * (deg_per_replay / max(t_len, 1)))
        title.set_text(f"t = {offset + (k - 1) / hz:+.2f} s   ·   "
                       f"replay {i // t_len + 1}/{repeats}")
        return lines + heads

    writer, dst = _writer(out_path, fps)
    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 // fps,
                                   blit=False)
    anim.save(dst, writer=writer)
    plt.close(fig)
    total_deg = deg_per_replay * repeats
    print(f"[write] {dst} ({n_frames} frames @ {fps}fps = {dur:.1f}s, "
          f"{repeats}x{t_len} @ {speed:g}x, {total_deg:g} deg total)", flush=True)
    return {"path": dst, "tap": tap, "task": task, "n_frames": n_frames, "fps": fps,
            "duration_s": dur, "t_len": t_len, "repeats": repeats, "speed": speed,
            "deg_per_replay": deg_per_replay, "total_deg": total_deg,
            "n_sessions": n_sessions, "n_curves": len(trajs),
            "n_pre": n_pre, "offset_s": offset}


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
    # Same origin as the figures and the page; see viz_common.center_per_session.
    ap.add_argument("--n-pre", type=int, default=0,
                    help="pre-stimulus frames to baseline against; 0 = time-mean origin")
    ap.add_argument("--loop", action="store_true",
                    help="one continuous section: replay the trajectory while orbiting, "
                         "instead of the two-phase draw-then-orbit clip")
    ap.add_argument("--speed", type=float, default=0.5,
                    help="playback rate against real time (--loop only); 0.5 = half speed")
    ap.add_argument("--repeats", type=int, default=5, help="replays per clip (--loop only)")
    ap.add_argument("--deg-per-replay", type=float, default=50.0,
                    help="camera degrees per replay (--loop only); the two-phase clip's "
                         "draw-phase rate is 25, so the default is twice that")
    ap.add_argument("--hz", type=float, default=32.0, help="frame rate of the data itself")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="seconds of the first frame; only a label, so a wrong value "
                         "mislabels every frame and nothing crashes")
    args = ap.parse_args()

    sessions = load_all(args.red_dir)
    lobes = shared_lobes(sessions)
    taps = [t for t in args.taps.split(",") if t and any(t in s.shapes for s in sessions)]
    tasks = [t for t in args.tasks.split(",") if t]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[load] {len(sessions)} sessions, taps {taps}, lobes {lobes}", flush=True)

    for tap in taps:
        for task in [t for t in args.animate.split(",") if t]:
            if args.loop:
                dst = os.path.join(args.out_dir, f"vid_loop_{tap}_{task}.mp4")
                animate_loop(sessions, lobes, tap, task, tasks, dst, hz=args.hz,
                             speed=args.speed, repeats=args.repeats,
                             deg_per_replay=args.deg_per_replay,
                             offset=args.offset, n_pre=args.n_pre or None)
            else:
                dst = os.path.join(args.out_dir, f"vid_trajectory_{tap}_{task}.mp4")
                animate_task(sessions, lobes, tap, task, tasks, dst, fps=args.fps,
                             n_pre=args.n_pre or None)


if __name__ == "__main__":
    main()
