"""PC-RGB painted onto each subject's electrode cloud in their own native space.

The DINOv3 panel figure asks whether the encoder's channel space is shared across subjects
when the rows are LOBES. This asks the same question with the rows left where they actually
are: individual contacts, at their own coordinates. Same three principal components, same
colour stretch, six different heads. If the space is shared, the same anatomy comes out the
same colour in subjects that were never aligned to each other.

What is deliberately NOT done here:

  * No template warp. Coordinates are native ``depth-wm.csv`` (L, I, P) per subject. There
    is no common brain in this figure and none is needed -- the claim is about the FEATURES
    agreeing, and warping anatomy first would let a template do work the encoder is being
    credited for. (MNI is banned in this project for exactly this class of reason.)
  * No cortical surface mesh. BrainTreebank ships no surface here, so drawing one would
    mean drawing somebody else's brain behind these electrodes. The cloud is the cloud.
  * No per-subject colour stretch. One percentile stretch over the pooled projection, so
    the same colour means the same thing in every head. Per-panel stretching would make
    disagreeing subjects look identical.

Colour is the class CONTRAST (class 1 minus class 0), not a single condition. Single-
condition trial averages are dominated by a large condition-INDEPENDENT evoked profile that
is shared across subjects whether or not the task decodes, so painting those would look like
a strong result for a task that sits at chance.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from scripts.neuroprobe.viz_common import pca_basis, to_rgb


def load_elec(path: str, tap: str, task: str):
    """One per-electrode reduction -> (contacts, T, C) class contrast, standardized."""
    z = np.load(path, allow_pickle=False)
    key = f"{tap}/shape"
    if key not in z.files:
        return None
    n_r, t, c = (int(v) for v in z[key])
    mu = z[f"{tap}/col_sum"].reshape(n_r, t, c).mean(axis=(0, 1))
    var = z[f"{tap}/col_sq"].reshape(n_r, t, c).mean(axis=(0, 1)) - mu ** 2
    sd = np.sqrt(np.maximum(var, 0.0)) + 1e-8
    a = z.get(f"{tap}/{task}/c1/all")
    b = z.get(f"{tap}/{task}/c0/all")
    if a is None or b is None:
        return None
    # standardization is linear, so it commutes with the subtraction; the channel MEAN
    # cancels in the contrast and only the per-channel gain survives
    return {
        "subject_id": int(z["subject_id"]), "trial_id": int(z["trial_id"]),
        "parcel": np.asarray(z["parcel_canon"], dtype=np.int64),
        "x": ((a - b) / sd).astype(np.float64),
    }


def build(red_dir: str, coords_npz: str, tap: str, task: str):
    """Shared 3-PC basis over pooled (contact x time) tokens -> per-session RGB."""
    zc = np.load(coords_npz, allow_pickle=True)
    sessions = []
    for p in sorted(glob.glob(os.path.join(red_dir, "red_s*_t*_*.npz"))):
        s = load_elec(p, tap, task)
        if s is None:
            continue
        key = f"s{s['subject_id']}_t{s['trial_id']}"
        if f"{key}/coords" not in zc.files:
            print(f"[skip] {key}: no coordinates", flush=True)
            continue
        s["coords"] = np.asarray(zc[f"{key}/coords"], dtype=np.float64)
        assert s["coords"].shape[0] == s["x"].shape[0], (
            f"{key}: {s['coords'].shape[0]} coords vs {s['x'].shape[0]} feature rows")
        assert len(s["parcel"]) == s["x"].shape[0]
        sessions.append(s)
    assert sessions, f"no usable sessions in {red_dir}"

    # one scale per session, so the pooled basis is not simply the loudest subject's basis
    for s in sessions:
        n = float(np.linalg.norm(s["x"]))
        s["x"] = s["x"] / n if n > 0 else s["x"]

    stack = np.concatenate([s["x"].reshape(-1, s["x"].shape[-1]) for s in sessions], axis=0)
    comps, mu, evr = pca_basis(stack, k=3)
    proj = [(s["x"].reshape(-1, s["x"].shape[-1]) - mu) @ comps.T for s in sessions]
    rgb = to_rgb(np.concatenate(proj, axis=0))          # ONE stretch over every subject
    off = 0
    for s, p in zip(sessions, proj):
        n = p.shape[0]
        s["rgb"] = rgb[off:off + n].reshape(s["x"].shape[0], s["x"].shape[1], 3)
        # the raw projection as well: ``rgb`` is percentile-stretched, so ranking contacts by
        # colour would rank them by where the stretch happened to put them, not by how
        # strongly the contrast actually loads
        s["proj"] = p.reshape(s["x"].shape[0], s["x"].shape[1], 3)
        off += n
    return sessions, evr


def anatomy_of_extremes(sessions, frame: int, *, q: float = 0.9) -> dict:
    """Which lobes carry the strongest class contrast, per subject, at one frame.

    The picture shows a warm cluster in most heads and it is tempting to call that "the same
    anatomy". This is the check that makes it a claim instead of an impression: rank each
    subject's contacts by projection magnitude, take the top decile, and report where they
    sit. Nothing here averages across subjects, so a lobe only looks common if it really is.
    """
    from scripts.neuroprobe.viz_common import lobe_of

    per_subject: dict[int, dict[str, int]] = {}
    for s in _one_per_subject(sessions):
        mag = np.linalg.norm(s["proj"][:, frame, :], axis=-1)
        top = mag >= np.quantile(mag, q)
        lobes = lobe_of(s["parcel"][top], pool_hemi=True)
        counts: dict[str, int] = {}
        for lb in lobes:
            counts[lb] = counts.get(lb, 0) + 1
        per_subject[s["subject_id"]] = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    shared = set.intersection(*(set(v) for v in per_subject.values())) if per_subject else set()
    return {"per_subject": per_subject, "shared": sorted(shared)}


def _one_per_subject(sessions):
    seen, out = set(), []
    for s in sorted(sessions, key=lambda d: (d["subject_id"], d["trial_id"])):
        if s["subject_id"] not in seen:
            seen.add(s["subject_id"])
            out.append(s)
    return out


# (L, I, P) are Left / Inferior / Posterior, so every axis points the "wrong" way for a
# reader: negate to get the conventional anterior-right, superior-up, right-right views.
VIEWS = (("sagittal", 2, 1, "posterior -> anterior", "inferior -> superior"),
         ("coronal", 0, 1, "left -> right", "inferior -> superior"),
         ("axial", 0, 2, "left -> right", "posterior -> anterior"))


def display_span(sessions) -> float:
    """One half-width in mm for every panel, so the heads are drawn at the SAME scale.

    Letting matplotlib autoscale each subplot to its own cloud makes a sparse montage fill
    the box and a dense one shrink, which reads as an anatomical difference that is really
    just an axis choice. Panels are centred on each subject's own centroid -- a translation
    for display only, not a registration; these stay native spaces.
    """
    return max(float(np.abs(s["coords"] - s["coords"].mean(axis=0)).max()) for s in sessions)


def figure_views(sessions, frame: int, times, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = _one_per_subject(sessions)
    lim = display_span(subs)
    fig, axes = plt.subplots(len(subs), len(VIEWS),
                             figsize=(3.1 * len(VIEWS), 2.9 * len(subs)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(subs):
        c3 = s["coords"] - s["coords"].mean(axis=0)
        for k, (name, ix, iy, xl, yl) in enumerate(VIEWS):
            ax = axes[r, k]
            ax.scatter(-c3[:, ix], -c3[:, iy], c=s["rgb"][:, frame, :], s=26,
                       edgecolors="k", linewidths=0.25)
            ax.set_aspect("equal")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10)
            if r == len(subs) - 1:
                ax.set_xlabel(xl, fontsize=7)
            if k == 0:
                ax.set_ylabel(f"S{s['subject_id']}\n{yl}", fontsize=8)
    fig.suptitle(f"t = {times[frame]:+.2f} s   (colour = 3 shared PCs of the class contrast)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[write] {out_path}", flush=True)


def figure_time(sessions, frames, times, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = _one_per_subject(sessions)
    lim = display_span(subs)
    fig, axes = plt.subplots(len(subs), len(frames),
                             figsize=(1.9 * len(frames), 2.0 * len(subs)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(subs):
        c3 = s["coords"] - s["coords"].mean(axis=0)
        for k, f in enumerate(frames):
            ax = axes[r, k]
            ax.scatter(-c3[:, 2], -c3[:, 1], c=s["rgb"][:, f, :], s=16,
                       edgecolors="k", linewidths=0.2)
            ax.set_aspect("equal")
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{times[f]:+.2f}s", fontsize=9)
            if k == 0:
                ax.set_ylabel(f"S{s['subject_id']}", fontsize=9)
    fig.suptitle("sagittal, colour = 3 shared PCs of the class contrast", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[write] {out_path}", flush=True)


def figure_3d(sessions, frame: int, times, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = _one_per_subject(sessions)
    lim = display_span(subs)
    cols = 3
    rows = (len(subs) + cols - 1) // cols
    fig = plt.figure(figsize=(4.0 * cols, 3.6 * rows))
    for i, s in enumerate(subs):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        c3 = s["coords"] - s["coords"].mean(axis=0)
        # zs by keyword: the 2D signature reads a third positional as the marker size
        ax.scatter(-c3[:, 2], -c3[:, 0], zs=-c3[:, 1], c=s["rgb"][:, frame, :], s=22,
                   edgecolors="k", linewidths=0.2, depthshade=False)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        # a shared CUBE is what equal-scale means in 3-D, but these clouds are anisotropic
        # (one subject spans 34/17/53 mm), so the cube is mostly empty. zoom crops the
        # viewport without touching the data limits -- the scale stays shared.
        ax.set_box_aspect((1, 1, 1), zoom=1.35)
        ax.set_xlabel("A", fontsize=7)
        ax.set_ylabel("R", fontsize=7)
        ax.set_zlabel("S", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f"S{s['subject_id']}", fontsize=10)
        ax.view_init(elev=14, azim=-62)
    fig.suptitle(f"native-space electrode clouds, t = {times[frame]:+.2f} s", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[write] {out_path}", flush=True)


def animate_brain(sessions, times, out_path: str, *, fps: int = 12,
                  orbit_frames: int = 90, hold: int = 6) -> dict:
    """Six native-space clouds, colour advancing in time, then a camera orbit.

    The still can only show one instant, and the interesting claim is that the SAME colour
    appears in the SAME anatomy at the SAME moment in heads that were never aligned. That is
    a claim about time, so it wants a time axis. The orbit then rules out the obvious
    objection to any 3-D scatter -- that agreement is an artefact of one projection.

    Every frame re-scatters rather than mutating face colours: Path3DCollection's colour
    handling depends on depth-sort state, and a frame that silently kept the previous
    colours would be indistinguishable from a correct one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    from scripts.neuroprobe.viz_video import _writer

    subs = _one_per_subject(sessions)
    lim = display_span(subs)
    n_t = subs[0]["rgb"].shape[1]
    cols = 3
    rows = (len(subs) + cols - 1) // cols
    fig = plt.figure(figsize=(3.7 * cols, 3.4 * rows))
    axes, clouds = [], []
    for i, s in enumerate(subs):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        c3 = s["coords"] - s["coords"].mean(axis=0)
        clouds.append((-c3[:, 2], -c3[:, 0], -c3[:, 1]))
        axes.append(ax)
    title = fig.suptitle("", fontsize=11)
    n_frames = n_t + hold + orbit_frames

    def draw(i: int):
        if i < n_t:
            f, phase = i, "time"
            spin = i * (25.0 / max(n_t, 1))
        else:
            f, phase = n_t - 1, "orbit"
            spin = 25.0 + (i - n_t - hold) * (360.0 / max(orbit_frames, 1))
        for ax, s, (x, y, zc) in zip(axes, subs, clouds):
            ax.clear()
            ax.scatter(x, y, zs=zc, c=s["rgb"][:, f, :], s=20, edgecolors="k",
                       linewidths=0.2, depthshade=False)
            # inside draw(): ax.clear() drops the limits, so setting them once outside would
            # silently give every frame a per-subject autoscale again
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_box_aspect((1, 1, 1), zoom=1.35)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xlabel("A", fontsize=7)
            ax.set_ylabel("R", fontsize=7)
            ax.set_zlabel("S", fontsize=7)
            ax.set_title(f"S{s['subject_id']}", fontsize=9)
            ax.view_init(elev=14, azim=-62 + spin)
        title.set_text(f"native-space electrode clouds · colour = 3 shared PCs of the class "
                       f"contrast\nt = {times[f]:+.2f} s · {phase}")
        return []

    writer, dst = _writer(out_path, fps)
    anim = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 // fps,
                                   blit=False)
    anim.save(dst, writer=writer)
    plt.close(fig)
    print(f"[write] {dst} ({n_frames} frames, {n_frames / fps:.1f}s)", flush=True)
    return {"path": dst, "n_frames": n_frames, "n_subjects": len(subs), "t_len": n_t}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--red-dir", required=True, help="per-electrode reductions")
    ap.add_argument("--coords", required=True, help="npz from viz_coords_dump.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tap", default="enc12_elec")
    ap.add_argument("--task", default="onset")
    ap.add_argument("--rate", type=float, default=32.0)
    ap.add_argument("--offset", type=float, default=0.0, help="seconds of the first frame")
    ap.add_argument("--frame", type=int, default=None, help="default: peak colour spread")
    ap.add_argument("--video", action="store_true", help="also render the orbiting clip")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    sessions, evr = build(args.red_dir, args.coords, args.tap, args.task)
    n_t = sessions[0]["x"].shape[1]
    times = args.offset + np.arange(n_t) / args.rate
    print(f"[check] {len(sessions)} sessions, {len(_one_per_subject(sessions))} subjects, "
          f"T={n_t}, evr={np.round(evr, 3).tolist()}", flush=True)

    if args.frame is None:
        # the frame where subjects disagree least about nothing: pick maximum colour spread,
        # i.e. where the contrast is actually doing something rather than sitting at baseline
        spread = np.array([np.mean([s["rgb"][:, f, :].std(axis=0).mean() for s in sessions])
                           for f in range(n_t)])
        frame = int(spread.argmax())
        print(f"[check] frame auto = {frame} (t={times[frame]:+.2f}s, spread={spread[frame]:.4f})",
              flush=True)
    else:
        frame = args.frame

    ext = anatomy_of_extremes(sessions, frame)
    for sid, counts in ext["per_subject"].items():
        top = ", ".join(f"{k} {v}" for k, v in list(counts.items())[:4])
        print(f"[check] S{sid} strongest-decile lobes: {top}", flush=True)
    print(f"[check] lobes in EVERY subject's strongest decile: "
          f"{ext['shared'] or 'NONE'}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.tap}_{args.task}"
    figure_views(sessions, frame, times, os.path.join(args.out_dir, f"brain_views_{stem}.png"))
    figure_3d(sessions, frame, times, os.path.join(args.out_dir, f"brain_3d_{stem}.png"))
    frames = np.linspace(0, n_t - 1, 6).astype(int).tolist()
    figure_time(sessions, frames, times, os.path.join(args.out_dir, f"brain_time_{stem}.png"))
    if args.video:
        animate_brain(sessions, times,
                      os.path.join(args.out_dir, f"vid_brain_{stem}.mp4"), fps=args.fps)


if __name__ == "__main__":
    main()
