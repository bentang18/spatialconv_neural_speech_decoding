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
        off += n
    return sessions, evr


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


def figure_views(sessions, frame: int, times, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subs = _one_per_subject(sessions)
    fig, axes = plt.subplots(len(subs), len(VIEWS),
                             figsize=(3.1 * len(VIEWS), 2.9 * len(subs)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(subs):
        c3 = s["coords"]
        for k, (name, ix, iy, xl, yl) in enumerate(VIEWS):
            ax = axes[r, k]
            ax.scatter(-c3[:, ix], -c3[:, iy], c=s["rgb"][:, frame, :], s=26,
                       edgecolors="k", linewidths=0.25)
            ax.set_aspect("equal")
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
    fig, axes = plt.subplots(len(subs), len(frames),
                             figsize=(1.9 * len(frames), 2.0 * len(subs)))
    axes = np.atleast_2d(axes)
    for r, s in enumerate(subs):
        c3 = s["coords"]
        for k, f in enumerate(frames):
            ax = axes[r, k]
            ax.scatter(-c3[:, 2], -c3[:, 1], c=s["rgb"][:, f, :], s=16,
                       edgecolors="k", linewidths=0.2)
            ax.set_aspect("equal")
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
    cols = 3
    rows = (len(subs) + cols - 1) // cols
    fig = plt.figure(figsize=(4.0 * cols, 3.6 * rows))
    for i, s in enumerate(subs):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        c3 = s["coords"]
        # zs by keyword: the 2D signature reads a third positional as the marker size
        ax.scatter(-c3[:, 2], -c3[:, 0], zs=-c3[:, 1], c=s["rgb"][:, frame, :], s=22,
                   edgecolors="k", linewidths=0.2, depthshade=False)
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

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{args.tap}_{args.task}"
    figure_views(sessions, frame, times, os.path.join(args.out_dir, f"brain_views_{stem}.png"))
    figure_3d(sessions, frame, times, os.path.join(args.out_dir, f"brain_3d_{stem}.png"))
    frames = np.linspace(0, n_t - 1, 6).astype(int).tolist()
    figure_time(sessions, frames, times, os.path.join(args.out_dir, f"brain_time_{stem}.png"))


if __name__ == "__main__":
    main()
