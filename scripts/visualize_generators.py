#!/usr/bin/env python3
"""Visualize synthetic generator outputs as grid heatmap filmstrips.

Generates a side-by-side comparison of generator levels alongside
a real uECOG trial (if data is available), using the same visualization
style as visualize_ecog.py.

Usage:
  # Static filmstrip (PNG) — no data needed
  python scripts/visualize_generators.py

  # With real data comparison
  python scripts/visualize_generators.py --paths configs/paths.yaml --patient S14

  # Animated MP4
  python scripts/visualize_generators.py --animate --fps 20

  # Custom grid size
  python scripts/visualize_generators.py --grid 12 22
"""
from __future__ import annotations

import argparse
import logging
import sys
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, FFMpegWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dead electrode templates (shared with synthetic_pipeline.py)
DEAD_TEMPLATES = {
    (12, 22): [
        (0, 0), (0, 21), (0, 1), (0, 20),
        (11, 0), (11, 21), (11, 1), (11, 20),
    ],
    (8, 16): [],
    (8, 32): [],
    (8, 34): [(r, c) for r in range(8) for c in [0, 33]],
}


def make_diverging_cmap(vmin: float, vmax: float) -> LinearSegmentedColormap:
    """Build diverging colormap matching visualize_ecog.py style.

    Blue (negative) → white (zero) → yellow → orange → red (positive).
    """
    n_neg = max(int(256 * abs(vmin) / (abs(vmin) + vmax)), 1) if vmin < 0 else 0
    n_pos = 256 - n_neg
    colors_neg = [(0.2, 0.3, 0.7), (0.5, 0.6, 0.9), (1.0, 1.0, 1.0)]
    colors_pos = [
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.6),
        (1.0, 0.85, 0.0),
        (1.0, 0.5, 0.0),
        (0.85, 0.1, 0.0),
        (0.5, 0.0, 0.0),
    ]
    if vmin < 0:
        neg_cmap = LinearSegmentedColormap.from_list("neg", colors_neg, N=n_neg)
        pos_cmap = LinearSegmentedColormap.from_list("pos", colors_pos, N=n_pos)
        neg_colors = [neg_cmap(i / max(n_neg - 1, 1)) for i in range(n_neg)]
        pos_colors = [pos_cmap(i / max(n_pos - 1, 1)) for i in range(n_pos)]
        return LinearSegmentedColormap.from_list("neural", neg_colors + pos_colors, N=256)
    return LinearSegmentedColormap.from_list("neural", colors_pos, N=256)


def make_dead_mask(grid_h: int, grid_w: int) -> np.ndarray:
    """Create dead electrode mask for known grid layouts."""
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for r, c in DEAD_TEMPLATES.get((grid_h, grid_w), []):
        if r < grid_h and c < grid_w:
            mask[r, c] = True
    return mask


def generate_all_levels(grid_h: int, grid_w: int, T: int, seed: int = 42) -> dict:
    """Generate example outputs from all available generator levels.

    Returns dict of {name: (H, W, T) ndarray}.
    """
    from speech_decoding.pretraining.generators import GENERATORS

    results = {}
    for name, gen_cls in GENERATORS.items():
        logger.info("Generating %s (%dx%d, T=%d)", name, grid_h, grid_w, T)
        gen = gen_cls(grid_h=grid_h, grid_w=grid_w, T=T)
        data = gen.generate(seed=seed)
        # Z-score per trial (matching synthetic pipeline)
        std = data.std()
        if std > 1e-8:
            data = (data - data.mean()) / std
        results[name] = data

    return results


def load_real_trial(patient: str, bids_root: str, trial_idx: int = 0) -> np.ndarray | None:
    """Load a single real trial for comparison. Returns (H, W, T) or None."""
    try:
        from speech_decoding.data.bids_dataset import load_patient_data
        ds = load_patient_data(patient, bids_root, task="PhonemeSequence",
                               n_phons=3, tmin=-0.5, tmax=1.0)
        grid, _, _ = ds[trial_idx]
        return grid  # (H, W, T)
    except Exception as e:
        logger.warning("Could not load real data: %s", e)
        return None


def render_filmstrip(
    generators: dict,
    real_trial: np.ndarray | None,
    grid_h: int, grid_w: int,
    n_frames: int = 8,
    vmin: float = -3.0, vmax: float = 3.0,
    dpi: int = 150,
    output_path: str = "results/visualizations/generator_filmstrip.png",
):
    """Render a filmstrip showing selected frames from each generator.

    Rows = generator levels (+ optional real data).
    Columns = evenly spaced time frames.
    """
    dead_mask = make_dead_mask(grid_h, grid_w)
    cmap = make_diverging_cmap(vmin, vmax)
    cmap.set_bad(color="0.5")

    # Collect all rows
    rows = {}
    if real_trial is not None:
        rows["Real uECOG"] = real_trial
    rows.update(generators)

    n_rows = len(rows)
    fig, axes = plt.subplots(
        n_rows, n_frames,
        figsize=(n_frames * 1.8, n_rows * 1.5 + 0.8),
        dpi=dpi,
        squeeze=False,
    )

    for row_idx, (name, data) in enumerate(rows.items()):
        H, W, T = data.shape
        frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)

        for col_idx, t_idx in enumerate(frame_indices):
            ax = axes[row_idx, col_idx]
            frame = data[:, :, t_idx]
            masked_frame = np.ma.masked_array(frame, mask=dead_mask[:H, :W])
            ax.imshow(
                masked_frame, cmap=cmap, vmin=vmin, vmax=vmax,
                aspect="equal", interpolation="nearest", origin="upper",
            )
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                t_sec = t_idx / 200.0 if "Real" in name else t_idx / 20.0
                ax.set_title(f"t={t_sec:.2f}s", fontsize=7, pad=2)

        # Row label
        axes[row_idx, 0].set_ylabel(name, fontsize=9, rotation=0,
                                     ha="right", va="center", labelpad=60)

    # Shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.08)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("z-score", fontsize=9)

    fig.suptitle(
        f"Synthetic Generator Outputs ({grid_h}×{grid_w} grid, z-scored)",
        fontsize=12, fontweight="bold", y=0.98,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved filmstrip: %s", output_path)


def render_animation(
    generators: dict,
    real_trial: np.ndarray | None,
    grid_h: int, grid_w: int,
    fps: int = 20, speed: float = 0.5,
    vmin: float = -3.0, vmax: float = 3.0,
    dpi: int = 100,
    output_path: str = "results/visualizations/generator_animation.mp4",
):
    """Render an animated MP4 showing all generators side-by-side."""
    dead_mask = make_dead_mask(grid_h, grid_w)
    cmap = make_diverging_cmap(vmin, vmax)
    cmap.set_bad(color="0.5")

    rows = {}
    if real_trial is not None:
        rows["Real uECOG"] = real_trial
    rows.update(generators)

    n_rows = len(rows)
    T_min = min(d.shape[2] for d in rows.values())
    duration = T_min / 20.0  # assume 20Hz for synthetic
    n_video_frames = int(ceil(duration / speed * fps))

    fig, axes = plt.subplots(
        1, n_rows,
        figsize=(n_rows * 3.2, 3.5),
        dpi=dpi,
        squeeze=False,
    )

    heatmaps = []
    titles_ax = []
    for col_idx, (name, data) in enumerate(rows.items()):
        ax = axes[0, col_idx]
        H, W, T = data.shape
        frame = np.ma.masked_array(data[:, :, 0], mask=dead_mask[:H, :W])
        hm = ax.imshow(
            frame, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest", origin="upper",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        t = ax.set_title(name, fontsize=10, fontweight="bold")
        heatmaps.append(hm)
        titles_ax.append(t)

    # Shared colorbar
    fig.subplots_adjust(right=0.90, wspace=0.15)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(heatmaps[0], cax=cbar_ax)
    cbar.set_label("z-score", fontsize=9)

    time_text = fig.suptitle("t = 0.000s", fontsize=11, y=0.98)

    data_list = list(rows.values())

    def update(frame_idx):
        t = frame_idx * (speed / fps)
        for col_idx, data in enumerate(data_list):
            H, W, T = data.shape
            # Real data at 200Hz, synthetic at 20Hz
            sr = 200.0 if col_idx == 0 and real_trial is not None else 20.0
            sample = min(int(t * sr), T - 1)
            frame = np.ma.masked_array(data[:, :, sample], mask=dead_mask[:H, :W])
            heatmaps[col_idx].set_data(frame)
        time_text.set_text(f"t = {t:.3f}s")
        return heatmaps

    anim = FuncAnimation(fig, update, frames=n_video_frames, blit=False, interval=1000 / fps)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=2000,
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    logger.info("Saved animation: %s", output_path)


def render_statistics(
    generators: dict,
    real_trial: np.ndarray | None,
    grid_h: int, grid_w: int,
    dpi: int = 150,
    output_path: str = "results/visualizations/generator_statistics.png",
):
    """Plot temporal PSD, spatial correlation, and amplitude distribution."""
    from scipy.signal import welch

    rows = {}
    if real_trial is not None:
        rows["Real uECOG"] = real_trial
    rows.update(generators)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=dpi)
    colors = plt.cm.Set2(np.linspace(0, 1, len(rows)))

    for idx, (name, data) in enumerate(rows.items()):
        H, W, T = data.shape
        sr = 200.0 if "Real" in name else 20.0
        color = colors[idx]

        # 1. Temporal PSD (average across cells)
        ax = axes[0]
        cell_psds = []
        for r in range(min(H, 4)):
            for c in range(min(W, 8)):
                if data[r, c, :].std() > 1e-8:
                    f, psd = welch(data[r, c, :], fs=sr, nperseg=min(T, 64))
                    cell_psds.append(psd)
        if cell_psds:
            mean_psd = np.mean(cell_psds, axis=0)
            ax.semilogy(f, mean_psd, color=color, label=name, linewidth=1.5)

        # 2. Spatial correlation vs distance
        ax = axes[1]
        frame_mid = data[:, :, T // 2]
        flat = frame_mid.ravel()
        n_pts = min(len(flat), 200)  # subsample for speed
        idxs = np.random.RandomState(42).choice(len(flat), n_pts, replace=False)
        coords = np.array([(i // W, i % W) for i in idxs])
        dists, corrs = [], []
        for i in range(len(idxs)):
            for j in range(i + 1, min(i + 30, len(idxs))):
                d = np.sqrt(((coords[i] - coords[j]) ** 2).sum()) * 1.33  # mm
                c = flat[idxs[i]] * flat[idxs[j]]
                dists.append(d)
                corrs.append(c)
        if dists:
            dists, corrs = np.array(dists), np.array(corrs)
            bins = np.linspace(0, dists.max(), 12)
            bin_means = []
            bin_centers = []
            for b in range(len(bins) - 1):
                mask = (dists >= bins[b]) & (dists < bins[b + 1])
                if mask.sum() > 2:
                    bin_means.append(corrs[mask].mean())
                    bin_centers.append((bins[b] + bins[b + 1]) / 2)
            if bin_centers:
                ax.plot(bin_centers, bin_means, 'o-', color=color, label=name,
                        markersize=3, linewidth=1.5)

        # 3. Amplitude distribution
        ax = axes[2]
        flat_all = data.ravel()
        ax.hist(flat_all, bins=60, density=True, alpha=0.4, color=color,
                label=name, histtype="stepfilled", linewidth=0.8)
        ax.hist(flat_all, bins=60, density=True, alpha=0.8, color=color,
                histtype="step", linewidth=1.2)

    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD")
    axes[0].set_title("Temporal Power Spectral Density")
    axes[0].legend(fontsize=7)
    axes[0].set_xlim(0, None)

    axes[1].set_xlabel("Distance (mm)")
    axes[1].set_ylabel("Spatial Correlation")
    axes[1].set_title("Spatial Correlation vs Distance")
    axes[1].legend(fontsize=7)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.5)

    axes[2].set_xlabel("z-score")
    axes[2].set_ylabel("Density")
    axes[2].set_title("Amplitude Distribution")
    axes[2].legend(fontsize=7)
    axes[2].set_xlim(-5, 5)

    fig.suptitle("Generator Statistics Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    logger.info("Saved statistics: %s", output_path)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize synthetic generator outputs")
    p.add_argument("--paths", type=str, default=None, help="paths.yaml for real data comparison")
    p.add_argument("--patient", type=str, default="S14", help="Patient for real data")
    p.add_argument("--trial", type=int, default=0, help="Trial index for real data")
    p.add_argument("--grid", nargs=2, type=int, default=[8, 16], help="Grid H W")
    p.add_argument("--T", type=int, default=30, help="Number of frames (at 20Hz = 1.5s)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vmin", type=float, default=-3.0)
    p.add_argument("--vmax", type=float, default=3.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--animate", action="store_true", help="Generate MP4 animation")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--speed", type=float, default=0.5)
    p.add_argument("--output-dir", default="results/visualizations")
    p.add_argument("--n-frames", type=int, default=8, help="Frames in filmstrip")
    return p.parse_args()


def main():
    args = parse_args()
    grid_h, grid_w = args.grid
    output_dir = Path(args.output_dir)

    # Generate synthetic data
    generators = generate_all_levels(grid_h, grid_w, args.T, seed=args.seed)

    # Optionally load real data
    real_trial = None
    if args.paths:
        import yaml
        with open(args.paths) as f:
            paths = yaml.safe_load(f)
        real_trial = load_real_trial(args.patient, paths["bids_root"], args.trial)

    # Filmstrip
    render_filmstrip(
        generators, real_trial, grid_h, grid_w,
        n_frames=args.n_frames,
        vmin=args.vmin, vmax=args.vmax, dpi=args.dpi,
        output_path=str(output_dir / "generator_filmstrip.png"),
    )

    # Statistics comparison
    render_statistics(
        generators, real_trial, grid_h, grid_w,
        dpi=args.dpi,
        output_path=str(output_dir / "generator_statistics.png"),
    )

    # Animation (optional)
    if args.animate:
        render_animation(
            generators, real_trial, grid_h, grid_w,
            fps=args.fps, speed=args.speed,
            vmin=args.vmin, vmax=args.vmax, dpi=args.dpi // 2,
            output_path=str(output_dir / "generator_animation.mp4"),
        )

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
