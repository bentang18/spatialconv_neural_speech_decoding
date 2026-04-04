#!/usr/bin/env python3
"""Visualize uECOG grid activity synchronized with speech audio.

Generates MP4 videos showing a 2D electrode heatmap (HGA z-score)
evolving over time, with the patient's speech audio embedded.

Usage:
  # Single trial, focused on speech production
  python scripts/visualize_ecog.py --patient S14 --trial 0

  # Full epoch window (includes resting baseline)
  python scripts/visualize_ecog.py --patient S14 --trial 0 --mode full

  # 10 consecutive trials concatenated into one long video
  python scripts/visualize_ecog.py --patient S14 --trial 0 --concat 10

  # All trials
  python scripts/visualize_ecog.py --patient S14 --trial all --speed 1.0
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import tempfile
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for video rendering

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.io import wavfile

from speech_decoding.data.audio_features import (
    extract_audio_segment,
    load_patient_audio,
    load_phoneme_timing,
)
from speech_decoding.data.bids_dataset import load_patient_data
from speech_decoding.data.grid import load_grid_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Phoneme annotation colors (consonants vs vowels)
PHON_COLORS = {
    "b": "#1f77b4", "p": "#2ca02c", "g": "#d62728", "k": "#9467bd", "v": "#8c564b",
    "a": "#ff7f0e", "ae": "#e377c2", "i": "#17becf", "u": "#bcbd22",
}
DEFAULT_PHON_COLOR = "#aaaaaa"


def build_atempo_filter(speed: float) -> str:
    """Build ffmpeg atempo filter chain for arbitrary speed factors.

    atempo accepts 0.5-100.0 per filter; chain for values outside this range.
    """
    if speed >= 0.5:
        return f"atempo={speed}"
    parts = []
    remaining = speed
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining}")
    return ",".join(parts)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Normalize audio to target peak amplitude."""
    peak = np.abs(audio).max()
    return audio / peak * target_peak if peak > 0 else audio


def save_video_with_audio(
    anim, fig, audio_seg, audio_sr, output_path, fps, speed,
    volume=1.0, bitrate=3000,
):
    """Save animation as MP4, optionally muxing audio via ffmpeg."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps, codec="libx264", bitrate=bitrate,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    if audio_seg is None:
        anim.save(str(output_path), writer=writer)
        plt.close(fig)
        logger.info("Saved (no audio): %s", output_path)
        return

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_v:
        tmp_video = tmp_v.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_a:
        tmp_wav = tmp_a.name

    try:
        anim.save(tmp_video, writer=writer)
        plt.close(fig)

        audio_norm = normalize_audio(audio_seg)
        audio_int16 = np.clip(audio_norm * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(tmp_wav, audio_sr, audio_int16)

        atempo = build_atempo_filter(speed)
        afilter = f"{atempo},volume={volume}" if volume != 1.0 else atempo
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_video, "-i", tmp_wav,
            "-filter:a", afilter,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-shortest", str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        logger.info("Saved: %s", output_path)
    finally:
        Path(tmp_video).unlink(missing_ok=True)
        Path(tmp_wav).unlink(missing_ok=True)


def load_audio_events(bids_root, subject):
    """Load per-trial speech onset times from the audio events TSV.

    These timestamps are on the AUDIO RECORDING clock, which may differ
    from the neural recording clock used by the phoneme CSV.
    """
    tsv_path = (
        Path(bids_root) / "derivatives" / "audio" / f"sub-{subject}" / "events"
        / f"sub-{subject}_task-phoneme_acq-01_run-01_desc-production_events.tsv"
    )
    if not tsv_path.exists():
        logger.warning("Audio events TSV not found: %s", tsv_path)
        return []
    events = []
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            events.append({
                "onset": float(row["onset"]),
                "duration": float(row["duration"]),
                "trial_type": row["trial_type"],
            })
    return events


def estimate_clock_offset(timing_list, audio_events):
    """Estimate clock relationship between neural and audio recordings.

    Fits a linear model: audio_time = neural_time + (drift * neural_time + intercept)
    to handle crystal oscillator drift between the two recording devices.

    Returns a callable clock_fn(neural_time) -> audio_offset that gives the
    offset to add to a neural timestamp to get the corresponding audio timestamp.
    """
    # Default: no correction
    def _zero(t):
        return 0.0

    if not audio_events or not timing_list:
        return _zero

    neural_times = sorted(t.response_onset for t in timing_list)
    audio_times = sorted(e["onset"] for e in audio_events)

    # Rough offset from first events
    rough = audio_times[0] - neural_times[0]

    # Match audio events to nearest neural event (adjusted by rough offset)
    neural_arr = np.array(neural_times)
    pairs = []  # (neural_time, offset)
    for at in audio_times:
        dists = np.abs((at - rough) - neural_arr)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 2.0:
            pairs.append((neural_arr[min_idx], at - neural_arr[min_idx]))

    if len(pairs) < 3:
        if abs(rough) > 10.0:
            logger.warning("Using rough constant clock offset: %.1fs", rough)
            c = rough
            return lambda t: c
        return _zero

    nt = np.array([p[0] for p in pairs])
    offsets = np.array([p[1] for p in pairs])

    # Fit linear model: offset = drift * time + intercept
    t0 = nt[0]
    coeffs = np.polyfit(nt - t0, offsets, 1)
    drift, intercept = float(coeffs[0]), float(coeffs[1])
    residual_std = float(np.std(offsets - np.polyval(coeffs, nt - t0)))

    logger.info(
        "Clock model: offset = %.4f ms/s * t + %.3fs "
        "(drift=%.3fs over session, residual_std=%.3fs, %d/%d pairs)",
        drift * 1000, intercept,
        drift * (nt[-1] - nt[0]), residual_std,
        len(pairs), len(audio_events),
    )

    def clock_fn(neural_time):
        return drift * (neural_time - t0) + intercept

    return clock_fn


def match_trial_to_audio_event(timing_info, audio_events, clock_fn):
    """Find the audio event matching a neural trial by time proximity.

    Uses clock_fn(neural_time) to predict the expected audio time, then
    finds the nearest audio event. Returns the per-trial audio-neural
    offset, or None if no match within 2s.
    """
    if not audio_events or timing_info is None:
        return None

    expected_audio_time = timing_info.response_onset + clock_fn(timing_info.response_onset)

    # Find nearest audio event in time
    best = min(audio_events, key=lambda ae: abs(ae["onset"] - expected_audio_time))
    residual = abs(best["onset"] - expected_audio_time)

    # Should be within 2s after clock correction
    if residual > 2.0:
        return None

    return best["onset"] - timing_info.response_onset


def load_stimulus_token(bids_root, syllable):
    """Load a clean stimulus token WAV for a given syllable."""
    token_path = Path(bids_root) / "stimuli" / "all_tokens" / f"{syllable}.wav"
    if not token_path.exists():
        return None, None
    sr, data = wavfile.read(str(token_path))
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sr


def build_trial_audio(bids_root, timing_info, tmin, tmax, target_sr=16000):
    """Build audio for a trial using the clean stimulus token.

    Places the token at the first phoneme onset in the display timeline.
    Resamples token to target_sr if needed.
    """
    duration = tmax - tmin
    n_samples = int(duration * target_sr)
    audio_out = np.zeros(n_samples, dtype=np.float32)

    if timing_info is None:
        return audio_out, target_sr

    token, token_sr = load_stimulus_token(bids_root, timing_info.syllable)
    if token is None:
        return audio_out, target_sr

    # Resample token to target_sr if needed
    if token_sr != target_sr:
        from scipy.signal import resample
        n_target = int(len(token) * target_sr / token_sr)
        token = resample(token, n_target).astype(np.float32)

    # Place token at the first phoneme onset in the display timeline.
    # phoneme_intervals are relative to response_onset (t=0 in display).
    first_phon_onset = timing_info.phoneme_intervals[0][0]
    insert_sample = int((first_phon_onset - tmin) * target_sr)
    insert_sample = max(0, insert_sample)

    end_sample = min(insert_sample + len(token), n_samples)
    token_len = end_sample - insert_sample
    if token_len > 0:
        audio_out[insert_sample:end_sample] = token[:token_len] * 0.9

    return audio_out, target_sr


def load_all_data(subject, bids_root, tmin, tmax):
    """Load neural data, audio, timing, and grid info for a patient."""
    ds = load_patient_data(
        subject, bids_root, task="PhonemeSequence", n_phons=3,
        tmin=tmin, tmax=tmax,
    )
    timing = load_phoneme_timing(subject, bids_root)
    # Raw audio for waveform display (has actual mic recording)
    audio, audio_sr = load_patient_audio(subject, bids_root, desc="raw")
    audio_events = load_audio_events(bids_root, subject)

    # Grid dead mask
    electrodes_tsv = (
        Path(bids_root) / f"sub-{subject}" / "ieeg"
        / f"sub-{subject}_acq-01_space-ACPC_electrodes.tsv"
    )
    grid_info = load_grid_mapping(electrodes_tsv)

    return ds, timing, audio, audio_sr, grid_info, audio_events


def render_single_trial_3d(
    grid_frame, audio_seg, audio_sr, mic_waveform,
    timing_info, dead_mask,
    tmin, tmax, patient, trial_idx,
    speed, fps, vmin, vmax, cmap_name, dpi, no_audio,
    output_path, threshold=0, volume=1.0,
):
    """Render one trial as 3D column visualization with audio."""
    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    H, W, T_neural = grid_frame.shape
    duration = tmax - tmin
    neural_sr = 200.0
    n_video_frames = int(ceil(duration / speed * fps))

    # Diverging colormap: blue (below threshold) → white (0 = threshold) → yellow → red (high activation)
    # Anchored so white is at 0 in the shifted data
    n_neg = max(int(256 * abs(vmin) / (abs(vmin) + vmax)), 1) if vmin < 0 else 0
    n_pos = 256 - n_neg
    colors_neg = [(0.2, 0.3, 0.7), (0.5, 0.6, 0.9), (1.0, 1.0, 1.0)]  # blue → white
    colors_pos = [
        (1.0, 1.0, 1.0),   # white — ground floor / threshold
        (1.0, 1.0, 0.6),   # light yellow
        (1.0, 0.85, 0.0),  # yellow
        (1.0, 0.5, 0.0),   # orange
        (0.85, 0.1, 0.0),  # red
        (0.5, 0.0, 0.0),   # dark red
    ]
    if vmin < 0:
        neg_cmap = LinearSegmentedColormap.from_list("neg", colors_neg, N=n_neg)
        pos_cmap = LinearSegmentedColormap.from_list("pos", colors_pos, N=n_pos)
        neg_colors = [neg_cmap(i / max(n_neg - 1, 1)) for i in range(n_neg)]
        pos_colors = [pos_cmap(i / max(n_pos - 1, 1)) for i in range(n_pos)]
        cmap_3d = LinearSegmentedColormap.from_list("neural", neg_colors + pos_colors, N=256)
    else:
        cmap_3d = LinearSegmentedColormap.from_list("neural", colors_pos, N=256)

    # Normalize audio
    mic_norm = mic_waveform.copy()
    mic_peak = np.abs(mic_norm).max()
    if mic_peak > 0:
        mic_norm = mic_norm / mic_peak

    # Bar positions — row/col on X/Y ground plane, height on Z
    xpos, ypos = np.meshgrid(np.arange(W), np.arange(H))
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    # Mask out dead electrodes
    alive = ~dead_mask.ravel()
    xpos_alive = xpos[alive]
    ypos_alive = ypos[alive]
    dx = dy = 0.8  # bar width

    fig = plt.figure(figsize=(18, 12), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.2, 0.8], hspace=0.15)
    ax_wave = fig.add_subplot(gs[0])
    ax_3d = fig.add_subplot(gs[1], projection="3d")

    # --- Top panel: waveform ---
    t_mic = np.linspace(tmin, tmax, len(mic_waveform))
    ax_wave.plot(t_mic, mic_norm, color="0.3", linewidth=0.5, alpha=0.8)
    ax_wave.set_xlim(tmin, tmax)
    ax_wave.set_ylabel("Mic")
    ax_wave.axvline(0, color="green", linestyle="--", linewidth=1, alpha=0.7)
    if timing_info is not None:
        for (on, off), phon in zip(timing_info.phoneme_intervals, timing_info.phonemes):
            color = PHON_COLORS.get(phon, DEFAULT_PHON_COLOR)
            ax_wave.axvspan(on, off, alpha=0.25, color=color)
            ax_wave.text((on + off) / 2, 0.85, f"/{phon}/",
                         ha="center", va="top", fontsize=9, fontweight="bold",
                         color=color, transform=ax_wave.get_xaxis_transform())
    playhead = ax_wave.axvline(tmin, color="red", linewidth=1.5, zorder=5)

    syllable = timing_info.syllable if timing_info else "?"
    phonemes_str = " ".join(f"/{p}/" for p in timing_info.phonemes) if timing_info else ""
    title = fig.suptitle(
        f"sub-{patient} | trial {trial_idx} | \"{syllable}\" ({phonemes_str}) | t = {tmin:.3f}s",
        fontsize=12, fontweight="bold",
    )

    # Threshold offset for relabeling z-axis to original SD values
    # If threshold was applied, displayed 0 = ±threshold in original,
    # displayed +1 = threshold+1 original, etc.
    thresh = threshold if threshold > 0 else 0

    def setup_3d_axes(ax):
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_zlim(vmin, vmax)
        ax.set_xlabel("Column", fontsize=9, labelpad=8)
        ax.set_ylabel("Row", fontsize=9, labelpad=8)
        ax.set_zlabel("HGA (SD)", fontsize=9, labelpad=8)
        ax.view_init(elev=35, azim=-60)
        # Force z-ticks every 1 unit, relabel to original SD values
        zticks = np.arange(int(np.ceil(vmin)), int(np.floor(vmax)) + 1, 1)
        ax.set_zticks(zticks)
        ax.tick_params(axis='z', which='major', pad=2, labelsize=7)
        if thresh > 0:
            labels = []
            for v in zticks:
                if v == 0:
                    labels.append(f"[{-thresh:.0f},{thresh:.0f}]")
                elif v > 0:
                    labels.append(f"{v + thresh:.0f}")
                else:
                    labels.append(f"{v - thresh:.0f}")
            ax.set_zticklabels(labels)
        # Prevent matplotlib from auto-culling ticks
        from matplotlib.ticker import FixedLocator
        ax.zaxis.set_major_locator(FixedLocator(zticks))
        # Force column and row axes to tick every 1
        ax.xaxis.set_major_locator(FixedLocator(np.arange(W)))
        ax.yaxis.set_major_locator(FixedLocator(np.arange(H)))
        ax.tick_params(axis='x', which='major', pad=2, labelsize=7)
        ax.tick_params(axis='y', which='major', pad=2, labelsize=7)

    setup_3d_axes(ax_3d)
    ax_3d.set_box_aspect([W / 8, H / 8, 1])

    def update(frame_idx):
        t = tmin + frame_idx * (speed / fps)
        t = min(t, tmax)
        sample = int(round((t - tmin) * neural_sr))
        sample = min(sample, T_neural - 1)

        frame = grid_frame[:, :, sample]
        heights = frame.ravel()[alive]
        heights_clipped = np.clip(heights, vmin, vmax)

        # Clear and redraw bars
        ax_3d.cla()
        setup_3d_axes(ax_3d)

        # Color based on value
        norm_vals = (heights_clipped - vmin) / (vmax - vmin)
        bar_colors = cmap_3d(norm_vals)

        # Draw bars on X/Y ground plane, height along Z
        # Only draw bars that are actually above/below ground (skip zero-height)
        z_bases = np.where(heights_clipped >= 0, 0, heights_clipped)
        dz = np.abs(heights_clipped)
        has_height = dz > 0.01  # skip flat bars

        if has_height.any():
            ax_3d.bar3d(
                xpos_alive[has_height], ypos_alive[has_height], z_bases[has_height],
                dx, dx, dz[has_height],
                color=bar_colors[has_height], alpha=0.9, shade=True,
            )

        # Draw zero plane on the ground
        ax_3d.plot_surface(
            np.array([[-.5, W-.5], [-.5, W-.5]]),
            np.array([[-.5, -.5], [H-.5, H-.5]]),
            np.array([[0, 0], [0, 0]]),
            alpha=0.08, color="gray",
        )

        playhead.set_xdata([t, t])
        title.set_text(
            f"sub-{patient} | trial {trial_idx} | \"{syllable}\" ({phonemes_str}) | t = {t:.3f}s"
        )
        return []

    anim = FuncAnimation(fig, update, frames=n_video_frames, blit=False, interval=1000 / fps)
    save_video_with_audio(
        anim, fig, None if no_audio else audio_seg, audio_sr,
        output_path, fps, speed, volume=volume, bitrate=3000,
    )


def render_single_trial(
    grid_frame, audio_seg, audio_sr, mic_waveform,
    timing_info, dead_mask,
    tmin, tmax, patient, trial_idx,
    speed, fps, vmin, vmax, cmap_name, dpi, no_audio,
    output_path, volume=1.0,
):
    """Render one trial as 2D heatmap MP4 with embedded audio."""
    H, W, T_neural = grid_frame.shape
    duration = tmax - tmin  # seconds
    neural_sr = 200.0
    n_video_frames = int(ceil(duration / speed * fps))

    # Set up figure
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="0.5")

    fig_width = max(10, W * 0.55)
    fig = plt.figure(figsize=(fig_width, 8), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.3, 0.7], hspace=0.25)
    ax_wave = fig.add_subplot(gs[0])
    ax_grid = fig.add_subplot(gs[1])

    # --- Top panel: raw microphone waveform (visual reference) ---
    t_mic = np.linspace(tmin, tmax, len(mic_waveform))
    mic_norm = mic_waveform.copy()
    mic_peak = np.abs(mic_norm).max()
    if mic_peak > 0:
        mic_norm = mic_norm / mic_peak
    ax_wave.plot(t_mic, mic_norm, color="0.3", linewidth=0.5, alpha=0.8)
    ax_wave.set_xlim(tmin, tmax)
    ax_wave.set_ylabel("Mic waveform")
    ax_wave.axvline(0, color="green", linestyle="--", linewidth=1, alpha=0.7, label="response onset")

    # Phoneme boundary annotations (relative to neural response_onset = display t=0)
    if timing_info is not None:
        for (on, off), phon in zip(
            timing_info.phoneme_intervals, timing_info.phonemes
        ):
            color = PHON_COLORS.get(phon, DEFAULT_PHON_COLOR)
            ax_wave.axvspan(on, off, alpha=0.25, color=color, zorder=1)
            mid = (on + off) / 2
            ax_wave.text(
                mid, ax_wave.get_ylim()[1] * 0.85, f"/{phon}/",
                ha="center", va="top", fontsize=10, fontweight="bold",
                color=color,
            )

    playhead = ax_wave.axvline(tmin, color="red", linewidth=1.5, zorder=5)

    # --- Bottom panel: electrode grid heatmap ---
    init_data = np.ma.masked_array(grid_frame[:, :, 0], mask=dead_mask)
    heatmap = ax_grid.imshow(
        init_data, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest", origin="upper",
    )
    ax_grid.set_xlabel("Column")
    ax_grid.set_ylabel("Row")
    cbar = fig.colorbar(heatmap, ax=ax_grid, shrink=0.8, pad=0.02)
    cbar.set_label("HGA z-score")

    # Title
    syllable = timing_info.syllable if timing_info else "?"
    phonemes_str = " ".join(f"/{p}/" for p in timing_info.phonemes) if timing_info else ""
    title = fig.suptitle(
        f"sub-{patient} | trial {trial_idx} | \"{syllable}\" ({phonemes_str}) | t = {tmin:.3f}s",
        fontsize=12, fontweight="bold",
    )

    # Animation update function
    def update(frame_idx):
        t = tmin + frame_idx * (speed / fps)
        t = min(t, tmax)
        sample = int(round((t - tmin) * neural_sr))
        sample = min(sample, T_neural - 1)

        frame_data = np.ma.masked_array(grid_frame[:, :, sample], mask=dead_mask)
        heatmap.set_data(frame_data)
        playhead.set_xdata([t, t])
        title.set_text(
            f"sub-{patient} | trial {trial_idx} | \"{syllable}\" ({phonemes_str}) | t = {t:.3f}s"
        )
        return [heatmap, playhead, title]

    anim = FuncAnimation(fig, update, frames=n_video_frames, blit=False, interval=1000/fps)
    save_video_with_audio(
        anim, fig, None if no_audio else audio_seg, audio_sr,
        output_path, fps, speed, volume=volume,
    )


def render_concat_trials(
    ds, timing, audio, audio_sr, grid_info, audio_events,
    start_trial, n_concat, tmin, tmax, patient,
    speed, fps, vmin, vmax, cmap_name, dpi, no_audio,
    output_path, volume=1.0, clock_fn=None,
):
    """Render multiple consecutive trials concatenated into one video."""
    if clock_fn is None:
        clock_fn = lambda t: 0.0
    n_trials = min(n_concat, len(ds) - start_trial)
    duration_per_trial = tmax - tmin
    gap_duration = 0.5  # seconds of separator between trials
    neural_sr = 200.0

    # Collect all trial data with audio offsets
    trial_data_list = []
    mute_end = -0.3  # mute stimulus before this time relative to response_onset
    for i in range(start_trial, start_trial + n_trials):
        grid_frame = ds.grid_data[i]  # (H, W, T)
        ti = timing[i] if i < len(timing) else None
        offset = match_trial_to_audio_event(ti, audio_events, clock_fn) if ti else None
        if offset is None:
            offset = clock_fn(ti.response_onset) if ti else 0.0
        # Extract audio using clock-corrected position
        audio_seg = extract_audio_segment(
            audio, audio_sr, ti.response_onset + offset,
            pre_s=abs(tmin), post_s=tmax,
        ) if ti else np.zeros(int((tmax - tmin) * audio_sr), dtype=np.float32)
        # Mute stimulus period, keep only patient response
        audio_seg = audio_seg.copy()
        mute_sample = int((mute_end - tmin) / (tmax - tmin) * len(audio_seg))
        mute_sample = max(0, min(mute_sample, len(audio_seg)))
        audio_seg[:mute_sample] = 0.0
        trial_data_list.append((grid_frame, audio_seg, ti, i, offset))

    # Calculate total video frames
    frames_per_trial = int(ceil(duration_per_trial / speed * fps))
    gap_frames = int(ceil(gap_duration / speed * fps))
    total_frames = n_trials * frames_per_trial + (n_trials - 1) * gap_frames

    # Concatenate audio with silence gaps, normalize volume
    gap_samples = int(gap_duration * audio_sr)
    silence = np.zeros(gap_samples, dtype=np.float32)
    all_audio = []
    for idx, (_, audio_seg, _, _, _) in enumerate(trial_data_list):
        all_audio.append(audio_seg)
        if idx < len(trial_data_list) - 1:
            all_audio.append(silence)
    concat_audio = np.concatenate(all_audio)
    # Normalize volume
    peak = np.abs(concat_audio).max()
    if peak > 0:
        concat_audio = concat_audio / peak * 0.9

    H, W = grid_info.grid_shape
    dead_mask = grid_info.dead_mask

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="0.5")

    fig_width = max(10, W * 0.55)
    fig = plt.figure(figsize=(fig_width, 8), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.3, 0.7], hspace=0.25)
    ax_wave = fig.add_subplot(gs[0])
    ax_grid = fig.add_subplot(gs[1])

    # Waveform — show per-trial segments
    total_audio_dur = (n_trials * duration_per_trial + (n_trials - 1) * gap_duration)
    t_audio_full = np.linspace(0, total_audio_dur, len(concat_audio))
    ax_wave.plot(t_audio_full, concat_audio, color="0.3", linewidth=0.5, alpha=0.8)
    ax_wave.set_xlim(0, total_audio_dur)
    ax_wave.set_ylabel("Audio")

    # Add phoneme annotations for each trial
    for trial_offset, (_, _, ti, trial_idx, _offset) in enumerate(trial_data_list):
        t_base = trial_offset * (duration_per_trial + gap_duration) + abs(tmin)
        if ti is not None:
            for (on, off), phon in zip(ti.phoneme_intervals, ti.phonemes):
                color = PHON_COLORS.get(phon, DEFAULT_PHON_COLOR)
                ax_wave.axvspan(
                    t_base + on, t_base + off,
                    alpha=0.2, color=color,
                )

        # Trial separator
        if trial_offset < n_trials - 1:
            sep_t = (trial_offset + 1) * duration_per_trial + trial_offset * gap_duration
            ax_wave.axvline(sep_t, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    playhead = ax_wave.axvline(0, color="red", linewidth=1.5, zorder=5)

    init_data = np.ma.masked_array(
        trial_data_list[0][0][:, :, 0], mask=dead_mask
    )
    heatmap = ax_grid.imshow(
        init_data, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest", origin="upper",
    )
    ax_grid.set_xlabel("Column")
    ax_grid.set_ylabel("Row")
    cbar = fig.colorbar(heatmap, ax=ax_grid, shrink=0.8, pad=0.02)
    cbar.set_label("HGA z-score")

    title = fig.suptitle(
        f"sub-{patient} | trials {start_trial}-{start_trial + n_trials - 1}",
        fontsize=12, fontweight="bold",
    )

    T_neural = trial_data_list[0][0].shape[2]

    def update(frame_idx):
        # Determine which trial and local frame we're in
        segment_len = frames_per_trial + gap_frames
        trial_offset = frame_idx // segment_len
        local_frame = frame_idx % segment_len

        # Global time for playhead
        global_t = frame_idx * (speed / fps)
        playhead.set_xdata([global_t, global_t])

        if trial_offset >= n_trials:
            trial_offset = n_trials - 1
            local_frame = frames_per_trial - 1

        grid_frame, _, ti, trial_idx, _ = trial_data_list[min(trial_offset, n_trials - 1)]

        if local_frame >= frames_per_trial:
            # In the gap — show gray
            frame_data = np.ma.masked_array(
                np.zeros((H, W)), mask=np.ones((H, W), dtype=bool)
            )
            heatmap.set_data(frame_data)
            syllable = "---"
            t_local = 0.0
        else:
            t_local = tmin + local_frame * (speed / fps)
            t_local = min(t_local, tmax)
            sample = int(round((t_local - tmin) * neural_sr))
            sample = min(sample, T_neural - 1)
            frame_data = np.ma.masked_array(grid_frame[:, :, sample], mask=dead_mask)
            heatmap.set_data(frame_data)
            syllable = ti.syllable if ti else "?"

        title.set_text(
            f"sub-{patient} | trial {trial_idx} | \"{syllable}\" | t = {t_local:.3f}s"
        )
        return [heatmap, playhead, title]

    anim = FuncAnimation(fig, update, frames=total_frames, blit=False, interval=1000/fps)
    save_video_with_audio(
        anim, fig, None if no_audio else concat_audio, audio_sr,
        output_path, fps, speed, volume=volume,
    )


def parse_trial_arg(trial_str: str, n_trials: int) -> list[int]:
    """Parse --trial argument: int, 'N-M' range, or 'all'."""
    if trial_str == "all":
        return list(range(n_trials))
    if "-" in trial_str:
        start, end = trial_str.split("-", 1)
        return list(range(int(start), min(int(end) + 1, n_trials)))
    return [int(trial_str)]


def main():
    parser = argparse.ArgumentParser(
        description="Visualize uECOG grid activity with audio."
    )
    parser.add_argument("--patient", default="S14")
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--trial", default="0",
                        help="Trial index, range '0-9', or 'all'")
    parser.add_argument("--mode", choices=["speech", "full"], default="speech",
                        help="'speech' = [-0.5, 1.0s], 'full' = [-1.0, 1.5s]")
    parser.add_argument("--concat", type=int, default=0,
                        help="Concatenate N consecutive trials into one video")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="Playback speed (1.0=realtime, 0.5=half speed)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vmin", type=float, default=-3.0)
    parser.add_argument("--vmax", type=float, default=3.0)
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--output-dir", default="results/visualizations")
    parser.add_argument("--output", default=None,
                        help="Override output filename (e.g. 'my_video.mp4')")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="Audio volume multiplier (0.3 = 30%%, 1.0 = full)")
    parser.add_argument("--raw-audio", action="store_true",
                        help="Use raw mic recording instead of stimulus token")
    parser.add_argument("--style", choices=["2d", "3d"], default="2d",
                        help="Visualization style: 2d heatmap or 3d columns")
    parser.add_argument("--lpf", type=float, default=0,
                        help="Low-pass filter cutoff in Hz (zero-phase Butterworth, 0=off)")
    parser.add_argument("--threshold", type=float, default=0,
                        help="Zero out values with abs below this (in z-score units, 0=off)")
    args = parser.parse_args()

    with open(args.paths) as f:
        paths = yaml.safe_load(f)
    bids_root = Path(paths["ps_bids_root"])

    # Mode-dependent time window
    if args.mode == "full":
        tmin, tmax = -1.0, 1.5
    else:
        tmin, tmax = -0.5, 1.0

    logger.info("Loading data for %s (mode=%s, window=[%.1f, %.1f]s)...",
                args.patient, args.mode, tmin, tmax)

    ds, timing, audio, audio_sr, grid_info, audio_events = load_all_data(
        args.patient, bids_root, tmin, tmax
    )
    logger.info("Loaded %d trials, grid %s, audio sr=%d, audio_events=%d",
                len(ds), grid_info.grid_shape, audio_sr, len(audio_events))

    # Optional zero-phase Butterworth low-pass filter (vectorized)
    if args.lpf > 0:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, args.lpf, btype="low", fs=200.0, output="sos")
        shape = ds.grid_data.shape  # (N, H, W, T)
        flat = ds.grid_data.reshape(-1, shape[-1])  # (N*H*W, T)
        ds.grid_data = sosfiltfilt(sos, flat, axis=1).reshape(shape)
        logger.info("Applied %d Hz zero-phase Butterworth LPF (order 4)", args.lpf)

    # Optional thresholding — clamp [-threshold, +threshold] to zero, shift remainder
    if args.threshold > 0:
        data = ds.grid_data
        frac = (np.abs(data) <= args.threshold).mean()
        ds.grid_data = np.sign(data) * np.maximum(np.abs(data) - args.threshold, 0)
        logger.info("Threshold ±%.1f SD: %.1f%% clamped to ground", args.threshold, frac * 100)

    output_dir = Path(args.output_dir)

    # Estimate clock relationship (linear model handles drift between devices).
    clock_fn = estimate_clock_offset(timing, audio_events)

    if args.concat > 0:
        # Concatenated mode
        trial_indices = parse_trial_arg(args.trial, len(ds))
        start = trial_indices[0]
        if args.output:
            out_name = args.output if args.output.endswith(".mp4") else args.output + ".mp4"
        else:
            out_name = (
                f"sub-{args.patient}_trials-{start:03d}-{start + args.concat - 1:03d}"
                f"_concat_{args.mode}_speed-{args.speed}.mp4"
            )
        render_concat_trials(
            ds, timing, audio, audio_sr, grid_info, audio_events,
            start_trial=start, n_concat=args.concat,
            tmin=tmin, tmax=tmax, patient=args.patient,
            speed=args.speed, fps=args.fps,
            vmin=args.vmin, vmax=args.vmax,
            cmap_name=args.cmap, dpi=args.dpi, no_audio=args.no_audio,
            output_path=output_dir / out_name, volume=args.volume,
            clock_fn=clock_fn,
        )
        return

    # Single trial or batch mode
    trial_indices = parse_trial_arg(args.trial, len(ds))
    for trial_idx in trial_indices:
        ti = timing[trial_idx] if trial_idx < len(timing) else None

        # Compute per-trial audio-neural offset using drift-corrected matching.
        # Falls back to linear clock model if no per-trial match found.
        audio_offset = match_trial_to_audio_event(ti, audio_events, clock_fn) if ti else None
        if audio_offset is None:
            audio_offset = clock_fn(ti.response_onset) if ti else 0.0
            logger.info("  Trial %d: no per-trial match, using clock model %.3fs", trial_idx, audio_offset)
        else:
            logger.info("  Trial %d audio offset: %.3fs", trial_idx, audio_offset)

        # Raw mic waveform — extract from audio clock position (neural time + offset)
        if ti is not None:
            mic_waveform = extract_audio_segment(
                audio, audio_sr, ti.response_onset + audio_offset,
                pre_s=abs(tmin), post_s=tmax,
            )
        else:
            mic_waveform = np.zeros(int((tmax - tmin) * audio_sr), dtype=np.float32)

        if args.raw_audio:
            # Use raw mic recording as audio track (normalized)
            audio_seg = mic_waveform.copy()
            audio_sr_used = audio_sr
        else:
            # Use clean stimulus token synced to response timing
            audio_seg, audio_sr_used = build_trial_audio(
                bids_root, ti, tmin, tmax,
            )

        syllable = ti.syllable if ti else "unknown"
        if args.output and len(trial_indices) == 1:
            out_name = args.output if args.output.endswith(".mp4") else args.output + ".mp4"
        else:
            out_name = (
                f"sub-{args.patient}_trial-{trial_idx:03d}_{syllable}"
                f"_{args.mode}_speed-{args.speed}.mp4"
            )

        logger.info("Rendering trial %d/%d (%s) [%s]...", trial_idx + 1, len(ds), syllable, args.style)
        render_fn = render_single_trial_3d if args.style == "3d" else render_single_trial
        kwargs = dict(
            grid_frame=ds.grid_data[trial_idx],
            audio_seg=audio_seg,
            audio_sr=audio_sr_used,
            mic_waveform=mic_waveform,
            timing_info=ti,
            dead_mask=grid_info.dead_mask,
            tmin=tmin, tmax=tmax,
            patient=args.patient, trial_idx=trial_idx,
            speed=args.speed, fps=args.fps,
            vmin=args.vmin, vmax=args.vmax,
            cmap_name=args.cmap, dpi=args.dpi, no_audio=args.no_audio,
            output_path=output_dir / out_name, volume=args.volume,
        )
        if args.style == "3d":
            kwargs["threshold"] = args.threshold
        render_fn(**kwargs)


if __name__ == "__main__":
    main()
