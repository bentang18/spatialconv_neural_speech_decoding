"""Tiered frame-alignment verification: BT reference PNG vs rip frame at the same fps-derived time.

BT ships `movie_frames/<slug>/frame_<idx>.png` at frame indices spaced every 10k frames
(1000, 11000, ... up to <=10 reference frames per film), encoded as 1280×720 PNG @ 23.976 fps.
For each reference frame, we:
  1. Compute the wall-clock time t = idx / 23.976 in the rip.
  2. Extract the rip's frame at t via ffmpeg (seeks accurately to nearest keyframe + fwd).
  3. Letterbox-crop both BT and rip frames so identical aspect ratios are compared.
  4. Compare via:
     - np.all() fast path (rare exact match)
     - phash 8x8 perceptual hash (DCT-based)
     - dhash 8x8 difference hash (gradient-based, independent signal)
     - SSIM tiebreaker if phash + dhash disagree.

Pass criteria per frame:
  exact match OR (phash_dist <= phash_thresh AND dhash_dist <= dhash_thresh)
  OR (one hash failing AND SSIM > ssim_thresh).

Hash thresholds are PER-FILM (calibrated separately; defaults below are starting points).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import imagehash
from PIL import Image
from skimage.metrics import structural_similarity as ssim

BT_FPS = 23.976
DEFAULT_PHASH_THRESH = 8
DEFAULT_DHASH_THRESH = 8
DEFAULT_SSIM_THRESH = 0.7
DEFAULT_HASH_SIZE = 8


@dataclass
class FrameCheckResult:
    bt_frame_path: str
    rip_video: str
    frame_idx: int
    t_seconds: float
    exact_match: bool
    phash_dist: int
    dhash_dist: int
    ssim: float | None
    pass_frame: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def crop_letterbox(img: np.ndarray, threshold: int = 16) -> np.ndarray:
    """Strip black bars at top/bottom (and left/right). Returns the cropped image."""
    gray = img.mean(axis=2) if img.ndim == 3 else img
    row_max = gray.max(axis=1)
    col_max = gray.max(axis=0)
    rows = np.where(row_max > threshold)[0]
    cols = np.where(col_max > threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def extract_rip_frame(rip_video: Path, t_seconds: float, out_png: Path) -> None:
    """Use ffmpeg to extract the rip's frame at t. Combined input-seek + filter-seek
    is the most accurate per ffmpeg docs."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{t_seconds:.4f}",
        "-i", str(rip_video),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_png),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0 or not out_png.exists():
        raise RuntimeError(f"ffmpeg frame extract failed: {res.stderr.strip()}")


def compare_frames(
    bt_png: Path,
    rip_png: Path,
    phash_thresh: int = DEFAULT_PHASH_THRESH,
    dhash_thresh: int = DEFAULT_DHASH_THRESH,
    ssim_thresh: float = DEFAULT_SSIM_THRESH,
    hash_size: int = DEFAULT_HASH_SIZE,
) -> dict:
    """Returns dict with exact_match, phash_dist, dhash_dist, ssim, pass_frame."""
    bt = np.array(Image.open(bt_png).convert("RGB"))
    rip = np.array(Image.open(rip_png).convert("RGB"))

    bt_c = crop_letterbox(bt)
    rip_c = crop_letterbox(rip)

    # Resize rip to BT's size for fair comparison
    bt_pil = Image.fromarray(bt_c)
    rip_pil = Image.fromarray(rip_c).resize(bt_pil.size, Image.Resampling.LANCZOS)

    # Exact match (rare)
    rip_arr = np.array(rip_pil)
    bt_arr = np.array(bt_pil)
    exact = bool(rip_arr.shape == bt_arr.shape and np.all(rip_arr == bt_arr))

    # Independent hashes
    p_bt = imagehash.phash(bt_pil, hash_size=hash_size)
    p_rip = imagehash.phash(rip_pil, hash_size=hash_size)
    d_bt = imagehash.dhash(bt_pil, hash_size=hash_size)
    d_rip = imagehash.dhash(rip_pil, hash_size=hash_size)
    p_dist = int(p_bt - p_rip)
    d_dist = int(d_bt - d_rip)

    # Conservative pass: both hashes under threshold
    hash_pass = (p_dist <= phash_thresh) and (d_dist <= dhash_thresh)

    # SSIM tiebreaker only if exactly one hash fails
    ssim_score: float | None = None
    if not exact and not hash_pass and (
        (p_dist <= phash_thresh) ^ (d_dist <= dhash_thresh)
    ):
        bt_gray = np.array(bt_pil.convert("L"))
        rip_gray = np.array(rip_pil.convert("L"))
        ssim_score = float(ssim(bt_gray, rip_gray, data_range=255))
        hash_pass = ssim_score >= ssim_thresh

    return {
        "exact_match": exact,
        "phash_dist": p_dist,
        "dhash_dist": d_dist,
        "ssim": ssim_score,
        "pass_frame": exact or hash_pass,
    }


def verify_frame(
    bt_frame_path: Path,
    rip_video: Path,
    fps: float = BT_FPS,
    phash_thresh: int = DEFAULT_PHASH_THRESH,
    dhash_thresh: int = DEFAULT_DHASH_THRESH,
    ssim_thresh: float = DEFAULT_SSIM_THRESH,
) -> FrameCheckResult:
    """Full pipeline: parse BT frame_<idx>.png, extract rip frame at idx/fps,
    compare via crop + hash + ssim. Returns FrameCheckResult."""
    name = Path(bt_frame_path).stem
    if not name.startswith("frame_"):
        raise ValueError(f"unexpected BT frame name: {name}")
    frame_idx = int(name.split("_")[1])
    t = frame_idx / fps

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        rip_png = Path(f.name)
    try:
        extract_rip_frame(rip_video, t, rip_png)
        cmp = compare_frames(
            Path(bt_frame_path), rip_png,
            phash_thresh=phash_thresh,
            dhash_thresh=dhash_thresh,
            ssim_thresh=ssim_thresh,
        )
    finally:
        if rip_png.exists():
            rip_png.unlink()

    return FrameCheckResult(
        bt_frame_path=str(bt_frame_path),
        rip_video=str(rip_video),
        frame_idx=frame_idx,
        t_seconds=t,
        exact_match=cmp["exact_match"],
        phash_dist=cmp["phash_dist"],
        dhash_dist=cmp["dhash_dist"],
        ssim=cmp["ssim"],
        pass_frame=cmp["pass_frame"],
    )
