"""V-JEPA 2 encoder-only feature extraction for the teacher-ceiling probe.

Per word: decode the [start, start+after_s] window from a 256-px movie proxy,
sample to 64 frames, forward the frozen V-JEPA 2 ViT-L encoder, and reduce the
(32 temporal x 256 spatial x 1024) token grid to two spatially-pooled streams:
mean-over-spatial and max-over-spatial, each (32, 1024). Temporal pooling (the
8 Hz triangular pool + mean/flatten conditions) happens downstream in the runner
so a single forward serves all conditions.

API facts (verified 2026-06-02, transformers 5.4.0):
  - encoder-only = ``model.get_vision_features(pixel_values_videos)`` -> (B, 8192, 1024).
    (The full ``VJEPA2Model.forward`` also runs the JEPA predictor — skip it.)
  - processor = ``AutoVideoProcessor``; key ``pixel_values_videos`` (B, 64, 3, 256, 256).
  - token order is temporal-major (modeling_vjepa2.py:115 conv3d -> flatten(2) -> transpose):
    token n -> temporal t = n // 256, spatial s = n % 256. reshape(32, 256, 1024).
"""
from __future__ import annotations

import numpy as np

VJEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
FRAMES_PER_CLIP = 64
N_TEMPORAL = 32      # frames_per_clip // tubelet_size (64 // 2)
N_SPATIAL = 256      # (crop // patch) ** 2 = (256 // 16) ** 2
HIDDEN = 1024


def tri_pool_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Bartlett (triangular) resampling matrix (n_out, n_in), rows sum to 1.

    Half-base = stride = n_in / n_out (COLA-correct Bartlett tiling). This is the
    SAME triangle shape as ``whisper_teacher_pool.triangular_pool_weight_matrix``,
    which hardcodes half_fwhm=6.25 because that equals the stride for Whisper's
    50 Hz -> 8-bucket/1 s case. V-JEPA's native temporal rate is 32 (not 50), so
    the half-base must track the stride (=4 for 32->8) to stay COLA-correct.
    """
    stride = n_in / n_out
    half = stride
    centres = np.arange(n_out, dtype=np.float64) * stride
    pos = np.arange(n_in, dtype=np.float64)
    dist = np.abs(pos[None, :] - centres[:, None])
    W = np.clip(1.0 - dist / half, 0.0, None)
    W /= np.clip(W.sum(axis=1, keepdims=True), 1e-8, None)
    return W


def _device_and_dtype():
    import torch
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


class VJepaClipExtractor:
    """Frozen V-JEPA 2 ViT-L encoder -> spatially-pooled (32, 1024) streams."""

    def __init__(self, model_id: str = VJEPA_MODEL_ID, device: str | None = None):
        import torch
        from transformers import AutoVideoProcessor, VJEPA2Model

        if device is None:
            device, dtype = _device_and_dtype()
        else:
            dtype = torch.float16 if device != "cpu" else torch.float32
        self.device = device
        self.dtype = dtype
        self.torch = torch
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        model = VJEPA2Model.from_pretrained(model_id, torch_dtype=dtype).eval()
        if device != "cpu":
            model = model.to(device)
        self.model = model

    def forward_clip(self, frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """frames: list of 64 (H, W, 3) RGB uint8 -> (sm, mx) each (32, 1024) fp32."""
        torch = self.torch
        enc = self.processor(frames, return_tensors="pt")
        pv = enc["pixel_values_videos"].to(self.device, self.dtype)
        with torch.no_grad():
            feat = self.model.get_vision_features(pv)        # (1, 8192, 1024)
        g = feat[0].float().cpu().numpy().reshape(N_TEMPORAL, N_SPATIAL, HIDDEN)
        return g.mean(axis=1), g.max(axis=1)                 # (32, 1024) each


def decode_clip(cap, fps: float, start_s: float, after_s: float = 1.0,
                n_frames: int = FRAMES_PER_CLIP) -> list[np.ndarray] | None:
    """Read [start_s, start_s+after_s] from an open cv2 VideoCapture, resample to
    n_frames RGB uint8 frames. One seek + a short sequential read, then index.

    Returns None if the window can't be read (past EOF / decode failure).
    """
    import cv2

    f0 = int(round(start_s * fps))
    f1 = int(round((start_s + after_s) * fps))
    f1 = max(f1, f0 + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    block = []
    for _ in range(f1 - f0):
        ok, frame = cap.read()
        if not ok:
            break
        block.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not block:
        return None
    # Resample the decoded block to exactly n_frames (linspace index, repeats ok).
    idx = np.linspace(0, len(block) - 1, n_frames).round().astype(int)
    return [block[i] for i in idx]


def open_video(path: str):
    """Open a proxy mp4 with cv2; return (cap, fps)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps <= 0:  # NaN/0 guard
        fps = 24000.0 / 1001.0
    return cap, fps
