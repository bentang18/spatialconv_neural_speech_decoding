"""Whisper teacher-feature cache writer (v14 P3 distillation).

For each (film, clip-window) anchor pair, extract a 1280-d feature at 50 Hz
native from the Whisper-large-v3 encoder. The default merge is a **plain,
unweighted mean over all 32 encoder layers** (``layer_merge="mean_all"``),
NOT a single layer-8 read. This replicates the ceiling-probe ``mean_all``
target, which beat every single-layer pick on the transfer splits that match
the leaderboard metric (cross-movie LOSO / cross-session;
``reports/neuroprobe_whisper_ceiling_2026_05_29/``), superseding the prior
B06 layer-8 default per ``project_v14_whisper_teacher_all_layer_mean_2026_05_30``.

Normalization note (the load-bearing detail). The ceiling probe did NOT
LN/L2-normalize each layer before averaging — ``mean_all`` is a raw mean over
the 32 post-block hidden states (same forward-hook capture used here). The
per-channel standardization that made ``mean_all`` win lived DOWNSTREAM, in the
probe's per-channel train-fit ``StandardScaler`` on the merged vector — a
property of the *consumer*, not the feature. Under B33 project-up (2026-05-30)
that ``StandardScaler`` role is filled by an explicit, mandatory, train-only
per-channel z-score — :func:`fit_channel_stats` + :class:`TargetStandardizer`
below — applied to the 1280-d target before the distillation loss. The teacher
carries NO trainable adapter (``WhisperAdapter`` is demoted to the
``R-project-down`` sister; the student projects 256→1280 instead). So we store
the raw mean here and do not pre-normalize. (A pre-norm residual stream grows
with depth, so this mean is late-layer weighted — that is exactly the dominance
the per-channel z-score rescales, and it is what won.) Falsifier
``R-no-target-standardize`` skips the z-score (raw 1280-d target).

v2→v3 upgrade 2026-05-28 (``project_v14_whisper_teacher_v3_upgrade_2026_05_28``):
same encoder topology as v2 (32 layers, d=1280, native 50 Hz); v3 differs only
in the mel front-end (80→128 bins) and produces −10–20% WER on noisy speech.
Features are 1280-d at 50 Hz native: the 160-sample mel hop @ 16 kHz = 10 ms
(100 Hz mel), and the encoder's 2× conv downsample → 50 Hz = one step per 20 ms.

Cache schema (per clip):
    out_dir / <film> / <clip_id>.pt   ->  dict({
        "features": Tensor of shape (T, 1280) float16,
        "rate_hz": 50,
        "t0_movie_s": float,  # absolute movie clock when feature[0] applies
        "model": "openai/whisper-large-v3",
        "layer_merge": "mean_all",  # or an int for the single-layer sister
    })

Implementation notes:
- Registers a forward hook on every hooked encoder.layers[L] and takes the
  plain mean of their outputs (``torch.stack(...).mean(0)``); for a single int
  the stack-mean is that one layer's output (a no-op mean), so both paths share
  one code path that matches the probe's capture exactly.
- For unit-testing on a CPU laptop, swap whisper-large-v3 (1280-d, 32 layers)
  for whisper-tiny (384-d, 4 layers). The cache writer is the same; only the
  model differs.
- This module DOES NOT load a model at import; the caller passes in a loaded
  Whisper. The DCC dispatch script wires that up.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn

# Default merge over the encoder's layer outputs. "mean_all" = plain unweighted
# mean over ALL encoder layers (faithful to the ceiling-probe mean_all that won
# the transfer splits 2026-05-29). An int selects a single layer (the prior B06
# layer-8 default, retained as the R-whisper-single-layer-L8 sister anchor).
DEFAULT_LAYER_MERGE: int | str = "mean_all"
SINGLE_LAYER_SISTER_INDEX = 8  # R-whisper-single-layer-L8 falsifier anchor (was the B06 default)
DEFAULT_TEACHER_HZ = 50
WHISPER_SR = 16000
WHISPER_HOP = 160  # mel hop: 160 / 16 kHz = 10 ms (mel 100 Hz; 2× conv → 50 Hz enc)


@dataclass
class TeacherCacheEntry:
    clip_id: str
    film: str
    t0_movie_s: float
    rate_hz: int
    n_frames: int
    d_model: int
    out_path: str


class WhisperFeatureExtractor:
    """Wraps a transformers Whisper model and forward hooks to grab encoder
    layer outputs, then merges them. Designed for batch-of-1 inference (one
    clip at a time).

    ``layer_merge="mean_all"`` (default) hooks every encoder layer and returns
    the plain unweighted mean of their outputs — the ceiling-probe ``mean_all``
    target. An int hooks that single layer (the prior B06 default, retained for
    the ``R-whisper-single-layer-L8`` sister). No per-layer normalization: see
    the module docstring for why the win is faithful without it.
    """

    def __init__(self, model, processor, layer_merge: int | str = DEFAULT_LAYER_MERGE):
        self.model = model
        self.processor = processor
        self.layer_merge = layer_merge
        encoder_layers = model.model.encoder.layers
        n_layers = len(encoder_layers)
        if layer_merge == "mean_all":
            self._layers: list[int] = list(range(n_layers))
        elif isinstance(layer_merge, int):
            if layer_merge >= n_layers:
                raise ValueError(
                    f"layer {layer_merge} >= encoder depth {n_layers} "
                    f"(use whisper-large-v3 for L8)"
                )
            self._layers = [layer_merge]
        else:
            raise ValueError(
                f"layer_merge must be 'mean_all' or an int layer index, "
                f"got {layer_merge!r}"
            )
        self._captures: dict[int, object] = {}
        # transformer EncoderLayer.forward returns a tuple; output[0] is hidden_states
        self._handles = [
            encoder_layers[L].register_forward_hook(self._make_hook(L))
            for L in self._layers
        ]

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            self._captures[layer_idx] = (
                output[0] if isinstance(output, tuple) else output
            )
        return hook

    def extract(self, wav: np.ndarray, sample_rate: int = WHISPER_SR):
        if sample_rate != WHISPER_SR:
            raise ValueError(f"Whisper expects {WHISPER_SR} Hz, got {sample_rate}")
        import torch
        inputs = self.processor(
            wav, sampling_rate=sample_rate, return_tensors="pt"
        )
        self._captures.clear()
        with torch.no_grad():
            self.model.model.encoder(inputs.input_features)
        if len(self._captures) != len(self._layers):
            raise RuntimeError(
                f"hooks captured {len(self._captures)}/{len(self._layers)} layers"
            )
        # Plain unweighted mean over hooked layers (no per-layer norm) — matches
        # the ceiling-probe mean_all merge exactly. Single-layer is a 1-element
        # stack-mean (no-op). Each capture is (1, T, d_model).
        feat = torch.stack([self._captures[L] for L in self._layers], dim=0).mean(dim=0)
        feat = feat.squeeze(0)  # (T_enc, d_model)
        # Whisper pads/crops every clip to 30 s, so the encoder ALWAYS emits 1500
        # frames; only the first round(clip_s × 50) are real audio — the rest is
        # pad-silence. Trim to real frames so the cache matches the (clip_s × 50,
        # d) contract the 50→8 Hz pool consumes, and so pad-silence does not
        # poison fit_channel_stats (H3). enc rate = sr / WHISPER_HOP / 2 (mel
        # 100 Hz, 2× conv downsample → 50 Hz); at the enforced 16 kHz this is 50.
        enc_frames_per_s = sample_rate / WHISPER_HOP / 2.0
        n_real = round(len(wav) / sample_rate * enc_frames_per_s)
        feat = feat[:n_real]
        return feat.cpu().to(torch.float16)

    def close(self):
        for handle in self._handles:
            handle.remove()


def write_clip_cache(
    feature_extractor: WhisperFeatureExtractor,
    wav: np.ndarray,
    sample_rate: int,
    clip_id: str,
    film: str,
    t0_movie_s: float,
    out_dir: Path,
    rate_hz: int = DEFAULT_TEACHER_HZ,
) -> TeacherCacheEntry:
    """Extract layer-merged features for one clip and save to out_dir/<film>/<clip_id>.pt.

    Precondition for the P3 path: the downstream 50→8 Hz pool requires exactly
    250 frames, so callers must feed exactly-5.0-s (80000-sample @ 16 kHz) clips.
    A non-whole-second clip trims to ``round(clip_s × 50) ≠ 250`` and the pool
    rejects it loudly (this is intended — a mis-sized clip should fail, not be
    silently re-pooled)."""
    import torch
    feat = feature_extractor.extract(wav, sample_rate)
    film_dir = out_dir / film
    film_dir.mkdir(parents=True, exist_ok=True)
    out_path = film_dir / f"{clip_id}.pt"
    torch.save({
        "features": feat,
        "rate_hz": rate_hz,
        "t0_movie_s": float(t0_movie_s),
        "model": "openai/whisper-large-v3",
        "layer_merge": feature_extractor.layer_merge,
    }, out_path)
    return TeacherCacheEntry(
        clip_id=clip_id, film=film, t0_movie_s=float(t0_movie_s),
        rate_hz=rate_hz, n_frames=int(feat.shape[0]), d_model=int(feat.shape[1]),
        out_path=str(out_path),
    )


def fit_channel_stats(
    feature_paths: list[Path],
    d_model: int = 1280,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Fit a fixed per-channel z-score over the P3 **train-pool** clip caches.

    B33 (project-up, 2026-05-30) makes per-channel target standardization
    mandatory: with the teacher-side adapter gone, nothing else plays the
    ``StandardScaler`` role the ceiling probe relied on. The standardizer is
    applied to the **8 Hz pooled** target (cache 50 Hz →
    :func:`~speech_decoding.extractors.whisper_teacher_pool.triangular_pool_50_to_8_hz`
    → z-score; see ``ssl.distill``), so we fit at that same rate: each cached
    ``features`` tensor ``(250, d)`` is pooled to ``(40, d)`` before
    accumulation. Fitting on the raw 50 Hz cache instead would leave the
    standardized target at std≈0.33 — the triangular pool averages ~12-13
    frames per bucket and shrinks the variance ~9× (H2). Accumulates in **fp32**
    (cache is fp16; single-pass fp16 variance over large N is unstable) over
    ``(clips × 8 Hz frames)`` — one scalar per channel, never per-timestep (that
    would erase the onset/temporal structure the student must predict) — and
    returns ``{'mean': (d,), 'inv_std': (d,)}``.

    Train-only: pass ONLY the training-split clip paths; the returned stats are
    frozen and reused for val/test, so no val/test clip enters the statistics
    (same discipline as the ceiling probe's ``StandardScaler.fit(train)``).
    Zero-variance guard: a channel with ``σ ≈ 0`` gets ``inv_std = 1`` (passes
    through unscaled), mirroring sklearn ``_handle_zeros_in_scale``.

    Two-pass (mean, then sum of squared deviations) for numerical stability —
    avoids the ``E[x²] − E[x]²`` catastrophic cancellation of a single pass. The
    caller saves the result alongside the cache as ``channel_stats.pt``;
    :class:`TargetStandardizer` consumes it at load/train time. Falsifier
    ``R-no-target-standardize`` skips standardization (raw 1280-d target).
    """
    from speech_decoding.extractors.whisper_teacher_pool import (
        triangular_pool_50_to_8_hz,
    )

    def _load_pooled(path: Path) -> Tensor:
        # Fit at the rate the standardizer is applied: the 8 Hz pooled target,
        # not the 50 Hz cache (H2). fp32 (cache is fp16); pool 250→40.
        feat = torch.load(path, weights_only=False)["features"].to(torch.float32)
        return triangular_pool_50_to_8_hz(feat.unsqueeze(0)).squeeze(0)

    # Pass 1 — per-channel mean over (clips × 8 Hz frames), fp32.
    total = torch.zeros(d_model, dtype=torch.float32)
    n_frames = 0
    for path in feature_paths:
        pooled = _load_pooled(path)
        total += pooled.sum(dim=0)
        n_frames += int(pooled.shape[0])
    if n_frames == 0:
        raise ValueError("fit_channel_stats: no frames in feature_paths")
    mean = total / n_frames
    # Pass 2 — sum of squared deviations (stable).
    ss = torch.zeros(d_model, dtype=torch.float32)
    for path in feature_paths:
        pooled = _load_pooled(path)
        ss += ((pooled - mean) ** 2).sum(dim=0)
    var = ss / n_frames
    inv_std = 1.0 / torch.sqrt(var + eps)
    # Zero-variance guard: σ≈0 → pass channel through unscaled.
    inv_std = torch.where(var <= eps, torch.ones_like(inv_std), inv_std)
    return {"mean": mean, "inv_std": inv_std}


class TargetStandardizer(nn.Module):
    """Frozen per-channel affine for the 1280-d Whisper distillation target.

    Buffers ``mean`` and ``inv_std`` ``(d,)`` from :func:`fit_channel_stats`
    (non-trainable). ``forward: (B, T, d) → (B, T, d)`` applies
    ``(x - mean) * inv_std``. Applied to the teacher target upstream of the
    Phase-3 loss; train-only stats, fixed for val/test.
    """

    def __init__(self, mean: Tensor, inv_std: Tensor) -> None:
        super().__init__()
        if mean.ndim != 1 or mean.shape != inv_std.shape:
            raise ValueError(
                f"mean {tuple(mean.shape)} and inv_std {tuple(inv_std.shape)} "
                f"must be matching 1-D tensors"
            )
        self.register_buffer("mean", mean.to(torch.float32))
        self.register_buffer("inv_std", inv_std.to(torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, d) → (B, T, d)``: per-channel ``(x - mean) * inv_std``."""
        return (x - self.mean) * self.inv_std
