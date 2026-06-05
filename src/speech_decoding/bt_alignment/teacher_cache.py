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
that ``StandardScaler`` role is filled by an explicit, mandatory, full-corpus
per-channel z-score — :func:`fit_channel_stats` + :class:`TargetStandardizer`
below — applied to the 1280-d target before the distillation loss (full-corpus
ratified 2026-06-04, ``project_v14_b33_channel_stats_full_corpus_2026_06_04``:
not a leak, transductive norm of a frozen-model target; train-only kept as the
``R-train-only-stats`` sister). The teacher
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

Cache schema (per clip). The model + layer-merge are folded into the PATH
(not just the payload) so two target-defining configs can never be mixed into
one stats fit / training set:
    out_dir / <model_slug> / <layer_merge> / <film> / <clip_id>.pt   ->  dict({
        "features": Tensor of shape (T, 1280) float16,
        "rate_hz": 50,
        "t0_movie_s": float,  # absolute movie clock when feature[0] applies
        "model": str,  # the wrapped model's name_or_path, e.g. "openai/whisper-large-v3"
        "layer_merge": "mean_all",  # or an int for the single-layer sister
    })
    # <model_slug> = model_name_or_path.replace("/", "_"); <layer_merge> = "mean_all" | f"L{int}"

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


def merge_slug(layer_merge: int | str) -> str:
    """Cache-path segment for a layer merge: ``mean_all`` or ``L{int}``."""
    return "mean_all" if layer_merge == "mean_all" else f"L{int(layer_merge)}"


def movie_cache_path(
    out_dir: Path | str, model: str, layer_merge: int | str, movie: str
) -> Path:
    """``out_dir/<model_slug>/<merge_slug>/<movie>.pt`` — the SINGLE source of
    truth for the whole-movie cache layout.

    Writer (:func:`write_movie_cache`), reader
    (:class:`~speech_decoding.extractors.whisper_target.WhisperTargetExtractor`),
    and the build-script skip check all route through here so they cannot drift
    apart: a one-character slug mismatch between writer and reader would make
    every clip silently miss its cache."""
    return Path(out_dir) / model.replace("/", "_") / merge_slug(layer_merge) / f"{movie}.pt"


@dataclass
class TeacherCacheEntry:
    clip_id: str
    film: str
    t0_movie_s: float
    rate_hz: int
    n_frames: int
    d_model: int
    out_path: str


# Whisper's encoder is architecturally fixed at a 30 s receptive window (1500
# frames @ 50 Hz); the mel front-end pads/crops every input to 30 s. Feeding a
# whole movie at once therefore SILENTLY truncates to its first 30 s. The
# whole-movie cache (write_movie_cache) chunks on this exact grid so each full
# chunk emits exactly round(30 × 50) = 1500 frames and the dense stream's frame
# f maps to movie-clock time f / rate_hz with no drift.
WHISPER_ENCODER_WINDOW_S: float = 30.0


@dataclass
class MovieCacheEntry:
    movie: str
    rate_hz: int
    n_frames: int
    d_model: int
    duration_s: float
    chunk_s: float
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
        # Move the mel input to the model's device AND dtype so the same code
        # path runs on a GPU node (large-v3) and on a CPU laptop (tiny smoke).
        # The dtype match is load-bearing under transformers v5, which loads a
        # checkpoint in its NATIVE dtype: large-v3's HF weights are fp16, so the
        # encoder conv1d sees Half weights while the processor emits a float32
        # mel — a raw ``.to(device)`` then raises "Input type (float) and bias
        # type (c10::Half) should be the same". Aligning to the param dtype is a
        # no-op when the model is fp32 (the build forces fp32; see
        # build_bt_teacher_cache) and the CPU tiny-smoke path.
        param = next(self.model.parameters())
        input_features = inputs.input_features.to(device=param.device, dtype=param.dtype)
        self._captures.clear()
        with torch.no_grad():
            self.model.model.encoder(input_features)
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
    """Extract layer-merged features for one clip and save to
    ``out_dir/<model_slug>/<layer_merge>/<film>/<clip_id>.pt``.

    The two fields that DEFINE the stored target — the layer merge
    (``mean_all`` vs the ``R-whisper-single-layer-L8`` sister) and the model —
    are folded into the path, not just the payload. Two configs that produce
    different Whisper targets therefore land in disjoint dirs and can never be
    mixed into one :func:`fit_channel_stats` fit or one training set (the
    payload alone did not protect against an alt-config reusing ``out_dir``).

    Precondition for the P3 path: the downstream 50→8 Hz pool requires exactly
    250 frames, so callers must feed exactly-5.0-s (80000-sample @ 16 kHz) clips.
    A non-whole-second clip trims to ``round(clip_s × 50) ≠ 250`` and the pool
    rejects it loudly (this is intended — a mis-sized clip should fail, not be
    silently re-pooled)."""
    import torch
    feat = feature_extractor.extract(wav, sample_rate)
    # Model identity comes from the wrapped model, NOT a constant — a constant
    # would collide an alt-model (e.g. the whisper-tiny CPU smoke) with large-v3
    # in one dir AND mislabel the payload's source. ``name_or_path`` is set by
    # ``from_pretrained`` (the only constructor in this codebase). Fail loud on
    # an empty name rather than collapse to ``out_dir/""/...`` and silently
    # collide — the exact silent-landmine class this batch closes.
    model_name = feature_extractor.model.name_or_path
    if not model_name:
        raise ValueError(
            "feature_extractor.model has no name_or_path; cannot key the cache "
            "by model. Load the Whisper model via from_pretrained(...)."
        )
    layer_merge = feature_extractor.layer_merge
    clip_dir = out_dir / model_name.replace("/", "_") / merge_slug(layer_merge) / film
    clip_dir.mkdir(parents=True, exist_ok=True)
    out_path = clip_dir / f"{clip_id}.pt"
    torch.save({
        "features": feat,
        "rate_hz": rate_hz,
        "t0_movie_s": float(t0_movie_s),
        "model": model_name,
        "layer_merge": layer_merge,
    }, out_path)
    return TeacherCacheEntry(
        clip_id=clip_id, film=film, t0_movie_s=float(t0_movie_s),
        rate_hz=rate_hz, n_frames=int(feat.shape[0]), d_model=int(feat.shape[1]),
        out_path=str(out_path),
    )


def write_movie_cache(
    feature_extractor: WhisperFeatureExtractor,
    wav: np.ndarray,
    sample_rate: int,
    movie: str,
    out_dir: Path,
    chunk_s: float = WHISPER_ENCODER_WINDOW_S,
    rate_hz: int = DEFAULT_TEACHER_HZ,
) -> MovieCacheEntry:
    """Encode a WHOLE movie into one dense ``(T, d)`` teacher stream at ``rate_hz``.

    The whole-movie form (vs the per-clip :func:`write_clip_cache`): run Whisper
    ONCE per movie over consecutive ``chunk_s`` grid windows and concatenate, so
    every training clip — at any ``t0_movie_s``, any Δlag, any sampler — is a
    free slice ``dense[round(t0 × rate_hz) : +round(clip_s × rate_hz)]`` of this
    stream, and the cost is shared across every subject/trial that watched the
    movie. The dense frame ``f`` maps to movie-clock time ``f / rate_hz``.

    Why chunk (not feed the whole movie): Whisper's encoder is fixed at a
    ``WHISPER_ENCODER_WINDOW_S`` (30 s) receptive window — the mel front-end
    pads/crops every input to 30 s, so a single forward over a 90-minute movie
    would silently keep only its first 30 s. We chunk on a 30 s grid (the
    encoder's native window, giving every frame full in-window context) so each
    full chunk emits exactly ``round(chunk_s × rate_hz)`` real frames and the
    frame↔time map carries no drift across chunks.

    Boundary caveat (flagged, accepted per the grid-aligned directive): a clip
    whose window straddles a chunk seam draws its teacher frames from two
    separate passes, each context-truncated at the seam. The ``R-teacher-overlap``
    falsifier (overlapping windows, keep center) is the fallback if seam effects
    surface.

    Path keyed by model + layer-merge exactly like :func:`write_clip_cache`
    (``out_dir/<model_slug>/<merge_slug>/<movie>.pt``) so two target-defining
    configs can never land in one file. Features fp16, like the per-clip cache.
    """
    import torch

    if sample_rate != WHISPER_SR:
        raise ValueError(f"Whisper expects {WHISPER_SR} Hz, got {sample_rate}")
    if wav.ndim != 1:
        raise ValueError(f"expected mono 1-D waveform, got shape {wav.shape}")
    chunk_samples = int(round(chunk_s * sample_rate))
    if chunk_samples <= 0:
        raise ValueError(f"chunk_s {chunk_s} × sr {sample_rate} → non-positive chunk")

    feats: list[Tensor] = []
    for start in range(0, len(wav), chunk_samples):
        chunk = wav[start : start + chunk_samples]
        feat = feature_extractor.extract(chunk, sample_rate)  # (n_real, d) fp16
        if feat.shape[0] > 0:
            feats.append(feat)
    if not feats:
        raise ValueError(f"movie {movie!r}: no frames extracted (empty waveform?)")
    dense = torch.cat(feats, dim=0)  # (T, d) fp16

    # Frame-count invariant (truncation guard — the headline silent-corruption
    # mode). For 30 s grid chunking at the fixed enc rate, dense frames ==
    # round(duration_s × rate_hz) EXACTLY: each full chunk emits
    # round(chunk_s × rate_hz) frames, k·(that) is integer, so the per-chunk
    # rounds sum to the whole-movie round. A mismatch means a chunk silently
    # truncated to its first 30 s (the exact trap this design exists to avoid),
    # a chunk got dropped, or enc rate ≠ rate_hz. Fail loud — never cache a
    # teacher stream whose frame↔movie-clock map is wrong (that would slice the
    # P3 target off-target, undetectably).
    duration_s = len(wav) / float(sample_rate)
    expected_frames = round(duration_s * rate_hz)
    # Tolerance of 1 frame (20 ms): the only LEGITIMATE disagreement between the
    # whole-movie round and the per-chunk-summed dense length is round-half-to-
    # even (banker's rounding) on the final partial chunk landing on a .5
    # boundary — at most ±1 frame, well past any word onset. A REAL truncation
    # (a 30 s chunk silently cropped, or a chunk dropped) is off by ~1500 frames,
    # so >1 still fails loud. Without the tolerance an unlucky movie length
    # crashes the build with a misleading "truncated" message (no real BT movie
    # hits it, but D-cohort / SWEC durations are unaudited).
    if abs(int(dense.shape[0]) - expected_frames) > 1:
        raise RuntimeError(
            f"movie {movie!r}: dense has {dense.shape[0]} frames but "
            f"{duration_s:.3f}s × {rate_hz}Hz ⇒ {expected_frames} expected "
            f"(off by {int(dense.shape[0]) - expected_frames}) — a chunk "
            f"truncated/dropped or enc rate ≠ {rate_hz}Hz. Refusing to cache a "
            f"teacher stream with a broken frame↔time map."
        )

    model_name = feature_extractor.model.name_or_path
    if not model_name:
        raise ValueError(
            "feature_extractor.model has no name_or_path; cannot key the cache "
            "by model. Load the Whisper model via from_pretrained(...)."
        )
    layer_merge = feature_extractor.layer_merge
    out_path = movie_cache_path(out_dir, model_name, layer_merge, movie)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": dense,
        "rate_hz": rate_hz,
        "t0_movie_s": 0.0,  # dense stream starts at the movie origin
        "duration_s": float(duration_s),
        "chunk_s": float(chunk_s),
        "movie": movie,
        "model": model_name,
        "layer_merge": layer_merge,
    }, out_path)

    # Re-open the just-written file (mmap = header read, no full materialize) and
    # confirm shape/dtype survived the write. Guards a truncated/corrupt .pt from
    # an OOM or full disk mid-save being recorded as a valid cache entry — a
    # downstream torch.load would then silently slice the wrong region.
    reloaded = torch.load(out_path, map_location="cpu", mmap=True, weights_only=False)
    rf = reloaded["features"]
    if tuple(rf.shape) != tuple(dense.shape) or rf.dtype != torch.float16:
        raise RuntimeError(
            f"movie {movie!r}: cache reloads as shape {tuple(rf.shape)} dtype "
            f"{rf.dtype}, expected {tuple(dense.shape)} float16 — write was "
            f"truncated/corrupt ({out_path})."
        )

    return MovieCacheEntry(
        movie=movie, rate_hz=rate_hz,
        n_frames=int(dense.shape[0]), d_model=int(dense.shape[1]),
        duration_s=float(duration_s), chunk_s=float(chunk_s),
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

    Corpus-agnostic: fits over whatever paths it is given. The shipped caller
    (:func:`fit_and_save_channel_stats`) passes the FULL corpus (all movies,
    train+test) — Ben 2026-06-04: leakage is second-order because the
    standardizer is consumed entirely within P3 distillation and is GONE at the
    frozen-probe P4 eval (no Whisper target at readout; audit-confirmed). The
    2560 label-free scalars cannot inflate the grouped-by-token CV metric. Pass
    ONLY the training-split paths for the ``R-train-only-stats`` sister (the
    original B33 discipline, ≡ the ceiling probe's ``StandardScaler.fit(train)``).
    Zero-variance guard: a channel with ``σ ≈ 0`` gets ``inv_std = 1`` (passes
    through unscaled), mirroring sklearn ``_handle_zeros_in_scale``.

    Two-pass (mean, then sum of squared deviations) for numerical stability —
    avoids the ``E[x²] − E[x]²`` catastrophic cancellation of a single pass.
    Save via :func:`fit_and_save_channel_stats` as ``channel_stats.pt``;
    :class:`TargetStandardizer` consumes it at load/train time. Falsifier
    ``R-no-target-standardize`` skips standardization (raw 1280-d target).
    """
    from speech_decoding.extractors.whisper_teacher_pool import (
        triangular_pool_50_to_8_hz,
        _EXPECTED_N_IN as _CLIP_FRAMES_50HZ,  # 250 = 5 s × 50 Hz (pool's locked input)
    )

    def _load_pooled(path: Path) -> Tensor:
        # Fit at the rate the standardizer is applied: the 8 Hz pooled target,
        # not the 50 Hz cache (H2). The shipped teacher cache is WHOLE-MOVIE
        # (~350k frames per movie), but triangular_pool_50_to_8_hz is locked to
        # the per-clip 250→40 contract (it rejects other lengths by design). So
        # tile the movie into non-overlapping 250-frame clips and batch-pool each
        # — the IDENTICAL op training applies per clip (same zero-padded edges),
        # so the fitted per-channel stats match the 8 Hz target distribution the
        # standardizer sees. The ≤249-frame tail is dropped (<0.02% of a movie).
        # A 250-frame input (the unit-test clip) is n_clips=1 → unchanged. fp32
        # (cache is fp16). (Whole-movie caller never hit before #44 was run live.)
        feat = torch.load(path, weights_only=False)["features"].to(torch.float32)
        n_clips = feat.shape[0] // _CLIP_FRAMES_50HZ
        if n_clips == 0:
            raise ValueError(
                f"fit_channel_stats: {path.name} has {feat.shape[0]} frames "
                f"(< one {_CLIP_FRAMES_50HZ}-frame clip); cannot pool to 8 Hz."
            )
        clips = feat[: n_clips * _CLIP_FRAMES_50HZ].reshape(
            n_clips, _CLIP_FRAMES_50HZ, -1,
        )
        pooled = triangular_pool_50_to_8_hz(clips)        # (n_clips, 40, d)
        return pooled.reshape(-1, pooled.shape[-1])       # (n_clips·40, d)

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


def fit_and_save_channel_stats(
    cache_dir: Path | str,
    *,
    model: str,
    layer_merge: int | str,
    out_path: Path | str,
    d_model: int = 1280,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Fit ONE global per-channel z-score over every whole-movie teacher cache
    and save it as ``channel_stats.pt`` for :class:`TargetStandardizer`.

    Globs the full ``movie_cache_path`` layout
    (``cache_dir/<model_slug>/<merge_slug>/*.pt``) and fits the FULL corpus
    (all movies) via :func:`fit_channel_stats` — the shipped P3 default (Ben
    2026-06-04; full-corpus, not train-only, leakage second-order per that
    function's docstring). Any ``channel_stats*.pt`` is excluded from the glob —
    not just this call's own ``out_path`` — because the default ``out_path`` lives
    IN this same dir, so a re-fit to a different location would otherwise ingest a
    prior stats file (keys ``{mean, inv_std}``, no ``features``) as a movie cache
    and crash. ``channel_stats`` is a reserved artifact name, never a movie slug.
    Returns the saved ``{'mean', 'inv_std'}``.
    """
    out_path = Path(out_path)
    movie_dir = Path(cache_dir) / model.replace("/", "_") / merge_slug(layer_merge)
    feature_paths = sorted(
        p for p in movie_dir.glob("*.pt")
        if p.resolve() != out_path.resolve()
        and not p.stem.startswith("channel_stats")
    )
    if not feature_paths:
        raise FileNotFoundError(
            f"fit_and_save_channel_stats: no movie caches under {movie_dir} "
            f"(model={model!r}, layer_merge={layer_merge!r}). Build the teacher "
            "cache first (write_movie_cache), or run the R-no-target-standardize "
            "sister (target_standardize=False)."
        )
    stats = fit_channel_stats(feature_paths, d_model=d_model, eps=eps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, out_path)
    return stats


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
