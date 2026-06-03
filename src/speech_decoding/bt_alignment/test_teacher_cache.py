"""Tests for Whisper teacher-feature cache writer.

Uses whisper-tiny (4 layers, 384-d) for the CPU smoke. The same interface drives
the DCC dispatch with whisper-large-v3 (32 layers, 1280-d); the default merge is
a plain mean over all encoder layers (``layer_merge="mean_all"``).
"""
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from speech_decoding.bt_alignment.teacher_cache import (
    WhisperFeatureExtractor, write_clip_cache,
    fit_channel_stats, TargetStandardizer,
    DEFAULT_LAYER_MERGE, SINGLE_LAYER_SISTER_INDEX, DEFAULT_TEACHER_HZ, WHISPER_SR,
)


def _save_features(path: Path, feat: torch.Tensor) -> None:
    """Minimal teacher-cache payload (fit_channel_stats reads only 'features');
    stored fp16 to mirror the real cache dtype."""
    torch.save({"features": feat.to(torch.float16)}, path)


def test_default_constants():
    assert DEFAULT_LAYER_MERGE == "mean_all"
    assert SINGLE_LAYER_SISTER_INDEX == 8
    assert DEFAULT_TEACHER_HZ == 50
    assert WHISPER_SR == 16000


@pytest.fixture(scope="module")
def whisper_tiny():
    """Whisper-tiny: 4 encoder layers, 384-d. Layer index 3 is the deepest."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model.eval()
    return model, proc


def test_extractor_raises_if_layer_too_deep(whisper_tiny):
    model, proc = whisper_tiny
    with pytest.raises(ValueError, match="encoder depth"):
        WhisperFeatureExtractor(model, proc, layer_merge=8)


def test_extractor_rejects_bad_layer_merge(whisper_tiny):
    model, proc = whisper_tiny
    with pytest.raises(ValueError, match="mean_all"):
        WhisperFeatureExtractor(model, proc, layer_merge="mean_top4")


def test_extract_returns_correct_shape_and_dtype(whisper_tiny):
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    sr = WHISPER_SR
    wav = np.zeros(int(sr * 30.0), dtype=np.float32)
    try:
        feat = fe.extract(wav, sample_rate=sr)
        # Whisper-tiny encoder produces (T=1500, 384) at 50 Hz for 30 s input
        assert feat.ndim == 2
        assert feat.shape[0] == 1500  # 30 s × 50 Hz
        assert feat.shape[1] == 384   # whisper-tiny hidden dim
        assert feat.dtype == torch.float16
    finally:
        fe.close()


def test_extract_trims_pad_silence_to_real_frames(whisper_tiny):
    """Whisper pads every clip to 30 s, so the encoder always emits 1500 frames;
    extract must trim to the real ``round(clip_s × 50)`` frames so the cache
    matches the ``(clip_s × 50, d)`` pool contract and pad-silence does not
    poison fit_channel_stats (H3). Real audio is left-aligned in the 30-s window,
    so the kept prefix is the speech frames."""
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    sr = WHISPER_SR
    try:
        for clip_s, expected in [(5.0, 250), (3.0, 150), (10.0, 500)]:
            wav = np.zeros(int(sr * clip_s), dtype=np.float32)
            feat = fe.extract(wav, sample_rate=sr)
            assert feat.shape[0] == expected, (
                f"{clip_s}s clip → {feat.shape[0]} frames, want {expected}"
            )
            assert feat.shape[1] == 384
    finally:
        fe.close()


def test_mean_all_is_plain_mean_over_all_layers(whisper_tiny):
    """Default mean_all hooks every layer and returns their plain mean — NOT a
    single layer, and NOT per-layer normalized. Reproduce the ceiling-probe
    mean_all merge: equal (within fp16) to the mean of the 4 single-layer reads,
    and distinct from any single layer."""
    model, proc = whisper_tiny
    sr = WHISPER_SR
    wav = (np.random.RandomState(1).randn(int(sr * 5.0)) * 0.1).astype(np.float32)

    fe_mean = WhisperFeatureExtractor(model, proc)  # default = "mean_all"
    try:
        assert fe_mean.layer_merge == "mean_all"
        merged = fe_mean.extract(wav, sample_rate=sr).float()
    finally:
        fe_mean.close()

    singles = []
    for L in range(4):
        fe_L = WhisperFeatureExtractor(model, proc, layer_merge=L)
        try:
            singles.append(fe_L.extract(wav, sample_rate=sr).float())
        finally:
            fe_L.close()

    ref = torch.stack(singles, dim=0).mean(dim=0)
    # 5 s × 50 Hz = 250 real frames after the H3 pad-silence trim.
    assert merged.shape == ref.shape == (250, 384)
    # Plain unweighted mean over all 4 layers. Compare by mean-abs-diff: the two
    # paths differ only in fp16 rounding order (mean-then-cast vs cast-then-mean),
    # so a few large-magnitude outlier dims would trip an elementwise max check
    # even though the underlying mean is identical.
    assert (merged - ref).abs().mean().item() < 5e-3
    # ...and genuinely a merge: each single layer is far from the all-layer mean.
    for L, single in enumerate(singles):
        assert (merged - single).abs().mean().item() > 5e-2, f"mean_all ~= layer {L}"


def test_extract_rejects_wrong_sample_rate(whisper_tiny):
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    try:
        with pytest.raises(ValueError, match="16000"):
            fe.extract(np.zeros(1000, dtype=np.float32), sample_rate=22050)
    finally:
        fe.close()


def test_write_clip_cache_round_trip(whisper_tiny):
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    sr = WHISPER_SR
    wav = (np.random.RandomState(0).randn(int(sr * 5.0)) * 0.1).astype(np.float32)
    try:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            entry = write_clip_cache(
                fe, wav, sample_rate=sr,
                clip_id="anchor_0042", film="megamind",
                t0_movie_s=42.5, out_dir=out_dir,
            )
            assert entry.n_frames > 0
            assert entry.d_model == 384
            saved_path = Path(entry.out_path)
            assert saved_path.exists()
            payload = torch.load(saved_path, weights_only=False)
            assert payload["rate_hz"] == 50
            assert payload["t0_movie_s"] == 42.5
            assert payload["layer_merge"] == 3
            assert payload["features"].shape == (entry.n_frames, 384)
    finally:
        fe.close()


# --- B33 per-channel target standardizer -------------------------------------


def test_fit_channel_stats_shape_keys_and_dtype(tmp_path):
    """Default d_model=1280; fp32 accumulation (not fp16). Clips are 250 frames
    (5 s × 50 Hz) — fit pools each 250→40 before accumulating (H2)."""
    torch.manual_seed(0)
    paths = []
    for i in range(3):
        p = tmp_path / f"clip_{i}.pt"
        _save_features(p, torch.randn(250, 1280))
        paths.append(p)
    stats = fit_channel_stats(paths)
    assert set(stats) == {"mean", "inv_std"}
    assert stats["mean"].shape == (1280,)
    assert stats["inv_std"].shape == (1280,)
    assert stats["mean"].dtype == torch.float32
    assert stats["inv_std"].dtype == torch.float32
    assert torch.isfinite(stats["mean"]).all()
    assert torch.isfinite(stats["inv_std"]).all()


def test_fit_channel_stats_is_train_only(tmp_path):
    """Stats use the passed paths ONLY; a held-out clip never enters them."""
    from speech_decoding.extractors.whisper_teacher_pool import (
        triangular_pool_50_to_8_hz,
    )

    torch.manual_seed(1)
    d = 8
    train_paths, train_feats = [], []
    for i in range(3):
        feat = torch.randn(250, d) + 2.0
        train_feats.append(feat.to(torch.float16).float())  # match fp16 round-trip
        p = tmp_path / f"train_{i}.pt"
        _save_features(p, feat)
        train_paths.append(p)
    val_p = tmp_path / "val_0.pt"
    _save_features(val_p, torch.randn(250, d) + 100.0)  # wildly different scale

    stats_train = fit_channel_stats(train_paths, d_model=d)
    # Reference fit = pool each train clip 250→40, then mean over pooled frames
    # (fit happens at the 8 Hz pooled rate, not the 50 Hz cache rate — H2).
    pooled = torch.cat(
        [triangular_pool_50_to_8_hz(f.unsqueeze(0)).squeeze(0) for f in train_feats],
        dim=0,
    )
    assert torch.allclose(stats_train["mean"], pooled.mean(dim=0), atol=1e-2)
    # Including the val clip must move the stats — proves train-only is real.
    stats_all = fit_channel_stats(train_paths + [val_p], d_model=d)
    assert not torch.allclose(stats_all["mean"], stats_train["mean"], atol=1.0)


def test_fit_channel_stats_zero_variance_guard(tmp_path):
    """A channel constant across all clips/timesteps → inv_std == 1 (pass-through)."""
    torch.manual_seed(2)
    d = 4
    paths = []
    for i in range(3):
        feat = torch.randn(250, d)
        feat[:, 0] = 7.0  # channel 0 constant everywhere (stays constant post-pool)
        p = tmp_path / f"c_{i}.pt"
        _save_features(p, feat)
        paths.append(p)
    stats = fit_channel_stats(paths, d_model=d)
    assert stats["inv_std"][0] == 1.0
    assert torch.isfinite(stats["inv_std"]).all()
    assert (stats["inv_std"][1:] != 1.0).any()  # varying channels are scaled


def test_target_standardizer_gives_unit_variance_on_pooled_target(tmp_path):
    """Stats are fit on the 8 Hz POOLED target, so standardizing the pooled
    frames gives mean 0 / unit variance. H2 regression: applying the same stats
    to the raw 50 Hz frames over-shrinks — the pool averages ~12-13 frames per
    bucket, so raw variance is ~9× the pooled variance, and fit-rate must match
    apply-rate (fitting on raw, applying to pooled gave the std≈0.33 bug)."""
    from speech_decoding.extractors.whisper_teacher_pool import (
        triangular_pool_50_to_8_hz,
    )

    torch.manual_seed(3)
    d = 16
    paths, feats = [], []
    for i in range(4):
        feat = torch.randn(250, d) * 3.0 + 5.0
        feats.append(feat.to(torch.float16).float())
        p = tmp_path / f"t_{i}.pt"
        _save_features(p, feat)
        paths.append(p)
    stats = fit_channel_stats(paths, d_model=d)
    standardizer = TargetStandardizer(stats["mean"], stats["inv_std"])

    pooled = torch.cat(
        [triangular_pool_50_to_8_hz(f.unsqueeze(0)).squeeze(0) for f in feats], dim=0
    ).unsqueeze(0)  # (1, N_pooled, d)
    z = standardizer(pooled)
    assert z.shape == pooled.shape
    assert torch.allclose(z.mean(dim=(0, 1)), torch.zeros(d), atol=1e-3)
    assert torch.allclose(z.var(dim=(0, 1), unbiased=False), torch.ones(d), atol=1e-2)

    # Apply the pooled-fit stats to the RAW 50 Hz frames → variance ≫ 1.
    raw = torch.cat(feats, dim=0).unsqueeze(0)  # (1, N_raw, d)
    z_raw = standardizer(raw)
    assert (z_raw.var(dim=(0, 1), unbiased=False) > 2.0).all()


def test_target_standardizer_buffers_not_params_and_preserves_shape():
    standardizer = TargetStandardizer(torch.zeros(1280), torch.ones(1280))
    assert sum(p.numel() for p in standardizer.parameters()) == 0  # non-trainable
    buffers = dict(standardizer.named_buffers())
    assert "mean" in buffers and "inv_std" in buffers
    x = torch.randn(2, 40, 1280)
    out = standardizer(x)
    assert out.shape == (2, 40, 1280)
    assert torch.allclose(out, x)  # identity affine = no-op


def test_target_standardizer_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="matching 1-D"):
        TargetStandardizer(torch.zeros(1280), torch.ones(256))
    with pytest.raises(ValueError, match="matching 1-D"):
        TargetStandardizer(torch.zeros(2, 1280), torch.ones(2, 1280))


def test_fit_channel_stats_empty_raises():
    with pytest.raises(ValueError, match="no frames"):
        fit_channel_stats([], d_model=8)
