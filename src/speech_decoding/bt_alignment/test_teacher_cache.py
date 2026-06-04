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
    WhisperFeatureExtractor, write_clip_cache, write_movie_cache,
    fit_channel_stats, fit_and_save_channel_stats, movie_cache_path,
    TargetStandardizer,
    DEFAULT_LAYER_MERGE, SINGLE_LAYER_SISTER_INDEX, DEFAULT_TEACHER_HZ, WHISPER_SR,
    WHISPER_ENCODER_WINDOW_S,
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


def test_extract_aligns_input_dtype_to_model_dtype(whisper_tiny):
    """Regression: transformers v5 loads large-v3 in its NATIVE fp16, so the
    encoder conv1d sees Half weights while the processor emits a float32 mel —
    a raw ``.to(device)`` raised "Input type (float) and bias type (c10::Half)".
    ``extract`` must cast the mel to the model's param dtype. Spied via a fake
    model so the assertion runs on CPU (real CPU fp16 conv is unsupported);
    checks BOTH dtypes to prove it matches the model, not "always fp16"."""
    model, proc = whisper_tiny
    seen: dict[str, torch.dtype] = {}

    class _Inner:
        def __init__(self, captures: dict, layers: list[int]) -> None:
            self._captures, self._layers = captures, layers

        def encoder(self, input_features: torch.Tensor) -> None:
            seen["dtype"] = input_features.dtype
            for layer in self._layers:  # stand in for the real forward hooks
                self._captures[layer] = torch.zeros(
                    1, 1500, 384, dtype=input_features.dtype
                )

    class _FakeModel:
        def __init__(self, dtype: torch.dtype, fe) -> None:
            self._p = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
            self.model = _Inner(fe._captures, fe._layers)

        def parameters(self):
            yield self._p

    wav = np.zeros(WHISPER_SR, dtype=np.float32)  # 1 s, float32 mel input

    for model_dtype in (torch.float16, torch.float32):
        fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
        try:
            fe.model = _FakeModel(model_dtype, fe)
            feat = fe.extract(wav, sample_rate=WHISPER_SR)
            assert seen["dtype"] == model_dtype  # mel cast to MATCH the model
            assert feat.dtype == torch.float16   # cache dtype is always fp16
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
            # TCACHE-KEY-1: model identity is the wrapped model's, NOT a hardcoded
            # large-v3 constant — payload + path slug both reflect whisper-tiny.
            assert payload["model"] == "openai/whisper-tiny"
            assert "openai_whisper-tiny" in entry.out_path
    finally:
        fe.close()


def test_write_clip_cache_keys_path_by_layer_merge(whisper_tiny):
    """TCACHE-KEY-1: two configs that change the stored target (mean_all vs a
    single layer) must land in DISJOINT paths under the same out_dir, even
    with identical clip_id/film. The payload alone did not protect against an
    alt-config silently reusing the dir and mixing two targets into one fit."""
    model, proc = whisper_tiny
    sr = WHISPER_SR
    wav = (np.random.RandomState(0).randn(int(sr * 5.0)) * 0.1).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        fe_mean = WhisperFeatureExtractor(model, proc)           # mean_all
        fe_l0 = WhisperFeatureExtractor(model, proc, layer_merge=0)
        try:
            e_mean = write_clip_cache(
                fe_mean, wav, sample_rate=sr, clip_id="anchor_0",
                film="megamind", t0_movie_s=1.0, out_dir=out_dir,
            )
            e_l0 = write_clip_cache(
                fe_l0, wav, sample_rate=sr, clip_id="anchor_0",
                film="megamind", t0_movie_s=1.0, out_dir=out_dir,
            )
        finally:
            fe_mean.close()
            fe_l0.close()
        assert e_mean.out_path != e_l0.out_path
        assert Path(e_mean.out_path).exists()
        assert Path(e_l0.out_path).exists()
        assert "mean_all" in e_mean.out_path
        assert "L0" in e_l0.out_path
        # Same filename, disjoint dirs → no collision.
        assert Path(e_mean.out_path).name == Path(e_l0.out_path).name == "anchor_0.pt"


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


def test_fit_and_save_channel_stats_globs_full_corpus_and_saves(tmp_path):
    """#44 (Ben 2026-06-04): globs the movie_cache_path layout, fits the FULL
    corpus, saves a loadable {mean, inv_std} == a direct fit over the same
    caches, and the result feeds TargetStandardizer."""
    torch.manual_seed(3)
    model, merge, d = "openai/whisper-large-v3", "mean_all", 16
    paths = []
    for m in ("megamind", "venom", "spiderman"):
        p = movie_cache_path(tmp_path, model, merge, m)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_features(p, torch.randn(250, d))
        paths.append(p)

    out = tmp_path / "channel_stats.pt"
    stats = fit_and_save_channel_stats(
        tmp_path, model=model, layer_merge=merge, out_path=out, d_model=d,
    )
    assert out.exists()
    loaded = torch.load(out, weights_only=True)
    assert torch.equal(loaded["mean"], stats["mean"])
    # Full-corpus == a direct fit over the same movie caches (all movies).
    direct = fit_channel_stats(sorted(paths), d_model=d)
    assert torch.allclose(stats["mean"], direct["mean"])
    assert torch.allclose(stats["inv_std"], direct["inv_std"])
    std = TargetStandardizer(loaded["mean"], loaded["inv_std"])
    assert std(torch.randn(2, 40, d)).shape == (2, 40, d)


def test_fit_and_save_channel_stats_excludes_output_file_from_glob(tmp_path):
    """A re-run must not ingest the saved channel_stats.pt as a movie cache,
    even when out_path lives in the same dir as the caches (no 'features' key
    → would KeyError if globbed)."""
    torch.manual_seed(4)
    model, merge, d = "m", "mean_all", 8
    for i in range(2):
        p = movie_cache_path(tmp_path, model, merge, f"movie_{i}")
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_features(p, torch.randn(250, d))
    out = movie_cache_path(tmp_path, model, merge, "movie_0").parent / "channel_stats.pt"
    first = fit_and_save_channel_stats(
        tmp_path, model=model, layer_merge=merge, out_path=out, d_model=d,
    )
    second = fit_and_save_channel_stats(  # out now exists in the glob dir
        tmp_path, model=model, layer_merge=merge, out_path=out, d_model=d,
    )
    assert torch.equal(first["mean"], second["mean"])
    assert torch.equal(first["inv_std"], second["inv_std"])


def test_fit_and_save_channel_stats_no_caches_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no movie caches"):
        fit_and_save_channel_stats(
            tmp_path, model="absent", layer_merge="mean_all",
            out_path=tmp_path / "x.pt", d_model=8,
        )


# --- whole-movie teacher cache (T14) -----------------------------------------


def _movie_wav(seconds: float, seed: int = 0) -> np.ndarray:
    """Deterministic non-silent mono movie waveform at 16 kHz."""
    n = int(round(seconds * WHISPER_SR))
    return (np.random.RandomState(seed).randn(n) * 0.05).astype(np.float32)


def test_write_movie_cache_chunks_whole_movie_not_truncated(whisper_tiny, tmp_path):
    """THE truncation guard. Whisper's encoder is fixed at a 30 s window, so a
    naive single forward over a >30 s movie keeps only its first 30 s (1500
    frames). write_movie_cache must chunk on the 30 s grid and concatenate, so a
    65 s movie yields round(65 × 50) = 3250 frames (1500 + 1500 + 250), NOT
    1500. Frame f maps to movie-time f / 50."""
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    try:
        wav = _movie_wav(65.0)
        entry = write_movie_cache(
            fe, wav, sample_rate=WHISPER_SR, movie="megamind", out_dir=tmp_path,
        )
        assert entry.n_frames == 3250, (
            f"65 s movie → {entry.n_frames} frames; want 3250 "
            "(truncated to 1500 == the 30 s-window bug)"
        )
        assert entry.n_frames != 1500
        assert entry.d_model == 384
        assert abs(entry.duration_s - 65.0) < 1e-3
        assert entry.chunk_s == WHISPER_ENCODER_WINDOW_S
        payload = torch.load(entry.out_path, weights_only=False)
        assert payload["features"].shape == (3250, 384)
        assert payload["features"].dtype == torch.float16
        assert payload["rate_hz"] == 50
        assert payload["t0_movie_s"] == 0.0
        assert payload["movie"] == "megamind"
    finally:
        fe.close()


def test_write_movie_cache_dense_equals_chunk_concat(whisper_tiny, tmp_path):
    """The dense stream is EXACTLY the per-chunk extracts concatenated in order;
    each full 30 s chunk contributes exactly 1500 frames. This is what makes the
    frame↔movie-time map exact across chunk boundaries (no drift, no overlap)."""
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    try:
        wav = _movie_wav(65.0, seed=1)
        entry = write_movie_cache(
            fe, wav, sample_rate=WHISPER_SR, movie="m", out_dir=tmp_path,
        )
        dense = torch.load(entry.out_path, weights_only=False)["features"]
        chunk_n = int(round(WHISPER_ENCODER_WINDOW_S * WHISPER_SR))
        expected = torch.cat(
            [fe.extract(wav[s : s + chunk_n], WHISPER_SR)
             for s in range(0, len(wav), chunk_n)],
            dim=0,
        )
        assert dense.shape == expected.shape == (3250, 384)
        torch.testing.assert_close(dense, expected, atol=0.0, rtol=0.0)
        # Full chunks are exactly 1500 frames → grid is exact.
        assert fe.extract(wav[:chunk_n], WHISPER_SR).shape[0] == 1500
    finally:
        fe.close()


def test_write_movie_cache_frame_maps_to_movie_time(whisper_tiny, tmp_path):
    """A clip at movie-onset t0 is the slice dense[round(t0×50) : +250]. Verify
    the slicing contract the segmenter (T20) will use, including a clip that
    STRADDLES a 30 s chunk seam (t0=29.5 s → frames 1475..1725, drawing 25
    frames from chunk 0 and 225 from chunk 1) — the concat must make that seam
    invisible to movie-time indexing."""
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    rate = DEFAULT_TEACHER_HZ
    try:
        wav = _movie_wav(65.0, seed=2)
        entry = write_movie_cache(
            fe, wav, sample_rate=WHISPER_SR, movie="m", out_dir=tmp_path,
        )
        dense = torch.load(entry.out_path, weights_only=False)["features"]
        chunk_n = int(round(WHISPER_ENCODER_WINDOW_S * WHISPER_SR))
        c0 = fe.extract(wav[:chunk_n], WHISPER_SR)
        c1 = fe.extract(wav[chunk_n : 2 * chunk_n], WHISPER_SR)

        # In-chunk clip at t0 = 10 s → frames [500:750] from chunk 0.
        f0 = round(10.0 * rate)
        torch.testing.assert_close(dense[f0 : f0 + 250], c0[500:750], atol=0.0, rtol=0.0)

        # Seam-straddling clip at t0 = 29.5 s → frames [1475:1725].
        fs = round(29.5 * rate)
        assert fs == 1475
        seam = torch.cat([c0[1475:1500], c1[0:225]], dim=0)
        torch.testing.assert_close(dense[fs : fs + 250], seam, atol=0.0, rtol=0.0)
    finally:
        fe.close()


def test_write_movie_cache_keys_path_by_model_and_merge(whisper_tiny, tmp_path):
    """Like write_clip_cache: model + layer-merge fold into the PATH so two
    target-defining configs can never share one movie file. One file per movie
    (no clip_id / film subdir)."""
    model, proc = whisper_tiny
    wav = _movie_wav(35.0, seed=3)
    fe_mean = WhisperFeatureExtractor(model, proc)            # mean_all
    fe_l0 = WhisperFeatureExtractor(model, proc, layer_merge=0)
    try:
        e_mean = write_movie_cache(fe_mean, wav, WHISPER_SR, "venom", tmp_path)
        e_l0 = write_movie_cache(fe_l0, wav, WHISPER_SR, "venom", tmp_path)
    finally:
        fe_mean.close()
        fe_l0.close()
    assert e_mean.out_path != e_l0.out_path
    assert "mean_all" in e_mean.out_path and "openai_whisper-tiny" in e_mean.out_path
    assert "L0" in e_l0.out_path
    assert Path(e_mean.out_path).name == Path(e_l0.out_path).name == "venom.pt"
    # 35 s movie → round(35×50)=1750 frames (1500 + 250), not 1500.
    assert e_mean.n_frames == 1750


def test_write_movie_cache_exact_multiple_has_no_partial_chunk(whisper_tiny, tmp_path):
    """A movie that is an EXACT multiple of the 30 s grid (90 s = 3 chunks) must
    yield exactly 3 × 1500 = 4500 frames — no spurious extra (empty) chunk, no
    dropped tail. Complements the 65 s partial-chunk case: this exercises the
    no-remainder branch of the chunk loop."""
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    try:
        wav = _movie_wav(90.0, seed=4)
        entry = write_movie_cache(fe, wav, WHISPER_SR, "exact", tmp_path)
        assert entry.n_frames == 4500, f"90 s → {entry.n_frames}, want 4500"
        assert abs(entry.duration_s - 90.0) < 1e-3
    finally:
        fe.close()


def test_write_movie_cache_non_round_duration_keeps_partial(whisper_tiny, tmp_path):
    """A non-grid-aligned, non-round duration (100.5 s) must give
    round(100.5 × 50) = 5025 frames = 3 full chunks (4500) + a 10.5 s partial
    (525). Guards the partial-chunk frame count at a length unlike the other
    tests, and the dense frame f → movie-time f/50 map at a non-round seam."""
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    rate = DEFAULT_TEACHER_HZ
    try:
        wav = _movie_wav(100.5, seed=5)
        entry = write_movie_cache(fe, wav, WHISPER_SR, "nonround", tmp_path)
        assert entry.n_frames == 5025, f"100.5 s → {entry.n_frames}, want 5025"
        dense = torch.load(entry.out_path, weights_only=False)["features"]
        assert dense.shape[0] == 5025
        # The partial tail (chunk 3) is the last 525 frames; verify it equals a
        # direct extract of the 10.5 s remainder audio.
        chunk_n = int(round(WHISPER_ENCODER_WINDOW_S * WHISPER_SR))
        tail = fe.extract(wav[3 * chunk_n :], WHISPER_SR)
        assert tail.shape[0] == round(10.5 * rate) == 525
        torch.testing.assert_close(dense[4500:], tail, atol=0.0, rtol=0.0)
    finally:
        fe.close()


def test_write_movie_cache_rejects_bad_inputs(whisper_tiny, tmp_path):
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    try:
        with pytest.raises(ValueError, match="16000"):
            write_movie_cache(fe, _movie_wav(5.0), 22050, "m", tmp_path)
        with pytest.raises(ValueError, match="mono"):
            write_movie_cache(
                fe, np.zeros((2, 16000), dtype=np.float32), WHISPER_SR, "m", tmp_path,
            )
    finally:
        fe.close()


class _FixedFramesExtractor:
    """Emits a fixed frame count for EVERY chunk regardless of its true duration
    — simulates a silent per-chunk truncation (the first-30-s trap) so the
    frame-count invariant in write_movie_cache has something to catch."""

    def __init__(self, frames_per_chunk: int = 100, d: int = 8):
        self._n = frames_per_chunk
        self._d = d

    def extract(self, wav, sample_rate):  # noqa: ARG002 - signature parity
        return torch.zeros(self._n, self._d, dtype=torch.float16)


def test_write_movie_cache_truncation_guard_fires(tmp_path):
    """A per-chunk truncation (fixed frames regardless of chunk length) must trip
    the frame-count invariant: a 65 s movie chunks to 3 windows → 300 frames, but
    65 s × 50 Hz ⇒ 3250 expected. Refuse to cache a broken frame↔time map."""
    fe = _FixedFramesExtractor(frames_per_chunk=100, d=8)
    with pytest.raises(RuntimeError, match="truncated/dropped"):
        write_movie_cache(fe, _movie_wav(65.0), WHISPER_SR, "trunc", tmp_path)  # type: ignore[arg-type]


def test_write_movie_cache_detects_corrupt_save(whisper_tiny, tmp_path, monkeypatch):
    """If the on-disk .pt reloads a different shape than what was written (a
    truncated/corrupt save from OOM or a full disk), the post-save reload check
    must raise rather than record a bad entry as valid."""
    model, proc = whisper_tiny
    fe = WhisperFeatureExtractor(model, proc, layer_merge=3)
    real_save = torch.save

    def corrupt_save(payload, path, *a, **k):
        bad = dict(payload)
        bad["features"] = payload["features"][:1]  # truncate the saved stream
        real_save(bad, path, *a, **k)

    monkeypatch.setattr(torch, "save", corrupt_save)
    try:
        with pytest.raises(RuntimeError, match="truncated/corrupt"):
            write_movie_cache(fe, _movie_wav(5.0), WHISPER_SR, "corrupt", tmp_path)
    finally:
        fe.close()
