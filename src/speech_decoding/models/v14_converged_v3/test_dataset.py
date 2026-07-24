"""v14_converged_v3 Phase D2/D4 — V3SessionDataset (TDD, synthetic caches).

Memo project-v3-pipeline-build-contract-2026-07-10: the map-style dataset over the
per-session continuous |STFT| spec caches. Each dataset index is one random-
CONTINUOUS clip draw from a session; ``__getitem__`` (1) samples a guard-2-valid t0
(``clip_sampler``), (2) memmap-reads the 3 band ``.npy`` caches at the survivor rows
(``keep_idx``) over ``[t0, t0+T)``, (3) applies the session's FROZEN robust-z
(reused ``SessionRobustZNormalizer.from_stats``), and returns a ``V3ClipSample``.
Session identity per index feeds the reused ``_SessionGroupedBatchSampler``.

Tested against tiny synthetic ``.npy`` caches — the real braintreebank file-format
adapter (labels / LOF / parcel support) is a thin layer validated on DeltaAI at F2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.models.v14_converged_v3.batch import v3_collate
from speech_decoding.models.v14_converged_v3.dataset import (
    NATIVE_FINE_BAND_RATES,
    R5_BAND_RATES,
    R6_BAND_RATES,
    UNIFORM_BAND_RATES,
    V3SessionDataset,
    assert_band_rates_match_cache,
    build_session_spec,
)
from speech_decoding.models.v14_converged_v3.session_setup import build_session_setup

F_SLOW, F_MID, F_HGA = 7, 6, 7
T_CLIP = 96
FPS = 32.0


def _write_band(tmp, name, c, f, t_total):
    arr = (np.random.RandomState(abs(hash(name)) % 2**32)
           .randn(c, f, t_total).astype(np.float32))
    path = str(tmp / f"{name}.npy")
    np.save(path, arr)
    return path, arr


def _spec(tmp, *, key=(1, 0), shaft_sizes=(4, 3, 3), t_total=4000,
          bad_spans=(), drop=frozenset()):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    c_full = len(labels)
    setup = build_session_setup(labels, torch.tensor(parcels), drop_labels=drop)
    band_paths, band_fbins = [], (F_SLOW, F_MID, F_HGA)
    for bname, fb in zip(("v3slow", "v3mid", "hga"), band_fbins):
        p, _ = _write_band(tmp, f"{key[0]}_{key[1]}_{bname}", c_full, fb, t_total)
        band_paths.append(p)
    # frozen robust-z stats over the FULL c_full rows (adapter slices to survivors)
    band_stats = [
        (torch.randn(c_full, fb, 1), torch.rand(c_full, fb, 1) + 0.5)
        for fb in band_fbins
    ]
    return build_session_spec(
        session_key=key,
        band_paths=tuple(band_paths),
        band_stats=tuple(band_stats),
        setup=setup,
        n_frames=t_total,
        bad_spans_s=list(bad_spans),
    )


def test_getitem_returns_three_bands_survivor_rows(tmp_path) -> None:
    spec = _spec(tmp_path)
    n = len(spec.setup.sidecar.labels)
    ds = V3SessionDataset([spec], clips_per_session=5, clip_frames=T_CLIP, fps=FPS)
    sample = ds[0]
    assert len(sample.bands) == 3
    assert sample.bands[0].shape == (n, F_SLOW, T_CLIP)
    assert sample.bands[1].shape == (n, F_MID, T_CLIP)
    assert sample.bands[2].shape == (n, F_HGA, T_CLIP)
    assert sample.session_key == (1, 0)


def test_len_is_total_clip_budget(tmp_path) -> None:
    a = _spec(tmp_path / "a" if False else tmp_path, key=(1, 0))
    b = _spec(tmp_path, key=(2, 1), shaft_sizes=(3, 3))
    ds = V3SessionDataset([a, b], clips_per_session=7, clip_frames=T_CLIP, fps=FPS)
    assert len(ds) == 14


def test_session_key_list_aligns_to_index(tmp_path) -> None:
    a = _spec(tmp_path, key=(1, 0))
    b = _spec(tmp_path, key=(2, 1), shaft_sizes=(3, 3))
    ds = V3SessionDataset([a, b], clips_per_session=3, clip_frames=T_CLIP, fps=FPS)
    keys = ds.session_key_list()
    assert len(keys) == 6
    assert keys[:3] == [(1, 0)] * 3
    assert keys[3:] == [(2, 1)] * 3


def test_getitem_reads_survivor_rows_after_drop(tmp_path) -> None:
    # Drop LB1 → survivors exclude it; the clip must have N-1 rows, in survivor order.
    spec = _spec(tmp_path, drop=frozenset({"LB1"}))
    n = len(spec.setup.sidecar.labels)
    assert "LB1" not in spec.setup.sidecar.labels
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    assert ds[0].bands[0].shape[0] == n


def test_robust_z_matches_reused_normalizer(tmp_path) -> None:
    # The clip the dataset returns must equal a direct read + from_stats transform.
    spec = _spec(tmp_path)
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    ds.set_epoch(0)
    # force a deterministic t0 by making the whole session valid + seeding
    sample = ds[0]
    # recompute band 0 with the same normalizer at the SAME t0 the dataset used
    from speech_decoding.extractors.normalize import SessionRobustZNormalizer

    t0 = ds._last_t0  # test hook: the t0 __getitem__ drew
    mm = np.load(spec.band_paths[0], mmap_mode="r")
    keep = spec.keep_idx.numpy()
    raw = torch.from_numpy(mm[keep, :, t0:t0 + T_CLIP].astype(np.float32))
    med, sig = spec.band_stats[0]
    norm = SessionRobustZNormalizer.from_stats(median=med, sigma=sig)
    assert torch.allclose(sample.bands[0], norm.transform(raw))


def test_guard2_never_reads_a_bad_span(tmp_path) -> None:
    # A bad span in the MIDDLE; over many draws no clip may overlap it.
    a, b = 50.0, 60.0  # seconds → frames [1600,1920)
    spec = _spec(tmp_path, t_total=4000, bad_spans=[(a, b)])
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    fa, fb = round(a * FPS), round(b * FPS)
    for ep in range(60):
        ds.set_epoch(ep)
        ds[0]
        t0 = ds._last_t0
        assert not (t0 < fb and t0 + T_CLIP > fa)


def test_epoch_changes_the_draw(tmp_path) -> None:
    spec = _spec(tmp_path, t_total=8000)
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    draws = set()
    for ep in range(5):
        ds.set_epoch(ep)
        ds[0]
        draws.add(ds._last_t0)
    assert len(draws) > 1  # epochs re-draw the random-continuous window


def test_collate_integration_produces_batch(tmp_path) -> None:
    spec = _spec(tmp_path)
    n = len(spec.setup.sidecar.labels)
    ds = V3SessionDataset([spec], clips_per_session=8, clip_frames=T_CLIP, fps=FPS)
    batch = v3_collate([ds[i] for i in range(4)])
    assert batch.bands[0].shape == (4, n, F_SLOW, T_CLIP)
    assert batch.session_key == (1, 0)


def test_build_session_spec_slices_stats_to_survivors(tmp_path) -> None:
    # band_stats come in at c_full rows; the spec must slice them to keep_idx so they
    # align with the survivor clip rows the mmap read returns.
    spec = _spec(tmp_path, drop=frozenset({"LA1"}))
    n = len(spec.setup.sidecar.labels)
    for med, sig in spec.band_stats:
        assert med.shape[0] == n and sig.shape[0] == n


def test_per_band_winsor_caps_each_normalizer(tmp_path) -> None:
    # v3 winsor is per-band (SLOW/MID 15, HGA 20): each band's frozen normalizer must
    # carry its own |z| cap so the looser HGA clamp doesn't leak onto SLOW/MID.
    spec = _spec_winsor(tmp_path, winsor=(15.0, 15.0, 20.0))
    caps = [nrm.winsor for nrm in spec.band_norms]
    assert caps == [15.0, 15.0, 20.0]
    # a scalar broadcasts to all bands (back-compat with the single-cap contract)
    spec_scalar = _spec_winsor(tmp_path, winsor=12.0)
    assert [nrm.winsor for nrm in spec_scalar.band_norms] == [12.0, 12.0, 12.0]
    # and the clamp actually bites: a +100σ outlier is capped at the band's cap
    z = spec.band_norms[2].transform(
        spec.band_stats[2][0] + 100.0 * spec.band_stats[2][1]
    )
    assert float(z.max()) <= 20.0 + 1e-5 and float(z.max()) >= 20.0 - 1e-5


# ── native-rate multi-rate read (fine-HGA, 2026-07-21) ─────────────────────────
# Under native rates the 3 band npys have DIFFERENT frame counts: SLOW T/8 (4 Hz),
# MID T/2 (16 Hz), HGA T·4 (128 Hz). A clip of clip_frames (32 Hz) reads each band at
# its own offset t0·num//den for len clip_frames·num//den. band_rates default (1,1)×3
# = uniform = byte-identical to arm0. Binding align = lcm(dens); clip_frames must be
# divisible so every per-band clip length is an integer.
FINE_RATES = ((1, 8), (1, 2), (4, 1))  # (slow, mid, hga) relative to 32 Hz


def _spec_multirate(tmp, *, key=(1, 0), t32=8000, band_rates=FINE_RATES, bad_spans=()):
    """Synthetic spec whose per-band npys carry NATIVE frame counts (t32·num//den)."""
    labels, parcels = [], []
    for s, n in enumerate((4, 3, 3)):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    c_full = len(labels)
    setup = build_session_setup(labels, torch.tensor(parcels), drop_labels=frozenset())
    band_paths, arrays, band_fbins = [], [], (F_SLOW, F_MID, 4)  # HGA fine = 4 bins
    for bname, fb, (num, den) in zip(("v3slow", "v3mid", "v3hga"), band_fbins, band_rates):
        t_band = t32 * num // den
        p, arr = _write_band(tmp, f"mr_{key[0]}_{key[1]}_{bname}", c_full, fb, t_band)
        band_paths.append(p)
        arrays.append(arr)
    band_stats = [
        (torch.zeros(c_full, fb, 1), torch.ones(c_full, fb, 1)) for fb in band_fbins
    ]
    # n_frames is the 32 Hz REFERENCE (what session_loader derives); here == t32.
    spec = build_session_spec(
        session_key=key, band_paths=tuple(band_paths), band_stats=tuple(band_stats),
        setup=setup, n_frames=t32, bad_spans_s=list(bad_spans),
    )
    return spec, arrays


def test_native_rates_read_per_band_at_own_offset(tmp_path) -> None:
    # SLOW [t0/8:+12], MID [t0/2:+48], HGA [t0·4:+384] for clip_frames=96, and the
    # returned clip must equal the raw native-slice (stats are 0/1 ⇒ identity norm).
    spec, arrays = _spec_multirate(tmp_path, t32=8000)
    n = len(spec.setup.sidecar.labels)
    keep = spec.keep_idx.numpy()
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS,
                          band_rates=FINE_RATES)
    sample = ds[0]
    t0 = ds._last_t0
    assert t0 % 8 == 0, f"t0={t0} not 8-aligned"
    assert sample.bands[0].shape == (n, F_SLOW, T_CLIP // 8)   # 12
    assert sample.bands[1].shape == (n, F_MID, T_CLIP // 2)    # 48
    assert sample.bands[2].shape == (n, 4, T_CLIP * 4)         # 384
    for bi, (num, den) in enumerate(FINE_RATES):
        lo, hi = t0 * num // den, (t0 + T_CLIP) * num // den
        expect = torch.from_numpy(arrays[bi][keep, :, lo:hi].astype(np.float32))
        assert torch.allclose(sample.bands[bi], expect), f"band {bi} native slice mismatch"


def test_uniform_rates_default_is_byte_identical(tmp_path) -> None:
    # Omitting band_rates (default uniform) must reproduce the single-rate read exactly.
    spec = _spec(tmp_path, t_total=4000)
    ds_default = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS)
    ds_explicit = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS,
                                   band_rates=((1, 1), (1, 1), (1, 1)))
    ds_default.set_epoch(3); ds_explicit.set_epoch(3)
    a, b = ds_default[0], ds_explicit[0]
    assert ds_default._last_t0 == ds_explicit._last_t0
    for x, y in zip(a.bands, b.bands):
        assert torch.equal(x, y)


def test_native_clip_len_indivisible_raises(tmp_path) -> None:
    # clip_frames=100 with SLOW den=8 ⇒ 100·1/8 not integer ⇒ band clip length undefined.
    spec, _ = _spec_multirate(tmp_path)
    import pytest
    with pytest.raises(ValueError, match="not integer|divisible"):
        V3SessionDataset([spec], clips_per_session=1, clip_frames=100, fps=FPS,
                         band_rates=FINE_RATES)


def test_native_guard2_span_still_excluded_in_32hz(tmp_path) -> None:
    # guard-2 stays on the 32 Hz clock (t0, clip_frames) — a bad span excludes t0 the
    # same way regardless of per-band rates.
    a, b = 50.0, 60.0  # → 32 Hz frames [1600,1920)
    spec, _ = _spec_multirate(tmp_path, t32=8000, bad_spans=[(a, b)])
    ds = V3SessionDataset([spec], clips_per_session=1, clip_frames=T_CLIP, fps=FPS,
                          band_rates=FINE_RATES)
    fa, fb = round(a * FPS), round(b * FPS)
    for ep in range(60):
        ds.set_epoch(ep)
        ds[0]
        t0 = ds._last_t0
        assert t0 % 8 == 0
        assert not (t0 < fb and t0 + T_CLIP > fa)


def _spec_winsor(tmp, *, winsor):
    labels, parcels = [], []
    for s, n in enumerate((4, 3, 3)):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    c_full = len(labels)
    setup = build_session_setup(labels, torch.tensor(parcels), drop_labels=frozenset())
    band_paths, band_fbins = [], (F_SLOW, F_MID, F_HGA)
    for bname, fb in zip(("v3slow", "v3mid", "hga"), band_fbins):
        p, _ = _write_band(tmp, f"w_{bname}", c_full, fb, 4000)
        band_paths.append(p)
    band_stats = [
        (torch.randn(c_full, fb, 1), torch.rand(c_full, fb, 1) + 0.5)
        for fb in band_fbins
    ]
    return build_session_spec(
        session_key=(1, 0),
        band_paths=tuple(band_paths),
        band_stats=tuple(band_stats),
        setup=setup,
        n_frames=4000,
        bad_spans_s=[],
        winsor=winsor,
    )


# ---------------------------------------------------------------------------
# band_rates vs cache-rate guard (r6 regression, 2026-07-23). R6_BAND_RATES declared
# SLOW 4 Hz / MID 16 Hz against caches that were all baked at 32 Hz, so the per-band
# window t0·num//den read a COMPRESSED, TIME-SHIFTED slice at the right SHAPE — three
# bands from three unrelated moments of the recording. Every downstream shape check
# passed, so only an explicit rate assert can catch it.
# ---------------------------------------------------------------------------
def test_band_rates_guard_rejects_r6_native_rates_on_32hz_caches() -> None:
    with pytest.raises(ValueError, match="but its cache is 32 Hz"):
        assert_band_rates_match_cache([32, 32, 32], R6_BAND_RATES, where="session 1/0")


def test_band_rates_guard_names_the_offending_band_and_session() -> None:
    with pytest.raises(ValueError) as ei:
        assert_band_rates_match_cache([32, 32, 32], R6_BAND_RATES, where="session 1/0")
    msg = str(ei.value)
    assert "session 1/0" in msg and "band 0" in msg  # SLOW is the first mismatch


def test_band_rates_guard_accepts_every_shipped_arm_against_its_real_bake() -> None:
    # arm0/r4: all three bands baked at 32 Hz, stem decimates ::8/::2/::1
    assert_band_rates_match_cache([32, 32, 32], UNIFORM_BAND_RATES)
    # r5/r5nf: both streams baked at 64 Hz = 2× the clip clock
    assert_band_rates_match_cache([64, 64], R5_BAND_RATES)
    # a TRUE native 3-band bake (SLOW 4 / MID 16 / HGA 32) is what R6_BAND_RATES meant
    assert_band_rates_match_cache([4, 16, 32], R6_BAND_RATES)
    # fine-HGA: SLOW 4 / MID 16 / HGA 128
    assert_band_rates_match_cache([4, 16, 128], NATIVE_FINE_BAND_RATES)


def test_band_rates_guard_rejects_a_half_rate_bake() -> None:
    # uniform rates against a cache baked at 16 Hz — the mirror of the r6 failure
    with pytest.raises(ValueError, match="but its cache is 16 Hz"):
        assert_band_rates_match_cache([32, 16, 32], UNIFORM_BAND_RATES)
