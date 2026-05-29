"""Tests for tiered frame-alignment verification.

Synthetic + real-data assertions for crop, hash, SSIM behavior.
"""
import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from speech_decoding.bt_alignment.verify_frame import (
    BT_FPS,
    DEFAULT_PHASH_THRESH, DEFAULT_DHASH_THRESH, DEFAULT_SSIM_THRESH,
    crop_letterbox,
    compare_frames,
)


def test_bt_fps_constant():
    assert BT_FPS == 23.976


def test_crop_letterbox_strips_black_bars():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[20:80, 30:170] = 200  # active picture inside black border
    out = crop_letterbox(img)
    assert out.shape == (60, 140, 3)


def test_crop_letterbox_no_black_bars():
    img = (np.random.rand(50, 100, 3) * 200 + 30).astype(np.uint8)
    out = crop_letterbox(img)
    assert out.shape == img.shape


def test_crop_letterbox_all_black_returns_full():
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    out = crop_letterbox(img)
    # all black has no content above threshold, so it should return the original
    assert out.shape == img.shape


def test_compare_identical_image_passes_exact():
    rng = np.random.default_rng(42)
    img = (rng.uniform(50, 200, (240, 320, 3))).astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.png"
        Image.fromarray(img).save(p)
        r = compare_frames(p, p)
    assert r["exact_match"] is True
    assert r["phash_dist"] == 0
    assert r["dhash_dist"] == 0
    assert r["pass_frame"] is True


def test_compare_slight_noise_passes_hash():
    rng = np.random.default_rng(42)
    img = (rng.uniform(50, 200, (240, 320, 3))).astype(np.uint8)
    noise = rng.integers(-3, 4, img.shape)
    noisy = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.png"; p2 = Path(td) / "b.png"
        Image.fromarray(img).save(p1)
        Image.fromarray(noisy).save(p2)
        r = compare_frames(p1, p2)
    assert r["exact_match"] is False
    assert r["phash_dist"] <= DEFAULT_PHASH_THRESH
    assert r["dhash_dist"] <= DEFAULT_DHASH_THRESH
    assert r["pass_frame"] is True


def test_compare_completely_different_images_fail():
    rng = np.random.default_rng(0)
    img1 = (rng.uniform(0, 60, (240, 320, 3))).astype(np.uint8)
    img2 = (rng.uniform(180, 240, (240, 320, 3))).astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.png"; p2 = Path(td) / "b.png"
        Image.fromarray(img1).save(p1)
        Image.fromarray(img2).save(p2)
        r = compare_frames(p1, p2)
    assert r["pass_frame"] is False
    # at least one hash should be way over threshold
    assert max(r["phash_dist"], r["dhash_dist"]) > DEFAULT_PHASH_THRESH


def test_compare_identical_bt_frame_self_passes():
    """Sanity: a real BT reference PNG vs itself must pass exact."""
    bt = Path("/Users/bentang/Documents/Code/speech/.cache/braintreebank/movie_frames/aquaman/frame_1000.png")
    if not bt.exists():
        pytest.skip("BT reference frame not present")
    r = compare_frames(bt, bt)
    assert r["exact_match"] is True
    assert r["pass_frame"] is True


def test_thresholds_are_starting_points():
    assert DEFAULT_PHASH_THRESH == 8
    assert DEFAULT_DHASH_THRESH == 8
    assert DEFAULT_SSIM_THRESH == 0.7
