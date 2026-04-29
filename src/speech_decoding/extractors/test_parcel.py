"""Tests for V14ParcelMetadataExtractor against a synthetic per-patient cache."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from speech_decoding.atlas.support import TIER1_COLUMNS, write_support_cache
from speech_decoding.extractors.parcel import V14ParcelMetadataExtractor


def _write_fake_caches(
    tmp_path: Path, subject: str, names: tuple[str, ...]
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    n_ch = len(names)
    n_cols = len(TIER1_COLUMNS)
    rng = np.random.default_rng(0)

    support_pct = (
        rng.uniform(0.0, 100.0, size=(n_ch, n_cols)).astype(np.float32)
    )
    support_dir = tmp_path / "support_cache"
    write_support_cache(
        support_dir / f"{subject}_support_tier1.csv", names, support_pct
    )

    coords = rng.uniform(-60.0, 60.0, size=(n_ch, 3)).astype(np.float32)
    coords_dir = tmp_path / "fsaverage_coords"
    coords_dir.mkdir(parents=True, exist_ok=True)
    coords_path = coords_dir / f"{subject}_fsaverage_pial.csv"
    with coords_path.open("w") as f:
        f.write("name,hemisphere,fsaverage_vertex,x,y,z\n")
        for i, name in enumerate(names):
            x, y, z = coords[i]
            f.write(f"{name},L,{i},{x:.6f},{y:.6f},{z:.6f}\n")

    return support_dir, coords_dir, support_pct, coords


def test_parcel_extractor_emits_argmax_and_xyz(tmp_path: Path) -> None:
    names = tuple(f"E{i:03d}" for i in range(8))
    support_dir, coords_dir, support_pct, coords = _write_fake_caches(
        tmp_path, "S14", names
    )

    ext = V14ParcelMetadataExtractor(
        event_types="Ieeg",
        support_cache_dir=str(support_dir),
        fsaverage_coords_dir=str(coords_dir),
    )
    event = SimpleNamespace(subject="S14")
    out = ext.get_static(event)  # type: ignore[arg-type]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (len(names), 5)
    assert out.dtype == torch.float32

    expected_parcel = support_pct.argmax(axis=1).astype(np.float32)
    expected_support = (support_pct.max(axis=1) / 100.0).astype(np.float32)
    np.testing.assert_array_equal(out[:, 0].numpy(), expected_parcel)
    np.testing.assert_allclose(out[:, 1].numpy(), expected_support, rtol=1e-5)
    np.testing.assert_allclose(out[:, 2:5].numpy(), coords, rtol=1e-5)


def test_parcel_extractor_raises_on_missing_coords(tmp_path: Path) -> None:
    names = tuple(f"E{i:03d}" for i in range(4))
    support_dir, coords_dir, _, _ = _write_fake_caches(tmp_path, "S14", names)
    (coords_dir / "S14_fsaverage_pial.csv").unlink()

    ext = V14ParcelMetadataExtractor(
        event_types="Ieeg",
        support_cache_dir=str(support_dir),
        fsaverage_coords_dir=str(coords_dir),
    )
    with pytest.raises(FileNotFoundError, match="fsaverage coords cache"):
        ext.get_static(SimpleNamespace(subject="S14"))  # type: ignore[arg-type]
