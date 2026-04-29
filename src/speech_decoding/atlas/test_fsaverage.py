from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from speech_decoding.atlas.fsaverage import (
    assert_full_bake,
    compute_argmax_assignments,
    compute_token_support,
    load_baked_atlas,
)
from speech_decoding.atlas.fsaverage import load_fsaverage_cache, project_single_hemisphere_to_fsaverage


def test_project_single_hemisphere_to_fsaverage_identity() -> None:
    sub_pial = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    fsavg_pial = np.array(
        [[100.0, 0.0, 0.0], [110.0, 0.0, 0.0], [100.0, 10.0, 0.0], [100.0, 0.0, 10.0]],
        dtype=np.float64,
    )
    sphere = 100.0 * np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    points = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0], [0.0, 10.1, 0.0]], dtype=np.float64)
    vids, xyz = project_single_hemisphere_to_fsaverage(points, sub_pial, sphere, sphere, fsavg_pial)
    np.testing.assert_array_equal(vids, [0, 1, 2])
    np.testing.assert_allclose(xyz, fsavg_pial[[0, 1, 2]])


def test_load_fsaverage_cache_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "S14_fsaverage_pial.csv"
    path.write_text(
        "name,hemisphere,fsaverage_vertex,x,y,z\n"
        "A1,L,3,1.0,2.0,3.0\n"
        "A2,R,7,4.0,5.0,6.0\n"
    )
    cache = load_fsaverage_cache("S14", tmp_path)
    assert cache.names == ("A1", "A2")
    assert cache.hemisphere == ("L", "R")
    np.testing.assert_array_equal(cache.vertex_ids, [3, 7])
    np.testing.assert_allclose(cache.coords, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def _write_fake_frame(path: Path, values: np.ndarray) -> None:
    img = nib.freesurfer.mghformat.MGHImage(values.reshape((-1, 1, 1)).astype(np.float32), affine=np.eye(4))
    nib.save(img, str(path))


def test_load_baked_atlas_from_frame_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "fsaverage_bake"
    (out_dir / "lh").mkdir(parents=True)
    (out_dir / "rh").mkdir(parents=True)
    (out_dir / "bake_manifest.json").write_text(
        "{\n"
        '  "frames": [0, 1],\n'
        '  "hemis": ["lh", "rh"],\n'
        '  "smooth_fwhm": 3.5\n'
        "}\n"
    )
    _write_fake_frame(out_dir / "lh" / "lh.frame000.fsaverage.mgh", np.array([1.0, 2.0]))
    _write_fake_frame(out_dir / "lh" / "lh.frame001.fsaverage.mgh", np.array([3.0, 4.0]))
    _write_fake_frame(out_dir / "rh" / "rh.frame000.fsaverage.mgh", np.array([5.0, 6.0]))
    _write_fake_frame(out_dir / "rh" / "rh.frame001.fsaverage.mgh", np.array([7.0, 8.0]))
    _write_fake_frame(out_dir / "lh" / "lh.frame000.fsaverage.fwhm3.5.mgh", np.array([10.0, 20.0]))
    _write_fake_frame(out_dir / "lh" / "lh.frame001.fsaverage.fwhm3.5.mgh", np.array([30.0, 40.0]))
    _write_fake_frame(out_dir / "rh" / "rh.frame000.fsaverage.fwhm3.5.mgh", np.array([50.0, 60.0]))
    _write_fake_frame(out_dir / "rh" / "rh.frame001.fsaverage.fwhm3.5.mgh", np.array([70.0, 80.0]))

    atlas = load_baked_atlas(out_dir, kind="smoothed", tree_path=tmp_path / "missing_tree.csv")
    assert atlas.frame_ids == (0, 1)
    assert atlas.kind == "smoothed"
    assert atlas.parcel_names == ("ROI_001", "ROI_002")
    np.testing.assert_allclose(atlas.values_by_hemi["lh"], [[10.0, 30.0], [20.0, 40.0]])
    np.testing.assert_allclose(atlas.values_by_hemi["rh"], [[50.0, 70.0], [60.0, 80.0]])


def test_load_baked_atlas_subset_preserves_frame_identity(tmp_path: Path) -> None:
    out_dir = tmp_path / "fsaverage_bake"
    (out_dir / "lh").mkdir(parents=True)
    (out_dir / "rh").mkdir(parents=True)
    (out_dir / "bake_manifest.json").write_text(
        "{\n"
        '  "frames": [4, 9],\n'
        '  "hemis": ["lh", "rh"],\n'
        '  "smooth_fwhm": 3.5\n'
        "}\n"
    )
    _write_fake_frame(out_dir / "lh" / "lh.frame004.fsaverage.fwhm3.5.mgh", np.array([1.0, 2.0]))
    _write_fake_frame(out_dir / "lh" / "lh.frame009.fsaverage.fwhm3.5.mgh", np.array([3.0, 4.0]))
    _write_fake_frame(out_dir / "rh" / "rh.frame004.fsaverage.fwhm3.5.mgh", np.array([5.0, 6.0]))
    _write_fake_frame(out_dir / "rh" / "rh.frame009.fsaverage.fwhm3.5.mgh", np.array([7.0, 8.0]))

    atlas = load_baked_atlas(out_dir, kind="smoothed", tree_path=tmp_path / "missing_tree.csv")
    assert atlas.frame_ids == (4, 9)
    assert atlas.parcel_names == ("ROI_005", "ROI_010")


def test_load_baked_atlas_missing_frame_fails(tmp_path: Path) -> None:
    out_dir = tmp_path / "fsaverage_bake"
    (out_dir / "lh").mkdir(parents=True)
    (out_dir / "rh").mkdir(parents=True)
    (out_dir / "bake_manifest.json").write_text(
        "{\n"
        '  "frames": [0],\n'
        '  "hemis": ["lh", "rh"],\n'
        '  "smooth_fwhm": 3.5\n'
        "}\n"
    )
    _write_fake_frame(out_dir / "lh" / "lh.frame000.fsaverage.fwhm3.5.mgh", np.array([1.0, 2.0]))
    with pytest.raises(FileNotFoundError, match="missing baked atlas frame"):
        load_baked_atlas(out_dir, kind="smoothed", tree_path=tmp_path / "missing_tree.csv")


def test_assert_full_bake_rejects_subset(tmp_path: Path) -> None:
    out_dir = tmp_path / "fsaverage_bake"
    (out_dir / "lh").mkdir(parents=True)
    (out_dir / "rh").mkdir(parents=True)
    (out_dir / "bake_manifest.json").write_text(
        "{\n"
        '  "frames": [0, 1],\n'
        '  "hemis": ["lh", "rh"],\n'
        '  "smooth_fwhm": 3.5\n'
        "}\n"
    )
    _write_fake_frame(out_dir / "lh" / "lh.frame000.fsaverage.fwhm3.5.mgh", np.array([1.0, 2.0]))
    _write_fake_frame(out_dir / "lh" / "lh.frame001.fsaverage.fwhm3.5.mgh", np.array([3.0, 4.0]))
    _write_fake_frame(out_dir / "rh" / "rh.frame000.fsaverage.fwhm3.5.mgh", np.array([5.0, 6.0]))
    _write_fake_frame(out_dir / "rh" / "rh.frame001.fsaverage.fwhm3.5.mgh", np.array([7.0, 8.0]))
    atlas = load_baked_atlas(out_dir, kind="smoothed", tree_path=tmp_path / "missing_tree.csv")
    with pytest.raises(ValueError, match="partial"):
        assert_full_bake(atlas, expected_n_rois=246)


def test_support_summaries() -> None:
    support = np.array(
        [
            [80.0, 10.0, 0.0],
            [5.0, 60.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    argmax_idx, max_support, valid = compute_argmax_assignments(support)
    np.testing.assert_array_equal(argmax_idx, [1, 2, 0])
    np.testing.assert_allclose(max_support, [80.0, 60.0, 0.0])
    np.testing.assert_array_equal(valid, [True, True, False])
    np.testing.assert_allclose(compute_token_support(support), [0.85, 0.70, 0.0])
