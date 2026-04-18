"""Unit + integration tests for the #12 coordinate-bridge verifier.

Unit tests drive the bridge with synthetic chanMap arrays and a fake
`.electrodeNames` layout — no Box mount, no .fif, no cache files. Integration
tests run against the real S14 `.fif` and the real S33/S62 chanMapAll
artifacts, skipping when those inputs aren't available locally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from speech_decoding.v14.channel_map import (
    GRID_SHAPES,
    PATIENT_KIND,
    PhysicalResolution,
    expected_grid_shape_for_patient,
    load_chanmap_8x16,
    load_chanmap_all_256,
    load_electrode_names,
    resolve_physical_names_for_patient,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHANNEL_MAPS_DIR = PROJECT_ROOT / "data" / "channel_maps"
BOX_ROOT = Path("/Users/bentang/Library/CloudStorage/Box-Box")


# -----------------------------------------------------------------------------
# Registry sanity


def test_grid_shapes_cover_core_patients() -> None:
    for pt in ("S14", "S26", "S33", "S62"):
        assert pt in GRID_SHAPES
        assert pt in PATIENT_KIND
    assert GRID_SHAPES["S14"] == (8, 16)
    assert GRID_SHAPES["S26"] == (8, 16)
    assert GRID_SHAPES["S33"] == (12, 22)
    assert GRID_SHAPES["S62"] == (12, 22)


def test_expected_grid_shape_raises_on_unknown() -> None:
    with pytest.raises(KeyError):
        expected_grid_shape_for_patient("S999")


# -----------------------------------------------------------------------------
# Synthetic chanMap (8×16) parsing + bridge


def _write_fake_electrode_names(patient_dir: Path, patient_id: str, n: int) -> None:
    elec_recon = patient_dir / "elec_recon"
    elec_recon.mkdir(parents=True, exist_ok=True)
    names = [f"EL{i}" for i in range(1, n + 1)]
    lines = [f"Subject {patient_id}", "# header"]
    for name in names:
        lines.append(f"{name} G L")
    (elec_recon / f"{patient_id}.electrodeNames").write_text("\n".join(lines) + "\n")


def test_load_chanmap_8x16_round_trip(tmp_path: Path) -> None:
    arr = np.arange(1, 129, dtype=np.uint8).reshape(8, 16)
    path = tmp_path / "X_channelMap.mat"
    scipy.io.savemat(str(path), {"chanMap": arr})
    out = load_chanmap_8x16(path)
    assert out.shape == (8, 16)
    np.testing.assert_array_equal(out, arr.astype(int))


def test_load_chanmap_8x16_rejects_wrong_shape(tmp_path: Path) -> None:
    bad = np.arange(1, 65).reshape(8, 8)
    path = tmp_path / "bad.mat"
    scipy.io.savemat(str(path), {"chanMap": bad})
    with pytest.raises(ValueError, match="shape"):
        load_chanmap_8x16(path)


def test_load_chanmap_8x16_rejects_non_permutation(tmp_path: Path) -> None:
    dup = np.ones((8, 16), dtype=np.uint8)
    path = tmp_path / "dup.mat"
    scipy.io.savemat(str(path), {"chanMap": dup})
    with pytest.raises(ValueError, match="permutation"):
        load_chanmap_8x16(path)


def test_resolve_map4_128_synthetic(tmp_path: Path, monkeypatch) -> None:
    patient_id = "SXX"
    # Register fake patient in both tables for the call
    monkeypatch.setitem(PATIENT_KIND, patient_id, "map4_128")
    monkeypatch.setitem(GRID_SHAPES, patient_id, (8, 16))

    # chanMap: identity-ish permutation. Use reverse to prove the lookup works.
    arr = np.arange(1, 129, dtype=np.uint8).reshape(8, 16)
    channel_maps_dir = tmp_path / "cmaps"
    channel_maps_dir.mkdir()
    scipy.io.savemat(
        str(channel_maps_dir / f"{patient_id}_channelMap.mat"), {"chanMap": arr}
    )

    box_root = tmp_path / "box"
    patient_dir = box_root / "ECoG_Recon" / patient_id
    _write_fake_electrode_names(patient_dir, patient_id, n=128)

    kept = ("1", "17", "128")
    out = resolve_physical_names_for_patient(
        patient_id, kept,
        channel_maps_dir=channel_maps_dir, box_root=box_root,
    )
    assert [r.phys_num for r in out] == [1, 17, 128]
    assert [r.phys_name for r in out] == ["EL1", "EL17", "EL128"]
    assert out[0].row == 0 and out[0].col == 0
    assert out[1].row == 1 and out[1].col == 0  # 17 = 1*16 + 0 + 1
    assert out[2].row == 7 and out[2].col == 15


def test_resolve_map4_raises_on_bad_amp(tmp_path: Path, monkeypatch) -> None:
    patient_id = "SXX"
    monkeypatch.setitem(PATIENT_KIND, patient_id, "map4_128")
    arr = np.arange(1, 129, dtype=np.uint8).reshape(8, 16)
    channel_maps_dir = tmp_path / "cmaps"
    channel_maps_dir.mkdir()
    scipy.io.savemat(
        str(channel_maps_dir / f"{patient_id}_channelMap.mat"), {"chanMap": arr}
    )
    box_root = tmp_path / "box"
    _write_fake_electrode_names(box_root / "ECoG_Recon" / patient_id, patient_id, n=128)
    with pytest.raises(KeyError, match="amp 999"):
        resolve_physical_names_for_patient(
            patient_id, ("999",),
            channel_maps_dir=channel_maps_dir, box_root=box_root,
        )


# -----------------------------------------------------------------------------
# 256-ch chanMapAll (real file, synthetic electrodeNames)


def test_load_chanmap_all_256_s33_bbox() -> None:
    path = CHANNEL_MAPS_DIR / "S33_channelMapAll.mat"
    if not path.exists():
        pytest.skip(f"{path} not present")
    bbox, origin = load_chanmap_all_256(path)
    assert bbox.shape == (12, 22)
    assert int((bbox >= 1).sum() + (bbox <= 256).sum() - bbox.size) == 256 or \
        int(((bbox >= 1) & (bbox <= 256)).sum()) == 256
    assert origin == (0, 1)  # row 0 valid, col origin shifted by 1 per inspection


def test_load_chanmap_all_256_s62_bbox() -> None:
    path = CHANNEL_MAPS_DIR / "S62_channelMapAll.mat"
    if not path.exists():
        pytest.skip(f"{path} not present")
    bbox, _origin = load_chanmap_all_256(path)
    assert bbox.shape == (12, 22)
    assert int(((bbox >= 1) & (bbox <= 256)).sum()) == 256
    # Padding cells (row 0 corners) are -1
    assert (bbox == -1).sum() == 8


def test_resolve_map3_256_from_real_s33(tmp_path: Path, monkeypatch) -> None:
    """Use real S33 chanMapAll with a synthetic electrodeNames file."""

    path = CHANNEL_MAPS_DIR / "S33_channelMapAll.mat"
    if not path.exists():
        pytest.skip(f"{path} not present")

    box_root = tmp_path / "box"
    _write_fake_electrode_names(box_root / "ECoG_Recon" / "S33", "S33", n=256)

    kept = ("1", "128", "256")
    out = resolve_physical_names_for_patient(
        "S33", kept,
        channel_maps_dir=CHANNEL_MAPS_DIR,
        box_root=box_root,
    )
    assert len(out) == 3
    assert [r.phys_num for r in out] == [1, 128, 256]
    assert [r.phys_name for r in out] == ["EL1", "EL128", "EL256"]
    H_p, W_p = GRID_SHAPES["S33"]
    for r in out:
        assert 0 <= r.row < H_p
        assert 0 <= r.col < W_p


def test_resolve_map3_rejects_amp_outside_range(tmp_path: Path) -> None:
    path = CHANNEL_MAPS_DIR / "S33_channelMapAll.mat"
    if not path.exists():
        pytest.skip(f"{path} not present")
    box_root = tmp_path / "box"
    _write_fake_electrode_names(box_root / "ECoG_Recon" / "S33", "S33", n=256)
    with pytest.raises(ValueError, match="outside"):
        resolve_physical_names_for_patient(
            "S33", ("500",),
            channel_maps_dir=CHANNEL_MAPS_DIR,
            box_root=box_root,
        )


# -----------------------------------------------------------------------------
# electrodeNames parsing


def test_load_electrode_names_parses_rows(tmp_path: Path) -> None:
    pt_dir = tmp_path / "S99"
    elec_recon = pt_dir / "elec_recon"
    elec_recon.mkdir(parents=True)
    (elec_recon / "S99.electrodeNames").write_text(
        "hdr line 1\nhdr line 2\nA1 G L\nA2 G L\nA3 D R\n"
    )
    names = load_electrode_names(pt_dir, "S99")
    assert names == ("A1", "A2", "A3")


def test_load_electrode_names_rejects_bad_row(tmp_path: Path) -> None:
    pt_dir = tmp_path / "S99"
    elec_recon = pt_dir / "elec_recon"
    elec_recon.mkdir(parents=True)
    (elec_recon / "S99.electrodeNames").write_text("hdr1\nhdr2\nA1 G\n")
    with pytest.raises(ValueError, match="3 tokens"):
        load_electrode_names(pt_dir, "S99")


# -----------------------------------------------------------------------------
# End-to-end bridge on real S14 data


@pytest.mark.slow
def test_resolve_s14_end_to_end_against_real_data() -> None:
    """Real S14 chanMap + real Box electrodeNames + full kept-channel bridge."""

    chanmap_path = CHANNEL_MAPS_DIR / "S14_channelMap.mat"
    box_elec_names = BOX_ROOT / "ECoG_Recon" / "S14" / "elec_recon" / "S14.electrodeNames"
    if not chanmap_path.exists() or not box_elec_names.exists():
        pytest.skip("S14 real artifacts not present locally")
    kept = tuple(str(i) for i in range(1, 129))
    out = resolve_physical_names_for_patient(
        "S14", kept, channel_maps_dir=CHANNEL_MAPS_DIR, box_root=BOX_ROOT,
    )
    assert len(out) == 128
    assert all(isinstance(r, PhysicalResolution) for r in out)
    # every phys_num hit exactly once
    assert sorted(r.phys_num for r in out) == list(range(1, 129))
    # every (r, c) inside (8, 16)
    for r in out:
        assert 0 <= r.row < 8
        assert 0 <= r.col < 16
