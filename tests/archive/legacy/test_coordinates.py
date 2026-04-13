"""Tests for electrode coordinate loading and mapping."""
import numpy as np
import pytest
import tempfile
from pathlib import Path

from speech_decoding.data.coordinates import (
    ElectrodeCoordinates,
    load_ras_coordinates,
    load_channel_map,
    load_talairach_transform,
    build_electrode_coordinates,
    apply_talairach_transform,
    RIGHT_HEMISPHERE_PATIENTS,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def ras_128ch(tmp_path):
    """Synthetic 128-electrode RAS file (left hemisphere)."""
    path = tmp_path / "S14_RAS.txt"
    lines = []
    for elec in range(1, 129):
        row = (elec - 1) // 16
        col = (elec - 1) % 16
        # Physically realistic: ~1mm pitch, left hemisphere (negative x)
        x = -50.0 + col * 1.0
        y = -30.0 + row * 1.0
        z = 40.0
        lines.append(f"SMC {elec} {x:.6f} {y:.6f} {z:.6f} L G\n")
    path.write_text("".join(lines))
    return path


@pytest.fixture
def ras_256ch(tmp_path):
    """Synthetic 256-electrode RAS file (left hemisphere, numbered 3-288)."""
    path = tmp_path / "S33_RAS.txt"
    lines = []
    # 12x24 grid with 8 dead corners → 280 positions, but only 256 mapped
    elec_num = 1
    mapped = 0
    for row in range(12):
        for col in range(24):
            # Skip 8 dead corners
            if row == 11 and col in (0, 1, 2, 3, 20, 21, 22, 23):
                elec_num += 1
                continue
            x = -45.0 + col * 2.0
            y = -25.0 + row * 2.0
            z = 35.0
            lines.append(f"SMC {elec_num} {x:.6f} {y:.6f} {z:.6f} L G\n")
            elec_num += 1
            mapped += 1
            if mapped >= 256:
                break
        if mapped >= 256:
            break
    path.write_text("".join(lines))
    return path


@pytest.fixture
def ras_right_hemi(tmp_path):
    """Synthetic RAS file with positive x (right hemisphere)."""
    path = tmp_path / "S22_RAS.txt"
    lines = []
    for elec in range(1, 129):
        row = (elec - 1) // 16
        col = (elec - 1) % 16
        x = 46.0 + col * 1.0
        y = 25.0 + row * 1.0
        z = 38.0
        lines.append(f"SMC {elec} {x:.6f} {y:.6f} {z:.6f} R G\n")
    path.write_text("".join(lines))
    return path


@pytest.fixture
def chanmap_128(tmp_path):
    """Synthetic 8x16 chanMap with scrambled amplifier wiring."""
    import scipy.io

    path = tmp_path / "S14_channelMap.mat"
    # Scrambled: amplifier channels 1-128 assigned to random grid positions
    rng = np.random.RandomState(42)
    chanmap = rng.permutation(np.arange(1, 129)).reshape(8, 16).astype(np.uint8)
    scipy.io.savemat(str(path), {"chanMap": chanmap})
    return path, chanmap


@pytest.fixture
def xfm_file(tmp_path):
    """Synthetic talairach.xfm file."""
    path = tmp_path / "S14_talairach.xfm"
    path.write_text(
        "MNI Transform File\n"
        "% avi2talxfm\n"
        "\n"
        "Transform_Type = Linear;\n"
        "Linear_Transform = \n"
        "1.0 0.0 0.0 -1.0\n"
        "0.0 1.0 0.0 2.0\n"
        "0.0 0.0 1.0 -3.0;\n"
    )
    return path


# ── RAS Loading ───────────────────────────────────────────────────


class TestLoadRASCoordinates:
    def test_loads_128_electrodes(self, ras_128ch):
        coords = load_ras_coordinates(ras_128ch)
        assert len(coords) == 128
        assert all(isinstance(k, int) for k in coords)
        assert all(v.shape == (3,) for v in coords.values())

    def test_coordinate_values(self, ras_128ch):
        coords = load_ras_coordinates(ras_128ch)
        # Electrode 1 should be at (-50, -30, 40)
        np.testing.assert_allclose(coords[1], [-50.0, -30.0, 40.0], atol=1e-4)
        # Electrode 17 (row 1, col 0) should be at (-50, -29, 40)
        np.testing.assert_allclose(coords[17], [-50.0, -29.0, 40.0], atol=1e-4)

    def test_electrode_numbers_are_keys(self, ras_128ch):
        coords = load_ras_coordinates(ras_128ch)
        assert set(coords.keys()) == set(range(1, 129))


# ── Channel Map Loading ──────────────────────────────────────────


class TestLoadChannelMap:
    def test_loads_8x16(self, chanmap_128):
        path, expected = chanmap_128
        chanmap = load_channel_map(path)
        assert chanmap.shape == (8, 16)
        np.testing.assert_array_equal(chanmap, expected.astype(int))

    def test_missing_key_raises(self, tmp_path):
        import scipy.io

        path = tmp_path / "bad.mat"
        scipy.io.savemat(str(path), {"wrongKey": np.zeros(5)})
        with pytest.raises(KeyError, match="No 'chanMap'"):
            load_channel_map(path)


# ── Talairach Transform ──────────────────────────────────────────


class TestLoadTalairachTransform:
    def test_loads_4x4_affine(self, xfm_file):
        affine = load_talairach_transform(xfm_file)
        assert affine.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_array_equal(affine[3, :], [0, 0, 0, 1])

    def test_translation_values(self, xfm_file):
        affine = load_talairach_transform(xfm_file)
        # Our synthetic xfm has translation (-1, 2, -3)
        np.testing.assert_allclose(affine[:3, 3], [-1.0, 2.0, -3.0])

    def test_real_xfm_file(self):
        """Test with real S14 talairach.xfm if available."""
        path = Path("data/transforms/S14_talairach.xfm")
        if not path.exists():
            pytest.skip("Real xfm file not available")
        affine = load_talairach_transform(path)
        assert affine.shape == (4, 4)
        # Real S14: first row starts with ~1.036
        assert 0.9 < affine[0, 0] < 1.1


class TestApplyTransform:
    def test_identity_transform(self):
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        identity = np.eye(4)
        result = apply_talairach_transform(coords, identity)
        np.testing.assert_allclose(result, coords)

    def test_translation(self):
        coords = np.array([[0.0, 0.0, 0.0]])
        affine = np.eye(4)
        affine[:3, 3] = [10.0, 20.0, 30.0]
        result = apply_talairach_transform(coords, affine)
        np.testing.assert_allclose(result, [[10.0, 20.0, 30.0]])


# ── Build Electrode Coordinates ──────────────────────────────────


class TestBuildElectrodeCoordinates128:
    def test_with_chanmap(self, ras_128ch, chanmap_128):
        cm_path, chanmap = chanmap_128
        ec = build_electrode_coordinates(
            "S14", ras_128ch, chanmap_path=cm_path
        )
        assert ec.patient_id == "S14"
        assert ec.hemisphere == "L"
        assert ec.n_mapped == 128

    def test_correct_mapping_chain(self, ras_128ch, chanmap_128):
        """Verify: fif ch → chanMap → phys_elec → RAS gives correct coords."""
        cm_path, chanmap = chanmap_128
        ec = build_electrode_coordinates(
            "S14", ras_128ch, chanmap_path=cm_path
        )
        ras = load_ras_coordinates(ras_128ch)

        # Pick a specific amplifier channel
        amp_ch = int(chanmap[0, 0])  # whatever channel is at grid (0,0)
        phys_elec = 0 * 16 + 0 + 1  # = 1

        coord = ec.coords[str(amp_ch)]
        expected = ras[phys_elec]
        np.testing.assert_allclose(coord, expected, atol=1e-4)

    def test_smooth_distances_along_row(self, ras_128ch, chanmap_128):
        """Adjacent grid positions should have ~1mm NN distances."""
        cm_path, chanmap = chanmap_128
        ec = build_electrode_coordinates(
            "S14", ras_128ch, chanmap_path=cm_path
        )

        # Get coords for row 0 in grid order
        row0_coords = []
        for c in range(16):
            amp_ch = str(int(chanmap[0, c]))
            if amp_ch in ec.coords:
                row0_coords.append(ec.coords[amp_ch])

        dists = [
            np.linalg.norm(row0_coords[i + 1] - row0_coords[i])
            for i in range(len(row0_coords) - 1)
        ]
        # Synthetic grid has exactly 1.0mm pitch
        assert all(d < 2.0 for d in dists), f"Distances: {dists}"

    def test_filter_by_fif_names(self, ras_128ch, chanmap_128):
        cm_path, _ = chanmap_128
        ec = build_electrode_coordinates(
            "S14", ras_128ch, chanmap_path=cm_path,
            fif_ch_names=["1", "2", "3"],
        )
        assert ec.n_mapped <= 3


class TestBuildElectrodeCoordinates256:
    def test_direct_mapping(self, ras_256ch):
        ec = build_electrode_coordinates("S33", ras_256ch)
        assert ec.hemisphere == "L"
        assert ec.n_mapped > 0

    def test_zero_indexed_offset(self, tmp_path):
        """S57 uses 0-indexed .fif channels → needs +1 for RAS lookup."""
        ras_path = tmp_path / "S57_RAS.txt"
        lines = []
        for elec in range(1, 257):
            lines.append(f"SMC {elec} -{50+elec*0.1:.6f} -20.0 40.0 L G\n")
        ras_path.write_text("".join(lines))

        # S57 .fif channels are 0-255
        fif_names = [str(i) for i in range(256)]
        ec = build_electrode_coordinates(
            "S57", ras_path, fif_ch_names=fif_names
        )
        # Channel "0" should map to RAS electrode 1
        assert "0" in ec.coords
        expected_x = -(50 + 1 * 0.1)  # electrode 1
        np.testing.assert_allclose(ec.coords["0"][0], expected_x, atol=1e-4)

    def test_non_zero_indexed_patient(self, ras_256ch):
        """S33 is 1-indexed — no offset applied."""
        ec = build_electrode_coordinates("S33", ras_256ch)
        # No channel "0" should exist
        assert "0" not in ec.coords


# ── Hemisphere Mirroring ─────────────────────────────────────────


class TestHemisphereMirroring:
    def test_right_to_left(self, ras_right_hemi):
        ec = build_electrode_coordinates("S22", ras_right_hemi)
        assert ec.hemisphere == "R"

        mirrored = ec.mirror_to_left()
        assert mirrored.hemisphere == "L"

        # All x-coords should now be negative
        for coord in mirrored.coords.values():
            assert coord[0] < 0, f"Expected negative x, got {coord[0]}"

    def test_left_is_noop(self, ras_128ch, chanmap_128):
        cm_path, _ = chanmap_128
        ec = build_electrode_coordinates(
            "S14", ras_128ch, chanmap_path=cm_path
        )
        assert ec.hemisphere == "L"
        mirrored = ec.mirror_to_left()
        assert mirrored is ec  # same object, no copy

    def test_mirror_preserves_yz(self, ras_right_hemi):
        ec = build_electrode_coordinates("S22", ras_right_hemi)
        mirrored = ec.mirror_to_left()
        for ch in ec.coords:
            np.testing.assert_allclose(
                mirrored.coords[ch][1:], ec.coords[ch][1:]
            )

    def test_known_right_hemisphere_patients(self):
        assert RIGHT_HEMISPHERE_PATIENTS == {"S22", "S58"}


# ── to_array ─────────────────────────────────────────────────────


class TestToArray:
    def test_aligned_output(self, ras_128ch, chanmap_128):
        cm_path, _ = chanmap_128
        ec = build_electrode_coordinates(
            "S14", ras_128ch, chanmap_path=cm_path
        )
        ch_names = [str(i) for i in range(1, 129)]
        arr = ec.to_array(ch_names)
        assert arr.shape == (128, 3)
        assert arr.dtype == np.float32
        # No NaN since all 128 channels are mapped
        assert not np.isnan(arr).any()

    def test_missing_channels_get_nan(self, ras_128ch, chanmap_128):
        cm_path, _ = chanmap_128
        ec = build_electrode_coordinates(
            "S14", ras_128ch, chanmap_path=cm_path
        )
        # Ask for channel "999" which doesn't exist
        arr = ec.to_array(["1", "999"])
        assert arr.shape == (2, 3)
        assert not np.isnan(arr[0]).any()  # ch 1 exists
        assert np.isnan(arr[1]).all()  # ch 999 does not


# ── Real Data Tests ──────────────────────────────────────────────


@pytest.mark.slow
class TestRealData:
    """Test with actual patient data files."""

    def test_s14_128ch_with_chanmap(self):
        ras_path = Path("data/mni_coords/S14_RAS.txt")
        cm_path = Path("data/channel_maps/S14_channelMap.mat")
        if not ras_path.exists() or not cm_path.exists():
            pytest.skip("Real data not available")

        ec = build_electrode_coordinates("S14", ras_path, chanmap_path=cm_path)
        assert ec.n_mapped == 128
        assert ec.hemisphere == "L"

        # Verify smooth distances: load chanmap and check row-0
        chanmap = load_channel_map(cm_path)
        row0_coords = []
        for c in range(16):
            amp_ch = str(int(chanmap[0, c]))
            if amp_ch in ec.coords:
                row0_coords.append(ec.coords[amp_ch])

        dists = [
            np.linalg.norm(row0_coords[i + 1] - row0_coords[i])
            for i in range(len(row0_coords) - 1)
        ]
        assert all(d < 3.0 for d in dists), f"Row-0 distances: {dists}"
        assert np.mean(dists) < 2.0

    def test_s33_256ch_direct(self):
        ras_path = Path("data/mni_coords/S33_RAS.txt")
        if not ras_path.exists():
            pytest.skip("Real data not available")

        ec = build_electrode_coordinates("S33", ras_path)
        assert ec.n_mapped > 200  # Should be ~256
        assert ec.hemisphere == "L"

    def test_s22_right_hemisphere(self):
        ras_path = Path("data/mni_coords/S22_RAS.txt")
        cm_path = Path("data/channel_maps/S22_channelMap.mat")
        if not ras_path.exists() or not cm_path.exists():
            pytest.skip("Real data not available")

        ec = build_electrode_coordinates("S22", ras_path, chanmap_path=cm_path)
        assert ec.hemisphere == "R"
        mirrored = ec.mirror_to_left()
        assert mirrored.hemisphere == "L"
        # Verify all mirrored x are negative
        for coord in mirrored.coords.values():
            assert coord[0] < 0

    def test_s14_talairach_transform(self):
        ras_path = Path("data/mni_coords/S14_RAS.txt")
        xfm_path = Path("data/transforms/S14_talairach.xfm")
        cm_path = Path("data/channel_maps/S14_channelMap.mat")
        if not all(p.exists() for p in [ras_path, xfm_path, cm_path]):
            pytest.skip("Real data not available")

        ec = build_electrode_coordinates("S14", ras_path, chanmap_path=cm_path)
        affine = load_talairach_transform(xfm_path)

        ch_names = list(ec.coords.keys())[:5]
        acpc = ec.to_array(ch_names)
        mni = apply_talairach_transform(acpc, affine)

        # MNI and ACPC should be close (transform is near-identity)
        assert mni.shape == acpc.shape
        # Differences should be small (< 10mm for typical brains)
        diff = np.abs(mni - acpc)
        assert diff.max() < 15.0
