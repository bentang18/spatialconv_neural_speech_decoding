"""Electrode coordinate loading and mapping for cross-patient models.

Maps .fif amplifier channel numbers → physical electrode positions → ACPC
coordinates. For 128-ch patients, chanMap is required (8.55mm mean error
without it). For 256-ch patients, direct mapping works (~85-100% overlap).

ACPC coordinates are per-patient, AC-PC aligned — a common reference frame
sufficient for cross-patient modeling. talairach.xfm available for MNI
conversion if needed (e.g., atlas lookups).

Coordinate chain (128-ch):
    .fif channel N → chanMap[r,c]==N → physical_elec = r*n_cols+c+1 → RAS(x,y,z)

Coordinate chain (256-ch):
    .fif channel N → RAS electrode N directly (with S57 +1 offset)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io


# Patients with 0-indexed .fif channel names (all others are 1-indexed)
_ZERO_INDEXED_PATIENTS = {"S57", "S58"}

# Right-hemisphere patients (positive x; all others are left/negative x)
RIGHT_HEMISPHERE_PATIENTS = {"S22", "S58"}


@dataclass
class ElectrodeCoordinates:
    """ACPC coordinates for a patient's electrodes, keyed by .fif channel name."""

    patient_id: str
    # fif_channel_name (str) → (x, y, z) in ACPC space
    coords: dict[str, np.ndarray]
    hemisphere: str  # "L" or "R"

    @property
    def n_mapped(self) -> int:
        return len(self.coords)

    def to_array(self, ch_names: list[str]) -> np.ndarray:
        """Return (n_channels, 3) coordinate array aligned to ch_names.

        Channels without coordinates get NaN. Caller should filter these out.
        """
        out = np.full((len(ch_names), 3), np.nan, dtype=np.float32)
        for i, name in enumerate(ch_names):
            if str(name) in self.coords:
                out[i] = self.coords[str(name)]
        return out

    def mirror_to_left(self) -> "ElectrodeCoordinates":
        """Return a copy with x-coordinates negated (right → left hemisphere)."""
        if self.hemisphere == "L":
            return self
        mirrored = {}
        for ch, coord in self.coords.items():
            mirrored[ch] = coord.copy()
            mirrored[ch][0] = -mirrored[ch][0]
        return ElectrodeCoordinates(
            patient_id=self.patient_id,
            coords=mirrored,
            hemisphere="L",
        )


def load_ras_coordinates(ras_path: str | Path) -> dict[int, np.ndarray]:
    """Load ACPC electrode coordinates from RAS brainshifted file.

    Format: prefix electrode_num x y z hemisphere type
    Example: SMC 16 -49.753086 -27.118988 44.393181 L G

    Returns:
        Dict mapping physical electrode number → (x, y, z) array.
    """
    coords = {}
    with open(ras_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            elec_num = int(parts[1])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            coords[elec_num] = np.array([x, y, z], dtype=np.float64)
    return coords


def load_channel_map(mat_path: str | Path) -> np.ndarray:
    """Load chanMap from .mat file.

    Returns:
        (n_rows, n_cols) int array where chanmap[r, c] = amplifier channel number.
    """
    mat = scipy.io.loadmat(str(mat_path))
    if "chanMap" in mat:
        return mat["chanMap"].astype(int)
    raise KeyError(f"No 'chanMap' key in {mat_path}. Keys: {list(mat.keys())}")


def load_talairach_transform(xfm_path: str | Path) -> np.ndarray:
    """Load FreeSurfer talairach.xfm as 4×4 affine matrix.

    Returns:
        (4, 4) float64 affine matrix for ACPC → Talairach/MNI.
    """
    rows = []
    reading = False
    with open(xfm_path) as f:
        for line in f:
            if "Linear_Transform" in line:
                reading = True
                continue
            if reading:
                # Remove trailing semicolon and parse
                cleaned = line.strip().rstrip(";").strip()
                if cleaned:
                    rows.append([float(v) for v in cleaned.split()])
                if len(rows) == 3:
                    break

    if len(rows) != 3:
        raise ValueError(f"Could not parse 3×4 matrix from {xfm_path}")

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.array(rows)
    return affine


def build_electrode_coordinates(
    patient_id: str,
    ras_path: str | Path,
    chanmap_path: str | Path | None = None,
    fif_ch_names: list[str] | None = None,
) -> ElectrodeCoordinates:
    """Build the complete .fif channel → ACPC coordinate mapping.

    For 128-ch patients (chanmap provided):
        fif ch N → chanMap position (r,c) → physical elec r*ncols+c+1 → RAS

    For 256-ch patients (no chanmap):
        fif ch N → RAS electrode N directly (with +1 for 0-indexed patients)

    Args:
        patient_id: e.g., "S14"
        ras_path: Path to *_RAS.txt file with ACPC coordinates.
        chanmap_path: Path to *_channelMap.mat. Required for 128-ch patients.
            If None, uses direct mapping (suitable for 256-ch).
        fif_ch_names: .fif channel names. If provided, only maps these channels.
            If None, maps all available channels.
    """
    ras_coords = load_ras_coordinates(ras_path)
    hemisphere = _detect_hemisphere(ras_coords)
    zero_indexed = patient_id in _ZERO_INDEXED_PATIENTS

    coords: dict[str, np.ndarray] = {}

    if chanmap_path is not None:
        # 128-ch path: use chanMap to bridge amplifier → physical electrode
        chanmap = load_channel_map(chanmap_path)
        n_rows, n_cols = chanmap.shape

        # Build amplifier ch → physical electrode mapping
        amp_to_phys: dict[int, int] = {}
        for r in range(n_rows):
            for c in range(n_cols):
                amp_ch = int(chanmap[r, c])
                phys_elec = r * n_cols + c + 1
                amp_to_phys[amp_ch] = phys_elec

        # Map each .fif channel
        channels = fif_ch_names if fif_ch_names else [str(a) for a in amp_to_phys]
        for ch_name in channels:
            amp_ch = int(ch_name)
            phys_elec = amp_to_phys.get(amp_ch)
            if phys_elec is not None and phys_elec in ras_coords:
                coords[str(ch_name)] = ras_coords[phys_elec].astype(np.float32)
    else:
        # 256-ch path: direct mapping with optional +1 offset
        offset = 1 if zero_indexed else 0
        channels = fif_ch_names if fif_ch_names else [str(e) for e in ras_coords]
        for ch_name in channels:
            fif_num = int(ch_name)
            ras_num = fif_num + offset
            if ras_num in ras_coords:
                coords[str(ch_name)] = ras_coords[ras_num].astype(np.float32)

    return ElectrodeCoordinates(
        patient_id=patient_id,
        coords=coords,
        hemisphere=hemisphere,
    )


def _detect_hemisphere(ras_coords: dict[int, np.ndarray]) -> str:
    """Detect hemisphere from median x-coordinate."""
    xs = [c[0] for c in ras_coords.values()]
    return "R" if np.median(xs) > 0 else "L"


def apply_talairach_transform(
    coords: np.ndarray, affine: np.ndarray
) -> np.ndarray:
    """Apply 4×4 affine to transform ACPC → Talairach/MNI.

    Args:
        coords: (N, 3) ACPC coordinates.
        affine: (4, 4) affine matrix from load_talairach_transform().

    Returns:
        (N, 3) transformed coordinates.
    """
    N = coords.shape[0]
    homogeneous = np.column_stack([coords, np.ones(N)])  # (N, 4)
    transformed = (affine @ homogeneous.T).T  # (N, 4)
    return transformed[:, :3]
