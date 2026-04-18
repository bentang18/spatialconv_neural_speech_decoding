"""Amplifier-to-physical-electrode bridge for v14-core (blocker #12).

Single source of truth for the channel-map layer of the loader. Used by the
Phase-A2 verifier, the B1 `V14TrialDataset`, and the E1 CLI.

Bridge contracts (per `docs/implementation_tasks.md` #12):

- **128-strip (Map 4)**: `data/channel_maps/<pt>_channelMap.mat` contains an
  8×16 int array `chanMap`. `amp → (r, c)` where `chanMap[r, c] == amp`;
  `phys_num = r * 16 + c + 1`. Grid shape `(8, 16)`, no padding.
- **256-grid (Map 3)**: `data/channel_maps/<pt>_channelMapAll.mat` contains a
  lab-wide 46×24 float array `chanMapAll` with NaN padding. The patient's 256
  physical electrodes sit inside a 12×22 bounding box. For 256-ch the amp
  number equals the physical number, so `(r, c)` comes directly from
  `np.argwhere(bbox == amp_num) - bbox_origin`.

`phys_name = <pt>.electrodeNames[phys_num - 1]`; that name is the stable key
into the fsaverage coord cache and the Tier-1 support cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.io


# Per-patient grid bounding rectangle (H_p, W_p). See docs/plans/v14-core.md N7.
# N7's (16, 16) entry for 256-ch patients was wrong — the bbox of the 1..256
# sub-region of chanMapAll is 12×22 (row 0 has 14 valid cells, rows 1..11 have
# 22 each, 8 padding cells total).
GRID_SHAPES: dict[str, tuple[int, int]] = {
    # 128-ch strips (Map 4)
    "S14": (8, 16),
    "S16": (8, 16),
    "S22": (8, 16),  # Phase 2 (RH)
    "S23": (8, 16),
    "S26": (8, 16),
    # 256-ch grids (Map 3)
    "S33": (12, 22),
    "S39": (12, 22),
    "S62": (12, 22),
    # S58 is a 12×24 row-slice of Map 3, deferred to Phase 2.
    "S58": (12, 24),
}


ChannelMapKind = Literal["map4_128", "map3_256"]

PATIENT_KIND: dict[str, ChannelMapKind] = {
    "S14": "map4_128",
    "S16": "map4_128",
    "S22": "map4_128",
    "S23": "map4_128",
    "S26": "map4_128",
    "S33": "map3_256",
    "S39": "map3_256",
    "S62": "map3_256",
}


@dataclass(frozen=True)
class PhysicalResolution:
    amp_name: str      # original channel label from `.fif.info.ch_names`
    amp_num: int       # parsed amplifier number
    phys_num: int      # 1-indexed physical electrode number
    phys_name: str     # from `<pt>.electrodeNames` at row (phys_num - 1)
    row: int           # 0-indexed grid row within per-patient bbox
    col: int           # 0-indexed grid col within per-patient bbox


def load_chanmap_8x16(path: Path) -> np.ndarray:
    """Load a 128-strip `chanMap` (Map 4). Assert shape (8, 16) and permutation of 1..128."""

    mat = scipy.io.loadmat(str(path), squeeze_me=True)
    if "chanMap" not in mat:
        raise KeyError(f"{path}: expected 'chanMap' key, got {list(mat.keys())}")
    arr = np.asarray(mat["chanMap"]).astype(int)
    if arr.shape != (8, 16):
        raise ValueError(f"{path}: chanMap shape {arr.shape} != (8, 16)")
    if sorted(int(v) for v in arr.ravel()) != list(range(1, 129)):
        raise ValueError(f"{path}: chanMap values must be a permutation of 1..128")
    return arr


def load_chanmap_all_256(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load `chanMapAll` (Map 3) and slice to the 1..256 bounding box.

    Returns `(bbox, (row_origin, col_origin))` where `bbox` is the `(H_p, W_p)`
    int array with cells outside [1, 256] set to -1 (padding), and
    `(row_origin, col_origin)` is the top-left of the bbox in the full
    chanMapAll frame.
    """

    mat = scipy.io.loadmat(str(path), squeeze_me=True)
    if "chanMapAll" not in mat:
        raise KeyError(f"{path}: expected 'chanMapAll' key, got {list(mat.keys())}")
    raw = np.asarray(mat["chanMapAll"], dtype=np.float64)
    arr = np.where(np.isnan(raw), -1, raw).astype(np.int64)
    in_range = (arr >= 1) & (arr <= 256)
    count = int(in_range.sum())
    if count != 256:
        raise ValueError(f"{path}: expected 256 values in [1, 256], got {count}")
    rows = np.where(in_range.any(axis=1))[0]
    cols = np.where(in_range.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    bbox = arr[r0:r1 + 1, c0:c1 + 1].copy()
    bbox[(bbox < 1) | (bbox > 256)] = -1
    return bbox, (r0, c0)


def load_electrode_names(patient_dir: Path, patient_id: str) -> tuple[str, ...]:
    """Read physical-electrode names from `<pt>.electrodeNames` (rows 3+)."""

    path = patient_dir / "elec_recon" / f"{patient_id}.electrodeNames"
    if not path.exists():
        raise FileNotFoundError(f"missing electrodeNames file: {path}")
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path}: expected at least 3 lines, got {len(lines)}")
    names: list[str] = []
    for row_idx, line in enumerate(lines[2:], start=3):
        tokens = line.split()
        if len(tokens) != 3:
            raise ValueError(
                f"{path}:{row_idx}: expected 3 tokens (name type hem), got {tokens!r}"
            )
        names.append(tokens[0])
    return tuple(names)


def resolve_physical_names_for_patient(
    patient_id: str,
    kept_amp_names: tuple[str, ...],
    *,
    channel_maps_dir: Path,
    box_root: Path,
) -> list[PhysicalResolution]:
    """Apply the #12 bridge for each amp channel label in order.

    `kept_amp_names` is an ordered list of `.fif.info.ch_names` after `bads`
    are filtered. Each label is parsed as an integer amplifier number. Raises
    if any channel fails to resolve.
    """

    kind = PATIENT_KIND.get(patient_id)
    if kind is None:
        raise KeyError(f"no channel-map kind registered for {patient_id!r}")
    patient_dir = box_root / "ECoG_Recon" / patient_id
    electrode_names = load_electrode_names(patient_dir, patient_id)

    if kind == "map4_128":
        chanmap = load_chanmap_8x16(channel_maps_dir / f"{patient_id}_channelMap.mat")
        n_cols = chanmap.shape[1]
        amp_to_rc: dict[int, tuple[int, int]] = {
            int(chanmap[r, c]): (r, c)
            for r in range(chanmap.shape[0])
            for c in range(n_cols)
        }
        out: list[PhysicalResolution] = []
        for amp_name in kept_amp_names:
            amp_num = int(amp_name)
            if amp_num not in amp_to_rc:
                raise KeyError(f"{patient_id}: amp {amp_num} not in chanMap")
            r, c = amp_to_rc[amp_num]
            phys_num = r * n_cols + c + 1
            if not (1 <= phys_num <= len(electrode_names)):
                raise IndexError(
                    f"{patient_id}: phys_num {phys_num} outside electrodeNames length "
                    f"{len(electrode_names)}"
                )
            out.append(
                PhysicalResolution(
                    amp_name=amp_name,
                    amp_num=amp_num,
                    phys_num=phys_num,
                    phys_name=electrode_names[phys_num - 1],
                    row=r,
                    col=c,
                )
            )
        return out

    # kind == "map3_256"
    bbox, _origin = load_chanmap_all_256(
        channel_maps_dir / f"{patient_id}_channelMapAll.mat"
    )
    phys_to_rc: dict[int, tuple[int, int]] = {}
    for r in range(bbox.shape[0]):
        for c in range(bbox.shape[1]):
            v = int(bbox[r, c])
            if 1 <= v <= 256:
                phys_to_rc[v] = (r, c)
    out = []
    for amp_name in kept_amp_names:
        amp_num = int(amp_name)
        if not (1 <= amp_num <= 256):
            raise ValueError(f"{patient_id}: amp {amp_num} outside [1, 256]")
        rc = phys_to_rc.get(amp_num)
        if rc is None:
            raise KeyError(f"{patient_id}: amp {amp_num} missing from Map 3 bbox")
        r, c = rc
        phys_num = amp_num  # identity for 256-ch arrays
        if not (1 <= phys_num <= len(electrode_names)):
            raise IndexError(
                f"{patient_id}: phys_num {phys_num} outside electrodeNames length "
                f"{len(electrode_names)}"
            )
        out.append(
            PhysicalResolution(
                amp_name=amp_name,
                amp_num=amp_num,
                phys_num=phys_num,
                phys_name=electrode_names[phys_num - 1],
                row=r,
                col=c,
            )
        )
    return out


def expected_grid_shape_for_patient(patient_id: str) -> tuple[int, int]:
    if patient_id not in GRID_SHAPES:
        raise KeyError(f"no grid shape registered for {patient_id!r}")
    return GRID_SHAPES[patient_id]
