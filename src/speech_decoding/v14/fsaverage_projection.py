"""Strict fsaverage pial projection for the Phase-1 spatial prototype.

This module is intentionally separate from ``coordinates.py``.

- ``coordinates.py`` remains the Zac/cvs_avg35 oracle.
- This module implements the first-pass pure fsaverage experiment:
  patient ``pial`` -> patient ``sphere.reg`` -> ``fsaverage`` ``pial``.

The algorithm mirrors the old projection logic structurally:

1. snap each electrode to the nearest vertex on the patient's ``pial``
2. read the corresponding point on the patient's ``sphere.reg``
3. find the nearest vertex on ``fsaverage``'s ``sphere.reg``
4. read out the corresponding point on ``fsaverage``'s ``pial``

This is a strict pial-centered path. ``pial-outer-smoothed`` is excluded on
purpose so the parity experiment cleanly tests the stock FreeSurfer route.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree


DEFAULT_BOX_ROOT = Path("/Users/bentang/Library/CloudStorage/Box-Box")
DEFAULT_FS_HOME = Path("/Applications/freesurfer/8.2.0")
DEFAULT_SUBJECTS_DIR = DEFAULT_FS_HOME / "subjects"
DEFAULT_TARGET_SUBJECT = "fsaverage"
_ALGORITHM_VERSION = "strict fsaverage pial projection v1 (2026-04-16)"


@dataclass(frozen=True)
class PatientElectrodes:
    names: tuple[str, ...]
    coords: np.ndarray
    is_left: np.ndarray
    is_subdural: np.ndarray
    brainshift_method: str

    def __post_init__(self) -> None:
        n = len(self.names)
        assert self.coords.shape == (n, 3)
        assert self.is_left.shape == (n,)
        assert self.is_subdural.shape == (n,)
        assert self.coords.dtype == np.float64


@dataclass(frozen=True)
class FsaverageCoordinateCache:
    names: tuple[str, ...]
    hemisphere: tuple[str, ...]
    vertex_ids: np.ndarray  # (N,) int64, per-hemisphere fsaverage pial vertex id
    coords: np.ndarray  # (N, 3) float64, fsaverage pial xyz in mm

    def __post_init__(self) -> None:
        n = len(self.names)
        assert len(self.hemisphere) == n
        assert self.vertex_ids.shape == (n,)
        assert self.coords.shape == (n, 3)
        assert self.coords.dtype == np.float64

    @property
    def is_left(self) -> np.ndarray:
        return np.array([hem == "L" for hem in self.hemisphere], dtype=bool)

    def lookup(self, name: str) -> tuple[str, int, np.ndarray]:
        try:
            idx = self.names.index(name)
        except ValueError as err:
            raise KeyError(f"electrode name not in fsaverage cache: {name!r}") from err
        return self.hemisphere[idx], int(self.vertex_ids[idx]), self.coords[idx]


def read_lepto(patient_dir: Path, patient_id: str) -> PatientElectrodes:
    """Read ``<pt>.LEPTO`` + ``<pt>.electrodeNames`` into a lightweight struct."""

    lepto_path = patient_dir / "elec_recon" / f"{patient_id}.LEPTO"
    names_path = patient_dir / "elec_recon" / f"{patient_id}.electrodeNames"
    if not lepto_path.exists():
        raise FileNotFoundError(f"missing LEPTO file: {lepto_path}")
    if not names_path.exists():
        raise FileNotFoundError(f"missing electrodeNames file: {names_path}")

    lepto_lines = lepto_path.read_text().splitlines()
    if len(lepto_lines) < 3:
        raise ValueError(f"{lepto_path}: expected at least 3 lines, got {len(lepto_lines)}")
    header = lepto_lines[0].split("\t")
    brainshift_method = header[1] if len(header) >= 2 else "unknown"
    coord_system = lepto_lines[1].strip()
    if coord_system != "R A S":
        raise ValueError(f"{lepto_path}: expected 'R A S' on line 2, got {coord_system!r}")
    coords = np.array(
        [[float(v) for v in line.split()] for line in lepto_lines[2:]],
        dtype=np.float64,
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{lepto_path}: coord rows must have exactly 3 floats")

    name_lines = names_path.read_text().splitlines()
    if len(name_lines) < 3:
        raise ValueError(f"{names_path}: expected at least 3 lines, got {len(name_lines)}")
    names: list[str] = []
    is_left: list[bool] = []
    is_subdural: list[bool] = []
    for row_idx, line in enumerate(name_lines[2:], start=3):
        tokens = line.split()
        if len(tokens) != 3:
            raise ValueError(
                f"{names_path}:{row_idx}: expected 3 tokens (name type hem), got {tokens!r}"
            )
        name, kind, hem = tokens
        if kind.upper() not in {"G", "S", "D"}:
            raise ValueError(
                f"{names_path}:{row_idx}: expected type in {{G,S,D}}, got {kind!r}"
            )
        if hem.upper() not in {"L", "R"}:
            raise ValueError(
                f"{names_path}:{row_idx}: expected hem in {{L,R}}, got {hem!r}"
            )
        names.append(name)
        is_subdural.append(kind.upper() != "D")
        is_left.append(hem.upper() == "L")

    if len(names) != coords.shape[0]:
        raise ValueError(
            f"row count mismatch: {lepto_path.name} has {coords.shape[0]}, "
            f"{names_path.name} has {len(names)}"
        )

    return PatientElectrodes(
        names=tuple(names),
        coords=coords,
        is_left=np.array(is_left, dtype=bool),
        is_subdural=np.array(is_subdural, dtype=bool),
        brainshift_method=brainshift_method,
    )


def _load_surface(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing surface file: {path}")
    verts, _faces = nib.freesurfer.io.read_geometry(str(path))
    return np.asarray(verts, dtype=np.float64)


def _assert_hemisphere_sign_consistent(elec: PatientElectrodes) -> None:
    xs = elec.coords[:, 0]
    tol = 5.0
    left_bad = [
        name
        for name, left, x in zip(elec.names, elec.is_left, xs)
        if left and x > tol
    ]
    right_bad = [
        name
        for name, left, x in zip(elec.names, elec.is_left, xs)
        if (not left) and x < -tol
    ]
    if left_bad or right_bad:
        raise ValueError(
            f"hemisphere label / x-sign mismatch (> {tol} mm from midline): "
            f"left-labeled with x>tol: {left_bad}; "
            f"right-labeled with x<-tol: {right_bad}"
        )


def project_single_hemisphere_to_fsaverage(
    points: np.ndarray,
    sub_pial_verts: np.ndarray,
    sub_sph_verts: np.ndarray,
    fsavg_sph_verts: np.ndarray,
    fsavg_pial_verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one hemisphere and return `(vertex_ids, coords)` on fsaverage."""

    if sub_pial_verts.shape[0] != sub_sph_verts.shape[0]:
        raise ValueError(
            f"subject pial/sphere vertex counts disagree: "
            f"{sub_pial_verts.shape[0]} vs {sub_sph_verts.shape[0]}"
        )
    if fsavg_sph_verts.shape[0] != fsavg_pial_verts.shape[0]:
        raise ValueError(
            f"fsaverage pial/sphere vertex counts disagree: "
            f"{fsavg_sph_verts.shape[0]} vs {fsavg_pial_verts.shape[0]}"
        )
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 3), dtype=np.float64)

    sub_tree = cKDTree(sub_pial_verts)
    _, sub_vids = sub_tree.query(points, k=1)

    fsavg_tree = cKDTree(fsavg_sph_verts)
    _, fsavg_vids = fsavg_tree.query(sub_sph_verts[sub_vids], k=1)
    fsavg_vids = np.asarray(fsavg_vids, dtype=np.int64)
    return fsavg_vids, fsavg_pial_verts[fsavg_vids].astype(np.float64, copy=True)


def project_to_fsaverage(
    sub_coords: np.ndarray,
    is_left: np.ndarray,
    sub_surf_dir: Path,
    subjects_dir: Path,
    target_subject: str = DEFAULT_TARGET_SUBJECT,
) -> tuple[np.ndarray, np.ndarray]:
    """Project subject electrodes to fsaverage pial via stock `sphere.reg`."""

    n = sub_coords.shape[0]
    if is_left.shape != (n,):
        raise ValueError(f"is_left shape {is_left.shape} != ({n},)")

    out_vids = np.full((n,), -1, dtype=np.int64)
    out_xyz = np.full((n, 3), np.nan, dtype=np.float64)
    fsavg_surf_dir = subjects_dir / target_subject / "surf"

    for hem, mask in (("lh", is_left), ("rh", ~is_left)):
        if not mask.any():
            continue
        sub_pial = _load_surface(sub_surf_dir / f"{hem}.pial")
        sub_sph = _load_surface(sub_surf_dir / f"{hem}.sphere.reg")
        fsavg_sph = _load_surface(fsavg_surf_dir / f"{hem}.sphere.reg")
        fsavg_pial = _load_surface(fsavg_surf_dir / f"{hem}.pial")
        vids, xyz = project_single_hemisphere_to_fsaverage(
            sub_coords[mask],
            sub_pial,
            sub_sph,
            fsavg_sph,
            fsavg_pial,
        )
        out_vids[mask] = vids
        out_xyz[mask] = xyz

    if (out_vids < 0).any() or np.isnan(out_xyz).any():
        raise RuntimeError("fsaverage projection left unset rows")
    return out_vids, out_xyz


def project_patient_to_fsaverage(
    patient_id: str,
    box_root: Path,
    cache_dir: Path,
    subjects_dir: Path = DEFAULT_SUBJECTS_DIR,
    target_subject: str = DEFAULT_TARGET_SUBJECT,
) -> Path:
    """Project one patient to fsaverage and write a cache CSV + sidecar."""

    patient_dir = box_root / "ECoG_Recon" / patient_id
    elec_recon = patient_dir / "elec_recon"
    sub_surf_dir = patient_dir / "surf"

    elec = read_lepto(patient_dir, patient_id)
    if not elec.is_subdural.all():
        depths = [n for n, s in zip(elec.names, elec.is_subdural) if not s]
        raise ValueError(
            f"{patient_id}: depth electrodes found, not supported in Phase 1: {depths}"
        )
    _assert_hemisphere_sign_consistent(elec)

    vertex_ids, xyz = project_to_fsaverage(
        sub_coords=elec.coords,
        is_left=elec.is_left,
        sub_surf_dir=sub_surf_dir,
        subjects_dir=subjects_dir,
        target_subject=target_subject,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{patient_id}_fsaverage_pial.csv"
    with out_path.open("w") as f:
        f.write("name,hemisphere,fsaverage_vertex,x,y,z\n")
        for name, is_left, vid, coord in zip(elec.names, elec.is_left, vertex_ids, xyz):
            hem = "L" if is_left else "R"
            f.write(
                f"{name},{hem},{int(vid)},{coord[0]:.6f},{coord[1]:.6f},{coord[2]:.6f}\n"
            )

    _update_sidecar(
        cache_dir=cache_dir,
        patient_id=patient_id,
        elec=elec,
        lepto_path=elec_recon / f"{patient_id}.LEPTO",
        names_path=elec_recon / f"{patient_id}.electrodeNames",
        subjects_dir=subjects_dir,
        target_subject=target_subject,
    )
    return out_path


def _update_sidecar(
    *,
    cache_dir: Path,
    patient_id: str,
    elec: PatientElectrodes,
    lepto_path: Path,
    names_path: Path,
    subjects_dir: Path,
    target_subject: str,
) -> None:
    meta_path = cache_dir / "_projection_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta[patient_id] = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm_version": _ALGORITHM_VERSION,
        "source_snap_surface": "pial",
        "target_subject": target_subject,
        "subjects_dir": str(subjects_dir),
        "brainshift_method": elec.brainshift_method,
        "n_electrodes": len(elec.names),
        "n_left": int(elec.is_left.sum()),
        "n_right": int((~elec.is_left).sum()),
        "lepto_mtime_utc": datetime.fromtimestamp(
            lepto_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "electrodenames_mtime_utc": datetime.fromtimestamp(
            names_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def load_fsaverage_cache(patient_id: str, cache_dir: Path) -> FsaverageCoordinateCache:
    path = cache_dir / f"{patient_id}_fsaverage_pial.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing fsaverage cache: {path}")

    names: list[str] = []
    hemis: list[str] = []
    vertex_ids: list[int] = []
    coords: list[list[float]] = []
    with path.open() as f:
        header = f.readline().rstrip("\n")
        if header != "name,hemisphere,fsaverage_vertex,x,y,z":
            raise ValueError(f"{path}: unexpected header {header!r}")
        for row_idx, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 6:
                raise ValueError(f"{path}:{row_idx}: expected 6 columns, got {fields!r}")
            name, hem, vid, x, y, z = fields
            if hem not in {"L", "R"}:
                raise ValueError(f"{path}:{row_idx}: hemisphere must be L or R, got {hem!r}")
            names.append(name)
            hemis.append(hem)
            vertex_ids.append(int(vid))
            coords.append([float(x), float(y), float(z)])

    return FsaverageCoordinateCache(
        names=tuple(names),
        hemisphere=tuple(hemis),
        vertex_ids=np.array(vertex_ids, dtype=np.int64),
        coords=np.array(coords, dtype=np.float64),
    )
