"""cvs_avg35 outer->pial bridge for the atlas-modernized Phase-1 prototype.

This module keeps Zac's patient-side projection logic untouched:

    patient native -> patient pial-outer-smoothed
    -> patient sphere-outer-mni.reg
    -> cvs_avg35_inMNI152 sphere-outer.reg
    -> cvs_avg35_inMNI152 pial-outer-smoothed

The only new step here is a deterministic template-local bridge from the
average-brain outer envelope to the average-brain pial surface. Phase 1 uses
the simplest accepted baseline: nearest-pial lookup on the same target brain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from .fsaverage_atlas import load_mni_cache


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOX_ROOT = Path("/Users/bentang/Library/CloudStorage/Box-Box")
DEFAULT_SUBJECTS_DIR = PROJECT_ROOT / "data" / "freesurfer_subjects"
DEFAULT_TARGET_SUBJECT = "cvs_avg35_inMNI152"
DEFAULT_ORACLE_CACHE_DIR = PROJECT_ROOT / "data" / "mni_coords"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cvsavg35_pial_coords"
_ALGORITHM_VERSION = "cvs_avg35 outer->pial nearest bridge v1 (2026-04-16)"


@dataclass(frozen=True)
class CvsAveragePialCache:
    names: tuple[str, ...]
    hemisphere: tuple[str, ...]
    outer_vertex_ids: np.ndarray  # (N,) int64
    pial_vertex_ids: np.ndarray  # (N,) int64
    outer_coords: np.ndarray  # (N, 3) float64
    pial_coords: np.ndarray  # (N, 3) float64
    outer_snap_mm: np.ndarray  # (N,) float64
    outer_to_pial_mm: np.ndarray  # (N,) float64

    def __post_init__(self) -> None:
        n = len(self.names)
        assert len(self.hemisphere) == n
        assert self.outer_vertex_ids.shape == (n,)
        assert self.pial_vertex_ids.shape == (n,)
        assert self.outer_coords.shape == (n, 3)
        assert self.pial_coords.shape == (n, 3)
        assert self.outer_snap_mm.shape == (n,)
        assert self.outer_to_pial_mm.shape == (n,)
        assert self.outer_coords.dtype == np.float64
        assert self.pial_coords.dtype == np.float64

    @property
    def vertex_ids(self) -> np.ndarray:
        return self.pial_vertex_ids

    @property
    def coords(self) -> np.ndarray:
        return self.pial_coords


def ensure_cvsavg35_subject(
    subjects_dir: Path = DEFAULT_SUBJECTS_DIR,
    box_root: Path = DEFAULT_BOX_ROOT,
    target_subject: str = DEFAULT_TARGET_SUBJECT,
) -> Path:
    """Ensure `subjects_dir/target_subject` exists, symlinking from Box if needed."""

    subjects_dir.mkdir(parents=True, exist_ok=True)
    target_path = subjects_dir / target_subject
    if target_path.exists():
        return target_path

    box_path = box_root / "ECoG_Recon" / target_subject
    if not box_path.exists():
        raise FileNotFoundError(
            f"missing {target_subject} under subjects_dir and Box: {target_path}, {box_path}"
        )
    target_path.symlink_to(box_path)
    return target_path


def _load_surface(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing surface file: {path}")
    verts, _faces = nib.freesurfer.io.read_geometry(str(path))
    return np.asarray(verts, dtype=np.float64)


def bridge_single_hemisphere(
    points_outer: np.ndarray,
    outer_verts: np.ndarray,
    pial_verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map outer-surface points onto pial via nearest outer then nearest pial."""

    if points_outer.shape[0] == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_x = np.zeros((0, 3), dtype=np.float64)
        empty_d = np.zeros((0,), dtype=np.float64)
        return empty_i, empty_i, empty_x, empty_d, empty_d

    outer_tree = cKDTree(outer_verts)
    outer_snap_mm, outer_vids = outer_tree.query(points_outer, k=1)
    outer_vids = np.asarray(outer_vids, dtype=np.int64)

    pial_tree = cKDTree(pial_verts)
    _unused, pial_vids = pial_tree.query(outer_verts[outer_vids], k=1)
    pial_vids = np.asarray(pial_vids, dtype=np.int64)
    pial_xyz = pial_verts[pial_vids].astype(np.float64, copy=True)
    outer_to_pial_mm = np.linalg.norm(outer_verts[outer_vids] - pial_xyz, axis=1)
    return outer_vids, pial_vids, pial_xyz, np.asarray(outer_snap_mm, dtype=np.float64), outer_to_pial_mm


def project_oracle_cache_to_cvsavg_pial(
    patient_id: str,
    oracle_cache_dir: Path = DEFAULT_ORACLE_CACHE_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    subjects_dir: Path = DEFAULT_SUBJECTS_DIR,
    box_root: Path = DEFAULT_BOX_ROOT,
    target_subject: str = DEFAULT_TARGET_SUBJECT,
) -> Path:
    """Bridge the Zac/cvs_avg35 oracle cache from outer coords to pial vertices."""

    ensure_cvsavg35_subject(subjects_dir=subjects_dir, box_root=box_root, target_subject=target_subject)
    oracle = load_mni_cache(patient_id, oracle_cache_dir)

    surf_dir = subjects_dir / target_subject / "surf"
    hemispheres = tuple("L" if is_left else "R" for is_left in oracle.is_left)
    n = len(oracle.names)
    outer_vids = np.full((n,), -1, dtype=np.int64)
    pial_vids = np.full((n,), -1, dtype=np.int64)
    pial_xyz = np.full((n, 3), np.nan, dtype=np.float64)
    outer_snap_mm = np.full((n,), np.nan, dtype=np.float64)
    outer_to_pial_mm = np.full((n,), np.nan, dtype=np.float64)

    for hem, mask in (("lh", oracle.is_left), ("rh", ~oracle.is_left)):
        if not mask.any():
            continue
        outer_verts = _load_surface(surf_dir / f"{hem}.pial-outer-smoothed")
        pial_verts = _load_surface(surf_dir / f"{hem}.pial")
        (
            out_hemi_vids,
            pial_hemi_vids,
            pial_hemi_xyz,
            snap_hemi_mm,
            bridge_hemi_mm,
        ) = bridge_single_hemisphere(oracle.coords[mask], outer_verts, pial_verts)
        outer_vids[mask] = out_hemi_vids
        pial_vids[mask] = pial_hemi_vids
        pial_xyz[mask] = pial_hemi_xyz
        outer_snap_mm[mask] = snap_hemi_mm
        outer_to_pial_mm[mask] = bridge_hemi_mm

    if (outer_vids < 0).any() or (pial_vids < 0).any() or np.isnan(pial_xyz).any():
        raise RuntimeError(f"{patient_id}: cvs_avg35 bridge left unset rows")

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{patient_id}_cvsavg35_pial.csv"
    with out_path.open("w") as f:
        f.write(
            "name,hemisphere,outer_vertex,pial_vertex,outer_x,outer_y,outer_z,"
            "pial_x,pial_y,pial_z,d_outer_mm,d_outer_to_pial_mm\n"
        )
        for idx, name in enumerate(oracle.names):
            f.write(
                f"{name},{hemispheres[idx]},{int(outer_vids[idx])},{int(pial_vids[idx])},"
                f"{oracle.coords[idx,0]:.6f},{oracle.coords[idx,1]:.6f},{oracle.coords[idx,2]:.6f},"
                f"{pial_xyz[idx,0]:.6f},{pial_xyz[idx,1]:.6f},{pial_xyz[idx,2]:.6f},"
                f"{outer_snap_mm[idx]:.6f},{outer_to_pial_mm[idx]:.6f}\n"
            )

    _update_sidecar(
        cache_dir=cache_dir,
        patient_id=patient_id,
        target_subject=target_subject,
        outer_snap_mm=outer_snap_mm,
        outer_to_pial_mm=outer_to_pial_mm,
    )
    return out_path


def _update_sidecar(
    *,
    cache_dir: Path,
    patient_id: str,
    target_subject: str,
    outer_snap_mm: np.ndarray,
    outer_to_pial_mm: np.ndarray,
) -> None:
    meta_path = cache_dir / "_projection_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta[patient_id] = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm_version": _ALGORITHM_VERSION,
        "target_subject": target_subject,
        "source_cache": "mni_coords oracle / cvs_avg35 outer",
        "bridge_rule": "nearest outer vertex then nearest pial vertex",
        "n_electrodes": int(outer_snap_mm.shape[0]),
        "outer_snap_median_mm": float(np.median(outer_snap_mm)),
        "outer_snap_max_mm": float(np.max(outer_snap_mm)),
        "outer_to_pial_median_mm": float(np.median(outer_to_pial_mm)),
        "outer_to_pial_max_mm": float(np.max(outer_to_pial_mm)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def load_cvsavg_pial_cache(
    patient_id: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> CvsAveragePialCache:
    path = cache_dir / f"{patient_id}_cvsavg35_pial.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing cvs_avg35 pial cache: {path}")

    names: list[str] = []
    hemisphere: list[str] = []
    outer_vertex_ids: list[int] = []
    pial_vertex_ids: list[int] = []
    outer_coords: list[list[float]] = []
    pial_coords: list[list[float]] = []
    outer_snap_mm: list[float] = []
    outer_to_pial_mm: list[float] = []
    with path.open() as f:
        header = f.readline().rstrip("\n")
        expected = (
            "name,hemisphere,outer_vertex,pial_vertex,outer_x,outer_y,outer_z,"
            "pial_x,pial_y,pial_z,d_outer_mm,d_outer_to_pial_mm"
        )
        if header != expected:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for row_idx, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 12:
                raise ValueError(f"{path}:{row_idx}: expected 12 columns, got {fields!r}")
            (
                name,
                hem,
                outer_vid,
                pial_vid,
                outer_x,
                outer_y,
                outer_z,
                pial_x,
                pial_y,
                pial_z,
                d_outer,
                d_bridge,
            ) = fields
            if hem not in {"L", "R"}:
                raise ValueError(f"{path}:{row_idx}: hemisphere must be L or R, got {hem!r}")
            names.append(name)
            hemisphere.append(hem)
            outer_vertex_ids.append(int(outer_vid))
            pial_vertex_ids.append(int(pial_vid))
            outer_coords.append([float(outer_x), float(outer_y), float(outer_z)])
            pial_coords.append([float(pial_x), float(pial_y), float(pial_z)])
            outer_snap_mm.append(float(d_outer))
            outer_to_pial_mm.append(float(d_bridge))

    return CvsAveragePialCache(
        names=tuple(names),
        hemisphere=tuple(hemisphere),
        outer_vertex_ids=np.array(outer_vertex_ids, dtype=np.int64),
        pial_vertex_ids=np.array(pial_vertex_ids, dtype=np.int64),
        outer_coords=np.array(outer_coords, dtype=np.float64),
        pial_coords=np.array(pial_coords, dtype=np.float64),
        outer_snap_mm=np.array(outer_snap_mm, dtype=np.float64),
        outer_to_pial_mm=np.array(outer_to_pial_mm, dtype=np.float64),
    )
