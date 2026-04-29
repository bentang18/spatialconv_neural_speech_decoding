"""fsaverage atlas + projection utilities.

Two pieces, merged from `fsaverage_projection.py` + `fsaverage_atlas.py`
during the pre-Stage-0 reorg:

1. **Projection** — strict patient pial -> patient sphere.reg ->
   fsaverage sphere.reg -> fsaverage pial. ``coordinates.py`` (now archived)
   was the cvs_avg35 oracle; this is the first-pass pure fsaverage path.
   ``pial-outer-smoothed`` is excluded on purpose so the parity experiment
   cleanly tests the stock FreeSurfer route.

2. **Atlas bake + support sampling** — load the baked Brainnetome surface
   stack and read per-electrode parcel support at projected fsaverage
   vertices.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BAKE_DIR = PROJECT_ROOT / "data" / "atlas" / "fsaverage_bake_v2c"
DEFAULT_BNA_TREE = Path.home() / "nilearn_data" / "bnatlas_tree.csv"
DEFAULT_ORACLE_PM_PATH = PROJECT_ROOT / "data" / "atlas" / "BNA_PM_dilated_8mm.nii.gz"
DEFAULT_ORACLE_SIGMA_MM = 1.5
DEFAULT_BNA_N_ROIS = 246
_SUPPORTED_HEMIS = ("lh", "rh")

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


@dataclass(frozen=True)
class BakedAtlasSurface:
    frame_ids: tuple[int, ...]
    parcel_names: tuple[str, ...]
    values_by_hemi: dict[str, np.ndarray]  # hemi -> (n_vertices, n_parcels)
    kind: str  # "raw" or "smoothed"
    smooth_fwhm: float | None
    source_dir: Path


@dataclass(frozen=True)
class MNICoordinateCache:
    names: tuple[str, ...]
    coords: np.ndarray  # (N, 3) float64, MNI152 in mm
    is_left: np.ndarray  # (N,) bool

    def lookup(self, name: str) -> np.ndarray:
        try:
            idx = self.names.index(name)
        except ValueError as err:
            raise KeyError(f"electrode name not in MNI cache: {name!r}") from err
        return self.coords[idx]


def load_bna_names(tree_path: Path, n_rois: int) -> tuple[str, ...]:
    if not tree_path.exists():
        return tuple(f"ROI_{idx:03d}" for idx in range(1, n_rois + 1))
    names_by_idx: dict[int, str] = {}
    with tree_path.open() as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                names_by_idx[int(row[0])] = row[1].strip()
    return tuple(names_by_idx.get(idx, f"ROI_{idx:03d}") for idx in range(1, n_rois + 1))


def load_bna_names_for_frames(tree_path: Path, frame_ids: list[int] | tuple[int, ...]) -> tuple[str, ...]:
    if not tree_path.exists():
        return tuple(f"ROI_{frame_idx + 1:03d}" for frame_idx in frame_ids)
    names_by_idx: dict[int, str] = {}
    with tree_path.open() as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                names_by_idx[int(row[0])] = row[1].strip()
    return tuple(names_by_idx.get(frame_idx + 1, f"ROI_{frame_idx + 1:03d}") for frame_idx in frame_ids)


def load_mni_cache(patient_id: str, cache_dir: Path) -> MNICoordinateCache:
    path = cache_dir / f"{patient_id}_MNI152.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing MNI cache: {path}")

    names: list[str] = []
    coords: list[list[float]] = []
    is_left: list[bool] = []
    with path.open() as f:
        header = f.readline().rstrip("\n")
        if header != "name,x,y,z,hemisphere":
            raise ValueError(f"{path}: unexpected header {header!r}")
        for row_idx, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split(",")
            if len(fields) != 5:
                raise ValueError(f"{path}:{row_idx}: expected 5 columns, got {fields!r}")
            name, x, y, z, hem = fields
            if hem not in {"L", "R"}:
                raise ValueError(f"{path}:{row_idx}: hemisphere must be L or R, got {hem!r}")
            names.append(name)
            coords.append([float(x), float(y), float(z)])
            is_left.append(hem == "L")
    return MNICoordinateCache(
        names=tuple(names),
        coords=np.array(coords, dtype=np.float64),
        is_left=np.array(is_left, dtype=bool),
    )


def load_bake_manifest(out_dir: Path) -> dict[str, object]:
    path = out_dir / "bake_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing bake manifest: {path}")
    return json.loads(path.read_text())


def _stack_file(out_dir: Path, kind: str) -> Path:
    return out_dir / f"{kind}_stack.npz"


def _frame_path(
    out_dir: Path,
    hemi: str,
    frame: int,
    kind: str,
    smooth_fwhm: float | None,
) -> Path:
    hemi_dir = out_dir / hemi
    if kind == "raw":
        return hemi_dir / f"{hemi}.frame{frame:03d}.fsaverage.mgh"
    if smooth_fwhm is None:
        raise ValueError("smooth_fwhm is required for smoothed atlas loads")
    return hemi_dir / f"{hemi}.frame{frame:03d}.fsaverage.fwhm{smooth_fwhm:g}.mgh"


def _multiframe_path(
    out_dir: Path,
    hemi: str,
    kind: str,
    smooth_fwhm: float | None,
) -> Path:
    hemi_dir = out_dir / hemi
    if kind == "raw":
        return hemi_dir / f"{hemi}.fsaverage.mgh"
    if smooth_fwhm is None:
        raise ValueError("smooth_fwhm is required for smoothed atlas loads")
    return hemi_dir / f"{hemi}.fsaverage.fwhm{smooth_fwhm:g}.mgh"


def _load_surface_stack(paths: list[Path]) -> np.ndarray:
    cols: list[np.ndarray] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"missing baked atlas frame: {path}")
        data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64).reshape(-1)
        cols.append(data)
    return np.stack(cols, axis=1)


def _load_multiframe_surface(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing baked atlas file: {path}")
    data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float64)
    data = np.squeeze(data)
    if data.ndim == 1:
        return data[:, None]
    if data.ndim != 2:
        raise ValueError(f"unexpected multiframe surface shape {data.shape} in {path}")
    return data


def consolidate_baked_atlas(out_dir: Path, tree_path: Path = DEFAULT_BNA_TREE) -> None:
    manifest = load_bake_manifest(out_dir)
    frames = [int(frame) for frame in manifest["frames"]]
    smooth_fwhm = float(manifest["smooth_fwhm"])
    parcel_names = load_bna_names_for_frames(tree_path, frames)

    for kind in ("raw", "smoothed"):
        arrays: dict[str, np.ndarray] = {}
        for hemi in manifest["hemis"]:
            hemi_str = str(hemi)
            multi_path = _multiframe_path(
                out_dir=out_dir,
                hemi=hemi_str,
                kind=kind,
                smooth_fwhm=smooth_fwhm,
            )
            if multi_path.exists():
                arrays[hemi_str] = _load_multiframe_surface(multi_path)
            else:
                paths = [
                    _frame_path(
                        out_dir=out_dir,
                        hemi=hemi_str,
                        frame=frame,
                        kind=kind,
                        smooth_fwhm=smooth_fwhm,
                    )
                    for frame in frames
                ]
                arrays[hemi_str] = _load_surface_stack(paths)
        np.savez_compressed(
            _stack_file(out_dir, kind),
            frame_ids=np.array(frames, dtype=np.int64),
            parcel_names=np.array(parcel_names, dtype=object),
            smooth_fwhm=np.array([smooth_fwhm], dtype=np.float64),
            lh=arrays["lh"],
            rh=arrays["rh"],
        )


def load_baked_atlas(
    out_dir: Path = DEFAULT_BAKE_DIR,
    *,
    kind: str,
    tree_path: Path = DEFAULT_BNA_TREE,
) -> BakedAtlasSurface:
    if kind not in {"raw", "smoothed"}:
        raise ValueError(f"kind must be 'raw' or 'smoothed', got {kind!r}")

    manifest = load_bake_manifest(out_dir)
    stack_path = _stack_file(out_dir, kind)
    if stack_path.exists():
        payload = np.load(stack_path, allow_pickle=True)
        frame_ids = tuple(int(v) for v in payload["frame_ids"])
        parcel_names = tuple(str(v) for v in payload["parcel_names"])
        smooth_fwhm = float(payload["smooth_fwhm"][0])
        values_by_hemi = {hemi: np.asarray(payload[hemi], dtype=np.float64) for hemi in _SUPPORTED_HEMIS}
        return BakedAtlasSurface(
            frame_ids=frame_ids,
            parcel_names=parcel_names,
            values_by_hemi=values_by_hemi,
            kind=kind,
            smooth_fwhm=None if kind == "raw" else smooth_fwhm,
            source_dir=out_dir,
        )

    frames = tuple(int(frame) for frame in manifest["frames"])
    smooth_fwhm = float(manifest["smooth_fwhm"])
    parcel_names = load_bna_names_for_frames(tree_path, frames)
    values_by_hemi: dict[str, np.ndarray] = {}
    for hemi in manifest["hemis"]:
        hemi_str = str(hemi)
        multi_path = _multiframe_path(
            out_dir=out_dir,
            hemi=hemi_str,
            kind=kind,
            smooth_fwhm=smooth_fwhm,
        )
        if multi_path.exists():
            values_by_hemi[hemi_str] = _load_multiframe_surface(multi_path)
        else:
            paths = [
                _frame_path(
                    out_dir=out_dir,
                    hemi=hemi_str,
                    frame=frame,
                    kind=kind,
                    smooth_fwhm=smooth_fwhm,
                )
                for frame in frames
            ]
            values_by_hemi[hemi_str] = _load_surface_stack(paths)

    return BakedAtlasSurface(
        frame_ids=frames,
        parcel_names=parcel_names,
        values_by_hemi=values_by_hemi,
        kind=kind,
        smooth_fwhm=None if kind == "raw" else smooth_fwhm,
        source_dir=out_dir,
    )


def sample_baked_support(
    cache: FsaverageCoordinateCache,
    atlas: BakedAtlasSurface,
) -> np.ndarray:
    support = np.zeros((len(cache.names), len(atlas.frame_ids)), dtype=np.float64)
    for row_idx, (hem, vid) in enumerate(zip(cache.hemisphere, cache.vertex_ids)):
        hemi_key = "lh" if hem == "L" else "rh"
        hemi_values = atlas.values_by_hemi[hemi_key]
        if vid < 0 or vid >= hemi_values.shape[0]:
            raise IndexError(
                f"{hemi_key} vertex {vid} out of range for atlas stack shape {hemi_values.shape}"
            )
        support[row_idx] = hemi_values[int(vid)]
    return support


def compute_argmax_assignments(support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    totals = support.sum(axis=1)
    argmax_idx = np.zeros((support.shape[0],), dtype=np.int64)
    max_support = np.zeros((support.shape[0],), dtype=np.float64)
    valid = totals > 0
    if valid.any():
        argmax_idx[valid] = support[valid].argmax(axis=1) + 1
        max_support[valid] = support[valid].max(axis=1)
    return argmax_idx, max_support, valid


def compute_token_support(support: np.ndarray) -> np.ndarray:
    return support.sum(axis=0) / 100.0


def compute_argmax_wins(support: np.ndarray) -> np.ndarray:
    argmax_idx, _max_support, valid = compute_argmax_assignments(support)
    wins = np.zeros((support.shape[1],), dtype=np.int64)
    for idx in argmax_idx[valid]:
        wins[int(idx) - 1] += 1
    return wins


def compute_any_coverage(support: np.ndarray) -> np.ndarray:
    return (support > 0).sum(axis=0).astype(np.int64)


def build_cohort_ranking_rows(
    *,
    support_by_patient: dict[str, np.ndarray],
    parcel_names: tuple[str, ...],
) -> list[dict[str, object]]:
    if not support_by_patient:
        return []
    parcel_count = len(parcel_names)
    cohort_support = np.concatenate(list(support_by_patient.values()), axis=0)
    token_support = compute_token_support(cohort_support)
    argmax_wins = compute_argmax_wins(cohort_support)
    any_cov = compute_any_coverage(cohort_support)
    max_support = cohort_support.max(axis=0)

    patient_count = np.zeros((parcel_count,), dtype=np.int64)
    for patient_support in support_by_patient.values():
        patient_count += (patient_support.sum(axis=0) > 0).astype(np.int64)

    rows: list[dict[str, object]] = []
    for idx, name in enumerate(parcel_names, start=1):
        rows.append(
            {
                "parcel_idx": idx,
                "parcel_name": name,
                "token_support": float(token_support[idx - 1]),
                "argmax_wins": int(argmax_wins[idx - 1]),
                "any_coverage": int(any_cov[idx - 1]),
                "n_patients": int(patient_count[idx - 1]),
                "max_support": float(max_support[idx - 1]),
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["argmax_wins"]),
            -float(row["token_support"]),
            -float(row["max_support"]),
        )
    )
    return rows


def voxel_sizes_mm(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def smooth_pm_gaussian(pm: np.ndarray, affine: np.ndarray, sigma_mm: float) -> np.ndarray:
    if sigma_mm <= 0:
        return pm
    vox_mm = voxel_sizes_mm(affine)
    sigma_vox = (sigma_mm / vox_mm).astype(np.float64)
    return gaussian_filter(
        pm,
        sigma=(float(sigma_vox[0]), float(sigma_vox[1]), float(sigma_vox[2]), 0.0),
        mode="constant",
        cval=0.0,
    )


def mni_to_voxel(mni_xyz: np.ndarray, affine: np.ndarray) -> np.ndarray:
    n = mni_xyz.shape[0]
    h = np.concatenate([mni_xyz, np.ones((n, 1))], axis=1)
    return (h @ np.linalg.inv(affine).T)[:, :3]


def sample_pm_trilinear(pm: np.ndarray, affine: np.ndarray, mni_xyz: np.ndarray) -> np.ndarray:
    vox = mni_to_voxel(mni_xyz, affine)
    shape = np.array(pm.shape[:3])
    in_bounds = np.all((vox >= 0) & (vox <= shape - 1), axis=1)
    samples = np.zeros((mni_xyz.shape[0], pm.shape[3]), dtype=np.float64)
    if not in_bounds.any():
        return samples
    coords = vox[in_bounds].T
    for roi_idx in range(pm.shape[3]):
        samples[in_bounds, roi_idx] = map_coordinates(
            pm[..., roi_idx],
            coords,
            order=1,
            mode="constant",
            cval=0.0,
        )
    return samples


def load_oracle_volume_support(
    *,
    points_mni: np.ndarray,
    pm_path: Path = DEFAULT_ORACLE_PM_PATH,
    sigma_mm: float = DEFAULT_ORACLE_SIGMA_MM,
) -> np.ndarray:
    img = nib.load(str(pm_path))
    pm = img.get_fdata(dtype=np.float32)
    pm = smooth_pm_gaussian(pm, np.asarray(img.affine), sigma_mm)
    return sample_pm_trilinear(pm, np.asarray(img.affine), points_mni)


def assert_full_bake(atlas: BakedAtlasSurface, *, expected_n_rois: int = DEFAULT_BNA_N_ROIS) -> None:
    if len(atlas.frame_ids) != expected_n_rois:
        raise ValueError(
            f"fsaverage bake is partial: expected {expected_n_rois} atlas frames, "
            f"found {len(atlas.frame_ids)}. Re-run the full bake before support/frame/parity scripts."
        )
    expected = tuple(range(expected_n_rois))
    if atlas.frame_ids != expected:
        raise ValueError(
            "fsaverage bake frame ids are not the full ordered 0..245 set. "
            "Support/frame/parity scripts require a full bake, not a subset smoke test."
        )
