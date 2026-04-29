"""ORACLE / MIGRATION REFERENCE — NOT PHASE-1 ACTIVE.

Per `#36` in `docs/tactics.md`, the active Phase-1 patient-side
projection is `src/speech_decoding/v14/fsaverage_projection.py` (stock pial +
sphere.reg → fsaverage). This module is kept as the cvs_avg35 migration
oracle: it feeds `scripts/compare_fsaverage_spatial_parity.py` and has an
S14 oracle regression in `tests/v14/test_coordinates.py`. Do not route
loader/model code through the cvs_avg35 cache produced by this file.

Python port of Zac Spalding's ``sub2AvgBrainClinical.m``, plus a frame
translation step (added 2026-04-16, v2) described below.

Projects subdural ECoG electrodes from subject FreeSurfer space to the
``cvs_avg35_inMNI152`` average brain via surface-based registration. The
algorithm has four steps per hemisphere:

1. Snap each electrode to the nearest vertex on the subject's
   ``lh.pial-outer-smoothed`` / ``rh.pial-outer-smoothed`` surface.
2. Look up the corresponding vertex on the subject's
   ``lh.sphere-outer-mni.reg`` / ``rh.sphere-outer-mni.reg`` sphere
   (the spherical registration to cvs_avg35).
3. Find the nearest vertex on the average subject's
   ``lh.sphere-outer.reg`` / ``rh.sphere-outer.reg`` sphere.
4. Read out the corresponding point from the average subject's
   ``lh.pial-outer-smoothed`` / ``rh.pial-outer-smoothed`` surface.
5. *Frame translation:* add the average subject's ``cras`` (read from
   its ``lh.pial`` header) so the output is in scanner RAS (≈ MNI).

The projection is not an affine transform. It is a nonlinear surface
registration; the output of step 4 is constrained to lie on the average
subject's outer-smoothed pial envelope. See ``docs/tactics.md``
blocker #1 for the full context and the post-trust PM coverage spot check.

Output is keyed by electrode *name* (from ``<pt>.electrodeNames``), not by
physical position. See blocker #12 for the reason and the downstream bridge
contract.

Frame translation rationale (v2, 2026-04-16):

    Box's ``cvs_avg35_inMNI152/surf/lh.pial-outer-smoothed`` stores
    ``cras = (0, 0, 0)`` in its FreeSurfer header, while the sibling
    ``lh.pial`` stores ``cras = (-1, -17, 19)``. Both surfaces are in
    the same subject tkrRAS; only ``lh.pial`` has the cras metadata to
    translate to scanner RAS. Without step 5, the output of step 4 is
    mislabeled — the file name is ``_MNI152.csv`` but the coords are
    tkrRAS, offset by +(-1, -17, 19) from true MNI. Sampling a volume
    that lives in MNI (e.g. ``BNA_PM_dilated_8mm.nii.gz``) at these
    coords lands ~22 mm off the electrode's actual location and flips
    whole sub-gyri (e.g., S14 sensorimotor argmax-to-PrG instead of
    PoG). The Python port inherited this from ``sub2AvgBrainClinical.m``.
    Verified via triangulation across S14/S26/S33/S62 surface-baked vs
    volumetric argmax on 2026-04-16.

    The fix is defensive and generalizes: for any average subject whose
    ``lh.pial`` has a correct cras (including fsaverage, where cras is
    zero and the step is a no-op), the output is always in scanner RAS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

import mne


DEFAULT_AVG_SUBJECT = "cvs_avg35_inMNI152"
_ALGORITHM_VERSION = "sub2AvgBrainClinical.m python port v2 (2026-04-16, cras-corrected)"


@dataclass(frozen=True)
class PatientElectrodes:
    names: tuple[str, ...]
    coords: np.ndarray  # (N, 3) float64, subject RAS in mm
    is_left: np.ndarray  # (N,) bool
    is_subdural: np.ndarray  # (N,) bool
    brainshift_method: str

    def __post_init__(self) -> None:
        n = len(self.names)
        assert self.coords.shape == (n, 3), f"coords shape {self.coords.shape} != ({n}, 3)"
        assert self.is_left.shape == (n,), f"is_left shape {self.is_left.shape}"
        assert self.is_subdural.shape == (n,), f"is_subdural shape {self.is_subdural.shape}"
        assert self.coords.dtype == np.float64


def read_lepto(patient_dir: Path, patient_id: str) -> PatientElectrodes:
    """Read ``<pt>.LEPTO`` + ``<pt>.electrodeNames`` into a PatientElectrodes.

    ``<pt>.LEPTO`` layout:
        line 1: ``<timestamp>\\t<brainshift_method>\\t<fs_version>``
        line 2: ``R A S``
        rows 3+: ``x y z`` (space-separated floats, subject RAS in mm)

    ``<pt>.electrodeNames`` layout:
        line 1: ``<timestamp>``
        line 2: ``Name, Depth/Strip/Grid, Hem``
        rows 3+: ``<name> <G|S|D> <L|R>``

    The two files must have the same number of data rows, in the same order.
    """

    lepto_path = patient_dir / "elec_recon" / f"{patient_id}.LEPTO"
    names_path = patient_dir / "elec_recon" / f"{patient_id}.electrodeNames"
    if not lepto_path.exists():
        raise FileNotFoundError(f"missing LEPTO file: {lepto_path}")
    if not names_path.exists():
        raise FileNotFoundError(f"missing electrodeNames file: {names_path}")

    with lepto_path.open() as f:
        lepto_lines = f.read().splitlines()
    if len(lepto_lines) < 3:
        raise ValueError(f"{lepto_path}: expected at least 3 lines, got {len(lepto_lines)}")
    header = lepto_lines[0]
    parts = header.split("\t")
    brainshift_method = parts[1] if len(parts) >= 2 else "unknown"
    coord_system = lepto_lines[1].strip()
    if coord_system != "R A S":
        raise ValueError(
            f"{lepto_path}: expected 'R A S' on line 2, got {coord_system!r}"
        )
    coords = np.array(
        [[float(v) for v in line.split()] for line in lepto_lines[2:]],
        dtype=np.float64,
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"{lepto_path}: coord rows must have exactly 3 floats")

    with names_path.open() as f:
        name_lines = f.read().splitlines()
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


def project_single_hemisphere(
    points: np.ndarray,
    sub_pial_verts: np.ndarray,
    sub_sph_verts: np.ndarray,
    avg_sph_verts: np.ndarray,
    avg_pial_verts: np.ndarray,
) -> np.ndarray:
    """Core algorithm, no file I/O. Vectorized over ``points``.

    ``sub_pial_verts`` and ``sub_sph_verts`` must have the same row count
    (same vertex indexing). Same for ``avg_sph_verts`` and ``avg_pial_verts``.
    """

    if sub_pial_verts.shape[0] != sub_sph_verts.shape[0]:
        raise ValueError(
            f"subject pial/sphere vertex counts disagree: "
            f"{sub_pial_verts.shape[0]} vs {sub_sph_verts.shape[0]}"
        )
    if avg_sph_verts.shape[0] != avg_pial_verts.shape[0]:
        raise ValueError(
            f"average pial/sphere vertex counts disagree: "
            f"{avg_sph_verts.shape[0]} vs {avg_pial_verts.shape[0]}"
        )
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    sub_tree = cKDTree(sub_pial_verts)
    _, sub_vids = sub_tree.query(points, k=1)

    avg_tree = cKDTree(avg_sph_verts)
    _, avg_vids = avg_tree.query(sub_sph_verts[sub_vids], k=1)

    return avg_pial_verts[avg_vids].astype(np.float64, copy=True)


def _load_surface(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing surface file: {path}")
    result = mne.read_surface(str(path), verbose=False)
    verts = result[0]
    return np.asarray(verts, dtype=np.float64)


def _read_surface_cras(path: Path) -> np.ndarray:
    """Read the FreeSurfer ``cras`` (surfaceRAS-to-scannerRAS translation) from a surface file.

    Raises if the file is missing or the header lacks a valid ``cras`` entry.
    The returned vector is a 3-vector in mm; adding it to vertex coordinates
    expressed in subject tkrRAS yields scanner RAS.
    """

    if not path.exists():
        raise FileNotFoundError(f"missing surface file: {path}")
    result = nib.freesurfer.io.read_geometry(str(path), read_metadata=True)
    if not isinstance(result, tuple) or len(result) < 3:
        raise RuntimeError(f"{path}: unexpected read_geometry return shape {type(result)!r}")
    meta: Any = result[2]
    if "cras" not in meta:
        raise ValueError(f"{path}: surface header missing cras metadata")
    return np.asarray(meta["cras"], dtype=np.float64)


def project_to_average(
    sub_coords: np.ndarray,
    is_left: np.ndarray,
    sub_surf_dir: Path,
    avg_surf_dir: Path,
) -> np.ndarray:
    """Project subject electrodes to the average subject's outer-smoothed pial.

    Assumes ``cvs_avg35_inMNI152`` as the target — subject side uses
    ``sphere-outer-mni.reg``, average side uses ``sphere-outer.reg``.
    Left- and right-hemisphere electrodes are projected through their
    respective surface files; output is merged back into input order.
    """

    n = sub_coords.shape[0]
    if is_left.shape != (n,):
        raise ValueError(f"is_left shape {is_left.shape} != ({n},)")
    out = np.full((n, 3), np.nan, dtype=np.float64)

    # Frame translation: read cras from the average subject's `lh.pial`. Step 4
    # returns a point on `<hem>.pial-outer-smoothed`, whose cras is unreliable
    # (zero on Box's cvs_avg35_inMNI152 even though the surface is stored in
    # tkrRAS). The sibling `lh.pial` has the correct cras; both surfaces share
    # the same subject coordinate frame. We read cras from `lh.pial` once and
    # assert it matches `rh.pial` so the translation is unambiguous.
    cras = _read_surface_cras(avg_surf_dir / "lh.pial")
    rh_cras = _read_surface_cras(avg_surf_dir / "rh.pial")
    if not np.allclose(cras, rh_cras, atol=1e-3):
        raise ValueError(
            f"average-subject lh.pial cras {cras} differs from rh.pial cras "
            f"{rh_cras}; frame translation is ambiguous"
        )

    for hem, mask in (("lh", is_left), ("rh", ~is_left)):
        if not mask.any():
            continue
        sub_pial = _load_surface(sub_surf_dir / f"{hem}.pial-outer-smoothed")
        sub_sph = _load_surface(sub_surf_dir / f"{hem}.sphere-outer-mni.reg")
        avg_sph = _load_surface(avg_surf_dir / f"{hem}.sphere-outer.reg")
        avg_pial = _load_surface(avg_surf_dir / f"{hem}.pial-outer-smoothed")
        out[mask] = project_single_hemisphere(
            sub_coords[mask], sub_pial, sub_sph, avg_sph, avg_pial
        ) + cras

    if np.isnan(out).any():
        raise RuntimeError("projection left NaN rows — hemisphere mask did not cover all electrodes")
    return out


def _assert_hemisphere_sign_consistent(elec: PatientElectrodes) -> None:
    """Sanity: left-labeled electrodes should have x <= 0 and right x >= 0.

    Electrodes near the midline can sit at |x| < ~5 mm on either side of
    zero; the check fails only on clearly-wrong-hemisphere assignments.
    """

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


def project_patient(
    patient_id: str,
    box_root: Path,
    cache_dir: Path,
    avg_subject: str = DEFAULT_AVG_SUBJECT,
) -> Path:
    """Full single-patient driver: read, project, write CSV + sidecar metadata.

    Returns the path of the written CSV.
    """

    if avg_subject != DEFAULT_AVG_SUBJECT:
        raise NotImplementedError(
            f"Phase 1 only supports {DEFAULT_AVG_SUBJECT}; got {avg_subject!r}"
        )

    patient_dir = box_root / "ECoG_Recon" / patient_id
    avg_dir = box_root / "ECoG_Recon" / avg_subject / "surf"
    sub_surf_dir = patient_dir / "surf"
    elec_recon = patient_dir / "elec_recon"

    elec = read_lepto(patient_dir, patient_id)

    if not elec.is_subdural.all():
        depths = [n for n, s in zip(elec.names, elec.is_subdural) if not s]
        raise ValueError(
            f"{patient_id}: depth electrodes found, not supported in Phase 1: {depths}"
        )
    _assert_hemisphere_sign_consistent(elec)

    avg_cras = _read_surface_cras(avg_dir / "lh.pial")

    mni = project_to_average(
        sub_coords=elec.coords,
        is_left=elec.is_left,
        sub_surf_dir=sub_surf_dir,
        avg_surf_dir=avg_dir,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{patient_id}_MNI152.csv"
    with out_path.open("w") as f:
        f.write("name,x,y,z,hemisphere\n")
        for name, (x, y, z), is_l in zip(elec.names, mni, elec.is_left):
            hem_char = "L" if is_l else "R"
            f.write(f"{name},{x:.6f},{y:.6f},{z:.6f},{hem_char}\n")

    _update_sidecar(
        cache_dir=cache_dir,
        patient_id=patient_id,
        avg_subject=avg_subject,
        elec=elec,
        lepto_path=elec_recon / f"{patient_id}.LEPTO",
        names_path=elec_recon / f"{patient_id}.electrodeNames",
        avg_cras=avg_cras,
    )
    return out_path


def _update_sidecar(
    cache_dir: Path,
    patient_id: str,
    avg_subject: str,
    elec: PatientElectrodes,
    lepto_path: Path,
    names_path: Path,
    avg_cras: np.ndarray,
) -> None:
    meta_path = cache_dir / "_projection_meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)
    else:
        meta = {}
    meta[patient_id] = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm_version": _ALGORITHM_VERSION,
        "avg_subject": avg_subject,
        "avg_cras_mm": [float(c) for c in avg_cras],
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
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")


@dataclass(frozen=True)
class MNICoordinateCache:
    names: tuple[str, ...]
    coords: np.ndarray  # (N, 3) float64, MNI152 in mm
    is_left: np.ndarray  # (N,) bool

    def lookup(self, name: str) -> np.ndarray:
        try:
            idx = self.names.index(name)
        except ValueError as err:
            raise KeyError(f"electrode name not in cache: {name!r}") from err
        return self.coords[idx]


def load_mni_cache(patient_id: str, cache_dir: Path) -> MNICoordinateCache:
    """Read a cached MNI152 CSV back into arrays. Used by the v14 loader."""

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
            names.append(name)
            coords.append([float(x), float(y), float(z)])
            if hem not in {"L", "R"}:
                raise ValueError(f"{path}:{row_idx}: hemisphere must be L or R, got {hem!r}")
            is_left.append(hem == "L")
    return MNICoordinateCache(
        names=tuple(names),
        coords=np.array(coords, dtype=np.float64),
        is_left=np.array(is_left, dtype=bool),
    )


def compare_against_oracle(
    cache: MNICoordinateCache,
    oracle_names: Sequence[str],
    oracle_coords: np.ndarray,
) -> dict[str, float]:
    """Per-electrode residual summary against a trusted oracle (Zac's MATLAB run).

    ``oracle_coords`` must be aligned with ``oracle_names``. The two name sets
    must be identical; ordering may differ. Returns max / mean / median
    per-electrode Euclidean residual in mm.
    """

    if len(oracle_names) != oracle_coords.shape[0]:
        raise ValueError("oracle names / coords row count mismatch")
    if set(oracle_names) != set(cache.names):
        missing_in_oracle = set(cache.names) - set(oracle_names)
        missing_in_cache = set(oracle_names) - set(cache.names)
        raise ValueError(
            f"electrode name sets differ: in cache but not oracle={sorted(missing_in_oracle)}; "
            f"in oracle but not cache={sorted(missing_in_cache)}"
        )
    oracle_by_name = {n: oracle_coords[i] for i, n in enumerate(oracle_names)}
    residuals = np.array(
        [np.linalg.norm(cache.coords[i] - oracle_by_name[n]) for i, n in enumerate(cache.names)],
        dtype=np.float64,
    )
    return {
        "max_mm": float(residuals.max()),
        "mean_mm": float(residuals.mean()),
        "median_mm": float(np.median(residuals)),
        "p95_mm": float(np.percentile(residuals, 95)),
        "n": int(residuals.shape[0]),
    }
