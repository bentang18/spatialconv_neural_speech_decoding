"""Cogan per-electrode DKT localization: sphere-sample a FreeSurfer atlas volume.

Produces the per-subject ``depth-wm.csv`` the v3 parcel_fn consumes, by replaying
the lab's own ECoG_Recon electrode-localization for every feed subject — because
only D23/D24 ship a ready DK CSV and NONE ship a DKT one, so DKT must be derived
(recon inspected on DCC 2026-07-13; see the ingestion-contract memo).

Method (verbatim from the shipped DK CSV's ``#Info`` header): for each electrode,
take every atlas voxel within a fixed-radius sphere around its coordinate, read
each voxel's integer label, and report BOTH the label at the sphere ORIGIN
(center voxel) and the per-label proportions over the sphere. The origin-vs-
majority choice for the final ``DKT`` column is a downstream selection (Ben-gated),
so this core returns both and picks neither.

The one convention this file does NOT hard-code is how a ``.LEPTOVOX`` coordinate
triple maps onto the nibabel array's ``(i, j, k)`` index order — the FreeSurfer
conformed-volume vox/tkr subtlety. That is pinned EMPIRICALLY on DCC by
``reproduce_dk_csv`` below: sample ``aparc+aseg.mgz`` and require a row-for-row
match against the shipped D23/D24 DK CSV; the ``axis_perm``/``flip`` that makes it
match is then reused for the DKT volume. So the geometry here is unit-testable on
a synthetic volume with zero FreeSurfer data, and the convention is fixed against
ground truth, not assumed.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# ECoG_Recon input parsers (D<id>.electrodeNames / D<id>.LEPTOVOX) + FS LUT.
#
# The two per-subject recon files are ROW-ALIGNED: line k of ``electrodeNames``
# names the contact whose coordinate is line k of ``LEPTOVOX``. Both carry two
# header lines (a timestamp line, then a column-caption line) before the data.
# Verified against D24 (2026-07-13): 2 headers + 52 contact rows each.
# ---------------------------------------------------------------------------

# electrodeNames data row: ``<name> <G|D|S> <L|R|?>`` (name, grid/depth/strip,
# hemisphere), whitespace-separated. We keep the name; kind/hemi are provenance.
_ELECNAME_RE = re.compile(
    r"^(?P<name>\S+)\s+(?P<kind>[GDS])\s+(?P<hem>[LRlr?])\s*$"
)


def read_electrode_names(path: str) -> list[tuple[str, str, str]]:
    """``D<id>.electrodeNames`` → ``[(name, kind, hem), …]`` in file order.

    Skips the two header lines (timestamp; ``Name, Depth/Strip/Grid, Hem``).
    Each data row is ``<name> <G/D/S> <L/R>``; a row that doesn't match is a
    hard error (the file is machine-generated, so a mismatch means a format
    drift we must not paper over).
    """
    out: list[tuple[str, str, str]] = []
    with open(path) as fh:
        lines = fh.read().splitlines()
    for ln in lines[2:]:
        if not ln.strip():
            continue
        m = _ELECNAME_RE.match(ln)
        if not m:
            raise ValueError(f"{path}: unparseable electrodeNames row {ln!r}")
        out.append((m.group("name"), m.group("kind"), m.group("hem").upper()))
    return out


def read_leptovox(path: str) -> np.ndarray:
    """``D<id>.LEPTOVOX`` → ``(N, 3)`` float array of ``R A S`` coordinates.

    Skips the two header lines (timestamp+tool; ``R A S``). These are the
    dykstra-projected coordinates in the FreeSurfer conformed volume's frame;
    how the triple maps onto the atlas array's ``(i, j, k)`` index order is NOT
    decided here — that is pinned empirically by ``reproduce_dk_csv`` on DCC.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()
    rows: list[tuple[float, float, float]] = []
    for ln in lines[2:]:
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"{path}: LEPTOVOX row not 3 floats: {ln!r}")
        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return np.array(rows, dtype=float).reshape(-1, 3)


def read_recon_electrodes(
    electrode_names_path: str, leptovox_path: str
) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    """Read + row-align the two recon files, asserting equal length.

    Returns ``(electrode_rows, coords_ras (N,3))`` with ``electrode_rows[k]``
    the contact at ``coords_ras[k]``. A length mismatch means the recon pair is
    inconsistent and localization would silently misassign coordinates.
    """
    names = read_electrode_names(electrode_names_path)
    coords = read_leptovox(leptovox_path)
    if len(names) != coords.shape[0]:
        raise ValueError(
            f"recon mismatch: {len(names)} electrodeNames rows vs "
            f"{coords.shape[0]} LEPTOVOX rows ({electrode_names_path})"
        )
    return names, coords


def read_fs_color_lut(path: str) -> dict[int, str]:
    """``FreeSurferColorLUT.txt`` → ``{label_id: label_name}``.

    Each non-comment, non-blank line is ``<id> <name> <r> <g> <b> <a>``; we keep
    the first two whitespace tokens. This is the SAME LUT for DK (``aparc+aseg``)
    and DKT (``aparc.DKTatlas+aseg``) — the DKT volume simply never emits the
    three DK-only cortical ids (bankssts/frontalpole/temporalpole), so one LUT
    resolves both atlases' integer labels to strings like ``ctx-lh-precentral``.
    """
    lut: dict[int, str] = {}
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            lut[int(parts[0])] = parts[1]
    return lut


def label_ids_to_names(
    origin_id: int,
    props: list[tuple[int, float]],
    lut: dict[int, str],
) -> tuple[str, list[tuple[str, float]]]:
    """Resolve a ``sphere_label_proportions`` result through the FS LUT.

    ``(origin_name, [(label_name, proportion), …])``, preserving the descending
    proportion order. An id absent from the LUT resolves to ``f"label-{id}"`` so
    the writer never crashes on an unmapped atlas value (surfaces as unmapped
    downstream rather than being silently dropped).
    """
    def name(i: int) -> str:
        return lut.get(i, f"label-{i}")

    return name(origin_id), [(name(i), p) for i, p in props]


def _signed_permutations() -> list[tuple[tuple[int, ...], tuple[bool, ...]]]:
    """All 48 signed axis permutations: 6 axis orders × 2^3 per-axis flips."""
    from itertools import permutations, product

    return [
        (perm, flips)
        for perm in permutations((0, 1, 2))
        for flips in product((False, True), repeat=3)
    ]


def apply_axis_convention(
    coords: np.ndarray,
    perm: Sequence[int],
    flips: Sequence[bool],
    shape: Sequence[int],
) -> np.ndarray:
    """Map ``(N,3)`` recon coords → array ``(i,j,k)`` under one signed permutation.

    ``ijk[:, a] = coords[:, perm[a]]``, then reflected within the volume extent
    (``shape[a]-1 - v``) where ``flips[a]``. This is the family the LEPTOVOX→array
    convention must live in for a 1 mm conformed FreeSurfer volume (the columns
    are captioned ``R A S`` but are actually a signed permutation of the CRS
    voxel index); ``find_axis_convention`` picks the member matching ground truth.
    """
    out = coords[:, list(perm)].astype(float)
    for a in range(3):
        if flips[a]:
            out[:, a] = (shape[a] - 1) - out[:, a]
    return out


def find_axis_convention(
    atlas_vol: np.ndarray,
    coords_ras: np.ndarray,
    expected_origin_names: Sequence[str],
    lut: dict[int, str],
    radius_vox: float,
) -> tuple[tuple[int, ...], tuple[bool, ...], float]:
    """Search the 48 signed permutations for the one reproducing ground truth.

    For each candidate convention, sample the origin (center-voxel) label at every
    electrode and score the fraction whose LUT name equals
    ``expected_origin_names[k]`` (the shipped D23/D24 DK CSV's far-left column).
    Returns ``(perm, flips, best_fraction)``; the caller asserts the fraction is
    ~1.0 before trusting it and reusing it on the DKT volume. Pure logic — the
    atlas array is passed in already loaded, so this is exercised on a synthetic
    volume with a planted convention and no FreeSurfer data.
    """
    if not (coords_ras.shape[0] == len(expected_origin_names)):
        raise ValueError(
            f"coords ({coords_ras.shape[0]}) vs expected names "
            f"({len(expected_origin_names)}) length mismatch"
        )
    shape = atlas_vol.shape
    best: tuple[tuple[int, ...], tuple[bool, ...], float] | None = None
    for perm, flips in _signed_permutations():
        ijk = apply_axis_convention(coords_ras, perm, flips, shape)
        hits = 0
        for k in range(ijk.shape[0]):
            origin_id, _ = sphere_label_proportions(atlas_vol, ijk[k], radius_vox)
            if lut.get(origin_id, f"label-{origin_id}") == expected_origin_names[k]:
                hits += 1
        frac = hits / ijk.shape[0] if ijk.shape[0] else 0.0
        if best is None or frac > best[2]:
            best = (perm, flips, frac)
    assert best is not None
    return best


def read_dk_csv_origins(path: str) -> list[tuple[str, str]]:
    """Shipped ``*_elec_location_radius_*mm_aparc+aseg.mgz.csv`` → ``[(name, origin)]``.

    The file has a leading ``#Info:`` comment line, then one row per electrode:
    column A is the origin (center-voxel) label, column B the electrode name, and
    columns C onward the descending ``(label, proportion)`` pairs. We return the
    ``(electrode_name, origin_label)`` pairs — the ground truth
    ``find_axis_convention`` scores against.
    """
    out: list[tuple[str, str]] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 2:
                continue
            origin, name = row[0].strip(), row[1].strip()
            out.append((name, origin))
    return out


def write_depth_wm_csv(
    path: str,
    electrode_rows: Sequence[tuple[str, str, str]],
    coords_ras: np.ndarray,
    origin_names: Sequence[str],
    majority_names: Sequence[str],
    majority_props: Sequence[float],
    *,
    label_choice: str,
    radius_mm: float,
) -> None:
    """Write the per-subject ``depth-wm.csv`` the v3 parcel_fn consumes.

    Columns: ``Electrode,DKT,DKT_origin,DKT_majority,majority_prop,R,A,S,radius_mm``.
    ``load_public_bt_anatomy`` reads only ``Electrode`` + the label column
    (``DKT``); the rest are provenance/audit and the two candidate labels.

    ``label_choice`` (``"origin"`` | ``"majority"``) selects which candidate fills
    the ``DKT`` column — REQUIRED, no default, because origin (center voxel) vs
    majority (top sphere proportion) is a Ben-gated decision that must not be made
    silently. Both candidates are always written, so the choice is auditable and
    re-derivable without re-sampling.
    """
    if label_choice not in ("origin", "majority"):
        raise ValueError(
            f"label_choice must be 'origin' or 'majority', got {label_choice!r}"
        )
    n = len(electrode_rows)
    if not (
        coords_ras.shape[0] == n == len(origin_names)
        == len(majority_names) == len(majority_props)
    ):
        raise ValueError(
            "write_depth_wm_csv: ragged inputs "
            f"(electrodes={n}, coords={coords_ras.shape[0]}, "
            f"origin={len(origin_names)}, majority={len(majority_names)}, "
            f"props={len(majority_props)})"
        )
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["Electrode", "DKT", "DKT_origin", "DKT_majority",
             "majority_prop", "R", "A", "S", "radius_mm"]
        )
        for k in range(n):
            name = electrode_rows[k][0]
            chosen = origin_names[k] if label_choice == "origin" else majority_names[k]
            r, a, s = (float(coords_ras[k, 0]), float(coords_ras[k, 1]),
                       float(coords_ras[k, 2]))
            w.writerow(
                [name, chosen, origin_names[k], majority_names[k],
                 f"{float(majority_props[k]):.4f}", f"{r:.4f}", f"{a:.4f}",
                 f"{s:.4f}", f"{float(radius_mm):.3f}"]
            )


def sphere_offsets(radius_vox: float) -> np.ndarray:
    """Integer ``(di, dj, dk)`` offsets whose Euclidean norm ≤ ``radius_vox``.

    For a 1 mm isotropic conformed volume ``radius_vox`` == radius in mm. Includes
    the origin ``(0,0,0)``. Returned sorted for determinism.
    """
    if radius_vox < 0:
        raise ValueError(f"radius_vox {radius_vox} < 0")
    r = int(np.floor(radius_vox))
    rng = range(-r, r + 1)
    offs = [
        (di, dj, dk)
        for di in rng
        for dj in rng
        for dk in rng
        if di * di + dj * dj + dk * dk <= radius_vox * radius_vox
    ]
    return np.array(sorted(offs), dtype=np.int64).reshape(-1, 3)


def sphere_label_proportions(
    atlas_vol: np.ndarray,
    center_ijk: Sequence[float],
    radius_vox: float,
) -> tuple[int, list[tuple[int, float]]]:
    """One electrode → ``(origin_label, [(label, proportion), …] desc by prop)``.

    ``atlas_vol`` is a 3-D integer label volume; ``center_ijk`` is the electrode's
    coordinate ALREADY in this array's index order (the caller resolves the
    LEPTOVOX→array convention). Voxels of the sphere that fall outside the volume
    are skipped (an edge electrode still gets its in-bounds proportions); the
    origin label is read at the rounded center, or ``0`` (``unknown``) if the
    center itself is out of bounds. Proportions are over the in-bounds voxels and
    sum to 1 (unless every voxel is out of bounds → empty list).
    """
    vol = atlas_vol
    if vol.ndim != 3:
        raise ValueError(f"atlas_vol must be 3-D, got shape {vol.shape}")
    center = np.asarray(center_ijk, dtype=float)
    if center.shape != (3,):
        raise ValueError(f"center_ijk must be length-3, got {center.shape}")

    ctr = np.rint(center).astype(np.int64)
    coords = ctr[None, :] + sphere_offsets(radius_vox)
    in_bounds = np.all((coords >= 0) & (coords < np.array(vol.shape)), axis=1)
    coords = coords[in_bounds]

    if coords.shape[0] == 0:
        return 0, []
    labels = vol[coords[:, 0], coords[:, 1], coords[:, 2]]
    uniq, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    props = sorted(
        ((int(lab), float(c) / float(total)) for lab, c in zip(uniq, counts)),
        key=lambda kv: (-kv[1], kv[0]),
    )

    if np.all((ctr >= 0) & (ctr < np.array(vol.shape))):
        origin = int(vol[ctr[0], ctr[1], ctr[2]])
    else:
        origin = 0
    return origin, props
