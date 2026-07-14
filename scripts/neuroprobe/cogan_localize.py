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


def read_postimploc_names(path: str) -> list[str]:
    """``D<id>PostimpLoc.txt`` → ``[electrode_name, …]`` in file order.

    Each row is ``<shaft> <num> <R> <A> <S> <hem> <kind>``; the electrode name is
    ``shaft+num`` (e.g. ``LAT 14`` → ``LAT14``). PostimpLoc is ROW-ALIGNED to
    ``LEPTOVOX`` (same order, identical coordinates) and names EVERY localized
    contact — including ones later dropped from the clinical ``electrodeNames``
    channel list. Used only to recover a subject whose two primary recon files
    disagree in length (see ``read_recon_electrodes_postimploc``).
    """
    names: list[str] = []
    for ln in open(path).read().splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 5:
            raise ValueError(f"{path}: PostimpLoc row not <shaft num R A S ...>: {ln!r}")
        names.append(f"{parts[0]}{parts[1]}")
    return names


def read_recon_electrodes_postimploc(
    electrode_names_path: str, leptovox_path: str, postimploc_path: str
) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    """Recover a subject whose ``electrodeNames``/``LEPTOVOX`` counts disagree.

    Some subjects (e.g. D59: 182 clinical names vs 184 localized coords) drop a
    couple of localized contacts from the EDF channel list, leaving ``LEPTOVOX``
    longer than ``electrodeNames`` so a naive row-align misassigns every coord.
    ``PostimpLoc`` is row-aligned to ``LEPTOVOX`` and names all coords, so we map
    each ``electrodeNames`` contact to its ``LEPTOVOX`` coord BY NAME and drop the
    localized-but-unrecorded extras. ``LEPTOVOX`` stays the coordinate authority
    (matching the row-aligned path used for every other subject); PostimpLoc only
    supplies the name→row index. Returns ``(electrode_rows, coords)`` like
    ``read_recon_electrodes``.
    """
    names = read_electrode_names(electrode_names_path)
    coords = read_leptovox(leptovox_path)
    pl_names = read_postimploc_names(postimploc_path)
    if len(pl_names) != coords.shape[0]:
        raise ValueError(
            f"PostimpLoc rows ({len(pl_names)}) != LEPTOVOX coords "
            f"({coords.shape[0]}) — not row-aligned, recovery unsafe."
        )
    index = {n: i for i, n in enumerate(pl_names)}
    if len(index) != len(pl_names):
        raise ValueError(f"{postimploc_path}: PostimpLoc names are not unique.")
    out_rows: list[tuple[str, str, str]] = []
    out_coords: list[np.ndarray] = []
    missing: list[str] = []
    for name, kind, hem in names:
        i = index.get(name)
        if i is None:
            missing.append(name)
            continue
        out_rows.append((name, kind, hem))
        out_coords.append(coords[i])
    if missing:
        raise ValueError(
            f"electrodeNames contacts absent from PostimpLoc: {missing[:10]}"
        )
    return out_rows, np.array(out_coords, dtype=float).reshape(-1, 3)


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
    nearest_gm_names: Sequence[str] | None = None,
    gm_reach_mm: Sequence[float] | None = None,
    n_gm_base: Sequence[int] | None = None,
    cap_mm: float | None = None,
) -> None:
    """Write the per-subject ``depth-wm.csv`` the v3 parcel_fn consumes.

    Columns: ``Electrode,DKT,DKT_nearest_gm,DKT_majority,DKT_origin,majority_prop,
    gm_reach_mm,n_gm_base,R,A,S,radius_mm,cap_mm``. ``load_public_bt_anatomy`` reads
    only ``Electrode`` + the label column (``DKT``); the rest are provenance/audit
    and the three candidate labelings, so the origin/majority/nearest-GM choice is
    fully re-derivable from one file (the counterfactual lives in the columns).

    ``label_choice`` (``"nearest_gm"`` | ``"majority"`` | ``"origin"``) selects which
    candidate fills ``DKT`` — REQUIRED, no default. ``"nearest_gm"`` is the
    physically-correct volumetric rule (:func:`nearest_gm_label`) and requires the
    ``nearest_gm_names``/``gm_reach_mm``/``n_gm_base`` columns; ``"origin"`` (center
    voxel) and ``"majority"`` (top sphere proportion incl. WM) are the naive
    volume labelings kept for the blind-vs-rule counterfactual. When the nearest-GM
    columns are absent they are written empty and ``"nearest_gm"`` is rejected.
    """
    if label_choice not in ("origin", "majority", "nearest_gm"):
        raise ValueError(
            "label_choice must be 'nearest_gm', 'majority', or 'origin', got "
            f"{label_choice!r}"
        )
    has_ng = nearest_gm_names is not None
    if has_ng and (gm_reach_mm is None or n_gm_base is None):
        raise ValueError(
            "nearest_gm_names/gm_reach_mm/n_gm_base must be provided together"
        )
    if label_choice == "nearest_gm" and not has_ng:
        raise ValueError(
            "label_choice='nearest_gm' requires nearest_gm_names/gm_reach_mm/"
            "n_gm_base"
        )
    # Narrowed non-optional views (all-or-none, guarded above).
    ng_names: Sequence[str] = nearest_gm_names if has_ng else ()
    ng_reach: Sequence[float] = gm_reach_mm if gm_reach_mm is not None else ()
    ng_nbase: Sequence[int] = n_gm_base if n_gm_base is not None else ()

    n = len(electrode_rows)
    lengths = [coords_ras.shape[0], n, len(origin_names),
               len(majority_names), len(majority_props)]
    if has_ng:
        lengths += [len(ng_names), len(ng_reach), len(ng_nbase)]
    if len(set(lengths)) != 1:
        raise ValueError(
            "write_depth_wm_csv: ragged inputs "
            f"(electrodes={n}, coords={coords_ras.shape[0]}, "
            f"origin={len(origin_names)}, majority={len(majority_names)}, "
            f"props={len(majority_props)}, "
            f"nearest={len(ng_names) if has_ng else '-'})"
        )

    def _reach(k: int) -> str:
        if not has_ng:
            return ""
        v = float(ng_reach[k])
        return "inf" if not np.isfinite(v) else f"{v:.4f}"

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["Electrode", "DKT", "DKT_nearest_gm", "DKT_majority", "DKT_origin",
             "majority_prop", "gm_reach_mm", "n_gm_base",
             "R", "A", "S", "radius_mm", "cap_mm"]
        )
        for k in range(n):
            name = electrode_rows[k][0]
            ng = ng_names[k] if has_ng else ""
            chosen = {
                "origin": origin_names[k],
                "majority": majority_names[k],
                "nearest_gm": ng,
            }[label_choice]
            r, a, s = (float(coords_ras[k, 0]), float(coords_ras[k, 1]),
                       float(coords_ras[k, 2]))
            w.writerow(
                [name, chosen, ng, majority_names[k], origin_names[k],
                 f"{float(majority_props[k]):.4f}", _reach(k),
                 str(int(ng_nbase[k])) if has_ng else "",
                 f"{r:.4f}", f"{a:.4f}", f"{s:.4f}",
                 f"{float(radius_mm):.3f}",
                 "" if cap_mm is None else f"{float(cap_mm):.3f}"]
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


# ---------------------------------------------------------------------------
# Volumetric nearest-GM parcel labeling (the physically-correct depth rule).
#
# An sEEG contact records LFP/HGA over a small volume (a few mm). The parcel it
# "belongs to" is the gray matter that dominates the tissue it actually records —
# the plurality of IN-VOCAB gray voxels inside the recording sphere, ignoring
# white matter / unknown / CSF / ventricle (all out of the parcel vocabulary). If
# no gray sits inside the base sphere, the contact is at the gray/white fringe:
# grow the sphere just far enough to reach the nearest gray voxel and vote there
# (nearest-GM), up to a physical reach cap. Beyond the cap the contact is in
# genuine deep white matter with no cortex in reach → WM sentinel (drops out of
# vocab downstream). This is strictly MORE faithful than a 2-D pial-surface
# projection (iELVis yangWang, BT's method): it respects the 3-D recording volume
# and reaches subcortical sEEG targets (hippocampus/amygdala/thalamus) a surface
# projection cannot represent — while reducing to the same nearest-GM answer in
# the near-cortical regime where BT's projection lives (BT max ShiftDist 7.5mm).
#
# The GM vocabulary MUST be the exact set the v3 parcel embedding uses, or the
# same physical location would map to a different parcel across BT and Cogan and
# break parcel identity. It is vendored here (this file runs standalone on DCC
# where the package is not importable) and pinned to the package's canonical
# tuple by ``test_vendored_dkt_vocab_matches_anatomy`` locally. Any drift fails
# that test. Mirrors ``anatomy.V14_DKT_PARCEL_LABELS`` (K=74).
# ---------------------------------------------------------------------------

_DKT_CORTICAL_BASES: tuple[str, ...] = (
    "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus", "entorhinal",
    "fusiform", "inferiorparietal", "inferiortemporal", "insula",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "paracentral", "parahippocampal",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "transversetemporal",
)
"""31 DKT cortical bases = the 34 DK aparc gyri minus DKT's 3 drops
(bankssts, frontalpole, temporalpole). Order matches ``anatomy._DK_APARC_BASE_LABELS``
with those three removed; the vocab tuple is order-sensitive downstream, so the
local parity test compares against the package, not just as a set."""

_ASEG_SUBCORTICAL_BASES: tuple[str, ...] = (
    "Hippocampus", "Amygdala", "Caudate", "Putamen", "Pallidum", "Thalamus-Proper",
)

V14_DKT_PARCEL_LABELS: tuple[str, ...] = (
    tuple(f"ctx-{h}-{b}" for h in ("lh", "rh") for b in _DKT_CORTICAL_BASES)
    + tuple(f"{p}-{b}" for p in ("Left", "Right") for b in _ASEG_SUBCORTICAL_BASES)
)
"""Vendored copy of the canonical K=74 DKT parcel vocabulary (62 cortical + 12
subcortical). Pinned to ``anatomy.V14_DKT_PARCEL_LABELS`` by a local test."""


def gm_ids_from_lut(
    lut: dict[int, str], vocab: Sequence[str] = V14_DKT_PARCEL_LABELS
) -> set[int]:
    """FS LUT ids whose names are in the parcel vocabulary (the in-vocab GM set).

    This is the ONLY place the raw atlas integers meet the parcel vocabulary:
    every id not in this set (white matter, unknown, CSF, ventricle, choroid, the
    3 DKT-dropped gyri) is treated as non-GM by ``nearest_gm_label`` and never
    fills a parcel. Passing the vocab in keeps the rule vocab-agnostic + testable.
    """
    names = set(vocab)
    return {i for i, n in lut.items() if n in names}


def nearest_gm_label(
    atlas_vol: np.ndarray,
    center_ijk: Sequence[float],
    gm_ids: set[int],
    r_base_vox: float,
    r_cap_vox: float,
) -> tuple[int, float, int]:
    """One depth contact → ``(gm_label_id, reach_vox, n_gm_base)`` by nearest-GM.

    ``atlas_vol`` is a 3-D integer label volume; ``center_ijk`` is the contact in
    this array's index order (caller resolves the LEPTOVOX→array convention).
    ``gm_ids`` is the in-vocab gray set from :func:`gm_ids_from_lut`. For a 1 mm
    conformed volume ``r_base_vox``/``r_cap_vox`` are radii in mm.

    Rule: take the plurality in-vocab gray label within the smallest sphere that
    (a) is at least ``r_base_vox`` and (b) contains ≥1 gray voxel, ignoring all
    non-gray voxels in the vote. If the nearest gray voxel is beyond ``r_cap_vox``
    (or the sphere is entirely out of bounds), the contact is deep white matter →
    return the ``0`` (``unknown``) sentinel, which maps out of vocab downstream.

    Returns:
      * ``gm_label_id`` — plurality in-vocab gray id, or ``0`` if none within cap.
      * ``reach_vox``   — distance (vox==mm) to the nearest in-vocab voxel; ``0``
        when the center voxel itself is gray (BT ``ShiftDist`` analogue). ``inf``
        if no gray within cap.
      * ``n_gm_base``   — count of in-vocab voxels inside the base sphere (``0`` if
        the contact only reached gray by growing past ``r_base_vox``).
    """
    if r_base_vox < 0 or r_cap_vox < r_base_vox:
        raise ValueError(
            f"require 0 <= r_base_vox ({r_base_vox}) <= r_cap_vox ({r_cap_vox})"
        )
    vol = atlas_vol
    if vol.ndim != 3:
        raise ValueError(f"atlas_vol must be 3-D, got shape {vol.shape}")
    center = np.asarray(center_ijk, dtype=float)
    if center.shape != (3,):
        raise ValueError(f"center_ijk must be length-3, got {center.shape}")

    ctr = np.rint(center).astype(np.int64)
    offs = sphere_offsets(r_cap_vox)
    dists = np.sqrt((offs.astype(float) ** 2).sum(axis=1))
    coords = ctr[None, :] + offs
    in_bounds = np.all((coords >= 0) & (coords < np.array(vol.shape)), axis=1)
    coords = coords[in_bounds]
    dists = dists[in_bounds]
    if coords.shape[0] == 0:
        return 0, float("inf"), 0

    labels = vol[coords[:, 0], coords[:, 1], coords[:, 2]]
    gm = (
        np.isin(labels, np.fromiter(gm_ids, dtype=np.int64))
        if gm_ids
        else np.zeros(labels.shape[0], dtype=bool)
    )
    if not gm.any():
        return 0, float("inf"), 0

    nearest = float(dists[gm].min())
    r_star = max(float(r_base_vox), nearest)
    vote = gm & (dists <= r_star + 1e-9)
    vlabels = labels[vote]
    vdists = dists[vote]
    # Plurality among in-vocab gray voxels; ties broken toward the nearer parcel,
    # then the smaller id, for determinism.
    best_id: int | None = None
    best_key: tuple[int, float, int] | None = None
    for u in np.unique(vlabels):
        m = vlabels == u
        key = (-int(m.sum()), float(vdists[m].min()), int(u))
        if best_key is None or key < best_key:
            best_key, best_id = key, int(u)
    assert best_id is not None
    n_gm_base = int((gm & (dists <= float(r_base_vox) + 1e-9)).sum())
    return best_id, nearest, n_gm_base
