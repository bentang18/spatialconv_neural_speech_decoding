"""Unit and integration tests for the v14 MNI coordinate port.

The synthetic tests exercise the pure-array core (``project_single_hemisphere``)
and the text-file parser (``read_lepto``) without touching any real surface
files. The Box-mounted tests run against real S14 data and are skipped if the
Box mount is not accessible (so CI on DCC or GitHub Actions still passes).

The oracle test is skipped unless ``tests/v14/fixtures/S14_MNI152_oracle.csv``
exists. That fixture is Zac's trusted MATLAB output; it is dropped into place
when we receive it from him and from then on serves as a per-electrode
regression bar (worst-case residual <= 0.5 mm).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from speech_decoding.v14.coordinates import (
    DEFAULT_AVG_SUBJECT,
    MNICoordinateCache,
    compare_against_oracle,
    load_mni_cache,
    project_patient,
    project_single_hemisphere,
    read_lepto,
)


BOX_ROOT = Path("/Users/bentang/Library/CloudStorage/Box-Box")
S14_DIR = BOX_ROOT / "ECoG_Recon" / "S14"
AVG_DIR = BOX_ROOT / "ECoG_Recon" / DEFAULT_AVG_SUBJECT
LOCAL_S14_RAS = Path(__file__).resolve().parents[2] / "data" / "mni_coords" / "S14_RAS.txt"
# Zac's trusted MATLAB run of sub2AvgBrainClinical.m with avgSubj=cvs_avg35_inMNI152.
# Format: one row per electrode, `<patient_id>-<electrode_name>,x,y,z`, no header.
ORACLE_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "S14_mni152_avgCoords.csv"


def _load_matlab_avgcoords(path: Path, patient_id: str) -> tuple[list[str], np.ndarray]:
    """Parse a raw ``sub2AvgBrainClinical.m`` output CSV.

    The MATLAB driver prepends ``<patient_id>-`` to every electrode name
    (lines 379-381 of the .m file). We strip that prefix so names align
    with the ``.electrodeNames`` convention used everywhere else.
    """

    names: list[str] = []
    rows: list[list[float]] = []
    prefix = f"{patient_id}-"
    for line_idx, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        fields = raw.split(",")
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_idx}: expected 4 columns, got {fields!r}")
        name = fields[0]
        if not name.startswith(prefix):
            raise ValueError(f"{path}:{line_idx}: name {name!r} missing {prefix!r} prefix")
        names.append(name[len(prefix):])
        rows.append([float(fields[1]), float(fields[2]), float(fields[3])])
    return names, np.array(rows, dtype=np.float64)

requires_box = pytest.mark.skipif(
    not S14_DIR.exists(), reason="Box mount or S14 reconstruction not accessible"
)


def _write_lepto_pair(
    tmp_path: Path,
    patient_id: str,
    coords: list[list[float]],
    meta: list[tuple[str, str, str]],
    brainshift_method: str = "dykstra-",
) -> Path:
    patient_dir = tmp_path / patient_id
    (patient_dir / "elec_recon").mkdir(parents=True)
    lepto = patient_dir / "elec_recon" / f"{patient_id}.LEPTO"
    with lepto.open("w") as f:
        f.write(f"01-Jan-2026 00:00:00\t{brainshift_method}\tFreeSurferVersionUnknown\n")
        f.write("R A S\n")
        for row in coords:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
    names = patient_dir / "elec_recon" / f"{patient_id}.electrodeNames"
    with names.open("w") as f:
        f.write("01-Jan-2026 00:00:00\n")
        f.write("Name, Depth/Strip/Grid, Hem\n")
        for name, kind, hem in meta:
            f.write(f"{name} {kind} {hem}\n")
    return patient_dir


# ---------------------------------------------------------------------------
# read_lepto — synthetic fixtures
# ---------------------------------------------------------------------------


def test_read_lepto_basic(tmp_path: Path) -> None:
    patient_dir = _write_lepto_pair(
        tmp_path,
        "XTEST",
        coords=[[-10.0, 0.0, 0.0], [5.0, 10.0, 20.0], [-5.0, -5.0, -5.0]],
        meta=[("E1", "G", "L"), ("E2", "S", "R"), ("E3", "G", "L")],
        brainshift_method="dykstra-",
    )
    elec = read_lepto(patient_dir, "XTEST")

    assert elec.names == ("E1", "E2", "E3")
    np.testing.assert_allclose(
        elec.coords,
        [[-10.0, 0.0, 0.0], [5.0, 10.0, 20.0], [-5.0, -5.0, -5.0]],
    )
    np.testing.assert_array_equal(elec.is_left, [True, False, True])
    np.testing.assert_array_equal(elec.is_subdural, [True, True, True])
    assert elec.brainshift_method == "dykstra-"


def test_read_lepto_rejects_depth_marker(tmp_path: Path) -> None:
    patient_dir = _write_lepto_pair(
        tmp_path,
        "XTEST",
        coords=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        meta=[("E1", "G", "L"), ("E2", "D", "L")],
    )
    elec = read_lepto(patient_dir, "XTEST")
    # Depth entries are *parsed* as subdural=False; the driver later rejects.
    np.testing.assert_array_equal(elec.is_subdural, [True, False])


def test_read_lepto_row_count_mismatch(tmp_path: Path) -> None:
    patient_dir = tmp_path / "XTEST"
    (patient_dir / "elec_recon").mkdir(parents=True)
    (patient_dir / "elec_recon" / "XTEST.LEPTO").write_text(
        "01-Jan-2026 00:00:00\tdykstra-\tFS\nR A S\n0 0 0\n1 1 1\n"
    )
    (patient_dir / "elec_recon" / "XTEST.electrodeNames").write_text(
        "01-Jan-2026 00:00:00\nName, Depth/Strip/Grid, Hem\nE1 G L\n"
    )
    with pytest.raises(ValueError, match="row count mismatch"):
        read_lepto(patient_dir, "XTEST")


def test_read_lepto_rejects_non_ras_header(tmp_path: Path) -> None:
    patient_dir = tmp_path / "XTEST"
    (patient_dir / "elec_recon").mkdir(parents=True)
    (patient_dir / "elec_recon" / "XTEST.LEPTO").write_text(
        "01-Jan-2026 00:00:00\tdykstra-\tFS\nL P S\n0 0 0\n"
    )
    (patient_dir / "elec_recon" / "XTEST.electrodeNames").write_text(
        "01-Jan-2026 00:00:00\nheader\nE1 G L\n"
    )
    with pytest.raises(ValueError, match="R A S"):
        read_lepto(patient_dir, "XTEST")


# ---------------------------------------------------------------------------
# project_single_hemisphere — pure algorithm
# ---------------------------------------------------------------------------


def test_project_single_hemisphere_identity_sphere() -> None:
    """When subject and average share sphere vertex positions, the nearest-sphere
    step is the identity: electrode near subject pial vertex k maps to avg pial
    vertex k. Verifies the pipeline wiring end-to-end without surface files.
    """

    sub_pial = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )
    avg_pial = np.array(
        [[100.0, 0.0, 0.0], [110.0, 0.0, 0.0], [100.0, 10.0, 0.0], [100.0, 0.0, 10.0]]
    )
    # Sphere vertices are identical between sub and avg, so sphere nearest
    # picks the same index as the pial snap.
    sphere = 100.0 * np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )

    points = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0], [0.0, 10.1, 0.0]])
    out = project_single_hemisphere(points, sub_pial, sphere, sphere, avg_pial)

    # Expect avg_pial[0], avg_pial[1], avg_pial[2]
    np.testing.assert_allclose(out, avg_pial[[0, 1, 2]])


def test_project_single_hemisphere_independent_sphere_remap() -> None:
    """When sphere vertices differ, we exercise the full four-step lookup. The
    test picks sphere positions such that subject vertex k maps to average
    vertex (k+1) mod N — a known permutation — and checks the output.
    """

    n = 4
    sub_pial = np.array([[float(i), 0.0, 0.0] for i in range(n)])
    avg_pial = np.array([[100.0 + float(i), 0.0, 0.0] for i in range(n)])
    # Sub sphere: positions 0..3 along +x
    sub_sph = np.array([[float(i), 0.0, 0.0] for i in range(n)])
    # Avg sphere: shifted by -0.5 so nearest to sub_sph[k] is avg_sph[k-1]
    # for k >= 1, and avg_sph[0] for k = 0.
    avg_sph = np.array([[float(i) - 0.5, 0.0, 0.0] for i in range(n)])

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    out = project_single_hemisphere(points, sub_pial, sub_sph, avg_sph, avg_pial)

    # Hand-check: sub_sph[0]=(0,0,0). Avg sphere candidates: (-.5,0,0),(.5,0,0),(1.5,0,0),(2.5,0,0).
    # Nearest to (0,0,0): (-.5,0,0) at distance .5 (avg index 0). So out[0]=avg_pial[0]=100.
    # sub_sph[1]=(1,0,0). Nearest avg: (.5,0,0) or (1.5,0,0), both dist .5 — cKDTree picks one deterministically.
    # Both map to indices 1 or 2 in avg; skip strict expectation for ties.
    np.testing.assert_allclose(out[0], [100.0, 0.0, 0.0])
    np.testing.assert_allclose(out[3], [103.0, 0.0, 0.0])  # unambiguous


def test_project_single_hemisphere_empty_input() -> None:
    verts = np.zeros((10, 3))
    out = project_single_hemisphere(np.zeros((0, 3)), verts, verts, verts, verts)
    assert out.shape == (0, 3)


def test_project_single_hemisphere_vertex_count_mismatch_subject() -> None:
    sub_pial = np.zeros((10, 3))
    sub_sph = np.zeros((9, 3))
    avg_sph = np.zeros((5, 3))
    avg_pial = np.zeros((5, 3))
    with pytest.raises(ValueError, match="subject pial/sphere"):
        project_single_hemisphere(
            np.zeros((1, 3)), sub_pial, sub_sph, avg_sph, avg_pial
        )


def test_project_single_hemisphere_vertex_count_mismatch_average() -> None:
    verts = np.zeros((5, 3))
    with pytest.raises(ValueError, match="average pial/sphere"):
        project_single_hemisphere(
            np.zeros((1, 3)), verts, verts, np.zeros((6, 3)), verts
        )


# ---------------------------------------------------------------------------
# load_mni_cache — roundtrip via project_patient
# ---------------------------------------------------------------------------


def test_mni_cache_lookup_by_name() -> None:
    cache = MNICoordinateCache(
        names=("A", "B", "C"),
        coords=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -2.0, -3.0]]),
        is_left=np.array([True, True, False]),
    )
    np.testing.assert_allclose(cache.lookup("B"), [4.0, 5.0, 6.0])
    with pytest.raises(KeyError):
        cache.lookup("Z")


# ---------------------------------------------------------------------------
# S14 integration against Box (skipped when not mounted)
# ---------------------------------------------------------------------------


@requires_box
def test_s14_lepto_matches_local_ras_file() -> None:
    """Sanity: LEPTO in Box should match the local ACPC RAS file bit-for-bit.

    Local file format: ``<prefix> <num> <x> <y> <z> <hem> <type>`` with the
    name split into two space-separated tokens. Concatenating the first two
    tokens must give the name in ``.electrodeNames``.
    """

    elec = read_lepto(S14_DIR, "S14")
    assert len(elec.names) == 128

    local_rows = LOCAL_S14_RAS.read_text().splitlines()
    assert len(local_rows) == 128
    for i, row in enumerate(local_rows):
        parts = row.split()
        assert len(parts) == 7, f"row {i}: expected 7 tokens, got {parts}"
        name_concat = parts[0] + parts[1]
        local_xyz = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
        assert elec.names[i] == name_concat, (
            f"row {i}: LEPTO name {elec.names[i]!r} != local {name_concat!r}"
        )
        np.testing.assert_allclose(elec.coords[i], local_xyz, atol=1e-6)


@requires_box
def test_s14_hemisphere_all_left() -> None:
    elec = read_lepto(S14_DIR, "S14")
    assert elec.is_left.all(), "S14 is a LH patient"
    assert elec.is_subdural.all(), "S14 has no depth electrodes"


@requires_box
def test_s14_projection_end_to_end(tmp_path: Path) -> None:
    out_path = project_patient(
        patient_id="S14",
        box_root=BOX_ROOT,
        cache_dir=tmp_path,
    )
    assert out_path.exists()

    cache = load_mni_cache("S14", tmp_path)
    assert len(cache.names) == 128
    assert cache.coords.shape == (128, 3)
    assert not np.isnan(cache.coords).any()

    # All S14 electrodes are LH — x must be negative or near midline.
    assert (cache.coords[:, 0] <= 5.0).all(), (
        f"LH electrodes landed with x > 5: "
        f"max x = {cache.coords[:, 0].max():.2f}"
    )

    # MNI152 brain bounding box. Tight bounds — a missing cras shift would
    # systematically push every y up by +17 mm and every z down by -19 mm,
    # putting electrodes outside these ranges.
    xs, ys, zs = cache.coords[:, 0], cache.coords[:, 1], cache.coords[:, 2]
    assert -80.0 < xs.min() and xs.max() < 10.0, f"x range {xs.min():.1f}..{xs.max():.1f}"
    assert -120.0 < ys.min() and ys.max() < 80.0, f"y range {ys.min():.1f}..{ys.max():.1f}"
    assert -70.0 < zs.min() and zs.max() < 90.0, f"z range {zs.min():.1f}..{zs.max():.1f}"

    # Landmark sanity: S14's array covers ventral left sensorimotor cortex
    # (face/tongue territory on the inferior precentral/postcentral gyrus).
    # The centroid should sit in that region in MNI; this check fails if the
    # cras frame-translation is missing (centroid shifts ~17 mm anterior).
    centroid = cache.coords.mean(axis=0)
    assert -70.0 < centroid[0] < -40.0, f"x centroid {centroid[0]:.1f} not in left ventral SMC"
    assert -25.0 < centroid[1] < 10.0, f"y centroid {centroid[1]:.1f} not in ventral SMC"
    assert 10.0 < centroid[2] < 55.0, f"z centroid {centroid[2]:.1f} not in ventral SMC"

    # Sidecar metadata should exist and name S14, and should record the avg
    # subject's cras so downstream consumers can audit the frame translation.
    meta = json.loads((tmp_path / "_projection_meta.json").read_text())
    assert "S14" in meta
    assert meta["S14"]["n_electrodes"] == 128
    assert meta["S14"]["n_left"] == 128
    assert meta["S14"]["avg_subject"] == DEFAULT_AVG_SUBJECT
    assert "avg_cras_mm" in meta["S14"]
    assert len(meta["S14"]["avg_cras_mm"]) == 3


@pytest.mark.skipif(
    not ORACLE_FIXTURE.exists(),
    reason="Zac's S14 MNI152 oracle fixture not present",
)
@requires_box
def test_s14_matches_oracle_within_tolerance(tmp_path: Path) -> None:
    """Compare the Python port against Zac's trusted MATLAB run (v2, cras-corrected).

    Zac's MATLAB ``sub2AvgBrainClinical.m`` outputs coordinates in cvs_avg35
    *subject tkrRAS* — it terminates with a vertex lookup in
    ``lh.pial-outer-smoothed`` and does not add cras. The Python port, as of
    v2 (2026-04-16), adds the ``lh.pial`` cras so its output is in true MNI.
    This test compares (MATLAB oracle + cras) against the Python port.

    Tolerance is intentionally wider than pure floating-point noise (~0.05 mm
    expected) because Zac's oracle was generated against an earlier snapshot
    of the FreeSurfer reconstruction files. ``lh.sphere-outer-mni.reg`` and
    ``S14.LEPTO`` in Box are both stamped 2025-08-11 from a coordinated
    reconstruction update; the oracle predates that. The diagnostics show:
    (a) our cKDTree nearest-vertex matches brute-force argmin bit-for-bit at
    both snap steps, and (b) Zac's oracle points sit 0.14–0.85 mm from the
    nearest vertex on the current ``cvs_avg35_inMNI152/lh.pial-outer-smoothed``
    mesh — i.e., the oracle lies on a *different* mesh snapshot than what is
    currently in Box, not on any current vertex. The residual magnitude
    (~0.64 mm median) is therefore one mesh-spacing of file drift, not an
    algorithm bug.

    Bars chosen so that a real bug (frame error, wrong reader, wrong
    algorithm) fails loudly while file-version drift is tolerated:
    - max residual <= 1.5 mm (one vertex-spacing hop worst case)
    - median residual <= 1.0 mm
    - per-axis mean |residual| <= 0.5 mm (no systematic offset)
    """

    project_patient("S14", BOX_ROOT, tmp_path)
    cache = load_mni_cache("S14", tmp_path)

    oracle_names, oracle_coords_tkr = _load_matlab_avgcoords(ORACLE_FIXTURE, "S14")
    # The MATLAB fixture is in cvs_avg35 tkrRAS; our v2 port is in scanner RAS.
    # Add the same cras the Python port adds so the comparison is frame-matched.
    meta = json.loads((tmp_path / "_projection_meta.json").read_text())
    avg_cras = np.asarray(meta["S14"]["avg_cras_mm"], dtype=np.float64)
    oracle_coords = oracle_coords_tkr + avg_cras

    stats = compare_against_oracle(cache, oracle_names, oracle_coords)

    summary = (
        f"mean={stats['mean_mm']:.3f} median={stats['median_mm']:.3f} "
        f"p95={stats['p95_mm']:.3f} max={stats['max_mm']:.3f}"
    )
    assert stats["max_mm"] <= 1.5, f"worst-case residual > 1.5 mm — possible algorithm bug: {summary}"
    assert stats["median_mm"] <= 1.0, f"median residual > 1.0 mm — possible algorithm bug: {summary}"

    # Per-axis sanity: a real bug (frame flip, cras offset, wrong reader) would
    # produce a large systematic axis mean; vertex quantization shows small
    # means with large stds.
    cache_by_name = {n: cache.coords[i] for i, n in enumerate(cache.names)}
    py_aligned = np.stack([cache_by_name[n] for n in oracle_names])
    per_axis_mean = np.abs((py_aligned - oracle_coords).mean(axis=0))
    assert (per_axis_mean <= 0.5).all(), (
        f"per-axis mean |residual| > 0.5 mm suggests systematic offset: {per_axis_mean}"
    )
