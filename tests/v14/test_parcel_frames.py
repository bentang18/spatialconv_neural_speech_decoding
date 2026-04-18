"""Unit tests for `src/speech_decoding/v14/parcel_frames.py` (#10)."""

from __future__ import annotations

import numpy as np
import pytest

from speech_decoding.v14.parcel_frames import (
    ParcelFrame,
    _choose_u_axis,
    build_parcel_frame,
    compute_vertex_normals,
)


def _flat_square_patch() -> tuple[np.ndarray, np.ndarray]:
    """Return a 3x3 flat mesh on z = 0 with z-aligned outward normals.

    9 vertices on the integer grid (x ∈ {-1,0,1}, y ∈ {-1,0,1}), z = 0.
    Eight triangles tile the square. All face normals point at +z by
    right-hand rule given the CCW winding below.
    """

    verts = np.array(
        [
            [-1.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [3, 4, 7],
            [3, 7, 6],
            [4, 5, 8],
            [4, 8, 7],
        ],
        dtype=np.int64,
    )
    return verts, faces


class TestComputeVertexNormals:
    def test_flat_patch_normals_are_plus_z(self) -> None:
        verts, faces = _flat_square_patch()
        normals = compute_vertex_normals(verts, faces)
        assert normals.shape == verts.shape
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
        assert np.allclose(normals[:, 2], 1.0, atol=1e-12)
        assert np.allclose(normals[:, :2], 0.0, atol=1e-12)


class TestChooseUAxis:
    def test_anterior_projection_preferred(self) -> None:
        z = np.array([0.0, 0.0, 1.0])
        u = _choose_u_axis(z)
        assert np.allclose(u, np.array([0.0, 1.0, 0.0]))  # +A

    def test_degenerate_anterior_falls_back_to_superior(self) -> None:
        z = np.array([0.0, 1.0, 0.0])  # z aligned with anterior -> u can't be anterior
        u = _choose_u_axis(z)
        assert np.allclose(u, np.array([0.0, 0.0, 1.0]))  # +S

    def test_returned_u_is_tangent_to_z(self) -> None:
        z = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        u = _choose_u_axis(z)
        assert abs(float(np.dot(u, z))) < 1e-12
        assert abs(np.linalg.norm(u) - 1.0) < 1e-12


class TestBuildParcelFrame:
    def test_flat_patch_frame_has_expected_axes(self) -> None:
        verts, faces = _flat_square_patch()
        weights = np.zeros(verts.shape[0], dtype=np.float64)
        # Support only the inner 3x3 block (all 9 here) with uniform weight 1.
        weights[:] = 1.0
        frame = build_parcel_frame(
            parcel_name="TEST_FLAT",
            parcel_idx_1based=1,
            hemisphere="L",
            verts=verts,
            faces=faces,
            weights=weights,
        )
        # z axis should be +z.
        z_axis = frame.basis[:, 2]
        assert np.allclose(z_axis, np.array([0.0, 0.0, 1.0]), atol=1e-9)
        # u axis should be the anterior direction tangent to z, i.e. +y.
        u_axis = frame.basis[:, 0]
        assert np.allclose(u_axis, np.array([0.0, 1.0, 0.0]), atol=1e-9)
        # v = normalize(z x u).
        v_axis = frame.basis[:, 1]
        assert np.allclose(v_axis, np.cross(z_axis, u_axis), atol=1e-9)

    def test_basis_is_orthonormal_and_right_handed(self) -> None:
        verts, faces = _flat_square_patch()
        weights = np.ones(verts.shape[0], dtype=np.float64)
        frame = build_parcel_frame(
            parcel_name="TEST_ORTHO",
            parcel_idx_1based=1,
            hemisphere="L",
            verts=verts,
            faces=faces,
            weights=weights,
        )
        basis = frame.basis
        assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-9)
        # Right-handed: det(basis) = +1.
        assert np.isclose(float(np.linalg.det(basis)), 1.0, atol=1e-9)

    def test_origin_snaps_to_in_parcel_vertex(self) -> None:
        verts, faces = _flat_square_patch()
        # Weight only the (0, 0, 0) center vertex: weighted centroid == that vertex.
        weights = np.zeros(verts.shape[0], dtype=np.float64)
        weights[4] = 1.0  # center index per _flat_square_patch() layout
        # The function requires >= 3 support vertices, so add two more with tiny weight.
        weights[0] = 1e-3
        weights[8] = 1e-3
        frame = build_parcel_frame(
            parcel_name="TEST_ORIGIN",
            parcel_idx_1based=1,
            hemisphere="L",
            verts=verts,
            faces=faces,
            weights=weights,
        )
        # Center vertex dominates the weighted centroid, so the origin must snap there.
        assert frame.origin_vertex == 4
        assert np.allclose(frame.origin_xyz, verts[4], atol=1e-12)

    def test_weights_shape_mismatch_raises(self) -> None:
        verts, faces = _flat_square_patch()
        with pytest.raises(ValueError, match="weights shape"):
            build_parcel_frame(
                parcel_name="BAD",
                parcel_idx_1based=1,
                hemisphere="L",
                verts=verts,
                faces=faces,
                weights=np.ones(verts.shape[0] - 1, dtype=np.float64),
            )

    def test_too_few_support_vertices_raises(self) -> None:
        verts, faces = _flat_square_patch()
        weights = np.zeros(verts.shape[0], dtype=np.float64)
        weights[0] = 1.0
        weights[1] = 1.0
        # Only 2 support vertices; #10 contract requires >= 3.
        with pytest.raises(ValueError, match="support vertices"):
            build_parcel_frame(
                parcel_name="BAD",
                parcel_idx_1based=1,
                hemisphere="L",
                verts=verts,
                faces=faces,
                weights=weights,
            )

    def test_sigma_u_sigma_v_are_weighted_std_of_local_coords(self) -> None:
        verts, faces = _flat_square_patch()
        weights = np.ones(verts.shape[0], dtype=np.float64)
        frame = build_parcel_frame(
            parcel_name="TEST_SIGMA",
            parcel_idx_1based=1,
            hemisphere="L",
            verts=verts,
            faces=faces,
            weights=weights,
        )
        # For a unit-weighted 3x3 grid on z=0 centered at origin, u = +y and v = -x,
        # so sigma_u is the weighted std of y, sigma_v is weighted std of x.
        # y and x take values {-1, 0, +1} three times each => weighted std = sqrt(2/3).
        expected = float(np.sqrt(2.0 / 3.0))
        assert np.isclose(frame.sigma_u, expected, atol=1e-9)
        assert np.isclose(frame.sigma_v, expected, atol=1e-9)

    def test_returned_dataclass_contract(self) -> None:
        verts, faces = _flat_square_patch()
        weights = np.ones(verts.shape[0], dtype=np.float64)
        frame = build_parcel_frame(
            parcel_name="TEST_CONTRACT",
            parcel_idx_1based=7,
            hemisphere="L",
            verts=verts,
            faces=faces,
            weights=weights,
        )
        assert isinstance(frame, ParcelFrame)
        assert frame.parcel_name == "TEST_CONTRACT"
        assert frame.parcel_idx_1based == 7
        assert frame.hemisphere == "L"
        assert frame.origin_xyz.shape == (3,)
        assert frame.basis.shape == (3, 3)
        assert frame.sigma_u > 0.0
        assert frame.sigma_v > 0.0
