from __future__ import annotations

import numpy as np

from speech_decoding.v14.cvsavg_projection import bridge_single_hemisphere


def test_bridge_single_hemisphere_nearest_outer_then_pial() -> None:
    outer = np.array(
        [
            [0.0, 0.0, 2.0],
            [10.0, 0.0, 2.0],
            [0.0, 10.0, 2.0],
        ],
        dtype=np.float64,
    )
    pial = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [50.0, 50.0, 0.0],
        ],
        dtype=np.float64,
    )
    points_outer = np.array(
        [
            [0.2, 0.0, 2.0],
            [9.7, 0.1, 2.0],
            [0.0, 9.6, 2.0],
        ],
        dtype=np.float64,
    )

    outer_vids, pial_vids, pial_xyz, outer_snap_mm, outer_to_pial_mm = bridge_single_hemisphere(
        points_outer,
        outer,
        pial,
    )

    np.testing.assert_array_equal(outer_vids, [0, 1, 2])
    np.testing.assert_array_equal(pial_vids, [0, 1, 2])
    np.testing.assert_allclose(pial_xyz, pial[:3])
    np.testing.assert_allclose(outer_to_pial_mm, [2.0, 2.0, 2.0])
    assert np.all(outer_snap_mm < 0.5)
