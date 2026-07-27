"""_shift_and_trim: the non-parity window used by the visualization encode.

Dropping edge clips renumbers the union axis. The hazard is silent: a split index left
pointing at its OLD position would select a different clip with no error anywhere, so the
remap is what these tests pin down.
"""
from __future__ import annotations

import numpy as np

from scripts.neuroprobe.v3_probe_encode_r4 import _shift_and_trim
from speech_decoding.experiments.pretrain_probe_labels import SessionTargets


def _targets(n: int = 6) -> SessionTargets:
    starts = np.arange(n, dtype=float)  # 0,1,...,5 s
    return SessionTargets(
        subject_id=1,
        trial_id=0,
        clip_starts=starts,
        clip_durations=np.full(n, 1.0),
        clip_movie_onsets=starts + 100.0,
        labels={"onset": np.arange(n, dtype=float)},
        ws_split={"onset": {0: {"train": np.array([0, 1, 4, 5]),
                                "val": np.array([2]),
                                "test": np.array([3])}}},
        cs_split={"onset": {"val": np.array([0, 2]), "test": np.array([3, 5])}},
    )


def test_shift_and_trim_drops_both_edges() -> None:
    # 2 s clip (64 frames @ 32 Hz), -0.5 s offset, session 192 frames long.
    # shifted starts -0.5,0.5,1.5,2.5,3.5,4.5 -> t0 -16,16,48,80,112,144
    # valid needs t0 >= 0 and t0+64 <= 192, so rows 1..4 survive.
    out = _shift_and_trim(_targets(), offset_s=-0.5, clip_frames=64, n_frames=192)
    np.testing.assert_allclose(out.clip_starts, [0.5, 1.5, 2.5, 3.5])
    np.testing.assert_allclose(out.labels["onset"], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(out.clip_movie_onsets, [101.0, 102.0, 103.0, 104.0])


def test_shift_and_trim_remaps_split_indices_to_new_axis() -> None:
    out = _shift_and_trim(_targets(), offset_s=-0.5, clip_frames=64, n_frames=192)
    # survivors are old rows 1,2,3,4 -> new 0,1,2,3. Old 0 and 5 are gone, not renumbered.
    np.testing.assert_array_equal(out.ws_split["onset"][0]["train"], [0, 3])  # old 1,4
    np.testing.assert_array_equal(out.ws_split["onset"][0]["val"], [1])       # old 2
    np.testing.assert_array_equal(out.ws_split["onset"][0]["test"], [2])      # old 3
    np.testing.assert_array_equal(out.cs_split["onset"]["val"], [1])          # old 2 (0 dropped)
    np.testing.assert_array_equal(out.cs_split["onset"]["test"], [2])         # old 3 (5 dropped)


def test_shift_and_trim_split_indices_still_address_the_right_clip() -> None:
    """The invariant that matters: an index must select the same clip it did before."""
    t = _targets()
    out = _shift_and_trim(t, offset_s=-0.5, clip_frames=64, n_frames=192)
    for name in ("train", "val", "test"):
        old = t.ws_split["onset"][0][name]
        new = out.ws_split["onset"][0][name]
        # labels are the row's original identity, so surviving rows must carry them through
        kept = [t.labels["onset"][i] for i in old if t.clip_starts[i] + -0.5 in out.clip_starts]
        np.testing.assert_allclose(out.labels["onset"][new], kept)


def test_shift_and_trim_is_a_noop_when_window_fits() -> None:
    out = _shift_and_trim(_targets(), offset_s=0.0, clip_frames=32, n_frames=1000)
    np.testing.assert_allclose(out.clip_starts, np.arange(6, dtype=float))
    np.testing.assert_array_equal(out.ws_split["onset"][0]["train"], [0, 1, 4, 5])
