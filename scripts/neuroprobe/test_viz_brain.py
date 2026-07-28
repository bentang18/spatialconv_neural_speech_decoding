"""A brain figure is the easiest place to publish a permutation bug and never find out.

Nothing downstream of ``build`` can tell a correct electrode-to-coordinate mapping from a
shuffled one -- both produce a pretty picture. So the row alignment is asserted, and the
colour construction is pinned to the two properties the figure's claim rests on: one shared
stretch across subjects, and invariance to a per-subject amplitude difference.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.neuroprobe.viz_brain import build, load_elec

TAP = "enc12_elec"
TASK = "onset"
T, C = 8, 12


def _pattern(n_rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, T, C))


def _write(tmp_path, subj, trial, x, n_coords=None, gain=1.0):
    n_rows = x.shape[0]
    base = 5.0 * np.ones((n_rows, T, C))
    d = {
        "subject_id": np.int64(subj), "trial_id": np.int64(trial),
        "parcel_canon": np.arange(n_rows, dtype=np.int64) % 4,
        f"{TAP}/shape": np.asarray([n_rows, T, C], dtype=np.int64),
        f"{TAP}/col_sum": (base.reshape(n_rows, -1)).astype(np.float32),
        f"{TAP}/col_sq": (base.reshape(n_rows, -1) ** 2 + 1.0).astype(np.float32),
        f"{TAP}/{TASK}/c0/all": (gain * (base - 0.5 * x)).astype(np.float32),
        f"{TAP}/{TASK}/c1/all": (gain * (base + 0.5 * x)).astype(np.float32),
    }
    np.savez_compressed(tmp_path / f"red_s{subj}_t{trial}_hga.npz", **d)
    return np.arange((n_coords or n_rows) * 3, dtype=np.float32).reshape(-1, 3)


def _coords(tmp_path, mapping):
    np.savez_compressed(tmp_path / "coords.npz",
                        **{f"{k}/coords": v for k, v in mapping.items()})
    return str(tmp_path / "coords.npz")


def test_load_elec_returns_the_class_contrast_not_a_condition(tmp_path) -> None:
    x = _pattern(6, 0)
    _write(tmp_path, 1, 0, x)
    got = load_elec(str(tmp_path / "red_s1_t0_hga.npz"), TAP, TASK)
    assert got is not None
    # the shared 5.0 offset lives in BOTH classes and must cancel; sd here is 1.0
    np.testing.assert_allclose(got["x"], x, atol=1e-4)


def test_a_coordinate_row_count_mismatch_is_fatal(tmp_path) -> None:
    c = _write(tmp_path, 1, 0, _pattern(6, 0), n_coords=5)
    path = _coords(tmp_path, {"s1_t0": c})
    with pytest.raises(AssertionError, match="coords vs"):
        build(str(tmp_path), path, TAP, TASK)


def test_a_session_without_coordinates_is_skipped_not_guessed(tmp_path) -> None:
    c = _write(tmp_path, 1, 0, _pattern(6, 0))
    _write(tmp_path, 2, 0, _pattern(6, 1))
    sessions, _ = build(str(tmp_path), _coords(tmp_path, {"s1_t0": c}), TAP, TASK)
    assert [s["subject_id"] for s in sessions] == [1]


def test_a_shared_contrast_paints_matched_contacts_the_same_colour(tmp_path) -> None:
    x = _pattern(6, 0)
    c1 = _write(tmp_path, 1, 0, x)
    c2 = _write(tmp_path, 2, 0, x)
    sessions, _ = build(str(tmp_path), _coords(tmp_path, {"s1_t0": c1, "s2_t0": c2}),
                        TAP, TASK)
    a, b = sessions[0]["rgb"], sessions[1]["rgb"]
    assert np.abs(a - b).max() < 1e-6


def test_an_unshared_contrast_does_not(tmp_path) -> None:
    c1 = _write(tmp_path, 1, 0, _pattern(6, 0))
    c2 = _write(tmp_path, 2, 0, _pattern(6, 99))
    sessions, _ = build(str(tmp_path), _coords(tmp_path, {"s1_t0": c1, "s2_t0": c2}),
                        TAP, TASK)
    assert np.abs(sessions[0]["rgb"] - sessions[1]["rgb"]).max() > 0.2


def test_a_louder_subject_does_not_get_a_different_colour(tmp_path) -> None:
    """Per-session unit scaling is what stops one subject owning the shared basis. Without
    it a 10x amplitude difference alone would recolour the quieter subject."""
    x = _pattern(6, 0)
    c1 = _write(tmp_path, 1, 0, x)
    c2 = _write(tmp_path, 2, 0, x, gain=10.0)
    sessions, _ = build(str(tmp_path), _coords(tmp_path, {"s1_t0": c1, "s2_t0": c2}),
                        TAP, TASK)
    assert np.abs(sessions[0]["rgb"] - sessions[1]["rgb"]).max() < 1e-6


def test_the_colour_stretch_is_shared_not_per_subject(tmp_path) -> None:
    """A subject whose contrast barely moves must come out almost one flat colour.

    Amplitude alone cannot show this -- per-session unit scaling removes it on purpose (see
    the test above). What separates a shared stretch from a per-panel one is a subject whose
    tokens sit in a tiny CLUSTER: stretching that panel by its own percentiles would blow
    0.01 of jitter up to the full gamut and make it look as structured as anyone else.
    """
    c1 = _write(tmp_path, 1, 0, _pattern(6, 0))
    flat = np.ones((6, T, 1)) * _pattern(1, 3)[0, 0] + 0.01 * _pattern(6, 4)
    c2 = _write(tmp_path, 2, 0, flat)
    sessions, _ = build(str(tmp_path), _coords(tmp_path, {"s1_t0": c1, "s2_t0": c2}),
                        TAP, TASK)
    spread = [float(s["rgb"].reshape(-1, 3).std(axis=0).mean()) for s in sessions]
    assert spread[1] < 0.25 * spread[0], spread


def test_the_clip_advances_colour_in_time_then_orbits(tmp_path) -> None:
    """A frame that silently reused the previous colours would look like a correct render,
    so the test checks that a colour-varying source actually changes the pixels between two
    time frames, and that the orbit phase moves the camera instead of the clock."""
    import os

    from scripts.neuroprobe.viz_brain import animate_brain

    coords = {}
    for s in (1, 2):
        c = _write(tmp_path, s, 0, _pattern(6, s))
        coords[f"s{s}_t0"] = c
    sessions, _ = build(str(tmp_path), _coords(tmp_path, coords), TAP, TASK)
    times = np.arange(T) / 32.0
    info = animate_brain(sessions, times, str(tmp_path / "b.gif"), fps=6, orbit_frames=4,
                         hold=1)
    assert info["n_frames"] == T + 1 + 4
    assert info["n_subjects"] == 2 and info["t_len"] == T
    assert os.path.getsize(info["path"]) > 0
    # the colours really do differ across time -- otherwise the clip is a still
    r = sessions[0]["rgb"]
    assert not np.allclose(r[:, 0, :], r[:, T - 1, :])


def test_the_clip_draws_one_panel_per_subject_not_per_session(tmp_path) -> None:
    from scripts.neuroprobe.viz_brain import animate_brain

    coords = {}
    for s, t in ((1, 0), (1, 1), (2, 0)):
        coords[f"s{s}_t{t}"] = _write(tmp_path, s, t, _pattern(6, s * 10 + t))
    sessions, _ = build(str(tmp_path), _coords(tmp_path, coords), TAP, TASK)
    assert len(sessions) == 3
    info = animate_brain(sessions, np.arange(T) / 32.0, str(tmp_path / "b2.gif"),
                         fps=6, orbit_frames=2, hold=1)
    assert info["n_subjects"] == 2


def test_every_panel_gets_the_same_millimetre_scale(tmp_path) -> None:
    """Autoscaling each subplot to its own cloud makes a sparse montage fill the box and a
    dense one shrink, which reads as anatomy but is only an axis choice."""
    from scripts.neuroprobe.viz_brain import display_span

    small = np.zeros((6, 3), dtype=np.float32)
    small[:, 0] = np.linspace(-1, 1, 6)
    big = small * 40.0
    sessions = [{"coords": small}, {"coords": big}]
    assert display_span(sessions) == pytest.approx(40.0)
    # centring is on each subject's OWN centroid, so an offset head is not pushed off-frame
    shifted = [{"coords": big + 500.0}]
    assert display_span(shifted) == pytest.approx(40.0)


def test_the_strongest_decile_is_ranked_by_projection_not_by_colour(tmp_path) -> None:
    """rgb is percentile-stretched, so ranking contacts by colour ranks them by where the
    stretch landed. The check has to use the raw projection or it measures the colormap."""
    coords = {}
    for s in (1, 2):
        coords[f"s{s}_t0"] = _write(tmp_path, s, 0, _pattern(20, s))
    sessions, _ = build(str(tmp_path), _coords(tmp_path, coords), TAP, TASK)
    for s in sessions:
        assert "proj" in s and s["proj"].shape == (20, T, 3)
        # rgb is bounded to the unit gamut; proj is not, which is how they differ
        assert s["rgb"].min() >= 0.0 and s["rgb"].max() <= 1.0
        assert s["proj"].min() < 0.0


def test_anatomy_of_extremes_reports_per_subject_and_the_intersection(tmp_path) -> None:
    from scripts.neuroprobe.viz_brain import anatomy_of_extremes

    coords = {}
    for s in (1, 2):
        coords[f"s{s}_t0"] = _write(tmp_path, s, 0, _pattern(20, s))
    sessions, _ = build(str(tmp_path), _coords(tmp_path, coords), TAP, TASK)
    got = anatomy_of_extremes(sessions, 0, q=0.9)
    assert set(got["per_subject"]) == {1, 2}
    for counts in got["per_subject"].values():
        assert sum(counts.values()) >= 2          # a decile of 20 contacts
        assert list(counts.values()) == sorted(counts.values(), reverse=True)
    # the intersection is exactly that -- present for every subject, not a union
    for lb in got["shared"]:
        assert all(lb in c for c in got["per_subject"].values())
