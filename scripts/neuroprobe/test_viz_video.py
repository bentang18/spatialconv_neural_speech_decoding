"""The animation must be the static figure in motion, not a second implementation of it.

A video drawn in a basis that is nearly the figure's basis would look right and be wrong,
and no reviewer could catch it. So the geometry is pinned to ``task_basis`` -- the same call
the panel makes -- and the shared axis range is pinned too, because per-task autoscaling in
a video is the same lie as in a still.
"""
from __future__ import annotations

import os

import numpy as np

from scripts.neuroprobe.test_viz_figures import TASK, TEMPORAL, _pattern, _write_classes
from scripts.neuroprobe.viz_common import load_all, shared_lobes
from scripts.neuroprobe.viz_figures import _traj, figure_tasks, task_basis
from scripts.neuroprobe.viz_video import animate_task


def _corpus(tmp_path, shared: bool):
    profile = 20.0 * _pattern(0)
    diff = _pattern(1)
    rng = np.random.default_rng(11)
    for s in (1, 2, 3, 4):
        d = diff if shared else rng.normal(size=diff.shape)
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: profile - 0.5 * d, 1: profile + 0.5 * d})
    return load_all(str(tmp_path))


def test_animate_writes_a_clip_covering_draw_then_orbit(tmp_path) -> None:
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    dst = str(tmp_path / "v.gif")          # gif: no ffmpeg dependency in the test
    info = animate_task(sessions, lobes, "enc12", TASK, [TASK], dst, fps=6,
                        orbit_frames=4, hold=1)
    assert info["n_frames"] == info["t_len"] + 1 + 4
    assert info["n_sessions"] == 4
    assert os.path.getsize(info["path"]) > 0


def test_the_animation_geometry_is_the_panel_geometry(tmp_path) -> None:
    """task_basis is the single source. If the panel and the clip ever diverged, the
    correlation the panel PRINTS would no longer describe the curves the clip DRAWS."""
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    per_task, comps, mu, _, lim = task_basis(sessions, lobes, "enc12", [TASK])
    trajs = [_traj(m, comps, mu) for _, m in per_task[TASK]]
    # identical subjects -> every trajectory coincides, and the panel says so
    for p in trajs[1:]:
        np.testing.assert_allclose(p, trajs[0], atol=1e-9)
    info = figure_tasks(sessions, lobes, "enc12", [TASK], str(tmp_path / "t.png"))
    assert info["align_3pc"][TASK] > 0.999
    assert lim >= float(np.abs(trajs[0]).max()) - 1e-12


def test_a_dud_task_is_not_rescaled_to_fill_the_frame(tmp_path) -> None:
    """The clip's axis range is shared across tasks, so a task whose contrast is weak has
    to render as a speck near the origin rather than a full-frame path.

    The dud is built the way a real one looks -- a class difference an order of magnitude
    smaller than the live task's, and unshared between subjects (frame_brightness has a
    split-half ceiling of .056). Scaling is per SESSION and pooled over tasks, so that
    weakness survives into the picture instead of being normalized away.
    """
    live = _pattern(1)
    profile = 20.0 * _pattern(0)
    rng = np.random.default_rng(3)
    for s in (1, 2, 3, 4):
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: profile - 0.5 * live, 1: profile + 0.5 * live})
        f = tmp_path / f"red_s{s}_t0_hga.npz"
        d = dict(np.load(f))
        weak = 0.05 * rng.normal(size=live.shape)
        for cls in (0, 1):
            sign = 1.0 if cls == 1 else -1.0
            dead = profile + sign * 0.5 * weak
            for name in ("all", "h0", "h1"):
                d[f"enc12/dead/c{cls}/{name}"] = dead[None].astype(np.float32)
                d[f"n/dead/c{cls}/{name}"] = np.int64(50)
        np.savez_compressed(f, **d)
    sessions = load_all(str(tmp_path))
    lobes = shared_lobes(sessions)
    per_task, comps, mu, _, lim = task_basis(sessions, lobes, "enc12", [TASK, "dead"])
    live_max = max(float(np.abs(_traj(m, comps, mu)).max()) for _, m in per_task[TASK])
    dead_max = max(float(np.abs(_traj(m, comps, mu)).max()) for _, m in per_task["dead"])
    assert dead_max < 0.5 * live_max, (dead_max, live_max)
    # and the shared range is set by the live task, so the dud cannot fill the box
    assert abs(lim - live_max) < 1e-12


def test_animate_returns_empty_for_a_task_with_no_data(tmp_path) -> None:
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    assert animate_task(sessions, lobes, "enc12", "nope", [TASK], str(tmp_path / "x.gif"),
                        fps=6, orbit_frames=2, hold=1) == {}


def test_pooling_collapses_a_subjects_sessions_into_one_curve(tmp_path) -> None:
    """Two trials of one subject share a montage, so drawing both would read as agreement
    the cross-subject metric never counted (it buckets same-subject pairs separately)."""
    from scripts.neuroprobe.viz_video import pool_subjects

    class S:
        def __init__(self, sid):
            self.subject_id = sid

    a, b = np.zeros((4, 3)), np.ones((4, 3))
    pooled = pool_subjects([(S(1), a), (S(1), b), (S(2), a)])
    assert [sid for sid, _ in pooled] == [1, 2]
    np.testing.assert_allclose(pooled[0][1], 0.5)
    np.testing.assert_allclose(pooled[1][1], 0.0)


def test_pooled_clip_draws_one_curve_per_subject_not_per_session(tmp_path) -> None:
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    info = animate_task(sessions, lobes, "enc12", TASK, [TASK], str(tmp_path / "v.gif"),
                        fps=6, orbit_frames=3, hold=1, per_subject=True)
    assert info["n_sessions"] == 4 and info["n_curves"] == 4     # 4 subjects, 1 trial each
    assert info["per_subject"] is True


def test_loop_schedule_turns_a_playback_rate_into_a_seamless_frame_budget() -> None:
    """The clip length is a consequence of speed and repeats, never typed in. 64 frames of
    32 Hz data at 0.5x is 16 fps, so one pass is 4 s and five passes are exactly 20 s."""
    from scripts.neuroprobe.viz_video import loop_schedule

    fps, n_frames, dur = loop_schedule(64, 5, 32.0, 0.5)
    assert (fps, n_frames) == (16, 320) and abs(dur - 20.0) < 1e-9
    assert loop_schedule(64, 5, 32.0, 1.0) == (32, 320, 10.0)     # 1x halves the duration


def test_loop_clip_replays_without_a_pause_between_passes(tmp_path) -> None:
    """A seamless wrap is the point: frame t_len must be pass 2 frame 1, not a held last
    frame. Off-by-one here shows up as a visible stutter every replay."""
    from scripts.neuroprobe.viz_video import animate_loop

    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    info = animate_loop(sessions, lobes, "enc12", TASK, [TASK], str(tmp_path / "l.gif"),
                        hz=32.0, speed=0.5, repeats=3, deg_per_replay=25.0)
    t_len = info["t_len"]
    assert info["n_frames"] == t_len * 3                  # no hold, no orbit tail
    assert [i % t_len + 1 for i in (t_len - 1, t_len)] == [t_len, 1]


def test_rotating_clip_removes_every_element_that_repicks_its_box_edge() -> None:
    """Spine, ticks, tick labels and axis label all get re-assigned to a different cube edge
    as the view angle crosses a threshold, which under rotation reads as flipping rather than
    motion. The loop clip must ship none of them, and must ship the triad that replaces them."""
    import matplotlib.pyplot as plt

    from scripts.neuroprobe.viz_video import _static_axis_chrome

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    n_grid = 6
    _static_axis_chrome(ax, 1.0, n_grid=n_grid)
    fig.canvas.draw()          # ticks regenerate on draw; the fix must survive that
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert axis.line.get_visible() is False
        assert axis._axinfo["tick"]["inward_factor"] == 0.0
        assert axis._axinfo["tick"]["outward_factor"] == 0.0
        assert axis.pane.get_visible() is False    # back walls swap as the camera turns
        assert all(not t.get_text() for t in axis.get_ticklabels())
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("", "", "")
    assert {t.get_text() for t in ax.texts} == {"PC1", "PC2", "PC3"}
    # 3 triad arms + a floor grid drawn as plain segments, and NOTHING else: the floor must
    # be real lines rather than matplotlib's grid, or it comes and goes with the panes.
    assert len(ax.lines) == 3 + 2 * (n_grid + 1)
    zs = [ln.get_data_3d()[2] for ln in ax.lines[:2 * (n_grid + 1)]]
    assert all(set(z) == {-1.0} for z in zs), "floor grid must sit at z = -lim, flat"
    plt.close(fig)
