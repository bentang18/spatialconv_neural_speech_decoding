"""The grid's argument is the shared colour scale and the null row. Both are testable."""
from __future__ import annotations

import numpy as np

from scripts.neuroprobe.test_viz_figures import BL, C, T, TEMPORAL, _pattern
from scripts.neuroprobe.viz_common import load_all, shared_lobes
from scripts.neuroprobe.viz_figures import retrieval, retrieval_sims
from scripts.neuroprobe.viz_retrieval_grid import _time_ticks, grid

TAPS = ("enc0", "enc12")
SIG, NULL = "onset", "frame_brightness"


def _write(dirpath, subject_id, per_tap_task):
    """One session carrying SEVERAL taps and tasks, which the shared fixture cannot do."""
    out = {
        "subject_id": np.int64(subject_id), "trial_id": np.int64(0),
        "present_parcels": np.asarray([TEMPORAL], dtype=np.int64),
        "parcel_counts": np.asarray([10], dtype=np.int64),
        "band_lengths": np.asarray(BL, dtype=np.int64), "n_windows": np.int64(100),
    }
    for tap in {tp for tp, _ in per_tap_task}:
        out[f"{tap}/shape"] = np.asarray([1, T, C], dtype=np.int64)
        out[f"{tap}/col_sum"] = np.zeros((1, T * C), dtype=np.float32)
        out[f"{tap}/col_sq"] = np.ones((1, T * C), dtype=np.float32)
    for (tap, task), per_class in per_tap_task.items():
        for cls in (0, 1):
            m = per_class[cls][None].astype(np.float32)
            for name in ("all", "h0", "h1"):
                out[f"{tap}/{task}/c{cls}/{name}"] = m
                out[f"n/{task}/c{cls}/{name}"] = np.int64(50)
            out[f"count/{task}/c{cls}"] = np.int64(50)
    np.savez_compressed(str(dirpath / f"red_s{subject_id}_t0_hga.npz"), **out)


def _corpus(tmp_path):
    """Signal task: the class difference is SHARED across subjects, so it retrieves. Null
    task: independent per subject, so it cannot. Depth is faked by making enc12 cleaner."""
    shared = _pattern(1)
    rng = np.random.default_rng(7)
    for s in (1, 2, 3):
        payload = {}
        for tap, noise in (("enc0", 1.6), ("enc12", 0.15)):
            d = shared + noise * rng.normal(size=shared.shape)
            payload[(tap, SIG)] = {0: -0.5 * d, 1: 0.5 * d}
            n = rng.normal(size=shared.shape)
            payload[(tap, NULL)] = {0: -0.5 * n, 1: 0.5 * n}
        _write(tmp_path, s, payload)
    return load_all(str(tmp_path))


def test_sims_and_scalars_come_from_one_computation(tmp_path) -> None:
    """The grid must not re-derive top-1: a montage whose numbers disagree with report.json
    is worse than no montage."""
    sessions = _corpus(tmp_path)
    lobes = shared_lobes(sessions)
    sims, stats = retrieval_sims(sessions, lobes, "enc12", SIG)
    direct = retrieval(sessions, lobes, "enc12", SIG)
    assert stats["top1"] == direct["top1"]
    assert stats["median_rank"] == direct["median_rank"]
    assert len(sims) == stats["n_pairs"]
    assert all(s.shape == (T, T) for s in sims)


def test_every_cell_shares_one_colour_scale(tmp_path) -> None:
    """Per-panel normalisation would sharpen every diagonal, the null included, purely by
    rescaling. One vmax for the whole grid is what makes the progression readable."""
    sessions = _corpus(tmp_path)
    lobes = shared_lobes(sessions)
    out = tmp_path / "grid.png"
    info = grid(sessions, lobes, list(TAPS), [SIG, NULL], str(out), null_task=NULL)
    assert out.exists() and out.stat().st_size > 0
    hi = max(float(np.abs(np.mean(retrieval_sims(sessions, lobes, tp, tk)[0],
                                  axis=0)).max())
             for tp in TAPS for tk in (SIG, NULL))
    assert np.isclose(info["vmax"], hi)


def test_the_grid_shows_depth_helping_the_task_and_not_the_null(tmp_path) -> None:
    """The whole point of the figure. If depth sharpened diagonals as a generic side effect
    it would lift the null row too, and the figure would be showing an artefact."""
    sessions = _corpus(tmp_path)
    lobes = shared_lobes(sessions)
    info = grid(sessions, lobes, list(TAPS), [SIG, NULL], str(tmp_path / "g.png"),
                null_task=NULL)
    c = info["cells"]
    assert c[f"{SIG}|enc12"]["top1"] > c[f"{SIG}|enc0"]["top1"]
    assert c[f"{NULL}|enc12"]["top1"] <= 3 * c[f"{NULL}|enc12"]["chance"]
    assert c[f"{SIG}|enc12"]["median_rank"] < c[f"{NULL}|enc12"]["median_rank"]


def test_the_null_row_is_labelled_and_reported(tmp_path) -> None:
    """The control only works if a reader can tell which row it is."""
    sessions = _corpus(tmp_path)
    lobes = shared_lobes(sessions)
    info = grid(sessions, lobes, ["enc12"], [SIG, NULL], str(tmp_path / "g1.png"),
                null_task=NULL)
    assert info["null_task"] == NULL
    assert "top1" in info["cells"][f"{NULL}|enc12"]


def test_a_task_absent_from_the_shards_is_dropped_not_crashed(tmp_path) -> None:
    """Reductions and menus carry different task lists; a missing row is a dropped row."""
    sessions = _corpus(tmp_path)
    lobes = shared_lobes(sessions)
    info = grid(sessions, lobes, ["enc12"], [SIG, "not_a_real_task"],
                str(tmp_path / "g2.png"))
    assert info["tasks"] == [SIG]


def test_time_ticks_put_zero_at_the_onset_frame() -> None:
    """The whole point of labelling the axis is locating t=0. A tick grid anchored on frame 0
    instead of on the onset would put the '0' label wherever the window happened to start."""
    ticks, labels = _time_ticks(64, 32.0, -0.5)
    # No "1.5" tick: frame 63 is CENTERED at 1.469 s, so imshow's last cell ends before the
    # 1.5 s mark. Drawing it would put a label outside the data it claims to index.
    assert labels == ["-0.5", "0", "0.5", "1"]
    assert ticks[labels.index("0")] == 16.0, "t=0 must land on the n_pre frame"
    assert all(-0.5 <= t <= 63.5 for t in ticks), "no tick may sit outside the panel"


def test_time_ticks_survive_a_window_that_does_not_straddle_zero() -> None:
    """The 1 s window has no pre-stimulus frames; the helper must not invent a negative tick."""
    ticks, labels = _time_ticks(32, 32.0, 0.0)
    assert labels[0] == "0" and ticks[0] == 0.0
    assert all(float(x) >= 0 for x in labels)


def test_time_ticks_are_empty_without_a_sampling_rate() -> None:
    assert _time_ticks(64, 0.0, -0.5) == ([], [])
