"""The page must report the same numbers as the figures.

A demo that recomputes its own version of the result is worse than no demo: it looks like
evidence and is not checked by anything. These tests pin the payload to the functions the
static figures and the printed table come from.
"""
from __future__ import annotations

import json

import numpy as np

from scripts.neuroprobe.test_viz_figures import TASK, TEMPORAL, _pattern, _write_classes
from scripts.neuroprobe.viz_common import load_all, shared_lobes
from scripts.neuroprobe.viz_demo import HTML, NULL_TASKS, build, page_prose
from scripts.neuroprobe.viz_figures import figure_tasks, retrieval

T = 32


def _corpus(tmp_path, shared: bool):
    profile = 20.0 * _pattern(0)
    diff = _pattern(1)
    rng = np.random.default_rng(5)
    for s in (1, 2, 3, 4):
        d = diff if shared else rng.normal(size=diff.shape)
        _write_classes(tmp_path, s, 0, TEMPORAL, 10,
                       {0: profile - 0.5 * d, 1: profile + 0.5 * d})
    return load_all(str(tmp_path))


def test_payload_align_matches_the_static_figure(tmp_path) -> None:
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    p = build(sessions, lobes, ["enc12"], [TASK], 32.0, 0.0)
    info = figure_tasks(sessions, lobes, "enc12", [TASK], str(tmp_path / "t.png"))
    assert abs(p["data"]["enc12"][TASK]["align"] - info["align_3pc"][TASK]) < 1e-9


def test_payload_retrieval_matches_the_reported_metric(tmp_path) -> None:
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    p = build(sessions, lobes, ["enc12"], [TASK], 32.0, 0.0)
    r = retrieval(sessions, lobes, "enc12", TASK)
    assert abs(p["retrieval"]["enc12"][TASK]["top1"] - r["top1"]) < 1e-4
    assert abs(p["chance"] - r["chance"]) < 1e-9


def test_payload_is_json_serializable_and_shaped_for_the_page(tmp_path) -> None:
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    p = build(sessions, lobes, ["enc12"], [TASK], 32.0, -0.5)
    blob = json.loads(json.dumps(p))          # numpy scalars would break the page silently
    assert blob["nframes"] == T
    assert len(blob["times"]) == T and abs(blob["times"][0] + 0.5) < 1e-9
    traj = blob["data"]["enc12"][TASK]["traj"]
    assert len(traj) == 4
    assert all(len(v) == T and len(v[0]) == 2 for v in traj.values())
    rgb = blob["data"]["enc12"][TASK]["rgb"]
    assert all(0 <= px <= 255 for img in rgb.values() for row in img
               for frame in row for px in frame)


def test_the_strip_panel_marks_the_event_only_when_the_window_leads_it(tmp_path) -> None:
    """t=0 is a fixed landmark, not a computed one, so the only thing that can go wrong is
    marking it in the wrong place -- or drawing it at all when the window opens ON the event,
    where a rule on the left border marks nothing and reads as a rendering artifact."""
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    lead = build(sessions, lobes, ["enc12"], [TASK], 32.0, -0.5)
    assert lead["t0"] == 16                       # 0.5 s x 32 Hz
    assert abs(lead["times"][lead["t0"]]) < 1e-9  # and it really is the zero-second frame
    assert build(sessions, lobes, ["enc12"], [TASK], 32.0, 0.0)["t0"] is None
    assert 'if (D.t0 !== null)' in HTML


def test_page_ships_a_player_driven_by_the_time_axis() -> None:
    """64 frames is too many to read by dragging, so the play control is part of the page.
    Its rate comes from ``times``, which is what makes 1x mean real time at either window."""
    for token in ('id="play"', 'id="speed"', "requestAnimationFrame(tick)",
                  "[0.1, 0.25, 0.5, 1, 2]",
                  "const HZ = 1 / (D.times[1] - D.times[0]);"):
        assert token in HTML, token


def test_unshared_subjects_give_the_page_a_near_zero_score(tmp_path) -> None:
    sessions = _corpus(tmp_path, shared=False)
    lobes = shared_lobes(sessions)
    p = build(sessions, lobes, ["enc12"], [TASK], 32.0, 0.0)
    assert abs(p["data"]["enc12"][TASK]["align"]) < 0.5


def test_board_auroc_averages_the_per_subject_cells(tmp_path) -> None:
    """CS cells are per-SUBJECT. Quoting one is the standard way to misreport the board, so
    the loader must average them and must not mix norms."""
    import json

    from scripts.neuroprobe.viz_common import board_cs_auroc

    blob = {"run|onset": {"cs": {"enc12|std": {"S1T0": 0.6, "S2T0": 0.8},
                                 "enc0|std": {"S1T0": 0.5, "S2T0": 0.5},
                                 "enc12|raw": {"S1T0": 0.99, "S2T0": 0.99}}}}
    p = tmp_path / "board.json"
    p.write_text(json.dumps(blob))
    got = board_cs_auroc(str(p))
    assert abs(got["onset"]["enc12"] - 0.7) < 1e-9      # the MEAN, not a cell
    assert abs(got["onset"]["enc0"] - 0.5) < 1e-9
    assert "raw" not in json.dumps(got)                  # norm=std only, raw is not reported


def test_payload_carries_decoding_and_the_honesty_columns(tmp_path) -> None:
    """The page shows geometry AND accuracy; a shared trajectory that decoded at chance
    would be a curiosity. LOSO and the peak/settle shape ride along in the same payload so
    the table cannot show a number the payload did not compute."""
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    p = build(sessions, lobes, ["enc12"], [TASK], 32.0, -0.25, n_pre=8,
              decode={TASK: {"enc12": 0.6021}})
    blob = json.loads(json.dumps(p))
    assert blob["decode"]["enc12"][TASK] == 0.6021
    assert blob["n_pre"] == 8
    d = blob["data"]["enc12"][TASK]
    assert d["loso"] is not None and -1.0 <= d["loso"] <= 1.0
    assert "peak_s" in d["shape"] and "settle_frac" in d["shape"]


def test_a_task_with_no_board_entry_renders_rather_than_crashing(tmp_path) -> None:
    """Board runs and reductions carry different task lists. A missing decode number is a
    null the page draws as an em dash, not a KeyError that blanks the whole page."""
    sessions = _corpus(tmp_path, shared=True)
    lobes = shared_lobes(sessions)
    p = build(sessions, lobes, ["enc12"], [TASK], 32.0, -0.25, n_pre=8, decode={})
    assert json.loads(json.dumps(p))["decode"]["enc12"][TASK] is None


def test_the_page_never_names_a_task_that_is_not_in_the_menu() -> None:
    """The 15-task copy shipped onto an 8-task page and claimed a `frame_brightness` floor
    that was not on it. --tasks is a free choice, so every task the prose names has to be a
    task the reader can actually select."""
    eight = ["onset", "delta_volume", "word_index", "word_gap", "gpt2_surprisal",
             "word_head_pos", "word_part_speech", "word_length"]
    prose = page_prose(eight, 16, -0.5, 32.0)
    blob = " ".join(prose.values())
    for absent in NULL_TASKS + ("speech", "volume"):
        assert absent not in blob, f"page names '{absent}', which is not in the menu"
    assert "no null task sits here as a floor" in blob


def test_the_page_keeps_its_floor_when_a_null_task_is_on_the_menu() -> None:
    """The other half: with a null present the sentence is TRUE and must survive, otherwise
    the fix for the 8-task page would have silently stripped the 15-task page's control."""
    prose = page_prose(["onset", "speech", "frame_brightness"], 16, -0.5, 32.0)
    blob = " ".join(prose.values())
    assert "frame_brightness" in blob and "Two controls" in blob
    assert "onset" in blob and "speech" in blob


def test_the_origin_illustration_needs_both_of_the_tasks_it_contrasts() -> None:
    """onset-vs-speech explains why a sustained contrast fails to return. With only one of
    them on the page the comparison is unverifiable, so the generic wording is used."""
    both = page_prose(["onset", "speech"], 16, -0.5, 32.0)["why"]
    one = page_prose(["onset", "word_length"], 16, -0.5, 32.0)["why"]
    assert "a word inside ongoing talk" in both
    assert "a word inside ongoing talk" not in one
    assert "re-references to the state at word onset" in one
