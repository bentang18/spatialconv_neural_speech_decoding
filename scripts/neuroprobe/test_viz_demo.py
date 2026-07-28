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
from scripts.neuroprobe.viz_demo import HTML, build
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
