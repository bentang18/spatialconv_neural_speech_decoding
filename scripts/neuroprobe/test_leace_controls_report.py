"""The verdict text is the deliverable, so each world has to produce the right call.

Synthetic result trees are cheap; a wrong verdict on real data is not recoverable, because by then
it is in a paper. The three worlds here are the three readings the controls exist to separate.
"""

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "leace_controls_report.py"
CELLS = ["S1T1", "S1T2", "S3T0", "S3T1", "S4T0", "S4T1", "S7T0", "S7T1", "S10T0", "S10T1"]


def write_tree(tmp: Path, *, between, var_removed, d_leace, d_shuf, d_top, tap="enc12"):
    """One task per cell; deltas are applied exactly so the paired mean is what we asked for."""
    for i, cell in enumerate(CELLS):
        base = 0.60 + 0.001 * i                       # a std baseline that varies across cells
        jitter = 1e-6 * (i - 4.5)                     # keeps sd > 0 so t is finite
        cells = {
            f"{tap}|std": {"test": base},
            f"{tap}|leace": {"test": base + d_leace + jitter},
            f"{tap}|leace_shuf": {"test": base + d_shuf + jitter},
            f"{tap}|leace_toppc": {"test": base + d_top + jitter},
        }
        checks = {tap: {"var_removed": var_removed, "dir_between_frac": between,
                        "cos_pc1": 0.9, "pc1_var_frac": 0.22, "pc_participation": 1.1,
                        "cos_domain_mean_shift": 1.0}}
        (tmp / f"ctrl_{cell}.json").write_text(
            json.dumps({"onset": {"cells": cells, "checks": checks, "n_parcels": 7}}))


def run(tmp: Path) -> str:
    r = subprocess.run([sys.executable, str(SCRIPT), "--dir", str(tmp)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_world_C_no_power_is_called_out_as_voiding_the_claim(tmp_path):
    """Everything is free, including a shuffled concept. The instrument reads zero on all inputs."""
    write_tree(tmp_path, between=0.3, var_removed=0.21, d_leace=-7e-6, d_shuf=-5e-6, d_top=-8e-6)
    out = run(tmp_path)
    assert "NO POWER" in out
    assert "voids the claim" in out
    assert "ALSO free" in out, "a free top-PC control must also be reported"


def test_world_B_offset_dominates_collapses_the_effective_variance(tmp_path):
    """The direction is 95% between-session offset, so the 21% headline is 1% of real content."""
    write_tree(tmp_path, between=0.95, var_removed=0.21, d_leace=-7e-6, d_shuf=-3e-3, d_top=-4e-3)
    out = run(tmp_path)
    assert "EFFECTIVE within-session erased variance = 1.050%" in out
    assert "does not survive" in out
    assert "HAS power" in out, "large control costs mean the test had power"


def test_world_A_survives_when_geometry_and_controls_both_hold(tmp_path):
    """Real within-session share, controls cost real money, identity is free. Reading A."""
    write_tree(tmp_path, between=0.15, var_removed=0.21, d_leace=-7e-6, d_shuf=-2e-3, d_top=-5e-3)
    out = run(tmp_path)
    assert "framing survives" in out
    assert "HAS power" in out
    assert "NOT free, so identity's freeness is specific" in out


def test_effective_variance_comparison_fires_across_taps(tmp_path):
    """The enc0-vs-enc12 size contrast has to be re-tested on effective, not raw, variance."""
    write_tree(tmp_path, between=0.95, var_removed=0.21, d_leace=-7e-6, d_shuf=-3e-3,
               d_top=-4e-3, tap="enc12")
    # enc0: small raw share, and mostly within-session -> effective 0.31%
    for i, cell in enumerate(CELLS):
        p = tmp_path / f"ctrl_{cell}.json"
        doc = json.loads(p.read_text())
        base = 0.55 + 0.001 * i
        doc["onset"]["cells"].update({
            "enc0|std": {"test": base}, "enc0|leace": {"test": base - 0.002},
            "enc0|leace_shuf": {"test": base - 1e-5}, "enc0|leace_toppc": {"test": base - 3e-3}})
        doc["onset"]["checks"]["enc0"] = {"var_removed": 0.0037, "dir_between_frac": 0.15,
                                          "cos_pc1": 0.4, "cos_domain_mean_shift": 1.0}
        p.write_text(json.dumps(doc))
    out = run(tmp_path)
    assert "enc0 vs enc12 on EFFECTIVE within-session erased variance" in out
    # enc12 effective 1.05% vs enc0 0.31% -> ratio 3.3x, below the 5x the published story needs
    assert "size gap largely CLOSES" in out


def test_partial_cells_are_dropped_not_silently_averaged(tmp_path):
    """A cell missing the std baseline cannot contribute a delta -- partial cells lie."""
    write_tree(tmp_path, between=0.5, var_removed=0.2, d_leace=-1e-3, d_shuf=-1e-3, d_top=-1e-3)
    p = tmp_path / f"ctrl_{CELLS[0]}.json"
    doc = json.loads(p.read_text())
    del doc["onset"]["cells"]["enc12|std"]
    p.write_text(json.dumps(doc))
    assert "9/9 neg" in run(tmp_path), "the unpaired cell must be dropped, leaving 9"


def _augment(tmp: Path, tap: str, extra: dict):
    for cell in CELLS:
        p = tmp / f"ctrl_{cell}.json"
        doc = json.loads(p.read_text())
        doc["onset"]["checks"][tap].update(extra)
        p.write_text(json.dumps(doc))


def test_the_null_normalised_section_divides_by_the_shuffled_arm(tmp_path):
    """`pc_participation` is not comparable across taps raw -- enc0 and enc12 have different
    row-space ranks -- so the ratio to the shuffled arm is what gets printed."""
    write_tree(tmp_path, between=0.99, var_removed=0.2, d_leace=-7e-6, d_shuf=-5e-6, d_top=-8e-6)
    _augment(tmp_path, "enc12", {"pc_participation_leace_shuf": 200.0,
                                 "cos_pc1_leace_shuf": 0.1, "dir_between_frac_leace_shuf": 0.3,
                                 "var_removed_leace_shuf": 0.0001})
    out = run(tmp_path)
    assert "against the matched null" in out
    # participation 1.1 against a null of 200 -> 0.0055
    assert "real   1.1000   null  200.0000   ratio   0.0055" in out
    assert "real   0.9000   null    0.1000   ratio   9.0000" in out, "cos_pc1 ratio too"


def test_alignment_is_reported_against_its_ceiling_and_the_rotation_reference(tmp_path):
    write_tree(tmp_path, between=0.99, var_removed=0.2, d_leace=-7e-6, d_shuf=-5e-6, d_top=-8e-6)
    _augment(tmp_path, "enc12", {"align_k8": 0.12, "align_k8_frac": 0.13, "align_k8_ceil": 0.92,
                                 "align_k8_floor": 0.001, "diag_k8": 0.55, "diag_k8_rot": 0.80})
    out = run(tmp_path)
    assert "share a coordinate system" in out
    assert "k=8    align frac 0.1300" in out
    assert "diag 0.5500 vs rotation 0.8000" in out, "the rotation reference must be printed beside"


def test_the_task_axis_section_counts_cells_beating_their_own_null(tmp_path):
    """A mean can be carried by one cell; the p95 count cannot."""
    write_tree(tmp_path, between=0.99, var_removed=0.2, d_leace=-7e-6, d_shuf=-5e-6, d_top=-8e-6)
    _augment(tmp_path, "enc12", {"task_cos": 0.40, "task_cos_null": 0.05,
                                 "task_cos_null_p95": 0.12, "task_vs_sess_t": 0.02,
                                 "task_vs_sess_chance": 0.012})
    out = run(tmp_path)
    assert "is the TASK axis shared" in out
    assert "cos 0.4000  vs null 0.0500 (x8.00)   10/10 beat their p95" in out
    assert "overlap with the session offset: 0.0200 (chance 0.0120)" in out


def _task_world(tmp: Path, speech_cos: float, visual_cos: float):
    """Same cell, two tasks, one from each modality group -- the shape the real menu has."""
    write_tree(tmp, between=0.99, var_removed=0.2, d_leace=-7e-6, d_shuf=-5e-6, d_top=-8e-6)
    _augment(tmp, "enc12", {"task_cos": speech_cos, "task_cos_null": 0.05,
                            "task_cos_null_p95": 0.12, "task_vs_sess_t": 0.02,
                            "task_vs_sess_chance": 0.012})
    for cell in CELLS:
        p = tmp / f"ctrl_{cell}.json"
        doc = json.loads(p.read_text())
        vis = json.loads(json.dumps(doc["onset"]))          # same scores, different task axis
        vis["checks"]["enc12"]["task_cos"] = visual_cos
        doc["global_flow"] = vis
        p.write_text(json.dumps(doc))


def test_the_task_axis_breakdown_separates_speech_from_visual(tmp_path):
    """A shared task axis only explains the cross-subject gain if it rises the way the gain does --
    for speech and not for visual. Pooling the menu averages that contrast away."""
    _task_world(tmp_path, speech_cos=0.60, visual_cos=0.06)
    out = run(tmp_path)
    assert "speech/language  cos 0.6000  vs null 0.0500   10/10 beat their p95" in out
    assert "cos 0.0600  vs null 0.0500   0/10 beat their p95" in out, "the visual group"
    assert "cos 0.3300  vs null 0.0500 (x6.60)   10/20 beat their p95" in out, \
        "the pooled line reads as a shared axis when only half the menu has one"


def test_the_breakdown_is_suppressed_when_the_menu_is_one_sided(tmp_path):
    """One group is not a contrast, and a bare group mean reads as selectivity that wasn't tested."""
    write_tree(tmp_path, between=0.99, var_removed=0.2, d_leace=-7e-6, d_shuf=-5e-6, d_top=-8e-6)
    _augment(tmp_path, "enc12", {"task_cos": 0.40, "task_cos_null": 0.05,
                                 "task_cos_null_p95": 0.12, "task_vs_sess_t": 0.02,
                                 "task_vs_sess_chance": 0.012})
    assert "speech/language" not in run(tmp_path)


def test_empty_directory_fails_loudly(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "no scored cells" in r.stderr
