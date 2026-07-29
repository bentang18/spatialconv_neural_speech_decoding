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


def test_empty_directory_fails_loudly(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--dir", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "no scored cells" in r.stderr
