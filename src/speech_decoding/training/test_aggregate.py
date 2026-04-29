"""Unit tests for the v14-core aggregator (Task E3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from speech_decoding.training.aggregate import (
    BASELINE_PER_PATIENT,
    BASELINE_POPULATION,
    GATE_TOLERANCE,
    aggregate,
    load_results,
    render_markdown,
)


def _write_row(
    root: Path,
    *,
    patient: str,
    fold: int,
    seed: int,
    depth: int,
    test_per: float,
) -> None:
    d = root / patient / f"depth{depth}"
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{patient}_fold{fold}_seed{seed}_depth{depth}"
    (d / f"{tag}.result.json").write_text(
        json.dumps(
            {
                "patient": patient,
                "fold": fold,
                "seed": seed,
                "grid_mixer_depth": depth,
                "best_val_per": test_per + 0.01,
                "test_per": test_per,
                "final_epoch": 50,
            }
        )
    )


def _full_cohort(root: Path, make_per) -> None:
    for pt in ("S14", "S26"):
        for d in (1, 2):
            for f in range(5):
                for s in range(3):
                    _write_row(
                        root, patient=pt, fold=f, seed=s, depth=d,
                        test_per=make_per(pt, d, f, s),
                    )


class TestLoadAndAggregate:
    def test_full_2pt_cohort_counts(self, tmp_path: Path) -> None:
        _full_cohort(tmp_path, lambda *_: 0.5)
        rows = load_results(tmp_path)
        assert len(rows) == 2 * 2 * 5 * 3

        report = aggregate(
            rows,
            patients=("S14", "S26"),
            depths=(1, 2),
            folds=range(5),
            seeds=range(3),
        )
        assert report.actual_n == 60
        assert report.expected_n == 60
        assert report.missing == []

    def test_missing_cells_tracked(self, tmp_path: Path) -> None:
        _full_cohort(tmp_path, lambda *_: 0.5)
        # Remove one file
        victim = tmp_path / "S14" / "depth1" / "S14_fold2_seed1_depth1.result.json"
        victim.unlink()
        rows = load_results(tmp_path)
        report = aggregate(
            rows,
            patients=("S14", "S26"),
            depths=(1, 2),
            folds=range(5),
            seeds=range(3),
        )
        assert report.actual_n == 59
        assert report.missing == [("S14", 2, 1, 1)]
        # Fold cell still computes mean from remaining 2 seeds
        fc = report.fold_cells[("S14", 1, 2)]
        assert len(fc.seed_pers) == 2

    def test_aggregation_math(self, tmp_path: Path) -> None:
        # S14 depth=1: fold mean = fold index / 10 (constant across seeds)
        # S14 depth=2: fold mean = fold index / 10 + 0.05
        def per(pt: str, d: int, f: int, s: int) -> float:
            return f / 10.0 + (0.05 if d == 2 else 0.0)

        _full_cohort(tmp_path, per)
        rows = load_results(tmp_path)
        report = aggregate(
            rows,
            patients=("S14", "S26"),
            depths=(1, 2),
            folds=range(5),
            seeds=range(3),
        )
        # S14 depth=1: fold means = [0, .1, .2, .3, .4], mean = .2
        pc = report.patient_cells[("S14", 1)]
        assert pc.mean == pytest.approx(0.2)
        # S14 depth=2: fold means shifted +0.05, mean = .25
        pc2 = report.patient_cells[("S14", 2)]
        assert pc2.mean == pytest.approx(0.25)
        # Winner: depth=1 (lower)
        w = report.patient_winner("S14", (1, 2))
        assert w["winner"] == 1
        # Population depth=1 = mean of per-patient means = (.2 + .2) / 2
        assert report.population_cells[1].mean == pytest.approx(0.2)


class TestCloseCall:
    def test_close_call_flagged_when_gap_below_std(self, tmp_path: Path) -> None:
        # Both depths ~0.4 but with per-fold variance ~0.05, gap ~0.02
        def per(pt: str, d: int, f: int, s: int) -> float:
            base = 0.4 + (0.01 if d == 2 else 0.0)
            return base + (0.05 if f % 2 == 0 else -0.05)

        _full_cohort(tmp_path, per)
        rows = load_results(tmp_path)
        report = aggregate(
            rows, patients=("S14",), depths=(1, 2),
            folds=range(5), seeds=range(3),
        )
        w = report.patient_winner("S14", (1, 2))
        assert w["close_call"] is True


class TestRenderMarkdown:
    def test_report_contains_expected_sections(self, tmp_path: Path) -> None:
        _full_cohort(tmp_path, lambda *_: 0.5)
        rows = load_results(tmp_path)
        report = aggregate(
            rows, patients=("S14", "S26"), depths=(1, 2),
            folds=range(5), seeds=range(3),
        )
        md = render_markdown(report, patients=("S14", "S26"), depths=(1, 2))
        assert "Per-patient PER" in md
        assert "Population PER per depth" in md
        assert "Plan-closure verdict" in md
        assert "S14" in md and "S26" in md

    def test_baseline_constants_are_surfaced(self) -> None:
        assert BASELINE_PER_PATIENT["S14"] == 0.734
        assert BASELINE_POPULATION == 0.825
        assert GATE_TOLERANCE == 0.02
