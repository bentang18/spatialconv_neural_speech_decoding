"""Aggregate v14-core First-Run Protocol results (Task E3).

Two-level aggregation over the 120-job array:

  (patient, depth, fold)  → mean PER across the 3 seeds      ("fold mean")
  (patient, depth)        → mean ± std across the 5 folds    ("patient mean")
  (depth,)                → mean of per-patient means (#33)   ("population mean")

The aggregator is generic over grouping keys so Phase F ablations (different
`d_model`, `num_blocks`, etc.) can reuse it by widening the config key set.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


BASELINE_POPULATION: float = 0.825  # per-phoneme MFA + flat head, 11-patient mean
BASELINE_PER_PATIENT: dict[str, float] = {"S14": 0.734}  # published S14 baseline
GATE_TOLERANCE: float = 0.02  # plan closure rule


@dataclass(frozen=True)
class ResultRow:
    patient: str
    fold: int
    seed: int
    grid_mixer_depth: int
    best_val_per: float
    test_per: float
    final_epoch: int


@dataclass
class FoldCell:
    patient: str
    depth: int
    fold: int
    seed_pers: list[float]  # length 3 (or fewer if incomplete)

    @property
    def mean(self) -> float:
        return statistics.mean(self.seed_pers)


@dataclass
class PatientCell:
    patient: str
    depth: int
    fold_means: list[float]  # length 5 (or fewer if incomplete)

    @property
    def mean(self) -> float:
        return statistics.mean(self.fold_means)

    @property
    def std(self) -> float:
        return statistics.stdev(self.fold_means) if len(self.fold_means) > 1 else 0.0


@dataclass
class PopulationCell:
    depth: int
    patient_means: dict[str, float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.patient_means.values())


@dataclass
class AggregateReport:
    fold_cells: dict[tuple[str, int, int], FoldCell]  # (patient, depth, fold)
    patient_cells: dict[tuple[str, int], PatientCell]  # (patient, depth)
    population_cells: dict[int, PopulationCell]  # depth
    missing: list[tuple[str, int, int, int]]  # (patient, fold, seed, depth)
    expected_n: int
    actual_n: int

    def patient_winner(self, patient: str, depths: Iterable[int]) -> dict:
        """Lower mean wins; flag `close_call` if gap < max(std_a, std_b)."""

        cells = {
            d: self.patient_cells[(patient, d)]
            for d in depths
            if (patient, d) in self.patient_cells
        }
        if not cells:
            return {"winner": None, "reason": "no data"}
        sorted_depths = sorted(cells, key=lambda d: cells[d].mean)
        winner = sorted_depths[0]
        info: dict = {"winner": winner, "mean": cells[winner].mean}
        if len(sorted_depths) >= 2:
            runner = sorted_depths[1]
            gap = cells[runner].mean - cells[winner].mean
            max_std = max(cells[winner].std, cells[runner].std)
            info["gap"] = gap
            info["close_call"] = gap < max_std
        return info


def load_results(results_dir: Path) -> list[ResultRow]:
    """Read every `*.result.json` under ``results_dir`` recursively."""

    rows: list[ResultRow] = []
    for p in Path(results_dir).rglob("*.result.json"):
        d = json.loads(p.read_text())
        rows.append(
            ResultRow(
                patient=d["patient"],
                fold=int(d["fold"]),
                seed=int(d["seed"]),
                grid_mixer_depth=int(d["grid_mixer_depth"]),
                best_val_per=float(d["best_val_per"]),
                test_per=float(d["test_per"]),
                final_epoch=int(d.get("final_epoch", -1)),
            )
        )
    return rows


def aggregate(
    rows: list[ResultRow],
    *,
    patients: Iterable[str],
    depths: Iterable[int],
    folds: Iterable[int],
    seeds: Iterable[int],
) -> AggregateReport:
    patients = list(patients)
    depths = list(depths)
    folds = list(folds)
    seeds = list(seeds)
    expected_n = len(patients) * len(depths) * len(folds) * len(seeds)

    by_key: dict[tuple[str, int, int, int], float] = {}
    for r in rows:
        by_key[(r.patient, r.grid_mixer_depth, r.fold, r.seed)] = r.test_per

    missing: list[tuple[str, int, int, int]] = []
    fold_cells: dict[tuple[str, int, int], FoldCell] = {}
    for pt in patients:
        for d in depths:
            for f in folds:
                pers: list[float] = []
                for s in seeds:
                    v = by_key.get((pt, d, f, s))
                    if v is None:
                        missing.append((pt, f, s, d))
                    else:
                        pers.append(v)
                if pers:
                    fold_cells[(pt, d, f)] = FoldCell(pt, d, f, pers)

    patient_cells: dict[tuple[str, int], PatientCell] = {}
    for pt in patients:
        for d in depths:
            fm = [
                fold_cells[(pt, d, f)].mean
                for f in folds
                if (pt, d, f) in fold_cells
            ]
            if fm:
                patient_cells[(pt, d)] = PatientCell(pt, d, fm)

    population_cells: dict[int, PopulationCell] = {}
    for d in depths:
        pm = {
            pt: patient_cells[(pt, d)].mean
            for pt in patients
            if (pt, d) in patient_cells
        }
        if pm:
            population_cells[d] = PopulationCell(d, pm)

    return AggregateReport(
        fold_cells=fold_cells,
        patient_cells=patient_cells,
        population_cells=population_cells,
        missing=missing,
        expected_n=expected_n,
        actual_n=len(rows),
    )


def _fmt_per(v: float) -> str:
    return f"{v:.3f}"


def render_markdown(
    report: AggregateReport,
    *,
    patients: Iterable[str],
    depths: Iterable[int],
) -> str:
    patients = list(patients)
    depths = list(depths)
    lines: list[str] = []
    lines.append("# v14-core First-Run Protocol results\n")
    lines.append(
        f"{report.actual_n} / {report.expected_n} result rows. "
        f"{len(report.missing)} missing.\n"
    )

    # Per-patient × depth table (mean ± std across 5 folds)
    lines.append("## Per-patient PER (mean ± std across folds)\n")
    header = "| Patient | " + " | ".join(f"depth={d}" for d in depths) + " | Winner | Baseline | Passes gate |"
    sep = "|" + "---|" * (len(depths) + 4)
    lines.append(header)
    lines.append(sep)
    for pt in patients:
        row = [pt]
        for d in depths:
            c = report.patient_cells.get((pt, d))
            row.append(
                f"{_fmt_per(c.mean)} ± {_fmt_per(c.std)}" if c else "—"
            )
        winner_info = report.patient_winner(pt, depths)
        winner = winner_info.get("winner")
        winner_cell = (
            f"depth={winner}"
            + (" (close)" if winner_info.get("close_call") else "")
            if winner is not None
            else "—"
        )
        row.append(winner_cell)
        baseline = BASELINE_PER_PATIENT.get(pt, BASELINE_POPULATION)
        row.append(_fmt_per(baseline))
        if winner is not None:
            best = report.patient_cells[(pt, winner)].mean
            passes = best <= baseline + GATE_TOLERANCE
            row.append(f"{'yes' if passes else 'no'} (Δ={best - baseline:+.3f})")
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Population per depth
    lines.append("## Population PER per depth (mean of per-patient means, #33)\n")
    lines.append("| Depth | Population PER | Baseline | Δ |")
    lines.append("|---|---|---|---|")
    for d in depths:
        pc = report.population_cells.get(d)
        if pc is None:
            lines.append(f"| {d} | — | {_fmt_per(BASELINE_POPULATION)} | — |")
        else:
            delta = pc.mean - BASELINE_POPULATION
            lines.append(
                f"| {d} | {_fmt_per(pc.mean)} | "
                f"{_fmt_per(BASELINE_POPULATION)} | {delta:+.3f} |"
            )
    lines.append("")

    # Closure verdict
    lines.append("## Plan-closure verdict\n")
    all_pass = all(
        (
            (winner_info := report.patient_winner(pt, depths)).get("winner") is not None
            and report.patient_cells[(pt, winner_info["winner"])].mean
            <= BASELINE_PER_PATIENT.get(pt, BASELINE_POPULATION) + GATE_TOLERANCE
        )
        for pt in patients
        if any((pt, d) in report.patient_cells for d in depths)
    )
    if all_pass:
        lines.append(
            "Per-patient PER is within ~0.02 of the baseline on every core "
            "patient. Proceed to Phase F ablations, then graduate to v14-full."
        )
    else:
        lines.append(
            "At least one core patient misses the ~0.02 tolerance. Invoke "
            "`superpowers:systematic-debugging`. Usual suspects: (1) GridMixer "
            "reshape, (2) SoftParcelEmbedding support normalization, (3) "
            "decoder BOS / prev_tokens wiring, (4) token_active_mask zeroing "
            "legitimate cells."
        )
    lines.append("")

    # Missing
    if report.missing:
        lines.append("## Missing jobs\n")
        for pt, f, s, d in report.missing[:50]:
            lines.append(f"- {pt} fold={f} seed={s} depth={d}")
        if len(report.missing) > 50:
            lines.append(f"... and {len(report.missing) - 50} more")
        lines.append("")

    return "\n".join(lines)
