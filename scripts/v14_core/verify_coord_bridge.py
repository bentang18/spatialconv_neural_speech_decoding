#!/usr/bin/env python3
"""Coordinate-bridge verifier for blocker #12 (docs/plans/v14-core.md A2).

For each core patient, runs the `.fif channel -> amp -> (r,c) -> phys name`
bridge through the real channel map and both caches (fsaverage coordinates +
Tier-1 support). A patient is PASS iff every retained `.fif` channel maps to
exactly one physical name, every name is present exactly once in each cache,
and every `(r, c)` sits inside the patient's declared grid shape.

Patients with no locally available `.fif` emit a SKIP row; the same checks run
on DCC in Phase E2 before sbatch submission.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.channel_map import (
    GRID_SHAPES,
    PATIENT_KIND,
    expected_grid_shape_for_patient,
    resolve_physical_names_for_patient,
)
from speech_decoding.v14.fsaverage_projection import load_fsaverage_cache
from speech_decoding.v14.support_cache import read_support_cache


DEFAULT_BOX_ROOT = Path("/Users/bentang/Library/CloudStorage/Box-Box")
DEFAULT_CHANNEL_MAPS_DIR = PROJECT_ROOT / "data" / "channel_maps"
DEFAULT_FSAVG_CACHE_DIR = PROJECT_ROOT / "data" / "fsaverage_coords"
DEFAULT_SUPPORT_CACHE_DIR = PROJECT_ROOT / "data" / "atlas" / "support_cache"
DEFAULT_QC_PATH = PROJECT_ROOT / "docs" / "qc" / "coord_bridge_verification.md"
DEFAULT_PATHS_YAML = PROJECT_ROOT / "configs" / "paths.yaml"
DEFAULT_PATIENTS = ("S14", "S26", "S33", "S62")


Verdict = Literal["PASS", "FAIL", "SKIP"]


@dataclass(frozen=True)
class PatientReport:
    patient: str
    verdict: Verdict
    n_fif_channels: int
    n_bads: int
    n_kept: int
    grid_shape: tuple[int, int]
    notes: tuple[str, ...]


def fif_path_for(bids_root: Path, patient: str) -> Path:
    return (
        bids_root
        / "derivatives"
        / "epoch(CAR)"
        / f"sub-{patient}"
        / "epoch(band)(power)"
        / f"sub-{patient}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"
    )


def verify_patient(
    patient_id: str,
    *,
    bids_root: Path,
    channel_maps_dir: Path,
    fsavg_cache_dir: Path,
    support_cache_dir: Path,
    box_root: Path,
) -> PatientReport:
    notes: list[str] = []
    if patient_id not in PATIENT_KIND:
        return PatientReport(
            patient=patient_id, verdict="SKIP", n_fif_channels=0, n_bads=0,
            n_kept=0, grid_shape=(0, 0),
            notes=(f"no channel-map kind registered for {patient_id!r}",),
        )

    grid_shape = expected_grid_shape_for_patient(patient_id)
    fif = fif_path_for(bids_root, patient_id)
    if not fif.exists():
        return PatientReport(
            patient=patient_id, verdict="SKIP", n_fif_channels=0, n_bads=0,
            n_kept=0, grid_shape=grid_shape,
            notes=(f".fif missing locally: {fif}",),
        )

    import mne

    epochs = mne.read_epochs(str(fif), preload=False, verbose="ERROR")
    ch_names = tuple(epochs.ch_names)
    bads = tuple(epochs.info["bads"])
    kept = tuple(n for n in ch_names if n not in set(bads))

    try:
        resolutions = resolve_physical_names_for_patient(
            patient_id,
            kept,
            channel_maps_dir=channel_maps_dir,
            box_root=box_root,
        )
    except (KeyError, ValueError, IndexError, FileNotFoundError) as exc:
        notes.append(f"bridge failed: {exc}")
        return PatientReport(
            patient=patient_id, verdict="FAIL",
            n_fif_channels=len(ch_names), n_bads=len(bads), n_kept=len(kept),
            grid_shape=grid_shape, notes=tuple(notes),
        )

    phys_names = [r.phys_name for r in resolutions]
    dup_phys = {n for n in phys_names if phys_names.count(n) > 1}
    if dup_phys:
        notes.append(f"duplicate physical names: {sorted(dup_phys)[:10]}")

    # (r, c) bounds check against declared grid shape
    H_p, W_p = grid_shape
    out_of_bounds = [
        (r.amp_name, r.row, r.col) for r in resolutions
        if not (0 <= r.row < H_p and 0 <= r.col < W_p)
    ]
    if out_of_bounds:
        notes.append(f"{len(out_of_bounds)} cells outside grid {grid_shape}")

    # fsaverage cache uniqueness
    try:
        fs_cache = load_fsaverage_cache(patient_id, fsavg_cache_dir)
    except FileNotFoundError as exc:
        notes.append(f"fsaverage cache missing: {exc}")
        return PatientReport(
            patient=patient_id, verdict="FAIL",
            n_fif_channels=len(ch_names), n_bads=len(bads), n_kept=len(kept),
            grid_shape=grid_shape, notes=tuple(notes),
        )
    fs_counts: dict[str, int] = {}
    for n in fs_cache.names:
        fs_counts[n] = fs_counts.get(n, 0) + 1
    missing_fs = [n for n in phys_names if fs_counts.get(n, 0) == 0]
    nonunique_fs = [n for n in phys_names if fs_counts.get(n, 0) > 1]
    if missing_fs:
        notes.append(f"{len(missing_fs)} phys names missing from fsaverage cache")
    if nonunique_fs:
        notes.append(f"{len(nonunique_fs)} phys names non-unique in fsaverage cache")

    # support cache uniqueness
    support_path = support_cache_dir / f"{patient_id}_support_tier1.csv"
    try:
        sc_names, _support = read_support_cache(support_path)
    except FileNotFoundError as exc:
        notes.append(f"support cache missing: {exc}")
        return PatientReport(
            patient=patient_id, verdict="FAIL",
            n_fif_channels=len(ch_names), n_bads=len(bads), n_kept=len(kept),
            grid_shape=grid_shape, notes=tuple(notes),
        )
    sc_counts: dict[str, int] = {}
    for n in sc_names:
        sc_counts[n] = sc_counts.get(n, 0) + 1
    missing_sc = [n for n in phys_names if sc_counts.get(n, 0) == 0]
    nonunique_sc = [n for n in phys_names if sc_counts.get(n, 0) > 1]
    if missing_sc:
        notes.append(f"{len(missing_sc)} phys names missing from support cache")
    if nonunique_sc:
        notes.append(f"{len(nonunique_sc)} phys names non-unique in support cache")

    verdict: Verdict = "PASS" if not notes else "FAIL"
    return PatientReport(
        patient=patient_id, verdict=verdict,
        n_fif_channels=len(ch_names), n_bads=len(bads), n_kept=len(kept),
        grid_shape=grid_shape, notes=tuple(notes),
    )


def write_qc_report(qc_path: Path, reports: list[PatientReport]) -> None:
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Coordinate-bridge verification (blocker #12)\n")
    lines.append(
        "PASS: every retained `.fif` channel resolves to exactly one physical name, "
        "uniquely present in both the fsaverage coord cache and the Tier-1 support cache, "
        "with `(r, c)` inside the declared grid shape.\n"
    )
    lines.append("SKIP: `.fif` not available locally; re-runs on DCC in Phase E2.\n")
    lines.append("\n## Summary\n")
    lines.append("| patient | verdict | n_fif | n_bads | n_kept | grid_shape | notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in reports:
        note_text = "; ".join(r.notes) if r.notes else "—"
        lines.append(
            f"| {r.patient} | {r.verdict} | {r.n_fif_channels} | {r.n_bads} | "
            f"{r.n_kept} | {r.grid_shape[0]}×{r.grid_shape[1]} | {note_text} |"
        )
    lines.append("")
    qc_path.write_text("\n".join(lines))


def resolve_bids_root(paths_yaml: Path) -> Path:
    if not paths_yaml.exists():
        raise FileNotFoundError(f"missing paths config: {paths_yaml}")
    data = yaml.safe_load(paths_yaml.read_text())
    root = data.get("ps_bids_root")
    if root is None:
        raise KeyError(f"{paths_yaml}: missing 'ps_bids_root' key")
    return Path(str(root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--paths-yaml", type=Path, default=DEFAULT_PATHS_YAML)
    parser.add_argument("--channel-maps-dir", type=Path, default=DEFAULT_CHANNEL_MAPS_DIR)
    parser.add_argument("--fsavg-cache-dir", type=Path, default=DEFAULT_FSAVG_CACHE_DIR)
    parser.add_argument("--support-cache-dir", type=Path, default=DEFAULT_SUPPORT_CACHE_DIR)
    parser.add_argument("--box-root", type=Path, default=DEFAULT_BOX_ROOT)
    parser.add_argument("--qc-path", type=Path, default=DEFAULT_QC_PATH)
    args = parser.parse_args(argv)

    bids_root = resolve_bids_root(args.paths_yaml)
    reports: list[PatientReport] = []
    for patient_id in args.patient:
        if patient_id not in GRID_SHAPES:
            print(f"[{patient_id}] no grid shape — treating as SKIP")
            reports.append(PatientReport(
                patient=patient_id, verdict="SKIP", n_fif_channels=0, n_bads=0,
                n_kept=0, grid_shape=(0, 0),
                notes=(f"no grid shape registered",),
            ))
            continue
        r = verify_patient(
            patient_id,
            bids_root=bids_root,
            channel_maps_dir=args.channel_maps_dir,
            fsavg_cache_dir=args.fsavg_cache_dir,
            support_cache_dir=args.support_cache_dir,
            box_root=args.box_root,
        )
        reports.append(r)
        print(f"[{patient_id}] {r.verdict} ({r.n_kept}/{r.n_fif_channels} kept) "
              f"grid {r.grid_shape} notes={list(r.notes) or '—'}")

    write_qc_report(args.qc_path, reports)
    print(f"wrote {args.qc_path}")

    has_fail = any(r.verdict == "FAIL" for r in reports)
    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
