"""A2: Events.tsv + muscle-artifact audit across all PS D-patients.

For each of the 50 D-patients under `CoganLab/BIDS-1.4_Phoneme_sequencing/BIDS/`:
  - concatenate all `sub-<D>/ieeg/*_task-PhonemeSequence_*_events.tsv` runs
  - count Listen/Audio/Speak/Response rows
  - extract unique `stim_file` (should be a subset of ps_tokens.csv's 52 tokens)
  - compute stimulus-to-response delay (Response.onset - Audio.onset per trial)
  - read `derivatives/muscle/sub-<D>/*_artifacts.csv` for muscle-ch counts

Outputs `reports/seeg_events_audit_2026_04_24/{per_patient.csv, summary.md}`.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


BIDS_ROOT = Path(
    "/Users/bentang/Library/CloudStorage/Box-Box/CoganLab/"
    "BIDS-1.4_Phoneme_sequencing/BIDS"
)
TOKENS_CSV = Path("data/ps_tokens.csv")
OUT_DIR = Path("reports/seeg_events_audit_2026_04_24")


def load_canonical_stim_files(tokens_csv: Path) -> set[str]:
    stim: set[str] = set()
    with tokens_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            stim.add(f"{row['ps_notation']}.wav")
    return stim


def read_events(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def audit_patient(bids_id: str, bids_root: Path, canonical: set[str]) -> dict:
    pt_dir = bids_root / bids_id / "ieeg"
    runs = sorted(pt_dir.glob(f"{bids_id}_task-PhonemeSequence*_events.tsv"))

    phase_counts: dict[str, int] = {"Listen": 0, "Audio": 0, "Speak": 0, "Response": 0}
    stim_files: set[str] = set()
    audio_onsets: dict[tuple[str, str], float] = {}  # (run_key, trial_num) -> onset
    delays: list[float] = []
    format_label = "bids"

    for run in runs:
        run_key = run.name
        rows = read_events(run)
        if not rows:
            continue

        if "trial_type" in rows[0]:
            # Standard BIDS 4-row/trial layout
            for row in rows:
                ttype = row["trial_type"]
                if ttype in phase_counts:
                    phase_counts[ttype] += 1
                if ttype == "Audio":
                    stim = row["stim_file"].strip()
                    if stim and stim != "n/a":
                        stim_files.add(stim)
                    audio_onsets[(run_key, row["trial_num"])] = float(row["onset"])
                if ttype == "Response":
                    key = (run_key, row["trial_num"])
                    aud_on = audio_onsets.get(key)
                    if aud_on is not None:
                        delays.append(float(row["onset"]) - aud_on)
        elif "sound" in rows[0] and "audioStart" in rows[0]:
            # Legacy MATLAB-export format (one row per trial)
            format_label = "legacy_matlab"
            for row in rows:
                stim = row["sound"].strip()
                if stim and stim != "n/a":
                    stim_files.add(stim)
                for phase in ("Listen", "Audio", "Speak", "Response"):
                    phase_counts[phase] += 1  # one trial = one of each phase
                # Delay = respStart - audioStart if both present and numeric
                try:
                    delays.append(float(row["respStart"]) - float(row["audioStart"]))
                except (KeyError, ValueError):
                    pass
        else:
            format_label = "unknown"

    in_vocab = sum(1 for s in stim_files if s in canonical)
    out_of_vocab = sorted(stim_files - canonical)

    return {
        "bids_id": bids_id,
        "format": format_label,
        "n_runs": len(runs),
        "n_listen": phase_counts["Listen"],
        "n_audio": phase_counts["Audio"],
        "n_speak": phase_counts["Speak"],
        "n_response": phase_counts["Response"],
        "n_unique_stim": len(stim_files),
        "n_stim_in_vocab": in_vocab,
        "n_stim_out_of_vocab": len(out_of_vocab),
        "oov_stims": ";".join(out_of_vocab),
        "delay_mean_s": round(mean(delays), 4) if delays else None,
        "delay_std_s": round(stdev(delays), 4) if len(delays) > 1 else None,
        "n_response_with_delay": len(delays),
    }


def muscle_ch_count(bids_id: str, bids_root: Path) -> int | None:
    muscle_dir = bids_root / "derivatives" / "muscle" / bids_id
    if not muscle_dir.exists():
        return None
    paths = list(muscle_dir.glob("*_artifacts.csv"))
    if not paths:
        return None
    total: int = 0
    for p in paths:
        with p.open() as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows and rows[0] and rows[0][0].strip().lower() == "channel":
            rows = rows[1:]
        total += sum(1 for r in rows if r and r[0].strip())
    return total


def enumerate_ps_patients(bids_root: Path) -> list[str]:
    return sorted(p.name for p in bids_root.glob("sub-D*") if p.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", type=Path, default=BIDS_ROOT)
    ap.add_argument("--tokens-csv", type=Path, default=TOKENS_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    canonical = load_canonical_stim_files(args.tokens_csv)
    print(f"canonical stim_files: {len(canonical)} (expected 52)")

    patients = enumerate_ps_patients(args.bids_root)
    print(f"found {len(patients)} PS D-patients under {args.bids_root}")

    rows: list[dict] = []
    for bids_id in patients:
        aud = audit_patient(bids_id, args.bids_root, canonical)
        aud["n_muscle_ch"] = muscle_ch_count(bids_id, args.bids_root)
        rows.append(aud)
        flags: list[str] = []
        if aud["n_stim_out_of_vocab"] > 0:
            flags.append(f"OOV={aud['n_stim_out_of_vocab']}")
        if aud["n_audio"] < 30:
            flags.append(f"few_trials={aud['n_audio']}")
        flag_str = f"  [{','.join(flags)}]" if flags else ""
        print(
            f"  {bids_id}: runs={aud['n_runs']} trials={aud['n_audio']} "
            f"unique_stim={aud['n_unique_stim']} in_vocab={aud['n_stim_in_vocab']} "
            f"delay={aud['delay_mean_s']}±{aud['delay_std_s']}s "
            f"muscle_ch={aud['n_muscle_ch']}{flag_str}"
        )

    csv_path = args.out_dir / "per_patient.csv"
    fieldnames = [
        "bids_id", "format", "n_runs", "n_listen", "n_audio", "n_speak", "n_response",
        "n_unique_stim", "n_stim_in_vocab", "n_stim_out_of_vocab", "oov_stims",
        "delay_mean_s", "delay_std_s", "n_response_with_delay", "n_muscle_ch",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_total = len(rows)
    n_52 = sum(1 for r in rows if r["n_unique_stim"] == 52)
    n_oov = sum(1 for r in rows if r["n_stim_out_of_vocab"] > 0)
    n_missing_muscle = sum(1 for r in rows if r["n_muscle_ch"] is None)
    trials = [r["n_audio"] for r in rows]
    delays = [r["delay_mean_s"] for r in rows if r["delay_mean_s"] is not None]
    top_oov = sorted((r for r in rows if r["oov_stims"]), key=lambda r: -r["n_stim_out_of_vocab"])[:10]

    lines: list[str] = [
        "# A2 — PS D-patient events + muscle-artifact audit (2026-04-24)",
        "",
        f"Audited {n_total} PS D-patients under `{BIDS_ROOT.name}/`.",
        "Canonical stim_files derived from `data/ps_tokens.csv` (52 CVC/VCV tokens).",
        "",
        "## Token-identity verdict",
        "",
        f"- Patients using exactly 52 canonical tokens: **{n_52}/{n_total}**",
        f"- Patients with ≥1 out-of-vocabulary stim: {n_oov}",
        f"- Patients with muscle-artifacts file: {n_total - n_missing_muscle}/{n_total}",
        "",
        "## Trial counts",
        "",
        f"- Trials/patient (Audio rows): median={sorted(trials)[len(trials)//2]}, "
        f"min={min(trials)}, max={max(trials)}",
        "",
        "## Audio → Response delay (s)",
        "",
        f"- Patient-mean delays: {round(mean(delays),3)}±{round(stdev(delays),3)} "
        f"(min={round(min(delays),3)}, max={round(max(delays),3)})",
        "",
    ]
    if top_oov:
        lines += [
            "## Patients with out-of-vocabulary stim_files",
            "",
            "| bids_id | n_audio | n_unique_stim | n_oov | oov_stims |",
            "|---|---:|---:|---:|---|",
        ]
        for r in top_oov:
            lines.append(
                f"| {r['bids_id']} | {r['n_audio']} | {r['n_unique_stim']} | "
                f"{r['n_stim_out_of_vocab']} | `{r['oov_stims']}` |"
            )
        lines.append("")
    summary = args.out_dir / "summary.md"
    summary.write_text("\n".join(lines))

    print(f"\nwrote {csv_path}  ({n_total} rows)")
    print(f"wrote {summary}")
    print(f"exact-52-token patients: {n_52}/{n_total}  (OOV patients: {n_oov})")


if __name__ == "__main__":
    main()
