"""A1: sig-channel inventory for sEEG D-patients from pre-computed Cogan stats.

Reads the cluster-corrected permutation-test masks under
`CoganLab/BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/stats/<D_bids>_{aud,go,resp}_mask-ave.fif`
and counts how many channels hit any significant timepoint per phase.

Joins against `reports/seeg_cohort_scoping_2026_04_21/coverage.csv` so each row
has (lh_tier1, rh_tier1, n_sig_aud, n_sig_go, n_sig_resp, frac_sig_resp).

Outputs:
    reports/seeg_sig_channels_2026_04_24/per_patient.csv
    reports/seeg_sig_channels_2026_04_24/summary.md

The `frac_sig_resp` column is the usability proxy — patients with low response-
phase sig fraction are flagged as probable "usability-low." This is a
Nanlin-independent first pass, not her authoritative tiering.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mne


STATS_ROOT = Path(
    "/Users/bentang/Library/CloudStorage/Box-Box/CoganLab/"
    "BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/stats"
)
COVERAGE_CSV = Path("reports/seeg_cohort_scoping_2026_04_21/coverage.csv")
OUT_DIR = Path("reports/seeg_sig_channels_2026_04_24")

PHASES = ("aud", "go", "resp")


def read_coverage(csv_path: Path) -> dict[str, dict]:
    """Map bids_id -> coverage row (PS task only)."""
    out: dict[str, dict] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["task"] != "phoneme_sequencing":
                continue
            out[row["bids_id"]] = row
    return out


def count_sig(mask_path: Path) -> tuple[int, int]:
    """Return (n_channels_any_sig, n_channels_total)."""
    evoked_list = mne.read_evokeds(mask_path, verbose="ERROR")
    evoked = evoked_list[0] if isinstance(evoked_list, list) else evoked_list
    data = evoked.data
    n_total = data.shape[0]
    n_sig = int((data > 0).any(axis=1).sum())
    return n_sig, n_total


def audit_patient(bids_id: str, stats_root: Path) -> dict | None:
    row: dict = {"bids_id": bids_id}
    for phase in PHASES:
        path = stats_root / f"{bids_id}_{phase}_mask-ave.fif"
        if not path.exists():
            return None
        n_sig, n_total = count_sig(path)
        row[f"n_sig_{phase}"] = n_sig
        row["n_ch_stats"] = n_total  # identical across phases
    return row


def enumerate_patients_with_stats(stats_root: Path) -> list[str]:
    """Return sorted list of D-IDs that have all 3 mask files."""
    all_ids = sorted({p.name.split("_")[0] for p in stats_root.glob("D*_*_mask-ave.fif")})
    complete: list[str] = []
    for did in all_ids:
        if all((stats_root / f"{did}_{ph}_mask-ave.fif").exists() for ph in PHASES):
            complete.append(did)
    return complete


def flag_usability(frac_sig_resp: float) -> str:
    if frac_sig_resp < 0.05:
        return "likely_bad"
    if frac_sig_resp < 0.20:
        return "low"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-root", type=Path, default=STATS_ROOT)
    ap.add_argument("--coverage-csv", type=Path, default=COVERAGE_CSV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    coverage = read_coverage(args.coverage_csv)
    patients = enumerate_patients_with_stats(args.stats_root)
    print(f"found {len(patients)} D-patients with complete stats under {args.stats_root}")

    rows: list[dict] = []
    for did in patients:
        audit = audit_patient(did, args.stats_root)
        if audit is None:
            continue
        cov = coverage.get(did)
        n_ch_stats = int(audit["n_ch_stats"])
        n_sig_resp = int(audit["n_sig_resp"])
        frac_sig_resp = n_sig_resp / n_ch_stats if n_ch_stats else 0.0
        row = {
            "bids_id": did,
            "recon_id": cov["recon_id"] if cov else "",
            "n_elec_recon": cov["n_elec"] if cov else "",
            "n_ch_stats": n_ch_stats,
            "lh_tier1": cov["lh_tier1"] if cov else "",
            "rh_tier1": cov["rh_tier1"] if cov else "",
            "n_sig_aud": int(audit["n_sig_aud"]),
            "n_sig_go": int(audit["n_sig_go"]),
            "n_sig_resp": n_sig_resp,
            "frac_sig_aud": round(int(audit["n_sig_aud"]) / n_ch_stats, 4),
            "frac_sig_resp": round(frac_sig_resp, 4),
            "usability_flag": flag_usability(frac_sig_resp),
        }
        rows.append(row)
        print(f"  {did}: {n_ch_stats}ch  aud={audit['n_sig_aud']}  go={audit['n_sig_go']}  "
              f"resp={n_sig_resp} ({100*frac_sig_resp:.1f}%)  -> {row['usability_flag']}")

    csv_path = out_dir / "per_patient.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Also flag PS patients missing from stats
    ps_patients_in_cohort = sorted(coverage.keys())
    missing = [p for p in ps_patients_in_cohort if p not in patients]

    summary = out_dir / "summary.md"
    total = len(rows)
    ok = sum(1 for r in rows if r["usability_flag"] == "ok")
    low = sum(1 for r in rows if r["usability_flag"] == "low")
    bad = sum(1 for r in rows if r["usability_flag"] == "likely_bad")
    top_resp = sorted(rows, key=lambda r: -r["n_sig_resp"])[:10]
    top_aud = sorted(rows, key=lambda r: -r["n_sig_aud"])[:10]
    lines: list[str] = [
        "# A1 — sEEG D-patient sig-channel inventory (2026-04-24)",
        "",
        "Reads pre-computed cluster-corrected permutation-test masks from ",
        f"`{STATS_ROOT.name}/` (Cogan Lab upstream pipeline).",
        "`n_sig_<phase>` = count of channels with any significant timepoint in that phase.",
        "",
        "## Coverage",
        "",
        f"- PS D-patients in cohort audit: {len(ps_patients_in_cohort)}",
        f"- PS D-patients with complete stats: {total}",
        f"- Missing stats (flagged 'unknown' usability): {len(missing)} — {', '.join(missing) or 'none'}",
        "",
        "## Usability-proxy tiering (response-phase)",
        "",
        "Rule: frac_sig_resp ≥ 20% → `ok`; 5–20% → `low`; < 5% → `likely_bad`.",
        "First-pass proxy only — Nanlin's authoritative tiering overrides.",
        "",
        f"- ok: **{ok}**",
        f"- low: {low}",
        f"- likely_bad: {bad}",
        "",
        "## Top 10 PS D-patients by production-phase sig channels (`n_sig_resp`)",
        "",
        "| bids_id | n_ch_stats | lh_tier1 | rh_tier1 | n_sig_aud | n_sig_resp | frac_resp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in top_resp:
        lines.append(
            f"| {r['bids_id']} | {r['n_ch_stats']} | {r['lh_tier1']} | {r['rh_tier1']} | "
            f"{r['n_sig_aud']} | {r['n_sig_resp']} | {r['frac_sig_resp']} |"
        )
    lines += [
        "",
        "## Top 10 PS D-patients by auditory-phase sig channels (`n_sig_aud`)",
        "",
        "| bids_id | n_ch_stats | lh_tier1 | rh_tier1 | n_sig_aud | n_sig_resp | frac_aud |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in top_aud:
        lines.append(
            f"| {r['bids_id']} | {r['n_ch_stats']} | {r['lh_tier1']} | {r['rh_tier1']} | "
            f"{r['n_sig_aud']} | {r['n_sig_resp']} | {r['frac_sig_aud']} |"
        )
    lines.append("")
    summary.write_text("\n".join(lines))

    print(f"\nwrote {csv_path}  ({len(rows)} rows)")
    print(f"wrote {summary}")
    print(f"usability: ok={ok} low={low} likely_bad={bad} unknown={len(missing)}")


if __name__ == "__main__":
    main()
