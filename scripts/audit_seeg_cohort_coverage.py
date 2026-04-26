"""sEEG D-cohort Stage-3 scoping audit.

Cross-references Box BIDS task folders with ECoG_Recon and the per-patient
pre-baked BNA CSVs (patient-space `aparc.BN_atlas+aseg.mgz`, 3mm-radius
argmax) to count Tier-1 speech-motor-parcel coverage per D-patient.

Tier-1 is the frozen 15-parcel set from `token_spec.DEFAULT_BASE_PARCELS`
(LH names only — `_L` / `_R` suffix handled here).

Outputs a ranked table CSV + summary counts.

Run from repo root:
    .venv/bin/python scripts/audit_seeg_cohort_coverage.py \
        --out reports/seeg_cohort_scoping_2026_04_21/coverage.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from speech_decoding.v14.token_spec import DEFAULT_BASE_PARCELS

BOX_ROOT = Path("/Users/bentang/Library/CloudStorage/Box-Box")
RECON_ROOT = BOX_ROOT / "ECoG_Recon"
BIDS_ROOTS = {
    "phoneme_sequencing": BOX_ROOT / "CoganLab/BIDS-1.4_Phoneme_sequencing/BIDS",
    "lex_delay": BOX_ROOT / "CoganLab/BIDS-1.0_LexicalDecRepDelay/BIDS",
    "lex_no_delay": BOX_ROOT / "CoganLab/BIDS-1.0_LexicalDecRepNoDelay/BIDS",
    "sentence_rep": BOX_ROOT / "CoganLab/BIDS-1.4_SentenceRep/BIDS",
}

TIER1_NAMES = [name for name, _ in DEFAULT_BASE_PARCELS]
TIER1_L = {f"{n}_L" for n in TIER1_NAMES}
TIER1_R = {f"{n}_R" for n in TIER1_NAMES}


def bids_to_recon_id(sub_name: str) -> str:
    """sub-D0023 -> D23 (strip leading zeros on the numeric part)."""

    pt = sub_name.replace("sub-", "")
    return "D" + str(int(pt[1:]))


def count_coverage(csv_path: Path) -> tuple[int, int, int]:
    """Return (N_total_electrodes, N_tier1_LH, N_tier1_RH) from a 3mm BNA CSV."""

    total = lh1 = rh1 = 0
    with csv_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            argmax = parts[0]
            total += 1
            if argmax in TIER1_L:
                lh1 += 1
            elif argmax in TIER1_R:
                rh1 += 1
    return total, lh1, rh1


def audit_task(task: str, bids_root: Path) -> list[dict]:
    rows = []
    for sub in sorted(bids_root.glob("sub-D*")):
        bids_id = sub.name.replace("sub-", "")
        recon_id = bids_to_recon_id(sub.name)
        recon_dir = RECON_ROOT / recon_id
        csvf = recon_dir / "elec_recon" / f"{recon_id}_elec_location_radius_3mm_aparc.BN_atlas+aseg.mgz.csv"

        has_pial = (recon_dir / "surf" / "lh.pial").exists()
        has_sphere = (recon_dir / "surf" / "lh.sphere.reg").exists()
        has_ras = list((recon_dir / "elec_recon").glob("*_elec_locations_RAS*")) if (recon_dir / "elec_recon").exists() else []
        projectable = has_pial and has_sphere and bool(has_ras)

        if not csvf.exists():
            rows.append({
                "task": task, "bids_id": bids_id, "recon_id": recon_id,
                "projectable": int(projectable),
                "n_elec": None, "lh_tier1": None, "rh_tier1": None, "status": "no_bna_csv",
            })
            continue

        total, lh1, rh1 = count_coverage(csvf)
        if lh1 == 0 and rh1 == 0:
            status = "no_tier1"
        elif lh1 > 0 and rh1 == 0:
            status = "lh_only"
        elif rh1 > 0 and lh1 == 0:
            status = "rh_only"
        else:
            status = "bilateral"
        rows.append({
            "task": task, "bids_id": bids_id, "recon_id": recon_id,
            "projectable": int(projectable),
            "n_elec": total, "lh_tier1": lh1, "rh_tier1": rh1, "status": status,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports/seeg_cohort_scoping_2026_04_21/coverage.csv"))
    args = ap.parse_args()

    all_rows: list[dict] = []
    for task, root in BIDS_ROOTS.items():
        if not root.exists():
            print(f"[warn] skipping {task}: {root} missing")
            continue
        all_rows.extend(audit_task(task, root))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "bids_id", "recon_id", "projectable", "n_elec", "lh_tier1", "rh_tier1", "status"])
        w.writeheader()
        w.writerows(all_rows)

    # Summary per task (real CSVs only, projectable)
    for task in BIDS_ROOTS:
        task_rows = [r for r in all_rows if r["task"] == task and r["n_elec"] is not None]
        if not task_rows:
            continue
        lh10 = sum(1 for r in task_rows if r["lh_tier1"] >= 10)
        rh10 = sum(1 for r in task_rows if r["rh_tier1"] >= 10)
        any10 = sum(1 for r in task_rows if max(r["lh_tier1"], r["rh_tier1"]) >= 10)
        zero = sum(1 for r in task_rows if r["lh_tier1"] == 0 and r["rh_tier1"] == 0)
        print(f"{task}: N={len(task_rows)} | LH≥10={lh10} | RH≥10={rh10} | any≥10={any10} | zero={zero}")

    print(f"\nWrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
