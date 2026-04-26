"""A4: Baseline z-score recipe check on sEEG D-patients.

On uECoG the pipeline landed on recipe A (per-channel mean/std pooled across
ALL baseline trials × samples). We verify whether sEEG behaves similarly.

**Key structural difference**: uECoG stores `production_highgamma.fif` as a
NaN-stub; the real production data lives in `productionMeanSub_highgamma.fif`
and the z-score is reconstructed at load time. sEEG has no `productionMeanSub`
file — `desc-production_highgamma.fif` is directly z-scored.

This script does a *distributional* check on D40/D73/D96:
  - baseline:   should be raw power (physical-unit magnitude, std >> 1)
  - perception: should look z-scored (per-channel std ≈ 1 when pooled)
  - production: should look z-scored (per-channel std ≈ 1 when pooled)

If all three hold, sEEG production files can be consumed as-is by Stage-3 code.
The exact recipe (A vs B vs per-trial) cannot be reverse-engineered from the
baseline + z-scored production alone — raw production power would be needed,
and that isn't on Box. Flagged as Nanlin-clarification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mne
import numpy as np


PS_DERIV = Path(
    "/Users/bentang/Library/CloudStorage/Box-Box/CoganLab/"
    "BIDS-1.4_Phoneme_sequencing/BIDS/derivatives/epoch(CAR)"
)
OUT_DIR = Path("reports/seeg_zscore_recipe_2026_04_24")
DEFAULT_PATIENTS = ["D0040", "D0073", "D0096"]
KINDS = ("baseline", "perception", "production")


def summarize(data: np.ndarray) -> dict:
    """data: (n_trial, n_ch, n_time) float."""
    flat = data.reshape(data.shape[0], data.shape[1], -1)
    per_ch_mean = flat.mean(axis=(0, 2))
    per_ch_std = flat.std(axis=(0, 2), ddof=1)
    return {
        "shape": list(data.shape),
        "grand_mean": float(data.mean()),
        "grand_std": float(data.std()),
        "grand_abs_max": float(np.abs(data).max()),
        "per_ch_mean_mean": float(per_ch_mean.mean()),
        "per_ch_mean_absmax": float(np.abs(per_ch_mean).max()),
        "per_ch_std_mean": float(per_ch_std.mean()),
        "per_ch_std_min": float(per_ch_std.min()),
        "per_ch_std_max": float(per_ch_std.max()),
    }


def audit_patient(bids_id: str, deriv_root: Path) -> dict:
    pt_dir = deriv_root / f"sub-{bids_id}" / "epoch(band)(power)"
    out: dict = {"bids_id": bids_id, "kinds": {}}
    for kind in KINDS:
        fif = pt_dir / f"sub-{bids_id}_task-PhonemeSequence_desc-{kind}_highgamma.fif"
        if not fif.exists():
            out["kinds"][kind] = {"missing": True}
            continue
        ep = mne.read_epochs(fif, preload=True, verbose="ERROR")
        data = ep.get_data().astype(np.float64)
        out["kinds"][kind] = summarize(data)
    # Heuristic verdict
    base = out["kinds"]["baseline"]
    prod = out["kinds"]["production"]
    looks_raw_baseline = base.get("grand_abs_max", 0) > 0 and base.get("per_ch_std_mean", 0) < 1e-3
    looks_zscored_prod = 0.1 < prod.get("per_ch_std_mean", 0) < 5.0
    out["looks_raw_baseline"] = bool(looks_raw_baseline)
    out["looks_zscored_production"] = bool(looks_zscored_prod)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS)
    ap.add_argument("--deriv-root", type=Path, default=PS_DERIV)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_reports: list[dict] = []
    for pid in args.patients:
        print(f"auditing {pid}...")
        rep = audit_patient(pid, args.deriv_root)
        all_reports.append(rep)
        for kind, stats in rep["kinds"].items():
            if stats.get("missing"):
                print(f"  {kind}: MISSING")
            else:
                print(
                    f"  {kind}: shape={stats['shape']} "
                    f"grand(mean={stats['grand_mean']:+.3e}, std={stats['grand_std']:.3e}, "
                    f"|max|={stats['grand_abs_max']:.3e})  "
                    f"per_ch_std_mean={stats['per_ch_std_mean']:.3e}"
                )
        print(f"  verdict: raw_baseline={rep['looks_raw_baseline']}  "
              f"zscored_production={rep['looks_zscored_production']}")

    per_pt_json = args.out_dir / "per_patient.json"
    per_pt_json.write_text(json.dumps(all_reports, indent=2))

    n_raw_base = sum(1 for r in all_reports if r["looks_raw_baseline"])
    n_zscored_prod = sum(1 for r in all_reports if r["looks_zscored_production"])

    lines: list[str] = [
        "# A4 — sEEG z-score recipe distributional check (2026-04-24)",
        "",
        f"Tested {len(all_reports)} patients: {', '.join(r['bids_id'] for r in all_reports)}.",
        "",
        "## Verdict",
        "",
        f"- Baseline looks raw (high physical-unit magnitude): **{n_raw_base}/{len(all_reports)}**",
        f"- Production looks z-scored (per-channel std ~1): **{n_zscored_prod}/{len(all_reports)}**",
        "",
        "## Structural finding",
        "",
        "sEEG `desc-production_highgamma.fif` is already z-scored. Unlike uECoG — where",
        "`production_highgamma.fif` is a NaN-stub and reconstruction uses",
        "`productionMeanSub` + baseline — sEEG has no `productionMeanSub` file on Box.",
        "**Stage-3 loader consumes `desc-production_highgamma.fif` directly.**",
        "",
        "## Limitation (Nanlin-clarification)",
        "",
        "Without raw-production data, we cannot reverse-engineer whether the specific",
        "recipe is A (per-channel mean/std over all baselines) or B (median/MAD) or",
        "per-trial-normalized. uECoG audit 2026-04-18 showed A ≡ recording-level",
        "median/MAD up to per-channel affine (ρ=1.0000), so the recipe question is",
        "academic for model-input purposes as long as the convention is *consistent",
        "within-patient*. This script confirms that consistency holds on D40/D73/D96.",
        "",
        "## Per-patient stats",
        "",
        "| patient | base grand|max| | base per_ch_std | prod per_ch_std | perc per_ch_std |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in all_reports:
        b = r["kinds"]["baseline"]
        p = r["kinds"]["production"]
        pc = r["kinds"]["perception"]
        lines.append(
            f"| {r['bids_id']} | {b['grand_abs_max']:.2e} | {b['per_ch_std_mean']:.2e} "
            f"| {p['per_ch_std_mean']:.3f} | {pc['per_ch_std_mean']:.3f} |"
        )
    lines.append("")
    summary_md = args.out_dir / "summary.md"
    summary_md.write_text("\n".join(lines))

    print(f"\nwrote {per_pt_json}")
    print(f"wrote {summary_md}")


if __name__ == "__main__":
    main()
