# pyright: reportMissingImports=false
"""BT shaft/depth geometry contract audit (Stage-0 §3).

Produces the three CSVs + a populated `README.md` that freeze the canonical
BT shaft parser, contact-order orientation policy, and signed-depth-feature
gate. Reads only public BT metadata (no neural data):

- ``electrode_labels/sub_{N}/electrode_labels.json`` — raw label list per subject.
- ``localization/sub_{N}/depth-wm.csv`` — Electrode, Hemisphere, DK label, L, P, I native.
- ``corrupted_elec.json`` — corrupted labels per subject (informational).

Cohort: ``BT_FULL_SESSIONS`` subject IDs minus S5 (DK-first-pass freeze 2026-05-13:
S5 dropped for frontal lesion, not in leaderboard subset).

Outputs (in ``--out-dir``):

- ``shaft_contact_inventory.csv`` — one row per electrode_label.
- ``shaft_label_transition_summary.csv`` — one row per (subject, shaft).
- ``contact_order_orientation_audit.csv`` — one row per (subject, shaft) with ≥3 contacts.
- ``parse_anomalies.csv`` — labels parse_shaft cannot resolve.
- ``README.md`` — frozen feature contract + audit summary.

Usage:
    .venv/bin/python scripts/neuroprobe/audit_shaft_geometry.py \\
        --bt-root /work/ht203/data/braintreebank \\
        --out-dir reports/neuroprobe_stage0_shaft_depth_geometry_$(date +%Y_%m_%d)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from speech_decoding.extractors.reference import parse_shaft
from speech_decoding.studies.braintreebank.anatomy import clean_bt_electrode_label
from speech_decoding.studies.braintreebank.manifest import BT_FULL_SESSIONS

DROPPED_SUBJECTS: tuple[int, ...] = (5,)
"""DK-first-pass freeze 2026-05-13: S5 has BT-flagged frontal lesion; not in
leaderboard subset {1,2,3,4,7,10}; dropping is free."""

# Two linearity thresholds. Above ``LINEARITY_OK`` a shaft is treated as well-
# modelled by its first principal axis (suffix↔PC1 correlation is interpretable);
# between ``LINEARITY_WARN`` and ``LINEARITY_OK`` we still report PC1 metrics but
# flag the shaft; below ``LINEARITY_WARN`` orientation is unreliable.
LINEARITY_OK = 0.97
LINEARITY_WARN = 0.90

# Below this absolute Pearson(suffix, PC1) we treat the integer suffix as NOT a
# monotonic projection of physical position along the shaft. Conservative
# threshold — sEEG shafts almost always have |r| > 0.99 when linear.
MONOTONIC_R_OK = 0.98


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bt_root = args.bt_root.resolve()
    subject_ids = sorted({sid for sid, _ in BT_FULL_SESSIONS} - set(DROPPED_SUBJECTS))

    corrupted = _load_corrupted(bt_root)
    inventory_rows: list[dict] = []
    anomaly_rows: list[dict] = []

    for sid in subject_ids:
        raw_labels = _load_raw_labels(bt_root, sid)
        anatomy = _load_anatomy(bt_root, sid)
        inv, anomalies = _build_inventory(sid, raw_labels, anatomy, corrupted.get(sid, set()))
        inventory_rows.extend(inv)
        anomaly_rows.extend(anomalies)

    inventory = pd.DataFrame(inventory_rows)
    transitions = _build_transitions(inventory)
    orientation = _build_orientation(inventory)
    anomalies = pd.DataFrame(anomaly_rows)

    inventory.to_csv(out_dir / "shaft_contact_inventory.csv", index=False)
    transitions.to_csv(out_dir / "shaft_label_transition_summary.csv", index=False)
    orientation.to_csv(out_dir / "contact_order_orientation_audit.csv", index=False)
    anomalies.to_csv(out_dir / "parse_anomalies.csv", index=False)

    readme = _render_readme(
        bt_root=bt_root,
        subject_ids=subject_ids,
        inventory=inventory,
        transitions=transitions,
        orientation=orientation,
        anomalies=anomalies,
    )
    (out_dir / "README.md").write_text(readme)

    print(f">> wrote {out_dir / 'shaft_contact_inventory.csv'} ({len(inventory)} rows)")
    print(f">> wrote {out_dir / 'shaft_label_transition_summary.csv'} ({len(transitions)} rows)")
    print(f">> wrote {out_dir / 'contact_order_orientation_audit.csv'} ({len(orientation)} rows)")
    print(f">> wrote {out_dir / 'parse_anomalies.csv'} ({len(anomalies)} rows)")
    print(f">> wrote {out_dir / 'README.md'}")


def _load_corrupted(bt_root: Path) -> dict[int, set[str]]:
    path = bt_root / "corrupted_elec.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[int, set[str]] = {}
    for key, labels in raw.items():
        if not key.startswith("sub_"):
            continue
        sid = int(key.removeprefix("sub_"))
        out[sid] = {clean_bt_electrode_label(label) for label in labels}
    return out


def _load_raw_labels(bt_root: Path, subject_id: int) -> list[str]:
    path = bt_root / "electrode_labels" / f"sub_{subject_id}" / "electrode_labels.json"
    if not path.exists():
        raise FileNotFoundError(f"missing electrode labels: {path}")
    return list(json.loads(path.read_text()))


def _load_anatomy(bt_root: Path, subject_id: int) -> pd.DataFrame:
    path = bt_root / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing depth-wm.csv: {path}")
    df = pd.read_csv(path)
    df["Electrode"] = df["Electrode"].map(clean_bt_electrode_label)
    return df


def _is_trigger(label: str) -> bool:
    upper = label.upper()
    return upper.startswith("DC") or upper.startswith("TRIG")


def _build_inventory(
    subject_id: int,
    raw_labels: list[str],
    anatomy: pd.DataFrame,
    corrupted: set[str],
) -> tuple[list[dict], list[dict]]:
    anatomy_by_label = anatomy.set_index("Electrode", drop=False)
    inventory: list[dict] = []
    anomalies: list[dict] = []
    for raw_idx, raw_label in enumerate(raw_labels):
        cleaned = clean_bt_electrode_label(raw_label)
        shaft, contact_index = parse_shaft(cleaned)
        trigger = _is_trigger(cleaned)
        parse_status = "ok"
        if contact_index is None:
            parse_status = "no_numeric_suffix"
            anomalies.append(
                {
                    "subject_id": subject_id,
                    "raw_label": raw_label,
                    "cleaned_label": cleaned,
                    "raw_index": raw_idx,
                    "is_trigger": trigger,
                    "reason": parse_status,
                }
            )
        anat_row = anatomy_by_label.loc[cleaned] if cleaned in anatomy_by_label.index else None
        if anat_row is None and not trigger:
            anomalies.append(
                {
                    "subject_id": subject_id,
                    "raw_label": raw_label,
                    "cleaned_label": cleaned,
                    "raw_index": raw_idx,
                    "is_trigger": False,
                    "reason": "missing_depth_wm_row",
                }
            )
        hemisphere = _safe_get(anat_row, "Hemisphere")
        dk_label = _safe_get(anat_row, "DesikanKilliany")
        L = _safe_get(anat_row, "L")
        P = _safe_get(anat_row, "P")
        I = _safe_get(anat_row, "I")
        inventory.append(
            {
                "subject_id": subject_id,
                "raw_label": raw_label,
                "electrode_label": cleaned,
                "raw_index": raw_idx,
                "shaft": shaft,
                "contact_index": contact_index,
                "hemisphere": hemisphere,
                "DesikanKilliany": dk_label,
                "L": L,
                "P": P,
                "I": I,
                "is_corrupted": cleaned in corrupted,
                "is_trigger": trigger,
                "parse_status": parse_status,
            }
        )
    return inventory, anomalies


def _safe_get(row: pd.Series | None, column: str):
    if row is None:
        return np.nan
    value = row.get(column, np.nan) if hasattr(row, "get") else row[column]
    return value


def _build_transitions(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    parsed = inventory[(~inventory["is_trigger"]) & inventory["contact_index"].notna()]
    for (sid, shaft), group in parsed.groupby(["subject_id", "shaft"], sort=True):
        group = group.sort_values("contact_index")
        labels = group["DesikanKilliany"].fillna("unknown").astype(str).tolist()
        hemis = group["hemisphere"].fillna("unknown").astype(str).tolist()
        unique_dk = sorted(set(labels))
        hemispheres_spanned = sorted(set(hemis))
        transitions = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        if labels:
            counts = Counter(labels)
            dom_label, dom_n = counts.most_common(1)[0]
            dom_coverage = dom_n / len(labels)
        else:
            dom_label, dom_coverage = "", 0.0
        rows.append(
            {
                "subject_id": sid,
                "shaft": shaft,
                "n_contacts": len(group),
                "hemispheres_spanned": ";".join(hemispheres_spanned),
                "cross_hemisphere": len(hemispheres_spanned) > 1,
                "unique_DK_labels": ";".join(unique_dk),
                "n_unique_DK_labels": len(unique_dk),
                "n_DK_transitions": transitions,
                "dominant_DK_label": dom_label,
                "dominant_label_coverage": round(dom_coverage, 4),
            }
        )
    return pd.DataFrame(rows)


def _build_orientation(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    parsed = inventory[
        (~inventory["is_trigger"])
        & inventory["contact_index"].notna()
        & inventory[["L", "P", "I"]].notna().all(axis=1)
    ]
    # Per-subject centroid: proxy for "external" direction (contacts far from
    # centroid are typically more superficial / closer to skull entry).
    subject_centroids = parsed.groupby("subject_id")[["L", "P", "I"]].mean()

    for (sid, shaft), group in parsed.groupby(["subject_id", "shaft"], sort=True):
        n = len(group)
        if n < 3:
            continue
        group = group.sort_values("contact_index").reset_index(drop=True)
        coords = group[["L", "P", "I"]].to_numpy(dtype=float)
        suffixes = group["contact_index"].to_numpy(dtype=float)

        centered = coords - coords.mean(axis=0, keepdims=True)
        _, sv, vt = np.linalg.svd(centered, full_matrices=False)
        var = (sv ** 2) / max(1, n - 1)
        var_ratio_1 = float(var[0] / var.sum()) if var.sum() > 0 else 0.0
        pc1 = vt[0]
        pc1_proj = centered @ pc1

        pearson = float(np.corrcoef(suffixes, pc1_proj)[0, 1]) if n >= 2 else np.nan
        monotonic_r_ok = abs(pearson) >= MONOTONIC_R_OK

        diffs_pc1 = np.diff(pc1_proj * np.sign(pearson)) if not np.isnan(pearson) else np.diff(pc1_proj)
        n_inversions = int(np.sum(diffs_pc1 < 0))

        neighbor_steps_mm = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        max_step_mm = float(neighbor_steps_mm.max()) if neighbor_steps_mm.size else np.nan
        mean_step_mm = float(neighbor_steps_mm.mean()) if neighbor_steps_mm.size else np.nan

        centroid = subject_centroids.loc[sid].to_numpy(dtype=float)
        dist_to_centroid = np.linalg.norm(coords - centroid, axis=1)
        suffix_vs_centroid_r = (
            float(np.corrcoef(suffixes, dist_to_centroid)[0, 1]) if n >= 2 else np.nan
        )
        if np.isnan(suffix_vs_centroid_r) or abs(suffix_vs_centroid_r) < 0.5:
            deeper_end = "unknown"
        elif suffix_vs_centroid_r > 0:
            deeper_end = "lower_suffix"
        else:
            deeper_end = "higher_suffix"

        rows.append(
            {
                "subject_id": sid,
                "shaft": shaft,
                "n_contacts": n,
                "pca_var_ratio_1": round(var_ratio_1, 5),
                "linear_ok": var_ratio_1 >= LINEARITY_OK,
                "linear_warn": var_ratio_1 >= LINEARITY_WARN and var_ratio_1 < LINEARITY_OK,
                "pearson_suffix_pc1": round(pearson, 5) if not np.isnan(pearson) else np.nan,
                "monotonic_r_ok": monotonic_r_ok,
                "n_inversions": n_inversions,
                "mean_neighbor_step_mm": round(mean_step_mm, 4) if not np.isnan(mean_step_mm) else np.nan,
                "max_neighbor_step_mm": round(max_step_mm, 4) if not np.isnan(max_step_mm) else np.nan,
                "pearson_suffix_centroid_dist": round(suffix_vs_centroid_r, 5)
                if not np.isnan(suffix_vs_centroid_r)
                else np.nan,
                "deeper_end": deeper_end,
            }
        )
    return pd.DataFrame(rows)


def _render_readme(
    *,
    bt_root: Path,
    subject_ids: list[int],
    inventory: pd.DataFrame,
    transitions: pd.DataFrame,
    orientation: pd.DataFrame,
    anomalies: pd.DataFrame,
) -> str:
    today = date.today().isoformat()
    n_contacts = len(inventory)
    n_triggers = int(inventory["is_trigger"].sum())
    n_corrupted = int(inventory["is_corrupted"].sum())
    n_parse_fail = int((inventory["parse_status"] != "ok").sum())
    n_shafts = len(transitions)
    multi_label = transitions["n_unique_DK_labels"] > 1
    pct_multi_label = 100.0 * multi_label.mean() if len(transitions) else 0.0
    multi_label_contacts = int(transitions.loc[multi_label, "n_contacts"].sum())
    contacts_on_multi = (
        100.0 * multi_label_contacts / max(1, transitions["n_contacts"].sum())
    )
    cross_hemi = int(transitions["cross_hemisphere"].sum()) if len(transitions) else 0

    if len(orientation):
        pct_linear_ok = 100.0 * orientation["linear_ok"].mean()
        pct_monotonic = 100.0 * orientation["monotonic_r_ok"].mean()
        deeper_consistency = orientation.groupby("subject_id")["deeper_end"].agg(
            lambda s: s.value_counts(normalize=True).max()
        )
        deeper_majority = orientation.groupby("subject_id")["deeper_end"].agg(
            lambda s: s.value_counts().idxmax()
        )
        per_subject = pd.DataFrame(
            {
                "n_shafts_audited": orientation.groupby("subject_id").size(),
                "majority_deeper_end": deeper_majority,
                "majority_consistency": deeper_consistency.round(3),
            }
        )
        per_subject_md = per_subject.to_markdown()
        cross_subject_majority = deeper_majority.value_counts().to_dict()
    else:
        pct_linear_ok = pct_monotonic = 0.0
        per_subject_md = "_(no orientation rows)_"
        cross_subject_majority = {}

    # Headline decisions
    signed_depth_allowed = (
        len(orientation) > 0
        and pct_monotonic >= 95.0
        and len(cross_subject_majority) == 1
    )

    return f"""# BT shaft/depth geometry contract

**Date**: {today}
**Audit script**: `scripts/neuroprobe/audit_shaft_geometry.py`
**BT root**: `{bt_root}`
**Cohort**: full BT minus S5 (DK-first-pass freeze 2026-05-13). Subjects = {subject_ids}.

This contract freezes:
1. the canonical shaft parser,
2. the contact-order orientation policy,
3. which shaft/depth features are admissible v14 inputs.

## Headline

- **Contacts audited**: {n_contacts} ({n_triggers} trigger, {n_corrupted} corrupted, {n_parse_fail} parse-failed).
- **Shafts parsed**: {n_shafts}.
- **Multi-label shafts**: {multi_label.sum()} of {n_shafts} ({pct_multi_label:.1f}%); these contain {multi_label_contacts} contacts ({contacts_on_multi:.1f}% of all parsed contacts). One shaft commonly traverses multiple DK regions — confirms that DK alone discards within-shaft position.
- **Cross-hemisphere shafts**: {cross_hemi}. DK-first-pass audit expects 0/140; deviation = parser bug.
- **Linear shafts** (PCA PC1 var-ratio ≥ {LINEARITY_OK}): {pct_linear_ok:.1f}% of orientation-audited shafts.
- **Suffix-monotonic shafts** (|r(suffix, PC1)| ≥ {MONOTONIC_R_OK}): {pct_monotonic:.1f}%.

## Frozen decisions

### 1. Canonical shaft parser
`parse_shaft(channel_name)` (regex `^(.*?)(\\d+)$`) defined at
`src/speech_decoding/extractors/reference.py:31`. Splits trailing integer
suffix as `contact_index` and preceding stem as `shaft`. Labels without a
numeric suffix → `(name, None)`; the audit reports them as
`parse_status="no_numeric_suffix"` and excludes them from shaft groupings.
Trigger channels (`DC*`, `TRIG*`) parse normally but are flagged
`is_trigger=True` and excluded from all reference / depth computations.

### 2. Contact-order orientation
Per-shaft principal axis (SVD of mean-centered (L, P, I) native coordinates)
gives PC1; `r(contact_index, PC1)` measures whether the integer suffix is a
monotonic projection of physical position. Sign of `r` tells the suffix→depth
direction *up to sign*.

The cross-subject majority of `deeper_end` per shaft (from
`pearson_suffix_centroid_dist`) is `{cross_subject_majority}`.

### 3. Signed-depth-feature policy
{"**ALLOWED** — orientation is consistent across the cohort. v14 may use" if signed_depth_allowed else "**FORBIDDEN by default** — orientation is not consistent across the cohort. v14 must use"}
orientation-invariant within-shaft features only:

- ordinal `contact_index` ✗ disallowed as a feature (subject-specific magnitude).
- normalized `contact_index / max_index` ✗ disallowed (sign-ambiguous).
- centered normalized position `2 * (i - 0.5*(max+min)) / range` ✗ disallowed (sign-ambiguous).
- same-shaft adjacency mask ✓ allowed.
- relative offset `|i - j|` on same shaft ✓ allowed.
- local-reference provenance metadata ✓ allowed (shaft-CAR / Laplacian indices).

### 4. Reference scheme compatibility
- **shaftCAR (R2, v14 default)**: orientation-INDEPENDENT. Uses `parse_shaft`
  stem grouping only. Cleared regardless of orientation outcome.
- **shaftLaplacian (R4, upstream Linear-Lap+spec parity)**: orientation-
  INDEPENDENT in BrainBERT's symmetric form (`S′ᵢ = Sᵢ − ½(Sᵢ₋₁ + Sᵢ₊₁)`).
  Boundary contacts fall back to one-sided difference (`S′ᵢ = Sᵢ − Sᵢ₋₁` or
  `Sᵢ − Sᵢ₊₁`) — the sign asymmetry is absorbed in spectral magnitude.
  Cleared.
- **Bipolar (R3)**: orientation-INDEPENDENT for inputs but creates virtual
  midpoint channels with interpolated coords. Already rejected for v14 on
  architectural grounds (parcel `support[i,p]` ill-defined at midpoints).
- **Signed depth embeddings**: gated on §3 above.

### 5. Rejected encodings
- **Learned `shaft_id` embedding**: rejected. `shaft` strings are patient-
  specific (`OFa12` on sub_1 ≠ `OFa12` on sub_2). A learned embedding would
  pattern-match subject identity. Permitted only as grouping key for local
  reference construction, never as a feature.
- **Raw `contact_index` as a numeric feature**: rejected. Sign-ambiguous and
  subject-specific magnitude.
- **`hemisphere` as a one-hot input feature**: rejected as a v14 default; the
  Graphormer-bias / parcel-id-tagged latents carry hemisphere via DK label
  scope (`lh_*` / `rh_*` parcels). May be revisited as a P2 ablation cell.

## Per-subject orientation summary

{per_subject_md}

## Anomalies

- Parse anomalies: `parse_anomalies.csv` ({len(anomalies)} rows).
- Subjects: {subject_ids}.
- Triggers excluded (informational, count above).

## Files

- `shaft_contact_inventory.csv` — one row per (subject, raw label).
- `shaft_label_transition_summary.csv` — one row per (subject, shaft).
- `contact_order_orientation_audit.csv` — one row per (subject, shaft) with ≥3 contacts.
- `parse_anomalies.csv` — labels parse_shaft cannot resolve (incl. triggers and missing-anatomy rows).

## Unfreeze triggers

- Chris MNI lands → re-run orientation audit against true brain-surface skull
  intersection; signed-depth policy may relax.
- BNA subcortical scope decision (parked blocker, 2026-05-12) → if v14 extends
  beyond DK Tier-1, re-validate non-cortical shafts (hippocampus depth probes
  with single-shaft trajectories).
- Subject pool changes → re-run; per-subject orientation consistency is
  load-bearing.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--bt-root",
        type=Path,
        required=True,
        help="BrainTreebank root directory (parent of `localization/`).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write the four CSVs + README.md.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
