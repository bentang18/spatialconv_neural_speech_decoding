"""Public BrainTreebank anatomy support.

BrainTreebank currently exposes DK/Destrieux-style per-electrode labels and
template plotting coordinates. The label path is useful as a hard-parcel
ablation, but it is not fsaverage/BNA support.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SupportKind = tp.Literal["soft_surface_bna", "hard_public_bt_label", "none"]
DEFAULT_BT_LABEL_COLUMN = "DesikanKilliany"


@dataclass(frozen=True)
class HardLabelSupport:
    """One-hot electrode support over public BrainTreebank anatomy labels."""

    kind: SupportKind
    electrode_labels: tuple[str, ...]
    parcel_labels: tuple[str, ...]
    support: np.ndarray
    label_column: str


def load_public_bt_anatomy(
    bt_root: str | Path,
    subject_id: int,
    *,
    label_column: str = DEFAULT_BT_LABEL_COLUMN,
) -> pd.DataFrame:
    """Load public `depth-wm.csv` labels for one BrainTreebank subject."""

    path = Path(bt_root) / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    if not path.exists():
        raise FileNotFoundError(f"BrainTreebank anatomy file missing: {path}")
    anatomy = pd.read_csv(path)
    required = {"Electrode", label_column}
    missing = sorted(required - set(anatomy.columns))
    if missing:
        raise KeyError(f"{path}: missing required columns {missing}")
    out = anatomy.copy()
    out["Subject"] = f"sub_{subject_id}"
    out["Electrode"] = out["Electrode"].map(clean_bt_electrode_label)
    out[label_column] = out[label_column].astype(str)
    return out


def bt_label_vocabulary(
    anatomy_tables: tp.Iterable[pd.DataFrame],
    *,
    label_column: str = DEFAULT_BT_LABEL_COLUMN,
    include_hemisphere: bool = False,
) -> tuple[str, ...]:
    """Return sorted public anatomy labels for a set of BT anatomy tables."""

    labels: set[str] = set()
    for table in anatomy_tables:
        _require_columns(table, {"Electrode", label_column})
        for row in table.itertuples(index=False):
            label = str(getattr(row, label_column))
            if include_hemisphere:
                hemi = getattr(row, "Hemisphere", None)
                label = f"{hemi}:{label}"
            labels.add(label)
    return tuple(sorted(labels))


def build_hard_public_bt_label_support(
    electrode_labels: tp.Sequence[str],
    anatomy: pd.DataFrame,
    parcel_labels: tp.Sequence[str],
    *,
    label_column: str = DEFAULT_BT_LABEL_COLUMN,
    include_hemisphere: bool = False,
) -> HardLabelSupport:
    """Build `(n_electrodes, n_labels)` one-hot support in electrode order."""

    required = {"Electrode", label_column}
    if include_hemisphere:
        required.add("Hemisphere")
    _require_columns(anatomy, required)

    cleaned = tuple(clean_bt_electrode_label(label) for label in electrode_labels)
    parcel_labels = tuple(str(label) for label in parcel_labels)
    parcel_index = {label: idx for idx, label in enumerate(parcel_labels)}
    if len(parcel_index) != len(parcel_labels):
        raise ValueError("parcel_labels must be unique")

    rows_by_electrode = {}
    for row in anatomy.itertuples(index=False):
        electrode = clean_bt_electrode_label(str(getattr(row, "Electrode")))
        label = str(getattr(row, label_column))
        if include_hemisphere:
            label = f"{getattr(row, 'Hemisphere')}:{label}"
        rows_by_electrode[electrode] = label

    support = np.zeros((len(cleaned), len(parcel_labels)), dtype=np.float32)
    missing_electrodes = []
    unknown_labels = []
    for electrode_idx, electrode in enumerate(cleaned):
        label = rows_by_electrode.get(electrode)
        if label is None:
            missing_electrodes.append(electrode)
            continue
        parcel_idx = parcel_index.get(label)
        if parcel_idx is None:
            unknown_labels.append(label)
            continue
        support[electrode_idx, parcel_idx] = 1.0

    if missing_electrodes:
        raise KeyError(
            "missing BT anatomy rows for electrodes: "
            f"{missing_electrodes[:10]}"
            + (
                f" (+{len(missing_electrodes) - 10} more)"
                if len(missing_electrodes) > 10
                else ""
            )
        )
    if unknown_labels:
        unique = sorted(set(unknown_labels))
        raise KeyError(
            "BT anatomy labels absent from parcel vocabulary: "
            f"{unique[:10]}"
            + (f" (+{len(unique) - 10} more)" if len(unique) > 10 else "")
        )

    return HardLabelSupport(
        kind="hard_public_bt_label",
        electrode_labels=cleaned,
        parcel_labels=parcel_labels,
        support=support,
        label_column=label_column,
    )


DEFAULT_SUPPORT_BIAS_EPS: float = 1e-2
"""v14 anatomy-prior strength. Controls cross-attn QK-bias headroom off-parcel:
1e-6 ≈ hard mask (off-parcel gradient ~ -14, dies), 1e-2 ≈ strong-but-finite
prior (off-parcel ~ -4.6, learnable), 0.5 ≈ weak prior, ∞ → vanilla cross-attn.
Default 1e-2 picked for first-pass robustness to residual DK label noise;
sweep {1e-4, 1e-3, 1e-2, 1e-1} on first dispatch."""


def support_attention_bias(
    support: np.ndarray,
    *,
    eps: float = DEFAULT_SUPPORT_BIAS_EPS,
) -> np.ndarray:
    """Convert support weights to Graphormer-style log attention bias.

    Used by v14 cross-attn as `softmax(QK / sqrt(d) + log(support + eps))`.
    `eps` is the anatomy-prior strength hyperparameter — not a numerical-
    stability constant. See ``DEFAULT_SUPPORT_BIAS_EPS``.
    """

    if eps <= 0.0:
        raise ValueError("eps must be positive")
    support = np.asarray(support, dtype=np.float32)
    if np.any(support < 0.0):
        raise ValueError("support must be non-negative")
    return np.log(support + np.float32(eps)).astype(np.float32)


def clean_bt_electrode_label(electrode_label: str) -> str:
    return str(electrode_label).replace("*", "").replace("#", "")


def _require_columns(table: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise KeyError(f"missing required columns {missing}")


_DK_APARC_BASE_LABELS: tuple[str, ...] = (
    "bankssts",
    "caudalanteriorcingulate",
    "caudalmiddlefrontal",
    "cuneus",
    "entorhinal",
    "frontalpole",
    "fusiform",
    "inferiorparietal",
    "inferiortemporal",
    "insula",
    "isthmuscingulate",
    "lateraloccipital",
    "lateralorbitofrontal",
    "lingual",
    "medialorbitofrontal",
    "middletemporal",
    "paracentral",
    "parahippocampal",
    "parsopercularis",
    "parsorbitalis",
    "parstriangularis",
    "pericalcarine",
    "postcentral",
    "posteriorcingulate",
    "precentral",
    "precuneus",
    "rostralanteriorcingulate",
    "rostralmiddlefrontal",
    "superiorfrontal",
    "superiorparietal",
    "superiortemporal",
    "supramarginal",
    "temporalpole",
    "transversetemporal",
)

_DK_ASEG_BASE_LABELS: tuple[str, ...] = (
    "Hippocampus",
    "Amygdala",
    "Caudate",
    "Putamen",
    "Pallidum",
    "Thalamus-Proper",
)

V14_DK_PARCEL_LABELS_CORTICAL: tuple[str, ...] = tuple(
    f"ctx-{hemi}-{base}" for hemi in ("lh", "rh") for base in _DK_APARC_BASE_LABELS
)
"""68 FreeSurfer DK aparc cortical labels (hemis-distinct), in BT depth-wm.csv string format."""

V14_DK_PARCEL_LABELS_SUBCORTICAL: tuple[str, ...] = tuple(
    f"{prefix}-{base}" for prefix in ("Left", "Right") for base in _DK_ASEG_BASE_LABELS
)
"""12 FreeSurfer aseg subcortical labels (Hippocampus/Amygdala/Caudate/Putamen/Pallidum/Thalamus-Proper, bilateral)."""

V14_DK_PARCEL_LABELS: tuple[str, ...] = (
    V14_DK_PARCEL_LABELS_CORTICAL + V14_DK_PARCEL_LABELS_SUBCORTICAL
)
"""Canonical K=80 v14 DK parcel vocabulary. Atlas-fixed, not cohort-derived —
keeps unpopulated parcels alive for cross-cohort portability."""


def parse_dk_label(label: str) -> tuple[str, str, str]:
    """Parse a BT depth-wm DK label into ``(kind, hemi, base)``.

    Returns
    -------
    kind : ``"cortical"`` or ``"subcortical"``.
    hemi : ``"lh"`` or ``"rh"``.
    base : the DK base region without prefix.
    """
    if label.startswith("ctx-lh-") or label.startswith("ctx-rh-"):
        return "cortical", label[4:6], label[7:]
    if label.startswith("Left-"):
        return "subcortical", "lh", label.removeprefix("Left-")
    if label.startswith("Right-"):
        return "subcortical", "rh", label.removeprefix("Right-")
    raise ValueError(f"unrecognised DK label: {label!r}")
