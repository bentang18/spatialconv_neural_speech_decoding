"""Public BrainTreebank anatomy support.

BrainTreebank currently exposes DK/Destrieux-style per-electrode labels and
template plotting coordinates. The label path is useful as a hard-parcel
ablation, but it is not fsaverage/BNA support.
"""

from __future__ import annotations

import functools
import json
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SupportKind = tp.Literal["soft_surface_bna", "hard_public_bt_label", "none"]
DEFAULT_BT_LABEL_COLUMN = "DesikanKilliany"


@dataclass(frozen=True)
class HardLabelSupport:
    """One-hot electrode support over public BrainTreebank anatomy labels.

    ``support[c]`` is the one-hot parcel assignment for ``electrode_labels[c]``;
    ``valid[c]`` is True iff that electrode was mapped to a parcel in the vocab
    (i.e. ``support[c]`` is nonzero). Under ``unmapped_policy="zero"`` an
    electrode with no anatomy row, or a DK label outside ``parcel_labels``, gets
    a zero ``support`` row and ``valid[c] = False`` **in place** — rows are never
    re-packed, so row ``c`` always names the same physical electrode as the
    voltage / token tensors it is collated beside (C1 fix).
    """

    kind: SupportKind
    electrode_labels: tuple[str, ...]
    parcel_labels: tuple[str, ...]
    support: np.ndarray
    valid: np.ndarray
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
    unmapped_policy: tp.Literal["raise", "zero"] = "raise",
) -> HardLabelSupport:
    """Build `(n_electrodes, n_labels)` one-hot support in ``electrode_labels`` order.

    ``unmapped_policy``:
    - ``"raise"`` (default): raise ``KeyError`` if any electrode has no anatomy
      row, or a DK label outside ``parcel_labels``.
    - ``"zero"``: leave a zero support row + ``valid=False`` for such electrodes,
      **in place** (no re-pack), so row ``c`` keeps naming ``electrode_labels[c]``.
    """

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
    valid = np.zeros(len(cleaned), dtype=bool)
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
        valid[electrode_idx] = True

    if unmapped_policy == "raise":
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
        valid=valid,
        label_column=label_column,
    )


_BT_MISSING_COORDINATE_ELECTRODES: dict[int, tuple[str, ...]] = {
    1: ("F3cId10",),
    2: (),
    3: (
        "F3c9", "F3c10", "T1aIc1", "T1aIc2", "P2a10",
        "O1aIb2", "O1aIb3", "O1aIb4", "O1aIb5", "O1aIb6", "O1aIb7", "O1aIb8",
    ),
    4: ("LT1aIb10", "LF3bIa12"),
    5: (),
    6: (),
    7: ("LF3aOFa16", "LF1cCb12"),
    8: ("F2bCb6", "F2bCb14"),
    9: ("P2a6", "P2a7", "P2a8"),
    10: ("T1aIa4", "P2cCc5"),
}
"""Per-subject electrodes upstream drops for missing coordinates when
``allow_missing_coordinates=False`` (the ``BrainTreebankSubject`` default the
v14 loader uses). Copied verbatim from the PINNED upstream
``neuroprobe.braintreebank_subject.BrainTreebankSubject._get_corrupted_electrodes``
(commit ``c7b955b0``). Drift from upstream is caught by
``test_voltage_order_matches_upstream`` (DCC-only; skipped when neuroprobe is
not importable)."""


def _is_trigger_label(electrode_label: str) -> bool:
    up = electrode_label.upper()
    return up.startswith("DC") or up.startswith("TRIG")


def voltage_electrode_order(
    bt_root: str | Path, subject_id: int
) -> tuple[str, ...]:
    """Canonical voltage channel order — exactly the electrodes (and order) the
    v14 loader feeds the front-end.

    This reproduces ``BrainTreebankSubject(subject_id).electrode_labels``: the
    cleaned ``electrode_labels.json`` order, with corrupted electrodes
    (``corrupted_elec.json``), missing-coordinate electrodes
    (``_BT_MISSING_COORDINATE_ELECTRODES``), and trigger (``DC*`` / ``TRIG*``)
    channels removed. It is derived from the same files upstream reads so the DK
    ``support`` / ``valid_mask`` extractors align to the VOLTAGE order, not the
    independent ``depth-wm.csv`` row order (C2 fix). Reading depth-wm row order
    instead silently routes every voltage into the wrong parcel for any subject
    whose corrupted/trigger/unmapped contacts shift the two orders apart.

    Replicated (rather than imported) so laptop-side audits run against the
    vendored fixtures without neuroprobe; guarded against upstream drift by
    ``test_voltage_order_matches_upstream`` (skipped when neuroprobe is absent).
    """
    root = Path(bt_root)
    sid = int(subject_id)
    labels_path = root / "electrode_labels" / f"sub_{sid}" / "electrode_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"BrainTreebank electrode_labels file missing: {labels_path}"
        )
    raw_labels = json.loads(labels_path.read_text())
    cleaned = [clean_bt_electrode_label(str(e)) for e in raw_labels]

    drop: set[str] = set()
    corrupted_path = root / "corrupted_elec.json"
    if corrupted_path.exists():
        corrupted = json.loads(corrupted_path.read_text())
        drop.update(
            clean_bt_electrode_label(str(e))
            for e in corrupted.get(f"sub_{sid}", ())
        )
    drop.update(
        clean_bt_electrode_label(e)
        for e in _BT_MISSING_COORDINATE_ELECTRODES.get(sid, ())
    )

    return tuple(
        e for e in cleaned if e not in drop and not _is_trigger_label(e)
    )


DEFAULT_SUPPORT_BIAS_EPS: float = 1e-2
"""Anatomy-prior strength for the soft ``log(support + ε)`` routing bias.

**NOT on the v14 default path.** B36 (2026-06-01) replaced the soft routing
bias with the hard block-diagonal per-parcel pool, which consumes the one-hot
DK ``support`` directly and has no ``ε``. This constant is retained only as
the default for the gated ``R-bna-soft`` routing sister (and as the vestigial
``--eps`` / encoder-forward ``eps`` default). When that sister runs, ``ε``
controls off-parcel QK-bias headroom: 1e-6 ≈ hard mask, 1e-2 ≈ strong-but-
finite prior, ∞ → vanilla cross-attn."""


def support_attention_bias(
    support: np.ndarray,
    *,
    eps: float = DEFAULT_SUPPORT_BIAS_EPS,
) -> np.ndarray:
    """Convert support weights to Graphormer-style log attention bias.

    **Sister-only helper — NOT on the v14 default path.** B36 (2026-06-01)
    replaced the soft routing bias with the hard per-parcel pool; this
    ``softmax(QK / sqrt(d) + log(support + ε))`` form is retained for the
    gated ``R-bna-soft`` routing sister. ``eps`` is the anatomy-prior strength
    hyperparameter — not a numerical-stability constant. See
    ``DEFAULT_SUPPORT_BIAS_EPS``.
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


@functools.lru_cache(maxsize=64)
def aligned_voltage_support(
    bt_root: str | Path,
    subject_id: int,
    *,
    parcel_labels: tuple[str, ...] = V14_DK_PARCEL_LABELS,
    unmapped_policy: tp.Literal["raise", "zero"] = "raise",
    label_column: str = DEFAULT_BT_LABEL_COLUMN,
) -> HardLabelSupport:
    """DK support aligned to the VOLTAGE electrode order.

    ``result.support[c]`` and ``result.valid[c]`` describe the same physical
    electrode as ``electrode_tokens[c]`` (= ``voltage_electrode_order(...)[c]``).
    Electrodes with no anatomy row or an out-of-vocab DK label get a zero row +
    ``valid=False`` in place under ``unmapped_policy="zero"`` (no re-pack). The
    result is memoized per ``(bt_root, subject_id, parcel_labels, policy)``;
    callers must not mutate the returned arrays.
    """
    order = voltage_electrode_order(bt_root, subject_id)
    anatomy = load_public_bt_anatomy(
        bt_root, int(subject_id), label_column=label_column
    )
    return build_hard_public_bt_label_support(
        order,
        anatomy,
        parcel_labels,
        label_column=label_column,
        unmapped_policy=unmapped_policy,
    )
