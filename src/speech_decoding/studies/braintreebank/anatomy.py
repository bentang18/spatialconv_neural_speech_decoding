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


_BT_V14_EXTRA_BAD_ELECTRODES: dict[int, tuple[str, ...]] = {
    2: ("LT2aA2", "LT3a8"),
    4: ("LT2aA11", "LT2bHb12", "LF3cIc10", "LF3aOFa16"),
    7: ("LF3bOFb1",),
    8: ("F3cIc8", "T3H12"),
    9: ("P2e6", "P2e8"),
}
"""Per-subject flaky-contact electrodes v14 statically excludes ON TOP of the
upstream corrupted / missing-coordinate / trigger drops.

These contacts produce brief, intense **spectral** transients — huge finite
``|STFT|`` robust-z (up to 1e4 σ) at isolated time-frequency bins — that spike
the SSL gradient. They are NOT high-amplitude dead channels: their whole-session
voltage MAD is ~subject-median, so the MNE-LOF voltage-domain guard (and any
voltage-amplitude rule) MISSES them. The detector is the ``|STFT|`` robust-z the
encoder actually sees, recurring above threshold across a MAJORITY of a subject's
trials (single-trial subjects use a higher one-shot bar).

Applied at BOTH electrode-order sites in lockstep so support / valid_mask stay
row-aligned with the front-end voltage (the DP4 contract): folded into
``voltage_electrode_order`` (support / valid_mask) AND dropped by ``bt_load_raw``
(front-end voltage) — the latter BEFORE shaft-CAR, so a bad contact can no longer
contaminate its shaft's CAR reference (which winsorising the cached output cannot
undo). Labels are cleaned (``clean_bt_electrode_label``) before matching.

Deliberate divergence from upstream ``BrainTreebankSubject.electrode_labels``;
``test_voltage_order_matches_upstream`` subtracts this set so the drift guard
still catches UNintended divergence.

The 11 contacts below are the finalized list (Ben-approved 2026-06-14). Selection
rule: an electrode is excluded iff it is separable from the clean population by
**(a) recurrence** — pre-CAR voltage events_100/min above the clean ceiling
(~0.14/min) in a MAJORITY of the subject's trials (single-trial subjects: the one
trial) — **OR (b) catastrophic amplitude** — post-CAR |STFT| robust-z max above the
clean ceiling (~2100), an event no biology reaches — **AND** the pre-CAR voltage
robust-z breaches its floor in the same trial(s) (de-confounds CAR-halo and
pure-spectral). The clean ceilings were measured across 25 BT sessions (voltage
573, |STFT| 2094, events 0.14/min). sub_8 T2A13 was dropped (12→11) — below the
clean ceiling on every axis, so the per-clip + winsor guards cover it instead.
Populating this changes the cache key, so the cache is rebuilt (and re-scanned to
confirm CAR-contaminated neighbors fall back to baseline)."""


def extra_bad_electrodes(subject_id: int) -> frozenset[str]:
    """Cleaned-label set of v14 statically-excluded flaky contacts for a subject
    (:data:`_BT_V14_EXTRA_BAD_ELECTRODES`). THE single source both electrode-order
    sites consult, so ``voltage_electrode_order`` (support/valid_mask) and
    ``bt_load_raw`` (front-end voltage) drop exactly the same contacts."""
    return frozenset(
        clean_bt_electrode_label(e)
        for e in _BT_V14_EXTRA_BAD_ELECTRODES.get(int(subject_id), ())
    )


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
    # v14 flaky-contact static exclusion — kept in lockstep with bt_load_raw via
    # the shared extra_bad_electrodes() source so support/valid_mask and the
    # front-end voltage drop the SAME contacts (DP4 row-alignment).
    drop.update(extra_bad_electrodes(sid))

    order = tuple(
        e for e in cleaned if e not in drop and not _is_trigger_label(e)
    )
    # Fail-loud on ambiguity (L4 single-source guard). This is THE single source
    # of electrode-row identity: support / valid_mask / the front-end scatter all
    # key off this order positionally. A duplicate label here means two physical
    # contacts are indistinguishable by name — exactly the precondition that lets a
    # name-keyed scatter collide rows (DP2). Raise rather than silently let row c
    # name two electrodes. See reports/bt_alignment/electrode_desync_damage_2026_06_09.md.
    if len(set(order)) != len(order):
        seen: set[str] = set()
        dups = sorted({e for e in order if e in seen or seen.add(e)})  # type: ignore[func-returns-value]
        raise ValueError(
            f"subject {sid}: voltage_electrode_order has duplicate electrode "
            f"labels {dups[:10]} after cleaning/drop — ambiguous row identity. "
            "Two contacts share a name; positional support/valid_mask/front-end "
            "alignment is undefined."
        )
    return order


def lite_voltage_mask(bt_root: str | Path, subject_id: int) -> np.ndarray:
    """Boolean mask over :func:`voltage_electrode_order` selecting the Neuroprobe
    Lite electrodes for this subject.

    Reproduces upstream ``BrainTreebankSubjectTrialBenchmarkDataset`` Lite
    subsetting (``datasets.py``: ``[full.index(e) for e in lite if e in full]``)
    as a **set**: the realized Lite subset is ``voltage_order ∩ lite_labels``.
    Upstream keeps the subset in Lite-list order; we keep voltage order, which is
    equivalent for the v14 encoder because the block-diagonal per-parcel pool is
    permutation-invariant within a parcel — only the *set* of electrodes feeding
    each parcel matters, not their order. Applying this one mask in lockstep to
    the study's voltage rows AND the DK support / valid-mask extractors keeps
    ``support[c]`` ↔ ``electrode_tokens[c]`` aligned after subsetting.

    Lite labels are already cleaned (no ``*``/``#``); we clean defensively so the
    intersection is in the same label space as ``voltage_electrode_order``.
    """
    from ._neuroprobe_lite_tables import NEUROPROBE_LITE_ELECTRODES

    sid = int(subject_id)
    key = f"btbank{sid}"
    if key not in NEUROPROBE_LITE_ELECTRODES:
        raise KeyError(
            f"subject {sid} ({key}) absent from vendored NEUROPROBE_LITE_ELECTRODES"
        )
    lite_set = {
        clean_bt_electrode_label(e) for e in NEUROPROBE_LITE_ELECTRODES[key]
    }
    order = voltage_electrode_order(bt_root, sid)
    return np.array([e in lite_set for e in order], dtype=bool)


def lite_voltage_order(bt_root: str | Path, subject_id: int) -> tuple[str, ...]:
    """The subject's Lite electrodes in voltage order (``voltage_order`` filtered
    to the Lite set). Set-equal to upstream's realized ``electrode_labels`` for
    the Lite Dataset; see :func:`lite_voltage_mask` for why order is free."""
    order = voltage_electrode_order(bt_root, subject_id)
    mask = lite_voltage_mask(bt_root, subject_id)
    return tuple(e for e, keep in zip(order, mask) if keep)


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


# --- DKT (Desikan-Killiany-Tourville) atlas -------------------------------- #
# The BT depth-wm.csv carries a NATIVE ``DKT`` column (FreeSurfer aparc.DKTatlas),
# parcellated directly — NOT derivable by dropping DK electrodes. The DKT protocol
# removes three DK gyral labels whose boundaries it could not define reliably
# (bankssts, frontalpole, temporalpole) and REASSIGNS those vertices to neighbouring
# gyri, so an electrode DK-labelled e.g. ``temporalpole`` re-appears under a DKT
# neighbour label — it is not lost. Same string format as the DK column.
_DKT_DROPPED_BASES: frozenset[str] = frozenset(
    {"bankssts", "frontalpole", "temporalpole"}
)
_DKT_APARC_BASE_LABELS: tuple[str, ...] = tuple(
    b for b in _DK_APARC_BASE_LABELS if b not in _DKT_DROPPED_BASES
)
"""31 DKT cortical base regions = the 34 DK aparc bases minus the 3 DKT drops."""

V14_DKT_PARCEL_LABELS_CORTICAL: tuple[str, ...] = tuple(
    f"ctx-{hemi}-{base}" for hemi in ("lh", "rh") for base in _DKT_APARC_BASE_LABELS
)
"""62 FreeSurfer DKT aparc cortical labels (hemis-distinct), BT depth-wm.csv format."""

V14_DKT_PARCEL_LABELS_SUBCORTICAL: tuple[str, ...] = V14_DK_PARCEL_LABELS_SUBCORTICAL
"""12 aseg subcortical labels — identical to DK (DKT only changes cortical gyri)."""

V14_DKT_PARCEL_LABELS: tuple[str, ...] = (
    V14_DKT_PARCEL_LABELS_CORTICAL + V14_DKT_PARCEL_LABELS_SUBCORTICAL
)
"""Canonical K=74 v14 DKT parcel vocabulary (62 cortical + 12 subcortical).
Atlas-fixed, not cohort-derived — keeps unpopulated parcels alive for
cross-cohort portability (mirrors :data:`V14_DK_PARCEL_LABELS`)."""


# Atlas registry — the SINGLE source pairing each atlas's CSV column with its
# parcel vocabulary. They MUST move together: using the DKT vocabulary against the
# ``DesikanKilliany`` column (or vice versa) silently mis-routes every electrode
# (a temporalpole DK label would land out-of-vocab, etc.). Thread an atlas name,
# never a bare (column, labels) pair, so the two can never desync.
_ATLAS_SPECS: dict[str, tuple[str, tuple[str, ...]]] = {
    "dk": ("DesikanKilliany", V14_DK_PARCEL_LABELS),
    "dkt": ("DKT", V14_DKT_PARCEL_LABELS),
}


def atlas_spec(atlas: str) -> tuple[str, tuple[str, ...]]:
    """Resolve an atlas name to its ``(label_column, parcel_labels)`` pair.

    The ONLY sanctioned way to pick an atlas — guarantees the depth-wm.csv column
    and the parcel vocabulary are the matched pair (``"dk"`` → DesikanKilliany /
    K=80, ``"dkt"`` → DKT / K=74). Raises on an unknown name rather than silently
    falling back to DK.
    """
    key = str(atlas).lower()
    if key not in _ATLAS_SPECS:
        raise ValueError(
            f"unknown atlas {atlas!r}; expected one of {sorted(_ATLAS_SPECS)}"
        )
    return _ATLAS_SPECS[key]


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


_MIN_PARCEL_VALID_FRACTION: float = 0.5
"""Per-subject floor on the fraction of voltage electrodes that map to an
in-vocab DK parcel. MEASURED 2026-06-13 across all 10 vendored BT subjects:
9/10 map at 1.000, sub_4 at 0.989 (2 ``Left-Inf-Lat-Vent`` ventricle contacts
legitimately outside the K=80 cortical+subcortical vocab), and 0 missing-anatomy
rows anywhere. The 0.5 floor sits ~18x below the worst real subject, so it never
false-fires on the live cohort but catches an S9-class silent collapse: a
per-subject ``depth-wm.csv`` namespace shift / re-rip / new subject whose DK
labels stop matching the ``electrode_labels.json`` voltage namespace would drop
most/all electrodes to ``valid=False`` (a zero support row each) with NO error
under ``unmapped_policy='zero'``, silently erasing that subject's entire
parcel-pool routing. See reports/bt_alignment/data_pipeline_bug_ledger.md LG15."""


def _assert_parcel_support_coverage(
    result: HardLabelSupport, subject_id: int
) -> None:
    """Fail loud if a subject's DK-parcel coverage collapses (ledger LG15).

    Under ``unmapped_policy='zero'`` an electrode with no anatomy row or an
    out-of-vocab DK label is silently zeroed (``valid=False``). A catastrophic
    drop — far below the measured 0.989-1.000 live range — means the electrode
    label namespace diverged for this subject, not a few stray ventricle
    contacts, so the whole subject's pooled features would be degenerate.
    """
    n_total = int(result.valid.shape[0])
    if n_total == 0:
        return
    n_valid = int(result.valid.sum())
    fraction = n_valid / n_total
    if fraction < _MIN_PARCEL_VALID_FRACTION:
        raise ValueError(
            f"subject {subject_id}: only {n_valid}/{n_total} voltage electrodes "
            f"({fraction:.3f}) mapped to an in-vocab {result.label_column} parcel "
            f"— below the {_MIN_PARCEL_VALID_FRACTION:.2f} floor (DK cohort is "
            "0.989-1.000; DKT 0.888-1.000, measured 2026-06-13). "
            f"The depth-wm.csv {result.label_column} labels likely stopped matching "
            "the voltage electrode namespace for this subject (S9-class); its "
            "parcel-pool routing would be silently zeroed under "
            "unmapped_policy='zero'. "
            f"Rebuild/verify localization/sub_{subject_id}/depth-wm.csv. "
            "See data_pipeline_bug_ledger.md LG15."
        )


def _drop_single_electrode_parcels(result: HardLabelSupport) -> HardLabelSupport:
    """Exclude single-electrode parcels (Ben 2026-06-13): a parcel covered by
    exactly ONE valid electrode is dropped ENTIRELY — the lone electrode's support
    row is zeroed and ``valid[c]=False``, leaving the parcel uncovered for this
    subject.

    Why: a parcel pooled from a single electrode has a degenerate within-parcel
    statistic — its mean IS that electrode and its std is identically 0, which
    poisons the heteroscedastic M4 precision weight (σ²→0 ⇒ unbounded weight). The
    parcel stays in the K-vocabulary (cross-subject parcel embeddings must align),
    so this changes only this subject's coverage, never the global K. Operates on
    the post-``unmapped_policy`` support, so already-invalid electrodes (zero rows)
    do not count toward a parcel's electrode tally. One-hot routing guarantees no
    cascade — dropping a parcel's lone electrode empties only that parcel.
    """
    support = result.support
    valid = result.valid
    # Per-parcel valid-electrode count (invalid electrodes already have zero rows).
    per_parcel = support.sum(axis=0)                       # (K,)
    single = np.flatnonzero(per_parcel == 1.0)             # parcels with exactly 1
    if single.size == 0:
        return result
    support = support.copy()
    valid = valid.copy()
    for k in single:
        rows = np.flatnonzero(support[:, k] != 0.0)        # the lone electrode row
        support[rows, :] = 0.0
        valid[rows] = False
    return HardLabelSupport(
        kind=result.kind,
        electrode_labels=result.electrode_labels,
        parcel_labels=result.parcel_labels,
        support=support,
        valid=valid,
        label_column=result.label_column,
    )


@functools.lru_cache(maxsize=64)
def aligned_voltage_support(
    bt_root: str | Path,
    subject_id: int,
    *,
    parcel_labels: tuple[str, ...] = V14_DK_PARCEL_LABELS,
    unmapped_policy: tp.Literal["raise", "zero"] = "raise",
    label_column: str = DEFAULT_BT_LABEL_COLUMN,
    exclude_single_electrode_parcels: bool = False,
) -> HardLabelSupport:
    """DK/DKT support aligned to the VOLTAGE electrode order.

    ``result.support[c]`` and ``result.valid[c]`` describe the same physical
    electrode as ``electrode_tokens[c]`` (= ``voltage_electrode_order(...)[c]``).
    Electrodes with no anatomy row or an out-of-vocab label get a zero row +
    ``valid=False`` in place under ``unmapped_policy="zero"`` (no re-pack). The
    result is memoized per ``(bt_root, subject_id, parcel_labels, policy,
    label_column, exclude_single_electrode_parcels)``; callers must not mutate
    the returned arrays.

    ``label_column`` selects the depth-wm.csv atlas column ("DesikanKilliany" or
    "DKT"); it MUST be the partner of ``parcel_labels`` (use :func:`atlas_spec` to
    get the matched pair — never mix a column with the wrong vocabulary).

    ``exclude_single_electrode_parcels`` (Ben 2026-06-13): after mapping, drop any
    parcel covered by exactly one valid electrode — zero that electrode's row and
    mark it invalid (see :func:`_drop_single_electrode_parcels`). Applied AFTER the
    coverage guard so the guard measures true anatomy health, not the intentional
    single-electrode drop.

    Fails loud (ledger LG15) if per-subject parcel coverage collapses below
    :data:`_MIN_PARCEL_VALID_FRACTION` — the S9-class silent-degradation guard.
    ``lru_cache`` does not cache exceptions, so the guard re-checks every call.
    """
    order = voltage_electrode_order(bt_root, subject_id)
    anatomy = load_public_bt_anatomy(
        bt_root, int(subject_id), label_column=label_column
    )
    result = build_hard_public_bt_label_support(
        order,
        anatomy,
        parcel_labels,
        label_column=label_column,
        unmapped_policy=unmapped_policy,
    )
    _assert_parcel_support_coverage(result, int(subject_id))
    if exclude_single_electrode_parcels:
        result = _drop_single_electrode_parcels(result)
    return result
