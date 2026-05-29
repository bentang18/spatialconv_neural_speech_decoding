"""Build and read the per-electrode Tier-1 support cache.

One CSV per patient at `data/atlas/support_cache/<pt>_support_tier1.csv`.
Columns: `name`, then 15 Tier-1 support columns in `DEFAULT_BASE_PARCELS`
order. Values are raw BNA probability in [0, 100] (float32); no normalization.

Also exposes :func:`compute_gated_log_support_bias` — the B29 Item 12
per-clip ``λ_anat`` gate on the ``log(support + ε)`` cross-attn bias.
The encoder forward calls this inline at every step; the helper exists
so callers can pre-compute / debug the bias outside the model.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fsaverage import (
    DEFAULT_BNA_N_ROIS,
    DEFAULT_BNA_TREE,
    assert_full_bake,
    load_baked_atlas,
    load_fsaverage_cache,
    sample_baked_support,
)
from .tokens import DEFAULT_BASE_PARCELS

try:
    import torch
    from torch import Tensor as _Tensor
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard dep, but guard for docs builds
    torch = None  # type: ignore[assignment]
    _Tensor = tp.Any  # type: ignore[misc, assignment]
    _TORCH_AVAILABLE = False


def compute_gated_log_support_bias(
    support: "_Tensor",
    *,
    eps: float,
    lambda_anat: "float | _Tensor" = 1.0,
) -> "_Tensor":
    """Return ``λ_anat * log(support + ε)`` with B29 per-clip broadcasting.

    Parameters
    ----------
    support: (B, C, K) float Tensor — per-electrode parcel-support mass.
    eps:     float — anatomy-prior strength (default 1e-2 lives in
             :data:`speech_decoding.studies.braintreebank.anatomy.DEFAULT_SUPPORT_BIAS_EPS`).
    lambda_anat:
        Either a scalar (B28 warmup) or a (B,) Tensor (B29 per-clip
        metadata gate). Broadcasts to ``(B, 1, 1)`` for the latter so
        each clip independently scales its bias. Must be ≥ 0.

    The encoder forward inlines this computation; this helper exists so
    monitor probes / debug scripts can reproduce the gated bias outside
    the model. See :class:`speech_decoding.extractors.subtype_meta.LambdaAnatExtractor`
    for the per-clip metadata source.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("torch is required for compute_gated_log_support_bias")
    if support.dim() != 3:
        raise ValueError(
            f"support must be (B, C, K); got shape {tuple(support.shape)}"
        )
    B = int(support.shape[0])
    if isinstance(lambda_anat, _Tensor):
        # Warmup schedule wrappers may pass a 0-d Tensor; treat that as
        # the scalar branch rather than rejecting it.
        if lambda_anat.dim() == 0:
            scalar = float(lambda_anat.item())
            if scalar < 0.0:
                raise ValueError(f"lambda_anat must be >= 0; got {scalar}")
            return scalar * torch.log(support + eps)
        # NeuralSet collates per-event ``(1,)`` TimedArrays into
        # ``(B, 1)``; squeeze the trailing singleton before validating.
        if lambda_anat.dim() == 2 and int(lambda_anat.shape[-1]) == 1:
            lambda_anat = lambda_anat.squeeze(-1)
        if lambda_anat.dim() != 1 or int(lambda_anat.shape[0]) != B:
            raise ValueError(
                f"lambda_anat tensor must have shape (B={B},); "
                f"got shape {tuple(lambda_anat.shape)}"
            )
        if (lambda_anat < 0).any():
            raise ValueError("lambda_anat tensor must be >= 0")
        # ``support`` is the device-of-truth (often CUDA); per-clip
        # metadata may arrive on CPU. Match both device AND dtype.
        gate = lambda_anat.to(device=support.device, dtype=support.dtype).view(B, 1, 1)
    else:
        if float(lambda_anat) < 0.0:
            raise ValueError(f"lambda_anat must be >= 0; got {lambda_anat}")
        gate = lambda_anat
    return gate * torch.log(support + eps)


def sanitize_parcel_name(name: str) -> str:
    """Replace characters that round-trip poorly in CSV headers."""

    return name.replace("/", "_")


TIER1_COLUMNS: tuple[str, ...] = tuple(
    f"support_{sanitize_parcel_name(n)}" for n, _ in DEFAULT_BASE_PARCELS
)
TIER1_BNA_INDICES_1BASED: tuple[int, ...] = tuple(idx for _, idx in DEFAULT_BASE_PARCELS)


@dataclass(frozen=True)
class PatientQC:
    patient: str
    n_electrodes: int
    argmax_in_tier1: int
    low_tier1_mass: int
    # Histogram over max-per-electrode support in bins
    # [0,1) [1,10) [10,25) [25,50) [50,75) [75,100]
    hist_counts: tuple[int, int, int, int, int, int]


_HIST_BIN_EDGES = (0.0, 1.0, 10.0, 25.0, 50.0, 75.0, 100.0 + 1e-6)
_HIST_BIN_LABELS = ("[0,1)", "[1,10)", "[10,25)", "[25,50)", "[50,75)", "[75,100]")


def build_patient_support(
    patient_id: str,
    *,
    coord_cache_dir: Path,
    bake_dir: Path,
    bna_tree: Path = DEFAULT_BNA_TREE,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Sample the baked atlas at this patient's fsaverage vertices, Tier-1 slice."""

    cache = load_fsaverage_cache(patient_id, coord_cache_dir)
    atlas = load_baked_atlas(bake_dir, kind="smoothed", tree_path=bna_tree)
    assert_full_bake(atlas, expected_n_rois=DEFAULT_BNA_N_ROIS)
    support_full = sample_baked_support(cache, atlas)  # (N_e, 246) float64 in [0, 100]
    col_idx = np.array([i - 1 for i in TIER1_BNA_INDICES_1BASED], dtype=np.int64)
    support_tier1 = support_full[:, col_idx].astype(np.float32, copy=False)
    return cache.names, support_tier1


def write_support_cache(
    out_path: Path,
    names: tuple[str, ...],
    support_tier1: np.ndarray,
) -> None:
    if support_tier1.shape != (len(names), len(TIER1_COLUMNS)):
        raise ValueError(
            f"support shape {support_tier1.shape} != expected {(len(names), len(TIER1_COLUMNS))}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write(",".join(("name",) + TIER1_COLUMNS) + "\n")
        for row_idx, name in enumerate(names):
            values = ",".join(f"{float(v):.6f}" for v in support_tier1[row_idx])
            f.write(f"{name},{values}\n")


def read_support_cache(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Inverse of `write_support_cache`. Returns (names, support[N_e, 15] float32)."""

    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"empty support cache: {path}")
    header = lines[0].split(",")
    expected_header = ("name",) + TIER1_COLUMNS
    if tuple(header) != expected_header:
        raise ValueError(f"{path}: header {header!r} != expected {list(expected_header)}")
    names: list[str] = []
    rows: list[list[float]] = []
    for row_idx, line in enumerate(lines[1:], start=2):
        if not line:
            continue
        fields = line.split(",")
        if len(fields) != len(expected_header):
            raise ValueError(
                f"{path}:{row_idx}: expected {len(expected_header)} columns, got {len(fields)}"
            )
        names.append(fields[0])
        rows.append([float(v) for v in fields[1:]])
    support = np.array(rows, dtype=np.float32)
    return tuple(names), support


def lookup_support_for_kept_channels(
    support_cache_path: Path,
    kept_names: tuple[str, ...],
) -> np.ndarray:
    """Return `support[N_e, 15]` in `kept_names` order. Raises if any name is missing."""

    names, support = read_support_cache(support_cache_path)
    name_to_row = {name: idx for idx, name in enumerate(names)}
    missing = [n for n in kept_names if n not in name_to_row]
    if missing:
        raise KeyError(
            f"{support_cache_path}: missing electrodes in support cache: {missing[:10]}"
            + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else "")
        )
    idx = np.array([name_to_row[n] for n in kept_names], dtype=np.int64)
    return support[idx]


def qc_row_for_patient(patient_id: str, support_tier1: np.ndarray) -> PatientQC:
    max_per_elec = support_tier1.max(axis=1)
    sum_per_elec = support_tier1.sum(axis=1)
    hist, _ = np.histogram(max_per_elec, bins=np.array(_HIST_BIN_EDGES))
    hist_tuple: tuple[int, int, int, int, int, int] = (
        int(hist[0]), int(hist[1]), int(hist[2]),
        int(hist[3]), int(hist[4]), int(hist[5]),
    )
    return PatientQC(
        patient=patient_id,
        n_electrodes=int(support_tier1.shape[0]),
        argmax_in_tier1=int((max_per_elec > 0).sum()),
        low_tier1_mass=int((sum_per_elec < 1.0).sum()),
        hist_counts=hist_tuple,
    )


def write_qc_report(qc_path: Path, qc_rows: list[PatientQC]) -> None:
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Tier-1 support cache QC report\n")
    lines.append("Per-patient summary of `data/atlas/support_cache/<pt>_support_tier1.csv`.\n")
    lines.append(
        "Source: fsaverage snap-to-pial + `fsaverage_bake_fast2/smoothed`, "
        "sliced to 15 `DEFAULT_BASE_PARCELS` columns (raw [0, 100] BNA probability).\n"
    )
    lines.append("\n## Tier-1 column order\n")
    for col, (name, bna_idx) in zip(TIER1_COLUMNS, DEFAULT_BASE_PARCELS):
        sanitized_hint = (
            f" (sanitized to `{col}`)" if name != col[len("support_"):] else ""
        )
        lines.append(f"- `{col}` ← `{name}`{sanitized_hint}, BNA index {bna_idx}")
    lines.append("\n## Per-patient summary\n")
    header_bins = " | ".join(f"max {lbl}" for lbl in _HIST_BIN_LABELS)
    lines.append(
        "| patient | n_elec | argmax_in_tier1 | low_tier1_mass (<1) | " + header_bins + " |"
    )
    lines.append("|" + "---|" * (4 + len(_HIST_BIN_LABELS)))
    for row in qc_rows:
        hist_cells = " | ".join(str(c) for c in row.hist_counts)
        lines.append(
            f"| {row.patient} | {row.n_electrodes} | {row.argmax_in_tier1} | "
            f"{row.low_tier1_mass} | {hist_cells} |"
        )
    lines.append("")
    qc_path.write_text("\n".join(lines))
