#!/usr/bin/env python3
"""Verify the BNA -> fsaverage bake against the Brainnetome authors' published annot.

Contract (#36 verification):

- Input A: our bake, loaded via
      load_baked_atlas("data/atlas/fsaverage_bake_fast2/", kind={raw,smoothed})
  Shape per hemi: (n_vert, 246) float.

- Input B: published fsaverage annot from the BNA FreeSurfer distribution,
      data/atlas/BN_Atlas_fsaverage/label/{lh,rh}.BN_Atlas.annot
  nib.freesurfer.io.read_annot returns (labels[n_vert], ctab, names) where
  names[0] = 'Unknown' and names[1..246] are the 246 BNA parcels in
  BNA-id order.

- Alignment: our bake frame k (0-based) corresponds to BNA id (k+1) (1-based).
  Per-vertex: our_argmax_1b[v] = argmax(bake[v, :]) + 1, compared to
  annot_labels[v].

- Valid-vertex mask M[v] = cortex[v] and (bake_max[v] > 0) and (annot[v] > 0).

- Metrics: overall, interior, boundary agreement on M; per-parcel Dice;
  raw vs smoothed decomposition to isolate smoothing drift.

Outputs under data/atlas/fsaverage_bake_fast2/verify_vs_published/:
  summary.json
  {lh,rh}.per_parcel.{raw,smoothed}.csv
  {lh,rh}.disagreement.{raw,smoothed}.mgh   (0=agree/out, 1=boundary, 2=interior)
  {lh,rh}.our_argmax.{raw,smoothed}.mgh
  {lh,rh}.published_argmax.mgh
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.fsaverage_atlas import (
    DEFAULT_BNA_N_ROIS,
    load_baked_atlas,
)

DEFAULT_FS_HOME = Path("/Applications/freesurfer/8.2.0")
DEFAULT_SUBJECTS_DIR = DEFAULT_FS_HOME / "subjects"
DEFAULT_BAKE_DIR = PROJECT_ROOT / "data" / "atlas" / "fsaverage_bake_fast2"
DEFAULT_ANNOT_DIR = PROJECT_ROOT / "data" / "atlas" / "BN_Atlas_freesurfer" / "fsaverage" / "label"
DEFAULT_OUT_DIR = DEFAULT_BAKE_DIR / "verify_vs_published"
HEMIS = ("lh", "rh")

# BNA fsaverage annot ships 210 cortical parcels + "Unknown" = 211 names.
# Parcels are odd = LH, even = RH. Validate ordering by checking that a
# handful of known LH ids land on names ending with "_L".
N_CORTICAL_BNA = 210
LH_ANCHORS: tuple[int, ...] = (29, 33, 53, 61, 155, 157)


def read_published_annot(path: Path) -> tuple[np.ndarray, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"missing published annot: {path}\n"
            f"Fetch the BNA FreeSurfer distribution from the Brainnetome atlas "
            f"release and place {{lh,rh}}.BN_Atlas.annot at {path.parent}/"
        )
    annot = nib.freesurfer.io.read_annot(str(path))
    labels = annot[0]
    name_bytes = annot[2]
    names = [
        n.decode("utf-8", errors="replace") if isinstance(n, bytes) else str(n)
        for n in name_bytes
    ]
    return np.asarray(labels, dtype=np.int64), names


def validate_annot_ordering(names: list[str], hemi: str, path: Path) -> None:
    # 211 = Unknown + 210 cortical parcels; 247 = Unknown + 246 (if subcortical included).
    if len(names) < N_CORTICAL_BNA + 1:
        raise ValueError(
            f"{path}: expected at least {N_CORTICAL_BNA + 1} names "
            f"(Unknown + {N_CORTICAL_BNA} cortical), got {len(names)}"
        )
    first = names[0].lower()
    if "unknown" not in first and "???" not in first:
        print(f"[warn] {path.name}: names[0] = {names[0]!r}, expected Unknown")
    # LH anchors should end in _L. If they don't, ordering is wrong.
    for bna_id in LH_ANCHORS:
        if bna_id >= len(names):
            continue
        got = names[bna_id]
        if not got.endswith("_L"):
            print(
                f"[warn] {path.name} {hemi}: names[{bna_id}] = {got!r}, "
                f"expected an LH parcel ending in '_L'. BNA-id ordering may be off."
            )


def cortex_mask(subjects_dir: Path, hemi: str, n_vertices: int) -> np.ndarray:
    label_path = subjects_dir / "fsaverage" / "label" / f"{hemi}.cortex.label"
    if not label_path.exists():
        raise FileNotFoundError(f"missing fsaverage cortex label: {label_path}")
    # .label files: line 0 comment, line 1 count, then `vno x y z stat` rows.
    with label_path.open() as f:
        f.readline()
        count = int(f.readline().strip())
        vids = np.empty((count,), dtype=np.int64)
        for i in range(count):
            vids[i] = int(f.readline().split(None, 1)[0])
    mask = np.zeros((n_vertices,), dtype=bool)
    mask[vids] = True
    return mask


def build_edge_adjacency(surf_path: Path, n_vertices: int):
    geom = nib.freesurfer.io.read_geometry(str(surf_path))
    faces = np.asarray(geom[1], dtype=np.int64)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data = np.ones(rows_sym.shape[0], dtype=np.uint8)
    adj = sparse.coo_matrix(
        (data, (rows_sym, cols_sym)),
        shape=(n_vertices, n_vertices),
    ).tocsr()
    return adj


def boundary_mask(labels: np.ndarray, adj) -> np.ndarray:
    coo = adj.tocoo()
    diff = labels[coo.row] != labels[coo.col]
    out = np.zeros(labels.shape[0], dtype=bool)
    out[coo.row[diff]] = True
    return out


def save_mgh(values: np.ndarray, out_path: Path) -> None:
    vol = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
    img = nib.freesurfer.mghformat.MGHImage(vol, affine=np.eye(4, dtype=np.float32))  # type: ignore[arg-type]
    img.to_filename(str(out_path))


def verify_hemi(
    *,
    hemi: str,
    bake: np.ndarray,
    annot_labels: np.ndarray,
    annot_names: list[str],
    cortex: np.ndarray,
    adj: sparse.csr_matrix,
    out_dir: Path,
    kind: str,
) -> dict:
    n_vert = bake.shape[0]
    if annot_labels.shape[0] != n_vert:
        raise ValueError(
            f"{hemi}: bake verts {n_vert} != annot verts {annot_labels.shape[0]}"
        )

    # Restrict to cortical BNA ids (1..210) so argmax is comparable to the annot.
    cortical_bake = bake[:, :N_CORTICAL_BNA]
    full_bake_max = bake.max(axis=1)
    full_argmax = bake.argmax(axis=1) + 1
    subcortical_dominant = (full_argmax > N_CORTICAL_BNA) & (full_bake_max > 0)

    ctx_max = cortical_bake.max(axis=1)
    our_argmax = cortical_bake.argmax(axis=1).astype(np.int64) + 1  # 1-based BNA id
    our_argmax[ctx_max <= 0] = 0

    # Unknown-on-either-side vertices are excluded from the agreement score.
    labeled = cortex & (our_argmax > 0) & (annot_labels > 0)
    agree = labeled & (our_argmax == annot_labels)

    is_boundary = boundary_mask(annot_labels, adj)
    interior = labeled & (~is_boundary)
    boundary = labeled & is_boundary

    n_labeled = int(labeled.sum())
    n_agree = int(agree.sum())
    n_interior = int(interior.sum())
    n_boundary = int(boundary.sum())
    n_interior_agree = int((agree & interior).sum())
    n_boundary_agree = int((agree & boundary).sum())

    # Per-parcel Dice for every BNA id that appears on this hemisphere.
    parcel_ids = np.unique(annot_labels)
    parcel_ids = parcel_ids[parcel_ids > 0]
    per_parcel: list[dict] = []
    for pid in parcel_ids:
        a = (annot_labels == pid) & cortex
        b = (our_argmax == pid) & cortex
        a_sum = int(a.sum())
        b_sum = int(b.sum())
        inter = int((a & b).sum())
        denom = a_sum + b_sum
        dice = float("nan") if denom == 0 else (2.0 * inter) / denom
        name = annot_names[int(pid)] if int(pid) < len(annot_names) else f"ROI_{int(pid):03d}"
        per_parcel.append(
            {
                "bna_id": int(pid),
                "name": name,
                "published_vertices": a_sum,
                "bake_vertices": b_sum,
                "intersection": inter,
                "dice": dice,
            }
        )

    # 0 = agree or out-of-mask; 1 = boundary disagreement; 2 = interior disagreement.
    dis = np.zeros((n_vert,), dtype=np.float32)
    disagree = labeled & (~agree)
    dis[disagree & is_boundary] = 1.0
    dis[disagree & (~is_boundary)] = 2.0
    out_dir.mkdir(parents=True, exist_ok=True)
    save_mgh(dis, out_dir / f"{hemi}.disagreement.{kind}.mgh")
    save_mgh(our_argmax.astype(np.float32), out_dir / f"{hemi}.our_argmax.{kind}.mgh")

    subcort_on_cortex = int((subcortical_dominant & cortex).sum())
    return {
        "hemi": hemi,
        "kind": kind,
        "n_cortex": int(cortex.sum()),
        "n_labeled": n_labeled,
        "overall_agreement": (n_agree / n_labeled) if n_labeled else float("nan"),
        "interior_agreement": (n_interior_agree / n_interior) if n_interior else float("nan"),
        "boundary_agreement": (n_boundary_agree / n_boundary) if n_boundary else float("nan"),
        "n_interior": n_interior,
        "n_boundary": n_boundary,
        "subcortical_full_argmax_on_cortex": subcort_on_cortex,
        "per_parcel": per_parcel,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bake-dir", type=Path, default=DEFAULT_BAKE_DIR)
    parser.add_argument("--annot-dir", type=Path, default=DEFAULT_ANNOT_DIR)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--kinds",
        nargs="+",
        default=["smoothed", "raw"],
        choices=["smoothed", "raw"],
        help="Which bake kinds to verify. Running both isolates smoothing drift.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load each annot once; both kinds compare against the same ground truth.
    annots: dict[str, tuple[np.ndarray, list[str]]] = {}
    for hemi in HEMIS:
        annot_path = args.annot_dir / f"{hemi}.BN_Atlas.annot"
        labels, names = read_published_annot(annot_path)
        validate_annot_ordering(names, hemi, annot_path)
        annots[hemi] = (labels, names)
        save_mgh(
            labels.astype(np.float32),
            out_dir / f"{hemi}.published_argmax.mgh",
        )

    # Fsaverage geometry per hemi.
    adj_by_hemi: dict = {}
    cortex_by_hemi: dict[str, np.ndarray] = {}
    for hemi in HEMIS:
        labels = annots[hemi][0]
        n_vert = labels.shape[0]
        cortex_by_hemi[hemi] = cortex_mask(args.subjects_dir, hemi, n_vert)
        adj_by_hemi[hemi] = build_edge_adjacency(
            args.subjects_dir / "fsaverage" / "surf" / f"{hemi}.pial",
            n_vert,
        )

    summary: dict = {
        "inputs": {
            "bake_dir": str(args.bake_dir),
            "annot_dir": str(args.annot_dir),
            "subjects_dir": str(args.subjects_dir),
        },
        "by_hemi_kind": {},
    }

    for kind in args.kinds:
        atlas = load_baked_atlas(args.bake_dir, kind=kind)
        if len(atlas.frame_ids) != DEFAULT_BNA_N_ROIS:
            raise ValueError(
                f"bake is partial: {len(atlas.frame_ids)} frames, expected {DEFAULT_BNA_N_ROIS}"
            )
        for hemi in HEMIS:
            bake = atlas.values_by_hemi[hemi]
            labels, names = annots[hemi]
            result = verify_hemi(
                hemi=hemi,
                bake=bake,
                annot_labels=labels,
                annot_names=names,
                cortex=cortex_by_hemi[hemi],
                adj=adj_by_hemi[hemi],
                out_dir=out_dir,
                kind=kind,
            )

            csv_path = out_dir / f"{hemi}.per_parcel.{kind}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "bna_id",
                        "name",
                        "published_vertices",
                        "bake_vertices",
                        "intersection",
                        "dice",
                    ],
                )
                writer.writeheader()
                for row in result["per_parcel"]:
                    writer.writerow(row)

            summary["by_hemi_kind"][f"{hemi}_{kind}"] = {
                k: v for k, v in result.items() if k != "per_parcel"
            }
            print(
                f"{hemi} {kind:>8s}:  "
                f"overall={result['overall_agreement']:.3f}  "
                f"interior={result['interior_agreement']:.3f}  "
                f"boundary={result['boundary_agreement']:.3f}  "
                f"(n_int={result['n_interior']}, n_bnd={result['n_boundary']})"
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
