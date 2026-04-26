#!/usr/bin/env python3
"""Build cvs_avg35 parcel frames and QC figures from the raw atlas bake."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from speech_decoding.v14.cvsavg_projection import (
    DEFAULT_BOX_ROOT,
    DEFAULT_SUBJECTS_DIR,
    DEFAULT_TARGET_SUBJECT,
    ensure_cvsavg35_subject,
)
from speech_decoding.v14.fsaverage_atlas import (
    DEFAULT_BNA_N_ROIS,
    DEFAULT_BNA_TREE,
    assert_full_bake,
    load_baked_atlas,
)
from speech_decoding.v14.parcel_frames import (
    build_parcel_frame,
    load_surface_geometry,
    plot_parcel_frame_qc,
    save_parcel_frames,
)
from speech_decoding.v14.token_spec import DEFAULT_BASE_PARCELS, assert_token_spec_frozen


DEFAULT_BAKE_DIR = PROJECT_ROOT / "data" / "atlas" / "cvsavg35_bake"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "atlas" / "cvsavg35_parity" / "parcel_frames"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bake-dir", type=Path, default=DEFAULT_BAKE_DIR)
    parser.add_argument("--bna-tree", type=Path, default=DEFAULT_BNA_TREE)
    parser.add_argument("--subjects-dir", type=Path, default=DEFAULT_SUBJECTS_DIR)
    parser.add_argument("--box-root", type=Path, default=DEFAULT_BOX_ROOT)
    parser.add_argument("--target-subject", default=DEFAULT_TARGET_SUBJECT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hemi", default="lh", choices=("lh",))
    args = parser.parse_args(argv)

    ensure_cvsavg35_subject(
        subjects_dir=args.subjects_dir,
        box_root=args.box_root,
        target_subject=args.target_subject,
    )
    assert_token_spec_frozen()
    atlas = load_baked_atlas(args.bake_dir, kind="raw", tree_path=args.bna_tree)
    assert_full_bake(atlas, expected_n_rois=DEFAULT_BNA_N_ROIS)
    verts, faces = load_surface_geometry(
        args.subjects_dir / args.target_subject / "surf" / f"{args.hemi}.pial"
    )

    frames = []
    qc_dir = args.out_dir / "qc"
    for parcel_name, parcel_idx_1based in DEFAULT_BASE_PARCELS:
        weights = atlas.values_by_hemi[args.hemi][:, parcel_idx_1based - 1]
        frame = build_parcel_frame(
            parcel_name=parcel_name,
            parcel_idx_1based=parcel_idx_1based,
            hemisphere="L",
            verts=verts,
            faces=faces,
            weights=weights,
        )
        frames.append(frame)
        plot_parcel_frame_qc(
            frame=frame,
            verts=verts,
            weights=weights,
            out_path=qc_dir / f"{parcel_name}.png",
        )
        print(f"{parcel_name}: qc -> {qc_dir / f'{parcel_name}.png'}")

    save_parcel_frames(
        frames,
        args.out_dir / "parcel_frames.npz",
        atlas_dir=args.bake_dir,
        target_subject=args.target_subject,
    )
    print(f"wrote {args.out_dir / 'parcel_frames.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
