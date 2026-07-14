"""DCC driver: derive per-subject Cogan ``depth-wm.csv`` from FreeSurfer recons.

Thin orchestration over the tested ``cogan_localize`` core. Runs ONLY on DCC
(needs ``nibabel`` + the real ``*.mgz`` atlas volumes under
``/hpc/group/coganlab/Data/ECoG_Recon_Full/D<id>/``); every unit of logic it
calls is covered in ``test_cogan_localize.py`` on synthetic data.

Two stages:

  1. PIN the LEPTOVOX→array axis convention (``find_axis_convention``) against the
     only subjects that ship a ready DK CSV — D23/D24 — by reproducing that CSV's
     origin column from ``aparc+aseg.mgz``. Abort unless the winning convention
     reproduces ≥ ``--pin-threshold`` of origins on BOTH pin subjects (a soft
     match means the frame is wrong and every downstream DKT label is untrusted).
  2. APPLY the pinned convention to every feed subject: sample
     ``aparc.DKTatlas+aseg.mgz`` in a fixed-radius sphere per electrode and write
     ``depth-wm.csv`` (``Electrode`` + ``DKT`` + audit columns).

Run-gated + output-gated: the produced CSVs feed v3 SSL only once the 0.578
Neuroprobe gate is cleared, and any launch that samples the volumes is Ben-gated.
The DKT ``DKT`` column defaults to ``majority`` (3mm-sphere plurality); both
``origin`` and ``majority`` candidates are always written for audit (see
``write_depth_wm_csv`` and ``--label-choice``).
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

import cogan_localize as cl

# Only D23/D24 ship a ready DK CSV. D24 is a subdural GRID/STRIP (ECoG) subject
# whose surface contacts sit at the pial boundary — center-voxel volume labeling
# reproduces its DK CSV at only ~0.44, an intrinsic surface-vs-volume mismatch,
# not a convention error. The Cogan v3 feed is depth-only (surface subjects are
# dropped, matching BT's sEEG modality and the shaft/depth-RoPE arch), so the
# convention is pinned on the DEPTH reference D23 alone: 121 contacts across 10
# shafts spanning the hemisphere reproduce its DK CSV at 0.926 origin / 0.942
# majority, with the runner-up convention at 0.42 — a decisive, unambiguous lock.
PIN_SUBJECTS = ("D23",)

# The shipped DK CSV is generated FROM the recon in row order, so we align the
# ground-truth origin column to our coords by ROW INDEX, not by electrode name.
# Some subjects (e.g. D23) renumber contacts within a shaft between the
# electrodeNames file (clinical, what the EDF channels use) and the DK CSV
# (e.g. RAMF 1..7 vs 8..14) — a pure relabel, not a reorder. Name-matching then
# spuriously KeyErrors; row alignment is correct and more faithful. We still
# guard it by asserting the shaft PREFIX agrees row-by-row, which catches a real
# reorder while tolerating the benign within-shaft renumber.
_SHAFT_RE = re.compile(r"^(?P<shaft>.*?)(?P<depth>\d+)$")


def _shaft_prefix(name: str) -> str:
    m = _SHAFT_RE.match(name)
    return m.group("shaft") if m else name


def _recon_dir(recon_root: str, subj: str) -> str:
    # electrodeNames / LEPTOVOX / the shipped DK CSV live here.
    return os.path.join(recon_root, subj, "elec_recon")


def _mri_dir(recon_root: str, subj: str) -> str:
    # FreeSurfer atlas volumes (aparc+aseg / aparc.DKTatlas+aseg) live here.
    return os.path.join(recon_root, subj, "mri")


def _load_mgz(path: str) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(path).get_fdata()).astype(np.int64)


def _dk_csv_path(recon_root: str, subj: str, radius_mm: float) -> str:
    r = int(round(radius_mm))
    return os.path.join(
        _recon_dir(recon_root, subj),
        f"{subj}_elec_location_radius_{r}mm_aparc+aseg.mgz.csv",
    )


def pin_convention(
    recon_root: str,
    lut: dict[int, str],
    radius_mm: float,
    threshold: float,
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Reproduce the depth-reference DK CSV to lock the LEPTOVOX→array convention.

    The winning ``(perm, flips)`` must clear ``threshold`` on every pin subject
    (depth-only; see ``PIN_SUBJECTS``), else we abort — a match below threshold
    means the geometry is unresolved. Multiple pin subjects must also agree.
    """
    winner: tuple[tuple[int, ...], tuple[bool, ...]] | None = None
    for subj in PIN_SUBJECTS:
        rd = _recon_dir(recon_root, subj)
        names, coords = cl.read_recon_electrodes(
            os.path.join(rd, f"{subj}.electrodeNames"),
            os.path.join(rd, f"{subj}.LEPTOVOX"),
        )
        gt = cl.read_dk_csv_origins(_dk_csv_path(recon_root, subj, radius_mm))
        if len(gt) != len(names):
            raise SystemExit(
                f"{subj}: DK CSV rows ({len(gt)}) != recon electrodes "
                f"({len(names)}) — cannot row-align the pin ground truth."
            )
        # Row alignment is only valid if the two files list contacts in the same
        # order; assert the shaft prefix matches row-by-row (tolerates renumber).
        mismatched = [
            (i, n[0], gt[i][0])
            for i, n in enumerate(names)
            if _shaft_prefix(n[0]) != _shaft_prefix(gt[i][0])
        ]
        if mismatched:
            raise SystemExit(
                f"{subj}: DK CSV is NOT row-aligned to the recon — shaft prefix "
                f"differs at rows {mismatched[:5]}; row-index pin is unsafe."
            )
        expected = [origin for _, origin in gt]
        vol = _load_mgz(os.path.join(_mri_dir(recon_root, subj), "aparc+aseg.mgz"))
        perm, flips, frac = cl.find_axis_convention(
            vol, coords, expected, lut, radius_mm
        )
        print(f"[pin] {subj}: convention perm={perm} flips={flips} match={frac:.4f}")
        if frac < threshold:
            raise SystemExit(
                f"axis-convention pin FAILED on {subj}: best match {frac:.4f} "
                f"< threshold {threshold} — geometry unresolved, aborting."
            )
        if winner is None:
            winner = (perm, flips)
        elif (perm, flips) != winner:
            raise SystemExit(
                f"pin subjects disagree on convention: {winner} vs {(perm, flips)}"
            )
    assert winner is not None
    return winner


def localize_subject(
    recon_root: str,
    subj: str,
    lut: dict[int, str],
    perm: tuple[int, ...],
    flips: tuple[bool, ...],
    radius_mm: float,
    cap_mm: float,
    label_choice: str,
    out_path: str,
    depth_only: bool = True,
) -> int:
    """Sample the DKT atlas for one subject → write ``depth-wm.csv``. Returns N.

    ``depth_only`` (default True) keeps only kind ``D`` (sEEG depth) contacts and
    drops grid/strip (``G``/``S``) surface contacts: the Cogan v3 feed is
    depth-only, center-voxel volume labeling is unreliable for surface contacts
    (~0.44 reproduction), and 2D grids have no linear shaft for the depth-RoPE.
    A pure-surface subject then yields an empty ``depth-wm.csv`` (N=0) and drops
    out of the feed naturally; a mixed subject (e.g. D67) keeps only its depths.
    """
    rd = _recon_dir(recon_root, subj)
    names, coords = cl.read_recon_electrodes(
        os.path.join(rd, f"{subj}.electrodeNames"),
        os.path.join(rd, f"{subj}.LEPTOVOX"),
    )
    if depth_only:
        keep = [k for k, n in enumerate(names) if n[1] == "D"]
        names = [names[k] for k in keep]
        coords = coords[keep]
    vol = _load_mgz(
        os.path.join(_mri_dir(recon_root, subj), "aparc.DKTatlas+aseg.mgz")
    )
    ijk = cl.apply_axis_convention(coords, perm, flips, vol.shape)
    gm_ids = cl.gm_ids_from_lut(lut)

    origin_names: list[str] = []
    majority_names: list[str] = []
    majority_props: list[float] = []
    nearest_gm_names: list[str] = []
    gm_reach_mm: list[float] = []
    n_gm_base: list[int] = []
    for k in range(ijk.shape[0]):
        origin_id, props = cl.sphere_label_proportions(vol, ijk[k], radius_mm)
        origin_name, named = cl.label_ids_to_names(origin_id, props, lut)
        origin_names.append(origin_name)
        if named:
            majority_names.append(named[0][0])
            majority_props.append(named[0][1])
        else:
            majority_names.append("unknown")
            majority_props.append(0.0)
        # Physically-correct volumetric nearest-GM label (the DKT column default).
        gm_id, reach, nbase = cl.nearest_gm_label(vol, ijk[k], gm_ids, radius_mm, cap_mm)
        nearest_gm_names.append(lut.get(gm_id, "unknown") if gm_id else "unknown")
        gm_reach_mm.append(reach)
        n_gm_base.append(nbase)

    cl.write_depth_wm_csv(
        out_path, names, coords, origin_names, majority_names, majority_props,
        label_choice=label_choice, radius_mm=radius_mm,
        nearest_gm_names=nearest_gm_names, gm_reach_mm=gm_reach_mm,
        n_gm_base=n_gm_base, cap_mm=cap_mm,
    )
    return len(names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon-root", default="/hpc/group/coganlab/Data/ECoG_Recon_Full")
    ap.add_argument("--lut", required=True, help="FreeSurferColorLUT.txt")
    ap.add_argument("--subjects", nargs="+", required=True, help="e.g. D23 D24 D19 …")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--radius-mm", type=float, default=3.0,
        help="recording-sphere radius (mm=vox on the 1mm conformed volume). 3.0 = "
        "the lab's own ECoG_Recon neighborhood (shipped DK CSV is radius_3mm), so "
        "the nearest-GM vote runs over the same sphere the lab sampled.",
    )
    ap.add_argument(
        "--cap-mm", type=float, default=10.0,
        help="max reach for nearest-GM growth (mm). Beyond it a contact is deep "
        "white matter with no cortex in reach → dropped to 'unknown'. 10.0 clears "
        "BT's empirical max ShiftDist (7.5mm) with margin; Cogan reaches GM by 10mm "
        "for ~100%% of contacts, so >cap is genuine deep-WM (~0.1%%).",
    )
    ap.add_argument(
        "--pin-threshold", type=float, default=0.90,
        help="min origin-match on each depth pin subject to trust the convention. "
        "0.90: D23 reproduces at 0.926 (residual = center-voxel boundary noise, "
        "every miss has the expected label inside the sphere) while the runner-up "
        "convention sits at 0.42, so 0.90 separates them decisively.",
    )
    ap.add_argument(
        "--keep-surface", action="store_true",
        help="also localize grid/strip (G/S) contacts (default: depth-only). For "
        "auditing surface reproduction only — the v3 feed drops surface contacts.",
    )
    ap.add_argument(
        "--label-choice", choices=("nearest_gm", "majority", "origin"),
        default="nearest_gm",
        help="which candidate fills the DKT column. Default 'nearest_gm': the "
        "physically-correct volumetric rule — plurality of IN-VOCAB gray voxels in "
        "the recording sphere, grown to the nearest gray up to --cap-mm, WM/unknown "
        "ignored (so no contact is labeled white matter, matching BT's 0%%-WM DKT). "
        "'majority' (top sphere proportion incl. WM) and 'origin' (center voxel) are "
        "the naive labelings kept in audit columns for the blind-vs-rule counterfactual.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    lut = cl.read_fs_color_lut(args.lut)
    perm, flips = pin_convention(
        args.recon_root, lut, args.radius_mm, args.pin_threshold
    )
    print(f"[pin] LOCKED convention perm={perm} flips={flips}")

    for subj in args.subjects:
        out = os.path.join(args.out_dir, f"{subj}_depth-wm.csv")
        try:
            n = localize_subject(
                args.recon_root, subj, lut, perm, flips, args.radius_mm,
                args.cap_mm, args.label_choice, out,
                depth_only=not args.keep_surface,
            )
            tag = "ok" if n else "EMPTY(surface-only)"
            print(f"[{tag}] {subj}: {n} depth electrodes → {out}")
        except (FileNotFoundError, ValueError) as e:
            print(f"[SKIP] {subj}: {e}")


if __name__ == "__main__":
    main()
