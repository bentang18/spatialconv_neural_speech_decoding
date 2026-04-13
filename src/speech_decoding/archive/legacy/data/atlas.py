"""Brainnetome atlas ROIs for speech motor cortex virtual electrodes.

MNI centroid coordinates computed from official Brainnetome atlas NIfTI
(Fan et al. 2016, Cerebral Cortex) and verified against our electrode data.

SPEECH_ROIS_CORE (16 ROIs) — the default set for v12. Top 16
left-hemisphere ROIs by patient reachability (<25mm threshold),
verified against all 11 patients (2026-04-06). Top 15 have ≥4 patients;
#16 (A2) has 3 patients. Categories: Motor(3), Sensory(3), Broca's(6),
Auditory(2), Insula(1), Executive(1).

SPEECH_ROIS_EXTENDED (8 additional ROIs) — for analysis/ablation.
These are poorly covered (≤3 patients) or unreachable (0 patients).

Right-hemisphere patients (S22, S58) have their electrodes mirrored to
left before computing distances.

Usage:
    from speech_decoding.data.atlas import get_virtual_electrode_positions
    positions = get_virtual_electrode_positions()       # (16, 3) core set
    positions = get_virtual_electrode_positions("all")  # (24, 3) full set
"""
from __future__ import annotations

import numpy as np


# Brainnetome atlas LEFT hemisphere MNI centroids for speech motor cortex.
# Source: Fan et al. 2016, Table 1 (Cerebral Cortex 26(8):3508-3526).
# Format: (label, full_name, x, y, z)

# Core set: top 16 ROIs by patient reachability (<25mm), verified 2026-04-06.
# Ranked by #patients reaching each ROI. Top 15 have ≥4 patients; #16 has 3.
# Atlas centroids from Brainnetome NIfTI, rounded to integers.
SPEECH_ROIS_CORE: list[tuple[str, str, float, float, float]] = [
    # --- Motor (3) ---
    ("A6cvl", "Ventral PMC (PrG_6_6)", -49, 6, 31),         # 9 pts, motor planning
    ("A4tl", "M1 tongue (PrG_6_5)", -52, 2, 9),             # 7 pts, articulatory execution
    ("A4hf", "M1 face (PrG_6_1)", -49, -6, 40),             # 6 pts, articulatory execution
    # --- Sensory (3) ---
    ("A1/2/3tonIa", "S1 tongue (PoG_4_2)", -56, -12, 17),   # 8 pts, sensory feedback
    ("A1/2/3ulhf", "S1 face (PoG_4_1)", -49, -15, 44),      # 5 pts, sensory feedback
    ("A2", "S1 proprioceptive (PoG_4_3)", -45, -29, 50),     # 3 pts, proprioceptive feedback
    # --- Broca's (6) ---
    ("A44d", "Broca BA44 dorsal (IFG_6_1)", -46, 15, 25),    # 8 pts, speech production
    ("A45c", "Broca BA45 caudal (IFG_6_3)", -52, 25, 13),    # 8 pts, speech production
    ("A44v", "Broca BA44 ventral (IFG_6_6)", -52, 15, 8),    # 7 pts, speech production
    ("A45i", "Broca BA45 mid (IFG_6_2)", -47, 34, 15),       # 7 pts, language production
    ("A45r", "Broca BA45 rostral (IFG_6_4)", -49, 38, -1),   # 5 pts, language comprehension
    ("A44op", "Broca opercular (IFG_6_5)", -39, 25, 5),      # 4 pts, opercular
    # --- Auditory (2) ---
    ("STGpp", "Planum polare (STG_6_6)", -55, -1, -9),       # 4 pts, auditory-motor
    ("STGa", "Anterior STG (STG_6_5)", -45, 13, -18),        # 4 pts, speech perception
    # --- Insula (1) ---
    ("INSa", "Anterior insula (INS_6_6)", -38, 7, 5),        # 4 pts, articulatory planning
    # --- Executive (1) ---
    ("MFG", "Dorsolateral PFC (MFG_7_2)", -41, 15, 37),      # 6 pts, working memory
]

# Extended set: poorly-covered or unreachable ROIs (for ablation/analysis)
_SPEECH_ROIS_EXTENDED: list[tuple[str, str, float, float, float]] = [
    # 3 patients (parietal / posterior coverage)
    ("SMG", "Supramarginal gyrus (IPL_6_1)", -55, -33, 34),  # 3 pts, phonological WM
    ("Spt", "Sylvian-parietal-temporal", -53, -35, 22),       # 3 pts, auditory-motor
    ("pSTG", "Posterior STG (STG_6_2)", -60, -30, 10),        # 3 pts, phoneme perception
    # 2 patients
    ("A6cdl", "Dorsal PMC (PrG_6_2)", -31, -9, 59),          # 2 pts, dorsal motor
    # Unreachable (0 patients)
    ("SMA", "SMA proper (SFG_7_1)", -5, -5, 55),             # 0 pts, too medial
    ("preSMA", "Pre-SMA (SFG_7_2)", -7, 10, 50),             # 0 pts, too medial
    ("A4ul", "M1 upper limb (PrG_6_3)", -26, -25, 63),       # 0 pts, too dorsal
    ("A1/2/3tru", "S1 trunk (PoG_4_4)", -21, -35, 68),       # 0 pts, too dorsal
]

SPEECH_ROIS_ALL = SPEECH_ROIS_CORE + _SPEECH_ROIS_EXTENDED

# Backward compatibility
SPEECH_ROIS = SPEECH_ROIS_ALL


ROI_CATEGORIES: dict[str, str] = {
    "A6cvl": "Motor",
    "A4tl": "Motor",
    "A4hf": "Motor",
    "A1/2/3tonIa": "Sensory",
    "A1/2/3ulhf": "Sensory",
    "A2": "Sensory",
    "A44d": "Broca",
    "A45c": "Broca",
    "A44v": "Broca",
    "A45i": "Broca",
    "A45r": "Broca",
    "A44op": "Broca",
    "STGpp": "Auditory",
    "STGa": "Auditory",
    "INSa": "Insula",
    "MFG": "Executive",
    "A6cdl": "Motor",
    "SMG": "Parietal",
    "Spt": "Parietal",
    "pSTG": "Auditory",
    "SMA": "Medial",
    "preSMA": "Medial",
    "A4ul": "Motor",
    "A1/2/3tru": "Sensory",
}

# Brainnetome LUT names and integer indices from bnatlas.nii.txt.
ROI_BRAINNETOME_NAME: dict[str, str] = {
    "A6cvl": "PrG_L_6_6",
    "A4tl": "PrG_L_6_5",
    "A4hf": "PrG_L_6_1",
    "A1/2/3tonIa": "PoG_L_4_2",
    "A1/2/3ulhf": "PoG_L_4_1",
    "A2": "PoG_L_4_3",
    "A44d": "IFG_L_6_1",
    "A45c": "IFG_L_6_3",
    "A44v": "IFG_L_6_6",
    "A45i": "IFG_L_6_2",
    "A45r": "IFG_L_6_4",
    "A44op": "IFG_L_6_5",
    "STGpp": "STG_L_6_6",
    "STGa": "STG_L_6_5",
    "INSa": "INS_L_6_6",
    "MFG": "MFG_L_7_2",
    "A6cdl": "PrG_L_6_2",
    "SMG": "IPL_L_6_1",
    "Spt": "SPL_L_5_1",
    "pSTG": "STG_L_6_2",
    "SMA": "SFG_L_7_1",
    "preSMA": "SFG_L_7_2",
    "A4ul": "PrG_L_6_3",
    "A1/2/3tru": "PoG_L_4_4",
}

ROI_BRAINNETOME_INDEX: dict[str, int] = {
    "A6cvl": 63,
    "A4tl": 61,
    "A4hf": 53,
    "A1/2/3tonIa": 157,
    "A1/2/3ulhf": 155,
    "A2": 159,
    "A44d": 29,
    "A45c": 33,
    "A44v": 39,
    "A45i": 31,
    "A45r": 35,
    "A44op": 37,
    "STGpp": 79,
    "STGa": 77,
    "INSa": 173,
    "MFG": 17,
    "A6cdl": 55,
    "SMG": 183,
    "Spt": 211,
    "pSTG": 71,
    "SMA": 1,
    "preSMA": 3,
    "A4ul": 57,
    "A1/2/3tru": 161,
}

# Official Brainnetome candidate omitted from the active v14 core set but worth
# inspection in the surface viewer because it is marginally reachable in this cohort.
OFFICIAL_CANDIDATE_ROIS = ["A6cdl"]


def _get_roi_list(variant: str = "core") -> list[tuple[str, str, float, float, float]]:
    if variant == "core":
        return SPEECH_ROIS_CORE
    elif variant == "all":
        return SPEECH_ROIS_ALL
    raise ValueError(f"Unknown variant: {variant!r}. Use 'core' or 'all'.")


def get_virtual_electrode_positions(variant: str = "core") -> np.ndarray:
    """Return (N_rois, 3) array of MNI centroid coordinates.

    Args:
        variant: "core" (16 ROIs, default) or "all" (24 ROIs including extended).
    """
    rois = _get_roi_list(variant)
    return np.array([[x, y, z] for _, _, x, y, z in rois], dtype=np.float64)


def get_roi_labels(variant: str = "core") -> list[str]:
    """Return ROI labels in the same order as positions."""
    return [label for label, _, _, _, _ in _get_roi_list(variant)]


def get_roi_names(variant: str = "core") -> list[str]:
    """Return human-readable ROI names in the same order as positions."""
    return [name for _, name, _, _, _ in _get_roi_list(variant)]


def get_roi_categories(variant: str = "core") -> list[str]:
    """Return functional categories aligned to the ROI order."""
    return [ROI_CATEGORIES[label] for label in get_roi_labels(variant)]


def get_brainnetome_roi_name(label: str) -> str:
    """Return Brainnetome LUT name for a project ROI label."""
    return ROI_BRAINNETOME_NAME[label]


def get_brainnetome_roi_index(label: str) -> int:
    """Return Brainnetome integer index for a project ROI label."""
    return ROI_BRAINNETOME_INDEX[label]


def select_active_virtual_electrodes(
    electrode_mni: np.ndarray,
    max_distance_mm: float = 25.0,
    variant: str = "core",
) -> np.ndarray:
    """Return boolean mask of virtual electrodes within range of real electrodes.

    For patient-adaptive virtual electrode selection: only compute
    cross-attention to virtual electrodes that the patient's array
    can actually reach.

    Args:
        electrode_mni: (N_elec, 3) MNI coordinates of real electrodes.
        max_distance_mm: Maximum distance to consider "reachable".
        variant: "core" or "all".

    Returns:
        (N_rois,) boolean mask — True for active virtual electrodes.
    """
    distances = compute_electrode_roi_distances(electrode_mni, variant=variant)
    nn_distances = distances.min(axis=0)  # closest real electrode per VE
    return nn_distances <= max_distance_mm


def compute_electrode_roi_distances(
    electrode_mni: np.ndarray,
    variant: str = "core",
) -> np.ndarray:
    """Compute pairwise distances between electrodes and virtual electrodes.

    Args:
        electrode_mni: (N_elec, 3) MNI coordinates of real electrodes.
        variant: "core" (10 ROIs) or "all" (16 ROIs).

    Returns:
        (N_elec, N_rois) distance matrix in mm.
    """
    roi_pos = get_virtual_electrode_positions(variant)  # (N_rois, 3)
    # Broadcast: (N_elec, 1, 3) - (1, N_rois, 3) → (N_elec, N_rois, 3)
    diff = electrode_mni[:, None, :] - roi_pos[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))  # (N_elec, N_rois)


def compute_coverage_stats(
    electrode_mni: np.ndarray,
    max_distance_mm: float = 15.0,
    variant: str = "core",
) -> dict:
    """Compute how well real electrodes cover the virtual electrode positions.

    Args:
        electrode_mni: (N_elec, 3) MNI coordinates of real electrodes.
        max_distance_mm: Maximum distance to consider "covered".
        variant: "core" (10 ROIs) or "all" (16 ROIs).

    Returns:
        Dict with coverage statistics per ROI and overall.
    """
    distances = compute_electrode_roi_distances(electrode_mni, variant=variant)
    nn_distances = distances.min(axis=0)  # closest electrode per ROI
    nn_electrode_idx = distances.argmin(axis=0)  # which electrode is closest

    roi_labels = get_roi_labels(variant)
    roi_names = get_roi_names(variant)

    per_roi = []
    for i, (label, name) in enumerate(zip(roi_labels, roi_names)):
        per_roi.append({
            "label": label,
            "name": name,
            "nearest_distance_mm": float(nn_distances[i]),
            "nearest_electrode_idx": int(nn_electrode_idx[i]),
            "covered": nn_distances[i] <= max_distance_mm,
        })

    n_covered = sum(1 for r in per_roi if r["covered"])
    rois = _get_roi_list(variant)
    return {
        "n_rois": len(rois),
        "n_covered": n_covered,
        "coverage_fraction": n_covered / len(rois),
        "mean_nn_distance_mm": float(nn_distances.mean()),
        "max_nn_distance_mm": float(nn_distances.max()),
        "per_roi": per_roi,
    }
