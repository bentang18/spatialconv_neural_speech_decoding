"""Frozen atlas/subparcel token layout for Neural Field Perceiver v14 Phase 1.

Rule (frozen 2026-04-14 per blocker #4):

    parcel p is Tier-1 iff argmax_wins(p) >= 10

where ``argmax_wins(p)`` is the number of Phase-1 LH electrodes for which
``p`` is the globally dominant parcel across all 246 BNA ROIs.

Current derivation (2026-04-16 late, blocker #36 closed):

- Spatial pipeline: strict fsaverage snap-to-pial via
  ``src/speech_decoding/v14/fsaverage_projection.py``
  (patient pial -> patient sphere.reg -> fsaverage sphere.reg -> fsaverage pial).
- Atlas bake: ``data/atlas/fsaverage_bake_fast2/`` (mri_vol2surf
  --projfrac-avg 0 1 0.1 from ICBM152_fs, then mri_surf2surf to fsaverage,
  then 2D geodesic smoothing FWHM 3.5 mm).
- Support statistic: direct atlas read at each electrode's fsaverage pial
  vertex. No extra query-time Gaussian — the PSF is baked in.
- Cohort: 7 Phase-1 LH patients (S14, S16, S23, S26, S33, S39, S62) =
  1280 electrodes.
- Ranking source: ``data/atlas/fsaverage_parity/fsaverage_cohort_ranking.csv``
  produced by ``scripts/summarize_fsaverage_support.py``.

Retired pipelines (kept as oracles, not for Tier-1 derivation):

- cvs_avg35 + BNA_PM_dilated_8mm + sigma=1.5 mm 3D Gaussian + trilinear
  (the ``rank_parcels_by_support.py`` flow). The 2026-04-16 cras-bug
  investigation disproved the earlier thesis that "volumetric vs surface
  lookup is load-bearing" — the real divergence was a coordinate frame
  error in ``sub2AvgBrainClinical.m``. See
  ``docs/session_cras_fix_2026-04-16.md``.

Rule history (kept for context):

1. "Top-12 by effective_N" — a quantity rule masquerading as a quality rule.
2. "effective_N >= 10 AND argmax_wins >= 1" — caught leakage sinks but
   excluded primary auditory cortex on a technicality.
3. "argmax_wins >= 10" (current) — directly measures "how many electrodes
   have this parcel as their true anatomical home." On the fsaverage
   cohort ranking the argmax band [5, 9] is empty: any threshold in [5, 10]
   yields the same 15 parcels.

Cardinality is an output of the rule, not an input. The current derivation
happens to land at 15 parcels again; 13 overlap with the cras-corrected
cvs_avg35 list (87%), confirming cross-registration stability.

Uniform ``k_parcel = 1`` (no 2-token splits in Phase 1). With the 2026-04-16
late B-1 amendment, ``DEFAULT_BASE_PARCELS`` is no longer used to build
per-parcel pooled tokens — it is the embedding-lookup table for the soft
parcel embedding (Stage 3 of the B-1 architecture). The 2-token-per-parcel
split variant is therefore not currently a tracked ablation. See
``docs/v14_core_contract_amendment_2026-04-16.md``.
"""

from __future__ import annotations


PROVISIONAL_TOKEN_SPEC: bool = False


def assert_token_spec_frozen() -> None:
    """Assert that the Phase-1 token spec is frozen. No-op since 2026-04-14."""

    if PROVISIONAL_TOKEN_SPEC:
        raise RuntimeError(
            "token_spec is still provisional — blocker #4 must close before "
            "DEFAULT_BASE_PARCELS / DEFAULT_SPLIT_COUNTS may enter a v14 code "
            "path. See docs/implementation_tasks.md #4."
        )


# 15 LH Brainnetome parcels selected by argmax_wins >= 10 on the fsaverage
# cohort ranking (7 patients, 1280 electrodes; derived 2026-04-16 late).
# Ordered by argmax_wins high -> low (primary), token_support high -> low
# (tiebreak). Tuple of (parcel_name, BNA_index_1based) pairs so name-to-index
# and ordering are both canonical in a single source.
DEFAULT_BASE_PARCELS: tuple[tuple[str, int], ...] = (
    ("A4hf",        53),   # PrG_L_6_1   wins=307  tsup=178.18  pts=7  face/head M1
    ("A1/2/3ulhf", 155),   # PoG_L_4_1   wins=246  tsup=169.47  pts=6  face/head S1
    ("A1/2/3tonIa", 157),  # PoG_L_4_2   wins=151  tsup=103.42  pts=6  tongue S1
    ("A4tl",        61),   # PrG_L_6_5   wins=110  tsup= 90.00  pts=7  tongue M1
    ("IFJ",         17),   # MFG_L_7_2   wins= 89  tsup= 49.16  pts=4  inferior frontal junction
    ("A44d",        29),   # IFG_L_6_1   wins= 65  tsup= 45.71  pts=5  Broca dorsal
    ("A45c",        33),   # IFG_L_6_3   wins= 60  tsup= 55.19  pts=5  Broca caudal
    ("A6cvl",       63),   # PrG_L_6_6   wins= 59  tsup= 67.74  pts=7  ventral premotor
    ("A9/46v",      21),   # MFG_L_7_4   wins= 38  tsup= 24.37  pts=4  ventral 9/46
    ("A40rd",      139),   # IPL_L_6_3   wins= 33  tsup= 26.69  pts=5  supramarginal rostrodorsal (PFt)
    ("IFS",         31),   # IFG_L_6_2   wins= 31  tsup= 42.75  pts=5  inferior frontal sulcus
    ("A2",         159),   # PoG_L_4_3   wins= 26  tsup= 56.95  pts=6  area 2 (dorsal S1)
    ("A40rv",      145),   # IPL_L_6_6   wins= 24  tsup= 24.80  pts=5  supramarginal rostroventral (PFop)
    ("A44v",        39),   # IFG_L_6_6   wins= 18  tsup= 24.32  pts=5  Broca ventral
    ("A22r",        79),   # STG_L_6_6   wins= 17  tsup=  8.17  pts=3  rostral area 22
)

# Uniform k_parcel = 1 in Phase 1. The 2-token variant is a named ablation
# and requires a future parcel elongation metric before it can be derived.
DEFAULT_SPLIT_COUNTS: dict[str, int] = {name: 1 for name, _ in DEFAULT_BASE_PARCELS}


def default_token_count() -> int:
    """Return the current default atlas/subparcel token count (N_tok = 15)."""

    return sum(DEFAULT_SPLIT_COUNTS.values())
