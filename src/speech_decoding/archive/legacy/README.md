# Legacy quarantine

Frozen 2026-04-13. Everything in this tree is **pre-v14** code from the Conv2d / grid / virtual-electrode / LOPO / NCA-JEPA era. It is preserved for historical reference only. v14 code **must not** import anything from this tree.

## What's here

- `data/` — grid-based BIDS loader (`bids_dataset.py`), grid reshape (`grid.py`), augmentation for grid inputs (`augmentation.py`), grid collation (`collate.py`), the centroid/distance virtual-electrode atlas logic (`atlas.py`), the untrusted ACPC→MNI coordinate path (`coordinates.py`), sig-channel detection (`sig_channels.py`), audio features (`audio_features.py`).
- `models/` — Conv2d grid read-in (`spatial_conv.py`), linear flatten read-in (`linear_readin.py`), old BiGRU backbone (`backbone.py`), articulatory and flat heads, YAML→model assembler.
- `training/` — per-patient trainer, LOPO trainer/adaptor, CTC utils, MFA-guided trainer, phonological-aux trainer.
- `pretraining/` — NCA-JEPA and related SSL pipeline (BYOL, DINO, VICReg, LeWM, synthetic generators, stage1/2/3).
- `evaluation/` — `content_collapse.py` (v12 diagnostic), `metrics.py` (mixed: edit-distance PER helper + v12 per-position accuracy + regression-pivot R² diagnostics).

## Why it's quarantined

Each of these modules silently embeds a pre-pivot contract that would corrupt v14 Phase 1 if imported:

- grid/channel space instead of token space
- centroid-based virtual electrodes + distance-threshold reachability instead of volumetric Brainnetome PM membership
- untrusted ACPC→MNI transform (top Phase-1 blocker as of 2026-04-13)
- Conv2d(H,W) spatial mechanism instead of within-parcel Perceiver summarizer
- BiGRU + stride=10 as the fixed temporal answer rather than an open blocker
- CTC supervision contract instead of AR decoder with 3 queries
- grid-shape-preserving artifact zeroing that biases any downstream parcel-support statistic
- right-hemisphere mirror_to_left hack that is wrong for volumetric parcel lookups
- SSL objectives designed around electrode neighborhoods, not atlas tokens

See `docs/objectives.md`, `docs/tactics.md`, and memory file `project_v14_phase1_contract_2026_04_13.md` for the Phase-1 contract and the active blocker list.

## If you need functionality from here

Rewrite it fresh inside `src/speech_decoding/v14/` against the Phase-1 contract. Do not copy-paste from this tree without going through the blocker list first. The things most likely to look reusable (artifact detection, grouped-by-token CV, phoneme label mapping, PER edit distance) should each be re-derived from their actual requirements, not inherited.

Grouped-by-token CV and the phoneme label space are the two pieces of legacy code that **were** promoted back into the active tree (`src/speech_decoding/evaluation/grouped_cv.py`, `src/speech_decoding/data/phoneme_map.py`) because they are architecture-neutral. Everything else stays here.
