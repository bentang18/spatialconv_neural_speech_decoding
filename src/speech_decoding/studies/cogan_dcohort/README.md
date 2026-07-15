# `cogan_dcohort` — Cogan sEEG D-cohort study/loader

Producer-side ingestion of the Cogan-lab (Duke) stereo-EEG D-cohort into the v3
(`v14_converged_v3`) SSL pipeline. Design goal: **byte-identical consumption** — the
Cogan bake emits the same 3-band `|STFT|` cache a Brain Treebank (BT) v3 session
produces, so `dispatch_v3 --session` reads a Cogan session through the *unchanged*
consumer. All new code is on the producer side; the consumer is untouched.

## Modules
- `loader.py` — `_load_raw` (EDF → `mne.io.RawArray` with clean `<shaft><contact>` names;
  `channels.tsv` opened `utf-8-sig` for BOM), `parse_shaft`, and `select_neural` (keep
  SEEG/ECOG, drop mistyped physio/scalp refs and guard-1 `extra_bad`).
- `study.py` — `DCohortStudy`: manifest-driven timeline enumeration (one row per
  `(subject, task, run)`); the extractor (`CARIeegExtractor`) owns notch / HPF / shaft-CAR
  / STFT.
- `guard1_static.py` — reads the static per-`(sid,tid)` bad-channel map (env
  `COGAN_GUARD1_STATIC`) and injects `extra_bad` at load.

## Pipeline (scripts in `scripts/neuroprobe/*cogan*`)
1. `build_cogan_manifest.py` → `cogan_manifest{,.work.csv}` (`.work.csv` = `/work`-staged
   path columns for compute nodes that don't mount the read-only source).
2. `precompute_guard1_static_cogan.py` → `collect_guard1_static_cogan.py` → static
   bad-channel map (spike/artifact detector over raw voltage).
3. `cogan_localize.py` / `localize_cogan_batch.py` → per-patient `D<num>_depth-wm.csv`
   (volumetric nearest-GM DKT parcellation; applied as the load-time `parcel_fn`, so the
   cache is localization-agnostic).
4. `submit_build_cogan_3band_v3_cache.py` (`--cache-only`) → 3-band `|STFT|` cache:
   resample → shaft-CAR → 60 Hz notch → STFT → robust-z → memmap. Uniform `hop=64 @ 2048 Hz
   → 32 fps`. Bands: v3slow (nperseg 1024, 2–14 Hz, 7 bins), v3mid (256, 16–56 Hz, 6 bins),
   hga (128, 64–160 Hz, 7 bins).
5. `precompute_bad_windows_cogan.py` (+ `submit_…`) → guard-2: slide a hot/cat/abs/flat
   detector over the *baked* cache on a 1 s grid → merged bad-window spans per session.

## Cache format
`band_{v3slow,v3mid,hga}/…/<hash>.{npy,stats.npz,json}` where `.npy` = raw `|STFT|`
`(C,F,T)` float32, `.stats.npz` = robust-z `median`/`sigma` `(C,F,1)` (applied at LOAD with
`sigma_floor=1e-6`), `.json` = frame/channel/rate metadata (`sample_rate=2048`, `band_hop=64`).
Ground-truth baked count = `find band_<b> -name '*.json' | wc -l` (files nest under the
cachekey subdir, so a top-level `*.npy` glob reads 0).

## Coordinates & notes
ACPC native coordinates (MNI is banned for intracranial). 60 Hz mains (Duke/US). SSL is
label-agnostic → all 7 cognitive tasks are kept. `global_subject_id 1XXX ↔ D<int(XXX)>`;
re-implant variants carry a trailing letter (`sub-D0107A` ↔ `D107A`).

Full corpus/QC reference (internal, not committed here per the repo's docs policy): see
`COGAN_INGESTION.md` staged with the corpus on the compute cluster.
