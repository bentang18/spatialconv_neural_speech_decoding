# SWEC iEEG dataset — per-subject manifest

`swec_per_subject_manifest.csv` — one row per folder in the HuggingFace release
`NeuroTec/SWEC_iEEG_Dataset`, built by reading the HDF5 metadata of all 68
`IDxx_total.h5` files directly (2026-05-20).

| column | meaning |
|---|---|
| `folder` | ID01–ID68 |
| `channels` | channel count of `data/ieeg` |
| `sampling_rate_hz` | 512 or 1024 |
| `n_samples` | exact sample count of `data/ieeg` |
| `hours` | `n_samples / sampling_rate_hz / 3600` |
| `n_seizures` | annotated ictal events |
| `duplicate_of` | the folder this one re-exports (blank if not a duplicate) |
| `counts_as_unique_subject` | True for the 50 distinct recordings |

**Finding.** 68 folders, but 50 distinct recordings: ID01–ID18 each match a folder
in ID20–ID40 on channel count, sampling rate, exact sample count, and the full
seizure onset/offset list. All 68 folders summed = 9328.0 h / 704 events; the 50
unique recordings = 6672.3 h / 460 events.

Full audit: `memory/reference_swec_ieeg_dataset_audit_2026_05_19.md`.
