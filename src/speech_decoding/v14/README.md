# v14 Package Layout

This package is the clean implementation boundary for the current architecture.

Planned modules:

- `config.py`: dataclass configs
- `token_spec.py`: default parcel/subparcel interface
- `calibration.py`: atlas-resource loading + per-patient gain/offset and geometry calibration
- `tokenizer.py`: shared temporal tokenizer
- `local_summarizer.py`: within-parcel Perceiver summarizer
- `backbone.py`: relational-temporal transformer
- `decoder.py`: `3`-query AR phoneme decoder
- `model.py`: top-level assembly

Atlas default for Phase 1:

- real Brainnetome 4D probability map:
  - `/Users/bentang/Documents/Code/speech/data/atlas/BNA_PM_4D.nii.gz`
- Brainnetome label/MPM map for ROI indexing and sanity checks:
  - `~/nilearn_data/bnatlas.nii.gz`

Phase 1 rule:

- use the real PM file as the active membership source
- do **not** silently fall back to the old smoothed-MPM proxy
- use the MPM file only for ROI indexing / sanity checks

The first implementation target is `uECoG` only.
`sEEG`, SSL, and external datasets should be added only after the end-to-end `uECoG` path is verified.
