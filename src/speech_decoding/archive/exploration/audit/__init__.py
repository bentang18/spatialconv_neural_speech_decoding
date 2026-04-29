"""v14 Phase-1 loader gate: phoneme-loading + trial-timing + audio-neural audit.

Closes blocker `#34`. The gate is deterministic per-patient and must return
`pass` (or `warn with workaround`) on every core patient before the v14-core
loader is written. See `docs/tactics.md` #34.

Ground truth (settled 2026-04-16 during S14 pilot):
- `eventsOLD.tsv` is the authoritative trial/timing/token source for PS uECoG.
- `events.tsv` is known-bad (53 unique tokens incl. non-PS, re-labeling bugs).
- Trial-level `.fif` epochs align 100% with `eventsOLD.tsv` and are cropped to
  `#29`'s `[-0.5, 1.0] s` window by the loader (the `.fif` on disk is wider).
- Audio clock offset from ECoG is real and must be estimated per-patient.
"""
from speech_decoding.v14.audit.runner import run_patient_audit

__all__ = ["run_patient_audit"]
