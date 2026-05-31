# SWEC-iEEG sourcing

Download + audit for the SWEC-iEEG dataset (`NeuroTec/SWEC_iEEG_Dataset` on
HuggingFace, released with MVPFormer arXiv:2506.20354). These scripts ARE the
documented regenerate spec for the `/work` cache (CLAUDE.md storage tiering):
`/work/ht203/data/swec/` is 75-day-purge bulk tier, so re-fetch is one command.

Full dataset audit + rationale: `memory/reference_swec_ieeg_dataset_audit_2026_05_19.md`.

## Storage

`/work/ht203/data/swec/` (mirrors `/work/ht203/data/braintreebank/`). The 50
unique folders are ~3.3 TB compressed. Persistent `/hpc/group/coganlab` has too
little free space (~389 GB) to hold it.

## Pinned revision

`584e9d29313ad6d2ed675b5d5202240f4ff75970` (re-verified 2026-05-30). Fetch +
audit bind to this so we source exactly the bytes we proved.

## Procedure (run on DCC: `.venv/bin/python scripts/swec/<x>.py`)

1. **Verify the dedup** (cheap, over HTTP — no bulk download):
   ```
   .venv/bin/python scripts/swec/verify_swec_dedup.py
   ```
   Re-derives the 50-unique / 18-duplicate split purely from live HF metadata and
   asserts it matches the committed manifest. The HF "68 subjects" headline
   double-counts: ID01-18 are content-identical re-exports (ID20,21,22,24,25,
   27-32,34-40). Bit-compares signal on the top dupe pairs. Pins the revision.

2. **Fetch the 50 unique folders** (skips the 18 dupes -> saves ~1.3 TB and
   avoids over-weighting 18 subjects in SSL):
   ```
   .venv/bin/python scripts/swec/fetch_swec.py             # all 50 unique
   .venv/bin/python scripts/swec/fetch_swec.py --subjects ID19   # one (smoke)
   ```
   Resumable. Set an `HF_TOKEN` (`huggingface-cli login`) for higher rate limits
   on the multi-hour pull.

3. **Audit on-disk integrity + contract**:
   ```
   .venv/bin/python scripts/swec/audit_swec.py --psd       # all unique on disk
   ```
   Per subject: size==HF-sibling + summed-part-length==total + boundary-chunk
   decode; schema/anatomy-blind; `(C, T)` orientation; reconcile vs the 5/19
   manifest; PSD roll-off confirming the 0.5-120 Hz band-limit. Writes
   `verified_manifest.csv`.

## Two dataset landmines (verified 2026-05-30) — affect the DP03 loader

- **Embedded `info/checksums` are STALE.** The parts were re-compressed before
  HF upload, so `b2sum(part)` != the stored checksum. Do NOT use them as an
  integrity gate. Use HF Xet content-verification + size + decode (what
  `audit_swec.py` does).

- **`total.h5` VDS source filenames are CORRUPT — read PART FILES directly.**
  The VDS cites *another patient's* parts (ID19_total -> `ID01_part_1.h5`;
  ID47_total -> `ID29_part_*`). The card's "recommended" total-file access
  returns fill-zeros here and could silently serve the WRONG patient's voltage.
  Authoritative = `info/files` (real part names) + parts concatenated in
  `part_1..part_N` order. The DP03 loader MUST read parts directly, never the VDS.
