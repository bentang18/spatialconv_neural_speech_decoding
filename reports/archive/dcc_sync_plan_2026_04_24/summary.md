# A5 — DCC sync diff for Stage-3 sEEG D-cohort (2026-04-24)

SSH-only enumeration of `/datacommons/coganlab/` + `/hpc/group/coganlab/` vs. the Box mount on laptop. Zero writes — this is a recipe, not an execution.

## Locations on DCC (read-only source-of-truth)

| Kind | DCC path | Count |
|---|---|---:|
| Recons (main) | `/datacommons/coganlab/ECoG_Recon_Full/D*` | 110 |
| Recons (newer, overflow) | `/hpc/group/coganlab/ECoG_Recon/D*` | 23 |
| BIDS · Phoneme sequencing | `/datacommons/coganlab/BIDS-1.4_Phoneme_sequencing/BIDS/sub-D*` | 50 |
| BIDS · LexicalDecRepDelay | `/datacommons/coganlab/BIDS-1.0_LexicalDecRepDelay/BIDS/sub-D*` (mirror at `/hpc/.../BIDS/`) | 21 |
| BIDS · LexicalDecRepNoDelay | *(not on DCC)* | 0 |
| BIDS · SentenceRep | `/datacommons/coganlab/BIDS-1.4_SentenceRep/BIDS/sub-D*` | 34 |

Box totals for comparison: 129 recons, 50/52/26/34 D-patients for PS / LexDelay / LexNoDelay / SentenceRep.

## Diff summary

| Asset | Box-only (push to DCC) | DCC-only | Intersect |
|---|---:|---:|---:|
| Recons | 18 | 1 | 111 |
| PS BIDS | 0 | 0 | 50 |
| LexDelay BIDS | 31 | 0 | 21 |
| LexNoDelay BIDS | 26 | 0 | 0 |
| SentenceRep BIDS | 0 | 0 | 34 |

- **PS and SentenceRep are fully on DCC.** Stage-3 PS training can start without any BIDS rsync.
- **LexicalDecRepNoDelay is entirely Box-only** — 26 D-patients, and the root doesn't exist on DCC at all. Must be created (`/hpc/group/coganlab/BIDS-1.0_LexicalDecRepNoDelay/BIDS/`) before rsync.
- **LexDelay missing 31 D-patients on DCC** — these are mostly lower-D (D0023–D0047) and a cluster of newer D-patients (D0100+).
- Only **D139** is on DCC (`/hpc/group/coganlab/ECoG_Recon/D139B`) and not on Box.
- 18 Box-only recons fall in D116–D144; most are newer than the cohort scoping reports.

## /work/ht203/data state (DCC-side cache)

Currently contains uECoG S-patient data only (7 recons, BIDS with `sub-S*`). Zero D-patients. Stage-3 training must either:

- (a) **read from `/datacommons/...` directly** (no copy needed, but jobs are slower when datacommons is throttled), **or**
- (b) **symlink the needed patients into `/work/ht203/data/BIDS/sub-D*`** matching the uECoG layout, **or**
- (c) **rsync the 85 scoped D-patients into `/work/ht203/data/BIDS/sub-D*`** (costly but autonomous — /work auto-purges 75 d, so `/hpc/group/coganlab/ht203/` is the durable home).

Recommendation: **(b) — symlink** `/datacommons/coganlab/BIDS-1.4_Phoneme_sequencing/BIDS/sub-D*` → `/work/ht203/data/BIDS/`. Free, fast, and avoids duplication. The 57 Box-only patients (LexDelay 31 + LexNoDelay 26) rsync into DCC first (see §rsync_commands.sh §2), then symlink with the rest.

## Artifacts we built locally that DCC doesn't have

- `data/atlas/support_cache_v2c_snap_dcohort/` — 122 Tier-1 support CSVs (from B1)
- `data/dcohort_coords/` — 128 RAS coord CSVs (from B2)
- `data/dcohort_manifest.csv` — manifest (from B3, pending)

These are ~2 MB total; §rsync_commands.sh §1 pushes them.

## Files produced

- `missing_recons.txt` — 18 D-patients: Box-only recons
- `dcc_only_recons.txt` — 1 D-patient (D139) DCC-only
- `missing_lexdelay.txt` — 31 D-patients: Box-only LexDelay BIDS
- `missing_lexnodelay.txt` — 26 D-patients: Box-only LexNoDelay BIDS (entire task)
- `dcc_ecog_recon_full.txt` — 110 D-patients on `/datacommons/coganlab/ECoG_Recon_Full/`
- `dcc_hpc_recon.txt` — 23 D-patients on `/hpc/group/coganlab/ECoG_Recon/`
- `rsync_commands.sh` — annotated dry-run rsync script

## Out-of-scope (Nanlin-clarification)

- `D_Data/` under `/datacommons/coganlab/` — has `Associative_Memory`, `Caruso`, `EnvelopeTracking`, `GlobalLocal`, etc. but no visible `Phoneme_Sequencing/`. `SCRIPTS_USAGE.md` at BIDS root says MFA/TextGrid materials are "in `D_Data/Phoneme_Sequencing`" — either the dir exists at a deeper path or the comment is stale. Ask Nanlin for the actual location before trying to rsync.
- Is `/hpc/group/coganlab/BIDS-1.0_LexicalDecRepDelay/BIDS/` an older mirror of `/datacommons/...BIDS/` or a superset? Both report 21 D; assume mirror. Confirm when rsyncing.
