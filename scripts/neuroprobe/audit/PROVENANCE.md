# Provenance — audit scripts and the run IDs behind their numbers

Every number these scripts print must be traceable to *file, line, run ID*. The LEACE and board
artifacts carry **no provenance field**, so run IDs below were recovered by matching each shard's
mtime to the `sacct` End time of the array element that wrote it — every match exact to ≤1 s.
Verified 2026-07-29. Full findings: `memory/project-number-provenance-audit-2026-07-29.md`.

## Artifacts → run IDs

| artifact (on Delta, `/projects/bhqk/htang13/`) | run ID | state | shape |
|---|---|---|---|
| `cs_leace_r6_40k` — enc0 + enc12 erasure | **20544396** | COMPLETED | 10 cells × 15 tasks |
| `cs_leace_r6_40k_enc3` — enc3 erasure | **20565621** | COMPLETED | 10 cells × 15 tasks |
| `leace_ctrl_r6_40k` — `dir_between_frac` etc. | **20566055** | COMPLETED | 10 cells × **1 task** (`onset`) |
| `leace_ctrl_r6_40k_null` — `leace_shuf` + `leace_toppc` arms | **20566474** | **TIMEOUT** | 10 cells, **RAGGED 5–14 tasks** ⇒ use the 5-task intersection |
| `board_shards_cdlin45k_PARCELUNIT` (`ws_*`) | **20580123** | COMPLETED | 12/12 cells |
| `board_shards_cdlin45k_PARCELUNIT` (`csession_*`) | **20580124** | in flight | load-bearing arm |
| vendored Neuroprobe leaderboard JSONs | commit **`f9b0842`** | clean tree | Release 0.1.8, 2026-05-17 |

**The LEACE arrays ran on the `r6_40k` checkpoint, not canonical `cdlin_45k`.** Confirmed two ways:
the directory names say `r6_40k`, and `enc12|std` is bit-identical 150/150 against the 40k board JSON
and 0/150 against 45k. `enc0|std` matches both (there is no encoder at enc0), which is what makes
that fingerprint valid rather than coincidental.

## Scripts

| script | what it recomputes | reads |
|---|---|---|
| `board_number_audit.py` | our board macros vs quoted values, with a 15-task invariant | `results/r6_era/board/*.json` |
| `leace_erasure_audit.py` | paired erasure delta per tap; asserts λ pairing and task completeness | `cs_leace_r6_40k*` |
| `enc3_erasure_free_check.py` | whether "erasure is free at enc3" survives at full N (**it does not**) | `cs_leace_r6_40k*` |
| `parcel_split_readout.py` | the pre-registered event/level split at parcel unit; refuses a <12-cell macro | `board_shards_cdlin45k_PARCELUNIT` |
| `gate_model.py` | gate-vs-penalty fit, `a_event`/`a_level` | `results/r6_era/board/shards_*` |

## Landmines these scripts encode

- **Read SHARDS, not merged board JSONs** — the merges carry `csession: {}` from a stale merge
  (`csession_mean` is literally EMPTY in `results_v3_board_cdlin_45k.json`). `board_number_audit.py`
  falls back to `shards_<tag>/` automatically, which is the **only** route by which the csession
  headline is traceable at all. All four headline macros recompute and PASS:
  CS enc0 **.5872**, CS enc12 **.6036**, WS enc12 **.6897**, CSession enc12 **.6862**
  (15 tasks each; 10 cells for CS, 12 for WS/CSession). Note **.6862, not .6846** — the latter was
  the 40k arm.
- **`parcel_split_readout.py` refuses a macro below 12 cells.** A partial cell set lies.
- **Three incompatible ratio definitions** exist for the event/level comparison: additive
  (cs .033), k-ratio (cs .609), and the gate coefficient `a`. Never swap them; the script prints all.
- **`gate_model.py`'s asymmetry appears only on the `ws→cs` row**, which the script itself prints
  `do not decide` on (it crosses elec→parcel) *and* which fails the script's own sign-agreement
  admissibility check (9/12 at 45k). On the unit-controlled `ws→csession` row there is no asymmetry
  (`a_event` 1.132 vs `a_level` 1.175) and PENALTY beats GATE. **Do not cite `a_event`/`a_level` as
  a fitted gate.**
- **Erasure results are confounded for the separability claim.** AUROC is invariant to a constant
  score shift, so any purely between-domain component of the removed direction is invisible by
  construction. `leace_toppc` (variance-matched, non-identity) is equally free at enc12, which rules
  out "identity got separated" as the explanation. Use `task_identity_overlap`
  (`scripts/neuroprobe/viz_figures.py:434`) to test the geometry directly — never run to date.
