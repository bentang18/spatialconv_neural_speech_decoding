# Stage-3 right-hemisphere parcel expansion — design stub

Docs-only. No code changes here.

## Purpose

Stage-3 brings the Cogan sEEG D-cohort alongside uECoG. Roughly a third of D-patients have right-hemisphere (RH) coverage dominance, and the Phase-1 frozen token set is LH-only (15 Tier-1 Brainnetome parcels, `DEFAULT_BASE_PARCELS` in `token_spec.py`). Keeping 15 LH-only tokens would drop RH contacts to zero-support rows — acceptable for pilots, wasteful for Stage-3 full-corpus scaling.

This doc enumerates the file-change order to go **LH-15 → bilateral-30** (same 15 parcels, both hemispheres) when Stage-3 kicks off. Do **not** execute any of this until a Stage-3 decision gate is met; blocker #4 ("token set frozen at LH-15") stays frozen for all of Stage-1 + Stage-2.

## Change order

Top-down, data-side first, model-side last. Each step lands a self-contained commit.

### 1. `src/speech_decoding/v14/token_spec.py`

Extend `DEFAULT_BASE_PARCELS` from the 15 LH entries to 30 `(name, bna_idx)` pairs covering both hemispheres. BNA 1-based indices: current 15 are LH-odd; add RH-even pairs (each LH area X maps to RH area X+1 in BNA).

Downstream touches: `TIER1_COLUMNS` in `support_cache.py:34` auto-becomes 30-long (derived); header of every cache CSV doubles in width.

### 2. Cache regeneration

- **uECoG**: re-run `scripts/v14_core/build_support_cache.py` across all 11 PS patients + lex patients. New caches overwrite `data/atlas/support_cache_v2c_snap/`.
- **D-cohort**: re-run `scripts/v14_core/build_dpatient_support_cache.py` across all 122 D-patients. New caches overwrite `data/atlas/support_cache_v2c_snap_dcohort/`.
- RAS coord caches (B2) do not change — coords aren't parcelwise.

### 3. `src/speech_decoding/v14/dataset.py` and `phoneme_dataset.py`

- `N_TIER1_PARCELS: int = 15` → `int = len(DEFAULT_BASE_PARCELS)` (derive, don't hard-code). Locations: `dataset.py:40`, `phoneme_dataset.py:38`.
- Shape-assertion messages (`dataset.py:114`, `phoneme_dataset.py` equiv) already interpolate the constant — no change once #1 lands.
- `PhonemeSample` / `TrialSample` docstrings mentioning `(N_e, 15)`: update to `(N_e, 30)` or `(N_e, n_parcels)` — strictly text.

### 4. Model side

- `src/speech_decoding/v14/parcel_embedding.py`: `P_emb` is `nn.Parameter(torch.empty(n_parcels, d_model))`. `n_parcels` is read from config, so doubling it is a config change not a source change. Double-check the init call site actually reads from `len(DEFAULT_BASE_PARCELS)` (or an equivalent dynamic source) and not a literal `15`.
- `src/speech_decoding/v14/config.py:112`: docstring `"n_parcels = 15 matches DEFAULT_BASE_PARCELS"` → drop the literal, keep the derivation.
- `phoneme_model.py:291,324,522`: docstring `(B, N_e, 15)` → `(B, N_e, n_parcels)`. Actual indexing is on shape, not the literal.
- `decoder.py`: sanity-check there is no hard-coded 15 in the temporal decoder or CE head. Current expectation: no dependency (decoder sees `(B, T, d_model)` tokens, not parcels).

### 5. Tests

- `tests/v14/test_phoneme_dataset.py` and `test_phoneme_model_shapes.py`: replace `15` with `n_parcels` via `len(DEFAULT_BASE_PARCELS)` import.
- `tests/v14/test_support_cache*.py` (if present — verify before asserting): regenerate fixtures if any small CSVs are hard-coded to 15 cols.

### 6. Ablation log compatibility

`scripts/v14_core/update_ablation_log.py` does not read the parcel count, but previous-experiment results were trained with 15 parcels. Any pooled/LOPO result post-expansion must land with a distinct `experiment_id` suffix (e.g. `*_rh30`) so the aggregator does not merge pre- and post-expansion runs into the same cell.

## Verification path

After all six steps:

```
.venv/bin/python -m pytest tests/v14/ -q
```

…should show the same number of tests green as before. Any shape-assertion test failing on `30 != 15` points to a literal that step #3 missed.

Smoke-train one fold of a known-good config (`per_cell + partialconv + pe2d + hier d=32 depth=3 pool=(4,8)`) on S14. Expect:
- Loss curve to behave qualitatively the same (higher capacity but LH electrodes still dominate). If it diverges, the new RH parcels are either mis-indexed or receiving wrong support values — trace back to #2.

## Out of scope for this doc

- **D-patient loader integration.** `phoneme_dataset.py` emits `electrode_grid_layout[N_e, 2]` + `electrode_grid_shape[H_p, W_p]` for the uECoG grid-scatter. sEEG depth probes have no 2D grid — either emit pseudo-grid layouts or switch to the per-electrode-token path (B-1 mode). This is a Stage-3 architecture decision, not an expansion-mechanics question.
- **Non-Tier-1 parcel additions.** If Stage-3 wants the full 210-parcel BNA, that's a separate `DEFAULT_BASE_PARCELS` redefinition and falls under the same change order but with 14× the parcel count. Pick bilateral-30 first and validate before broadening.
- **Cross-hemisphere weight sharing.** `P_emb` currently treats LH parcel X and RH parcel X as independent rows. A constraint `P_emb[X_R] = P_emb[X_L]` (tied weights) is a separate architecture call.
