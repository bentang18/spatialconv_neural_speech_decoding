# Neural Field Perceiver v14-Core Implementation Plan

**Plan written:** 2026-04-16. **Amended:** 2026-04-16 late (post-architecture-rewrite). **Scope:** Phase-1 `v14-core` — supervised phoneme decoding on intra-op `uECoG`, fixed-atlas, per-patient baseline (`#31`). Cross-patient shared-weights training (`v14-full`) is a separate follow-up plan.

**Goal:** End-to-end supervised v14-core → per-patient slot-averaged PER on S14, S26, S33, S62 via 5-fold × 3-seed grouped-by-token CV, comparable to the 0.734 S14 baseline.

**Architecture:** Per-electrode tokens (B-1) with soft parcel embedding, whole-grid Conv2d, combined spatiotemporal attention. Frozen by `docs/v14_core_contract_amendment_2026-04-16.md` and the updated `docs/implementation_tasks.md` (`#1-#36` + First-Run Protocol). SC/FC bias is **deferred to ablation** — not in the v14-core baseline.

**Open implementation notes:** `docs/plans/v14-core_open_notes.md` tracks the soft gaps surfaced during plan review (SDPA / FlashAttention routing, GridMixer depth as co-equal baseline, soft-embedding normalization variants, RoPE inline impl, CSV column sanitization, per-patient grid shapes, BNA SC/FC source for A4, `.tex` rewrite). Check it before starting each phase.

**Tech stack:** Python ≥3.11 via `uv`, PyTorch ≥2.0, MNE, MNE-BIDS, pytest. Training on DCC.

**Conventions:**
- Every step cites the freezing blocker (`#N`) in `docs/implementation_tasks.md`.
- New code under `src/speech_decoding/v14/` or `scripts/v14_core/`. Tests under `tests/v14/`. No writes to `src/speech_decoding/archive/legacy/`.
- TDD: write the test, watch it fail, implement, watch it pass, commit. One commit per task.
- Run tests with `.venv/bin/python -m pytest tests/v14/ -q`.

---

## Phase A — Pre-flight gates

Do not start Phase B until both A1 and A2 are green.

### Task A1 — Per-electrode Tier-1 support cache (GATE for `#5` + soft parcel embedding)

**Files:** `scripts/v14_core/build_support_cache.py` · `docs/qc/support_cache_qc_report.md` · `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv` · `tests/v14/test_support_cache.py`

Replaces the deprecated parcel-frames QC task — the new B-1 architecture has no parcel-frame chart. The new pre-flight is to materialize the per-electrode Tier-1 support tensor that the loader needs.

- [ ] Script reads each electrode's assigned fsaverage pial vertex from `data/fsaverage_coords/<pt>_fsaverage_pial.csv`, looks up the baked atlas row at that vertex from `data/atlas/fsaverage_bake_v2c/`, and writes a per-patient CSV with columns `name, support_A4hf, support_A1/2/3ulhf, ..., support_A22r` (15 Tier-1 columns in `DEFAULT_BASE_PARCELS` order). Raw support values, not normalized.
- [ ] Markdown QC report: per patient, the count of electrodes whose `argmax over Tier-1 ∈ Tier-1` (sanity check that there's any overlap), the histogram of `max over Tier-1` values, and the count of electrodes with `sum over Tier-1 < 1` (low Tier-1 mass — would contribute weakly to all parcel embeddings).
- [ ] Test asserts: cache exists for all 4 core patients; column count is 16 (`name` + 15 parcels); column order matches `DEFAULT_BASE_PARCELS`; values are float32 in `[0, 100]`; no NaNs.
- [ ] Commit when cache + report + test exist.

### Task A2 — Coordinate-bridge verifier (GATE for `#12`)

**Files:** `scripts/v14_core/verify_coord_bridge.py` · `docs/qc/coord_bridge_verification.md` · `tests/v14/test_coord_bridge.py`

- [ ] Script implements the `#12` bridge for every core patient: `.fif` channel → amp idx → `(r, c)` via Map 4 (`S14 S26`) or Map 3 (`S33 S62`) → `phys_idx = r * ncols + c + 1` → name from `<pt>.electrodeNames` → lookup in `data/fsaverage_coords/<pt>_fsaverage_pial.csv` (used only for the parcel argmax + support lookup, not coord PE) AND in the new `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv` from A1.
- [ ] Verifier contract per `#12`: every retained `.fif` channel maps to exactly one physical name; every name is in the fsaverage cache exactly once; every name is in the support cache exactly once; the (row, col) on the device grid is recovered for every electrode (Stage 2 needs this). Failures are loud.
- [ ] Test asserts the report exists and every core patient (`S14 S26 S33 S62`) reads `PASS`.
- [ ] Commit.

---

## Phase B — Data layer

### Task B1 — `v14-core` dataset + loader (revised `#13` + `#16`/`#17`/`#18` + `#29`)

**Files:** `src/speech_decoding/v14/dataset.py` · `tests/v14/test_dataset.py` · `tests/v14/fixtures/fake_fif.py`

Closes the v14-core loader contract. Loader contract is REVISED per the 2026-04-16 amendment — no `coords`, no `token_mask`, no `token_support`. Five composable pieces, all unit-testable:

1. `V14Sample` — frozen dataclass with:
   - `signal[N_e, T=301]` float32
   - `patient_id: str`
   - `label[3]` long, alphabetical ARPABET indices
   - `electrode_grid_layout[N_e, 2]` int (row, col) on patient device grid
   - `electrode_grid_shape: tuple[int, int]` per-patient bounding rect (H_p, W_p)
   - `electrode_active_mask[N_e]` bool — non-artifact AND not pad
   - `support[N_e, 15]` float32 — raw BNA prob over Tier-1
2. `decompose_token_to_slot_indices(token: str) -> list[int]` — greedy left-to-right with `ae` as the only 2-char PS symbol; raises if the token doesn't decompose to exactly 3 ARPABET indices. Uses `phoneme_map.normalize_label` + `ARPA_PHONEMES` for the alphabetical index (`#16`/`#17`).
3. `build_sample_from_epoch(...)` — hard-asserts `T == 301` (`#29`), produces a `V14Sample`.
4. `lookup_support_for_kept_channels(support_cache_path, kept_names) -> ndarray[N_e, 15]` — reads `data/atlas/support_cache_v2c_snap/<pt>_support_tier1.csv` (built by Task A1), returns the raw support matrix in `kept_names` order. Raises if any kept name is missing from the cache.
5. `V14TrialDataset` — per-patient `torch.utils.data.Dataset`. Asserts `.fif.event_id` has 52 keys (`#18`); filters `info["bads"]` (`#11`); applies the SOFT `#3` rule (no hard exclusion — `support` carries the suppression naturally; an electrode with `max over Tier-1 == 0` is still emitted but its `parcel_emb` is the zero vector); reads grid layout from the `#12` channel-map bridge for each kept channel; returns one `V14Sample` per trial, reading tokens from `.fif.event_id` per the `#34` closure (`.fif` is authoritative).

**Tests (behavior, not implementation):**
- `V14Sample` field dtypes/shapes per the new contract.
- `build_sample_from_epoch` on a fake 2-trial `mne.EpochsArray` with tokens `"bak", "gup"` produces labels `[2, 0, 5]` and `[3, 7, 6]` (alphabetical ARPABET indices).
- Rejects a 4-phoneme token.
- `support` shape `(N_e, 15)` and dtype `float32`.
- `electrode_grid_layout` rows are within `[0, H_p) × [0, W_p)`.

**Gate:** tests pass; fake-fixture path is end-to-end.

### Task B2 — Batch collator (`#31` + revised loader contract)

**Files:** `src/speech_decoding/v14/dataset.py` (append `collate_v14_batch`) · `tests/v14/test_collate.py`

Simplified vs the prior plan — no `parcel_frames.npz` lookup, no `(u,v)` chart projection, no `argmax_parcel` derivation. The collator just stacks per-trial fields and constructs teacher-forcing `prev_tokens`.

- [ ] Collator stacks samples in a batch:
  - `signal[B, N_e, 301]`
  - `electrode_grid_layout[B, N_e, 2]`
  - `electrode_grid_shape: (H_p, W_p)` shared across batch
  - `electrode_active_mask[B, N_e]`
  - `support[B, N_e, 15]`
  - `labels[B, 3]`
  - `prev_tokens[B, 3]` = `[[-1, label[0], label[1]] for label in labels]` for teacher forcing
  - `patient_id: str` (shared across batch per `#31`)
- [ ] Hard assertion: all samples in a batch share one `patient_id` (grouped-by-patient batches, `#31`), one `N_e`, and one `electrode_grid_shape`.
- [ ] Output dict keys consumed by the model: `signal`, `electrode_grid_layout`, `electrode_grid_shape`, `electrode_active_mask`, `support`, `labels`, `prev_tokens`, `patient_id`.
- [ ] Test builds two fake `V14Sample`s with labels `[2, 0, 5]` and `[3, 7, 6]`, asserts shapes + `prev_tokens == [[-1, 2, 0], [-1, 3, 7]]` + shared `patient_id`/`N_e`/`grid_shape`.

---

## Phase C — Model modules

Each task follows the same discipline: write a shape/behavior test → run to fail → implement → run to pass → commit. No implementation sketches in this plan; the contract below is what the test checks.

### Task C1 — Temporal tokenizer (`#2`/`#6`)

**Files:** `src/speech_decoding/v14/tokenizer.py` · `tests/v14/test_tokenizer.py`

- [ ] Shared per-electrode `Conv1d` (weight sharing by folding `N_e` into batch). Kernel `30` samples, stride `10` samples (hard-asserted). Input `(B, N_e, 301)` → output `(B, N_e, d_model, 28)`. Supports variable `N_e` per forward.

### Task C2 — Whole-grid Conv2d + soft parcel embedding (Stages 2–3, NEW)

**Files:** `src/speech_decoding/v14/grid_mixer.py` · `src/speech_decoding/v14/parcel_embedding.py` · `tests/v14/test_grid_mixer.py` · `tests/v14/test_parcel_embedding.py`

Replaces the deprecated within-parcel Perceiver summarizer (`#26`). Two small modules:

**Stage 2 — `GridMixer`:**
- [ ] Per-time-step 2D Conv with shared weights across `t`. Kernel `3×3`, stride `1`, padding `1`, full channel mixing, `bias=True`. Pre-norm LayerNorm over channel dim, GELU activation, residual `x + Conv(LayerNorm(x)) * mask`.
- [ ] `forward(tokens, electrode_grid_layout, electrode_grid_shape, electrode_active_mask)`:
  - Reshape `(B, N_e, d, T)` → `(B, d, H_p, W_p, T)` using `electrode_grid_layout` (scatter inactive cells to zero, padded cells stay zero).
  - Per-time-step Conv2d with shared weights (fold `T` into batch).
  - Apply `electrode_active_mask` to the post-conv contribution so pad/artifact cells stay zero.
  - Reshape back to `(B, N_e, d, T)` via gather.
- [ ] Single conv layer for baseline. `num_layers` is a config knob (default 1, ablation 2).
- [ ] Tests cover: output shape preserved; padded cells stay zero after forward; backward through the conv is non-NaN; `num_layers=2` works.

**Stage 3 — `SoftParcelEmbedding`:**
- [ ] `nn.Module` holding a learnable `P_emb: nn.Parameter` of shape `(15, d)`, init Xavier uniform.
- [ ] `forward(support: (B, N_e, 15)) -> (B, N_e, d)`: returns `support @ P_emb`. Support is RAW (per the amendment); electrodes with low Tier-1 mass naturally produce small embeddings.
- [ ] Tests cover: output shape `(B, N_e, d)`; zero support input → zero output; gradient flows to `P_emb`.

**Composition in the top-level model (Task C5):** the per-time-step parcel embedding is broadcast across `T` and added to the conv-enriched per-electrode features, then flattened to `(B, N_e × T, d)` for the backbone.

### Task C3 — Combined-attention backbone (revised `#27`)

**Files:** `src/speech_decoding/v14/backbone.py` · `tests/v14/test_backbone.py`

Revised per the 2026-04-16 amendment: factored → combined; `B = 3` blocks; **no SC/FC bias** (deferred to ablation A4).

- [ ] `B = 3` blocks. Each block is one pre-norm combined-attention layer over the flat sequence of `(N_e × T)` tokens, then pre-norm FFN. Heads = 4, `head_dim = 16` (= 64/4), FFN width `4d`, GELU, dropout `0.1`.
- [ ] Combined attention: every `(electrode, time)` token attends to every other `(electrode, time)` token in one shot. RoPE applied to the temporal axis only (rotation depends on `t_a - t_b`, not on the electrode axis).
- [ ] Active-cell mask: `(B, N_e × T)` derived as `electrode_active_mask` broadcast across time. Mask applies on both the key axis (mask out attention TO inactive cells) and the query axis (mask out attention FROM inactive cells); inactive query rows are zeroed in the block output.
- [ ] No SC/FC bias parameters in the baseline backbone. Ablation A4 will add per-(layer, head) `α_SC` and `α_FC` scalars and a precomputed `(N_e, N_e)` soft bias derived from `support @ SC_norm @ support.T` and similar for FC.
- [ ] Tests cover: input/output shape `(B, N_e × T, d)` preserved; inactive-cell rows stay zero after forward; RoPE applied only on temporal axis (active-cell pairs at the same `t` get no rotation); no parameters named `*sc_fc*` or `*sc_gain*` or `*fc_gain*` exist in baseline.

### Task C4 — 3-slot AR decoder (`#9`/`#28`)

**Files:** `src/speech_decoding/v14/decoder.py` · `tests/v14/test_decoder.py`

- [ ] One AR block: causal self-attention over 3 slot queries + cross-attention over flattened backbone memory `(N_e × T, d)`, then FFN, then shared linear vocab head. Shared base query + per-slot embedding + previous-token embedding (BOS is a dedicated parameter, not a vocab index; BOS sentinel `-1` in the `prev_tokens` argument).
- [ ] `forward_teacher(memory, prev_tokens)` returns `(B, 3, 9)` for teacher-forced training (`#9`).
- [ ] `exhaustive_decode(memory)` enumerates all `9^3 = 729` sequences per batch item and returns the argmax of the summed slot log-probs — no greedy, no beam.
- [ ] Tests cover: teacher-forced logits shape; exhaustive decode shape and range `[0, 9)`; exhaustive decode matches a slow `itertools.product` reference on a random memory.

### Task C5 — Top-level model assembly

**Files:** `src/speech_decoding/v14/model.py` · `tests/v14/test_model.py`

- [ ] `NeuralFieldPerceiver(V14Config)` composes tokenizer → grid_mixer → soft_parcel_embedding → backbone → decoder.
- [ ] `forward(batch, mode)` with `mode in {"train", "eval"}`:
  1. `temporal_tokens = tokenizer(batch["signal"])` → `(B, N_e, d, 28)`
  2. `mixed = grid_mixer(temporal_tokens, batch["electrode_grid_layout"], batch["electrode_grid_shape"], batch["electrode_active_mask"])` → `(B, N_e, d, 28)`
  3. `parcel_emb = soft_parcel_embedding(batch["support"])` → `(B, N_e, d)`
  4. `tokens = mixed + parcel_emb.unsqueeze(-1)` (broadcast across `T=28`) → `(B, N_e, d, 28)`
  5. Reshape `(B, N_e, d, 28) → (B, N_e × 28, d)`; build `token_active_mask: (B, N_e × 28)` by broadcasting `electrode_active_mask` across `T`.
  6. `memory = backbone(tokens, token_active_mask)` → `(B, N_e × 28, d)`
  7. If `mode == "train"`: `logits = decoder.forward_teacher(memory, batch["prev_tokens"])` → `(B, 3, 9)`.
     If `mode == "eval"`: `pred = decoder.exhaustive_decode(memory)` → `(B, 3)` long.
- [ ] Tests cover: teacher-forced forward returns `(B, 3, 9)`; eval forward returns `(B, 3)` of long indices in `[0, 9)`; works on a fake batch with `N_e = 100`, `H_p × W_p = 8 × 16 = 128` (some cells masked).

---

## Phase D — Training + evaluation

### Task D1 — Grouped-by-token CV (First-Run Protocol)

**Files:** `src/speech_decoding/v14/cv.py` · `tests/v14/test_cv.py`

- [ ] `make_outer_folds(tokens_per_trial, n_folds=5, seed=0)` assigns each unique token to exactly one fold by round-robin on a seeded shuffle; returns `[(train_idx, test_idx)] * 5` with no token leakage across train/test.
- [ ] `make_val_split(tokens_per_trial, train_idx, val_frac=0.2, seed)` carves 20% of the training tokens (not trials) into a val set — same seed across all 3 training seeds of a given outer fold (First-Run Protocol).
- [ ] Tests cover: folds disjoint and token-disjoint; val split disjoint from remaining train and token-disjoint.

### Task D2 — Slot-averaged PER (`#33`)

**Files:** `src/speech_decoding/v14/eval.py` · `tests/v14/test_eval.py`

- [ ] `slot_averaged_per(pred: (B, 3), true: (B, 3)) -> float` — wrong-slot count / `3·B`.
- [ ] Tests cover: 0.0 on exact match; 1.0 on all-wrong; 1/6 on one wrong slot out of 6.

### Task D3 — Training loop (First-Run Protocol)

**Files:** `src/speech_decoding/v14/train.py` · `tests/v14/test_train_step.py`

- [ ] `train_one_fold`: AdamW LR `1e-3`, WD `1e-4`, cosine decay, 20-epoch linear warmup, grad clip `1.0`, plain slot-wise CE (`#9` — no focal, no smoothing, no mixup, no aug), mixed precision on CUDA, `trials_per_batch=8` with grad-accum to effective 32, max 300 epochs, val every 5 epochs, early stop after 10 non-improving val checks.
- [ ] `evaluate(model, loader)` runs exhaustive decode and returns PER.
- [ ] Test asserts a 10-step `train_step` loop on a random fixed batch drops loss by ≥ 0.1.

### Task D4 — Tiny-subset overfit smoke gate

**Files:** `tests/v14/test_overfit_smoke.py`

- [ ] 4 random trials, 200 AdamW steps, no val. Acceptance: final CE < 0.05; exhaustive-decode exact-match rate > 0.75.
- [ ] **Gate:** do not submit to DCC until this passes locally. A failure here is a structural bug (likely grid reshape in `GridMixer`, broadcast of `parcel_emb` across `T`, decoder BOS, or `token_active_mask` zeroing legitimate cells).

---

## Phase E — DCC benchmark run

### Task E1 — Single-patient CLI + DCC array sbatch

**Files:** `scripts/v14_core/train_v14_core.py` · `scripts/v14_core/v14_core_dcc.sh`

- [ ] CLI takes `--patient --fold --seed --grid-mixer-depth --out-dir`, builds `V14TrialDataset` for one patient, materializes folds via `make_outer_folds(..., seed=0)` (fixed across all v14 comparisons per First-Run Protocol), val split via `make_val_split(..., seed=7)`, trains one fold with one seed at the given GridMixer depth, evaluates on the held-out fold, writes a single `.result.json` row (`patient, fold, seed, grid_mixer_depth, best_val_per, test_per, final_epoch`) plus a jsonl training log.
- [ ] sbatch array over `4 patients × 5 folds × 3 seeds × {depth=1, depth=2} = 120` jobs on `coganlab-gpu`. Per-job: 1 GPU, 32 GB RAM, 8 h wall. Outputs under `/work/ht203/results/v14core/<pt>/depth<d>/`. **GridMixer depth-1 and depth-2 are co-equal first-pass baselines per `v14-core_open_notes.md` N2** — both report into Phase E3 aggregation; the better one carries forward into Phase F. Depth-3 stays as a Phase F follow-on if needed.
- [ ] CLI reuses `resolve_physical_names_for_patient` from the A2 verifier (single source of truth for the `#12` bridge) and `lookup_support_for_kept_channels` from the A1 cache.

### Task E2 — DCC sync + submit

- [ ] `rsync` repo to `/work/ht203/repo/speech` (excluding `.venv`, retired `cvsavg35_bake*`).
- [ ] `rsync` `data/fsaverage_coords/`, `data/atlas/fsaverage_bake_v2c/`, `data/atlas/support_cache_v2c_snap/`, `data/channel_maps/` to `/work/ht203/data/`.
- [ ] Write DCC `configs/paths.yaml` pointing at `/work/ht203/data/BIDS`.
- [ ] Run full `tests/v14/` on DCC before submission — all must pass.
- [ ] `sbatch scripts/v14_core/v14_core_dcc.sh`; monitor with `squeue -u ht203`.
- [ ] `rsync` results back to `results/v14core/` when the array completes.

### Task E3 — Aggregation + report + plan closure

**Files:** `scripts/v14_core/aggregate_v14_core.py` · `docs/results/v14_core_first_run.md`

- [ ] Aggregator: for each `(patient, fold, depth)` take mean PER across 3 seeds; for each `(patient, depth)` report mean ± std of the 5 fold means; population mean = mean of per-patient means (`#33`). Report depth-1 and depth-2 side-by-side per N2; declare the per-patient winner per the closure rule below.
- [ ] Report compares to the 0.734 S14 baseline (per-phoneme MFA + flat head) and the 0.825 population mean from that same baseline recipe.
- [ ] **Plan closure checkpoint:**
  - If per-patient PER is within ~0.02 of the baselines → run the planned ablations (see Phase F), then graduate to `v14-full` (cross-patient shared weights).
  - If PER ≫ baseline on every patient → invoke `superpowers:systematic-debugging`. Usual suspects, in priority order: (1) `GridMixer` reshape on the patient grid (especially non-rectangular S58 once it joins), (2) `SoftParcelEmbedding` support normalization (we use raw, not normalized — verify), (3) decoder BOS / `prev_tokens` wiring, (4) `token_active_mask` zeroing legitimate cells in the backbone.
  - Do not start `v14-full` until per-patient numbers are defensible.

---

## Phase F — Ablations (after v14-core baseline lands)

Run only after Phase E reports defensible per-patient PER. Each ablation is one knob change against the baseline; otherwise identical recipe (same folds, seeds, optimizer, schedule).

### Ablation A1 — `d_model = 128`
- Frozen as the first width ablation (`#15`). One DCC array (60 jobs) at the wider width. Compare per-patient PER mean ± std.

### Ablation A2 — Block count
- Two runs: `B = 2` and `B = 4` (vs `B = 3` baseline). Tests whether combined attention's expressiveness scales with depth on our data.

### Ablation A3 — Conv depth-3 (depth-1 and depth-2 promoted to baseline)
- **Promoted out of Phase F**: `num_layers = 1` and `num_layers = 2` are co-equal first-pass baselines in Phase E1 per `v14-core_open_notes.md` N2 — receptive field ±1 cell vs ±2 cells, identical downstream shapes (stride=1, padding=1 preserves grid).
- Phase F A3 now tests `num_layers = 3` (±3 cells RF) only if either E1 baseline saturates and a deeper local mixing run is warranted.

### Ablation A4 — SC/FC additive logit bias (revised `#8`)
- Re-introduce SC/FC bias in the backbone. Per-(layer, head) `α_SC` and `α_FC` scalars init `0.1`. Soft parcel-pair bias `(N_e, N_e)` precomputed per batch as `support @ SC_norm @ support.T` (and similar for FC), broadcast across temporal pairs. SC = `log1p` then z-score of `BNA SC[15,15]`; FC = Fisher-z then z-score of `BNA FC[15,15]`. One-time data prep script: `scripts/v14_core/build_sc_fc_tier1.py` → `data/atlas/sc_fc_tier1_15.npz`.

### Ablation A5 — Mean+gradient pool linear baseline
- Replace `SoftParcelEmbedding` + `GridMixer` + backbone with: per-electrode tokenizer → per-parcel mean + spatial-gradient pool → linear projection to atlas tokens → linear classification head. The historical `#26` linear ablation, executed as a sanity floor.

### Ablation A6 — Combined vs factored attention head-to-head
- Re-implement the historical factored backbone (`#27` original: spatial-then-temporal) with otherwise identical config. Direct comparison of attention pattern.

### Ablation A7 — Patient-grid bbox intra-parcel PE (PE-2)
- Add per-electrode `(row_in_parcel_bbox / H_p, col_in_parcel_bbox / W_p) ∈ [0,1]²` as additional input to `SoftParcelEmbedding`. Tests whether explicit within-parcel position helps despite the registration-noise argument that motivated dropping it.

---

## Decisions folded into this plan

| Decision | Resolution | Source |
|----------|-----------|--------|
| SC+FC bias | **Deferred to Phase F ablation A4.** Baseline backbone has no SC/FC parameters. | 2026-04-16 amendment |
| Token construction | B-1 with soft parcel embedding (per-electrode tokens, no compression to 15). | 2026-04-16 amendment |
| Spatial mixing | Whole-grid Conv2d (M2, no gating); kernel `3×3`, 1 layer baseline. | 2026-04-16 amendment |
| Backbone attention | Combined spatiotemporal (revised `#27`), `B = 3` blocks. | 2026-04-16 amendment |
| Within-parcel position | None. Sub-parcel cross-patient alignment infeasible per noise floor. | 2026-04-16 amendment |
| Parcel routing | Soft probability via raw `support[N_e, 15]`; no hard argmax, no Tier-1 hard mask. | 2026-04-16 amendment |
| Parcel frames (`#10`) | **Deprecated.** No `parcel_frames.npz` in pipeline. | 2026-04-16 amendment |
| Loader contract (`#13`) | Revised to `signal, electrode_grid_layout, electrode_grid_shape, electrode_active_mask, support, label, patient_id`. No `coords`, no `token_mask`, no `token_support`. | 2026-04-16 amendment |
| Scope | End-to-end: loader + model + trainer + eval + DCC + first 5×3 benchmark run + Phase F ablations. | User answer Q2 |
| Plan location + branch | `docs/plans/v14-core.md`, stay on `main`, no worktree. | User answer Q4 |
| Cohort | Core only (`S14 S26 S33 S62`). Extended LH (`S16 S23 S39`) deferred. | `#30` + user answer Q2 |
| Training mode | Per-patient baseline first; shared-weights cross-patient (`v14-full`) is the *next* plan. | `#31` + user answer Q2 |
| Eval protocol | 5 outer folds fixed once, 3 seeds (`42, 137, 256`), val = 20% of training tokens per fold, same partition across seeds. | First-Run Protocol |

## Explicitly out of scope

- Cross-patient shared-weights training (`v14-full`) — separate plan after v14-core lands.
- Extended LH cohort (`S16 S23 S39`).
- Learned per-patient calibration (Phase 2).
- SSL on continuous uECoG (Phase 1.5).
- Re-implementation of `parcel_frames.npz` builder (deprecated by amendment).
- Hard-argmax / hard-Tier-1 routing path (replaced by soft probability).

## Self-review

**Spec coverage** — every blocker in `docs/implementation_tasks.md` is consumed by exactly one task:
- `#1 #5 #12 #36` → A1, A2, B1, E1 (fsaverage snap, support stat, channel-map bridge)
- `#2 #6` → C1 (temporal tokenizer)
- `#3` → B1 (soft probability rule, no hard exclusion)
- `#4` → C2 via `DEFAULT_BASE_PARCELS` (Tier-1 as embedding-lookup keys)
- `#7` → C2 / C5 (token_support implicit in soft parcel embedding)
- `#8` → Phase F ablation A4 (deferred from baseline)
- `#9 #28` → C4 (decoder)
- `#10` → DEPRECATED, no task
- `#11 #13 #16 #17 #18 #29 #32 #34` → B1 (loader)
- `#15` → already frozen in `config.py`
- `#26` → DEPRECATED, replaced by C2 (GridMixer + SoftParcelEmbedding)
- `#27` → C3 (revised: combined attention, no SC/FC)
- `#30 #31` → E1 (cohort + per-patient batches)
- `#33` → D2 (slot-averaged PER)
- First-Run Protocol → D1, D3, E1

**Type consistency** — model input dict keys (`signal`, `electrode_grid_layout`, `electrode_grid_shape`, `electrode_active_mask`, `support`, `labels`, `prev_tokens`, `patient_id`) are identical between B2 collator, C5 model, and D3 training loop. `V14Sample`, `V14TrialDataset`, `collate_v14_batch`, `slot_averaged_per`, `train_one_fold`, `resolve_physical_names_for_patient`, `lookup_support_for_kept_channels` are reused consistently. Tensor-shape pipeline `(B, N_e, 301) → (B, N_e, d=64, 28) → (B, d, H_p, W_p, 28) [GridMixer] → (B, N_e, d, 28) → +parcel_emb → (B, N_e × 28, d) [backbone] → (B, 3, 9) [decoder]` is consistent end-to-end.
