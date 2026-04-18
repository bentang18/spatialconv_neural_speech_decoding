# v14-core Implementation Open Notes

Living capture of soft gaps surfaced during the 2026-04-16 late plan review. Each note has a target phase and a decision needed before that phase can finish. This is a notepad, not a contract — once a decision lands, fold it back into `docs/plans/v14-core.md` or `docs/implementation_tasks.md` and remove from here.

Authority: `docs/v14_core_contract_amendment_2026-04-16.md` is the architecture contract; `docs/plans/v14-core.md` is the task-level plan; this doc is the open-questions queue.

---

## Architecture / Stage decisions

### N1. Combined-attention memory path → use PyTorch SDPA (FlashAttention)
**Phase**: C3 + E pre-flight.
**Risk**: At `d_model = 128` (ablation A1), v14-full padding to `N_e ≈ 288` (S58) gives `L = N_e × T = 8064`. Naive attention materializes an `L² = 65M`-entry map per head per sample. At fp16 × 4 heads × batch 8 × 3 layers ≈ 3.1 GB just for attention maps — does not fit on RTX 5000 (32 GB).
**Decision**: backbone attention uses `torch.nn.functional.scaled_dot_product_attention` (FlashAttention 2 / mem-efficient backend). Memory drops to `O(B·H·L·d)` ≈ tens of MB per layer. Do **not** hand-roll `softmax(QK^T / sqrt(d_k)) @ V`.
**Mask compatibility**: SDPA preserves FlashAttention routing only when `attn_mask` is boolean and broadcastable to `(B, H, L, L)`. Keep `token_active_mask` boolean. The "inactive query rows zeroed in block output" step from the amendment is post-attention — not part of the SDPA call.
**Verification (Phase E pre-flight)**: 30-line script constructs worst-case shapes (`B=8, L=8064, d=128, H=4, fp16, num_blocks=3`), runs forward + backward, reports `torch.cuda.max_memory_allocated()`. Must fit comfortably under 32 GB before E1 sbatch.

### N2. GridMixer depth-1 and depth-2 as co-equal baselines
**Phase**: C2 + E1.
**Reason**: With `num_layers = 1` (k=3), each electrode aggregates info from ±1 grid neighbor (3×3 RF) before being handed to the backbone, where attention has no spatial locality prior (RoPE temporal only). Spatial inductive bias = "±1 cell of conv mixing + global set-attention." The missing middle is medium-range spatial structure (3-7 cell radius). With depth=2 the RF grows to ±2 cells (5×5).
**Decision**: run depth-1 (current baseline) **and** depth-2 GridMixer in the first DCC array (Phase E1) as two co-equal baselines, not depth-1 first and depth-2 later as ablation A3. Depth-3 stays as a Phase F follow-on.
**Cost**: ~37K params per GridMixer layer at d=64; doubling depth is trivial in compute and storage. Wall-clock impact is dominated by backbone, not Stage 2.

### N3. Soft parcel embedding normalization — three A7-family variants
**Phase**: A1 (data) + C2 (model) + F (ablation A7 family).
**Risk**: Raw `support` in `[0, 100]` gives boundary electrodes (split 30+50) ~2× lower parcel-identity signal magnitude than dead-center electrodes (support ~80) after `support @ P_emb`. Conv2d-mixed HGA feature carries 64 channels at its own scale; magnitude mismatch implicitly down-weights parcel evidence on boundary electrodes.
**Decision**: keep raw as the baseline (per amendment); add three A7-family variants as ablations:
- **A7a**: `support` row-sum normalized (`support / sum(support, dim=-1, keepdim=True)`). Separates "which parcels" from "how much evidence."
- **A7b**: `support` L2-normalized + total-mass scalar concatenated as one extra input feature. Same factorization, unit-norm parcel mix.
- **A7c**: pre-LayerNorm on `parcel_emb` before `mixed + parcel_emb.unsqueeze(-1)`. Model-side fix; does not change the loader.
**Loader cost**: A7a/b are pure data transforms — can live behind a `support_normalization: Literal["raw", "row_sum", "l2"]` knob in the loader, default `"raw"`.

### N4. RoPE implementation — inline, temporal-axis only
**Phase**: C3.
**Decision**: inline RoPE in `backbone.py` (~30 lines). Do **not** use a generic library RoPE that rotates the entire sequence axis — that would rotate across the flattened `(electrode × time)` axis as if it were 1D temporal, mixing electrode identity into rotation phase.
**Implementation**: rotation depends on `t_a − t_b`, not on the electrode index. Precompute `cos/sin` tables of shape `(T, head_dim)` once; gather per-token by `t = token_idx % T` assuming flatten order `e * T + t`. Apply RoPE to `q` and `k` **before** the SDPA call (SDPA is RoPE-agnostic).
**Test**: pair of active-cell tokens at the same `t` but different `e` must receive identical rotation phase; pair at different `t` same `e` must receive `t_a − t_b`-dependent phase.

### N5. Backbone block uses SDPA (pairs with N1, N4)
**Phase**: C3.
**Decision**: each backbone block computes attention via `F.scaled_dot_product_attention(q_rot, k_rot, v, attn_mask=mask_bool, dropout_p=cfg.dropout if training else 0.0)` where `q_rot`/`k_rot` are RoPE-rotated. No custom attention math.

---

## Data layer decisions

### N6. CSV column-name sanitization in support cache
**Phase**: A1.
**Issue**: 3 of 15 Tier-1 parcel names contain `/` (`A1/2/3ulhf`, `A1/2/3tonIa`, `A9/46v`); literal use as CSV headers is round-trip fragile (some readers escape, others choke).
**Decision**: replace `/` with `_` in CSV header (`support_A1_2_3ulhf`); document the mapping in a `# header_map: A1/2/3ulhf -> A1_2_3ulhf, ...` row-0 comment so the loader's `kept_names` lookup is deterministic. Loader uses `DEFAULT_BASE_PARCELS` for canonical column order, sanitization at read time.
**Alternative considered**: parquet (preserves slashes natively). Rejected to avoid a new dependency for one cache file.

### N7. Per-patient `electrode_grid_shape` enumeration
**Phase**: A2.
**Decision**: verifier output `docs/qc/coord_bridge_verification.md` must include the bounding rectangle per patient. From the `#12` channel-map contract:

| Patient(s) | Channel map | `(H_p, W_p)` |
|---|---|---|
| S14, S16, S23, S26 | Map 4 (`*_channelMap.mat`) | `(8, 16)` |
| S33, S39, S62 | Map 3 (`*_channelMapAll.mat`) | `(16, 16)` |
| S58 (Phase 2) | Map 3 row-slice | `(12, 24)` |

Pin into a `GRID_SHAPES: dict[str, tuple[int, int]]` constant in `dataset.py`, asserted at loader construction against the verifier's per-patient report.

### N8. Test fixture file
**Phase**: B1.
**Decision**: build `tests/v14/fixtures/fake_fif.py` as a minimal `mne.EpochsArray` factory: 2 trials, tokens `bak` and `gup`, `N_e = 16`, `sfreq = 200 Hz`, `tmin = -0.5 s`, `tmax = 1.0 s`. Under 100 lines. Used by `test_dataset.py` and `test_collate.py` only.

---

## Deferred-only (does not block baseline)

### N9. BNA SC[15,15] / FC[15,15] source for ablation A4
**Phase**: F (A4 only).
**Status**: source 246×246 BNA SC and FC matrices are not in `data/atlas/`. Brainnetome distributes these in the BNA atlas pack; the lab may already have a copy.
**Decision needed before A4**: locate the source (ask Zac first), write `scripts/v14_core/build_sc_fc_tier1.py` to produce `data/atlas/sc_fc_tier1_15.npz` (`log1p` then z-score for SC, Fisher-z then z-score for FC).
**Does not block baseline.**

### N10. Design doc `.tex` rewrite
**Phase**: doc pass after Phase E lands.
**Status**: `docs/neural_field_perceiver_v14.tex` still describes the pre-amendment factored / within-parcel-summarizer design. The amendment doc + `[AMENDED]` markers in `implementation_tasks.md` are the de facto contract; CLAUDE.md and `current_direction.md` are now consistent with the amendment.
**Decision**: rewrite the `.tex` against the B-1 amendment **after** first DCC results land. The `.tex` is what gets shared externally, but rewriting before E1 risks a second rewrite if the baseline misbehaves and the architecture moves again.

---

## Closure log

When a note's decision lands, move it here with a one-line outcome and a commit pointer.

- **N2 (GridMixer depth co-equal baseline)** — closed 2026-04-16 late. Confirmed no spatial compression (stride=1, padding=1, shape-preserving). Promoted depth-1 and depth-2 to co-equal first-pass baselines in Phase E1 (sbatch array doubles to 120 jobs). A3 narrowed to depth-3 only, conditional on E1 saturating. Plan updated in `docs/plans/v14-core.md`.
