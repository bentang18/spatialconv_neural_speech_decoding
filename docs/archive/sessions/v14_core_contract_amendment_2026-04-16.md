# v14-core Contract Amendment — 2026-04-16 late

This amendment supersedes the prior Phase-1 contract for spatial token construction and the backbone attention pattern. It is the result of an end-of-day design discussion that walked back from `N_tok = 15` atlas-pool tokens to per-electrode tokens with soft parcel embedding, and from factored to combined attention.

The design doc (`docs/neural_field_perceiver_v14.tex`) and `implementation_tasks.md` are updated to match. CLAUDE.md will be updated next.

## What forced the rewrite

1. **Vertex collision audit (run 2026-04-16 late).** The strict fsaverage snap collides on average **23.6% of electrodes per patient** (302/1280 across 7 LH patients), with up to 4 electrodes per vertex on the densest 128-ch grids. The within-parcel summarizer cannot separate collided electrodes in the spatial path. The snap is provably aliasing the rigid uECoG grid.

2. **uECoG ≠ sEEG compression calculus.** B-3-style 15 atlas tokens compress per-active-parcel by 22-30× for our dense grid, vs ~5× for sEEG. Sub-parcel somatotopy (Bouchard 2013, Conant 2018) sits at exactly this scale and is what the dense device was meant to capture. The atlas-token thesis is right for sparse modalities; wrong for dense uECoG.

3. **Position-driven cross-patient transfer is structurally impossible.** Registration noise (~3-5 mm RMS) plus inter-individual anatomical variability (~10 mm RMS) gives ~11 mm joint uncertainty in localizing sub-parcel functional landmarks. The clusters we'd want to align are 5-10 mm wide. Signal-to-noise ratio for position-based sub-parcel alignment is ≤ 1. Any architecture that relies on aligning sub-parcel position across patients is fighting physics.

4. **Electrodes don't conform to sulci.** uECoG arrays sit on the exposed cortical surface; the fsaverage `lh.pial` reconstruction includes all sulcal walls but the device does not record from them. Two grid-adjacent electrodes that "look like opposite banks of a sulcus" in the snapped representation are actually both sitting on a flat patch of cortex. The grid is the truth; the snapped surface is an artifact.

These four findings together mean: **the rigid uECoG grid is the correct spatial substrate, the parcel is the correct cross-patient anchor, and sub-parcel positional alignment is infeasible.** The architecture must respect all three.

## Architecture (locked)

```
── Stage 1 ──  Per-electrode temporal tokenizer (#2/#6, unchanged)
── Stage 2 ──  Whole-grid Conv2d (M2, NEW)
── Stage 3 ──  B-1 with soft parcel embedding (NEW)
── Stage 4 ──  Combined spatiotemporal attention backbone (REVISED #27)
── Stage 5 ──  AR decoder (#28, unchanged)
```

### Stage 1 — Per-electrode temporal tokenizer (frozen #2/#6)

```
Input:  signal[B, N_e, 301]                     # 1.5 s @ 200 Hz
Conv1d: kernel = 30 samples (150 ms)
        stride = 10 samples (50 ms)
        in_channels = 1, out_channels = d_model = 64
        per-electrode (no spatial mixing)
Output: tokens[B, N_e, d=64, 28]                # 28 tokens at 20 Hz
```

Compression: 10.7× temporal. Matches HGA envelope timescale.

### Stage 2 — Whole-grid Conv2d (M2, NEW)

```
Input:  tokens reshaped to grid:
        x[B, d, H_p, W_p, T=28]
        with H_p × W_p ≥ N_e (padded to per-patient bounding rect:
        S14 = 8×16, S33 = 16×16, S58 = 12×24, etc.)
        plus a per-cell active mask[B, H_p, W_p]

Per-time-step block (shared weights across t):
  x_normed = LayerNorm(x, dim=channel)
  x_conv   = GELU(Conv2d(x_normed, kernel=3, stride=1, padding=1, channel_mixing=full))
  x_out    = x + x_conv * active_mask

Output: x[B, d, H_p, W_p, T] — same shape, locally enriched
        plus active_mask
```

**Justifications:**
- Kernel 3×3 with stride 1: receptive field ±1.33-3 mm, matches HGA correlation length
- Full conv (not depthwise-separable): 64×64×9 ≈ 37K params, trivial at our scale
- Per-time-step with shared weights: temporal mixing belongs to Stage 4
- Pre-norm LayerNorm + GELU + residual: standard transformer-adjacent block
- Active mask zeroes pad cells and artifact channels from the residual contribution
- Default depth = 1 layer; 2 layers is an ablation

### Stage 3 — B-1 with soft parcel embedding (NEW)

```
Inputs (per electrode e):
  conv_feature[e, t] in R^d                     # from Stage 2
  support[e, p] in [0, 100]                     # raw BNA prob, p in Tier-1 (15)

Soft parcel embedding (per electrode):
  parcel_emb[e] = Σ_{p in Tier1} support[e, p] · P_emb[p]
  
  where P_emb is a learnable lookup table of shape (15, d=64).
  support is RAW (not row-normalized); electrodes outside Tier-1 get
  weak embeddings naturally.

Token construction (per electrode, broadcast across time):
  token[e, t] = conv_feature[e, t] + parcel_emb[e]

Flatten to backbone input:
  tokens[B, N_e × T, d=64]
  + token_active_mask[B, N_e × T]               # active electrode AND not pad
```

**What this contains:**
- Sub-parcel info: implicit in `conv_feature` (Stage 2 mixed grid neighbors)
- Parcel identity: explicit via soft `parcel_emb` (cross-patient anchor)
- Boundary-aware: electrodes split across two parcels get convex combination of both embeddings

**What this deliberately does NOT contain:**
- Grid coordinate embedding (Conv2d already encoded local position; not patient-transferable)
- Intra-parcel coordinate (PE-2 / PE-3 / parcel frames) — registration noise dominates anatomical variability, so cross-patient sub-parcel alignment is infeasible
- fsaverage vertex coordinate — collision-prone and not transferable
- Hard parcel argmax — replaced by soft probability weighting

**Loader contract change:** `support[N_e, 15]` is the new per-electrode field. `token_mask[N_tok]` and `token_support[N_tok]` are dropped from the sample dict — they were per-parcel quantities and there are no per-parcel tokens anymore.

### Stage 4 — Combined attention backbone (revised #27)

```
Input: tokens[B, N_e × T, d=64], token_active_mask[B, N_e × T]

For B_blocks = 3 blocks:
  q = LayerNorm(x)
  attn = MultiHeadAttention(
    q, q, q,
    num_heads = 4,
    head_dim = 16,                              # 64/4
    rope_axis = "temporal",                     # RoPE between t_a and t_b only
    attn_mask = token_active_mask broadcast to pairs,
  )
  x = x + attn

  z = LayerNorm(x)
  ffn = Linear(d -> 4d) -> GELU -> Linear(4d -> d)
  x = x + ffn

Output: tokens[B, N_e × T, d=64]
```

**Per-block attention pattern is COMBINED:** every (electrode, time) token attends to every other (electrode, time) token in one shot. NOT factored into spatial-then-temporal.

**No SC/FC bias in baseline.** SC/FC bias is deferred to ablation — see "Deferred until v14-core is built" below.

**Justifications:**
- Combined > factored on uECoG: at our scale (256 ch × 28 t = 7,168 tokens worst case), combined attention is ~13B FLOPs per layer, ~80B per forward — well within budget. Compute is not binding.
- Combined captures (electrode_i, t_a) → (electrode_j, t_b) cross-spatial-temporal interactions in 1 layer, where factored requires 2-layer composition. Speech production has lag-coupled cross-region patterns (planning → execution → feedback), making combined the structurally honest choice.
- 3 blocks (down from 6 factored half-blocks): each combined block is more expressive, fewer needed.
- 4 heads × head_dim = 16: stays at d_model = 64 baseline.
- RoPE on temporal only: temporal locality prior; spatial relationships emerge from data plus parcel embedding plus Conv2d mixing.

### Stage 5 — AR decoder (frozen #28)

```
Input: backbone output tokens[B, N_e × T, d]

3 phoneme queries with shared base + per-slot embedding + previous-token embedding
1 causal self-attention over the 3 slot queries
1 cross-attention to (N_e × T) backbone memory
Shared linear vocab head (9 phonemes)
No auxiliary head
```

Unchanged. Cross-attention naturally handles variable `N_e × T` per patient.

## What's deprecated

| Item | Status | Reason |
|---|---|---|
| `parcel_frames.npz` (`#10`) | DEPRECATED | No intra-parcel positional coord in B-1 |
| `coordinates.py` v2 (cras-corrected port) | RETAINED as parity oracle only, NOT loaded by training | Coords no longer used in loader |
| `cvsavg_projection.py` | RETAINED as parity oracle only | Same reason |
| `coords[N_ch, 3]` in loader | DROPPED from sample dict | No spatial coord PE in tokens |
| `token_mask[N_tok]` in loader | DROPPED from sample dict | No per-parcel tokens; mask is per-electrode |
| `token_support[N_tok]` in loader | DROPPED from sample dict | Implicit in soft parcel embedding |
| Argmax + hard Tier-1 mask (`#3`) | RELAXED | Soft probability bias; no hard exclusion |
| Within-parcel Perceiver summarizer (`#26`) | DEPRECATED | Replaced by Stage 3 soft embedding + Stage 4 combined attention |
| Factored spatial-then-temporal attention (`#27` original) | REPLACED by combined | Combined more expressive at our scale |
| SC/FC bias as baseline (`#8`) | DEFERRED to ablation | Build v14-core first; add as ablation |
| `parcel-frame chart` section in design doc | REWRITTEN to reflect deprecation | Same |

## What's preserved unchanged

| Item | Status |
|---|---|
| `#1` ACPC → fsaverage pipeline (snap-to-pial) | Unchanged; reduced role to parcel argmax + probability lookup |
| `#2` Per-electrode Conv1d temporal tokenizer | Unchanged |
| `#4` Tier-1 = 15 LH parcels | Unchanged in identity; role changes from partition labels to embedding-lookup keys |
| `#5` `support(e, p) = fsaverage_bake[v_e, p]` | Unchanged in computation; now feeds soft embedding instead of parcel-pool |
| `#6` Temporal output contract `(B, N_e, d, 28)` | Unchanged |
| `#9` Supervised training contract (3-slot CE, exhaustive eval) | Unchanged |
| `#11` Channel inclusion (all non-artifact) | Unchanged |
| `#12` Amp → physical electrode → coordinate bridge | Unchanged |
| `#13` Loader contract | UPDATED — see new spec below |
| `#15` Width budget `d_model = 64` baseline, 128 ablation | Unchanged |
| `#16-#25` Label/phoneme contracts | Unchanged |
| `#28` AR decoder (3 queries, cross-attention) | Unchanged |
| `#29` Epoching (`tmin=-0.5s`, `tmax=1.0s`, 200 Hz) | Unchanged |
| `#30` LH-only Phase 1 | Unchanged |
| `#31-#33` Batching, normalization, metric | Unchanged |
| `#36` fsaverage spatial base (closed) | Unchanged; reduced role to parcel argmax/prob lookup only |

## Updated loader contract (replaces `#13`)

The v14-core loader emits per sample:

```
signal[N_e, 301]                   # float32, productionZscore HGA, 200 Hz, 1.5 s
patient_id                         # str
label[3]                           # int, 0-indexed alphabetical ARPABET
electrode_grid_layout[N_e, 2]      # int (row, col) on patient grid
electrode_grid_shape               # tuple (H_p, W_p) per-patient bounding rect
electrode_active_mask[N_e]         # bool: True iff non-artifact
support[N_e, 15]                   # float32, raw BNA probability over Tier-1
```

Notes:
- `signal` is z-scored HGA from `productionZscore_highgamma.fif` (per `#32`).
- `electrode_grid_layout` is the (row, col) of each electrode on the patient's bounding rectangle. Padded cells fill out the rectangle and have `active_mask = False`.
- `electrode_grid_shape` is the bounding-rect dimensions for Stage 2 reshaping.
- `electrode_active_mask` covers both artifact channels (per `#11`) and any pad cells.
- `support` comes from `data/atlas/fsaverage_bake_fast2/` evaluated at each electrode's assigned fsaverage pial vertex (per `#5`). Tier-1 columns only (15 wide).
- `label` is the 3-phoneme target (per `#16`).
- No `coords`, no `token_mask`, no `token_support` — those fields are removed.

## Deferred until v14-core is built

These are not blockers. They become ablations once the v14-core baseline is running.

1. **SC/FC additive logit bias** (revised `#8`):
   - Per-head per-layer learnable scalars (`α_SC`, `α_FC`)
   - Soft parcel-pair bias from raw `support` and z-scored 15×15 SC/FC matrices
   - Broadcasts across temporal pairs
   - Will be added as Phase-1 ablation A4
2. **`d_model = 128` width** (per `#15`): ablation A1
3. **`B = 2` and `B = 4` block counts** (vs `B = 3` baseline): ablation A2
4. **`num_conv_layers = 2`** (vs 1 baseline): ablation A3
5. **Mean+gradient pool baseline** (linear comparator): ablation A5
6. **Combined vs factored attention head-to-head**: ablation A6 (factored as historical reference)
7. **Patient-grid bbox intra-parcel PE** (PE-2): ablation A7 (test whether explicit position helps despite the alignment-noise argument)

## Hyperparameter table (locked baseline)

```
Stage 1 — Per-electrode temporal Conv1d:
  in_channels         = 1
  out_channels (= d)  = 64
  kernel_size         = 30 samples (150 ms)
  stride              = 10 samples (50 ms)
  bias                = True

Stage 2 — Grid Conv2d (per time-step):
  in_channels (= d)   = 64
  out_channels (= d)  = 64
  kernel_size         = 3
  stride              = 1
  padding             = 1
  bias                = True
  channel_mixing      = full
  num_layers          = 1
  activation          = GELU
  norm                = LayerNorm (channel dim, pre-norm)
  residual            = pre-norm + post-conv add (masked)
  init                = Kaiming on conv weights, zero bias

Stage 3 — Soft parcel embedding:
  P_emb shape         = (15, 64)
  init                = Xavier uniform
  support source      = baked fsaverage atlas, raw (not normalized)
  Tier-1              = src/speech_decoding/v14/token_spec.DEFAULT_BASE_PARCELS

Stage 4 — Backbone (combined attention):
  num_blocks          = 3
  d_model             = 64
  num_heads           = 4
  head_dim            = 16
  ffn_dim             = 256 (= 4 × d)
  ffn_activation      = GELU
  norm                = pre-norm LayerNorm
  attention           = combined spatiotemporal
  rope_axis           = temporal only
  dropout             = 0.1
  attn_mask           = token_active_mask broadcast to pairs

Stage 5 — AR decoder (#28, unchanged):
  num_slots           = 3
  base_query          = learned, shared
  slot_emb            = learned, per slot
  prev_token_emb      = learned, vocab-sized
  causal_self_attn    = 1 layer
  cross_attn          = 1 layer over (N_e × T, d)
  vocab_head          = shared linear, 9 phonemes
```

## Why this is the right place to land

- **Physics-matched compression.** Encodes structure at scales where transfer is feasible (parcel cm-scale, signal patterns) and accepts that sub-parcel positional alignment is below the noise floor (~11 mm).
- **Maximally expressive within those constraints.** Per-electrode tokens, combined attention, no premature compression below electrode resolution.
- **Anatomically informed without registration dependence.** Soft parcel embedding (BNA probabilities) is the only anatomical input; everything else is rigid grid + signal.
- **Elegant.** Single token stream, single embedding mechanism, no positional embedding to argue about. No `parcel_frames.npz` to maintain.
- **BarISTA-validated.** The parcel-embedding mechanism + zero per-patient parameters are the published parts of this design (Oganesian 2025).
- **uECoG-appropriate.** Preserves dense per-electrode resolution where the device's value lives. Does not impose sEEG-style compression.

## See also

- `docs/neural_field_perceiver_v14.tex` — design doc (will be updated next pass)
- `docs/implementation_tasks.md` — blocker updates (this commit)
- `docs/v14_core_implementation_plan.md` — implementation order with verification checks (this commit)
- `pastwork/summaries/oganesian2025_barista.md` — the closest published architecture
- Vertex collision audit: in-conversation 2026-04-16 late, 23.6% loss across 7 LH patients
