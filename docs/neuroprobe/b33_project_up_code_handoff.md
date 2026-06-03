# B33 Project-Up Phase-3 — Engineer Memo / Implementation Handoff

**Date:** 2026-05-30 · **Status:** doc-locked, code unwired · **Owner of decision:** Ben
**Locks:** [[project_v14_b33_project_up_phase3_2026_05_30]] (B33), [[project_v14_whisper_teacher_all_layer_mean_2026_05_30]] (teacher = all-layer mean, already landed), [[project_v14_b31_vjepa2_canonical_loss_2026_05_28]] (B31, PMA trained at P3 only)
**Spec:** `docs/neuroprobe/training_recipe.md` §5 (Phase 3) + §6 (Phase 4 banner)

This memo is the code pass for flipping Phase-3 cross-modal distillation from **project-DOWN** to **project-UP**, plus the **MLP student head** and the **mandatory per-channel target z-score**. It implements *modules*, not the training loop. The P3 dispatch wiring (warmup→unfreeze schedule, LR groups, early-stop) and the P4 attentive probe are explicitly **out of scope** (see §7).

---

## 1. What changes in one paragraph

Today the teacher squeezes 1280→256 with a trainable `WhisperAdapter` and the student is an identity passthrough; the loss is in 256-d. B33 flips this: the teacher target is the **fixed full 1280-d** Whisper all-layer mean (no trainable teacher module), the **student** projects **up** 256→1280 with a new 2-layer MLP head, and the loss is in **1280-d**. Because removing the teacher adapter removes the only thing that was playing a `StandardScaler` role, a **per-channel, train-only z-score of the 1280-d target becomes mandatory**.

```
                          PROJECT-DOWN (current code)              PROJECT-UP (B33 target)
teacher  Whisper mean_all (B,250,1280)                    Whisper mean_all (B,250,1280)
         → tri-pool 50→8Hz → (B,40,1280)                  → tri-pool 50→8Hz → (B,40,1280)
         → WhisperAdapter MLP 1280→256 → (B,40,256)       → per-channel z-score → (B,40,1280)   [FIXED, detached]
student  encoder → PMA k=1 → (B,40,256)  [identity]       encoder → PMA k=1 → (B,40,256)
                                                          → StudentWhisperProjector 256→1280 → (B,40,1280)
loss     Smooth-L1 in 256-d                               Smooth-L1 in 1280-d
```

The teacher-side **triangular pool already outputs 1280-d** (`extractors/whisper_teacher_pool.py`), so it is unchanged. The only teacher-side addition is the standardizer.

---

## 2. Current code state (as landed — the "from")

| File | Symbol | Current behavior | Action |
|---|---|---|---|
| `models/whisper_adapter.py` | `WhisperAdapter(1280,256,256)` | teacher-side MLP, **project-down default** | **demote** → `R-project-down` sister (keep class, fix docstring) |
| `models/whisper_adapter.py` | `WhisperLayerMerge` | layer-merge sister (mean/weighted) | **keep** (direction-agnostic) |
| `extractors/whisper_teacher_pool.py` | `triangular_pool_50_to_8_hz`, `triangular_pool_weight_matrix` | teacher pool `(B,250,1280)→(B,40,1280)` | **keep** (already 1280-d; direction-agnostic) |
| `bt_alignment/teacher_cache.py` | `WhisperFeatureExtractor`, `write_clip_cache` | extract + cache raw `(T,1280)` fp16, `mean_all` | **keep extract/cache**; **add** per-channel standardizer (fit + apply) |
| `ssl/distill.py` | `phase3_distillation_loss`, `PhaseThreeDistillationConfig` | shape-agnostic Smooth-L1/MSE/cosine; detaches teacher; `target_instance_norm` per-token hook (default False) | **docstrings only** (256→1280); clarify the two normalization axes |
| `bt_alignment/contract.py` | `WHISPER_CONTRACT` | 1280-d, `mean_all`, 50 Hz | **add** project-up fields; fix stale "adapter = StandardScaler" docstring |

No standardizer/z-score exists anywhere in `src/` today — it is entirely new.

---

## 3. New module — `StudentWhisperProjector` (the "upsample adapter")

Add to `models/whisper_adapter.py`. This is the connector head trained at P3, **discarded at P4**.

```python
class StudentWhisperProjector(nn.Module):
    """B33 project-up student head: (B, T, 256) -> (B, T, 1280).

    Default mode="mlp" (LLaVA-1.5 shape, ~1.97M); mode="linear" is the
    R-head-linear falsifier (~0.33M). Discarded at P4 (P4 re-learns its own
    attentive probe over the encoder grid).
    """
    def __init__(self, d_in: int = 256, d_out: int = 1280, mode: str = "mlp") -> None:
        super().__init__()
        if mode == "mlp":
            self.net = nn.Sequential(
                nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out)
            )
        elif mode == "linear":
            self.net = nn.Linear(d_in, d_out)
        else:
            raise ValueError(f"mode must be 'mlp' or 'linear', got {mode!r}")
        self.mode, self.d_in, self.d_out = mode, d_in, d_out

    def forward(self, x: Tensor) -> Tensor:   # (B, T, d_in) -> (B, T, d_out)
        return self.net(x)
```

- **Default = `mode="mlp"`** (Ben 2026-05-30; rationale = connector base rate + brain→audio nonlinearity + warmup de-risk + the P4-change retired the pro-linear argument).
- Param counts to pin in tests: **mlp = 1,968,640** (`256·1280+1280 + 1280·1280+1280`); **linear = 328,960**.
- Attaches **after** PMA k=1 (`V14ParcelCollapsePMA`, already in the encoder; B31 set its `freeze` default False). Connector = `{PMA, StudentWhisperProjector}`.

---

## 4. New — per-channel target z-score (the rigorous part)

Mandatory default. A **fixed, train-only, per-channel** affine over the 1280-d axis, applied to the teacher target before the loss.

### 4.1 Algorithm
For each channel `c ∈ [0,1280)`: `z[t,c] = (x[t,c] − μ[c]) / σ[c]`, with `(μ,σ)` computed **once** over the P3 **training-pool** clips, pooled over **(clips × timesteps)**, then frozen.

### 4.2 The four rigor must-dos (do not skip any)
1. **Train-only.** Fit `(μ,σ)` on the P3 training split only; freeze for val/test. No val/test clip enters the stats. (Same discipline as the ceiling probe's `StandardScaler.fit(train)`.)
2. **Pool over (clips × timesteps), one scalar per channel.** Never per-timestep (`μ[t,c]` would erase onset/temporal structure the student must predict).
3. **Zero-variance guard.** `inv_std[c] = 1/√(σ²[c] + ε)`; where `σ[c]≈0`, set `inv_std[c]=1` (channel passes through unscaled — mirrors sklearn `_handle_zeros_in_scale`).
4. **fp32 accumulation.** Two-pass or Welford in fp32 (cache is fp16; single-pass fp16 variance over large N is unstable).

### 4.3 Why it is load-bearing (not cosmetic)
Smooth-L1's `β=1.0` is **not scale-invariant**. Without unit per-channel variance, the quadratic↔linear threshold means something different per channel — high-norm late-layer channels sit permanently in the robust/linear regime, low-norm channels in the quadratic regime. Z-scoring makes `β=1.0` a uniform threshold across all 1280 channels; the net objective becomes reduced-rank linear regression onto the top PCs of the standardized target. (data2vec normalizes targets for the same reason.)

### 4.4 Suggested interface (in `teacher_cache.py`)
```python
def fit_channel_stats(feature_paths: list[Path], d_model: int = 1280,
                      eps: float = 1e-8) -> dict[str, Tensor]:
    """Stream the train-pool clip caches, accumulate fp32 sum/sumsq over
    (clips × timesteps), return {'mean': (d,), 'inv_std': (d,)} with the
    zero-variance guard applied. Save alongside the cache as channel_stats.pt."""

class TargetStandardizer(nn.Module):
    """Frozen per-channel affine. Buffers mean, inv_std (non-trainable).
    forward: (B, T, d) -> (x - mean) * inv_std."""
```
The cache itself **stays raw 1280-d** (regenerate-cheap); standardization is applied at load/train time from the frozen `channel_stats.pt`. Falsifier `R-no-target-standardize` = skip the standardizer (raw target).

---

## 5. `ssl/distill.py` — docstrings only

The loss is already shape-agnostic (`phase3_distillation_loss` only asserts `student.shape == teacher.shape`). Under project-up both are `(B,40,1280)`. **No math change.** Required:
- Update the module + function docstrings: student `(B,40,256)→head→(B,40,1280)`, teacher `(B,40,1280)` per-channel-z'd; loss in 1280-d.
- Clarify the **two distinct normalization axes** so they are not conflated:
  - **Mandatory (new):** per-**channel** corpus z-score, applied **upstream** (in the data pipeline via `TargetStandardizer`), train-only.
  - **Opt-in `target_instance_norm` (existing, default False):** per-**token** instance-norm across the 1280 channels (M05 candidate). Different axis; stays opt-in.
- Note the `detach()` is now trivially correct (target is a fixed cached + standardized tensor, no teacher-side trainable module to starve).

---

## 6. `bt_alignment/contract.py` — `WHISPER_CONTRACT`

Add fields and fix the stale standardization sentence in the module docstring (it currently says "the v14 analog of StandardScaler is the trainable adapter" — false under project-up).

```python
WHISPER_CONTRACT = {
    ...                                    # existing: variant, n_mels, d_model=1280, etc.
    "project_direction": "up",             # B33: student 256→1280, no teacher-side adapter
    "target_standardization": "per_channel_zscore_train_only",  # mandatory
    "teacher_down_adapter": False,         # R-project-down sister only
    "student_head": "mlp_256_1280",        # R-head-linear sister = "linear_256_1280"
}
```

---

## 7. Out of scope (explicitly NOT in this pass)

- **P3 dispatch / training-loop wiring** (`dispatch_v14.py` + the NeuralTrain Experiment): the warmup→unfreeze schedule (3a freeze-encoder train PMA+head ~3–5k; 3b unfreeze A@LR/10, B@/3, PMA+head@full ~18k), the LR param-groups, EMA=none, early-stop on val Smooth-L1. P3 is still unwired into dispatch — this memo lands the modules so that wiring has something to call.
- **P4 attentive probe** ([[project_v14_b33_project_up_phase3_2026_05_30]] §3 / `training_recipe.md` §6 banner): PMA is dropped at P4 and replaced by a V-JEPA-style learned-query attentive probe over the encoder grid. **Open form pending Ben** (query count; single vs deeper cross-attn; frozen-encoder A1 vs unfrozen A2). Do **not** implement until specced.

---

## 8. Sisters to expose as flags

| Sister | Flag | Mechanism |
|---|---|---|
| `R-project-down` (prior default) | restore teacher `WhisperAdapter` + 256-d loss | `WhisperAdapter` already exists; wire as a teacher-side mode |
| `R-head-linear` | `StudentWhisperProjector(mode="linear")` | done by the new module |
| `R-no-target-standardize` | skip `TargetStandardizer` | bypass the standardizer |
| `R-frozen-throughout` (BLIP-2 A1) | freeze encoder all of P3, train PMA+head only | training-loop flag (§7 wiring) |
| `R-head-mlp`→ now default | — | (default) |

Decision gate: `R-head-linear` vs MLP default is judged on **P4 downstream transfer, never on P3 distillation loss**. Both must-run.

---

## 9. Tests

- **`models/test_whisper_adapter.py`** — keep `WhisperAdapter` tests (now the R-project-down sister). **Add** `StudentWhisperProjector`: mlp default shape `(2,40,256)→(2,40,1280)`; param count `== 1_968_640`; linear mode `== 328_960`; mlp nonlinearity (GeLU active); gradient flow to both Linears.
- **`bt_alignment/test_teacher_cache.py`** — **add** standardizer tests: `fit_channel_stats` returns `(1280,)` mean/inv_std; train-only (val clips never touched); zero-variance channel → `inv_std==1`; fp32 accumulation; `TargetStandardizer` output has ~unit per-channel variance + ~zero mean on the train pool; preserves `(B,T,1280)` shape.
- **`ssl/test_distill.py`** — existing tests pass (shape-agnostic). **Add** one `(B,40,1280)` shape test to pin the project-up contract.
- **`bt_alignment/test_contract.py`** — assert the new `WHISPER_CONTRACT` fields (`project_direction=="up"`, `target_standardization`, `teacher_down_adapter is False`, `student_head=="mlp_256_1280"`).

**Verify:** `.venv/bin/python -m pytest -q src/speech_decoding/models/test_whisper_adapter.py src/speech_decoding/bt_alignment/test_teacher_cache.py src/speech_decoding/ssl/test_distill.py src/speech_decoding/bt_alignment/test_contract.py` then the full suite `.venv/bin/python -m pytest -q`. Check LSP diagnostics on the four edited files.

---

## 10. Drift-table bookkeeping

Update `docs/neuroprobe/v14_blockers.md`: the B33 rows (project-up target, student head, per-channel z-score) move from "unimplemented in `src/`" toward "modules landed, dispatch wiring pending." The `R-project-down`/`R-head-linear`/`R-no-target-standardize` sisters become flag-gated. P4 attentive-probe stays an open row (pending Ben).
