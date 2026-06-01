# Handoff — B35: revert B34, Phase-4 readout = frozen-P3-PMA → mean-over-time → linear

**Date**: 2026-05-31
**From**: architecture/docs (Ben + Claude)
**To**: implementation engineer
**Scope**: code-side rewire only. The architecture/docs half (B35 memo, MEMORY.md, training_recipe.md, v14_blockers.md) is already landed — do **not** re-touch those.

Canonical decision memo: `memory/project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31.md`.
**Supersedes** `reports/v14_b34_p4_attentive_pooler_handoff_2026_05_30.md` — that handoff's design (drop PMA from P4, default to a per-task attentive query) is reverted. Its code landed in the working tree (uncommitted) and never ran; B35 rewires it before first dispatch.

---

## What the change is

**Current code (B34, uncommitted):** PMA removed from Phase 4; P4 readout is `V14PerTaskAttentivePooler` (default, 792,074 trainable params) attending over the full `(B, L, T_p, d)` parcel×time field, with `V14MeanPoolLinearHead` as the `readout="meanpool"` baseline.

**Target (B35):** PMA comes **back at P4, frozen** (loaded from the P3 checkpoint). The default readout is **frozen-PMA → mean-over-time → Linear**:

```
latents (B, L, T_p, d)
  → frozen PMA collapses the parcel/slot axis (L=80 DK slots, M=1 per B29) per time-patch,
    key-masked by latent_valid (B30)                         → (B, T_p, d)
  → mean over T_p                                            → (B, d)
  → per-task Linear(d, n_classes)   [the ONLY trainable P4 module]   → (B, n_classes)
```

This is the B31-era shape with the time-readout pinned to **mean** (B31's rationale text said "PMA → mean → linear"; its *code* shipped a flat-head — B35 resolves that drift in favor of mean). The attentive query and the flat-head both survive as deferred sisters.

**Why revert.** The encoder is frozen at P4, so the readout is the only thing that trains, on tiny per-task splits (`NEUROPROBE_LITE_MAX_SAMPLES=3500`, balanced; CrossSubject trains on S2/trial-4 only ≤3500; Within-Session 2-fold ~1750 train). A 792K-param pooling operator from random init over ≤3500 samples (~152–453× params-over-data) overfits or fits subject identity — the exact CSubject leakage the gate punishes. A frozen aggregator + tiny linear is the only readout that actually measures what the P3/SSL encoder learned. Full reasoning + the PopT-precedent and bucket-A leaderboard grounding are in the B35 memo.

---

## Param accounting (measured 2026-05-31, d=256, n_classes=10, T_p=4)

| readout | trainable | total | note |
|---|---|---|---|
| **B35 default: frozen PMA + mean-over-time + linear** | **514** (binary) / **2,570** (10-way) | ~12.12M | PMA 263,424 frozen; mean = 0 params; linear is the only trainable module |
| R-p4-flatten (was B31 code): frozen PMA + flatten + linear | 10,250 (10-way) | ~12.13M | `V14Phase4FlatHead`, 4·256·10+10 |
| R-p4-attentive (= retired B34 default) | 792,074 | 792,074 | `V14PerTaskAttentivePooler` |
| R-p4-meanpool-no-pma (= B34 meanpool baseline) | 2,570 | 2,570 | `V14MeanPoolLinearHead`, mean over parcel×time, NO PMA |
| Full B35 P4 model (F=38/T=17/K=80/d=256/nc=10) | 514–2,570 | **~12.12M** | encoder ~11.86M + frozen PMA 0.263M + linear; 12,119,296 frozen; under 30M cap |

**CORRECTION carried in — the B34 docs/tests misstate the PMA count.** The committed B34 memo + handoff (this file's predecessor, line 85) and the recipe's old "≈0.8M" say **PMA = 789,760**. The real `V14ParcelCollapsePMA(256, 8)` = **263,424** (query 256 + ln_q 512 + ln_kv 512 + q_proj 65,536 + kv_proj 131,072 + out_proj 65,536; **no rFF MLP**). 789,760 was a mis-transcription of the attentive pooler's 792,074. Fix every assertion/comment that carries 789,760 (or 800,010, which was 789,760 + 10,250) → use 263,424. The superseded B34 handoff/memo are left as-is (marked superseded, not re-touched).

---

## Code re-scope

All targets are in `src/speech_decoding/models/v14_encoder.py` unless noted. Line numbers are from the current (B34) working tree; confirm before editing.

### 1. `V14ParcelCollapsePMA` — return to P4, frozen
- Docstring (around `v14_encoder.py:1275`, esp. the "**P4 (Neuroprobe probe)** — PMA is **NOT used**" block at ~1290–1294): revert to **"P3-trained, P4-frozen"** — used both at P3 (Whisper distillation per B33) and at P4 (frozen, parcel-collapse before the probe). The forward already produces `(B, T_p, d)` from `(B, L, T_p, d)` keyed by `latent_valid` — that is the same contract P3 uses, so reuse it; don't add a second collapse path.

### 2. New default readout head
Add a small head module that holds the **frozen** PMA and applies the temporal op + linear. Suggested:

```python
class V14PmaReadout(nn.Module):
    # frozen P3-PMA → {mean|flatten|timeattn} over T_p → Linear(d, n_classes)
    def __init__(self, pma: V14ParcelCollapsePMA, temporal: Literal["mean","flatten","timeattn"],
                 d_model, n_classes, n_time_bins): ...
    def forward(self, latents, *, latent_valid):
        collapsed = self.pma(latents, latent_valid=latent_valid)   # (B, T_p, d), PMA params frozen
        pooled = self._temporal(collapsed)                         # mean→(B,d); flatten→(B,T_p·d); timeattn→(B,d)
        return self.linear(pooled)
```
- Freeze the PMA inside this head (`requires_grad_(False)`); the linear (and, for `timeattn`, the ~257-param query) are the only trainable params at P4.
- The PMA must be **loaded from the P3 checkpoint**, not re-initialised — same load path B31 used at P4. Verify the checkpoint key wiring.

### 3. `V14ParcelPerceiver` config — readout selector
- `v14_encoder.py:1655` currently: `readout: tp.Literal["attentive", "meanpool"] = "attentive"`.
- Replace with: `readout: tp.Literal["pma_mean_linear", "pma_flatten_linear", "pma_timeattn_linear", "attentive", "meanpool"] = "pma_mean_linear"`.
- `build()` if/elif (`v14_encoder.py:1727–1739`): construct
  - `pma_mean_linear` (default) → `V14PmaReadout(pma, "mean", ...)`
  - `pma_flatten_linear` → `V14PmaReadout(pma, "flatten", ...)` (this restores the B31 flat-head shape; you may reuse `V14Phase4FlatHead` on the collapsed `(B,T_p,d)` instead)
  - `pma_timeattn_linear` → `V14PmaReadout(pma, "timeattn", ...)`
  - `attentive` → `V14PerTaskAttentivePooler(...)` (kept, no longer default — the `R-p4-attentive` sister)
  - `meanpool` → `V14MeanPoolLinearHead(...)` (kept, no longer baseline-by-default — the `R-p4-meanpool-no-pma` sister)
- The first three need the PMA built + frozen; `attentive`/`meanpool` take the full field directly (unchanged).
- `V14ParcelPerceiverWithHead.forward` (`v14_encoder.py:1587`) already calls `self.readout(latents, latent_valid=...)` — leave it; the new head matches that signature.

### 4. Keep the existing heads as sisters (do not delete)
- `V14PerTaskAttentivePooler` (`v14_encoder.py:1379`) → docstring: "no longer the P4 default; `R-p4-attentive` sister (revisit only with a larger per-task train budget)."
- `V14MeanPoolLinearHead` (`v14_encoder.py:1481`) → docstring: "`R-p4-meanpool-no-pma` sister — means over parcel×time, **skips the PMA**; distinct from the default `pma_mean_linear` which means over time *after* the PMA collapses parcels. Leakage/ablation reference."
- `V14Phase4FlatHead` (`v14_encoder.py:1356`) → stays a primitive; used by `pma_flatten_linear` / `R-p4-flatten`.

### 5. `dispatch_v14.py`
- `readout: str = "attentive"` (`dispatch_v14.py:353`) → `"pma_mean_linear"`.
- CLI `--readout` choices (`dispatch_v14.py:885–889`): `["attentive","meanpool"]` → `["pma_mean_linear","pma_flatten_linear","pma_timeattn_linear","attentive","meanpool"]`, default `pma_mean_linear`; update the help text.
- Comments at 346–349, 679–682, 881–889 describe the B34 "PMA gone, attentive default" decision — rewrite to B35.

### 6. `models/__init__.py`
- Export `V14PmaReadout`. Keep `V14PerTaskAttentivePooler` + `V14MeanPoolLinearHead` exported (they're now sisters, still selectable).

### 7. Comment relabel B34 → B35
All `B34 (2026-05-30)` P4-readout comments in `v14_encoder.py`, `dispatch_v14.py`, `models/__init__.py`, and the touched tests describe the now-reverted decision — relabel to **B35 (2026-05-31)** and point wikilinks at `project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31`. (The encoder module docstring at `v14_encoder.py:47–52` describes the B34 attentive readout — rewrite to the B35 frozen-PMA→mean→linear shape.) Verify every B34 hit is the P4-readout change before replacing; B33 (P3 project-up) is unrelated and must not be touched.

---

## Tests

- `models/test_v14_encoder.py`: the build tests now assert `readout="attentive"` default — flip to `readout="pma_mean_linear"` default; add a build test per readout option. Param-budget test still asserts `<30M` (passes at ~12.12M). Add/repair an assertion on the frozen-PMA count (**263,424**, not 789,760) and the trainable-linear count (514 binary / 2,570 10-way).
- `models/test_v14_readout_mask.py`: the masked-invariance / unmasked-attends-everywhere tests were rewritten for the attentive pooler. Re-point the default-readout case at `pma_mean_linear`; keep the attentive and meanpool cases as sister coverage. The `latent_valid` mask now flows through the **PMA** softmax for the default — assert invalid latents don't affect the collapsed `(B,T_p,d)`.
- `experiments/test_v14_dispatch_wired.py`: `readout=` assertion default `"attentive"` → `"pma_mean_linear"`; keep a meanpool/attentive smoke for the sisters.
- Grep for the bad param literal before running: `grep -rn "789760\|789,760\|800010\|800,010" src/ docs/` — fix any in `src/`.

## Loose end — the one B34 failing test
`models/test_v14_encoder.py::test_v14_head_wrapper_forwards_b29_conditioning_to_encoder` was failing at B34 handoff (logit delta 1.87e-5 < 1e-4 threshold) because the **attentive readout's extra LayerNorms + softmax attenuated** the anatomy-gate-induced latent shift. B35's frozen-PMA → mean → linear is closer to the old flat-head (no post-LN/softmax pile-up over the full field), so **re-run this test after the swap before changing it** — the delta may recover above 1e-4 on its own. If it still fails, apply the B34-recommended fix: make the test's `graded_support` peakier (sharper anatomy contrast, ~2 lines at `test_v14_encoder.py:664–667`). Do **not** lower the `1e-4` threshold (it guards an unrelated `subject_subtype`/`ref_idx` regression in the same test).

---

## Engineer checklist
1. [ ] Revert `V14ParcelCollapsePMA` docstring to "P3-trained, P4-frozen".
2. [ ] Add `V14PmaReadout` (frozen PMA → mean/flatten/timeattn → Linear); load PMA from P3 ckpt; freeze it.
3. [ ] Swap the `readout` config to the 5-option enum, default `pma_mean_linear`; rewire `build()`.
4. [ ] Update `dispatch_v14.py` default + CLI choices + comments.
5. [ ] Export `V14PmaReadout`; keep attentive/meanpool exported as sisters.
6. [ ] Relabel B34 → B35 comments/wikilinks in the touched files (verify each is the P4 change, not B33).
7. [ ] Fix param assertions (frozen PMA 263,424; linear 514/2,570; total ~12.12M); grep-kill 789,760.
8. [ ] Re-run the conditioning test; apply peakier-`graded_support` fix only if it still fails.
9. [ ] Clear `__pycache__` and run the full suite: `.venv/bin/python -m pytest -q src/` — expect 0 fail.
10. [ ] Commit on a branch (NOT main, NOT the DCC clone). Do not push unless Ben asks.

## Guardrails (project policy)
- All training on DCC, never local; never commit on the DCC clone.
- Do not push to remote unless explicitly asked.
- Never skip hooks (`--no-verify`) or bypass signing.
- Surgical changes only — don't refactor adjacent code. The encoder, SSL loss surface, and P3 distillation (B33) are untouched by B35; only the P4 readout changes.
