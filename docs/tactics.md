# Tactics — Concrete Task List

Tactics layer of the triad (objectives → strategy → tactics). Operational: what's running, what to do when it lands, what's blocked. Updated 2026-04-19 late.

- Objectives: `objectives.md`
- Current-stage strategy: `strategy/stage_1.md`
- DCC tooling: `references/dcc_setup.md`

**Current stage:** Stage 1 (Phase 1). Architectural ablation wave in flight. When it drains, Stage 1 pauses; pivot to Stage 2 prerequisite work while waiting on data unblock.

---

## In flight (DCC queue)

Poll with `scripts/ablation/status.py`. Tail logs with `scripts/ablation/logs.py <job_id>`. Peek mid-flight JSONs with `scripts/ablation/peek.py <job_id>`.

| Job | Arm | Decides |
|---|---|---|
| 45769580 | T3.4 LOPO: `per_cell + pe2d + hier + partialconv` | Whether hierarchical joins Stage-1 default |
| 45769655 | T3.3_frozen LOPO: `per_cell + pe2d_frozen + partialconv` | Whether pe2d's 192 learned params are load-bearing under transfer |
| 45768642 | Default LOPO: `per_cell + partialconv + pe2d` | Fallback default if T3.4 disappoints |
| 45769582 | T3.1 pooled: `hierarchical_atlas + partialconv + pe2d` | Anatomy-indexed query vs cell-indexed free query |
| 45768472 | T1.2 amp_only pooled | Whether single-op amp augmentation passes pooled gate |
| 45768473 | T1.2 dropout_only pooled | Whether single-op dropout augmentation passes pooled gate |
| 45768474 | T1.2 noise_only pooled | Whether single-op noise augmentation passes pooled gate |
| 45768489 | T2.2 per_electrode d=32 depth=4 pooled | Whether depth helps per_electrode path |

---

## When jobs land (post-wave actions)

In order, per landing batch:

1. **Aggregate** — `scripts/ablation/collect.py <job_ids>` pulls `*.result.json` from DCC and runs `update_ablation_log.py`. Result rows land in `experiments/v14_ablation_log.csv`.
2. **Update Stage-1 scoreboard** — fill the matching row in `strategy/stage_1.md` §Current scoreboard with pooled / LOPO numbers + 1-line verdict. Remove the job row from the "In flight" table above.
3. **Fire follow-on LOPOs** — any T1.2 variant that cleared the pooled gate (< 0.800) gets a LOPO: `sbatch scripts/v14_core/v14_lopo_<variant>_dcc.sh` after `rsync_repo`; record in `.ablation_submissions.jsonl`; add row to "In flight" above.
4. **Update Stage-1 default** — if composed LOPOs confirm hierarchical + pe2d_frozen, rewrite §Default architecture in `strategy/stage_1.md` with the confirmed version and note the verdict.
5. **Pause architectural ablation** — per §Discipline in `strategy/stage_1.md`. Further Stage-1 arch ablations below the reliability horizon are not actionable.

---

## Stage-2 prerequisites (unblocked; work in parallel)

### Blockers (ask Zac)

- [ ] **13 missing lexical FreeSurfer recons.** Projectable today = 3/16 (S76, S78, S81). Blocks lexical-supervised Stage 2. Press Zac — may exist on Box under a folder not on my mount, or on DCC group storage.
- [ ] **Lexical cohort quality assessment.** Stage-2 supervised expansion is gated on Zac's pass/fail per lexical patient (up to 16). Needs per-subject HG response, signal quality, alignment sanity checks.
- [ ] **Cogan internal speech sEEG patient list.** Stage-3 data source (~33 h). Need patient IDs, access path, LH speech motor subset scope.

### Infrastructure (self-contained)

- [ ] **Continuous-sample loader.** New loader reading raw `.fif` for PS + lexical corpora (not phoneme-epoched). Wire in z-score recipe A (per-channel mean/std pooled across all pre-auditory baselines; verified 2026-04-18). Target: `src/speech_decoding/v14/continuous_dataset.py`.
- [ ] **SSL objective.** Mask generator (spatial / temporal / spatiotemporal patch). Reconstruction head. Loss (MSE on HGA or contrastive). Checkpoint interop so pretrained weights load cleanly into the per-phoneme loader.
- [ ] **Calibration module stub.** `src/speech_decoding/v14/calibration.py` signature-only (no-op default). Reserves the interface for Phase-2 per-patient residuals without baking assumptions into Stage-2 code.
- [ ] **Per-electrode token path at sparse layouts.** Verified for pool training (T2.2 ran). Needs test under sEEG-like sparse layouts + a mixed-cohort dataloader that interleaves uECoG + lexical patients.

---

## Backlog (deferred to later stages)

- **T3.6 decomposed path** (old tasks #10/#11/#12) — dual-stream cross-attn + 611-token backbone. Subsumed into the Stage-2 atlas-mechanism ablation, where cross-attn on per-electrode tokens is strictly cleaner than the per_cell dual-stream variant. Likely retired as originally specified.
- **Phase-2 learned per-patient calibration** — enabled by the `calibration.py` stub above.
- **RH patient re-inclusion (S22, S58)** — Stage 3 with sEEG join.

---

## Reference

- DCC helpers: `scripts/ablation/{submit,status,logs,collect,query,dcc_sync_check,peek}.py`. Each has `--help`.
- DCC setup + rsync recipe: `references/dcc_setup.md`.
- Raw ablation log: `experiments/v14_ablation_log.csv` (authoritative results).
- Submissions ledger: `.ablation_submissions.jsonl` (gitignored; decodes task ids → fold/seed).
