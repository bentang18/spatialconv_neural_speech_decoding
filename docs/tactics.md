# Tactics — Concrete Task List

Tactics layer of the triad (objectives → strategy → tactics). Operational: what's running, what to do when it lands, what's blocked. Updated 2026-04-20 (Stage 1 closed).

- Objectives: `objectives.md`
- Current-stage strategy: `strategy/stage_1.md`
- DCC tooling: `references/dcc_setup.md`

**Current stage:** Stage 1 closed 2026-04-20. Default frozen at `per_cell + partialconv + pe2d_frozen` @ d=32, depth=3, pool=(4,8). H1.2 confirmed at full Phase-1 LH scope (0.833 ± 0.060). Atlas-mechanism LOPO-inert at 4-core scale (mechanism-claim deferred to Stage 2). Pivoted to Stage-2 prerequisite work.

---

## In flight (DCC queue)

Nothing. Close-out wave landed 2026-04-20.

Poll with `scripts/ablation/status.py`. Tail logs with `scripts/ablation/logs.py <job_id>`. Peek mid-flight JSONs with `scripts/ablation/peek.py <job_id>`.

## Immediate housekeeping (Stage-1 close-out admin)

- [x] **Copy 7-LH pooled checkpoint off `/work`.** Done 2026-04-20: 15 `.ckpt.pt` + 15 `.result.json` at `/hpc/group/coganlab/ht203/stage1_ckpt/`. Warm-start / SSL init source.

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
