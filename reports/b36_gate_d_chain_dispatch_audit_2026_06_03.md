# Gate-D audit — Chunk D: P3 dispatch + P1→P2→P3a→P3b→P4 chain driver

4-agent adversarial audit + 2-agent focused re-audit, 2026-06-03. Chunk D of the
BT-full-run prep. Branch `bt-full-run-prep`, commits `2f163aa` (chunk) →
`9673914` (fixes) → `b19eb6e` (F4 test).

## Verdict: PASS (after 1 fix round)

Round-1 4-agent audit **FAILED** — all four auditors converged on real,
mostly-introduced defects (two HIGH that produce silently-invalid runs). Fixes
landed; the 2-agent re-audit confirmed every finding closed, value-preserving,
and regression-tested. Full experiments suite: **402 pass**.

| Audit | Round 1 (4-agent) | Re-audit (2-agent) |
|---|---|---|
| #1 routing correctness | **FAIL** (F1 HIGH) | — |
| #2 handoff / exca semantics | **FAIL** (F4 HIGH) | — |
| #3 config drift / leakage | **FAIL** (F2 HIGH + 2 MED) | — |
| #4 elegance / regression / DCC | **FAIL** (F2 HIGH, coverage HIGH, DCC footgun) | — |
| fix-verification | — | **PASS** (5/5 closed, mutation-tested) |
| regression / new-bug | — | **PASS** (value-preserving, 401→402) |

## Findings + fixes

| ID | Sev | Introduced? | Finding | Fix |
|---|---|---|---|---|
| **F1** | HIGH | yes | Frozen-probe P4 (`--frozen-probe`/`--resume-from`) bypassed the `neural_lag_s=0` leaderboard-parity guard — its `elif` branch short-circuits the base-P4 guard. A manual `--phase 4 --frozen-probe --neural-lag-s X` silently slid the probe window off `[onset, onset+1 s]`. (Chain P4 was safe — hardcodes 0.0.) | Re-assert the guard inside the `phase4_frozen_probe` branch. |
| **F2** | HIGH | yes | `_build_v14_chain`'s inline `common` dict had drifted: it dropped `loss_variant` / `latent_valid_override` / `sa_mask_mode` / `binary_tasks` that the single-phase call passes → `--chain --loss-variant X` ran the **default** arm while `main()` printed the sister as applied. `--no-binary-tasks` also desynced the SSL/distill clip population from the P4 eval set. | Factor `_common_build_kwargs(args)` consumed by **both** call sites — a forgotten flag is now structurally impossible. Key-set verified exact (57 = 43 helper ∪ 14 per-phase, zero dropped/dup). |
| **F4** | HIGH | (latent) | `_snapshot` runs inside the exca-cached `run()` body → a cache-hit phase doesn't rewrite its `.ckpt`; on a re-run with a purged `--work-dir` the next phase's strict load died with an opaque `FileNotFoundError`. | `_load_pretrained` now raises an actionable error naming the cause (cache-hit on purged work_dir) before `torch.load`. + regression test. |
| **F6** | MED | pre-existing | A single `--phase 4` defaulted `clip_len=5.0`, not the 1 s parity window (Gate-B flag 3). | `--clip-len` → None sentinel; `main()` resolves to 1.0 for phase 4, 5.0 otherwise, before every consumer. Explicit `--clip-len` still honored. |
| **F7** | MED | yes | `--phase 4 --snapshot-ckpt-to` (no resume) built the base `Experiment`, which lacks the transferable protocol → runtime `TypeError` after a full DCC train. | `--snapshot-ckpt-to` now also implies the frozen-probe readout on P4 (the only P4 variant with the protocol), symmetric with `--resume-from`. |
| F11 | LOW | yes | Stale test-module docstrings ("Phases 1/2/3 must raise NotImplementedError"; dead xref to a renamed test). | Rewritten. |

### Not fixed (flagged, out of scope for the maiden run)
- **[LOW] `projector_mode` / `parcel_lr_scale` have no CLI surface** — the B33
  `R-project-down` / `R-head-linear` and 3b-LR sisters are unreachable from the
  dispatch CLI (every P3 run is locked to `mlp` + `parcel_lr=1/3`). Not needed
  for the maiden run; add `--projector-mode` / `--parcel-lr-scale` when those
  sisters are dispatched.
- **[LOW] base supervised P4 metrics hardcode `num_classes=2`** (pre-existing) —
  a 3-class base-supervised P4 would loudly crash (not silently corrupt); the
  frozen-probe P4 + P3 use their own internal metrics, so the maiden path is
  unaffected. Derive from `n_outputs` when a multiclass base-P4 is needed.
- **[LOW] stray `--whisper-target-cache-dir` on a non-P3 single-phase run**
  attaches an unused teacher extractor (wasteful, not wrong). Chain is correct.

## DCC launch model — the operational footgun (auditor #4)

`run_phase_pipeline` runs `[phase.run() for phase in configured]` sequentially
**in-process**. `Experiment.run()` is `@infra.apply`:
- **`--cluster None`** (the default): every `phase.run()` runs synchronously in
  the calling process. Launched via `scripts/dcc/dispatch` (which SSH-runs on the
  **login node**), that trains on the login node — unacceptable.
- **`--cluster slurm`**: each `phase.run()` submits a submitit job and the
  login-node driver **blocks** until it finishes, for the whole multi-hour chain
  — fragile to SSH drops (exca caching lets a re-run resume completed phases).

**Recommendation for the maiden run**: launch `--cluster None` *inside* an sbatch
GPU allocation (NOT via `scripts/dcc/dispatch`) on `coganlab-gpu` — the whole
chain runs in one allocation on a real GPU, handoff ckpts land on the shared FS,
no login-node training. Reserve `--cluster slurm` for the incremental
one-sbatch-per-phase path (`--phase N --resume-from --snapshot-ckpt-to`), which
is the de-risking path for the very first end-to-end run. Sync from the separate
clone `/work/ht203/repo/speech_bt` (the shared `/work/ht203/repo/speech` is a
collaborator's sweep target — do not `git reset --hard` it).

## What re-audit verified (mutation-tested)

Each fix was mutation-tested (revert the fix → the new test fails):
`test_b36_neural_lag_rejected_on_frozen_probe_p4_path` (F1),
`test_chain_threads_sister_flags` + `test_single_phase_passes_sister_flags` (F2),
`test_e3_load_missing_snapshot_raises_actionable_error` (F4),
`test_p4_clip_len_defaults_to_one_second` (F6),
`test_snapshot_ckpt_to_implies_frozen_probe_on_p4` (F7). The 4 chain/guard tests
provably FAIL against pre-fix source (run in a `9673914^` worktree). No
regressions: `_common_build_kwargs` extraction is key-set-exact, no
duplicate-keyword collisions, clip_len sentinel resolved before every consumer.
