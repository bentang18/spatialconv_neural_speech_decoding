# v14 Blocker Audit — Closing Report (2026-05-23)

Closing report for the recursive convergence-driven blocker audit launched 2026-05-22 under the directive *"Continue recursively in discovering all load-bearing gaps | converging returns (new audits add <5 distinct additions)."* Hand-off from discovery phase to gap-resolution phase.

## Audit summary

**244 enumerated gaps** across **19 orthogonal lenses** in **5 waves** (May 22–23, 2026):

| Wave | Date | Lenses | Raw → New | New/lens |
|------|------|--------|-----------|----------|
| W1 (first-pass) | pre-5/22 | Implicit (B/M/S top-level) | — / 50 | n/a |
| W2 (second-pass) | 5/22 PM | IE / AB / EV / EX | — / 44 | 11.0 |
| W3 (third-pass) | 5/22 PM | DP / PT / NT / IM / CR | 65 / 42 | 8.4 |
| W4 (fourth-pass) | 5/22 evening | PF / HB / RT / BP / CQ | 85 / 52 | 10.4 |
| W5 (fifth-pass) | 5/23 | TST / VIS / ARG / TIME / DOC | 75 / 56 | 11.2 |

**Convergence criterion not literally met.** The `<5 new/lens` floor was never reached; yield trended **upward** wave-over-wave (8.4 → 10.4 → 11.2). However, the **kind of gap** shifted: W2 yielded 100% technical-design lenses; W5 yielded only 2/5 technical (TST + VIS) with the remaining 3 (ARG, TIME, DOC) being process/coordination/cleanup surface rather than architectural unknowns. Discovery saturated in the technical-design space; the curve continues to rise only because additional process-flavored lenses keep being added.

**Audit terminates here per Ben's 2026-05-23 decision.** Further lens-fanout would mostly surface process/coordination gaps not architectural ones.

## Severity breakdown across 244 gaps

| Severity | Count | Share |
|----------|-------|-------|
| BIG | 51 | 21% |
| MEDIUM | ~143 | 59% |
| SMALL | ~50 | 20% |

## Top-30 pre-Phase-1 critical-path blockers

These must resolve before M0 (Lite-cell post-BTWordEvents-fix rerun) or before the first Phase-1 dispatch. Grouped by cluster.

### Cluster A: Code-not-written (gates all dispatches)
1. **NT01** — Phase-specific `Experiment` class shape (single vs sub-classed). [BIG]
2. **NT02** — Phase-aware `Data` class for 4-phase corpus differences. [BIG]
3. **DP03** — `SWECStudy` + `DCohortStudy` + `AJILE12Study` NeuralFetch class scaffolding. [BIG]

### Cluster B: Data-loader contract
4. **DP01** — Variable-T RoPE + variable-C collate across phases. [BIG]
5. **DP02** — Corpus-balanced sampler implementation (`WeightedRandomSampler` spec). [BIG]
6. **B02** — Cross-corpus batch composition rule (√h or h + α·N). [BIG]

### Cluster C: Phase-1 training contract
7. **B01** — Phase 1 + Phase 2 optimizer / LR / schedule. [BIG, gates first NT Experiment]
8. **B08** — Phase 1 mask block sizes. [BIG]
9. **B11** — EMA teacher layer-K averaging count. [BIG]
10. **B07** — PMA k=1 query training timeline (when does it receive gradient). [BIG, also Phase-2 dependency]

### Cluster D: Compute / storage feasibility
11. **HB01** — `/hpc/group/coganlab/` persistent quota vs Multi-STFT cache size (~18 TB estimate). [BIG]
12. **HB02** — Phase-1 GPU-hour estimate on DCC Ada-5000 32GB (NOT A100 80GB — corrects prior planning assumption). [BIG]

### Cluster E: Pre-dispatch verification
13. **TST01** — Pre-dispatch pytest marker for BIG/MEDIUM blockers. [BIG]
14. **TST05** — Phase-1 loss NaN detector (Multi-STFT + log stability). [BIG]
15. **TST10** — DCC pre-flight pytest run in `scripts/dcc/dispatch`. [BIG]

### Cluster F: Schedule reality check
16. **TIME01** — Phase-1 wall-clock conversion (GPU-h → wall-day) — must measure on M0. [BIG]
17. **TIME04** — Critical-path bottleneck — phases serial, slip cascades. [BIG]
18. **TIME07** — Fallback fork-gate at Jul 4 undefined. [BIG]
19. **TIME11** — Ben's writing cadence unanchored — surgical edits add multiplier. [BIG]

### Cluster G: Cross-memo coherence (cheap; clear before dispatch)
20. **ARG03** — PMA k=1 query training timeline ambiguous across memos. [BIG]
21. **ARG04** — Phase-4 readout dimensionality — T=15 (1s) vs T=73 (5s) clip-length mismatch with Phase 3. [BIG]

### Cluster H: Pretraining-corpus leakage audit
22. **BP20** — Pretraining-corpus subject-overlap leakage audit (Phase 1 vs Neuroprobe eval). [BIG]

### Cluster I: Statistical power for primary claims
23. **VIS13** — Statistical power + multiple-comparison correction unregistered across interp suite. [BIG, gates paper-section design]

### Cluster J: Pre-Phase-2 BIG dependencies that should be settled at the same gate to avoid revisits
24. **B03** — Phase 2 electrode-mask mechanism. [BIG, pre-Phase-2]
25. **B04** — Phase 2 loss weighting (L_recon_A + λ·L_recon_utt). [BIG, pre-Phase-2]
26. **B09** — Latent-stack parcel SA bias — keep / drop / different. [BIG, pre-Phase-2]
27. **CQ12** — D-cohort electrode count distribution — max 168 exceeds C_MAX=120. [BIG, pre-Phase-2]
28. **RT10** — Phase-boundary checkpoint `strict=False` silent-skip risk. [BIG, pre-Phase-2]
29. **TST03** — Phase-1 ↔ Phase-2 checkpoint strict-mode compatibility test. [BIG, pre-Phase-2]
30. **IM11** — Phase-1 SWEC high-bin under-training cascade into Phase 2. [BIG, pre-Phase-2]

## BIG gaps deferred to later gates (21 remaining)

**Pre-Phase-3** (3): B05 (triangular-pool spec), B06 (preflight protocol), CQ17 (BT films in Whisper pretrain corpus → distillation leakage).

**Pre-Phase-4 / eval** (5): B10 (linear-probe optimizer), PT01 (eval window 1s vs task-specific), IM04 (iMINDBench flatten convention), RT11 (robust-z held-out scope leakage), BP01 (PopT apples-to-apples).

**Paper-draft time** (5): IM05 (iMINDBench logistic baseline is raw-spectral parity break), PF01 (anatomy-vs-per-subject-params defense), PF04 (DIVER-1 within-session vs v14 cross-subject framing), BP02 (DIVER-1 protocol-mismatch caveat), CQ08 (AJILE12 anatomy-unavailable framing), DOC05 (per-corpus data-release policy).

**Schedule** (2): TIME02 (Phase-2 PoC), TIME03 (paper-writing turnaround vs Aug 29).

**Interp-output spec** (4): VIS01 (parcel connectivity figure), VIS02 (band re-discovery metric), VIS03 (temporal latency cascade test), VIS04 (PMA attention visualization). VIS13 is in the top-30 because it gates the whole interp section's statistical spec.

**Hardware** (1): HB03 (Phase-2 GPU-hours + 224h convergence risk).

## What converged vs what stayed open

**Converged in W2–W5**: front-end (Multi-STFT spec confirmed across waves), atlas routing (DK-first pivot stable), corpus-diet decisions (P1 vs P2 lock), submission-gate numerics (0.667/0.628), arch param-count (13M default with 13/25/40M sister).

**Stayed open**: per-phase implementation contracts (B01–B11), data-loader scaffolding (DP01–DP03 + NT01/02), per-corpus Study classes, pre-dispatch test infrastructure (TST01–TST11), schedule feasibility (TIME01–TIME11), interp-output specs (VIS01–VIS16), methods-section documentation (DOC01–DOC08), and cross-memo coherence cleanup (ARG01–ARG10).

## Recommended walkthrough order

The walk-through protocol from the blocker doc (pick lowest open ID, decide or refine, mark ✅, cascade to canonical memo) applies. Suggested order:

1. **Bundle 1 — Code scaffolding (5 items, target: 1 week)**: NT01, NT02, DP01, DP02, DP03 → unblocks all dispatches.
2. **Bundle 2 — Cross-memo coherence cleanup (10 items, target: 1 day)**: ARG01–ARG10 → cheap, removes contradiction drag.
3. **Bundle 3 — Pre-dispatch tests (5 items, target: 3 days)**: TST01, TST03, TST05, TST09, TST10 → CI gate operational.
4. **Bundle 4 — Phase-1 training contract (5 items, target: 2 days discussion)**: B01, B02, B08, B11 + DP02 (already in Bundle 1) → Phase-1 dispatchable.
5. **Bundle 5 — Compute fit (3 items)**: HB01, HB02 → quota OK + budget anchored.
6. **Bundle 6 — Schedule + fork-gate (4 items)**: TIME01, TIME04, TIME07, TIME11 → Jul 4 fork gate defined; M0 measures wall-clock to validate critical path.

This bundling lands the top-30 in ~2 weeks of focused discussion + code work, opening M0 dispatch.

## Limitations of this audit

- Lens choice was non-exhaustive; remaining unaudited surface includes per-task-head architecture, distributed-training numerics, mixed-precision contracts, label-derivation pipelines, IRB/ethics submission, real-time-inference feasibility, and a deeper cite-graph audit. These were de-prioritized because the technical-design yield curve was already saturating.
- Many MEDIUM and SMALL gaps were not enumerated to the same depth as BIG; some MEDIUM may upgrade to BIG once their adjacent BIG is settled (e.g., M-gaps on Phase-1 step counts).
- ARG (cross-memo coherence) gaps are detection-quality, not exhaustive — only the high-signal contradictions were captured; minor wording drift wasn't enumerated.
- TIME estimates rely on unmeasured throughput; the picture clarifies after M0.
- Process-flavored gaps (TIME, DOC, ARG) interact with downstream resolution of technical gaps — re-audit may be needed once the top-30 are walked.

## Pointers

- Living blocker doc: `docs/neuroprobe/v14_blockers.md` (244 entries, walkthrough protocol at bottom).
- Plan doc (open-contract items above the blocker-doc level): `docs/neuroprobe/plan.md` §"Open questions" + §"Open contract decisions still pending".
- Canonical sources for resolution: post-v3 amendment, 5/22 iMINDBench pivot, 3-phase recipe, cross-subject-pretrain strategy memos.
