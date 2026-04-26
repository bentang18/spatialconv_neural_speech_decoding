# Strategy — Stage Index

Strategy layer of the triad (objectives → strategy → tactics). Each stage has its own doc under `strategy/` with default architecture, frozen contract, patient scope, live scoreboard, rejected paths, and discipline — scoped to that stage only.

Stage hypotheses and advance gates live in `objectives.md`; they are stage-stable. Architectural defaults are revision-prone and belong here.

## Stages

| Stage | Scope | Strategy doc |
|---|---|---|
| Stage 1 (Phase 1) | Single-sensor supervised correctness pass on uECoG | [`strategy/stage_1.md`](strategy/stage_1.md) — *closed 2026-04-20* |
| **Stage 2 (Phase 1.5)** | In-sensor scaling: uECoG supervised expansion (PS + lex) + continuous-corpus SSL | [`strategy/stage_2.md`](strategy/stage_2.md) |
| Stage 3 (Phase 2) | Cross-sensor join (uECoG + Cogan sEEG D-cohort) | *TBD* |
| Stage 4 (Phase 3) | External-lab validation | *TBD* |

Stage-N strategy is written only when Stage-(N−1) has concluded enough to define the architectural entry point. Pre-writing downstream stages bakes in assumptions that experiments haven't validated yet.
