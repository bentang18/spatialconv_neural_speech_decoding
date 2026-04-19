# Docs

Organized as Sun Tzu's objectives → strategy → tactics. Exactly **three working docs** live here; everything else is reference or archive.

## Working docs (the triad)

| layer | doc | what it answers |
|---|---|---|
| **Objectives** | [`objectives.md`](objectives.md) | program hypothesis, stage roadmap, evaluation philosophy, stage-advance gates. Stage-stable. |
| **Strategy** | [`strategy.md`](strategy.md) (index) + [`strategy/stage_<N>.md`](strategy/stage_1.md) | per-stage: default architecture, frozen contract, patient scope, live scoreboard, rejected paths, discipline. Revision-prone; one doc per stage. |
| **Tactics** | [`tactics.md`](tactics.md) | concrete task list: in-flight jobs, post-landing actions, blockers. Refreshed when jobs land. |

Do not create additional planning or tracker docs. If a new question arises, extend the relevant triad doc. A Stage-N strategy doc is written only when Stage-(N−1) has concluded enough to define the architectural entry point. Doc surplus breaks this organization.

## References (static, consult as needed)

- [`references/data_reference.md`](references/data_reference.md) — per-patient tables: array layouts, sig/artifact channels, Brainnetome parcel list, raw-corpus sizes.
- [`references/dcc_setup.md`](references/dcc_setup.md) — Duke DCC cluster setup, rsync recipes, submission workflow.
- [`experiments/v14_ablation_log.csv`](experiments/v14_ablation_log.csv) — authoritative raw results.
- [`qc/`](qc/) — short per-check QC reports (coord bridge, support cache, phoneme audit).
- [`figures/`](figures/) — generated images and HTML interactives.

## Archive

Historical material lives under `archive/`:

- `archive/plans/` — superseded plan docs.
- `archive/experiments/` — frozen-in-time experiment trackers.
- `archive/design_docs/` — historical design documents (pre-B-1-amendment `neural_field_perceiver_v14.tex/.pdf`).
- `archive/nca_jepa/`, `archive/v12_era/`, `archive/lopo_autoresearch/`, etc. — pre-v14 directions.
- `archive/experiment_log.md`, `archive/training_log.md`, `archive/research_synthesis.md` — long-form histories.
- `archive/implementation_tasks_*.md` — pre-closure blocker logs.

Archive is append-only. Move superseded material in; never move it back out without a rewrite.
