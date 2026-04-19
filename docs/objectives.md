# Objectives — Research Direction

Objectives layer of the triad (objectives → strategy → tactics). This doc stays stable; revision-prone architecture and task detail live in `docs/strategy.md` and `docs/tactics.md`.

## Program hypothesis

**Parcels, not electrodes, are the shared representation across patients and sensors.** An atlas-grounded parcel-token architecture should transfer cross-patient (within a sensor), cross-sensor (within an anatomy), and cross-lab (within a modality).

No iEEG foundation model has tested cross-sensor representational transfer rigorously. POYO+ and Charmander do cross-session within a single modality; BrainWave / FoME are joint EEG+iEEG without a sensor-to-sensor transfer claim. Cross-patient within a sensor is the stepping stone; cross-sensor within an anatomy is the destination.

**Design rule for every decision: does this structure survive a sensor change?**

## Evaluation philosophy

Every architectural change and data-scaling step reports two protocols, always:

1. **Pooled joint** — one model trained on all patients; each patient's held-out fold evaluated on the shared model. Tests whether weight-sharing during training helps each patient.
2. **LOPO warm-start** — pretrain pooled on `N−1` patients (held-out never seen), then finetune per-patient on held-out fold-train, loading the pretrained checkpoint. Tests whether pretraining transfers to a *new* patient. Load-bearing for Phase 1.5 SSL, Phase 2 cross-sensor transfer, and Phase 3 external-corpus transfer.

**The informative signal is the gap between pooled joint and LOPO warm-start.** Warm-start matching pooled → backbone learns transferable structure. Warm-start matching scratch → no transfer, regardless of how good pooled looks.

## Stage roadmap

Stages defined by *data scaling + hypotheses only*. Per-stage architectural defaults (revision-prone) live in `docs/strategy.md`.

### Stage 1 — Single-sensor correctness pass (Phase 1)

**Data:** 4-7 core LH PS uECoG, ~1 min/patient supervised. ~7 min total.

**Hypotheses:**
- **H1.1 (primary):** Parcel-token pipeline decodes 9-phoneme uECoG at or above the 0.734 S14 baseline (0.825 population mean across 11 patients).
- **H1.2:** Cross-patient pooled-joint training matches or beats solo per-patient training.

### Stage 2 — In-sensor scaling (Phase 1.5)

**Data:** up to 23 LH patients supervised (7 PS + up to 16 lexical, gated on Zac's quality assessment of the lexical cohort); SSL on continuous uECoG + lexical raw corpus (~7 h base).

**Hypotheses:**
- **H2.1 (primary):** More patients → better LOPO warm-start on held-out patients.
- **H2.2 (primary):** SSL pretrain on the continuous corpus → better LOPO warm-start than supervised-only pretrain.
- **H2.3:** PS-pretrained encoder transfers to the lexical corpus (different task, different phoneme inventory).

### Stage 3 — Cross-sensor join (Phase 2)

**Data:** + Cogan internal speech sEEG (~33 h; patient selection TBD with Zac). **Pretrain direction: sEEG → uECoG** — the larger, sparser, noisier corpus pretrains the backbone; the denser, cleaner uECoG is the finetune / eval target (Charmander-style noisy-pretrain → clean-finetune).

**Hypotheses:**
- **H3.1 (headline claim):** Backbone pretrained on sEEG transfers to uECoG without re-training the backbone. *Parcels are the shared representation across sensor types within an anatomy.*
- **H3.2:** Joint sEEG + uECoG training → positive transfer both directions (no interference).

### Stage 4 — External validation (Phase 3)

Loose menu, pending PI-level access. Kept deliberately under-specified — specifics are set when Stage 3 lands.

**Candidate data:**
- **AJILE12** (DANDI 000055, 12 ECoG pts, ~1,280 h total; ~15-40 h speech-on-cortex after talking-segment extraction) — massive SSL.
- **Flinker full speech ECoG cohort** (Nat Mach Intell 2024, 48 pts) / **Chang sentence-level ECoG** (Nat Neurosci 2020, Makin et al) / **Bouchard-Chang DANDI 000019** (4 pts, 256-ch HD ECoG, CV syllable production) — external speech production.
- **Auditory Naming EC** (eegdash DS006234 / DS006910), **Picture Naming EC** (DS006233, 108 pts), **Visual Naming EC** (Asano, 110 pts) — task generalization beyond non-word repetition.

**Hypothesis:**
- **H4.1:** Representation transfers to external-lab chronic ECoG on speech tasks.

## Stage-advance gates

Coarse gates only; fine-grained gates (architecture thresholds, per-ablation decisions) live in `docs/strategy.md`.

- **Stage 1 → 2:** pooled PER ≤ 0.825 population baseline; LOPO warm-start non-regressive vs scratch.
- **Stage 2 → 3:** H2.2 confirmed (SSL pretrain improves LOPO warm-start over supervised-only).
- **Stage 3 → 4:** H3.1 or H3.2 confirmed.

## Pointers

- **Strategy** (per-stage architecture, ablations, live wave scoreboard, frozen contract, patient scope): `docs/strategy.md`
- **Tactics** (task list, in-flight jobs, blockers): `docs/tactics.md`
- **References**:
  - Patient tables, corpus sizes, channel maps, field-landscape audit: `docs/references/data_reference.md`
  - DCC cluster + submission recipes: `docs/references/dcc_setup.md`
- **Experiment log** (authoritative results): `docs/experiments/v14_ablation_log.csv`
