# Data Acquisition Matrix

Updated: 2026-04-12

This document ranks the highest-value datasets for scaling `v14` and related cross-patient speech-decoding work.

Ranking criterion is **marginal value for `v14`**, not raw subject count. The most valuable datasets are the ones that best help with one or more of:

1. learning shared **speech-motor / articulatory** dynamics
2. learning shared **field-potential temporal priors** at scale
3. stress-testing **cross-patient spatial calibration**
4. providing a realistic **held-out transfer target**

The main distinction is:

- **Production / naming data**: highest value for articulatory-state learning and downstream decoder development
- **Naturalistic listening data**: high value for SSL and temporal/language priors, but weaker for motor speech decoding
- **Generic iEEG data**: useful only if scale is large enough and localization metadata are usable

## Recommended Training Roles

- **Supervised / task-adaptive pretraining**: overt speech production, naming, syllable/word articulation
- **SSL pretraining only**: passive listening, naturalistic stories, movies, generic language tasks
- **Held-out evaluation**: datasets behaviorally close to target speech production but not used in model selection

## Ranked Matrix

| Rank | Dataset | Scale | Modality / task | Access | Best use in stack | Why it is high value | Main risks / caveats | First action |
|---|---|---:|---|---|---|---|---|---|
| 1 | **Internal speech sEEG clinic corpus** | ~50 pts × ~40 min = ~33 h | sEEG, speech task(s), clinic recordings | Internal | **Primary SSL pretraining** and cross-patient transfer study | Best immediately accessible scale in the lab; same broad modality family as target; enough to test whether shared temporal priors help uECoG | sEEG is not uECoG; unclear task heterogeneity; likely weaker surface motor sampling than ECoG/uECoG | Confirm patient count, exact task mix, and localization completeness; prioritize BIDS + coordinate audit |
| 2 | **Flinker full speech ECoG cohort** | 48 participants | ECoG, overt speech across multiple tasks | Request / collaboration | **Highest-value external production corpus** | Best external match to target problem: overt speech, ECoG, sizeable cohort, directly useful for speech-motor priors and decoder pretraining | Full cohort is not openly posted; need collaboration / data use agreement; metadata harmonization likely nontrivial | Greg/PI email + data sharing agreement; ask specifically for electrode localizations, stimulus timing, and audio availability |
| 3 | **Chang / Makin / Moses sentence-level speech corpus** | multi-patient, sentence-level sequence decoding | ECoG, overt continuous speech | Request / collaboration | **Sequence-decoder pretraining / evaluation** | Most relevant external corpus for genuine sequence decoding, not just classification or naming | Access friction; likely heterogeneous preprocessing and licensing; may not be easy to harmonize quickly | Contact corresponding authors; ask about deidentified iEEG, audio, transcript alignment, and coordinates |
| 4 | **Auditory Naming EC / Auditory naming** (Asano group; verify canonical OpenNeuro release) | 119-121 participants | Intracranial EEG, auditory naming with response onset | Open | **Large-scale response-locked speech-network pretraining** | Massive open cohort with event markers and MNI-305 coordinates; excellent for cross-patient naming dynamics and speech onset modeling | There appear to be multiple closely related OpenNeuro releases; naming is not phoneme sequence decoding; whole-brain epilepsy coverage is heterogeneous | Verify which auditory-naming release is canonical before bulk ingest, then quantify response timing quality and perisylvian coverage |
| 5 | **Picture naming** (`DS006233`) | 108 participants | Intracranial EEG, picture naming with response onset | Open | **Production-adjacent supervised pretraining** | Large open overt naming dataset; likely closer to motor output than passive listening; good for response-locked HGA models | Not phoneme-labeled; visual object semantics confound speech motor activity; montage heterogeneity | Inspect event structure, response alignment, and language regions reached across subjects |
| 6 | **Visual Naming EC** (Asano group; dataset ID to verify) | 110 participants | Intracranial EEG, visual naming with response onset | Open | **Companion naming corpus** | Same value proposition as the other Asano datasets: large, open, coordinate-aware, language-task relevant | Need to verify the exact OpenNeuro dataset ID and metadata before scripting around it | Verify dataset ID, license, and event-code schema before ingestion |
| 7 | **Internal Cogan raw continuous field-potential corpus** | 29 patients, ~456 min | uECoG + related field-potential recordings, raw continuous | Internal | **Bridge corpus for modality adaptation** | Already local; closest operational path from generic SSL to target preprocessing; valuable for HGA extraction pipeline and modality adaptation | Small by SSL standards; task heterogeneity; may still be too little for standalone invariance learning | Finish HGA extraction pipeline and build a clean patient/task manifest |
| 8 | **DESIS / SingleWordProductionDutch** | 10 participants, 1103 electrodes | sEEG, read-aloud single-word production | Open | **Open speech-production transfer corpus** | Best openly available speech-production sEEG dataset with localization/anatomy; directly useful for production-focused transfer experiments | Smaller than the naming corpora; Dutch labels/tasks differ from target; probably limited total duration | Pull OSF metadata and confirm audio/transcript alignment plus coordinate completeness |
| 9 | **Bouchard-Chang DANDI 000019** | 4 subjects, 256-ch HD ECoG | ECoG, consonant-vowel syllable production | Open | **High-value articulatory fine-tuning set** | Small but extremely relevant: overt syllable production with dense ECoG over speech motor cortex; one of the best open articulatory datasets | Very small cohort; narrow task; may not provide much invariance on its own | Ingest immediately; use as a clean external motor-speech benchmark |
| 10 | **Brain Treebank** | 10 subjects, ~43.5 h | Intracranial recordings during naturalistic movie watching | Open | **Large-scale language / temporal SSL** | Best open naturalistic intracranial language corpus currently available; rich annotations; enough duration to matter for SSL | Perception, not production; invariances learned here may not transfer fully to articulatory decoding | Use for JEPA pretraining only; do not let it dominate architectural decisions for motor speech |
| 11 | **SWEC iEEG dataset** | **~10,000 h** | Long-term heterogeneous clinical iEEG, largely seizure / background dynamics | Open | **Massive generic iEEG foundation-model pretraining** | By far the largest open iEEG pretraining opportunity currently on the table; ideal for testing whether scale alone can improve temporal robustness, denoising, and heterogeneity handling | Extremely weak task match to speech production; seizure-dominated objective may teach the wrong invariances unless carefully staged; likely no speech/audio supervision | Treat as a dedicated generic pretraining stage, not pooled speech data; evaluate only through transfer to speech/naming corpora |
| 12 | **Podcast ECoG** (`ds005574`) | 9 participants, 30-min story | ECoG, naturalistic story listening | Open | **Audio-text-aligned SSL** | Includes high-gamma derivatives, transcript alignment, acoustic features, and LLM features; very useful for neural-audio/language auxiliaries | Still perception-only; modest scale; likely biased toward temporal/language rather than motor speech | Add as a second-stage SSL corpus after Brain Treebank if audio-text auxiliary experiments begin |
| 13 | **sEEG Passive listening to natural speech** (`ds004703`) | 10 participants, 2 sessions | sEEG, passive conversational speech listening | Open with non-commercial restriction | **sEEG temporal SSL** | Good open speech-listening sEEG with clear task structure and usable metadata; helps modality-relevant SSL | Non-commercial restriction in README; not suitable for any broadly commercialized model release; passive listening only | Resolve license terms internally before major investment; use only for research SSL |
| 14 | **Open multimodal iEEG-fMRI film dataset** (`ds003688`) | 51 participants | iEEG / HD-ECoG / sEEG, naturalistic audiovisual film | Open | **Heterogeneity robustness pretraining** | Very large open intracranial cohort; ideal for stress-testing modality and coverage heterogeneity | Weakest match to speech production; movie-driven dynamics may teach generic sensory priors more than articulatory state | Use only after speech/naming corpora are in place; valuable for robustness, not first-pass decoder training |
| 15 | **UPenn RAM / delayed free recall ecosystem** (`ds004789` + RAM public archive) | 251-273 participants, large public release (`~871 GB` on NEMAR) | Mixed grid / strip / depth iEEG, delayed free recall and memory tasks | Open / public consortium archive | **Late-stage generic human iEEG pretraining** | One of the only other public intracranial ecosystems with truly large patient count and file volume; useful for robustness and generic human iEEG representation learning | Very weak task match to speech motor decoding; more fragmented than SWEC; coordinates / metadata quality vary by recording; behavioral task is memory-heavy and far from articulatory state | Treat as optional late-stage generic pretraining only; audit harmonization cost before investing heavily |

## Best SSL Sources for `v14`

This ordering is by **expected transfer to patient-invariant articulatory decoding**, not by raw recording hours.

1. **Internal speech sEEG clinic corpus**
2. **Brain Treebank**
3. **SWEC iEEG dataset**
4. **UPenn RAM / delayed free recall ecosystem**

Why this ordering:

- **Internal speech sEEG** is the best modality-and-task match. Even at only `~33 h`, it is still speech-related intracranial field potentials and should teach the most relevant temporal priors for `v14`.
- **Brain Treebank** is the best open compromise between language relevance, clean annotations, and enough duration to matter for SSL.
- **SWEC** is the strongest generic pretraining source by scale, but the underlying objective is mostly seizure / background clinical dynamics rather than speech or language.
- **UPenn RAM** is valuable as a large human iEEG ecosystem, but its delayed-recall paradigm is even farther from articulatory-state learning than Brain Treebank and more fragmented than SWEC.

If ranked by **raw generic pretraining horsepower** instead, the order flips to roughly:

1. **SWEC**
2. **UPenn RAM**
3. **Brain Treebank**
4. **Internal speech sEEG**

That is precisely why the staged curriculum matters: the biggest corpus is not automatically the best corpus for `v14`.

## Recommended Priority Tiers

### Tier A — Acquire / ingest first

These datasets are most likely to move `v14` materially in the next 1-2 months.

1. Internal speech sEEG clinic corpus
2. Internal Cogan raw continuous field-potential corpus
3. Auditory Naming EC
4. Picture naming
5. Bouchard-Chang DANDI 000019
6. DESIS / SingleWordProductionDutch

Why this tier:

- immediately accessible or nearly so
- enough task relevance to inform articulatory-state modeling
- enough scale to make early SSL / transfer experiments meaningful

### Tier B — Highest-value request-only targets

1. Flinker full speech ECoG cohort
2. Chang / Makin / Moses sentence-level speech corpus

Why this tier:

- best external match to the end task
- likely the biggest gain for overt speech decoding if obtained
- worth PI-level effort because open data alone probably will not close the gap

### Tier C — SSL scale-up after the production core is in place

1. Brain Treebank
2. SWEC iEEG dataset
3. Podcast ECoG
4. sEEG Passive listening to natural speech
5. Open multimodal iEEG-fMRI film dataset
6. UPenn RAM / delayed free recall ecosystem

Why this tier:

- large and useful for temporal / language priors
- not close enough to target behavior to drive the whole architecture
- should be used to improve robustness and sample efficiency, not to define the final decoder

## Practical Acquisition Strategy

### Stage 1 — Build the production/naming core

Target outcome: a pooled corpus that is behaviorally close enough to speech motor decoding to test cross-patient transfer honestly.

Acquire / ingest:

1. Internal speech sEEG clinic corpus
2. Internal Cogan continuous field-potential corpus
3. Auditory Naming EC
4. Picture naming
5. Visual Naming EC
6. DESIS
7. DANDI 000019

Use for:

- response-locked HGA modeling
- supervised pretraining on naming / production-adjacent tasks
- small-scale JEPA pretraining in a speech-relevant regime

### Stage 2 — Add the high-value private speech cohorts

Target outcome: enough overt speech ECoG to make the shared dynamics model credible.

Acquire:

1. Flinker full cohort
2. Chang sentence-level speech corpus

Use for:

- speech-motor pretraining
- sequence-decoder development
- true external held-out speech evaluation

### Stage 3 — Add naturalistic SSL scale

Target outcome: stronger temporal priors, better robustness to coverage and patient heterogeneity.

Acquire:

1. Brain Treebank
2. SWEC iEEG dataset
3. Podcast ECoG
4. ds004703
5. ds003688
6. UPenn RAM / delayed free recall ecosystem

Use for:

- JEPA / masked prediction pretraining
- generic heterogeneous iEEG foundation-model pretraining
- audio/text auxiliary tasks
- robustness under mixed coverage and modality

## What Not To Do

- Do **not** blindly pool production, naming, listening, and movie datasets into one undifferentiated corpus.
- Do **not** mix SWEC-style seizure/background pretraining into the speech stack without a staged transfer plan; it is pretraining fuel, not target-matched supervision.
- Do **not** mistake UPenn RAM scale for speech relevance; it is a broad human iEEG reservoir, not a speech corpus.
- Do **not** let passive-listening corpora dominate model selection for an articulatory decoding paper.
- Do **not** assume more subjects means more useful signal; target match matters more than raw count.
- Do **not** drop localization quality from the ranking; datasets without usable coordinates or region annotations are much less valuable for `v14`.

## Short Version

If only a few datasets can be tackled soon, the highest-leverage order is:

1. Internal speech sEEG clinic corpus
2. Internal Cogan continuous field-potential corpus
3. Auditory Naming EC
4. Picture naming
5. Flinker full cohort
6. Chang sentence-level corpus
7. Bouchard-Chang DANDI 000019
8. DESIS
9. Brain Treebank
10. SWEC iEEG dataset

That ordering best balances:

- immediate feasibility
- closeness to articulatory decoding
- usefulness for SSL
- value for proving or disproving patient-invariant shared dynamics

## Useful Links

- Flinker cohort paper: <https://www.nature.com/articles/s42256-024-00824-8>
- Flinker public single-subject release: <https://data.mendeley.com/datasets/fp4bv9gtwk/2>
- Chang sentence-level speech paper: <https://www.nature.com/articles/s41593-020-0608-8>
- Picture naming (`DS006233`): <https://eegdash.org/api/dataset/eegdash.dataset.DS006233.html>
- Auditory naming releases to verify: <https://eegdash.org/api/dataset/eegdash.dataset.DS006234.html>, <https://eegdash.org/api/dataset/eegdash.dataset.DS006910.html>
- DESIS / SingleWordProductionDutch paper: <https://www.nature.com/articles/s41597-022-01542-9>
- DESIS OSF: <https://osf.io/nrgx6/>
- Bouchard-Chang DANDI 000019: <https://dandiarchive.org/dandiset/000019>
- Brain Treebank: <https://braintreebank.dev/>
- SWEC / MVPFormer paper: <https://openreview.net/forum?id=5M1YOW3bRq>
- SWEC dataset release: <https://huggingface.co/datasets/NeuroTec/SWEC_iEEG_Dataset>
- UPenn RAM public archive: <https://memory.psych.upenn.edu/RAM_Public_Data>
- Delayed free recall OpenNeuro summary (`ds004789`): <https://eegdash.org/api/dataset/eegdash.dataset.DS004789.html>
- Delayed free recall NEMAR entry: <https://nemar-dev.ucsd.edu/dataexplorer/detail?dataset_id=ds004789>
- Podcast ECoG (`ds005574`): <https://openneuro.org/datasets/ds005574>
- sEEG passive listening (`ds004703`): <https://openneuro.org/datasets/ds004703>
- Naturalistic film iEEG (`ds003688`): <https://openneuro.org/datasets/ds003688>
