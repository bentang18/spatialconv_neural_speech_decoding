# Data Acquisition Matrix

Updated: 2026-05-28

Curated world-inventory of iEEG datasets for `v14`'s cross-patient + cross-sensor speech-decoding program.

**Scope**: Human intracranial **field potentials only** — ECoG, uECoG/micro-ECoG, sEEG, epidural ECoG. **Utah intracortical arrays (spiking activity) are out of scope** — different signal modality, not a foundation-model target for uECoG. The BrainGate Dryad chronic-speech series (Willett T12, Card T15, Wairagkar, Kunz) is therefore excluded despite being the only open chronic speech BCI data in the field.

**Ranking criterion**: marginal value for `v14`'s cross-patient + cross-sensor speech representation. Value drivers:

1. Shared **speech-motor / articulatory** dynamics (production > perception)
2. **Field-potential modality match** (ECoG / uECoG preferred for same-modality SSL; sEEG for cross-sensor validation)
3. **Onset labels minimum** — stim, response, or vocalization timestamps
4. **Scale** — subjects, hours, and annotated events
5. **Chronic recordings** — uniquely load-bearing for stability claims; almost nonexistent in field-potential mode

The primary distinctions are:

- **Production / naming**: highest value for articulatory-state learning
- **Naturalistic listening**: good for SSL + temporal / language priors, weaker for motor speech
- **Chronic implant**: uniquely valuable for stability, rare in field-potential mode
- **Generic iEEG**: SSL fuel, not a decoder training target

## Recommended Training Roles

- **Supervised / task-adaptive pretraining**: overt speech production, naming, syllable/word articulation
- **SSL pretraining only**: passive listening, naturalistic stories, movies, generic continuous iEEG
- **Held-out evaluation**: behaviorally close to target speech production but not used in model selection

## Ranked Matrix

| Rank | Dataset | Scale | Modality | Task + onsets | Access | Best use | Why high value | Main risks | First action |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Internal Cogan PS + lex uECoG corpus** | **27 disjoint pts, 6.79 h** (PS = 11 pts / 2.83 h, lex = 16 pts / 3.96 h); fsaverage-projectable today = 14 pts / 3.57 h | uECoG (PS) + uECoG/sEEG (lex) | **Phoneme-level production (PS nonword repetition) + word-level production (lex word reading)**; response-locked, audio aligned via MFA | Internal (already in pipeline) | **Phase 1 + 1.5 primary working corpus** | The actual task v14 is built on; PS + lex are zero-overlap so already a disjoint cross-patient testbed; phoneme-level `.fif` exists for 11/11 PS + 15/16 lex | Small in absolute hours (6.79 h); 13/16 lex pts lack FreeSurfer recons (blocks fsaverage projection until Zac shares) | Continue Phase 1.5 SSL + supervised lex expansion; resolve missing lex recons |
| 2 | **Internal Cogan sEEG D-cohort** (4 speech tasks; full audit complete 2026-04-24) | **87 unique D-pts, 180.59 h continuous** (PS 40.73 h / 50 pts; LexDelay 65.87 h / 52 pts; LexNoDelay 31.68 h / 26 pts; SentenceRep 42.31 h / 34 pts); **122 D-pts with full prep artifacts** (support cache + coord cache + manifest in `data/dcohort_manifest.csv`); 128 D-recons enumerated | sEEG (DIXI depth, 3.3–3.5 mm inter-contact median verified A6) | **PS uses identical 52 CVC/VCV tokens to uECoG**, same 9-phoneme label map (no remap); LexDelay / LexNoDelay / SentenceRep production-locked event onsets. **No phoneme-level MFA on Box** (Nanlin asks: location of `D_Data/Phoneme_Sequencing/`); SSL viable today, supervised phoneme decode pending alignments | Internal (Box + DCC) | **Cross-sensor expansion priority + Neuroprobe Tier-2 pretrain corpus** (`docs/neuroprobe/tier2_dcohort_integration.md`) | **26.6× more raw hours, 3.2× more patients than uECoG** (180.59 h vs 6.79 h); pre-baked 3mm BNA CSVs convention-compatible with uECoG Tier-1 (parity-verified 2026-04-21); cross-sensor (depth ↔ surface) stress-test for v14's atlas-anchored thesis; full self-answerable Stage-3 prep landed (A1 sig-channel, A2 events+muscle, A3 corpus, A4 z-score recipe, A5 DCC sync recipe, A6 sensor geometry, B1/B2 caches, B3 manifest, C1 RH-expansion stub, C2 Nanlin-asks list) | Different sensor type vs uECoG (depth, no 2D grid — Stage-3 loader is an architecture decision: pseudo-grid vs per-electrode-token); preprocessing reference choice across CAR / WM / M1 / STG / HIPP / LING is Nanlin-asks (BT cross-subject baseline uses Laplacian — no equivalent in Cogan tree); **58.35 h still Box-only on DCC** (LexDelay missing 31 D-pts + entire LexNoDelay 26 D-pts) | (1) Rsync 58.35 h Box→DCC per `reports/dcc_sync_plan_2026_04_24/rsync_commands.sh`; (2) Email Nanlin (Laplacian/bipolar reference variant + MFA location); (3) Build `data/neuroprobe/tier2_corpus_manifest.csv`; (4) Begin SSL pretrain on continuous corpus once Tier 1 (BT-only) clears 0.539 |
| 3 | **AJILE12** (Peterson et al. 2022; DANDI 000055) | **12 pts, ~1,280 h ECoG**, 55 recording days, 500 Hz, ≥64 elec/pt | ECoG (grids/strips + depth); Precentral (motor + ventral speech motor), Postcentral, MTG/ITG | Naturalistic ADL incl. **"talk"** segments (video-annotated, not audio-aligned); no phoneme labels | Open (DANDI NWB) | **Primary generic iEEG-FM pretraining (Charmander-validated)**; **best open speech-motor SSL corpus** | By far the largest open ECoG; direct modality match to uECoG; ~24 h/pt rivals SWEC without seizure bias; "talk" label gives est. **15–40 h speech-on-cortex** (5–14× PS+lex); Charmander used 8 of 12 pts | No aligned audio or transcripts (SSL only, no supervised phoneme decoding); 500 Hz cap vs 2 kHz source | Ingest via DANDI; run Charmander-style masked-channel SSL; consider "talk"-segment detection as auxiliary |
| 4 | **Auditory Naming EC** (Asano group; OpenNeuro ds006234) | **~119–121 pts**, iEEG (ECoG + sEEG mixed), naming response onsets | iEEG, **auditory naming** (subject hears definition → names aloud) | Stim + response onsets | Open | **Large-scale response-locked speech-network pretraining** | Massive open cohort with event markers and MNI-305 coordinates; strongest open production-adjacent corpus by patient count | Word-level (not phoneme) labels; heterogeneous whole-brain epilepsy coverage; multiple closely-related OpenNeuro releases | Audit canonical release, MFA audio if available, quantify perisylvian coverage |
| 5 | **Picture Naming** (ds006233) | **108 pts**, iEEG, picture naming with response onset | iEEG, **visual → vocal naming** | Stim + response onsets | Open | **Production-adjacent supervised pretraining** | Closer to motor output than passive listening; response-locked HGA; companion to Auditory Naming | Visual object semantics confound motor activity; montage heterogeneity; not phoneme-labeled | Inspect event structure, response alignment, language regions reached |
| 6 | **Visual Naming EC** (Asano group; ds ID to verify) | ~110 pts, iEEG visual naming | iEEG, visual naming | Stim + response onsets | Open | **Companion naming corpus** | Third pillar of the Asano naming ecosystem | Verify dataset ID; potential overlap with Picture Naming | Verify ID, license, event schema |
| 7 | **Du-IN / Du-IN2** (Zheng et al. 2024, NeurIPS; arXiv 2405.11459) | **12 sEEG pts**, Mandarin 61-word vocabulary; targeted at vSMC + STG | sEEG, overt word-reading | Trial + response onsets, audio present | Open (with benchmark release) | **Largest open speech-PRODUCTION iEEG benchmark** | Purpose-built cross-patient benchmark; vSMC + STG targeting closest to our speech-motor + STG electrodes; scripted for reproducibility | Mandarin ≠ English (phoneme/tone inventory differs); 61-word closed vocabulary | Ingest via Du-IN benchmark repo; reproduce author's cross-patient baselines before introducing `v14` |
| 8 | **Verwoert 2022 SingleWordProductionDutch** (ds003194) | **10 sEEG, 1,103 elec**, Dutch read-aloud + 48 kHz mic | sEEG, overt word production | Stim + vocal onsets, BIDS `_events.tsv`, NWB | Open (OpenNeuro) | **Default "onset + audio + sEEG" benchmark** | Cleanest open reproducible production benchmark; BIDS-compliant; audio + NWB | Small; Dutch; 10 patients | Ingest immediately; use as external production-transfer benchmark |
| 9 | **Bouchard-Chang DANDI 000019** | **4 pts, 256-ch HD ECoG**, consonant-vowel syllable production | ECoG (HD) | Overt CV syllable production, onsets present | Open | **High-value articulatory fine-tuning / held-out eval** | Small but uniquely relevant: overt syllables with dense ECoG over speech motor cortex | 4 subjects; narrow task | Ingest immediately; use as clean external motor-speech benchmark |
| 10 | **Verwoert / Herff 2025 whole-brain sEEG** (Comms Biology 2025) | **15 sEEG pts**, Dutch speech production, audio-aligned | sEEG, Dutch word production with 48 kHz mic | Stim + vocal onsets, audio aligned | EBRAINS, DUA | **Strict upgrade over Verwoert 2022** | Same lab, larger cohort, same clean "onset + audio + sEEG" contract | DUA required (not instant download); Dutch labels differ from PS | Request EBRAINS DUA; stage after 2022 release |
| 11 | **Flinker full speech ECoG cohort** | 48 pts, overt speech across multiple tasks | ECoG | Multi-task speech, onsets present | **Request / collaboration** | **Highest-value external production corpus** | Best external match to target problem; direct modality + task match | Not openly posted; need DUA; metadata harmonization effort | PI-level email (Greg/Flinker); ask for electrode localizations, stimulus timing, audio |
| 12 | **Chang / Makin / Moses sentence-level speech corpus** | Multi-patient, sentence-level | ECoG, overt continuous speech | Sentence-level production onsets | **Request / collaboration** | **Sequence-decoder pretraining + evaluation** | Most relevant external corpus for genuine sequence decoding | Access friction; heterogeneous preprocessing | Contact corresponding authors; request deidentified iEEG + audio + transcripts + coordinates |
| 13 | **Littlejohn / Cho 2025 UCSF streaming brain-to-voice** (Nat Neurosci 2025) | UCSF BRAVO subset, ECoG; n and hours not disclosed | ECoG | Attempted/overt streaming speech | **Harvard Dataverse DOI:10.7910/DVN/8TQKC8, request-based/gated** | **First Chang-lab slice with any deposit** | First UCSF/Chang release with any kind of repository; all other BRAVO data is pure-request | Gated (request-based, not open download); still subject to UCSF IRB | Request access via Harvard Dataverse; low-risk first UCSF ask |
| 14 | **Goldstein / Zada / Flinker NYU 24/7 ECoG conversations** (Nat Hum Behav 2025) | ~100 h / 4 pts, dense-sampling 24/7 | ECoG | Continuous conversational speech in hospital | **Private** | Chronic conversational speech reference | Closest thing to chronic naturalistic speech in field-potential mode | No public deposit found | Contact Adeen Flinker (NYU) |
| 15 | **UPenn RAM / delayed free recall** (ds004789 + RAM public archive) | **251 pts, ~1,100+ sessions, >1,000 h**; large public release (~871 GB on NEMAR) | ECoG + sEEG mixed | Delayed free recall, categorized recall, paired associates; **vocalization timestamps on recalled words** + stimulus onsets | Open / request (consortium) | **Largest open human iEEG ecosystem with any kind of vocal onsets** | Unique scale for SSL + vocal-onset learning; memory-driven vocal responses | Behavior is memory-driven, not controlled reading; brief isolated vocalizations; fragmented metadata per site | Use as optional generic pretraining + vocal-onset SSL auxiliary; audit harmonization cost before investing heavily |
| 16 | **UPenn categorized free recall** (ds004809) | **258 pts, 140 ECoG + 22 sEEG ch**, 768.7 GB | ECoG + sEEG mixed | Study word-list → arithmetic distractor → free vocal recall; recall-vocalization onsets present | Open (OpenNeuro) | **RAM-family vocal-onset SSL auxiliary** | Largest open iEEG free-recall corpus with vocal recall events; companion to RAM (row 15) | Likely sub-ID overlap with RAM ds004789 — verify independence before pooling; memory-driven brief vocalizations, not controlled production | Audit sub-ID overlap vs ds004789; treat as RAM-family member until disambiguated |
| 17 | **Angrick chronic ALS ECoG** ★ (Sci Rep 2024, Crone lab NCT03567213) | **1 pt, 13 sessions over ~7 months**, SRT (780 trials) + KRT | Chronically implanted ECoG (Johns Hopkins) | Stim + response onsets + time-synced microphone audio | **Open (GitHub `cronelab/delayed-speech-synthesis`)** | **The only open chronic human speech field-potential dataset that exists** | Uniquely load-bearing for the cross-sensor/chronic stability story; n=1 but longitudinal; code + data both released | Single participant; low external validity; chronic stability claims from 1 pt are suggestive not conclusive | Phase 2+ chronic-stability reference; not Phase-1.5 priority |
| 18 | **Chronic RNS iEEG** (Frontiers Neurosci. 2026, doi.org/10.3389/fnins.2026.1815732) | Multi-patient chronic recordings via NeuroPace RNS device (scale per paper) | Chronic intracranial via RNS device electrodes (depth + strip) | RNS event-triggered snippets only; **no speech labels, no audio** | Open (data release per paper) | **Second open chronic field-potential reference** (Phase 2+ stability complement to Angrick row 17) | Adds a non-Utah, non-speech chronic reference; weakens "only Angrick" monopoly for open chronic field-potential data; multi-patient | Event-triggered snippets, not continuous; clinical seizure-detection focus, not speech; tiny windows per device trigger | Audit data-release terms + sample lengths; treat as Phase 2+ chronic-stability complement, not primary corpus |
| 19 | **Kanno 2025 Auditory Naming** (OpenNeuro ds005545) | **106 iEEG**, stim + response event codes | iEEG (ECoG + sEEG), auditory naming | Stim + response onsets | Open CC0 | **Independent open naming release (verify rebadge first)** | Second large open iEEG naming cohort; CC0 license; stackable with ds006234 IF independent | **Verify whether this is a sibling release, a rebadge, or an independent cohort** before pooling to avoid patient-duplication bias | Cross-reference sub-IDs against ds006234 + ds006910; confirm provenance before any use |
| 20 | **Auditory Naming EC ds006910** (Asano group, third release) | **121 pts**, MNI-305 coords, 128-ch, 6 sessions | iEEG (ECoG + sEEG mixed) | Auditory naming, stim + response onsets (event codes 401 / 402 / 501) | Open (OpenNeuro) | **Third Asano-naming candidate — pool only after rebadge audit** | Stacks with ds006234 (row 4) + ds005545 (row 19) IF independent; could be the canonical or latest release in the family | **Three releases now in the Asano auditory-naming family** (ds006234, ds005545, ds006910) — patient overlap unverified; pooling without sub-ID audit risks triple-duplication | Cross-reference sub-IDs across all three releases before any use |
| 21 | **BrainStratify** (Zheng 2025, arXiv 2505.20480) | 2 sEEG + 1 epidural ECoG release | sEEG + **epidural ECoG** | Vocal production + perception, trial onsets | Open (GitHub `liulab-repository/BrainStratify`) | **Only open epidural-ECoG slice for cross-sensor validation** | Epidural ECoG is the field-potential modality closest to future WIMAGINE-style chronic implants | Very small; Chinese-lab cohort | Ingest for epidural slice only |
| 22 | **VocalMind** (He et al. 2025 Sci Data; CUHK-SZ, Haizhou Li) | **3 sEEG pts, ~67 min/pt**, Mandarin | sEEG | **Vocalized + mimed + imagined** speech, word + sentence | Open (Zenodo 14696348) | **Only open dataset with imagined-speech mode** | Unique task: the "imagined" condition is rare in field-potential data; reserve for Phase 2+ inner-speech ablations | Very small (3 pts) | Ingest for imagined-speech sanity checks; do not let small n drive architecture |
| 23 | **NeuroListen** (Zhang 2025 NeurIPS D&B) | **5 sEEG, >10 h**, natural-speech listening + audio | sEEG | Natural-speech listening, audio-aligned, semantic categories | Open (Zenodo 17426506 + GitHub `NeuroListen/NeuroListen`) | **Open sEEG listening + audio SSL** | Clean audio-aligned open sEEG listening release | Perception only; small n | Ingest for audio-aligned SSL experiments |
| 24 | **Brain Treebank** (Hu/Feng et al. 2024) | **10 pts, ~43.5 h**, naturalistic movie | sEEG | Passive listening to movies; transcript + audio aligned | Open | **iEEG-FM shared eval benchmark** (BrainBERT / PopT / Charmander / BaRISTA / MVPFormer all use it) | Field-comparable benchmark; audio-text alignment; useful for temporal + language SSL priors | Perception, not production; invariances may not transfer to articulation | Use for JEPA pretraining + cross-sensor eval; not for motor-speech model selection |
| 25 | **SWEC iEEG** (Carzaniga/IBM + Schindler 2025; HuggingFace NeuroTec) | **68 pts, 9,328 h** (verified) | ECoG + sEEG strips/grids/depth | Long-term clinical monitoring; ictal annotations only; **no speech labels, no audio** | Open CDLA-Permissive (HuggingFace) | **Massive generic iEEG foundation-model pretraining** | Largest field-potential open iEEG by hours; enables testing whether scale alone improves temporal robustness | Weak task match; seizure-dominated objective; patient mostly non-speaking for the 9,328 h | Treat as dedicated generic pretraining stage; evaluate only through transfer |
| 26 | **Omni-iEEG** (arXiv **2502.16072**, 2026) | **302 pts, ~178 h pre-surgical**, BIDS-harmonized, 36K events | ECoG + sEEG | Pre-surgical iEEG; task + rest segments | Open (recent release) | **Next-wave cross-patient harmonization corpus** | Largest open iEEG cohort by patient count (2.5× UPenn RAM); BIDS-harmonized day one; mixed ECoG+sEEG enables modality-agnostic FMs | Too recent to be cited in any published iEEG FM; task-heterogeneous; metadata untested | Audit the release; prioritize if BIDS harmonization is as clean as claimed |
| 27 | **Cogitate iEEG** (Ferrante et al. *Sci Data* 2025) | **38 pts across 3 centers**, 4,771 electrodes (1,238 ECoG + 3,533 sEEG), 512–2048 Hz | ECoG + sEEG mixed | Go/No-Go visual detection (faces / objects / letters / false fonts × 3 orientations × 3 durations) + **finger-localizer motor task** | Open (XNAT account + BIDS bundle; CC license) | **Multi-center harmonized iEEG SSL fuel + motor-localizer reference** | Largest open multi-center iEEG with a unified protocol across 3 sites day one; finger-motor localizer is the only speech-motor-adjacent component; BIDS + extensive metadata | Visual task is not speech; finger localizer is hand not orofacial; no published v14-era iEEG-FM has used it yet; XNAT account-creation gate | Ingest via Cogitate GitHub release; treat as Tier-C generic-SSL fuel + motor-localizer parity check |
| 28 | **Podcast ECoG** (ds005574; Zada et al. 2025) | 9 pts, 30 min story, ~4.5 h total, 1,330 elec | ECoG + sEEG mixed | Naturalistic story listening | Audio-aligned; high-gamma derivatives; LLM features | Open CC0 | **Audio-text-aligned SSL** | Clean audio-text alignment for cross-modal auxiliaries | Perception only; small scale; biased toward temporal / language | Add as second-stage SSL after Brain Treebank for audio-text experiments |
| 29 | **sEEG Passive listening to natural speech** (ds004703) | 10 pts, 2 sessions | sEEG | Passive conversational speech listening | Stim + audio onsets | Open (non-commercial) | **sEEG temporal SSL** | Good open sEEG listening with clear task structure | Non-commercial license restriction; passive listening only | Resolve license terms internally before major investment |
| 30 | **Naturalistic film iEEG** (ds003688) | 51 pts | iEEG / HD-ECoG / sEEG mixed | Naturalistic audiovisual film | Stim onsets | Open | **Heterogeneity robustness pretraining** | Large open cohort; stress-tests modality and coverage heterogeneity | Weakest task match to speech production | Use only after speech/naming corpora are in place |
| 31 | **Kucewicz pupillometry + memory** (Sci Data 2022) | 10 pts, 4 memory tasks | iEEG | Memory tasks with correct/incorrect vocalization timing | Response + vocalization onsets | Open BIDS | **Small memory-task vocal-onset set** | Nicely annotated; BIDS-clean | Small; memory task, not speech | Include as complement to UPenn RAM; not primary |
| 32 | **HUP iEEG Epilepsy Dataset** (ds004100) | **58 pts**, ECoG + sEEG, interictal + ictal, MNI152 localizations | ECoG + sEEG | Epilepsy clinical monitoring | **No speech onsets** | Open | **Localization / metadata reference** | Clean MNI152 localizations across 58 pts; useful as a localization prior donor | No speech labels | Use for localization/coverage audits, not training |
| 33 | **MNI Open iEEG Atlas** (Frauscher 2018 + extensions) | 106–110 pts rest; 91 pts sleep; normative (healthy-tissue only) | sEEG + ECoG | Rest + sleep | **No speech onsets** | Open | **Healthy-tissue iEEG atlas** | Best-localized healthy-tissue iEEG atlas in the field; useful as anatomical prior | No speech; normative (excludes epileptogenic tissue) | Use for atlas/spatial priors |
| 34 | **Mayo Clinic + FNUSA iEEG** (multi-center seizure corpora) | Hundreds of hours each, multi-patient | sEEG | Epilepsy / seizure detection | Seizure annotations only | Open (Mayo + FNUSA) | **External seizure-detection transfer benchmarks** | Used by BrainWave / MVPFormer as OOD seizure eval; cross-hospital robustness anchors | Seizure-only; weak speech relevance | Low priority; only if reproducing iEEG-FM seizure benchmarks |
| 35 | **TUEG — Temple University EEG Corpus** (reference only) | **27,063 h / 14,987 indiv / 1,643 GB** | **Scalp EEG** (not iEEG) | Mixed clinical | — | Open (TUH registration) | **Joint EEG+iEEG FM reference scale** | Sets the baseline scale every joint FM (BrainWave, FoME, LaBraM, Neuro-GPT, NeuroLM) trains on | Scalp EEG, not iEEG | Do not acquire for v14; listed only as reference scale |

## Paper-only watch list (cohorts documented in 2024-2026 papers, data not yet publicly released)

These are the highest-value **releases to monitor or email authors about**. Many will become the next open datasets worth ingesting.

| Dataset | First author / venue | Scale | Key distinguisher | Contact |
|---|---|---|---|---|
| **Evanson "Minutes to Days"** (arXiv 2512.15830) | Evanson / King et al. 2025 (Meta FAIR + APHP) | **3 sEEG × ~168 h/subj** week-long + 120 min audiobook | **Closest to open chronic field-potential + audio dataset** | Jean-Rémi King |
| **Evanson "Emergence of Language"** (arXiv 2512.05718) | Evanson 2025 | **46 pts sEEG + ECoG**, audiobook ("Little Prince"), 7,400 elec | Large multi-site sEEG+ECoG listening with phoneme/word onsets | French epilepsy centers |
| **WIMAGINE epidural ECoG** (arXiv 2512.04618) | Ben Ticha 2025 (Clinatec/Yvert) | 1 epidural WIMAGINE + 1 subdural pt | **First epidural speech BCI pilot** — direct field-potential chronic parallel to Utah BCIs | Tetiana Aksenova / Tristan Tetrel (Clinatec) |
| **MIBRAIN internal Mandarin phoneme sEEG** (arXiv 2506.12055) | Wu et al. 2025 (Westlake/ZJU) | **11 sEEG pts**, Mandarin phoneme articulation | Methodologically closest neighbor to v14 (per-subject input tokenizer) | Jie Yang / Mohamad Sawan |
| **HUST-MIND / SACM** (arXiv 2505.19652) | Wang 2025 | 8 sEEG pts, 48 Mandarin words, audio-synced | Request-only Chinese-lab release (mentioned at GitHub SACM) | GitHub `WangHongbinary/SACM` |
| **Flinker Speech-Arrest ESM ECoG** (arXiv 2509.08703) | Emami / Flinker 2025 | 16 ECoG pts, ESM + speech task | Subset of the Flinker cohort with ESM labels | Adeen Flinker (NYU) |
| **Neuro2Semantic** (arXiv 2506.00381) | Shams et al. 2025 (Mesgarani / Flinker) | Cohort size undisclosed; >30 min perceived speech per pt | Continuous-language iEEG with aligned transcripts | Nima Mesgarani (Columbia) |
| **Lexical-tone sEEG corpus** (arXiv 2410.12866) | Wu et al. 2024/25 | Multi-patient Mandarin sEEG tonal production | Only open-adjacent tonal-speech release to watch | Authors on arXiv |

## Gated chronic speech BCI cohorts (NOT publicly released)

All Utah-array cohorts (BrainGate T12/T15, Kunz inner-speech) excluded per scope. The remaining gated **field-potential** chronic speech cohorts are:

- **UCSF Chang lab BRAVO trials** (all under NCT03698149):
  - **BRAVO-1 "Pancho"** (Moses 2021 NEJM)
  - **BRAVO-2 "Ann Johnson"** (Metzger 2023 Nature, avatar speech synthesis)
  - **BRAVO-3** (Silva 2024 Nat Biomed Eng, bilingual)
  - Uniform data availability: "restricted access per clinical-trial protocol; by reasonable request to E.F.C. (edward.chang@ucsf.edu)"
  - Littlejohn / Cho 2025 (row 13 above) is the first slice with any deposit (Harvard Dataverse, gated).
- **Angrick JHU follow-up** (J Neural Eng 2025, PMID 40972658) — promises release "upon publication"; repository not live as of 2026-04-25.
- **BrainGate chronic cohort longitudinal analysis** (Willett et al. medRxiv 2025.07.02.25330310) — 14 participants, 20 arrays, 2,319 sessions, up to 7.6 y. *Array-longevity analysis* is public; full neural data not released as a bundle.

**Bottom line**: for open chronic **field-potential** *speech* data, **Angrick 2024 (row 17) is effectively the only option that exists**. The closest non-speech chronic field-potential reference is **Chronic RNS iEEG (row 18)** — event-triggered snippets, no speech labels, but multi-patient (Phase 2+ stability complement). All other chronic *speech* data is Utah (out of scope), gated (UCSF BRAVO, Littlejohn), paper-only (Evanson, WIMAGINE), or private (NYU).

## Best SSL Sources for `v14`

Ordered by **expected transfer to patient-invariant articulatory decoding**, not raw hours.

1. **Internal Cogan PS + lex uECoG** — actual task; same modality, same preprocessing pipeline.
2. **Internal Cogan sEEG D-cohort** — best modality + task match for the cross-sensor expansion; **180.59 h / 87 D-pts**, 122 ready (full Stage-3 audit complete 2026-04-24).
3. **AJILE12** — largest open ECoG; direct modality match; real speech-on-cortex via "talk" ADL; Charmander-validated recipe.
4. **Du-IN** — largest open sEEG production benchmark; vSMC + STG targeting.
5. **Brain Treebank** — iEEG-FM shared eval benchmark; audio-text alignment at 43.5 h.
6. **UPenn RAM** — 1,000+ h with vocal onsets; large human iEEG ecosystem.
7. **SWEC** — largest field-potential generic pretraining fuel.

If ranked by **raw generic pretraining horsepower** instead, the order flips to:

1. **SWEC** (9,328 h)
2. **AJILE12** (~1,280 h)
3. **UPenn RAM** (1,000+ h)
4. **Internal Cogan sEEG D-cohort** (~180.59 h, 87 D-pts)
5. **Omni-iEEG** (~178 h)
6. **Brain Treebank** (~43.5 h)

The staged-curriculum ordering matters: biggest corpus is not automatically best corpus for `v14`.

## Recommended Priority Tiers

### Tier A — Acquire / ingest first

Most likely to move `v14` materially in the next 1–2 months. Filter: **open or already-local + in-modality production + meaningful scale + verified availability** (no DUA friction, no rebadge-risk, no Phase-2+ chronic). Ordered by in-modality strength × scale.

1. Internal Cogan PS + lex uECoG corpus (already in pipeline; primary working corpus)
2. Internal Cogan sEEG D-cohort (cross-sensor expansion; 180.59 h / 87 D-pts, 122 ready, audit-complete 2026-04-24)
3. AJILE12 (open ECoG SSL, 1,280 h, "talk" speech-on-cortex)
4. Auditory Naming EC (ds006234, 119+ pts open production-adjacent)
5. Picture Naming (ds006233, 108 pts companion)
6. Visual Naming EC (~110 pts, third Asano companion)
7. Du-IN (12 sEEG production benchmark, Mandarin)
8. Verwoert 2022 SingleWordProductionDutch (ds003194, 10 sEEG, cleanest "onset + audio + sEEG")
9. Bouchard-Chang DANDI 000019 (4 HD ECoG syllables, held-out articulatory eval)

### Tier B — Highest-value request / DUA targets

1. Flinker full speech ECoG cohort
2. Chang / Makin / Moses sentence-level speech corpus
3. Verwoert / Herff 2025 whole-brain sEEG (EBRAINS DUA — 15-pt upgrade over 2022)
4. Littlejohn / Cho 2025 UCSF (Harvard Dataverse gated)
5. UPenn RAM (memory consortium)

### Tier C — SSL scale-up + niche after the production core is in place

1. Brain Treebank (field-comparable eval)
2. SWEC iEEG (9,328 h generic pretraining)
3. Omni-iEEG (302 pts, BIDS-harmonized)
4. **Cogitate iEEG** (Ferrante 2025; 38 pts, 3-center harmonized BIDS + finger-localizer motor)
5. NeuroListen (open sEEG listening + audio)
6. Podcast ECoG (ds005574)
7. sEEG Passive listening (ds004703)
8. Naturalistic film iEEG (ds003688)
9. BrainStratify (epidural ECoG slice)
10. VocalMind (imagined-speech mode)
11. **UPenn categorized free recall (ds004809)** — RAM-family vocal-onset; audit sub-ID overlap vs ds004789 first
12. **Angrick chronic ALS ECoG** ★ (only open chronic field-potential **speech** data; Phase 2+ stability ablations)
13. **Chronic RNS iEEG** (Frontiers 2026; non-speech chronic field-potential reference; Phase 2+ complement to Angrick)
14. **Kanno 2025 Auditory Naming (ds005545)** + **Auditory Naming EC ds006910** (verify cross-release sub-ID overlap vs ds006234 *before* any use)

### Tier D — Reference / metadata only

1. HUP iEEG (ds004100) — localization prior donor
2. MNI Open iEEG Atlas — healthy-tissue iEEG atlas
3. Mayo + FNUSA — OOD seizure benchmarks
4. Kucewicz memory (Sci Data 2022)
5. TUEG — scalp reference scale only

## Practical Acquisition Strategy

### Stage 1 — Build the production / naming core

Target outcome: a pooled corpus behaviorally close enough to speech-motor decoding to test cross-patient transfer honestly.

Acquire / ingest: Internal Cogan PS+lex (already in pipeline), Internal Cogan sEEG D-cohort (audit-complete 2026-04-24, 122 D-pts ready), AJILE12, Auditory Naming (ds006234), Picture Naming (ds006233), Visual Naming EC, Du-IN, Verwoert 2022 (ds003194), Bouchard-Chang (DANDI 000019).

Use for: response-locked HGA modeling, supervised pretraining on naming / production-adjacent tasks, small-scale JEPA in speech-relevant regime.

### Stage 2 — Add high-value private + gated speech cohorts

Acquire: Flinker full cohort, Chang sentence-level corpus, Verwoert / Herff 2025 (EBRAINS DUA), Littlejohn / Cho UCSF (Harvard Dataverse).

Use for: speech-motor pretraining, sequence-decoder development, external held-out evaluation.

### Stage 3 — Add chronic + naturalistic SSL scale

Acquire: UPenn RAM, Brain Treebank, SWEC, Omni-iEEG, NeuroListen, Podcast ECoG, ds004703, ds003688, BrainStratify, VocalMind, Angrick chronic ALS ECoG, Kanno 2025 ds005545 (after rebadge verification).

Use for: JEPA / masked prediction pretraining, generic heterogeneous iEEG foundation-model pretraining, audio/text auxiliary tasks, chronic-stability ablations (Angrick), robustness under mixed coverage.

## What Not To Do

- Do **not** blindly pool production, naming, listening, and movie datasets into one undifferentiated corpus.
- Do **not** mix SWEC-style seizure pretraining into the speech stack without a staged transfer plan.
- Do **not** mistake UPenn RAM scale for speech relevance.
- Do **not** let passive-listening corpora dominate model selection for an articulatory decoding paper.
- Do **not** pool Kanno 2025 ds005545 and Auditory Naming EC ds006234 before verifying sub-ID overlap.
- Do **not** assume more subjects means more useful signal; target match matters more than raw count.
- Do **not** drop localization quality from ranking; datasets without usable coordinates are much less valuable for `v14`.
- Do **not** include Utah intracortical datasets in cross-sensor speech FM comparisons — different signal modality.

## Short Version

If only a few datasets can be tackled soon, the highest-leverage order is:

1. Internal Cogan PS + lex uECoG corpus (our task; already in pipeline)
2. Internal Cogan sEEG D-cohort (cross-sensor expansion; **180.59 h / 87 D-pts, 122 ready**)
3. AJILE12 (open ECoG SSL + speech-on-cortex, 1,280 h)
4. Auditory Naming EC (ds006234, 119+ pts)
5. Picture Naming (ds006233, 108 pts)
6. Visual Naming EC (~110 pts)
7. Du-IN (open sEEG production benchmark, 12 pts)
8. Verwoert 2022 SingleWordProductionDutch (ds003194, cleanest production)
9. Bouchard-Chang DANDI 000019 (held-out articulatory eval)
10. Flinker full cohort (request)
11. Chang sentence-level corpus (request)
12. Verwoert / Herff 2025 (EBRAINS DUA, 15-pt upgrade over 2022)
13. Brain Treebank (field-comparable eval)
14. UPenn RAM
15. SWEC iEEG
16. **Angrick chronic ALS ECoG** (the only open chronic field-potential speech dataset; Phase 2+)

That ordering balances immediate feasibility, closeness to articulatory decoding, usefulness for SSL, and value for proving the cross-patient + cross-sensor hypothesis.

## iEEG Foundation Models & Their Pretraining Corpora

Audit as of 2026-04-25. Shows which corpora each iEEG-FM uses, so the matrix tracks what "the field trains on" alongside what we could train on.

| Model | Year / Venue | Pretraining data | Total hours | Patients | Electrode type | In our matrix? |
|---|---|---|---:|---:|---|---|
| **BrainBERT** (Wang et al.) | ICLR 2023 | Internal 10-subj sEEG (Brain Treebank subset) | ~43.7 h | 10 | sEEG | Via Brain Treebank |
| **Brant** (Zhang et al.) | NeurIPS 2023 | Internal ZJU SAHZU sEEG | **2,528 h** (1.01 TB) | (large, undisclosed) | sEEG | N (internal) |
| **PopT / Population Transformer** (Chau et al.) | ICLR 2025 | Same BrainBERT corpus | ~43.7 h | 10 | sEEG | Via Brain Treebank |
| **Seegnificant** (Mentzelopoulos et al.) | NeurIPS 2024 | Internal Penn sEEG RT task | ~100+ electrode-hours | 21 | sEEG | N (internal) |
| **Charmander** (Mahato et al.) | NeurIPS-W 2025 | **AJILE12** + **Brain Treebank** | ~1,280 h + 43.5 h | 8 + 10 | ECoG + sEEG | Y |
| **MVPFormer** (Carzaniga/IBM) | ICLR 2026 | **SWEC iEEG** (9,328 h verified) | 9,328 h | 68 | sEEG (long-term) | Y (row 25) |
| **BaRISTA** (Oganesian/Shanechi) | NeurIPS 2025 | **Brain Treebank** only | ~43.5 h | 10 | sEEG | Y (row 24) |
| **MIBRAIN** (Wu et al., Westlake/ZJU, arXiv 2506.12055) | 2025 | Internal Mandarin sEEG | (not quoted) | 11 | sEEG | N (internal, closest methodological neighbor to v14) |
| **Du-IN / Du-IN2** (Zheng et al., arXiv 2405.11459) | NeurIPS 2024 | Own 12-pt Mandarin production sEEG | (not quoted) | 12 | sEEG | Y (row 3) |
| **BrainWave / Brant-2** (Yuan/Zhang) | CoRR 2024, updated 2025 | TUEG + multi-source EEG+iEEG (15 datasets) | **40,907 h (13.79 TB)** | ~15,997 individuals | **joint scalp EEG + sEEG** | Partial (TUEG as ref only) |
| **FoME** (Shi et al.) | arXiv 2409.12454, 2024 | Joint scalp EEG + iEEG | ~26,000 h | (undisclosed) | joint EEG + iEEG | N (joint; TUEG as ref) |
| **DIVER-1** (Mahato et al.) | ICLR 2026, arXiv 2512.19097 | **AJILE12 + internal 25-pt ECoG/sEEG** (4,028 h, 227k channel-hr) + **54k h scalp EEG** | **5,310 h iEEG + 54k h EEG** (352k iEEG channel-hr; 1.6M total channel-hr) | 37 iEEG + ~17.7k EEG | joint ECoG + sEEG + scalp EEG | Y (AJILE12 row 3; internal undisclosed) |

### Key convergence signals

- **Brain Treebank is the iEEG-FM shared benchmark.** 5 of 10 FMs audited (BrainBERT, PopT, Charmander, BaRISTA, MVPFormer-eval) pretrain on it or use it as headline eval. Any v14 SSL work should evaluate on Brain Treebank for apples-to-apples comparison.
- **AJILE12 is the largest open ECoG pretraining corpus.** Charmander validated the masked-channel reconstruction recipe on 8 of 12 pts. That recipe + corpus is the most direct template for v14's Phase-1.5 SSL.
- **Scale is not the bottleneck at the FM stage; data curation is.** Charmander's scaling table reports 8M ≈ 33M ≈ 142M params at 1,000+ h — capacity wasn't the gap. Implication: at 2.83 h we should stay small (d=32, depth=3) and invest in more data before more capacity.
- **Joint EEG+iEEG FMs (BrainWave, FoME)** dominate raw scale but use scalp EEG for most of it — reference scale only.
- **Du-IN is the only published open sEEG production-focused FM** — methodologically closest to v14 among open-release FMs.

### Per-patient conditioning — what iEEG FMs actually do

Audit 2026-04-19 (for the v14 per-patient-default question):

- **POYO / POYO+ / Charmander**: additive learned per-unit (neuron/channel) embedding; optional additive per-session embedding. No multiplicative affine, no softsign.
- **Seegnificant**: **shared trunk + per-subject shallow MLP readout head** (2,081 params × 21 subjects). Input-side: additive 3D MNI RBF PE only. **No γ/β scaling, no softsign, no input-side patient conditioning.**
- **BrainBERT / PopT**: no explicit per-patient mechanism beyond session-wise training.
- **MIBRAIN**: per-subject full 1D CNN tokenizer (not diagonal+softsign).
- **Du-IN**: per-subject linear projection; no diagonal+softsign.

Implication: **"per-patient diagonal affine + softsign" has no direct precedent among the iEEG FMs we audited.** The closest in-modality patterns are (a) additive per-channel / per-session embedding, (b) shared trunk + per-subject readout head, (c) per-subject parametric input tokenizer. Our proposed form should be reduced to one of these or the specific citation we were relying on needs direct verification before defaulting.

## Useful Links

### Production / naming / speech corpora

- AJILE12 paper: <https://www.nature.com/articles/s41597-022-01280-y>
- AJILE12 on DANDI (`000055`): <https://dandiarchive.org/dandiset/000055>
- Cogitate iEEG paper (Ferrante 2025 *Sci Data*): <https://www.nature.com/articles/s41597-025-04833-z> ; code + release: <https://github.com/Cogitate-consortium/iEEG-data-release>
- UPenn categorized free recall (`ds004809`): <https://nemar.org/dataexplorer/detail?dataset_id=ds004809>
- Auditory Naming EC (`ds006910`, Asano group third release): <https://openneuro.org/datasets/ds006910>
- Du-IN (Zheng 2024 NeurIPS, arXiv): <https://arxiv.org/abs/2405.11459>
- Auditory Naming EC (OpenNeuro ds006234): <https://openneuro.org/datasets/ds006234>
- Picture Naming (`ds006233`): <https://openneuro.org/datasets/ds006233>
- Kanno 2025 Auditory Naming (ds005545): <https://openneuro.org/datasets/ds005545>
- Verwoert / Herff 2025 whole-brain sEEG (Comms Biology): <https://doi.org/10.1038/s42003-025-07862-x>
- Verwoert 2022 SingleWordProductionDutch paper: <https://www.nature.com/articles/s41597-022-01542-9>
- Verwoert 2022 OSF: <https://osf.io/nrgx6/>
- Verwoert 2022 OpenNeuro: <https://openneuro.org/datasets/ds003194>
- Bouchard-Chang DANDI 000019: <https://dandiarchive.org/dandiset/000019>
- Flinker cohort paper: <https://www.nature.com/articles/s42256-024-00824-8>
- Flinker public single-subject release: <https://data.mendeley.com/datasets/fp4bv9gtwk/2>
- Chang sentence-level speech paper: <https://www.nature.com/articles/s41593-020-0608-8>
- Littlejohn / Cho 2025 UCSF Harvard Dataverse (gated): <https://doi.org/10.7910/DVN/8TQKC8>
- Angrick chronic ALS ECoG (Sci Rep 2024): <https://www.nature.com/articles/s41598-024-60277-2> ; code + data: <https://github.com/cronelab/delayed-speech-synthesis>
- BrainStratify (Zheng 2025): <https://arxiv.org/abs/2505.20480> ; code: <https://github.com/liulab-repository/BrainStratify>
- VocalMind: <https://doi.org/10.5281/zenodo.14696348>

### Listening / SSL corpora

- Brain Treebank: <https://braintreebank.dev/>
- NeuroListen: <https://zenodo.org/records/17426506> ; <https://github.com/NeuroListen/NeuroListen>
- Podcast ECoG (`ds005574`): <https://openneuro.org/datasets/ds005574>
- sEEG Passive listening (`ds004703`): <https://openneuro.org/datasets/ds004703>
- Naturalistic film iEEG (`ds003688`): <https://openneuro.org/datasets/ds003688>

### Generic / localization / reference

- SWEC / MVPFormer paper: <https://openreview.net/forum?id=5M1YOW3bRq>
- SWEC dataset release (HuggingFace NeuroTec): <https://huggingface.co/datasets/NeuroTec/SWEC_iEEG_Dataset>
- UPenn RAM public archive: <https://memory.psych.upenn.edu/RAM_Public_Data>
- Delayed free recall OpenNeuro (`ds004789`): <https://openneuro.org/datasets/ds004789>
- Delayed free recall NEMAR: <https://nemar-dev.ucsd.edu/dataexplorer/detail?dataset_id=ds004789>
- Kucewicz pupillometry + memory (Sci Data 2022): <https://www.nature.com/articles/s41597-022-01628-4>
- HUP iEEG Epilepsy (`ds004100`): <https://openneuro.org/datasets/ds004100>
- MNI Open iEEG Atlas: <https://ieegatlas.loris.ca/>
- Omni-iEEG (arXiv 2502.16072): <https://arxiv.org/abs/2502.16072>
- Mayo Clinic iEEG portal: <https://msel.mayo.edu/data.html>
- FNUSA iEEG release: <https://www.fnusa-icrc.org/research/data-sharing>
- TUEG (Temple University EEG Corpus): <https://isip.piconepress.com/projects/tuh_eeg/>
- Chronic RNS iEEG (Frontiers Neurosci. 2026): <https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1815732/abstract>

### Paper-only watch list

- Evanson "Minutes to Days" (arXiv 2512.15830): <https://arxiv.org/abs/2512.15830>
- Evanson "Emergence of Language" (arXiv 2512.05718): <https://arxiv.org/abs/2512.05718>
- WIMAGINE epidural ECoG (arXiv 2512.04618): <https://arxiv.org/abs/2512.04618>
- MIBRAIN (arXiv 2506.12055): <https://arxiv.org/abs/2506.12055>
- HUST-MIND / SACM (arXiv 2505.19652): <https://arxiv.org/abs/2505.19652> ; <https://github.com/WangHongbinary/SACM>
- Flinker Speech-Arrest ESM (arXiv 2509.08703): <https://arxiv.org/abs/2509.08703>
- Neuro2Semantic (arXiv 2506.00381): <https://arxiv.org/abs/2506.00381>
- Lexical-tone sEEG Wu 2024/25 (arXiv 2410.12866): <https://arxiv.org/abs/2410.12866>

### iEEG foundation-model papers

- BrainBERT (Wang 2023): <https://arxiv.org/abs/2302.14367>
- Brant (Zhang, NeurIPS 2023): <https://papers.neurips.cc/paper_files/paper/2023/file/535915d26859036410b0533804cee788-Paper-Conference.pdf>
- PopT / Population Transformer (Chau, ICLR 2025): <https://arxiv.org/abs/2406.03044>
- Seegnificant (Mentzelopoulos, NeurIPS 2024): <https://arxiv.org/abs/2411.10458>
- Charmander (Mahato, NeurIPS-W 2025): <https://openreview.net/pdf?id=CdP8Y4K4fz>
- MVPFormer (Carzaniga/IBM, ICLR 2026): <https://arxiv.org/abs/2506.20354>
- BaRISTA (Oganesian/Shanechi, NeurIPS 2025): <https://arxiv.org/abs/2512.12135>
- BrainWave / Brant-2: <https://arxiv.org/abs/2402.10251>
- FoME (Shi 2024): <https://arxiv.org/abs/2409.12454>
- EEG foundation models review (Li 2025): <https://arxiv.org/abs/2507.11783>
- Neuroprobe benchmark (2025): <https://arxiv.org/abs/2509.21671>
- DIVER-1 (Mahato et al., ICLR 2026): <https://arxiv.org/abs/2512.19097>
