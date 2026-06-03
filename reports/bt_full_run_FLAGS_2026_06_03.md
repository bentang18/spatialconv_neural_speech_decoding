# BT full-live-run — FLAGS for Ben (autonomous push 2026-06-03)

Surprises / inconsistencies surfaced while driving toward a live BT P1→P4 run.
Read top-to-bottom; newest flags appended.

---

## FLAG 1 — "Trial exclusion" signals are buggy / mis-scoped. Do NOT exclude trials on them. (the "exclusion thing")

You asked what the trial-exclusion thing means. A 4-agent adversarial audit (2026-06-03)
**refuted** my own earlier characterization. Findings, all code-grounded:

**A. Coverage % is wrong for restart trials (a real bug).**
`cut_coverage.py` computes `coverage_pct = last_movie_time_s / film_duration`. But
`pause_audit.py:56-60` sets `last_movie_time_s` = the **last `end`-row** movie-time. When a
recording session RESTARTED (multiple beginning/end segments), that `end` row is an
*intermediate* boundary, not the trial's true end:
- **sub_1_trial002**: reported **0.31 %** (24.4 s) — but triggers actually run to ~6364 s →
  **~81 % real coverage**. The "dead 0.3 % trial" is a mirage (single `end` row at 24 s, then
  63 k triggers continue after a restart).
- **sub_6_trial004**: reported 31.5 % → actually ~50 %.
- **sub_4_trial002 (44 %)** and **sub_6_trial000 (43 %)**: single-segment, genuinely partial.

**B. "sub_2_trial004 = 99.9 % droppable" is a granularity confusion.**
The 82582/82672 counts **photodiode TRIGGER rows** in `subject_timings.csv`, NOT words. The
trial's actual **words (7,845)** live in a separate `words_df` that has **no `trig_type` column**.
Word clip slicing uses `est_idx` directly and does **not** depend on trigger `trig_type`. And the
droppable/`est` analysis is consumed by **nothing** in the train/eval path (only by
`cut_coverage`'s buggy `last_movie_time_s`). So "nearly all words suspect" is unsupported.

**C. 4 no-`trig_type` trials (sub_1_t000/001/002, sub_10_t001): CONFIRMED** they lack the
`trig_type` column → trigger *provenance* unknowable. But that's trigger-level and (per B) does
not gate word clip extraction.

**What it MEANS / decision:** "Trial exclusion" was meant to drop trials whose neural↔movie
alignment is too poor to give valid clips. The signals I cited are buggy (coverage) or mis-scoped
(trigger-droppable ≠ word quality). **→ Do NOT exclude any trial on these numbers.** The only
defensible gates are (a) subject 5 (already excluded — frontal lesion) and (b) per-clip validity
the loader already enforces via `est_idx`. Before ANY trial exclusion we'd re-derive coverage
from the true trigger extent (not the last-end-row) and accept that trigger-provenance doesn't
invalidate word clips. **T18 (exclusions) is OFF the critical path to the live run** — deferred,
not wired. (Upstream-bug candidate: `pause_audit.last_movie_time_s` restart handling.)

---

## FLAG 2 — hop128 geometry: cosmetic off-by-one in my description (not a code bug)
A 5 s clip @ hop 128 gives **T_bin = 81** STFT frames (not 80); the (3,2) stem floors
`(81−2)//2+1 = 40` = teacher's 40. Conclusion (frame-for-frame P3 alignment, no 10 s clips /
no special stem / no teacher re-pin) holds and is enforced by a runtime assert in
`v14_phase3.py:303-311`. No action — just correcting my "≈80" wording.

---

## FLAG 3 — MON-MASK-002 kill-criterion is silently dead on the B36 path
`v14_joint_module.py:679` early-returns when `shaft_mask is None`; B36 dropped `shaft_mask` from
the default SSL path, so the orphan/visible-MSE ∈ [0.7,1.5] kill-criterion never fires on a real
B36 run. RankMe/effective-rank collapse monitors ARE wired (fire from step 0), so the collapse
tripwire is covered — but this one documented kill-criterion is inert. Re-deriving the tube-mask
analog needs the first BT-Lite run (can't fully land offline). Tracking; not a fire blocker.

---

## FLAG 4 — HB02 GPU-hour compute re-cost still OPEN
The B36 redesign changed HB02 trigger fields; no task budgets the campaign and `training_recipe.md`
still says HB02 is open. bs=32 OOMs on coganlab-gpu (32 GB); bs=8 is the working batch. First run
will use a conservative bs=8 config and I'll record actual GPU-h — but a formal HB02 re-estimate is
owed before scaling.

---

## FLAG 5 — Two stale "fit-on-train-only" doc strings contradict the B13 lock (no code impact, but confusing)
While wiring C3 (per-(C,F) robust-z) I found two docs that say the normalizer is "fit on the **train
split** only," which **contradicts** the controlling B13 lock (`v14_blockers.md:225`): C3 is
per-session-**own-recording**, computed **identically train and eval, NO split filter** (it's
physical-unit calibration: impedance × amp-gain). The code correctly follows B13 (fits over each
session's whole recording, same procedure at train and eval) — verified by an adversarial
spec-fidelity audit. The stale strings are:
- `docs/neuroprobe/b36_implementation_plan.md` C3 TEST line (~line 64) — "stats fit on train only".
- `src/speech_decoding/extractors/normalize.py:130-133` — `SessionRobustZNormalizer` docstring "fit
  on the train split."

These are **doc-only** (code is right). I did not edit them this pass to avoid touching a triad/plan
doc without your sign-off — flagging so you can reconcile the wording to B13 when convenient.

## FLAG 6 — B13-vs-M10 CSubject normalization scope tension — must resolve BEFORE any CrossSubject run (does NOT block BT)
Two locks disagree on how to normalize a **held-out subject** in CrossSubject:
- **B13** (`v14_blockers.md:225`): held-out CSubject ALSO fits per-session-own-recording; cohort-pooled
  is the `R-norm-cohort-pooled` **sister** (P1 ablation).
- **M10** (`v14_blockers.md:347`): CSubject **default = cohort-pooled** over the training cohort.

The code implements **per-session-own always** — there is no cohort-pooled path. This is faithful to
B13 (which the §Status precedence rule makes the winner) and B13's reasoning is sound (pooling
cross-electrode impedance manufactures a systematic z-shift). **This does NOT block the BT run** — BT
is CrossSession (every eval session's own recording is seen at prepare-time → gets its own stats; no
held-out-subject case arises). But the tension **must be settled before any CrossSubject leaderboard
cell**, and `R-norm-cohort-pooled` is currently **unbuilt**.

---

## FLAG 7 — C3 fit-time memory peak (mitigated; not a Lite blocker)
The C3 fit reads each session's **full recording** once to compute the per-(C,F) median/MAD. A 4-agent
substrate audit measured worst-case fit peak ≈ **25 GB** for a 2 h / 250-ch session (full waveform +
concatenated STFT frames + the median's sort buffer all alive at once); typical BT lands at **3–5 GB**.
I applied the audit's cheap mitigation — free the full-session waveform and per-chunk specs *before*
the fit allocates its sort buffer — which drops worst-case to **~7 GB** with zero numeric change
(`view.py:_fit_session_robust_z`). Serial, one session at a time. **Not a Lite blocker**; noting in case
a very wide/long BT (or later D-cohort/SWEC) session shows up.

## FLAG 8 — Unbounded MapInfra RAM cache, now FRONT-LOADED by C3's prepare (watch on Full)
The substrate's `CacheDict._ram_data` has **no eviction** (pre-existing NeuralSet property,
`exca/cachedict/core.py:139` — "LRU-like" docstring is aspirational). C3's `prepare()` now eagerly reads
**every** session up front to fit stats, so the entire RAM cache of full-session preprocessed waveforms
is resident before training step 1. Aggregate ceiling (per process): **Lite (12 sessions) ≈ 42 GB
typical / ~180 GB worst-case** (256-ch 2 h); Full (26) ≈ 92 GB / ~390 GB. Under DataLoader **spawn** it
multiplies by `num_workers`; DCC default is **fork** (copy-on-write), which avoids that. **This memory
would be touched anyway during training** — C3 only front-loads it — so it is not C3's bug, but it IS
the dominant memory cost and could OOM a 128 GB node on a worst-case Full run. coganlab-gpu host RAM
should be checked, and `keep_in_ram` bounded / post-fit eviction added, **before scaling past Lite**.
The Phase-4 supervised smoke ran green on coganlab-gpu, so Lite-scale data fits; Full is the watch.

---

## FLAG 9 — ★ HEADLINE ★ The Whisper-teacher join uses a DIFFERENT clock than the neural slice. Getting it wrong silently destroys P3.

This is the most important thing I found while wiring P3. The P3 cross-modal anchor pairs each
neural clip with the Whisper encoding of the **movie audio the subject was hearing at that moment**.
That requires each clip's **movie-clock onset** `t0_movie_s`. There are TWO clocks in BT, and they
are **not** the same:

- **Neural clock** — where the clip is sliced from the neural stream: `est_idx / 2048 Hz`
  (`word_events.py:183`). This is what the encoder sees.
- **Movie clock** — where the word actually occurs in the movie audio: the `words_df` **`start`**
  column (seconds). This is what Whisper must be sliced at.

They diverge by a per-trial offset that **DRIFTS within a single trial**. Measured directly on the
bundled `subject4_trial1_words_df.csv` (8,223 words):

| | first word | last word | drift |
|---|---|---|---|
| `est_idx/2048` (neural) | 274.16 s | 6051.62 s | — |
| `start` (movie) | 39.02 s | 5147.19 s | — |
| **gap (neural − movie)** | **235.1 s** | **904.4 s** | **grows 235→904 s** |

If the teacher join had used `est_idx/2048` as `t0_movie_s` (the naive read — and exactly what my
first code-trace concluded before I checked the data), every BT teacher feature would be sliced
**235–900 s off the actual word** — i.e. the P3 distillation target would be acoustically unrelated
noise, the loss would still go down (the student would learn to predict whatever-is-there), and the
run would "succeed" while teaching the encoder nothing about speech. Pure silent corruption. The
`word_events.py:146-155` comment already documents this divergence for the NEURAL-slice decision; it
had not been carried into the (unwritten) teacher join.

**Decision (confirmed, not a guess):** the teacher-cache join key is the `words_df` **`start`**
(movie-clock) onset. The neural slice stays at `est_idx` (+ Δlag). They are independent by
construction. I'm threading a `movie_onset_s` field through the word events so the `whisper_target`
extractor can look it up; I'll validate offline against the bundled CSVs (no GPU) before any cache
build is consumed. **This is load-bearing label-derivation** — flagging for your sign-off even though
the data makes the choice unambiguous.

---

## FLAG 10 — No P3 teacher cache exists yet; DCC clone is on a stale branch (both expected, both handled)

- **No P3 teacher cache on DCC.** Only the older `whisper_ceiling` probe artifacts exist
  (`cache_neuroai/whisper_ceiling`, from the 5/28–5/29 ceiling probe). The P3 distillation cache must
  be built — `teacher_cache.write_clip_cache` exists but has **zero production callers**. I'm building
  the *whole-movie* form you asked for (run Whisper-v3 once per movie over 30 s grid chunks → dense
  50 Hz stream; every clip/Δlag is a free slice) rather than the per-clip primitive. **20 movies** are
  staged at `/hpc/group/coganlab/ht203/data/braintreebank_wavs/` (ant-man…venom, all 16 kHz, T13).
- **Caveat to flag:** 30 s grid-aligned chunks (your directive) mean a clip whose 5 s window straddles
  a chunk boundary gets teacher frames from two separate Whisper passes, each with truncated context
  at the seam. Accepted per "grid-aligned"; I'll add an `R-teacher-overlap` sister (overlapping
  windows, keep center) as the falsifier if boundary effects show up. Storage ≈ 1 GB/movie fp16 →
  ~19 GB for 20 movies (fits persistent tier; far below the neural-cache budget).
- **Branch state (resolved):** the laptop is on **`main` @ `8221d97`**, not `b36-impl` as the
  session-start snapshot said. `b36-impl` (`87ac460`) is the *merge-base* and is **fully contained in
  `main`** (`b36-impl..main = 0`; `main..b36-impl = 6` fe-sweep commits) — so `main` has all the B36
  code (committed `v14_phase3.py` contains `V14Phase3DistillModule`) plus the later filterbank work.
  The DCC clone (`8221d97`) is **in sync with laptop `main`, NOT stale**. My uncommitted work (C3 +
  teacher cache + prior-session edits to `v14_phase3.py`/`v14_joint*`/`study.py`/docs) stacks on top.
  Per "branch off the default branch first," I'll land this on a feature branch off `main`, push, and
  `scripts/dcc/sync` resets DCC to it before dispatch. (DCC also carries stray untracked files —
  `.tmp_*.sbatch`, `bench_dataloader.py` — that sync's hard-reset will clear.)

## FLAG 11 — Teacher-cache build is now self-verifying (two guards added after the 4-agent audit)

The 4-agent adversarial audit on the cache build returned **Correctness SOUND / Spec FAITHFUL**;
the verification lens flagged a real **BLIND-SPOT**: the build trusted the in-memory tensor and never
re-checked what hit disk, so a movie that silently truncated to its first 30 s (1500 frames) or a
torch.save truncated by OOM/full-disk would be recorded as a *valid* cache entry and then slice the P3
target off-target — the same silent-corruption class as FLAG 9. Two guards added to
`teacher_cache.write_movie_cache` (both test-covered, 25 tests green):
- **Frame-count invariant**: `dense.shape[0] == round(duration_s × rate_hz)` is exact for 30 s grid
  chunking (k full chunks × 1500 frames is integer, so per-chunk rounds sum to the whole-movie round).
  A mismatch ⇒ a chunk truncated/dropped or enc rate ≠ 50 Hz → `RuntimeError`, refuse to cache. This
  directly catches the first-30-s truncation trap the chunking design exists to prevent.
- **Post-save reload check**: re-open the just-written `.pt` (mmap, header read only — no full
  materialize) and assert shape/dtype survived. Catches a truncated/corrupt write.
- A **post-build verifier checklist** (reload every `.pt`, assert `n_frames == round(dur×50)`, dtype
  fp16, count == 20) was produced by the audit and will be run on DCC after the job; it is the
  acceptance gate for T14, not just the build's exit code.

**Deferred (gold-plating for a 20-movie one-shot build, noted not implemented):** per-file checksums
in the manifest; an fp16→fp32 variance-preservation unit test (fp16 rel-error ~1e-3 is well below what
an L1 distill target resolves — standard V-JEPA/data2vec practice); an empty/tiny-WAV duration sanity
floor. **Non-issue corrected:** the audit's "BLOCKER 2" (build script `--dry-run` imports torch on the
login node) is not a blocker — importing torch ≠ loading a model; `from_pretrained` is never called on
the dry-run path, and the normal dispatch path runs only the stdlib-only *submitter* on the login node.
