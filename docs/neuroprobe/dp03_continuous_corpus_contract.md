# DP03 — Continuous Intrinsic-SSL Corpus Loader Contract

**Status**: ACCEPTED 2026-05-31 (Ben). **No code written.** Discuss-before-code
artifact for the SWEC (and AJILE12) loaders. The one open item at acceptance — clip
tiling vs the locked M18 sampler — was resolved as a **SWEC-only override of M18**
(§2 row 3); falsifier sister `R-swec-wrs-sampler`.

**Scope**: SWEC-iEEG (primary) + AJILE12 (sibling). Both are NeuralFetch-style
`Study` scaffolds today (`studies/swec/study.py`, `studies/ajile12/study.py`) —
every data method raises `NotImplementedError`. They are the only two **continuous,
multi-day, sub-2048 Hz, intrinsic-SSL** corpora in the joint cohort, blocked on
the identical set (DP03/DP02/DP06/M07/M18). BrainTreebank (`Wang2024Treebank`) is
the only fully-implemented loader; it is bounded-trial and broadband, so it does
**not** share this contract.

**Closes** (for SWEC): DP03 (subpackage contract) + DP06 (session-stratified clip
extraction). **Settles** M21 (anatomy-blind) via B30. **Adds** the SWEC-specific
layer the clip/session blockers never covered: direct-part I/O, native-CMR
reference, 512/1024→2048 resample, and the **seizure guard** (§4).

**Inherits — does NOT re-decide — these already-closed blockers (all 2026-05-26)**:
M17 (session = whole multi-day recording), M18 (clip sampling = random continuous
offset `Uniform(0, dur−5s)`, WRS **with replacement** — *not* a pre-tiled grid),
M07 (SSL train/val split = session-held-out; SWEC has 1 recording/subject → 5%
clip-stratified fallback), S06 (checkpointing every 5k, best-3 by probe-r²). This
contract layers the SWEC I/O + guard on top of them. **One amendment**: §2 row 3 —
Ben overrides M18 *for SWEC only* to non-overlapping stride-5 s tiling (broadband
corpora keep M18's random-offset WRS); falsifier sister `R-swec-wrs-sampler`.

---

## 1. Why one contract for two corpora

Scoping SWEC alone would either duplicate the work for AJILE12 or bake
SWEC-only assumptions that don't generalize. The loader *shape* is shared;
corpus differences become parameters (mains notch, native rate, valid bins,
referencing, presence of ictal annotations). §5 is the parameter table.

The contract is fully nailed for SWEC. AJILE12 reuses the shape; its
corpus-specific values are marked **CONFIRM-AT-AJILE12** where not yet verified.

---

## 2. Locked decisions (this session)

| # | Decision | Value | Rationale / source |
|---|---|---|---|
| 1 | **Timeline unit** | one **part file** | The natural physical unit (~10 GB), bounded, respects the "read parts directly, never the corrupt VDS" landmine (audit Finding B). **Nests under M17**: session = whole recording (all parts of a subject); the part is the I/O + no-cross-boundary clip unit; robust-z + held-out keys stay per whole-recording session (M17/B13). Clip positions are the non-overlapping stride-5 s tile grid per part (§2 row 3, SWEC override of M18); B02's WRS draws over this discrete per-subject tile pool (weighting + locality sharding unchanged). |
| 2 | **Clip length** | **5 s** (T=40 @ 8 Hz) | v14 SSL clip length, verified `training_recipe.md:169` + `whisper_teacher_pool.py:36`. SWEC/AJILE12 are SSL-only → 5 s (the 1 s P4 probe clip never applies). |
| 3 | **Clip tiling** | **non-overlapping** stride-5 s grid (SWEC **override of M18**) | Leakage-cleanliness over M18 conformity: a fixed non-overlapping tile pool gives exact corpus count + whole-tile held-out (no train/val partial-overlap leakage). **Scope**: overrides M18's random-continuous-offset *for SWEC only* — broadband (BT/AJILE12/D) keep M18's `Uniform` WRS. **B02 ripple is bounded**: SWEC clip-source becomes the discrete tile pool and its vb-eh weight becomes *exact* tile-count (no repeat-rate estimate, a DP02-measurement detail) — α=0.5 macro split, locality sharding, StatefulDataLoader, WRS-with-replacement all unchanged (non-overlap is a property of the *pool*, not the draw). Falsifier: `R-swec-wrs-sampler` reverts to M18. |
| 4 | **Seizure guard** (SWEC) | exclude **[onset − 30 min, offset + 90 min]** | Literature-grounded; see §4. |
| 5 | **Sample rate** | resample **512/1024 → 2048 Hz** in the loader | The Multi-STFT front-end fixes `hop`/`nperseg` in *samples*; only ~2048 Hz yields the locked 8 Hz frame rate. Upsampling is content-lossless (SWEC content ≤120 Hz ≪ native Nyquist); the empty high band is already masked by valid bins (k0–k21). Keeps one front-end config across all corpora. |
| 6 | **Referencing** | **accept native, re-reference NONE** | SWEC ships common-median-referenced (CMR), baked in — verified from 3 sources (§4.1). Anatomy-blind → no montage → shaftCAR/bipolar/Laplacian are undefined. Re-referencing would double-reference. (Q4 option A.) |
| 7 | **Part-file memory** | OPEN warning, not locked | Eager `mne.io.RawArray` won't fit (§7.1). Keep solution open. |

---

## 3. The loader contract (method by method)

The substrate path is `Study → Events DataFrame → Transforms/Chain → Segmenter →
Dataset/DataLoader`. A `Study` exposes three load hooks + `_download`.

### `_download`
**No-op / verify-present.** SWEC is fetched out-of-band onto `/work/ht203/data/swec/`
by `scripts/swec/fetch_swec.sbatch` (Slurm; login node SIGKILLs hf_xet). This
method asserts the 50 unique folders are present + reconcile to the committed
manifest; it does **not** download on demand.

### `iter_timelines`
Yields **one dict per part file**, across the 50 unique SWEC subjects (skip the 18
duplicates per the dedup map). Each timeline dict carries:
`{subject_id, part_index, part_path, n_samples, native_sr, mains_hz,
part_t_start_s}` — where `part_t_start_s` is the part's start time in
**full-recording seconds** (= Σ lengths of preceding parts; parts concatenate in
`part_1..part_N` order, vbounds contiguous). `part_t_start_s` is load-bearing for
the seizure guard (§4.2).

### `_load_timeline_events`
Returns the **clip manifest** for the part: the non-overlapping stride-5 s tile grid
over `[0, n_samples)` (SWEC override of M18, §2 row 3), **minus** any tile whose
span intersects a seizure-guard window (§4). Each row is
`{onset_sample, n_samples = 5 × native_sr}`; no class labels (SSL). B02's WRS then
draws over this discrete tile pool. Realizes DP06; overrides M18 for SWEC only
(broadband keeps M18's random offset).

### `_load_raw`
Returns the part's neural signal for the requested clip(s). **This is the open
item** — the NeuralFetch contract returns an in-memory `mne.io.RawArray`, which
does not fit for a ~10 GB part (§7.1). The loader must read each 5 s clip's
samples on demand from the HDF5 (parts directly, never the VDS), apply per-corpus
notch → resample → reference pass-through, and hand the front-end a 2048 Hz array.
Exact mechanism is §7.1 (open).

---

## 4. Seizure guard (SWEC only — AJILE12 has no ictal events)

**Lock: exclude `[seizure_onset − 30 min, seizure_offset + 90 min]` from the clip
pool.** Asymmetric because the physiology is asymmetric: pre-onset changes center
~30 min, post-ictal recovery is far longer.

- **Pre = 30 min** covers the central preictal window. Seizure-prediction studies
  put the optimal preictal period at 5–173 min, **average 25–48 min**
  (arXiv:2407.14876, reviewing the field).
- **Post = 90 min** covers the *average return of background EEG to baseline*
  (adults **84 min**; children 120 min) — i.e. exclude until the signal is back to
  representative resting iEEG, not merely past acute suppression. Focal-seizure
  postictal change alone is mean 275 s (range 7 s to >40 min); SWEC patients are
  focal-epilepsy surgical candidates. Source: Pottkämper et al. 2020, *Epilepsia*,
  "The postictal state — what do we know?" (PMC7317965). Rare tail to 24 h is not
  worth the data cost.

We deliberately reject the conservative seizure-prediction "clean interictal"
convention (4 h before + 1 h after) as the default: it's built for *labeling*
interictal-vs-preictal, not for stripping pathology from an SSL corpus, and costs
~30%+ of the data vs ~10% here.

### 4.1 Why "accept native reference" is correct — SWEC = CMR, baked in
Verified from three independent sources:
1. **Originator (primary)** — ieeg-swez.ethz.ch: *"the iEEG signals were
   median-referenced and digitally band-pass filtered between 0.5 and 120 Hz using
   a fourth-order Butterworth filter prior to analysis and **written onto disk** at
   a rate of 512 or 1024 Hz."* ("written onto disk" ⇒ in the released bytes; the
   512/1024 Hz + 0.5–120 Hz signature uniquely identifies the long-term DB = our
   release. The short-term DB on the same page is 0.5–150 Hz and **not**
   median-referenced.)
2. **Re-exporter** — MVPFormer §4 dataset description, verbatim match, stated as a
   signal property (distinct from their "512 Hz before training" step).
3. **The bytes** — band-pass at 120 Hz confirmed (PSD audit); across-channel median
   is reduced but not exactly zero (RMS 6.6 vs channel RMS 31.6 on ID19), which is
   exactly what "median-referenced **then** band-pass filtered (+ artifact-channel
   removal)" predicts — not raw, not double-zeroed.

### 4.2 Guard is computed in full-recording time, then intersected per part
Seizure annotations (`data/seizures`, onsets/offsets in seconds) are relative to
the **full recording**, but clips are generated **per part**. So: compute each
guard window in full-recording seconds, map the part's span via `part_t_start_s`,
and drop clips that intersect. A seizure near a part boundary guards **into the
adjacent part** — handle in full-recording time, not per-part-local time.

---

## 5. Per-corpus parameter table

| Parameter | SWEC | AJILE12 |
|---|---|---|
| Native sample rate | 512 Hz (17 subj) / 1024 Hz (33 subj) | 1000 Hz `SAMPLE_RATE_HZ` ClassVar — **CONFIRM** vs plan.md "500 Hz" note |
| Resample target | 2048 Hz | 2048 Hz |
| Mains notch | **50 Hz** (CH; MASK-01 per-corpus — dispatch must pass `mains_notch_hz=50.0`) | 60 Hz (US) |
| Released reference | **common-median (CMR)** — accept, no re-ref | **CONFIRM-AT-AJILE12** (verify Peterson 2022 release referencing before assuming) |
| Valid filterbank bins | k0–k21 (0.5–120 Hz) | k0–k20 (per plan.md) — **CONFIRM** |
| Ictal annotations | yes → seizure guard §4 | **none** (motor-BCI cohort) → no guard |
| Anatomy | none → `latent_valid` all-False, front-end-only (B30) | mixed (89.7% ECoG) — per-subject `latent_valid` from support |
| Channels / subject | 22–128 (≤ C_MAX=384) | per Peterson 2022 |
| Artifact channels | pre-removed by expert (paper) → LOF optional/secondary | **CONFIRM** |

Preproc order per part: HPF (0.5 Hz, already baked for SWEC) → per-corpus notch at
native fs → reference pass-through (accept native) → **resample → 2048 Hz** →
Multi-STFT front-end (→ 8 Hz frames) → valid-bin mask → robust-z. (Notch at native
fs, *then* resample, so the notch filter is designed for the native rate.)

---

## 6. Settled facts (do not relitigate)

- **Anatomy-blind → front-end-only.** SWEC contributes `L_pre_frame_masked` only;
  zero slot-axis contribution (`latent_valid` all-False, B30). No parcel routing,
  no `log(support+ε)` bias, no cross-attn ❺.
- **Read PART FILES directly, never `total.h5` VDS** (audit Finding B — VDS cites
  foreign patients' parts; would silently serve the wrong subject's voltage).
- **Ignore embedded `info/checksums`** (audit Finding A — stale after
  re-compression). Integrity = size==HF + Σ(part lengths)==total + boundary-chunk
  decode.
- **50 unique subjects** (skip 18 byte-identical dupes). SWEC-pretrain → BT/D-eval
  is leakage-clean (different institutions). [The MVPFormer 18-of-50 test leakage
  is *their* result's problem, not ours — see `reference_mvpformer_carzaniga_2025`.]
- **Mains notch is per-corpus** (MASK-01): SWEC dispatch passes `mains_notch_hz=50.0`.

---

## 7. Open / deferred items

### 7.1 Part-file memory (OPEN — keep solution space open)
A SWEC part is ~10 GB compressed (ID19 part 1 = 29 ch × 80 M samples = **21.7 h**);
the NeuralFetch `_load_raw → mne.io.RawArray` contract is in-memory (float64 →
~20 GB; worse for 128-ch parts), times N DataLoader workers → untenable. The
timeline-as-part-file *indexing* unit is fine; only the eager materialization is
the problem. Candidate solutions (not yet chosen):
- lazy windowed HDF5 reads per clip (read only the 5 s span);
- memory-mapped / chunked-iterable dataset over the part;
- a thinner `_load_raw` that returns a lazy handle rather than a full `RawArray`.
Pick at implementation; Ben flagged "there may be better solutions available."

### 7.2 M07 — SSL train/val split (already CLOSED 2026-05-26 — not deferred)
M07 is closed: **session-held-out val** (hold 1 whole session/subject). SWEC has
**one** multi-day recording per subject (M17), so it takes M07's stated fallback —
**5% clip-level stratified hold** within the recording. *Not* chronological-tail,
*not* set-at-implementation. Best-checkpoint criterion is S06 probe-r² (primary),
SSL `L_recon` secondary. Loader work = tag the 5% held-aside clips; nothing to decide.

### 7.3 DP02 — corpus sampling weight (a measurement, not a decision)
α=0.3 temperature sampling is locked (B29). With the §2-row-3 override, SWEC's clip
count is now **exact**: `Σ_part floor(part_dur / 5 s) − guarded tiles` — no
continuous-offset repeat-rate estimate. Feeds the B02 vb-eh weight directly at
sampler-build (this is the only B02-adjacent effect of the override).

### 7.4 Implementation-time verifications (don't assert until checked)
- **Seizure-annotation storage**: confirm `data/seizures` is readable from
  `total.h5` (small non-VDS structured array — should be intact despite the corrupt
  ieeg VDS) vs per-part; read accordingly.
- **AJILE12** native rate / bandpass / valid bins / referencing / artifact
  handling (the **CONFIRM** cells in §5).
- **Part → full-recording time mapping** exactness (`part_t_start_s` from
  `info/files` part order).

---

## 8. Sisters / ablations

- **`R-swec-guard-conservative`** — re-run the guard at the seizure-prediction
  convention [onset − 4 h, offset + 1 h]. Fires if SSL feature stats still look
  seizure-contaminated. (Sensitivity check on §4.)
- **`R-swec-global-car`** — apply a global-CAR on top of native CMR (Q4 option B),
  the old `plan.md` "SWEC degenerates to global-CAR" intent. Default is accept-native.
- **`R-swec-wrs-sampler`** — falsifier for the §2-row-3 override: revert SWEC to
  M18's random-continuous-offset `Uniform(0, dur−5s)` WRS-with-replacement. Tests
  whether deterministic non-overlap tiling actually beats the locked sampler. The
  override is the default; this is the M18-conformist check.
- Existing roster unchanged: `R-drop-swec` (B30), `R-item-12-all-true`,
  `R-sa-key-only` (retracted per orphan-slot audit), `R-sampler-*` (B02).

---

## Cross-references
- `memory/reference_swec_ieeg_dataset_audit_2026_05_19.md` — dataset audit, dedup,
  integrity landmines, full-pull manifest.
- `memory/reference_mvpformer_carzaniga_2025.md` — the model paper; SWEC test leakage.
- `memory/project_v14_anatomy_gated_symmetric_2026_05_28.md` (B30) — `latent_valid`,
  SWEC front-end-only.
- `memory/project_v14_preproc_recipe_2026_05_12.md` — per-corpus preproc chain.
- `docs/neuroprobe/plan.md` §Phase-1, `docs/neuroprobe/training_recipe.md` — clip
  length, frame rate, sampler shares.
- `docs/neuroprobe/v14_blockers.md` §DP03 — the blocker this closes.
- `scripts/swec/` — fetch/audit/dedup + `confirm_mvpformer_leakage.py`.
