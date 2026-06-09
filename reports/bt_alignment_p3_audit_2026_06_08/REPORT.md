# v14 P3 Alignment Audit — 2026-06-08

**Trigger**: P3 (Whisper-distillation) shows no signs of training. Goal: rule out — or find — any neural↔movie↔word data-loading/alignment bug. Run as a 23-agent workflow (`wf_e0976e31-440`), then every load-bearing claim re-verified by hand against code + the vendored CSVs + the empirical gate JSONs in this directory.

## Bottom line

**A concrete, live alignment bug was found and confirmed in code + data.** It is a sufficient cause of flat P3 loss. We cannot rule out a "rookie" data bug — we hit one.

---

## PRIMARY — nonverbal teacher-target clock bug (CONFIRMED)

`word_events.py:271` sets the Whisper-teacher slice key
```python
"movie_onset_s": float(source["start"]),
```
for **both** verbal and nonverbal anchors. For nonverbal anchors `source = nonverbal_df.iloc[src_idx]` (`word_events.py:246`), and `nonverbal_df["start"]` is the **NEURAL clock**, not the movie clock.

**Hand-verified against all 26 vendored sessions** (`.cache/neuroprobe_upstream/.../braintreebank_features_time_alignment/`):

| df | columns | `start` vs `est_idx/2048` |
|---|---|---|
| `nonverbal_df` | `start, end, est_idx, est_end_idx` (**no movie-clock column**) | `start == est_idx/2048` for **100% of 115,990 rows**, max diff **0.00 s** |
| `words_df` | `…, est_idx, …, start, …` | diverges for **100% of rows**, max diff **3616 s** (real movie clock) |

So for every nonverbal anchor the teacher cache is sliced at the neural-clock time, which in movie time is hundreds-to-thousands of seconds off the actual content (or past the movie end → clamped to a near-constant tail).

**It is live on the P3 path**: `dispatch_v14.py:1053-1064` sets `ssl_phase=True` → `balance=False` for `p3_distill`, with the explicit comment "keep EVERY word + nonverbal anchor." Nonverbal anchors are **~36–42%** of the corpus (115,990 nonverbal / 318,782 total across all sessions; the synthesis computed 62,257/147,418 = 42.2% over the 12 distill sessions). Roughly 40% of the distillation gradient pulls toward decorrelated targets. Only P3 attaches `WhisperTargetExtractor`/`movie_onset_s` — P1/P2 masked-JEPA never use movie time — which is exactly why this surfaces as *P3* not training.

**The in-code comment at `word_events.py:269-270` is false** ("Both words_df and nonverbal_df carry the movie-clock column `start`"). The test fixture `test_word_events.py:81` builds nonverbal stubs as `est_idx = (start+200)*2048`, giving them a fake 200 s movie offset — which is why no test ever caught this.

### Fix options (a real fork — re-keying IS possible)

The synthesis claimed re-keying is "impossible" because `nonverbal_df` has no movie column. That is too strong: the **same-session `words_df` carries dense `(est_idx → start)` pairs** that define the neural→movie mapping, so a nonverbal window's movie time can be recovered by `np.interp(nonverbal.est_idx, words_df.est_idx_sorted, words_df.start_sorted)` using only data already loaded.

- **Option A — exclude `<nonverbal>` from the P3 distill loss.** Simplest; drops ~40% of frames; makes P3 speech-only.
- **Option B — re-key `movie_onset_s` for nonverbal via `words_df` interpolation.** Preserves full corpus coverage; principled; ~5 lines in `word_events.py`.

Either way: fix the false comment (269-270) and replace the masking test fixture with one that mirrors real `nonverbal_df` (`start == est_idx/2048`) + a regression assertion that no nonverbal `start` is ever emitted as `movie_onset_s` unre-keyed.

---

## SECONDARY — two P3 films are CLOCK-OFFSET, not corrupt (RESOLVED 2026-06-08)

**UPDATE (2026-06-08, supersedes the original "wrong cut/edition" verdict below).**
The first gate (`audio_gate_results_all21.json`) searched only a constant
per-quarter SHIFT over a **±1.0 s grid**. A scale-aware re-probe
(`run_scale_gate.py` → `scale_gate_results.json`, + a fine refine) shows both
flagged films are the **right audio content at the wrong clock offset** — fully
recoverable in software, **no re-rip needed**:

- **fantastic-mr-fox** (sub1_trial0): scale=1.0, **constant +1.75 s offset** → whole-film log-RMS r **0.85**, uniform across all four quarters (0.83–0.87). The first gate's r=0.20 was an artifact: fox's true offset (~1.75 s extra lead-in in the rip) sits *outside* the ±1.0 s search window, so it never found it. NOT a wrong edition.
- **lotr-2** (sub3_trial2): scale=1.0, **+0.1 s for t<100 min, +1.0 s for t≥100 min** (reel-join step localized to the ~100-min mark) → r 0.88–0.92 on both sides.
- 18/20 films remain clean at best-shift 0.0–0.2 s (within the ~1-frame@8Hz tolerance band).

**WAV provenance RESOLVED**: the production cache WAVs (`/hpc/.../braintreebank_wavs`) are **md5-identical** to the validated laptop `audio/bt_16k` rips (DCC check), so the cache carries these same offsets. Fix path is therefore a per-film `movie_onset_s` offset correction (fox: +1.75 s; lotr-2: piecewise +0.1/+1.0 @100 min) applied at teacher-slice time — OR exclude the 2 sessions. The nonverbal bug above is independent of this.

<details><summary>Original (superseded) verdict</summary>

Empirical audio gate (`audio_gate_results_all21.json`, per-word RMS-log recompute vs BT `features.csv` rms over `[start,end]`, run against `audio/bt_16k`):

- **18/20 films PASS**: per-quarter best-shift 0.0–0.2 s, r 0.82–0.93, no misaligned regions.
- **fantastic-mr-fox** (P3 session sub1_trial0): r=**0.20** even after shift search, 76 misaligned regions — the laptop rip is the **wrong cut/edition**, off-word throughout.
- **lotr-2** (P3 session sub3_trial2): front half clean (r 0.90 @ +0.1 s), back half jumps to **+1.0 s** past ~100 min, 24 misaligned regions — a reel-join step.

The original ±1.0 s shift cap is why fox read as "wrong edition." Corrected above.
</details>

---

## What is CLEAN (densely verified — do not chase these)

| Link | Verdict | Evidence |
|---|---|---|
| Raw neural load / windowing (A) | ✅ | `round(est_idx/2048*2048)==est_idx` max err 0 over 147,418 anchors; byte-parity with upstream `datasets.py:300-302`; channel axis label-keyed. |
| Verbal event construction (B) | ✅ | `est_idx` is **trigger-offset-mapped, NOT naive `start*2048`** (clock gate residual ~2.5 ms; naive would be 100–2544 s off); `start`=neural, `movie_onset_s`=movie, no swap. |
| Segmenter / clip window (C) | ✅ | segmenter start = neural; `WhisperTargetExtractor` reads `movie_onset_s`, never segmenter start; missing key raises (no silent frame-0). |
| Teacher cache build/index (D) | ✅ core | whole-movie dense @50 Hz, `t0_movie_s==0`, frame-count invariant; `mean_all` over 32 layers; z-score retains structure. (Residual: WAV provenance, below.) |
| P3 module geometry (E) | ✅ module | T_p=40 == teacher 40, teacher detached, no degenerate target in code, 3a/3b freeze correct — but fed the corrupt targets from B. |
| Video alignment | ✅ | brightness vs BT `mean_pixel_brightness`: the-martian r 0.926, lotr-1 0.881, coraline 0.942, sharp peak <0.25 s. |
| DK support (electrode→parcel) | ✅ | `test_voltage_order_matches_upstream` 10/10; one-hot integrity holds. |

---

## Open DCC-side checks (not yet run — h5/cache are DCC-only)

See `p3_alignment_hypotheses` + `dcc_commands` in the workflow result for exact commands. Priority:
1. **WAV provenance**: do `/hpc/.../braintreebank_wavs` match the validated `audio/bt_16k` rips? (md5/duration; re-run the RMS gate against `/hpc`.) Resolves whether fox/lotr-2 (and any other) are corrupt in the actual cache.
2. **Teacher target health**: per-movie `features.std()>0`, `n_frames==round(dur*50)`, `t0_movie_s==0`; `channel_stats.pt` finite, `inv_std` not all 1, fit @ 8 Hz. (Rules out a degenerate/wrong-rate target.)
3. **`--no-target-standardize`** on the failing run? (non-alignment co-factor: one teacher channel ~777× median energy can saturate SmoothL1.)

## Non-alignment co-factors (out of asked scope; may co-contribute, cannot make 40% learnable)

Raw un-standardized teacher target; student init output ~16–40× too small vs unit target; low production peak LR; `train_loss` logged on_epoch-only (watch val_loss).

---

## Artifacts in this directory

- `run_audio_gate.py` / `audio_gate_results_all21.json` (all 20 films) / `audio_gate_results.json`
- `run_video_gate.py` / `video_gate_results.json`
- `run_clock_check.py` / `clock_check.json`

Note: the runner scripts need `librosa`/`soundfile`/`opencv` for the full feature set (RMS path works on `scipy`/`numpy` alone). The `mel` (128-d) per-word drift probe in `features.csv` remains **unconsumed** — a denser future check.
