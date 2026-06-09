# v14 P3 Alignment — Engineer Handoff (2026-06-08)

Hand-off for the P3 ("Whisper-distillation shows no signs of training") alignment
fix. Read `REPORT.md` in this directory first — it is the *audit* (what was
broken). This memo is the *fix*: what changed, why, how it was verified, and what
is deliberately left for later.

---

## TL;DR

- **Root cause of flat P3 loss = a teacher-target clock bug.** ~36–42% of the P3
  distill corpus (the `<nonverbal>` anchors) had their Whisper teacher sliced at
  the **wrong movie position**, so ~40% of the distillation gradient pulled toward
  decorrelated targets.
- **Fix = one unified mechanism.** Every anchor's movie-clock onset
  (`movie_onset_s`, the teacher slice key) is now derived from BT's authoritative
  per-session **trigger track** via `np.interp(est_idx → movie_time)`. This is the
  same map BT itself inverts to build `words_df.est_idx`, so it is the most
  authoritative neural↔movie source that exists, and it is the only source that
  **freezes movie time across recording pauses** (the residual that broke the
  nonverbal anchors).
- **Verified exhaustively, locally, from 9 independent angles + an independent
  adversarial re-audit (all 26 sessions) + the full test suite (1475 pass).** No
  remaining correctness gap on this path.
- **Two known issues are deliberately deferred** (S9 half-rate neural *window*
  length; multi-subject channel desync). Neither blocks P3; both are scoped below.

---

## 1. What was broken

`word_events.py` keyed the teacher slice off `source["start"]` for **both** verbal
and nonverbal anchors. For nonverbal anchors `source = nonverbal_df.iloc[...]`, and
`nonverbal_df["start"]` is the **NEURAL clock** (`est_idx/2048`), not a movie clock
— `nonverbal_df` has no movie-clock column at all. So every nonverbal teacher was
sliced hundreds-to-thousands of seconds off the real content. Live only on P3
(`balance=False` keeps every nonverbal anchor; P1/P2 masked-JEPA never touch movie
time) — which is exactly why it surfaced as *P3 not training*. Full evidence:
`REPORT.md` §PRIMARY.

The naive fix (REPORT.md "Option B": recover nonverbal movie time by interpolating
the **sparse `words_df`** est_idx→start map) is *also* wrong in a subtler way: the
sparse word map has no pause rows, so it **linearly bridges recording pauses**. A
nonverbal anchor that lands inside a pause gets a movie time interpolated across
the pause gap — up to **89 s** off. This is the residual the unified map removes.

## 2. The fix (unified trigger-track clock map)

BT records a per-session **trigger track**
`subject_timings/sub_{N}_trial{T:03d}_timings.csv` with explicit
`(index, movie_time)` pairs plus `pause`/`unpause` rows. BT's own
`trial_data_reader.add_estimated_sample_index` builds `words_df.est_idx` *from*
this track (`est_idx = round(near_trig + (start−near_t)·samp_frequency)`). So
`np.interp(est_idx → movie_time)` literally **inverts BT's forward map**, and
because the track carries pause rows, `movie_time` is constant between a `pause`
index and its `unpause` index → it freezes instead of bridging.

Code (`src/speech_decoding/studies/braintreebank/word_events.py`):
- New `_load_neural_to_movie_map(subject_id, trial_id, bt_root)` — reads the
  trigger track, `[index, movie_time]`, dropna, sort by index,
  `drop_duplicates(subset="index", keep="last")` (np.interp needs strictly
  increasing xp), returns `(idx_array, movie_time_array)`.
- `_word_event_rows` takes `neural_to_movie` and computes `movie_onset_s` via one
  closure `_movie_onset(est_idx, is_nonverbal, source)`:
  - **trigger track present (production)** → `np.interp(est_idx, idx, movie_time)`
    for **both** verbal and nonverbal anchors (authoritative path).
  - **trigger track absent (laptop tests, no BT mounted)** → fallback: nonverbal →
    sparse words_df interp, verbal → `source["start"]`.
- `_run` loads the map once and threads it through; `bt_root` set but file missing
  → **raises** (a misconfigured BT root fails loudly, never silently regresses).
- `dispatch_v14.py` already requires `bt_root`/`ROOT_DIR_BRAINTREEBANK` and passes
  it through, so **production always takes the trigger-track path**; the fallback
  only fires on a laptop with no BT (and no teacher cache), where it is inert.

The rip↔BT per-film offset (`_MOVIE_CLOCK_OFFSET_S`: fox +1.75 s, lotr-2
+0.1/+1.0 @100 min) in `whisper_target.py` is a **separate, composable** correction
applied after `movie_onset_s` at slice time. It stays. See REPORT.md §SECONDARY.

## 3. Why unified, not hybrid (the one real design choice)

A hybrid — verbal via exact `words_df.start`, nonverbal via trigger interp — would
match `words_df.start` exactly on verbal anchors. The unified map instead routes
verbal through the trigger track too, which differs from `words_df.start` by ~2.5 ms
on normal sessions and **~51 ms on S9** (the half-rate session). I chose unified
**on purpose**, for a teacher-neural **self-consistency** property:

> The neural window is sliced at `est_idx`. Keying the teacher at
> `trigger_interp(est_idx)` references the **same** neural sample, so any error in
> `est_idx` itself (e.g. S9's 2048-vs-1024 extrapolation overshoot) shifts the
> teacher and the neural window **identically** and cancels in the alignment.
> Using `words_df.start` for verbal would instead de-sync the teacher (true word
> time) from the neural window (at the overshot `est_idx`) by ~51 ms on S9.

The distillation target must describe the movie content the **neural window**
captured, not the transcript word time. So unified is the correct reference, not a
regression. On normal sessions the two agree to ~2.5 ms (sub-frame) anyway. The
~51 ms S9 figure is **0.41 frames at the 8 Hz teacher rate** — sub-bin either way.

## 4. Verification (all local, all reproducible)

Nine independent angles over all 26 vendored sessions
(`$CLAUDE_JOB_DIR/tmp/audit_*.py`, ephemeral — rebuild from this list):

| # | Angle | Result |
|---|---|---|
| 1 | Verbal self-consistency: `interp(est_idx)` vs `words_df.start` | ~2.5 ms median normal; S9 51 ms (= est_idx overshoot, §3) |
| 2 | `index` strictly increasing after dedup | 26/26 PASS |
| 3 | Nonverbal-in-pause freezes at the pause movie_time | **26210/26210 = 100%** (OLD sparse bridged 61.4% > 2.5 s) |
| 4 | New-vs-old onset changes concentrate at pauses | changes localize to pause neighborhoods |
| 5 | S9 teacher-neural self-consistency | est_idx overshoot cancels; 0.41 frames @ 8 Hz |
| 6 | Anchors outside the trigger span (would clamp) | **0/318782** |
| 7 | `np.interp` vs `scipy.interp1d` cross-impl | exact (max diff 0.0) |
| 8 | `movie_time` monotonicity | largest reversal −0.057 s at pause edges, fp-tolerable |
| 9 | **End-to-end through the SHIPPED code** (`_load_neural_to_movie_map` + `_word_event_rows` on real vendored data) | `movie_onset_s` matches independent interp to **0.000 µs** for all sessions incl. S9; fallback verbal == `words_df.start` (0 orphans) |

Independent adversarial re-audit (separate agent, fresh eyes, told to *break* it):
**all 6 attacked claims CONFIRMED** — 183/183 pause pairs freeze (max drift 0.0 ms,
neural index jumps 1–1785 s); strictly-increasing index in all 26 sessions;
23.9 % of nonverbal anchors shift > 1 s vs the buggy map (max 89.4 s). It found
**no correctness bug**; its three notes (two stale docstring numbers + the S9
caveat) are addressed (docstrings fixed) or documented (§3, §5).

Tests: `test_word_events.py` 22 pass / 1 skip; **full suite 1475 pass / 5 skip**.

## 5. Deferred (NOT in this changeset — scope + why safe to defer)

1. **S9 half-rate neural WINDOW length.** S9_T0's neural stream is physically
   ~1024 Hz but BT/Neuroprobe hardcode `SAMPLING_RATE=2048` ("do not change this").
   A 5 s window = `round(5·2048)` = 10240 samples = **10 real seconds** of S9
   neural. This is a **window-length** bug, orthogonal to `movie_onset_s` (which the
   unified map handles correctly for S9, §3). Fixing it is a separate **atomic
   loader/view change** (per-session sample rate + resample-to-2048) that must not
   partially change the `start = est_idx/SR` eval-parity contract — needs a DCC
   h5-shape confirm first. Upstream Neuroprobe also ships this unhandled, so we
   match the leaderboard convention until fixed.
2. **Multi-subject channel desync** (REPORT.md-adjacent, confirmed TRUE by a
   subagent). `channel_order='original'` + a shared last-writer-wins `_channels`
   dict + a single multi-subject `dataset.prepare()` scatter voltage rows to global
   indices ≠ the per-subject `voltage_electrode_order` used by dk_support/valid_mask
   → the hard per-parcel pool averages the wrong electrodes. **Single-subject is
   SAFE**; only multi-subject pooled runs corrupt. Orthogonal to P3 alignment; own
   fix (order voltage by per-subject `voltage_electrode_order`).
3. **fox/lotr-2 rip offsets** — already handled by `_MOVIE_CLOCK_OFFSET_S` (stays;
   film-intrinsic, composes with the unified map). Not corrupt, no re-rip.

## 6. Changeset + commit/sync status

Uncommitted, one coherent P3-alignment changeset (377 +/22 −):
- `studies/braintreebank/word_events.py` + `test_word_events.py` — unified map.
- `extractors/whisper_target.py` + `test_whisper_target.py` — rip-offset table +
  docstring (movie_onset_s is now the trigger-track interp).

**NOT yet committed/synced** — awaiting Ben's review of this memo (per the
discuss/handoff protocol). When cleared: commit on the laptop branch
`bt-full-run-prep` → push → `scripts/dcc/sync`. **Live P3 job `47919036` must not
be disrupted** — `scripts/dcc/sync` is a `git reset --hard` on the DCC clone; do
not sync while that job's working tree matters. Re-launch P3 only after the synced
code is in place.

## 7. Durable records to update (post-review)

- `MEMORY.md` §Status + `memory/project_v14_p3_nonverbal_clock_bug_2026_06_08.md`
  — record the unified trigger-track map as the resolution (supersedes Option B).
- `REPORT.md` §Fix options — mark Option B superseded by the unified map.
