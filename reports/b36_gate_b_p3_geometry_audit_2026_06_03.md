# Gate-B audit — P3 input geometry (hop=128 → T_p=40 = teacher)

4-agent adversarial audit, 2026-06-03. Chunk B of the BT-full-run prep.
Branch `bt-full-run-prep`. Code is mechanically correct; the items below are
**flags for Ben** — two touch the load-bearing P3 distillation contract, so
they are decisions, not unilateral fixes.

## Verdict

| Audit | Verdict | Real bug? |
|---|---|---|
| Arithmetic / off-by-one | CONFIRMED | No |
| Phase consistency / RoPE | CONFIRMED | No |
| Config-drift / default leakage | CONFIRMED | No |
| Student/teacher temporal alignment | BROKEN (claim wording) | No execution bug; distillation-quality concern |

The geometry **executes correctly**: 5 s → 81 STFT frames → (3,2) stem → T_p=40
== teacher pooled 40 (verified numerically with the real modules); 1 s → 17 →
T_p=8. RoPE is sized to T_p=40 (not the pre-stem 81), 40→8 is a **bit-identical
exact prefix** (maxdiff 0.0), no misindex, no silent T_p drift path, and the P3
step asserts `student.shape[-2]==teacher.shape[-2]` (loud crash on drift). Two
previously-untested invariants are now pinned by `test_p3_geometry_lock.py`
(commit 1c7cee0).

## Flags for Ben

### 1. [DESIGN — P3 contract] 31.25 ms constant student/teacher frame offset
The teacher 8 Hz grid is anchored at 0 ms (`whisper_teacher_pool.py:52`,
`centres = arange(40)*6.25`), so teacher frame *i* center = `i·125 ms`. The
student's (3,2) stem averages STFT frames `[2j, 2j+1]` (centers 0/62.5 ms),
receptive-field midpoint = `j·125 + 31.25 ms`. **Net: student frame i is
centered 31.25 ms later than teacher frame i, constant across all 40 frames.**

- **Severity: LOW-MED.** It is a quarter-frame, well under the ±300 ms
  physiological neural-lag jitter (Duraivel/Cogan). The stem conv weights are
  *learned* (effective center is adjustable) and the encoder's time
  self-attention can route content temporally, so the offset is largely
  absorbable — it is not a hard misalignment.
- **Options:** (a) accept + document as sub-physiological-lag noise
  (my lean — re-centering buys little and risks pool edge effects); or
  (b) shift the pool centers by +1.5625 input frames to align the two
  midpoints. Either way, **do not change the pool silently** — it is the P3
  target contract.

### 2. [DESIGN — P3 validity] Cross-clock join is word-identity, not frame-precise
The neural window opens at `est_idx/2048` (NEURAL clock); the teacher window at
`movie_onset_s = words_df['start']` (MOVIE clock). They are joined by **word
identity**, not a shared timeline (this is the FLAG-9 clock trap, working as
designed). The audit measured, on real `subject2_trial4_words_df.csv`:
- local neural-span/movie-span ratio ≈ **1.0004** over 10-word windows of
  continuous speech → frame-for-frame is locally meaningful there;
- ratio spikes to **6.887** when a 5 s window straddles a movie gap / scene-cut
  / silence → frame *i* ↔ frame *i* then compares unrelated content;
- gross offset `est_idx/SR − start` drifts 393→3184 s within the one trial
  (accumulated during gaps, not during speech).

So frame-for-frame P3 distillation is valid for clips that sit inside a
continuous-speech run and degrades for clips spanning a discontinuity. This is
a **data property**, not a code bug, but it bears on P3 loss quality and
connects directly to the deferred clip/trial-exclusion task (#18). **Decision
for Ben:** accept the per-clip noise (SmoothL1 + many clips averages it), or
exclude clips whose neural↔movie local ratio departs from 1 by > threshold.
Recommend deciding the exclusion threshold here alongside #18.

### 3. [OPERATOR] P4 needs explicit `--clip-len 1.0`
`--clip-len` defaults to 5.0 for **all** phases incl. `--phase 4`; there is no
phase→clip_len coupling. If P4 is launched without `--clip-len 1.0` it runs at
T_p=40 — geometrically valid (mean readout absorbs it, no crash) but **not the
1 s leaderboard-parity** cell. Add to the chain-dispatch (Chunk D) so the P4
stage always passes `--clip-len 1.0`; meanwhile it's an operator checklist item.

### 4. [DOC drift] Stale hop=256 in superseded docs
`docs/neuroprobe/v14_implementation_fix_list.md:46,555` and
`docs/neuroprobe/training_recipe.md:790` still prescribe the reverted hop=256 /
8 Hz plan. Live code is hop=128 (re-locked 2026-06-03). MEMORY already flags the
fix-list as stale; clean these when convenient. MEMORY §Status B20 row
("Conv2d (3,2), hop 256/8 Hz") is a historical snapshot — reconcile on next edit.

## What changed in code (safe, audit-driven)
- `test_p3_geometry_lock.py` (commit 1c7cee0): pins T_p(5s)==40==teacher 40,
  T_p(1s)==8, the one-frame cliff location, all-front-ends hop=128 parity, and
  the RoPE cross-phase prefix property (maxdiff 0.0). Nothing else changed —
  the alignment items above are deliberately left for Ben.
