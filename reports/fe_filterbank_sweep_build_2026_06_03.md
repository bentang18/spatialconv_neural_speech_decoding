# Front-end filterbank sweep — build doc

**For:** the engineer standing up the cheap front-end probe sweep.
**From:** the 2026-06-03 front-end design thread (Ben + Claude), grounded in `iMINDBench_iEEG_Multi_Ins.pdf` (read pp. 1–20).
**Status:** proposal — needs Ben sign-off on the grid before dispatch. Then engineer-buildable (no Ben-approval-gate files touched).

## 1. Question

Does v14's log-octave (constant-Q) filterbank earn its place over iMINDBench's raw-STFT-bins front end, and if so at what `{f0, high-cap, octave_step, half_bw}`? Settle it with a cheap logistic probe **before** committing the front end to pretraining. The within-vs-cross split is the payload: the broadband-power prior predicts the filterbank (and especially trimming/over-smoothing high-gamma) helps **cross-subject more than within-session**.

This sweep settles **filterbank knobs only**. Conv patch stem + masking are SSL-coupled → deferred to the B36 `R-*` sisters. CQT front end → deferred low-priority sister (`R-cqt-frontend`; the 1 s eval window caps its only real advantage, see thread).

## 2. iMINDBench parity (the harness is frozen to their setup)

Verified from the paper, with receipts:

| Element | iMINDBench spec | Source |
|---|---|---|
| STFT | 3 windows, nperseg 1024/512/256 @ 2048 Hz, hop=128 (68 ms) | Fig 8, §3.4 |
| Multi-STFT front end | **raw magnitude bins** per band (low 2–40, mid 20–148, high 80–248 Hz), concatenated, **no log filterbank**; retains 2–250 Hz | Fig 8, L182–183 |
| Normalization | per **channel-and-frequency** session-level stats | L182–183, Fig 8 caption |
| Pre-STFT | notch + **Laplacian** re-reference | L177 |
| Probe (Logistic) | sklearn linear classifier on **flattened** aligned-channel inputs, `max_iter=10000`, `tol=1e-3`, `seed=42` | L434–435 |
| Tasks | 1 s window aligned to word onset, **binary** classification (low-vs-high contrast), ROC-AUC | §3.2, Table 3 |
| Within-session split | movie ≈ split in half, train/eval halves **swapped across 2 folds**, score = mean of folds | L133–139 |
| Main-unit threshold | logistic-reference ROC-AUC **> 0.55** | L162 |
| Scoring | mean unit-level ROC-AUC across subject/session units | L211 |
| Sweep methodology precedent | Fig 4b/4c: sweep front-end params, logistic probe, report ΔROC-AUC ± 95% CI | §5.2 |

**Anchor / correctness gate (cell G).** Run **one** iMINDBench-exact cell — **Laplacian re-ref + raw bins** — and confirm it reproduces iMINDBench Multi-STFT-Logistic on BT: **Neuroprobe column 0.685** (Table 1; 0.663 is the cross-dataset overall). If cell G lands far from ~0.685, the harness is wrong — **fix that before trusting any sweep number.** Cell G is the only Laplacian cell; the actual sweep runs under shaft-CAR (§3).

## 3. Held constant (every cell) vs swept

**Frozen across all cells (iMINDBench parity):** STFT (1024/512/256, hop 128, Hann, raw `|STFT|`), per-corpus notch, per-`(C,F)` session robust-z, the exact logistic probe above, the eval splits, ROC-AUC, the BT subject subset, `exclude_artifacts=True`, grouped-by-token.

**Re-reference (Ben call 2026-06-03):** the sweep (control + cells 1–N) runs under **shaft-CAR** (the v14 recipe's spatial reference), so the filterbank ranking is read in v14-relevant conditions. The **only** Laplacian cell is the harness gate (cell G, §2) — its job is to reproduce iMINDBench's ~0.685 once and prove the harness is correct, after which the sweep's Δ-baseline is the shaft-CAR raw-bins control (cell 0). Filterbank ranking is assumed ~orthogonal to the spatial reference; `R-fe-shaft-car-confirm` is moot now (sweep already shaft-CAR) and is replaced by the gate.

**Swept — the filterbank only** (sits on the frozen multi-STFT grid):

| # | knob | what it controls | values | why this many |
|---|---|---|---|---|
| K0 | **filterbank presence** | raw bins (iMINDBench) vs constant-Q log filterbank (v14) | control vs filterbank | the headline control axis |
| K1 | `f0` | lowest bin center (low edge) | **2, 4 Hz** | physics-bounded: <2 dead (Gabor), >4 eats theta |
| K2 | high cap | highest bin center (→ sets `n_bins`); **octave-clean** = f0·2ⁿ | **128, 256 Hz** | physics-bounded: 64 kills high-γ, 512 pure noise. 256 keeps the noisy top octave, 128 trims it |
| K3 | `octave_step` | bin **spacing** (density on log axis) | **½, ⅓, ¼ oct** | free → 3 values |
| K4 | `half_bw_octaves` | triangular **width** / per-bin SNR | **0.5 (matched), 0.75, 1.0** | free → 3 values |
| — | `n_bins` | **derived** from {f0, cap, step}, not free | — | — |
| — | routing | **derived** per cell (see §6), not free | — | — |

**Granularity policy (Ben 2026-06-03):** 2 values where physics brackets the knob (K1, K2), 3 where it doesn't (K3, K4) — matching iMINDBench's own 3–4-per-axis sweeps (Fig 4b). **Coarse-first, refine-on-signal:** run the one-at-a-time pass below, read the within-vs-cross deltas, then add interior points / interactions **only** on the 1–2 knobs that actually move. Do not pre-pay for a full factorial.

## 4. Grid (coarse first pass — one-at-a-time off the default, 9 cells)

| cell | ref | f0 | cap | step / half_bw | n_bins | role |
|---|---|---|---|---|---|---|
| **G gate** | Laplacian | — | 2–250 | raw bins | ~70 | iMINDBench-exact; must reproduce ~0.685 (harness check, run once) |
| **0 control** | shaft-CAR | — | 2–250 | raw bins | ~70 | Δ baseline for the sweep |
| **1 default** | shaft-CAR | 2 | 256 | ½ / 0.5 | 15 | constant-Q, matched, in-limits |
| 2 drop-delta | shaft-CAR | **4** | 256 | ½ / 0.5 | 13 | does dropping noisy delta help? |
| 3 trim-octave | shaft-CAR | 2 | **128** | ½ / 0.5 | 13 | does dropping the noisy 128–256 octave help cross > within? |
| 4 finer-⅓ | shaft-CAR | 2 | 256 | **⅓ / ⅓** | 22 | finer low-mid resolution (matched) |
| 5 finer-¼ | shaft-CAR | 2 | 256 | **¼ / ¼** | 29 | finer still (matched) — curvature on resolution |
| 6 smoother-0.75 | shaft-CAR | 2 | 256 | ½ / **0.75** | 15 | extra averaging at fixed spacing |
| 7 smoother-1.0 | shaft-CAR | 2 | 256 | ½ / **1.0** | 15 | max averaging — curvature on width |

All cells 0–7 report **Δ ROC-AUC vs cell 0**, separately for within and cross. Cells 4–5 vary `octave_step` **with matched `half_bw`** (a pure resolution axis); cells 6–7 widen `half_bw` at fixed ½ spacing (a pure smoothing axis). Refine-on-signal: only the knob(s) showing a real Δ get a second pass (interior points / interactions).

## 5. Eval cells + task set (locked 2026-06-03)

**Task set (Ben 2026-06-03):**
- **Ranking set (4) — drives the filterbank decision:** `speech`, `sentence_onset`, `volume`, `pitch`. High-decodability (speech/onset ~0.92–0.93 on BT, Fig 7), high-gamma-dominant so the cap/smoothing knobs get real signal, + pitch for band diversity. Mean ROC-AUC over these 4 is the headline metric.
- **Report-all (15) — secondary robustness readout, ~free:** compute and report all 15 iMINDBench binary tasks (Table 3). Features are task-independent (shared STFT), so all-15 adds only cheap logistic fits. Used to confirm the ranking holds; near-chance/visual tasks do **not** drive the decision.

**Eval cells (both):**
- **Within-session:** iMINDBench 2-fold movie-half-swap, ROC-AUC, ~5 BT subjects.
- **Cross-subject:** Neuroprobe-**native train-one (sub 2) / test-one** (leaderboard-parity per L.2; **not** LOSO), binary AUROC. Train on sub 2, test each other BT subject, average.

Reported separately for within and cross, for both the ranking-4 mean and the all-15 mean.

## 6. Routing-derivation rule (the one non-obvious piece)

Routing (which STFT tier feeds each filterbank bin) is **not** the hardcoded `[0]*15+[1]*7+[2]*8` — it is re-derived per cell from the new bin centers, at **iMINDBench-parity crossovers**:

- bin center **< 32 Hz** → low tier (1024)
- **32 ≤ center < 152 Hz** → mid tier (512)
- center **≥ 152 Hz** → hi tier (256)

(32 / 152 Hz ≈ the midpoints of iMINDBench's overlapping bands and the current v14 crossovers.) **Guard:** a bin whose center falls below its tier's 1-cycle floor (low 2 Hz / mid 4 Hz / hi 8 Hz) is **invalid** → dropped (the FE-MULTISTFT-1 dead-bin guard; with f0≥2 this never fires on the low tier). Cell 0 (raw bins) uses iMINDBench's band assignment directly, no routing.

## 7. Compute (the cheap trick)

The three STFTs are **shared across all cells** — only the filterbank re-bin differs (a matmul). So: cache the raw STFT grid **once** (MapInfra, reuse the Multi-STFT precompute), then apply each cell's filterbank on the fly. Six filterbanks over one cached grid, not six precomputes. Logistic fits are CPU-light. Run on DCC (feature precompute is a real job); the fits can be a small follow-on job.

HB02 re-cost applies **only to the final locked config** that enters pretraining (it changes F-bin count) — not to the probe sweep.

## 8. Collector (exists before dispatch — no-sweep-without-a-collector)

Write `docs/experiments/fe_filterbank_sweep_<date>.csv` with one row per (cell × eval × task):
`cell, f0, cap, octave_step, half_bw, n_bins, eval={within,cross}, task, subject, roc_auc`
plus a derived summary table keyed `cell × eval` with **both** aggregations: `rank4_mean_auc` (the {speech, sentence_onset, volume, pitch} headline), `all15_mean_auc` (robustness), and `Δrank4_vs_ctrl`, `Δall15_vs_ctrl` with 95% CI. The filterbank decision is read off `Δrank4`; `Δall15` is a confirm-only column. Append the dispatch row to `docs/experiments/dispatch_log.csv`; clear the loop (analyze + record) before any further sweep.

## 9. Deferred sisters (logged, not run here)

- `R-cqt-frontend` — continuous per-bin window. Hypothesis: helps 2–8 Hz despite the 1 s cap. Low priority (see thread; bet against).
- `R-fe-multiclass` — add a within-session multiclass cell; binary ROC-AUC is the parity instrument, multiclass is downstream.
- Geeling/CNN probe — richer probe; add only if the logistic deltas are ambiguous.
