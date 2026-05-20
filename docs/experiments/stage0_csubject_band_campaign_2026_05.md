# Stage-0 CrossSubject + band campaign (2026-05-13 → 05-18)

Durable record for the 5/13–5/18 sweep campaign. 17 DCC report dirs, 860
`metrics.json`, ran to completion but were never collected or pulled until
2026-05-19. Aggregated by `scripts/neuroprobe/collect_stage0_grid.py`; each
report dir now carries an `aggregate_summary.csv`. All numbers are mean
`test_roc_auc` over sessions (each session = mean over its 15 Neuroprobe
tasks). Multiclass unless noted.

## Normalization × CrossSubject (the "N1 vs N8" question)

`l1_normalization_cross_subject_2026_05_15` runs N0,N2–N7 (R4 × stft_abs);
N1 and N8 live only in the `sn_*` spectral grids.

| grid | view | top cell | top AUROC | N1 | N8 | N8 − N1 |
|---|---|---|---|---|---|---|
| sn_spectral_norm_grid | log_stft | R1_bipolar N2 | 0.5427 | 0.5348–0.5356 | 0.5351–0.5377 | +0.000 to +0.003 |
| sn_stft_abs_norm_grid | stft_abs | R1_bipolar N8 | 0.5458 | 0.5310–0.5392 | 0.5407–0.5458 | +0.004 to +0.010 |
| l1_normalization_csub | stft_abs | N2 per_session_fixed | 0.5417 | — | — | — |

**N8 (per-channel z) beats N1 (pooled StandardScaler) at CrossSubject, but
only under `stft_abs`.** R4 × stft_abs: N8 0.5410 vs N1 0.5310 = **+0.0100** —
exactly the "+1pp" figure in the L.1-freeze memory's 5/15 amendment, now
backed by the formal grid. Under `log_stft` the gap nearly vanishes
(≈ +0.001): log compression already removes most of the per-channel scale
spread that N8 strips. N2 (per-session transductive) tops every grid; N4/none
is the floor (0.522). `sn_log_stft_none` (N4 only) ≈ 0.535 — normalization is
near-no-op once the view is log-compressed.

CrossSubject multiclass linear ceiling from this campaign ≈ **0.546**
(R1_bipolar × stft_abs × N8).

## Reference × view × CrossSubject — `l2_reference_view_cross_subject_2026_05_15`

| rank | cell | AUROC |
|---|---|---|
| 1 | R1 bipolar × stft_abs | 0.5392 |
| 2 | R5 shaft_car × stft_abs | 0.5369 |
| 3 | R4 shaft_lap × log_stft | 0.5353 |
| 7–8 | R4 × instantaneous_phase / raw_voltage | ≈ 0.500 (chance) |

Spectral view is mandatory cross-subject too — raw voltage and phase decode at
chance. Reference choice spans only ~0.02. `tier_c_alts` / `tier_c_n8`
reproduce the same cells (C.3 bipolar_stft 0.5392; C.5 n8 0.5410).

## Band sweeps

| sweep | protocol | finding |
|---|---|---|
| l6_nr_band_sweep_cross_subject | CSubject | monotone decay: 30–70 Hz 0.521 → 300–500 Hz 0.502 (chance). Low bands carry the cross-subject signal. |
| l6_nr_extra_bands | CSession | 30–70 Hz 0.574 > 150–300 Hz 0.565 > 300–500 Hz 0.533 |
| l6_nr_hg_subbands | CSession | 70–90 Hz 0.557 ≳ 90–120 0.556 > 120–150 0.549 |
| l6_nr_merged_bands | both | 70–500 Hz best; CSession 0.59–0.60 vs CSubject 0.51–0.52 — the protocol gap in one sweep |
| envelope_n8_cross_subject | CSubject | HG envelopes under N8 all ≈ 0.52 (weak) |

## L.6.BI band isolation — `l6_band_isolation_cross_session_2026_05_18` (CrossSession, binary)

| rank | cell | AUROC |
|---|---|---|
| 1 | BI.6 | 0.9095 |
| 2 | BI.C1 | 0.9094 |
| 3 | BI.C0 | 0.9076 |
| … | BI.4 / BI.5 | 0.77 |
| 10 | BI.7 (300–1000 Hz spike) | 0.6392 |

CrossSession binary is the easy protocol (0.91 ceiling). The 300–1000 Hz spike
band carries the least isolated signal. Submitter for this sweep is on
`origin/worktree-band-isolation-csession` (commits 41d4a00 / b4aae71 / 474a34b).

## Window length — `l6_wl_window_length_2026_05_13` + `l4_window_sweep_w2_w5_2026_05_13`

1.0–1.5 s optimal (≈ 0.613 CSession multiclass); 0.25 s costs ~5 pp; ≥ 2.0 s no
gain. `l5_p4_prestim` (pre-stimulus window) still decodes at 0.568 — kept as a
leakage probe.

## Provenance

Submitters for the normalization / reference / band-NR / window cells are on
`origin/worktree-band-ablation` (6 commits, unmerged). Band-isolation
submitter on `origin/worktree-band-isolation-csession` (3 commits, unmerged).
Both branches are pushed to origin; nothing is stranded. Source report dirs on
DCC at `/work/ht203/repo/speech/reports/neuroprobe_stage0_*` (75-day purge —
the laptop copies + this digest are the surviving record).
