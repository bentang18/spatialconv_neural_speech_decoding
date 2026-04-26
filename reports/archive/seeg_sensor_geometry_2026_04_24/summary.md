# A6 — sEEG inter-contact spacing (2026-04-24)

128 D-patients × all shafts, consecutive-contact distances on `<D>_elec_locations_RAS_brainshifted.txt`.

## Pooled distribution (mm, consecutive contacts within a shaft)

- n = 22280 consecutive-contact gaps across 128 patients
- median: **3.51 mm**
- p90: 4.97 mm
- p95: 6.01 mm
- min: 0.0 mm, max: 85.18 mm

## Implication for Stage-3 per-electrode attention

Typical sEEG shaft inter-contact spacing (~3–5 mm) is order-of-magnitude
similar to uECoG 128-strip spacing (~2 mm) but with much larger cross-shaft
gaps. Any Gaussian distance-bias on per-electrode attention should use
bandwidth ≥ p95 of the pooled spacing distribution so neighboring-contact
attention is not suppressed by the kernel. Compare to the MV-BrainFM
14→152 mm head specialization (`memory/reference_mv_brainfm_xu_2026_04.md`)
— bandwidths in the 10–30 mm range cover the typical sEEG shaft; a
long-range head (100+ mm) is still needed for cross-shaft structure.
