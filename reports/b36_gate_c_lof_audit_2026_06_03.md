# Gate-C audit — D2 MNE-LOF bad-channel drop (BT) + drop-count reporting

4-agent adversarial audit + 1 focused re-audit, 2026-06-03. Chunk C of the
BT-full-run prep. Branch `bt-full-run-prep`, commits `a2eb93f` → `de92549` →
`f1ba798`.

## Verdict: PASS (after 2 fix rounds)

The first audit round FAILED on a cache-poison showstopper; the re-audit round
FAILED again on an incomplete fix; both are now fixed and re-verified. The
**maiden BT run is LOF-OFF and proven byte-identical** — safe to ship as-is.

| Audit | Round 1 | Round 2 (re-audit) |
|---|---|---|
| #1 cache/exca semantics | **FAIL** (showstopper) | (refined) latent footgun, fixed |
| #2 guards (validator/uid/except) | PASS | — |
| #3 robust-z/CAR/memory interaction | PASS for maiden; 3 LOF-ON flags | — |
| #4 elegance/regression/coverage | PASS (maiden byte-identical) | — |
| fresh re-audit of the fix | — | **PASS** (poison closed) |

## What "report back how many channels are dropped" means now (your MAKE-SURE)

The drop-count reporting you asked me to guarantee is wired and **enforced at
construction**: when `lof_bad_channels=True`, a pydantic `model_validator` now
*requires* both `drop_bads=True` and a `lof_report_path` — you cannot run LOF
without producing the report. On every run the fit logs, per session and
per subject, `N/total channels flagged bad [names]` plus a SUMMARY line with the
overall % dropped, and writes a JSON (`{threshold, n_neighbors, total_bad,
total_channels, by_subject, sessions:[...]}`) to `lof_report_path`. **But no drop
counts exist yet** — the maiden run is LOF-OFF (see below), so the first real
counts arrive only when you green-light a LOF-ON run.

## The showstopper (found + fixed)

`self._get_data` is an **exca-cached property** whose cache uid does NOT include
the runtime `_bads_ready` flag. Three compounding defects, all fixed:

1. **Order (round 1).** `prepare()` ran `super().prepare()` first — which
   materializes `_get_data` for every session — *before* arming the bad sets, so
   it memoized the pre-LOF no-drop raw per session; later apply/robust-z read it
   stale. LOF detected + logged but **never dropped**. Fix: fit + arm bads
   *before* `super().prepare()` so the first materialization already drops.
2. **Shallow copy (round 2).** The fit reads a `car=None` sibling for pre-CAR
   voltage via `self.model_copy(update={"car": None})`. A *shallow* copy shares
   exca's infra (`infra._obj` stays bound to the `car=shaft` parent) → the
   sibling preprocessed *with* CAR (wrong LOF input) and wrote a no-drop entry
   under the production uid → poison reintroduced. Fix: `deep=True` forks the
   infra (verified: sibling uid == an independently-built `car=None` view's).
3. **Memoized uid (round 2 latent).** `deep=True` also deep-copies exca's cached
   uid; if `self.infra.uid()` were computed before the copy (a future pre-build
   uid log would), the sibling silently inherits the `car=shaft` uid. Fix:
   `_reset_infra_uid_cache()` clears the sibling's uid memo after the copy.

Plus: `lof_report_path` excluded from the cache uid (output-location, not
data-determining); `subject_id` resolution narrowed from blanket `except` and now
logs the `-1` fallback. 199 extractor tests pass; new pins:
`test_fit_bads_armed_before_super_prepare`,
`test_precar_sibling_has_distinct_cache_namespace`,
`test_precar_fork_survives_precomputed_uid`, construction-guard pair.

## Maiden run is safe (LOF OFF) — proven, not asserted

`lof_bad_channels` defaults False; the dispatch never sets it. With it off:
`prepare()` skips the LOF block, `_stamp_lof_bads` is a pure no-op, and
`_exclude_from_cache_uid` adds nothing to the uid (`exclude_defaults` drops the
default-off field). The default-off uid hash is **byte-identical** to the
pre-LOF commit — the existing multi-TB STFT cache is not invalidated. BT
`_load_raw` marks no `info["bads"]`, so zero channels drop and there is no
re-pack / singleton / extra-RAM path.

## DECISIONS FOR YOU — blockers before the first LOF-ON scored run

These do **not** affect the maiden run. They block turning LOF on, and all three
touch load-bearing infra (atlas/support, CAR, recipe compute) that is under your
approval gate, so I did not change them unilaterally.

1. **[ATLAS/SUPPORT — HIGH] token↔support positional misalignment after a drop.**
   The dispatch builds the view with `channel_order="original"`, which re-packs
   the electrode token C-dim (closes gaps) when a channel drops. But
   `V14DKHardSupportExtractor` / `ElectrodeValidMask` are built in full voltage
   order and never drop. The collate stacks them position-by-position → every
   electrode at/after a dropped index gets the WRONG DK parcel + valid bit. Fix
   options: (a) make the support/valid extractors consume the same post-drop set
   (share `_session_bads`), or (b) keep tokens full-width (zero-fill + mark
   dropped rows invalid, don't re-pack). This is the one I most want your call on.

2. **[CAR PREPROC — HIGH] singleton-shaft CAR zeroing.** If LOF leaves a shaft
   with 1 survivor, shaft-CAR subtracts its own value → silent all-zero (flat
   STFT). BT exposure is small (smallest clean shaft = 2; only sub_2's 2-contact
   shaft is one drop from a singleton), but real. Proposed fix: in `_apply_car`
   shaft mode, skip subtraction (passthrough) when a shaft has < 2 contacts and
   log it. Cheap + isolated, but it's a CAR-recipe change → your call. (The raw
   `*`/`#` sub_2 singleton bug is already fixed at the loader boundary on this
   branch.)

3. **[RECIPE COMPUTE — HIGH] LOF fit peak RAM ~70–84 GB / largest BT session.**
   The fit loads each full session (≈2 h, up to ~238 ch at 2048 Hz) and
   `quality.py` round-trips to float64 (+ MNE's internal copies). It also adds a
   full extra pass over all sessions. Default sbatch `--mem=24G` (and 64G) OOM;
   needs ≥160G host RAM + likely chunking the LOF load (as robust-z already
   chunks). Per the "recipe amendments need compute re-cost" rule this is an
   HB02 re-estimate before landing LOF-on. Decimation note: keep any RAM-saving
   downsample ≤4× (8×+ shifts the detected bad set off the locked recipe).

## Pre-existing (not introduced here) — FYI

- `CARIeegExtractor._preprocess_raw` applies shaft-CAR BEFORE notch/HPF
  (pick → drop_bads → CAR → filter). Harmless (CAR and linear notch/HPF commute)
  and the LOF fit dodges it via the `car=None` sibling, but the recipe text reads
  "filter → CAR". Flagging, not fixing.
- `lof_ch_type` does not change `find_bad_channels_lof`'s output for any data
  channel type; kept in the cache uid (harmless, over-conservative).

## Nice-to-have (future LOF-on hardening, non-blocking)

- A synthetic end-to-end test that a stamped channel actually disappears from an
  emitted clip (the seam the original bug lived in is proven only by reading code
  + the ordering pin, not an MNE round-trip test).
