"""v3 board-comparable readout — the Neuroprobe-Lite leaderboard number (WS + CS).

Sibling of ``v3_probe_readout_r4.py``: SAME ridge/parcel-pool mechanism on the SAME pooled
caches, so the erosion diagnostic and the board claim are one probe differing ONLY in the eval
universe. Consumes ``v3_probe_encode_r4.py --sessions board --tasks board15 --electrode-set
lite`` (per session a ``(n_win, |P|, F)`` fp16 parcel-mean feature per tap, + 15 label vectors,
+ the board splits).

Four deliberate changes from the diagnostic, each forced by board-comparability:

  1. CS ANCHOR = subject-2 TRIAL-4 — upstream's own ``DS_DM_TRAIN_SUBJECT_ID/TRIAL_ID``
     (config.py:36), verified. The diagnostic used (2,1). (2,4) is NOT in our pretrain, so
     using upstream's exact anchor costs no leakage — it removes a divergence.
  2. CS TEST = the 10 Lite CrossSubject cells = every Lite (s,t) with s != 2. Subjects 7 & 10
     are in NO pretrain session of ours ⇒ a TRUE held-out-subject transfer.
  3. TASKS = the 15 leaderboard tasks (verified == upstream NEUROPROBE_TASKS).
  4. ELECTRODES = the Lite montage, applied at ENCODE (see the encode's --electrode-set lite).

λ — and ONLY λ (Ben 2026-07-17) — is SELECTED ON THE VAL HALF; every number is reported on the
TEST half. Tap (enc0/3/6/12, per-electrode vs parcel-mean) and norm (std/raw/std_target) are
REPORTED AXES: the output is the full grid, one complete protocol per entry scored over every
cell, so nothing is discarded and no cell's protocol is picked by a max across taps or norms.
λ is the exception because it is not a result but a per-fit regularization parameter that must
be set somehow — and the val half is upstream's own design for exactly that, not a divergence
of ours. ``generate_splits_cross_subject`` (train_test_splits.py:65)
and ``generate_splits_within_session`` BOTH carve the eval set in half:
    test_size = len(test_dataset); val_size = test_size // 2
    val = range(0, val_size)   test = range(val_size, test_size)
The val half comes out of the TEST session, NOT the anchor — so selecting on it costs the fit
set ZERO rows. (An earlier draft of this file chose GCV-on-the-anchor to avoid "wasting"
training samples; that rationale was a misreading — there is nothing to waste.) The reported
test half is never fit on and never selected on.

``build_session_targets`` already emits exactly these halves (``ws_split[task][fold]`` →
train/val/test, ``cs_split[task]`` → val/test), so the splits are upstream-faithful by
construction and this file just has to USE the val key.

Ridge/metric primitives INLINED — zero speech_decoding dep, runs on the stock NCSA Delta
pytorch module. fp32 GEMM into a fp64 n×n eigendecomposition (a fp64 z_anchor at |P|·F is
multi-GB and sends the array NUMA-bound). ONE eigh per (task, cell, tap, norm) serves the whole
λ grid: the ridge smoother's eigenvalues are w/(w+λ), so every λ reuses the same (w, V, c) and
costs one O(n²) mat-vec instead of a fresh O(n³) solve.

Usage (CPU, NCSA Delta; --mem>=64G):
  python scripts/neuroprobe/v3_board_readout.py \
      --cache-dir /projects/bhqk/htang13/v3_board_cache --tags board_r4_20k \
      --mode all --out results_v3_board.json
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np
import torch

# Reserved DKT id for an electrode the atlas could not place (== len(parcel vocabulary)).
UNKNOWN_PARCEL_ID = 74

# ── phase timing ───────────────────────────────────────────────────────────────────
# The 20263259 array ran shards at 98-581% of 8 cores: some were single-threaded for
# 20 min while others sat in threaded BLAS. I diagnosed that serial phase twice from
# the armchair (torch.load; then the fp16->fp32 gather) and was WRONG BOTH TIMES —
# read_bytes showed the file fully read while state=R, and the gather measured 4.8
# min/shard. So the code now MEASURES instead of me guessing: every shard prints its
# own phase budget, and the next optimization argues from this table or not at all.
_PH: dict = {}


@contextlib.contextmanager
def _timed(name):
    t = time.perf_counter()
    try:
        yield
    finally:
        _PH[name] = _PH.get(name, 0.0) + (time.perf_counter() - t)


def _phase_report(label) -> None:
    tot = sum(_PH.values()) or 1.0
    print(f"[phase] {label} — wall budget by phase (total {tot / 60:.1f} min):", flush=True)
    for k, v in sorted(_PH.items(), key=lambda kv: -kv[1]):
        print(f"[phase]   {k:22s} {v / 60:7.2f} min  {100 * v / tot:5.1f}%", flush=True)

# The 15 Neuroprobe leaderboard tasks — verified == upstream neuroprobe.config.NEUROPROBE_TASKS.
BOARD_TASKS = (
    "onset", "speech", "volume", "delta_volume", "pitch", "word_index",
    "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)
# Upstream DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID (config.py:36) — verified.
CS_TRAIN_ANCHOR = (2, 4)
# Every Lite (s,t) with s != 2 — upstream generate_splits_cross_subject asserts test != anchor.
CS_TEST_CELLS = ((1, 1), (1, 2), (3, 0), (3, 1), (4, 0), (4, 1),
                 (7, 0), (7, 1), (10, 0), (10, 1))
# The 12 Lite sessions — verified == upstream NEUROPROBE_LITE_SUBJECT_TRIALS.
LITE_SESSIONS = ((1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
                 (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1))
ENCODERS = ("enc0", "enc3", "enc6", "enc9", "enc12")  # parcel-mean (electrodes pooled at encode)
ELEC_TAPS = ("enc0_elec", "enc12_elec")   # per-electrode: enc0_elec = depth-0 parity FLOOR for enc12_elec
ALL_TAPS = ELEC_TAPS + ENCODERS           # universe for --taps validation
# Per-regime feature UNIT (Ben 2026-07-19): compute only the unit that is defensible for each
# regime, so no column is a distraction on the leaderboard grid.
#   WS / CSession → ELECTRODE-level only. The same electrodes are available (WS trivially;
#     CSession via elec_labels identity-intersect within subject), so the per-electrode ladder
#     is the meaningful unit — parcel-mean would only discard spatial resolution here.
#   CSubject      → PARCEL-mean only. Electrode identity is NOT shared across subjects, so the
#     anatomical parcel bridge is the only cross-subject-valid unit.
WS_TAPS = ELEC_TAPS
CS_TAPS = ENCODERS
CSESSION_TAPS = ELEC_TAPS
CSESSION_CELLS = LITE_SESSIONS            # every Lite session is a cross-session test cell, sibling-trained


def _sibling(cell):
    """The other Lite trial of the SAME subject — upstream cross_session's train trial
    (train_test_splits.py:146: the one other NEUROPROBE_LITE_SUBJECT_TRIALS entry for the subject)."""
    s, _ = cell
    sibs = [c for c in LITE_SESSIONS if c[0] == s and c != cell]
    assert len(sibs) == 1, f"subject {s} must have exactly one sibling Lite trial, got {sibs}"
    return sibs[0]
# std_target (per-domain/AdaBN CS column) is REPORTED by default (preserves the r4 board), but
# Ben's r5mod board is raw+std ONLY — toggle off with --no-std-target.
REPORT_STD_TARGET = True

# Back-fill electrode labels for caches encoded before ``elec_labels`` was stored (e.g. arm0/r4b).
# Labels are a pure function of (channels, drop-set), independent of weights (session_setup.py:114/
# 119), so a sibling cache's labels are valid for a same-set session — proven per session by
# count + present_parcels equality in build_arm0_label_sidecar.py before the sidecar is written.
# {"s{S}_t{T}": np.ndarray[str]}; attached in _load only when the record lacks its own.
_ELEC_LABELS_SIDECAR: dict | None = None
NORMS = ("std",)   # std-only default (Ben 2026-07-20); raw retired
# norm is a REPORTED axis, not a selected one (Ben 2026-07-17): both columns are computed over
# every cell and both are printed. Measured on our own r4 20k probe (results_v3_probe_r4_20k.json,
# 16 paired tap×task cells): std beats raw 16/16 WS (meanΔ +0.0277) but only 9/16 CS (meanΔ
# +0.0026, median +0.0007 — a coin flip), and the whole CS edge is carried by enc0 (the already-
# normalized input floor, +0.0281): every real encoder tap mildly PREFERS raw in CS (enc3 −0.0046,
# enc6 −0.0091 @ 25% std-wins, enc12 −0.0041). Mechanism for the regime split: WS fits μ/σ on the
# same session it scores, so the scaler is valid; CS fits μ/σ on the ANCHOR and applies it to a
# DIFFERENT subject, where a scale mismatch makes z-scoring actively harmful. That is a RESULT,
# and val-selecting over it would bury it — and would make the headline a per-cell mixture of two
# protocols. CS additionally reports "std_target" (per-domain/AdaBN; see _standardize_per_domain).
# (Upstream's pip package ships no baseline readout, so nothing external forces a choice.)
# λ multipliers on base = trace(G)/n.
#
# ⚠️ WIDENING THIS IS A LIVE, UNDECIDED QUESTION (07-29) — DO NOT WIDEN WITHOUT READING THIS.
# The pin audit found LO-pinning is asymmetric across taps (ws enc12 39 vs enc0 16; csession 14 vs
# 1), which SUGGESTS reported depth gains are lower bounds. But that inference is CONFOUNDED, and a
# trial widening to logspace(-8, 4, 37) measured the confound directly:
#   • _select_lam takes argmax with a strict `>` while iterating this tuple in ASCENDING order, so
#     on a val TIE it silently keeps the SMALLEST λ — the selected λ then depends on how far down
#     this grid runs, not on the data.
#   • Ties are the COMMON case, not the exception: on the standard test fixture 32 of 37 λ tie at
#     val=1.0, spanning 4.6e-7..1e4, and their TEST AUROCs range 0.906..0.969.
#   ⇒ a `lam_pinned` cell may be a TIE resolved to the floor rather than an optimum below it, and
#     widening moves numbers in cells where the data expressed NO preference, in either direction.
# So widening alone is not sufficient and not safe on the headline table. Measure the tie fraction
# first (see test_val_ties_make_the_selected_lambda_depend_on_the_GRID_not_the_DATA).
# If it is widened later: keep the 1/3-decade spacing so the old 25 points survive exactly, and
# stop at 1e-8 — G is a fp32 GEMM cast to fp64, so eigenvalues below ~1e-7·base are numerical
# noise and a pin surviving past that means the cell is DEGENERATE, not that the optimum is lower.
LAM_MULTS = tuple(np.logspace(-4.0, 4.0, 25))


def auroc(scores, labels) -> float:
    """Verbatim from online_probe.auroc. NaN if the eval half is single-class."""
    from sklearn.metrics import roc_auc_score

    y = (np.asarray(labels) > 0).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores)))


def _finite(y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    r = np.asarray(rows, dtype=np.int64)
    return r[np.isfinite(y[r])]


def _standardize(z_tr, others):
    """Per-feature z-score on TRAIN stats only (never fit on val/test). σ=0 → 1.

    This is the canonical frozen-FM linear probe: the MAE/MoCo-v3/DINO lineage puts a
    BatchNorm WITHOUT affine before the linear head, and BN at eval uses statistics
    accumulated over probe TRAINING — i.e. train-set stats, which in CS means the ANCHOR's.
    """
    with _timed("standardize"):
        mu = z_tr.mean(axis=0)
        sd = z_tr.std(axis=0)
        sd[sd == 0] = 1.0
        return (z_tr - mu) / sd, [(z - mu) / sd for z in others]


# Columns per standardize pass. The reduction is PER COLUMN, so blocking the column axis cannot
# change any number — it only changes what is resident while the number is computed. Measured on
# Delta (n=1750, d=120000, tr+val+test, OMP=8), whole-array vs blocked: 6.39 s -> 0.70 s @512,
# 0.73 s @1024, 0.75 s @2048, 0.99 s @4096, i.e. a ~9x plateau that decays once a block leaves L3.
# 1024 sits mid-plateau at a 7 MB train-block footprint (+3.5 MB each for val/test).
# FLOOR OF 2: at blk=1 the slice is a contiguous 1-D vector and numpy sums it PAIRWISE instead of
# accumulating a SIMD row-vector across columns — a different summation order (drift 1.2e-7), which
# would break the bitwise guarantee. Every width >= 2 is exact; see test_v3_board_readout.py.
_STD_BLOCK = 1024


def _standardize_inplace(z_tr, others, blk=_STD_BLOCK):
    """Bit-identical to _standardize but MUTATES its inputs — no second copy of the design
    matrix ever exists. `z -= mu; z /= sd` is the same two fp32 ufuncs in the same order as
    `(z - mu) / sd`, so every downstream number is unchanged; the only difference is that the
    ~tens-of-GB std copy is never allocated. That copy is what pushed heavy CSession cells (two
    sessions' design matrices resident at once) past the node memory cap into thrash.

    COLUMN-BLOCKED, and the reason is measured (07-27): `standardize` was 81% of an eager WS
    shard (68.2 of 83.7 min, 20298989_0) and 10-39% of every CSession shard. The whole cost is
    memory traffic, not arithmetic — `np.std(axis=0)` materializes its own (n, d) fp32 `(z-mu)**2`
    temp, which at enc12_elec is an 11 GB allocation streamed once more on top of the four passes
    the mean/std/subtract/divide already owe. Blocking the COLUMN axis keeps each block's train +
    val + test slices in L3 across all four passes and shrinks that temp from (n, d) to (n, blk)
    — 11 GB down to ~7 MB. Measured ~9x on the phase, and it lowers the shard's transient peak,
    which is the same pressure that forces --mem=176G.

    Every reduction is along axis 0 and therefore independent per column, so chunking columns
    sums exactly the same values in exactly the same order: the result is BIT-identical, asserted
    against the whole-array path in test_v3_board_readout.py.

    CONSUMES z_tr and every array in `others` — the caller must run any norm that needs the RAW
    features (raw, std_target) BEFORE this one. `_run_norms` enforces that ordering.
    """
    with _timed("standardize"):
        d = z_tr.shape[1]
        edges = list(range(0, d, blk)) + [d]
        if len(edges) > 2 and edges[-1] - edges[-2] == 1:
            edges.pop(-2)                      # never leave a WIDTH-1 trailing block (see floor above)
        for lo, hi in zip(edges, edges[1:]):
            b = z_tr[:, lo:hi]
            mu = b.mean(axis=0)
            sd = b.std(axis=0)
            sd[sd == 0] = 1.0
            b -= mu
            b /= sd
            for z in others:
                zb = z[:, lo:hi]
                zb -= mu
                zb /= sd
        return z_tr, others


def _standardize_per_domain(z_tr, z_va, z_te):
    """CS ablation: each SUBJECT z-scored in its OWN frame (AdaBN-style target adaptation).

    Anchor uses anchor stats (identical to _standardize's train side, so G/eigh/α are unchanged);
    the target subject uses the target's OWN stats — fit on its VAL half ONLY and applied to both
    val and test, so no test-half statistic is ever touched. Motivation: plain std maps the target
    into the ANCHOR's coordinate frame ((z_t − μ_a)/σ_a), which is wrong exactly when subjects'
    feature scales differ — i.e. in the CS regime the probe is measuring. Per-subject target
    normalization is standard in cross-subject EEG/BCI transfer (AdaBN; the Euclidean-Alignment
    family), though NOT part of the canonical vision linear probe.

    REPORTED, NEVER SELECTED (Ben 2026-07-17): it is a third CS norm column on the same footing
    as std/raw, scored over every cell. It is a DIFFERENT CLAIM from the other two — "transfer
    GIVEN target statistics" rather than "frozen features transfer" — which is precisely why it
    must not be fused into them by an argmax: that would make the number target-adapted on some
    cells and not others.

    Rules check: SUBMIT.md constrains only (1) splits from train_test_splits.py and (2) no
    pretraining on eval data — nothing about normalization. Fitting an UNSUPERVISED scaler on the
    val half uses strictly less information than the λ/tap selection upstream already sanctions
    on the same half.
    """
    mu_a, sd_a = z_tr.mean(axis=0), z_tr.std(axis=0)
    sd_a[sd_a == 0] = 1.0
    mu_t, sd_t = z_va.mean(axis=0), z_va.std(axis=0)   # target stats: VAL half only
    sd_t[sd_t == 0] = 1.0
    return (z_tr - mu_a) / sd_a, [(z_va - mu_t) / sd_t, (z_te - mu_t) / sd_t]


CAT = "cat:"          # virtual tap: "cat:enc6+enc9+enc12" — depth-concatenated features

# ── TIME POOLING ──────────────────────────────────────────────────────────────────────────────
# The cache stores (n, |P|, k_full·d): per unit, k_full TIME TOKENS of width d, flattened whole.
# On the Lite board that is 52·256 = 13312 per electrode, so a WS fit carries 119·13312 = 1.58M
# features against ~1750 train rows — p/n ~ 900. The MAE/V-JEPA convention is the opposite: mean
# -pool the final-layer tokens and fit d. These prefixes make that an option instead of a rewrite.
#
#   gpool:  mean over ALL k_full tokens        -> d per unit          (the MAE convention)
#   bpool:  mean WITHIN each band, concatenated -> n_bands·d per unit
#
# bpool exists because the token axis is NOT homogeneous: pack_r4 lays tokens out per contact as
# [SLOW; MID; HGA] with per-band rates (4/16/32 at 1 s), so a global mean is dominated by the band
# with the most tokens -- HGA outnumbers SLOW 8:1 -- and silently reweights the bands whose
# separate behaviour is the whole point of the per-band architecture. bpool keeps band identity.
#
# NOT combinable with cat:. Both rewrite the feature axis, and stacking them raises the question
# of whether to pool before or after the hstack; nothing needs the combination, so it is refused.
POOL_PREFIXES = ("gpool:", "bpool:")


def _pool_spec(tap):
    """(prefix, base_tap) — ('', tap) when `tap` carries no pooling prefix."""
    for p in POOL_PREFIXES:
        if tap.startswith(p):
            return p, tap[len(p):]
    return "", tap


# ── fm: frontend-ablation column masks (2026-08-01) ────────────────────────────────
# A virtual tap `fm:<arm>:<enc0 tap>` fits the SAME frozen ridge on a SUBSET of the enc0
# spectrogram's feature axis. It answers ONE question: which axis of the multirate |STFT|
# front end carries the zero-parameter enc0 result -- band presence, HGA temporal rate, or
# HGA frequency resolution. Nothing else moves: same splits, same standardize, same λ grid,
# same val-only selection. An arm is a REPORTED grid key like any tap, never a selected one,
# so the published `enc0_elec|std` path is bit-identical whether or not arms are requested.
#
# LAYOUT. enc0/enc0_elec store, per unit, the three bands concatenated TIME-MAJOR
# (v3_probe_encode_r4.py:367-400): [SLOW T=4 x F=7 | MID T=16 x F=6 | HGA T=32 x F=7] = 348
# at 1 s. T_b and F_b are read from the record's OWN `band_lengths`/`band_fdims` and asserted
# against the stored width -- never hardcoded, because F_b is not recoverable from the
# flattened width alone (viz_reduce.py:56-68 makes the same point for the same cache).
#
# ENC0 ONLY. Encoder taps are (k tokens x d 256); they have no frequency axis, so the same
# arm name would mean a different operation at each tap. Refused at parse time.
FM = "fm:"
_KEEP = (1, "all")          # keep every frame, every bin
# arm -> per-band (time_stride, freq_mode) in [SLOW, MID, HGA] order; None DROPS the band.
#   time_stride : keep frames ``[::stride]`` -- the band's own token rate, coarsened
#   freq_mode   : "all" keeps the band's bins, "mean" averages them to ONE channel
FEAT_ARMS: dict = {
    "full":   (_KEEP, _KEEP, _KEEP),        # == the plain tap; bit-identity asserted by test
    # band necessity -- where does the information live?
    "slow":   (_KEEP, None, None),
    "mid":    (None, _KEEP, None),
    "hga":    (None, None, _KEEP),
    "noslow": (None, _KEEP, _KEEP),
    "nomid":  (_KEEP, None, _KEEP),
    "nohga":  (_KEEP, _KEEP, None),
    # HGA temporal rate (32 Hz native) -- the multirate hypothesis, stated as a ladder
    "hgat16": (_KEEP, _KEEP, (2, "all")),
    "hgat8":  (_KEEP, _KEEP, (4, "all")),
    "hgat4":  (_KEEP, _KEEP, (8, "all")),
    "hgat1":  (_KEEP, _KEEP, (32, "all")),
    # HGA frequency resolution -- 7 bins collapsed to one envelope, time axis untouched
    "hgaf1":  (_KEEP, _KEEP, (1, "mean")),
    # is "match the rate to the band" general, or is it HGA-specific?
    "midt4":  (_KEEP, (4, "all"), _KEEP),
    # every band at the SLOW rate: the single-rate straw man inside our own bands
    "allt4":  (_KEEP, (4, "all"), (8, "all")),
}
FM_TAPS = ("enc0", "enc0_elec")   # the only taps whose feature axis is a spectrogram


def _fm_spec(tap):
    """(arm, base_tap) — ('', tap) when `tap` carries no `fm:` prefix."""
    if not tap.startswith(FM):
        return "", tap
    arm, sep, base = tap[len(FM):].partition(":")
    if not sep:
        raise SystemExit(f"{tap!r} names no tap — spell it '{FM}<arm>:<tap>'")
    return arm, base


def _fm_apply(x, arm, band_lengths, band_fdims):
    """(r, U, F) fp32 → (r, U, F') under `arm`. Bands are sliced from the record's own layout.

    The width assertion is the invariant: if `sum_b T_b*F_b` does not equal the stored width
    the layout does not describe this cache, and every masked column after it would be a
    wrong-but-plausible number rather than a crash.
    """
    spec = FEAT_ARMS[arm]
    tl = [int(t) for t in band_lengths]
    fd = [int(f) for f in band_fdims]
    if len(tl) != len(fd) or len(spec) != len(tl):
        raise SystemExit(f"{FM}{arm}: {len(spec)} band slots vs band_lengths {tl} / fdims {fd}")
    total = sum(t * f for t, f in zip(tl, fd))
    if total != x.shape[-1]:
        raise SystemExit(f"{FM}{arm}: sum_b T_b*F_b = {total} != stored width {x.shape[-1]} — "
                         f"band_lengths {tl} / band_fdims {fd} do not describe this cache")
    lead, out, off = x.shape[:-1], [], 0
    for (t_b, f_b), s in zip(zip(tl, fd), spec):
        blk = x[..., off:off + t_b * f_b].reshape(lead + (t_b, f_b))
        off += t_b * f_b
        if s is None:
            continue
        stride, fmode = s
        if t_b % stride:
            raise SystemExit(f"{FM}{arm}: stride {stride} does not divide band frames {t_b}")
        blk = blk[..., ::stride, :]
        if fmode == "mean":
            blk = blk.mean(axis=-1, keepdims=True)
        out.append(blk.reshape(lead + (-1,)))
    if not out:
        raise SystemExit(f"{FM}{arm}: drops every band — nothing left to fit")
    return np.concatenate(out, axis=-1)


def _fm_width(arm, band_lengths, band_fdims) -> int:
    """Columns per unit `arm` leaves — the width printed next to each arm's result."""
    spec, out = FEAT_ARMS[arm], 0
    for (t_b, f_b), s in zip(zip(band_lengths, band_fdims), spec):
        if s is None:
            continue
        stride, fmode = s
        out += (int(t_b) // stride) * (1 if fmode == "mean" else int(f_b))
    return out


def _cat_parts(tap):
    """The taps a virtual `cat:` tap is built from, or () if `tap` is an ordinary tap."""
    tap = _pool_spec(_fm_spec(tap)[1])[1]
    return tuple(tap[len(CAT):].split("+")) if tap.startswith(CAT) else ()


def _base_tap(tap) -> str:
    """The cached tap under any combination of virtual prefixes."""
    return _pool_spec(_fm_spec(tap)[1])[1]


def _have(rec, tap) -> bool:
    """Is `tap` (ordinary, `cat:`, pooled or `fm:`) computable from this record's features?"""
    return all(p in rec["feats"] for p in (_cat_parts(tap) or (_base_tap(tap),)))


def _validate_taps(taps) -> None:
    """Refuse a bad tap at PARSE time, naming the flag the user typed.

    A `cat:` tap must (a) name only known taps and (b) keep to ONE unit. Mixing e.g.
    `enc12_elec` with `enc6` would hstack an (n, |E|·F) block onto an (n, |P|·F) one and then
    index both with a single `col_idx` — the parcel intersection in cs, the electrode-label
    intersection in csession — silently gathering the wrong columns from one of them. That is a
    WRONG NUMBER, not a crash, so it is refused here rather than discovered in a shard.
    A single-part cat is refused too: it is just the plain tap under a second name, and would
    split one result across two grid keys.
    """
    for t in taps:
        arm, rest = _fm_spec(t)
        if arm:
            # An fm: arm is defined over the enc0 spectrogram layout ONLY, and it rewrites the
            # feature axis exactly as cat:/pool do — so, for the same reason those two are
            # refused together, it may not stack with either.
            if arm not in FEAT_ARMS:
                raise SystemExit(f"unknown arm {arm!r} in {t!r}; choose from "
                                 f"{tuple(FEAT_ARMS)}")
            if rest not in FM_TAPS:
                raise SystemExit(f"{t!r} masks {rest!r}, but {FM!r} arms are defined over the "
                                 f"enc0 spectrogram layout only — choose from {FM_TAPS} "
                                 f"(encoder taps are k tokens x d and have no frequency axis)")
        pre, base = _pool_spec(rest)
        if pre and base.startswith(CAT):
            raise SystemExit(f"{t!r} combines {pre!r} with {CAT!r}; both rewrite the feature axis "
                             f"and nothing needs the combination — pool a single tap")
        parts = _cat_parts(t)
        if not parts:
            if base not in ALL_TAPS:
                raise SystemExit(f"unknown tap {base!r} in {t!r}; choose from {ALL_TAPS}, "
                                 f"'{CAT}a+b' over them, or one of {POOL_PREFIXES} on one of them")
            continue
        unknown = [p for p in parts if p not in ALL_TAPS]
        if unknown:
            raise SystemExit(f"unknown taps {unknown} inside {t!r}; choose from {ALL_TAPS}")
        if len(parts) < 2:
            raise SystemExit(f"{t!r} concatenates one tap — drop the '{CAT}' prefix")
        if len(set(parts)) != len(parts):
            raise SystemExit(f"{t!r} repeats a tap; duplicate columns only inflate the width")
        if len({p in ELEC_TAPS for p in parts}) != 1:
            raise SystemExit(f"{t!r} mixes electrode and parcel taps, which live in different "
                             f"column spaces ({ELEC_TAPS} vs {ENCODERS}) — a single col_idx "
                             f"cannot index both, so this would silently gather wrong columns")


def _is_elec(tap) -> bool:
    """Does `tap` live in the ELECTRODE index space? A `cat:` tap inherits its parts' unit, and
    `_validate_taps` has already refused a mixed one. Written over `_cat_parts(tap) or (tap,)`
    rather than `all(...)` alone because `all()` over the empty parts tuple of an ordinary PARCEL
    tap is vacuously True, which would route it down the electrode branch. Pooling collapses the
    FEATURE axis and never the unit axis, so a pooled tap keeps its base tap's unit -- and an
    `fm:` mask is a feature-axis mask for the same reason."""
    return all(p in ELEC_TAPS for p in (_cat_parts(tap) or (_base_tap(tap),)))


# Token layout of one unit's cached feature block, needed to pool the time axis. Set from
# --pool-d / --pool-bands; ASSERTED against the real width on first use rather than trusted, and
# printed, so a wrong layout is a loud failure instead of a silently mis-shaped mean.
POOL_D = 256
POOL_BANDS: tuple = (4, 16, 32)      # SLOW, MID, HGA tokens at 1 s (pack_r4 order: [SLOW;MID;HGA])
_POOL_ANNOUNCED: set = set()


def _pool_feats(x, pre, tap):
    """(r, U, k·d) fp32 → pooled (r, U, ·). `pre` is a POOL_PREFIXES member.

    The reshape is the whole risk: k and d are NOT recoverable from the cached width alone, so a
    wrong --pool-d silently means the mean is taken over the wrong axis and every downstream
    number is wrong-but-plausible. Hence the width assertion and the one-time print.
    """
    d, bands = POOL_D, POOL_BANDS
    f = x.shape[-1]
    if f % d:
        raise SystemExit(f"--pool-d {d} does not divide the {tap!r} feature width {f}")
    k = f // d
    if pre == "bpool:" and sum(bands) != k:
        raise SystemExit(f"--pool-bands {bands} sums to {sum(bands)} but {tap!r} has {k} tokens "
                         f"(width {f} / d {d}) — the band split does not describe this cache")
    x = x.reshape(x.shape[0], x.shape[1], k, d)
    if pre == "gpool:":
        out = x.mean(axis=2)
    else:
        cuts = np.cumsum((0,) + tuple(bands))
        out = np.concatenate([x[:, :, cuts[i]:cuts[i + 1], :].mean(axis=2)
                              for i in range(len(bands))], axis=2)
    if tap not in _POOL_ANNOUNCED:
        _POOL_ANNOUNCED.add(tap)
        print(f"[check] {pre}{tap}: {f} = {k} tokens x d {d} -> {out.shape[-1]} per unit "
              f"({'mean over all tokens' if pre == 'gpool:' else f'per-band means {tuple(bands)}'})"
              f"  [{f / out.shape[-1]:.1f}x reduction]", flush=True)
    return out


def _feat(rec, enc, rows, col_idx=None) -> np.ndarray:
    """(n,|P|,F) fp16 cache → rows (and optionally parcel columns) → flat fp32 (r, ·).

    Under --mmap this is where the cache is actually READ: the gather touches only the
    requested rows' pages, so a tap never gathered is never paged in at all.

    DEPTH CONCATENATION. `enc` may be a virtual `cat:a+b+c` tap, which hstacks the parts along
    the feature axis. This is the layer-combination readout (SUPERB fits a weighted sum over
    layers; a ridge over the concatenation is its unconstrained form, and the λ-selection already
    in this grid is what keeps the extra width from simply overfitting). It stays a LINEAR probe
    on FROZEN features fit PER TASK, so it does not spend the protocol parity a leaderboard entry
    needs -- unlike a multitask driver, which does.

    `col_idx` applies to EVERY part, which is why a cat may not mix units: parcel taps are indexed
    by the anchor∩test parcel intersection and electrode taps by the label intersection, and the
    two index spaces are unrelated. Validated at parse time (`_validate_taps`), not here, so the
    error names the flag the user typed.
    """
    arm, enc = _fm_spec(enc)
    parts = _cat_parts(enc)
    if parts:
        return np.hstack([_feat(rec, p, rows, col_idx) for p in parts])
    pre, base = _pool_spec(enc)
    with _timed("gather_fp16"):
        x = rec["feats"][base]["raw"][np.asarray(rows, dtype=np.int64)]
        if col_idx is not None:
            x = x[:, np.asarray(col_idx, dtype=np.int64)]
    with _timed("to_fp32"):
        x = x.to(torch.float32).numpy()
        # Mask BEFORE the flatten, for the same reason pooling is: after reshape(r,-1) the
        # unit and (time,freq) axes are indistinguishable and a mask over the wrong one would
        # still produce a well-shaped array.
        if arm:
            with _timed("fm_mask"):
                x = _fm_apply(x, arm, rec["band_lengths"], rec["band_fdims"])
        # Pool BEFORE the flatten: after reshape(r,-1) the unit and token axes are indisting-
        # uishable, so a mean over the wrong one would still produce a well-shaped array.
        if pre:
            with _timed("pool"):
                x = _pool_feats(x, pre, base)
        return x.reshape(x.shape[0], -1)


# --rbf: report an RBF KERNEL RIDGE column beside the linear one, fit on the SAME standardized
# matrices, the same splits and the same λ grid. It is a REPORTED column, never a selected one --
# "std" stays exactly what it is today, so the control is the untouched published path rather than
# a re-run that has to be argued to be equivalent.
#
# WHY THIS IS THE ONE SINGLE-MODEL LEVER LEFT. Everything that stays inside one model has come back
# null (depth concat, time pooling, tap/λ ensembles, weight-averaged fine-tunes); the only thing
# that has ever moved the board is rank-averaging N models, which is an ensemble and not defensible
# as an entry. A kernel ridge is still ONE model, ONE closed-form fit, ONE forward pass, and the
# nonlinearity is the one axis the linear probe has never been allowed to use. Protocol parity is
# unaffected: it is fit PER TASK on FROZEN features, and Neuroprobe's own baselines (CNN, PopT) are
# nonlinear decoders, so this spends none of the parity a multitask driver would.
RBF = False
# γ = mult / median(d²_train). The median heuristic is what makes a fixed grid meaningful across
# taps whose widths span 41k (enc0_elec) to 1.6e6 (enc12_elec) -- an absolute γ would be a
# different model at every tap. Only TRAIN distances set the scale, exactly as λ's `base` is
# anchor-side only, so no eval set can enter through the bandwidth.
GAMMA_MULTS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
# Which γ the val half actually picked, accumulated over every fit and printed once per shard.
# A grid whose selections pile up on an END is a grid that is too narrow to have answered the
# question; that has to be visible in the log, not inferred later from the AUROCs.
_RBF_STATS: dict = {"fits": 0, "picked": {}, "conc": []}
# Top-k PC projections to ALSO report, each as its own '<tap>|std_rbfpc<k>' column. 32 because the
# separability work measured that keeping the 32 highest-variance directions PRESERVES held-out
# performance -- so if the content really does live there, this is the dimension at which an RBF
# kernel can still tell two trials apart. The neighbours bracket it rather than trusting it.
RBF_PCS = (16, 32, 64)


def _pc_project(g, kern_lin, k):
    """Top-k principal components, STRAIGHT OUT OF THE GRAM WE ALREADY HAVE. No extra GEMM.

    Z = U S Vᵀ ⇒ G = Z Zᵀ = U S² Uᵀ, so the train scores on the top-k right-singular directions
    are ZV = U S, and an eval block projects as Z_e V = (Z_e Zᵀ) U S⁻¹ = K_e U S⁻¹. Both are
    read off the eigendecomposition of G and the eval kernels — the two things `_linear_grams`
    already returns — so denoising costs an n×n eigh, not a pass over the 1.6e6-wide features.

    WHY IT IS HERE: the RBF column's one failure mode is distance concentration, and the cure for
    concentration is to stop measuring distance in directions that carry no signal. Projecting to
    the leading PCs is the cheapest honest way to do that, and it is ANCHOR-SIDE ONLY — V comes
    from the train Gram, so no eval row informs the basis it is projected onto.

    Returns (None, None) when the Gram has no usable spectrum.
    """
    w, u = np.linalg.eigh(g)
    order = np.argsort(w)[::-1][:k]
    w_k, u_k = w[order], u[:, order]
    if w_k.size == 0 or w_k[0] <= 0:
        return None, None
    keep = w_k > w_k[0] * 1e-10          # drop numerically-null directions, never invert them
    w_k, u_k = w_k[keep], u_k[:, keep]
    if w_k.size == 0:
        return None, None
    s = np.sqrt(w_k)
    return u_k * s[None, :], {name: (ke @ u_k) / s[None, :] for name, ke in kern_lin.items()}


def _lam_grid_rbf(z_tr, y_tr, evals, grams=None, n_pc=None):
    """RBF kernel ridge over the same λ grid, with γ selected on val.

    `n_pc` first projects to the top-n_pc principal components of the TRAIN Gram (see
    _pc_project). That is the concentration-cure column; n_pc=None is the raw-feature one.

    NEARLY FREE, BY ALGEBRA. Squared distances are a rearrangement of the products the linear fit
    already forms: d²(i,j) = ||z_i||² + ||z_j||² − 2 z_i·z_j, and the diagonal of G supplies the
    norms. So given `grams` the whole γ grid costs one exp() and one n×n eigendecomposition per γ
    -- against a GEMM that is four orders of magnitude larger. K = exp(−γ d²).

    γ AND λ ARE SELECTED JOINTLY ON VAL, and the nesting below IS that joint argmax:
    max over (γ,λ) == max over γ of (max over λ). Returning the winning γ's FULL λ curve and
    letting _select_lam take its argmax therefore picks exactly the point a flat sweep would,
    while keeping the return shape, the tie census and the lam_pin diagnostics identical to the
    linear path. ⚠️ Selecting on val needs no defense -- λ, and every entry's early stopping,
    already do. Selecting on TEST would, and nothing here reads test.

    INVARIANT, ASSERTED NOT ASSUMED: diag(K) == exp(0) == 1 ⇒ trace(K)/n == 1 exactly, so `base`
    is 1 and a λ multiplier means the same absolute shrinkage at every γ. Without that the γ and λ
    axes would be entangled and the grid would not be a grid.
    """
    nan = {name: {m: float("nan") for m in LAM_MULTS} for name in evals}
    if len(y_tr) < 2:
        return nan
    g, kern_lin = _linear_grams(z_tr, evals) if grams is None else grams
    if g.shape[0] < 2:
        return nan
    if n_pc:
        p_tr, p_e = _pc_project(g, kern_lin, n_pc)
        if p_tr is None or p_e is None:
            return nan
        g = np.asarray(p_tr @ p_tr.T, dtype=np.float64)
        kern_lin = {name: np.asarray(p_e[name] @ p_tr.T, dtype=np.float64) for name in evals}
        ev_sq = {name: np.einsum("ij,ij->i", p_e[name], p_e[name], dtype=np.float64)
                 for name in evals}
    else:
        ev_sq = {name: np.einsum("ij,ij->i", z, z, dtype=np.float64)
                 for name, (z, _) in evals.items()}
    n = g.shape[0]
    sq = np.diag(g).copy()
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    np.maximum(d2, 0.0, out=d2)                                 # kill fp64 round-off below zero
    iu = np.triu_indices(n, k=1)
    med = float(np.median(d2[iu]))
    if not np.isfinite(med) or med <= 0.0:
        # Every train row identical ⇒ no length scale exists. Report NaN rather than invent one.
        return nan
    # THE PRECONDITION FOR THE WHOLE COLUMN, recorded per fit rather than argued. An RBF kernel
    # can only separate points its distances distinguish; as the number of UNINFORMATIVE
    # directions grows, every pairwise distance converges to the same value, K flattens toward
    # constant and the fit degenerates to (a worse) linear one. Measured on synthetic XOR:
    # sd/mean .72 at d=4 (RBF .96 vs linear .53) -> .022 at d=4000 (RBF dead, and NO γ rescues
    # it). So a near-zero ratio here means the column cannot work, and that has to be READ OFF
    # THE LOG, not inferred afterwards from a null result.
    _RBF_STATS["conc"].append(float(np.std(d2[iu]) / np.mean(d2[iu])))
    d2e = {}
    for name in evals:
        de = ev_sq[name][:, None] + sq[None, :] - 2.0 * kern_lin[name]
        np.maximum(de, 0.0, out=de)
        d2e[name] = de
    y = np.asarray(y_tr, dtype=np.float64)
    best = None
    for gm in GAMMA_MULTS:
        gamma = gm / med
        with _timed("rbf_eigh"):
            w, V = np.linalg.eigh(np.exp(-gamma * d2))
        base = float(np.sum(w) / max(n, 1))
        assert abs(base - 1.0) < 1e-6, f"RBF trace(K)/n = {base}, must be 1 (diag(K)==1)"
        c = V.T @ y
        kern = {name: np.exp(-gamma * d2e[name]) for name in evals}
        out: dict = {name: {} for name in evals}
        sc: dict = {name: {} for name in evals}
        with _timed("rbf_lam_sweep"):
            for m in LAM_MULTS:
                alpha = V @ (c / (w + m * base))
                for name, (_, yy) in evals.items():
                    s = kern[name] @ alpha
                    if _capturing():
                        sc[name][m] = s
                    out[name][m] = (auroc(s, yy) if len(yy) >= 2 else float("nan"))
        if _capturing():
            out["_scores"] = sc
        vals = [v for v in out.get("val", {}).values() if np.isfinite(v)]
        key = max(vals) if vals else -np.inf
        # Strict `>` over an ASCENDING γ grid ⇒ a val tie keeps the SMALLEST γ, i.e. the WIDEST
        # kernel, i.e. the one closest to linear. Same tie convention as `argmax` on λ, and it
        # breaks toward the incumbent model rather than toward the new one.
        if best is None or key > best[0]:
            best = (key, out, gm)
    assert best is not None, "GAMMA_MULTS is non-empty, so a γ is always selected"
    _RBF_STATS["fits"] += 1
    _RBF_STATS["picked"][best[2]] = _RBF_STATS["picked"].get(best[2], 0) + 1
    return best[1]


def _linear_grams(z_tr, evals):
    """(G = Z_trZ_trᵀ, {name: Z_eZ_trᵀ}) — the ONLY GEMMs either kernel needs.

    Split out so the linear and RBF fits can SHARE them. At the board's headline tap this is the
    entire cost of a fit: enc12_elec is d = 1.58e6 against n <= 1750, so G alone is ~10 TFLOP
    while everything downstream (an n×n eigendecomposition, a 25-point λ sweep) is milliseconds.
    Recomputing these to add a second kernel would DOUBLE a shard for nothing.
    """
    with _timed("gram_gemm"):
        g = np.asarray(z_tr @ z_tr.T, dtype=np.float64)         # fp32 GEMM → fp64 Gram
    with _timed("eval_kernels"):
        kern = {name: np.asarray(z @ z_tr.T, dtype=np.float64)
                for name, (z, _) in evals.items()}
    return g, kern


# ── λ SELECTED ON THE TRAIN HALF (--lam-cv) ───────────────────────────────────────────────────
# WHY THIS IS NOT RE-OPENING THE CLOSED λ AXIS. The λ memo closes "every VAL-DEFINABLE rule", and
# it is right: tie-break, slack, and all three ensemble rules were measured on the CS board and
# every one failed to reach the per-unit oracle (.6094 published vs .6164 oracle, +.0070 on
# 100/150 units). But every one of those rules reads the SAME val half, so they all inherit the
# same noise -- they re-weight one noisy estimate instead of replacing it. `tiemax` moved ZERO of
# 150 units, which says CS val is not TIED, it is picking a strictly-best-on-val λ that is not
# best on test. That is selection VARIANCE, and the only cure for variance is more data.
#
# The train half is the one source never used, and it is ~14x larger: upstream's KFold gives train
# = (k-1)/k of the session while val is HALF of one held-out fold = 1/(2k)
# (train_test_splits.py:230-238). Selecting λ there is also strictly MORE protocol-safe than the
# published rule, not less -- it touches no held-out data at all.
#
# 🔮 PRIOR, STATED BEFORE THE MEASUREMENT so the result cannot be rationalized afterwards. I expect
# this to select LARGER λ than val does, for two compounding reasons:
#   1. each CV model fits 0.8n rows, and the optimal λ grows as n shrinks;
#   2. at d >> n (enc12_elec is d=1.58e6 against n<=1750) the ridge INTERPOLATES as λ→0, so a
#      held-out fold punishes small λ harder than the val half does.
# WS wants the opposite -- it is LO-pinned on 19.4% of units, i.e. it wants LESS shrinkage. So the
# honest prior is that this HELPS CS (where λ is interior and the oracle gap is measured) and
# HURTS WS. That asymmetry is the result to look for; a uniform win would be the surprise.
# ⇒ the median selected λ is printed for BOTH rules per shard, so the mechanism is visible in the
# log rather than inferred from the AUROCs afterwards.
LAM_CV = False
LAM_CV_K = 5            # fixed in advance; it is not a knob to tune against test.
_CV_STATS: dict = {"pairs": []}


def _cv_folds(n: int, k: int):
    """CONTIGUOUS k-fold indices over 0..n-1.

    Contiguous, not shuffled, deliberately: upstream builds its own folds with
    ``KFold(shuffle=False)`` and comments that shuffling would "avoid correlated train/test
    splits". Trial windows adjacent in time share a movie context, so a SHUFFLED fold leaks that
    context across the split and would report an optimistic λ curve — the same leak class as the
    M14 window-overlap bug. Contiguous blocks reproduce the train→held-out relationship the board
    actually scores.
    """
    b = np.linspace(0, n, k + 1).astype(int)
    return [(np.r_[0:b[i], b[i + 1]:n], np.arange(b[i], b[i + 1])) for i in range(k)]


def _cv_curve(g, y_tr, base, k=None):
    """{lam_mult: mean held-out AUROC} over contiguous CV folds of the TRAIN half.

    Costs NO new GEMM: a fold's Gram is the submatrix ``g[ix_(tr,tr)]`` of the one already formed,
    so the added work is k eigendecompositions of a (0.8n)² block — ~2s against the headline tap's
    ~10 TFLOP GEMM.

    ``base`` is the FULL-train trace(G)/n, deliberately shared with the main sweep rather than
    recomputed per fold: a λ MULTIPLIER only transfers to the full-train fit if it denotes the same
    absolute shrinkage in both places. (The two differ negligibly anyway — both are means of the
    same diagonal — but sharing makes that exact instead of approximate.)
    """
    k = LAM_CV_K if k is None else k
    y = np.asarray(y_tr, dtype=np.float64)
    acc: dict = {m: [] for m in LAM_MULTS}
    for tr_i, te_i in _cv_folds(len(y), k):
        yte = y[te_i]
        # A single-class held-out block scores NaN, not 0.5: it carries no ranking information, and
        # averaging a fabricated 0.5 into the curve would flatten it toward indifference.
        if len(tr_i) < 2 or len(yte) < 2 or (yte > 0).min() == (yte > 0).max():
            continue
        w2, v2 = np.linalg.eigh(g[np.ix_(tr_i, tr_i)])
        c2 = v2.T @ y[tr_i]
        gte = g[np.ix_(te_i, tr_i)]
        for m in LAM_MULTS:
            acc[m].append(auroc(gte @ (v2 @ (c2 / (w2 + m * base))), yte))
    return {m: (float(np.nanmean(v)) if v else float("nan")) for m, v in acc.items()}


def _lam_grid(z_tr, y_tr, evals, grams=None):
    """Fit ridge on (z_tr, y_tr); score every λ in LAM_MULTS on each eval set.

    Returns {eval_name: {lam_mult: auroc}}. One fp64 eigendecomposition of G=Z_trZ_trᵀ serves
    the whole grid — the ridge solution is α = V diag(1/(w+λ)) Vᵀ y, so sweeping λ reuses
    (w, V, c=Vᵀy) and costs one mat-vec each. GEMMs are fp32 (memory), G/solve are fp64.
    λ NEVER enters through an eval set: only through w, which is anchor-side only.

    `grams` accepts an already-computed (G, eval kernels) pair so the RBF column can be fit off
    the SAME products. Passing it forces the dual, which is why the caller only ever passes it
    when d >= n — the branch the primal would have taken is never the one being shared.
    """
    if len(y_tr) < 2:
        return {name: {m: float("nan") for m in LAM_MULTS} for name in evals}
    if grams is None and z_tr.shape[1] < z_tr.shape[0]:
        return _lam_grid_primal(z_tr, y_tr, evals)
    g, kern = _linear_grams(z_tr, evals) if grams is None else grams
    n = g.shape[0]
    with _timed("eigh"):
        w, V = np.linalg.eigh(g)                                # G symmetric PSD ⇒ w >= 0
    c = V.T @ np.asarray(y_tr, dtype=np.float64)
    base = float(np.sum(w) / max(n, 1))                         # trace(G)/n — the λ scale
    out: dict = {name: {} for name in evals}
    sc: dict = {name: {} for name in evals}
    with _timed("lam_sweep"):
        for m in LAM_MULTS:
            alpha = V @ (c / (w + m * base))
            for name, (_, y) in evals.items():
                s = kern[name] @ alpha
                if _capturing():
                    sc[name][m] = s
                out[name][m] = (auroc(s, y) if len(y) >= 2 else float("nan"))
    if LAM_CV:
        # Keyed with a leading underscore so it can never be mistaken for an eval set: _grid_cells
        # iterates real eval names, and a "cv" entry there would be reported as if it were a
        # held-out score. It is a SELECTION curve computed entirely inside the train half.
        out["_cv"] = _cv_curve(g, y_tr, base)
    if _capturing():
        out["_scores"] = sc
    return out


def _cv_selected(d):
    """Re-key a λ grid so _select_lam picks on the TRAIN-CV curve and reports the SAME test curve.

    Nothing is refit. The published column and this one differ only in which λ is read off an
    identical test curve, so the two are paired on the same fit by construction — the comparison
    cannot be contaminated by a difference in features, splits, or arithmetic.
    """
    return None if "_cv" not in d else {"val": d["_cv"], "test": d["test"]}


def _lam_grid_primal(z_tr, y_tr, evals):
    """Same fit as _lam_grid, solved in the PRIMAL — taken whenever d < n.

    ``Zᵀ(ZZᵀ + λI)⁻¹ y == (ZᵀZ + λI)⁻¹ Zᵀ y`` exactly, and ``trace(ZᵀZ) == trace(ZZᵀ)``, so
    ``base`` — and therefore what each λ multiplier MEANS — is unchanged. Three costs move:
    the Gram is n·d² instead of n²·d, the eigendecomposition is O(d³) instead of O(n³), and the
    eval kernels vanish entirely (score is ``z_e @ β``, an (n_e, d)·(d, |LAM|) GEMM, not
    ``(z_e z_trᵀ) @ α``). All 25 λ share one GEMM per eval set by stacking β columnwise.

    WHERE THIS ACTUALLY FIRES (measured on the r6_40k cache, 07-27): the ONE tap where d < n is
    CS enc0 — 7 anchor∩test parcels × 348 = d 2436 against n_train 2096-3500. Every other tap is
    d >> n (CS enc3/6/12 = 93184; WS/CSession enc0_elec = 41412, enc12_elec = 1584128, both
    against n <= 1750), so the dual is correct there and this branch is never taken. Benchmarked
    at the CS enc0 shape: 5.91 s dual -> 1.35 s primal, 4.4x — but enc0 is ~0.9% of a CS shard's
    GEMM work, so the SHARD-level saving is ~1-2%. Kept because it is a few lines and free, NOT
    because it is the lever; the lever was the standardize blocking in _standardize_inplace.

    NOT bit-identical to the dual path: the fp32 GEMM accumulates a different set of products.
    Measured max |AUROC_dual − AUROC_primal| = 2.7e-5 at the CS enc0 shape — three orders below
    the ±.002 probe noise floor, but it does mean a re-run of an old board can move CS enc0 in
    the 5th decimal.
    """
    with _timed("gram_gemm"):
        a_mat = np.asarray(z_tr.T @ z_tr, dtype=np.float64)      # (d, d)
    n = z_tr.shape[0]
    with _timed("eigh"):
        w, V = np.linalg.eigh(a_mat)
    c = V.T @ (z_tr.T @ np.asarray(y_tr, dtype=np.float64))
    base = float(np.trace(a_mat) / max(n, 1))                    # == sum(eig(G))/n, the dual's scale
    with _timed("lam_sweep"):
        lam = np.asarray(LAM_MULTS, dtype=np.float64) * base
        beta = V @ (c[:, None] / (w[:, None] + lam[None, :]))    # (d, |LAM_MULTS|)
        out: dict = {}
        sc: dict = {}
        for name, (z, y) in evals.items():
            if len(y) < 2:
                out[name] = {m: float("nan") for m in LAM_MULTS}
                continue
            s = np.asarray(z, dtype=np.float64) @ beta           # (n_e, |LAM_MULTS|)
            out[name] = {m: auroc(s[:, i], y) for i, m in enumerate(LAM_MULTS)}
            if _capturing():
                sc[name] = {m: s[:, i] for i, m in enumerate(LAM_MULTS)}
    if _capturing():
        out["_scores"] = sc
    return out


LAM_RULE = "argmax"     # published tie-break. --lam-rule overrides; see _select_lam.
LAM_RULES = ("argmax", "tiemax")

# --dump-lam-grid: record the WHOLE 25-point (val, test) λ curve per cell instead of only the
# selected point. Same bargain --dump-epoch-test struck on the FT side, and it paid there: the
# epoch curve is what proved the +.0105 headroom was in SELECTION, not in the schedule. Here it
# answers the λ-axis version of that question -- how much does val-argmax-λ cost against the best
# λ on the grid? -- and it answers it for every λ rule, offline, forever, off one CPU run.
# 🔴 OBSERVATION ONLY. The grid is already computed for the sweep; dumping it changes no selection
# and no reported number (test_dump_lam_grid_does_not_move_the_selected_point). Picking λ to
# maximise the dumped test curve is an ORACLE and must be reported as a ceiling, never a result.
LAM_GRID_DUMP = False

# --tap-ensemble: rank-average the PREDICTIONS of the per-tap ridges into extra reported columns.
#
# 🚫 THIS IS NOT DEPTH CONCAT, which is CLOSED AND NEGATIVE (cs enc9+enc12 -.0007, +enc6 -.0023,
# +enc3 -.0046, monotone harm over 150 units). Concat fuses taps INSIDE one kernel, where extra
# blocks dilute the Gram and the single shared λ has to regularize all of them at once. This fits
# each tap SEPARATELY, with its OWN val-selected λ, and combines only at the score level -- the
# ordinary ensemble bargain, which pays exactly when the members' ERRORS decorrelate. The
# interesting pair is therefore not adjacent depths (nearly the same features) but the two UNITS,
# enc12 vs enc12_elec: parcel-mean and per-electrode are different spatial aggregations of the
# same tap, so they can be wrong on different trials.
#
# Ranks, not raw scores, for the same reason the FT epoch ensemble uses them: AUROC depends only
# on the ORDER of scores, so a tap whose selected λ is smaller emits larger-magnitude scores and
# would outvote the rest of a raw mean for a reason that carries no information.
#
# ✅ SUBMITTABLE. Every rule below reads the VAL half only -- the same half that already selects λ
# -- and never test. `ens:all` has no selection at all.
TAP_ENSEMBLE = False
ENS_RULES = ("ens:all", "ens:top2", "ens:top3", "ens:auto")

# --lam-ensemble: rank-average the predictions ACROSS λ within a single tap, as extra columns
# "lamall:<tap>|<norm>" / "lam3:<tap>|<norm>" / "lamge:<tap>|<norm>".
#
# WHY THIS AXIS. The λ-grid dump measured the ceiling on the real CS board: val-argmax-λ scores
# .6094 where the best λ PER UNIT would score .6164 (+.0070, better on 100/150 units). That gap is
# an ORACLE and not claimable, but it is evidence that the val half picks λ NOISILY -- and
# averaging over a val-defined NEIGHBOURHOOD of λ is the standard, submittable way to spend that
# noise. It costs nothing: every λ's scores are already computed for the sweep.
#
# ⚠️ NOT the same bet as --tap-ensemble. Members here are the SAME features at adjacent shrinkage,
# so they are far more correlated than two taps are, and the honest prior is a smaller effect.
LAM_ENSEMBLE = False
LAM_ENS_RULES = ("lamall", "lam3", "lamge")
LAM_ENS_SLACK = 0.01        # `lamge` band: every λ whose val is within this of the best


def _capturing() -> bool:
    """Both ensemble levers need _lam_grid to keep its per-λ score vectors."""
    return TAP_ENSEMBLE or LAM_ENSEMBLE

# Tie census, accumulated across every ridge fit in a shard and printed once at the end. The
# LAM_MULTS comment says to MEASURE THE TIE FRACTION FIRST; this is that measurement, taken on the
# real board rather than on the synthetic fixture where the plateau was discovered.
_TIE_STATS = {"fits": 0, "tied_ge2": 0, "tied_total": 0}


def _select_lam(d, rule=None) -> dict:
    """One (tap,norm)'s λ-grid {"val": {m: auroc}, "test": {m: auroc}} → pick argmax VAL.

    λ is the ONLY val-selected axis (Ben 2026-07-17). Tap and norm are REPORTED, not selected:
    every (tap, norm) keeps its own complete number over every cell, so the depth ladder
    (enc0/3/6/12), the per-electrode-vs-parcel-mean contrast, and the std/raw/std_target
    normalization contrast all survive intact instead of collapsing into one argmax. λ is the
    exception because it is not a result — it is a per-fit regularization parameter that has to
    be set somehow, and upstream's own val half exists for exactly that.

    All-NaN val (single-class val half) → NaN, reported as such rather than silently defaulting
    to a λ, so a degenerate cell cannot masquerade as a scored one.

    ``lam_pin`` records which grid boundary the selected λ sits on, and the two sides are NOT
    the same failure (measured 07-17):

      "hi" → BENIGN. AUROC saturates as λ→∞: the smoother w/(w+λ) → 0 uniformly, so the scores
             converge to a fixed ranking (AUROC(1e4) == AUROC(1e16) exactly, and both equal the
             AUROC of K@y_tr). A HI pin means "maximal shrinkage is best", which the grid already
             reports faithfully. Widening the grid cannot change the number.
      "lo"  → REAL truncation. There is no such limit at the bottom; λ→0 keeps moving, so the
             optimum really is outside the grid and the AUROC is an artifact.

    ``lam_pinned`` is therefore LO-only — it is the one that invalidates a fit. Conflating the two
    costs a full re-run of a 22-shard board for nothing.
    ``rule`` decides ONLY what happens on a val TIE, and both rules are functions of the val half
    alone — neither can see test, so switching is not a selection-on-test move.

      "argmax" (DEFAULT, the PUBLISHED rule) — strict `>` over an ascending grid ⇒ a tie keeps the
               SMALLEST λ, i.e. the LEAST regularized model. Every board number reported to date
               uses this; it must stay the default so past runs stay reproducible.
      "tiemax" — a tie keeps the LARGEST λ. Among models the val half rates identically, take the
               most shrunk one. This is the ordinary convention (cf. the 1-SE rule) and it is
               a-priori defensible WITHOUT reference to the outcome: a tie carries no information,
               so the arbitrary order of a tuple should not be what picks the operating point.
               It has NO free parameter, so it adds no knob to tune against test.

    ⚠️ THE TWO RULES ARE NOT COMPARABLE AS "BETTER" ON ONE BOARD. Whichever wins, it wins partly by
    luck over an under-determined plateau (see test_widening_the_grid_can_LOWER_a_pinned_cell).
    Report the rule, fixed in advance; do not run both and quote the max.
    """
    rule = LAM_RULE if rule is None else rule
    finite = [(m, va) for m, va in d["val"].items() if not np.isnan(va)]
    if not finite:
        return {"val": float("nan"), "test": float("nan"), "lam_mult": float("nan"),
                "lam_pin": "", "lam_pinned": False, "n_tied": 0}
    best_val = max(va for _, va in finite)
    # Insertion order is LAM_MULTS ascending, so tied[0] is the smallest tied λ — exactly what the
    # strict-`>` loop this replaced selected. That equivalence is what keeps "argmax" byte-faithful.
    tied = [m for m, va in finite if va == best_val]
    m = tied[0] if rule == "argmax" else max(tied)
    _TIE_STATS["fits"] += 1
    _TIE_STATS["tied_ge2"] += len(tied) > 1
    _TIE_STATS["tied_total"] += len(tied)
    pin = "lo" if m == LAM_MULTS[0] else ("hi" if m == LAM_MULTS[-1] else "")
    out = {"val": best_val, "test": d["test"][m], "lam_mult": float(m),
           "lam_pin": pin, "lam_pinned": pin == "lo", "n_tied": len(tied)}
    if LAM_GRID_DUMP:
        # Appended AFTER the selected point is built, from the same `d` the selection read, so the
        # dump cannot influence it. Ascending λ, one [mult, val, test] row per grid point.
        # Over the grid THIS FIT ACTUALLY SWEPT, not the module-level LAM_MULTS: they are the same
        # tuple on every board run, but keying off `d` means the dump can never silently drop a
        # point or KeyError on a grid that differs from the global one.
        out["lam_grid"] = [[float(mm), float(d["val"][mm]), float(d["test"][mm])]
                           for mm in sorted(d["val"])]
    if "_scores" in d:
        # The val/test score vectors AT THE SELECTED λ, for _tap_ensembles. Carried on the cell
        # only until the ensemble is built; _grid_cells strips it before anything is written.
        out["_sc"] = {nm: s[m] for nm, s in d["_scores"].items() if m in s}
    return out


def _cell_key(tap, norm) -> str:
    return f"{tap}|{norm}"


def _rank01(s):
    """Scores → ranks in [0,1]. Order-only, so a tap cannot outvote the others by score SCALE."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    r = np.empty(n, dtype=np.float64)
    r[np.argsort(s, kind="stable")] = np.arange(n, dtype=np.float64)
    return r / (n - 1)


def _ens_member_sets(names, vals):
    """{rule → member tap keys}, from the VAL AUROCs and NOTHING else.

    `ens:auto` includes the best SINGLE tap as a candidate, so the rule is allowed to decline to
    ensemble. Without that it would be forced to average even where averaging hurts, and a rule
    that cannot say "no" is not a rule -- it is a result waiting to be quoted selectively.
    """
    ok = [k for k in names if np.isfinite(vals[k])]
    if len(ok) < 2:
        return {}
    order = sorted(ok, key=lambda k: (-float(vals[k]), k))
    return {"ens:all": sorted(ok), "ens:top2": sorted(order[:2]), "ens:top3": sorted(order[:3])}


def _tap_ensembles(cells, y_va, y_te) -> dict:
    """Rank-averaged cross-TAP ensembles, per norm. Returns extra cell entries.

    Grouped BY NORM because std and raw are different reported columns, not interchangeable
    members: averaging a std tap with a raw tap would fuse two columns the board reports
    separately, which is the same defect class as pooling ws with csession.
    """
    if y_va is None or y_te is None or len(y_te) < 2:
        return {}
    by_norm: dict = {}
    for key, c in cells.items():
        if "_sc" not in c or "val" not in c["_sc"] or "test" not in c["_sc"]:
            continue
        by_norm.setdefault(key.split("|", 1)[1], []).append(key)
    out: dict = {}
    for norm, keys in by_norm.items():
        if len(keys) < 2:
            continue
        vals = {k: cells[k]["val"] for k in keys}
        rk = {k: {nm: _rank01(cells[k]["_sc"][nm]) for nm in ("val", "test")} for k in keys}

        def score(members):
            return {nm: auroc(np.mean([rk[k][nm] for k in members], axis=0),
                              y_va if nm == "val" else y_te) for nm in ("val", "test")}

        sets = _ens_member_sets(keys, vals)
        got = {r: score(ms) for r, ms in sets.items()}
        # `ens:auto` picks among the ensembles AND the best single tap, on VAL alone.
        best_single = max(keys, key=lambda k: (float(vals[k]), k))
        cand = list(got.items()) + [("single", {"val": vals[best_single],
                                                "test": cells[best_single]["test"]})]
        pick = max(cand, key=lambda kv: (float(kv[1]["val"]), kv[0]))
        got["ens:auto"] = pick[1]
        for rule, r in got.items():
            members = ("|".join(sets[rule]) if rule in sets else
                       (pick[0] if pick[0] != "single" else best_single))
            out[f"{rule}|{norm}"] = {"val": float(r["val"]), "test": float(r["test"]),
                                     "lam_mult": float("nan"), "lam_pin": "",
                                     "lam_pinned": False, "n_tied": 0, "ens_members": members}
    return out


def _lam_ensembles(grid, y_va, y_te) -> dict:
    """Rank-averaged WITHIN-tap ensembles over λ. Returns extra cells keyed "<rule>:<tap>|<norm>".

    Every member set is read off the VAL curve -- the same curve that already picks λ -- so no
    rule here sees test. `lamall` has no selection at all: it averages the whole grid.
    """
    if y_va is None or y_te is None or len(y_te) < 2:
        return {}
    out: dict = {}
    for (tap, norm), d in grid.items():
        scs = d.get("_scores")
        if not scs or "val" not in scs or "test" not in scs:
            continue
        ms = [m for m in sorted(d["val"]) if np.isfinite(d["val"][m]) and m in scs["test"]]
        if len(ms) < 2:
            continue
        best = max(d["val"][m] for m in ms)
        order = sorted(ms, key=lambda m: (-float(d["val"][m]), m))
        sets = {"lamall": ms,
                "lam3": sorted(order[:3]),
                "lamge": [m for m in ms if d["val"][m] >= best - LAM_ENS_SLACK]}
        rk = {nm: {m: _rank01(scs[nm][m]) for m in ms} for nm in ("val", "test")}
        for rule, mem in sets.items():
            if not mem:
                continue
            out[f"{rule}:{tap}|{norm}"] = {
                "val": float(auroc(np.mean([rk["val"][m] for m in mem], axis=0), y_va)),
                "test": float(auroc(np.mean([rk["test"][m] for m in mem], axis=0), y_te)),
                "lam_mult": float("nan"), "lam_pin": "", "lam_pinned": False,
                "n_tied": 0, "n_lam": len(mem)}
    return out


def _grid_cells(grid, y_va=None, y_te=None) -> dict:
    """{(tap,norm): λ-grid} → {"tap|norm": λ-selected result}. Nothing is dropped or fused.

    Under --tap-ensemble the per-tap columns are untouched and ADDITIONAL "ens:*|norm" columns are
    appended; the score vectors that build them are stripped here so nothing extra is written.
    """
    cells = {_cell_key(t, nm): _select_lam(d) for (t, nm), d in grid.items()}
    if LAM_CV:
        # The MECHANISM census, paired per (tap, fit). The stated prior is that the train-CV rule
        # picks LARGER λ than val does; if the column then loses, this says whether it lost for the
        # predicted reason (over-shrinkage) or for some other one. Recorded here because this is
        # the only place both selections for one fit are in scope at once.
        for (t, nm) in grid:
            if nm != "std" or _cell_key(t, "std_cv") not in cells:
                continue
            a, b = cells[_cell_key(t, "std")], cells[_cell_key(t, "std_cv")]
            if np.isfinite(a["lam_mult"]) and np.isfinite(b["lam_mult"]):
                _CV_STATS["pairs"].append((float(a["lam_mult"]), float(b["lam_mult"])))
    if TAP_ENSEMBLE:
        cells.update(_tap_ensembles(cells, y_va, y_te))
    if LAM_ENSEMBLE:
        # Built from `grid`, not from `cells`, because it needs EVERY λ's scores, not the selected
        # one. Ordered after the tap ensemble so a λ-ensemble column can never become a tap member.
        cells.update(_lam_ensembles(grid, y_va, y_te))
    for c in cells.values():
        c.pop("_sc", None)
    return cells


def _parcel_cols(anchor_rec, test_rec):
    """Anchor∩test parcel columns, aligned BY ATLAS ID (not by position).

    The reserved 'unknown' id is NOT an anatomical location: two electrodes carrying it are
    not in the same place, so aligning subjects on it would be a free unearned column. It is
    a no-op on the Lite board (anchor S2T4 has no unmapped electrodes) but fires on any
    corpus that does, so it is excluded here rather than assumed away.
    """
    a_p = np.asarray(anchor_rec["present_parcels"], dtype=np.int64)
    t_p = np.asarray(test_rec["present_parcels"], dtype=np.int64)
    common = np.intersect1d(a_p, t_p)
    if UNKNOWN_PARCEL_ID in common:
        print(f"[check] dropping unknown parcel {UNKNOWN_PARCEL_ID} from intersection", flush=True)
        common = common[common != UNKNOWN_PARCEL_ID]
    if common.size == 0:
        return None, None, common
    a_idx = [int(np.where(a_p == c)[0][0]) for c in common]
    t_idx = [int(np.where(t_p == c)[0][0]) for c in common]
    return a_idx, t_idx, common


def _run_norms(grid, enc, z_tr, z_va, z_te, y_tr, y_va, y_te, cs=False):
    """Score one tap's λ-grid under every REPORTED norm, holding only ONE copy of each design
    matrix in memory.

    ORDER IS LOAD-BEARING: raw and std_target both read the RAW
    features, so they run BEFORE the in-place std that consumes (mutates) z_tr/z_va/z_te. With
    this ordering every norm's numbers are bit-identical to the old copy-based path; the only
    change is that the std copy — the ~tens-of-GB duplicate that thrashed heavy CSession cells —
    is never allocated. std_target (_standardize_per_domain) makes its own copies and never
    mutates its inputs, so it is safe to run before the in-place std."""
    def evals(b, c):
        return {"val": (b, y_va), "test": (c, y_te)}
    if "raw" in NORMS:
        grid[(enc, "raw")] = _lam_grid(z_tr, y_tr, evals(z_va, z_te))
    if cs and REPORT_STD_TARGET:
        a, (b, c) = _standardize_per_domain(z_tr, z_va, z_te)
        grid[(enc, "std_target")] = _lam_grid(a, y_tr, evals(b, c))
    if "std" in NORMS:
        a, (b, c) = _standardize_inplace(z_tr, [z_va, z_te])
        ev = evals(b, c)
        # Share ONE set of GEMMs between the two kernels when both are reported. Only when d >= n,
        # because that is the branch _lam_grid would have taken anyway -- passing grams where the
        # primal is cheaper would make the linear column slower to serve the RBF one.
        gr = _linear_grams(a, ev) if (RBF and a.shape[1] >= a.shape[0] and len(y_tr) >= 2) else None
        grid[(enc, "std")] = _lam_grid(a, y_tr, ev, grams=gr)
        if LAM_CV:
            cv = _cv_selected(grid[(enc, "std")])
            if cv is not None:
                grid[(enc, "std_cv")] = cv
        if RBF:
            grid[(enc, "std_rbf")] = _lam_grid_rbf(a, y_tr, ev, grams=gr)
            for k in RBF_PCS:
                grid[(enc, f"std_rbfpc{k}")] = _lam_grid_rbf(a, y_tr, ev, grams=gr, n_pc=k)


def _ws_cell(rec, task, taps) -> dict:
    """Within-session: board KFold(2). Per fold fit train, λ-select on the val half, report the
    test half; average the two folds' test AUROCs, per (tap, norm)."""
    y = np.asarray(rec["labels"][task], dtype=np.float64)
    folds = []
    for _fold, sp in sorted(rec["ws_split"][task].items()):
        tr, va, te = (_finite(y, sp["train"]), _finite(y, sp["val"]), _finite(y, sp["test"]))
        if len(tr) < 2 or len(te) < 2:
            continue
        grid = {}
        for enc in taps:
            if not _have(rec, enc):
                continue
            z_tr = _feat(rec, enc, tr)
            z_va, z_te = _feat(rec, enc, va), _feat(rec, enc, te)
            _run_norms(grid, enc, z_tr, z_va, z_te, y[tr], y[va], y[te])
        if grid:
            folds.append(_grid_cells(grid, y[va], y[te]))
    if not folds:
        return {"cells": {}}
    keys = sorted({k for f in folds for k in f})
    out = {}
    for k in keys:
        vals = [f[k]["test"] for f in folds if k in f]
        out[k] = {"test": float(np.nanmean(vals)) if vals else float("nan"),
                  "lam_pinned": bool(any(f[k]["lam_pinned"] for f in folds if k in f)),
                  "lam_sat": bool(any(f[k].get("lam_pin") == "hi" for f in folds if k in f)),
                  "lam_mult": [f[k]["lam_mult"] for f in folds if k in f]}
        # 🔴 THIS DICT IS A FIXED FIELD LIST, so anything a cell grew elsewhere is DROPPED here --
        # which is exactly why --dump-lam-grid produced empty WS shards while CS/CSession (which
        # return _grid_cells directly, unfolded) dumped fine. The ensemble AUROCs survive because
        # they are their own cells with their own "test"; what needs carrying is the audit
        # metadata that says WHICH members produced them, per fold.
        for fld in ("ens_members", "n_lam"):
            got = [f[k][fld] for f in folds if k in f and fld in f[k]]
            if got:
                out[k][fld] = got
    return {"cells": out}


def _cs_cell(anchor_rec, test_rec, task, taps) -> dict:
    """Cross-subject: fit the anchor's finite rows, λ-select on the test cell's val half, report
    its test half. Features are the anchor∩test parcel intersection (atlas-id aligned)."""
    y_a = np.asarray(anchor_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {"cells": {}}
    a_idx, t_idx, common = _parcel_cols(anchor_rec, test_rec)
    if common.size == 0:
        return {"cells": {}}
    grid: dict = {}
    for enc in taps:
        if not (_have(anchor_rec, enc) and _have(test_rec, enc)):
            continue
        z_tr = _feat(anchor_rec, enc, tr, a_idx)
        z_va, z_te = _feat(test_rec, enc, va, t_idx), _feat(test_rec, enc, te, t_idx)
        # cs=True reports the AdaBN-style per-domain norm (std_target) as a third column; it makes
        # its own copies and reads the RAW features, so _run_norms runs it before the in-place std.
        # Gated by REPORT_STD_TARGET (Ben's r5mod board is std-only via --no-std-target).
        _run_norms(grid, enc, z_tr, z_va, z_te, y_a[tr], y_t[va], y_t[te], cs=True)
    if not grid:
        return {"cells": {}}
    return {"cells": _grid_cells(grid, y_t[va], y_t[te]), "n_parcels": int(common.size)}


def _elec_cols(train_rec, test_rec):
    """Shared electrodes between two SAME-SUBJECT sessions, aligned BY LABEL (elec_labels).

    Sibling Lite trials drop DIFFERENT bad channels (measured: 4/6 subjects differ, e.g. subj-3
    100 vs 102), so the per-electrode axis is NOT positionally aligned across sessions —
    intersecting by identity is the only correct alignment. Returns (train_idx, test_idx,
    n_shared); (None, None, 0) if either cache lacks elec_labels (pre-edit cache) or no overlap."""
    a = train_rec.get("elec_labels")
    t = test_rec.get("elec_labels")
    if a is None or t is None:
        return None, None, 0
    a = np.asarray(a)
    t = np.asarray(t)
    common = np.intersect1d(a, t)
    if common.size == 0:
        return None, None, 0
    a_idx = [int(np.where(a == c)[0][0]) for c in common]
    t_idx = [int(np.where(t == c)[0][0]) for c in common]
    return a_idx, t_idx, int(common.size)


def _csession_cell(train_rec, test_rec, task, taps) -> dict:
    """Cross-session: train on the SIBLING trial of the SAME subject, λ-select on this cell's val
    half, report its test half.

    Upstream ``generate_splits_cross_session`` halves the test session IDENTICALLY to
    ``generate_splits_cross_subject`` (train_test_splits.py:153-156 == 66-69: val=range(size//2),
    test=range(size//2,size)), so ``cs_split``'s val/test are reused VERBATIM — only the train
    anchor differs (the sibling trial, not S2T4). Parcel taps align by atlas id (``_parcel_cols``);
    per-electrode taps align by electrode IDENTITY (``_elec_cols``), on the shared-electrode subset.
    """
    y_a = np.asarray(train_rec["labels"][task], dtype=np.float64)
    y_t = np.asarray(test_rec["labels"][task], dtype=np.float64)
    tr = _finite(y_a, np.arange(len(y_a)))
    va = _finite(y_t, test_rec["cs_split"][task]["val"])
    te = _finite(y_t, test_rec["cs_split"][task]["test"])
    if len(tr) < 2 or len(te) < 2:
        return {"cells": {}}
    p_a, p_t, p_common = _parcel_cols(train_rec, test_rec)
    e_a, e_t, n_elec = _elec_cols(train_rec, test_rec)
    grid: dict = {}
    for enc in taps:
        if not (_have(train_rec, enc) and _have(test_rec, enc)):
            continue
        if _is_elec(enc):
            if e_a is None:
                continue
            col_a, col_t = e_a, e_t
        else:
            if p_a is None:
                continue
            col_a, col_t = p_a, p_t
        z_tr = _feat(train_rec, enc, tr, col_a)
        z_va, z_te = _feat(test_rec, enc, va, col_t), _feat(test_rec, enc, te, col_t)
        _run_norms(grid, enc, z_tr, z_va, z_te, y_a[tr], y_t[va], y_t[te])
    if not grid:
        return {"cells": {}}
    return {"cells": _grid_cells(grid, y_t[va], y_t[te]),
            "n_parcels": int(p_common.size) if p_a is not None else 0,
            "n_elec": n_elec}


# mmap default is PER MODE, and the reason is measured, not aesthetic (07-17, on enc_s2_t4
# under a Lustre carrying 734 jobs):
#     cold scattered gather, 1750 elec rows : 231.55 s for 5.5 GB  =  24 MB/s
#     warm gather, same rows (page cache)   :   2.31 s             = 100x faster
#     eager sequential load (shard 0, live) : ~660 s for 56.7 GB   = ~86 MB/s
# So mmap does NOT make loading free — it DEFERS it into the gathers, at ~1/4 the bandwidth of
# one sequential stream. ("mmap loads in 0.5 s at 0.5 GB" is an artifact of measuring an open()
# that reads nothing; the bytes are still owed.)
#
# mmap was originally ON for CS, to cut RESIDENT memory (~12 GB of parcel taps instead of 43 GB)
# and buy concurrency. That whole premise was WRONG, and the correction is the lesson worth
# keeping: the shards were not stalling because they were too big. They were stalling because
# --cpus-per-task=8 pins to NUMA node 0 (31.9 GB of 251 GB) and AutoNUMA then thrashes forever
# trying to migrate spilled pages home. The fix is `numactl --interleave=all` in the sbatch — see
# reference-delta-numa-node0-starvation-interleave-2026-07-17. Once memory is no longer scarce,
# mmap buys nothing and costs plenty: measured 15 MB/s with 3.7M major faults and RSS collapsed
# to 1 GB, vs ~86 MB/s eager. A 43-min shard produced zero output under it.
# Eager is right for BOTH modes. Keep the knob for A/B, but neither default is mmap.
MMAP_DEFAULT = {"ws": False, "cs": False, "csession": False}


def _load(cache_dir, session, tag, mmap=False):
    """Load a session cache. ``mmap=True`` defers the read into the gathers (see MMAP_DEFAULT).

    Pages arrive only where a tensor is actually indexed, so selectivity is free: a CS shard
    never gathers enc12_elec and therefore never reads those 34 GB. No tap-filter argument is
    needed — not touching IS not loading. But laziness is not a speedup, and for WS it is a
    slowdown; pick with MMAP_DEFAULT and A/B with --mmap/--no-mmap before trusting a change.
    """
    s, t = session
    rec = torch.load(f"{cache_dir}/enc_s{s}_t{t}_{tag}.pt", map_location="cpu",
                     weights_only=False, mmap=mmap)
    if _ELEC_LABELS_SIDECAR is not None and rec.get("elec_labels") is None:
        lab = _ELEC_LABELS_SIDECAR.get(f"s{s}_t{t}")
        if lab is not None:
            if lab.shape[0] != rec["feats"]["enc12_elec"]["raw"].shape[1]:
                raise ValueError(
                    f"sidecar labels ({lab.shape[0]}) != enc12_elec electrodes "
                    f"({rec['feats']['enc12_elec']['raw'].shape[1]}) for s{s}_t{t}")
            rec["elec_labels"] = lab
    return rec


# ── sharded units (one SLURM array task each; all cells are independent) ────────────
# The 15 board tasks within a shard are independent and share the loaded cache, so a shard is
# itself embarrassingly parallel — but the 20263259 array ran them SERIALLY on 8 cores, which
# is why shards sat at ~100% CPU (1 core of 8) for long stretches. _map_tasks forks AFTER the
# cache is opened: children inherit it copy-on-write (and under mmap share the page cache
# outright), so the per-task serial work (gather / fp32 / standardize) of one task overlaps
# the threaded BLAS of another instead of stalling behind it.
#
# Memory is the ceiling, not cores: a WS worker's private fp32 slices are ~22 GB for the
# enc12_elec tap and ~44 GB while a norm's standardized copy coexists with the raw one, so
# --workers must be set against --mem (WS ~3, CS ~8 — CS's parcel taps are ~40x smaller).
# Give each worker its own BLAS threads via OMP_NUM_THREADS = cpus-per-task / workers.
_SHARED: dict = {}


def _task_worker(task):
    """Runs in a forked child. Reads the cache from _SHARED — NEVER take it as an argument:
    Pool pickles arguments, which would serialize a multi-GB cache per task."""
    fn, taps = _SHARED["fn"], _SHARED["taps"]
    return task, fn(task, taps)


def _map_tasks(fn, taps, workers) -> dict:
    """{task: cell} over BOARD_TASKS, optionally across forked workers."""
    if workers <= 1:
        return {task: fn(task, taps) for task in BOARD_TASKS}
    _SHARED["fn"], _SHARED["taps"] = fn, taps          # set BEFORE fork ⇒ inherited, not pickled
    with mp.get_context("fork").Pool(workers) as pool:
        return dict(pool.map(_task_worker, BOARD_TASKS))


def _ws_shard(cache_dir, tag, session, taps=WS_TAPS, workers=1,
              mmap=MMAP_DEFAULT["ws"]) -> dict:
    rec = _load(cache_dir, session, tag, mmap=mmap)
    out = _map_tasks(lambda task, tp: _ws_cell(rec, task, tp), taps, workers)
    return {"kind": "ws", "name": f"S{session[0]}T{session[1]}",
            "cells": {f"{tag}|{k}": v for k, v in out.items()}}


def _cs_shard(cache_dir, tag, cell, taps=CS_TAPS, workers=1,
              mmap=MMAP_DEFAULT["cs"]) -> dict:
    taps = tuple(t for t in taps if t not in ELEC_TAPS)   # CS is parcel-bridged by necessity
    anchor_rec = _load(cache_dir, CS_TRAIN_ANCHOR, tag, mmap=mmap)
    test_rec = _load(cache_dir, cell, tag, mmap=mmap)
    out = _map_tasks(lambda task, tp: _cs_cell(anchor_rec, test_rec, task, tp), taps, workers)
    return {"kind": "cs", "name": f"S{cell[0]}T{cell[1]}",
            "cells": {f"{tag}|{k}": v for k, v in out.items()}}


def _csession_shard(cache_dir, tag, cell, taps=CSESSION_TAPS, workers=1,
                    mmap=MMAP_DEFAULT["csession"]) -> dict:
    """Cross-session cell: train on the sibling trial (same subject), test on this session's
    held-out half. Keeps the per-electrode taps (electrode identity IS shared within subject)."""
    train_rec = _load(cache_dir, _sibling(cell), tag, mmap=mmap)
    test_rec = _load(cache_dir, cell, tag, mmap=mmap)
    out = _map_tasks(lambda task, tp: _csession_cell(train_rec, test_rec, task, tp), taps, workers)
    return {"kind": "csession", "name": f"S{cell[0]}T{cell[1]}",
            "cells": {f"{tag}|{k}": v for k, v in out.items()}}


def _blank(tags) -> dict:
    return {f"{tag}|{t}": {"ws": {}, "cs": {}, "csession": {}, "pinned": {}, "sat": {},
                           "n_parcels": {}, "n_elec": {}}
            for tag in tags for t in BOARD_TASKS}


def _absorb(res, sh) -> None:
    """Fold one shard in. res[task][kind]["tap|norm"][cell_name] = test AUROC — the full grid,
    every entry populated over every cell. No axis is collapsed at merge time either."""
    kind = sh["kind"]
    for k, val in sh["cells"].items():
        for gk, s in (val.get("cells") or {}).items():
            res[k][kind].setdefault(gk, {})[sh["name"]] = s["test"]
            if s.get("lam_pinned"):
                res[k]["pinned"].setdefault(f"{kind}:{gk}", []).append(sh["name"])
            if s.get("lam_sat"):
                res[k]["sat"].setdefault(f"{kind}:{gk}", []).append(sh["name"])
        if val.get("n_parcels") is not None:
            res[k]["n_parcels"][sh["name"]] = val["n_parcels"]
        if val.get("n_elec") is not None:
            res[k]["n_elec"][sh["name"]] = val["n_elec"]


def _merge(tags, shard_dir) -> dict:
    res = _blank(tags)
    for kind in ("ws", "cs", "csession"):
        for path in sorted(glob.glob(f"{shard_dir}/{kind}_*.json")):
            with open(path) as f:
                _absorb(res, json.load(f))
    return _finalize(res)


def _finalize(res: dict) -> dict:
    """Cohort-mean each grid entry over its cells (12 WS sessions / 10 CS cells / 12 cross-session)."""
    for c in res.values():
        for kind in ("ws", "cs", "csession"):
            c[f"{kind}_mean"] = {gk: float(np.nanmean(list(d.values())))
                                 for gk, d in c[kind].items() if d}
    return res


def _compute_all(cache_dir, tags, ws_taps=WS_TAPS, cs_taps=CS_TAPS) -> dict:
    res = _blank(tags)
    for tag in tags:
        for session in LITE_SESSIONS:
            sh = _ws_shard(cache_dir, tag, session, ws_taps)
            _absorb(res, sh)
            print(f"[{tag}] WS done {sh['name']}", flush=True)
        for cell in CS_TEST_CELLS:
            sh = _cs_shard(cache_dir, tag, cell, cs_taps)
            _absorb(res, sh)
            print(f"[{tag}] CS done {sh['name']}", flush=True)
        for cell in CSESSION_CELLS:
            sh = _csession_shard(cache_dir, tag, cell, CSESSION_TAPS)
            _absorb(res, sh)
            print(f"[{tag}] CSession done {sh['name']}", flush=True)
    return _finalize(res)


def _macro(res, tag, kind, gk) -> float:
    """Macro over the 15 board tasks of one grid entry's cohort mean."""
    v = [res[f"{tag}|{t}"].get(f"{kind}_mean", {}).get(gk, np.nan) for t in BOARD_TASKS]
    return float(np.nanmean(v)) if not all(np.isnan(x) for x in v) else float("nan")


def _report(tags, res) -> None:
    """Print the GRID, not a headline (Ben 2026-07-17).

    Every (tap, norm) is a complete protocol scored over every cell: 12 WS sessions / 10 CS
    cells × 15 tasks, λ chosen on each cell's val half. Nothing is selected across taps or
    norms and nothing is discarded, so the depth ladder (enc0/3/6/12), the per-electrode vs
    parcel-mean feature-unit diff, and the std/raw/std_target normalization contrast are all
    readable side by side, and each column is one protocol describable in a single sentence.
    Which column is THE board number is a claim, and claims are Ben's to make — this file's
    job is to put every number on the table honestly.
    """
    for tag in tags:
        for kind, label in (("cs", "CS (anchor S2T4 → 10 cells)"),
                            ("csession", "CSession (12 cells, sibling-trained)"),
                            ("ws", "WS (12 sessions)")):
            gks = sorted({g for t in BOARD_TASKS for g in res[f"{tag}|{t}"].get(kind, {})},
                         key=lambda g: (g.split("|")[1], g.split("|")[0]))
            if not gks:
                continue
            print(f"\n=== {label} test-half AUROC — tag={tag} ===", flush=True)
            print(f"  {'task':18s}" + "".join(f"{g:>18s}" for g in gks), flush=True)
            for t in BOARD_TASKS:
                m = res[f"{tag}|{t}"].get(f"{kind}_mean", {})
                print(f"  {t:18s}" + "".join(f"{m.get(g, float('nan')):18.4f}" for g in gks),
                      flush=True)
            print(f"  {'MACRO(15)':18s}"
                  + "".join(f"{_macro(res, tag, kind, g):18.4f}" for g in gks), flush=True)

        # Contrasts the grid above already contains, stated as differences so the ordering is
        # not left to the reader's eye. These are READS of the table, never separate fits.
        for kind, taps in (("cs", ENCODERS), ("csession", CSESSION_TAPS), ("ws", WS_TAPS)):
            norms = sorted({g.split("|")[1] for t in BOARD_TASKS
                            for g in res[f"{tag}|{t}"].get(kind, {})})
            if not norms:
                continue
            print(f"\n=== {kind.upper()} contrasts (macro over 15 tasks), tag={tag} ===",
                  flush=True)
            for nm in norms:
                if nm == "std":
                    continue
                for tp in taps:
                    x, y = (_macro(res, tag, kind, f"{tp}|{nm}"),
                            _macro(res, tag, kind, f"{tp}|std"))
                    if np.isnan(x) or np.isnan(y):
                        continue
                    print(f"  {tp:12s} {nm:11s} {x:.4f}  vs std {y:.4f}  Δ {x - y:+.4f}",
                          flush=True)
            # per-electrode vs parcel-mean at enc12 — meaningful where electrodes are shared
            # (WS trivially; CSession via identity intersection). CS has no shared electrodes.
            if kind in ("ws", "csession"):
                for nm in norms:
                    e, q = (_macro(res, tag, kind, f"enc12_elec|{nm}"),
                            _macro(res, tag, kind, f"enc12|{nm}"))
                    if not (np.isnan(e) or np.isnan(q)):
                        print(f"  [diff] enc12 per-electrode − parcel-mean ({nm}) = {e - q:+.4f}"
                              f"  ({e:.4f} vs {q:.4f})", flush=True)
                # depth-0 parity floor: enc0_elec vs enc12_elec, both per-electrode
                for nm in norms:
                    f0, f12 = (_macro(res, tag, kind, f"enc0_elec|{nm}"),
                               _macro(res, tag, kind, f"enc12_elec|{nm}"))
                    if not (np.isnan(f0) or np.isnan(f12)):
                        print(f"  [diff] enc12_elec − enc0_elec ({nm}) = {f12 - f0:+.4f}"
                              f"  ({f12:.4f} vs {f0:.4f})  [per-electrode depth gain]", flush=True)

    # λ pin check. The two boundaries are NOT the same failure and must not be counted together:
    #   HI (λ=LAM_MULTS[-1]) → BENIGN. AUROC saturates as λ→∞ (α→(1/λ)·y ⇒ scores→(1/λ)·K@y, a
    #      positive rescale AUROC ignores; asserted in test_auroc_saturates_at_high_lambda).
    #      "Maximal shrinkage won" is the true answer; widening the grid cannot change the number.
    #   LO (λ=LAM_MULTS[0])  → REAL truncation. No limit at the bottom, so the optimum is genuinely
    #      off-grid and the AUROC is an artifact.
    # Conflating them once cost a near-re-run of the whole 22-shard board for nothing.
    pinned = {f"{k}  {gk}": cells for k, c in res.items()
              for gk, cells in c.get("pinned", {}).items() if cells}
    sat = {f"{k}  {gk}": cells for k, c in res.items()
           for gk, cells in c.get("sat", {}).items() if cells}
    n_fits = sum(len(d) for c in res.values() for kind in ("ws", "cs")
                 for d in c.get(kind, {}).values())
    n_pin = sum(len(v) for v in pinned.values())
    n_sat = sum(len(v) for v in sat.values())
    if pinned:
        print(f"\n[check] λ grid: VIOLATED — {n_pin}/{n_fits} fits selected the LO boundary "
              f"({LAM_MULTS[0]:.1e}); their optimum is BELOW the grid and those AUROCs are "
              f"truncation artifacts. Lower LAM_MULTS[0] and re-run those cells."
              f" First 10: {list(pinned)[:10]}", flush=True)
    else:
        print(f"\n[check] λ grid: OK — 0/{n_fits} fits pinned to the LO boundary "
              f"({LAM_MULTS[0]:.1e}); no fit is truncated from below.", flush=True)
    print(f"[check] λ saturated (HI, benign): {n_sat}/{n_fits} fits chose λ={LAM_MULTS[-1]:.1e}, "
          f"i.e. maximal shrinkage. AUROC is constant past that point, so these are faithful "
          f"reports, NOT artifacts — do not widen the grid for them.", flush=True)
    print("[check] selection: every number above is a TEST-half AUROC. λ is the ONLY axis "
          "chosen on the val half (upstream train_test_splits.py:65); tap and norm are "
          "reported in full, never selected.", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--tags", default="board_r4_20k")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("all", "ws", "cs", "csession", "merge"), default="all")
    p.add_argument("--index", type=int, help="shard index (mode=ws|cs|csession)")
    p.add_argument("--no-std-target", dest="std_target", action="store_false",
                   help="drop the CS std_target (AdaBN) column — Ben's r5mod board is raw+std only.")
    p.add_argument("--shard-dir")
    p.add_argument("--workers", type=int, default=1,
                   help="fork this many task-workers per shard (memory-bound: WS ~3, CS ~8). "
                        "Set OMP_NUM_THREADS = cpus-per-task / workers.")
    p.add_argument("--mmap", dest="mmap", default=None, action="store_true",
                   help=f"force lazy paging (default per mode: {MMAP_DEFAULT}). Measured: it "
                        f"DEFERS the read into the gathers at ~1/4 sequential bandwidth — a "
                        f"win for CS (skips the 34 GB elec tap) and a loss for WS.")
    p.add_argument("--no-mmap", dest="mmap", action="store_false",
                   help="force one eager sequential read of the whole cache.")
    p.add_argument("--taps", default="",
                   help=f"comma-separated subset of {ALL_TAPS} (default: per-regime; CS drops "
                        f"{ELEC_TAPS} automatically)")
    p.add_argument("--elec-labels-sidecar",
                   help="pickle {'s{S}_t{T}': labels} to attach to records that lack elec_labels "
                        "(caches encoded before the field was stored, e.g. arm0/r4b). Validated "
                        "same-set upstream; _load re-asserts count-match before attaching.")
    p.add_argument("--lam-rule", choices=LAM_RULES, default="argmax",
                   help="how a VAL TIE picks λ. argmax (default, the PUBLISHED rule) keeps the "
                        "smallest tied λ; tiemax keeps the largest. Both read val only. Fix this "
                        "in advance and report it — do not run both and quote the better one.")
    p.add_argument("--dump-lam-grid", action="store_true",
                   help="record the WHOLE 25-point (val, test) λ curve per cell as `lam_grid`, "
                        "not just the selected point. Costs nothing — the grid is already swept — "
                        "and buys every λ rule offline, forever, off one run. Observation only: "
                        "no selection and no reported number moves. 🚫 picking λ to maximise the "
                        "dumped TEST curve is an ORACLE, a ceiling, never a result.")
    p.add_argument("--tap-ensemble", action="store_true",
                   help="ALSO report rank-averaged cross-TAP ensemble columns (ens:all / ens:top2 "
                        "/ ens:top3 / ens:auto, per norm). Each tap keeps its own val-selected λ "
                        "and is fit separately; only the predictions are combined, so this is NOT "
                        "depth concat (closed, negative). Every rule reads the val half only. "
                        "Per-tap columns are unchanged.")
    p.add_argument("--lam-ensemble", action="store_true",
                   help="ALSO report rank-averaged WITHIN-tap ensembles over λ as "
                        "lamall:/lam3:/lamge: columns. The λ-grid dump measured a +.0070 ORACLE "
                        "gap on the CS board, i.e. val picks λ noisily; averaging a val-defined "
                        "neighbourhood is the submittable way to spend that. Val-only, free "
                        "(the grid is already swept), per-tap columns unchanged.")
    p.add_argument("--lam-cv", action="store_true",
                   help="ALSO report '<tap>|std_cv', identical to '<tap>|std' except that λ is "
                        f"selected by contiguous {LAM_CV_K}-fold CV INSIDE THE TRAIN HALF instead "
                        "of on the val half. Not another val-defined rule (those are closed and "
                        "negative) — it replaces the noisy estimate rather than re-weighting it, "
                        "using ~14x more rows, and touches no held-out data at all. Free: a fold's "
                        "Gram is a submatrix of the one already formed.")
    p.add_argument("--rbf", action="store_true",
                   help="ALSO report an RBF kernel-ridge column '<tap>|std_rbf' beside the linear "
                        "'<tap>|std', fit on the same standardized matrices, splits and λ grid, "
                        "sharing one set of GEMMs. γ = mult/median(d²_train) selected on val "
                        "jointly with λ. The linear column is untouched, so it is the control.")
    p.add_argument("--pool-d", type=int, default=256,
                   help="model width d, used to reshape a unit's cached (k_full*d) block before a "
                        "gpool:/bpool: mean. Asserted to divide the real width.")
    p.add_argument("--pool-bands", default="4,16,32",
                   help="tokens per band in pack_r4 order [SLOW,MID,HGA] for bpool:. Must sum to "
                        "k_full = width/d; asserted, never assumed.")
    args = p.parse_args()

    global REPORT_STD_TARGET, _ELEC_LABELS_SIDECAR, LAM_RULE, POOL_D, POOL_BANDS, LAM_GRID_DUMP
    global TAP_ENSEMBLE, LAM_ENSEMBLE, RBF, LAM_CV
    REPORT_STD_TARGET = args.std_target
    RBF = args.rbf
    LAM_CV = args.lam_cv
    # REFUSED rather than silently mis-scored: both ensemble builders treat every (tap, norm) in
    # the grid as a member, so std_cv — which shares std's test curve exactly — would enter as a
    # near-duplicate member and inflate its own weight.
    assert not (LAM_CV and (args.tap_ensemble or args.lam_ensemble)), \
        "--lam-cv cannot be combined with an ensemble flag: std_cv shares std's test curve"
    LAM_RULE = args.lam_rule
    LAM_GRID_DUMP = args.dump_lam_grid
    TAP_ENSEMBLE = args.tap_ensemble
    LAM_ENSEMBLE = args.lam_ensemble
    POOL_D = args.pool_d
    POOL_BANDS = tuple(int(b) for b in args.pool_bands.split(",") if b.strip())
    print(f"[check] lam_rule={LAM_RULE} (published board = argmax) "
          f"lam_grid_dump={LAM_GRID_DUMP} tap_ensemble={TAP_ENSEMBLE} "
          f"lam_ensemble={LAM_ENSEMBLE}", flush=True)
    if args.elec_labels_sidecar:
        with open(args.elec_labels_sidecar, "rb") as fh:
            _ELEC_LABELS_SIDECAR = pickle.load(fh)
        print(f"[sidecar] attaching elec_labels for {len(_ELEC_LABELS_SIDECAR)} sessions",
              flush=True)

    tags = tuple(t.strip() for t in args.tags.split(","))
    # Default taps are per-REGIME (WS/CSession electrode-only, CS parcel-only); --taps overrides.
    _mode_taps = {"ws": WS_TAPS, "cs": CS_TAPS, "csession": CSESSION_TAPS}
    taps = (tuple(t.strip() for t in args.taps.split(",") if t.strip())
            or _mode_taps.get(args.mode, ALL_TAPS))
    _validate_taps(taps)

    if args.mode in ("ws", "cs", "csession"):
        cells = {"ws": LITE_SESSIONS, "cs": CS_TEST_CELLS, "csession": CSESSION_CELLS}[args.mode]
        cell = cells[args.index]
        fn = {"ws": _ws_shard, "cs": _cs_shard, "csession": _csession_shard}[args.mode]
        t0 = time.perf_counter()
        use_mmap = MMAP_DEFAULT[args.mode] if args.mmap is None else args.mmap
        sh = fn(args.cache_dir, tags[0], cell, taps, workers=args.workers, mmap=use_mmap)
        os.makedirs(args.shard_dir, exist_ok=True)
        out = f"{args.shard_dir}/{args.mode}_{sh['name']}.json"
        with open(out, "w") as f:
            json.dump(sh, f, indent=2)
        print(f"wrote {out}", flush=True)
        # THE PRECONDITION FOR --lam-rule, printed rather than assumed. If tied_ge2 is ~0 on the
        # real board then argmax and tiemax are the SAME function here and the switch is not worth
        # a run -- the tied plateau was found on a synthetic fixture and need not survive at
        # n_va~875. Like _phase_report this is the PARENT's census only: with --workers>1 the
        # children's counters die with them, so read it from a workers=1 shard.
        _f, _t = _TIE_STATS["fits"], _TIE_STATS["tied_ge2"]
        print(f"[check] lam ties: {_t}/{_f} fits had >=2 tied λ "
              f"({100.0 * _t / _f if _f else 0.0:.1f}%), mean tied λ per fit "
              f"{_TIE_STATS['tied_total'] / _f if _f else 0.0:.2f} of {len(LAM_MULTS)} "
              f"[workers={args.workers}: parent-only if >1]", flush=True)
        if RBF and _RBF_STATS["fits"]:
            # A γ grid whose picks pile up on an END has not answered the question -- it has run
            # out of room. Printed so that is visible in the log, not reverse-engineered later.
            _pk = _RBF_STATS["picked"]
            _hist = " ".join(f"{gm:g}:{_pk.get(gm, 0)}" for gm in GAMMA_MULTS)
            _ends = _pk.get(GAMMA_MULTS[0], 0) + _pk.get(GAMMA_MULTS[-1], 0)
            print(f"[check] rbf γ picks over {_RBF_STATS['fits']} fits: {_hist} "
                  f"| on a grid END {100.0 * _ends / _RBF_STATS['fits']:.0f}% "
                  f"(high ⇒ WIDEN GAMMA_MULTS before quoting)", flush=True)
            _c = np.asarray(_RBF_STATS["conc"], dtype=np.float64)
            print(f"[check] rbf distance concentration sd(d²)/mean(d²): "
                  f"min {_c.min():.4f} median {np.median(_c):.4f} max {_c.max():.4f} "
                  f"over {_c.size} fits — BELOW ~0.05 THE KERNEL CANNOT SEPARATE and a null "
                  f"result says nothing about nonlinearity, only about dimension", flush=True)
        if LAM_CV and _CV_STATS["pairs"]:
            _p = np.asarray(_CV_STATS["pairs"], dtype=np.float64)
            _up = int((_p[:, 1] > _p[:, 0]).sum())
            print(f"[check] λ rule over {len(_p)} paired fits: median mult val {np.median(_p[:, 0]):.3g} "
                  f"vs train-CV {np.median(_p[:, 1]):.3g} | CV picked LARGER on {_up}/{len(_p)} "
                  f"({100.0 * _up / len(_p):.0f}%) — the PREDICTED direction was larger; if the "
                  f"column loses while this is high, it lost by over-shrinking", flush=True)
        # Phase totals are per-PROCESS: with --workers>1 the children's timers die with them,
        # so this table is the parent's view (load + merge) only. Profile with --workers 1.
        _phase_report(f"{args.mode} {sh['name']} workers={args.workers} "
                      f"mmap={use_mmap} wall={(time.perf_counter() - t0) / 60:.1f} min")
        return

    res = _merge(tags, args.shard_dir) if args.mode == "merge" else _compute_all(
        args.cache_dir, tags, taps, tuple(t for t in taps if t not in ELEC_TAPS))
    _report(tags, res)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {args.out}\nMERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
