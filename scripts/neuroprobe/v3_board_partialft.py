"""Board partial fine-tune: block-12 MLP + leave-one-out ridge driver, all three board regimes.

Ports the WS pilot's winning recipe (``v3_ws_partialft_pilot.py``, +.0320 over 28 cells, 24/28,
p=.0002) to the actual leaderboard: 15 tasks, 3 regimes, and — for ws/csession — the ELECTRODE
unit the board reports, which the pilot never measured.

WHAT IS BEING MEASURED. Per (regime, cell, task[, fold]):
  A = frozen block 12 + ridge          <- the published board entry, recomputed in-path
  C = fine-tuned block 12 + THE SAME ridge
and the headline is d(C - A). There is no separate "D" cell: under ``--driver ridge`` the logistic
head is never in the graph, so a trained-head number would be the zero-init head's ~0.5 dressed up
as a readout. The pilot documents this at :901 and it is why D was banned from that report.

🔴 λ IS VAL-SELECTED ON BOTH SIDES, over the BOARD's grid. ``v3_board_readout._grid_cells`` ->
``_select_lam`` picks λ on the val half, so the published frozen entry is val-selected and A must
be too. The board grid is ``LAM_MULTS = logspace(-4, 4, 25)`` (v3_board_readout.py:168); the
pilot's is 7 multipliers spanning 0.03-30. Both are MULTIPLIERS on the same base
(``basel = trace(G)/n``, pilot :234 == board ``base``) — the trap is not scale but SPAN: the
board's selected λ routinely lands outside the pilot's 7-point range, so scoring A on the pilot
grid would silently report a different frozen number than the one on the leaderboard. This script
scores both A and C on ``BRD.LAM_MULTS``.

WHY THIS IMPORTS THE PILOT INSTEAD OF RE-DERIVING IT. The block-split numerics (``_run_tail``,
``_ridge_eval``, ``_loo_ridge_risk``, ``_pool_t``) are exactly what the L0/L1/L2 parity gates
certify — L0 proves block12(cached tap-11) reproduces the on-disk enc12 to 5e-4 relative. Retyping
them would put that equivalence back at risk for no gain. The BOARD logic (regimes, cells, splits,
units, λ grid) is written fresh here.

THROUGHPUT — what this costs and what was actually done about it.
Per epoch the loop is ``4*n_tr + n_va + n_te`` block-12 forward-units: 3*n_tr for the training
fwd+bwd, then three no-grad eval passes. Measured on the pilot (job 2789630_1, S2T1): 4.2 s/epoch,
and 1,066 s of that 1,182 s job was the epoch loop — the tap-11 encode is only ~10%.
  1. Tap-11 is encoded ONCE per session and reused by every task, fold and (for cs) every test
     cell. Correct, and mandatory, but it buys ~10%, not the bulk.
  2. Only tap 11 -> block 12 -> enc12. No enc0/3/6/9: we neither fine-tune nor report them.
  3. The TEST ridge is computed ONCE, at the end, on the restored best-val state — not every
     epoch. Exactly ``n_te / (4*n_tr + n_va + n_te)`` = 875/8748 = 10.0% of every epoch, free.
  4. The λ sweep stays: ``_ridge_eval`` does one ``eigh`` and every λ reuses the eigenvalues, so
     the 25-point grid costs the same as one λ. It is also what makes A comparable to the board.
  5. Task grouping was MEASURED AND REJECTED, not assumed. All 15 board tasks have independent
     partitions: per session/fold there are ~2,100-6,400 rows one task calls train and another
     calls test (``audit/board_split_signature.py``). One multitask fine-tune cannot serve 15
     evaluations without training on other tasks' test rows. Multitask survives only in its safe
     form — extra label COLUMNS on the primary task's own train rows (pilot :677), never extra
     rows.
  6. Patience stays 15. Dropping to 10 is a real 15.7% (712 pilot cell-runs, mean ep*=16.9,
     median 12, p90 40) but only 48% of cells have ep* <= 10, so selection is provably unchanged
     on barely half — not a trade worth making against a +.0304 bar. Instead the per-epoch val
     trace is LOGGED, so patience can be retuned from the logs for free next time.

CS SCOPE IS PER TEST CELL, on the anchor∩test parcel intersection — Ben's contract. The anchor's
tap-11 is still encoded once and amortized across all 10 test cells x 15 tasks; only the fine-tune
itself repeats, because the intersection changes the features and therefore the gradients.
"""
from __future__ import annotations

import argparse
import copy
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch


def _load_sibling(name):
    """Load a sibling script as a module (they are scripts, not package members)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FTP = _load_sibling("v3_ws_partialft_pilot")
BRD = _load_sibling("v3_board_readout")
ENC = FTP.ENC
RDO = FTP.RDO

TAP = 12
SPLIT_AT = 11
CERT_ROWS = 256          # rows parity-L0 probes; kept in fp32 regardless of --x11-dtype
# Every regime's REPORTED tap. ws/csession report the per-electrode tap, cs reports the parcel
# tap (v3_board_readout.py:115-117) -- this asymmetry is the whole reason --unit exists.
REGIME_UNIT = {"ws": "elec", "csession": "elec", "cs": "parcel"}
REGIME_CELLS = {"ws": BRD.LITE_SESSIONS, "csession": BRD.CSESSION_CELLS, "cs": BRD.CS_TEST_CELLS}


# ── features ────────────────────────────────────────────────────────────────────────────────
def _unit_feats(y, grid, cols, unit, sel):
    """(B, M, d) tail output -> (B, n_units, k*d), in the encode cache's EXACT layout.

    Mirrors ``v3_probe_encode_r4.py:440-442`` verbatim, including the asymmetry that bites:
    the parcel tap keeps only contacts whose parcel is in ``present`` and MEANS them, while the
    elec tap keeps ALL n canonical contacts unpooled. ``sel`` is the cross-record column subset
    (parcel intersection, or shared-electrode set) and is applied AFTER the unit reduction,
    because both index the reduced axis.
    """
    b = y.shape[0]
    z = y.float().reshape(b, -1, grid.k_full, y.shape[-1])       # (B, n, k, d)
    z = FTP._pool_t(z, cols) if unit == "parcel" else z.reshape(b, z.shape[1], -1)
    return z if sel is None else z[:, sel]


def _finite(y, idx):
    idx = np.asarray(idx, dtype=np.int64)
    return idx[np.isfinite(y[idx])]


# ── one fine-tuned cell ─────────────────────────────────────────────────────────────────────
def _stub_rows(y):
    """A tiny two-class row subset, used as a placeholder test set on val-only epochs.

    ``FTP._ridge_eval`` scores val and test from one Gram, so it always wants a test block. Passing
    the real val matrix there would cost an extra n_va x F x n_tr kernel — ~2.6 TFLOP at the
    electrode unit, which is the same order as the block-12 forward the 10% cut exists to avoid.
    Four rows cost nothing. Two classes so AUROC is defined rather than nan-ing through the guard.
    """
    y = np.asarray(y)
    pos = np.where(y > 0)[0][:2]
    neg = np.where(y <= 0)[0][:2]
    idx = np.concatenate([pos, neg]) if pos.size and neg.size else np.arange(min(4, len(y)))
    return np.sort(idx.astype(np.int64))


# ── epoch ENSEMBLING (stop selecting an epoch; average over several) ────────────────────────
# The epoch-curve arm measured a +.0105 oracle gap on WS (112/0/8) that no val-based STOPPING rule
# recovers: le4/le6/le8/le12 and smooth3 all LOSE to plain argmax. The val trace is too noisy to
# PICK the best epoch -- which does not mean the epochs are bad. Averaging their predictions can
# beat any single one even when val cannot say which one to take.
#
# 🔴 THE ONLY THING THAT MAKES THIS SUBMITTABLE is that the epoch SET is a function of the val
# trace alone. That is enforced structurally: `_ensemble_index_sets` takes `vals` and nothing else,
# so it cannot read test data even by accident (test_board_partialft_epoch_ensemble.py).
# `ens_top1` is the SELF-CHECK -- averaging one epoch IS the published rule, so it must reproduce
# `test_c`; the run prints the difference rather than assuming it.

def _rank01(s):
    """Scores -> ranks in [0, 1]. AUROC depends only on the ORDER of scores, so a mean of RAW
    scores is not order-only: an epoch whose val-selected λ happens to be smaller emits
    larger-magnitude scores and outvotes the rest for reasons that have nothing to do with how well
    it ranks the test set. Ranks put every epoch on one scale. Ties are broken by position, which
    is measure-zero for a continuous ridge score and cannot bias a mean either way."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    r = np.empty(n, dtype=np.float64)
    r[np.argsort(s, kind="stable")] = np.arange(n, dtype=np.float64)
    return r / (n - 1)


def _ensemble_index_sets(vals):
    """{rule -> list of epoch INDICES to average}, from the val trace and NOTHING else.

    Index 0 is the frozen entry (epoch 0), matching `trace`'s ordering. Non-finite vals are failed
    fits and are dropped: left in, a nan would compare False against everything and could win a
    top-k slot by default. Ties break toward the EARLIER epoch, the same way the training loop's
    strict `>` does, so `ens_top1` picks exactly the epoch the loop selected."""
    keys = ("ens_all", "ens_valge0", "ens_top3", "ens_top1", "ens_last3", "ens_last5", "ens_swa")
    ok = [i for i, v in enumerate(vals) if np.isfinite(v)]
    if not ok:
        return {k: [] for k in keys}
    order = sorted(ok, key=lambda i: (-float(vals[i]), i))
    base = float(vals[ok[0]])
    return {"ens_all": list(ok),
            "ens_valge0": [i for i in ok if float(vals[i]) >= base],
            "ens_top3": sorted(order[:3]),
            "ens_top1": [order[0]],
            # THE CANONICAL WEIGHT-AVERAGING RULE, and the only one here that never reads val:
            # average the last N checkpoints of the run (Vaswani et al. 2017 used N=5 for the base
            # transformer; SWA averages the trajectory's tail). Every other rule above is one we
            # chose and therefore have to defend; this one is the literature's default and is the
            # head-to-head a reviewer will ask for. It is reported for the prediction ensembles
            # too, at no extra cost, so weight- and prediction-averaging can be compared under the
            # SAME epoch set instead of each at its own favourite rule.
            "ens_last3": ok[-3:],
            "ens_last5": ok[-5:],
            # SWA with swa_start = the val-argmax epoch: average from the optimum to the END of
            # the run. This is how patience and the averaging window are reconciled -- the window
            # length IS patience, so there is no second constant to defend, and the ~15 epochs
            # patience currently computes and throws away become the ingredient list. Unlike
            # last-N the window starts AT the optimum instead of landing 15 epochs past it.
            "ens_swa": [i for i in ok if i >= order[0]]}


def _epoch_ensembles(vals, scores, y_te, auroc):
    """{rule -> test AUROC of the rank-averaged prediction} for every rule in `_ensemble_index_sets`."""
    sets = _ensemble_index_sets(vals)
    ranks = {}
    out = {}
    for name, idxs in sets.items():
        if not idxs:
            out[name] = float("nan")
            continue
        for i in idxs:
            if i not in ranks:
                ranks[i] = _rank01(scores[i])
        out[name] = float(auroc(np.mean([ranks[i] for i in idxs], axis=0), y_te))
    return out


def _average_states(states):
    """Elementwise mean of N tail-block state lists -> ONE tail-block state list.

    Entries IDENTICAL across every member are passed through UNTOUCHED. Only block 12's MLP is
    unfrozen, so most of the block holds the same value at every epoch, and a floating mean of
    equal values does not reliably return that value (x+x+x need not be 3x, and dividing by 3 need
    not undo it). Passing them through is what keeps `soup_top1 == test_c` EXACT and stops the
    frozen half of the block drifting a ulp per rule.

    Non-floating entries are never averaged: an index buffer has no meaningful mean, so one that
    disagrees across epochs is an upstream bug and this raises rather than rounding it away.

    ⚠️ Sound because block 12 is LayerNorm-only (attention.py:80-81) and its positional buffers are
    persistent=False, hence absent from state_dict. A BatchNorm here would need SWA's separate
    running-stat re-estimation pass; see test_v3_board_partialft_weight_soup.py."""
    first = states[0]
    out = []
    for bi, sd0 in enumerate(first):
        avg = {}
        for k, v0 in sd0.items():
            members = [s[bi][k] for s in states]
            if all(torch.equal(m, v0) for m in members[1:]):
                avg[k] = v0.clone()
                continue
            if not torch.is_floating_point(v0):
                raise ValueError(
                    f"non-float state entry {k!r} differs across epochs; refusing to average it")
            avg[k] = torch.stack([m.float() for m in members]).mean(0).to(v0.dtype)
        out.append(avg)
    return out


def _greedy_soup(vals, states, soup_val):
    """Wortsman et al. 2022 greedy soup, over the epochs of one fine-tune.

    Sort candidates by val, seed the soup with the best one, then walk the rest in order and keep
    a candidate ONLY if souping it in raises the val of the resulting AVERAGE. The criterion is
    the soup's val, not the candidate's own -- that is the whole difference from `valge0` and
    `top-k`, which score each epoch in isolation and can therefore average in members that are
    individually good but redundant or in tension with what is already in the pot.

    WHY THIS IS THE RULE WE LEAD WITH. It is published, it is a weight-average (so it ships ONE
    model), and it has NO free parameter -- the val comparison decides how many members go in.
    `last-N` carries an N that Vaswani et al. themselves tuned per model (5 base / 20 big), and it
    additionally assumes the run ENDS near its best, which ours does not: patience-15 stopping
    leaves the tail ~15 epochs past the val optimum at ~80% of peak LR.

    `soup_val` re-measures the averaged weights down the SAME ridge path the training loop used,
    so the seed's score is directly comparable to the rest. Non-finite vals are failed fits and
    can never enter. Ties break toward the earlier epoch, matching the loop's strict `>`.

    Returns the ingredient INDICES; the caller averages them and scores test once."""
    ok = [i for i, v in enumerate(vals) if np.isfinite(v)]
    if not ok:
        return []
    order = sorted(ok, key=lambda i: (-float(vals[i]), i))
    ing = [order[0]]
    cur = float(soup_val(_average_states([states[order[0]]])))
    for i in order[1:]:
        cand = ing + [i]
        v = float(soup_val(_average_states([states[j] for j in cand])))
        # STRICT. A member has to earn its place; on a tie the smaller soup is the simpler model.
        if np.isfinite(v) and v > cur:
            ing, cur = cand, v
    return sorted(ing)


def _weight_soups(vals, states, refit):
    """{soup_<rule> -> test AUROC after LOADING the averaged weights}, over the SAME val-only epoch
    sets `_epoch_ensembles` uses.

    Sharing `_ensemble_index_sets` is what makes soup-vs-ensemble apples-to-apples structurally,
    instead of two hand-written rule tables that can drift apart. `refit` loads an averaged state
    and returns the test AUROC at the val-selected lambda; it is injected so the rule logic is
    testable without a GPU."""
    sets = _ensemble_index_sets(vals)
    out = {}
    for name, idxs in sets.items():
        key = "soup_" + name[len("ens_"):]
        if not idxs:
            out[key] = float("nan")
            continue
        out[key] = float(refit(_average_states([states[i] for i in idxs])))
    return out


def _snap_tail(enc):
    """CPU copy of the recomputed tail's state. CPU because GPU memory is the binding constraint
    here -- the big ws sessions already OOM at one GPU's share."""
    return [{k: v.detach().to("cpu", copy=True) for k, v in b.state_dict().items()}
            for b in enc.blocks[SPLIT_AT:TAP]]


def _load_tail(enc, st):
    for b, sd_ in zip(enc.blocks[SPLIT_AT:TAP], st):
        b.load_state_dict(sd_)


def _run_cell(args, enc, feats_tr, feats_ev, rows, ybin_tr, ybin_ev, msk_tr, col, dcols, lams,
              tag):
    """Fine-tune block 12's MLP on ``rows['tr']``; return A, C and the epoch trace.

    ``feats_tr``/``ybin_tr`` read the TRAIN record; ``feats_ev``/``ybin_ev`` read the TEST record.
    They are kept as separate arguments on purpose: in ws they happen to be the same objects, but
    in cs/csession the train anchor is a DIFFERENT subject with its own label array, and row
    indices are valid in both — so any routing cleverness would mis-read labels silently rather
    than raise.
    """
    tr, va, te = rows["tr"], rows["va"], rows["te"]
    params = FTP._arm_params(enc, "mlp")
    for q in enc.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    # CONSTANT LR, DELIBERATELY. This replaced CosineAnnealingLR(T_max=args.epochs) on 07-31.
    #
    # The cosine was never actually annealing: T_max was the 80-epoch BUDGET while patience-15
    # stopping ends runs at ~24, so we only ever traversed the first 30% and the LR stayed between
    # 80% and 100% of peak. Worse, because the stop epoch varies per cell and T_max did not, a
    # cell that stopped at 15 epochs trained at a systematically HIGHER average LR than one that
    # ran 40 -- the effective schedule was a function of how long early stopping happened to let
    # the cell run. That is a confound across the very cells the board test pairs over.
    #
    # Constant removes it, and it is the schedule the analysis actually assumes: averaging weights
    # across checkpoints (Polyak-Ruppert, SWA) is justified when the iterates sample a STATIONARY
    # distribution around the basin, which is what a constant LR produces and a decaying one does
    # not. We were relying on that regime by accident; now we ask for it.
    #
    # 🔴 This CHANGES TRAINING. Every partial-FT number produced before 07-31 (R21/R23/R25, incl.
    # the .7014 WS ensemble read) is on the old truncated cosine and is NOT comparable to a run
    # from this file -- A and C both move. Re-measure, never mix.
    rng = np.random.default_rng(args.seed)
    # SPLIT BY CONSUMER, NOT BY CONVENIENCE. ``_ridge_eval`` uses y_tr as a TORCH tensor on z's
    # device (it forms v.T @ y in the dual solve) but hands y_va/y_te to RDO.auroc, which calls
    # np.asarray -- and that raises on a CUDA tensor (v3_probe_readout_r4.py:93). Keeping all
    # three on the GPU crashed at the very first ridge_now(True), before any weight moved.
    yt = ybin_tr[tr][:, col]
    yv = ybin_ev[va][:, col].cpu().numpy()
    ye = ybin_ev[te][:, col].cpu().numpy()
    stub = _stub_rows(yv)

    def ridge_now(want_test, want_scores=False):
        """val (+ optionally test) AUROC at val-selected λ on the CURRENT weights."""
        with torch.no_grad():
            z_tr = FTP._flat16(feats_tr(tr, False))
            z_va = FTP._flat16(feats_ev(va, False))
            if want_test:
                z_te, y_te = FTP._flat16(feats_ev(te, False)), ye
            else:
                z_te, y_te = z_va[stub], yv[stub]
            return FTP._ridge_eval(z_tr, z_va, z_te, yt, yv, y_te, lams, want_scores)

    # ── A: the frozen board entry, recomputed in-path (ep 0) ────────────────────────────────
    # Scores are collected only under --dump-epoch-test, which already pays the per-epoch test
    # forward; the ensembles therefore cost the rank sort and nothing else on the GPU.
    ens_scores: list = []
    soup_states: list = []
    if args.dump_epoch_test:
        v0, _t0const, a_test, a_df, s0 = ridge_now(True, True)
        ens_scores.append(s0)
    else:
        v0, _t0const, a_test, a_df = ridge_now(True)
    if args.weight_soup:
        soup_states.append(_snap_tail(enc))
    best = {"val": float(v0), "epoch": 0, "state": None}
    trace = [(0, float(v0), float(a_test) if args.dump_epoch_test else None)]
    t0 = time.time()
    n_ep = 0
    for ep in range(1, args.epochs + 1):
        warming = ep <= args.warmup_epochs
        for q in params:
            q.requires_grad_(not warming)
        enc.eval() if warming else enc.train()
        order = rng.permutation(len(tr))
        for s in range(0, len(order), args.train_batch):
            pos = order[s:s + args.train_batch]
            z = feats_tr(tr[pos], True).reshape(len(pos), -1)
            # WHICH LABEL COLUMNS DRIVE THE UPDATE — see --driver-tasks. Always on the primary
            # task's own train rows; extra ROWS would be leakage, since the 15 board tasks do NOT
            # share a partition (audit/board_split_signature.py).
            loss = FTP._loo_ridge_risk(z, ybin_tr[tr[pos]][:, dcols], msk_tr[tr[pos]][:, dcols])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()
        enc.eval()
        # THE 10% CUT: val only. Test is computed once, at the end, on the restored best state.
        # --dump-epoch-test BUYS THE CUT BACK (n_te/(4*n_tr+n_va+n_te) = ~10% per epoch) to record
        # the test curve alongside val. OBSERVATION ONLY, and provably so: _ridge_eval builds
        # val_const from k_va and y_va alone (v3_ws_partialft_pilot.py:250), so the val trace,
        # `best`, and the selected epoch are bit-identical either way -- asserted in
        # test_board_partialft_epoch_curve.py. 🚫 The curve is a MEASUREMENT of how much headroom
        # val-selection leaves on the table; selecting an epoch ON it is the winner's curse that
        # already cost us a 10x shrink, so the reported number still comes from a val-fixed rule.
        if args.dump_epoch_test:
            cv, _, ct, _, s_ep = ridge_now(True, True)
            ens_scores.append(s_ep)
        else:
            cv, ct = ridge_now(False)[0], None
        trace.append((ep, float(cv), None if ct is None else float(ct)))
        if args.weight_soup:
            soup_states.append(_snap_tail(enc))
        if np.isfinite(cv) and cv > best["val"]:
            best = {"val": float(cv), "epoch": ep,
                    "state": copy.deepcopy([b.state_dict() for b in enc.blocks[SPLIT_AT:TAP]])}
        n_ep = ep
        if ep - best["epoch"] >= args.patience:
            break

    if best["epoch"] == 0:
        c_test = a_test            # FT never beat frozen on val -> C IS A, no extra forward
    else:
        for b, sd_ in zip(enc.blocks[SPLIT_AT:TAP], best["state"]):
            b.load_state_dict(sd_)
        c_test = ridge_now(True)[2]
    # ── E: epoch ensembles (val-only rules, no extra forward) ───────────────────────────────
    ens: dict = {}
    if args.dump_epoch_test and len(ens_scores) == len(trace):
        ens = _epoch_ensembles([v for _, v, _ in trace], ens_scores, ye, RDO.auroc)
        # SELF-CHECK, PRINTED NOT ASSUMED. ens_top1 averages exactly one epoch -- the one the loop
        # selected -- so it must reproduce c_test. A non-zero gap means the ensemble is reading a
        # different curve than the run selected on, which would invalidate every other rule here.
        print(f"[enscheck] {tag} ens_top1={ens['ens_top1']:.6f} c_test={c_test:.6f} "
              f"d={ens['ens_top1'] - c_test:+.2e} n_ep_in_trace={len(trace)}", flush=True)
        print("[ens] " + tag + " " + " ".join(f"{k}={v:.4f}" for k, v in sorted(ens.items())),
              flush=True)

    # ── F: weight soups (same val-only sets, but averaged in PARAMETER space) ────────────────
    # Runs LAST because it mutates the encoder; `c_test` is already fixed above and the caller
    # re-runs FTP._restore(enc, pristine) before the next cell.
    soup: dict = {}
    if args.weight_soup and len(soup_states) == len(trace):
        def _refit(st):
            _load_tail(enc, st)
            return ridge_now(True)[2]
        soup = _weight_soups([v for _, v, _ in trace], soup_states, _refit)
        # GREEDY SOUP — the headline rule, and the only one here with no free parameter. It costs
        # one VAL evaluation per candidate (not test), which is the cheaper half of ridge_now.
        vtrace = [v for _, v, _ in trace]

        def _soup_val(st):
            _load_tail(enc, st)
            return ridge_now(False)[0]

        ing = _greedy_soup(vtrace, soup_states, _soup_val)
        soup["soup_greedy"] = float(_refit(_average_states([soup_states[i] for i in ing]))) \
            if ing else float("nan")
        soup["soup_greedy_n"] = float(len(ing))
        print(f"[greedy] {tag} n_ing={len(ing)} of {len(vtrace)} ing={ing} "
              f"test={soup['soup_greedy']:.4f}", flush=True)
        # SELF-CHECK, PRINTED NOT ASSUMED, and STRICTER than the ensemble's. Averaging one state is
        # an identity (frozen entries pass through, and the lone member of a mean is itself), so
        # this reloads exactly the weights the loop selected and must reproduce c_test to the bit
        # -- no rank-tie slack. A non-zero d means the per-epoch snapshots are not the states the
        # run selected on, and every other soup number is then meaningless.
        print(f"[soupcheck] {tag} soup_top1={soup['soup_top1']:.6f} c_test={c_test:.6f} "
              f"d={soup['soup_top1'] - c_test:+.2e} n_snap={len(soup_states)}", flush=True)
        print("[soup] " + tag + " " + " ".join(f"{k}={v:.4f}" for k, v in sorted(soup.items())),
              flush=True)

    sec = (time.time() - t0) / max(n_ep, 1)
    print(f"[cell] {tag} ep*={best['epoch']:3d} A={a_test:.4f} C={c_test:.4f} "
          f"d={c_test - a_test:+.4f} | df={a_df:.1f}/{len(tr)} | K_drv={len(dcols)} | "
          f"{n_ep}ep {sec:.2f}s/ep n_tr={len(tr)} n_va={len(va)} n_te={len(te)}", flush=True)
    print(f"[trace] {tag} " + " ".join(f"{e}:{v:.4f}" for e, v, _ in trace), flush=True)
    if args.dump_epoch_test:
        print(f"[tracet] {tag} " + " ".join(f"{e}:{t:.4f}" for e, _, t in trace), flush=True)
    return {"test_frozen_vallam": float(a_test), "test_c": float(c_test),
            "epoch_curve": [[e, v, t] for e, v, t in trace],
            "d": float(c_test - a_test), "best_epoch": int(best["epoch"]),
            "n_epochs_run": int(n_ep), "sec_per_epoch": float(sec), "ridge_df": float(a_df),
            "n_tr": int(len(tr)), "n_va": int(len(va)), "n_te": int(len(te)), **ens, **soup}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--regime", choices=("ws", "csession", "cs"), required=True)
    p.add_argument("--cell-index", type=int, required=True, help="shard: index into the regime's cells")
    p.add_argument("--board-cache-dir", required=True, help="encode cache of the SAME ckpt (parity + splits)")
    p.add_argument("--board-tag", required=True)
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--elec-labels-sidecar",
                   help="pickle {'s{S}_t{T}': labels} for caches that predate inline elec_labels "
                        "(board_r6_40k). REQUIRED for csession — without it the electrode "
                        "identity-intersect has nothing to intersect on.")
    p.add_argument("--unit", choices=("parcel", "elec"), default=None,
                   help="override the regime default; for the one-session smoke test only")
    p.add_argument("--tasks", default=None, help="comma list; default = all 15 board tasks")
    p.add_argument("--driver-tasks", default="report",
                   help="WHICH label columns enter _loo_ridge_risk: 'report' (default, ONLY the "
                        "reported task), 'all' (every board task), or a comma list. This is the "
                        "single knob that was silently different from the pilot: the pilot's "
                        "--multitask meant K=4 HOMOGENEOUS language/acoustic columns, the board "
                        "made it K=15 including visual tasks whose frozen AUROC is at chance "
                        "(frame_brightness A=0.507). _loo_ridge_risk normalizes by m2.sum(), so a "
                        "column is weighted by its LABEL COUNT, not by how much signal it "
                        "carries: unfittable columns contribute large irreducible residuals and "
                        "near-noise gradient. 'report' is also the only setting under which the "
                        "driver is what its own docstring claims -- 'the generalization risk of "
                        "EXACTLY the estimator we report' -- and the only one that touches no "
                        "other task's labels, which matters for a leaderboard submission.")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--dump-epoch-test", action="store_true",
                   help="Also evaluate TEST every epoch and record it in `epoch_curve` + a "
                        "[tracet] log line. Costs ~10%% wall (one extra n_te forward per epoch) "
                        "and buys the whole selection-rule family offline, forever, off one run. "
                        "Observation only -- val, `best` and the selected epoch are unchanged. "
                        "🚫 NOT a selection input: picking the epoch that maximises this curve is "
                        "an ORACLE, and must be reported as a ceiling, never as a result.")
    p.add_argument("--weight-soup", action="store_true",
                   help="Snapshot the tail's weights every epoch and additionally report, per "
                        "val-only rule, the test AUROC of the AVERAGED WEIGHTS (`soup_*`). Same "
                        "epoch sets as the prediction ensembles, averaged in parameter space "
                        "instead of rank space -- so it ships ONE model, at one forward pass, "
                        "which is the cheaper thing to defend in the paper. Costs one extra ridge "
                        "fit per rule and ~3 MB of host RAM per epoch. Sound here only because "
                        "block 12 is LayerNorm-only: a BatchNorm would need SWA's running-stat "
                        "re-estimation pass, which this does NOT do.")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="0: the ridge driver has no head to align, so REVE's first step is a no-op")
    p.add_argument("--fwd-batch", type=int, default=256)
    p.add_argument("--train-batch", type=int, default=128)
    # 🔴 CPU IS THE DEFAULT AND cuSOLVER IS WHY. At fp16 the tap is 40.1 GiB, which "fits" in a
    # 96 GB GH200 -- and then torch.linalg.eigh dies with CUSOLVER_STATUS_INTERNAL_ERROR on
    # cusolverDnCreate (job 2790794). cuSOLVER allocates its handle and workspace OUTSIDE PyTorch's
    # caching allocator, so a tap large enough to make the allocator claim most of HBM starves it
    # even though the eigh itself is tiny (n x n, n ~ 1.6k). Host residency costs an index_select
    # copy per minibatch and buys back the whole eigensolver. Keeping the tap off the GPU also
    # leaves --mem as the only thing to reason about, which is what the billing math wants.
    p.add_argument("--x11-device", choices=("cuda", "cpu"), default="cpu")
    p.add_argument("--x11-dtype", choices=("fp32", "fp16"), default="fp32",
                   help="dtype of the cached tap-11. fp32 is the safe default and what the pilot "
                        "used. fp16 HALVES it (measured 80 GiB -> 40 GiB on a 13,592-window board "
                        "session), which is the difference between fitting one GPU's memory share "
                        "(bills 1000) and needing double (bills 2000) — and it is what makes the "
                        "TWO resident caches of cs/csession fit at all. MEASURED at 1.910e-3 "
                        "relative on the reported features (job 2790650) -- 10 mantissa bits "
                        "~4.9e-4, amplified ~3.9x by block 12. That does NOT gate: parity-L0 runs "
                        "on fp32 certificate rows, so the split is certified independently of "
                        "storage. ws does not need this (fp32 MaxRSS 93.8 GiB already fits the "
                        "110.9 GiB share); cs/csession do.")
    p.add_argument("--tower", choices=("online", "teacher"), default="online")
    p.add_argument("--parity-only", action="store_true")
    p.add_argument("--seed", type=int, default=33)
    args = p.parse_args()

    FTP.SPLIT_AT = SPLIT_AT
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unit = args.unit or REGIME_UNIT[args.regime]
    # 🔴 THE ROW SPACE IS DEFINED BY ALL 15 TASKS, ALWAYS. ``_load_targets`` derives clip_starts
    # from the task list it is given, so asking it for one task yields a DIFFERENT, SHORTER row
    # space (measured: 3,134 windows for onset alone vs 13,592 for the full board set on S1T1).
    # The board cache's ws_split/cs_split index the full-set space, so a restricted list silently
    # misaligns every split index and the L0 comparison. --tasks restricts only what is REPORTED.
    # Caught by parity-L1b on the first GPU run rather than by a wrong number.
    tasks = list(BRD.BOARD_TASKS)
    report_tasks = [t for t in (args.tasks.split(",") if args.tasks else tasks) if t]
    unknown = [t for t in report_tasks if t not in tasks]
    if unknown:
        raise SystemExit(f"--tasks names non-board tasks: {unknown}")
    cell = REGIME_CELLS[args.regime][args.cell_index]
    train_cell = (cell if args.regime == "ws"
                  else BRD._sibling(cell) if args.regime == "csession"
                  else BRD.CS_TRAIN_ANCHOR)
    lams = list(BRD.LAM_MULTS)

    print(f"[board-ft] regime={args.regime} cell=S{cell[0]}T{cell[1]} "
          f"train=S{train_cell[0]}T{train_cell[1]} unit={unit} "
          f"rowspace_tasks={len(tasks)} reporting={len(report_tasks)} "
          f"driver_tasks={args.driver_tasks} "
          f"lams={len(lams)} (board grid) lr={args.lr} sched=CONSTANT wd={args.wd} "
          f"patience={args.patience} "
          f"device={device}", flush=True)
    print("[board-ft] A = frozen block12 + val-selected ridge (the published entry, recomputed "
          "in-path); C = fine-tuned block12 + THE SAME ridge. Headline = d(C-A). No D cell: the "
          "ridge driver never touches the head.", flush=True)

    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    parcel_fn = make_bt_parcel_fn(args.bt_root)
    clip_frames = int(round(ENC.CLIP_DUR_S * ENC.FPS))
    sd = ENC._load_ckpt(args.ckpt)
    tower = ENC._load_teacher(
        sd, device=device,
        pref="objective.online." if args.tower == "online" else "objective.teacher.model.")
    enc = tower.encoder
    pristine = FTP._pristine(enc)

    def build(session):
        """tap-11 cache + feature closure for ONE session. Encoded once, reused by every task."""
        spec = load_v3_sessions(
            sessions=[session], band_cache_dirs=args.band_cache_dirs, span_dir=args.span_dir,
            parcel_fn=parcel_fn, lof_report_path=None, winsor=(15.0, 15.0, 20.0),
            keep_labels_fn=None)[0]
        targets = ENC._load_targets(session, args.bt_root, tasks)
        bands = ENC._window_bands(spec, targets.clip_starts, clip_frames, rate_mult=1)
        geom = spec.setup.geom.to(device)
        parcel_id = spec.setup.parcel_id.to(device)
        grid = build_r4_grid(geom, n_time=clip_frames)
        parcel_packed = parcel_id[grid.contact]
        _c, parcel_canon, present = ENC._canon_parcels(grid, parcel_id)
        cols = [torch.as_tensor(np.where(parcel_canon == q)[0], device=device) for q in present]
        n_win = int(bands[0].shape[0])
        x11 = None
        x11_cert = None
        for s in range(0, n_win, args.fwd_batch):
            e = min(s + args.fwd_batch, n_win)
            bb = [b[s:e].to(device) for b in bands]
            # 🔴 bf16 AUTOCAST IS NOT OPTIONAL — IT IS PART OF THE PARITY CONTRACT. The board cache
            # was written under it (v3_probe_encode_r4.py:434), so tap 11 must be produced the same
            # way or the 11 blocks below the split diverge from the ones that made the cached
            # enc12. Running this loop in fp32 cost a full parity cycle: L0 printed rel=7.6e-3
            # (bf16's own error accumulated over the stack) and, diagnostically, fp32 and fp16 x11
            # storage gave the IDENTICAL rel to 4 digits -- storage precision cannot move an error
            # that was baked in before the tap was stored.
            with torch.no_grad(), torch.autocast(device_type=device.type,
                                                 dtype=torch.bfloat16,
                                                 enabled=(device.type == "cuda")):
                _o, taps = tower.forward(bb, grid, parcel_packed, tap_blocks=(SPLIT_AT,))
            t = taps[SPLIT_AT]
            if x11 is None:
                dev = torch.device(args.x11_device if torch.cuda.is_available() else "cpu")
                dt = t.dtype if args.x11_dtype == "fp32" else torch.float16
                need = n_win * t.shape[1] * t.shape[2] * torch.empty((), dtype=dt).element_size()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    if need > 0.55 * torch.cuda.mem_get_info()[0]:
                        dev = torch.device("cpu")
                x11 = torch.empty((n_win, t.shape[1], t.shape[2]), dtype=dt, device=dev)
            x11[s:e] = t.to(device=x11.device, dtype=x11.dtype)
            # THE CERTIFICATE ROWS STAY fp32 NO MATTER HOW THE BULK IS STORED. parity-L0 probes
            # only the first CERT_ROWS windows, so an exact copy of just those costs ~1.5 GiB
            # against ~17 GiB of headroom and keeps the STRUCTURAL question (is the block split
            # right?) answerable at full precision even when the bulk tap is fp16. Without this the
            # gate conflates two unrelated things: a wrong split (observed at 5.4e-3 and 7.6e-3)
            # and fp16 storage noise (1.9e-3) -- and tightening it against the latter costs a real
            # 2x in billing on cs/csession for no correctness gain.
            if s < CERT_ROWS:
                if x11_cert is None:
                    x11_cert = torch.empty((min(CERT_ROWS, n_win), t.shape[1], t.shape[2]),
                                           dtype=torch.float32, device=x11.device)
                m = min(e, CERT_ROWS)
                x11_cert[s:m] = t[:m - s].to(device=x11_cert.device, dtype=torch.float32)
        # The windowed bands and the session spec are consumed ONLY by the encode loop above, but
        # Python keeps them alive to the end of build(), so they sit next to the 40-80 GiB tap
        # cache at peak. Nothing below closes over them (feats reads x11, grid, cols). Dropping
        # them here is free and it is a large fraction of the OOM headroom.
        del bands, spec
        ctx_cache: dict = {}

        def ctx(b):
            if b not in ctx_cache:
                ctx_cache[b] = FTP._flat_ctx(enc, grid, b, device)
            return ctx_cache[b]

        def _feats_from(src, rows, grad, sel=None):
            out = []
            for s in range(0, len(rows), args.fwd_batch):
                idx = torch.as_tensor(rows[s:s + args.fwd_batch], device=src.device)
                xb = src.index_select(0, idx).to(device)
                with torch.set_grad_enabled(grad), torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
                    y = FTP._run_tail(enc, xb, ctx(xb.shape[0]))
                z = _unit_feats(y, grid, cols, unit, sel)
                out.append(z if grad else z.detach())
            return torch.cat(out, 0)

        def feats(rows, grad, sel=None):
            return _feats_from(x11, rows, grad, sel)

        def feats_cert(rows, grad, sel=None):
            """parity-L0 only. Row-aligned with x11[:CERT_ROWS] and always fp32."""
            return _feats_from(x11_cert, rows, grad, sel)

        feats.cert = feats_cert

        print(f"[board-ft] S{session[0]}T{session[1]} n_win={n_win} |P|={len(present)} "
              f"M={grid.total} k={grid.k_full} tap{SPLIT_AT} "
              f"{x11.numel() * x11.element_size() / 2**30:.1f} GiB {x11.dtype} on {x11.device}",
              flush=True)
        return targets, feats, n_win

    # Board cache records: the object the frozen number was computed from. Loaded with mmap --
    # they carry enc12_elec and run 50-65 GB, and eager torch.load is the documented OOM.
    sidecar = None
    if args.elec_labels_sidecar:
        import pickle
        with open(args.elec_labels_sidecar, "rb") as fh:
            sidecar = pickle.load(fh)
        print(f"[sidecar] elec_labels available for {len(sidecar)} sessions", flush=True)

    def rec_of(session):
        hits = glob.glob(os.path.join(args.board_cache_dir,
                                      f"enc_s{session[0]}_t{session[1]}_{args.board_tag}.pt"))
        if not hits:
            raise SystemExit(f"no board cache for S{session[0]}T{session[1]} in {args.board_cache_dir}")
        rec = torch.load(hits[0], map_location="cpu", mmap=True, weights_only=False)
        # Caches encoded before elec_labels was stored (board_r6_40k is one: the field is None)
        # carry them in a sidecar pickle instead. Same attach + same shape check as
        # v3_board_readout.py:615-622 — without it _elec_cols returns (None, None, 0) and every
        # csession cell would die claiming "no shared electrodes", which is a MISSING LABEL
        # problem wearing the costume of a data problem.
        if sidecar is not None and rec.get("elec_labels") is None:
            lab = sidecar.get(f"s{session[0]}_t{session[1]}")
            if lab is not None:
                n_e = rec["feats"]["enc12_elec"]["raw"].shape[1]
                if lab.shape[0] != n_e:
                    raise SystemExit(f"sidecar labels ({lab.shape[0]}) != enc12_elec electrodes "
                                     f"({n_e}) for s{session[0]}_t{session[1]}")
                rec["elec_labels"] = lab
        return rec

    test_rec = rec_of(cell)
    train_rec = test_rec if train_cell == cell else rec_of(train_cell)

    # ── column alignment, mirroring the frozen readout exactly ──────────────────────────────
    if args.regime == "ws":
        sel_tr = sel_ev = None
    elif unit == "elec":
        if train_rec.get("elec_labels") is None or test_rec.get("elec_labels") is None:
            raise SystemExit(
                f"S{cell[0]}T{cell[1]}: a record has no elec_labels, so electrodes cannot be "
                f"identity-intersected. Pass --elec-labels-sidecar (board_r6_40k stores None).")
        a_idx, t_idx, n_sh = BRD._elec_cols(train_rec, test_rec)
        if not n_sh:
            raise SystemExit(f"no shared electrodes for S{cell[0]}T{cell[1]}")
        sel_tr, sel_ev = a_idx, t_idx
        print(f"[align] shared electrodes = {n_sh} (by LABEL, not position)", flush=True)
    else:
        a_idx, t_idx, common = BRD._parcel_cols(train_rec, test_rec)
        if common.size == 0:
            raise SystemExit(f"no shared parcels for S{cell[0]}T{cell[1]}")
        sel_tr, sel_ev = a_idx, t_idx
        print(f"[align] shared parcels = {int(common.size)} (by atlas id)", flush=True)

    tgt_ev, feats_ev_raw, _n = build(cell)
    if train_cell == cell:
        tgt_tr, feats_tr_raw = tgt_ev, feats_ev_raw
    else:
        tgt_tr, feats_tr_raw, _n2 = build(train_cell)

    # ── L1b: the splits we fine-tune against MUST be the ones the frozen number used ────────
    # _load_targets rebuilds them; the board cache stored them at encode time. If these ever
    # diverge, every d(C-A) below is measured against a different partition than the bar.
    for name, tg, rc in (("test", tgt_ev, test_rec), ("train", tgt_tr, train_rec)):
        for t in tasks:
            a = np.asarray(tg.labels[t], dtype=np.float64)
            b = np.asarray(rc["labels"][t], dtype=np.float64)
            if a.shape != b.shape or not np.array_equal(np.isfinite(a), np.isfinite(b)) \
                    or not np.allclose(a[np.isfinite(a)], b[np.isfinite(b)]):
                raise SystemExit(f"PARITY-L1b FAIL labels {name} {t}")
    print(f"[parity-L1b] labels match the board cache for {len(tasks)} tasks x 2 records OK",
          flush=True)

    # ── L0: block12(cached tap-11) == the on-disk enc12 tap, AT THE REPORTED UNIT ───────────
    # The elec tap is keyed "enc12_elec" (== BRD.ELEC_TAPS[1]), NOT "elec_enc12", and every tap is
    # a dict whose "raw" holds the tensor (v3_board_readout.py:618). Both were verified against a
    # real record's key list rather than inferred from the encode script's _write() call.
    key = f"enc{TAP}" if unit == "parcel" else f"enc{TAP}_elec"
    if key not in test_rec["feats"]:
        raise SystemExit(f"L0: board cache has no '{key}' tap (has {sorted(test_rec['feats'])}) "
                         f"-- cannot certify the {unit} unit")
    # 🔴 TWO DIFFERENT QUESTIONS, TWO DIFFERENT BARS. THE GATE IS ON THE FIRST ONE ONLY.
    #   (a) STRUCTURAL -- is the split right? Measured on the fp32 certificate rows, so storage
    #       dtype cannot enter. Every real bug this has ever caught landed far above 1e-3: an
    #       earlier bf16 cache draft (pilot:40), rel=5.371e-3 on job 2779110_0 (pilot:551), and a
    #       missing bf16 autocast on the tap-11 encode at 7.641e-3. In fp32 the correct split is
    #       BIT-EXACT (rel=0.000e+00, job 2790649), so this bar has enormous margin and should
    #       stay tight. It is the only thing standing between a block-split bug and a number.
    #   (b) STORAGE NOISE -- how much does --x11-dtype perturb the features? fp16 measures
    #       1.910e-3 (job 2790650): 10 mantissa bits ~4.9e-4, amplified ~3.9x by block 12. That is
    #       NOT a bug, and gating it at 1e-3 was a category error on my part -- it would have cost
    #       2x billing on every cs/csession task (two resident taps -> 227120M -> bills 2000) to
    #       buy nothing. Reported every run, never fatal. What it perturbs is C, whose features
    #       move far more than this under fine-tuning; A is read from the on-disk cache, which is
    #       itself stored fp16 (v3_probe_encode_r4.py:442) and rounded again by FTP._flat16.
    ref = test_rec["feats"][key]["raw"]
    probe_rows = np.arange(min(CERT_ROWS, ref.shape[0]), dtype=np.int64)
    r = torch.as_tensor(np.asarray(ref[probe_rows.tolist()])).float().reshape(len(probe_rows), -1)
    rmax = float(r.abs().max()) + 1e-12
    with torch.no_grad():
        zc = FTP._flat16(feats_ev_raw.cert(probe_rows, False)).cpu()
    d0 = float((r - zc.float()).abs().max())
    rel = d0 / rmax
    print(f"[parity-L0] unit={unit} cached {key} vs block12(tap{SPLIT_AT}) fp32-cert "
          f"max|d|={d0:.3e} rel={rel:.3e} {'OK' if rel < 1e-3 else 'FAIL'}", flush=True)
    if rel >= 1e-3:
        raise SystemExit("PARITY-L0 FAIL — the block split does not reproduce the reported tap.")
    if args.x11_dtype != "fp32":
        with torch.no_grad():
            zs = FTP._flat16(feats_ev_raw(probe_rows, False)).cpu()
        ds = float((r - zs.float()).abs().max())
        print(f"[parity-L0s] storage={args.x11_dtype} max|d|={ds:.3e} rel={ds / rmax:.3e} "
              f"(INFORMATIONAL -- split already certified exact above)", flush=True)
    if args.parity_only:
        return

    # ── labels: (n, K) for the multitask driver; column `col` is the reported task ──────────
    y_tr_all = np.stack([np.asarray(tgt_tr.labels[t], dtype=np.float64) for t in tasks], 1)
    y_ev_all = np.stack([np.asarray(tgt_ev.labels[t], dtype=np.float64) for t in tasks], 1)
    ybin_tr = torch.as_tensor((y_tr_all > 0).astype(np.float32), device=device)
    msk_tr = torch.as_tensor(np.isfinite(y_tr_all), device=device)
    ybin_ev = (ybin_tr if train_cell == cell
               else torch.as_tensor((y_ev_all > 0).astype(np.float32), device=device))

    results = []
    for task in report_tasks:
        # Column index is into the FULL board task list, which is what the label matrices and the
        # multitask driver are built over — not into the reporting subset.
        ti = tasks.index(task)
        y_a, y_t = y_tr_all[:, ti], y_ev_all[:, ti]
        # Resolve the DRIVER columns for this cell. Built per-task because 'report' depends on
        # which task is being reported. Printed per cell so the log, not a memo, is the record of
        # what actually drove the weights.
        if args.driver_tasks == "report":
            dcols = [ti]
        elif args.driver_tasks == "all":
            dcols = list(range(len(tasks)))
        else:
            want = [t.strip() for t in args.driver_tasks.split(",") if t.strip()]
            bad = [t for t in want if t not in tasks]
            if bad:
                raise SystemExit(f"--driver-tasks names unknown board tasks {bad}")
            dcols = sorted({tasks.index(t) for t in want} | {ti})
        if args.regime == "ws":
            folds = sorted(tgt_ev.ws_split[task].items())
            plan = [(f, {"tr": _finite(y_a, sp["train"]), "va": _finite(y_t, sp["val"]),
                         "te": _finite(y_t, sp["test"])}) for f, sp in folds]
        else:
            sp = test_rec["cs_split"][task]
            plan = [(0, {"tr": _finite(y_a, np.arange(len(y_a))),
                         "va": _finite(y_t, sp["val"]), "te": _finite(y_t, sp["test"])})]
        for fold, rows in plan:
            if len(rows["tr"]) < 2 or len(rows["te"]) < 2:
                print(f"[skip] {task} f{fold}: n_tr={len(rows['tr'])} n_te={len(rows['te'])}",
                      flush=True)
                continue
            FTP._restore(enc, pristine)
            res = _run_cell(
                args, enc,
                lambda r_, g_, s_=sel_tr: feats_tr_raw(r_, g_, s_),
                lambda r_, g_, s_=sel_ev: feats_ev_raw(r_, g_, s_),
                rows, ybin_tr, ybin_ev, msk_tr, ti, dcols, lams,
                f"S{cell[0]}T{cell[1]} {task:17s} f{fold}")
            results.append(dict(res, regime=args.regime, cell=f"S{cell[0]}T{cell[1]}",
                                task=task, fold=int(fold), unit=unit,
                                train_cell=f"S{train_cell[0]}T{train_cell[1]}"))
            json.dump(results, open(args.out, "w"), indent=1)

    d = np.array([r["d"] for r in results], dtype=float)
    pos = int((d > 0).sum())
    print(f"\n=== {args.regime} S{cell[0]}T{cell[1]} unit={unit} — {len(d)} cells === "
          f"mean d={d.mean():+.4f} positive={pos}/{len(d)} p={FTP._sign_p(pos, len(d)):.4f}",
          flush=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[board-ft] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
