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


def _run_cell(args, enc, feats_tr, feats_ev, rows, ybin_tr, ybin_ev, msk_tr, col, lams, tag):
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    rng = np.random.default_rng(args.seed)
    yt = ybin_tr[tr][:, col]
    yv = ybin_ev[va][:, col]
    ye = ybin_ev[te][:, col]
    stub = _stub_rows(yv.cpu().numpy())

    def ridge_now(want_test):
        """val (+ optionally test) AUROC at val-selected λ on the CURRENT weights."""
        with torch.no_grad():
            z_tr = FTP._flat16(feats_tr(tr, False))
            z_va = FTP._flat16(feats_ev(va, False))
            if want_test:
                z_te, y_te = FTP._flat16(feats_ev(te, False)), ye
            else:
                z_te, y_te = z_va[stub], yv[stub]
            return FTP._ridge_eval(z_tr, z_va, z_te, yt, yv, y_te, lams)

    # ── A: the frozen board entry, recomputed in-path (ep 0) ────────────────────────────────
    v0, _t0const, a_test, a_df = ridge_now(True)
    best = {"val": float(v0), "epoch": 0, "state": None}
    trace = [(0, float(v0))]
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
            # Multitask in its ONLY safe form: all 15 label COLUMNS on the primary task's own
            # train rows (pilot :677). Extra rows would be leakage — the 15 board tasks do NOT
            # share a partition (audit/board_split_signature.py).
            loss = FTP._loo_ridge_risk(z, ybin_tr[tr[pos]], msk_tr[tr[pos]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            opt.step()
        sched.step()
        enc.eval()
        # THE 10% CUT: val only. Test is computed once, at the end, on the restored best state.
        cv = ridge_now(False)[0]
        trace.append((ep, float(cv)))
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
    sec = (time.time() - t0) / max(n_ep, 1)
    print(f"[cell] {tag} ep*={best['epoch']:3d} A={a_test:.4f} C={c_test:.4f} "
          f"d={c_test - a_test:+.4f} | df={a_df:.1f}/{len(tr)} | {n_ep}ep {sec:.2f}s/ep "
          f"n_tr={len(tr)} n_va={len(va)} n_te={len(te)}", flush=True)
    print(f"[trace] {tag} " + " ".join(f"{e}:{v:.4f}" for e, v in trace), flush=True)
    return {"test_frozen_vallam": float(a_test), "test_c": float(c_test),
            "d": float(c_test - a_test), "best_epoch": int(best["epoch"]),
            "n_epochs_run": int(n_ep), "sec_per_epoch": float(sec), "ridge_df": float(a_df),
            "n_tr": int(len(tr)), "n_va": int(len(va)), "n_te": int(len(te))}


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
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="0: the ridge driver has no head to align, so REVE's first step is a no-op")
    p.add_argument("--fwd-batch", type=int, default=256)
    p.add_argument("--train-batch", type=int, default=128)
    p.add_argument("--x11-device", choices=("cuda", "cpu"), default="cuda")
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
    tasks = [t for t in (args.tasks.split(",") if args.tasks else BRD.BOARD_TASKS) if t]
    cell = REGIME_CELLS[args.regime][args.cell_index]
    train_cell = (cell if args.regime == "ws"
                  else BRD._sibling(cell) if args.regime == "csession"
                  else BRD.CS_TRAIN_ANCHOR)
    lams = list(BRD.LAM_MULTS)

    print(f"[board-ft] regime={args.regime} cell=S{cell[0]}T{cell[1]} "
          f"train=S{train_cell[0]}T{train_cell[1]} unit={unit} tasks={len(tasks)} "
          f"lams={len(lams)} (board grid) lr={args.lr} wd={args.wd} patience={args.patience} "
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
        for s in range(0, n_win, args.fwd_batch):
            e = min(s + args.fwd_batch, n_win)
            bb = [b[s:e].to(device) for b in bands]
            with torch.no_grad():
                _o, taps = tower.forward(bb, grid, parcel_packed, tap_blocks=(SPLIT_AT,))
            t = taps[SPLIT_AT]
            if x11 is None:
                dev = torch.device(args.x11_device if torch.cuda.is_available() else "cpu")
                need = n_win * t.shape[1] * t.shape[2] * t.element_size()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    if need > 0.55 * torch.cuda.mem_get_info()[0]:
                        dev = torch.device("cpu")
                x11 = torch.empty((n_win, t.shape[1], t.shape[2]), dtype=t.dtype, device=dev)
            x11[s:e] = t.to(x11.device)
        ctx_cache: dict = {}

        def ctx(b):
            if b not in ctx_cache:
                ctx_cache[b] = FTP._flat_ctx(enc, grid, b, device)
            return ctx_cache[b]

        def feats(rows, grad, sel=None):
            out = []
            for s in range(0, len(rows), args.fwd_batch):
                idx = torch.as_tensor(rows[s:s + args.fwd_batch], device=x11.device)
                xb = x11.index_select(0, idx).to(device)
                with torch.set_grad_enabled(grad), torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
                    y = FTP._run_tail(enc, xb, ctx(xb.shape[0]))
                z = _unit_feats(y, grid, cols, unit, sel)
                out.append(z if grad else z.detach())
            return torch.cat(out, 0)

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
    ref = test_rec["feats"][key]["raw"]
    probe_rows = np.arange(min(256, ref.shape[0]), dtype=np.int64)
    with torch.no_grad():
        z16 = FTP._flat16(feats_ev_raw(probe_rows, False)).cpu()
    r = torch.as_tensor(np.asarray(ref[probe_rows.tolist()])).float().reshape(len(probe_rows), -1)
    d0 = float((r - z16.float()).abs().max())
    rel = d0 / (float(r.abs().max()) + 1e-12)
    print(f"[parity-L0] unit={unit} cached {key} vs block12(tap{SPLIT_AT}) max|d|={d0:.3e} "
          f"rel={rel:.3e} {'OK' if rel < 1e-3 else 'FAIL'}", flush=True)
    if rel >= 1e-3:
        raise SystemExit("PARITY-L0 FAIL — the block split does not reproduce the reported tap.")
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
    for ti, task in enumerate(tasks):
        y_a, y_t = y_tr_all[:, ti], y_ev_all[:, ti]
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
                rows, ybin_tr, ybin_ev, msk_tr, ti, lams,
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
