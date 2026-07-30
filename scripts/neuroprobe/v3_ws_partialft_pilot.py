"""WS partial fine-tuning pilot — does unfreezing the LAST ENCODER BLOCK beat the frozen ridge?

THE QUESTION (Ben 2026-07-30): our readout is a kernel ridge on frozen enc12 features. MAE
(He et al. 2021 §4.3) reports that a frozen linear probe and partial fine-tuning are *largely
uncorrelated*, and that unfreezing ONE transformer block moves ImageNet 73.5 -> 81.0. Every
number we have ever quoted is the frozen-probe column. This asks whether our frozen column
leaves the same kind of gap on the table.

ARM: pbs50 @ 20k (`v3_ckpt_v3_r6_pbspace111_sf50_lr6e-3_40k/ladder-step=20000.ckpt`, job
2764203) — the adoption candidate, so the FT delta lands on top of the number we are actually
choosing, and pairs directly against the pbs50/pbs25 double dissociation read at the same step.
Its frozen probe already exists (`results_v3_probe_pbs50_20k.json`) AND so does its encode cache
(`v3_probe_cache_pbs50_20k`), which is what makes the parity gate below cheap and strict.

WHAT IS HELD FIXED — the delta is block-12 weights and NOTHING else
-------------------------------------------------------------------
Same checkpoint, same 7 sessions, same 4 tasks, same per-task ``ws_split`` folds, same parcel
pooling, same fp16 feature rounding, same ``std`` conditioning, same CONSTANT-lambda dual ridge,
same rows. The frozen arm is not re-derived for the headline: it is read off the probe JSON, and
the parity gate proves this script reproduces it before any weight moves.

PARITY GATE (feedback-build-the-invariant-into-the-probe) — runs FIRST, refuses to train on fail
-------------------------------------------------------------------------------------------------
L0  FEATURE level. The cached ``enc12`` tensor from ``v3_probe_cache_pbs50_20k`` vs the enc12
    this script computes as block12(cached tap-11). Proves the hand-split of the block stack is
    bit-equivalent to the tower's own forward. This is the check that matters — everything else
    here rests on that split being exact.
L1  METRIC level. Zero-step ridge vs the on-disk ``ws_per_session`` enc12 values.
L2  GPU ridge (TF32 OFF) vs the readout's own CPU numpy ridge, so the fast per-epoch selection
    loop is certified against the slow reference before it is trusted.

WHY BLOCK 12 IS RUN ALONE
--------------------------
A full-tower forward every epoch costs ~12x what is needed. ``V3Tower._run_flat``
(towers.py:281-317) adds the parcel embed ONCE before block 0 and then loops blocks, capturing
``taps[i+1] = xf`` as the RAW block output. So tap 11 already carries the parcel embed and every
positional effect, and block 12 applied to it reproduces tap 12 exactly. We cache tap 11 once
per session (bf16, the autocast residual dtype, so the numerics match) and re-run only block 12.
L0 is what certifies this.

WHAT IS REPORTED: A AND D ONLY (Ben, 2026-07-30)
-------------------------------------------------
An earlier draft treated the trained head as a gradient scaffold and reported a refit ridge on the
fine-tuned features. That was wrong: MAE's headline numbers ARE the fine-tuned classifier's own
accuracy (mae.tex:424, "All self-supervised methods are evaluated by end-to-end fine-tuning") —
they never refit a probe. So the head is the reported readout.

  A = frozen enc12 + constant-lambda ridge   <- the number every result of ours has quoted
  D = fine-tuned block 12 + the trained head <- the form MAE reports

The two intermediate cells (frozen+head, fine-tuned+ridge) are NOT run. Ben: "I think we should
only do A and D only." STATED PLAINLY: A vs D therefore moves weights AND readout family
together, so a positive result says "fine-tuning the last block with a logistic head beats our
frozen ridge", not "the features got better". That is the deployable claim, and it is the one the
decision map below is written against.

Consequences for cost: the ridge is fit ONCE per cell, at epoch 0, where it IS A and the L1
parity check. Nothing per-epoch touches a Gram, and the per-epoch eval scores only val+test —
never the ~3.6k train rows — so the inner loop is one train fwd+bwd plus one short eval forward.

Each epoch is selected on the head's val AUROC, the same quantity D reports.

THE ASYMMETRY, STATED: the FT arm gets one val-selected hyperparameter (the epoch) the frozen arm
does not. So a THIRD column is reported -- frozen features, lambda selected on the SAME val -- to
price val selection on its own. Fine-tuning must beat THAT, not just the constant-lambda number.

EPOCH BUDGET: 80 with patience 15, not 40/8. MAE's partial-FT appendix (mae.tex:694) reports that
"tuning fewer blocks requires a longer schedule" and sweeps {50, 100, 200} epochs for that reason.
Truncating a late head optimum would look exactly like a negative result.

PRE-REGISTERED NULL AND DECISION MAP (feedback-compute-the-null / name-the-decision)
------------------------------------------------------------------------------------
n = 28 cells (7 sessions x 4 tasks), paired, two-sided sign test on the per-cell delta. Under the
null the sign is a fair coin: 20/28 -> p=.036, 21/28 -> p=.013.
Scale for "worth it": pbs50@20k's own frozen enc0->enc12 WS gain is +.0518 (that is ALL of
pretraining, on this arm). The board WS margin over the leaderboard is +.0120.
  d >= +.010 AND >= 20/28 positive -> BUILD THE CS VERSION (15 fine-tunes) and take it to the
                                     board. Headline path.
  0 < d < +.010, or n.s.           -> negative result; ridge stays; close the thread.
  d <= 0                           -> close the thread; the readout is not the bottleneck.
  parity gate fails                -> fix the forward; read nothing.
A/B differ in all four branches.

Two stages so a bad lr cannot cost the grid:
  A  --stage a: one session, lr swept, val-selected. Shard by lr.
  B  --stage b --lr <winner>: the full 7x4x2 grid. Shard by session (--session-index).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _load_sibling(name: str):
    """Import a sibling script by path (they are scripts, not a package)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ENC = _load_sibling("v3_probe_encode_r4")
RDO = _load_sibling("v3_probe_readout_r4")

PROBE_TASKS = RDO.PROBE_TASKS
PROBE_COHORT_7 = RDO.PROBE_COHORT_7
TAP = 12          # reported tap = RAW output of block 12 (towers.py:313)
SPLIT_AT = 11     # cache this tap; re-run only the blocks after it


# ── ridge, on GPU, numerically matched to the readout's numpy primitives ─────────────────────
def _gpu_ridge(z_tr, y_tr, z_te, lam_mult=None, lams=None):
    """G=ZZ^T in fp32 (TF32 OFF), solve in fp64 — the same two-dtype split as
    ``v3_probe_readout_r4.dual_ridge_scores``. With ``lams``, ONE eigendecomposition of G serves
    every lambda (the board readout's trick, v3_board_readout.py:298-302)."""
    if lam_mult is None:
        lam_mult = RDO.CONST_LAM_MULT
    g = (z_tr @ z_tr.T).double()
    k = (z_te @ z_tr.T).double()
    n = g.shape[0]
    base = float(torch.diagonal(g).sum() / max(n, 1))
    y = y_tr.double()
    if lams is None:
        eye = torch.eye(n, dtype=torch.float64, device=g.device)
        return (k @ torch.linalg.solve(g + (lam_mult * base) * eye, y)).cpu().numpy()
    w, v = torch.linalg.eigh(g)
    vty = v.T @ y
    return {lm: (k @ (v @ (vty / (w + lm * base)))).cpu().numpy() for lm in lams}


def _std_gpu(z_tr, z_te):
    """Per-feature z-score on TRAIN stats only — ``RDO._standardize`` on GPU."""
    mu = z_tr.mean(0)
    sd = z_tr.std(0, unbiased=False)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    return (z_tr - mu) / sd, (z_te - mu) / sd


def _ridge_eval(z_tr, z_va, z_te, y_tr, y_va, y_te, lams):
    """val + test AUROC from ONE Gram and ONE eigendecomposition.

    Both readouts standardize on the SAME train statistics, so the standardized train matrix —
    and therefore G = ZZ^T, the 5.7 TFLOP term — is identical for the val and test scorings.
    Computing it twice per epoch was pure waste. eigh additionally makes the lambda sweep free,
    which is what pays for the val-lambda control column.

    Returns (val@const-lambda, test@const-lambda, test@val-selected-lambda)."""
    mu = z_tr.mean(0)
    sd = z_tr.std(0, unbiased=False)
    sd = torch.where(sd == 0, torch.ones_like(sd), sd)
    a = (z_tr - mu) / sd
    g = (a @ a.T).double()
    k_va = (((z_va - mu) / sd) @ a.T).double()
    k_te = (((z_te - mu) / sd) @ a.T).double()
    n = g.shape[0]
    basel = float(torch.diagonal(g).sum() / max(n, 1))
    y = y_tr.double()
    w, v = torch.linalg.eigh(g)
    vty = v.T @ y

    def sc(k, lm):
        return (k @ (v @ (vty / (w + lm * basel)))).cpu().numpy()

    c = RDO.CONST_LAM_MULT
    val_const = RDO.auroc(sc(k_va, c), y_va)
    test_const = RDO.auroc(sc(k_te, c), y_te)
    best_lm = max(lams, key=lambda L: RDO.auroc(sc(k_va, L), y_va))
    return val_const, test_const, RDO.auroc(sc(k_te, best_lm), y_te)


def _pool_t(x, cols):
    """Differentiable twin of ``ENC._pool_parcels``: (B,n,k,d) -> (B,|P|,k*d), NO fp16 cast.
    Same math, same parcel order; the fp16 round is applied later, only on reported features."""
    B = x.shape[0]
    return torch.stack([x[:, c].mean(1).reshape(B, -1) for c in cols], dim=1)


def _flat16(z):
    """Match the cache's numeric path exactly: fp16 store -> fp32 read (``RDO._feat``)."""
    return z.reshape(z.shape[0], -1).to(torch.float16).to(torch.float32)


def _sign_p(k, n):
    if n == 0:
        return float("nan")
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2.0 ** n)


# ── the flat-forward context, reproduced from towers.py:281-301 ──────────────────────────────
def _flat_ctx(enc, grid, B, device):
    """(depth_b, time_b, cu_b, cu_drop_b, max_seqlen, rope_cs) for a batch of B clips.

    Verbatim from ``V3Tower._run_flat``; it is a pure function of (grid, B), so it is built once
    per distinct batch size and reused. The parcel embed is NOT here — it is added before block 0
    and is therefore already inside the cached tap."""
    M = grid.total
    cu_static = grid.cu_seqlens
    offsets = torch.arange(B, device=device, dtype=torch.int64) * M
    cu_b = torch.cat([
        (cu_static[:-1].to(torch.int64)[None, :] + offsets[:, None]).reshape(-1),
        torch.tensor([B * M], dtype=torch.int64, device=device),
    ])
    keep = torch.ones_like(cu_b, dtype=torch.bool)
    keep[1:] = cu_b[1:] != cu_b[:-1]
    cu_drop_b = cu_b[keep]
    cu_b = cu_b.to(cu_static.dtype)
    depth_b = grid.depth[None, :].expand(B, M).reshape(B * M)
    time_b = grid.time_pos[None, :].expand(B, M).reshape(B * M)
    cos, sin = enc.blocks[0].rope.cos_sin(depth_b, time_b)
    return depth_b, time_b, cu_b, cu_drop_b, grid.max_seqlen, (cos, sin)


def _run_tail(enc, x11, ctx):
    """Blocks SPLIT_AT+1 .. end applied to a cached tap. Mirrors the loop body at
    towers.py:306-311 (``blk.forward_flat``), which is the ONLY thing between tap 11 and tap 12."""
    B, M, d = x11.shape
    xf = x11.reshape(B * M, d)
    depth_b, time_b, cu_b, cu_drop_b, max_seqlen, rope_cs = ctx
    for blk in enc.blocks[SPLIT_AT:TAP]:
        xf = blk.forward_flat(xf, depth_b, time_b, cu_b, cu_drop_b, max_seqlen, rope_cs=rope_cs)
    return xf.reshape(B, M, d)


# ── trainable parameter arms ─────────────────────────────────────────────────────────────────
def _arm_params(enc, arm: str):
    """Parameters to unfreeze. The ladder is by PARAMETER COUNT, which is the axis that binds
    here: n_train is ~3.6k windows, so the full block (~790k params) is heavily overparameterized
    and norm-only (BitFit) is the honest floor."""
    blk = enc.blocks[TAP - 1]
    if arm == "norm":
        return [p for _, p in blk.named_parameters() if p.ndim == 1]
    if arm == "mlp":
        return list(blk.mlp.parameters())
    if arm == "block12":
        return list(blk.parameters())
    raise SystemExit(f"unknown arm {arm!r}")


class _Head(torch.nn.Module):
    """Linear on the SAME flattened (|P|*F) feature the ridge sees. Overparameterized vs n_train
    on purpose — with weight decay it is an L2-regularized logistic regression on the readout's
    own feature space, so the gradient block 12 receives is about the features the ridge will
    actually use. (A parcel-pooled head would be better conditioned but would optimize a
    different representation than the one reported.)"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fc = torch.nn.Linear(dim, 1)
        torch.nn.init.zeros_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

    def forward(self, z):
        return self.fc(self.norm(z)).squeeze(-1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--baseline-json", required=True)
    p.add_argument("--baseline-prefix", default="pbs50_20k")
    p.add_argument("--parity-cache-dir", default=None,
                   help="encode cache of the SAME ckpt; enables the L0 feature-level gate")
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stage", choices=("a", "b"), default="b")
    p.add_argument("--session-index", type=int, default=None, help="stage b: shard by session")
    p.add_argument("--arms", default="block12")
    p.add_argument("--lrs", default="1e-4", help="stage a sweeps these; stage b takes one")
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--fwd-batch", type=int, default=64)
    p.add_argument("--train-batch", type=int, default=16)
    p.add_argument("--val-margin", type=int, default=2,
                   help="windows dropped off the VAL side of the val|test seam (M14 hop "
                        "overlap). test is never touched — it must stay the frozen arm's rows.")
    p.add_argument("--x11-device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--parity-only", action="store_true")
    # r6 is MAE — there is NO EMA teacher, so the deployed r6 encode sbatch passes --online and
    # the tower comes from ``objective.online.*``. Loading the r4 default would fail the strict
    # state_dict check, not silently mis-load, but the default belongs on the arm we probe.
    p.add_argument("--tower", choices=("online", "teacher"), default="online")
    p.add_argument("--seed", type=int, default=33)
    args = p.parse_args()
    pref = "objective.online." if args.tower == "online" else "objective.teacher.model."

    torch.backends.cuda.matmul.allow_tf32 = False   # fp32 must MEAN fp32 for the parity gate
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = json.load(open(args.baseline_json))
    arms = [a for a in args.arms.split(",") if a]
    lrs = [float(x) for x in args.lrs.split(",") if x]
    if args.stage == "b" and len(lrs) != 1:
        raise SystemExit("stage b takes exactly one --lrs (the stage-a winner)")

    cohort = list(PROBE_COHORT_7)
    if args.session_index is not None:
        cohort = [cohort[args.session_index]]

    print(f"[ft-pilot] ckpt={args.ckpt}\n[ft-pilot] stage={args.stage} tap=enc{TAP} "
          f"split_at={SPLIT_AT} arms={arms} lrs={lrs} wd={args.wd} epochs<={args.epochs} "
          f"patience={args.patience} sessions={[f'S{a}T{b}' for a, b in cohort]} device={device}",
          flush=True)
    print("[ft-pilot] DECISION MAP: mean d >= +.010 AND >= 20/28 positive -> build the CS "
          "version; 0 < d < +.010 or n.s. -> negative, ridge stays; d <= 0 -> close the thread. "
          "NULL: sign of d is a fair coin over 28 paired cells (20/28 p=.036).", flush=True)

    # These live in the package, not in the encode SCRIPT — v3_probe_encode_r4 imports them
    # inside its own main() (:526-532), so they are not module attributes and ENC.<name> raises.
    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    parcel_fn = make_bt_parcel_fn(args.bt_root)
    clip_frames = int(round(ENC.CLIP_DUR_S * ENC.FPS))
    sd = ENC._load_ckpt(args.ckpt)
    results: list[dict] = []

    for session in cohort:
        subject_id, trial_id = session
        skey = f"S{subject_id}T{trial_id}"
        spec = load_v3_sessions(
            sessions=[session], band_cache_dirs=args.band_cache_dirs, span_dir=args.span_dir,
            parcel_fn=parcel_fn, lof_report_path=None, winsor=(15.0, 15.0, 20.0),
            keep_labels_fn=None,
        )[0]
        targets = ENC._load_targets(session, args.bt_root, PROBE_TASKS)
        bands = ENC._window_bands(spec, targets.clip_starts, clip_frames, rate_mult=1)
        geom = spec.setup.geom.to(device)
        parcel_id = spec.setup.parcel_id.to(device)
        grid = build_r4_grid(geom, n_time=clip_frames)
        parcel_packed = parcel_id[grid.contact]
        _canon, parcel_canon, present = ENC._canon_parcels(grid, parcel_id)
        n_win = int(bands[0].shape[0])
        cols = [torch.as_tensor(np.where(parcel_canon == q)[0], device=device) for q in present]

        tower = ENC._load_teacher(sd, device=device, pref=pref)
        enc = tower.encoder
        pristine = {k: v.detach().clone() for k, v in enc.blocks[TAP - 1].state_dict().items()}
        ctx_cache: dict[int, tuple] = {}

        def ctx(B):
            if B not in ctx_cache:
                ctx_cache[B] = _flat_ctx(enc, grid, B, device)
            return ctx_cache[B]

        # ── cache tap 11 once (bf16 = the autocast residual dtype, so numerics match) ────────
        x11_dev = torch.device(args.x11_device if torch.cuda.is_available() else "cpu")
        x11 = None      # allocated on the first batch, once the tap's shape is known
        for s in range(0, n_win, args.fwd_batch):
            e = min(s + args.fwd_batch, n_win)
            bb = [b[s:e].to(device) for b in bands]
            with torch.no_grad(), torch.autocast(device_type=device.type,
                                                 dtype=torch.bfloat16,
                                                 enabled=(device.type == "cuda")):
                _out, taps = tower.forward(bb, grid, parcel_packed, tap_blocks=(SPLIT_AT,))
            t = taps[SPLIT_AT].to(torch.bfloat16)
            if x11 is None:
                x11 = torch.empty((n_win, t.shape[1], t.shape[2]), dtype=torch.bfloat16,
                                  device=x11_dev)
            x11[s:e] = t.to(x11_dev)
        gib = x11.numel() * 2 / 1024 ** 3
        print(f"\n[ft-pilot] === {skey} n_win={n_win} |P|={len(present)} M={grid.total} "
              f"k_full={grid.k_full} d={x11.shape[-1]} tap{SPLIT_AT} cache {gib:.1f} GiB on "
              f"{x11_dev} ===", flush=True)

        def feats(rows, grad):
            """block12(cached tap 11) -> parcel-pooled (r, |P|, F)."""
            out = []
            for s in range(0, len(rows), args.fwd_batch):
                idx = torch.as_tensor(rows[s:s + args.fwd_batch], device=x11.device)
                xb = x11.index_select(0, idx).to(device)
                with torch.set_grad_enabled(grad), torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda")):
                    y = _run_tail(enc, xb, ctx(xb.shape[0]))
                z = _pool_t(y.float().reshape(len(idx), -1, grid.k_full, y.shape[-1]), cols)
                out.append(z if grad else z.detach())
            return torch.cat(out, 0)

        # ── L0: feature-level parity against the encode cache ───────────────────────────────
        allrows = np.arange(n_win, dtype=np.int64)
        with torch.no_grad():
            z_all = feats(allrows, False)
        z16 = z_all.to(torch.float16).cpu()
        if args.parity_cache_dir:
            hits = glob.glob(os.path.join(args.parity_cache_dir,
                                          f"enc_s{subject_id}_t{trial_id}_*.pt"))
            if not hits:
                raise SystemExit(f"L0: no cache for {skey} in {args.parity_cache_dir}")
            rec = torch.load(hits[0], map_location="cpu", weights_only=False)
            ref = rec["feats"][f"enc{TAP}"]["raw"]
            d0 = float((ref.float() - z16.float()).abs().max())
            rel = d0 / (float(ref.float().abs().max()) + 1e-12)
            print(f"[parity-L0] {skey} cached enc{TAP} vs block12(tap{SPLIT_AT}) "
                  f"max|d|={d0:.3e} rel={rel:.3e} {'OK' if rel < 1e-3 else 'FAIL'}", flush=True)
            if rel >= 1e-3:
                raise SystemExit("PARITY-L0 FAIL — the block split is not equivalent to the "
                                 "tower forward; every FT number below would be meaningless.")
            del rec, ref

        # ── L1: metric-level parity against the on-disk probe JSON ──────────────────────────
        z_np = z16.float().numpy().reshape(n_win, -1)
        for task in PROBE_TASKS:
            y = np.asarray(targets.labels[task], dtype=np.float64)
            folds = []
            for sp in targets.ws_split[task].values():
                tr, te = RDO._finite(y, sp["train"]), RDO._finite(y, sp["test"])
                folds.append(RDO._ridge_test(z_np[tr], y[tr], z_np[te], y[te], "std"))
            got = float(np.nanmean(folds))
            want = float(base[f"{args.baseline_prefix}|enc{TAP}|std|{task}"]["ws_per_session"][skey])
            d = abs(got - want)
            print(f"[parity-L1] {skey} {task:16s} here={got:.6f} disk={want:.6f} |d|={d:.2e} "
                  f"{'OK' if d < 1e-4 else 'FAIL'}", flush=True)
            if d >= 1e-4:
                raise SystemExit(f"PARITY-L1 FAIL {skey} {task}: |d|={d:.3e}")

        # ── L2: GPU ridge vs the CPU reference ──────────────────────────────────────────────
        task = PROBE_TASKS[0]
        y = np.asarray(targets.labels[task], dtype=np.float64)
        sp = targets.ws_split[task][0]
        tr, te = RDO._finite(y, sp["train"]), RDO._finite(y, sp["test"])
        zf = _flat16(z_all)
        a, b_ = _std_gpu(zf[tr], zf[te])
        gpu = RDO.auroc(_gpu_ridge(a, torch.as_tensor(y[tr], device=device), b_), y[te])
        cpu = RDO._ridge_test(z_np[tr], y[tr], z_np[te], y[te], "std")
        print(f"[parity-L2] {skey} {task} gpu={gpu:.6f} cpu={cpu:.6f} |d|={abs(gpu-cpu):.2e} "
              f"{'OK' if abs(gpu-cpu) < 1e-4 else 'FAIL'}", flush=True)
        if abs(gpu - cpu) >= 1e-4:
            raise SystemExit("PARITY-L2 FAIL — GPU ridge does not match the CPU reference.")
        del z_np, z_all, z16, zf, a, b_
        if args.parity_only:
            del x11, bands, spec, tower
            torch.cuda.empty_cache()
            continue

        # ── the cells ───────────────────────────────────────────────────────────────────────
        for task in PROBE_TASKS:
            y_all = np.asarray(targets.labels[task], dtype=np.float64)
            for fold, sp in targets.ws_split[task].items():
                tr = RDO._finite(y_all, sp["train"])
                va = RDO._finite(y_all, sp["val"])
                te = RDO._finite(y_all, sp["test"])
                # val and test are ADJACENT contiguous blocks (held-out fold halved), so the
                # last val window and the first test window can overlap in time — the M14 STFT
                # hop-overlap leak. Drop the margin off the VAL side only: `te` must stay
                # byte-identical to the frozen arm's test rows or the pairing is broken.
                if args.val_margin and len(va) > args.val_margin:
                    va = np.sort(va)[: -args.val_margin]
                if min(len(tr), len(va), len(te)) < 2:
                    print(f"[cell] {skey} {task} f{fold} SKIP (n<2)", flush=True)
                    continue
                yt = torch.as_tensor(y_all[tr], device=device)
                for arm in arms:
                    for lr in lrs:
                        enc.blocks[TAP - 1].load_state_dict(pristine)   # fresh weights per cell
                        r = _run_cell(enc, feats, tr, va, te, y_all, yt, device=device,
                                      arm=arm, lr=lr, args=args)
                        r.update(session=skey, task=task, fold=int(fold), arm=arm, lr=lr,
                                 wd=float(args.wd),
                                 n_train=len(tr), n_val=len(va), n_test=len(te))
                        results.append(r)
                        print(f"[cell] {skey} {task:16s} f{fold} {arm:8s} lr={lr:g} "
                              f"ep*={r['best_epoch']:2d} val={r['val']:.4f} "
                              f"ft={r['test_ft']:.4f} frozen={r['test_frozen']:.4f} "
                              f"vallam={r['test_frozen_vallam']:.4f} "
                              f"d={r['test_ft'] - r['test_frozen']:+.4f} "
                              f"n_tr={len(tr)} n_va={len(va)} n_te={len(te)}", flush=True)
                        json.dump(results, open(args.out, "w"), indent=1)
        enc.blocks[TAP - 1].load_state_dict(pristine)
        del x11, bands, spec, tower
        torch.cuda.empty_cache()

    if args.stage == "a":
        # Selected on the head's val AUROC — the same quantity the headline (D) reports.
        by: dict[tuple, list] = {}
        for r in results:
            by.setdefault((r["arm"], r.get("wd", args.wd), r["lr"]), []).append(r["val"])
        for arm in sorted({k[0] for k in by}):
            sel = {(w, lr): v for (a, w, lr), v in by.items() if a == arm}
            for w, lr in sorted(sel):
                print(f"[stage-a] arm={arm:8s} wd={w:g} lr={lr:g} "
                      f"mean val={float(np.nanmean(sel[(w, lr)])):.4f} n={len(sel[(w, lr)])}")
            bw, blr = max(sel, key=lambda k: float(np.nanmean(sel[k])))
            print(f"[stage-a] WINNER arm={arm} wd={bw:g} lr={blr:g}")
    else:
        _report(results, base, args)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[ft-pilot] wrote {args.out}", flush=True)


def _run_cell(enc, feats, tr, va, te, y_all, yt, *, device, arm, lr, args):
    """One fine-tune. Epoch 0 is evaluated BEFORE any step, so the frozen control is measured
    inside the same code path as the FT number — not read from elsewhere."""
    params = _arm_params(enc, arm)
    for q in enc.parameters():
        q.requires_grad_(False)
    for q in params:
        q.requires_grad_(True)
    n_par = sum(q.numel() for q in params)

    with torch.no_grad():
        dim = int(feats(tr[:1], False).reshape(1, -1).shape[1])
    head = _Head(dim).to(device)
    opt = torch.optim.AdamW(list(head.parameters()) + params, lr=lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    ybin = torch.as_tensor((y_all[tr] > 0).astype(np.float32), device=device)
    lossf = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)

    # A and D ONLY (Ben 07-30: "I think we should only do A and D only"). A is the frozen-ridge
    # number we quote; D is the fine-tuned head's own AUROC, which is what MAE reports. The two
    # intermediate cells are dropped, so the ridge is fit ONCE, at epoch 0, where it IS cell A
    # and the L1 parity gate. Epoch selection is by the head's own val AUROC.
    best = {"val": -1.0, "test_ft": float("nan"), "best_epoch": -1, "n_params": n_par}
    frozen_test = frozen_vallam = float("nan")
    for ep in range(args.epochs + 1):
        if ep > 0:
            enc.train()
            order = rng.permutation(len(tr))
            for s in range(0, len(order), args.train_batch):
                pos = order[s:s + args.train_batch]
                z = feats(tr[pos], True).reshape(len(pos), -1)
                loss = lossf(head(z), ybin[pos])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(head.parameters()) + params, 3.0)
                opt.step()
            sched.step()
        enc.eval()
        head.eval()
        with torch.no_grad():
            if ep == 0:
                # Cell A, measured in THIS code path rather than imported, plus the val-lambda
                # control that prices the one val-selected hyperparameter D gets and A does not.
                zs = [feats(r, False).reshape(len(r), -1).to(torch.float16).to(torch.float32)
                      for r in (tr, va, te)]
                _vs, frozen_test, frozen_vallam = _ridge_eval(
                    *zs, yt, y_all[va], y_all[te], [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
                del zs
            hv = RDO.auroc(head(feats(va, False).reshape(len(va), -1)).float().cpu().numpy(),
                           y_all[va])
            ht = RDO.auroc(head(feats(te, False).reshape(len(te), -1)).float().cpu().numpy(),
                           y_all[te])
        head.train()
        if np.isfinite(hv) and hv > best["val"]:
            best = {"val": float(hv), "test_ft": float(ht), "best_epoch": ep, "n_params": n_par}
        elif ep - best["best_epoch"] >= args.patience:
            break
    for q in enc.parameters():
        q.requires_grad_(False)
    best.update(test_frozen=float(frozen_test), test_frozen_vallam=float(frozen_vallam))
    return best


def _report(results, base, args):
    """28 paired cells (7 sessions x 4 tasks), fold-meaned — the same reduction the frozen
    readout does (``_ws_session`` nanmeans over folds), so the two are comparable.

    A vs D only. `ft-head` is D: the fine-tuned block-12 features read by the trained logistic
    head, which is the form MAE reports. `frozen` is A: the same rows, frozen features, constant-
    lambda ridge — the number every result of ours has quoted, recomputed in-path. `disk` is that
    same A read off the probe JSON, so the two must agree (that is parity L1, printed again here).
    `val-lam` prices the one val-selected hyperparameter D gets and A does not."""
    for arm in sorted({r["arm"] for r in results}):
        cells: dict[tuple, list] = {}
        for r in results:
            if r["arm"] == arm:
                cells.setdefault((r["session"], r["task"]), []).append(r)
        rows = []
        for (sess, task), rs in sorted(cells.items()):
            disk = float(base[f"{args.baseline_prefix}|enc{TAP}|std|{task}"]["ws_per_session"][sess])
            rows.append((sess, task,
                         float(np.nanmean([x["test_ft"] for x in rs])),
                         float(np.nanmean([x["test_frozen"] for x in rs])),
                         float(np.nanmean([x["test_frozen_vallam"] for x in rs])),
                         disk))
        if not rows:
            continue
        print(f"\n=== WS partial-FT, arm={arm} — {len(rows)} paired cells "
              f"({rows[0][0] if len(rows) < 8 else '7 sessions'} x {len(PROBE_TASKS)} tasks) ===")
        print(f"{'session':8s} {'task':16s} {'D ft-head':>9s} {'A frozen':>9s} {'val-lam':>8s} "
              f"{'disk':>8s} {'d(D-A)':>9s}")
        for sess, task, d_ft, a_fr, vl, disk in rows:
            print(f"{sess:8s} {task:16s} {d_ft:9.4f} {a_fr:9.4f} {vl:8.4f} {disk:8.4f} "
                  f"{d_ft - a_fr:+9.4f}")
        for label, j in (("D vs A const-lam", 3), ("D vs A val-lam", 4)):
            ds = [r[2] - r[j] for r in rows]
            nz = [x for x in ds if abs(x) > 1e-9]
            k = sum(x > 0 for x in nz)
            print(f"  {label:22s} mean d={float(np.mean(ds)):+.4f}  {k}/{len(nz)} positive  "
                  f"p={_sign_p(k, len(nz)):.4f}")
        ds = [r[2] - r[3] for r in rows]
        mean_d, k = float(np.mean(ds)), sum(x > 0 for x in ds)
        verdict = ("BUILD THE CS VERSION" if mean_d >= 0.010 and k >= 20 and len(rows) == 28
                   else "NEGATIVE — ridge stays" if mean_d > 0 else "CLOSE THE THREAD")
        print(f"  DECISION -> {verdict}"
              + ("" if len(rows) == 28 else f"  (PARTIAL: {len(rows)}/28 cells — not the gate)"))


if __name__ == "__main__":
    main()
