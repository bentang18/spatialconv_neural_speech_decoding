#!/usr/bin/env python
"""Cross-subject DECODER sweep on the board cache: MLP / CNN-naive / CNN-per-band on enc0|enc12.

WHY THIS EXISTS. The Neuroprobe leaderboard's published SOTA margin is decoder, not features.
On IDENTICAL `laplacian-stft_abs_nperseg512_poverlap0.75_maxfreq150` features the harness reports
Logistic .539 -> MLP .566 -> CNN .578 (neuroprobe/examples/eval_utils.py, `--classifier_type`).
Our board CS number uses a ridge, so our headline +.024 over the leader compares OUR LINEAR
decoder against THEIR CONVOLUTIONAL one; the matched linear-vs-linear frontend comparison is
enc0-ridge .5872 vs their-logistic .539 = +.048. This script swaps ONLY the decoder while holding
the split and parcel-intersect contract byte-identical to `_cs_cell`, so any delta is
attributable to the decoder alone.

TAP LAYOUT — verified 2026-07-26 against the live board cache and `stem.PER_BAND_SPECS`
`((7,8),(6,2),(7,1))` = (n_bins, lattice_stride) per band in SLOW, MID, HGA order:

    enc0   (n, |P|, 348)    concat_b (T_b x F_b), TIME-MAJOR within band  [_enc0_bands_canon
                            yields (.., T_b, F_b) then _pool_parcels flattens prod(feat)]
                            T_b = (4, 16, 32),  F_b = (7, 6, 7)
                            check: 4*7 + 16*6 + 32*7 = 28 + 96 + 224 = 348               OK
    enc12  (n, |P|, 13312)  (T_total, d_model) with T_total = 4+16+32 = 52 band-concatenated
                            check: 52 * 256 = 13312                                      OK

THE BAND-SEAM PROBLEM (why `perband` exists). Our 52-bin time axis is NOT uniform -- it is three
bands concatenated at different rates (4 Hz / 16 Hz / 32 Hz). The harness spectrogram has a single
uniform time axis, so a verbatim Conv2d slides kernels straight across the SLOW->MID and MID->HGA
seams, convolving bins that are not temporally adjacent. `naive` keeps that flaw deliberately --
it is the faithful "we ran THEIR decoder" control and the only legitimate head-to-head. `perband`
runs one stack per band so no kernel ever straddles a seam; it is the arm that may actually win.

Splits are `_cs_cell`'s exactly: train on the anchor's finite rows, model-select on the TEST
cell's val half, report its test half. Note this differs from the harness CNNClassifier, which
carves its early-stopping val out of the last 20% of TRAIN; we use our own val half so the number
stays apples-to-apples with OUR ridge (which lambda-selects on that same val half).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_board_readout import (  # noqa: E402
    BOARD_TASKS,
    CS_TEST_CELLS,
    CS_TRAIN_ANCHOR,
    _finite,
    _load,
    _parcel_cols,
)

# ── tap layout (see module docstring; do NOT hardcode elsewhere) ──────────────
BAND_T: tuple[int, ...] = (4, 16, 32)          # per-band token counts, SLOW/MID/HGA
BAND_F0: tuple[int, ...] = (7, 6, 7)           # enc0 per-band freq bins (stem.PER_BAND_SPECS)
D_MODEL = 256                                  # enc12 token width
ENC0_DIM = sum(t * f for t, f in zip(BAND_T, BAND_F0))   # 348
ENC12_DIM = sum(BAND_T) * D_MODEL                        # 13312


def tap_bands(enc: str) -> tuple[tuple[int, int], ...]:
    """Per-band (T_b, D_b) blocks for a tap, in cache flatten order.

    enc0 blocks are (T_b, F_b) with F_b band-specific; enc12 blocks are (T_b, d_model) since the
    encoder emits one d_model token per (band, position). Blocks tile the flat feature axis in
    SLOW, MID, HGA order, so `np.split` at their cumulative sizes recovers them exactly.
    """
    if enc == "enc0":
        return tuple(zip(BAND_T, BAND_F0))
    if enc in ("enc3", "enc6", "enc12"):
        return tuple((t, D_MODEL) for t in BAND_T)
    raise ValueError(f"unsupported tap {enc!r} (expected enc0|enc3|enc6|enc12)")


def tap_dim(enc: str) -> int:
    return sum(t * d for t, d in tap_bands(enc))


# ── decoders ─────────────────────────────────────────────────────────────────
class HarnessCNN(nn.Module):
    """The Neuroprobe `CNNClassifier` architecture VERBATIM (eval_utils.py:525-570).

    3-D input (P, T, D) -> Conv2d branch; 2-D input (P, F) -> Conv1d branch. Channel axis is the
    parcel axis, matching the harness, whose `combine_regions` also emits region-as-channel.
    enc12 takes the Conv2d branch on (P, 52, 256); enc0 takes the Conv1d branch on (P, 348),
    because 348 does not factor as (T x uniform-F) -- the bands carry different freq-bin counts,
    so there is no well-formed 2-D image to hand a Conv2d. Both branches are the harness's own.
    """

    def __init__(self, input_shape: tuple[int, ...], n_classes: int) -> None:
        super().__init__()
        self.two_d = len(input_shape) == 3
        if self.two_d:
            self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            flat = (input_shape[1] // 8) * (input_shape[2] // 8) * 128
        else:
            self.conv1 = nn.Conv1d(input_shape[0], 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool1d(2)
            flat = (input_shape[1] // 8) * 128
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flat, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class PerBandCNN(nn.Module):
    """Same 3-conv stack as `HarnessCNN`, but ONE STACK PER BAND -- no kernel crosses a seam.

    Pooling is on the FEATURE axis only: SLOW has T_b=4, so the harness's three time-halvings
    would annihilate it (4 -> 0). Each band ends in AdaptiveAvgPool2d(1) -> 128 channels, the
    three are concatenated -> 384, and the harness head (dropout -> 256 -> dropout -> classes)
    is reused verbatim so the arms differ ONLY in the convolutional body.
    """

    def __init__(self, n_parcels: int, blocks: tuple[tuple[int, int], ...], n_classes: int) -> None:
        super().__init__()
        self.blocks = blocks
        self.stacks = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(n_parcels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((1, 2)),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d((1, 2)),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            for _ in blocks
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * len(blocks), 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, P, F_flat) -> per-band (B, P, T_b, D_b)
        outs, off = [], 0
        for stack, (t_b, d_b) in zip(self.stacks, self.blocks):
            span = t_b * d_b
            blk = x[:, :, off : off + span].reshape(x.size(0), x.size(1), t_b, d_b)
            outs.append(stack(blk).flatten(1))
            off += span
        z = torch.cat(outs, dim=1)
        z = self.dropout(z)
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        return self.fc2(z)


class HarnessMLP(nn.Module):
    """The Neuroprobe `MLPClassifier` architecture VERBATIM (eval_utils.py:693, hidden [1024,1024]).

    Eats the flat (P*F) feature vector, so it never touches the band-seam question at all -- the
    zero-design-risk arm, and the one that discriminates "nonlinear readout" from "convolutional
    structure": if MLP ~= CNN on our features, the win is not architectural.
    """

    def __init__(self, input_size: int, n_classes: int, hidden=(1024, 1024)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


ATTN_TAPS: tuple[str, ...] = ("enc0", "enc3", "enc6", "enc12")


class AttnPoolDecoder(nn.Module):
    """Attention-pool ACROSS TAPS with a learned query -> one global token -> concat with the
    per-parcel tokens -> harness head.

    This is REVE's layer-pooling (arXiv 2510.21585 p.5: "attention pooling over the outputs of all
    MHA layers ... concatenated and attended by a learned query token") moved from PRETRAINING to
    READOUT. The pretraining version is a closed finding for us -- it is the documented provenance
    of the r4 secondary loss, which flatlined and was CUT -- so this deliberately does NOT reopen
    it. Nothing is retrained; the four taps are already sitting in the board cache.

    The motivation is the tap-depth result: enc12 > enc6 > enc3 > enc0 is monotone, so every ridge
    and every other arm here reads ONE depth and discards three. Depth carries complementary
    information (enc0 is the raw spectral envelope, enc12 is the contextualized code), and which
    depth is best is task-dependent. A learned query lets the decoder weight depths per example
    instead of us picking one globally.

        per tap t:  Linear(F_t -> d) shared across parcels  ->  (P, d)
                    mean over parcels                       ->  tap token tok_t  (d,)
        query   q:  a = softmax(q . tok_t / sqrt(d)) over the 4 taps   [convex, sums to 1]
        global  g:  sum_t a_t tok_t                                    (d,)
        parcels h:  the enc12 projection, kept per-parcel               (P, d)
        head:       flatten(concat([g], h)) -> 256 -> classes   (harness fc1/fc2/dropout verbatim)

    Only the tap-pooling is new: the head is byte-identical to the one `PerBandCNN` and
    `HarnessCNN` end in, so an attnpool-vs-perband delta is attributable to the pooling alone.
    """

    def __init__(self, n_parcels: int, tap_dims: tuple[int, ...], n_classes: int,
                 d: int = D_MODEL, main_tap: int = -1) -> None:
        super().__init__()
        self.d = d
        self.main_tap = main_tap                      # which tap supplies the per-parcel tokens
        self.proj = nn.ModuleList(nn.Linear(f, d) for f in tap_dims)
        self.query = nn.Parameter(torch.randn(d) * d ** -0.5)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear((n_parcels + 1) * d, 256)
        self.fc2 = nn.Linear(256, n_classes)
        self.relu = nn.ReLU()

    def tap_weights(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """(B, n_taps) convex attention over taps -- exposed so the fit loop can log which depth
        the query actually selected, which is the whole scientific point of the arm."""
        pooled = torch.stack([p(x).mean(1) for p, x in zip(self.proj, xs)], dim=1)  # (B, T, d)
        return torch.softmax(pooled @ self.query / self.d ** 0.5, dim=1)

    def forward(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        toks = [p(x) for p, x in zip(self.proj, xs)]              # each (B, P, d)
        pooled = torch.stack([t.mean(1) for t in toks], dim=1)    # (B, T, d)
        att = torch.softmax(pooled @ self.query / self.d ** 0.5, dim=1)   # (B, T)
        g = (att.unsqueeze(-1) * pooled).sum(1)                   # (B, d)
        h = toks[self.main_tap]                                   # (B, P, d)
        z = torch.cat([g.unsqueeze(1), h], dim=1).flatten(1)      # (B, (P+1)*d)
        z = self.dropout(z)
        z = self.relu(self.fc1(z))
        z = self.dropout(z)
        return self.fc2(z)


# harness hyperparameters, per decoder (eval_utils.py class defaults)
ARM_HP = {
    "mlp": {"lr": 1e-5, "batch": 200, "max_iter": 100, "patience": 100},
    "naive": {"lr": 1e-4, "batch": 128, "max_iter": 100, "patience": 10},
    "perband": {"lr": 1e-4, "batch": 128, "max_iter": 100, "patience": 10},
    # attnpool is not a harness decoder, so there is no default to copy; it shares the CNN arms'
    # schedule so a win cannot be attributed to a longer/hotter fit.
    "attnpool": {"lr": 1e-4, "batch": 128, "max_iter": 100, "patience": 10},
}

MULTI_ARMS = ("attnpool",)          # arms that consume ALL of ATTN_TAPS at once, not one tap


def build(arm: str, enc, n_parcels: int, n_classes: int) -> nn.Module:
    if arm == "attnpool":
        taps = tuple(enc)
        return AttnPoolDecoder(n_parcels, tuple(tap_dim(t) for t in taps), n_classes)
    blocks = tap_bands(enc)
    if arm == "mlp":
        return HarnessMLP(n_parcels * tap_dim(enc), n_classes)
    if arm == "perband":
        return PerBandCNN(n_parcels, blocks, n_classes)
    if arm == "naive":
        if enc == "enc12":
            return HarnessCNN((n_parcels, sum(BAND_T), D_MODEL), n_classes)
        return HarnessCNN((n_parcels, tap_dim(enc)), n_classes)
    raise ValueError(f"unknown arm {arm!r}")


def shape_for(arm: str, enc: str, n_parcels: int) -> tuple[int, ...]:
    """Input tensor shape (excluding batch) each arm expects, given the flat (P, F) cache tap."""
    if arm == "mlp":
        return (n_parcels, tap_dim(enc))
    if arm == "perband":
        return (n_parcels, tap_dim(enc))
    if enc == "enc12":
        return (n_parcels, sum(BAND_T), D_MODEL)
    return (n_parcels, tap_dim(enc))


# ── train / eval ─────────────────────────────────────────────────────────────
def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _batches(n: int, bs: int, shuffle: bool, gen: np.random.Generator):
    idx = gen.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs]


def fit_score(arm, enc, z_tr, y_tr, z_va, y_va, z_te, y_te, n_parcels, device, seed=42):
    """Train one decoder; model-select on val AUROC; return (val, test) AUROC at the best epoch."""
    hp = ARM_HP[arm]
    torch.manual_seed(seed)
    gen = np.random.default_rng(seed)
    multi = arm in MULTI_ARMS          # z_* are LISTS of per-tap arrays, not one array
    shape: tuple[int, ...] = () if multi else shape_for(arm, enc, n_parcels)
    model = build(arm, enc, n_parcels, 2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    lossf = nn.CrossEntropyLoss()

    def _prep(z, rows):
        if multi:
            return tuple(torch.from_numpy(zi[rows]).to(device, torch.float32) for zi in z)
        t = torch.from_numpy(z[rows]).to(device, torch.float32)
        return t.view(len(rows), *shape)

    yt = torch.from_numpy(y_tr.astype(np.int64)).to(device)
    best_va, best_te, bad = -1.0, float("nan"), 0
    best_w: list[float] = []
    for _ in range(hp["max_iter"]):
        model.train()
        for b in _batches(len(y_tr), hp["batch"], True, gen):
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(_prep(z_tr, b)), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = np.concatenate(
                [
                    torch.softmax(model(_prep(z_va, b)), 1)[:, 1].float().cpu().numpy()
                    for b in _batches(len(y_va), 512, False, gen)
                ]
            )
            st = np.concatenate(
                [
                    torch.softmax(model(_prep(z_te, b)), 1)[:, 1].float().cpu().numpy()
                    for b in _batches(len(y_te), 512, False, gen)
                ]
            )
        va = _auroc(y_va, sv)
        if va > best_va:
            best_va, best_te, bad = va, _auroc(y_te, st), 0
            if multi:
                # which depth the learned query actually selected, at the SELECTED epoch
                with torch.no_grad():
                    w = torch.cat([model.tap_weights(_prep(z_va, b))
                                   for b in _batches(len(y_va), 512, False, gen)])
                best_w = [round(float(x), 4) for x in w.mean(0).cpu()]
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    return best_va, best_te, best_w


def _feat_struct(rec, enc, rows, cols) -> np.ndarray:
    """(n, |P|, F) fp32 -- like `_board_readout._feat` but WITHOUT the final flatten, because the
    decoders need the parcel axis (conv channels) and the per-band structure preserved."""
    x = rec["feats"][enc]["raw"][np.asarray(rows, dtype=np.int64)]
    x = x[:, np.asarray(cols, dtype=np.int64)]
    return x.to(torch.float32).numpy()


def _standardize_inplace(z_tr: np.ndarray, *others: np.ndarray) -> None:
    """Per-feature z on TRAIN stats only (harness StandardScaler; our ridge's _standardize).

    IN-PLACE: z_tr for enc12 is (n_tr, |P|, 13312) fp32 ~ 11 GB at a full anchor, so an
    out-of-place `(z - mu) / sd` would transiently double it and OOM the ghx4 fair-share.
    """
    mu = z_tr.mean(0, keepdims=True)
    sd = z_tr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    for z in (z_tr, *others):
        np.subtract(z, mu, out=z)
        np.divide(z, sd, out=z)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--index", type=int, required=True, help="index into CS_TEST_CELLS")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--arms", default="mlp,naive,perband")
    ap.add_argument("--taps", default="enc0,enc12")
    ap.add_argument("--tasks", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_te, t_te = CS_TEST_CELLS[args.index]
    tasks = tuple(args.tasks.split(",")) if args.tasks else BOARD_TASKS
    arms = tuple(args.arms.split(","))
    multi_arms = tuple(a for a in arms if a in MULTI_ARMS)
    plain_arms = tuple(a for a in arms if a not in MULTI_ARMS)
    # multi-tap arms fix their own tap set (ATTN_TAPS); --taps only governs the per-tap arms
    taps = tuple(args.taps.split(",")) if plain_arms else ()

    # mmap=True is LOAD-BEARING, not an optimization: a board cache record is 40-64 GB on disk and
    # an eager torch.load pulls ALL taps, including the ~34 GB enc12_elec this script never touches.
    # Two eager records = ~100 GB, which OOM-killed every enc0 cell (32G) in <45 s and most enc12
    # cells (113560M) on 2026-07-26. Under mmap, pages arrive only where a tensor is indexed, so
    # the resident set is just the parcel taps we actually gather.
    anchor = _load(args.cache_dir, CS_TRAIN_ANCHOR, args.tag, mmap=True)
    test = _load(args.cache_dir, (s_te, t_te), args.tag, mmap=True)
    a_idx, t_idx, common = _parcel_cols(anchor, test)
    n_parcels = int(common.size)
    print(f"[cell] anchor=S{CS_TRAIN_ANCHOR[0]}T{CS_TRAIN_ANCHOR[1]} "
          f"test=S{s_te}T{t_te} parcels={n_parcels} device={device}", flush=True)
    if n_parcels == 0:
        raise SystemExit("empty parcel intersection")

    # INVARIANT: the cached flat feature axis must equal the layout this module assumes, or every
    # unflatten below is silently wrong. Assert it once, loudly, before any fitting happens.
    for enc in dict.fromkeys(taps + (ATTN_TAPS if multi_arms else ())):
        got = int(anchor["feats"][enc]["raw"].shape[-1])
        want = tap_dim(enc)
        ok = got == want
        print(f"[check] {enc} flat_dim={got} expected={want} -> {'OK' if ok else 'VIOLATED'}",
              flush=True)
        if not ok:
            raise SystemExit(f"{enc} layout mismatch: cache {got} != contract {want}")

    out: dict = {"tag": args.tag, "cell": f"S{s_te}T{t_te}", "n_parcels": n_parcels, "cells": {}}
    os.makedirs(args.out_dir, exist_ok=True)
    dest = os.path.join(args.out_dir, f"dec_S{s_te}T{t_te}.json")

    def _flush() -> None:
        """Checkpoint after every task: a cell that overruns its walltime still yields the tasks
        it finished, instead of discarding hours of GPU work at the SIGKILL."""
        tmp = dest + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, dest)

    for task in tasks:
        y_a = np.asarray(anchor["labels"][task], dtype=np.float64)
        y_t = np.asarray(test["labels"][task], dtype=np.float64)
        tr = _finite(y_a, np.arange(len(y_a)))
        va = _finite(y_t, test["cs_split"][task]["val"])
        te = _finite(y_t, test["cs_split"][task]["test"])
        if len(tr) < 2 or len(te) < 2:
            continue
        for enc in taps:
            z_tr = _feat_struct(anchor, enc, tr, a_idx)
            z_va = _feat_struct(test, enc, va, t_idx)
            z_te = _feat_struct(test, enc, te, t_idx)
            _standardize_inplace(z_tr, z_va, z_te)
            for arm in plain_arms:
                t0 = time.time()
                v, s, _ = fit_score(arm, enc, z_tr, y_a[tr], z_va, y_t[va], z_te, y_t[te],
                                    n_parcels, device)
                out["cells"][f"{enc}|{arm}|{task}"] = {"val": v, "test": s}
                print(f"  {task:<18} {enc:<6} {arm:<8} val={v:.4f} test={s:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            del z_tr, z_va, z_te

        for arm in multi_arms:
            # all four taps resident at once; standardized per tap on anchor stats, as above
            zs_tr, zs_va, zs_te = [], [], []
            for enc in ATTN_TAPS:
                a = _feat_struct(anchor, enc, tr, a_idx)
                b = _feat_struct(test, enc, va, t_idx)
                c = _feat_struct(test, enc, te, t_idx)
                _standardize_inplace(a, b, c)
                zs_tr.append(a)
                zs_va.append(b)
                zs_te.append(c)
            t0 = time.time()
            v, s, w = fit_score(arm, ATTN_TAPS, zs_tr, y_a[tr], zs_va, y_t[va], zs_te, y_t[te],
                                n_parcels, device)
            out["cells"][f"multitap|{arm}|{task}"] = {"val": v, "test": s, "tap_w": w}
            print(f"  {task:<18} {'4tap':<6} {arm:<8} val={v:.4f} test={s:.4f} "
                  f"w={w} ({time.time()-t0:.0f}s)", flush=True)
            del zs_tr, zs_va, zs_te
        _flush()

    _flush()
    print(f"[done] wrote {dest} ({len(out['cells'])} fits)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
