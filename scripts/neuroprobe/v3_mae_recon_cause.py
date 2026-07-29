"""Split r6 MAE reconstruction quality by MASK CAUSE x BAND — no training required.

The question (Ben 2026-07-29): the r6 spatial mask is a TUBE — ``contact_mask`` is (R, N), so
deleting a contact deletes it across all three bands and all time. Is the model able to
reconstruct a spatially-deleted contact at all, and does that ability differ by band? SLOW is
volume-conducted and should interpolate along a shaft; HGA is focal, and at BrainTreebank's
measured 3.61 mm pitch (M6, 2026-07-15) Duraivel 2023 Fig 2d puts inter-electrode HG
correlation near r ~ 0.2-0.35. If spatial-HGA comes back near zero while spatial-SLOW is
high, the tube is the culprit and per-band spatial widths get set from data, not argument.

Two causes, and they are DISJOINT by construction:

  SPACE — the contact is in ``contact_mask``. EVERY token of that contact is masked (all 3
          bands, all time), because ``token_flags_r6`` ORs contact_mask into the band masks.
  TIME  — the contact SURVIVED spatially, and this band-token fell in a width-4 band-time
          block on its own sensor's grid.

``masked = contact_masked | temporal_masked`` and ``in_loss == masked`` (no margin gate), so
"space" and "time-only" partition the scored set exactly. The partition is ASSERTED below,
not assumed.

Metrics per (band, cause), pooled over all valid (token, bin) pairs:
  r                  Pearson correlation of prediction against target
  std(pred)/std(tgt) shrinkage. An MAE trained on an unpredictable target learns the
                     conditional mean, which drives BOTH r and this ratio toward 0 together.
  mse/token          the actual per-token loss contribution
  loss share         fraction of the total MAE loss this cell carries

The objective is NOT modified. ``_mae_output`` is wrapped here exactly as in
``v3_mae_recon.py`` — a debug flag in the training path would be a permanent hazard, and the
live dtai queue imports that tree. This script only READS from it.

GPU, on dtai.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np
import torch

from scripts.neuroprobe.v3_mae_recon import clip_starts_seconds
from scripts.neuroprobe.v3_probe_encode_r4 import (
    _load_ckpt, _lite_keep_labels_fn, _subtree, _window_bands,
)
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig, sample_masks_r6

HZ = 32.0
BAND_NAMES = ("SLOW", "MID", "HGA")  # grid.band: 0=SLOW 1=MID 2=HGA (pack_r4.R4Grid)


def cell_stats(pred: np.ndarray, tgt: np.ndarray) -> dict:
    """Pearson r + shrinkage over a flat set of (token, bin) pairs."""
    if pred.size < 2:
        return {"n_pairs": int(pred.size), "r": float("nan"), "std_ratio": float("nan"),
                "std_pred": float("nan"), "std_tgt": float("nan")}
    sp, st = float(pred.std()), float(tgt.std())
    # a constant prediction has zero variance => r is undefined, NOT zero. Report it as nan
    # and let std_ratio carry the "it collapsed to the conditional mean" signal, rather than
    # letting a divide-by-zero print a spurious correlation.
    r = float(np.corrcoef(pred, tgt)[0, 1]) if sp > 0 and st > 0 else float("nan")
    return {"n_pairs": int(pred.size), "r": r, "std_ratio": (sp / st) if st > 0 else float("nan"),
            "std_pred": sp, "std_tgt": st}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--n-clips", type=int, default=32)
    p.add_argument("--clip-dur", type=float, default=2.0)
    p.add_argument("--electrode-set", choices=("pretrain", "lite"), default="pretrain",
                   help="pretrain = the FULL montage the model was trained on (default). "
                        "lite restricts to the Neuroprobe-Lite montage, which changes shaft "
                        "sizes and therefore the realized mask geometry — wrong for a "
                        "diagnostic about the training objective.")
    p.add_argument("--space-frac", type=float, default=None,
                   help="override V3MaskConfig.space_frac (default: the locked 0.50)")
    p.add_argument("--block-w-space", type=int, default=None,
                   help="override V3MaskConfig.block_w_space (default: the locked 4)")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_frames = int(round(args.clip_dur * HZ))
    print(f"[check] device={device} clip_frames={clip_frames} ({args.clip_dur}s @ {HZ}Hz) "
          f"n_clips={args.n_clips}", flush=True)

    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    # the montage decides shaft sizes, and shaft sizes decide the realized spatial mask
    # (d_s = round(space_frac * n_s) contiguous width-4 blocks PER SHAFT). Running the Lite
    # montage here would measure a mask geometry the model never trained under.
    keep_labels_fn = _lite_keep_labels_fn(args.bt_root) if args.electrode_set == "lite" else None
    print(f"[check] electrode_set={args.electrode_set} "
          f"keep_labels_fn={'lite' if keep_labels_fn else 'None (full pretrain montage)'}",
          flush=True)
    spec = load_v3_sessions(
        sessions=[(args.subject, args.trial)], band_cache_dirs=args.band_cache_dirs,
        span_dir=args.span_dir, parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=None, winsor=(15.0, 15.0, 20.0), keep_labels_fn=keep_labels_fn,
    )[0]

    # shaft sizes ARE the spatial-mask geometry, so record them next to the result
    _cs = spec.setup.geom.valid.sum(1)
    print(f"[check] montage: {int(spec.setup.geom.valid.sum())} contacts / "
          f"{spec.setup.geom.n_shafts} shafts, contacts-per-shaft "
          f"min={int(_cs.min())} median={int(_cs.median())} max={int(_cs.max())}", flush=True)

    starts = clip_starts_seconds(int(spec.n_frames), clip_frames, args.n_clips)
    bands = [b.to(device) for b in _window_bands(spec, starts, clip_frames, rate_mult=1)]
    geom = spec.setup.geom.to(device)
    parcel_id = spec.setup.parcel_id.to(device)

    sd = _load_ckpt(args.ckpt)
    from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
    n_parcels = int(parcel_id.max().item()) + 1
    obj = V3JepaObjective(n_parcels=max(n_parcels, 75), mae=True, r6=True).to(device)
    sub = _subtree(sd, "objective.")
    assert sub, f"no objective.* subtree in the ckpt; sample keys {list(sd)[:5]}"
    missing, unexpected = obj.load_state_dict(sub, strict=False)
    # strict=False makes a prefix mistake SILENT -- it would forward RANDOM weights and report
    # a spatial-HGA r of ~0, which is exactly the result we are testing for. This assert is
    # what stops an untrained model from confirming the hypothesis.
    assert not missing, f"{len(missing)} params not in the ckpt: {missing[:8]}"
    print(f"[check] loaded objective: matched={len(sub)} missing=0 unexpected={len(unexpected)}",
          flush=True)
    obj.eval()

    cfg_kw = {}
    if args.space_frac is not None:
        cfg_kw["space_frac"] = args.space_frac
    if args.block_w_space is not None:
        cfg_kw["block_w_space"] = args.block_w_space
    cfg = V3MaskConfig(**cfg_kw)
    print(f"[check] mask cfg: space_frac={cfg.space_frac} block_w_space={cfg.block_w_space} "
          f"band fracs slow/mid/hga={cfg.slow_mask_frac}/{cfg.mid_mask_frac}/{cfg.hga_mask_frac} "
          f"block_w_band={cfg.block_w_band}", flush=True)

    gen = torch.Generator().manual_seed(args.seed)
    masks = sample_masks_r6(spec.setup.geom, int(parcel_id.shape[0]), n_time=clip_frames,
                            n_rows=len(starts), generator=gen, cfg=cfg)
    masks = dataclasses.replace(masks, **{f.name: getattr(masks, f.name).to(device)
                                          for f in dataclasses.fields(masks)})

    seen: dict = {}
    orig = obj._mae_output

    def spy(bands_, grid_, h_, in_loss_, **kw):
        seen.update(bands=bands_, grid=grid_, h=h_, in_loss=in_loss_)
        return orig(bands_, grid_, h_, in_loss_, **kw)

    obj._mae_output = spy
    with torch.no_grad():
        out = obj.forward(bands, geom, parcel_id, masks)
    obj._mae_output = orig
    assert seen, "the MAE arm never ran -- is this a JEPA checkpoint?"

    with torch.no_grad():
        target, feat_valid, feat_count = obj._mae_gather_target(seen["bands"], seen["grid"])
        pred = obj._mae_pred(seen["h"], seen["grid"])
    g = seen["grid"]
    in_loss = seen["in_loss"].bool()                      # (B, total)
    space = masks.contact_mask[:, g.contact, g.band].bool()  # (B, total) cause = SPACE (per band)
    time_only = in_loss & ~space                          # (B, total) cause = TIME

    # ── invariants, printed BEFORE any result is read ────────────────────────────
    frac = float(in_loss.float().mean())
    band_np = g.band.cpu().numpy()
    tok_share = {BAND_NAMES[b]: float((band_np == b).mean()) for b in range(3)}
    # the tube: a spatially-masked contact must have EVERY one of its tokens scored. If this
    # fails, "space" and "time-only" are not a partition and every number below is mixed.
    space_all_scored = bool((space & ~in_loss).sum().item() == 0)
    per_tok_mse = (((pred - target) ** 2) * feat_valid[None].to(pred.dtype)).sum(-1) \
        / feat_count[None].to(pred.dtype)                 # (B, total)
    w = in_loss.to(pred.dtype)
    loss_tot = float((per_tok_mse * w).sum())
    # per_tok_mse is RECOMPUTED here from pred/target; the objective already reduced the same
    # quantity to its loss. If these disagree the recomputation is wrong and every per-cell
    # number below is wrong with it, so check it before reading any result.
    loss_recomputed = loss_tot / max(float(w.sum()), 1.0)
    print(f"[check] loss: objective={float(out.loss):.6f} recomputed={loss_recomputed:.6f} "
          f"({'OK match' if abs(float(out.loss) - loss_recomputed) < 1e-4 else 'MISMATCH'})",
          flush=True)
    assert abs(float(out.loss) - loss_recomputed) < 1e-4, (
        f"recomputed loss {loss_recomputed} != objective loss {float(out.loss)}")
    print(f"[check] in_loss fraction = {frac:.4f}  "
          f"({'OK ~0.75' if 0.70 < frac < 0.80 else 'UNEXPECTED for r6'})", flush=True)
    print(f"[check] every spatially-masked token is scored: {space_all_scored} "
          f"({'OK — space/time-only partition the loss' if space_all_scored else 'BROKEN'})",
          flush=True)
    print(f"[check] token share by band: " +
          "  ".join(f"{k}={v:.3f}" for k, v in tok_share.items()) +
          "   (expect SLOW .077 MID .308 HGA .615 at 2s)", flush=True)
    print(f"[check] loss share space-caused = "
          f"{float((per_tok_mse * space.to(pred.dtype)).sum()) / loss_tot:.3f} "
          f"(arithmetic predicts .667 at space_frac=0.50)", flush=True)
    assert space_all_scored, "contact_mask tokens are not all in_loss — cause split is invalid"

    # ── the split ────────────────────────────────────────────────────────────────
    fv = feat_valid.cpu().numpy().astype(bool)            # (total, F_MAX)
    pr = pred.float().cpu().numpy()                       # (B, total, F_MAX)
    tg = target.float().cpu().numpy()
    mse = per_tok_mse.float().cpu().numpy()               # (B, total)
    rows = []
    for b, name in enumerate(BAND_NAMES):
        bsel = band_np == b                               # (total,)
        for cause, m in (("space", space.cpu().numpy()), ("time", time_only.cpu().numpy())):
            sel = m & bsel[None]                          # (B, total)
            if not sel.any():
                continue
            # gather only the VALID bins of the selected tokens
            bins = fv[None].repeat(sel.shape[0], 0) & sel[..., None]
            st = cell_stats(pr[bins], tg[bins])
            st.update(band=name, cause=cause, n_tokens=int(sel.sum()),
                      mse_per_token=float(mse[sel].mean()),
                      loss_share=float(mse[sel].sum() / loss_tot))
            rows.append(st)

    hdr = (f"{'band':>5s} {'cause':>6s} {'tokens':>9s} {'loss%':>7s} {'r':>7s} "
           f"{'std(p)/std(t)':>14s} {'mse/token':>10s}")
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r_ in sorted(rows, key=lambda x: -x["loss_share"]):
        print(f"{r_['band']:>5s} {r_['cause']:>6s} {r_['n_tokens']:>9d} "
              f"{r_['loss_share']*100:6.1f}% {r_['r']:7.3f} {r_['std_ratio']:14.3f} "
              f"{r_['mse_per_token']:10.4f}", flush=True)

    payload = {
        "subject": args.subject, "trial": args.trial, "ckpt": args.ckpt,
        "n_clips": args.n_clips, "clip_dur": args.clip_dur, "seed": args.seed,
        "space_frac": cfg.space_frac, "block_w_space": cfg.block_w_space,
        "in_loss_frac": frac, "token_share": tok_share, "loss_total": loss_tot,
        "space_partition_ok": space_all_scored, "rows": rows,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[write] {args.out} ({os.path.getsize(args.out)} B)", flush=True)


if __name__ == "__main__":
    main()
