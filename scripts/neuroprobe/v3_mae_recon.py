"""Dump one session's r6 MAE reconstruction so the masked-recon strip can be drawn locally.

What the model was actually trained to do: 75% of the tokens are hidden and it predicts each
hidden token's own |STFT| bins. This writes ground truth, prediction and the mask for a
handful of real clips; the plotting is a separate local step because it is cheap and the
forward is not.

The objective is NOT modified to expose its internals. ``_mae_output`` is wrapped HERE, in
this script, to record the arguments it already receives, and the target/prediction are then
recomputed with the objective's own methods. A debug flag in the training path would be a
permanent hazard for a one-off figure; a local wrapper cannot affect any run.

GPU, on dtai. CPU-only work does not belong here.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from scripts.neuroprobe.v3_probe_encode_r4 import (
    _load_ckpt, _lite_keep_labels_fn, _subtree, _window_bands,
)
from speech_decoding.models.v14_converged_v3.masking import V3MaskConfig, sample_masks_r6

HZ = 32.0


def clip_starts_seconds(n_frames: int, clip_frames: int, n_clips: int) -> np.ndarray:
    """Evenly spaced clip starts, in SECONDS, spanning the session.

    ``_window_bands`` takes seconds and multiplies by FPS itself. Handing it frame indices
    instead is a unit bug that does not look like one -- it raises "start=30763.0000s out of
    bounds", which reads as a corrupt cache rather than a caller mistake. The spacing is
    computed in frames and divided down so ``rint(start * FPS)`` recovers the exact integer
    frame, leaving no chance that the last clip rounds one frame past the end.
    """
    last = n_frames - clip_frames
    if last <= 0:
        raise ValueError(f"session has {n_frames} frames, shorter than a {clip_frames}-frame clip")
    frames = np.rint(np.linspace(0, last, n_clips)).astype(np.int64)
    return frames / HZ


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--trial", type=int, required=True)
    p.add_argument("--n-clips", type=int, default=8)
    p.add_argument("--clip-dur", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_frames = int(round(args.clip_dur * HZ))
    print(f"[check] device={device} clip_frames={clip_frames} ({args.clip_dur}s @ {HZ}Hz)",
          flush=True)

    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    keep_labels_fn = _lite_keep_labels_fn(args.bt_root)
    spec = load_v3_sessions(
        sessions=[(args.subject, args.trial)], band_cache_dirs=args.band_cache_dirs,
        span_dir=args.span_dir, parcel_fn=make_bt_parcel_fn(args.bt_root),
        lof_report_path=None, winsor=(15.0, 15.0, 20.0), keep_labels_fn=keep_labels_fn,
    )[0]

    # clips spread across the session, not the first N in a row: consecutive windows overlap
    # and would make the strip look more consistent than the data is
    starts = clip_starts_seconds(int(spec.n_frames), clip_frames, args.n_clips)
    print(f"[check] n_frames={spec.n_frames} ({spec.n_frames / HZ:.1f}s) "
          f"starts_s={[round(float(s), 2) for s in starts]}", flush=True)
    # rate_mult=1, no per-band rates: r4 and r6 share the uniform 32 Hz caches
    # (v3_probe_encode_r4.py:610). Declaring per-band rates here would misalign SLOW/MID.
    bands = _window_bands(spec, starts, clip_frames, rate_mult=1)
    bands = [b.to(device) for b in bands]

    geom = spec.setup.geom.to(device)
    parcel_id = spec.setup.parcel_id.to(device)
    # no grid built here: objective.forward builds its own from geom + n_time, and the spy
    # below hands back exactly the grid the loss was computed on

    sd = _load_ckpt(args.ckpt)
    from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
    n_parcels = int(parcel_id.max().item()) + 1
    obj = V3JepaObjective(n_parcels=max(n_parcels, 75), mae=True, r6=True).to(device)
    # _subtree, not a bare startswith("objective."): the keeper was trained under
    # torch.compile, so every key is "model._orig_mod.objective.…". A plain prefix test
    # matches nothing, and strict=False turns that into a SILENT random-weight forward --
    # the run would produce a reconstruction figure of an untrained model.
    sub = _subtree(sd, "objective.")
    # the namespace is printed BEFORE the load, not inferred from the failure afterwards: the
    # dtai login node cannot import torch right now (Errno 5 on /sw/user/python), so a
    # checkpoint that does not match has to explain itself from inside the job or not at all
    def _roots(keys):
        r: dict[str, int] = {}
        for k in keys:
            r[k.split(".")[0]] = r.get(k.split(".")[0], 0) + 1
        return dict(sorted(r.items()))
    print(f"[check] ckpt keys={len(sd)} sample={list(sd)[:2]}", flush=True)
    print(f"[check] objective subtree={len(sub)} roots={_roots(sub)}", flush=True)
    print(f"[check] fresh objective roots={_roots(obj.state_dict())}", flush=True)
    assert sub, f"no objective.* subtree in the ckpt; sample keys {list(sd)[:5]}"

    missing, unexpected = obj.load_state_dict(sub, strict=False)
    print(f"[check] load_state_dict matched={len(sub)} missing={len(missing)} "
          f"unexpected={len(unexpected)}", flush=True)
    # missing is the one that has to be zero. strict=False makes a prefix mistake SILENT --
    # it would forward random weights and draw a reconstruction figure of an untrained model,
    # which looks like a bad result rather than a bug. unexpected is only reported: a ckpt
    # carrying extra keys (an EMA teacher, say) still leaves every parameter here loaded.
    assert not missing, f"{len(missing)} params not in the ckpt: {missing[:8]}"
    if unexpected:
        print(f"[check] ckpt carries {len(unexpected)} keys the objective does not "
              f"(ignored): {unexpected[:5]}", flush=True)
    obj.eval()

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    masks = sample_masks_r6(geom, int(parcel_id.shape[0]), n_time=clip_frames,
                            n_rows=len(starts), generator=gen, cfg=V3MaskConfig())

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
    print(f"[check] loss={float(out.loss):.4f} n_masked={float(out.n_masked):.0f}", flush=True)

    with torch.no_grad():
        target, feat_valid, feat_count = obj._mae_gather_target(seen["bands"], seen["grid"])
        pred = obj._mae_pred(seen["h"], seen["grid"])
    in_loss = seen["in_loss"]
    g = seen["grid"]
    frac = float(in_loss.float().mean())
    print(f"[check] masked fraction (in_loss) = {frac:.4f}  "
          f"-> {'OK' if 0.6 < frac < 0.85 else 'UNEXPECTED (r6 masks ~0.75)'}", flush=True)

    np.savez_compressed(
        args.out,
        target=target.float().cpu().numpy(), pred=pred.float().cpu().numpy(),
        in_loss=in_loss.bool().cpu().numpy(), feat_valid=feat_valid.bool().cpu().numpy(),
        feat_count=feat_count.cpu().numpy(),
        band=g.band.cpu().numpy(), contact=g.contact.cpu().numpy(),
        # both are recoverable from band/feat_count, but storing them keeps the plotting
        # step from having to re-derive the token layout it is trying to display
        band_lengths=np.asarray([int(x) for x in g.band_lengths], dtype=np.int64),
        band_fdims=np.asarray([int(b.shape[-2]) for b in bands], dtype=np.int64),
        k_full=np.int64(g.k_full), clip_frames=np.int64(clip_frames),
        starts_s=starts, subject_id=np.int64(args.subject), trial_id=np.int64(args.trial),
        labels=np.asarray(spec.setup.sidecar.labels, dtype=object).astype(str),
    )
    print(f"[write] {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
