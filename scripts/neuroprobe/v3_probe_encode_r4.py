"""r4 (v14_converged_v3 Design B) depth-ladder probe — Stage 1 encode.

Adapts the r1 CS-erosion probe (``v3_probe_encode.py``) to the r4 tower. r4 is a DIFFERENT
architecture — ``PerBandStem`` decimates each band to its own token rate (HGA 32 / MID 16 /
SLOW 4 tokens per 1 s clip) and a flat-L1 encoder attends over the RAGGED per-(contact,band)
tokens packed varlen per shaft (``pack_r4.build_r4_grid``). There is no ``forward_padded``;
the teacher runs ``encoder.forward_flat`` over the packed grid.

Taps (Ben 2026-07-15): enc0, enc3, enc6, enc12.
  - enc0 = the pre-projection DECIMATED raw band bins (``x[..., ::stride]`` per band, the exact
    tensor ``PerBandStem``'s per-band Linear consumes) — the M9 input-linear floor, decimated to
    what the model actually sees. No model, CPU-side (computed here so bands aren't re-read).
  - enc3/6/12 = raw block outputs of the EMA teacher (``_TargetTower``, the shipped
    representation), read in ONE forward via ``tap_blocks=(3,6,12)``.

Feature = keep-time, NATIVE ragged (no hold-up: linear ridge, frame alignment irrelevant),
ELECTRODES POOLED TO PARCELS at encode (Ben 2026-07-15, OOM guard): per (band-slot, parcel)
MEAN over electrodes-in-parcel (Ben 2026-07-16: mean is the transferable parcel summary; std
conflates with per-subject electrode sampling, a CS nuisance). WS uses all present parcels; CS
intersects anchor/test parcels at readout. Pooling is per-(row, band-slot, parcel) hence
row-independent, so pre-pooling then row-subsetting is numerically identical to pool-at-readout
— but it shrinks the cache from ~14 GB/session (per-electrode keep-time) to well under 1 GB.

Runs from the LIVE r4 tree /projects/bhqk/htang13/speech (@ 2d3f52d = the ckpt's commit), so
the teacher state_dict matches by construction. GPU for enc3/6/12; enc0 + pooling are CPU.

Usage (one 1-GPU allocation):
  .venv/bin/python -m scripts.neuroprobe.v3_probe_encode_r4 \
      --ckpt /projects/bhqk/htang13/v3_ckpt_r4/ladder-step=10000.ckpt --tag r4_10k \
      --out-dir /projects/bhqk/htang13/v3_probe_cache_r4_10k \
      --band-cache-dir <slow> --band-cache-dir <mid> --band-cache-dir <hga> \
      --span-dir <spans> --bt-root <bt_root>
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

PROBE_TASKS: tuple[str, ...] = ("onset", "delta_volume", "word_index", "gpt2_surprisal")
FPS = 32.0
CLIP_DUR_S = 1.0
N_PARCELS = 75
GPU_TAPS: tuple[int, ...] = (3, 6, 12)   # raw block outputs read in one teacher forward


def _load_teacher(ckpt_path: str, *, device: torch.device):
    """Load ONLY the EMA teacher tower (`_TargetTower` = PerBandStem + encoder) from the ckpt.

    Filter the LightningModule state_dict to the ``objective.teacher.model.*`` subtree and
    load it into a fresh ``_TargetTower`` (strict) — no need to build the full model / secondary
    head, so the load is independent of the objective's post-launch changes (#46 mean floor)."""
    from speech_decoding.models.v14_converged_v3.objective import _TargetTower

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if "state_dict" in raw else raw
    pref = "objective.teacher.model."
    tsd = {}
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk.startswith("model."):
            kk = kk[len("model."):]
        if kk.startswith(pref):
            tsd[kk[len(pref):]] = v
    if not tsd:
        raise RuntimeError(f"no '{pref}*' keys in {ckpt_path}; wrong ckpt layout")
    peek = [v.shape[0] for kk, v in tsd.items() if kk.endswith("parcel_embed.embed.weight")]
    if peek and int(peek[0]) != N_PARCELS:
        raise ValueError(f"ckpt parcel table {peek[0]} != expected {N_PARCELS}")
    tower = _TargetTower(n_parcels=N_PARCELS, deep_sup=True)
    missing, unexpected = tower.load_state_dict(tsd, strict=False)
    bad = [m for m in missing if "num_batches_tracked" not in m]
    if bad or unexpected:
        raise RuntimeError(f"teacher state_dict mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
    tower.eval().to(device)
    for p in tower.parameters():
        p.requires_grad_(False)
    return tower


def _load_targets(session, bt_root):
    from scripts.neuroprobe.run_pretrain_probe_suite import _label_events
    from speech_decoding.experiments.pretrain_probe_labels import build_session_targets

    subject_id, trial_id = session
    events = _label_events(subject_id, trial_id, f"btbank{subject_id}_trial{trial_id}",
                           PROBE_TASKS, bt_root, lite_cap=True)
    return build_session_targets(events, subject_id=subject_id, trial_id=trial_id)


def _window_bands(spec, starts, clip_frames):
    """Slice + robust-z every union window from the continuous 32 Hz spec caches. Returns 3
    tensors, each (n_windows, N, F_band, T32=clip_frames).

    Bulk-loads each band's survivor rows into RAM ONCE (``mm[keep]`` → ~1 GB/band), then slices
    windows from RAM and robust-z's the whole (n,N,F,T) batch in one vectorized call. The naive
    per-window ``mm[keep, :, a:b]`` did ~n·N scattered Lustre reads (2.8M for one session) and
    dominated wall-clock; this is numerically identical (mm[keep] then slice == mm[keep,:,a:b];
    the normalizer is elementwise broadcast)."""
    keep = spec.keep_idx.numpy()
    t0 = np.rint(np.asarray(starts, dtype=float) * FPS).astype(np.int64)
    end = t0 + clip_frames
    oob = np.where((t0 < 0) | (end > spec.n_frames))[0]
    if len(oob):
        raise RuntimeError(
            f"{spec.session_key}: {len(oob)} union windows out of cache bounds "
            f"(n_frames={spec.n_frames}, first bad start={float(starts[oob[0]]):.4f}s)"
        )
    bands = []
    for path, norm in zip(spec.band_paths, spec.band_norms):
        mm = np.load(path, mmap_mode="r")
        full = np.asarray(mm[keep], dtype=np.float32)              # (N, F, T_total) bulk → RAM
        del mm
        clips = np.stack([full[:, :, a:b] for a, b in zip(t0.tolist(), end.tolist())], axis=0)
        bands.append(norm.transform(torch.from_numpy(clips)))     # (n, N, F_b, T32) vectorized
    return bands                                                  # 3 × (n, N, F_b, T32)


def _canon_parcels(grid, parcel_id):
    """Canonical (grid-order) contact indices + their parcel ids + present parcel atlas ids.

    build_r4_grid lays tokens contact-major (k_full block per contact); the first token of
    each block carries that contact's index (``grid.contact``), so reshaping to (n, k_full)
    and taking column 0 recovers the n canonical contacts and their parcels."""
    k = grid.k_full
    canon = grid.contact.reshape(-1, k)[:, 0].cpu().numpy()         # (n,) contact index into N
    parcel_canon = parcel_id.cpu().numpy()[canon]                   # (n,) DKT tag per canon contact
    present = np.unique(parcel_canon)                              # sorted present atlas ids
    return canon, parcel_canon, present


def _pool_parcels(x, parcel_canon, present):
    """Pool electrodes→parcels: x (B, n, *feat) → (B, |P|, prod(feat)) flattened last dim.

    Per present parcel, MEAN over its electrodes (Ben 2026-07-16: mean is the transferable
    parcel summary; std-over-electrodes conflates with per-subject electrode count/placement,
    a CS nuisance). Returns (B, |P|, F) fp16, parcel order == present."""
    B = x.shape[0]
    blocks = []
    for p in present:
        cols = np.where(parcel_canon == p)[0]
        sub = x[:, cols]                                          # (B, |cols|, *feat)
        blocks.append(sub.mean(1).reshape(B, -1))                 # (B, prod(feat))
    return torch.stack(blocks, dim=1).to(torch.float16)           # (B, |P|, F)


def _enc0_pooled(bands, canon, parcel_canon, present):
    """enc0 input floor: per band decimate ``x[..., ::stride]`` (the model's own input frames),
    reorder to canonical contacts, pool to parcels, concat bands → (n_win, |P|, F0)."""
    from speech_decoding.models.v14_converged_v3.pack_r4 import BAND_STRIDES

    per_band = []
    for x, st in zip(bands, BAND_STRIDES):                        # x (n, N, F_b, T32)
        xd = x[..., ::st]                                         # (n, N, F_b, T_b) decimated
        xd = xd.transpose(-1, -2).contiguous()                   # (n, N, T_b, F_b) time-major
        xd = xd[:, canon]                                        # (n, n_canon, T_b, F_b)
        per_band.append(_pool_parcels(xd, parcel_canon, present))  # (n, |P|, T_b·2F_b)
    return torch.cat(per_band, dim=-1)                            # (n, |P|, F0)


@torch.no_grad()
def _encode_taps(teacher, bands, grid, parcel_packed, parcel_canon, present,
                 *, device, batch_size):
    """One forward of the teacher over all windows → per-tap parcel-pooled keep-time features.

    Cache stores the raw parcel-mean feature (n,|P|,k_full·d) — the most flexible storage: a
    readout can standardize columns on train stats (the FM linear-probe convention) or feed it
    raw (r1/M9-comparable), but neither is recoverable from a baked per-token LN. So we keep raw
    only. Returns {tap: {'raw': (n,|P|,k_full·d)}} for tap in GPU_TAPS."""
    n = bands[0].shape[0]
    k = grid.k_full
    acc = {t: [] for t in GPU_TAPS}
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        bb = [b[s:e].to(device) for b in bands]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            _, taps = teacher.forward(bb, grid, parcel_packed, tap_blocks=GPU_TAPS)
        Bb = e - s
        for t in GPU_TAPS:
            enc = taps[t].float().reshape(Bb, -1, k, taps[t].shape[-1]).cpu()  # (Bb, n, k, d)
            acc[t].append(_pool_parcels(enc, parcel_canon, present))
    return {t: {"raw": torch.cat(acc[t], 0)} for t in GPU_TAPS}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True,
                   help="3× in v3 concat order: slow, mid, hga")
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    from speech_decoding.experiments.pretrain_probe_suite import PROBE_COHORT_7
    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    if len(args.band_cache_dirs) != 3:
        raise SystemExit(f"need 3 --band-cache-dir (slow, mid, hga), got {len(args.band_cache_dirs)}")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_frames = round(CLIP_DUR_S * FPS)
    parcel_fn = make_bt_parcel_fn(args.bt_root)
    teacher = _load_teacher(args.ckpt, device=device)
    print(f"[encode-r4] tag={args.tag} device={device} gpu_taps={GPU_TAPS} + enc0", flush=True)

    for session in PROBE_COHORT_7:
        subject_id, trial_id = session
        path = os.path.join(args.out_dir, f"enc_s{subject_id}_t{trial_id}_{args.tag}.pt")
        if os.path.exists(path):
            print(f"[encode-r4] {session}: exists, skip -> {path}", flush=True)
            continue
        spec = load_v3_sessions(
            sessions=[session], band_cache_dirs=args.band_cache_dirs, span_dir=args.span_dir,
            parcel_fn=parcel_fn, lof_report_path=None, winsor=(15.0, 15.0, 20.0),
        )[0]
        targets = _load_targets(session, args.bt_root)
        bands = _window_bands(spec, targets.clip_starts, clip_frames)

        geom = spec.setup.geom.to(device)
        parcel_id = spec.setup.parcel_id.to(device)
        grid = build_r4_grid(geom, n_time=clip_frames)
        parcel_packed = parcel_id[grid.contact]
        canon, parcel_canon, present = _canon_parcels(grid, parcel_id)

        feats = {"enc0": {"raw": _enc0_pooled(bands, canon, parcel_canon, present)}}
        tap_pooled = _encode_taps(teacher, bands, grid, parcel_packed, parcel_canon,
                                  present, device=device, batch_size=args.batch_size)
        for t in GPU_TAPS:
            feats[f"enc{t}"] = tap_pooled[t]

        payload = {
            "subject_id": subject_id, "trial_id": trial_id, "ckpt_tag": args.tag,
            "present_parcels": np.asarray(present, dtype=np.int64),   # (|P|,) atlas ids, feature order
            "band_lengths": tuple(int(x) for x in grid.band_lengths),
            "feats": {k: {v: t for v, t in d.items()} for k, d in feats.items()},
            "clip_starts": np.asarray(targets.clip_starts),
            "labels": {lt: np.asarray(v) for lt, v in targets.labels.items()},
            "ws_split": targets.ws_split,
            "cs_split": targets.cs_split,
            "n_windows": int(bands[0].shape[0]),
        }
        torch.save(payload, path)
        shp = {k: tuple(next(iter(d.values())).shape) for k, d in feats.items()}
        print(f"[encode-r4] {session}: |P|={len(present)} n={payload['n_windows']} "
              f"shapes={shp} -> {path}", flush=True)
        del bands, feats, tap_pooled, payload


if __name__ == "__main__":
    main()
