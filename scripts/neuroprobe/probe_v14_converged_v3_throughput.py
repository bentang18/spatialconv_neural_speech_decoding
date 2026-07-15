#!/usr/bin/env python
"""Throughput probe for the r4 (v14_converged_v3) SSL assembly.

Times the FULL optimizer step — accum micro-batches of fwd + bwd, then optimizer.step
+ EMA teacher advance — of :class:`V3ConvergedModel` on ONE synthetic session-homogeneous
batch at a representative production shape. This is the model-only s/opt-step gate: v3
d256 measured 1.080 s/opt-step launch-bound on a GH200; r4 adds the secondary write-only
Perceiver Gaussian-NLL, and this probe measures the delta.

The secondary is ON by default (synthetic per-parcel frozen stats activate it, exactly as
the launch config's --state-stats-dir would). --no-secondary measures the JEPA-only
baseline for an A/B on the head's cost.

Mask sampling happens INSIDE forward (per-step generator), so it is included in the timing
— that is the real per-step cost. Single-device; run on a DeltaAI ghx4 GH200:

  python -m scripts.neuroprobe.probe_v14_converged_v3_throughput \
      --device cuda --steps 300 --warmup 50 --precision bf16 --compile
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.objective import LAMBDA_NLL
from speech_decoding.models.v14_converged_v3.session_setup import build_session_setup

# input |STFT| bin counts per band (SLOW, MID, HGA) at the 32 Hz clock — the stem
# decimates internally to 4/16/32 Hz (per-band strides 8/2/1), so the INPUT T is the
# shared 32 Hz length for every band.
_BAND_F = (7, 6, 7)
_N_PARCELS = 75  # global DKT table (74 + reserved unknown), matches dispatch_v3._n_parcels


def _synth_session(n_shafts: int, per_shaft: int, parcels_per_shaft: int):
    """A realistic ragged montage: ``n_shafts`` shafts of ``per_shaft`` contacts, each
    shaft split into ``parcels_per_shaft`` DKT parcels (median BT ≈ 6 elec/parcel). Returns
    (labels, parcel_id) in the FULL voltage order build_session_setup consumes."""
    labels: list[str] = []
    parcels: list[int] = []
    pid = 0
    for s in range(n_shafts):
        pre = f"L{chr(65 + s % 26)}{s // 26}"
        for c in range(1, per_shaft + 1):
            labels.append(f"{pre}-{c}")
            # split the shaft's contacts into contiguous parcel runs
            parcels.append(pid + (c - 1) * parcels_per_shaft // per_shaft)
        pid += parcels_per_shaft
    return labels, torch.tensor(parcels, dtype=torch.long)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=200, help="timed opt-steps (post-warmup)")
    ap.add_argument("--warmup", type=int, default=30, help="untimed warmup opt-steps")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4, help="grad-accum micro-batches per opt-step")
    ap.add_argument("--clip-len-s", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=32, help="shared clock (Hz)")
    ap.add_argument("--n-shafts", type=int, default=10)
    ap.add_argument("--per-shaft", type=int, default=13, help="contacts per shaft (~130 total)")
    ap.add_argument("--parcels-per-shaft", type=int, default=2)
    ap.add_argument("--lambda-nll", type=float, default=LAMBDA_NLL)
    ap.add_argument("--secondary", action=argparse.BooleanOptionalAction, default=True,
                    help="activate the secondary Gaussian-NLL (default ON = the r4 config)")
    ap.add_argument("--deep-sup", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--compile", action="store_true", help="torch.compile(dynamic=False)")
    ap.add_argument("--seed", type=int, default=33)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    T = round(args.clip_len_s * args.fps)

    labels, parcel_id = _synth_session(args.n_shafts, args.per_shaft, args.parcels_per_shaft)
    N = len(labels)
    # by-parcel-VALUE frozen stats (n_parcels, 6): mean 0 / std 1 ⇒ z-score is identity, so
    # the synthetic target stays finite. build_session_setup gathers the present parcels.
    if args.secondary:
        sm = torch.zeros(_N_PARCELS, 6)
        ss = torch.ones(_N_PARCELS, 6)
    else:
        sm = ss = None
    setup = build_session_setup(labels, parcel_id, drop_labels=set(), stat_mean=sm, stat_std=ss)

    model = V3ConvergedModel(
        n_parcels=_N_PARCELS, deep_sup=args.deep_sup, lambda_nll=args.lambda_nll,
    ).to(device).train()
    if not args.secondary and model.objective.perceiver is not None:
        for p in model.objective.perceiver.parameters():  # mirror the module's OFF freeze
            p.requires_grad_(False)
    if args.compile:
        import torch._dynamo as _dynamo
        _dynamo.config.cache_size_limit = 256
        model = torch.compile(model, dynamic=False)  # type: ignore[assignment]

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=6e-3, betas=(0.9, 0.95), weight_decay=0.04,
    )

    geom = setup.geom.to(device)
    pid_dev = setup.parcel_id.to(device)
    sm_dev = None if setup.stat_mean is None else setup.stat_mean.to(device)
    ss_dev = None if setup.stat_std is None else setup.stat_std.to(device)

    def _bands():
        return [
            torch.randn(args.batch_size, N, _BAND_F[b], T, device=device)
            for b in range(3)
        ]

    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.precision == "bf16"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    def _opt_step(step: int) -> None:
        opt.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            g = torch.Generator(device=device)
            g.manual_seed((args.seed * 1_000_003 + step * args.accum + micro) & 0x7FFF_FFFF)
            with autocast:
                out = model(
                    _bands(), geom, pid_dev, generator=g,
                    stat_mean=sm_dev, stat_std=ss_dev,
                )
                (out.loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3.0)
        opt.step()
        model.objective.update_teacher()

    print(
        f"[cfg] device={device} N={N} parcels_present={int(pid_dev.unique().numel())} "
        f"B={args.batch_size} accum={args.accum} T={T} secondary={args.secondary} "
        f"deep_sup={args.deep_sup} lambda_nll={args.lambda_nll} precision={args.precision} "
        f"compile={args.compile}"
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cfg] trainable params = {n_trainable/1e6:.2f}M")

    for s in range(args.warmup):
        _opt_step(s)
    _sync(device)

    times: list[float] = []
    for s in range(args.steps):
        _sync(device)
        t0 = time.perf_counter()
        _opt_step(args.warmup + s)
        _sync(device)
        times.append(time.perf_counter() - t0)

    times.sort()
    mean = statistics.fmean(times)
    median = statistics.median(times)
    p90 = times[int(0.9 * (len(times) - 1))]
    tokens = args.batch_size * args.accum * N * T  # per opt-step, all bands share N×T grid
    print(
        f"[result] s/opt-step mean={mean:.4f} median={median:.4f} p90={p90:.4f} "
        f"min={times[0]:.4f} (n={len(times)}, accum={args.accum})"
    )
    print(f"[result] ~{tokens/median/1e3:.1f}k (contact·frame)/s at the median opt-step")


if __name__ == "__main__":
    main()
