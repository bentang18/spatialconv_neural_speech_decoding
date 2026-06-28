#!/usr/bin/env python
"""Throughput probe for the converged-v2 SSL assembly (P3.2).

Times forward(+backward+ema) of :class:`V14ConvergedV2` on a SYNTHETIC homogeneous
session at the representative production shape (impl-plan memo: B32 C120 P16 k2
d256 N150@5s). No cache / no Experiment needed — this is the model-only sec/step
gate Ben wants <1 s on a GH200 before any long run. Run it on DeltaAI after the
cache port; locally it validates the model runs end-to-end at production scale.

Mask SAMPLING (the separately-tracked ~455 ms CPU bubble) is excluded from the
per-step timing — masks are session-constant in shape and sampled once here; the
bubble is an orthogonal dataloader-overlap concern, not model compute. The number
this reports is pure frontend→pool→latent→predictors fwd(+bwd) time.

Example (GH200):
  python -m scripts.neuroprobe.probe_v14_converged_v2_throughput \
      --steps 500 --device cuda --compile
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    V14ConvergedV2Config,
    active_parcels,
    bands_for_clip_len,
    compute_static_shapes_v2,
)


def _parcel_of_electrode(n_elec: int, n_active: int, n_parcels: int) -> torch.Tensor:
    """Assign ``n_elec`` electrodes to ``n_active`` distinct DKT labels (< n_parcels)
    as evenly as possible — a realistic ragged ``n_p`` (e.g. 120/16 ≈ 7-8 each)."""
    if n_active > n_parcels:
        raise ValueError(f"n_active {n_active} > n_parcels {n_parcels}")
    labels = torch.linspace(0, n_parcels - 1, n_active).round().long()
    if labels.unique().numel() != n_active:  # collisions if n_active near n_parcels
        labels = torch.arange(n_active)
    return labels[torch.arange(n_elec) % n_active]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--frontend-layers", type=int, default=6)
    ap.add_argument("--latent-layers", type=int, default=8)
    ap.add_argument("--m2-pred-layers", type=int, default=3)
    ap.add_argument("--m4-pred-layers", type=int, default=6)
    ap.add_argument("--pred-dim", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-parcels", type=int, default=62, help="DKT embedding table size")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--electrodes", type=int, default=120)
    ap.add_argument("--active-parcels", type=int, default=16)
    ap.add_argument("--clip-len-s", type=float, default=5.0)
    ap.add_argument("--tube-ratio", type=float, default=0.25)
    ap.add_argument("--untie-lfs", action="store_true", help="n_op=4 ablation")
    ap.add_argument("--m3-pred-layers", type=int, default=None,
                    help="Run-A master switch: M3 pool-inpaint head depth "
                         "(None => legacy 2-head assembly)")
    ap.add_argument("--qk-norm", action="store_true",
                    help="per-head QK-norm on all towers (Run-A)")
    ap.add_argument("--support-weight", action="store_true",
                    help="Run-A: electrode-count weighted M3/M4 per-head loss")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-backward", action="store_true", help="forward-only timing")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="autocast bf16 (GH200)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    device = torch.device(dev)
    torch.manual_seed(args.seed)

    cfg = V14ConvergedV2Config(
        d_model=args.d_model,
        n_heads=args.n_heads,
        frontend_layers=args.frontend_layers,
        latent_layers=args.latent_layers,
        m2_pred_layers=args.m2_pred_layers,
        m4_pred_layers=args.m4_pred_layers,
        pred_dim=args.pred_dim,
        n_parcels=args.n_parcels,
        k=args.k,
        tube_ratio=args.tube_ratio,
        tie_lfs=not args.untie_lfs,
        m3_pred_layers=args.m3_pred_layers,
        qk_norm=args.qk_norm,
        support_weight=args.support_weight,
    )
    model = V14ConvergedV2(cfg).to(device)
    model.train(not args.no_backward)

    bands = bands_for_clip_len(args.clip_len_s)
    lfs_b, hga_b = bands
    B, C = args.batch, args.electrodes
    poe = _parcel_of_electrode(C, args.active_parcels, args.n_parcels).to(device)
    _, membership = active_parcels(poe)

    lfs = torch.randn(B, C, lfs_b.n_freq_bins, lfs_b.n_time_frames, device=device)
    hga = torch.randn(B, C, hga_b.n_freq_bins, hga_b.n_time_frames, device=device)

    gen = torch.Generator().manual_seed(args.seed + 1)
    m2_mask, tube_mask = model.sample_masks(B, membership.cpu(), bands, gen)
    m2_mask, tube_mask = m2_mask.to(device), tube_mask.to(device)
    sh = compute_static_shapes_v2(m2_mask, tube_mask, membership, bands, k=cfg.k)

    fwd = model
    if args.compile:
        fwd = torch.compile(model)

    opt = None if args.no_backward else torch.optim.AdamW(model.parameters(), lr=1e-4)

    def one_step() -> None:
        ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if args.bf16
            else torch.autocast(device_type=device.type, enabled=False)
        )
        with ctx:
            out = fwd(lfs, hga, poe, m2_mask, tube_mask, clip_len_s=args.clip_len_s)
            loss = out["loss"]
        if not args.no_backward:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.ema_step()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"device={device} | params(trainable)={n_params/1e6:.1f}M | "
        f"compile={args.compile} bf16={args.bf16} backward={not args.no_backward}"
    )
    print(
        f"static header: B={sh.b} C={sh.c} P={sh.P} k={sh.k} S={sh.S} "
        f"n_mask={sh.n_mask} s_vis={sh.s_vis} n_tube={sh.n_tube} P_vis={sh.P_vis}"
    )
    if cfg.run_a:
        q_line = f"m2_q={sh.m2_q} m3_q={sh.m3_q} m4_q_tubed={sh.m4_q_tubed}"
    else:
        q_line = f"m2_q={sh.m2_q} m4_q={sh.m4_q}"
    print(
        f"  latent_student={sh.latent_student} latent_teacher={sh.latent_teacher} "
        f"{q_line}"
    )

    for _ in range(args.warmup):
        one_step()
    _sync(device)

    times: list[float] = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        one_step()
        _sync(device)
        times.append(time.perf_counter() - t0)

    times.sort()
    med = statistics.median(times)
    p10 = times[max(0, int(0.10 * len(times)) - 1)]
    p90 = times[min(len(times) - 1, int(0.90 * len(times)))]
    print(
        f"sec/step: median={med:.3f}  p10={p10:.3f}  p90={p90:.3f}  "
        f"(n={args.steps}, CV={statistics.pstdev(times)/med*100:.0f}%)"
    )
    print(f"  -> {'<1s PASS' if med < 1.0 else '>=1s'} | eff-batch effective B={B}")


if __name__ == "__main__":
    main()
