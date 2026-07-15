#!/usr/bin/env python
"""Loss-balance diagnostic for r4 — the ‖g_nll‖/‖g_jepa‖ gradient-norm ratio (#43).

Answers the standing launch question: how does the secondary Gaussian-NLL compete with
the primary JEPA-L1 for the SHARED online tower (stem + encoder), and does λ (default
0.2) bring λ·‖g_nll‖ into balance with ‖g_jepa‖?

Why grad-norm, not the loss-value ratio: jepa_loss is an L1 (≥0) but the Gaussian NLL is
sign-indefinite (it goes negative as the predicted variance shrinks), so a loss-VALUE
ratio is NOT a balance readout. What actually competes for the encoder is the GRADIENT
each objective puts on the shared trunk — that is what this measures, w.r.t. the online
tower's grad-carrying params.

SINGLE-PROCESS by design (Ben 2026-07-15: "single-process smoke only"): no DDP, no
torch.compile, so it is free of the r3 static_graph×grad-accum crash surface. It is a
pure diagnostic, NOT a live training callback — running it under DDP+compile is exactly
the surface that killed r3, and the number does not need per-step tracking.

Caveat printed with the result: this is a SYNTHETIC session (realistic shapes, random
weights + random |STFT| bands, self-consistent z-scored targets). It gives the ORDER OF
MAGNITUDE and whether λ is in the right ballpark; the exact in-run number would come from
a live monitor on real data (deferred — the live monitor is the DDP+compile surface we
are avoiding). We also report after a few Adam steps to see whether the ratio drifts as
the head begins to fit.

  python -m scripts.neuroprobe.probe_r4_grad_ratio --device cuda
"""

from __future__ import annotations

import argparse

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks
from speech_decoding.models.v14_converged_v3.objective import LAMBDA_NLL, V3JepaObjective
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.state_target import (
    SLOT_STRIDE,
    raw_state_vectors,
)


def _gnorm(loss: torch.Tensor, params: list[torch.Tensor], dev: torch.device) -> float:
    """‖∂loss/∂params‖₂ over the given params (allow_unused: a param the loss doesn't
    touch contributes 0, not an error)."""
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    sq = torch.zeros((), device=dev)
    for x in gs:
        if x is not None:
            sq = sq + (x.detach() ** 2).sum()
    return float(sq.sqrt().item())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-shafts", type=int, default=12)
    ap.add_argument("--per-shaft", type=int, default=10, help="contacts per shaft")
    ap.add_argument("--elec-per-parcel", type=int, default=4)
    ap.add_argument("--n-time", type=int, default=64, help="32 Hz frames (multiple of SLOW stride 8)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--warm-steps", type=int, default=20, help="Adam steps before the drift re-measure")
    ap.add_argument("--lambda-nll", type=float, default=LAMBDA_NLL)
    ap.add_argument("--seed", type=int, default=33)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    N = args.n_shafts * args.per_shaft
    T = args.n_time
    B = args.batch_size

    names = [f"S{s}-{c}" for s in range(args.n_shafts) for c in range(args.per_shaft)]
    parcel_id_cpu = torch.arange(N) // args.elec_per_parcel
    n_parcels_table = int(parcel_id_cpu.max()) + 1
    sc = build_sidecar(names, parcel_id=parcel_id_cpu)
    geom = build_l1_geometry(sc).to(dev)
    parcel_id = sc.parcel_id.to(dev)

    g = torch.Generator(device=dev).manual_seed(args.seed + 7)
    bands = [
        torch.rand(B, N, 7, T, generator=g, device=dev),
        torch.rand(B, N, 6, T, generator=g, device=dev),
        torch.rand(B, N, 7, T, generator=g, device=dev),
    ]

    obj = V3JepaObjective(n_parcels=n_parcels_table, lambda_nll=args.lambda_nll).to(dev).train()

    # Per-(parcel,6) frozen stats. The real launch supplies train-set stats; here a
    # self-consistent estimate off the synthetic target so the z-scored NLL targets are
    # ~unit-scale (the regime the head sees on real data), not artificially huge.
    with torch.no_grad():
        raw, _parcels, _n_elec = raw_state_vectors(bands, parcel_id, slot_stride=SLOT_STRIDE)
        P = raw.shape[1]
        stat_mean = raw.mean(dim=(0, 2))  # (P, 6)
        stat_std = raw.std(dim=(0, 2)).clamp_min(1e-3)  # (P, 6)

    def enc_params() -> list[torch.Tensor]:
        # the SHARED online tower both objectives pull on (stem + encoder), grad-carrying.
        return [p for p in obj.online.parameters() if p.requires_grad]

    def measure(tag: str) -> None:
        gen = torch.Generator(device=dev).manual_seed(1234)
        masks = sample_masks(geom, N, n_time=T, n_rows=B, generator=gen)
        out = obj(bands, geom, parcel_id, masks, stat_mean=stat_mean, stat_std=stat_std)
        assert out.jepa_loss is not None and out.nll_loss is not None
        jl = float(out.jepa_loss.detach())
        nl = float(out.nll_loss.detach())
        params = enc_params()
        gj = _gnorm(out.jepa_loss, params, dev)
        gn = _gnorm(out.nll_loss, params, dev)  # UNWEIGHTED nll grad
        lam = args.lambda_nll
        ratio = gn / gj if gj > 0 else float("nan")
        bal = lam * ratio  # λ·‖g_nll‖ / ‖g_jepa‖ — the effective balance on the trunk
        state = (
            "BALANCED" if 0.33 <= bal <= 3.0
            else "secondary DOMINATES" if bal > 3.0
            else "secondary NEGLIGIBLE"
        )
        print(
            f"[{tag}] jepa_L1={jl:+.4f} nll={nl:+.4f} | "
            f"‖g_jepa‖={gj:.4e} ‖g_nll‖={gn:.4e} | "
            f"ratio ‖g_nll‖/‖g_jepa‖={ratio:.3f} | λ={lam} → λ·ratio={bal:.3f} ({state})"
        )

    print(
        f"[setup] N={N} elec, {args.n_shafts} shafts, P={P} parcels, T={T}, B={B}, "
        f"λ={args.lambda_nll} device={dev} (SYNTHETIC at-init session — order-of-magnitude only)"
    )
    measure("init")

    # a few Adam steps on the TOTAL loss, then re-measure — does the balance drift as the
    # head begins to fit (NLL can fall / go negative)?
    opt = torch.optim.AdamW([p for p in obj.parameters() if p.requires_grad], lr=6e-3)
    for step in range(args.warm_steps):
        gen = torch.Generator(device=dev).manual_seed(step)
        masks = sample_masks(geom, N, n_time=T, n_rows=B, generator=gen)
        out = obj(bands, geom, parcel_id, masks, stat_mean=stat_mean, stat_std=stat_std)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        obj.update_teacher()
    measure(f"after-{args.warm_steps}-steps")
    print("[done]")


if __name__ == "__main__":
    main()
