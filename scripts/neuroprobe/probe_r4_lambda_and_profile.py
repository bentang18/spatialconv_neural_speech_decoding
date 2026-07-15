#!/usr/bin/env python
"""r4 (v14_converged_v3) λ-balance + secondary-cost profiler — ONE GPU run, two answers.

(1) λ BALANCE. total = jepa_loss + λ·nll_loss. The two terms are in different units (L1
    reconstruction error vs Gaussian NLL in nats), so their MAGNITUDE ratio is not the
    balance. The balance is the GRADIENT each sends into the shared encoder trunk. We
    autograd.grad each loss w.r.t. the encoder params and report:
      - raw magnitudes |jepa|, |nll|, λ·|nll|
      - ‖g_jepa‖, ‖g_nll‖ at the encoder, and the gradient ratio
      - λ* that gives JEPA-dominant balance ρ = λ·‖g_nll‖/‖g_jepa‖ for ρ∈{0.3,0.5,1.0}
    Measured at init AND after --drift-steps opt-steps (the ratio drifts as NLL settles).

(2) SECONDARY COST. torch.profiler over a few steps → top ops by CUDA time, localizing
    where r4's +0.6 s/opt-step over v3 goes (perceiver attn / cholesky_solve / state-target
    build / teacher forward). Guides which knob to turn, rather than guessing.

Eager (no --compile): autograd.grad needs a clean retained graph and the profiler wants
per-op attribution. Run on a DeltaAI ghx4 GH200:

  python -m scripts.neuroprobe.probe_r4_lambda_and_profile \
      --device cuda --precision bf16 --drift-steps 300 --profile-steps 20
"""

from __future__ import annotations

import argparse
import math

import torch

from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.objective import LAMBDA_NLL
from speech_decoding.models.v14_converged_v3.session_setup import build_session_setup
from scripts.neuroprobe.probe_v14_converged_v3_throughput import _BAND_F, _N_PARCELS, _synth_session


def _grad_norm(loss, params, retain: bool) -> float:
    gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    sq = sum(float(g.detach().float().pow(2).sum()) for g in gs if g is not None)
    return math.sqrt(sq)


def _report_balance(model, out, lam: float, tag: str) -> None:
    enc_params = [
        p for n, p in model.named_parameters()
        if "online.encoder" in n and p.requires_grad
    ]
    gj = _grad_norm(out.jepa_loss, enc_params, retain=True)
    gn = _grad_norm(out.nll_loss, enc_params, retain=True)
    j = float(out.jepa_loss.detach().float())
    n = float(out.nll_loss.detach().float())
    print(f"\n=== λ balance @ {tag} ===")
    print(f"  |jepa|={j:.4f}  |nll|={n:.4f}  λ·|nll|={lam * n:.4f}  (λ={lam})")
    print(f"  ‖g_jepa‖={gj:.5f}  ‖g_nll‖(unweighted)={gn:.5f}  ratio ‖g_nll‖/‖g_jepa‖={gn / gj:.4f}")
    print(f"  at λ={lam}: encoder-grad influence ρ = λ·‖g_nll‖/‖g_jepa‖ = {lam * gn / gj:.4f}")
    for rho in (0.3, 0.5, 1.0):
        lam_star = rho * gj / gn if gn > 0 else float("nan")
        print(f"  λ* for ρ={rho:.1f} (JEPA{'-dominant' if rho < 1 else '=secondary'}): {lam_star:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--clip-len-s", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=32)
    ap.add_argument("--n-shafts", type=int, default=10)
    ap.add_argument("--per-shaft", type=int, default=13)
    ap.add_argument("--parcels-per-shaft", type=int, default=2)
    ap.add_argument("--lambda-nll", type=float, default=LAMBDA_NLL)
    ap.add_argument("--drift-steps", type=int, default=300, help="opt-steps before the 2nd balance read")
    ap.add_argument("--profile-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=33)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    T = round(args.clip_len_s * args.fps)

    labels, parcel_id = _synth_session(args.n_shafts, args.per_shaft, args.parcels_per_shaft)
    N = len(labels)
    sm = torch.zeros(_N_PARCELS, 6)
    ss = torch.ones(_N_PARCELS, 6)
    setup = build_session_setup(labels, parcel_id, drop_labels=set(), stat_mean=sm, stat_std=ss)

    model = V3ConvergedModel(
        n_parcels=_N_PARCELS, deep_sup=True, lambda_nll=args.lambda_nll,
    ).to(device).train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=6e-3, betas=(0.9, 0.95), weight_decay=0.04,
    )

    geom = setup.geom.to(device)
    pid_dev = setup.parcel_id.to(device)
    sm_dev = setup.stat_mean.to(device)
    ss_dev = setup.stat_std.to(device)

    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.precision == "bf16"
        else torch.autocast(device_type=device.type, enabled=False)
    )

    def _bands():
        return [torch.randn(args.batch_size, N, _BAND_F[b], T, device=device) for b in range(3)]

    def _fwd(step: int):
        g = torch.Generator(device=device)
        g.manual_seed((args.seed * 1_000_003 + step) & 0x7FFF_FFFF)
        with autocast:
            return model(_bands(), geom, pid_dev, generator=g, stat_mean=sm_dev, stat_std=ss_dev)

    print(
        f"[cfg] device={device} N={N} parcels_present={int(pid_dev.unique().numel())} "
        f"B={args.batch_size} accum={args.accum} T={T} precision={args.precision} "
        f"lambda_nll={args.lambda_nll} drift_steps={args.drift_steps}"
    )

    # (1a) balance at init
    _report_balance(model, _fwd(0), args.lambda_nll, "init")

    # drift: run real opt-steps so NLL settles toward its floor, then re-measure balance
    for s in range(args.drift_steps):
        opt.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            out = _fwd(s * args.accum + micro + 1)
            (out.loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 3.0
        )
        opt.step()
        model.objective.update_teacher()
    _report_balance(model, _fwd(10 ** 7), args.lambda_nll, f"after {args.drift_steps} steps")

    # (2) profiler — where does the secondary time go
    print(f"\n=== secondary-cost profile ({args.profile_steps} steps, self CUDA time) ===")
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if device.type == "cuda":
        acts.append(ProfilerActivity.CUDA)
    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    with profile(activities=acts) as prof:
        for s in range(args.profile_steps):
            opt.zero_grad(set_to_none=True)
            for micro in range(args.accum):
                out = _fwd(2 * 10 ** 7 + s * args.accum + micro)
                (out.loss / args.accum).backward()
            opt.step()
            model.objective.update_teacher()
        if device.type == "cuda":
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
    print("[MARKER] probe exit=0")


if __name__ == "__main__":
    main()
