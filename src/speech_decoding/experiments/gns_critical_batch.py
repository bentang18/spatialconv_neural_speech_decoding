"""Gradient noise scale → critical batch size for the converged-arch SSL step
(Ben 2026-06-22).

Measures whether the production eff-128 batch is below, at, or above the
*critical batch size* ``B_crit`` — the batch beyond which more samples/step buy
diminishing reductions in steps-to-target (McCandlish et al. 2018, "An Empirical
Model of Large-Batch Training"). The estimator is the paper's *simple* form:

    E‖g_B‖² = ‖G‖² + tr(Σ)/B          (linear in 1/B)
    B_crit  = tr(Σ) / ‖G‖²

``‖G‖²`` is the true (full-data) gradient norm² (the 1/B→0 intercept) and
``tr(Σ)`` the per-example gradient-covariance trace (the slope). Two batch sizes
pin the line; we also fit the whole ``E‖g_B‖²`` vs ``1/B`` curve for robustness.

**Faithfulness.** The "examples" are real production micro-batches drawn from the
SAME ``group_by_session`` sampler the run uses, pushed through the REAL ``_step``
(fresh tube masks per forward + EMA-teacher targets), so the measured ``B_crit``
is the one that governs the actual run — NOT an IID-clip idealization. Because a
single micro-batch is one session and the accumulated big batch is several
DIFFERENT sessions, the ``b → n_accum·b`` axis is exactly the single-session-vs-
multi-session gradient-variance gap (which also informs the same-session-per-step
throughput question): a large gap ⇒ cross-session variance dominates ⇒ eff-128 is
genuinely below ``B_crit`` and single-session steps would be noisy; a small gap ⇒
eff-128 is already past the knee and bigger batch (or more diversity) buys little.

The pure-math core (:func:`gradient_noise_scale`, :func:`fit_gns_curve`) is
laptop-TDD'd (``test_gns_critical_batch``); the model forward/backward driver
(:func:`run_gns_probe`) is DCC/DeltaAI-only (needs BT voltage + a GPU).
"""

from __future__ import annotations

import typing as tp

import torch
from torch import Tensor

__all__ = [
    "gradient_noise_scale",
    "fit_gns_curve",
    "flatten_grads",
    "run_gns_probe",
]


def gradient_noise_scale(
    e_small: float, b_small: int, e_big: float, b_big: int
) -> dict[str, float]:
    """Two-point McCandlish simple estimator.

    ``e_small``/``e_big`` are E‖g_B‖² at batch sizes ``b_small`` < ``b_big``.
    Returns ``g_sq`` (‖G‖², the intercept), ``tr_sigma`` (tr(Σ), the slope), and
    ``b_crit`` = tr(Σ)/‖G‖². ‖G‖² is a small difference of large noisy numbers,
    so average ``e_*`` over many rounds before calling this (the driver does).
    """
    if b_big <= b_small:
        raise ValueError(f"need b_small < b_big, got {b_small}, {b_big}")
    g_sq = (b_big * e_big - b_small * e_small) / (b_big - b_small)
    tr_sigma = (e_small - e_big) / (1.0 / b_small - 1.0 / b_big)
    b_crit = tr_sigma / g_sq if g_sq != 0.0 else float("inf")
    return {"g_sq": g_sq, "tr_sigma": tr_sigma, "b_crit": b_crit}


def fit_gns_curve(
    batch_sizes: tp.Sequence[int], e_of_b: tp.Sequence[float]
) -> dict[str, float]:
    """Least-squares fit of E‖g_B‖² = ‖G‖² + tr(Σ)·(1/B) over ALL measured B.

    More robust than the 2-point estimate when ≥3 batch sizes are available
    (the curve's intercept/slope use every point, damping the ‖G‖² noise).
    Returns the same keys as :func:`gradient_noise_scale` plus ``r2``.
    """
    x = torch.tensor([1.0 / b for b in batch_sizes], dtype=torch.float64)
    y = torch.tensor(list(e_of_b), dtype=torch.float64)
    n = x.numel()
    if n < 2:
        raise ValueError("need >=2 batch sizes for a fit")
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    slope = ((x - xm) * (y - ym)).sum() / sxx  # tr(Sigma)
    intercept = ym - slope * xm  # ||G||^2
    yhat = intercept + slope * x
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - ym) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
    g_sq = float(intercept)
    tr_sigma = float(slope)
    b_crit = tr_sigma / g_sq if g_sq != 0.0 else float("inf")
    return {"g_sq": g_sq, "tr_sigma": tr_sigma, "b_crit": b_crit, "r2": r2}


def flatten_grads(params: tp.Iterable[torch.nn.Parameter]) -> Tensor:
    """1-D fp32 concat of all populated ``.grad`` (trainable params, grad set).

    The grad vector lives in fixed param-space, so it is comparable across
    micro-batches of DIFFERENT sessions (different electrode count C) — exactly
    what lets us accumulate single-session grads into a multi-session big-batch
    grad. fp32 so the running-sum norm is numerics-clean even under a bf16 run.
    """
    flat = [p.grad.reshape(-1).float() for p in params if p.grad is not None]
    if not flat:
        raise RuntimeError("no gradients populated — did backward() run?")
    return torch.cat(flat)


def run_gns_probe(  # pragma: no cover - DCC/DeltaAI-only driver
    xp: tp.Any,
    *,
    out_path: str,
    ckpt_path: str | None = None,
    micro_batch: int | None = None,
    n_accum: int = 8,
    rounds: int = 64,
    device: torch.device | None = None,
) -> dict[str, tp.Any]:
    """Build the real converged module + train loader, optionally warm-start from
    ``ckpt_path`` (full Lightning state → student AND EMA teacher restored), then
    estimate ``B_crit`` from ``rounds`` accumulations of ``n_accum`` single-session
    micro-batches. Writes ``out_path`` (json) incrementally. fp32 (autocast off)
    so the measured noise is the data/model/masking GNS, not bf16 kernel jitter
    (which only ADDS to it). Returns the result dict."""
    import json
    import time

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gns] device={device} ckpt={ckpt_path}")

    loaders = xp.data.build(worker_seed=xp.seed)
    train_loader = loaders["train"]
    module = xp._build_brain_module(train_loader).to(device)
    module.train()  # masking + EMA-target path active; we only read grads

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        incompat = module.load_state_dict(state, strict=False)
        own = set(module.state_dict().keys())
        real_missing = [k for k in incompat.missing_keys if k in own]
        # non-persistent geometry buffers are rebuilt at init → expected gaps
        if real_missing:
            raise RuntimeError(
                f"ckpt missing persistent params {real_missing[:8]} "
                f"({len(real_missing)} total) — not weight-complete."
            )
        print(f"[gns] loaded ckpt ({len(incompat.unexpected_keys)} unexpected keys ignored).")

    b = micro_batch if micro_batch is not None else int(xp.data.batch_size)
    sizes = [k * b for k in range(1, n_accum + 1)]
    # running E‖g_B‖² accumulators, one per k=1..n_accum (B=k*b)
    sum_e = [0.0] * n_accum
    n_done = 0

    it = iter(train_loader)

    def _next_batch() -> tp.Any:
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(train_loader)
            return next(it)

    t0 = time.perf_counter()
    for r in range(rounds):
        running: Tensor | None = None  # Σ of per-micro grad vectors
        for k in range(n_accum):
            batch = _next_batch()
            data = {
                key: (v.to(device) if torch.is_tensor(v) else v)
                for key, v in batch.data.items()
            }
            module.zero_grad(set_to_none=True)
            out = module._step(data)
            out["loss"].backward()
            g = flatten_grads(module.parameters())
            running = g.clone() if running is None else running + g
            mean_g = running / (k + 1)  # grad of the size-(k+1)*b big batch
            sum_e[k] += float(mean_g.dot(mean_g))  # one sample of E‖g_{(k+1)b}‖²
        n_done += 1
        if (r + 1) % max(1, rounds // 8) == 0:
            e_now = [s / n_done for s in sum_e]
            two = gradient_noise_scale(e_now[0], sizes[0], e_now[-1], sizes[-1])
            fit = fit_gns_curve(sizes, e_now)
            print(
                f"[gns] round {r + 1}/{rounds}  B_crit(2pt)={two['b_crit']:.0f} "
                f"B_crit(fit)={fit['b_crit']:.0f} r2={fit['r2']:.3f} "
                f"({time.perf_counter() - t0:.0f}s)"
            )

    e_final = [s / n_done for s in sum_e]
    two = gradient_noise_scale(e_final[0], sizes[0], e_final[-1], sizes[-1])
    fit = fit_gns_curve(sizes, e_final)
    eff_batch = b * 4  # 4-GPU eff-128 at micro-batch b (world_size 4)
    result: dict[str, tp.Any] = {
        "ckpt": ckpt_path,
        "micro_batch": b,
        "n_accum": n_accum,
        "rounds": n_done,
        "batch_sizes": sizes,
        "e_of_b": e_final,
        "two_point": two,
        "curve_fit": fit,
        "eff_batch_4gpu": eff_batch,
        "b_crit_vs_eff": fit["b_crit"] / eff_batch if eff_batch else None,
        "verdict": (
            "eff-batch ABOVE B_crit (diminishing returns — bigger batch wastes FLOPs)"
            if fit["b_crit"] < eff_batch
            else "eff-batch BELOW B_crit (more batch still buys near-linear speedup)"
        ),
    }
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(
        f"[gns] DONE  B_crit(fit)={fit['b_crit']:.0f} vs eff-{eff_batch}  "
        f"→ {result['verdict']}  (wrote {out_path})"
    )
    return result
