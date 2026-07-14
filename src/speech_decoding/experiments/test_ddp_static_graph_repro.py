"""CPU/gloo repro of the r3 killer: DDP ``static_graph=True`` × the first backward.

r3 (`v3_pretrain_r3`, DeltaAI 2653154, 2026-07-14) died at the FIRST backward on all
4 ranks with ``expect_autograd_hooks_ INTERNAL ASSERT FAILED (c10d/reducer.cpp:1703)``
— 0 usable steps. The knowledge that ``static_graph=False`` survives lived only in a
``dispatch_v3`` code comment; this file is that repro, committed and executable: 2
ranks, gloo, no GPU, a ~2k-param stand-in for the v3 model's STRUCTURE (online tower +
no-grad EMA teacher + predictor + an OPTIONAL head whose loss is scaled by a λ that may
be a 0-valued 0-d tensor), never the 11.95M-param model itself (it needs caches we do
not have on a laptop).

WHAT THIS FILE PINS — and what it does NOT:

  * ``static_graph=False`` survives EVERY configuration tested (accum or not, optional
    head live or frozen). That is the ONLY claim the r3 gate rests on, and it is the
    claim the fix (``--no-ddp-static-graph``) depends on.
  * ``static_graph=True`` raises THE r3 ASSERT — but on this torch (2.10) the trigger is
    GRADIENT ACCUMULATION, not the optional head: it fires identically with the head
    FROZEN (the r1/r2 config). Under ``accumulate_grad_batches>1`` Lightning wraps every
    non-final micro-batch in ``DDP.no_sync()``
    (``loops/optimization/automatic.py:184`` → ``strategies/parallel.py:117``), so the
    run's FIRST forward/backward happens with ``require_backward_grad_sync=False``:
    ``_post_forward`` then skips ``reducer.prepare_for_backward()`` (leaving
    ``expect_autograd_hooks_`` False) while STILL installing ``_DDPSink``, whose backward
    enqueues ``reducer._delay_all_reduce`` → ``finalize_backward()`` →
    ``TORCH_INTERNAL_ASSERT(expect_autograd_hooks_)`` blows.
  * ⇒ this repro does NOT reproduce r3's *discriminator*. r1 ran ~43k steps on DeltaAI
    with static_graph=True, devices=4, accum=4 — i.e. DeltaAI's torch does not take this
    path (older torch gates the ``_DDPSink`` install on ``num_iterations == 1``, which
    only advances on a SYNCED forward). The exact ingredient that flips the assert on is
    therefore torch-version-dependent; ``static_graph=True`` is the fragile
    ingredient common to every sighting, and ``static_graph=False`` is the invariant that
    survives. Do not read these tests as "the context head is innocent on DeltaAI" — read
    them as "static_graph is a loaded gun; the gate is the safety".
"""
from __future__ import annotations

import contextlib
import copy
import os
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

_D = 8  # feature width; the structure is the subject of the test, not the size
_WORLD = 2


class _MiniJepa(nn.Module):
    """The STRUCTURE of ``V3JepaObjective`` in ~2k params: an online tower, its no-grad
    EMA deepcopy, a predictor, ``pred_to_target``, and the OPTIONAL
    ``pred_to_target_context`` head gated by ``_static_off`` (a TYPE test — a 0-valued
    0-d tensor keeps the head graph-connected, a python 0.0 folds it away)."""

    def __init__(self) -> None:
        super().__init__()
        self.online = nn.Sequential(nn.Linear(_D, _D), nn.GELU(), nn.Linear(_D, _D))
        self.teacher = copy.deepcopy(self.online)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(_D, _D), nn.GELU(), nn.Linear(_D, _D))
        self.pred_to_target = nn.Linear(_D, _D)
        self.pred_to_target_context = nn.Linear(_D, _D)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, _D))

    def forward(
        self, x: Tensor, cell_masked: Tensor, lambda_context: float | Tensor
    ) -> Tensor:
        z = self.online(x)
        with torch.no_grad():
            tgt = F.layer_norm(self.teacher(x), (x.shape[-1],))
        h = self.predictor(torch.where(cell_masked[..., None], self.mask_token, z))
        pred = self.pred_to_target(h)
        w = cell_masked.to(pred.dtype)
        ae = (pred - tgt).abs().mean(-1)
        loss = (ae * w).sum() / w.sum().clamp(min=1.0)
        static_off = isinstance(lambda_context, (int, float)) and lambda_context == 0.0
        if not static_off:
            ctx = self.pred_to_target_context(h)
            vis = ~cell_masked
            loss = loss + lambda_context * (ctx[vis] - tgt[vis]).abs().mean()
        return loss

    @torch.no_grad()
    def update_teacher(self) -> None:
        for t, s in zip(self.teacher.parameters(), self.online.parameters()):
            t.mul_(0.99925).add_(s, alpha=1.0 - 0.99925)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _worker(
    rank: int,
    port: int,
    static_graph: bool,
    head_trainable: bool,
    accum: int,
    out: dict,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD)
    try:
        torch.manual_seed(0)
        model = _MiniJepa()
        if not head_trainable:
            # r1/r2: V14ConvergedV3Module freezes the context head when hold == 0.0.
            for p in model.pred_to_target_context.parameters():
                p.requires_grad_(False)
        ddp = DDP(model, find_unused_parameters=False, static_graph=static_graph)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        gen = torch.Generator().manual_seed(rank + 1)
        for _ in range(3):  # 3 optimizer steps
            for micro in range(accum):
                x = torch.randn(2, 5, _D, generator=gen)
                cell_masked = torch.rand(2, 5, generator=gen) < 0.5
                # head trainable ⇒ the module's on-schedule λ: a 0-d TENSOR (value 0
                # pre-warmup) that keeps the head graph-connected. Frozen ⇒ python 0.0.
                lam: float | Tensor = (
                    torch.tensor(0.0) if head_trainable else 0.0
                )
                # Lightning blocks the DDP sync on every non-final accumulation
                # micro-batch (loops/optimization/automatic.py:184).
                sync = micro == accum - 1
                with contextlib.nullcontext() if sync else ddp.no_sync():
                    (ddp(x, cell_masked, lam) / accum).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            model.update_teacher()
        out[rank] = "OK"
    except Exception:  # noqa: BLE001 — the failure mode IS the observable
        out[rank] = traceback.format_exc()
    finally:
        dist.destroy_process_group()


def _run(*, static_graph: bool, head_trainable: bool, accum: int) -> list[str]:
    """Spawn 2 gloo ranks; return each rank's outcome ("OK" or its traceback)."""
    import torch.multiprocessing as mp

    with mp.Manager() as mgr:
        out = mgr.dict()
        with contextlib.suppress(Exception):  # a child assert re-raises here
            mp.spawn(
                _worker,
                args=(_free_port(), static_graph, head_trainable, accum, out),
                nprocs=_WORLD,
                join=True,
            )
        res = dict(out)
    assert len(res) == _WORLD, f"a rank produced no outcome: {res}"
    return [res[r] for r in range(_WORLD)]


_ASSERT = "expect_autograd_hooks_"


@pytest.mark.parametrize("accum", [1, 4])
@pytest.mark.parametrize("head_trainable", [True, False])
def test_static_graph_false_survives_every_configuration(
    accum: int, head_trainable: bool
) -> None:
    """THE claim the r3 gate rests on: ``static_graph=False`` completes 3 optimizer
    steps clean with the optional head live (r3) or frozen (r1/r2), with and without
    Lightning-style accumulation. This is why ``--no-ddp-static-graph`` is the fix."""
    for rank_out in _run(
        static_graph=False, head_trainable=head_trainable, accum=accum
    ):
        assert rank_out == "OK", rank_out


def test_static_graph_true_asserts_at_first_backward_under_accumulation() -> None:
    """static_graph=True + Lightning's no_sync accumulation ⇒ the r3 assert, on EVERY
    rank, at the first backward. Same signature as the crash that killed r3."""
    outs = _run(static_graph=True, head_trainable=True, accum=4)
    for rank_out in outs:
        assert _ASSERT in rank_out, rank_out
        assert "reducer.cpp" in rank_out


def test_static_graph_true_asserts_with_the_optional_head_frozen_too() -> None:
    """NEGATIVE RESULT, pinned so nobody re-derives the wrong story: with the context
    head FROZEN (the exact r1/r2 config) static_graph=True raises the IDENTICAL assert
    on this torch. The optional head is therefore NOT the discriminator here — the
    trigger is static_graph × the un-synced first backward. See the module docstring."""
    outs = _run(static_graph=True, head_trainable=False, accum=4)
    for rank_out in outs:
        assert _ASSERT in rank_out, rank_out


def test_static_graph_true_survives_without_accumulation() -> None:
    """Isolates the trigger on this torch: with accum=1 (every backward synced)
    static_graph=True is fine even with the optional head live. So the minimal model
    does NOT reproduce r3's *differentiator* (the head joining the trainable set) —
    it reproduces the assert by the accumulation route."""
    for rank_out in _run(static_graph=True, head_trainable=True, accum=1):
        assert rank_out == "OK", rank_out
