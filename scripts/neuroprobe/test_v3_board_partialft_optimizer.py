"""Adam's second-moment horizon must be sized to THIS run's step count, not inherited from torch.

WHY THIS FILE EXISTS. `1/(1 - beta2)` is a horizon in OPTIMIZER STEPS. This fine-tune takes ~250-330
of them end to end -- n_tr runs 1122-1750 at `--train-batch 128`, so 9-14 steps per epoch over the
~24 epochs a patience-15 run reaches, with the val-argmax landing near step 100-140. torch's default
beta2=0.999 sets a 1000-step horizon, which is 3-4x the entire run: v-hat never accumulates enough
samples to be well conditioned, and the whole fine-tune sits in the high-variance regime that LR
warmup exists to paper over (Liu et al. 2020, RAdam). beta2=0.95 is a 20-step horizon and is the
value v3 pretraining already uses.

The bug this guards is SILENT AND EXACTLY THE ONE WE SHIPPED: `AdamW(params, lr=..., weight_decay=
...)` is a perfectly ordinary-looking line that quietly selects 0.999. Nothing crashes, nothing logs,
and the pilot's lr sweep then tunes lr *given* that horizon -- so the misconfiguration hides inside a
number that looks measured. A value test alone would not catch a regression to the implicit default,
because a re-inherited 0.999 and an explicit 0.999 are indistinguishable at the parser. So the
load-bearing assertion here is STRUCTURAL: the AdamW call site must pass `betas` by name.

🔴 The default changed 0.999 -> 0.95 on 07-31. Every partial-FT number produced before that commit
is on 0.999 and is NOT comparable to one produced after it -- both A and C move. Never mix.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


BFT = _mod("v3_board_partialft")


def test_beta2_is_passed_explicitly_to_adamw():
    """THE LOAD-BEARING CHECK. Dropping `betas=` silently restores 0.999 and nothing else changes,
    so pin the call site rather than the value."""
    src = inspect.getsource(BFT._run_cell)
    assert "betas=(0.9, args.beta2)" in src, (
        "the AdamW call must pass betas by name -- omitting it silently re-inherits torch's "
        "beta2=0.999, a 1000-step horizon on a ~300-step run")


def test_beta2_default_is_sized_to_the_run_not_torchs_default():
    d = {a.dest: a.default for a in BFT._parser()._actions}
    assert d["beta2"] == 0.95, f"beta2 default must be 0.95, got {d['beta2']}"


def test_the_beta2_horizon_is_shorter_than_the_runs_step_count():
    """The invariant stated as arithmetic rather than as a pinned constant, so it survives a future
    change to batch size or patience: whatever beta2 we ship, `1/(1-beta2)` must fit INSIDE the run.

    Step count uses the SMALLEST observed session (n_tr=1122) and the epoch count a patience-15 run
    typically reaches, i.e. the least favourable case -- if the horizon fits there it fits
    everywhere. torch's 0.999 fails this by a factor of ~5."""
    d = {a.dest: a.default for a in BFT._parser()._actions}
    steps_per_epoch = -(-1122 // d["train_batch"])          # ceil, smallest board session
    run_steps = steps_per_epoch * 24                        # ~24 epochs under patience 15
    horizon = 1.0 / (1.0 - d["beta2"])
    assert horizon < run_steps, (
        f"beta2={d['beta2']} has a {horizon:.0f}-step horizon but the shortest run is only "
        f"{run_steps} steps -- v-hat would never be conditioned")
    assert 1.0 / (1.0 - 0.999) > run_steps, (
        "fixture broken: torch's default was supposed to FAIL this bound")


def test_beta2_is_reachable_so_the_old_arm_can_be_reproduced():
    """0.999 must stay expressible. The pre-07-31 numbers are not comparable, but they are still
    numbers we may need to regenerate to make the non-comparability concrete."""
    req = ["--ckpt", "x", "--regime", "ws", "--cell-index", "0", "--board-cache-dir", "x",
           "--board-tag", "x", "--band-cache-dir", "x", "--span-dir", "x", "--bt-root", "x",
           "--out", "x"]
    args = BFT._parser().parse_args(req + ["--beta2", "0.999"])
    assert args.beta2 == 0.999
    assert BFT._parser().parse_args(req).beta2 == 0.95


def test_beta1_is_not_silently_changed_alongside_beta2():
    """Only the SECOND moment is being resized. beta1 is a momentum timescale, it was never part of
    the diagnosis, and moving it too would confound the arm."""
    assert "betas=(0.9, " in inspect.getsource(BFT._run_cell)
