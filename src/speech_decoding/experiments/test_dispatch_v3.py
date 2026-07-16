"""v14_converged_v3 launcher — E1/E2 assembly + resume/gate/forensics wiring (TDD).

Mostly cheap, pure-Python contract tests over the launcher assembly (E1), the
trainer knobs (E2), the r3 static_graph launch gate, and the r4 heartbeat.

The three standalone CPU smoke tests that ran ``trainer.fit`` on the real ~12M-param
model (F1) were removed 2026-07-16 (Ben): they pinned the whole laptop for minutes on
a bare ``pytest``. End-to-end coverage did NOT go with them —
``test_module_resumes_model_and_step_from_checkpoint`` still drives synthetic band
caches ─▶ load_v3_sessions ─▶ build_v3_training ─▶ fit, so a real V3Batch still goes
through model + objective + optimizer + EMA here. What IS gone: the only test of
``main()``'s argv glue.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

from speech_decoding.experiments.dispatch_v3 import (
    _build_trainer,
    _parse_sessions,
    _StepTimeCallback,
    build_v3_optim_cfg,
    build_v3_training,
)
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

_BAND_F = (7, 6, 7)


def _shaft_labels(shaft_sizes):
    labels = []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
    return labels


def _stub_parcel_fn(subject_id, trial_id, labels):
    prefixes: dict[str, int] = {}
    pid = []
    for lab in labels:
        pre = lab.rstrip("0123456789")
        prefixes.setdefault(pre, len(prefixes))
        pid.append(prefixes[pre])
    return torch.tensor(pid, dtype=torch.long)


def _write_caches(tmp_path, sessions, *, n_frames=400):
    band_dirs = []
    for b in range(3):
        d = tmp_path / f"band{b}"
        d.mkdir(parents=True, exist_ok=True)
        F = _BAND_F[b]
        for subject_id, trial_id, labels in sessions:
            stem = f"btbank{subject_id}_t{trial_id}"
            key = (
                '{"cls":"Wang2024Treebank","method":"_load_raw","timeline":'
                f'{{"extra_bad":[],"subject":"btbank{subject_id}",'
                f'"subject_id":{subject_id},"trial_id":{trial_id}}}}}_0.000_6867.860'
            )
            C = len(labels)
            (d / f"{stem}.json").write_text(json.dumps({
                "key": key, "ch_names": labels, "total_frames": n_frames,
                "sample_rate": 32,
            }))
            rng = np.random.default_rng(subject_id * 100 + trial_id + b)
            np.save(str(d / f"{stem}.npy"),
                    rng.standard_normal((C, F, n_frames)).astype(np.float32))
            np.savez(str(d / f"{stem}.stats.npz"),
                     median=np.zeros((C, F, 1), np.float32),
                     sigma=np.ones((C, F, 1), np.float32))
        band_dirs.append(str(d))
    span_dir = tmp_path / "spans"
    span_dir.mkdir()
    for subject_id, trial_id, _ in sessions:
        (span_dir / f"btbank{subject_id}_t{trial_id}.json").write_text(json.dumps({
            "subject_id": subject_id, "trial_id": trial_id, "bad_windows_s": [],
        }))
    return band_dirs, str(span_dir)


def _smoke_args(**over):
    a = dict(
        lr=6e-3, weight_decay=0.04, warmup_steps=5000, min_lr_ratio=1.0,
        adam_beta2=0.95, seed=33, monitor_every_n_steps=1,
        batch_size=2, clips_per_session=4, clip_len=3.0, num_workers=0,
        ssl_max_steps=2, precision="32-true", accelerator="cpu", devices=1,
        grad_clip=3.0, accumulate_grad_batches=2, log_every_n_steps=1,
        ckpt_dir=None, ckpt_ladder_every=0, wandb_project=None, run_name=None,
        compile=False, same_session_ranks=False, sdpa_backend="auto",
        ddp_static_graph=False, grad_ratio_every_n_steps=0,
        state_stats_dir=None, deep_sup=True,
        nll_warmup_start_step=0, nll_warmup_steps=0, resume_ckpt=None,
    )
    a.update(over)
    return argparse.Namespace(**a)


def test_parse_sessions() -> None:
    assert _parse_sessions(["6:4", "1:0"]) == [(6, 4), (1, 0)]
    with pytest.raises(ValueError, match="S:T"):
        _parse_sessions(["6"])


def test_build_optim_cfg_is_non_fused_adamw_warmupcosine() -> None:
    o = build_v3_optim_cfg(
        lr=6e-3, weight_decay=0.04, warmup_steps=5000, min_lr_ratio=1.0,
        adam_beta2=0.95,
    )
    assert type(o.optimizer).__name__ == "AdamW"  # discriminator resolved the subclass
    assert o.optimizer.lr == 6e-3
    assert "fused" not in o.optimizer.kwargs  # non-fused (bf16 grad-clip compat)
    assert list(o.optimizer.kwargs["betas"]) == [0.9, 0.95]
    assert o.optimizer.kwargs["weight_decay"] == 0.04
    assert type(o.scheduler).__name__ == "WarmupCosine"


# --- resume / fork wiring ----------------------------------------------------
# r4b forks r4 at step 10000 (--lambda-nll 0) by restoring r4's ladder ckpt. That
# needs dispatch to thread a ckpt_path into trainer.fit — v3 runs dispatch directly
# (no exca), so the resume is plain Lightning ckpt_path, and it was never plumbed.


def test_resume_ckpt_arg_defaults_none_and_parses() -> None:
    from speech_decoding.experiments.dispatch_v3 import build_arg_parser

    base = [
        "--bt-root", "x", "--band-cache-dir", "a", "--band-cache-dir", "b",
        "--band-cache-dir", "c", "--span-dir", "s", "--session", "1:0",
        "--ssl-max-steps", "1",
    ]
    assert build_arg_parser().parse_args(base).resume_ckpt is None
    got = build_arg_parser().parse_args(base + ["--resume-ckpt", "/p/ladder-step=10000.ckpt"])
    assert got.resume_ckpt == "/p/ladder-step=10000.ckpt"


def _cpu_trainer(max_steps, ckpt_dir=None):
    """A tiny CPU trainer that honors a small max_steps — build_v3_training's own
    trainer floors max_steps to 100k (a real-run stop-point), so resume tests build
    their own."""
    import lightning.pytorch as pl

    cbs = []
    if ckpt_dir is not None:
        from lightning.pytorch.callbacks import ModelCheckpoint

        cbs.append(ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="ladder-{step}",
            every_n_train_steps=1, save_last=True, save_top_k=-1,
        ))
    return pl.Trainer(
        max_steps=max_steps, max_epochs=-1, accelerator="cpu", devices=1,
        precision="32-true", accumulate_grad_batches=2, log_every_n_steps=1,
        callbacks=cbs, logger=False, num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1, enable_checkpointing=ckpt_dir is not None,
        gradient_clip_val=3.0, use_distributed_sampler=False,
    )


def test_module_resumes_model_and_step_from_checkpoint(tmp_path) -> None:
    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_caches(tmp_path, sess)

    def _build():
        specs = load_v3_sessions(
            sessions=[(1, 0)], band_cache_dirs=band_dirs, span_dir=span_dir,
            parcel_fn=_stub_parcel_fn,
        )
        module, dm, _ = build_v3_training(specs, _smoke_args(clips_per_session=4))
        return module, dm

    ckpt_dir = tmp_path / "ck"
    m1, dm1 = _build()
    _cpu_trainer(2, ckpt_dir=ckpt_dir).fit(m1, datamodule=dm1)
    last = ckpt_dir / "last.ckpt"
    assert last.exists()
    trained = torch.cat([p.detach().flatten() for p in m1.model.parameters()
                         if p.requires_grad])

    # a FRESH module (independent init) restored from the ckpt at its own stop-step
    # must end holding the ckpt's exact weights after ZERO further opt-steps — the
    # unambiguous proof that ckpt_path restored model + global_step (not a fresh run).
    m2, dm2 = _build()
    fresh = torch.cat([p.detach().flatten() for p in m2.model.parameters()
                       if p.requires_grad])
    assert not torch.allclose(trained, fresh)  # genuinely different before restore
    t2 = _cpu_trainer(2)  # max_steps == restore step ⇒ resume then stop immediately
    t2.fit(m2, datamodule=dm2, ckpt_path=str(last))
    assert t2.global_step == 2
    restored = torch.cat([p.detach().flatten() for p in m2.model.parameters()
                          if p.requires_grad])
    assert torch.allclose(trained, restored, atol=1e-5)


# --- r3 static_graph launch gate --------------------------------------------
# The gate fires on ANY optional module joining the trainable set under multi-GPU
# static_graph, not on one specific head: r4's secondary write-only Perceiver
# Gaussian-NLL (enabled by --state-stats-dir) is such a module and must be caught by
# the SAME gate. The crash it guards is reproduced on CPU in test_ddp_static_graph_repro.py.


def _ddp_args(**over):
    base = dict(
        devices=4, ddp_static_graph=True, ack_r3_static_graph=False,
        state_stats_dir=None, deep_sup=True,
    )
    base.update(over)
    return _smoke_args(**base)


def test_r3_gate_fires_on_secondary_multi_gpu_static_graph() -> None:
    with pytest.raises(SystemExit, match="secondary Perceiver Gaussian-NLL"):
        _build_trainer(_ddp_args(state_stats_dir="/some/stats"))


def test_r3_gate_is_generic_over_the_optional_module(monkeypatch) -> None:
    """Pin the SHAPE: ANY non-empty optional-module list trips the gate, independent of
    which module produced it."""
    import speech_decoding.experiments.dispatch_v3 as d3

    monkeypatch.setattr(
        d3, "_enabled_optional_modules", lambda args: ["perceiver_latents (--r4)"]
    )
    with pytest.raises(SystemExit, match="perceiver_latents"):
        _build_trainer(_ddp_args())


def test_r3_gate_silent_on_the_r1_r2_baseline() -> None:
    """r1 ran ~43k steps multi-GPU with static_graph=True and no optional module —
    the gate must not break that launch (no --state-stats-dir ⇒ empty optional list)."""
    assert _build_trainer(_ddp_args()) is not None


def test_r3_gate_ack_flag_overrides() -> None:
    assert _build_trainer(
        _ddp_args(state_stats_dir="/some/stats", ack_r3_static_graph=True)
    ) is not None


def test_r3_gate_ignores_single_device() -> None:
    """No DDPStrategy at devices=1 ⇒ no static_graph ⇒ nothing to gate."""
    assert _build_trainer(_ddp_args(devices=1, state_stats_dir="/some/stats")) is not None


def test_r3_gate_silent_with_no_static_graph() -> None:
    """--no-ddp-static-graph IS the fix; it must launch."""
    assert _build_trainer(
        _ddp_args(state_stats_dir="/some/stats", ddp_static_graph=False)
    ) is not None


# ------------------------------------------------------- heartbeat (r4 forensics)
class _HbTrainer:
    """Minimal trainer stand-in: the heartbeat reads only host ints."""

    def __init__(self, *, global_step: int, global_rank: int = 0, current_epoch: int = 7) -> None:
        self.global_step = global_step
        self.global_rank = global_rank
        self.current_epoch = current_epoch


class _HbModule:
    def log(self, *_: object, **__: object) -> None:  # the wandb scalar; not under test
        return None


def _hb_lines(capsys) -> list[str]:
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[hb]")]


def test_heartbeat_prints_on_cadence_and_is_silent_off_cadence(capsys) -> None:
    """The r4 post-mortem gap: the .log held ZERO step lines across 14 h, so the crash
    step was unknowable. Heartbeat must emit exactly on the N-step cadence."""
    cb = _StepTimeCallback(heartbeat_every=100)
    for step in (100, 137, 200):
        cb.on_train_batch_end(_HbTrainer(global_step=step), _HbModule())
    lines = _hb_lines(capsys)
    assert len(lines) == 2, lines  # 137 is off-cadence
    assert "step=100" in lines[0] and "step=200" in lines[1]


def test_heartbeat_dedupes_under_grad_accum(capsys) -> None:
    """``on_train_batch_end`` fires once per MICRO-batch, so r4's accumulate=4 hands the
    SAME global_step 4×. Without dedup the log would carry 4 identical lines per step."""
    cb = _StepTimeCallback(heartbeat_every=100)
    for _ in range(4):  # accumulate_grad_batches=4 micro-batches, one optimizer step
        cb.on_train_batch_end(_HbTrainer(global_step=100), _HbModule())
    assert len(_hb_lines(capsys)) == 1


def test_heartbeat_is_per_rank_and_names_the_rank(capsys) -> None:
    """Rank identity is the WHOLE point: r4 died with ranks 0/2/3 stalled at collective
    266865 while rank 1 ran ahead into a broadcast. Per-rank lines make a divergence
    readable directly, instead of inferred from NCCL sequence numbers."""
    for rank in range(4):
        _StepTimeCallback(heartbeat_every=100).on_train_batch_end(
            _HbTrainer(global_step=100, global_rank=rank), _HbModule()
        )
    lines = _hb_lines(capsys)
    assert len(lines) == 4
    assert {f"rank={r}" for r in range(4)} == {ln.split()[1] for ln in lines}


def test_heartbeat_reports_step_epoch_elapsed_and_never_syncs(capsys) -> None:
    """Fields needed to bracket a stall; and the values it reads are host ints — a
    GPU ``.item()`` here would undo the #37 per-step sync kill."""
    cb = _StepTimeCallback(heartbeat_every=100)
    cb.on_train_batch_end(_HbTrainer(global_step=100, current_epoch=520), _HbModule())
    (line,) = _hb_lines(capsys)
    for field in ("rank=0", "step=100", "epoch=520", "sec_per_step=", "elapsed="):
        assert field in line, line
    assert "sec_per_step=nan" in line  # first mark: no prior perf_counter


def test_heartbeat_disabled_when_zero(capsys) -> None:
    cb = _StepTimeCallback(heartbeat_every=0)
    cb.on_train_batch_end(_HbTrainer(global_step=100), _HbModule())
    assert _hb_lines(capsys) == []


def test_step_time_scalar_still_skips_first_step_then_logs() -> None:
    """The PRE-EXISTING _StepTimeCallback contract, pinned cheaply (no 12M-param CPU
    fit): first batch-end has no prior perf_counter mark ⇒ no scalar; every later one
    logs a finite train_sec_per_step with sync_dist=False (no cross-rank barrier)."""
    logged: list[tuple[str, float, bool]] = []

    class _Rec(_HbModule):
        def log(self, key, value, **kw):  # type: ignore[override]
            logged.append((key, float(value), bool(kw.get("sync_dist", False))))

    cb, mod = _StepTimeCallback(heartbeat_every=0), _Rec()
    cb.on_train_batch_end(_HbTrainer(global_step=1), mod)
    assert logged == []  # first step: no prior mark
    cb.on_train_batch_end(_HbTrainer(global_step=2), mod)
    assert len(logged) == 1
    key, value, sync = logged[0]
    assert key == "train_sec_per_step" and value >= 0.0 and sync is False
