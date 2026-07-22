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
    _frontend_config,
    _parse_sessions,
    _StepTimeCallback,
    build_arg_parser,
    build_v3_optim_cfg,
    build_v3_training,
)
from speech_decoding.models.v14_converged_v3.dataset import (
    NATIVE_FINE_BAND_RATES,
    R5_BAND_RATES,
    UNIFORM_BAND_RATES,
)
from speech_decoding.experiments.fork_point_ckpt import (
    fork_ckpt_general,
    fork_to_point_ckpt,
    optimizer_param_names,
)
from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
from speech_decoding.models.v14_converged_v3.state_target import STATE_DIM

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


def test_frontend_flag_parses_and_defaults_to_v3() -> None:
    base = ["--bt-root", "/b", "--band-cache-dir", "/s", "--band-cache-dir", "/m",
            "--band-cache-dir", "/h", "--span-dir", "/sp", "--session", "1/0",
            "--ssl-max-steps", "10"]
    assert build_arg_parser().parse_args(base).frontend == "v3"
    assert build_arg_parser().parse_args(base + ["--frontend", "v3fine"]).frontend == "v3fine"
    assert build_arg_parser().parse_args(base + ["--frontend", "v3r5"]).frontend == "v3r5"


def test_frontend_config_maps_flag_to_native_flag_and_rates() -> None:
    # (native_fine_hga, early_fusion, band_rates)
    assert _frontend_config(_smoke_args()) == (False, False, UNIFORM_BAND_RATES)
    assert _frontend_config(_smoke_args(frontend="v3")) == (False, False, UNIFORM_BAND_RATES)
    assert _frontend_config(_smoke_args(frontend="v3fine")) == (True, False, NATIVE_FINE_BAND_RATES)
    assert _frontend_config(_smoke_args(frontend="v3r5")) == (False, True, R5_BAND_RATES)


def test_v3fine_threads_native_into_model_and_dataset(tmp_path) -> None:
    # The flag must reach BOTH the stem (model.native_fine_hga) AND the loader's per-band
    # read (dataset.band_rates); either alone silently mis-reads. Uniform default is proven
    # in every other build_v3_training test.
    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_caches(tmp_path, sess)
    specs = load_v3_sessions(
        sessions=[(1, 0)], band_cache_dirs=band_dirs, span_dir=span_dir,
        parcel_fn=_stub_parcel_fn,
    )
    m_fine, dm_fine, _ = build_v3_training(specs, _smoke_args(frontend="v3fine"))
    assert m_fine.model.native_fine_hga is True
    assert m_fine.model.objective.native_fine_hga is True
    assert dm_fine.dataset.band_rates == NATIVE_FINE_BAND_RATES
    assert dm_fine.dataset.start_align == 8  # lcm(8,2,1)

    m_uni, dm_uni, _ = build_v3_training(specs, _smoke_args())
    assert m_uni.model.native_fine_hga is False
    assert dm_uni.dataset.band_rates == UNIFORM_BAND_RATES


def _write_r5_caches(tmp_path, sessions, *, n_frames_64=400):
    """Two 64 Hz caches (v3hga F=4, v3lfs F=1) — the Chang 2-stream r5 input. total_frames
    is the 64 Hz count (= 2× the 32 Hz clock); the loader/dataset apply R5_BAND_RATES."""
    band_dirs = []
    for b, F in enumerate((4, 1)):  # (v3hga, v3lfs) — HGA first
        d = tmp_path / f"r5band{b}"
        d.mkdir(parents=True, exist_ok=True)
        for subject_id, trial_id, labels in sessions:
            stem = f"btbank{subject_id}_t{trial_id}"
            key = (
                '{"cls":"Wang2024Treebank","method":"_load_raw","timeline":'
                f'{{"extra_bad":[],"subject":"btbank{subject_id}",'
                f'"subject_id":{subject_id},"trial_id":{trial_id}}}}}_0.000_6867.860'
            )
            C = len(labels)
            (d / f"{stem}.json").write_text(json.dumps({
                "key": key, "ch_names": labels, "total_frames": n_frames_64,
                "sample_rate": 64,
            }))
            rng = np.random.default_rng(subject_id * 100 + trial_id + 17 + b)
            np.save(str(d / f"{stem}.npy"),
                    rng.standard_normal((C, F, n_frames_64)).astype(np.float32))
            np.savez(str(d / f"{stem}.stats.npz"),
                     median=np.zeros((C, F, 1), np.float32),
                     sigma=np.ones((C, F, 1), np.float32))
        band_dirs.append(str(d))
    span_dir = tmp_path / "r5spans"
    span_dir.mkdir()
    for subject_id, trial_id, _ in sessions:
        (span_dir / f"btbank{subject_id}_t{trial_id}.json").write_text(json.dumps({
            "subject_id": subject_id, "trial_id": trial_id, "bad_windows_s": [],
        }))
    return band_dirs, str(span_dir)


def test_v3r5_threads_early_fusion_into_model_and_dataset(tmp_path) -> None:
    # v3r5 must reach the stem (model.early_fusion) AND the loader's per-band read
    # (dataset.band_rates == R5_BAND_RATES). 2 caches (v3hga F=4, v3lfs F=1) @64 Hz;
    # the 32 Hz clip clock is total_frames//2, so clip_len 3s (96 tok) needs 192 frames.
    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_r5_caches(tmp_path, sess)
    specs = load_v3_sessions(
        sessions=[(1, 0)], band_cache_dirs=band_dirs, span_dir=span_dir,
        parcel_fn=_stub_parcel_fn, band_rates=R5_BAND_RATES,
    )
    assert specs[0].n_frames == 200  # 400 // 2 (64 Hz cache → 32 Hz clock)
    m_r5, dm_r5, _ = build_v3_training(specs, _smoke_args(frontend="v3r5"))
    assert m_r5.model.early_fusion is True
    assert m_r5.model.objective.early_fusion is True
    assert m_r5.model.native_fine_hga is False
    assert dm_r5.dataset.band_rates == R5_BAND_RATES
    assert dm_r5.dataset.start_align == 1  # lcm(1,1) — single-rate lattice


def test_clips_per_session_default_is_the_long_epoch() -> None:
    """r4b passes NO --clips-per-session, so this default IS Arm 0's epoch length. It must
    match the 40000 arms 1/2/4 pass explicitly, or the control arm silently inherits r4's
    every-52-step loader-rebuild churn (r4 died at 26414 on a suspected IO stall)."""
    from speech_decoding.experiments.dispatch_v3 import build_arg_parser

    base = ["--bt-root", "x", "--band-cache-dir", "a", "--band-cache-dir", "b",
            "--band-cache-dir", "c", "--span-dir", "s", "--session", "1:0",
            "--ssl-max-steps", "1"]
    assert build_arg_parser().parse_args(base).clips_per_session == 40_000
    # still overridable — the smoke launch and probes pass small values
    assert build_arg_parser().parse_args(
        base + ["--clips-per-session", "32"]).clips_per_session == 32


def test_resume_survives_a_changed_clips_per_session(tmp_path) -> None:
    """The r4 resume + arms raise --clips-per-session 40000 while r4 (and r4b) ran the
    default 2000, so the ckpt is restored into a datamodule whose EPOCH LENGTH differs
    20x. clips_per_session is documented operational (epoch length only), but "is X
    wired?" gets asserted, not assumed: the restored global_step must be the ckpt's,
    NOT 0 (a silent cold start is exactly the failure Ben hit with the v2 exca recipe)
    and NOT rescaled by the new epoch geometry."""
    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_caches(tmp_path, sess)

    def _build(clips):
        specs = load_v3_sessions(
            sessions=[(1, 0)], band_cache_dirs=band_dirs, span_dir=span_dir,
            parcel_fn=_stub_parcel_fn,
        )
        module, dm, _ = build_v3_training(specs, _smoke_args(clips_per_session=clips))
        return module, dm

    ckpt_dir = tmp_path / "ck_epoch"
    m1, dm1 = _build(4)  # short epochs: 4 clips/session -> epoch flips fast
    _cpu_trainer(4, ckpt_dir=ckpt_dir).fit(m1, datamodule=dm1)
    last = ckpt_dir / "last.ckpt"
    saved_epoch = torch.load(last, weights_only=False)["epoch"]
    trained = torch.cat([p.detach().flatten() for p in m1.model.parameters()
                         if p.requires_grad])

    m2, dm2 = _build(64)  # 16x longer epoch, same ckpt
    t2 = _cpu_trainer(4)  # max_steps == restore step => resume then stop immediately
    t2.fit(m2, datamodule=dm2, ckpt_path=str(last))
    restored = torch.cat([p.detach().flatten() for p in m2.model.parameters()
                          if p.requires_grad])
    ok_step = t2.global_step == 4
    ok_w = torch.allclose(trained, restored, atol=1e-5)
    print(f"[check] saved epoch={saved_epoch} (clips=4) -> resumed into clips=64 "
          f"epoch={t2.current_epoch}; global_step={t2.global_step} (want 4, cold=0) "
          f"{'OK' if ok_step and ok_w else 'VIOLATED'}")
    assert ok_step, f"changed epoch length broke step restore: {t2.global_step}"
    assert ok_w  # weights are the ckpt's, not a fresh init


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


# --- r5 Arm 2 wiring: --no-nll-floor -----------------------------------------


def test_nll_floor_flag_defaults_on_and_parses_no_variant() -> None:
    """Default MUST stay ON: r1-r4 and the r4 resume all assume the measured floor, so a
    flipped default would silently change a live run's objective mid-matrix."""
    from speech_decoding.experiments.dispatch_v3 import build_arg_parser

    base = [
        "--bt-root", "x", "--band-cache-dir", "a", "--band-cache-dir", "b",
        "--band-cache-dir", "c", "--span-dir", "s", "--session", "1:0",
        "--ssl-max-steps", "1",
    ]
    assert build_arg_parser().parse_args(base).nll_floor is True
    assert build_arg_parser().parse_args(base + ["--no-nll-floor"]).nll_floor is False
    assert build_arg_parser().parse_args(base + ["--nll-floor"]).nll_floor is True


def test_nll_floor_threads_from_args_through_model_to_objective() -> None:
    """'Is X wired?' -> assert it end-to-end, don't trust the kwarg chain by eye. The flag
    crosses dispatch -> V3ConvergedModel -> V3JepaObjective; a drop anywhere is silent
    (the run trains happily WITH the floor and Arm 2 measures nothing)."""
    from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel

    assert V3ConvergedModel(n_parcels=4).objective.nll_floor is True  # default = r4
    assert V3ConvergedModel(n_parcels=4, nll_floor=False).objective.nll_floor is False


def _write_state_stats(tmp_path, subject_id: int, n_parcels: int = 64):
    """Frozen per-(parcel-value, dim) state-norm table — identity normalization. Presence
    of the dir is what opts the secondary IN (session_loader._load_state_stats)."""
    import numpy as np

    d = tmp_path / "stats"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"sub-{subject_id}.npz",
             stat_mean=np.zeros((n_parcels, STATE_DIM), dtype=np.float32),
             stat_std=np.ones((n_parcels, STATE_DIM), dtype=np.float32))
    return str(d)


def test_arm3_forks_an_nll_checkpoint_despite_owning_no_covariance_head(tmp_path) -> None:
    """r5 Arm 3 forks r4's step-10000 ckpt — written by a model that HAD a chol_head that
    Arm 3's point head does not own. Two ways that can go wrong, both silent-to-reason-about
    and neither allowed to be assumed:

      1. STATE_DICT: a strict load must reject the 2 unexpected chol_head keys. If it does,
         the fork needs those keys stripped, and this test is what proves the stripping is
         necessary AND sufficient.
      2. OPTIMIZER: AdamW state is restored BY POSITION within each param group, so 2 fewer
         trainable params can shift every later index. The restored ``global_step`` must be
         the ckpt's (10000 in the real run, 4 here) and NOT 0 — a silent cold start is the
         exact failure Ben hit with the v2 exca recipe.

    The perceiver is at INIT in the real fork (λ=0 for the whole 0→10k prefix, so it took
    no gradient), which is WHY dropping its covariance weights costs nothing scientifically.
    """
    import torch

    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_caches(tmp_path, sess)
    stats_dir = _write_state_stats(tmp_path, subject_id=1)

    def _build(secondary_loss):
        specs = load_v3_sessions(
            sessions=[(1, 0)], band_cache_dirs=band_dirs, span_dir=span_dir,
            parcel_fn=_stub_parcel_fn, state_stats_dir=stats_dir,
        )
        module, dm, _ = build_v3_training(
            specs, _smoke_args(state_stats_dir=stats_dir, secondary_loss=secondary_loss)
        )
        return module, dm

    ckpt_dir = tmp_path / "ck_nll"
    m_nll, dm_nll = _build("nll")
    _cpu_trainer(4, ckpt_dir=ckpt_dir).fit(m_nll, datamodule=dm_nll)
    last = ckpt_dir / "last.ckpt"
    sd = torch.load(last, weights_only=False)["state_dict"]
    chol_keys = [k for k in sd if "chol_head" in k]

    # (1) does a strict load of the NLL ckpt into the POINT model actually reject?
    m_l1, dm_l1 = _build("l1")
    strict_rejected = False
    try:
        m_l1.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        strict_rejected = "chol_head" in str(e)

    # (2) a NAIVE strip (state_dict only) must NOT be enough — it leaves the optimizer
    #     param groups the wrong size. Pinning this is the point: it is why
    #     fork_point_ckpt exists rather than a one-line dict comprehension.
    naive = ckpt_dir / "fork-naive.ckpt"
    ck = torch.load(last, weights_only=False)
    ck["state_dict"] = {k: v for k, v in ck["state_dict"].items() if "chol_head" not in k}
    torch.save(ck, naive)
    naive_err = ""
    try:
        m_n, dm_n = _build("l1")
        _cpu_trainer(4).fit(m_n, datamodule=dm_n, ckpt_path=str(naive))
    except Exception as e:
        naive_err = type(e).__name__
    naive_rejected = naive_err != ""

    # (3) the REAL fork: strip + remap the optimizer state BY NAME.
    m_old, _ = _build("nll")
    m_new, _ = _build("l1")
    old_names = optimizer_param_names(m_old)
    new_names = optimizer_param_names(m_new)
    forked_ck, rep = fork_to_point_ckpt(
        torch.load(last, weights_only=False), old_names, new_names
    )
    good = ckpt_dir / "fork-point.ckpt"
    torch.save(forked_ck, good)

    m_l1b, dm_l1b = _build("l1")
    t = _cpu_trainer(4)  # max_steps == restore step => resume then stop immediately
    forked, err = True, ""
    try:
        t.fit(m_l1b, datamodule=dm_l1b, ckpt_path=str(good))
    except Exception as e:
        forked, err = False, f"{type(e).__name__}: {e}"

    ok_step = forked and t.global_step == 4
    only_chol = all("chol_head" in n for n in rep["optimizer_params_dropped"])
    ok = bool(chol_keys) and strict_rejected and naive_rejected and forked and ok_step \
        and only_chol
    print(f"[check] fork: ckpt chol keys={len(chol_keys)}; strict load into the point model "
          f"REJECTED={strict_rejected}; NAIVE state_dict-only strip also rejected="
          f"{naive_rejected} ({naive_err or 'no error!'}) => the optimizer remap is REQUIRED")
    print(f"[check] remapped fork: dropped optimizer params={rep['optimizer_params_dropped']} "
          f"(only chol_head={only_chol}); {rep['n_params_in']}->{rep['n_params_out']} params; "
          f"loaded={forked}{(' err=' + err) if err else ''}; restored global_step="
          f"{t.global_step if forked else 'n/a'} (want 4, cold=0) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_fork_reinits_shape_changed_mu_head_for_diag_nll() -> None:
    """r5-mod (diag_nll) forks the SAME r4@10k ckpt as Arm 3, but its state is 5-dim
    ([slow_mu, mid_mu, hga_mu, relmod48, relmod816]) where the ckpt's NLL head is 6-dim
    (3 mean + 3 std). So mu_head is a SHAPE CHANGE (6,d)->(5,d), not a drop: carrying its
    tensor would fail the strict load and carrying its Adam moment would fail the optimizer
    load. The fork must (a) drop chol_head, (b) reseed mu_head from the fresh 5-dim module,
    (c) drop mu_head's carried optimizer state (lazy re-init), (d) carry the backbone BY
    NAME and preserve global_step. Reseeding costs nothing at the fork point: the whole
    0->10k prefix ran lambda=0, so the head took zero gradient (moments are zero, weights
    are decayed init).

    Unit-tested on a SYNTHETIC 6-dim ckpt because current code (STATE_DIM=5) can no longer
    write one; the same-dim (l1) path stays covered by the Arm-3 end-to-end test above."""
    import torch

    d = 4
    sd = {
        "model.enc.weight": torch.arange(9.0).reshape(3, 3),
        "model.head.mu_head.weight": torch.full((6, d), 7.0),  # the OLD 6-dim head
        "model.head.mu_head.bias": torch.full((6,), 7.0),
        "model.head.chol_head.weight": torch.zeros(21, d),  # 6-dim tril = 21
        "model.head.chol_head.bias": torch.zeros(21),
    }
    # optimizer index order (as maybe_split_no_decay lays it out): wd group = the >=2-D
    # weights [enc, mu_head.w, chol_head.w]; no-wd group = the 1-D biases [mu_head.b, chol_head.b].
    old_names = [
        "model.enc.weight", "model.head.mu_head.weight", "model.head.chol_head.weight",
        "model.head.mu_head.bias", "model.head.chol_head.bias",
    ]
    # a distinguishable Adam moment per old param, so carry-vs-drop is checkable by value.
    old_state = {i: {"step": torch.tensor(10000.0), "tag": torch.tensor(float(i))}
                 for i in range(5)}
    ckpt = {
        "global_step": 10000, "epoch": 5, "state_dict": dict(sd),
        "optimizer_states": [{
            "state": old_state,
            "param_groups": [
                {"params": [0, 1, 2], "lr": 1e-3, "weight_decay": 0.04},  # wd (weights)
                {"params": [3, 4], "lr": 1e-3, "weight_decay": 0.0},       # no-wd (biases)
            ],
        }],
    }
    # the fresh 5-dim diag_nll module: mu_head (5,d), no chol.
    new_names = ["model.enc.weight", "model.head.mu_head.weight", "model.head.mu_head.bias"]
    seed = {
        "model.enc.weight": torch.zeros(3, 3),  # backbone in seed is IGNORED (only reinit keys used)
        "model.head.mu_head.weight": torch.ones(5, d),
        "model.head.mu_head.bias": torch.ones(5),
    }
    out, rep = fork_to_point_ckpt(
        ckpt, old_names, new_names, seed_state_dict=seed, reinit_substrings=("mu_head",)
    )

    # (a) chol fully gone; (b) mu_head reseeded to 5-dim from the fresh module.
    assert not any("chol_head" in k for k in out["state_dict"])
    assert out["state_dict"]["model.head.mu_head.weight"].shape == (5, d)
    assert torch.equal(out["state_dict"]["model.head.mu_head.weight"], torch.ones(5, d))
    assert torch.equal(out["state_dict"]["model.head.mu_head.bias"], torch.ones(5))
    # (c) backbone carried VERBATIM (value untouched, NOT taken from seed's zeros).
    assert torch.equal(out["state_dict"]["model.enc.weight"], sd["model.enc.weight"])
    assert out["global_step"] == 10000

    st = out["optimizer_states"][0]
    # new order [enc(0), mu_head.w(1), mu_head.b(2)]: only the backbone carries a moment;
    # both mu_head params are fresh (no state entry) so AdamW lazily re-inits them at (5,d).
    assert 0 in st["state"] and st["state"][0]["tag"].item() == 0.0  # enc carried from old idx 0
    assert 1 not in st["state"] and 2 not in st["state"]
    # group sizes: wd [enc, mu_head.w]=2, no-wd [mu_head.b]=1 (chol removed from each).
    assert [len(g["params"]) for g in st["param_groups"]] == [2, 1]
    assert set(rep["state_dict_keys_reseeded"]) == {
        "model.head.mu_head.weight", "model.head.mu_head.bias"
    }
    assert all("chol_head" in n for n in rep["optimizer_params_dropped"])
    print(f"[check] diag_nll fork: chol dropped={rep['state_dict_keys_dropped']}; "
          f"mu_head reseeded 6->5={rep['state_dict_keys_reseeded']}; backbone moment carried "
          f"(tag=0), mu_head moments dropped (fresh); global_step={out['global_step']} OK")


def test_fork_reseeds_shape_changed_buffer_not_optimizer() -> None:
    """The 5-dim head resizes a frozen BUFFER as well as a Linear: ``noise_var`` is
    register_buffer'd at STATE_DIM, so it went (6,)->(5,) alongside mu_head. A buffer is in
    the state_dict but NOT in the optimizer's param groups, so the fork must reseed it from
    the fresh module (else the carried 6-dim buffer fails the strict load — the exact reject
    the CPU resume smoke hit) WITHOUT ever trying to touch it as an optimizer param. This
    pins that a reinit substring naming a buffer reseeds the tensor and leaves the optimizer
    remap untouched."""
    import torch

    d = 4
    sd = {
        "model.enc.weight": torch.arange(9.0).reshape(3, 3),
        "model.head.mu_head.weight": torch.full((6, d), 7.0),
        "model.head.mu_head.bias": torch.full((6,), 7.0),
        "model.head.chol_head.weight": torch.zeros(21, d),
        "model.head.chol_head.bias": torch.zeros(21),
        "model.head.noise_var": torch.full((6,), 9.0),  # frozen buffer, OLD 6-dim
    }
    old_names = [
        "model.enc.weight", "model.head.mu_head.weight", "model.head.chol_head.weight",
        "model.head.mu_head.bias", "model.head.chol_head.bias",
    ]
    old_state = {i: {"step": torch.tensor(10000.0), "tag": torch.tensor(float(i))}
                 for i in range(5)}
    ckpt = {
        "global_step": 10000, "epoch": 5, "state_dict": dict(sd),
        "optimizer_states": [{
            "state": old_state,
            "param_groups": [
                {"params": [0, 1, 2], "lr": 1e-3, "weight_decay": 0.04},
                {"params": [3, 4], "lr": 1e-3, "weight_decay": 0.0},
            ],
        }],
    }
    new_names = ["model.enc.weight", "model.head.mu_head.weight", "model.head.mu_head.bias"]
    seed = {
        "model.enc.weight": torch.zeros(3, 3),
        "model.head.mu_head.weight": torch.ones(5, d),
        "model.head.mu_head.bias": torch.ones(5),
        "model.head.noise_var": torch.full((5,), 0.5),  # fresh 5-dim floor
    }
    out, rep = fork_to_point_ckpt(
        ckpt, old_names, new_names, seed_state_dict=seed,
        reinit_substrings=("mu_head", "noise_var"),
    )

    # buffer reseeded to the fresh 5-dim floor, not the carried 6-dim one.
    assert out["state_dict"]["model.head.noise_var"].shape == (5,)
    assert torch.equal(out["state_dict"]["model.head.noise_var"], torch.full((5,), 0.5))
    assert "model.head.noise_var" in rep["state_dict_keys_reseeded"]
    # buffer is NOT an optimizer param: it never appears in the (dropped) optimizer names,
    # and the remap still only drops the two chol params.
    assert all("chol_head" in n for n in rep["optimizer_params_dropped"])
    st = out["optimizer_states"][0]
    assert [len(g["params"]) for g in st["param_groups"]] == [2, 1]
    assert 0 in st["state"] and 1 not in st["state"] and 2 not in st["state"]
    print(f"[check] buffer fork: noise_var reseeded 6->5 (state_dict only, not optimizer)="
          f"{out['state_dict']['model.head.noise_var'].tolist()}; opt drops still chol-only OK")


def test_fork_ckpt_general_carries_reshapes_drops_and_adds() -> None:
    """The context arm needs the fully general fork: it forks the SAME r4@10k (6-dim NLL)
    ckpt but the new module BOTH reshapes the diag head 6->5 (like r5-mod) AND INJECTS a
    ``pred_to_target_context`` head the ckpt never had. ``fork_to_point_ckpt`` can't add a
    param (it refuses names the ckpt lacks), so ``fork_ckpt_general`` classifies every new
    param CARRY / RESHAPE / ADD / (implicit DROP) against the ckpt.

    Pins, on a synthetic COMPILED-prefix ckpt (``model._orig_mod.``) so the cross-prefix
    match + inject-under-ckpt-front both get exercised:
      * enc + pred_to_target CARRY verbatim (value + Adam moment BY NAME),
      * mu_head (Linear) and noise_var (buffer) RESHAPE 6->5 from the fresh module (moment
        dropped for the Linear; buffer never touches the optimizer),
      * chol_head DROP (absent from out state_dict AND optimizer),
      * pred_to_target_context ADD from the fresh module under the ckpt's ``_orig_mod`` front,
        with NO carried moment (fresh at the lambda=0 fork point)."""
    import torch

    d, t = 4, 3
    F = "model._orig_mod."  # compiled-run front the ckpt carries
    ck_sd = {
        F + "enc.weight": torch.arange(9.0).reshape(3, 3),
        F + "head.mu_head.weight": torch.full((6, d), 7.0),
        F + "head.mu_head.bias": torch.full((6,), 7.0),
        F + "head.chol_head.weight": torch.zeros(21, d),
        F + "head.chol_head.bias": torch.zeros(21),
        F + "head.noise_var": torch.full((6,), 9.0),  # frozen buffer, OLD 6-dim
        F + "pred_to_target.weight": torch.full((t, d), 5.0),
        F + "pred_to_target.bias": torch.full((t,), 5.0),
    }
    # ckpt optimizer index order: wd group (>=2-D weights) then no-wd (1-D biases). noise_var
    # is a buffer, so it is NOT here.
    old_names = [
        F + "enc.weight", F + "head.mu_head.weight", F + "head.chol_head.weight",
        F + "pred_to_target.weight",
        F + "head.mu_head.bias", F + "head.chol_head.bias", F + "pred_to_target.bias",
    ]
    old_state = {i: {"step": torch.tensor(10000.0), "tag": torch.tensor(float(i))}
                 for i in range(7)}
    ckpt = {
        "global_step": 10000, "epoch": 5, "state_dict": dict(ck_sd),
        "optimizer_states": [{
            "state": old_state,
            "param_groups": [
                {"params": [0, 1, 2, 3], "lr": 1e-3, "weight_decay": 0.04},  # wd weights
                {"params": [4, 5, 6], "lr": 1e-3, "weight_decay": 0.0},       # no-wd biases
            ],
        }],
    }
    # the fresh module: 5-dim diag head, NO chol, PLUS a context head. Uncompiled "model."
    # keys (different front from the ckpt) to prove the stripped match works cross-prefix.
    new_group_names = [
        ["model.enc.weight", "model.head.mu_head.weight",
         "model.pred_to_target.weight", "model.pred_to_target_context.weight"],
        ["model.head.mu_head.bias", "model.pred_to_target.bias",
         "model.pred_to_target_context.bias"],
    ]
    new_sd = {
        "model.enc.weight": torch.zeros(3, 3),                 # CARRY target IGNORED
        "model.head.mu_head.weight": torch.ones(5, d),         # RESHAPE seed
        "model.head.mu_head.bias": torch.ones(5),              # RESHAPE seed
        "model.head.noise_var": torch.full((5,), 0.5),         # RESHAPE seed (buffer)
        "model.pred_to_target.weight": torch.zeros(t, d),      # CARRY target IGNORED
        "model.pred_to_target.bias": torch.zeros(t),           # CARRY target IGNORED
        "model.pred_to_target_context.weight": torch.full((t, d), 2.0),  # ADD
        "model.pred_to_target_context.bias": torch.full((t,), 2.0),      # ADD
    }
    out, rep = fork_ckpt_general(
        ckpt, old_names=old_names, new_group_names=new_group_names, new_state_dict=new_sd
    )
    osd = out["state_dict"]

    # CARRY: verbatim ckpt value, under the ckpt's own compiled front (NOT the seed's zeros).
    assert torch.equal(osd[F + "enc.weight"], ck_sd[F + "enc.weight"])
    assert torch.equal(osd[F + "pred_to_target.weight"], ck_sd[F + "pred_to_target.weight"])
    # RESHAPE: reseeded to the fresh 5-dim tensors, keyed under the ckpt front.
    assert osd[F + "head.mu_head.weight"].shape == (5, d)
    assert torch.equal(osd[F + "head.mu_head.weight"], torch.ones(5, d))
    assert osd[F + "head.noise_var"].shape == (5,)
    assert torch.equal(osd[F + "head.noise_var"], torch.full((5,), 0.5))
    # ADD: injected under the ckpt's compiled front, taken from the fresh module.
    assert torch.equal(osd[F + "pred_to_target_context.weight"], torch.full((t, d), 2.0))
    assert torch.equal(osd[F + "pred_to_target_context.bias"], torch.full((t,), 2.0))
    # DROP: chol gone entirely.
    assert not any("chol_head" in k for k in osd)
    assert out["global_step"] == 10000

    st = out["optimizer_states"][0]
    # new order [enc(0), mu.w(1), ptt.w(2), pttc.w(3), mu.b(4), ptt.b(5), pttc.b(6)].
    # carried moments: enc<-old0, pred_to_target.w<-old3, pred_to_target.b<-old6.
    assert st["state"][0]["tag"].item() == 0.0
    assert st["state"][2]["tag"].item() == 3.0
    assert st["state"][5]["tag"].item() == 6.0
    # reshaped (mu.w=1, mu.b=4) and added (pttc.w=3, pttc.b=6) carry NO moment.
    assert all(j not in st["state"] for j in (1, 3, 4, 6))
    assert [len(g["params"]) for g in st["param_groups"]] == [4, 3]

    assert sorted(rep["state_dict_keys_reshaped"]) == [
        "head.mu_head.bias", "head.mu_head.weight", "head.noise_var"
    ]
    assert sorted(rep["state_dict_keys_added"]) == [
        "pred_to_target_context.bias", "pred_to_target_context.weight"
    ]
    assert all("chol_head" in k for k in rep["state_dict_keys_dropped"])
    print(f"[check] general fork: carry={{enc,pred_to_target}} moments tag 0/3/6; "
          f"reshape={rep['state_dict_keys_reshaped']}; add={rep['state_dict_keys_added']}; "
          f"drop={rep['state_dict_keys_dropped']}; groups {[len(g['params']) for g in st['param_groups']]} "
          f"global_step={out['global_step']} OK")
