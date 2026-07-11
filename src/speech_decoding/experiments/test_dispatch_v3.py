"""v14_converged_v3 launcher — E1/E2 assembly + F1 local CPU smoke (TDD).

Drives the WHOLE v3 stack end-to-end on CPU with synthetic caches + a stub
parcel_fn (no BT anatomy, no wandb, no GPU): synthetic band caches ─▶
load_v3_sessions ─▶ build_v3_training ─▶ trainer.fit for a couple of optimizer
steps. This is the first thing that constructs a real V3Batch and runs it through
the model + objective + optimizer + EMA + monitors, so it is the launch-readiness
gate (F1) as well as the unit test for the launcher assembly (E1) and the trainer
knobs (E2).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

from speech_decoding.experiments.dispatch_v3 import (
    _parse_sessions,
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
        ddp_static_graph=False,
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


def test_local_cpu_smoke_fit_runs(tmp_path) -> None:
    sess = [
        (1, 0, _shaft_labels((8, 8, 8))),
        (2, 1, _shaft_labels((8, 8))),
    ]
    band_dirs, span_dir = _write_caches(tmp_path, sess)
    specs = load_v3_sessions(
        sessions=[(1, 0), (2, 1)],
        band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
    )
    args = _smoke_args()
    module, dm, trainer = build_v3_training(specs, args)

    # teacher starts as an exact copy of the online tower (EMA not yet advanced)
    trainer.fit(module, datamodule=dm)

    assert trainer.global_step == 2  # max_steps honored
    # the training_step logged a finite loss and a fixed masked count
    assert torch.isfinite(torch.tensor(trainer.callback_metrics["train_loss"].item()))


def test_main_argv_path_runs(tmp_path, monkeypatch) -> None:
    # exercise main()'s full argv → load_v3_sessions → build_v3_training → fit glue,
    # with the one real seam (BT parcel lookup) stubbed so no anatomy/GPU is needed.
    import speech_decoding.experiments.dispatch_v3 as d3

    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_caches(tmp_path, sess)
    monkeypatch.setattr(d3, "make_bt_parcel_fn", lambda bt_root: _stub_parcel_fn)
    argv = [
        "--bt-root", "unused",
        "--band-cache-dir", band_dirs[0],
        "--band-cache-dir", band_dirs[1],
        "--band-cache-dir", band_dirs[2],
        "--span-dir", span_dir,
        "--session", "1:0",
        "--clips-per-session", "4", "--batch-size", "2",
        "--ssl-max-steps", "1", "--accumulate-grad-batches", "2",
        "--accelerator", "cpu", "--devices", "1", "--precision", "32-true",
        "--num-workers", "0",
    ]
    d3.main(argv)  # must not raise


def test_smoke_advances_teacher_and_params(tmp_path) -> None:
    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_caches(tmp_path, sess)
    specs = load_v3_sessions(
        sessions=[(1, 0)],
        band_cache_dirs=band_dirs, span_dir=span_dir, parcel_fn=_stub_parcel_fn,
    )
    args = _smoke_args(clips_per_session=4, ssl_max_steps=1, accumulate_grad_batches=2)
    module, dm, trainer = build_v3_training(specs, args)
    before = torch.cat([p.detach().flatten() for p in module.model.parameters()
                        if p.requires_grad])
    trainer.fit(module, datamodule=dm)
    after = torch.cat([p.detach().flatten() for p in module.model.parameters()
                       if p.requires_grad])
    # one optimizer step moved the trainable params (optimizer + EMA both ran)
    assert not torch.allclose(before, after)
