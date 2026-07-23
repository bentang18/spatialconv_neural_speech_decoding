"""v14_converged_v3 launcher — E1/E2 assembly + resume/forensics wiring (TDD).

Mostly cheap, pure-Python contract tests over the launcher assembly (E1), the
trainer knobs (E2), and the r4 heartbeat.

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
    _frontend_config,
    _nf_decimate,
    _parse_sessions,
    _StepTimeCallback,
    R5NF_BAND_RATES,
    build_arg_parser,
    build_v3_optim_cfg,
    build_v3_training,
)
from speech_decoding.models.v14_converged_v3.dataset import (
    NATIVE_FINE_BAND_RATES,
    R5_BAND_RATES,
    UNIFORM_BAND_RATES,
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
        ddp_static_graph=False, deep_sup=True, resume_ckpt=None,
        batch_unit=None, contact_budget=None, shaft_alpha=0.5,
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
    # (native_fine_hga, early_fusion, no_fusion, band_rates)
    assert _frontend_config(_smoke_args()) == (False, False, False, UNIFORM_BAND_RATES)
    assert _frontend_config(_smoke_args(frontend="v3")) == (False, False, False, UNIFORM_BAND_RATES)
    assert _frontend_config(_smoke_args(frontend="v3fine")) == (True, False, False, NATIVE_FINE_BAND_RATES)
    assert _frontend_config(_smoke_args(frontend="v3r5")) == (False, True, False, R5_BAND_RATES)
    assert _frontend_config(_smoke_args(frontend="v3r5nf")) == (False, False, True, R5NF_BAND_RATES)


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
    # r5 (early_fusion) DEFAULTS to shaft-level cross-patient batching; shaft mode requires an
    # explicit --contact-budget (the per-pack contact count that pins grid.total — no invented
    # numeric default). The ShaftPackDataset still threads the same per-band read contract.
    from speech_decoding.models.v14_converged_v3.shaft_dataset import ShaftPackDataset

    m_r5, dm_r5, _ = build_v3_training(
        specs, _smoke_args(frontend="v3r5", contact_budget=16)
    )
    assert m_r5.model.early_fusion is True
    assert m_r5.model.objective.early_fusion is True
    assert m_r5.model.native_fine_hga is False
    assert dm_r5.batch_unit == "shaft"  # early_fusion ⇒ shaft is the default batch unit
    assert isinstance(dm_r5.dataset, ShaftPackDataset)
    assert dm_r5.dataset.contact_budget == 16
    assert dm_r5.dataset.band_rates == R5_BAND_RATES
    assert dm_r5.dataset.start_align == 1  # lcm(1,1) — single-rate lattice


def test_v3r5nf_threads_no_fusion_and_stream_weight_then_fits(tmp_path) -> None:
    # v3r5nf reads the SAME 2 caches as v3r5 (v3hga F=4, v3lfs F=1) but must reach the
    # NoFusionStem (model.no_fusion) and default to shaft batching. --mae-stream-weight
    # threads to the objective (default 'equal'); a couple of real CPU fit steps prove the
    # full trainer path — shaft-pack dataset + session_plan static shapes + backward — the
    # layer where the 07-22 plan-cache clock bug lived.
    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_r5_caches(tmp_path, sess)
    specs = load_v3_sessions(
        sessions=[(1, 0)], band_cache_dirs=band_dirs, span_dir=span_dir,
        parcel_fn=_stub_parcel_fn, band_rates=R5NF_BAND_RATES,
    )
    m, dm, _ = build_v3_training(
        specs, _smoke_args(frontend="v3r5nf", contact_budget=16, objective="mae")
    )
    assert m.model.no_fusion is True
    assert m.model.objective.no_fusion is True
    assert m.model.early_fusion is False and m.model.native_fine_hga is False
    assert m.model.objective.mae_stream_weight == "equal"  # NEW DEFAULT
    assert dm.batch_unit == "shaft"  # two-stream ⇒ shaft is the default batch unit
    assert dm.dataset.band_rates == R5NF_BAND_RATES

    # --mae-stream-weight pooled threads through too.
    m_pooled, _, _ = build_v3_training(
        specs, _smoke_args(frontend="v3r5nf", contact_budget=16, objective="mae",
                           mae_stream_weight="pooled")
    )
    assert m_pooled.model.objective.mae_stream_weight == "pooled"

    _cpu_trainer(2).fit(m, datamodule=dm)  # runs; no shape/plan-cache error


def test_v3r5nffast_threads_decimate_4_and_per_stream_block_w_then_fits(tmp_path) -> None:
    # v3r5nffast = v3r5nf with the first stem conv at stride 2 (net 4× → 16 Hz tokens). It maps to
    # the SAME (no_fusion, R5NF_BAND_RATES) config as v3r5nf but with nf_decimate=4. Temporal block
    # widths are now PER-STREAM (Ben 2026-07-22): HGA 3 / LFS 5 by default on BOTH no-fusion arms —
    # LFS wider so the slow stream can't trivially in-fill the masked run.
    assert _frontend_config(_smoke_args(frontend="v3r5nffast")) == (
        False, False, True, R5NF_BAND_RATES
    )  # same tuple as v3r5nf — the decimate is the only difference
    assert _nf_decimate(_smoke_args(frontend="v3r5nffast")) == 4
    assert _nf_decimate(_smoke_args(frontend="v3r5nf")) == 2
    assert _nf_decimate(_smoke_args(frontend="v3")) == 2

    sess = [(1, 0, _shaft_labels((8, 8, 8)))]
    band_dirs, span_dir = _write_r5_caches(tmp_path, sess)
    specs = load_v3_sessions(
        sessions=[(1, 0)], band_cache_dirs=band_dirs, span_dir=span_dir,
        parcel_fn=_stub_parcel_fn, band_rates=R5NF_BAND_RATES,
    )
    m, dm, _ = build_v3_training(
        specs, _smoke_args(frontend="v3r5nffast", contact_budget=16, objective="mae")
    )
    assert m.model.no_fusion is True
    assert m.model.nf_decimate == 4
    assert m.model.objective.nf_decimate == 4
    assert m.model.objective.online.stem.decimate == 4
    assert m.model.objective.mae_head_hga.out_features == 16  # DEC·4
    assert m.model.objective.mae_head_lfs.out_features == 4  # DEC·1
    assert m.model.mask_cfg.hga_block_w == 3  # per-stream default (HGA)
    assert m.model.mask_cfg.lfs_block_w == 5  # per-stream default (LFS, wider)
    assert dm.batch_unit == "shaft"

    # v3r5nf gets the same per-stream defaults (3 / 5).
    m_nf, _, _ = build_v3_training(
        specs, _smoke_args(frontend="v3r5nf", contact_budget=16, objective="mae")
    )
    assert m_nf.model.nf_decimate == 2
    assert (m_nf.model.mask_cfg.hga_block_w, m_nf.model.mask_cfg.lfs_block_w) == (3, 5)

    # Per-stream flags override independently.
    m_ov, _, _ = build_v3_training(
        specs, _smoke_args(frontend="v3r5nffast", contact_budget=16, objective="mae",
                           hga_block_w=4, lfs_block_w=8)
    )
    assert (m_ov.model.mask_cfg.hga_block_w, m_ov.model.mask_cfg.lfs_block_w) == (4, 8)

    # --temporal-block-w sets BOTH streams (back-compat) when no per-stream flag is given.
    m_both, _, _ = build_v3_training(
        specs, _smoke_args(frontend="v3r5nf", contact_budget=16, objective="mae",
                           temporal_block_w=6)
    )
    assert (m_both.model.mask_cfg.hga_block_w, m_both.model.mask_cfg.lfs_block_w) == (6, 6)

    # A per-stream flag wins over --temporal-block-w for that stream only.
    m_mix, _, _ = build_v3_training(
        specs, _smoke_args(frontend="v3r5nf", contact_budget=16, objective="mae",
                           temporal_block_w=6, lfs_block_w=9)
    )
    assert (m_mix.model.mask_cfg.hga_block_w, m_mix.model.mask_cfg.lfs_block_w) == (6, 9)

    _cpu_trainer(2).fit(m, datamodule=dm)  # 16 Hz token grid fits end-to-end (plan cache + backward)


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
