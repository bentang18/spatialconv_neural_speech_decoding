"""B36 WS-F tests for the Phase-3 Whisper-distillation module + experiment.

F1 (module / loss):
* construction freezes the encoder under 3a (connector trains) and unfreezes
  it under 3b;
* ``_step`` returns a finite Smooth-L1 scalar with student AND teacher both at
  1280-d × 40 frames (8 Hz × 5 s);
* the target standardizer is fit on the TRAIN pool only (adding a clip changes
  the stats — train-only is a load-bearing choice, not cosmetic).

F2 (3a/3b LR groups):
* 3a optimizes the connector ``{PMA, projector}`` only — the encoder is absent
  from the param groups and grad never reaches it;
* 3b builds 3 discriminative-LR groups in the locked {base/10, base/3, base}
  ratio and grad reaches the encoder.

Plus the E4 handoff (encoder+pma+projector export; encoder-only P2 load leaves
the connector fresh; 3a→3b round-trip) and the 8 Hz rate-match guard.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.bt_alignment.teacher_cache import (
    TargetStandardizer,
    fit_channel_stats,
)
from speech_decoding.experiments.v14_phase3 import (
    V14Phase3DistillModule,
    V14Phase3Experiment,
)
from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel

# 8 Hz × 5 s = 40 teacher frames; the student front-end must emit T_p=40 to
# align with the Whisper pool. With non-overlap patch_kernel_time=2 that means
# n_time_bins=80.
_WHISPER_D = 1280
_N_TIME_BINS = 80
_T_P = 40


def _optim_config(lr: float = 1e-3) -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "Adam", "lr": lr})


def _make_tiny_encoder(
    *,
    n_freq_bins: int = 4,
    n_time_bins: int = _N_TIME_BINS,
    k_parcels: int = 5,
    m_sub_slots: int = 1,
    d_model: int = 16,
    n_heads: int = 4,
    depth_self_attn: int = 1,
    n_token_blocks: int = 1,
    patch_kernel_freq: int = 2,
    patch_kernel_time: int = 2,
) -> V14ParcelPerceiverModel:
    return V14ParcelPerceiverModel(
        n_freq_bins=n_freq_bins,
        n_time_bins=n_time_bins,
        k_parcels=k_parcels,
        m_sub_slots=m_sub_slots,
        d_model=d_model,
        n_heads=n_heads,
        depth_self_attn=depth_self_attn,
        n_token_blocks=n_token_blocks,
        patch_kernel_freq=patch_kernel_freq,
        patch_kernel_time=patch_kernel_time,
    )


def _make_synthetic_batch(
    *, B: int = 2, C: int = 5, K: int = 5, whisper_d: int = _WHISPER_D,
) -> SimpleNamespace:
    torch.manual_seed(0)
    electrode_tokens = torch.randn(B, C, _N_TIME_BINS, 4)
    support = torch.zeros(B, C, K)
    for i in range(min(C, K)):
        support[:, i, i] = 1.0
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    # Raw 50 Hz Whisper cache feature (B, 250, 1280) — the module pools it.
    whisper_target = torch.randn(B, 250, whisper_d)
    return SimpleNamespace(
        data={
            "electrode_tokens": electrode_tokens,
            "support": support,
            "valid_mask": valid_mask,
            "whisper_target": whisper_target,
        }
    )


def _make_module(
    encoder=None, *, stage: str = "3a", standardizer=None, **kwargs,
) -> V14Phase3DistillModule:
    if encoder is None:
        encoder = _make_tiny_encoder()
    return V14Phase3DistillModule(
        encoder=encoder,
        optim_config=_optim_config(),
        stage=stage,  # type: ignore[arg-type]
        pma_n_heads=4,
        standardizer=standardizer,
        **kwargs,
    )


def _identity_standardizer(d: int = _WHISPER_D) -> TargetStandardizer:
    return TargetStandardizer(torch.zeros(d), torch.ones(d))


def _write_cache(path: Path, *, seed: int, d: int = _WHISPER_D) -> Path:
    gen = torch.Generator().manual_seed(seed)
    feat = torch.randn(250, d, generator=gen).to(torch.float16)
    torch.save(
        {
            "features": feat,
            "rate_hz": 50,
            "t0_movie_s": 0.0,
            "model": "openai/whisper-large-v3",
            "layer_merge": "mean_all",
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Construction / stage freeze
# ---------------------------------------------------------------------------


def test_construct_3a_freezes_encoder_connector_trains() -> None:
    module = _make_module(stage="3a")
    assert all(not p.requires_grad for p in module.encoder.parameters())
    assert all(p.requires_grad for p in module.pma.parameters())
    assert all(p.requires_grad for p in module.projector.parameters())


def test_construct_3b_unfreezes_encoder() -> None:
    module = _make_module(stage="3b")
    assert all(p.requires_grad for p in module.encoder.parameters())
    assert all(p.requires_grad for p in module.pma.parameters())
    assert all(p.requires_grad for p in module.projector.parameters())


def test_rejects_unknown_stage() -> None:
    with pytest.raises(ValueError, match="stage"):
        _make_module(stage="3c")


# ---------------------------------------------------------------------------
# F1 — finite Smooth-L1 in 1280-d; both sides 40 frames
# ---------------------------------------------------------------------------


def test_p3_step_finite_smooth_l1_1280d() -> None:
    module = _make_module(stage="3a", standardizer=_identity_standardizer())
    loss = module._step(_make_synthetic_batch().data)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    # Non-trivial: a zero/degenerate loss would also be finite and ≥ 0, so the
    # headline distillation test must assert the loss is actually positive.
    assert float(loss.detach()) > 0.0


def test_p3_loss_depends_on_the_student_not_only_the_teacher() -> None:
    """Anti-vacuity guard: zeroing the projector makes the student emit ~0, so
    the SmoothL1 against the same teacher must change. If it did not, the loss
    would not depend on the student path (an identity/dead projector would pass
    the finite-and-positive checks above)."""
    module = _make_module(stage="3a", standardizer=_identity_standardizer())
    batch = _make_synthetic_batch()
    real_loss = float(module._step(batch.data).detach())
    with torch.no_grad():
        for p in module.projector.parameters():
            p.zero_()
    zeroed_loss = float(module._step(batch.data).detach())
    assert real_loss > 0.0
    assert abs(real_loss - zeroed_loss) > 1e-6


def test_student_and_teacher_both_1280d_and_40_frames() -> None:
    module = _make_module(stage="3a", standardizer=_identity_standardizer())
    batch = _make_synthetic_batch()
    enc_kwargs = module._extract_encoder_kwargs(batch.data)
    student = module._student_readout(enc_kwargs)
    teacher = module._teacher_target(batch.data)
    assert student.shape == (2, _T_P, _WHISPER_D)
    assert teacher.shape == (2, _T_P, _WHISPER_D)


def test_step_finite_without_standardizer() -> None:
    """R-no-target-standardize path: a None standardizer is a no-op, still
    finite (the raw 1280-d pooled target distills directly)."""
    module = _make_module(stage="3a", standardizer=None)
    loss = module._step(_make_synthetic_batch().data)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# F1 — target standardizer fit on the TRAIN pool only
# ---------------------------------------------------------------------------


def test_standardizer_fit_train_pool_only(tmp_path: Path) -> None:
    train_paths = [
        _write_cache(tmp_path / "c0.pt", seed=1),
        _write_cache(tmp_path / "c1.pt", seed=2),
    ]
    val_path = _write_cache(tmp_path / "c2.pt", seed=3)

    stats_train = fit_channel_stats(train_paths, d_model=_WHISPER_D)
    stats_with_val = fit_channel_stats(train_paths + [val_path], d_model=_WHISPER_D)

    # Train-only is a real choice: folding the val clip into the fit moves the
    # stats. (If they were identical, "train-only" would be cosmetic.)
    assert not torch.allclose(stats_train["mean"], stats_with_val["mean"])

    # The module distills against the TRAIN-only standardizer and stays finite.
    standardizer = TargetStandardizer(stats_train["mean"], stats_train["inv_std"])
    module = _make_module(stage="3a", standardizer=standardizer)
    loss = module._step(_make_synthetic_batch().data)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# F1 — grad scope by stage
# ---------------------------------------------------------------------------


def test_3a_grad_reaches_connector_not_encoder() -> None:
    module = _make_module(stage="3a", standardizer=_identity_standardizer())
    module._step(_make_synthetic_batch().data).backward()
    # Connector trains.
    assert module.pma.query.grad is not None
    assert torch.isfinite(module.pma.query.grad).all()
    assert any(p.grad is not None for p in module.projector.parameters())
    # Encoder is frozen — no grad anywhere.
    assert all(p.grad is None for p in module.encoder.parameters())


def test_3b_grad_reaches_encoder_and_connector() -> None:
    module = _make_module(stage="3b", standardizer=_identity_standardizer())
    module._step(_make_synthetic_batch().data).backward()
    assert module.encoder.encoder_ln.weight.grad is not None
    assert torch.isfinite(module.encoder.encoder_ln.weight.grad).all()
    assert module.pma.query.grad is not None


# ---------------------------------------------------------------------------
# F2 — 3a/3b optimizer param groups
# ---------------------------------------------------------------------------


def test_3a_param_groups_connector_only() -> None:
    module = _make_module(stage="3a")
    params = module._phase_param_groups()
    # Flat list of params (no group dicts).
    assert all(isinstance(p, torch.nn.Parameter) for p in params)
    ids = {id(p) for p in params}
    connector_ids = {
        id(p)
        for p in list(module.pma.parameters()) + list(module.projector.parameters())
    }
    assert ids == connector_ids
    # No encoder param is optimized at 3a.
    assert not any(id(p) in ids for p in module.encoder.parameters())


def test_3b_param_groups_three_groups_discriminative_lr() -> None:
    base_lr = 1e-3
    module = V14Phase3DistillModule(
        encoder=_make_tiny_encoder(),
        optim_config=_optim_config(base_lr),
        stage="3b",
        pma_n_heads=4,
    )
    groups = module._phase_param_groups()
    assert len(groups) == 3
    frontend_g, parcel_g, connector_g = groups
    assert frontend_g["lr"] == pytest.approx(base_lr * 0.1)
    assert parcel_g["lr"] == pytest.approx(base_lr / 3.0)
    assert connector_g["lr"] == pytest.approx(base_lr)

    # The connector group is exactly {PMA, projector}; encoder params split
    # across the front-end + parcel groups (partition covers every encoder param).
    connector_ids = {
        id(p)
        for p in list(module.pma.parameters()) + list(module.projector.parameters())
    }
    assert {id(p) for p in connector_g["params"]} == connector_ids
    enc_in_groups = {id(p) for p in frontend_g["params"]} | {
        id(p) for p in parcel_g["params"]
    }
    assert enc_in_groups == {id(p) for p in module.encoder.parameters()}


# ---------------------------------------------------------------------------
# 8 Hz rate-match guard
# ---------------------------------------------------------------------------


def test_rate_mismatch_raises_clear_error() -> None:
    # n_time_bins=8 → T_p=4 ≠ 40 teacher frames → loud 8 Hz-contract error.
    encoder = _make_tiny_encoder(n_time_bins=8)
    module = _make_module(encoder, stage="3a", standardizer=_identity_standardizer())
    batch = _make_synthetic_batch()
    batch.data["electrode_tokens"] = torch.randn(2, 5, 8, 4)
    with pytest.raises(ValueError, match="8 Hz"):
        module._step(batch.data)


def test_missing_whisper_target_raises() -> None:
    module = _make_module(stage="3a", standardizer=_identity_standardizer())
    batch = _make_synthetic_batch()
    del batch.data["whisper_target"]
    with pytest.raises(KeyError, match="whisper_target"):
        module._step(batch.data)


# ---------------------------------------------------------------------------
# E4 cross-phase handoff
# ---------------------------------------------------------------------------


def test_transferable_state_has_encoder_pma_projector() -> None:
    module = _make_module(stage="3a")
    state = module.transferable_state()
    assert set(state.keys()) == {"encoder", "pma", "projector"}


def test_load_from_p2_encoder_only_keeps_connector_fresh() -> None:
    # P2 exports encoder only; loading it must succeed and leave PMA/projector
    # at their fresh init (the 3a warmup starts the connector cold).
    src = _make_module(stage="3a")
    with torch.no_grad():
        src.encoder.encoder_ln.weight.fill_(3.0)
    p2_state = {"encoder": src.encoder.state_dict()}

    dst = _make_module(stage="3a")
    pma_query_before = dst.pma.query.detach().clone()
    dst.load_transferable_state(p2_state)
    torch.testing.assert_close(
        dst.encoder.encoder_ln.weight.detach(), torch.full_like(
            dst.encoder.encoder_ln.weight, 3.0,
        ),
    )
    # Connector untouched (no 'pma'/'projector' key in the P2 snapshot).
    torch.testing.assert_close(dst.pma.query.detach(), pma_query_before)
    # 3a freeze re-applied after load.
    assert all(not p.requires_grad for p in dst.encoder.parameters())


def test_load_roundtrip_3a_to_3b_carries_connector() -> None:
    src = _make_module(stage="3a")
    with torch.no_grad():
        src.pma.query.fill_(0.5)
    state = src.transferable_state()

    dst = _make_module(stage="3b")
    dst.load_transferable_state(state)
    torch.testing.assert_close(
        dst.pma.query.detach(), torch.full_like(dst.pma.query, 0.5),
    )
    # 3b unfreezes the encoder even after load.
    assert all(p.requires_grad for p in dst.encoder.parameters())


def test_load_missing_encoder_raises() -> None:
    module = _make_module(stage="3a")
    with pytest.raises(KeyError, match="encoder"):
        module.load_transferable_state({"pma": module.pma.state_dict()})


# ---------------------------------------------------------------------------
# V14Phase3Experiment config-layer guards + standardizer build
#
# Construct via ``model_construct`` (skips field validation, fills defaults,
# and triggers ``model_post_init`` in pydantic v2) so the guards are exercised
# without standing up the full data/brain-module pipeline — mirrors the
# V14JointExperiment tests in test_v14_joint.py. Full P3 end-to-end
# construction (``_build_brain_module``) is the WS-I2 capstone.
# ---------------------------------------------------------------------------


def test_experiment_rejects_phase_outside_3a_3b() -> None:
    """The experiment models Phase-3 distillation only (phase ∈ {3a, 3b}); a
    Phase-4 (or any other) phase must raise at config time."""
    for bad_phase in (1, 2, 4):
        with pytest.raises(ValueError, match="3a"):
            V14Phase3Experiment.model_construct(
                phase=bad_phase, target_standardize=False,
            )


def test_experiment_requires_channel_stats_when_standardizing() -> None:
    """B33 default target_standardize=True needs the train-only stats path; a
    missing path is a config error, not a silent raw-target run."""
    with pytest.raises(ValueError, match="channel_stats_path"):
        V14Phase3Experiment.model_construct(
            phase="3a", target_standardize=True, channel_stats_path=None,
        )


def test_experiment_accepts_valid_3b_without_standardize() -> None:
    """R-no-target-standardize: target_standardize=False needs no stats path."""
    xp = V14Phase3Experiment.model_construct(
        phase="3b", target_standardize=False,
    )
    xp.model_post_init(None)  # no raise
    assert xp.phase == "3b"


def test_build_standardizer_none_when_disabled() -> None:
    xp = V14Phase3Experiment.model_construct(
        phase="3a", target_standardize=False,
    )
    assert xp._build_standardizer() is None


def test_build_standardizer_roundtrips_train_only_stats(tmp_path: Path) -> None:
    """With target_standardize=True the experiment loads {mean, inv_std} from
    channel_stats_path and wires a TargetStandardizer. Identity stats (zero
    mean, unit inv_std) make it a passthrough — proves the load + construct
    path, not a no-op."""
    stats_path = tmp_path / "channel_stats.pt"
    torch.save(
        {"mean": torch.zeros(_WHISPER_D), "inv_std": torch.ones(_WHISPER_D)},
        stats_path,
    )
    xp = V14Phase3Experiment.model_construct(
        phase="3a", target_standardize=True, channel_stats_path=str(stats_path),
    )
    standardizer = xp._build_standardizer()
    assert standardizer is not None
    x = torch.randn(2, _T_P, _WHISPER_D)
    torch.testing.assert_close(standardizer(x), x)
