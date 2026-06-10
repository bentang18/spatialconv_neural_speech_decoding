"""B37 D7/D9 — :class:`V14JointBrainModule` ``ssl_mode="joint"`` SSL.

One masked student forward producing BOTH the M2 (front-end) and M4 (parcel)
taps, scored by TWO predictors under a single composite mask
(``L = L_M2 + λ·L_M4``). Pins (spec §3, tests lines 247-251):

  * the two predictors are built with the right identity axes + depths
    (M2: freq id, depth 3; M4: parcel+freq 2-axis id, depth 2) and the single
    staged ``predictor`` is gone;
  * the composite mask is band ∧ tube (M2 bands only on surviving parcels);
  * gradient reaches the stem from BOTH losses, and both predictors train;
  * the EMA teacher is updated exactly once per step and always sees the FULL
    input (no mask leak);
  * λ scales L_M4 in the total;
  * joint requires ``pool="mean"``;
  * both the M2 and M4 collapse-rank monitors fire (joint trains both).
"""

from __future__ import annotations

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_joint_module import (
    JointJepaBreakdown,
    V14JointBrainModule,
)
from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel


def _optim_config() -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3})


# Token grid F_p=(15-3)//3+1=5, T_p=(10-2)//2+1=5 — big enough that the M2
# band mask (time-band floor 2) leaves real VISIBLE context cells in every
# surviving parcel (a 2×2 grid would be all-masked → degenerate, no stem grad).
def _mean_encoder(**over) -> V14ParcelPerceiverModel:
    kw = dict(
        n_freq_bins=15, n_time_bins=10, k_parcels=6, m_sub_slots=1, d_model=32,
        n_heads=4, depth_self_attn=2, n_token_blocks=2, patch_kernel_freq=3,
        pool="mean",
    )
    kw.update(over)
    torch.manual_seed(0)
    return V14ParcelPerceiverModel(**kw)  # type: ignore[arg-type]


def _cross_attn_encoder() -> V14ParcelPerceiverModel:
    torch.manual_seed(0)
    return V14ParcelPerceiverModel(
        n_freq_bins=15, n_time_bins=10, k_parcels=6, m_sub_slots=1, d_model=32,
        n_heads=4, depth_self_attn=2, patch_kernel_freq=3,  # cross_attn default
    )


def _joint_module(encoder=None, **kwargs) -> V14JointBrainModule:
    if encoder is None:
        encoder = _mean_encoder()
    return V14JointBrainModule(
        encoder=encoder, optim_config=_optim_config(), ssl_mode="joint", **kwargs,
    )


def _batch(*, B: int = 2, C: int = 6, T_bins: int = 10, F_bins: int = 15, K: int = 6):
    torch.manual_seed(1)
    electrode_tokens = torch.randn(B, C, T_bins, F_bins)
    # diagonal support → all K parcels covered (so the 0.20 tube masks exactly
    # 1 of 6 covered parcels while ≥3 stay visible — real default geometry).
    support = torch.zeros(B, C, K)
    for i in range(min(C, K)):
        support[:, i, i] = 1.0
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    return {
        "electrode_tokens": electrode_tokens,
        "support": support,
        "valid_mask": valid_mask,
    }


# --------------------------------------------------------------------------- #
# construction — two predictors, D9 config
# --------------------------------------------------------------------------- #
def test_joint_builds_two_predictors_with_correct_config() -> None:
    enc = _mean_encoder()
    m = _joint_module(enc)
    assert m.predictor is None                          # the staged single is gone
    # M2 predictor: ORDERED freq id (mirrors encoder.freq_pos), depth 3, 1 axis.
    assert m.m2_predictor.n_identity == enc.n_freq_patches
    assert m.m2_predictor.id_pos == enc.freq_pos
    assert m.m2_predictor.n_identity_2 is None
    assert len(m.m2_predictor.blocks) == 3
    # M4 predictor: 2-axis id = parcel (learned) + freq (sinusoidal), depth 2.
    assert m.m4_predictor.n_identity == enc.k_parcels
    assert m.m4_predictor.id_pos == "learned"
    assert m.m4_predictor.n_identity_2 == enc.n_freq_patches
    assert m.m4_predictor.id_pos_2 == "sinusoidal"
    assert len(m.m4_predictor.blocks) == 2


def test_joint_predictor_depths_are_configurable() -> None:
    m = _joint_module(m2_predictor_depth=4, m4_predictor_depth=3)
    assert len(m.m2_predictor.blocks) == 4
    assert len(m.m4_predictor.blocks) == 3


def test_joint_requires_mean_pool() -> None:
    with pytest.raises(ValueError, match="pool='mean'"):
        _joint_module(_cross_attn_encoder())


def test_joint_rejects_injected_single_predictor() -> None:
    from speech_decoding.models.v14_encoder import JepaPredictor
    enc = _mean_encoder()
    pred = JepaPredictor(enc.d_model, n_identity=enc.n_freq_patches, hidden=8,
                         n_heads=2, depth=1)
    with pytest.raises(ValueError, match="two predictors"):
        V14JointBrainModule(encoder=enc, optim_config=_optim_config(),
                            ssl_mode="joint", predictor=pred)


# --------------------------------------------------------------------------- #
# the joint step — composite loss
# --------------------------------------------------------------------------- #
def test_joint_step_returns_joint_breakdown() -> None:
    m = _joint_module()
    bd = m._step(_batch())
    assert isinstance(bd, JointJepaBreakdown)
    assert bd.phase == "joint"
    assert torch.isfinite(bd.total) and torch.isfinite(bd.m2_total)
    assert torch.isfinite(bd.m4_total)
    assert bd.n_masked_m2 > 0, "no band-masked M2 cells — composite mask empty"
    assert bd.n_masked_m4 > 0, "no tube-masked M4 cells — tube empty"
    assert bd.n_masked == bd.n_masked_m2 + bd.n_masked_m4


def test_joint_total_is_m2_plus_lambda_m4() -> None:
    for lam in (0.0, 1.0, 2.5):
        m = _joint_module(lambda_m4=lam)
        bd = m._step(_batch())
        torch.testing.assert_close(
            bd.total, bd.m2_total + lam * bd.m4_total, rtol=1e-6, atol=1e-6,
        )
    # λ=0 → the M4 term drops out of the total entirely.
    m0 = _joint_module(lambda_m4=0.0)
    bd0 = m0._step(_batch())
    torch.testing.assert_close(bd0.total, bd0.m2_total, rtol=1e-6, atol=1e-6)


def test_joint_grad_reaches_stem_from_both_losses() -> None:
    """The crux of D7: a SINGLE student forward, so BOTH the M2 and M4 losses
    must backprop into the shared per-parcel stem (not just one of them)."""
    m = _joint_module()
    stem = m.student.encoder.patch_stem

    def stem_grad_from(which: str) -> float:
        # Recompute a FRESH forward graph per loss so each backward is
        # independent (no retain_graph / double-backward). _step is fully
        # deterministic here — fixed batch seed + step-0 mask seed — so the two
        # forwards are identical, and this stays robust to global torch.compile
        # donated-buffer state if the full suite runs a compile test first.
        m.zero_grad(set_to_none=True)
        bd = m._step(_batch())
        getattr(bd, which).backward()
        return sum(
            float(p.grad.abs().sum()) for p in stem.parameters() if p.grad is not None
        )

    # M2-only and M4-only each move the stem → both losses reach it.
    assert stem_grad_from("m2_total") > 0.0, "M2 loss does not reach the stem"
    assert stem_grad_from("m4_total") > 0.0, "M4 loss does not reach the stem"


def test_joint_both_predictors_train() -> None:
    m = _joint_module()
    bd = m._step(_batch())
    bd.total.backward()
    for name, pred in (("m2", m.m2_predictor), ("m4", m.m4_predictor)):
        got = [
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for n, p in pred.named_parameters()
            if "id_embed" not in n  # learned id rows only update for sampled ids
        ]
        assert got and all(got), f"{name}_predictor received no gradient"


def test_joint_teacher_is_full_input_and_detached() -> None:
    """The teacher forward carries NO mask (the B7 guard would raise if a
    visibility mask leaked in) and its target is detached (no teacher grad)."""
    m = _joint_module()
    bd = m._step(_batch())               # would raise if a mask leaked to teacher
    bd.total.backward()
    for p in m.teacher.parameters():
        assert p.grad is None            # stop-grad on the EMA teacher


# --------------------------------------------------------------------------- #
# EMA discipline — one update per step
# --------------------------------------------------------------------------- #
def test_joint_ema_updates_once_per_step() -> None:
    m = _joint_module()
    calls = {"n": 0}
    real = m.teacher.update_from

    def counting(student):
        calls["n"] += 1
        return real(student)

    m.teacher.update_from = counting  # type: ignore[method-assign]
    m._step(_batch()).total.backward()
    m.on_before_zero_grad(optimizer=None)
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# composite mask sampled by the module
# --------------------------------------------------------------------------- #
def test_joint_module_composite_mask_band_only_on_surviving() -> None:
    m = _joint_module()
    data = _batch()
    tok, ptm = m._sample_composite_mask(
        electrode_tokens=data["electrode_tokens"], support=data["support"],
    )
    B, K, F_p, T_p = tok.shape
    assert ptm.shape == (B, K, T_p)
    tubed = ptm.any(dim=-1)                              # (B, K)
    # tube is whole-parcel-all-time; bands never fall on a tubed parcel.
    assert (ptm == tubed.unsqueeze(-1)).all()
    assert int((tok & tubed.unsqueeze(-1).unsqueeze(-1)).sum()) == 0


# --------------------------------------------------------------------------- #
# optimizer param groups + monitors
# --------------------------------------------------------------------------- #
def test_joint_param_groups_cover_encoder_and_both_predictors() -> None:
    m = _joint_module()
    groups = m._phase_param_groups()
    assert len(groups) == 1                              # single base-LR group (D8 in Chunk E)
    ids = {id(p) for p in groups[0]["params"]}
    for p in m.m2_predictor.parameters():
        assert id(p) in ids
    for p in m.m4_predictor.parameters():
        assert id(p) in ids
    # Every TRAINABLE encoder param is optimized; the mean-pool path freezes the
    # unused cross-attn latent-init embeds, which are correctly excluded.
    for n, p in m.student.encoder.named_parameters():
        if p.requires_grad:
            assert id(p) in ids, n
        else:
            assert id(p) not in ids, f"frozen {n} leaked into the optimizer"
    # No frozen param sits in the group at all.
    assert all(p.requires_grad for p in groups[0]["params"])


def test_meanpool_freezes_unused_latent_init_embeds() -> None:
    """The hard mean-pool never reads the cross-attn latent-init embeds, so
    they must be frozen (no grad → no DDP-static-graph unused-param hazard, no
    weight-decay drift) — but a cross_attn encoder keeps them trainable."""
    mean = _mean_encoder()
    assert not mean.learnable_parcel_embed.requires_grad
    cross = _cross_attn_encoder()
    assert cross.learnable_parcel_embed.requires_grad
    # And the frozen embed never receives a gradient on a joint step.
    m = _joint_module(mean)
    m._step(_batch()).total.backward()
    assert m.student.encoder.learnable_parcel_embed.grad is None


def test_joint_monitors_emit_both_m2_and_m4_rank() -> None:
    m = _joint_module()
    logged: dict[str, float] = {}
    m.log = lambda key, value, **_kw: logged.update(  # type: ignore[method-assign]
        {key: float(value.detach() if hasattr(value, "detach") else value)})
    m._monitor_from_step(_batch(), step_name="train")
    # joint probes BOTH collapse monitors (front-end M2 rank + parcel M4 rank).
    assert any(k.startswith("train_mon_frontend_rankme") for k in logged)
    assert any(k == "train_mon_rankme" or k.startswith("train_mon_rankme_") for k in logged)
    assert "train_mon_coverage_active_mean" in logged


def test_joint_training_step_logs_component_losses() -> None:
    m = _joint_module()
    logged: dict[str, float] = {}
    m.log = lambda key, value, **_kw: logged.update(  # type: ignore[method-assign]
        {key: float(value.detach() if hasattr(value, "detach") else value)})
    from types import SimpleNamespace
    out = m.training_step(SimpleNamespace(data=_batch()), batch_idx=0)
    assert torch.isfinite(out)
    for key in ("train_loss", "train_loss_m2", "train_loss_m4",
                "train_n_masked_m2", "train_n_masked_m4"):
        assert key in logged, key
