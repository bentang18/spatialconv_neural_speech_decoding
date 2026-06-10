"""#95 — A-JEPA fixed sinusoidal positional encoding on the FREQUENCY axis.

The frequency-patch axis is an ORDERED metric axis (low→high Hz), so — like
A-JEPA / AudioMAE (arXiv 2311.15830) — the default positional code is a fixed
1-D MAE sincos table (a non-learned buffer): an absolute per-patch code PLUS a
free smoothness/ordering prior. The pre-lock learned table survives as the
``R-freq-learned-embed`` sister (``freq_pos="learned"`` / ``id_pos="learned"``).

The lock touches BOTH freq positional sites:

  * the ENCODER front-end ``freq_embed`` (added per freq patch, broadcast over
    time) — ``V14ParcelPerceiverModel(freq_pos=...)``;
  * the P1 PREDICTOR identity tag (masked queries tagged by freq-patch) —
    ``JepaPredictor(id_pos=...)``, driven from ``encoder.freq_pos`` by the joint
    module.

The PARCEL axis stays learned everywhere (unordered anatomy — sincos would
impose a false metric); TIME stays RoPE (relative). So the P2 predictor's
identity tag is ALWAYS a learned table regardless of ``encoder.freq_pos``.

Pinned here:
  * ``_sincos_1d`` is deterministic, correct-shape, even-dim-guarded.
  * sinusoidal ⇒ a FIXED non-grad buffer absent from ``.parameters()`` and equal
    to ``_sincos_1d`` at both sites; learned ⇒ a trainable ``Parameter`` /
    ``nn.Embedding``.
  * the encoder forward still runs under the buffer (the additive site is
    mode-agnostic), and is deterministic across two same-config builds.
  * the predictor forward runs + stays finite under the sincos id-tag.
  * the joint module mirrors ``encoder.freq_pos`` onto the P1 predictor and
    forces the P2 predictor to ``learned``.
  * invalid mode strings fail loud at construction.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from speech_decoding.models.v14_encoder import (
    JepaPredictor,
    V14ParcelPerceiverModel,
    _sincos_1d,
)


# --------------------------------------------------------------------------- #
# _sincos_1d
# --------------------------------------------------------------------------- #
def test_sincos_shape_and_determinism() -> None:
    a = _sincos_1d(10, 32)
    b = _sincos_1d(10, 32)
    assert a.shape == (10, 32)
    assert a.requires_grad is False
    torch.testing.assert_close(a, b)  # no RNG → bit-identical


def test_sincos_odd_dim_raises() -> None:
    with pytest.raises(ValueError, match="even"):
        _sincos_1d(8, 7)


def test_sincos_first_position_is_sin0_cos0() -> None:
    """Position 0 → all sin terms 0, all cos terms 1 (the MAE/Vaswani anchor)."""
    out = _sincos_1d(4, 16)
    half = 16 // 2
    torch.testing.assert_close(out[0, :half], torch.zeros(half))
    torch.testing.assert_close(out[0, half:], torch.ones(half))


# --------------------------------------------------------------------------- #
# encoder front-end freq_embed
# --------------------------------------------------------------------------- #
def _enc_kw() -> dict:
    # patch_kernel_freq=3 avoids the FE-RAW-1 F=50 guard; d_model even for sincos.
    return {
        "n_freq_bins": 6,
        "n_time_bins": 4,
        "k_parcels": 6,
        "d_model": 32,
        "n_heads": 4,
        "depth_self_attn": 2,
        "m_sub_slots": 1,
        "patch_kernel_freq": 3,
    }


def test_encoder_default_freq_pos_is_sinusoidal_buffer() -> None:
    enc = V14ParcelPerceiverModel(**_enc_kw())
    assert enc.freq_pos == "sinusoidal"
    # A buffer, not a Parameter: fixed, no grad, out of the param set.
    assert not isinstance(enc.freq_embed, nn.Parameter)
    assert enc.freq_embed.requires_grad is False
    param_ids = {id(p) for p in enc.parameters()}
    assert id(enc.freq_embed) not in param_ids
    # Exactly the MAE sincos table over the F_p freq patches.
    torch.testing.assert_close(
        enc.freq_embed, _sincos_1d(enc.n_freq_patches, enc.d_model)
    )


def test_encoder_learned_freq_pos_is_parameter() -> None:
    enc = V14ParcelPerceiverModel(**_enc_kw(), freq_pos="learned")
    assert enc.freq_pos == "learned"
    assert isinstance(enc.freq_embed, nn.Parameter)
    assert enc.freq_embed.requires_grad is True
    param_ids = {id(p) for p in enc.parameters()}
    assert id(enc.freq_embed) in param_ids


def test_encoder_invalid_freq_pos_raises() -> None:
    with pytest.raises(ValueError, match="freq_pos"):
        V14ParcelPerceiverModel(**_enc_kw(), freq_pos="rope")  # type: ignore[arg-type]


def test_encoder_forward_runs_under_sincos_and_is_deterministic() -> None:
    """The additive freq site is mode-agnostic, so the buffer path forwards
    cleanly; two same-config sincos builds give an identical freq table (the
    only freq-positional state is the deterministic buffer)."""
    torch.manual_seed(0)
    e1 = V14ParcelPerceiverModel(**_enc_kw())
    torch.manual_seed(0)
    e2 = V14ParcelPerceiverModel(**_enc_kw())
    torch.testing.assert_close(e1.freq_embed, e2.freq_embed)

    B, C = 2, 8
    kw = _enc_kw()
    g = torch.Generator().manual_seed(1)
    et = torch.randn(B, C, kw["n_time_bins"], kw["n_freq_bins"], generator=g)
    support = torch.zeros(B, C, kw["k_parcels"])
    for c in range(C):
        support[:, c, c % kw["k_parcels"]] = 1.0
    valid = torch.ones(B, C, dtype=torch.bool)
    out = e1(et, support, valid_mask=valid)
    # Whatever the readout tensor is, it must be finite (smoke test).
    leaf = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.isfinite(leaf).all()


# --------------------------------------------------------------------------- #
# predictor identity tag (id_pos)
# --------------------------------------------------------------------------- #
def _predictor(*, id_pos: str, n_identity: int = 4, hidden: int = 32) -> JepaPredictor:
    torch.manual_seed(0)
    return JepaPredictor(
        d_model=16,
        n_identity=n_identity,
        hidden=hidden,
        n_heads=4,
        depth=2,
        max_time_patches=6,
        id_pos=id_pos,  # type: ignore[arg-type]
    )


def test_predictor_default_id_pos_is_learned() -> None:
    pred = _predictor(id_pos="learned")
    assert pred.id_pos == "learned"
    assert isinstance(pred.id_embed, nn.Embedding)
    assert pred._id_table is None
    # The learned embedding weight is in the param set.
    param_ids = {id(p) for p in pred.parameters()}
    assert id(pred.id_embed.weight) in param_ids


def test_predictor_sinusoidal_id_pos_is_fixed_buffer() -> None:
    n_identity, hidden = 10, 32
    pred = _predictor(id_pos="sinusoidal", n_identity=n_identity, hidden=hidden)
    assert pred.id_pos == "sinusoidal"
    assert pred.id_embed is None
    assert pred._id_table is not None
    assert pred._id_table.requires_grad is False
    # Fixed sincos over the n_identity freq-patch slots, in predictor `hidden`-d.
    torch.testing.assert_close(pred._id_table, _sincos_1d(n_identity, hidden))
    # The table is a buffer → absent from .parameters().
    param_ids = {id(p) for p in pred.parameters()}
    assert id(pred._id_table) not in param_ids


def test_predictor_id_tag_dispatches_on_mode() -> None:
    ids = torch.tensor([0, 1, 2, 1])
    learned = _predictor(id_pos="learned")
    torch.testing.assert_close(learned._id_tag(ids), learned.id_embed(ids))
    sincos = _predictor(id_pos="sinusoidal")
    torch.testing.assert_close(sincos._id_tag(ids), sincos._id_table[ids])


def test_predictor_invalid_id_pos_raises() -> None:
    with pytest.raises(ValueError, match="id_pos"):
        _predictor(id_pos="rope")


def test_predictor_sinusoidal_forward_finite() -> None:
    """Full ragged P1-style forward under the sincos id-tag stays finite."""
    B, N, d, F_p, max_time = 3, 12, 16, 4, 6
    pred = JepaPredictor(
        d_model=d, n_identity=F_p, hidden=32, n_heads=4, depth=2,
        max_time_patches=max_time, id_pos="sinusoidal",
    )
    torch.manual_seed(1)
    context = torch.randn(B, N, d)
    out = pred(
        context,
        context_time_ids=torch.randint(0, max_time, (N,)),
        query_time_ids=torch.randint(0, max_time, (N,)),
        query_id=torch.randint(0, F_p, (N,)),
        context_key_padding_mask=torch.rand(B, N) < 0.4,
        query_valid=torch.rand(B, N) < 0.5,
    )
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# joint-module integration: P1 mirrors encoder.freq_pos; P2 always learned
# --------------------------------------------------------------------------- #
def _joint_kw() -> dict:
    # d_model 256 keeps the predictor-hidden sincos (128) + parcel grid sane;
    # F=30/kernel-3 → F_p=10 freq patches.
    return {
        "n_freq_bins": 30,
        "n_time_bins": 80,
        "k_parcels": 80,
        "d_model": 256,
        "n_heads": 8,
        "depth_self_attn": 2,
        "m_sub_slots": 1,
        "n_token_blocks": 2,
        "patch_kernel_freq": 3,
        "patch_kernel_time": 2,
    }


def _joint_module(*, phase: str, freq_pos: str):
    from neuraltrain.optimizers import LightningOptimizer
    from neuraltrain.optimizers.base import AdamW

    from speech_decoding.experiments.v14_joint_module import V14JointBrainModule

    enc = V14ParcelPerceiverModel(**_joint_kw(), freq_pos=freq_pos)
    return V14JointBrainModule(
        encoder=enc,
        optim_config=LightningOptimizer(optimizer=AdamW(lr=1e-3)),
        phase=phase,  # type: ignore[arg-type]
    )


def test_p1_predictor_mirrors_encoder_sinusoidal() -> None:
    m = _joint_module(phase="p1", freq_pos="sinusoidal")
    assert m.predictor.id_pos == "sinusoidal"
    assert m.predictor._id_table is not None
    assert m.predictor.id_embed is None


def test_p1_predictor_mirrors_encoder_learned() -> None:
    m = _joint_module(phase="p1", freq_pos="learned")
    assert m.predictor.id_pos == "learned"
    assert isinstance(m.predictor.id_embed, nn.Embedding)
    assert m.predictor._id_table is None


def test_p2_predictor_always_learned_even_under_sincos_encoder() -> None:
    """Parcel axis is unordered anatomy → the P2 identity tag stays learned
    regardless of the encoder's (freq-axis) sincos default."""
    m = _joint_module(phase="p2", freq_pos="sinusoidal")
    assert m.predictor.id_pos == "learned"
    assert isinstance(m.predictor.id_embed, nn.Embedding)
    assert m.predictor._id_table is None
