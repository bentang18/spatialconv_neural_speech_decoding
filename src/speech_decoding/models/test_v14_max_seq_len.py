"""IE02: RoPE table sized to ``max_seq_len`` so the same model instance can
ingest clips of different T (Phase-1/2 at 5 s vs Phase-3/4 at 1 s)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel


def test_max_seq_len_default_matches_n_time_bins() -> None:
    """Backward-compat: when ``max_seq_len`` is not passed, the raw-frame
    limit equals ``n_time_bins`` (semantic unchanged across the FE-02 patch
    stem refactor).

    After FE-02 the latent-stack RoPE table is sized in POST-PATCH frames
    ``T_p = max_n_time_patches``, NOT raw frames. ``max_seq_len`` continues
    to track the raw-frame input ceiling so callers reason about the input
    contract independently of the patch stem.
    """
    model = V14ParcelPerceiverModel(
        n_freq_bins=4, n_time_bins=8, k_parcels=4,
        d_model=32, n_heads=4, depth_self_attn=1,
        m_sub_slots=1, n_token_blocks=1,
    )
    assert model.max_seq_len == 8
    expected_T_p = model.patch_stem.n_time_patches(8)
    assert model.max_n_time_patches == expected_T_p
    assert model.key_rope.shape[1] == expected_T_p


def test_max_seq_len_overrides_rope_table_size() -> None:
    model = V14ParcelPerceiverModel(
        n_freq_bins=4, n_time_bins=8, k_parcels=4,
        d_model=32, n_heads=4, depth_self_attn=1,
        m_sub_slots=1, n_token_blocks=1,
        max_seq_len=64,
    )
    assert model.max_seq_len == 64
    expected_T_p = model.patch_stem.n_time_patches(64)
    assert model.max_n_time_patches == expected_T_p
    assert model.key_rope.shape[1] == expected_T_p


def test_forward_accepts_smaller_T_than_n_time_bins() -> None:
    """The encoder must accept T < n_time_bins when max_seq_len >= n_time_bins.
    (Phase-3/4 short-clip readout against a model declared at the longer
    pretraining T.)

    FE-02 floor: input T must be ≥ ``patch_kernel_time`` so the Conv2d stem
    can emit at least one time-patch. With the default kernel_time=2,
    the smallest valid T_in is 2.
    """
    n_freq, n_time, K = 4, 8, 4
    model = V14ParcelPerceiverModel(
        n_freq_bins=n_freq, n_time_bins=n_time, k_parcels=K,
        d_model=32, n_heads=4, depth_self_attn=1,
        m_sub_slots=1, n_token_blocks=1,
        max_seq_len=n_time,
    )
    B, C = 2, 3
    for T_in in (2, 4, n_time):
        tokens = torch.randn(B, C, T_in, n_freq)
        support = torch.rand(B, C, K) + 0.1
        out = model(tokens, support)
        T_p = model.patch_stem.n_time_patches(T_in)
        assert out.shape == (B, K * 1, T_p, 32)


def test_forward_rejects_T_above_max_seq_len() -> None:
    model = V14ParcelPerceiverModel(
        n_freq_bins=4, n_time_bins=8, k_parcels=4,
        d_model=32, n_heads=4, depth_self_attn=1,
        m_sub_slots=1, n_token_blocks=1,
        max_seq_len=8,
    )
    tokens = torch.randn(1, 2, 9, 4)
    support = torch.rand(1, 2, 4) + 0.1
    with pytest.raises(ValueError, match="max_seq_len"):
        model(tokens, support)


def test_max_seq_len_must_not_be_smaller_than_n_time_bins() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        V14ParcelPerceiverModel(
            n_freq_bins=4, n_time_bins=8, k_parcels=4,
            d_model=32, n_heads=4, depth_self_attn=1,
            m_sub_slots=1, n_token_blocks=1,
            max_seq_len=4,
        )
