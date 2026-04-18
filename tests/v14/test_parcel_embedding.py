"""Unit tests for the soft parcel embedding (Task C2, Stage 3)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.config import SoftParcelEmbeddingConfig
from speech_decoding.v14.parcel_embedding import SoftParcelEmbedding


class TestSoftParcelEmbedding:
    def test_output_shape(self) -> None:
        m = SoftParcelEmbedding()
        support = torch.rand(2, 8, 15)
        out = m(support)
        assert out.shape == (2, 8, 64)

    def test_zero_support_produces_zero_output(self) -> None:
        m = SoftParcelEmbedding()
        support = torch.zeros(1, 4, 15)
        out = m(support)
        assert torch.all(out == 0)

    def test_gradient_flows_to_P_emb(self) -> None:
        m = SoftParcelEmbedding()
        support = torch.rand(1, 4, 15)
        m(support).sum().backward()
        assert m.P_emb.grad is not None
        assert torch.isfinite(m.P_emb.grad).all()

    def test_xavier_init_not_zero(self) -> None:
        m = SoftParcelEmbedding()
        assert not torch.all(m.P_emb == 0)

    def test_rejects_wrong_last_dim(self) -> None:
        m = SoftParcelEmbedding()
        with pytest.raises(ValueError, match="15"):
            m(torch.zeros(1, 4, 10))

    def test_custom_d_model(self) -> None:
        m = SoftParcelEmbedding(SoftParcelEmbeddingConfig(d_model=32, n_parcels=15))
        out = m(torch.rand(1, 4, 15))
        assert out.shape == (1, 4, 32)
