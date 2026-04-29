"""Shape + overfit tests for NeuralFieldPerceiverPerPhoneme (plan P4)."""

from __future__ import annotations

from dataclasses import replace

import torch

from speech_decoding.training.config import PerPhonemeConfig
from speech_decoding.studies.cogan_ps.dataset import (
    BOS_TOKEN,
    N_TIER1_PARCELS,
    T_RAW_SAMPLES,
)
from speech_decoding.models.phoneme import NeuralFieldPerceiverPerPhoneme


def _fake_batch(
    *, B: int = 4, N_e: int = 16, H_p: int = 8, W_p: int = 16, seed: int = 0
) -> dict:
    """Fake per-phoneme batch shaped like the collator output."""
    torch.manual_seed(seed)
    signal = torch.randn(B, N_e, T_RAW_SAMPLES)
    rows = torch.arange(H_p).repeat_interleave(W_p)[:N_e]
    cols = torch.arange(W_p).repeat(H_p)[:N_e]
    layout_per = torch.stack([rows, cols], dim=1)
    layout = layout_per.unsqueeze(0).expand(B, -1, -1).contiguous()
    active = torch.ones(B, N_e, dtype=torch.bool)
    support = torch.rand(B, N_e, N_TIER1_PARCELS)
    labels = torch.randint(0, 9, (B,), dtype=torch.long)
    prev_tokens = torch.tensor(
        [BOS_TOKEN, 0, 1, 2][:B], dtype=torch.long
    )[:B]
    # Pad if B > 4
    if B > 4:
        extra = torch.randint(0, 9, (B - 4,), dtype=torch.long)
        prev_tokens = torch.cat([prev_tokens, extra], dim=0)
    return {
        "signal": signal,
        "electrode_grid_layout": layout,
        "electrode_grid_shape": (H_p, W_p),
        "electrode_active_mask": active,
        "support": support,
        "labels": labels,
        "prev_tokens": prev_tokens,
        "phoneme_pos": torch.zeros(B, dtype=torch.long),
        "trial_id": torch.arange(B, dtype=torch.long),
        "patient_id": "SXX",
    }


class TestForwardShape:
    def test_full_128_strip_shapes(self) -> None:
        model = NeuralFieldPerceiverPerPhoneme()
        batch = _fake_batch(B=4, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (4, 9)

    def test_256_grid_shapes(self) -> None:
        model = NeuralFieldPerceiverPerPhoneme()
        batch = _fake_batch(B=2, N_e=256, H_p=16, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_encode_memory_shape_per_cell(self) -> None:
        """Default (per_cell) memory is (B, n_cells * T_tokens, d)."""
        model = NeuralFieldPerceiverPerPhoneme()
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        memory = model.encode_memory(batch)
        assert memory.shape == (2, 32 * 11, 32)

    def test_encode_memory_shape_flat(self) -> None:
        """Flat (ablation) memory is (B, T_tokens, d). Conv1d 130→11 tokens; d=32."""
        cfg = replace(PerPhonemeConfig(), temporal_frontend="flat")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        memory = model.encode_memory(batch)
        assert memory.shape == (2, 11, 32)


class TestParamBudget:
    def test_total_params_per_cell_near_plan_target(self) -> None:
        """per_cell default matches the original ~45k plan target."""
        model = NeuralFieldPerceiverPerPhoneme()
        total = sum(p.numel() for p in model.parameters())
        assert 30_000 <= total <= 60_000, f"got {total}, expected ~45k"

    def test_total_params_flat_ablation(self) -> None:
        """Flat bakes spatial mixing into the temporal conv — ~285k."""
        cfg = replace(PerPhonemeConfig(), temporal_frontend="flat")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        total = sum(p.numel() for p in model.parameters())
        assert 250_000 <= total <= 320_000, f"got {total}, expected ~285k"


class TestPartialConv:
    """Partial-conv renormalization (Liu 2018) — masking_mode='partial_conv'."""

    def test_equivalent_to_zero_fill_when_fully_active(self) -> None:
        """All-active grid → mask_sum == k² everywhere → scale == 1 everywhere.
        Memory must match bit-for-bit between zero_fill and partial_conv.
        """
        torch.manual_seed(0)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)

        cfg_zero = PerPhonemeConfig()
        cfg_part = replace(PerPhonemeConfig(), masking_mode="partial_conv")

        model_zero = NeuralFieldPerceiverPerPhoneme(cfg_zero)
        model_part = NeuralFieldPerceiverPerPhoneme(cfg_part)
        # Copy weights so only the masking path differs.
        model_part.load_state_dict(model_zero.state_dict())

        model_zero.eval()
        model_part.eval()
        with torch.no_grad():
            m_zero = model_zero.encode_memory(batch)
            m_part = model_part.encode_memory(batch)
        assert torch.allclose(m_zero, m_part, atol=1e-5)

    def test_renormalizes_when_artifacts_present(self) -> None:
        """With artifacts, partial_conv must diverge from zero_fill.
        Specifically, positions with partially-valid RFs get scale > 1.
        """
        torch.manual_seed(0)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        active = batch["electrode_active_mask"].clone()
        # Mark ~20% of electrodes as artifacts.
        active[:, ::5] = False
        batch["electrode_active_mask"] = active

        cfg_zero = PerPhonemeConfig()
        cfg_part = replace(PerPhonemeConfig(), masking_mode="partial_conv")

        model_zero = NeuralFieldPerceiverPerPhoneme(cfg_zero)
        model_part = NeuralFieldPerceiverPerPhoneme(cfg_part)
        model_part.load_state_dict(model_zero.state_dict())

        model_zero.eval()
        model_part.eval()
        with torch.no_grad():
            m_zero = model_zero.encode_memory(batch)
            m_part = model_part.encode_memory(batch)
        assert not torch.allclose(m_zero, m_part, atol=1e-4)

    def test_no_new_parameters(self) -> None:
        """Partial-conv adds zero learnable params."""
        n_zero = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme().parameters())
        cfg_part = replace(PerPhonemeConfig(), masking_mode="partial_conv")
        n_part = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme(cfg_part).parameters())
        assert n_part == n_zero


class TestReadoutModes:
    """CLS + hierarchical readouts — alternatives to mean_pool."""

    def test_cls_shapes(self) -> None:
        cfg = replace(PerPhonemeConfig(), readout_mode="cls")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_cls_memory_has_extra_token(self) -> None:
        """CLS-enabled memory has exactly one more token than mean_pool."""
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)

        cfg_mean = PerPhonemeConfig()
        cfg_cls = replace(PerPhonemeConfig(), readout_mode="cls")
        m_mean = NeuralFieldPerceiverPerPhoneme(cfg_mean).encode_memory(batch)
        m_cls = NeuralFieldPerceiverPerPhoneme(cfg_cls).encode_memory(batch)
        assert m_cls.shape == (2, m_mean.shape[1] + 1, 32)

    def test_cls_adds_d_model_params(self) -> None:
        n_base = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme().parameters())
        cfg = replace(PerPhonemeConfig(), readout_mode="cls")
        n_cls = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme(cfg).parameters())
        assert n_cls - n_base == 32  # one (1, 1, d=32) learnable CLS vector.

    def test_hierarchical_shapes(self) -> None:
        cfg = replace(PerPhonemeConfig(), readout_mode="hierarchical")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_hierarchical_adds_two_d_model_params(self) -> None:
        """Hierarchical adds exactly q_temporal (d) + q_cell (d) = 2·d params."""
        n_base = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme().parameters())
        cfg = replace(PerPhonemeConfig(), readout_mode="hierarchical")
        n_h = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme(cfg).parameters())
        assert n_h - n_base == 2 * 32

    def test_hierarchical_rejects_flat_frontend(self) -> None:
        cfg = replace(
            PerPhonemeConfig(), readout_mode="hierarchical", temporal_frontend="flat"
        )
        import pytest
        with pytest.raises(ValueError, match="requires temporal_frontend='per_cell'"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_hierarchical_atlas_shapes(self) -> None:
        cfg = replace(PerPhonemeConfig(), readout_mode="hierarchical_atlas")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_hierarchical_atlas_adds_only_temporal_query(self) -> None:
        """Atlas-anchored cell query reuses the existing parcel embedding, so
        it only adds the temporal query (d params) — no q_cell."""
        n_base = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme().parameters())
        cfg = replace(PerPhonemeConfig(), readout_mode="hierarchical_atlas")
        n_h = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme(cfg).parameters())
        assert n_h - n_base == 32

    def test_hierarchical_atlas_rejects_flat_frontend(self) -> None:
        cfg = replace(
            PerPhonemeConfig(),
            readout_mode="hierarchical_atlas",
            temporal_frontend="flat",
        )
        import pytest
        with pytest.raises(ValueError, match="requires temporal_frontend='per_cell'"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_hierarchical_atlas_rejects_no_parcel_embedding(self) -> None:
        cfg = replace(
            PerPhonemeConfig(),
            readout_mode="hierarchical_atlas",
            use_parcel_embedding=False,
        )
        import pytest
        with pytest.raises(ValueError, match="use_parcel_embedding=True"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_hierarchical_atlas_compute_cell_query(self) -> None:
        """compute_cell_query returns (B, n_cells, d) for hier_atlas, else None."""
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        cfg = replace(PerPhonemeConfig(), readout_mode="hierarchical_atlas")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        cq = model.compute_cell_query(batch)
        assert cq is not None and cq.shape == (2, 32, 32)
        model_base = NeuralFieldPerceiverPerPhoneme()
        assert model_base.compute_cell_query(batch) is None


class TestAttentionHeads:
    """Heads ablation — d=32 splits as 1×32, 2×16 (baseline), 4×8."""

    def _model(self, num_heads: int) -> NeuralFieldPerceiverPerPhoneme:
        from speech_decoding.training.config import BackboneConfig

        base = PerPhonemeConfig()
        cfg = replace(
            base,
            backbone=replace(
                base.backbone,
                num_heads=num_heads,
                head_dim=base.d_model // num_heads,
            )
            if isinstance(base.backbone, BackboneConfig)
            else base.backbone,
        )
        return NeuralFieldPerceiverPerPhoneme(cfg)

    def test_heads_1x32_forwards(self) -> None:
        model = self._model(1)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_heads_4x8_forwards(self) -> None:
        model = self._model(4)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_heads_param_count_matches_between_splits(self) -> None:
        """Heads × head_dim covers the same projection volume for any valid split."""
        m1 = self._model(1)
        m2 = self._model(2)
        m4 = self._model(4)
        n1 = sum(p.numel() for p in m1.parameters())
        n2 = sum(p.numel() for p in m2.parameters())
        n4 = sum(p.numel() for p in m4.parameters())
        assert n1 == n2 == n4  # Q/K/V/O are all d×d regardless of heads split


class TestFactorized2DSpatialPE:
    """Rigid-grid ViT-style PE on the (H_pool, W_pool) cell grid."""

    def test_forward_shape(self) -> None:
        cfg = replace(PerPhonemeConfig(), spatial_pe_mode="factorized_2d")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_param_count_matches_expected(self) -> None:
        """Adds H_pool*d/2 + W_pool*d/2 params at pool (4,8), d=32 → 4·16+8·16=192."""
        cfg = replace(PerPhonemeConfig(), spatial_pe_mode="factorized_2d")
        n_base = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme().parameters())
        n_pe = sum(p.numel() for p in NeuralFieldPerceiverPerPhoneme(cfg).parameters())
        d = PerPhonemeConfig().d_model
        H_pool, W_pool = PerPhonemeConfig().pool.pool_shape
        expected = H_pool * (d // 2) + W_pool * (d // 2)
        assert n_pe - n_base == expected

    def test_rejects_flat_frontend(self) -> None:
        cfg = replace(
            PerPhonemeConfig(),
            spatial_pe_mode="factorized_2d",
            temporal_frontend="flat",
        )
        import pytest
        with pytest.raises(ValueError, match="requires temporal_frontend='per_cell'"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_changes_memory_vs_baseline(self) -> None:
        """Adding non-zero PE must shift memory; with std=0.02 init it's non-zero."""
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        m_base = NeuralFieldPerceiverPerPhoneme()
        cfg_pe = replace(PerPhonemeConfig(), spatial_pe_mode="factorized_2d")
        m_pe = NeuralFieldPerceiverPerPhoneme(cfg_pe)
        # Copy shared weights so only the PE path differs.
        base_state = m_base.state_dict()
        pe_state = m_pe.state_dict()
        for k, v in base_state.items():
            pe_state[k] = v
        m_pe.load_state_dict(pe_state, strict=False)
        m_base.eval()
        m_pe.eval()
        with torch.no_grad():
            mem_base = m_base.encode_memory(batch)
            mem_pe = m_pe.encode_memory(batch)
        assert not torch.allclose(mem_base, mem_pe, atol=1e-5)


class TestPerElectrodePath:
    """Grid-agnostic scalable path — no scatter, no Conv2d, no pool."""

    def test_forward_shape(self) -> None:
        cfg = replace(PerPhonemeConfig(), spatial_path="per_electrode")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        logits = model(batch)
        assert logits.shape == (2, 9)

    def test_encode_memory_shape(self) -> None:
        """Per-electrode memory is (B, N_e * T_tokens, d)."""
        cfg = replace(PerPhonemeConfig(), spatial_path="per_electrode")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        memory = model.encode_memory(batch)
        assert memory.shape == (2, 128 * 11, 32)

    def test_works_with_non_rectangular_N_e(self) -> None:
        """No grid dependency — arbitrary N_e forwards cleanly."""
        cfg = replace(PerPhonemeConfig(), spatial_path="per_electrode")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        batch = _fake_batch(B=2, N_e=53, H_p=8, W_p=16)
        memory = model.encode_memory(batch)
        assert memory.shape == (2, 53 * 11, 32)

    def test_no_conv2d_allocated(self) -> None:
        cfg = replace(PerPhonemeConfig(), spatial_path="per_electrode")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        assert model.conv2d is None
        assert model.conv1d is None
        assert model.per_electrode_conv1d is not None

    def test_rejects_flat_frontend(self) -> None:
        cfg = replace(
            PerPhonemeConfig(),
            spatial_path="per_electrode",
            temporal_frontend="flat",
        )
        import pytest
        with pytest.raises(ValueError, match="requires\n*.*temporal_frontend='per_cell'"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_rejects_factorized_2d_pe(self) -> None:
        cfg = replace(
            PerPhonemeConfig(),
            spatial_path="per_electrode",
            spatial_pe_mode="factorized_2d",
        )
        import pytest
        with pytest.raises(ValueError, match="incompatible.*factorized_2d"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_rejects_hierarchical_readout(self) -> None:
        cfg = replace(
            PerPhonemeConfig(),
            spatial_path="per_electrode",
            readout_mode="hierarchical",
        )
        import pytest
        with pytest.raises(ValueError, match="incompatible.*hierarchical"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_inactive_electrodes_are_masked(self) -> None:
        """Setting some electrodes inactive must change the memory output."""
        cfg = replace(PerPhonemeConfig(), spatial_path="per_electrode")
        model = NeuralFieldPerceiverPerPhoneme(cfg)
        model.eval()
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        with torch.no_grad():
            mem_all = model.encode_memory(batch)
        batch["electrode_active_mask"] = batch["electrode_active_mask"].clone()
        batch["electrode_active_mask"][:, ::4] = False
        with torch.no_grad():
            mem_some = model.encode_memory(batch)
        assert not torch.allclose(mem_all, mem_some, atol=1e-5)

    def test_cls_adds_one_token(self) -> None:
        batch = _fake_batch(B=2, N_e=128, H_p=8, W_p=16)
        cfg = replace(PerPhonemeConfig(), spatial_path="per_electrode")
        cfg_cls = replace(cfg, readout_mode="cls")
        m_base = NeuralFieldPerceiverPerPhoneme(cfg)
        m_cls = NeuralFieldPerceiverPerPhoneme(cfg_cls)
        mem_base = m_base.encode_memory(batch)
        mem_cls = m_cls.encode_memory(batch)
        assert mem_cls.shape[1] == mem_base.shape[1] + 1


class TestFourierElectrodePE:
    """Random-Fourier PE on electrode xyz — scalable bucket, per-electrode only."""

    def _cfg(self) -> PerPhonemeConfig:
        return replace(
            PerPhonemeConfig(),
            spatial_path="per_electrode",
            electrode_pe_mode="fourier_mni",
        )

    def _batch_with_xyz(self, seed: int = 0) -> dict:
        batch = _fake_batch(B=2, N_e=64, H_p=8, W_p=16, seed=seed)
        torch.manual_seed(seed)
        batch["electrode_xyz"] = torch.randn(2, 64, 3) * 50.0
        return batch

    def test_forward_shape(self) -> None:
        model = NeuralFieldPerceiverPerPhoneme(self._cfg())
        logits = model(self._batch_with_xyz())
        assert logits.shape == (2, 9)

    def test_rejects_missing_xyz_in_batch(self) -> None:
        model = NeuralFieldPerceiverPerPhoneme(self._cfg())
        batch = _fake_batch(B=2, N_e=64, H_p=8, W_p=16)  # no electrode_xyz
        import pytest
        with pytest.raises(KeyError, match="electrode_xyz"):
            model(batch)

    def test_rejects_on_pool_path(self) -> None:
        cfg = replace(PerPhonemeConfig(), electrode_pe_mode="fourier_mni")
        import pytest
        with pytest.raises(ValueError, match="requires spatial_path='per_electrode'"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_changes_memory_vs_no_pe(self) -> None:
        batch = self._batch_with_xyz()
        cfg_no = replace(PerPhonemeConfig(), spatial_path="per_electrode")
        cfg_pe = self._cfg()
        m_no = NeuralFieldPerceiverPerPhoneme(cfg_no)
        m_pe = NeuralFieldPerceiverPerPhoneme(cfg_pe)
        # Copy overlapping weights so only the PE differs.
        m_pe.load_state_dict(m_no.state_dict(), strict=False)
        m_no.eval()
        m_pe.eval()
        with torch.no_grad():
            mem_no = m_no.encode_memory(batch)
            mem_pe = m_pe.encode_memory(batch)
        assert not torch.allclose(mem_no, mem_pe, atol=1e-5)

    def test_buffer_is_frozen(self) -> None:
        """fourier_W must be a buffer (no grad), not a trainable parameter."""
        model = NeuralFieldPerceiverPerPhoneme(self._cfg())
        param_names = {n for n, _ in model.named_parameters()}
        assert "fourier_W" not in param_names
        assert "fourier_W" in dict(model.named_buffers())


class TestDistanceBiasAttention:
    """Learnable -α · pairwise-distance additive bias in backbone attention."""

    def _cfg(self) -> PerPhonemeConfig:
        return replace(
            PerPhonemeConfig(),
            spatial_path="per_electrode",
            electrode_pe_mode="distance_bias",
        )

    def _batch_with_xyz(self, seed: int = 0) -> dict:
        batch = _fake_batch(B=2, N_e=64, H_p=8, W_p=16, seed=seed)
        torch.manual_seed(seed)
        batch["electrode_xyz"] = torch.randn(2, 64, 3) * 50.0
        return batch

    def test_forward_shape(self) -> None:
        model = NeuralFieldPerceiverPerPhoneme(self._cfg())
        logits = model(self._batch_with_xyz())
        assert logits.shape == (2, 9)

    def test_rejects_missing_xyz_in_batch(self) -> None:
        model = NeuralFieldPerceiverPerPhoneme(self._cfg())
        batch = _fake_batch(B=2, N_e=64, H_p=8, W_p=16)
        import pytest
        with pytest.raises(KeyError, match="electrode_xyz"):
            model(batch)

    def test_rejects_on_pool_path(self) -> None:
        cfg = replace(PerPhonemeConfig(), electrode_pe_mode="distance_bias")
        import pytest
        with pytest.raises(ValueError, match="requires spatial_path='per_electrode'"):
            NeuralFieldPerceiverPerPhoneme(cfg)

    def test_alpha_is_zero_initialized(self) -> None:
        """Init α=0 means the bias is the additive identity at start."""
        model = NeuralFieldPerceiverPerPhoneme(self._cfg())
        assert float(model.dist_alpha.item()) == 0.0

    def test_alpha_zero_matches_baseline_memory(self) -> None:
        """With α=0, distance_bias must produce identical memory to no bias."""
        batch = self._batch_with_xyz()
        cfg_no = replace(PerPhonemeConfig(), spatial_path="per_electrode")
        cfg_db = self._cfg()
        m_no = NeuralFieldPerceiverPerPhoneme(cfg_no)
        m_db = NeuralFieldPerceiverPerPhoneme(cfg_db)
        m_db.load_state_dict(m_no.state_dict(), strict=False)
        m_no.eval()
        m_db.eval()
        with torch.no_grad():
            mem_no = m_no.encode_memory(batch)
            mem_db = m_db.encode_memory(batch)
        assert torch.allclose(mem_no, mem_db, atol=1e-5)

    def test_alpha_nonzero_changes_memory(self) -> None:
        """After setting α > 0, memory must diverge from the α=0 baseline."""
        batch = self._batch_with_xyz()
        m_db = NeuralFieldPerceiverPerPhoneme(self._cfg())
        m_db.eval()
        with torch.no_grad():
            mem_a0 = m_db.encode_memory(batch)
            m_db.dist_alpha.data.fill_(0.1)
            mem_a1 = m_db.encode_memory(batch)
        assert not torch.allclose(mem_a0, mem_a1, atol=1e-5)


class TestOverfit:
    def test_ten_steps_drop_loss_by_at_least_0_1(self) -> None:
        """10 AdamW steps on a fixed batch should drive CE down by ≥0.1."""
        torch.manual_seed(0)
        model = NeuralFieldPerceiverPerPhoneme()
        batch = _fake_batch(B=4, N_e=128, H_p=8, W_p=16)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        losses: list[float] = []
        for _ in range(10):
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        assert losses[0] - losses[-1] >= 0.1, f"losses={losses}"
