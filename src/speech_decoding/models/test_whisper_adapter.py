"""Tests for the Whisper distillation projection heads (v14 P3).

B33 (2026-05-30) default = student-side :class:`StudentWhisperProjector`
project-UP ``(B, 40, 256) → (B, 40, 1280)`` (LLaVA-1.5 2-layer MLP). The
teacher-side :class:`WhisperAdapter` project-DOWN ``(B, 40, 1280) → (B, 40, 256)``
(LLaVA-1.5 ``Linear(1280, 256) → GeLU → Linear(256, 256)``) is now the
``R-project-down`` sister. :class:`WhisperLayerMerge` (layer merge over the 32
encoder layers) is the ``R-whisper-layer-weighted-sum`` sister. Student stays at
8 Hz native (40 frames per 5 s clip).
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.whisper_adapter import (
    StudentWhisperProjector,
    WhisperAdapter,
    WhisperLayerMerge,
)


def test_whisper_adapter_default_shapes_match_llava_1_5() -> None:
    """LLaVA-1.5 shape: in=1280, hidden=256, out=256."""
    adapter = WhisperAdapter()
    assert adapter.in_dim == 1280
    assert adapter.hidden_dim == 256
    assert adapter.out_dim == 256


def test_whisper_adapter_forward_shape() -> None:
    """``(B, 40, 1280) → (B, 40, 256)`` is the canonical contract."""
    adapter = WhisperAdapter()
    x = torch.randn(2, 40, 1280)
    y = adapter(x)
    assert y.shape == (2, 40, 256)
    assert torch.isfinite(y).all()


def test_whisper_adapter_param_count_around_393k() -> None:
    """LLaVA-1.5 shape gives ~327k + ~65k = ~393k parameters
    (1280·256 + 256 = 327,936 ; 256·256 + 256 = 65,792 → 393,728).
    Pin the count so a future refactor that changes the topology
    surfaces immediately."""
    adapter = WhisperAdapter()
    n_params = sum(p.numel() for p in adapter.parameters())
    expected = (1280 * 256 + 256) + (256 * 256 + 256)
    assert n_params == expected, f"expected {expected:,} params; got {n_params:,}"
    assert 390_000 <= n_params <= 400_000  # human-readable sanity band


def test_whisper_adapter_is_nonlinear() -> None:
    """The GeLU between the two Linears makes the map non-affine —
    output of (x + y) ≠ output(x) + output(y) - output(0)."""
    adapter = WhisperAdapter()
    adapter.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 40, 1280)
    y = torch.randn(1, 40, 1280)
    with torch.no_grad():
        f_x = adapter(x)
        f_y = adapter(y)
        f_xy = adapter(x + y)
        f_0 = adapter(torch.zeros_like(x))
    assert not torch.allclose(f_xy, f_x + f_y - f_0, atol=1e-4), (
        "Adapter is suspiciously affine; the GeLU appears inactive"
    )


def test_whisper_adapter_gradient_flows() -> None:
    """End-to-end gradient through both Linears + the GeLU."""
    adapter = WhisperAdapter()
    x = torch.randn(2, 40, 1280, requires_grad=True)
    y = adapter(x)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Both Linear layers receive gradient.
    for name, p in adapter.named_parameters():
        assert p.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is non-finite"


# --- StudentWhisperProjector (B33 project-up default) ------------------------


def test_student_projector_mlp_default_shape() -> None:
    """B33 default: ``(B, 40, 256) → (B, 40, 1280)`` project-UP."""
    head = StudentWhisperProjector()
    assert head.mode == "mlp"
    assert head.d_in == 256
    assert head.d_out == 1280
    x = torch.randn(2, 40, 256)
    y = head(x)
    assert y.shape == (2, 40, 1280)
    assert torch.isfinite(y).all()


def test_student_projector_mlp_param_count_is_1_968_640() -> None:
    """LLaVA-1.5 shape Linear(256,1280)→GeLU→Linear(1280,1280):
    (256·1280 + 1280) + (1280·1280 + 1280) = 328,960 + 1,639,680 = 1,968,640."""
    head = StudentWhisperProjector(mode="mlp")
    n_params = sum(p.numel() for p in head.parameters())
    expected = (256 * 1280 + 1280) + (1280 * 1280 + 1280)
    assert expected == 1_968_640
    assert n_params == expected, f"expected {expected:,} params; got {n_params:,}"


def test_student_projector_linear_mode_shape_and_param_count() -> None:
    """R-head-linear falsifier: single Linear(256,1280) = 256·1280 + 1280 = 328,960."""
    head = StudentWhisperProjector(mode="linear")
    assert head.mode == "linear"
    x = torch.randn(2, 40, 256)
    assert head(x).shape == (2, 40, 1280)
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == 256 * 1280 + 1280 == 328_960


def test_student_projector_mlp_is_nonlinear() -> None:
    """The GeLU between the two Linears makes the map non-affine —
    f(x+y) ≠ f(x) + f(y) - f(0)."""
    head = StudentWhisperProjector(mode="mlp")
    head.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 40, 256)
    y = torch.randn(1, 40, 256)
    with torch.no_grad():
        f_x, f_y = head(x), head(y)
        f_xy, f_0 = head(x + y), head(torch.zeros_like(x))
    assert not torch.allclose(f_xy, f_x + f_y - f_0, atol=1e-4), (
        "Projector is suspiciously affine; the GeLU appears inactive"
    )


def test_student_projector_gradient_flows_to_both_linears() -> None:
    head = StudentWhisperProjector(mode="mlp")
    x = torch.randn(2, 40, 256, requires_grad=True)
    head(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(p.grad).all(), f"{name} gradient is non-finite"


def test_student_projector_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        StudentWhisperProjector(mode="conv")


# --- WhisperLayerMerge (R-whisper-layer-weighted-sum sister) -----------------


def test_layer_merge_mean_is_param_free_plain_mean() -> None:
    merge = WhisperLayerMerge(n_layers=32, mode="mean")
    assert sum(p.numel() for p in merge.parameters()) == 0
    x = torch.randn(2, 32, 40, 256)
    out = merge(x)
    assert out.shape == (2, 40, 256)
    assert torch.allclose(out, x.mean(dim=1), atol=1e-6)
    assert torch.allclose(merge.layer_weights(), torch.full((32,), 1.0 / 32))


def test_layer_merge_weighted_has_n_layer_params_and_inits_to_mean() -> None:
    """SUPERB-style: ~n_layers params; zero-init weights → uniform softmax →
    exact plain mean at init (recovers the default merge)."""
    merge = WhisperLayerMerge(n_layers=32, mode="weighted")
    assert sum(p.numel() for p in merge.parameters()) == 32
    assert torch.allclose(merge.layer_weights(), torch.full((32,), 1.0 / 32))
    x = torch.randn(2, 32, 40, 256)
    assert torch.allclose(merge(x), x.mean(dim=1), atol=1e-5)


def test_layer_merge_weighted_recovers_single_layer_when_one_hot() -> None:
    """A near-one-hot weighting collapses to a single layer — so the sister
    strictly generalizes both the all-layer mean and a single-layer read."""
    merge = WhisperLayerMerge(n_layers=4, mode="weighted")
    assert merge.weights is not None
    with torch.no_grad():
        merge.weights.copy_(torch.tensor([0.0, 0.0, 50.0, 0.0]))  # layer 2
    x = torch.randn(2, 4, 40, 256)
    assert torch.allclose(merge(x), x[:, 2], atol=1e-4)


def test_layer_merge_weighted_gradient_flows_to_weights() -> None:
    merge = WhisperLayerMerge(n_layers=4, mode="weighted")
    assert merge.weights is not None
    x = torch.randn(2, 4, 40, 256)
    merge(x).sum().backward()
    assert merge.weights.grad is not None
    assert torch.isfinite(merge.weights.grad).all()


def test_layer_merge_rejects_wrong_layer_count_and_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        WhisperLayerMerge(mode="topk")
    merge = WhisperLayerMerge(n_layers=32, mode="mean")
    with pytest.raises(ValueError, match="expected L=32"):
        merge(torch.randn(2, 8, 40, 256))
