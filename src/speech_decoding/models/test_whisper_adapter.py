"""P3-01 tests: Whisper-side 2-layer MLP adapter.

LLaVA-1.5 shape: ``Linear(1280, 256) → GeLU → Linear(256, 256)``.
Operates Whisper-side on the frozen Whisper-L8 features:
``(B, 40, 1280) → (B, 40, 256)``. Student stays at 8 Hz native (40
frames per 5 s clip).
"""

from __future__ import annotations

import torch

from speech_decoding.models.whisper_adapter import WhisperAdapter


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
