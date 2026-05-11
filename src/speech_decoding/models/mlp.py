from __future__ import annotations

from torch import Tensor, nn

from neuraltrain.models.base import BaseModelConfig


class TinyMLPModel(nn.Module):
    """Two-layer MLP with global average pooling over time.

    Forward expects `(B, n_in_channels, T)`; collapses time via mean pool then
    feeds through `Linear → GELU → Linear` to yield `(B, n_outputs)`.
    """

    def __init__(self, n_in_channels: int, n_outputs: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.mean(dim=-1))


class TinyMLP(BaseModelConfig):
    """Smoke-test MLP for substrate-integration runs (no v14 novelty)."""

    hidden: int = 16

    def build(self, n_in_channels: int, n_outputs: int) -> nn.Module:
        return TinyMLPModel(n_in_channels=n_in_channels, n_outputs=n_outputs, hidden=self.hidden)
