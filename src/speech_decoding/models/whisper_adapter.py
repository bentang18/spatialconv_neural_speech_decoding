"""P3-01: Whisper-side 2-layer MLP adapter (~393k params).

LLaVA-1.5 shape: ``Linear(1280, 256) → GeLU → Linear(256, 256)``.
Operates Whisper-side on the frozen Whisper-L8 hidden state to project
the 1280-d Whisper feature into the v14 student's 256-d latent space::

    (B, 40, 1280)  →  (B, 40, 256)

40 frames = 8 Hz × 5 s clip (B05 + B06 lock 2026-05-25 PM). Student
side is identity passthrough at 8 Hz native.

Per the B25 / Phase-3 distillation contract, the loss is Smooth-L1
β=1.0 between the adapter output and the student's PMA-k=1 → flatten
``(40, 256)`` rep, computed by :mod:`speech_decoding.ssl.distill`.

Sister cell ``R-adapter-linear`` collapses the two-Linear MLP to a
single ``Linear(1280, 256)``; this module exposes the MLP variant as
the default per the B05 + B06 lock.
"""

from __future__ import annotations

from torch import Tensor, nn


class WhisperAdapter(nn.Module):
    """Whisper-side 2-layer MLP adapter.

    Parameters
    ----------
    in_dim
        Whisper-L8 hidden-state width; default ``1280`` (Whisper-large-v3
        layer-8 hidden size; v2 had the same width but was upgraded to v3
        on 2026-05-28).
    hidden_dim
        Intermediate width; default ``256`` (matches student d_model).
    out_dim
        Output width; default ``256`` (matches student d_model).
    """

    def __init__(
        self,
        in_dim: int = 1280,
        hidden_dim: int = 256,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, in_dim) → (B, T, out_dim)``."""
        return self.fc2(self.act(self.fc1(x)))
