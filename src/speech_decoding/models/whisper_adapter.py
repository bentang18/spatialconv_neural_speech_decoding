"""Whisper distillation projection heads (v14 P3).

B33 (2026-05-30) flips Phase-3 cross-modal distillation from project-DOWN to
project-UP. The **default** is now the student-side :class:`StudentWhisperProjector`,
which projects the student's 256-d PMA readout UP to the fixed 1280-d Whisper
all-layer-mean target; the loss lives in 1280-d. The teacher carries no trainable
module — its target is the raw 1280-d feature run through a fixed, train-only
per-channel z-score (:class:`speech_decoding.bt_alignment.teacher_cache.TargetStandardizer`),
which is what now plays the role the ceiling probe's downstream ``StandardScaler``
filled (and what let the raw, unnormalized all-layer mean win the transfer splits).

This module hosts three pieces:

  * :class:`StudentWhisperProjector` — **default**. Student-side 256→1280 head
    (LLaVA-1.5 2-layer MLP, ~1.97M params). Trained at P3 as part of the
    connector ``{PMA, StudentWhisperProjector}`` (B31: PMA unfrozen at P3);
    discarded at P4.
  * :class:`WhisperAdapter` — the ``R-project-down`` sister (prior default):
    teacher-side 1280→256 MLP, loss in 256-d. Kept for the falsifier.
  * :class:`WhisperLayerMerge` — the ``R-whisper-layer-weighted-sum`` sister;
    direction-agnostic merge over the 32 encoder layers.

40 frames = 8 Hz × 5 s clip. The teacher feature is the **plain mean over all 32
encoder layers** (``layer_merge="mean_all"`` per
``project_v14_whisper_teacher_all_layer_mean_2026_05_30``). The Phase-3
distillation loss is Smooth-L1 β=1.0, computed by
:mod:`speech_decoding.ssl.distill`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WhisperAdapter(nn.Module):
    """Teacher-side 2-layer MLP adapter — the ``R-project-down`` sister.

    Pre-B33 default: projects the 1280-d Whisper teacher feature DOWN into the
    student's 256-d latent space (``(B, 40, 1280) → (B, 40, 256)``), with the
    loss in 256-d and the student an identity passthrough. B33 (2026-05-30)
    demoted this to a falsifier — the project-up default carries no teacher-side
    trainable module (see :class:`StudentWhisperProjector`). Kept so the
    ``R-project-down`` sweep cell can be wired teacher-side. LLaVA-1.5 shape
    ``Linear(1280, 256) → GeLU → Linear(256, 256)`` (~393k); sub-sister
    ``R-adapter-linear`` collapses the two Linears to one.

    Parameters
    ----------
    in_dim
        Whisper teacher-feature width; default ``1280`` (Whisper-large-v3
        hidden size; the default teacher is the all-layer mean, which keeps
        the 1280-d width; v2→v3 upgrade 2026-05-28).
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


class StudentWhisperProjector(nn.Module):
    """B33 project-up student head: ``(B, T, 256) → (B, T, 1280)`` — the default.

    Projects the student's 256-d PMA-k=1 readout UP to the fixed 1280-d Whisper
    all-layer-mean target, so the Phase-3 distillation loss lives in 1280-d.
    ``mode="mlp"`` (default, LLaVA-1.5 shape ``Linear(256,1280) → GeLU →
    Linear(1280,1280)``, ~1.97M params) is the locked default; ``mode="linear"``
    (a single ``Linear(256,1280)``, ~0.33M) is the ``R-head-linear`` falsifier.

    Trained at P3 as part of the connector ``{PMA, StudentWhisperProjector}``
    (B31: PMA unfrozen at P3); discarded at P4, which re-learns its own
    attentive probe over the encoder grid. The mlp-vs-linear decision is judged
    on P4 downstream transfer, never on the P3 distillation loss; both must-run.

    Parameters
    ----------
    d_in
        Student readout width; default ``256`` (student d_model).
    d_out
        Whisper target width; default ``1280`` (Whisper-large all-layer mean).
    mode
        ``"mlp"`` (default) or ``"linear"`` (``R-head-linear`` sister).
    """

    def __init__(self, d_in: int = 256, d_out: int = 1280, mode: str = "mlp") -> None:
        super().__init__()
        if mode == "mlp":
            self.net: nn.Module = nn.Sequential(
                nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out)
            )
        elif mode == "linear":
            self.net = nn.Linear(d_in, d_out)
        else:
            raise ValueError(f"mode must be 'mlp' or 'linear', got {mode!r}")
        self.mode = mode
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x: Tensor) -> Tensor:
        """``(B, T, d_in) → (B, T, d_out)``."""
        return self.net(x)


class WhisperLayerMerge(nn.Module):
    """Merge per-layer Whisper features ``(B, L, T, d) → (B, T, d)``.

    The default teacher cache pre-merges with a plain mean, so the default
    path needs no module. This module is the ``R-whisper-layer-weighted-sum``
    sister: it consumes an all-layer teacher cache and learns a SUPERB-style
    softmax weighting over the ``L`` encoder layers (~``n_layers`` params).

    ``mode="mean"`` is the parameter-free plain mean (identical to the default
    cache merge). ``mode="weighted"`` learns one scalar per layer; the weights
    init to zeros so the softmax starts uniform — i.e. it *starts* as the plain
    mean and can learn toward a single layer (one-hot) or any blend. It strictly
    generalizes both the all-layer mean and a single-layer read.
    """

    def __init__(self, n_layers: int = 32, mode: str = "mean") -> None:
        super().__init__()
        if mode not in ("mean", "weighted"):
            raise ValueError(f"mode must be 'mean' or 'weighted', got {mode!r}")
        self.n_layers = n_layers
        self.mode = mode
        # Zeros → uniform softmax → exact plain mean at init.
        self.weights = (
            nn.Parameter(torch.zeros(n_layers)) if mode == "weighted" else None
        )

    def layer_weights(self) -> Tensor:
        """Current per-layer softmax weights ``(n_layers,)`` (for logging /
        interpretability). Uniform ``1/n_layers`` in ``mean`` mode."""
        if self.weights is None:
            return torch.full((self.n_layers,), 1.0 / self.n_layers)
        return torch.softmax(self.weights, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, L, T, d) → (B, T, d)``, merging over the layer axis ``L``."""
        if x.shape[1] != self.n_layers:
            raise ValueError(
                f"expected L={self.n_layers} layers, got {x.shape[1]}"
            )
        if self.weights is None:
            return x.mean(dim=1)
        w = torch.softmax(self.weights, dim=0)
        return torch.einsum("l,bltd->btd", w, x)
