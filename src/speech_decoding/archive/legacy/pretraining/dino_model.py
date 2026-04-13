"""DINO self-distillation for uECOG data.

Emerging Properties in Self-Supervised Vision Transformers (Caron et al. 2021)
adapted for temporal neural data.

Three collapse-prevention mechanisms targeting alignment + uniformity:
1. EMA teacher — prevents trivial collapse
2. Centering — subtracts running mean of teacher outputs → uniformity
3. Sharpening — low teacher temperature → peaked predictions → cluster separation

Student sees augmented input, teacher sees different augmented input.
Cross-entropy loss in prototype probability space (not MSE in latent space).

After pretraining: transfer online encoder weights to PretrainModel.
Projector + teacher encoder/projector are discarded.
"""
from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from speech_decoding.data.augmentation import augment_from_config
from speech_decoding.models.spatial_conv import SpatialConvReadIn


DEFAULT_SSL_AUGMENTATION = {
    "time_shift_frames": 30,
    "amp_scale_std": 0.3,
    "channel_dropout_max": 0.2,
    "noise_frac": 0.05,
    "temporal_stretch": True,
    "temporal_stretch_max_rate": 0.15,
}


class DINOModel(nn.Module):
    """DINO self-distillation for uECOG temporal representation learning.

    Architecture:
        Student (online): augmented input → encoder → mean-pool → projector → P logits
        Teacher (EMA):    augmented input → encoder → mean-pool → projector → P logits

    Training objective:
        teacher_probs = softmax(center(teacher_logits) / tau_t)  [sharpened]
        student_probs = log_softmax(student_logits / tau_s)
        L = -sum(teacher_probs * student_probs)  [cross-entropy]

    Key difference from JEPA: loss in discrete probability space over P prototypes,
    not MSE in continuous latent space. Centering + sharpening directly optimize
    for k-NN-friendly geometry (Wang & Isola 2020).
    """

    def __init__(self, config: dict, grid_shape: tuple[int, int]):
        super().__init__()
        self.config = config
        self.grid_shape = grid_shape
        H, W = grid_shape

        # --- Student encoder ---
        sc = config.get("spatial_conv", {})
        channels = sc.get("channels", 8)
        pool_h = sc.get("pool_h", 4)
        pool_w = sc.get("pool_w", 8)
        num_layers = sc.get("num_layers", 1)

        self.readin = SpatialConvReadIn(
            grid_h=H, grid_w=W,
            C=channels, pool_h=pool_h, pool_w=pool_w,
            num_layers=num_layers,
        )
        d_flat = self.readin.out_dim

        d = config.get("d", 64)
        stride = config.get("temporal_stride", 10)
        self.d = d
        self.stride = stride

        gru_hidden = config.get("gru_hidden", 32)
        gru_layers = config.get("gru_layers", 2)
        self.gru_hidden = gru_hidden

        self.ln = nn.LayerNorm(d_flat)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_flat, d, kernel_size=stride, stride=stride),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            d, gru_hidden, num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=0.0,
        )

        gru_out_dim = gru_hidden * 2

        # --- Student projector: 3-layer MLP → P prototype logits ---
        proj_hidden = config.get("dino_proj_hidden", 256)
        n_prototypes = config.get("dino_n_prototypes", 256)
        self.n_prototypes = n_prototypes

        self.projector = nn.Sequential(
            nn.Linear(gru_out_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, n_prototypes),
        )

        # --- Teacher: deep copy of encoder + projector, no gradients ---
        self.teacher_readin = deepcopy(self.readin)
        self.teacher_ln = deepcopy(self.ln)
        self.teacher_temporal_conv = deepcopy(self.temporal_conv)
        self.teacher_gru = deepcopy(self.gru)
        self.teacher_projector = deepcopy(self.projector)
        for p in self._teacher_params():
            p.requires_grad = False

        # --- Centering: running mean of teacher logits ---
        self.register_buffer("center", torch.zeros(n_prototypes))
        self.center_momentum = config.get("dino_center_momentum", 0.9)

        # --- Temperatures ---
        self.tau_student = config.get("dino_tau_student", 0.1)
        self.tau_teacher = config.get("dino_tau_teacher", 0.04)

        # --- EMA state ---
        self.ema_step = 0
        self.ema_total_steps = config.get("ema_total_steps", 5000)
        self.ema_momentum = config.get("ema_momentum", 0.996)
        self.ema_momentum_end = config.get("ema_momentum_end", 1.0)

        # --- Augmentation config ---
        self.aug_config = dict(DEFAULT_SSL_AUGMENTATION)

    def _teacher_params(self):
        """Iterator over all teacher parameters."""
        for module in (self.teacher_readin, self.teacher_ln,
                       self.teacher_temporal_conv, self.teacher_gru,
                       self.teacher_projector):
            yield from module.parameters()

    def _student_teacher_pairs(self):
        """Yield (student_param, teacher_param) pairs for EMA update."""
        for s_mod, t_mod in (
            (self.readin, self.teacher_readin),
            (self.ln, self.teacher_ln),
            (self.temporal_conv, self.teacher_temporal_conv),
            (self.gru, self.teacher_gru),
            (self.projector, self.teacher_projector),
        ):
            for sp, tp in zip(s_mod.parameters(), t_mod.parameters()):
                yield sp, tp

    def _get_ema_momentum(self) -> float:
        """Cosine-scheduled EMA momentum: ramps from start to end."""
        start = self.ema_momentum
        end = self.ema_momentum_end
        if self.ema_total_steps <= 0:
            return end
        progress = min(self.ema_step / self.ema_total_steps, 1.0)
        return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2

    @torch.no_grad()
    def ema_update(self):
        """Update teacher via EMA of student."""
        tau = self._get_ema_momentum()
        for sp, tp in self._student_teacher_pairs():
            tp.data.mul_(tau).add_(sp.data, alpha=1 - tau)
        self.ema_step += 1

    @torch.no_grad()
    def _update_center(self, teacher_logits: torch.Tensor):
        """Update running mean of teacher logits (centering buffer)."""
        batch_center = teacher_logits.mean(dim=0)
        m = self.center_momentum
        self.center = m * self.center + (1 - m) * batch_center

    def _encode_student(self, x: torch.Tensor) -> torch.Tensor:
        """Student encoder: (B, H, W, T) → (B, T', 2*gru_hidden)."""
        spatial = self.readin(x)
        h = self.ln(spatial.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.temporal_conv(h)
        h = h.permute(0, 2, 1)
        gru_out, _ = self.gru(h)
        return gru_out

    @torch.no_grad()
    def _encode_teacher(self, x: torch.Tensor) -> torch.Tensor:
        """Teacher encoder: (B, H, W, T) → (B, T', 2*gru_hidden). No gradients."""
        spatial = self.teacher_readin(x)
        h = self.teacher_ln(spatial.permute(0, 2, 1)).permute(0, 2, 1)
        h = self.teacher_temporal_conv(h)
        h = h.permute(0, 2, 1)
        gru_out, _ = self.teacher_gru(h)
        return gru_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full student encoder (no augmentation, no projector) for downstream.

        Args:
            x: (B, H, W, T) raw grid input.
        Returns:
            (B, T', 2*gru_hidden) features.
        """
        return self._encode_student(x)

    def forward(
        self, x: torch.Tensor, compute_loss: bool = True,
    ) -> dict[str, torch.Tensor | float]:
        """Forward pass with DINO self-distillation objective.

        Two augmented views → student and teacher encode separately.
        Teacher logits are centered and sharpened. Student logits are softened.
        Loss is cross-entropy between teacher and student distributions.

        Args:
            x: (B, H, W, T) input grid data.
            compute_loss: whether to compute DINO loss.

        Returns:
            dict with keys:
                loss: DINO cross-entropy loss (scalar), if compute_loss
                teacher_std: std of teacher logits (collapse diagnostic)
        """
        if not compute_loss:
            z = self.encode(x)
            return {"embeddings": z.mean(dim=1)}

        # Two augmented views
        x1 = augment_from_config(x, self.aug_config, training=True)
        x2 = augment_from_config(x, self.aug_config, training=True)

        # Student: encode → pool → project
        s1 = self._encode_student(x1).mean(dim=1)  # (B, 2H)
        s2 = self._encode_student(x2).mean(dim=1)
        s1_logits = self.projector(s1)  # (B, P)
        s2_logits = self.projector(s2)

        # Teacher: encode → pool → project (no gradients)
        with torch.no_grad():
            t1 = self._encode_teacher(x1).mean(dim=1)
            t2 = self._encode_teacher(x2).mean(dim=1)
            t1_logits = self.teacher_projector(t1)
            t2_logits = self.teacher_projector(t2)

        # DINO loss: cross-view distillation
        # Teacher view 1 → student view 2, teacher view 2 → student view 1
        loss_12 = self._dino_loss(t1_logits, s2_logits)
        loss_21 = self._dino_loss(t2_logits, s1_logits)
        loss = (loss_12 + loss_21) / 2

        # Update center with both teacher outputs
        with torch.no_grad():
            self._update_center(torch.cat([t1_logits, t2_logits], dim=0))

        # Diagnostic: teacher std (should stay > 0)
        teacher_std = t1_logits.std().item()

        return {
            "loss": loss,
            "teacher_std": teacher_std,
        }

    def _dino_loss(
        self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """DINO cross-entropy: sharpened centered teacher → softened student.

        Args:
            teacher_logits: (B, P) raw teacher prototype logits.
            student_logits: (B, P) raw student prototype logits.
        Returns:
            Scalar loss.
        """
        # Teacher: center + sharpen
        teacher_probs = F.softmax(
            (teacher_logits - self.center) / self.tau_teacher, dim=-1
        )
        # Student: soften
        student_log_probs = F.log_softmax(
            student_logits / self.tau_student, dim=-1
        )
        # Cross-entropy
        loss = -torch.sum(teacher_probs * student_log_probs, dim=-1).mean()
        return loss

    def transfer_encoder_weights(self, pretrain_model: nn.Module) -> None:
        """Transfer student encoder weights to a PretrainModel.

        Uses the student encoder (not teacher) because:
        - Student is the one optimized by gradients
        - Teacher is a smoothed copy for generating targets

        Projector, teacher encoder/projector, and center buffer are discarded.
        """
        pretrain_model.readin.load_state_dict(self.readin.state_dict())
        pretrain_model.backbone.layernorm.load_state_dict(self.ln.state_dict())
        pretrain_model.backbone.temporal_conv.load_state_dict(
            self.temporal_conv.state_dict()
        )
        pretrain_model.backbone.gru.load_state_dict(self.gru.state_dict())
