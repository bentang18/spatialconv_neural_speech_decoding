"""B36 WS-F — Phase-3 Whisper-distillation module + experiment.

Phase 3 is the cross-modal anchor (B33 project-up, 2026-05-30
[[project_v14_b33_project_up_phase3_2026_05_30]]): the SSL-pretrained encoder
(warm-started from P2) is distilled toward the frozen Whisper all-layer-mean
target. Unlike P1/P2 there is **no EMA teacher and no JEPA predictor** — the
target is a fixed cached feature, not a moving-average of the student. The
trainable connector is the parcel-collapse PMA (k=1) + the
:class:`~speech_decoding.models.whisper_adapter.StudentWhisperProjector`
256→1280 head; the loss lives in 1280-d (Smooth-L1 β=1.0).

Forward chain (per :mod:`speech_decoding.ssl.distill`)::

    encoder(return_taps=True)["M4"]   (B, L, T_p, d)   full-input, no mask
      → PMA k=1 (latent_valid-keyed)  (B, T_p, d)
      → StudentWhisperProjector       (B, T_p, 1280)   = student
    whisper_target (cache, 50 Hz)     (B, 250, 1280)
      → triangular_pool_50_to_8_hz    (B, 40, 1280)
      → TargetStandardizer (train-only) (B, 40, 1280)  = teacher (detached)
    loss = SmoothL1(student, teacher)   in 1280-d

Both rates meet at 8 Hz: the student's ``T_p`` must equal the teacher pool's
40 frames (8 Hz × 5 s). A mismatch is the symptom of a mis-sized clip or a
front-end patch-rate drift, so :meth:`V14Phase3DistillModule._step` rejects it
loudly before the distill call rather than letting the shape-guard inside the
loss fire with a less specific message.

Two-stage schedule (B33 §5):

* **3a warmup** (``stage="3a"``) — the encoder is FROZEN; only the connector
  ``{PMA, StudentWhisperProjector}`` trains. The encoder runs under
  ``no_grad`` so no encoder activations are retained.
* **3b unfreeze** (``stage="3b"``) — the encoder unfreezes with a
  discriminative LR: front-end @ ``base/10`` (pretrained twice, in P1 and as
  the P2 LR/10 group), parcel side @ ``base/3``, connector @ full ``base``.

Cross-phase handoff (E4): :meth:`transferable_state` exports
``{encoder, pma, projector}`` so 3a→3b carries the warmed connector forward;
P4 loads only the ``encoder``+``pma`` intersection and drops the projector.
"""

from __future__ import annotations

import typing as tp
from contextlib import nullcontext

import pydantic
import torch
from lightning import pytorch as pl
from torch import Tensor, nn

from neuraltrain.optimizers import BaseOptimizer

from speech_decoding.bt_alignment.teacher_cache import TargetStandardizer
from speech_decoding.experiments.v14_experiment import V14Experiment
from speech_decoding.experiments.v14_joint_module import _maybe_drop_singleton_trailing
from speech_decoding.extractors.whisper_teacher_pool import triangular_pool_50_to_8_hz
from speech_decoding.models.v14_encoder import (
    V14ParcelCollapsePMA,
    V14ParcelPerceiverModel,
    _compute_latent_valid,
)
from speech_decoding.models.whisper_adapter import StudentWhisperProjector
from speech_decoding.ssl.distill import PhaseThreeDistillationConfig


_Stage = tp.Literal["3a", "3b"]

# 8 Hz × 5 s teacher pool output (whisper_teacher_pool 250 → 40). The student's
# T_p must match this for the 1280-d distillation loss to align frame-for-frame.
_TEACHER_POOL_FRAMES = 40

# Whisper-large-v3 hidden width (the teacher cache emits 1280; not a sweep axis).
_WHISPER_D = 1280
# Canonical batch key for the cached 50 Hz Whisper all-layer-mean feature.
_WHISPER_TARGET_KEY = "whisper_target"


class V14Phase3DistillModule(pl.LightningModule):
    """Phase-3 Whisper-distillation Lightning module (WS-F).

    Owns the warm-started ``encoder`` (a
    :class:`~speech_decoding.models.v14_encoder.V14ParcelPerceiverModel`), an
    unfrozen parcel-collapse ``pma`` (k=1), and the
    ``projector`` (:class:`StudentWhisperProjector`, 256→1280). There is no EMA
    teacher and no predictor — the distillation target is the fixed cached
    Whisper feature, pooled to 8 Hz and per-channel standardized
    (train-only) by the optional ``standardizer``.

    ``stage`` selects the freeze/optimizer regime: ``"3a"`` freezes the encoder
    (connector-only warmup), ``"3b"`` unfreezes it with a discriminative LR.
    """

    # B33 §5 discriminative-LR scales for 3b (pinned, not a sweep axis — mirrors
    # the sibling V14JointBrainModule's ClassVar convention). Front-end rides at
    # base/10 (pretrained in P1 + as the P2 LR/10 group); parcel side at base/3.
    _FRONTEND_LR_SCALE: tp.ClassVar[float] = 0.1
    _PARCEL_LR_SCALE: tp.ClassVar[float] = 1.0 / 3.0

    def __init__(
        self,
        *,
        encoder: V14ParcelPerceiverModel,
        optim_config: BaseOptimizer,
        stage: _Stage,
        pma_n_heads: int,
        projector_mode: str = "mlp",
        distill_config: tp.Optional[PhaseThreeDistillationConfig] = None,
        standardizer: tp.Optional[TargetStandardizer] = None,
        pma: tp.Optional[V14ParcelCollapsePMA] = None,
        projector: tp.Optional[StudentWhisperProjector] = None,
    ) -> None:
        super().__init__()
        if stage not in tp.get_args(_Stage):
            raise ValueError(f"stage={stage!r} not in {tp.get_args(_Stage)}")

        self.encoder = encoder
        # Default freeze=False ⇒ the P3 connector trains; the encoder freeze
        # is applied separately by stage below so 3a/3b share one construction.
        self.pma = pma or V14ParcelCollapsePMA(encoder.d_model, pma_n_heads)
        self.projector = projector or StudentWhisperProjector(
            d_in=encoder.d_model, d_out=_WHISPER_D, mode=projector_mode,
        )
        self.standardizer = standardizer
        self.distill = distill_config or PhaseThreeDistillationConfig()

        self.optim_config = optim_config
        self._stage: _Stage = stage
        self._m_sub_slots = encoder.m_sub_slots
        self._d_model = encoder.d_model

        self._apply_stage_freeze()

    # ------------------------------------------------------------------
    # Stage freeze
    # ------------------------------------------------------------------

    def _apply_stage_freeze(self) -> None:
        """3a freezes the encoder (connector-only warmup); 3b unfreezes it.

        The PMA + projector always train. ``requires_grad`` is the freeze
        contract the optimizer param-groups and the unit tests both read; 3a
        additionally runs the encoder forward under ``no_grad`` (see
        :meth:`_encoder_taps`) so frozen-encoder activations are never retained.
        """
        train_encoder = self._stage == "3b"
        self.encoder.requires_grad_(train_encoder)
        self.pma.requires_grad_(True)
        self.projector.requires_grad_(True)

    # ------------------------------------------------------------------
    # Cross-phase weight handoff (E4)
    # ------------------------------------------------------------------

    def transferable_state(self) -> dict[str, dict[str, Tensor]]:
        """Components that carry forward (E4): encoder + PMA + projector.

        The projector is exported so the 3a→3b boundary carries the warmed
        connector intact. P4 owns only an encoder + PMA, so its
        ``load_transferable_state`` loads the ``encoder``/``pma`` intersection
        and silently drops the P3-only ``projector`` (E4 "projector keys
        dropped").
        """
        return {
            "encoder": self.encoder.state_dict(),
            "pma": self.pma.state_dict(),
            "projector": self.projector.state_dict(),
        }

    def load_transferable_state(
        self, state: dict[str, dict[str, Tensor]], *, strict: bool = True,
    ) -> None:
        """Warm-start from a prior phase's :meth:`transferable_state` (E4).

        ``encoder`` is required and loaded strict — a missing/unexpected key
        is the early warning that the encoder topology drifted between phases.
        ``pma`` / ``projector`` are optional: loading from P2 (encoder-only)
        leaves the freshly-built connector at its init for the 3a warmup;
        loading from 3a carries the warmed connector into 3b. After loading,
        the stage freeze is re-applied so a loaded encoder honours 3a's freeze.
        """
        if "encoder" not in state:
            raise KeyError(
                "transferable state has no 'encoder' component; cannot "
                f"warm-start the P3 encoder. Got keys: {sorted(state)}."
            )
        self.encoder.load_state_dict(state["encoder"], strict=strict)
        if "pma" in state:
            self.pma.load_state_dict(state["pma"], strict=strict)
        if "projector" in state:
            self.projector.load_state_dict(state["projector"], strict=strict)
        # A loaded encoder must still obey the current stage's freeze.
        self._apply_stage_freeze()

    # ------------------------------------------------------------------
    # Batch ingest (mirrors V14JointBrainModule._extract_student_kwargs)
    # ------------------------------------------------------------------

    def _extract_encoder_kwargs(
        self, batch_data: dict[str, Tensor],
    ) -> dict[str, tp.Any]:
        if "electrode_tokens" not in batch_data:
            raise KeyError(
                "V14Phase3DistillModule batch missing 'electrode_tokens'."
            )
        if "support" not in batch_data:
            raise KeyError(
                "V14Phase3DistillModule batch missing 'support' (B30 "
                "latent_valid contract needs the per-clip support tensor)."
            )
        kwargs: dict[str, tp.Any] = {
            "electrode_tokens": batch_data["electrode_tokens"],
            "support": batch_data["support"],
        }
        if "valid_mask" in batch_data:
            kwargs["valid_mask"] = batch_data["valid_mask"]
        if "subject_subtype" in batch_data:
            kwargs["subject_subtype"] = _maybe_drop_singleton_trailing(
                batch_data["subject_subtype"],
            )
        if "ref_idx" in batch_data:
            kwargs["ref_idx"] = _maybe_drop_singleton_trailing(
                batch_data["ref_idx"],
            )
        return kwargs

    # ------------------------------------------------------------------
    # Per-step composition
    # ------------------------------------------------------------------

    def _encoder_taps(self, encoder_kwargs: dict[str, tp.Any]) -> Tensor:
        """Full-input encoder M4 tap ``(B, L, T_p, d)``. Frozen under 3a."""
        ctx = torch.no_grad() if self._stage == "3a" else nullcontext()
        with ctx:
            taps = self.encoder(return_taps=True, **encoder_kwargs)
        if not isinstance(taps, dict):
            raise RuntimeError(
                "V14ParcelPerceiverModel returned a single tensor under "
                "return_taps=True; expected the M2/M3/M4 tap dict."
            )
        return taps["M4"]

    def _student_readout(self, encoder_kwargs: dict[str, tp.Any]) -> Tensor:
        """Encoder M4 → PMA collapse → projector. Returns ``(B, T_p, 1280)``."""
        m4 = self._encoder_taps(encoder_kwargs)
        latent_valid = _compute_latent_valid(
            support=encoder_kwargs["support"],
            valid_mask=encoder_kwargs.get("valid_mask"),
            m_sub_slots=self._m_sub_slots,
        )
        collapsed = self.pma(m4, latent_valid)            # (B, T_p, d)
        return self.projector(collapsed)                  # (B, T_p, 1280)

    def _teacher_target(self, batch_data: dict[str, Tensor]) -> Tensor:
        """Cached Whisper feature → 8 Hz pool → train-only z-score. Detached.

        Reads the raw 50 Hz cache feature ``(B, 250, 1280)`` from the batch,
        pools to ``(B, 40, 1280)`` and (if a standardizer is wired) applies the
        fixed per-channel z-score. The result is the distillation target; the
        loss's own ``detach`` makes the stop-grad explicit but the target is
        already a fixed tensor.
        """
        if _WHISPER_TARGET_KEY not in batch_data:
            raise KeyError(
                f"V14Phase3DistillModule batch missing "
                f"{_WHISPER_TARGET_KEY!r}; the P3 data pipeline must emit "
                "the cached 50 Hz Whisper all-layer-mean feature (B, 250, "
                f"{_WHISPER_D}). See docs/neuroprobe/v14_blockers.md."
            )
        target = batch_data[_WHISPER_TARGET_KEY]
        pooled = triangular_pool_50_to_8_hz(target)        # (B, 40, 1280)
        if self.standardizer is not None:
            pooled = self.standardizer(pooled)
        return pooled

    def _step(self, batch_data: dict[str, Tensor]) -> Tensor:
        encoder_kwargs = self._extract_encoder_kwargs(batch_data)
        student = self._student_readout(encoder_kwargs)    # (B, T_p, 1280)
        teacher = self._teacher_target(batch_data)         # (B, 40, 1280)

        if student.shape[-2] != teacher.shape[-2]:
            raise ValueError(
                "P3 rate mismatch: student emits T_p="
                f"{student.shape[-2]} frames but the Whisper teacher pool emits "
                f"{teacher.shape[-2]} (8 Hz × 5 s = {_TEACHER_POOL_FRAMES}). "
                "The student and teacher must both be at 8 Hz — check the "
                "front-end patch rate (n_time_bins / patch_kernel_time) and "
                "the clip length feeding the Whisper cache."
            )
        teacher = teacher.to(student.dtype)
        return self.distill(student, teacher)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        loss = self._step(batch.data)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:  # noqa: ARG002
        loss = self._step(batch.data)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:  # noqa: ARG002
        loss = self._step(batch.data)
        self.log("test_loss", loss, on_epoch=True)

    # ------------------------------------------------------------------
    # Optimizer (3a connector-only; 3b discriminative LR)
    # ------------------------------------------------------------------

    def _connector_parameters(self) -> list[nn.Parameter]:
        return list(self.pma.parameters()) + list(self.projector.parameters())

    def _base_lr(self) -> float:
        inner = getattr(self.optim_config, "optimizer", None)
        lr = getattr(inner, "lr", None)
        if lr is None:
            raise RuntimeError(
                "P3 needs optim_config.optimizer.lr (3b discriminative LR scales "
                "the encoder groups off it); got None."
            )
        return float(lr)

    def _phase_param_groups(self) -> list[tp.Any]:
        """3a: connector params only. 3b: 3 discriminative-LR groups.

        3b groups (B33 §5): front-end @ ``base·frontend_lr_scale`` (=base/10),
        parcel side @ ``base·parcel_lr_scale`` (=base/3), connector @ full base.
        """
        if self._stage == "3a":
            return self._connector_parameters()
        frontend, parcel = self.encoder.partition_parameters_for_staging()
        base_lr = self._base_lr()
        return [
            {"params": frontend, "lr": base_lr * self._FRONTEND_LR_SCALE},
            {"params": parcel, "lr": base_lr * self._PARCEL_LR_SCALE},
            {"params": self._connector_parameters(), "lr": base_lr},
        ]

    def _estimated_total_steps(self) -> int | None:
        try:
            return int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            return None

    def configure_optimizers(self):  # type: ignore[override]
        params = self._phase_param_groups()
        total_steps = self._estimated_total_steps()
        if total_steps is None:
            return self.optim_config.build(params)
        try:
            return self.optim_config.build(params, total_steps=total_steps)
        except TypeError:
            return self.optim_config.build(params)


class V14Phase3Experiment(V14Experiment):
    """Phase-3 Whisper-distillation experiment (WS-F).

    Pinned to ``phase ∈ {"3a", "3b"}``. Mirrors
    :class:`speech_decoding.experiments.v14_joint.V14JointExperiment`: the
    parent's CE-classifier ``BrainModule`` is replaced by a
    :class:`V14Phase3DistillModule` built around the encoder extracted from
    ``brain_model_config.build(...).encoder``.

    The base ``loss`` / ``metrics`` fields are required by the
    :class:`Experiment` schema (dispatch fills them with a vestigial
    ``CrossEntropyLoss`` + ``Accuracy``); the distillation loss is composed
    inside the module, so they are unused here.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    projector_mode: tp.Literal["mlp", "linear"] = "mlp"
    # B33: per-channel target standardization is mandatory by default; the
    # falsifier R-no-target-standardize sets this False (raw 1280-d target).
    target_standardize: bool = True
    # Path to the train-only channel stats ({"mean", "inv_std"}) saved by
    # fit_channel_stats. Required when target_standardize is True.
    channel_stats_path: str | None = None
    distill: PhaseThreeDistillationConfig = pydantic.Field(
        default_factory=PhaseThreeDistillationConfig,
    )

    def model_post_init(self, _ctx) -> None:  # type: ignore[override]
        super().model_post_init(_ctx)
        if self.phase not in ("3a", "3b"):
            raise ValueError(
                "V14Phase3Experiment models Phase-3 distillation only "
                f"(phase ∈ {{'3a', '3b'}}); got phase={self.phase!r}."
            )
        if self.target_standardize and self.channel_stats_path is None:
            raise ValueError(
                "target_standardize=True (B33 default) needs channel_stats_path "
                "(train-only stats from fit_channel_stats). Set it, or run the "
                "R-no-target-standardize sister with target_standardize=False."
            )

    def _build_standardizer(self) -> TargetStandardizer | None:
        if not self.target_standardize:
            return None
        stats = torch.load(
            self.channel_stats_path, map_location="cpu", weights_only=True,
        )
        return TargetStandardizer(stats["mean"], stats["inv_std"])

    def _build_brain_module(self, train_loader):  # type: ignore[override]
        batch = next(iter(train_loader))
        input_name = self._input_tensor_name()
        x = batch.data[input_name]
        head_model = self.brain_model_config.build(
            n_in_channels=int(x.shape[1]),
            n_outputs=1,
        )
        encoder = getattr(head_model, "encoder", None)
        if not isinstance(encoder, V14ParcelPerceiverModel):
            raise RuntimeError(
                "V14Phase3Experiment expected brain_model_config.build to "
                "return a model with an ``.encoder`` of type "
                f"V14ParcelPerceiverModel; got {type(head_model).__name__}."
            )
        pma_n_heads = getattr(self.brain_model_config, "n_heads", None)
        if pma_n_heads is None:
            raise RuntimeError(
                "V14Phase3Experiment needs brain_model_config.n_heads to size "
                "the parcel-collapse PMA."
            )
        return V14Phase3DistillModule(
            encoder=encoder,
            optim_config=self.optim,
            stage=self.phase,  # type: ignore[arg-type]
            pma_n_heads=int(pma_n_heads),
            projector_mode=self.projector_mode,
            distill_config=self.distill,
            standardizer=self._build_standardizer(),
        )


__all__ = [
    "V14Phase3DistillModule",
    "V14Phase3Experiment",
]
