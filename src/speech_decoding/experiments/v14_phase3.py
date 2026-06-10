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

from speech_decoding.bt_alignment.teacher_cache import (
    DEFAULT_LAYER_MERGE,
    TargetStandardizer,
    merge_slug,
)
from speech_decoding.experiments.monitors import grad_spike_monitor
from speech_decoding.experiments.optim_param_groups import maybe_split_no_decay
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
        frontend_lr_scale: float = 0.1,
        parcel_lr_scale: float = 1.0 / 3.0,
        wd_exclude_norms: bool = True,
    ) -> None:
        super().__init__()
        if stage not in tp.get_args(_Stage):
            raise ValueError(f"stage={stage!r} not in {tp.get_args(_Stage)}")
        # B33 §5 discriminative-LR scales for 3b, now hyperparameters. Front-end
        # rides at base·frontend_lr_scale (default 0.1 = base/10; pretrained in
        # P1 + the P2 LR/10 group); parcel side at base·parcel_lr_scale
        # (default 1/3 = base/3). 0.0 freezes that sub-group in 3b
        # (R-3b-frontend-frozen / R-3b-parcel-frozen); a uniform 1.0/1.0 is
        # R-3b-lr-uniform. >1.0 would out-pace the connector — rejected.
        for _nm, _sc in (
            ("frontend_lr_scale", frontend_lr_scale),
            ("parcel_lr_scale", parcel_lr_scale),
        ):
            if not 0.0 <= _sc <= 1.0:
                raise ValueError(f"{_nm} must lie in [0.0, 1.0]; got {_sc}")

        self.encoder = encoder
        self._frontend_lr_scale = frontend_lr_scale
        self._parcel_lr_scale = parcel_lr_scale
        self._wd_exclude_norms = bool(wd_exclude_norms)
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

        # Per-step grad-divergence monitor (readiness-audit finding c). P3a/P3b
        # previously had only val-cadence guarding; mirror P1/P2's
        # on_before_optimizer_step so the CollapseGuardCallback's per-step
        # train_mon_grad_diverged abort is live in the two longest phases too.
        # Persisted so the EMA baseline survives a fit-loop restart.
        self.register_buffer(
            "_grad_ema_l2",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )

    # ------------------------------------------------------------------
    # Stage freeze
    # ------------------------------------------------------------------

    def _apply_stage_freeze(self) -> None:
        """3a freezes the encoder (connector-only warmup); 3b unfreezes it.

        The PMA + projector always train. ``requires_grad`` is the freeze
        contract the optimizer param-groups and the unit tests both read; 3a
        additionally runs the encoder forward under ``no_grad`` (see
        :meth:`_encoder_taps`) so frozen-encoder activations are never retained.

        In 3b a ``0.0`` LR scale freezes that encoder sub-group
        (R-3b-frontend-frozen / R-3b-parcel-frozen): the sub-group keeps
        ``requires_grad=False`` and is dropped from the optimizer.
        """
        train_encoder = self._stage == "3b"
        self.encoder.requires_grad_(train_encoder)
        if train_encoder:
            frontend, parcel = self.encoder.partition_parameters_for_staging()
            if self._frontend_lr_scale == 0.0:
                for p in frontend:
                    p.requires_grad_(False)
            if self._parcel_lr_scale == 0.0:
                for p in parcel:
                    p.requires_grad_(False)
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
                "return_taps=True; expected the M2/M4 tap dict (M3 opt-in via return_m3)."
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

    def on_before_optimizer_step(
        self, optimizer: tp.Any,   # noqa: ARG002 — Lightning hook signature
    ) -> None:
        """Per-step grad-L2 + divergence probe (readiness-audit finding c).

        Mirrors :meth:`V14JointBrainModule.on_before_optimizer_step` so the
        CollapseGuardCallback's per-step ``train_mon_grad_diverged`` abort is
        live in P3a/P3b (which otherwise only validate-cadence guard). Frozen
        params (the 3a encoder; 3b ``0.0``-scale sub-groups) carry no grad and
        are skipped, so the global L2 is over the actually-trained params.
        """
        sq_sum = torch.zeros(
            (), device=self._grad_ema_l2.device, dtype=torch.float32,
        )
        for p in self.parameters():
            if p.grad is not None:
                sq_sum = sq_sum + p.grad.detach().to(torch.float32).pow(2).sum()
        grad_l2 = float(sq_sum.sqrt().item())
        verdict = grad_spike_monitor(
            grad_l2=grad_l2,
            prior_ema_l2=float(self._grad_ema_l2.item()),
        )
        self._grad_ema_l2.fill_(verdict.new_grad_ema_l2)
        self.log("train_mon_grad_l2", verdict.grad_l2, on_step=True)
        self.log("train_mon_grad_ema_l2", verdict.grad_ema_l2, on_step=True)
        self.log("train_mon_grad_spike_ratio", verdict.spike_ratio, on_step=True)
        self.log(
            "train_mon_grad_spike", 1.0 if verdict.is_spike else 0.0, on_step=True,
        )
        self.log(
            "train_mon_grad_diverged",
            1.0 if verdict.is_diverged else 0.0,
            on_step=True,
        )

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
        """3a: connector params only. 3b: up to 3 discriminative-LR groups.

        3b groups (B33 §5, now hyperparameters): front-end @
        ``base·frontend_lr_scale`` (default base/10), parcel side @
        ``base·parcel_lr_scale`` (default base/3), connector @ full base. A
        ``0.0`` scale drops that group (the sub-group is frozen in
        :meth:`_apply_stage_freeze`), so the frozen falsifiers run with 2
        groups. The connector group is always present.
        """
        if self._stage == "3a":
            return self._connector_parameters()
        frontend, parcel = self.encoder.partition_parameters_for_staging()
        base_lr = self._base_lr()
        groups: list[tp.Any] = []
        if self._frontend_lr_scale > 0.0:
            groups.append({"params": frontend, "lr": base_lr * self._frontend_lr_scale})
        if self._parcel_lr_scale > 0.0:
            groups.append({"params": parcel, "lr": base_lr * self._parcel_lr_scale})
        groups.append({"params": self._connector_parameters(), "lr": base_lr})
        return groups

    def _estimated_total_steps(self) -> int | None:
        try:
            return int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            return None

    def configure_optimizers(self):  # type: ignore[override]
        params = self._phase_param_groups()
        # §7/B01 no-WD split (task #40): exempt biases / LN γβ / embeds from a
        # non-zero weight_decay. No-op at wd=0 or under --no-wd-exclude-norms.
        params = maybe_split_no_decay(
            params,
            modules=(self,),
            optim_config=self.optim_config,
            exclude=self._wd_exclude_norms,
        )
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

    # #82: P3 distillation must never pretrain on a Neuroprobe off-limits eval
    # session. Enforced fail-closed in Experiment._train_and_test.
    enforces_pretrain_leakage_guard: tp.ClassVar[bool] = True
    # Anti-starvation floor: P3 drops (8,0) (subject 8's only session, no Whisper
    # teacher cache), so the distill corpus is 6 subjects {1,2,3,4,6,9} / 12
    # sessions = V14_P3_DISTILL_SESSIONS.
    pretrain_cohort_floor: tp.ClassVar[tuple[int, int] | None] = (6, 12)

    projector_mode: tp.Literal["mlp", "linear"] = "mlp"
    # B33: per-channel target standardization is mandatory by default; the
    # falsifier R-no-target-standardize sets this False (raw 1280-d target).
    target_standardize: bool = True
    # Path to the train-only channel stats ({"mean", "inv_std"}) saved by
    # fit_channel_stats. Required when target_standardize is True.
    channel_stats_path: str | None = None
    # CN-3 consumer self-check: the channel-stats affine is 1280-d wide for BOTH
    # mean_all and L8, so a stats file fit under the wrong layer merge would
    # silently z-score the distill target with the wrong frozen affine (no shape
    # error). The dispatch CLI guards this at the junction, but a directly
    # constructed experiment (tests/notebook/alternate dispatcher) bypasses that;
    # _build_standardizer asserts the loaded stats' provenance matches this.
    whisper_layer_merge: int | str = DEFAULT_LAYER_MERGE
    distill: PhaseThreeDistillationConfig = pydantic.Field(
        default_factory=PhaseThreeDistillationConfig,
    )
    # B33 §5 3b discriminative-LR scales, now hyperparameters. Front-end @
    # base·frontend_lr_scale (default 0.1 = base/10); parcel side @
    # base·parcel_lr_scale (default 1/3 = base/3); connector @ full base.
    # Falsifiers: 0.0 freezes that sub-group (R-3b-frontend-frozen /
    # R-3b-parcel-frozen); 1.0/1.0 = R-3b-lr-uniform. No effect under 3a
    # (connector-only warmup).
    frontend_lr_scale: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)
    parcel_lr_scale: float = pydantic.Field(default=1.0 / 3.0, ge=0.0, le=1.0)

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
        # CN-3: assert-if-present provenance match (legacy untagged stats pass →
        # no cache invalidation, same contract as the dispatch-junction guard).
        stats_merge = stats.get("layer_merge") if isinstance(stats, dict) else None
        if stats_merge is not None and merge_slug(stats_merge) != merge_slug(self.whisper_layer_merge):
            raise ValueError(
                f"channel_stats_path {self.channel_stats_path!r} was fit on "
                f"layer_merge={stats_merge!r} ({merge_slug(stats_merge)}) but the "
                f"experiment's whisper_layer_merge={self.whisper_layer_merge!r} "
                f"({merge_slug(self.whisper_layer_merge)}) — the 1280-d affine "
                "would z-score the distill target with the wrong frozen stats. "
                "Re-fit channel stats for this merge, or set whisper_layer_merge "
                "to match (CN-3)."
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
            frontend_lr_scale=self.frontend_lr_scale,
            parcel_lr_scale=self.parcel_lr_scale,
            wd_exclude_norms=self.wd_exclude_norms,
        )


__all__ = [
    "V14Phase3DistillModule",
    "V14Phase3Experiment",
]
