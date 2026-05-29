"""B2.2 — :class:`V14JointBrainModule` Lightning module.

Composes the B28/B29 4-term v14 joint SSL loss via
:func:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses`:

* student encoder forwarded with ``return_taps=True`` →
  ``{"M2", "M3", "M4"}``.
* EMA teacher (full-input contract per B26) forwarded on the unmasked
  batch → ``{"M2", "M3", "M4"}`` under ``torch.no_grad``.
* ``LN_mid`` / ``LN_frame`` / ``LN_utt`` (student-owned) and the EMA-
  mirrored teacher copies normalise the M3 / M4 taps before they flow
  into the aggregator.
* :class:`V14ParcelCollapsePMA` (student-owned) + EMA-mirrored teacher
  PMA feed the utterance term via :func:`pma_then_mean` — the B30
  single-source-of-truth ``latent_valid`` is threaded everywhere.
* :func:`compute_v14_ssl_losses` returns the
  :class:`V14TotalLossBreakdown`; ``.total`` is the scalar the optimiser
  steps on.

EMA discipline (B26 lock 2026-05-27 PM): fixed τ=0.999 via
:func:`speech_decoding.ssl.ema.fixed_ema_schedule`; ``on_train_batch_end``
applies :meth:`EmaTeacher.update_from` so the teacher trails the student
exactly one optimiser step behind.

Predictor (B03c paradigm-B) is wired *optionally* — when the dispatch
plumbs a :class:`speech_decoding.models.v14_encoder.Predictor2Block` into
the module via ``predictor=...``, ``L_pre_frame`` is computed on the
predictor's output at masked positions per the B03 lock. Until B2.3
lands the masking extractors + the patch-drop / shaft-drop forward
threading, the predictor stays ``None`` and ``L_pre_frame`` falls back
to a direct ``F.l1_loss(student_M2, stop_grad(teacher_M2))`` over the
full M2 tap — a degenerate fallback that keeps the aggregator path
unit-testable end-to-end without the B2.3 plumbing. The fallback's
graph-level structure (per-element L1 between student-pred and detached
teacher-target) IS the structural contract the predictor path will
preserve once it lands.

Batch contract (B30 single source of truth) — ``batch.data`` keys read::

  electrode_tokens : (B, C, T_bins, F_bins)  required
  support          : (B, C, K)               required
  valid_mask       : (B, C) bool             optional (default: all-True)
  shaft_mask       : (B, C) bool             optional — student-only,
                                             never passed to teacher.
  subject_subtype  : (B,) or (B, 1[, 1]) int optional
  ref_idx          : (B,) or (B, 1[, 1]) int optional
  lambda_anat      : (B,) float              optional (default: 1.0)

Sister-flag gating: ``V14JointBrainModule`` honours
``latent_valid_override`` / ``sa_mask_mode`` only at their lock-default
values (``"support"`` / ``"bidirectional"``); non-default values raise
:class:`NotImplementedError` at construction so the
``B30-dispatch-sister-flags`` drift row's runtime branch stays explicit.
"""

from __future__ import annotations

import typing as tp

import torch
from lightning import pytorch as pl
from torch import Tensor, nn

from neuraltrain.optimizers import BaseOptimizer

from speech_decoding.experiments.monitors import (
    compute_orphan_parcels,
    mask_orphan_ratio_monitor,
)
from speech_decoding.models.v14_encoder import (
    V14ParcelCollapsePMA,
    V14ParcelPerceiverModel,
    _compute_latent_valid,
)
from speech_decoding.ssl.aggregator import compute_v14_ssl_losses
from speech_decoding.ssl.ema import (
    EmaTeacher,
    assert_teacher_full_input,
    fixed_ema_schedule,
    stop_grad,
)
from speech_decoding.ssl.total_loss import V14TotalLossBreakdown


_LossForm = tp.Literal["l1", "mse"]


class _V14StudentBundle(nn.Module):
    """All student-side modules that EMA-mirror into the teacher.

    Holds the encoder, the 3 SSL LN heads (``ln_mid`` / ``ln_frame`` /
    ``ln_utt``), and the parcel-collapse PMA. ``EmaTeacher`` deepcopies
    this bundle so the teacher's LN heads + PMA also mirror.

    The bundle's ``forward`` is the encoder forward; LN heads + PMA are
    accessed as attributes by :class:`V14JointBrainModule` per-tap.
    """

    def __init__(
        self,
        *,
        encoder: V14ParcelPerceiverModel,
        d_model: int,
        pma_n_heads: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.ln_mid = nn.LayerNorm(d_model)
        self.ln_frame = nn.LayerNorm(d_model)
        self.ln_utt = nn.LayerNorm(d_model)
        # PMA freeze=False so the student's PMA query trains during the
        # joint phase per B19 (P1+P2 train PMA; P4 freezes). EMA teacher
        # mirrors it.
        self.pma = V14ParcelCollapsePMA(d_model, pma_n_heads, freeze=False)

    def forward(self, **encoder_kwargs: tp.Any) -> dict[str, Tensor]:
        out = self.encoder(return_taps=True, **encoder_kwargs)
        if not isinstance(out, dict):
            raise RuntimeError(
                "V14ParcelPerceiverModel returned a single tensor under "
                "return_taps=True; expected the M2/M3/M4 tap dict."
            )
        return out


def _maybe_drop_singleton_trailing(t: tp.Optional[Tensor]) -> tp.Optional[Tensor]:
    """Mirror the encoder's NeuralSet-collate normalisation: strip
    trailing singleton axes from ``(B, 1[, 1])`` → ``(B,)``."""
    if t is None:
        return None
    # Bind to a non-Optional local so pyright keeps the narrowing across
    # the reassignment inside the loop body.
    out: Tensor = t
    while out.dim() > 1 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    return out


def _latent_valid_from_batch(
    *,
    support: Tensor,
    valid_mask: tp.Optional[Tensor],
    m_sub_slots: int,
) -> Tensor:
    """B30 single source of truth: ``(B, L) bool`` from
    ``(support.sum(over electrodes) > 0)`` expanded across ``M`` sub-slots.

    Mirrors the encoder's :func:`_compute_latent_valid` so the aggregator
    receives the SAME tensor the encoder's latent-SA ``attn_mask``
    consumed inside ``forward``.
    """
    return _compute_latent_valid(
        support=support, valid_mask=valid_mask, m_sub_slots=m_sub_slots,
    )


class V14JointBrainModule(pl.LightningModule):
    """B29 Item 1 + B30 lock — joint SSL Lightning module.

    Owns:

    * ``student`` — :class:`_V14StudentBundle` with the encoder + 3 LN
      heads + PMA.
    * ``teacher`` — :class:`EmaTeacher` deepcopy of ``student``,
      ``requires_grad=False`` on every parameter, τ=0.999 fixed.
    * ``predictor`` — optional :class:`Predictor2Block` for the B03c
      paradigm-B masked-patch path. ``None`` until B2.3 lands the patch-
      drop forward threading; the fallback L_pre_frame path is documented
      on the module docstring.

    Sister-flag gates: non-default ``latent_valid_override`` /
    ``sa_mask_mode`` raise :class:`NotImplementedError` at construction
    so the drift-table row ``B30-dispatch-sister-flags`` is enforced at
    the runtime boundary as well.

    Notes
    -----
    The Lightning training-step pulls the batch from a
    :class:`~speech_decoding.experiments.data.Data`-built loader. The
    BrainModule does NOT depend on the data pipeline at construction
    time — it can be instantiated and exercised with a synthetic batch
    (used by unit tests).
    """

    def __init__(
        self,
        *,
        encoder: V14ParcelPerceiverModel,
        optim_config: BaseOptimizer,
        pma_n_heads: int = 8,
        ema_tau: float = 0.999,
        loss_form: _LossForm = "l1",
        predictor: tp.Optional[nn.Module] = None,
        latent_valid_override: str = "support",
        sa_mask_mode: str = "bidirectional",
    ) -> None:
        super().__init__()
        # B30 sister-flag runtime gates. Drift row B30-dispatch-sister-flags
        # documents these as wired at the dispatch surface but not yet at
        # the runtime path; this BrainModule is the runtime path.
        if latent_valid_override != "support":
            raise NotImplementedError(
                f"latent_valid_override={latent_valid_override!r} is a B30 "
                "sister falsifier (R-item-12-all-true / "
                "R-parcels-supervised-gating); the runtime branch lives in "
                "V14JointBrainModule._run_step and has not landed — see "
                "docs/neuroprobe/v14_blockers.md row B30-dispatch-sister-flags."
            )
        if sa_mask_mode != "bidirectional":
            raise NotImplementedError(
                f"sa_mask_mode={sa_mask_mode!r} is the B30 sister "
                "falsifier R-sa-key-only; the encoder's key-only branch "
                "has not landed — see docs/neuroprobe/v14_blockers.md "
                "row B30-dispatch-sister-flags."
            )
        if not 0.0 < ema_tau < 1.0:
            raise ValueError(
                f"ema_tau must be in (0.0, 1.0); got {ema_tau}"
            )

        self.student = _V14StudentBundle(
            encoder=encoder,
            d_model=encoder.d_model,
            pma_n_heads=pma_n_heads,
        )
        # B26 lock: fixed τ via EmaTeacher's coeff_schedule.
        self.teacher = EmaTeacher(
            self.student, coeff_schedule=fixed_ema_schedule(tau=ema_tau),
        )
        self.predictor = predictor
        self.optim_config = optim_config
        self._m_sub_slots = encoder.m_sub_slots
        self._d_model = encoder.d_model
        self._loss_form: _LossForm = loss_form

    # ------------------------------------------------------------------
    # Batch ingest
    # ------------------------------------------------------------------

    def _extract_student_kwargs(
        self, batch_data: dict[str, Tensor],
    ) -> tuple[dict[str, tp.Any], tp.Optional[Tensor]]:
        """Pull student-forward kwargs from the batch dict.

        Returns ``(kwargs, shaft_mask)`` — ``shaft_mask`` is kept
        separate so the teacher forward can drop it (B26 full-input
        contract).
        """
        if "electrode_tokens" not in batch_data:
            raise KeyError(
                "V14JointBrainModule batch missing 'electrode_tokens'; "
                "the SCAFFOLD-02 data pipeline must emit this key."
            )
        if "support" not in batch_data:
            raise KeyError(
                "V14JointBrainModule batch missing 'support'; the B30 "
                "latent_valid contract requires the per-clip support "
                "tensor in the batch."
            )

        kwargs: dict[str, tp.Any] = {
            "electrode_tokens": batch_data["electrode_tokens"],
            "support": batch_data["support"],
        }

        if "valid_mask" in batch_data:
            kwargs["valid_mask"] = batch_data["valid_mask"]

        # Per-clip conditioning + λ_anat are looked up by the encoder
        # itself; the encoder forward tolerates ``None``. Trailing
        # NeuralSet-collated singleton axes get stripped to match the
        # encoder's internal handling.
        if "subject_subtype" in batch_data:
            kwargs["subject_subtype"] = _maybe_drop_singleton_trailing(
                batch_data["subject_subtype"],
            )
        if "ref_idx" in batch_data:
            kwargs["ref_idx"] = _maybe_drop_singleton_trailing(
                batch_data["ref_idx"],
            )
        if "lambda_anat" in batch_data:
            kwargs["lambda_anat"] = _maybe_drop_singleton_trailing(
                batch_data["lambda_anat"],
            )

        shaft_mask = batch_data.get("shaft_mask")
        return kwargs, shaft_mask

    # ------------------------------------------------------------------
    # Per-step composition
    # ------------------------------------------------------------------

    def _compose_l_pre_frame(
        self,
        *,
        student_m2: Tensor,
        teacher_m2: Tensor,
    ) -> Tensor:
        """B03c paradigm-B predictor path when wired; fallback otherwise.

        Until B2.3 lands the patch-drop forward threading and the
        masking extractor, ``self.predictor is None`` and the fallback
        applies pure L1 between student M2 and detached teacher M2 over
        the full tap — degenerate but graph-correct so the joint trainer
        can be exercised end-to-end without B2.3.
        """
        if self.predictor is None:
            # Fallback: pure L1 over the full M2 tap. Detach teacher
            # branch per B26 JEPA contract.
            return torch.nn.functional.l1_loss(
                student_m2, stop_grad(teacher_m2),
                reduction="mean",
            )
        # Predictor path (B03c). The predictor accepts ``(B, N, d)``;
        # the caller is responsible for flattening the per-electrode M2
        # tap into the predictor's expected shape. This branch is here
        # for forward-compatibility but is not yet exercised — it's
        # gated on B2.3 wiring the patch-mask into the encoder
        # forward + threading visible/masked indices through.
        raise NotImplementedError(
            "V14JointBrainModule.predictor branch is gated on B2.3 "
            "(patch-mask + visible/masked-index threading from the "
            "extractor through to predictor input). See "
            "docs/neuroprobe/v14_blockers.md row B03 shaft-mask + "
            "REF-01/REF-02."
        )

    def _step(
        self, batch_data: dict[str, Tensor],
    ) -> V14TotalLossBreakdown:
        """One forward + loss compose pass. Returns the breakdown so the
        Lightning step can log per-term scalars."""
        student_kwargs, shaft_mask = self._extract_student_kwargs(batch_data)

        # ── Student forward (with shaft_mask if present) ──
        student_taps_kwargs = dict(student_kwargs)
        if shaft_mask is not None:
            student_taps_kwargs["shaft_mask"] = shaft_mask
        student_taps = self.student(**student_taps_kwargs)

        # ── Teacher forward (FULL input, no shaft, no mask). B26 ──
        # The teacher's ``self.teacher.model`` is the deepcopy of the
        # student bundle, whose forward already injects
        # ``return_taps=True`` into the encoder call — passing it again
        # here would raise "multiple values for keyword argument".
        assert_teacher_full_input(patch_mask=None, shaft_mask=None)
        with torch.no_grad():
            teacher_taps = self.teacher.model(**student_kwargs)
        if not isinstance(teacher_taps, dict):
            raise RuntimeError(
                "EMA teacher returned a single tensor; expected the M2/M3/M4 "
                "tap dict (return_taps=True)."
            )

        m2_s, m3_s, m4_s = student_taps["M2"], student_taps["M3"], student_taps["M4"]
        m2_t, m3_t, m4_t = teacher_taps["M2"], teacher_taps["M3"], teacher_taps["M4"]

        # ── LN heads (student + EMA-mirrored teacher) ──
        m3_s_lnmid = self.student.ln_mid(m3_s)
        m4_s_lnframe = self.student.ln_frame(m4_s)
        m4_s_lnutt = self.student.ln_utt(m4_s)

        m3_t_lnmid = self.teacher.model.ln_mid(m3_t).detach()
        m4_t_lnframe = self.teacher.model.ln_frame(m4_t).detach()
        m4_t_lnutt = self.teacher.model.ln_utt(m4_t).detach()

        # ── B30 single source of truth: latent_valid ──
        latent_valid = _latent_valid_from_batch(
            support=student_kwargs["support"],
            valid_mask=student_kwargs.get("valid_mask"),
            m_sub_slots=self._m_sub_slots,
        )

        # ── L_pre_frame (predictor path or fallback) ──
        # The aggregator computes L_pre_frame from (student_m2_pred,
        # m2_target) via recon_loss; for the fallback path we hand-roll
        # it here and stub the aggregator's L_pre_frame to zero, then
        # add ours into the total. That keeps the aggregator's surface
        # (predictor output vs target) clean once B2.3 wires the
        # predictor. To avoid double-counting, we route the predictor
        # path through the aggregator and the fallback path around it.
        if self.predictor is None:
            l_pre_frame_fallback = self._compose_l_pre_frame(
                student_m2=m2_s, teacher_m2=m2_t,
            )
            # Compose 3 slot/utterance terms via the aggregator with a
            # zero scalar M2 pair to make L_pre_frame structurally 0;
            # add the fallback term to .total below.
            zero_pair = torch.zeros((), device=m2_s.device, dtype=m2_s.dtype)
            breakdown, _clip_valid = compute_v14_ssl_losses(
                student_m2_pred=zero_pair,
                m2_target=zero_pair,
                m2_valid_mask=None,
                student_m3_lnmid=m3_s_lnmid,
                teacher_m3_lnmid=m3_t_lnmid,
                student_m4_lnframe=m4_s_lnframe,
                teacher_m4_lnframe=m4_t_lnframe,
                student_m4_lnutt=m4_s_lnutt,
                teacher_m4_lnutt=m4_t_lnutt,
                latent_valid=latent_valid,
                pma_student=self.student.pma,
                pma_teacher=self.teacher.model.pma,
                loss_form=self._loss_form,
            )
            # Materialize the fallback L_pre_frame term into the total +
            # breakdown by rebuilding the breakdown (frozen dataclass).
            total = (
                breakdown.total
                - breakdown.l_pre_frame                # subtract zero stub
                + l_pre_frame_fallback                 # add fallback term
            )
            from dataclasses import replace
            return replace(
                breakdown, total=total, l_pre_frame=l_pre_frame_fallback,
            )

        # Predictor path: route through the aggregator's L_pre_frame
        # branch. (Reached only after B2.3 lands the patch-drop
        # threading + the predictor input pipeline.)
        student_m2_pred = self._compose_l_pre_frame(
            student_m2=m2_s, teacher_m2=m2_t,
        )  # raises NotImplementedError; reserved for future use
        breakdown, _clip_valid = compute_v14_ssl_losses(
            student_m2_pred=student_m2_pred,
            m2_target=m2_t,
            m2_valid_mask=None,
            student_m3_lnmid=m3_s_lnmid,
            teacher_m3_lnmid=m3_t_lnmid,
            student_m4_lnframe=m4_s_lnframe,
            teacher_m4_lnframe=m4_t_lnframe,
            student_m4_lnutt=m4_s_lnutt,
            teacher_m4_lnutt=m4_t_lnutt,
            latent_valid=latent_valid,
            pma_student=self.student.pma,
            pma_teacher=self.teacher.model.pma,
            loss_form=self._loss_form,
        )
        return breakdown

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def _log_breakdown(
        self,
        breakdown: V14TotalLossBreakdown,
        *,
        step_name: str,
    ) -> None:
        self.log(f"{step_name}_loss", breakdown.total, on_epoch=True, prog_bar=True)
        self.log(f"{step_name}_l_pre_frame", breakdown.l_pre_frame, on_epoch=True)
        self.log(f"{step_name}_l_mid_slot", breakdown.l_mid_slot, on_epoch=True)
        self.log(f"{step_name}_l_post_frame", breakdown.l_post_frame, on_epoch=True)
        self.log(f"{step_name}_l_post_utterance", breakdown.l_post_utterance, on_epoch=True)

    def _run_mask_orphan_monitor(
        self,
        *,
        batch_data: dict[str, Tensor],
        student_m4_lnframe: Tensor,
        teacher_m4_lnframe: Tensor,
        step_name: str,
    ) -> None:
        """MON-MASK-002 (B03d): orphan/visible MSE ratio per step.

        Called from training_step / validation_step when ``shaft_mask``
        is present in the batch. Pure-function monitor; the sustain
        window + sister-escalation logic lives in the training-loop
        callback, NOT here. This hook only emits the instantaneous
        ratio + escalation tags as Lightning scalars.
        """
        shaft_mask = batch_data.get("shaft_mask")
        if shaft_mask is None:
            return
        support = batch_data["support"]
        orphan_parcels = compute_orphan_parcels(shaft_mask, support)  # (K,)
        # Expand parcel-level (K,) → slot-level (K*M,) so the orphan
        # vector aligns with the post-frame tap's parcel-axis = slot.
        if self._m_sub_slots > 1:
            orphan_slots = orphan_parcels.repeat_interleave(self._m_sub_slots)
        else:
            orphan_slots = orphan_parcels
        verdict = mask_orphan_ratio_monitor(
            student_post_frame=student_m4_lnframe,
            teacher_post_frame=teacher_m4_lnframe,
            orphan_parcels=orphan_slots,
            parcel_dim=1,  # (B, P, T, d) → parcel at axis 1
        )
        if verdict.ratio == verdict.ratio:  # not nan
            self.log(
                f"{step_name}_mon_mask_002_ratio", float(verdict.ratio),
                on_epoch=True,
            )
            self.log(
                f"{step_name}_mon_mask_002_in_band",
                1.0 if verdict.in_band else 0.0,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        breakdown = self._step(batch.data)
        self._log_breakdown(breakdown, step_name="train")
        self._monitor_from_step(batch.data, step_name="train")
        return breakdown.total

    def validation_step(self, batch, batch_idx: int) -> None:  # noqa: ARG002
        breakdown = self._step(batch.data)
        self._log_breakdown(breakdown, step_name="val")
        self._monitor_from_step(batch.data, step_name="val")

    def test_step(self, batch, batch_idx: int) -> None:  # noqa: ARG002
        breakdown = self._step(batch.data)
        self._log_breakdown(breakdown, step_name="test")
        self._monitor_from_step(batch.data, step_name="test")

    def _monitor_from_step(
        self, batch_data: dict[str, Tensor], *, step_name: str,
    ) -> None:
        """Re-run the encoder + teacher forward in eval/no-grad mode to
        feed MON-MASK-002 without retaining the training-step graph.

        Re-doing the forward keeps the monitor independent of any
        training-step caching layout; the cost is two extra forwards on
        ~5% of steps (Lightning's val cadence). For training_step the
        caller can override the cadence by gating the call.
        """
        if "shaft_mask" not in batch_data:
            return
        student_kwargs, _ = self._extract_student_kwargs(batch_data)
        with torch.no_grad():
            student_taps = self.student(**student_kwargs)
            teacher_taps = self.teacher.model(**student_kwargs)
            m4_s_lnframe = self.student.ln_frame(student_taps["M4"])
            m4_t_lnframe = self.teacher.model.ln_frame(teacher_taps["M4"])
            self._run_mask_orphan_monitor(
                batch_data=batch_data,
                student_m4_lnframe=m4_s_lnframe,
                teacher_m4_lnframe=m4_t_lnframe,
                step_name=step_name,
            )

    def on_train_batch_end(
        self,
        outputs: tp.Any,           # noqa: ARG002 — Lightning hook signature
        batch: tp.Any,             # noqa: ARG002
        batch_idx: int,            # noqa: ARG002
    ) -> None:
        # B26 EMA step. ``update_from`` advances the schedule step
        # internally; coeff is fixed at τ=0.999 under the B26 lock.
        self.teacher.update_from(self.student)

    def configure_optimizers(self):  # type: ignore[override]
        try:
            return self.optim_config.build(
                self.student.parameters(),
                total_steps=self.trainer.estimated_stepping_batches,
            )
        except TypeError:
            return self.optim_config.build(self.student.parameters())


__all__ = [
    "V14JointBrainModule",
    "_V14StudentBundle",  # exported for tests / dispatch override
]
