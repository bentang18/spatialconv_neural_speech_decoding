"""B2.2 — :class:`V14JointBrainModule` Lightning module.

Composes the B31 V-JEPA-2-canonical 2-term v14 joint SSL loss via
:func:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses`
(see [[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]):

* **Default ``loss_variant="b31_default"``** computes 2 terms — pure-L1
  ``L_pre_frame @ M2`` (V-JEPA 2 §2.1 Eq 1) + pure-L1
  ``L_post_frame @ M4`` gated by the B30 single-source-of-truth
  ``latent_valid``. ``L_mid_slot`` and ``L_post_utterance`` are dropped
  from the default surface; PMA is constructed only for the P3 readout.
* **Sister variants** (P0 falsifiers) re-add the dropped terms:
  ``b31_plus_m3`` adds ``L_mid_slot @ LN_mid(M3)``; ``b31_plus_utt``
  adds ``L_post_utterance @ LN_utt(M4)-PMA``; ``b31_plus_both`` restores
  the pre-B31 4-term default.
* student encoder forwarded with ``return_taps=True`` →
  ``{"M2", "M3", "M4"}`` (M3 tap kept for diagnostics + sisters).
* EMA teacher (full-input contract per B26) forwarded on the unmasked
  batch → same taps under ``torch.no_grad``.
* ``LN_frame`` (and conditionally ``LN_mid`` / ``LN_utt`` per sister)
  are student-owned with EMA-mirrored teacher copies.
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
    grad_spike_monitor,
    mask_orphan_ratio_monitor,
    parcel_coverage_monitor,
    teacher_rank_monitor,
)
from speech_decoding.models.v14_encoder import (
    V14ParcelCollapsePMA,
    V14ParcelPerceiverModel,
    _compute_latent_valid,
)
from speech_decoding.ssl.aggregator import (
    LossVariant,
    compute_v14_ssl_losses,
    variant_wants_m3,
    variant_wants_utt,
)
from speech_decoding.ssl.ema import (
    EmaTeacher,
    assert_teacher_full_input,
    fixed_ema_schedule,
    stop_grad,
)
from speech_decoding.ssl.total_loss import V14TotalLossBreakdown


_LossForm = tp.Literal["l1", "mse"]

# B31 sister-arm gating is single-sourced in
# ``speech_decoding.ssl.aggregator``; we just rebind under the local
# pre-fix names so existing call sites stay legible.
_variant_wants_m3 = variant_wants_m3
_variant_wants_utt = variant_wants_utt


class _V14StudentBundle(nn.Module):
    """All student-side modules that EMA-mirror into the teacher.

    Holds the encoder, ``ln_frame`` (always present — half of the B31
    2-term default), and, conditionally on ``loss_variant``, ``ln_mid``,
    ``ln_utt``, and the parcel-collapse PMA. ``EmaTeacher`` deepcopies
    this bundle so whatever heads exist on the student also mirror on
    the teacher.

    **B31 lock 2026-05-28 PM**: under ``loss_variant="b31_default"`` the
    ``ln_mid`` / ``ln_utt`` / ``pma`` modules are NOT constructed. The
    PMA query stops accumulating gradient in the joint SSL phase; it
    gets its first gradient at P3 (Whisper-L8 distillation, separate
    Experiment) and is frozen at P4. Sister variants reconstruct only
    the heads the chosen falsifier needs.

    The bundle's ``forward`` is the encoder forward; the heads above are
    accessed as attributes by :class:`V14JointBrainModule` per-tap, with
    runtime guards on the loss-variant gating.
    """

    def __init__(
        self,
        *,
        encoder: V14ParcelPerceiverModel,
        d_model: int,
        pma_n_heads: int,
        loss_variant: LossVariant = "b31_default",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        # ln_frame is always present — it's the LN applied to M4 for the
        # always-on L_post_frame term in the B31 default.
        self.ln_frame = nn.LayerNorm(d_model)
        # Sister-only heads. Construction is conditional so the default
        # joint SSL phase ships with no PMA / LN_mid / LN_utt on the
        # graph (closes the B19-vs-code PMA-training drift).
        if _variant_wants_m3(loss_variant):
            self.ln_mid = nn.LayerNorm(d_model)
        else:
            self.ln_mid = None  # type: ignore[assignment]
        if _variant_wants_utt(loss_variant):
            self.ln_utt = nn.LayerNorm(d_model)
            # PMA freeze=False: when the utterance sister fires, its PMA
            # query trains alongside the rest of the student. EMA
            # teacher mirrors it.
            self.pma = V14ParcelCollapsePMA(d_model, pma_n_heads, freeze=False)
        else:
            self.ln_utt = None  # type: ignore[assignment]
            self.pma = None  # type: ignore[assignment]

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
        loss_variant: LossVariant = "b31_default",
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
        if loss_variant not in tp.get_args(LossVariant):
            raise ValueError(
                f"loss_variant={loss_variant!r} not in "
                f"{tp.get_args(LossVariant)}"
            )

        self.student = _V14StudentBundle(
            encoder=encoder,
            d_model=encoder.d_model,
            pma_n_heads=pma_n_heads,
            loss_variant=loss_variant,
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
        self._loss_variant: LossVariant = loss_variant

        # MON-GRAD-SPIKE-DIVERGENCE persistent EMA buffer. ``0.0`` seeds
        # the first step (the monitor skips spike detection until the
        # EMA has a baseline). Persisted in checkpoints so the buffer
        # survives ``trainer.fit_loop`` restarts.
        self.register_buffer(
            "_grad_ema_l2",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )

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
        # ln_frame is always on (B31 default + every sister). LN_mid +
        # LN_utt + PMA are constructed only when the chosen
        # ``loss_variant`` selects them; we mirror that conditional
        # construction at the per-tap forward.
        m4_s_lnframe = self.student.ln_frame(m4_s)
        m4_t_lnframe = self.teacher.model.ln_frame(m4_t).detach()

        m3_s_lnmid: tp.Optional[Tensor] = None
        m3_t_lnmid: tp.Optional[Tensor] = None
        if _variant_wants_m3(self._loss_variant):
            m3_s_lnmid = self.student.ln_mid(m3_s)
            m3_t_lnmid = self.teacher.model.ln_mid(m3_t).detach()

        m4_s_lnutt: tp.Optional[Tensor] = None
        m4_t_lnutt: tp.Optional[Tensor] = None
        pma_student: tp.Optional[nn.Module] = None
        pma_teacher: tp.Optional[nn.Module] = None
        if _variant_wants_utt(self._loss_variant):
            m4_s_lnutt = self.student.ln_utt(m4_s)
            m4_t_lnutt = self.teacher.model.ln_utt(m4_t).detach()
            pma_student = self.student.pma
            pma_teacher = self.teacher.model.pma

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
            # Compose remaining terms via the aggregator with a zero
            # scalar M2 pair to make L_pre_frame structurally 0; add the
            # fallback term to .total below.
            zero_pair = torch.zeros((), device=m2_s.device, dtype=m2_s.dtype)
            breakdown = compute_v14_ssl_losses(
                student_m2_pred=zero_pair,
                m2_target=zero_pair,
                m2_valid_mask=None,
                student_m4_lnframe=m4_s_lnframe,
                teacher_m4_lnframe=m4_t_lnframe,
                latent_valid=latent_valid,
                loss_variant=self._loss_variant,
                student_m3_lnmid=m3_s_lnmid,
                teacher_m3_lnmid=m3_t_lnmid,
                student_m4_lnutt=m4_s_lnutt,
                teacher_m4_lnutt=m4_t_lnutt,
                pma_student=pma_student,
                pma_teacher=pma_teacher,
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
        return compute_v14_ssl_losses(
            student_m2_pred=student_m2_pred,
            m2_target=m2_t,
            m2_valid_mask=None,
            student_m4_lnframe=m4_s_lnframe,
            teacher_m4_lnframe=m4_t_lnframe,
            latent_valid=latent_valid,
            loss_variant=self._loss_variant,
            student_m3_lnmid=m3_s_lnmid,
            teacher_m3_lnmid=m3_t_lnmid,
            student_m4_lnutt=m4_s_lnutt,
            teacher_m4_lnutt=m4_t_lnutt,
            pma_student=pma_student,
            pma_teacher=pma_teacher,
            loss_form=self._loss_form,
        )

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
        self.log(f"{step_name}_l_post_frame", breakdown.l_post_frame, on_epoch=True)
        # B31 sister-only terms are ``None`` under the 2-term default.
        if breakdown.l_mid_slot is not None:
            self.log(f"{step_name}_l_mid_slot", breakdown.l_mid_slot, on_epoch=True)
        if breakdown.l_post_utterance is not None:
            self.log(
                f"{step_name}_l_post_utterance",
                breakdown.l_post_utterance,
                on_epoch=True,
            )

    # MON-TEACHER-FEATURE-RANK validation-time subsample cap. The SVD
    # cost is O(min(N, d) * N * d); capping N at 4096 keeps it under
    # ~30 ms on H100 while staying well above the d=256 ceiling for
    # the rank estimate to converge.
    _RANKME_N_MAX: tp.ClassVar[int] = 4096

    def _run_parcel_coverage_monitor(
        self, *, latent_valid: Tensor, step_name: str,
    ) -> None:
        """MON-PARCEL-COVERAGE-VARIANCE (5/28 P0).

        Cheap — bool reductions over the (B, L) ``latent_valid`` mask.
        Run on every train/val/test step so a degenerate ref-aug or
        shaft-mask combo trips an alarm immediately.
        """
        verdict = parcel_coverage_monitor(latent_valid)
        self.log(
            f"{step_name}_mon_coverage_active_mean",
            verdict.active_slots_per_clip_mean,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_active_cv",
            verdict.active_slots_per_clip_cv,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_slot_var",
            verdict.slot_usage_fraction_var,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_degenerate_frac",
            verdict.degenerate_clip_fraction,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_swec_frac",
            verdict.front_end_only_clip_fraction,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_alarm",
            1.0 if verdict.is_alarm else 0.0,
            on_epoch=True,
        )

    def _run_teacher_rank_monitor(
        self,
        *,
        teacher_m4_lnframe: Tensor,
        latent_valid: Tensor,
        step_name: str,
    ) -> None:
        """MON-TEACHER-FEATURE-RANK (5/28 P0).

        Computes RankMe on the EMA teacher's LN_frame-normalised M4 tap,
        masked by ``latent_valid`` so SWEC / front-end-only positions
        don't dilute the rank estimate. Sub-samples to
        ``_RANKME_N_MAX`` rows to keep the SVD cheap.

        Wired into ``validation_step`` via :meth:`_monitor_from_step`
        with the val-cadence (Lightning's default = every train epoch),
        which closely approximates the spec's 10k-step probe.
        """
        if teacher_m4_lnframe.dim() != 4:
            return  # not a (B, L, T, d) tap — skip silently
        B, L, T, d = teacher_m4_lnframe.shape
        if L != latent_valid.shape[1] or B != latent_valid.shape[0]:
            return  # shape mismatch — skip rather than crash a probe

        # Expand (B, L) → (B, L, T), flatten, gather only valid rows.
        expanded = latent_valid.unsqueeze(-1).expand(B, L, T)
        flat = teacher_m4_lnframe.reshape(B * L * T, d)
        mask_flat = expanded.reshape(B * L * T)
        valid_rows = flat[mask_flat]
        if valid_rows.shape[0] > self._RANKME_N_MAX:
            # Deterministic head-slice keeps the probe reproducible
            # across resumes; the rank estimate is permutation-invariant.
            valid_rows = valid_rows[: self._RANKME_N_MAX]

        verdict = teacher_rank_monitor(valid_rows.detach())
        self.log(
            f"{step_name}_mon_rankme", verdict.rankme, on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_rankme_normalised",
            verdict.rankme_normalised,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_rankme_warn",
            1.0 if verdict.is_warn else 0.0,
            on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_rankme_alarm",
            1.0 if verdict.is_alarm else 0.0,
            on_epoch=True,
        )

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
        """Run per-step monitors that need the encoder/teacher forwards.

        Re-doing the forward keeps each monitor independent of any
        training-step caching layout; the cost is two extra forwards on
        ~5% of steps (Lightning's val cadence). For training_step the
        caller can override the cadence by gating the call.

        Monitors fired here:

        * MON-PARCEL-COVERAGE-VARIANCE (5/28 P0) — cheap; no forward
          needed. Runs on every step.
        * MON-MASK-002 (B03d) — requires ``shaft_mask`` in the batch.
        * MON-TEACHER-FEATURE-RANK (5/28 P0) — RankMe on the EMA
          teacher's LN_frame M4 tap, masked by ``latent_valid``. Run
          on validation/test steps only (default Lightning cadence
          ≈ 10k-step probe spec).
        """
        student_kwargs, _ = self._extract_student_kwargs(batch_data)

        latent_valid = _latent_valid_from_batch(
            support=student_kwargs["support"],
            valid_mask=student_kwargs.get("valid_mask"),
            m_sub_slots=self._m_sub_slots,
        )
        self._run_parcel_coverage_monitor(
            latent_valid=latent_valid, step_name=step_name,
        )

        needs_orphan = "shaft_mask" in batch_data
        needs_rankme = step_name in ("val", "test")
        if not (needs_orphan or needs_rankme):
            return

        with torch.no_grad():
            teacher_taps = self.teacher.model(**student_kwargs)
            m4_t_lnframe = self.teacher.model.ln_frame(teacher_taps["M4"])

            if needs_orphan:
                student_taps = self.student(**student_kwargs)
                m4_s_lnframe = self.student.ln_frame(student_taps["M4"])
                self._run_mask_orphan_monitor(
                    batch_data=batch_data,
                    student_m4_lnframe=m4_s_lnframe,
                    teacher_m4_lnframe=m4_t_lnframe,
                    step_name=step_name,
                )
            if needs_rankme:
                self._run_teacher_rank_monitor(
                    teacher_m4_lnframe=m4_t_lnframe,
                    latent_valid=latent_valid,
                    step_name=step_name,
                )

    def on_before_optimizer_step(
        self, optimizer: tp.Any,   # noqa: ARG002 — Lightning hook signature
    ) -> None:
        """MON-GRAD-SPIKE-DIVERGENCE (5/28 P0) — SPAM grad-L2 + spike
        probe before ``optimizer.step()``.

        Reads the post-backward grads off the student parameters (the
        teacher has no grads), computes the global L2, compares it to
        the persistent EMA buffer, and updates the buffer in-place.

        Done before the optimiser step so a flagged step still gets the
        update (the alarm is informational; the dispatch callback owns
        any rollback / LR backoff escalation per SPAM §3.1).
        """
        sq_sum = torch.zeros(
            (), device=self._grad_ema_l2.device, dtype=torch.float32,
        )
        for p in self.student.parameters():
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
            "train_mon_grad_spike",
            1.0 if verdict.is_spike else 0.0,
            on_step=True,
        )
        self.log(
            "train_mon_grad_diverged",
            1.0 if verdict.is_diverged else 0.0,
            on_step=True,
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
