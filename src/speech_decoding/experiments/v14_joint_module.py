"""B36 WS-B — :class:`V14JointBrainModule` masked-JEPA Lightning module.

Composes the B36 paradigm-B masked-JEPA SSL loss (one active term per phase,
B9 — see [[project_v14_b36_perparcel_pool_structured_jepa_2026_06_01]]),
replacing the inert B31 2-term self-distill (full input both sides, no mask,
no predictor):

* **P1 (``phase="p1"``)** — front-end M2 masked JEPA. A structured 1D
  spectro-temporal band ``token_mask`` (6/03 lock, held-out 0.50: whole
  time-columns ∪ whole freq-rows) zeroes the masked front-end cells UPSTREAM
  of the token blocks (paradigm A — the token blocks self-predict). Loss = L1
  between the student's masked M2 tokens and the EMA teacher's full-input M2
  (post-``frontend_ln``). The pool / inter-parcel encoder / predictor are
  downstream of M2 → no gradient. The per-cell Bernoulli ``"random"`` shape is
  the R-m2-random must-beat sister.
* **P2 (``phase="p2"``)** — parcel M4 masked JEPA (paradigm B). A parcel
  ``"tube"`` mask (6/03 lock: a uniform-random 0.20 subset of COVERED parcels,
  each masked across ALL time-patches) drops the masked parcels' electrodes
  upstream AND excludes them from the inter-parcel self-attention keys
  (visible-only encoder). The
  :class:`~speech_decoding.models.v14_encoder.JepaPredictor` (cross-time)
  predicts the masked parcel-time M4 cells from the visible parcel tokens;
  target = EMA teacher full-input M4 (post-``encoder_ln``). The
  contiguous-time-block shape is the ``"time_block"`` sister (pairs with a
  ``co_temporal`` predictor — never ``cross_time``; that is the H1 leak).

Both targets are ``detach()``ed (stop-grad on the EMA teacher, V-JEPA 2 §2.1)
and normalized only by the encoder's own terminal LayerNorm — there is NO
separate ``ln_frame`` head (B6 / B36 §4 canonical V-JEPA target-norm). The
predictor is student-only (never EMA-mirrored).

EMA discipline (B26 lock 2026-05-27 PM): fixed τ=0.99925 via
:func:`speech_decoding.ssl.ema.fixed_ema_schedule`; ``on_train_batch_end``
applies :meth:`EmaTeacher.update_from` so the teacher trails the student
exactly one optimiser step behind.

Batch contract (B30 single source of truth) — ``batch.data`` keys read::

  electrode_tokens : (B, C, T_bins, F_bins)  required
  support          : (B, C, K)               required
  valid_mask       : (B, C) bool             optional (default: all-True)
  subject_subtype  : (B,) or (B, 1[, 1]) int optional
  ref_idx          : (B,) or (B, 1[, 1]) int optional

(``shaft_mask`` is dropped from the B36 default SSL path — WS-H5; the mask is
the SSL signal now, not a shaft drop.)

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
    JepaPredictor,
    V14ParcelPerceiverModel,
    _compute_latent_valid,
    compute_latent_valid_3way,
)
from speech_decoding.ssl.ema import (
    EmaTeacher,
    assert_teacher_full_input,
    fixed_ema_schedule,
)
from speech_decoding.ssl.mask import (
    sample_m2_mask,
    sample_m4_mask,
    validate_m4_coupling,
)
from speech_decoding.ssl.masked_jepa import (
    MaskedJepaBreakdown,
    p1_frontend_m2_loss,
    p2_parcel_m4_loss,
)


_LossForm = tp.Literal["l1", "mse"]
_Phase = tp.Literal["p1", "p2"]


class _V14StudentBundle(nn.Module):
    """The student encoder, wrapped so the :class:`EmaTeacher` deepcopy
    mirrors exactly the modules that supply the JEPA targets.

    B36 §4 (canonical V-JEPA target-norm) deletes the per-loss-head
    ``ln_frame`` / ``ln_mid`` / ``ln_utt`` and the SSL-phase PMA: the
    target is the EMA teacher's terminal-LN tap (``frontend_ln`` at M2,
    ``encoder_ln`` at M4), both INSIDE the encoder. So the bundle is now
    just the encoder. The predictor is student-only and lives on the
    :class:`V14JointBrainModule` (never EMA-mirrored — V-JEPA predictors
    are not part of the teacher).
    """

    def __init__(self, *, encoder: V14ParcelPerceiverModel) -> None:
        super().__init__()
        self.encoder = encoder

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
    """B36 WS-B — staged masked-JEPA SSL Lightning module.

    Owns:

    * ``student`` — :class:`_V14StudentBundle` (the encoder; ``frontend_ln``
      / ``encoder_ln`` are inside it and supply the terminal-LN targets).
    * ``teacher`` — :class:`EmaTeacher` deepcopy of ``student``,
      ``requires_grad=False`` on every parameter, τ=0.99925 fixed.
    * ``predictor`` — the student-only :class:`JepaPredictor` (paradigm-B
      parcel predictor). Constructed here if not supplied; used only in P2
      (P1 is paradigm A — the front-end token blocks self-predict). Never
      EMA-mirrored.

    ``phase`` selects which single term is active (B9):
    ``"p1"`` → front-end M2 (``m2_mask_type`` bands/random, held-out
    ``m2_mask_ratio``); ``"p2"`` → parcel M4 (``m4_mask_type`` tube/time_block,
    ``m4_mask_ratio`` of covered parcels). ``predictor_scope`` must couple with
    ``m4_mask_type`` (tube↔cross_time, time_block↔co_temporal); the constructor
    enforces this and rejects the H1-leak pairing.

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
        phase: _Phase = "p1",
        m2_mask_type: str = "bands",
        m2_mask_ratio: float = 0.50,
        m2_time_band_floor: int = 2,
        m2_freq_band_floor: int = 1,
        m2_time_freq_split: tp.Optional[tuple[float, float]] = None,
        m4_mask_type: str = "tube",
        m4_mask_ratio: float = 0.20,
        m4_n_min_visible: int = 3,
        predictor_scope: str = "cross_time",
        mask_seed: int = 0,
        ema_tau: float = 0.99925,
        loss_form: _LossForm = "l1",
        predictor: tp.Optional[JepaPredictor] = None,
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
            raise ValueError(f"ema_tau must be in (0.0, 1.0); got {ema_tau}")
        if phase not in tp.get_args(_Phase):
            raise ValueError(f"phase={phase!r} not in {tp.get_args(_Phase)}")
        # 6/03 masking lock: mask shape ↔ predictor scope must move as a pair,
        # or the M4 SSL task leaks (time_block+cross_time = the H1 leak). Cheap
        # insurance at construction, before any step runs.
        validate_m4_coupling(m4_mask_type, predictor_scope)
        # The co_temporal predictor (the time_block sister's other half) is not
        # built — only the cross_time JepaPredictor exists. Gate it the same way
        # as the other not-yet-landed sister falsifiers above so the time_block
        # SSL path fails loud at construction rather than silently mis-predicting.
        if predictor_scope != "cross_time":
            raise NotImplementedError(
                f"predictor_scope={predictor_scope!r} (the M4 'time_block' "
                "sister's co_temporal predictor) has not landed — only the "
                "cross_time JepaPredictor is built. See WS-B6 / "
                "reports/b36_masking_handoff_2026_06_03.md."
            )

        self.student = _V14StudentBundle(encoder=encoder)
        # B26 lock: fixed τ via EmaTeacher's coeff_schedule. The teacher
        # deepcopies the encoder (incl. frontend_ln / encoder_ln) — the
        # terminal-LN targets are EMA-mirrored automatically (B1).
        self.teacher = EmaTeacher(
            self.student, coeff_schedule=fixed_ema_schedule(tau=ema_tau),
        )
        # Student-only paradigm-B predictor (B2). NOT part of the teacher.
        self.predictor = predictor or JepaPredictor(encoder.d_model)
        self.optim_config = optim_config
        self._phase: _Phase = phase
        self._m2_mask_type = m2_mask_type
        self._m2_mask_ratio = m2_mask_ratio
        self._m2_time_band_floor = m2_time_band_floor
        self._m2_freq_band_floor = m2_freq_band_floor
        self._m2_time_freq_split = m2_time_freq_split
        self._m4_mask_type = m4_mask_type
        self._m4_mask_ratio = m4_mask_ratio
        self._m4_n_min_visible = m4_n_min_visible
        self._predictor_scope = predictor_scope
        self._mask_seed = mask_seed
        self._m_sub_slots = encoder.m_sub_slots
        self._d_model = encoder.d_model
        self._loss_form: _LossForm = loss_form

        # MON-GRAD-SPIKE-DIVERGENCE persistent EMA buffer. ``0.0`` seeds
        # the first step (the monitor skips spike detection until the
        # EMA has a baseline). Persisted in checkpoints so the buffer
        # survives ``trainer.fit_loop`` restarts.
        self.register_buffer(
            "_grad_ema_l2",
            torch.zeros((), dtype=torch.float32),
            persistent=True,
        )

    def _trainable_parameters(self) -> tp.Iterator[nn.Parameter]:
        """Student encoder + predictor params (the teacher is frozen)."""
        yield from self.student.parameters()
        yield from self.predictor.parameters()

    # ------------------------------------------------------------------
    # B36 WS-E (E3/E4): cross-phase weight handoff
    # ------------------------------------------------------------------

    def transferable_state(self) -> dict[str, dict[str, Tensor]]:
        """Components that carry forward to the next phase (E4).

        P1/P2 export ONLY the shared encoder. The student-only predictor and
        the EMA teacher are not transferred — the predictor is re-trained per
        phase, and the next phase re-syncs its teacher from the loaded
        student. P3/P4 modules extend this with a ``"pma"`` entry; the handoff
        loader matches on the intersection of component names, so the P3-only
        ``StudentWhisperProjector`` is simply absent here and never crosses to
        P4 (E4 "projector keys dropped").
        """
        return {"encoder": self.student.encoder.state_dict()}

    def load_transferable_state(
        self, state: dict[str, dict[str, Tensor]], *, strict: bool = True,
    ) -> None:
        """Warm-start from a prior phase's :meth:`transferable_state` (E4).

        Loads the ``"encoder"`` component into the student encoder with
        ``strict=True`` — a missing/unexpected encoder key is a hard error
        (the early warning that the encoder topology drifted between phases,
        per SCAFFOLD-08). After the student is warm-started the EMA teacher is
        re-synced to it so the V-JEPA target starts == the student (the
        construction-time deepcopy held the cold init). Components this module
        does not own (e.g. ``"pma"`` from a P3/P4 snapshot) are ignored.
        """
        if "encoder" not in state:
            raise KeyError(
                "transferable state has no 'encoder' component; cannot "
                f"warm-start the SSL encoder. Got keys: {sorted(state)}."
            )
        # strict=True raises RuntimeError on any missing/unexpected key.
        self.student.encoder.load_state_dict(state["encoder"], strict=strict)
        # Re-sync the EMA teacher to the freshly-loaded student (the __init__
        # deepcopy captured the cold init). Teacher params stay frozen.
        self.teacher.model.load_state_dict(self.student.state_dict(), strict=True)
        for p in self.teacher.model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Batch ingest
    # ------------------------------------------------------------------

    def _extract_student_kwargs(
        self, batch_data: dict[str, Tensor],
    ) -> dict[str, tp.Any]:
        """Pull student-forward kwargs from the batch dict.

        ``shaft_mask`` is intentionally NOT threaded — B36 (WS-H5) drops
        the shaft drop from the default SSL path; the JEPA token /
        parcel-time mask IS the SSL signal now, sampled per-step in
        :meth:`_sample_phase_mask` rather than read from the batch.

        ``freq_patch_valid`` (B36 C5, SWEC corpus-valid freq prefix) is also
        NOT threaded yet, by design. The capability is fully built downstream
        — the M2 sampler (:func:`sample_token_band_mask`), the front-end loss
        (:func:`p1_frontend_m2_loss`) and the encoder forward all accept it —
        but the live wiring (which batch key carries it, the per-clip vs
        shared-prefix shape in a mixed-corpus batch, and the valid-bin RANGE
        itself) is a Ben-gated WS-H decision tied to the unbuilt SWEC/AJILE12
        loaders (all ``NotImplementedError`` today). BT (the capstone corpus)
        is all-valid ⇒ ``None`` ⇒ no-op, so leaving it ``None`` here is exact
        for every currently-runnable path. WS-H threads it through this method
        + :meth:`_sample_phase_mask` + :meth:`_step` when SWEC P1 lands.
        See ``docs/neuroprobe/v14_blockers.md`` (C5 / freq_patch_valid row).
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

        # Per-clip conditioning is looked up by the encoder itself; the
        # encoder forward tolerates ``None``. Trailing NeuralSet-collated
        # singleton axes get stripped to match the encoder's handling.
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

    def _sample_phase_mask(
        self, *, electrode_tokens: Tensor, support: Tensor,
    ) -> dict[str, Tensor]:
        """Sample the active phase's JEPA mask (B3/B4).

        Returns the encoder-forward mask kwarg for the student
        (``token_mask`` in P1, ``parcel_time_mask`` in P2). The mask is drawn
        from a per-step :class:`torch.Generator` seeded by
        ``mask_seed + global_step`` so a given (seed, step) reproduces it
        bit-for-bit while successive steps differ. Falls back to
        ``global_step = 0`` when no trainer is attached (unit tests call
        ``_step`` directly).
        """
        device = electrode_tokens.device
        C, F_p, T_p = self.student.encoder.patch_grid_shape(electrode_tokens)
        B = electrode_tokens.shape[0]
        try:
            step = int(self.global_step)
        except (RuntimeError, AttributeError):
            step = 0
        gen = torch.Generator(device=device)
        gen.manual_seed(self._mask_seed + step)
        if self._phase == "p1":
            # freq_patch_valid omitted (= None = all-valid) — see
            # _extract_student_kwargs: SWEC corpus-valid confinement is the
            # Ben-gated WS-H wiring; BT (capstone) is all-valid ⇒ no-op.
            token_mask = sample_m2_mask(
                (B, C, F_p, T_p),
                mask_type=self._m2_mask_type,
                held_out_ratio=self._m2_mask_ratio,
                generator=gen,
                time_band_floor=self._m2_time_band_floor,
                freq_band_floor=self._m2_freq_band_floor,
                time_freq_split=self._m2_time_freq_split,
                device=device,
            )
            return {"token_mask": token_mask}
        parcel_time_mask, _drop = sample_m4_mask(
            support,
            n_time_patches=T_p,
            mask_type=self._m4_mask_type,
            mask_ratio=self._m4_mask_ratio,
            n_min_visible=self._m4_n_min_visible,
            generator=gen,
        )
        return {"parcel_time_mask": parcel_time_mask}

    def _step(self, batch_data: dict[str, Tensor]) -> MaskedJepaBreakdown:
        """One masked-JEPA forward + loss pass (B36 WS-B, B5/B6/B7/B8).

        P1 (paradigm A): the front-end masked cells are zeroed UPSTREAM of
        the token blocks, so the blocks self-predict masked M2; loss =
        L1(student M2[masked], sg teacher M2[masked]) — pool / inter-parcel
        encoder / predictor are downstream of M2 and get no gradient.

        P2 (paradigm B): the visible-only student encoder produces M4; the
        :class:`JepaPredictor` predicts the masked parcel-time M4 cells from
        the visible parcel tokens; loss = L1(prediction, sg teacher M4[masked]).

        Exactly one term is active per phase (B9). The teacher always
        encodes the FULL input (B7 — :func:`assert_teacher_full_input`).
        """
        student_kwargs = self._extract_student_kwargs(batch_data)
        mask_kwargs = self._sample_phase_mask(
            electrode_tokens=student_kwargs["electrode_tokens"],
            support=student_kwargs["support"],
        )

        # P1 (paradigm A) reads ONLY M2, so skip the downstream pool /
        # inter-parcel encoder on both the student and teacher forwards
        # (M2 is taken pre-pool and carries no downstream P1 gradient). P2
        # (paradigm B) needs the full encoder output M4.
        m2_only = self._phase == "p1"

        # ── Student forward (masked / visible-only) ──
        student_taps = self.student(**student_kwargs, **mask_kwargs, m2_only=m2_only)

        # ── Teacher forward (FULL input — no mask, no shaft). B26 / B7 ──
        teacher_taps = self._teacher_forward(dict(student_kwargs), m2_only=m2_only)

        if self._phase == "p1":
            # freq_patch_valid omitted (all-valid) — WS-H threads the SWEC
            # corpus-valid prefix here alongside the sampler/encoder (C5).
            return p1_frontend_m2_loss(
                student_m2=student_taps["M2"],
                teacher_m2=teacher_taps["M2"],
                token_mask=mask_kwargs["token_mask"],
                loss_form=self._loss_form,
            )

        # ── P2: B8 three-way latent_valid → visible context + masked targets ──
        visible, target_mask, _teacher_valid = compute_latent_valid_3way(
            support=student_kwargs["support"],
            valid_mask=student_kwargs.get("valid_mask"),
            m_sub_slots=self._m_sub_slots,
            parcel_time_mask=mask_kwargs["parcel_time_mask"],
        )
        return p2_parcel_m4_loss(
            predictor=self.predictor,
            student_m4=student_taps["M4"],
            teacher_m4=teacher_taps["M4"],
            visible=visible,
            target_mask=target_mask,
            loss_form=self._loss_form,
        )

    def _teacher_forward(
        self, teacher_kwargs: dict[str, tp.Any], *, m2_only: bool = False,
    ) -> dict[str, Tensor]:
        """EMA-teacher FULL-input forward with the B7 tripwire (B26 /
        V-JEPA 2 §2.1).

        The teacher builds the V-JEPA target from the full UNMASKED input,
        so ``teacher_kwargs`` (a copy of the student kwargs) must carry NO
        JEPA mask — the per-step mask lives in a separate ``mask_kwargs``
        spread into the student ONLY. :func:`assert_teacher_full_input`
        inspects exactly the kwargs the teacher will receive: if a future
        refactor ever threads a (partial) ``token_mask`` /
        ``parcel_time_mask`` into this pass, the guard sees ``False``
        visibility entries (``~mask``) and raises instead of silently
        leaking the mask into the target. On the live path both keys are
        absent ⇒ ``None`` ⇒ full input ⇒ the guard passes.
        """
        _t_tok = teacher_kwargs.get("token_mask")
        _t_par = teacher_kwargs.get("parcel_time_mask")
        assert_teacher_full_input(
            patch_mask=None if _t_tok is None else ~_t_tok,
            shaft_mask=None if _t_par is None else ~_t_par,
        )
        with torch.no_grad():
            teacher_taps = self.teacher.model(**teacher_kwargs, m2_only=m2_only)
        if not isinstance(teacher_taps, dict):
            raise RuntimeError(
                "EMA teacher returned a single tensor; expected the M2/M3/M4 "
                "tap dict (return_taps=True)."
            )
        return teacher_taps

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def _log_breakdown(
        self,
        breakdown: MaskedJepaBreakdown,
        *,
        step_name: str,
    ) -> None:
        # B9: exactly one active term per phase — log the scalar + the
        # masked-cell count (n_masked == 0 ⇒ total is an exact 0).
        self.log(f"{step_name}_loss", breakdown.total, on_epoch=True, prog_bar=True)
        self.log(
            f"{step_name}_n_masked", float(breakdown.n_masked), on_epoch=True,
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
        teacher_m4: Tensor,
        latent_valid: Tensor,
        step_name: str,
    ) -> None:
        """MON-TEACHER-FEATURE-RANK (5/28 P0).

        Computes RankMe on the EMA teacher's M4 tap (already post
        ``encoder_ln`` — the canonical terminal-LN target, B6), masked by
        ``latent_valid`` so SWEC / front-end-only positions don't dilute
        the rank estimate. Sub-samples to ``_RANKME_N_MAX`` rows to keep
        the SVD cheap.

        Wired into ``training_step`` (from global step 0, at the
        ``log_every_n_steps`` cadence — I1, B36 WS-I) as well as every
        ``validation_step`` / ``test_step``, so a collapse is caught at
        the start of pretraining rather than only at the first val epoch.
        """
        if teacher_m4.dim() != 4:
            return  # not a (B, L, T, d) tap — skip silently
        B, L, T, d = teacher_m4.shape
        if L != latent_valid.shape[1] or B != latent_valid.shape[0]:
            return  # shape mismatch — skip rather than crash a probe

        # Expand (B, L) → (B, L, T), flatten, gather only valid rows.
        expanded = latent_valid.unsqueeze(-1).expand(B, L, T)
        flat = teacher_m4.reshape(B * L * T, d)
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
        student_m4: Tensor,
        teacher_m4: Tensor,
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
            student_post_frame=student_m4,
            teacher_post_frame=teacher_m4,
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

    def training_step(self, batch, batch_idx: int) -> Tensor:
        breakdown = self._step(batch.data)
        self._log_breakdown(breakdown, step_name="train")
        # 2026-05-30 speedup audit (Tier-2): cadence-gate the per-step
        # monitors on the train loop. ``_monitor_from_step`` re-runs the
        # (no_grad) teacher + student encoder forwards every step — ~2
        # extra forward-equivalents on top of ``_step``'s student+teacher,
        # i.e. ~4 forwards/train-step where only 2 carry gradient. The
        # monitors are diagnostic-log-only (never enter loss/grads), so
        # firing them every ``log_every_n_steps`` steps instead of every
        # step removes the redundant compute with ZERO effect on the B31
        # loss path (``_step`` above is untouched). The
        # ``_monitor_from_step`` docstring explicitly blesses this gating.
        # val/test monitor every step (their cadence is already sparse).
        if self._train_monitor_due(batch_idx):
            self._monitor_from_step(batch.data, step_name="train")
        return breakdown.total

    def _train_monitor_due(self, batch_idx: int) -> bool:
        """Train-step monitor cadence (2026-05-30 speedup audit).

        True on step 0 and every ``trainer.log_every_n_steps`` steps
        thereafter, so the forward-heavy diagnostic monitors run at the
        logging cadence rather than every step. Falls back to every step
        (cadence 1) when no trainer is attached — unit tests that call
        ``training_step`` directly keep their prior every-step behavior,
        and ``fast_dev_run`` (batch_idx 0) still exercises the monitors
        once.
        """
        try:
            cadence = int(self.trainer.log_every_n_steps)
        except (RuntimeError, AttributeError):
            cadence = 1
        return batch_idx % max(cadence, 1) == 0

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
          teacher's post-``encoder_ln`` M4 tap (the canonical terminal-LN
          target, B6 — there is no separate ``ln_frame`` head any more),
          masked by ``latent_valid``. Fired from TRAIN step 0 (I1) plus
          every val/test step.
        """
        student_kwargs = self._extract_student_kwargs(batch_data)

        latent_valid = _latent_valid_from_batch(
            support=student_kwargs["support"],
            valid_mask=student_kwargs.get("valid_mask"),
            m_sub_slots=self._m_sub_slots,
        )
        self._run_parcel_coverage_monitor(
            latent_valid=latent_valid, step_name=step_name,
        )

        needs_orphan = "shaft_mask" in batch_data
        # I1 (B36 WS-I): RankMe fires from TRAIN step 0, not just val/test, so a
        # teacher-feature collapse is caught the moment pretraining starts rather
        # than at the first validation epoch. The train caller already gates the
        # whole monitor pass to ``log_every_n_steps`` cadence (batch_idx 0
        # included), so this only adds one capped SVD per logging step.
        needs_rankme = True
        if not (needs_orphan or needs_rankme):
            return

        # The M4 tap is already post-``encoder_ln`` (B6 canonical terminal
        # LN) — the monitors read it directly; there is no separate
        # ``ln_frame`` head any more. The monitor forwards are FULL-input
        # (no JEPA mask) so they measure the unmasked feature geometry.
        with torch.no_grad():
            teacher_taps = self.teacher.model(**student_kwargs)
            m4_t = teacher_taps["M4"]

            if needs_orphan:
                student_taps = self.student(**student_kwargs)
                self._run_mask_orphan_monitor(
                    batch_data=batch_data,
                    student_m4=student_taps["M4"],
                    teacher_m4=m4_t,
                    step_name=step_name,
                )
            if needs_rankme:
                self._run_teacher_rank_monitor(
                    teacher_m4=m4_t,
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
        # Student encoder + predictor both carry grads (the teacher is
        # frozen). P1 leaves the predictor grad-free (paradigm A); P2
        # exercises it — iterating both keeps the L2 correct either way.
        for p in self._trainable_parameters():
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
        # internally; coeff is fixed at τ=0.99925 under the B26 lock.
        self.teacher.update_from(self.student)

    # B36 WS-E (E2): discriminative-LR factor for the front-end param group
    # in P2. The front-end was pretrained in P1, so it rides at base_lr/10
    # while the freshly-trained pool / inter-parcel encoder / predictor get
    # the full base LR (B36 §7 "front-end @ LR/10; discriminative unfreeze").
    _FRONTEND_LR_SCALE: tp.ClassVar[float] = 0.1

    def _phase_param_groups(self) -> list[tp.Any]:
        """Phase-conditional optimizer parameters (B36 WS-E, E1/E2).

        * **P1** — front-end params ONLY (E1). The masked-JEPA loss flows
          only through the M2 tap (``m2_only``), so the pool / inter-parcel
          encoder / predictor are already grad-free; excluding them from the
          optimizer makes "front-end only" the explicit, update-level
          contract (no stray weight-decay drift on a grad-free param).
        * **P2** — two param groups (E2): the front-end at ``base_lr / 10``
          and the parcel side + the student predictor at the full base LR.
          PyTorch reads each group's ``lr`` as its ``base_lr``, so a
          downstream scheduler scales both proportionally and the 10:1 ratio
          holds for the whole run.
        """
        frontend, parcel = self.student.encoder.partition_parameters_for_staging()
        if self._phase == "p1":
            return frontend
        predictor_params = list(self.predictor.parameters())
        base_lr = self._base_lr()
        return [
            {"params": frontend, "lr": base_lr * self._FRONTEND_LR_SCALE},
            {"params": parcel + predictor_params},
        ]

    def _base_lr(self) -> float:
        """Base LR from the optim config (E2 needs it to scale the front-end
        group). ``LightningOptimizer.optimizer`` is a ``BaseTorchOptimizer``
        with a required ``lr`` field; raise loudly if it is somehow absent so
        a misconfigured P2 never silently runs without discriminative LR."""
        inner = getattr(self.optim_config, "optimizer", None)
        lr = getattr(inner, "lr", None)
        if lr is None:
            raise RuntimeError(
                "P2 discriminative LR (B36 WS-E E2) needs "
                "optim_config.optimizer.lr; got None. Pass a LightningOptimizer "
                "whose inner optimizer config exposes a base lr."
            )
        return float(lr)

    def _estimated_total_steps(self) -> int | None:
        """``trainer.estimated_stepping_batches`` when attached, else ``None``.

        Mirrors the ``_sample_phase_mask`` / ``_train_monitor_due`` fallback:
        unit tests call ``configure_optimizers`` with no trainer attached, so
        the scheduler's ``total_steps`` is simply omitted there.
        """
        try:
            return int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            return None

    def configure_optimizers(self):  # type: ignore[override]
        # B36 WS-E: phase-conditional param groups (P1 front-end only; P2
        # two-group discriminative LR). The teacher is EMA-frozen and never
        # optimised.
        params = self._phase_param_groups()
        total_steps = self._estimated_total_steps()
        if total_steps is None:
            return self.optim_config.build(params)
        try:
            return self.optim_config.build(params, total_steps=total_steps)
        except TypeError:
            return self.optim_config.build(params)


__all__ = [
    "V14JointBrainModule",
    "_V14StudentBundle",  # exported for tests / dispatch override
]
