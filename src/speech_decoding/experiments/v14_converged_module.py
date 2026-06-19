"""LightningModule wrapping the converged-arch SSL model (Philosophy B).

Wires `speech_decoding.models.v14_converged.V14ConvergedSSL` into the
NeuralTrain/Lightning training loop WITHOUT touching the live B37/2STFT
`V14JointBrainModule`. The converged model is self-contained — it owns its
frontend-only EMA teacher, the two paradigm-B predictors, and the M2/M4 losses
with the locked see-vs-predict scopes (`project_v14_m2_m4_predictor_scopes_2026_06_18`).
So this module is intentionally thin: batch-ingest → per-step mask draw →
`model.forward` → log; EMA in `on_before_zero_grad`; encoder export for the
frozen-readout handoff.

Run numerics are NOT pre-committed here: `ema_tau` and `optim_config` are
required args supplied by the caller (dispatch). The mask configs default to the
LOCKED structural `M2MaskConfig`/`M4MaskConfig` (the FE-spec §8 geometry), not
run hyperparameters.

Batch contract (NeuralSet `batch.data` dict). Time-axis frame counts below are
the **1 s** Phase-4 eval geometry; SSL pretrain runs 5 s clips (slow 21 / beta 81
/ HG 161 frames), so the trailing time dim scales with `clip_len` — the slow
split and the stem patch-grid are clip-length-parameterized (`bands_for_clip_len`),
the freq axis is invariant:
  electrode_tokens_slow : (B, C, 12, T)     slow band, Re/Im as [Re(6) ++ Im(6)] on
                                            the freq axis; `_converged_inputs` splits
                                            it into the (B, C, 2, 6, T) channel form
                                            the slow stem (in_channels=2) consumes
  electrode_tokens_beta : (B, C, 6, T)      beta band (|STFT| mag, 1ch)
  electrode_tokens_hg   : (B, C, 9, T)      HG band (|STFT| mag, 1ch)
  support               : (B, C, K)         DK one-hot → parcel_per_electrode = argmax
  valid_mask            : (B, C) bool        optional → electrode_mask (real electrodes)
"""

from __future__ import annotations

import os
import typing as tp

import torch
from lightning import pytorch as pl
from torch import Tensor

from neuraltrain.optimizers import BaseOptimizer

from speech_decoding.experiments.optim_param_groups import maybe_split_no_decay
from speech_decoding.models.v14_converged import (
    M2MaskConfig,
    M4MaskConfig,
    V14ConvergedSSL,
    sample_ssl_masks,
)


class V14ConvergedBrainModule(pl.LightningModule):
    """Thin Lightning shell around a self-contained `V14ConvergedSSL`.

    The converged model owns the teacher/predictors/losses; this shell owns the
    training-loop plumbing (batch ingest, per-step mask sampling, logging, the
    EMA tick, optimizer construction, and the encoder handoff).
    """

    def __init__(
        self,
        *,
        model: V14ConvergedSSL,
        optim_config: BaseOptimizer,
        ema_tau: float,
        m2_cfg: M2MaskConfig = M2MaskConfig(),
        m4_cfg: M4MaskConfig = M4MaskConfig(),
        mask_seed: int = 0,
        wd_exclude_norms: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim_config = optim_config
        self.ema_tau = float(ema_tau)
        self.m2_cfg = m2_cfg
        self.m4_cfg = m4_cfg
        self._wd_exclude_norms = wd_exclude_norms
        # Mask RNG: own CPU generator so the per-step mask draws are reproducible
        # and independent of the global RNG (sample_ssl_masks loops in Python with
        # a torch.Generator). DDP rank-offset is a run-prep refinement, noted: a
        # shared seed across ranks only lowers mask diversity, it is not a
        # correctness bug (the model never assumes distinct masks per rank).
        self._mask_seed = int(mask_seed)
        self._mask_gen = torch.Generator()
        self._mask_gen.manual_seed(self._mask_seed)

        # ── torch.compile forward override (ported verbatim from the live
        # V14JointBrainModule, 2026-06-18) ──
        # The converged forward is ALSO ragged + DDP + find_unused: the M2/M4
        # heads early-return a scalar zero on empty-target steps, so a predictor
        # can be UNUSED on one rank and USED on another in the same step. Run
        # eager, that data-dependent param-usage divergence + the AccumulateGrad
        # cross-stream stash hangs DDP (observed: job 48523709 trained ~8 min then
        # NCCL-watchdog-stuck). The joint module survives the SAME raggedness only
        # because it compiles with DDPOptimizer DISABLED, which compiles the
        # forward into ONE graph (no bucket-split reorder) and routes the backward
        # through inductor (no AccumulateGrad cross-stream stash). So mirror it.
        #
        # Reads the env vars dispatch_v14 sets (V14_COMPILE / V14_COMPILE_MODE /
        # V14_COMPILE_DYNAMIC / V14_COMPILE_DDP_OPTIMIZER) — env, NOT pydantic
        # fields, so a compiled run shares the eager run's exca uid + cache. The
        # OptimizedModule is stored in a PLAIN DICT (not an attribute) so the
        # LightningModule never re-registers it as a submodule: params stay
        # registered once via self.model (no `_orig_mod.` key-prefix on
        # checkpoints, no double-registration of optimizer/EMA params). Unset env
        # (unit tests / direct construction) → eager, byte-identical, zero blast.
        self._compiled_fwd: dict[str, tp.Callable[..., tp.Any]] = {}
        _compile_flag = os.environ.get("V14_COMPILE", "").strip().lower()
        if _compile_flag not in ("", "0", "false", "no", "off"):
            _mode = os.environ.get("V14_COMPILE_MODE") or "default"
            _dyn_flag = os.environ.get("V14_COMPILE_DYNAMIC", "").strip().lower()
            _dynamic: bool | None = (
                True if _dyn_flag not in ("", "0", "false", "no", "off") else None
            )
            # DDPOptimizer × dynamic-shapes fix (2026-06-11): the bucket-split
            # optimizer hands a symbolic-shape SymInt back as a bare python int and
            # crashes inductor under dynamic=True; disabling it compiles a single
            # graph (cost: lost allreduce/compute overlap, negligible for this
            # ~16M-param model on single-node 4-GPU DDP). Default OFF = disabled,
            # because the production sweep IS compile+DDP+dynamic.
            import torch._dynamo as _dynamo_mod

            _ddp_opt = os.environ.get(
                "V14_COMPILE_DDP_OPTIMIZER", "").strip().lower()
            _dynamo_mod.config.optimize_ddp = _ddp_opt in ("1", "true", "yes", "on")
            self._compiled_fwd["model"] = torch.compile(
                self.model, mode=_mode, dynamic=_dynamic,
            )

    def _call_model(self, *args: tp.Any, **kwargs: tp.Any) -> dict[str, Tensor]:
        """Run the model forward through the compiled override when present, else
        eager. Falls back to ``self.model`` when V14_COMPILE was unset (tests /
        1-GPU), so the eager path is byte-identical to pre-compile."""
        return self._compiled_fwd.get("model", self.model)(*args, **kwargs)

    # ------------------------------------------------------------------ ingest
    def _converged_inputs(
        self, data: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Map a batch dict to the model's `(slow, beta, hg, parcel_per_electrode,
        electrode_mask)`. `parcel_per_electrode = support.argmax(-1)` (DK support
        is one-hot, so argmax is the exact hard parcel id); `electrode_mask` =
        `valid_mask` (all-real when absent).

        The cartesian slow band is cached as the two real components CONCATENATED on
        the freq axis (`[Re(F) ++ Im(F)]` → `(B, C, 2F, T)`; see
        `view.py::_single_stft_raw_view` `cartesian=True`). The frontend slow stem is
        built for `in_channels=2` and consumes a SEPARATE Re/Im channel axis
        `(B, C, 2, F, T)`, so split the freq-concatenated pair back into that channel
        axis (row-major → channel 0 = Re(F), channel 1 = Im(F)). Beta/HG are |STFT|
        magnitude (1 channel) and pass through 4-D unchanged."""
        slow = data["electrode_tokens_slow"]
        if slow.ndim != 4 or slow.shape[2] % 2 != 0:
            raise ValueError(
                "cartesian slow band must be (B, C, 2F, T) with an even freq axis "
                f"([Re ++ Im]); got shape {tuple(slow.shape)}"
            )
        Bs, Cs, F2, Ts = slow.shape
        slow = slow.reshape(Bs, Cs, 2, F2 // 2, Ts)
        beta = data["electrode_tokens_beta"]
        hg = data["electrode_tokens_hg"]
        support = data["support"]
        B, C = support.shape[0], support.shape[1]
        valid = data.get("valid_mask")
        if valid is None:
            electrode_mask = torch.ones(B, C, dtype=torch.bool, device=support.device)
        else:
            electrode_mask = valid.to(torch.bool)
        parcel_per_electrode = support.argmax(dim=-1)
        return slow, beta, hg, parcel_per_electrode, electrode_mask

    def _step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """The pure loss path (testable without a trainer): ingest → sample masks
        → `model.forward`. Returns `{loss, l_m2, l_m4}` plus the per-band
        diagnostics `l_m2_{beta,hg}` / `l_m4_{slow,beta,hg}`."""
        slow, beta, hg, ppe, emask = self._converged_inputs(data)
        # Mask sampling runs on CPU (a CPU generator can't drive CUDA randperm);
        # move the masks back to the feature device for the forward.
        masks = sample_ssl_masks(
            ppe.cpu(), emask.cpu(), self._mask_gen,
            m2_cfg=self.m2_cfg, m4_cfg=self.m4_cfg, bands=self.model.bands,
        )
        masks = {k: v.to(slow.device) for k, v in masks.items()}
        return self._call_model(slow, beta, hg, ppe, emask, **masks)

    # ------------------------------------------------------------------- loops
    def training_step(self, batch: tp.Any, batch_idx: int) -> Tensor:  # noqa: ARG002
        out = self._step(batch.data)
        self._log_losses(out, "train", on_step=True, on_epoch=False)
        return out["loss"]

    def validation_step(self, batch: tp.Any, batch_idx: int) -> None:  # noqa: ARG002
        out = self._step(batch.data)
        self._log_losses(out, "val", on_step=False, on_epoch=True)

    def _log_losses(
        self, out: dict[str, Tensor], name: str, *, on_step: bool, on_epoch: bool,
    ) -> None:
        """Log every scalar the forward emits: loss + the two head scalars, the
        per-band/aggregate monitor diagnostics (loss / explained_var / target_var
        per band {slow,beta,hg}), and the per-band stem-output norms. Non-finite
        scalars are skipped — ev/tv are NaN when a band has < 2 scored cells this
        step, and an undefined-variance step must not poison the epoch mean."""
        for key, val in out.items():
            if val.dim() != 0 or not bool(torch.isfinite(val)):
                continue
            self.log(
                f"{name}_{key}", val,
                on_step=on_step, on_epoch=on_epoch,
                prog_bar=(key == "loss"),
            )

    def on_before_zero_grad(self, optimizer: tp.Any) -> None:  # noqa: ARG002
        """EMA tick once per optimiser step (after the step, before zero_grad).
        Placed here, not `on_train_batch_end`, so `accumulate_grad_batches=K`
        applies ONE update/step (τ, not τ^K) — the #46 lesson from the live
        module."""
        self.model.update_teacher(self.ema_tau)

    # -------------------------------------------------------------- phase handoff
    def transferable_state(self) -> dict[str, dict[str, Tensor]]:
        """Carry the frozen ENCODER (frontend + latent) to the readout phase.
        Predictors are head-specific (re-trained) and the EMA teacher re-syncs
        from the loaded student — neither transfers (mirrors the live module)."""
        return {
            "student_frontend": self.model.student_frontend.state_dict(),
            "latent": self.model.latent.state_dict(),
        }

    def load_transferable_state(
        self, state: dict[str, dict[str, Tensor]], *, strict: bool = True,
    ) -> None:
        """Warm-start the encoder from a prior phase's `transferable_state`, then
        re-sync the EMA teacher to the freshly-loaded student frontend (the
        construction-time deepcopy held the cold init). Teacher stays frozen."""
        for comp in ("student_frontend", "latent"):
            if comp not in state:
                raise KeyError(
                    f"transferable state missing '{comp}'; cannot warm-start the "
                    f"converged encoder. Got keys: {sorted(state)}."
                )
        self.model.student_frontend.load_state_dict(
            state["student_frontend"], strict=strict,
        )
        self.model.latent.load_state_dict(state["latent"], strict=strict)
        self.model.teacher_frontend.load_state_dict(
            self.model.student_frontend.state_dict(), strict=True,
        )
        for p in self.model.teacher_frontend.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------------------- optimizer
    def _estimated_total_steps(self) -> int | None:
        try:
            return int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            return None

    def configure_optimizers(self):  # type: ignore[override]
        # Optimise only the trainable params (the EMA teacher is requires_grad
        # False, so it is excluded by construction).
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        params: tp.Any = [{"params": trainable}]
        params = maybe_split_no_decay(
            params, modules=(self.model,),
            optim_config=self.optim_config, exclude=self._wd_exclude_norms,
        )
        total_steps = self._estimated_total_steps()
        if total_steps is None:
            return self.optim_config.build(params)
        try:
            return self.optim_config.build(params, total_steps=total_steps)
        except TypeError:
            return self.optim_config.build(params)


__all__ = ["V14ConvergedBrainModule"]
