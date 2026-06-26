"""Lightning shell around the self-contained :class:`V14ConvergedV2` SSL model.

The v2 model OWNS the EMA teacher, the dual-depth shared-denominator loss, the
static drop-not-pad forward, and mask sampling — so this shell is THIN: ingest
the 2-band batch → sample masks (CPU) → ``model.forward`` → log → return loss,
with the EMA tick in ``on_before_zero_grad`` (one update per optimiser step, so
``accumulate_grad_batches=K`` gives τ not τ^K — the #46 lesson). It does NOT need
the 3STFT ``V14ConvergedBrainModule``'s compile / DDP-find-unused / static-shape
threading: v2's forward is static-shape BY CONSTRUCTION (drop-not-pad) and every
parameter is used every step (M2+M4 predictors always fire, teacher always runs),
so a stock static DDP graph is correct.

Batch contract (the v2 cache, session-homogeneous via the same-session sampler):
  electrode_tokens_lfs : (B, C, 28, T_lfs)   |STFT| magnitude, robust-z'd
  electrode_tokens_hga : (B, C,  7, T_hga)   |STFT| magnitude
  support              : (B, C, K) DK one-hot → parcel_of_electrode = argmax → (C,)
  valid_mask           : (B, C) bool, optional → MUST be all-real (no padding path)
"""

from __future__ import annotations

import typing as tp

import torch
from lightning import pytorch as pl
from torch import Tensor

from neuraltrain.optimizers import BaseOptimizer

from speech_decoding.experiments.optim_param_groups import maybe_split_no_decay
from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    active_parcels,
    bands_for_clip_len,
)


class V14ConvergedV2BrainModule(pl.LightningModule):
    """Thin Lightning shell — ingest, sample masks, forward, log, EMA tick."""

    def __init__(
        self,
        *,
        model: V14ConvergedV2,
        optim_config: BaseOptimizer,
        clip_len_s: float,
        mask_seed: int = 0,
        wd_exclude_norms: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim_config = optim_config
        self.clip_len_s = float(clip_len_s)
        self._wd_exclude_norms = wd_exclude_norms
        # Own CPU generator: the per-clip mask draws (CUDA randperm needs a CPU
        # generator) are reproducible + independent of the global RNG. A shared
        # seed across DDP ranks only lowers mask diversity, not correctness — the
        # model never assumes distinct masks per rank.
        self._mask_seed = int(mask_seed)
        self._mask_gen = torch.Generator()
        self._mask_gen.manual_seed(self._mask_seed)

    # ----------------------------------------------------------- batch ingest
    def _v2_inputs(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Map a batch dict to ``(lfs, hga, parcel_of_electrode (C,))``.

        ``parcel_of_electrode = support.argmax(-1)`` (DK support is one-hot, so
        argmax is the exact hard parcel id). The v2 model is session-homogeneous —
        one ``(C,)`` parcel vector for the whole batch — so the per-clip support
        must be CONSTANT across clips; fail loud otherwise. v2 has no
        electrode-padding path, so a ``valid_mask`` with padded electrodes is also
        a fail-loud contract violation (the same-session sampler is zero-padding)."""
        lfs = data["electrode_tokens_lfs"]
        hga = data["electrode_tokens_hga"]
        support = data["support"]
        ppe = support.argmax(dim=-1)                              # (B, C)
        if not torch.equal(ppe, ppe[:1].expand_as(ppe)):
            raise ValueError(
                "v2 requires a session-homogeneous batch: parcel_of_electrode must "
                "be constant across clips (use the same-session sampler)"
            )
        poe = ppe[0]                                             # (C,)
        valid = data.get("valid_mask")
        if valid is not None and not bool(valid.all()):
            raise ValueError(
                "v2 has no electrode-padding path: valid_mask has padded electrodes "
                "(use the zero-padding-free same-session sampler)"
            )
        return lfs, hga, poe

    # ------------------------------------------------------------- loss path
    def _step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Pure loss path (testable without a trainer): ingest → sample masks →
        ``model.forward``. Returns the converged-v2 loss dict (loss + per-term
        diagnostics)."""
        lfs, hga, poe = self._v2_inputs(data)
        bands = bands_for_clip_len(self.clip_len_s)
        # Masks sampled on CPU (CUDA randperm needs a CPU generator), membership
        # derived from the SAME deterministic active_parcels ⇒ matches the model's
        # internal session layout (unique-sorted labels, identical P + ordering).
        _, membership = active_parcels(poe.cpu())
        B = lfs.shape[0]
        m2, tube = self.model.sample_masks(B, membership, bands, self._mask_gen)
        m2, tube = m2.to(lfs.device), tube.to(lfs.device)
        return self.model(lfs, hga, poe, m2, tube, clip_len_s=self.clip_len_s)

    def training_step(self, batch: tp.Any, batch_idx: int) -> Tensor:  # noqa: ARG002
        out = self._step(batch.data)
        self._log_losses(out, "train", on_step=True, on_epoch=False)
        return out["loss"]

    def validation_step(self, batch: tp.Any, batch_idx: int) -> None:  # noqa: ARG002
        out = self._step(batch.data)
        self._log_losses(out, "val", on_step=False, on_epoch=True)

    def _log_losses(
        self, out: dict[str, Tensor], name: str, *, on_step: bool, on_epoch: bool
    ) -> None:
        """Log every finite scalar the loss emits (loss + per-term diagnostics).
        Non-finite (e.g. an empty-tubed batch's tubed diagnostic) are skipped so a
        single undefined step can't poison the epoch mean."""
        for key, val in out.items():
            if val.dim() == 0 and bool(torch.isfinite(val)):
                self.log(f"{name}_{key}", val, on_step=on_step, on_epoch=on_epoch)

    # ------------------------------------------------------------- lifecycle
    def on_before_zero_grad(self, optimizer: tp.Any) -> None:  # noqa: ARG002
        """EMA tick once per optimiser step (after step, before zero_grad)."""
        self.model.ema_step()

    def configure_optimizers(self):  # type: ignore[override]
        # Only the trainable params (the EMA teacher is requires_grad False, so it
        # is excluded by construction).
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        params: tp.Any = [{"params": trainable}]
        params = maybe_split_no_decay(
            params, modules=(self.model,),
            optim_config=self.optim_config, exclude=self._wd_exclude_norms,
        )
        return self.optim_config.build(params)


__all__ = ["V14ConvergedV2BrainModule"]
