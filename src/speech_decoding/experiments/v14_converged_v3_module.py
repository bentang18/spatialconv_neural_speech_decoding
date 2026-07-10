"""v14_converged_v3 — Lightning training module.

Adopts v2's training MECHANICS (not its module: ``V14JointBrainModule`` is welded
to the v2 encoder interface — pool / M2·M4 taps / per_band_stem — none of which v3
has). This is a lean wrapper around :class:`V3ConvergedModel` that reuses the
model-agnostic substrate v2 relies on:

  * ``optim_param_groups.maybe_split_no_decay`` — the shape-only no-WD split
    (``ndim<=1`` exempt; every ≥2-D matrix INCLUDING the parcel identity table is
    decayed — exact V-JEPA 2 ``init_opt`` rule).
  * the neuraltrain ``LightningOptimizer`` config (``optim_config.build(params,
    total_steps=...)``) carrying AdamW β₂ / weight-decay and the ``warmup_cosine``
    schedule — the same object dispatch builds from the CLI flags.
  * ``ssl.ema.EmaTeacher`` — the teacher is EMA-advanced once per optimiser step in
    ``on_before_zero_grad`` (v2 #46: off the per-micro-batch hook so grad-accum
    can't apply it K×).

Hyperparameters are the frozen v2 set (Ben 2026-07-10): lr 6e-3, wd 0.04,
grad-clip 3.0, warmup_cosine with min_lr_ratio 1.0 (flat peak after 5k warmup),
ema-tau 0.99925, β₂ 0.95, seed 33, bs 32 × accum 4. target_ln / predictor
terminal-LN / QK-norm are ON by construction in :class:`V3ConvergedModel`.

Masking is the sole augmentation and is sampled INSIDE ``model.forward`` from a
per-step generator seeded ``f(seed, global_step)`` — resume-stable (same step ⇒
same mask) yet fresh each step. The SSL-health monitors attach as a separate
``pl.Callback`` (``SSLHealthMonitorV3``), reading the attributes this module
exposes (``model``, ``_last_batch_size``, ``_last_taps``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from lightning import pytorch as pl
from torch import Tensor

from speech_decoding.experiments.optim_param_groups import maybe_split_no_decay
from speech_decoding.models.v14_converged_v3.geometry import L1Geometry
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel

# Base offset so the per-step mask seed decorrelates across seeds/steps without
# ever aliasing step N of seed A onto step M of seed B for small N.
_MASK_SEED_STRIDE = 1_000_003


@dataclass
class V3Batch:
    """One session's worth of clips — the contract the Phase-0 cache dataloader
    must produce (batch = ONE session; ``n_rows`` = clips in this micro-batch).

      * ``bands``     — the 3 multi-res |STFT| bands, each ``(n_rows, N, F_b, T_b)``
        in v3 concat order (SLOW, MID, HGA); ``F_b`` = (7, 6, 7).
      * ``geom``      — the per-session ``L1Geometry`` (shaft gather + depths),
        built once upstream and shared across every clip of the session.
      * ``parcel_id`` — ``(N,)`` long DKT hard tag, per contact.
    """

    bands: Sequence[Tensor]
    geom: L1Geometry
    parcel_id: Tensor


class V14ConvergedV3Module(pl.LightningModule):
    def __init__(
        self,
        *,
        model: V3ConvergedModel,
        optim_config,  # neuraltrain LightningOptimizer config
        seed: int = 33,
        monitor_every_n_steps: int = 1,
        wd_exclude_norms: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim_config = optim_config
        self.seed = int(seed)
        self.monitor_every_n_steps = max(int(monitor_every_n_steps), 1)
        self._wd_exclude_norms = wd_exclude_norms
        # Read by SSLHealthMonitorV3 (grad_noise_scale B_eff) and the tap monitors.
        self._last_batch_size: int = 0
        self._last_taps: dict[str, Tensor] | None = None

    # --------------------------------------------------------------- optimizer
    def _trainable_parameters(self) -> list[Tensor]:
        """Every param that carries gradient. The EMA teacher's deepcopy is
        ``requires_grad_(False)`` at construction, so this cleanly excludes it —
        no name matching needed."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def _estimated_total_steps(self) -> int | None:
        """``trainer.estimated_stepping_batches`` when attached, else ``None``
        (unit tests call ``configure_optimizers`` with no trainer)."""
        try:
            return int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            return None

    def configure_optimizers(self):  # type: ignore[override]
        # One group of all trainable params (v3 has no P1/P2 phases or
        # discriminative-LR split). The no-WD split then exempts biases / LN γβ
        # (ndim<=1); the parcel identity table (2-D) is DECAYED, matching upstream.
        params: list = [{"params": self._trainable_parameters()}]
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

    # ------------------------------------------------------------------- train
    def _step_generator(self, device: torch.device) -> torch.Generator:
        """Per-step mask generator seeded ``f(seed, global_step)`` — resume-stable
        (same step ⇒ same mask) and fresh each step."""
        g = torch.Generator(device=device)
        try:
            step = int(self.global_step)
        except (RuntimeError, AttributeError):
            step = 0
        g.manual_seed((self.seed * _MASK_SEED_STRIDE + step) & 0x7FFF_FFFF_FFFF_FFFF)
        return g

    def _monitor_due(self, step: int) -> bool:
        return step % self.monitor_every_n_steps == 0

    def training_step(self, batch: V3Batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        device = batch.bands[0].device
        gen = self._step_generator(device)
        # Taps (the SVD/rankme + tier-EV monitor inputs) are computed only on the
        # monitor cadence — off-cost otherwise (the <5% budget). Presence of
        # ``_last_taps`` IS the cadence signal the callback's on_train_batch_end
        # reads (its own global_step is post-step / off-by-one).
        try:
            step = int(self.global_step)
        except (RuntimeError, AttributeError):
            step = 0
        collect = self._monitor_due(step)
        out = self.model(
            batch.bands, batch.geom, batch.parcel_id,
            generator=gen, collect_taps=collect,
        )
        self._last_batch_size = int(batch.bands[0].shape[0])
        self._last_taps = out.taps if collect else None
        self.log("train_loss", out.loss, on_step=True, prog_bar=True)
        self.log("train_n_masked", float(out.n_masked), on_step=True)
        return out.loss

    def on_before_zero_grad(self, optimizer) -> None:  # noqa: ARG002
        # Runs after optimizer.step(), once per optimiser step (accum-safe) — the
        # v2 #46 placement. Advances the EMA teacher toward the just-stepped online
        # tower; the coeff schedule (fixed τ 0.99925) advances internally.
        self.model.update_teacher()
