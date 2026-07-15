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
ema-tau 0.99925, β₂ 0.95, seed 33, bs 32 × accum 4. target_ln and the predictor
terminal-LN are ON by construction in :class:`V3ConvergedModel`; QK-norm is OFF
(intentional — see attention.py rationale, not on the locked-divergences list).

Masking is the sole augmentation and is sampled INSIDE ``model.forward`` from a
per-step generator seeded ``f(seed, global_step)`` — resume-stable (same step ⇒
same mask) yet fresh each step. The SSL-health monitors attach as a separate
``pl.Callback`` (``SSLHealthMonitorV3``), reading the attributes this module
exposes (``model``, ``_last_batch_size``, ``_last_taps``).
"""

from __future__ import annotations

import torch
from lightning import pytorch as pl
from torch import Tensor

from speech_decoding.experiments.optim_param_groups import maybe_split_no_decay
from speech_decoding.models.v14_converged_v3.batch import V3Batch
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel

# Base offset so the per-step mask seed decorrelates across seeds/steps without
# ever aliasing step N of seed A onto step M of seed B for small N.
_MASK_SEED_STRIDE = 1_000_003


# V3Batch (bands / geom / parcel_id, shared per session) is the data contract the
# datamodule's ``v3_collate`` produces; re-exported here so callers that imported it
# from this module keep working after the definition moved to the models package.
__all__ = ["V3Batch", "V14ConvergedV3Module"]


class V14ConvergedV3Module(pl.LightningModule):
    def __init__(
        self,
        *,
        model: V3ConvergedModel,
        optim_config,  # neuraltrain LightningOptimizer config
        seed: int = 33,
        monitor_every_n_steps: int = 1,
        wd_exclude_norms: bool = True,
        secondary_active: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim_config = optim_config
        self.seed = int(seed)
        self.monitor_every_n_steps = max(int(monitor_every_n_steps), 1)
        self._wd_exclude_norms = wd_exclude_norms
        # Secondary Gaussian-NLL (contract §5–7) is OPT-IN: it fires only when the batch
        # carries per-session frozen state-stats, which the datamodule supplies iff a
        # stats dir was configured. ``secondary_active`` mirrors that config so the module
        # knows at construction whether the write-only Perceiver head will receive gradient.
        self._secondary_active = bool(secondary_active)
        # Secondary OFF ⇒ the Perceiver head never receives gradient (no stats ever reach
        # forward). Freeze it so DDP's reducer and the optimizer both skip it — the same
        # requires_grad=False mechanism that excludes the EMA teacher. Without this, DDP
        # with find_unused_parameters=False aborts on the unused head. deep_sup=False has
        # no Perceiver at all (objective.perceiver is None) ⇒ nothing to freeze.
        perceiver = self.model.objective.perceiver
        if perceiver is not None and not self._secondary_active:
            for p in perceiver.parameters():
                p.requires_grad_(False)
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

    # -------------------------------------------------------------- device move
    def transfer_batch_to_device(  # type: ignore[override]
        self, batch: V3Batch, device: torch.device, dataloader_idx: int
    ) -> V3Batch:
        """Move the ``V3Batch`` to ``device`` field-by-field. Lightning's default
        ``apply_to_collection`` refuses a FROZEN dataclass (can't reconstruct it in
        place), so the custom batch type moves itself: the band tensors + the
        per-session geometry (``L1Geometry.to``) + ``parcel_id``; ``session_key`` is
        metadata and stays put."""
        # non_blocking H2D: pin_memory is on (datamodule), so async copies overlap
        # transfer with compute. geom is a per-session static that Lightning may move
        # before the pinned batch tensors exist; keep it blocking (its own .to).
        # stat_mean/stat_std (per-session frozen state-stats, (P,6)) ride along like the
        # other per-session statics when present; None ⇒ secondary OFF (JEPA-only).
        sm = batch.stat_mean
        ss = batch.stat_std
        return V3Batch(
            bands=[b.to(device, non_blocking=True) for b in batch.bands],
            geom=batch.geom.to(device),
            parcel_id=batch.parcel_id.to(device, non_blocking=True),
            session_key=batch.session_key,
            stat_mean=None if sm is None else sm.to(device, non_blocking=True),
            stat_std=None if ss is None else ss.to(device, non_blocking=True),
        )

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

    def training_step(self, batch: V3Batch, batch_idx: int) -> Tensor:
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
        # Collect taps ONLY on the last accumulation micro-batch. ``global_step`` is
        # constant across an accum window, so ``_monitor_due`` alone would set
        # collect=True on every micro-batch — but the callback's on_train_batch_end
        # acts once per opt-step (guards ``(batch_idx+1) % accum == 0``) and discards
        # taps from the other accum-1 micro-batches. Materialising them there is pure
        # waste (measured +0.47s/opt-step at accum=4, monitor_every=1). Collecting only
        # on the used micro-batch keeps per-opt-step monitoring at monitor_every=1.
        try:
            accum = max(1, int(self.trainer.accumulate_grad_batches or 1))
        except (RuntimeError, AttributeError):
            accum = 1  # no trainer attached (unit tests) ⇒ every call is "last"
        is_last_microbatch = (batch_idx + 1) % accum == 0
        collect = self._monitor_due(step) and is_last_microbatch
        # stat_mean/stat_std turn ON the secondary Gaussian-NLL when present (secondary
        # opt-in); absent ⇒ JEPA-only. They flow per-session like geom/parcel_id.
        out = self.model(
            batch.bands, batch.geom, batch.parcel_id,
            generator=gen, collect_taps=collect,
            stat_mean=batch.stat_mean, stat_std=batch.stat_std,
        )
        self._last_batch_size = int(batch.bands[0].shape[0])
        self._last_taps = out.taps if collect else None
        self.log("train_loss", out.loss, on_step=True, prog_bar=True)
        self.log("train_n_masked", float(out.n_masked), on_step=True)
        # Two-loss monitors (Ben 2026-07-15): when the secondary is active, log the primary
        # JEPA L1 and the secondary Gaussian-NLL SEPARATELY so the two objectives are
        # tracked independently (λ-balance readout). Both None on the JEPA-only arm ⇒ only
        # train_loss (== jepa_loss) is logged. LOG-ONLY — nothing here kills the run.
        if out.jepa_loss is not None:
            self.log("train_jepa_loss", out.jepa_loss, on_step=True, prog_bar=True)
        if out.nll_loss is not None:
            self.log("train_nll_loss", out.nll_loss, on_step=True, prog_bar=True)
            # The weighted contribution the total actually carries (λ·NLL) — reads the
            # objective's live λ, so a change to --lambda-nll shows up here directly.
            self.log(
                "train_nll_weighted",
                float(self.model.objective.lambda_nll) * out.nll_loss.detach(),
                on_step=True,
            )
        return out.loss

    def on_before_zero_grad(self, optimizer) -> None:  # noqa: ARG002
        # Runs after optimizer.step(), once per optimiser step (accum-safe) — the
        # v2 #46 placement. Advances the EMA teacher toward the just-stepped online
        # tower; the coeff schedule (fixed τ 0.99925) advances internally.
        self.model.update_teacher()
