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

from speech_decoding.experiments.monitors.grad_ratio import loss_grad_ratio
from speech_decoding.experiments.optim_param_groups import maybe_split_no_decay
from speech_decoding.models.v14_converged_v3.batch import V3Batch
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.stem import clock_length_32hz

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
        grad_ratio_every_n_steps: int = 0,
        nll_warmup_start_step: int = 0,
        nll_warmup_steps: int = 0,
        ctx_warmup_start_step: int = 0,
        ctx_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.optim_config = optim_config
        self.seed = int(seed)
        self.monitor_every_n_steps = max(int(monitor_every_n_steps), 1)
        self._wd_exclude_norms = wd_exclude_norms
        # Live loss-balance readout (#43): ‖g_nll‖/‖g_jepa‖ on the shared online tower,
        # measured every ``grad_ratio_every_n_steps`` opt-steps (0 ⇒ OFF, the launch default).
        # SINGLE-PROCESS ONLY — the two extra autograd.grad passes re-enter the DDP reducer
        # over the shared graph (the r3 static_graph×grad-accum crash surface), so this is
        # gated to world_size==1 in ``training_step`` and is a 1-GPU diagnostic lever only.
        self.grad_ratio_every_n_steps = max(int(grad_ratio_every_n_steps), 0)
        # Secondary-NLL λ RAMP (Ben 2026-07-15): λ_nll is applied in EAGER training_step,
        # never inside the compiled objective forward — a per-step python-float scalar in
        # a compiled graph thrashes dynamo guards (mirrors v2's _context_lambda). λ=0 below
        # ``nll_warmup_start_step``, linear 0→hold over the next ``nll_warmup_steps`` opt-steps,
        # then hold (the objective's fixed lambda_nll). Defaults (0, 0) ⇒ constant λ==hold from
        # step 0 — backward-compatible with a run that passes no --nll-warmup-* flags.
        self._nll_warmup_start_step = max(int(nll_warmup_start_step), 0)
        self._nll_warmup_steps = max(int(nll_warmup_steps), 0)
        # Context-loss λ RAMP (same eager pattern as λ_nll): λ_ctx=0 below
        # ``ctx_warmup_start_step``, linear 0→hold over the next ``ctx_warmup_steps`` opt-steps,
        # then hold (the objective's fixed lambda_ctx). Only bites when the objective was built
        # with context_loss=True; defaults (0, 0) ⇒ constant λ==hold from step 0.
        self._ctx_warmup_start_step = max(int(ctx_warmup_start_step), 0)
        self._ctx_warmup_steps = max(int(ctx_warmup_steps), 0)
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
        # Per-session cache of the (grid_max_seqlen, m_vis, pack_max_seqlen) Python-int shape
        # constants (``model.session_plan``): computed once per session_key here (uncompiled),
        # then passed into the compiled forward so it skips the per-step ``.item()`` host syncs.
        # Batches are session-homogeneous (v3_collate), so a batch's session_key keys the plan.
        self._session_plan_cache: dict[tuple, tuple[int, int, int]] = {}

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

    def _grad_ratio_due(self, step: int) -> bool:
        return self.grad_ratio_every_n_steps > 0 and step % self.grad_ratio_every_n_steps == 0

    def _nll_lambda(self, step: int | None = None) -> float:
        """The secondary-NLL weight at ``step``: 0 below ``_nll_warmup_start_step``,
        linear 0→hold over the next ``_nll_warmup_steps`` opt-steps, then hold. ``hold``
        is the objective's fixed ``lambda_nll``. Read in EAGER code so λ never enters the
        compiled forward (v2's ``_context_lambda`` pattern). ``step=None`` reads
        ``global_step`` (0 when no trainer is attached)."""
        if step is None:
            try:
                step = int(self.global_step)
            except (RuntimeError, AttributeError):
                step = 0
        hold = float(self.model.objective.lambda_nll)
        s = self._nll_warmup_start_step
        w = self._nll_warmup_steps
        if w > 0:
            frac = min(max(step - s, 0) / w, 1.0)
        else:
            frac = 1.0 if step >= s else 0.0
        return hold * frac

    def _ctx_lambda(self, step: int | None = None) -> float:
        """The context-loss weight at ``step``: the λ_nll ramp's twin over the objective's fixed
        ``lambda_ctx`` hold (0 below ``_ctx_warmup_start_step``, linear over the next
        ``_ctx_warmup_steps``, then hold). Read in EAGER code so λ_ctx never enters the compiled
        forward. hold is 0 when the objective was built without context_loss (no ``lambda_ctx``)."""
        if step is None:
            try:
                step = int(self.global_step)
            except (RuntimeError, AttributeError):
                step = 0
        hold = float(getattr(self.model.objective, "lambda_ctx", 0.0))
        s = self._ctx_warmup_start_step
        w = self._ctx_warmup_steps
        if w > 0:
            frac = min(max(step - s, 0) / w, 1.0)
        else:
            frac = 1.0 if step >= s else 0.0
        return hold * frac

    def _plan_for(self, batch: V3Batch) -> tuple[int | None, int | None, int | None]:
        """The cached ``(grid_max_seqlen, m_vis, pack_max_seqlen)`` for this batch's session,
        computed once per ``session_key`` via ``model.session_plan``. A batch with no
        session_key (unit tests build V3Batch directly) returns all-None ⇒ the eager forward
        derives the constants itself (one host sync/step) — behaviour unchanged there."""
        key = batch.session_key
        if key is None:
            return (None, None, None)
        # Shaft packs (session_key = ("shaft_pack", ΣN)): the geometry CHANGES every step even
        # at constant ΣN — different shafts ⇒ different max_seqlen and m_vis — so the
        # per-session_key cache would hand every step the FIRST pack's STALE plan. Compute
        # fresh each step (uncompiled; m_vis/max_seqlen are deterministic from THIS pack's
        # shaft sizes). grid.total is pinned by the budget, so the compiled shape count stays
        # bounded to the handful of distinct m_vis values.
        is_pack = isinstance(key, tuple) and len(key) >= 1 and key[0] == "shaft_pack"
        plan = None if is_pack else self._session_plan_cache.get(key)
        if plan is None:
            # n_time is the 32 Hz CLOCK length T, not the raw band frame count: r5's bands
            # arrive at 64 Hz (2T) and native-fine SLOW/MID/HGA at T/8 / T/2 / 4T, so
            # bands[0].shape[-1] != T for those frontends. Derive T the same way the model's
            # forward does (clock_length_32hz) or session_plan grids a mis-sized T and m_vis
            # overflows the visible pack (arm0 works only because SLOW is already at 32 Hz).
            n_time = clock_length_32hz(
                batch.bands,
                native_fine_hga=self.model.native_fine_hga,
                early_fusion=self.model.early_fusion,
            )
            plan = self.model.session_plan(batch.geom, batch.parcel_id, n_time)
            if not is_pack:
                self._session_plan_cache[key] = plan
        return plan

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
        # grid_max_seqlen/m_vis/pack_max_seqlen: per-session Python-int shape constants,
        # computed once per session_key here (uncompiled) and passed in so the compiled
        # forward skips their per-step .item() host syncs (None ⇒ eager self-syncing path).
        gms, mvis, pms = self._plan_for(batch)
        out = self.model(
            batch.bands, batch.geom, batch.parcel_id,
            generator=gen, collect_taps=collect,
            stat_mean=batch.stat_mean, stat_std=batch.stat_std,
            grid_max_seqlen=gms, m_vis=mvis, pack_max_seqlen=pms,
        )
        self._last_batch_size = int(batch.bands[0].shape[0])
        self._last_taps = out.taps if collect else None
        # Combine the two losses in EAGER code with the RAMPED λ (compile-safe: the compiled
        # objective already produced jepa_loss/nll_loss separately; the λ multiply happens
        # here, outside the graph). When the secondary is off (nll_loss None) the objective's
        # own total (== jepa_loss) is used unchanged. Crucially, when the secondary is ON we
        # ALWAYS write jepa + λ·nll — even at λ=0, the ``0·nll`` term keeps the Perceiver head
        # in the autograd graph (zero grad, not None) so DDP's reducer sees it "used" and the
        # graph structure never mutates when the ramp opens.
        lam = 0.0
        lam_ctx = 0.0
        if out.jepa_loss is not None:  # a ramped secondary (nll and/or context) is active
            total = out.jepa_loss
            if out.nll_loss is not None:
                lam = self._nll_lambda(step)
                total = total + lam * out.nll_loss
            if out.ctx_loss is not None:
                lam_ctx = self._ctx_lambda(step)
                total = total + lam_ctx * out.ctx_loss
        else:
            total = out.loss
        self.log("train_loss", total, on_step=True, prog_bar=True)
        # n_masked is a 0-dim tensor (sync deferred to the logger cadence, not the compiled
        # forward); Lightning reduces/syncs it at its own log interval.
        self.log("train_n_masked", out.n_masked, on_step=True)
        # Two-loss monitors (Ben 2026-07-15): when the secondary is active, log the primary
        # JEPA L1 and the secondary Gaussian-NLL SEPARATELY so the two objectives are
        # tracked independently (λ-balance readout). Both None on the JEPA-only arm ⇒ only
        # train_loss (== jepa_loss) is logged. LOG-ONLY — nothing here kills the run.
        if out.jepa_loss is not None:
            self.log("train_jepa_loss", out.jepa_loss, on_step=True, prog_bar=True)
        if out.nll_loss is not None:
            self.log("train_nll_loss", out.nll_loss, on_step=True, prog_bar=True)
            # The live ramped λ and the weighted contribution the total actually carries
            # (λ·NLL) — both track the eager ramp directly, so the schedule is visible in wandb.
            self.log("train_nll_lambda", lam, on_step=True)
            self.log("train_nll_weighted", lam * out.nll_loss.detach(), on_step=True)
        # Context-loss monitors: the raw context L1, the live ramped λ_ctx, and the weighted
        # contribution (λ_ctx·ctx) the total carries — all track the eager ramp so the schedule
        # is visible in wandb. LOG-ONLY. Absent on every arm that did not opt into context_loss.
        if out.ctx_loss is not None:
            self.log("train_ctx_loss", out.ctx_loss, on_step=True, prog_bar=True)
            self.log("train_ctx_lambda", lam_ctx, on_step=True)
            self.log("train_ctx_weighted", lam_ctx * out.ctx_loss.detach(), on_step=True)
        # #43 live grad-balance: on cadence, on the last micro-batch, ONLY single-process.
        # Both losses must be graph-connected (jepa_loss/nll_loss are; the weighted log above
        # detached its own copy). retain_graph inside the helper keeps Lightning's subsequent
        # backward(out.loss) intact. Under DDP this would re-enter the reducer → r3 crash, so
        # hard-refuse world_size>1 rather than silently mis-measure.
        if (
            self._grad_ratio_due(step)
            and is_last_microbatch
            and out.jepa_loss is not None
            and out.nll_loss is not None
        ):
            try:
                world_size = int(self.trainer.world_size)
            except (RuntimeError, AttributeError):
                world_size = 1  # no trainer attached (unit tests) ⇒ single-process
            if world_size > 1:
                raise RuntimeError(
                    "grad_ratio_every_n_steps>0 requires single-process (world_size==1): "
                    "the extra autograd.grad passes re-enter the DDP reducer (r3 crash surface). "
                    "Use it only on the 1-GPU diagnostic."
                )
            gr = loss_grad_ratio(
                out.jepa_loss,
                out.nll_loss,
                list(self.model.objective.online.parameters()),
                lambda_nll=lam,
            )
            self.log("train_mon_grad_ratio", gr["grad_ratio"], on_step=True)
            self.log("train_mon_grad_ratio_weighted", gr["grad_ratio_weighted"], on_step=True)
            self.log("train_mon_g_jepa", gr["loss_g_jepa"], on_step=True)
            self.log("train_mon_g_nll", gr["loss_g_nll"], on_step=True)
        return total

    def on_before_zero_grad(self, optimizer) -> None:  # noqa: ARG002
        # Runs after optimizer.step(), once per optimiser step (accum-safe) — the
        # v2 #46 placement. Advances the EMA teacher toward the just-stepped online
        # tower; the coeff schedule (fixed τ 0.99925) advances internally.
        self.model.update_teacher()
