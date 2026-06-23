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

import contextlib
import os
import time
import typing as tp

import torch
from lightning import pytorch as pl
from torch import Tensor, nn

from neuraltrain.optimizers import BaseOptimizer

from speech_decoding.experiments.monitors import (
    RANKME_NORMALISED_ALARM,
    RANKME_NORMALISED_WARN,
    grad_spike_monitor,
    parcel_coverage_monitor,
    teacher_rank_monitor,
)
from speech_decoding.experiments.optim_param_groups import maybe_split_no_decay
from speech_decoding.models.v14_converged import (
    M2MaskConfig,
    M4MaskConfig,
    TightPackConfig,
    V14ConvergedSSL,
    compute_static_shapes,
    cudnn_sdpa_context,
    sample_ssl_masks,
    sample_ssl_masks_static,
)


def _infer_batch_size(batch: tp.Any) -> tp.Optional[int]:
    """Leading-dim batch size from a NeuralSet batch (dict of tensors) or a bare
    tensor; ``None`` if indeterminate. Feeds the per-step throughput monitor and
    the gradient-noise-scale eff-batch ``n``."""
    if isinstance(batch, dict):
        for v in batch.values():
            if isinstance(v, Tensor) and v.dim() >= 1:
                return int(v.shape[0])
    elif isinstance(batch, Tensor) and batch.dim() >= 1:
        return int(batch.shape[0])
    return None


# Joint-canonical metric aliases so the converged run lights up the SAME wandb
# panels as the live 2STFT module (the panels watch the joint names, e.g.
# ``train_loss_m2``, ``train_m4_explained_var``). Per-band diagnostics
# (``l_m2_beta``, ``ev_m4_slow`` …) have no joint analog and keep their native
# names. See V14JointBrainModule._log_breakdown / _log_term_stats.
_LOSS_ALIASES: dict[str, str] = {
    "l_m2": "loss_m2",
    "l_m4": "loss_m4",
    "ev_m2": "m2_explained_var",
    "tv_m2": "m2_target_var",
    "ev_m4": "m4_explained_var",
    "tv_m4": "m4_target_var",
    # collapse triad (the joint module's _log_term_stats names)
    "pv_m2": "m2_pred_var",
    "tn_m2": "m2_target_norm",
    "ptvr_m2": "m2_pred_target_var_ratio",
    "pv_m4": "m4_pred_var",
    "tn_m4": "m4_target_norm",
    "ptvr_m4": "m4_pred_target_var_ratio",
}


def _sdpa_forward_context(name: str | None) -> tp.ContextManager[None]:
    """Context manager that selects the SDPA kernel for the wrapped forward.

    ``"cudnn"`` forces the cuDNN-first priority list (``set_priority=True``)
    around the WHOLE model forward: masked attention runs the Hopper-native sm90
    cuDNN kernel (probe 2026-06-21, GH200: latent masked fwd+bwd 5.51ms vs
    mem-efficient sm80 14.94ms — 2.7×), with mem-efficient as the graceful
    fallback for any call cuDNN declines (NOT math, which OOMs at the latent's
    L). Backward inherits the forward's chosen kernel, so wrapping only the
    forward suffices. ``"cudnn_latent"`` returns ``nullcontext`` here — the
    cuDNN force is instead scoped INSIDE ``LatentEncoder`` to just the large-L
    cross-electrode block loops (the small-L frontend / predictor / time-SA
    calls then keep the cheaper default dispatcher, avoiding cuDNN's
    ~19 ms/call host plan-building where its GPU win is marginal). All other
    names / unset ⇒ ``nullcontext`` (the process-global flags set by
    ``_apply_sdpa_backend`` govern). Identical attention math throughout.
    """
    key = (name or "").strip().lower()
    return cudnn_sdpa_context(key == "cudnn")


def _apply_sdpa_backend(name: str | None) -> None:
    """Science-neutral SDPA backend preference (env-gated, process-global,
    idempotent). Identical attention math — only swaps which fused kernel
    ``F.scaled_dot_product_attention`` dispatches to. Unset/``"default"`` ⇒ no-op.

    ``"cudnn"`` (global force, whole forward) and ``"cudnn_latent"`` (force
    scoped inside ``LatentEncoder`` to the large-L cross-electrode blocks) are
    both handled per-call by the cuDNN context (forcing cuDNN beats the
    auto-dispatcher, which rejects cuDNN for the masked+grad latent call) and are
    NO-OPs here — they must not touch the global flags (disabling mem-efficient
    globally sends cuDNN-declined calls to MATH → OOM at the latent's L).
    ``"flash"``/``"efficient"``/``"math"`` set the process-global enable flags
    directly — diagnostic single-backend forcing for probe runs.
    """
    key = (name or "").strip().lower()
    if not key or key in ("default", "cudnn", "cudnn_latent"):
        return
    if key not in ("flash", "efficient", "math"):
        raise ValueError(
            f"unknown V14_SDPA_BACKEND={name!r} "
            "(expected one of: default, cudnn, cudnn_latent, flash, efficient, math)"
        )
    if not torch.cuda.is_available():
        return
    be = torch.backends.cuda
    be.enable_math_sdp(True)
    if key == "flash":
        be.enable_cudnn_sdp(False)
        be.enable_flash_sdp(True)
        be.enable_mem_efficient_sdp(False)
    elif key == "efficient":
        be.enable_cudnn_sdp(False)
        be.enable_flash_sdp(False)
        be.enable_mem_efficient_sdp(True)
    else:  # key == "math"
        be.enable_cudnn_sdp(False)
        be.enable_flash_sdp(False)
        be.enable_mem_efficient_sdp(False)


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
        tube_cfg: TightPackConfig | None = None,
        static_forward: bool = False,
        mask_seed: int = 0,
        wd_exclude_norms: bool = True,
        monitor_every_n_steps: int | None = None,
    ) -> None:
        super().__init__()
        # Science-neutral SDPA kernel preference (env, NOT a pydantic/uid field, so
        # a cudnn run shares the stock run's exca cache). flash/efficient/math set
        # process-global flags here; "cudnn" is applied per-forward via the
        # priority context manager in `_call_model`. Unset ⇒ no-op. See
        # `_apply_sdpa_backend` / `_sdpa_forward_context`.
        self._sdpa_backend_name = os.environ.get("V14_SDPA_BACKEND")
        _apply_sdpa_backend(self._sdpa_backend_name)
        self.model = model
        self.optim_config = optim_config
        self.ema_tau = float(ema_tau)
        self.m2_cfg = m2_cfg
        self.m4_cfg = m4_cfg
        # None ⇒ legacy variable-count masks (`sample_ssl_masks`). Set ⇒ static
        # V-JEPA-2 masks (`sample_ssl_masks_static`): tight-pack tube + rand_unmask
        # → constant n_vis/N_mask per session, one compiled graph per session.
        self.tube_cfg = tube_cfg
        # Static-shape forward (step B / V-JEPA-2): when True AND tube_cfg is set,
        # `_step` derives the CPU-known gather lengths (StaticShapes) from the CPU
        # masks and threads them through `model.forward` → every ragged gather slices
        # to a fixed length (no .item()/nonzero GPU sync) and the all-real sites run
        # maskless (cuDNN-nomask). REQUIRES the session-homogeneous batch sampler
        # (compute_static_shapes fails loud on a heterogeneous batch). OFF ⇒ legacy
        # variable-count ragged forward (bit-identical). Default OFF until DTAI
        # step-D validates the compiled step-time.
        self.static_forward = bool(static_forward)
        self._wd_exclude_norms = wd_exclude_norms
        # Forward-tap monitor cadence (RankMe / coverage / input-stats). These are
        # CHEAP: _monitor_from_step REUSES the activations the training forward
        # already stashed in model.last_rank_taps — NO extra forward (the #245
        # double-forward regression was removed 2026-06-19). Safe to run every step;
        # kept as a knob only to thin the wandb log, not for throughput. None ⇒ fall
        # back to log_every_n_steps.
        self._monitor_every_n_steps = (
            int(monitor_every_n_steps) if monitor_every_n_steps is not None else None
        )
        # Mask RNG: own CPU generator so the per-step mask draws are reproducible
        # and independent of the global RNG (sample_ssl_masks loops in Python with
        # a torch.Generator). DDP rank-offset is a run-prep refinement, noted: a
        # shared seed across ranks only lowers mask diversity, it is not a
        # correctness bug (the model never assumes distinct masks per rank).
        self._mask_seed = int(mask_seed)
        self._mask_gen = torch.Generator()
        self._mask_gen.manual_seed(self._mask_seed)

        # ── Monitor instrumentation (ported from V14JointBrainModule so the
        # converged run lights up the SAME wandb panels — grad routing, true
        # update-ratio, grad-noise-scale, EMA gap, RankMe, throughput) ──
        # Persistent EMA-of-grad-L2 buffer for the spike monitor (0.0 seeds the
        # first step; survives fit_loop restarts via checkpoint).
        self.register_buffer(
            "_grad_ema_l2", torch.zeros((), dtype=torch.float32), persistent=True,
        )
        self._update_snapshot: dict[int, Tensor] | None = None
        self._last_micro_bsz: int | None = None
        self._last_batch_end_time: float | None = None
        # Diagnostic per-step host-gap split (env `V14_STEP_TIMING=1`, OFF in
        # production ⇒ zero added logs). Splits the existing end-to-end
        # ``step_time_s`` into ``data_wait_s`` (gap before the step = dataloader
        # wait + between-step host work) and ``compute_s`` (in-step fwd/bwd/opt),
        # to attribute the all-GPU-idle bubble seen at 100ms nvidia-smi.
        self._step_timing = os.environ.get("V14_STEP_TIMING") is not None
        self._last_batch_start_time: float | None = None
        # Optional component-split profiler (teacher / student frontend / latent /
        # M2 / M4), gated by env `V14_PROFILE_STEPS="<wait>,<active>"` (or
        # "<wait>:<active>" — colon survives sbatch --export's comma split) — unset ⇒ None
        # ⇒ zero overhead. The forward stamps `record_function("v14/...")` ranges
        # (kept ON always, ~µs each); when this is set, `on_train_start` profiles an
        # `active`-step window AFTER `wait` steps (skip the ~12 static-shape compile
        # graphs) on rank 0 only, and `_dump_profile` prints a CUDA-time table keyed
        # by those ranges + a chrome trace. Compile-safe: Dynamo traces through
        # `record_function`, so the labels survive `--compile --sdpa-backend cudnn`.
        self._profile_spec = os.environ.get("V14_PROFILE_STEPS")
        self._profiler: tp.Any = None
        # Kineto-free per-section mask timing (env `V14_MASK_TIMING=<N>` ⇒ print a
        # median/p10/p90 table over every N train steps; unset ⇒ zero overhead).
        # Pure `perf_counter` + ONE `cuda.synchronize` after the forward — names the
        # all-GPU-idle bubble (mask_sample + mask_static_shapes are pure CPU with
        # NOTHING queued on the GPU) WITHOUT torch.profiler/kineto, which OOMs (CUDA
        # activity) or HANGS (chrome export) at cycle-end on aarch64 GH200 + torch
        # 2.10. mask_d2h absorbs the prior step's GPU-drain; mask_h2d is the upload;
        # fwd is the forward GPU time (the post-forward sync makes it real, at the
        # cost of one sync this probe-only path adds — production leaves it OFF).
        # Rank-0 + training only.
        _mt = os.environ.get("V14_MASK_TIMING")
        self._mask_timing_every = 0
        if _mt is not None:
            try:
                _mt_n = int(_mt)
            except ValueError:
                _mt_n = 50
            self._mask_timing_every = _mt_n if _mt_n > 0 else 50
        self._mask_timing_buf: dict[str, list[float]] = {
            "d2h": [], "sample": [], "static": [], "h2d": [], "fwd": [],
        }
        # RankMe normalised warn/alarm thresholds (the teacher_rank_monitor
        # defaults; surfaced as attrs so a future run can override per-arch).
        self._rankme_warn_threshold = float(RANKME_NORMALISED_WARN)
        self._rankme_alarm_threshold = float(RANKME_NORMALISED_ALARM)

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
        # Resolved (mode, dynamic, optimize_ddp) when compile is requested, else
        # None. Kept as a plain tuple so it survives the job-pickle (see
        # __getstate__) and _call_model can rebuild the OptimizedModule lazily.
        self._compile_spec: tuple[str, bool | None, bool] | None = None
        _compile_flag = os.environ.get("V14_COMPILE", "").strip().lower()
        if _compile_flag not in ("", "0", "false", "no", "off"):
            _mode = os.environ.get("V14_COMPILE_MODE") or "default"
            # Three states, NOT two: True = symbolic-shapes (compile once), False
            # = FULLY static (recompile per concrete shape, NO symbolic reasoning),
            # unset = None = torch's automatic-dynamic (static on shape #1, then
            # symbolic on shape #2). On torch 2.10 / GH200 both True AND None storm
            # in the symbolic solver (`pow_by_natural([VR[.., int_oo], -1])` spam,
            # never reaches a step) on this model's data-dependent gather; False is
            # the escape that keeps every dim concrete. So "0"/"false" must map to
            # False, not None.
            _dyn_flag = os.environ.get("V14_COMPILE_DYNAMIC", "").strip().lower()
            if _dyn_flag in ("1", "true", "yes", "on"):
                _dynamic: bool | None = True
            elif _dyn_flag in ("0", "false", "no", "off"):
                _dynamic = False
            else:
                _dynamic = None
            # DDPOptimizer × dynamic-shapes fix (2026-06-11): the bucket-split
            # optimizer hands a symbolic-shape SymInt back as a bare python int and
            # crashes inductor under dynamic=True; disabling it compiles a single
            # graph (cost: lost allreduce/compute overlap, negligible for this
            # ~16M-param model on single-node 4-GPU DDP). Default OFF = disabled,
            # because the production sweep IS compile+DDP+dynamic.
            _ddp_opt = os.environ.get(
                "V14_COMPILE_DDP_OPTIMIZER", "").strip().lower()
            self._compile_spec = (
                _mode, _dynamic, _ddp_opt in ("1", "true", "yes", "on"),
            )
            self._build_compiled_model()

    def _build_compiled_model(self) -> None:
        """Compile ``self.model`` per ``self._compile_spec`` and register it in the
        (plain, non-submodule) ``_compiled_fwd`` dict. Re-applies the global
        DDPOptimizer switch because unpickling bypasses ``__init__``: the
        in-allocation-ddp / submitit job-pickle drops the un-picklable
        OptimizedModule via ``__getstate__``, then ``_call_model`` calls this on
        the first forward to rebuild it on the live (un-pickled) module."""
        if self._compile_spec is None:
            return
        _mode, _dynamic, _optimize_ddp = self._compile_spec
        import torch._dynamo as _dynamo_mod

        _dynamo_mod.config.optimize_ddp = _optimize_ddp
        # Per-frame recompile cap. Under the static-shape forward (step B) +
        # `dynamic=False`, dynamo compiles ONE concrete graph per distinct session
        # geometry — a bounded, LEGITIMATE set (~one per BT session, not a dynamism
        # bug). The default cap (8) is below the corpus session count (~19-25
        # distinct n_vis), so without a raise dynamo silently falls back to eager
        # past the 8th shape. Raise to cover the corpus + margin (env-tunable).
        # torch 2.10 renamed cache_size_limit → recompile_limit (alias); set both.
        _cap = int(os.environ.get("V14_COMPILE_CACHE_LIMIT", "64"))
        for _attr in ("cache_size_limit", "recompile_limit"):
            if hasattr(_dynamo_mod.config, _attr):
                setattr(_dynamo_mod.config, _attr, _cap)
        self._compiled_fwd["model"] = torch.compile(
            self.model, mode=_mode, dynamic=_dynamic,
        )

    def __getstate__(self) -> dict[str, tp.Any]:
        """Make the module cloudpickle-safe even when compiled. exca pickles the
        live job graph on the ``--in-allocation-ddp`` path (and submitit pickles
        for the remote run), but the compiled forward carries ``torch._dynamo``
        guard weakrefs that cloudpickle cannot serialize (``cannot pickle
        'weakref.ReferenceType'``). Drop the OptimizedModule from the pickled
        state — ``_compile_spec`` (a plain tuple) survives, so ``_call_model``
        rebuilds it lazily on the first forward after unpickle. ``__getstate__``
        operates on a copy, so the in-process original keeps its compiled forward
        (single-GPU / cluster runs that never round-trip are unaffected)."""
        state = self.__dict__.copy()
        state["_compiled_fwd"] = {}
        return state

    def _call_model(self, *args: tp.Any, **kwargs: tp.Any) -> dict[str, Tensor]:
        """Run the model forward through the compiled override when present, else
        eager. Rebuilds the compiled forward lazily when ``_compile_spec`` is set
        but the OptimizedModule is absent (first forward after an un-pickle).
        Falls back to ``self.model`` when V14_COMPILE was unset (tests / 1-GPU),
        so the eager path is byte-identical to pre-compile."""
        if self._compile_spec is not None and "model" not in self._compiled_fwd:
            self._build_compiled_model()
        # "cudnn" forces the cuDNN-first SDPA priority for this forward (backward
        # inherits the kernel); other modes ⇒ nullcontext. Science-neutral.
        with _sdpa_forward_context(self._sdpa_backend_name):
            return self._compiled_fwd.get("model", self.model)(*args, **kwargs)

    # ==================================================================
    # Monitor instrumentation — ported from V14JointBrainModule.
    # Param-routing partition for the converged arch. Unlike the joint module
    # (which lumps both heads into one "predictor" bucket), the converged arch
    # has TWO distinct predictors, so the routing splits FOUR ways —
    # student_frontend / latent / m2_predictor / m4_predictor — so the grad
    # balance and update-ratio of each head are independently visible. The four
    # buckets EXACTLY tile the trainable set (teacher_frontend is frozen, so
    # requires_grad excludes it), so the per-group squared norms sum to global.
    # ==================================================================
    #: Routing groups, in log order. Frozen tuple so every routing method
    #: (grad norms, weight norms, lrs, true-update-ratio) iterates the same set.
    _ROUTING_GROUPS: tp.ClassVar[tuple[str, ...]] = (
        "frontend", "latent", "m2_predictor", "m4_predictor",
    )

    def _predictor_parameters(self) -> list[nn.Parameter]:
        return [
            *self.model.m2_predictor.parameters(),
            *self.model.m4_predictor.parameters(),
        ]

    def _trainable_parameters(self) -> tp.Iterator[nn.Parameter]:
        """Every param that carries a gradient (student frontend + latent +
        the two predictors; the EMA teacher_frontend is frozen)."""
        for p in self.model.parameters():
            if p.requires_grad:
                yield p

    def _group_of(
        self, p: nn.Parameter, *, front_ids: set[int],
        m2_ids: set[int], m4_ids: set[int],
    ) -> str:
        """Route a param to its routing group. ``latent`` is the residual bucket
        (everything trainable that is neither the frontend stem nor a head)."""
        i = id(p)
        if i in front_ids:
            return "frontend"
        if i in m2_ids:
            return "m2_predictor"
        if i in m4_ids:
            return "m4_predictor"
        return "latent"

    def _routing_id_sets(self) -> tuple[set[int], set[int], set[int]]:
        return (
            {id(p) for p in self.model.student_frontend.parameters()},
            {id(p) for p in self.model.m2_predictor.parameters()},
            {id(p) for p in self.model.m4_predictor.parameters()},
        )

    def _grad_routing_norms(
        self,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], Tensor]:
        """Per-group grad-L2² + weight-L2², single pass (the joint module's #119
        routing, split four ways for the converged arch's two heads)."""
        front_ids, m2_ids, m4_ids = self._routing_id_sets()
        dev = self._grad_ema_l2.device

        def _z() -> Tensor:
            return torch.zeros((), device=dev, dtype=torch.float32)

        grad_sq = {g: _z() for g in self._ROUTING_GROUPS}
        weight_sq = {g: _z() for g in self._ROUTING_GROUPS}
        for p in self._trainable_parameters():
            key = self._group_of(
                p, front_ids=front_ids, m2_ids=m2_ids, m4_ids=m4_ids,
            )
            weight_sq[key] = weight_sq[key] + p.detach().to(
                torch.float32
            ).pow(2).sum()
            if p.grad is not None:
                grad_sq[key] = grad_sq[key] + p.grad.detach().to(
                    torch.float32
                ).pow(2).sum()
        total_grad_sq = sum(
            (grad_sq[g] for g in self._ROUTING_GROUPS), start=_z(),
        )
        return grad_sq, weight_sq, total_grad_sq

    def _group_lrs(self, optimizer: tp.Any) -> dict[str, float]:
        """Live (scheduled) lr per routing group off ``optimizer.param_groups``;
        NaN for a group absent from the optimizer (no trainer / unit test)."""
        nan = float("nan")
        if optimizer is None or not getattr(optimizer, "param_groups", None):
            return {g: nan for g in self._ROUTING_GROUPS}
        lr_by_id: dict[int, float] = {}
        for pg in optimizer.param_groups:
            lr = float(pg.get("lr", nan))
            for p in pg["params"]:
                lr_by_id[id(p)] = lr
        front_ids, m2_ids, m4_ids = self._routing_id_sets()
        out = {g: nan for g in self._ROUTING_GROUPS}
        for p in self._trainable_parameters():
            key = self._group_of(
                p, front_ids=front_ids, m2_ids=m2_ids, m4_ids=m4_ids,
            )
            if out[key] != out[key]:  # first param seen for this group
                out[key] = lr_by_id.get(id(p), nan)
        return out

    def _eff_batch_size(self) -> int | None:
        """``n = micro×accum×world`` — the gradient-noise-scale batch unit."""
        micro = self._last_micro_bsz
        if micro is None:
            return None
        try:
            accum = int(self.trainer.accumulate_grad_batches)
            world = int(self.trainer.world_size)
        except (RuntimeError, AttributeError):
            accum, world = 1, 1
        return int(micro) * max(accum, 1) * max(world, 1)

    def _grad_noise_scale(
        self, optimizer: tp.Any,
    ) -> tuple[float, float, float]:
        """Gradient noise scale ``B_simple`` (McCandlish 2018) off AdamW's moment
        EMAs: ``n·Σ(v̂−m̂²)₊/Σm̂²``. All-NaN before the first AdamW step / under a
        non-Adam optimizer / when ``n`` is unknown. Pure optimizer-state read."""
        inner = optimizer
        seen_inner: set[int] = set()
        while inner is not None and id(inner) not in seen_inner:
            seen_inner.add(id(inner))
            nxt = getattr(inner, "optimizer", None)
            if nxt is None or nxt is inner:
                break
            inner = nxt
        state = getattr(inner, "state", None)
        if not state:
            return (float("nan"), float("nan"), float("nan"))
        betas_by_id: dict[int, tuple[float, float]] = {}
        for pg in getattr(inner, "param_groups", []):
            b = pg.get("betas", (0.9, 0.999))
            for p in pg["params"]:
                betas_by_id[id(p)] = (float(b[0]), float(b[1]))
        dev = self._grad_ema_l2.device
        signal = torch.zeros((), device=dev, dtype=torch.float32)
        var = torch.zeros((), device=dev, dtype=torch.float32)
        seen = False
        for p in self._trainable_parameters():
            st = state.get(p)
            if st is None or "exp_avg" not in st or "exp_avg_sq" not in st:
                continue
            seen = True
            b1, b2 = betas_by_id.get(id(p), (0.9, 0.999))
            step = st.get("step", 0.0)
            t = float(step.item()) if hasattr(step, "item") else float(step)
            c1 = 1.0 - b1 ** t if t > 0.0 else 1.0
            c2 = 1.0 - b2 ** t if t > 0.0 else 1.0
            m_hat = st["exp_avg"].detach().to(torch.float32) / c1
            v_hat = st["exp_avg_sq"].detach().to(torch.float32) / c2
            signal = signal + m_hat.pow(2).sum()
            var = var + (v_hat - m_hat.pow(2)).sum()
        if not seen:
            return (float("nan"), float("nan"), float("nan"))
        sig = float(signal.item())
        vr = max(0.0, float(var.item()))
        n = self._eff_batch_size()
        if n is None or sig <= 0.0:
            return (float("nan"), sig, vr)
        return (n * vr / sig, sig, vr)

    def _ema_weight_gap(self) -> float:
        """``‖θ_student − θ_teacher‖₂ / ‖θ_student‖₂`` over the EMA-mirrored
        FRONTEND (the only EMA-tracked module in the converged arch — the latent
        and predictors are student-only)."""
        dev = self._grad_ema_l2.device
        num = torch.zeros((), device=dev, dtype=torch.float32)
        den = torch.zeros((), device=dev, dtype=torch.float32)
        for ps, pt in zip(
            self.model.student_frontend.parameters(),
            self.model.teacher_frontend.parameters(),
        ):
            s = ps.detach().to(torch.float32)
            num = num + (s - pt.detach().to(torch.float32)).pow(2).sum()
            den = den + s.pow(2).sum()
        return float((num.sqrt() / (den.sqrt() + 1e-12)).item())

    def _train_monitor_due(self, batch_idx: int) -> bool:
        """Monitor cadence: True on step 0 and every ``log_every_n_steps`` steps;
        every step (cadence 1) when no trainer is attached (unit tests)."""
        try:
            cadence = int(self.trainer.log_every_n_steps)
        except (RuntimeError, AttributeError):
            cadence = 1
        return batch_idx % max(cadence, 1) == 0

    def _monitor_tap_due(self, batch_idx: int) -> bool:
        """Cadence for the forward-tap monitors (RankMe / coverage / input-stats).
        These are CHEAP — they reuse the training forward's stashed activations, no
        extra forward (see _monitor_from_step). ``monitor_every_n_steps`` only thins
        the wandb log; ``None`` falls back to ``log_every_n_steps``."""
        if self._monitor_every_n_steps is None:
            return self._train_monitor_due(batch_idx)
        return batch_idx % max(self._monitor_every_n_steps, 1) == 0

    def _update_ratio_due(self) -> bool:
        try:
            step = int(self.global_step)
        except (RuntimeError, AttributeError):
            step = 0
        return self._train_monitor_due(step)

    def _maybe_log_true_update_ratio(self) -> None:
        """True per-group ``‖Δθ‖/‖θ‖`` across two consecutive optimizer steps —
        the LR-calibration readback that is MONOTONIC in LR (unlike the clip
        fraction, which is U-shaped under the collapse-thrash confound)."""
        snap = self._update_snapshot
        if snap is not None:
            front_ids, m2_ids, m4_ids = self._routing_id_sets()
            dev = self._grad_ema_l2.device

            def _z() -> Tensor:
                return torch.zeros((), device=dev, dtype=torch.float32)

            delta_sq = {g: _z() for g in self._ROUTING_GROUPS}
            base_sq = {g: _z() for g in self._ROUTING_GROUPS}
            for p in self._trainable_parameters():
                before = snap.get(id(p))
                if before is None:
                    continue
                key = self._group_of(
                    p, front_ids=front_ids, m2_ids=m2_ids, m4_ids=m4_ids,
                )
                after = p.detach().to(torch.float32)
                b = before.to(torch.float32)
                delta_sq[key] = delta_sq[key] + (after - b).pow(2).sum()
                base_sq[key] = base_sq[key] + b.pow(2).sum()
            for group in self._ROUTING_GROUPS:
                bn = float(base_sq[group].sqrt().item())
                if bn > 0.0:
                    self.log(
                        f"train_mon_true_update_ratio_{group}",
                        float(delta_sq[group].sqrt().item()) / (bn + 1e-12),
                        on_step=True,
                    )
            self._update_snapshot = None
        if self._update_ratio_due():
            self._update_snapshot = {
                id(p): p.detach().clone() for p in self._trainable_parameters()
            }

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
        # Kineto-free per-section wall timing (V14_MASK_TIMING). Only when armed +
        # in training mode; reads `perf_counter` between the four mask ranges and
        # syncs once after the forward. Off ⇒ `mt` False ⇒ not a single extra call.
        mt = bool(self._mask_timing_every) and self.training
        _t: list[float] = [time.perf_counter()] if mt else []
        # Mask sampling runs on CPU (a CPU generator can't drive CUDA randperm);
        # move the masks back to the feature device for the forward. Copy ppe/emask
        # to host ONCE and reuse for both mask sampling and the static-shape
        # reduction below — same tensor + same CPU generator ⇒ bit-identical masks,
        # one fewer D2H copy than the old per-call ``.cpu()`` form.
        #
        # The four ``v14/mask_*`` ranges below name the host-side mask build for the
        # bubble probe: ``mask_d2h`` self-CPU absorbs the GPU-DRAIN sync (``.cpu()``
        # blocks until the prior step's queued GPU work finishes — "good" idle, the
        # GPU was busy); ``mask_sample`` + ``mask_static_shapes`` are PURE CPU work
        # with NOTHING queued on the GPU (= the all-GPU-idle bubble); ``mask_h2d`` is
        # the upload. Annotations only — inert when no profiler is active (~ns).
        with torch.profiler.record_function("v14/mask_d2h"):
            ppe_cpu = ppe.cpu()
            emask_cpu = emask.cpu()
        if mt:
            _t.append(time.perf_counter())
        with torch.profiler.record_function("v14/mask_sample"):
            if self.tube_cfg is None:
                masks = sample_ssl_masks(
                    ppe_cpu, emask_cpu, self._mask_gen,
                    m2_cfg=self.m2_cfg, m4_cfg=self.m4_cfg, bands=self.model.bands,
                )
            else:
                masks = sample_ssl_masks_static(
                    ppe_cpu, emask_cpu, self._mask_gen,
                    m2_cfg=self.m2_cfg, tube_cfg=self.tube_cfg, bands=self.model.bands,
                )
        if mt:
            _t.append(time.perf_counter())
        # Static-shape forward: derive the CPU-known gather lengths from the CPU
        # masks (cheap CPU reductions, no GPU sync) BEFORE the device move, then pass
        # them through so every gather slices to a fixed length. Only when the static
        # mask regime is active (tube_cfg set) and the flag is on.
        static = None
        if self.static_forward and self.tube_cfg is not None:
            with torch.profiler.record_function("v14/mask_static_shapes"):
                static = compute_static_shapes(
                    emask_cpu.to(torch.bool), masks["m2_mask"], masks["tube_mask"],
                    self.tube_cfg.p_fixed,
                )
        if mt:
            _t.append(time.perf_counter())
        with torch.profiler.record_function("v14/mask_h2d"):
            masks = {k: v.to(slow.device) for k, v in masks.items()}
        if mt:
            _t.append(time.perf_counter())
        out = self._call_model(slow, beta, hg, ppe, emask, **masks, static=static)
        if mt:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t.append(time.perf_counter())
            self._mask_timing_record(_t)
        return out

    def _mask_timing_record(self, t: list[float]) -> None:
        """Accumulate one step's per-section walls (perf_counter stamps in `t`,
        6 entries: start, +d2h, +sample, +static, +h2d, +forward); flush a
        median/p10/p90 table every `_mask_timing_every` steps. Prints on rank 0;
        clears on every rank so non-zero ranks stay bounded."""
        if len(t) < 6:
            return
        buf = self._mask_timing_buf
        buf["d2h"].append((t[1] - t[0]) * 1e3)
        buf["sample"].append((t[2] - t[1]) * 1e3)
        buf["static"].append((t[3] - t[2]) * 1e3)
        buf["h2d"].append((t[4] - t[3]) * 1e3)
        buf["fwd"].append((t[5] - t[4]) * 1e3)
        if len(buf["d2h"]) < self._mask_timing_every:
            return
        is_rank0 = not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
            and torch.distributed.get_rank() != 0
        )
        if is_rank0:
            def _sm(xs: list[float]) -> tuple[float, float, float, float]:
                s = sorted(xs)
                n = len(s)
                med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
                p10 = s[max(0, int(0.1 * (n - 1)))]
                p90 = s[min(n - 1, int(0.9 * (n - 1)))]
                return med, p10, p90, sum(s) / n
            n = len(buf["d2h"])
            bubble = [a + b for a, b in zip(buf["sample"], buf["static"])]
            total = [d + s + st + h + f for d, s, st, h, f in zip(
                buf["d2h"], buf["sample"], buf["static"], buf["h2d"], buf["fwd"])]
            rows = [
                ("mask_d2h(drain)", buf["d2h"]),
                ("mask_sample", buf["sample"]),
                ("mask_static", buf["static"]),
                ("mask_h2d", buf["h2d"]),
                ("forward", buf["fwd"]),
                ("BUBBLE samp+stat", bubble),
                ("step sum(5)", total),
            ]
            lines = [
                f"=== V14 mask timing (kineto-free) — {n} steps, ms ===",
                f"  {'section':<18}{'median':>9}{'p10':>9}{'p90':>9}{'mean':>9}",
            ]
            for name, xs in rows:
                med, p10, p90, mean = _sm(xs)
                lines.append(
                    f"  {name:<18}{med:>9.2f}{p10:>9.2f}{p90:>9.2f}{mean:>9.2f}")
            print("\n".join(lines), flush=True)
        for k in buf:
            buf[k].clear()

    # ------------------------------------------------------------------- loops
    def training_step(self, batch: tp.Any, batch_idx: int) -> Tensor:
        # Stash this rank's micro-batch for the gradient-noise-scale eff-batch n.
        self._last_micro_bsz = _infer_batch_size(batch.data)
        out = self._step(batch.data)
        self._log_losses(out, "train", on_step=True, on_epoch=False)
        # Forward-tap diagnostics (RankMe / coverage / input stats) reuse the
        # forward's stashed activations — cheap, no extra forward. The cadence only
        # thins the wandb log; decoupled from the per-step loss logging above.
        if self._monitor_tap_due(batch_idx):
            self._monitor_from_step(batch.data, step_name="train")
        return out["loss"]

    def validation_step(self, batch: tp.Any, batch_idx: int) -> None:  # noqa: ARG002
        out = self._step(batch.data)
        self._log_losses(out, "val", on_step=False, on_epoch=True)
        self._monitor_from_step(batch.data, step_name="val")

    def _log_losses(
        self, out: dict[str, Tensor], name: str, *, on_step: bool, on_epoch: bool,
    ) -> None:
        """Log every scalar the forward emits: loss + the two head scalars, the
        per-band/aggregate monitor diagnostics (loss / explained_var / target_var
        per band {slow,beta,hg}), and the per-band stem-output norms. Non-finite
        scalars are skipped — ev/tv are NaN when a band has < 2 scored cells this
        step, and an undefined-variance step must not poison the epoch mean."""
        # Collapse the per-scalar ``bool(torch.isfinite(val))`` D2H syncs (one per
        # logged scalar, ~10-30/step) into ONE: stack the 0-d scalars and read the
        # finite mask in a single host transfer. ``.float()`` guards against a
        # mixed-dtype stack; the ORIGINAL ``val`` is still what gets logged. Skip
        # semantics are unchanged — a non-finite (e.g. <2-cell band variance) scalar
        # is dropped so it can't poison the epoch mean.
        scalars = [(k, v) for k, v in out.items() if v.dim() == 0]
        if not scalars:
            return
        finite = torch.isfinite(
            torch.stack([v.float() for _, v in scalars])
        ).tolist()
        for (key, val), ok in zip(scalars, finite):
            if not ok:
                continue
            metric = _LOSS_ALIASES.get(key, key)
            self.log(
                f"{name}_{metric}", val,
                on_step=on_step, on_epoch=on_epoch,
                prog_bar=(key == "loss"),
            )

    # ----------------------------------------------------- forward-tap monitors
    _RANKME_N_MAX: tp.ClassVar[int] = 4096
    _RANKME_SUBSAMPLE_SEED: tp.ClassVar[int] = 0x9E3779B1

    def _rankme_subsample(self, flat: Tensor) -> Tensor:
        """Uniform-random subsample to ≤``_RANKME_N_MAX`` rows for a cheap SVD.

        RankMe's singular spectrum is permutation-invariant over rows, so any
        subset is valid IN EXPECTATION — but it must be unbiased w.r.t. row
        STRUCTURE. ``flat`` is ordered (batch, electrode, time) with time
        innermost, so a fixed STRIDE (the old impl) phase-locks onto a single
        time-bin whenever the stride lands near a multiple of the per-electrode
        token period S — i.e. whenever ``B·C / _RANKME_N_MAX`` is near an
        integer — annihilating temporal diversity and crashing the rank
        estimate ("needle"). That resonance moves with batch size
        (``stride = n_rows // N_MAX`` ∝ B), so bs32 needles where bs8 does not —
        a sampling artifact, not a real rank gap. A uniform-random draw has no
        phase to lock onto: stride-free and count-stable. Fixed-seed generator →
        same input tensor ⇒ same subset (resume-stable, reproducible). Mirrors
        the v14_joint_module fix (0d6692e), which never reached this module.
        """
        n = flat.shape[0]
        if n > self._RANKME_N_MAX:
            g = torch.Generator(device=flat.device)
            g.manual_seed(self._RANKME_SUBSAMPLE_SEED)
            idx = torch.randperm(n, generator=g, device=flat.device)
            flat = flat[idx[: self._RANKME_N_MAX]]
        return flat

    def _log_rankme(self, verdict, *, step_name: str, key: str = "") -> None:
        prefix = f"{step_name}_mon_{key}rankme"
        self.log(prefix, verdict.rankme, on_epoch=True)
        self.log(f"{prefix}_normalised", verdict.rankme_normalised, on_epoch=True)
        self.log(f"{prefix}_warn", 1.0 if verdict.is_warn else 0.0, on_epoch=True)
        self.log(f"{prefix}_alarm", 1.0 if verdict.is_alarm else 0.0, on_epoch=True)

    def _log_feature_stats(
        self, rows: Tensor, *, step_name: str, key: str = "",
    ) -> None:
        """VICReg-style per-dim feature health on the ``(N, d)`` RankMe rows:
        ``feat_std_mean`` / ``feat_std_min`` (a dim whose std→0 has died) and
        ``feat_cov_offdiag`` (dimensional redundancy). No-op for <2 / non-2-D."""
        rows = rows.detach()
        if rows.dim() != 2 or rows.shape[0] < 2:
            return
        prefix = f"{step_name}_mon_{key}feat"
        std = rows.std(dim=0, unbiased=False)
        self.log(f"{prefix}_std_mean", std.mean(), on_epoch=True)
        self.log(f"{prefix}_std_min", std.min(), on_epoch=True)
        d = rows.shape[1]
        denom = d * (d - 1)
        if denom > 0:
            centred = rows - rows.mean(dim=0, keepdim=True)
            cov = (centred.t() @ centred) / (rows.shape[0] - 1)
            off_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
            self.log(f"{prefix}_cov_offdiag", off_sq / denom, on_epoch=True)

    def _run_rank_monitor(
        self, *, tap: Tensor, valid: Tensor, step_name: str, key: str,
        token_valid: Tensor | None = None,
    ) -> None:
        """RankMe + feature stats on a per-electrode ``(B, C, S, d)`` tap, gating
        out the non-real rows so padding/masked tokens don't dilute the rank.
        ``key="frontend_"`` → ``mon_frontend_rankme*`` (teacher frontend M2 target,
        gated per-ELECTRODE by ``valid (B, C)`` — every token of a real electrode
        is real); ``key=""`` → ``mon_rankme*`` (student latent). When ``token_valid
        (B, C, S)`` is given it gates per-TOKEN instead — the ragged latent writes
        zeros off the visible set, so ranking those zero rows would corrupt the
        spectrum; ``token_valid`` keeps only the rows the latent actually wrote."""
        if tap.dim() != 4:
            return
        B, C, S, d = tap.shape
        flat = tap.reshape(B * C * S, d)
        if token_valid is not None and tuple(token_valid.shape) == (B, C, S):
            flat = flat[token_valid.reshape(-1).bool()]
        elif tuple(valid.shape[:2]) == (B, C):
            keep = valid.reshape(B, C, 1).expand(B, C, S).reshape(-1)
            flat = flat[keep]
        sub = self._rankme_subsample(flat)
        self._log_feature_stats(sub, step_name=step_name, key=key)
        verdict = teacher_rank_monitor(
            sub.detach(),
            warn_threshold=self._rankme_warn_threshold,
            alarm_threshold=self._rankme_alarm_threshold,
        )
        self._log_rankme(verdict, step_name=step_name, key=key)

    def _run_parcel_coverage_monitor(
        self, *, parcel_per_electrode: Tensor, electrode_mask: Tensor,
        n_parcels: int, step_name: str,
    ) -> None:
        """Active-parcel coverage per clip — a parcel is active iff ≥1 VALID
        electrode maps to it. Scatters ``parcel_per_electrode`` (gated by the
        electrode mask) into a ``(B, K)`` bool occupancy and feeds the shared
        coverage monitor (degenerate / low-variance coverage → alarm)."""
        B = parcel_per_electrode.shape[0]
        dev = parcel_per_electrode.device
        valid = electrode_mask.to(torch.bool)
        idx = parcel_per_electrode.clamp(0, n_parcels - 1)
        # Occupancy ``(B, K)``: parcel p active in clip b iff a VALID electrode
        # maps to it. scatter-add the valid-electrode count into per-parcel bins,
        # then threshold > 0 (a bool scatter isn't an OR, so count then compare).
        counts = torch.zeros(B, n_parcels, dtype=torch.long, device=dev)
        counts.scatter_add_(1, idx, valid.to(torch.long))
        latent_valid = counts > 0
        verdict = parcel_coverage_monitor(latent_valid)
        self.log(
            f"{step_name}_mon_coverage_active_mean",
            verdict.active_slots_per_clip_mean, on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_active_cv",
            verdict.active_slots_per_clip_cv, on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_slot_var",
            verdict.slot_usage_fraction_var, on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_degenerate_frac",
            verdict.degenerate_clip_fraction, on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_swec_frac",
            verdict.front_end_only_clip_fraction, on_epoch=True,
        )
        self.log(
            f"{step_name}_mon_coverage_alarm",
            1.0 if verdict.is_alarm else 0.0, on_epoch=True,
        )

    def _run_input_stats_monitor(
        self, *, data: dict[str, Tensor], step_name: str,
    ) -> None:
        """Sanity stats on the model's ACTUAL per-band inputs (the bad-electrode /
        normalization tripwire): under prod robust-z each |STFT| band lands ~O(1)
        mean/std, so a drifting mean/std flags a normalization break, a giant
        absmax flags a spectral transient past the STATIC/WINSOR guards, and a
        non-zero nonfinite-frac flags a corrupt-cache batch before it zeros the
        loss. Reductions over FINITE values only."""
        on_step = step_name == "train"

        def _band(tensor: tp.Optional[Tensor], suffix: str) -> None:
            if not isinstance(tensor, Tensor) or tensor.numel() == 0:
                return
            f = tensor.detach().to(torch.float32)
            finite = torch.isfinite(f)
            n = f.numel()
            n_finite = int(finite.sum().item())
            self.log(
                f"{step_name}_mon_input_{suffix}_nonfinite_frac",
                float((n - n_finite) / n), on_step=on_step, on_epoch=True,
            )
            if n_finite < 1:
                return
            ff = f[finite]
            self.log(
                f"{step_name}_mon_input_{suffix}_absmax", ff.abs().max(),
                on_step=on_step, on_epoch=True,
            )
            if n_finite >= 2:
                self.log(
                    f"{step_name}_mon_input_{suffix}_mean", ff.mean(),
                    on_step=on_step, on_epoch=True,
                )
                self.log(
                    f"{step_name}_mon_input_{suffix}_std", ff.std(unbiased=False),
                    on_step=on_step, on_epoch=True,
                )

        _band(data.get("electrode_tokens_slow"), "slow")
        _band(data.get("electrode_tokens_beta"), "beta")
        _band(data.get("electrode_tokens_hg"), "hg")

    @torch.no_grad()
    def _monitor_from_step(self, data: dict[str, Tensor], *, step_name: str) -> None:
        """Forward-tap diagnostics (RankMe on the teacher-frontend M2 target + the
        student-latent representation, parcel coverage, input stats).

        The two RankMe taps REUSE the activations the just-run training forward
        already produced (``model.last_rank_taps``) instead of re-running a second
        dense full-input forward every step — that double-forward was the #245
        throughput regression (Ben 2026-06-19). The teacher-frontend tap is
        byte-identical to a fresh ``teacher_frontend(slow,beta,hg)`` (electrode-
        isolated, teacher saw all tokens). The student-latent tap now reflects the
        MASKED-context visible-token representation the loss actually shapes (vs
        the old full-context dense pass) — the more faithful collapse tripwire, and
        free. Falls back to a fresh dense encode only if the stash is empty (a
        monitor call not preceded by a forward, e.g. a unit test)."""
        slow, beta, hg, ppe, emask = self._converged_inputs(data)
        self._run_input_stats_monitor(data=data, step_name=step_name)
        n_parcels = int(data["support"].shape[-1])
        self._run_parcel_coverage_monitor(
            parcel_per_electrode=ppe, electrode_mask=emask,
            n_parcels=n_parcels, step_name=step_name,
        )
        taps = getattr(self.model, "last_rank_taps", {}) or {}
        # Frontend M2 rank on the EMA teacher (the post-frontend target P-space).
        teacher_m2 = taps.get("teacher_frontend")
        if teacher_m2 is None:
            teacher_m2 = self.model.teacher_frontend(slow, beta, hg)
        self._run_rank_monitor(
            tap=teacher_m2, valid=emask, step_name=step_name, key="frontend_",
        )
        # Deep-representation rank on the student latent.
        student_latent = taps.get("student_latent")
        latent_valid = taps.get("student_latent_valid")
        if student_latent is None:
            student_latent = self.model.encode_latent(
                slow, beta, hg, ppe, electrode_mask=emask,
            )
            latent_valid = None
        self._run_rank_monitor(
            tap=student_latent, valid=emask, token_valid=latent_valid,
            step_name=step_name, key="",
        )

    # ----------------------------------------------------------- gradient hooks
    def on_before_optimizer_step(self, optimizer: tp.Any) -> None:
        """Grad-routing + spike/divergence + clip + grad-noise-scale + EMA-gap +
        true-update-ratio probe before ``optimizer.step()`` (ported from the
        joint module's #119/SPAM monitor). Pure observability — reads grads +
        optimizer state, mutates only the persistent EMA buffer."""
        try:
            _step = int(self.global_step)
        except (RuntimeError, AttributeError):
            _step = 0
        if not self._train_monitor_due(_step):
            self._maybe_log_true_update_ratio()
            return
        grad_sq, weight_sq, total_grad_sq = self._grad_routing_norms()
        grad_l2 = float(total_grad_sq.sqrt().item())
        verdict = grad_spike_monitor(
            grad_l2=grad_l2, prior_ema_l2=float(self._grad_ema_l2.item()),
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
            1.0 if verdict.is_diverged else 0.0, on_step=True,
        )
        for group in self._ROUTING_GROUPS:
            self.log(
                f"train_mon_grad_l2_{group}",
                float(grad_sq[group].sqrt().item()), on_step=True,
            )
            self.log(
                f"train_mon_wnorm_{group}",
                float(weight_sq[group].sqrt().item()), on_step=True,
            )
        group_lrs = self._group_lrs(optimizer)
        for group in self._ROUTING_GROUPS:
            lr_g = group_lrs[group]
            if lr_g == lr_g:  # not NaN → group present
                gl2 = float(grad_sq[group].sqrt().item())
                wn = float(weight_sq[group].sqrt().item())
                self.log(
                    f"train_mon_update_ratio_{group}",
                    lr_g * gl2 / (wn + 1e-12), on_step=True,
                )
        w_total_sq = sum(
            (weight_sq[g] for g in self._ROUTING_GROUPS),
            start=torch.zeros((), device=total_grad_sq.device),
        )
        all_finite = bool(torch.isfinite(total_grad_sq).item()) and bool(
            torch.isfinite(w_total_sq).item()
        )
        self.log("train_mon_nonfinite", 0.0 if all_finite else 1.0, on_step=True)
        self.log("train_mon_ema_weight_gap", self._ema_weight_gap(), on_step=True)
        b_noise, gn_signal, gn_var = self._grad_noise_scale(optimizer)
        if b_noise == b_noise:  # not NaN
            self.log("train_mon_grad_noise_scale", b_noise, on_step=True)
            self.log("train_mon_grad_noise_signal", gn_signal, on_step=True)
            self.log("train_mon_grad_noise_var", gn_var, on_step=True)
        try:
            clip_val = float(self.trainer.gradient_clip_val or 0.0)
        except (RuntimeError, AttributeError):
            clip_val = 0.0
        if clip_val > 0.0:
            self.log(
                "train_mon_grad_clipped",
                1.0 if grad_l2 > clip_val else 0.0, on_step=True,
            )
            self.log(
                "train_mon_grad_clip_scale",
                min(1.0, clip_val / (grad_l2 + 1e-12)), on_step=True,
            )
        self._maybe_log_true_update_ratio()

    def on_train_batch_start(
        self, batch: tp.Any, batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Stamp the step-compute start so ``on_train_batch_end`` can split the
        end-to-end ``step_time_s`` into host-wait vs in-step compute. Diagnostic
        only (``V14_STEP_TIMING``); a single ``perf_counter`` read otherwise."""
        if self._step_timing:
            self._last_batch_start_time = time.perf_counter()

    def on_train_batch_end(
        self, outputs: tp.Any, batch: tp.Any, batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Throughput + GPU-memory observability for the GH200-hr budget: per-step
        wall-clock (``step_time_s`` / ``steps_per_sec`` / ``samples_per_sec``) +
        the per-step peak allocation (``gpu_mem_gb``)."""
        now = time.perf_counter()
        prev = self._last_batch_end_time
        self._last_batch_end_time = now
        if prev is not None:
            dt = now - prev
            if dt > 0.0:
                self.log("train_mon_step_time_s", dt, on_step=True)
                self.log("train_mon_steps_per_sec", 1.0 / dt, on_step=True)
                bsz = _infer_batch_size(getattr(batch, "data", batch))
                if bsz is not None:
                    self.log("train_mon_samples_per_sec", bsz / dt, on_step=True)
                # Host-gap split (V14_STEP_TIMING): data_wait = gap before this
                # step's compute (dataloader + between-step host work); compute =
                # fwd/bwd/opt inside the step. data_wait + compute == step_time_s.
                start = self._last_batch_start_time
                if self._step_timing and start is not None:
                    data_wait = max(0.0, start - prev)
                    compute = max(0.0, now - start)
                    self.log("train_mon_data_wait_s", data_wait, on_step=True)
                    self.log("train_mon_compute_s", compute, on_step=True)
        if torch.cuda.is_available():
            self.log(
                "train_mon_gpu_mem_gb",
                torch.cuda.max_memory_allocated() / 1e9, on_step=True,
            )
            torch.cuda.reset_peak_memory_stats()
        if self._profiler is not None:
            self._profiler.step()

    # ---------------------------------------------- component-split profiler arm
    def on_train_start(self) -> None:
        """Build + start the component-split profiler when `V14_PROFILE_STEPS` is
        set (rank 0 only). `wait` skips the static-shape compile graphs; after a
        2-step warmup it records `active` steps, then `_dump_profile` fires with a
        CUDA-time table keyed by the forward's `record_function("v14/...")` ranges.
        Run the profiling probe at accumulate_grad_batches=1 so one batch == one
        opt-step (the schedule counts batches)."""
        if self._profile_spec is None:
            return
        if (torch.distributed.is_available() and torch.distributed.is_initialized()
                and torch.distributed.get_rank() != 0):
            return
        try:
            # Accept ":" as well as "," so the spec survives SLURM `sbatch
            # --export=VAR=a,b` (the CLI splits the export LIST on commas, which
            # silently truncates a comma-valued env to its first field). A colon
            # ("1450:20") passes through --export intact.
            spec = self._profile_spec.replace(":", ",")
            wait_s, active_s = (int(x) for x in spec.split(","))
        except ValueError:
            return
        # CPU-only by default: the per-step bubble is HOST-SIDE (GPU idle while the
        # CPU builds masks), so CPU self-time + the v14/mask_* ranges name it, and we
        # AVOID kineto's CUDA-activity (CUPTI) post-processing, which OOMs (large
        # `active`) or HANGS (chrome export) at cycle-end on aarch64 GH200 + torch
        # 2.10. Opt back into CUDA kernel timing with V14_PROFILE_CUDA=1 (at risk).
        activities = [torch.profiler.ProfilerActivity.CPU]
        if os.environ.get("V14_PROFILE_CUDA"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait_s, warmup=2, active=active_s, repeat=1),
            on_trace_ready=self._dump_profile,
            record_shapes=False, with_stack=False,
        )
        self._profiler.start()

    def _dump_profile(self, prof: tp.Any) -> None:
        # Default CPU-only: sort by self-CPU time so the host-side bubble (the
        # v14/mask_* ranges + their aten ops) sits at the top. Only sort by CUDA
        # time when V14_PROFILE_CUDA opted CUDA activity in.
        have_cuda = bool(os.environ.get("V14_PROFILE_CUDA"))
        sort_key = "self_cuda_time_total" if have_cuda else "self_cpu_time_total"
        label = "self CUDA time" if have_cuda else "self CPU time"
        table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
        print(f"=== V14 component profile ({label}) ===\n" + table, flush=True)
        # Chrome trace export is the kineto step that HANGS on aarch64 GH200 +
        # torch 2.10 — opt-in only (V14_PROFILE_TRACE=1), never on the default path.
        if not os.environ.get("V14_PROFILE_TRACE"):
            return
        out_dir = os.environ.get("EXCA_CACHE_FOLDER", ".")
        path = os.path.join(out_dir, f"v14_profile_trace_{os.getpid()}.json")
        try:
            prof.export_chrome_trace(path)
            print(f"=== V14 profile chrome trace -> {path} ===", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"=== V14 profile trace export failed: {exc} ===", flush=True)

    def on_train_end(self) -> None:
        if self._profiler is not None:
            try:
                self._profiler.stop()
            except Exception:  # noqa: BLE001
                pass
            self._profiler = None

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
