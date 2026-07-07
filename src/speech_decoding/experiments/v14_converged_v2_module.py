"""Lightning shell around the self-contained :class:`V14ConvergedV2` SSL model.

The v2 model OWNS the EMA teacher, the dual-depth shared-denominator loss, the
static drop-not-pad forward, and mask sampling — so this shell is THIN: ingest
the 2-band batch → sample masks (CPU) → ``model.forward`` → log → return loss,
with the EMA tick in ``on_before_zero_grad`` (one update per optimiser step, so
``accumulate_grad_batches=K`` gives τ not τ^K — the #46 lesson).

Speedups (env-gated by ``dispatch_v14``, byte-identical when unset ⇒ unit tests /
eager runs unaffected): ``V14_COMPILE`` ``torch.compile``s the whole forward
(OptimizedModule in a plain dict, never a submodule, rebuilt lazily after an
un-pickle) and ``V14_SDPA_BACKEND=cudnn`` forces the cuDNN-first SDPA priority for
the forward. v2's forward is static-shape BY CONSTRUCTION (drop-not-pad) and every
parameter is used every step (M2+M4 predictors always fire, teacher always runs),
so the compile is simpler than the 3STFT one — but ``dynamic=False``
(``--no-compile-dynamic``) is still REQUIRED: the data-dependent drop-not-pad
gathers storm dynamo's symbolic solver under ``dynamic=True``/``None``.

Batch contract (the v2 cache, session-homogeneous via the same-session sampler):
  electrode_tokens_lfs : (B, C, 28, T_lfs)   |STFT| magnitude, robust-z'd
  electrode_tokens_hga : (B, C,  7, T_hga)   |STFT| magnitude
  support              : (B, C, K) DKT one-hot → parcel_of_electrode = argmax → (C,)
  valid_mask           : (B, C) bool, optional → unmapped electrodes (False) are
                         DROPPED at ingest (v2 has no masked-electrode path); the
                         session-homogeneous batch drops the same rows for every clip
"""

from __future__ import annotations

import contextlib
import os
import time
import typing as tp

import torch
from lightning import pytorch as pl
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from neuraltrain.optimizers import BaseOptimizer

from speech_decoding.experiments.optim_param_groups import maybe_split_no_decay
from speech_decoding.models.v14_converged_v2 import (
    V14ConvergedV2,
    active_parcels,
    bands_for_clip_len,
)

# cuDNN-first SDPA priority (cuDNN → mem-efficient → math fallback), forced only
# when V14_SDPA_BACKEND=cudnn; otherwise the default dispatcher governs.
_CUDNN_SDPA_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _sdpa_forward_context(name: str | None) -> tp.ContextManager[None]:
    """Select the SDPA kernel for the wrapped v2 forward. ``"cudnn"`` forces the
    cuDNN-first priority (Hopper sm90 kernel; backward inherits the forward's
    kernel, so wrapping the forward suffices) around the WHOLE forward — v2 has no
    internal latent scoping, so a whole-forward force is the v2 analogue of the
    3STFT ``"cudnn"``. All other names / unset ⇒ ``nullcontext`` (the auto-resolved
    ``"cudnn_latent"`` too — v2 has nowhere to scope it). The recon head uses an
    explicit softmax, NOT SDPA, so it is unaffected either way. Identical math."""
    key = (name or "").strip().lower()
    if key == "cudnn" and torch.cuda.is_available():
        return sdpa_kernel(_CUDNN_SDPA_PRIORITY, set_priority=True)
    return contextlib.nullcontext()


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
        monitor_every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        # ── Science-neutral speedups (env-gated by dispatch_v14; byte-identical
        # when unset ⇒ unit tests / eager 1-GPU runs unaffected) ──
        # SDPA preference read once; applied per-forward via the context in
        # _call_model. Only frontend/latent/predictor attentions are affected
        # (the recon head uses an explicit softmax, not SDPA).
        self._sdpa_backend_name = os.environ.get("V14_SDPA_BACKEND")
        # torch.compile override: the OptimizedModule lives in a PLAIN dict (never a
        # submodule ⇒ no `_orig_mod.` ckpt prefix, no double param/EMA registration)
        # and is rebuilt lazily after an un-pickle (the exca/submitit job-pickle
        # drops it in __getstate__ — its dynamo guards carry unpicklable weakrefs).
        self._compiled_fwd: dict[str, tp.Callable[..., tp.Any]] = {}
        self._compile_spec: tuple[str, bool | None] | None = None
        _compile_flag = os.environ.get("V14_COMPILE", "").strip().lower()
        if _compile_flag not in ("", "0", "false", "no", "off"):
            _mode = os.environ.get("V14_COMPILE_MODE") or "default"
            # Three states: True = symbolic (compile once), False = fully static
            # (recompile per concrete shape, no symbolic reasoning), unset = torch's
            # automatic-dynamic. v2's data-dependent gathers storm the symbolic
            # solver under True/None, so --no-compile-dynamic ("0") ⇒ False is the
            # working setting; keep the three-state map so it is selectable.
            _dyn = os.environ.get("V14_COMPILE_DYNAMIC", "").strip().lower()
            if _dyn in ("1", "true", "yes", "on"):
                _dynamic: bool | None = True
            elif _dyn in ("0", "false", "no", "off"):
                _dynamic = False
            else:
                _dynamic = None
            self._compile_spec = (_mode, _dynamic)
            self._build_compiled_model()
        self.optim_config = optim_config
        self.clip_len_s = float(clip_len_s)
        self._wd_exclude_norms = wd_exclude_norms
        # SSL-health monitor cadence (steps). On a due step the forward runs with
        # ``return_taps=True`` and the detached tap subdict is stashed on
        # ``self._last_taps`` for the SSLHealthMonitor callback (Family B); off a
        # due step the forward is byte-identical to the no-tap path.
        self._monitor_every_n_steps = max(int(monitor_every_n_steps), 1)
        self._last_taps: dict[str, Tensor] | None = None
        self._last_batch_size = 0  # per-rank micro-batch, read by the monitor's B_eff
        # Own CPU generator: the per-clip mask draws (CUDA randperm needs a CPU
        # generator) are reproducible + independent of the global RNG. A shared
        # seed across DDP ranks only lowers mask diversity, not correctness — the
        # model never assumes distinct masks per rank.
        self._mask_seed = int(mask_seed)
        self._mask_gen = torch.Generator()
        self._mask_gen.manual_seed(self._mask_seed)
        # Diagnostic step timing (env-gated, OFF in production ⇒ zero added logs,
        # uid-neutral — read from env, not a pydantic field). `V14_STEP_TIMING`
        # logs a NON-OVERLAPPING per-step split to wandb ⇒ median sec/step:
        #   fetch_s    = host gap before the forward (pure dataloader wait +
        #                between-step host overhead) — measured from the END of the
        #                FULL prior step (`on_train_batch_end`), so it does NOT
        #                absorb backward/opt the way an end-of-forward stamp would.
        #   compute_s  = forward only (cuda.sync'd ⇒ real GPU wall).
        #   post_fwd_s = backward + optimizer.step + EMA tick + grad-clip
        #                (cuda.sync'd at `on_train_batch_end`).
        #   step_time_s = fetch_s + compute_s + post_fwd_s (true per-step wall).
        # `V14_MASK_TIMING=<N>` prints a median/p10/p90 ms table over every N train
        # steps splitting the in-`_step` work into mask_sample(CPU) / mask_h2d /
        # forward — names where the step goes WITHOUT torch.profiler/kineto (which
        # OOMs/hangs on aarch64 GH200 + torch 2.10). v2 masks are vectorized, so
        # mask_sample should be small (no 3STFT 455ms per-electrode-loop bubble).
        # Rank-0 + training only.
        self._step_timing = os.environ.get("V14_STEP_TIMING") is not None
        self._last_batch_end_time: float | None = None   # END of prior FULL step
        self._fwd_end_time: float | None = None           # end of this step's fwd
        self._fetch_s = 0.0
        self._compute_s = 0.0
        _mt = os.environ.get("V14_MASK_TIMING")
        self._mask_timing_every = 0
        if _mt is not None:
            try:
                _mt_n = int(_mt)
            except ValueError:
                _mt_n = 50
            self._mask_timing_every = _mt_n if _mt_n > 0 else 50
        self._mask_timing_buf: dict[str, list[float]] = {
            "sample": [], "h2d": [], "fwd": [],
        }

    # ----------------------------------------------------------- batch ingest
    def _v2_inputs(
        self, data: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Map a batch dict to ``(lfs, hga, parcel_of_electrode (C,), coords (C,3)|None)``.

        ``parcel_of_electrode = support.argmax(-1)`` (DKT support is one-hot, so
        argmax is the exact hard parcel id). The v2 model is session-homogeneous —
        one ``(C,)`` parcel vector for the whole batch — so the per-clip support
        must be CONSTANT across clips; fail loud otherwise.

        Unmapped electrodes (no DKT parcel ⇒ all-zero ``support`` row ⇒ a spurious
        ``argmax``-0 parcel id) are marked False by ``ElectrodeValidMask``. v2 has
        no masked-electrode path (unlike the 3STFT converged forward, which carries
        an ``electrode_mask`` and biases those rows to ``-inf``), so it DROPS them
        at ingest. Because the batch is session-homogeneous the same rows are
        invalid for every clip — assert that, then drop one ``(C,)`` row set,
        preserving the uniform-C contract. shaft-CAR is untouched: it ran in the
        loader over the post-STATIC montage (which still includes these contacts)
        before the cache; this drop is model-side, after the cache."""
        lfs = data["electrode_tokens_lfs"]
        hga = data["electrode_tokens_hga"]
        support = data["support"]
        coords = data.get("electrode_coords")                    # (B,C,3) native-RAS
        if coords is not None and coords.shape[1] != support.shape[1]:
            raise ValueError(
                f"electrode_coords C={coords.shape[1]} must align with support "
                f"C={support.shape[1]} (same voltage order) — alignment invariant"
            )
        valid = data.get("valid_mask")
        if valid is not None:
            valid = valid.to(torch.bool)
            if not torch.equal(valid, valid[:1].expand_as(valid)):
                raise ValueError(
                    "v2 requires a session-homogeneous batch: valid_mask must be "
                    "constant across clips (use the same-session sampler)"
                )
            keep = valid[0]                                       # (C,)
            if not bool(keep.all()):
                lfs = lfs[:, keep]
                hga = hga[:, keep]
                support = support[:, keep]
                if coords is not None:
                    coords = coords[:, keep]
        ppe = support.argmax(dim=-1)                              # (B, C')
        if not torch.equal(ppe, ppe[:1].expand_as(ppe)):
            raise ValueError(
                "v2 requires a session-homogeneous batch: parcel_of_electrode must "
                "be constant across clips (use the same-session sampler)"
            )
        poe = ppe[0]                                              # (C',)
        # Coords are session-static ⇒ constant across clips; take one row set.
        coords1: Tensor | None = None
        if coords is not None:
            coords1 = coords[0]                                   # (C',3)
        return lfs, hga, poe, coords1

    def _build_compiled_model(self) -> None:
        """Compile ``self.model`` per ``_compile_spec`` into the plain
        ``_compiled_fwd`` dict. Re-applies the global dynamo switches because
        un-pickling bypasses ``__init__`` — the in-allocation-ddp / submitit
        job-pickle drops the OptimizedModule (``__getstate__``) and ``_call_model``
        rebuilds it on the first forward after the un-pickle."""
        if self._compile_spec is None:
            return
        _mode, _dynamic = self._compile_spec
        import torch._dynamo as _dynamo_mod

        # DDPOptimizer OFF (default): the bucket-split optimizer reorders the graph
        # and can desync a find_unused DDP run — compiling ONE graph avoids it (cost:
        # lost allreduce/compute overlap, negligible for this small model on
        # single-node 4-GPU DDP). Env-tunable, mirrors the 3STFT module.
        _ddp_opt = os.environ.get("V14_COMPILE_DDP_OPTIMIZER", "").strip().lower()
        _dynamo_mod.config.optimize_ddp = _ddp_opt in ("1", "true", "yes", "on")
        # Recompile cap: dynamic=False compiles one concrete graph per distinct
        # session geometry (~one per BT session); the default cap (8) is below the
        # corpus session count, so dynamo would silently fall back to eager past the
        # 8th shape without a raise. torch 2.10 renamed cache_size_limit →
        # recompile_limit (alias); set both.
        _cap = int(os.environ.get("V14_COMPILE_CACHE_LIMIT", "64"))
        for _attr in ("cache_size_limit", "recompile_limit"):
            if hasattr(_dynamo_mod.config, _attr):
                setattr(_dynamo_mod.config, _attr, _cap)
        self._compiled_fwd["model"] = torch.compile(
            self.model, mode=_mode, dynamic=_dynamic,
        )

    def __getstate__(self) -> dict[str, tp.Any]:
        """cloudpickle-safe when compiled: the OptimizedModule carries
        ``torch._dynamo`` guard weakrefs cloudpickle cannot serialize. Drop it (a
        copy — the in-process original keeps its compiled forward, so 1-GPU / cluster
        runs that never round-trip are unaffected); ``_compile_spec`` survives, so
        ``_call_model`` rebuilds it lazily on the first forward after the un-pickle."""
        state = self.__dict__.copy()
        state["_compiled_fwd"] = {}
        return state

    def _call_model(self, *args: tp.Any, **kwargs: tp.Any) -> dict[str, Tensor]:
        """Forward through the compiled override when present (rebuilding lazily
        after an un-pickle), else eager — byte-identical when ``V14_COMPILE`` is
        unset. Wraps the call in the SDPA context so ``--sdpa-backend cudnn`` forces
        the cuDNN-first priority for this forward (backward inherits the kernel)."""
        if self._compile_spec is not None and "model" not in self._compiled_fwd:
            self._build_compiled_model()
        with _sdpa_forward_context(self._sdpa_backend_name):
            return self._compiled_fwd.get("model", self.model)(*args, **kwargs)

    def _monitor_due(self) -> bool:
        """Monitor cadence: ``global_step % N == 0``. ``global_step`` is 0 (and may
        raise) when no trainer is attached (unit tests calling ``_step`` directly),
        so guard it — a no-trainer call lands on step 0 ⇒ due."""
        try:
            step = int(self.global_step)
        except (RuntimeError, AttributeError):
            step = 0
        return step % self._monitor_every_n_steps == 0

    def _context_lambda(
        self, start: int | None = None, steps: int | None = None
    ) -> float:
        """Delayed-ramp context-loss coefficient: λ(t) = 0 for t < ``S``, then linear
        0→``context_lambda`` over the next ``W`` = ``context_warmup_steps`` steps (S =
        ``context_warmup_start_step``). The 0-hold lets the masked pretext establish
        before context is introduced (avoids the trivial-copy basin). ``start``/``steps``
        override the shared schedule (used for the decoupled M4 ramp); None ⇒ shared.
        Read from ``global_step`` in eager code so λ never enters the compiled forward
        (a per-step scalar would thrash dynamo guards). No trainer / step 0 ⇒ 0.0."""
        try:
            step = int(self.global_step)
        except (RuntimeError, AttributeError):
            step = 0
        s = self.model.cfg.context_warmup_start_step if start is None else start
        w = self.model.cfg.context_warmup_steps if steps is None else steps
        if w > 0:
            frac = min(max(step - s, 0) / w, 1.0)
        else:
            frac = 1.0 if step >= s else 0.0
        return self.model.cfg.context_lambda * frac

    # ------------------------------------------------------------- loss path
    def _step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Pure loss path (testable without a trainer): ingest → sample masks →
        ``model.forward``. Returns the converged-v2 loss dict (loss + per-term
        diagnostics)."""
        lfs, hga, poe, coords = self._v2_inputs(data)
        bands = bands_for_clip_len(self.clip_len_s)
        # Kineto-free per-section wall timing (V14_MASK_TIMING). Armed + training
        # only ⇒ off ⇒ not a single extra call. 4 perf_counter stamps split the
        # in-`_step` work into mask_sample(CPU) / mask_h2d / forward, with ONE
        # post-forward cuda.sync so `forward` is real GPU time, not async-dispatch.
        mt = bool(self._mask_timing_every) and self.training
        _t: list[float] = [time.perf_counter()] if mt else []
        # Masks sampled on CPU (CUDA randperm needs a CPU generator), membership
        # derived from the SAME deterministic active_parcels ⇒ matches the model's
        # internal session layout (unique-sorted labels, identical P + ordering).
        _, membership = active_parcels(poe.cpu())
        B = lfs.shape[0]
        self._last_batch_size = B  # per-rank micro-batch, for the monitor's B_eff
        m2, tube = self.model.sample_masks(B, membership, bands, self._mask_gen)
        # Run-B: whole-electrode drop (dataloader-side, count-uniform per parcel) +
        # native-RAS coords required for the masked-electrode reconstruction head.
        drop: Tensor | None = None
        m2_elec: Tensor | None = None
        if self.model.cfg.run_b:
            if coords is None:
                raise ValueError(
                    "Run-B requires electrode_coords in the batch (register "
                    "V14ElectrodeCoordsExtractor); none present"
                )
            drop = self.model.sample_drop(B, membership, self._mask_gen)
            if self.model.cfg.m2_hetero:
                m2_elec = self.model.sample_m2_hetero(
                    B, poe.shape[0], bands, self._mask_gen
                )
        if mt:
            _t.append(time.perf_counter())
        m2, tube = m2.to(lfs.device), tube.to(lfs.device)
        if coords is not None:
            coords = coords.to(lfs.device)
        if drop is not None:
            drop = drop.to(lfs.device)
        if m2_elec is not None:
            m2_elec = m2_elec.to(lfs.device)
        if mt:
            _t.append(time.perf_counter())
        # On a monitor-cadence training step request detached health taps; stash
        # them for the callback and STRIP the ``_tap_*`` keys before the dict flows
        # into `_log_losses` (so they aren't logged as losses). Off-cadence ⇒ no
        # taps requested ⇒ the forward is byte-identical to the no-tap path.
        want_taps = self.training and self._monitor_due()
        out = self._call_model(
            lfs, hga, poe, m2, tube,
            clip_len_s=self.clip_len_s, coords=coords, drop_mask=drop,
            m2_elec_mask=m2_elec, return_taps=want_taps,
        )
        if want_taps:
            self._last_taps = {
                k: out.pop(k) for k in list(out) if k.startswith("_tap_")
            }
        else:
            self._last_taps = None
        if mt:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t.append(time.perf_counter())
            self._mask_timing_record(_t)
        # Context loss (m4_recon_m3): the forward returns the per-head context terms
        # as diagnostics; fold them into the training loss here with the ramped λ(t),
        # so λ stays out of the compiled graph. Grad flows through the compiled
        # context outputs. Absent terms ⇒ no-op.
        if self.model.cfg.context_loss:
            # M2 + MELEC context ride the shared ramp; M4 context can ride a SEPARATE,
            # later ramp (sentinel -1 ⇒ inherit the shared schedule, byte-identical).
            lam = self._context_lambda()
            m4_s = self.model.cfg.m4_context_warmup_start_step
            m4_w = self.model.cfg.m4_context_warmup_steps
            lam_m4 = self._context_lambda(
                start=None if m4_s < 0 else m4_s,
                steps=None if m4_w < 0 else m4_w,
            )
            ctx = out["loss"].new_zeros(())
            for key in ("loss_m2_ctx", "loss_melec_ctx"):
                if key in out:
                    ctx = ctx + out[key]
            out["loss"] = out["loss"] + lam * ctx
            if "loss_m4_ctx" in out:
                out["loss"] = out["loss"] + lam_m4 * out["loss_m4_ctx"]
        return out

    def _mask_timing_record(self, t: list[float]) -> None:
        """Accumulate one step's per-section walls (4 perf_counter stamps: start,
        +sample, +h2d, +forward); flush a median/p10/p90 ms table every
        ``_mask_timing_every`` steps. Prints on rank 0; clears on every rank so
        non-zero ranks stay bounded."""
        if len(t) < 4:
            return
        buf = self._mask_timing_buf
        buf["sample"].append((t[1] - t[0]) * 1e3)
        buf["h2d"].append((t[2] - t[1]) * 1e3)
        buf["fwd"].append((t[3] - t[2]) * 1e3)
        if len(buf["sample"]) < self._mask_timing_every:
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
            n = len(buf["sample"])
            total = [a + b + c for a, b, c in
                     zip(buf["sample"], buf["h2d"], buf["fwd"])]
            rows = [
                ("mask_sample(CPU)", buf["sample"]),
                ("mask_h2d", buf["h2d"]),
                ("forward", buf["fwd"]),
                ("step sum(3)", total),
            ]
            print(f"\n[V14_MASK_TIMING v2] median/p10/p90/mean ms over {n} steps:")
            for name, xs in rows:
                med, p10, p90, mean = _sm(xs)
                print(f"  {name:18s} med={med:8.2f} p10={p10:8.2f} "
                      f"p90={p90:8.2f} mean={mean:8.2f}")
        for k in buf:
            buf[k].clear()

    def training_step(self, batch: tp.Any, batch_idx: int) -> Tensor:  # noqa: ARG002
        # Non-overlapping split (V14_STEP_TIMING): fetch_s = gap since the prior
        # FULL step ended (`on_train_batch_end` stamp ⇒ pure dataloader + host
        # wait, NOT backward); compute_s = forward only (cuda.sync ⇒ real GPU
        # wall). post_fwd_s + step_time_s are logged in `on_train_batch_end` once
        # backward+opt+EMA have run. Armed only.
        st = self._step_timing
        if st:
            now = time.perf_counter()
            self._fetch_s = (
                now - self._last_batch_end_time
                if self._last_batch_end_time is not None else 0.0
            )
            t0 = now
        out = self._step(batch.data)
        self._log_losses(out, "train", on_step=True, on_epoch=False)
        if st:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            self._compute_s = end - t0
            self._fwd_end_time = end
        return out["loss"]

    def on_train_batch_end(
        self, outputs: tp.Any, batch: tp.Any, batch_idx: int  # noqa: ARG002
    ) -> None:
        # Close the step AFTER Lightning has run backward + optimizer.step + the
        # EMA tick (on_before_zero_grad). A cuda.sync here makes post_fwd_s the
        # real backward+opt GPU wall, and stamping `_last_batch_end_time` at the
        # FULL-step end means the next step's fetch_s is pure dataloader/host wait.
        if not self._step_timing or self._fwd_end_time is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bend = time.perf_counter()
        post_fwd = bend - self._fwd_end_time
        step_time = self._fetch_s + self._compute_s + post_fwd
        self.log("fetch_s", self._fetch_s, on_step=True, on_epoch=False)
        self.log("compute_s", self._compute_s, on_step=True, on_epoch=False)
        self.log("post_fwd_s", post_fwd, on_step=True, on_epoch=False)
        self.log("step_time_s", step_time, on_step=True, on_epoch=False)
        self._last_batch_end_time = bend

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
    def configure_callbacks(self) -> list[pl.Callback]:  # type: ignore[override]
        """Attach the LEAN SSL-health monitor. Lightning MERGES these with any
        Trainer-level callbacks (module callbacks take precedence on conflict)."""
        from speech_decoding.experiments.monitors.ssl_health import SSLHealthMonitor

        return [SSLHealthMonitor(every_n_steps=self._monitor_every_n_steps)]

    def on_before_zero_grad(self, optimizer: tp.Any) -> None:  # noqa: ARG002
        """EMA tick once per optimiser step (after step, before zero_grad)."""
        self.model.ema_step()

    def _estimated_total_steps(self) -> int | None:
        try:
            return int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            return None

    def configure_optimizers(self):  # type: ignore[override]
        # Only the trainable params (the EMA teacher is requires_grad False, so it
        # is excluded by construction).
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        params: tp.Any = [{"params": trainable}]
        params = maybe_split_no_decay(
            params, modules=(self.model,),
            optim_config=self.optim_config, exclude=self._wd_exclude_norms,
        )
        # Thread the training horizon into the scheduler build. Without it,
        # WarmupCosine.build() gets total_steps=None and silently degrades to a
        # CONSTANT LR (warmup_steps clamped to 0) — i.e. --warmup-steps becomes a
        # no-op. This dropped-total_steps was the cause of the flat-1e-2 Run-B
        # smoke (grad storm at the pre-decollapse window with zero warmup cushion).
        total_steps = self._estimated_total_steps()
        if total_steps is None:
            return self.optim_config.build(params)
        try:
            return self.optim_config.build(params, total_steps=total_steps)
        except TypeError:
            return self.optim_config.build(params)


__all__ = ["V14ConvergedV2BrainModule"]
