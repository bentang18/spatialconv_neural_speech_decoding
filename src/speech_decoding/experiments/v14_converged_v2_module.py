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
  support              : (B, C, K) DKT one-hot → parcel_of_electrode = argmax → (C,)
  valid_mask           : (B, C) bool, optional → unmapped electrodes (False) are
                         DROPPED at ingest (v2 has no masked-electrode path); the
                         session-homogeneous batch drops the same rows for every clip
"""

from __future__ import annotations

import os
import time
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
    def _v2_inputs(self, data: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Map a batch dict to ``(lfs, hga, parcel_of_electrode (C,))``.

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
        ppe = support.argmax(dim=-1)                              # (B, C')
        if not torch.equal(ppe, ppe[:1].expand_as(ppe)):
            raise ValueError(
                "v2 requires a session-homogeneous batch: parcel_of_electrode must "
                "be constant across clips (use the same-session sampler)"
            )
        poe = ppe[0]                                              # (C',)
        return lfs, hga, poe

    # ------------------------------------------------------------- loss path
    def _step(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Pure loss path (testable without a trainer): ingest → sample masks →
        ``model.forward``. Returns the converged-v2 loss dict (loss + per-term
        diagnostics)."""
        lfs, hga, poe = self._v2_inputs(data)
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
        m2, tube = self.model.sample_masks(B, membership, bands, self._mask_gen)
        if mt:
            _t.append(time.perf_counter())
        m2, tube = m2.to(lfs.device), tube.to(lfs.device)
        if mt:
            _t.append(time.perf_counter())
        out = self.model(lfs, hga, poe, m2, tube, clip_len_s=self.clip_len_s)
        if mt:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _t.append(time.perf_counter())
            self._mask_timing_record(_t)
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
