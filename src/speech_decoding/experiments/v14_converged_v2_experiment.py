"""``V14ConvergedV2Experiment`` — the thin NeuralTrain experiment for the
self-contained converged-v2 SSL arch (2-band magnitude frontend + set-pool +
dual-depth M2/M4 + frozen EMA teacher).

Mirrors :class:`V14ConvergedExperiment` but is THINNER still: the v2 model
(:class:`speech_decoding.models.v14_converged_v2.V14ConvergedV2`) owns the EMA
teacher *momentum* (``ema_tau`` lives on the model config), the science knobs
(``k`` / ``tube_ratio`` / ``tie_lfs``), the shared-denominator loss, the static
drop-not-pad forward, AND mask sampling — so this experiment threads only the
SSL ``clip_len_s`` clock and the per-step ``mask_seed`` into the
:class:`V14ConvergedV2BrainModule`. Everything else (study chain, extractors,
sampler, segmenter, callbacks, trainer) is inherited from :class:`V14Experiment`.

The converged-v2 forward is intrinsically ONE joint phase (M2 + M4 in a single
student pass), so the inherited ``phase`` discriminator is INERT (pinned to ``1``
for the v14-family default; the base ``_train_and_test`` never branches on it).
The #82 pretrain-leakage guard is ON (SSL must never train on a Neuroprobe eval
session); the anti-starvation cohort floor stays ``None`` until the v2 run corpus
is named at launch — the hard leak check does not depend on it.

This subclasses :class:`V14Experiment` DIRECTLY (not ``V14ConvergedExperiment``)
to avoid inheriting the 3STFT-specific mask-geometry knobs and the 3STFT-cache-
coupled online-probe build; v2 carries its own minimal surface.
"""

from __future__ import annotations

from typing import ClassVar

import pydantic

from speech_decoding.experiments.module import BrainModule
from speech_decoding.experiments.v14_experiment import V14Experiment, V14Phase


class V14ConvergedV2Experiment(V14Experiment):
    """Single-phase joint-SSL experiment around the self-contained
    :class:`V14ConvergedV2`. The model computes its own dual-depth
    shared-denominator loss; the experiment supplies the data, the optimizer, the
    SSL clip-length clock, and the mask-sampling seed."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # #82: SSL pretraining must never see a Neuroprobe off-limits eval session.
    enforces_pretrain_leakage_guard: ClassVar[bool] = True
    # Anti-starvation floor deferred to launch (the v2 run corpus is Ben's to
    # name). The hard leak guard above is independent of this.
    pretrain_cohort_floor: ClassVar[tuple[int, int] | None] = None

    # The v2 forward is intrinsically joint M2+M4 — ``phase`` is inert (base
    # ``_train_and_test`` never reads it). Pinned to the v14-family default.
    phase: V14Phase = 1

    # The peek key ``_build_brain_module`` uses to read the electrode count C. The
    # v2 module reads both band keys (electrode_tokens_lfs / _hga) directly from
    # ``batch.data``; this only sources ``n_in_channels`` (which
    # ``V14ConvergedV2Net.build`` ignores — the band geometry is fixed by BANDS_V2,
    # not the electrode count). The LFS band's axis 1 is C, so it is the C-peek key.
    x_name: str | tuple[str, ...] = "electrode_tokens_lfs"

    # SSL clip-length clock (seconds). The v2 frontend is clip-len-parameterized
    # (RoPE tables built on the 5 s ``bands_for_clip_len(5.0)`` grid so they cover
    # the longest clock); the pretrain forward runs at the LOCKED 5 s SSL clip
    # (1 s is the eval clip, not the pretrain clip — the clip-len memory). The
    # dispatch ``--clip-len-s`` threads an override; the module revalidates by
    # re-deriving ``bands_for_clip_len(clip_len_s)`` each step.
    clip_len_s: float = pydantic.Field(default=5.0, gt=0.0)

    # Per-step mask RNG seed (own CPU generator in the module — CUDA randperm needs
    # a CPU generator; a shared seed across DDP ranks only lowers mask diversity,
    # never correctness).
    mask_seed: int = 0

    # SSL-health monitor cadence (steps). The module forwards with detached health
    # taps on `global_step % N == 0` and the SSLHealthMonitor callback logs the LEAN
    # diagnostic set on that cadence; off-cadence steps carry zero added cost.
    # Default 1: every-step intuition (rankme/feat_std/explained_var/GNS) is worth
    # the ~5-6% tap-forward + SVD overhead — the long-standing v14 convention.
    monitor_every_n_steps: int = 1

    def _build_brain_module(self, train_loader) -> BrainModule:  # type: ignore[override]
        """Build the converged-v2 Lightning module: resolve the model from the
        config (the C-peek is ignored by ``V14ConvergedV2Net.build``), fail loud if
        it is not a ``V14ConvergedV2``, then wrap it with the optimizer + clip-len
        clock + mask seed. Replaces the parent's CE-classifier ``BrainModule``."""
        from speech_decoding.experiments.v14_converged_v2_module import (
            V14ConvergedV2BrainModule,
        )
        from speech_decoding.models.v14_converged_v2 import V14ConvergedV2

        batch = next(iter(train_loader))
        x = batch.data[self._input_tensor_name()]
        model = self.brain_model_config.build(
            n_in_channels=int(x.shape[1]),
            n_outputs=1,
        )
        if not isinstance(model, V14ConvergedV2):
            raise RuntimeError(
                "V14ConvergedV2Experiment expected brain_model_config.build to "
                "return a V14ConvergedV2 (brain_model_config name "
                f"'V14ConvergedV2Net'); got {type(model).__name__}."
            )
        return V14ConvergedV2BrainModule(
            model=model,
            optim_config=self.optim,
            clip_len_s=self.clip_len_s,
            mask_seed=self.mask_seed,
            wd_exclude_norms=self.wd_exclude_norms,
            monitor_every_n_steps=self.monitor_every_n_steps,
        )


__all__ = ["V14ConvergedV2Experiment"]
