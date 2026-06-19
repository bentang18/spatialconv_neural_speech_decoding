"""``V14ConvergedExperiment`` — the Philosophy-B NeuralTrain experiment for the
converged v14 SSL arch.

Mirrors :class:`speech_decoding.experiments.v14_joint.V14JointExperiment` but is
intentionally THIN: the converged model
(:class:`speech_decoding.models.v14_converged.V14ConvergedSSL`) is self-contained
— it owns the frontend-only EMA teacher, the two paradigm-B predictors, and the
M2/M4 losses with the locked see-vs-predict scopes — so ``_build_brain_module``
only constructs the model from the config and wraps it in a
:class:`V14ConvergedBrainModule`. It does NOT touch the live ``V14JointExperiment``
/ ``V14JointBrainModule`` relaunch path.

The converged forward is intrinsically ONE joint phase (M2 + M4 in a single
student pass), so the inherited ``phase`` discriminator is INERT here (pinned to
``1`` for the v14-family default; the base ``_train_and_test`` never branches on
it). The #82 pretrain-leakage guard is ON (SSL must never train on a Neuroprobe
eval session); the anti-starvation cohort floor stays ``None`` until the
converged run corpus is named at launch — the hard leak check does not depend on
it.

Run numerics are NOT pre-committed: ``ema_tau`` and ``optim`` come from the
caller (dispatch). The mask-config scalars default to the FE-spec §8 LOCKED
structural starting points (the same values frozen in ``M2MaskConfig`` /
``M4MaskConfig``), which the §8.7 sister sweeps override.
"""

from __future__ import annotations

from typing import ClassVar

import pydantic

from speech_decoding.experiments.module import BrainModule
from speech_decoding.experiments.v14_experiment import V14Experiment, V14Phase


class V14ConvergedExperiment(V14Experiment):
    """Single-phase joint-SSL experiment around the self-contained
    :class:`V14ConvergedSSL`. The model computes its own ``L = λ_m2·L_M2 +
    λ_m4·L_M4``; the experiment supplies the data, the optimizer, the EMA τ, and
    the mask-sampling config."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # #82: SSL pretraining must never see a Neuroprobe off-limits eval session.
    enforces_pretrain_leakage_guard: ClassVar[bool] = True
    # Anti-starvation floor deferred to launch (the converged run corpus is Ben's
    # to name). The hard leak guard above is independent of this.
    pretrain_cohort_floor: ClassVar[tuple[int, int] | None] = None

    # The converged forward is intrinsically joint M2+M4 — ``phase`` is inert
    # (base ``_train_and_test`` never reads it). Pinned to the v14-family default.
    phase: V14Phase = 1

    # The peek key ``_build_brain_module`` uses to read the electrode count C.
    # The converged module reads the band keys directly from ``batch.data``; this
    # only sources ``n_in_channels`` (which ``V14Converged.build`` ignores).
    x_name: str | tuple[str, ...] = "electrode_tokens_slow"

    # EMA teacher momentum τ (mirrors V14JointExperiment / the ema.py lock). The
    # dispatch ``--ema-tau`` threads an override; the BrainModule revalidates.
    ema_tau: float = pydantic.Field(default=0.99925, gt=0.0, lt=1.0)

    # M2/M4 mask-sampling config (FE-spec §8 LOCKED structural starting points;
    # equal to the M2MaskConfig / M4MaskConfig frozen defaults). Exposed so the
    # §8.7 sisters (hg_start_rate coverage, beta_span redundancy) sweep from
    # dispatch. These are mask GEOMETRY, not run hyperparameters.
    m2_hg_start_rate: float = pydantic.Field(default=0.20, ge=0.0, le=1.0)
    m2_hg_span: int = pydantic.Field(default=3, ge=1)
    m2_beta_start_rate: float = pydantic.Field(default=0.30, ge=0.0, le=1.0)
    m2_beta_span: int = pydantic.Field(default=2, ge=1)
    m4_parcel_mask_ratio: float = pydantic.Field(default=0.20, gt=0.0, le=1.0)

    # Per-step mask RNG seed (own CPU generator in the module).
    mask_seed: int = 0

    # Heavy forward-tap monitor cadence (RankMe / coverage / input-stats — each
    # re-runs a no_grad extra forward; the post-latent RankMe tap is a DENSE
    # full-input latent pass over ~C·190 tokens). None ⇒ gate on log_every_n_steps
    # (pre-decouple behavior). Set it to fire the step-doubling extra forward
    # sparsely while loss curves stay per-step — measured ~2.1× step at cadence 1
    # (reports/converged_3stft_throughput_profile_2026_06_19.md). Diagnostic knob.
    monitor_every_n_steps: int | None = pydantic.Field(default=None, ge=1)

    # --- online linear probe (diagnostic; spec reports/online_probe_spec_2026_06_18.md)
    # OFF by default: registering the experiment can never perturb a live run. When
    # enabled, _callbacks() builds the probe dataset IN-WORKER from (seed, manifest)
    # so it is serialization-safe and deterministic across resume (spec §8); the
    # build is DCC-only (BT voltage + the 3STFT cache). cadence/k/batch tune the
    # callback; n_cap/seed feed the deterministic dataset build (Neuroprobe-Lite
    # parity N_cap=3500). These are diagnostic knobs, not run hyperparameters.
    online_probe_enabled: bool = False
    online_probe_cadence: int = pydantic.Field(default=1000, gt=0)
    online_probe_k_list: tuple[int, ...] = (2, 1)
    online_probe_batch_size: int = pydantic.Field(default=256, gt=0)
    online_probe_n_cap: int = pydantic.Field(default=3500, gt=0)
    online_probe_seed: int = 0

    def _callbacks(self):  # type: ignore[override]
        """Base callbacks (LR monitor, checkpoints) plus the diagnostic online
        probe when enabled. The probe append is the ONLY change vs the parent."""
        callbacks = super()._callbacks()
        callbacks.extend(self._online_probe_callbacks())
        return callbacks

    def _online_probe_callbacks(self) -> list:
        """``[]`` unless ``online_probe_enabled`` (spec §1, OFF by default). When on,
        build the probe dataset deterministically from ``(seed, n_cap, manifest)``
        — DCC-only (BT voltage + 3STFT cache) — and return a single configured,
        enabled :class:`OnlineLinearProbe`. Kept separate from ``_callbacks`` so the
        decision logic is unit-testable without a full trainer assembly.

        **Fail-soft (spec §1):** the probe is a DIAGNOSTIC; building its dataset must
        never abort a live training run. If the DCC-only extraction raises (a bad
        cache path, a missing band key, an empty corpus), log the error and register
        NO probe — the run proceeds exactly as if the probe were off."""
        if not self.online_probe_enabled:
            return []
        import logging

        from speech_decoding.experiments.online_probe import OnlineLinearProbe
        from speech_decoding.experiments.online_probe_dataset import build_probe_dataset

        # Reuse THIS run's data chain (study + wired 3STFT/support extractors +
        # bad_window_dir) re-segmented at the 1 s probe contract — so the probe
        # respects the run's CLIP layer and never hand-copies the data config.
        try:
            dataset = build_probe_dataset(
                self.data, n_cap=self.online_probe_n_cap, seed=self.online_probe_seed
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "online probe DISABLED: build_probe_dataset failed — training "
                "continues without the diagnostic probe (it never perturbs the run)."
            )
            return []
        return [
            OnlineLinearProbe(
                dataset=dataset,
                enabled=True,
                cadence=self.online_probe_cadence,
                k_list=self.online_probe_k_list,
                batch_size=self.online_probe_batch_size,
            )
        ]

    def _build_brain_module(self, train_loader) -> BrainModule:  # type: ignore[override]
        """Build the converged Lightning module: resolve the model from the
        config, then wrap it with the optimizer + EMA τ + mask config. Replaces
        the parent's CE-classifier ``BrainModule``."""
        from speech_decoding.experiments.v14_converged_module import (
            V14ConvergedBrainModule,
        )
        from speech_decoding.models.v14_converged import (
            M2MaskConfig,
            M4MaskConfig,
            V14ConvergedSSL,
        )

        batch = next(iter(train_loader))
        x = batch.data[self._input_tensor_name()]
        model = self.brain_model_config.build(
            n_in_channels=int(x.shape[1]),
            n_outputs=1,
        )
        if not isinstance(model, V14ConvergedSSL):
            raise RuntimeError(
                "V14ConvergedExperiment expected brain_model_config.build to "
                "return a V14ConvergedSSL (brain_model_config name "
                f"'V14Converged'); got {type(model).__name__}."
            )
        return V14ConvergedBrainModule(
            model=model,
            optim_config=self.optim,
            ema_tau=self.ema_tau,
            m2_cfg=M2MaskConfig(
                hg_start_rate=self.m2_hg_start_rate,
                hg_span=self.m2_hg_span,
                beta_start_rate=self.m2_beta_start_rate,
                beta_span=self.m2_beta_span,
            ),
            m4_cfg=M4MaskConfig(parcel_mask_ratio=self.m4_parcel_mask_ratio),
            mask_seed=self.mask_seed,
            wd_exclude_norms=self.wd_exclude_norms,
            monitor_every_n_steps=self.monitor_every_n_steps,
        )


__all__ = ["V14ConvergedExperiment"]
