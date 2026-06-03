"""B36 WS-G — Phase-4 readout module + experiment (B35 frozen-PMA → mean → Linear).

Phase 4 is the frozen-encoder per-task probe (B35 lock 2026-05-31,
[[project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31]]). The SSL+distill
pretrained encoder AND the P3-trained parcel-collapse PMA are both FROZEN; the
only trainable parameter is the per-task readout ``Linear``. The chain
(:class:`~speech_decoding.models.v14_encoder.V14ParcelPerceiverWithHead` +
:class:`~speech_decoding.models.v14_encoder.V14PmaReadout`)::

    encoder(...)              (B, L, T_p, d)   frozen, full-input
      → frozen P3-PMA k=1     (B, T_p, d)      key-masked by latent_valid
      → mean-over-time        (B, d)           B35 default
      → Linear(d, n_classes)  (B, n_classes)   the ONLY trainable P4 params

This module is the *trainer* the plan's WS-G calls out as missing: the B35
readout module already exists and is correct, but the supervised ``BrainModule``
trains the whole encoder unfrozen with a random PMA and no checkpoint. This
module fixes exactly that, in three ways:

1. **load** the P3 checkpoint (encoder + PMA strict; the P3-only
   ``StudentWhisperProjector`` keys are dropped — E4);
2. **freeze** the encoder (``requires_grad_(False)``); the PMA arrives frozen
   from :class:`V14PmaReadout`;
3. **optimize the readout ``Linear`` only** — the param group is the trainable
   set, not ``self.parameters()``.

Cross-phase handoff (E4): :meth:`load_transferable_state` reads a P3
``transferable_state`` (``{encoder, pma, projector}``), loads the encoder/PMA
intersection, and drops the projector. :meth:`transferable_state` exports only
the frozen foundation (``{encoder, pma}``); the per-task ``Linear`` is re-fit
per Neuroprobe task and is NOT a transferable weight.

Eval (G3): the test epoch accumulates logits over the FULL probe test set and
emits the Neuroprobe metric once — micro-accuracy always, plus binary AUROC for
2-class tasks (rank-based, over all samples, NOT a per-batch average). The
CrossSubject leave-2-subjects-out split + the BP20 no-electrode-overlap guard
are :func:`leave_two_subjects_out` / :func:`assert_no_electrode_overlap`.
"""

from __future__ import annotations

import typing as tp

import pydantic
import torch
from lightning import pytorch as pl
from torch import Tensor, nn
from torchmetrics.functional.classification import binary_auroc, multiclass_accuracy

from neuraltrain.optimizers import BaseOptimizer

from speech_decoding.experiments.v14_experiment import V14Experiment
from speech_decoding.models.v14_encoder import (
    V14ParcelPerceiverWithHead,
    V14PmaReadout,
)

_Electrode = tp.Tuple[tp.Hashable, tp.Hashable]


# ---------------------------------------------------------------------------
# CrossSubject leave-2-subjects-out split + BP20 electrode-overlap guard (G3)
# ---------------------------------------------------------------------------


def leave_two_subjects_out(
    subject_ids: tp.Sequence[tp.Hashable],
    held_out: tuple[tp.Hashable, tp.Hashable],
) -> tuple[list[int], list[int]]:
    """CrossSubject leave-2-subjects-out split of clip indices.

    Clips whose subject is in ``held_out`` form the test pool; the rest train.
    Neuroprobe's CrossSubject protocol trains on the remaining cohort and tests
    on the held-out subjects, so the two pools share no subject — and hence (see
    :func:`assert_no_electrode_overlap`) no electrode. Raises if the two
    held-out subjects are not distinct or are absent from the cohort.
    """
    a, b = held_out
    if a == b:
        raise ValueError(
            f"leave-2-out needs two DISTINCT held-out subjects; got {held_out!r}."
        )
    present = set(subject_ids)
    missing = [s for s in held_out if s not in present]
    if missing:
        raise ValueError(
            f"held-out subject(s) {missing!r} absent from the cohort "
            f"{sorted(map(repr, present))!r}."
        )
    held = set(held_out)
    train = [i for i, s in enumerate(subject_ids) if s not in held]
    test = [i for i, s in enumerate(subject_ids) if s in held]
    return train, test


def assert_no_electrode_overlap(
    subject_ids: tp.Sequence[tp.Hashable],
    electrode_ids: tp.Sequence[tp.Sequence[tp.Hashable]],
    train_idx: tp.Sequence[int],
    test_idx: tp.Sequence[int],
) -> tuple[set[_Electrode], set[_Electrode]]:
    """BP20: a CrossSubject probe must not share an electrode train↔test.

    Electrode identity is the ``(subject_id, electrode_id)`` pair — the same
    local contact index on two subjects is two different electrodes, so the
    subject id is part of the key. Returns the (train, test) electrode sets for
    logging; raises ``AssertionError`` on any overlap.
    """

    def pool(indices: tp.Sequence[int]) -> set[_Electrode]:
        return {(subject_ids[i], e) for i in indices for e in electrode_ids[i]}

    train_e = pool(train_idx)
    test_e = pool(test_idx)
    overlap = train_e & test_e
    if overlap:
        raise AssertionError(
            f"BP20 violated: {len(overlap)} electrode(s) shared between the "
            "CrossSubject probe train and test pools (e.g. "
            f"{sorted(map(repr, overlap))[:3]})."
        )
    return train_e, test_e


# ---------------------------------------------------------------------------
# Phase-4 Lightning module
# ---------------------------------------------------------------------------


class V14Phase4ReadoutModule(pl.LightningModule):
    """Phase-4 frozen-encoder probe Lightning module (WS-G, B35).

    Wraps a built :class:`V14ParcelPerceiverWithHead` whose readout is a
    :class:`V14PmaReadout`. Freezes the encoder + PMA on construction; only the
    readout ``Linear`` (and the ``timeattn`` score vector for that sister)
    train. Loads the P3 encoder+PMA via :meth:`load_transferable_state`.
    """

    def __init__(
        self,
        *,
        model: V14ParcelPerceiverWithHead,
        loss: nn.Module,
        optim_config: BaseOptimizer,
        x_name: str | tuple[str, ...] = "input",
        y_name: str = "target",
    ) -> None:
        super().__init__()
        if not isinstance(model.readout, V14PmaReadout):
            raise TypeError(
                "V14Phase4ReadoutModule requires a V14PmaReadout (the B35 "
                "frozen-PMA → temporal → Linear head; the P3-PMA loads into "
                f"readout.pma). Got readout={type(model.readout).__name__}. The "
                "meanpool / attentive B34 sisters have no PMA to load — run them "
                "through the supervised BrainModule, not this P4 module."
            )
        self.model = model
        self.loss = loss
        self.optim_config = optim_config
        self.x_name = x_name
        self.y_name = y_name
        self._n_classes = int(model.readout.classifier.out_features)
        self._apply_freeze()
        # Test-epoch accumulators: the Neuroprobe metric is computed once over
        # the full probe test set (a per-batch AUROC averaged across batches is
        # not the dataset AUROC), so logits/targets are gathered here.
        self._test_logits: list[Tensor] = []
        self._test_targets: list[Tensor] = []

    # ------------------------------------------------------------------
    # Freeze (encoder + PMA); only the readout Linear trains
    # ------------------------------------------------------------------

    def _apply_freeze(self) -> None:
        """Freeze the encoder + PMA. Idempotent: :class:`V14PmaReadout` already
        froze the PMA, and ``load_state_dict`` preserves ``requires_grad``, so
        this re-asserts the P4 contract after every weight load."""
        self.model.encoder.requires_grad_(False)
        self.model.readout.pma.requires_grad_(False)

    def _trainable_parameters(self) -> list[nn.Parameter]:
        """The P4 trainable set: with the encoder + PMA frozen this is exactly
        the readout ``Linear`` (plus the ``timeattn`` score vector for that
        sister) — the only params the optimizer ever sees."""
        return [p for p in self.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Cross-phase weight handoff (E4)
    # ------------------------------------------------------------------

    def transferable_state(self) -> dict[str, dict[str, Tensor]]:
        """The frozen foundation only (E4): ``{encoder, pma}``.

        P4 is terminal in the pretrain chain, but the WS-E driver snapshots
        every phase, so this is exercised. The per-task ``Linear`` is NOT
        exported — it is re-fit per Neuroprobe task, not a foundation weight —
        which keeps save/load symmetric on ``{encoder, pma}``.
        """
        return {
            "encoder": self.model.encoder.state_dict(),
            "pma": self.model.readout.pma.state_dict(),
        }

    def load_transferable_state(
        self, state: dict[str, dict[str, Tensor]], *, strict: bool = True,
    ) -> None:
        """Warm-start from a P3 :meth:`transferable_state` (E4).

        Loads ``encoder`` (required, strict) and ``pma`` (the P3-trained query,
        strict) into ``readout.pma``; the P3-only ``projector`` key is silently
        dropped (P4 owns no projector — E4 "projector keys dropped"). The
        encoder freeze is re-applied after load.
        """
        if "encoder" not in state:
            raise KeyError(
                "transferable state has no 'encoder'; cannot warm-start the P4 "
                f"encoder. Got keys: {sorted(state)}."
            )
        self.model.encoder.load_state_dict(state["encoder"], strict=strict)
        if "pma" in state:
            self.model.readout.pma.load_state_dict(state["pma"], strict=strict)
        # 'projector' (P3-only) is intentionally not loaded — P4 has no projector.
        self._apply_freeze()

    # ------------------------------------------------------------------
    # Forward / target
    # ------------------------------------------------------------------

    def forward(self, batch):  # type: ignore[override]
        if isinstance(self.x_name, str):
            return self.model(batch.data[self.x_name])
        return self.model(**{name: batch.data[name] for name in self.x_name})

    def _target(self, batch) -> Tensor:
        target = batch.data[self.y_name]
        if target.ndim > 1 and target.shape[-1] == 1:
            target = target.squeeze(-1)
        return target.long()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx: int) -> Tensor:  # noqa: ARG002
        logits = self.forward(batch)
        loss = self.loss(logits, self._target(batch))
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:  # noqa: ARG002
        logits = self.forward(batch)
        loss = self.loss(logits, self._target(batch))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self._test_logits = []
        self._test_targets = []

    def test_step(self, batch, batch_idx: int) -> None:  # noqa: ARG002
        logits = self.forward(batch)
        target = self._target(batch)
        loss = self.loss(logits, target)
        self.log("test_loss", loss, on_epoch=True)
        self._test_logits.append(logits.detach())
        self._test_targets.append(target.detach())

    def on_test_epoch_end(self) -> None:
        if not self._test_logits:
            return
        logits = torch.cat(self._test_logits)
        target = torch.cat(self._test_targets)
        for name, value in self._probe_metrics(logits, target).items():
            self.log(name, value, prog_bar=True)
        self._test_logits = []
        self._test_targets = []

    def _probe_metrics(self, logits: Tensor, target: Tensor) -> dict[str, Tensor]:
        """Neuroprobe eval scalars for the trained readout over the FULL probe
        test set: ``test_acc`` (micro-accuracy) always, plus ``test_auroc``
        (binary, rank-based) for 2-class tasks. Pure — no logging — so the
        capstone + G3 assert finiteness directly, and so the metric is computed
        once over all samples rather than averaged per batch.
        """
        metrics: dict[str, Tensor] = {
            "test_acc": multiclass_accuracy(
                logits, target, num_classes=self._n_classes, average="micro",
            )
        }
        if self._n_classes == 2:
            scores = logits.softmax(dim=-1)[:, 1]
            metrics["test_auroc"] = binary_auroc(scores, target)
        return metrics

    # ------------------------------------------------------------------
    # Optimizer (readout Linear only)
    # ------------------------------------------------------------------

    def _estimated_total_steps(self) -> int | None:
        try:
            return int(self.trainer.estimated_stepping_batches)
        except (RuntimeError, AttributeError):
            return None

    def configure_optimizers(self):  # type: ignore[override]
        params = self._trainable_parameters()
        total_steps = self._estimated_total_steps()
        if total_steps is None:
            return self.optim_config.build(params)
        try:
            return self.optim_config.build(params, total_steps=total_steps)
        except TypeError:
            return self.optim_config.build(params)


# ---------------------------------------------------------------------------
# Phase-4 experiment
# ---------------------------------------------------------------------------


class V14Phase4ReadoutExperiment(V14Experiment):
    """Phase-4 frozen-encoder probe experiment (WS-G).

    Pinned to ``phase == 4``. Replaces the parent CE-classifier ``BrainModule``
    with a :class:`V14Phase4ReadoutModule` built from the same
    ``brain_model_config.build(...)`` head (whose readout must be the B35
    :class:`V14PmaReadout`). The P3 checkpoint is loaded by the parent
    ``Experiment._load_pretrained`` → :meth:`V14Phase4ReadoutModule.load_transferable_state`
    before fit, so the optimizer/scheduler see the loaded weights.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def model_post_init(self, _ctx) -> None:  # type: ignore[override]
        super().model_post_init(_ctx)
        if self.phase != 4:
            raise ValueError(
                "V14Phase4ReadoutExperiment models Phase-4 only (phase == 4); "
                f"got phase={self.phase!r}."
            )

    def _build_brain_module(self, train_loader):  # type: ignore[override]
        batch = next(iter(train_loader))
        x = batch.data[self._input_tensor_name()]
        head_model = self.brain_model_config.build(
            n_in_channels=int(x.shape[1]),
            n_outputs=self._infer_n_outputs(train_loader),
        )
        if not isinstance(head_model, V14ParcelPerceiverWithHead):
            raise RuntimeError(
                "V14Phase4ReadoutExperiment expected brain_model_config.build to "
                "return a V14ParcelPerceiverWithHead; got "
                f"{type(head_model).__name__}."
            )
        return V14Phase4ReadoutModule(
            model=head_model,
            loss=self.loss.build(),
            optim_config=self.optim,
            x_name=self.x_name,
            y_name=self.y_name,
        )


__all__ = [
    "V14Phase4ReadoutModule",
    "V14Phase4ReadoutExperiment",
    "leave_two_subjects_out",
    "assert_no_electrode_overlap",
]
