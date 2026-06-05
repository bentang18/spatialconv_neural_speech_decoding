"""SCAFFOLD-01 (NT01): ``V14Experiment`` — phase-parametrized NeuralTrain
experiment for the v14 cross-subject SSL stack.

Per ``docs/neuroprobe/v14_implementation_fix_list.md`` (SCAFFOLD-01) and
``docs/neuroprobe/v14_blockers.md`` NT01, the v14 program has five phase
*values* — Phase-1 / Phase-2 (SSL), Phase-3a (Whisper-L8 MLP-adapter
warmup), Phase-3b (slow-LR student unfreeze), and Phase-4 (frozen-encoder
linear probe). B29 Item 1 collapsed Phase-1 ∪ Phase-2 into a single joint
SSL phase as the default (run by :class:`V14JointExperiment`, pinned to
``phase == 1``); the split P1→P2 path survives only as the
``R-keep-phase-split`` sister, which instantiates this class directly with
``phase=1`` then ``phase=2``. One class branches per phase via the
``phase`` discriminator. (Per-subject anatomy gating is via B30
``latent_valid``, not a P1-anatomy-OFF / P2-anatomy-ON phase split; the
shaft-block electrode mask was dropped in B03.)

This module is structural: the class declaration and the phase
discriminator are stable. The full runtime path (V14Data, EMA teacher,
checkpoint reload) is gated on SCAFFOLD-02..06 and the downstream blockers.

The per-step loss is composed by
:func:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses` — the loss
single-source-of-truth; its ``loss_variant`` argument selects the B31
2-term default (``b31_default``) or its sisters — NOT by a per-phase
coefficient tuple on this class.
"""

from __future__ import annotations

from typing import Literal

import pydantic

from speech_decoding.experiments.experiment import Experiment


V14Phase = Literal[1, 2, "3a", "3b", 4]


class V14Experiment(Experiment):
    """Phase-parametrized NeuralTrain experiment for v14 SSL.

    Inherits the full NeuralTrain/Exca scaffolding from
    :class:`speech_decoding.experiments.experiment.Experiment` (logger,
    callbacks, trainer, infra). Adds the ``phase`` discriminator.

    The runtime branching (EMA teacher in the joint SSL phase; Whisper
    adapter+pool in P3; frozen-encoder probe in P4) lives in
    ``_train_and_test`` once SCAFFOLD-02 lands.

    The active loss is composed by
    :func:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses` (the
    single source of truth), so the trainer never hard-codes a per-phase
    coefficient tuple on this class.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    phase: V14Phase

    # §7 / B01 no-weight-decay split (task #40). When True (default), and a
    # non-zero optimizer weight_decay is configured, biases / LayerNorm γβ /
    # named embedding tables are placed in a ``weight_decay: 0.0`` param group
    # (nanoGPT / timm convention) while matmul weights take the swept decay.
    # ``--no-wd-exclude-norms`` flips this off (decay every param uniformly) so
    # the M0 sweep can falsify whether the exclusion matters. No effect while
    # weight_decay == 0 (the split is a no-op there — see
    # ``optim_param_groups.maybe_split_no_decay``).
    wd_exclude_norms: bool = True
