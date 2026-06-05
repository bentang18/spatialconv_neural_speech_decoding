"""V14 first-pass DCC dispatch entrypoint.

Composes the v14 NeuralTrain Experiment: BT Wang2024Treebank study + DK-hard
support extractor + V14ParcelPerceiver + DETR readout, with first-pass defaults
locked in ``memory/project_v14_encoder_design_2026_05_13.md``.

DCC invocation (via ``scripts/dcc/dispatch``):

    scripts/dcc/dispatch -m speech_decoding.experiments.dispatch_v14 \\
        --mode lite --eps 1e-2 --m-sub-slots 4 --d-model 128 --depth 6

Smoke-test (laptop, no BT data):

    .venv/bin/python -m speech_decoding.experiments.dispatch_v14 --dry-run

Default electrode-tokens extractor is :class:`MultiStftView` (WS-C / C2,
B36): **F=50 raw ``|STFT|`` bins (FE-RAW-1, ``front_end="raw"`` default; LANDED
2026-06-04)**, hop=128 → 16 Hz (8 Hz latent), ``apply_log=False`` (raw |X|),
0.5 Hz HPF, and ``scaler=None`` (robust-z normalizes downstream). The F=30
⅓-octave filterbank is the demoted ``front_end="fbank"`` / ``R-filterbank-30bin``
sister; :class:`LogStftView` is the ``F-single-STFT`` sister; ``apply_log``
recovers the log-amplitude sister. Default support is
:class:`V14DKHardSupportExtractor` (K=80 DK, ``c_max=384`` padded), default
valid-mask is :class:`ElectrodeValidMask` (``c_max=384``). Caller can pass
``electrode_tokens_extractor=...`` to override the default (e.g. for the
P2 defensive sister run using the linear-baseline recipe).
"""

from __future__ import annotations

import argparse
import os
import typing as tp
from pathlib import Path

import neuralset as ns

import speech_decoding.models  # noqa: F401  # registers V14ParcelPerceiver with BaseModelConfig
from speech_decoding.experiments import Data, Experiment
from speech_decoding.experiments.lr_schedule import WarmupCosine  # noqa: F401  # registers WarmupCosine BaseLRScheduler for the optim discriminator
from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
from speech_decoding.extractors.ref_aug import (
    REF_MODES,
    RefAugMultiStftView,
    RefIdxExtractor,
)
from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor
from speech_decoding.extractors.subtype_meta import SubjectSubtypeExtractor
from speech_decoding.extractors.valid_mask import ElectrodeValidMask
from speech_decoding.extractors.view import MultiStftView
from speech_decoding.extractors.whisper_target import WhisperTargetExtractor
from speech_decoding.studies.braintreebank.anatomy import (
    DEFAULT_SUPPORT_BIAS_EPS,
    V14_DK_PARCEL_LABELS,
)
from speech_decoding.studies.braintreebank.manifest import V14_TRAIN_SUBJECT_IDS
from speech_decoding.studies.braintreebank.study import Wang2024Treebank
from speech_decoding.studies.braintreebank.word_events import BTWordEvents


# v4 amendment defaults (5/19 §3) + B28 (5/27 PM) + B29 Item 13 (5/27 PM-late):
# d=256, heads=8, depth=6, ~14.235M params at K=80, M=1 (B29 default), N=6,
# 1 cross-attn (B28 default). Sister R-m4-slots flips m_sub_slots back to 4
# via dispatch.
DEFAULT_D_MODEL = 256
DEFAULT_DEPTH = 6
DEFAULT_N_HEADS = 8
# B29 Item 13 lock 2026-05-27 PM-late: M=1 default (was 4). Sister
# R-m4-slots P0 flips via dispatch.
DEFAULT_M_SUB_SLOTS = 1
DEFAULT_K_PARCELS = len(V14_DK_PARCEL_LABELS)  # 80
# WS-C / C2 (B36): default front-end flips single-STFT → Multi-STFT.
# FE-RAW-1 (2026-06-04): the Multi-STFT default front end is the RAW |STFT|
# bins (F=50, §4.2), not the const-Q filterbank — encoder F_p = (50−5)//5+1
# = 10 at the kernel-5 patch stem (byte-shape-identical to the old F=30/kernel-3
# grid). The F=30 filterbank is the demoted ``front_end="fbank"`` sister.
DEFAULT_N_FREQ_BINS = 50
# WS-C / C1 (B36, hop=128 re-lock 2026-06-03): phase-conditional clip window.
# 5 s SSL clips (P1/P2/P3) → T_p=40; the P4 readout driver passes clip_len=1.0
# → T_p=8. n_time_bins (the RoPE ceiling) is derived from clip_len × the
# Multi-STFT frame geometry (1 + L//hop, center=True) — 5 s → 81, 1 s → 17 —
# not hardcoded.
DEFAULT_CLIP_LEN_S = 5.0
# B36 (2026-06-03) neural-response-lag knob (Δlag). The Whisper teacher cache is
# keyed by AUDIO movie-time and is immutable; sliding the NEURAL clip start by
# +Δlag aligns the lagged cortical response (neural-frame t reflects audio at
# t−lag) to the stimulus-time teacher frame. Default 0.0 = the 1:1 baseline
# (current behavior) and the falsifier null; the P3-distill sweep tries
# {0.075, 0.15, 0.30} s against it (R-distill-lag-*). Because the teacher is
# audio-keyed, the sweep is a pure neural re-slice — NO teacher recache. No-op
# for P1/P2 (student + EMA-teacher shift together); meaningful only for P3's
# frame-for-frame distill. The P4 probe MUST keep 0.0 (leaderboard parity) — a
# non-default raises under --phase 4 (guard below).
DEFAULT_NEURAL_LAG_S = 0.0
DEFAULT_BATCH_SIZE = 32
# DataLoader worker count (B1.4e, 2026-05-29). The per-sample extractor stack
# (CAR + torch.stft in LogStftView is recomputed per __getitem__; only the raw
# waveform load is MapInfra-cached) is CPU-bound, so num_workers=0 starves the
# GPU (1.48 it/s on the 5/29 Lite baseline). 4 workers overlap that CPU STFT
# with GPU compute; pair with --cpus-per-task >= num_workers + 1.
DEFAULT_NUM_WORKERS = 4
DEFAULT_N_EPOCHS = 100
# P4 frozen-probe early-stop patience (epochs on val_loss). P4 is a tiny
# supervised readout (≤3500 samples, ~514–2570 trainable params); it converges
# in a handful of epochs and the --n-epochs cap (100) would otherwise overfit
# it. Early-stopping on val_loss is correct ONLY for P4 (the actual downstream
# task); the SSL/distill phases never early-stop. Patience 10 is the operative
# limit; the epoch cap is a safety ceiling. <=0 disables (run to the cap).
DEFAULT_P4_EARLY_STOP_PATIENCE = 10
# Ship-first task default — `speech` is the highest-signal binary task
# requiring zero transcript enrichment (Sentence Onset = 0.780 CS-SOTA,
# Speech = 0.751; full ship-first set is {onset, speech, delta_volume,
# word_index} per the v2 paper's CS-above-chance four).
DEFAULT_TASK = "speech"
DEFAULT_EVAL_MODE = "CrossSession"
DEFAULT_TEST_SUBJECT_ID = 2
DEFAULT_TEST_TRIAL_ID = 4
DEFAULT_C_MAX = 384  # Locked 2026-05-23 PM per CQ12/B14 close. Covers all four
                     # Phase-1 corpora: D-cohort max=366 (n=128 manifest),
                     # AJILE12 max≈200 (146 surface + ~50 depth per Peterson
                     # 2022), BT max=256 (Wang2024Treebank raw), SWEC max=128.
                     # ValueError already raised in dk_support.py, view.py,
                     # valid_mask.py if any subject's n_real > c_max.

# MASK-01 per-corpus mains-notch field (v14_implementation_fix_list.md §A.3).
# Lifted from the formerly hardcoded `notch_filter=60.0`. SWEC pretrain (CH
# site) MUST pass `mains_notch_hz=50.0`; BT / D-cohort / AJILE12 (US sites)
# default to 60.0. Per-corpus call-out also propagated to
# training_recipe.md §2 and v14_blockers.md.
DEFAULT_MAINS_NOTCH_HZ = 60.0
MAINS_NOTCH_BY_CORPUS: dict[str, float] = {
    "braintreebank": 60.0,
    "wang2024treebank": 60.0,
    "d_cohort": 60.0,
    "cogan_dcohort": 60.0,
    "ajile12": 60.0,
    "swec": 50.0,
}

# B28 DKoleo demotion 2026-05-27 PM (B28 Item 1) + B29 sister-set
# expansion 2026-05-27 PM-late: DKoleo @ M4 is OFF by default; the four
# modes select which collapse-prevention sister arms the loss.
#
#   * ``off`` (default) — the B31 2-term base loss; no DKoleo arm at all.
#   * ``intra_clip_slots`` — B21-original per-clip × 80 slots (was 320
#     under M=4) unit, kept as the falsifier sister ``R-dkoleo-intra-clip-slots``.
#   * ``batch_cls_unit`` — DINOv2-faithful per-batch × CLS-analog
#     (utterance PMA-pooled vectors) unit (``R-dkoleo-batch-cls-unit``).
#   * ``vicreg_slot_variance`` — per-dim variance hinge per VICReg
#     (Bardes 2022), gated by MON-SLOT-REDUNDANCY's diag-zeroed mean
#     threshold (``R-vicreg-slot-variance``).
#
# Pre-B29 alias: ``batch_cls`` is accepted and silently maps to
# ``batch_cls_unit`` for back-compat with the 2026-05-27 PM (pre-late)
# dispatch surface. The composer in ``ssl/total_loss.py`` applies the
# locked ``W_DKOLEO_M4=0.1`` weight when a tensor is supplied; the unit
# choice is made upstream of the composer.
DKOLEO_MODES: tuple[str, ...] = (
    "off", "intra_clip_slots", "batch_cls_unit", "vicreg_slot_variance",
)
_DKOLEO_MODE_ALIASES: dict[str, str] = {"batch_cls": "batch_cls_unit"}
DEFAULT_DKOLEO_MODE: str = "off"

# B29 Item 11 lock 2026-05-27 PM-late: subtype embed vocab choice.
SUBTYPE_EMBED_VOCABS: tuple[str, ...] = ("binary", "three_way")
DEFAULT_SUBTYPE_EMBED_VOCAB: str = "binary"

# B29 Item 14 lock 2026-05-27 PM-late + MoE-FFN audit 2026-05-28: dense
# FFN preserved (default); ``soft_moe_4`` is the P2-if-budget sister
# pending a separate ``models/soft_moe.py`` build (Puigcerver 2024).
FFN_VARIANTS: tuple[str, ...] = ("dense", "soft_moe_4")
DEFAULT_FFN_VARIANT: str = "dense"

# B36 phase-mode relabel 2026-06-03: B36 reverses the B29 joint phase back
# to a staged P1 (front-end M2, all corpora) -> P2 (parcel M4, anatomy
# corpora, front-end LR/10) regime, so the recorded-only default regime is
# now ``split_p1_p2``. Behavioral stage selection is via ``--jepa-phase``
# (p1/p2), not this axis. ``joint_b29`` survives as the B29-collapse
# falsifier sister (R-joint-ssl).
PHASE_MODES: tuple[str, ...] = ("joint_b29", "split_p1_p2")
DEFAULT_PHASE_MODE: str = "split_p1_p2"

# B36 staged masked-JEPA sub-phase (within the joint SSL experiment, --phase 1).
# ``p1`` = front-end M2 masked prediction (all corpora); ``p2`` = parcel M4
# masked prediction (anatomy corpora, front-end LR/10). WS-E owns the staged
# P1->P2 checkpoint handoff; this axis selects which stage the run trains.
# ``V14JointExperiment.jepa_phase`` (v14_joint.py) is the field it threads onto.
JEPA_PHASES: tuple[str, ...] = ("p1", "p2")
DEFAULT_JEPA_PHASE: str = "p1"

# B36 WS-E E2: P2 front-end discriminative-LR scale. 0.1 = base/10 default;
# 0.2 = R-p2-frontend-lr-5; 0.0 = R-p2-freeze-frontend (front-end frozen).
DEFAULT_FRONTEND_LR_SCALE: float = 0.1

# B31 lock 2026-05-28 PM-late (V-JEPA-2-canonical loss simplification):
# joint SSL default is a 2-term surface (L_pre_frame @ M2 + L_post_frame
# @ LN_frame(M4), both pure L1 per V-JEPA 2 §2.1 Eq 1). The three
# falsifier sisters reinstate the B28/B29 dropped L_mid_slot / L_post_utt:
#
#   * ``b31_plus_m3``   → R-add-m3-loss         P0
#   * ``b31_plus_utt``  → R-add-utterance-loss  P0 (EAT-faithful, ≥0.02
#                          AUROC promotion gate vs default at BT-Lite Cell-0)
#   * ``b31_plus_both`` → R-add-both            P0
#
# The brain module builds the dropped heads only when the variant
# selects them; the SSL aggregator only computes the corresponding
# term when the matching tensors are supplied. ``LOSS_VARIANTS`` is
# derived from the canonical ``LossVariant`` Literal so a new sister
# arm added in ``ssl/aggregator.py`` is automatically exposed at the
# CLI without a parallel edit here.
from speech_decoding.ssl.aggregator import LOSS_VARIANTS  # noqa: E402

DEFAULT_LOSS_VARIANT: str = "b31_default"

# B36 (2026-06-01): the soft λ_anat·log(support+ε) routing bias and its
# schedule selector (ANATOMY_BIAS_MODES) were removed — the hard
# block-diagonal per-parcel pool consumes the one-hot DK ``support`` directly,
# with no bias to schedule. See
# project_v14_b36_perparcel_pool_structured_jepa_2026_06_01.

# B29 Item 5 corpus sampler-weight (α-hierarchical). 0.3 = the B29 lock
# default; sisters sweep α via dispatch.
DEFAULT_REF_OPERATOR_ALPHA: float = 0.3

# B29 corpus mix (HB02 doc-quality fix 2026-05-28): normalize the B29
# share-table headline (SWEC 35 / AJILE12 22 / D 18 / BT 12 — sums to
# 87) against per-corpus vb_eh totals. The HB02 memo flagged that the
# headline does not sum to 1.0; we land the normalized default so
# downstream code can assert sum == 1.0 ± 1e-4. When wiring against
# actual vb_eh totals, override via the ``corpus_mix`` dispatch kwarg.
DEFAULT_CORPUS_MIX: dict[str, float] = {
    "swec": 35.0 / 87.0,
    "ajile12": 22.0 / 87.0,
    "d_cohort": 18.0 / 87.0,
    "braintreebank": 12.0 / 87.0,
}

# B29 AJILE12 inclusion: re-included after being dropped in earlier
# memos (sensor-gap was reversed same-day per Agent 2's
# Charmander/DIVER-1 evidence).
DEFAULT_INCLUDE_AJILE12: bool = True


def _validate_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}; got {value!r}")


def _apply_extractor_cache(
    extractor: tp.Any, name: str, root_folder: str | Path | None
) -> None:
    """Point an extractor's ``infra.folder`` at the persistent two-tier cache.

    B1.5 (task #120, 2026-05-29): extractor outputs cached under
    ``{root_folder}/{name}/`` survive across Experiment-config changes
    (precision, batch_size, model knobs). Without this wiring every
    dispatch re-runs notch + STFT + per-clip metadata extractors (~16 min
    on Lite) because exca's TaskInfra slot binds the Experiment uid to a
    single folder containing both extractor outputs AND model
    checkpoints. CLAUDE.md storage tiering: this folder belongs on the
    DCC ``/work`` regenerate-cheap cache tier (75-day purge; set via
    ``EXCA_EXTRACTOR_CACHE_FOLDER``, concrete path in CLAUDE.md) — separate
    from the durable ``/hpc/group/coganlab/`` Experiment cache.

    No-op when ``root_folder`` is None (laptop dry-runs, tests). Skips
    extractors that don't inherit ``infra: MapInfra`` (BaseStatic
    subclasses like DK support / ValidMask carry no infra field).
    """
    if root_folder is None:
        return
    if not hasattr(extractor, "infra"):
        return
    folder = Path(root_folder) / name
    folder.parent.mkdir(parents=True, exist_ok=True)
    extractor.infra.folder = folder


def _build_optim_cfg(
    *,
    lr: float,
    lr_schedule: str,
    warmup_steps: int,
    min_lr_ratio: float,
    weight_decay: float,
    optimizer_name: str,
    adam_betas: tuple[float, float] | None,
) -> dict[str, tp.Any]:
    """Assemble the ``LightningOptimizer`` config dict (#37, audit 2026-06-03).

    NOT fused: a fused torch optimizer sets ``_step_supports_amp_scaling=True``,
    which makes Lightning's bf16-mixed AMP precision plugin REFUSE gradient
    clipping (``precision/amp.py`` raises "does not allow for gradient clipping
    because it performs unscaling of gradients internally"). The §7 recipe locks
    BOTH ``grad_clip=1.0`` and ``bf16-mixed``, so fused is incompatible and must
    stay off (verified live: job 47686013 crashed at the first optimizer step on
    exactly this combo; pre-#37 runs only got away with fused because grad-clip
    was silently OFF). fused was a 2026-05-30 speedup, never a recipe lock — the
    non-fused path is the standard, with negligible cost at this model size.
    ``betas`` / ``weight_decay`` are added ONLY when set, so the default (Adam,
    lr=1e-3, no β/wd) is the prior plain-Adam config (sans the fused speedup).
    With ``lr_schedule="warmup_cosine"`` the §7
    locked linear-warmup → cosine→0 shape is attached via the custom
    ``WarmupCosine`` scheduler — fields are TOP-LEVEL (it is a plain
    ``BaseLRScheduler``, NOT a ``kwargs``-dict ``BaseTorchLRScheduler``); its
    ``total_steps`` is supplied at ``configure_optimizers`` time from
    ``trainer.estimated_stepping_batches`` (which ``--ssl-max-steps`` pins).
    """
    _validate_choice("lr_schedule", lr_schedule, ("constant", "warmup_cosine"))
    _validate_choice("optimizer_name", optimizer_name, ("Adam", "AdamW"))
    # §7 LOCK GUARD (audit 2026-06-03, lens-2 FAIL): the B01 recipe locks TWO
    # param groups — no weight-decay on biases, LayerNorm/RMSNorm γβ, and the
    # freq embedding. The phase modules' ``_phase_param_groups`` split by stage
    # for discriminative LR but do NOT yet carry a ``weight_decay: 0.0`` override
    # for that no-WD subset, so a non-zero ``weight_decay`` here would be applied
    # UNIFORMLY to every param — a silent lock violation. Refuse loudly until the
    # no-WD groups are implemented (a Ben-gated training-orchestration follow-up,
    # paired with the M0 weight_decay value). wd=0.0 (the default) is unaffected
    # and a valid first-pass center; this never fires on the maiden run.
    if weight_decay:
        raise ValueError(
            f"weight_decay={weight_decay} is not yet supported: the §7 no-WD "
            "param-group split (biases / LayerNorm γβ / freq_embed exempt) is "
            "NOT implemented in the phase modules' configure_optimizers, so a "
            "non-zero weight_decay would decay those params too — violating the "
            "B01 lock. Pass --weight-decay 0 (the locked-shape-compliant first "
            "pass), or implement the no-WD groups first. See "
            "reports/b01_lr_schedule_unimplemented_lock_2026_06_03.md."
        )
    # NOT fused — fused optimizers are incompatible with Lightning's bf16-mixed
    # AMP gradient clipping (see docstring; both are §7 locks). Empty default →
    # standard non-fused torch optimizer, which clears _step_supports_amp_scaling
    # so the AMP clip guard does not fire.
    optim_kwargs: dict[str, tp.Any] = {}
    if weight_decay:
        # Unreachable while the guard above stands (wd is falsy here); retained as
        # the exact post-guard-removal path — when the no-WD groups land (task #40)
        # the guard is deleted and this forwards the M0 weight_decay to either
        # optimizer family.
        optim_kwargs["weight_decay"] = weight_decay
    elif optimizer_name == "AdamW":
        # AdamW's torch default is weight_decay=0.01 — pin it to 0.0 EXPLICITLY so
        # the §7 no-WD lock is not violated via the torch family default on the
        # lock-recommended ``--optimizer AdamW`` path (the guard only sees the flag
        # value, which is 0.0 here; without this pin AdamW would decay biases / LN
        # γβ / freq_embed at 0.01 uniformly). Adam's torch default is already 0.0,
        # so its kwargs stay empty.
        optim_kwargs["weight_decay"] = 0.0
    if adam_betas is not None:
        optim_kwargs["betas"] = list(adam_betas)
    cfg: dict[str, tp.Any] = {
        "optimizer": {"name": optimizer_name, "lr": lr, "kwargs": optim_kwargs}
    }
    if lr_schedule == "warmup_cosine":
        cfg["scheduler"] = {
            "name": "WarmupCosine",
            "warmup_steps": warmup_steps,
            "min_lr_ratio": min_lr_ratio,
        }
        cfg["interval"] = "step"
    return cfg


def _resolve_ddp_strategy(tasks_per_node: int | None) -> str | None:
    """The Lightning ``strategy`` for a given srun-rank topology.

    A multi-rank run (``tasks_per_node>1``) MUST use the find-unused DDP
    strategy: the staged B36 SSL phases leave whole submodules out of the
    active loss (P1 trains the front-end only; the pool / inter-parcel encoder /
    predictor stay ``requires_grad=True`` but get no gradient — the predictor is
    P2-only). Plain DDP's reducer rejects unused grad-requiring params on the
    2nd iteration ("parameters that were not used in producing the loss"); a
    1-batch ``--fast-dev-run`` cannot catch it (the reducer only rebuilds
    buckets from the 2nd forward). Single-GPU/local stays non-DDP (``None`` →
    Lightning auto-selects single-device).

    Single source of truth: both ``build_v14_experiment`` (what actually
    reaches the Trainer) and ``main()``'s run summary (what gets printed)
    derive the strategy here, so the printed value can never drift from the
    applied one.
    """
    return (
        "ddp_find_unused_parameters_true"
        if tasks_per_node is not None and tasks_per_node > 1
        else None
    )


def build_v14_experiment(
    *,
    bt_root: str | None = None,
    mode: tp.Literal["nano", "lite", "full"] = "lite",
    task: str = DEFAULT_TASK,
    eval_mode: tp.Literal["CrossSession", "CrossSubject"] = DEFAULT_EVAL_MODE,
    test_subject_id: int = DEFAULT_TEST_SUBJECT_ID,
    test_trial_id: int = DEFAULT_TEST_TRIAL_ID,
    binary_tasks: bool = True,
    electrode_tokens_extractor: tp.Any | None = None,
    # C3 (WS-C, B13): per-(electrode, freq, session) robust-z on the default
    # MultiStftView. True (the real-run default) fits frozen median/MAD per
    # session over its own full recording in prepare() and applies it per clip;
    # False emits the raw filterbank (only valid for the synthetic capstone /
    # plumbing smokes, where per-token encoder LN absorbs scale — plan §C3).
    # Ignored when a custom ``electrode_tokens_extractor`` is supplied (set the
    # flag on that extractor directly).
    session_robust_z: bool = True,
    mains_notch_hz: float = DEFAULT_MAINS_NOTCH_HZ,
    # #17 (D2 / T1.7): per-session MNE-LOF bad-channel drop on the default
    # MultiStftView. Default OFF so the existing multi-TB STFT cache uid is
    # untouched (the lof_* fields sit out of the uid via exclude_defaults). When
    # flipped True the dispatch also forces ``drop_bads=True`` (the view validator
    # requires it) and a ``lof_report_path`` so the per-subject drop counts are
    # written to JSON for the BEN REVIEW GATE ("report back how many channels are
    # dropped") before the run is scored. ``lof_threshold`` / ``lof_n_neighbors``
    # enter the cache uid (they change which channels drop) — only passed when LOF
    # is on. Ignored when a custom ``electrode_tokens_extractor`` is supplied (set
    # lof_* on that extractor directly). The 1.5 / 20 defaults MIRROR the
    # MultiStftView field defaults (extractors/view.py — the single source of
    # truth; the CLI defaults mirror them too, so all three must move in
    # lock-step. test_dispatch_lof_threshold_inert_when_lof_off pins the match).
    lof_bad_channels: bool = False,
    lof_threshold: float = 1.5,
    lof_n_neighbors: int = 20,
    lof_report_path: str | None = None,
    # #35: padded electrode-slot count C. Default 384 (DEFAULT_C_MAX) covers all
    # four Phase-1 corpora (D-cohort max=366) so the multi-corpus cache is shared.
    # BT-only runs can pass 256 (BT raw max=256, the exact safe floor — the
    # extractors raise if any subject's n_real > c_max) to drop the 128 pure-pad
    # slots the per-electrode front-end would otherwise process (~33% wasted
    # FLOPs/activations, an OOM lever). c_max is in the extractor-cache uid
    # (MultiStftView/dk_support/valid_mask/shaft_mask) → a non-default value forces
    # a fresh STFT cache and is a recipe-amendment (HB02 re-cost). When a custom
    # ``electrode_tokens_extractor`` is supplied its c_max must match this value.
    c_max: int = DEFAULT_C_MAX,
    eps: float = DEFAULT_SUPPORT_BIAS_EPS,
    d_model: int = DEFAULT_D_MODEL,
    depth: int = DEFAULT_DEPTH,
    n_heads: int = DEFAULT_N_HEADS,
    m_sub_slots: int = DEFAULT_M_SUB_SLOTS,
    n_freq_bins: int = DEFAULT_N_FREQ_BINS,
    # WS-C / C1: phase-conditional clip window (seconds). 5 s for the SSL
    # phases (P1/P2/P3 → T_p=40); the P4 readout driver passes 1.0 (→ T_p=8).
    clip_len: float = DEFAULT_CLIP_LEN_S,
    # B36 (2026-06-03) Δlag neural-response-lag (seconds), wired to the
    # segmenter clip ``start`` offset. 0.0 (default) = 1:1 stimulus-onset
    # baseline / falsifier null; P3-distill sweep sisters R-distill-lag-{75,150,
    # 300}ms. Audio-keyed teacher ⇒ no recache. Non-default raises under the
    # supervised Phase-4 path (leaderboard parity); no-op for P1/P2.
    neural_lag_s: float = DEFAULT_NEURAL_LAG_S,
    # ``None`` → derive the RoPE ceiling from ``clip_len`` × the front-end's
    # frame geometry (``electrode_tokens_extractor.n_time_bins_for_duration``).
    # Pass an explicit int only to override (e.g. a custom front-end).
    n_time_bins: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    # DataLoader worker count (B1.4e, 2026-05-29). >0 overlaps the per-sample
    # CPU STFT with GPU compute; pass --cpus-per-task >= num_workers + 1.
    num_workers: int = DEFAULT_NUM_WORKERS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    # Step budget (SSL/distill phases). None → epoch budget (``n_epochs``).
    max_steps: int | None = None,
    # Validation cadence as a step count (#54). On a ``max_steps`` SSL phase
    # that ends mid-epoch, epoch-boundary validation never runs, so the
    # collapse-guard soft panel would never evaluate. The chain sets this for
    # the SSL phases; None → Lightning default (epoch-boundary only).
    val_check_interval: int | float | None = None,
    # Cap on validation batches per check (#66). An uncapped SSL val set
    # (~875 batches) made each collapse-guard validation ~8 min → ~80% of
    # wall-clock under the (now-fixed) over-frequent cadence. The guard panel
    # estimates RankMe/coverage from a small sample, so a cap is lossless for
    # it. None → full val set (correct for P4, where val_loss IS the metric).
    limit_val_batches: int | float | None = None,
    # Early-stopping patience on ``val_loss``. None → no early-stop (the SSL /
    # distill phases train to a fixed budget; their val loss is a pretext
    # reconstruction/distill objective, NOT the downstream metric, so val-loss-min
    # is the wrong stop signal). The supervised P4 probe sets this — there
    # val_loss IS the downstream task, so val-loss-min is correct.
    early_stopping_patience: int | None = None,
    # #54 audit M1: the collapse guard is an SSL/distill kill-switch. P4 is a
    # frozen linear probe that cannot "collapse" in the SSL sense and already
    # has EarlyStopping on its real downstream val_loss, so the chain disables
    # the guard there (its loss-blowup criterion would only duplicate
    # EarlyStopping while risking an exca-cache-poisoning abort on benign
    # probe over-fit). True for P1/P2/P3a/P3b.
    collapse_guard: bool = True,
    seed: int = 33,
    exca_folder: str | None = None,
    # #54 audit C1: exca's default ``mode="cached"`` re-RAISES a stored failure
    # (e.g. a CollapseStop abort) on a same-config relaunch instead of
    # recomputing — so a code-level fix (which does not change the config-derived
    # uid) would never actually run. ``retry`` recomputes on a cached *error*
    # while keeping cached *successes*; it is the correct mode for the abort →
    # diagnose → fix → relaunch loop. This builder default stays ``cached``
    # (back-compat for direct callers); ``main()`` resolves an unset
    # ``--exca-mode`` to ``retry`` for ``--chain`` and ``cached`` otherwise, so
    # the capstone chain is C1-safe by default.
    exca_mode: str = "cached",
    # B1.5 (task #120, 2026-05-29) two-tier extractor cache root. When set,
    # each extractor's ``infra.folder`` is pointed at
    # ``{extractor_cache_folder}/{extractor_name}/`` so its outputs survive
    # across Experiment-config changes (precision, batch_size, model knobs)
    # — the previous behavior re-ran ~16 min of notch + STFT + per-clip
    # metadata prep on every OOM-debug iteration. Resolved from
    # ``EXCA_EXTRACTOR_CACHE_FOLDER`` if unset; ``None`` means no caching
    # (laptop dry-runs / tests).
    extractor_cache_folder: str | None = None,
    cluster: str | None = None,
    # Slurm resource knobs (B1.4a, 2026-05-29). All default ``None`` so
    # they only override exca's TaskInfra / submitit defaults when set.
    # Without them, ``--cluster slurm`` falls through to submitit's
    # ``common`` partition + ~2GB mem, which OOM-kills any real BT data
    # prep. See submitit field defs in ``exca/slurm.py``.
    slurm_partition: str | None = None,
    slurm_account: str | None = None,
    mem_gb: float | None = None,
    gpus_per_node: int | None = None,
    cpus_per_task: int | None = None,
    timeout_min: int | None = None,
    # 4-GPU DDP (#33). exca/submitit default tasks_per_node=1, so a bare
    # gpus_per_node>1 gives N GPUs to ONE srun task and Lightning hangs waiting
    # for N ranks. Real DDP needs one srun rank per GPU (tasks_per_node=N); exca
    # requires slurm_use_srun=True whenever tasks_per_node>1. main() auto-derives
    # both from gpus_per_node for a slurm run; left None/False here so single-GPU
    # and local paths are byte-for-byte unchanged.
    tasks_per_node: int | None = None,
    slurm_use_srun: bool = False,
    # Lightning trainer precision. v14 first-pass default is bf16-mixed
    # per 2026-05-29 OOM diagnosis on RTX 5000 Ada (31 GiB): factorized
    # per-electrode SA over C=384 padded electrodes at d=256 exhausts
    # fp32 activations even at BS=8. bf16-mixed halves activation memory
    # at the standard transformer-training precision floor.
    precision: str | None = "bf16-mixed",
    fast_dev_run: bool | int = False,
    # B28 DKoleo demotion 2026-05-27 PM: select the DKoleo @ M4 unit
    # (or disable). Plumbed onto the brain-model config so the SSL
    # training loop sees it; the downstream Phase-4 path treats it as
    # informational.
    dkoleo_mode: str = DEFAULT_DKOLEO_MODE,
    # B28 cross-attn collapse 2026-05-27 PM: ``[0]`` Perceiver IO default,
    # ``[0, 3]`` opt-in via ``R-perceiver-original-2-cross-attns`` sister.
    cross_attn_positions: list[int] | None = None,
    # B29 Item 11 + 5/28 PM precedent-audit flip 2026-05-28: subtype default
    # ON → OFF (Agent 2 found M3AE precedent net-neutral on iEEG via DIVER-1
    # §4.1).
    # B32 5/28 PM-late first-pass-no-input-aug lock: ref_embed default ON →
    # OFF. With LogStftView (single static shaft-CAR) as the default
    # dispatch extractor, ref_embed always indexes the same row; the
    # additive contribution is a no-op (best case) or pure distribution
    # drift (worst case). Sister `R-ref-aug-3-cell` re-enables the lookup
    # paired with RefAugMultiStftView. See:
    # memory/project_v14_b32_first_pass_no_input_aug_2026_05_28.md
    subtype_embed_enabled: bool = False,
    subtype_embed_reuse_kv: bool = True,
    subtype_embed_vocab: str = DEFAULT_SUBTYPE_EMBED_VOCAB,
    ref_embed_enabled: bool = False,
    ref_embed_reuse_kv: bool = True,
    # B29 phase-mode + corpus mix lock 2026-05-27 PM-late.
    phase_mode: str = DEFAULT_PHASE_MODE,
    include_ajile12: bool = DEFAULT_INCLUDE_AJILE12,
    ref_operator_alpha: float = DEFAULT_REF_OPERATOR_ALPHA,
    corpus_mix: dict[str, float] | None = None,
    notch_filter_hz_by_corpus: dict[str, float] | None = None,
    # B29 Item 14 + MoE-FFN audit 2026-05-28: dense default; soft_moe_4
    # is the P2-if-budget sister.
    ffn_variant: str = DEFAULT_FFN_VARIANT,
    # B2.1 (#96) phase-switch hook. When ``joint_phase=True`` the builder
    # returns a :class:`V14JointExperiment` (pinned to ``phase=1`` via its
    # ``model_post_init`` check, B29 Item 1 P1+P2 collapse) instead of a
    # vanilla supervised :class:`Experiment`. The joint subclass overrides
    # ``_build_brain_module`` (builds the masked-JEPA ``V14JointBrainModule``)
    # and ``model_post_init`` (validates phase + quarantined sisters); it does
    # NOT override ``_train_and_test`` / ``run``. Quarantined sisters
    # (non-default ``latent_valid_override`` / ``sa_mask_mode`` / ``loss_variant``)
    # raise at *construction* (model_post_init), so wiring the dispatch never
    # silently downgrades to Phase-4 CE.
    joint_phase: bool = False,
    # B30-dispatch-sister-flags (drift-table row added 2026-05-28 by R12
    # wiring audit). Default values match the B30 lock; non-default
    # values flag :class:`NotImplementedError` from
    # :class:`V14JointExperiment.model_post_init` until the corresponding
    # runtime branch (B2.2 aggregator-call / encoder latent-SA key-only)
    # lands. Persisted onto the run record so the choice is grep-able.
    latent_valid_override: str = "support",
    sa_mask_mode: str = "bidirectional",
    # B31 lock 2026-05-28 PM-late (V-JEPA-2-canonical 2-term joint SSL
    # default). ``"b31_default"`` is the locked first-pass shape; the
    # three sisters reinstate the B28/B29 dropped terms for
    # falsification. Only effective under ``joint_phase=True``; the
    # supervised Phase-4 path raises if a non-default is requested.
    loss_variant: str = DEFAULT_LOSS_VARIANT,
    # B36 staged masked-JEPA sub-phase (H4 2026-06-03): threads
    # ``V14JointExperiment.jepa_phase`` so P2 (parcel M4) is launchable from the
    # CLI — previously only the default ``p1`` (front-end M2) could run.
    # Effective under ``joint_phase=True`` only; a non-default under the
    # supervised Phase-4 path raises (same guard as the B30/B31 sister flags).
    jepa_phase: str = DEFAULT_JEPA_PHASE,
    # B36 WS-E E2: P2 front-end discriminative-LR scale (hyperparameter).
    # 0.1 = base/10 default; 0.2 = R-p2-frontend-lr-5; 0.0 =
    # R-p2-freeze-frontend. Effective under ``joint_phase=True`` +
    # ``jepa_phase="p2"`` only (no effect at P1, where the front-end is the
    # sole trained group).
    frontend_lr_scale: float = DEFAULT_FRONTEND_LR_SCALE,
    # WS-H / T20 (B33 P3 distillation): root of the whole-movie Whisper teacher
    # cache built by scripts/neuroprobe/build_bt_teacher_cache.py. When set, the
    # segmenter emits a per-clip ``whisper_target`` (B, 250, 1280) the P3 loss
    # consumes; ``None`` (P1/P2/P4) omits it. The join is keyed by the trigger
    # Word's MOVIE-clock onset (``movie_onset_s``), NOT the Δlag-shifted neural
    # window — see the WhisperTargetExtractor docstring (FLAG 9). ``layer_merge``
    # must match the cache build ("mean_all" locked default; "8" = the
    # R-whisper-single-layer-L8 sister).
    whisper_target_cache_dir: str | None = None,
    whisper_layer_merge: str = "mean_all",
    # B35 (2026-05-31) Phase-4 readout selector (reverts B34).
    # "pma_mean_linear" (default) = V14PmaReadout: frozen P3-PMA collapses
    # parcels → (B, T_p, d), then mean-over-time → Linear (only the linear
    # trains). "pma_flatten_linear" (R-p4-flatten) / "pma_timeattn_linear"
    # (R-p4-time-attn-pool) vary the temporal op. "attentive"
    # (R-p4-attentive) = V14PerTaskAttentivePooler over the full parcel×time
    # field (792k trainable — demoted, not fittable at the eval budget);
    # "meanpool" (R-p4-meanpool-no-pma) = V14MeanPoolLinearHead. The P3 PMA
    # weights load into readout.pma.* and the encoder-freeze for a real
    # probe are trainer-level concerns (load checkpoint → freeze), not
    # build flags.
    readout: str = "pma_mean_linear",
    # 2026-05-30 speedup audit (Tier-2, #119): default-off activation
    # checkpointing on the encoder block stacks. Threaded onto the
    # brain-model config; the encoder forward gates it on training + grad
    # so the no_grad teacher pass is never checkpointed. Numerics-safe.
    gradient_checkpointing: bool = False,
    # WS-F / B33 Phase-3 distillation routing (#21). ``p3_distill=True`` returns
    # a :class:`V14Phase3Experiment` (Whisper all-layer-mean SmoothL1 distill)
    # instead of the joint / supervised path; ``p3_stage`` picks 3a (encoder
    # frozen, PMA+projector connector warmup) or 3b (encoder unfrozen,
    # discriminative LR). P3 REQUIRES ``whisper_target_cache_dir`` (the teacher
    # stream) and, when ``target_standardize`` (B33 default), a
    # ``channel_stats_path`` (train-only per-channel z-score stats from
    # fit_channel_stats). Mutually exclusive with ``joint_phase``.
    p3_distill: bool = False,
    p3_stage: tp.Literal["3a", "3b"] = "3a",
    channel_stats_path: str | None = None,
    target_standardize: bool = True,
    projector_mode: tp.Literal["mlp", "linear"] = "mlp",
    # B33 §5 3b parcel-side discriminative-LR scale (base/3 lock). No effect
    # under 3a / the non-P3 paths; persisted onto the run record either way.
    parcel_lr_scale: float = 1.0 / 3.0,
    # WS-G B35 Phase-4 frozen-probe routing (#21). ``phase4_frozen_probe=True``
    # returns a :class:`V14Phase4ReadoutExperiment` (frozen encoder + frozen
    # P3-PMA → mean → trainable Linear, the B35 readout) carrying the
    # transferable-state protocol so it can warm-start from a P3 snapshot — the
    # base supervised ``Experiment`` (default P4) builds a from-scratch CE
    # ``BrainModule`` with no handoff. The chain / any ``--resume-from`` P4 run
    # uses the frozen probe. Mutually exclusive with ``joint_phase`` / ``p3_distill``.
    phase4_frozen_probe: bool = False,
    # Optimizer / LR-schedule (#37, 4-agent audit 2026-06-03). The v14 §7 recipe
    # (closed B01 lock) fixes the LR *shape* as linear-warmup → cosine→0 + AdamW
    # + β=(0.9, 0.95) + grad_clip=1.0 as non-swept universals; the peak-LR value
    # and weight_decay were re-classified to M0-measured sweep centers on
    # 2026-06-01. Defaults here REPRODUCE the prior constant-Adam behavior bit-
    # for-bit (lr=1e-3, schedule="constant", optimizer="Adam", wd=0, no β
    # override) so nothing silently changes; the locked warmup+cosine+AdamW path
    # is opt-in via ``lr_schedule="warmup_cosine"``. ``gradient_clip_val`` is the
    # one exception — the CLI defaults it to 1.0 (restoring the locked-universal
    # that was silently OFF), but this function-level default stays None for the
    # tiny test experiments. EMA τ is fixed at 0.99925 (B26/B27); the §9 EMA
    # ramp is DEAD and is NOT wired here.
    lr: float = 1e-3,
    # Same CLI-vs-function split as gradient_clip_val (above): the CLI flipped
    # these to the §7/B01 lock (warmup_cosine + AdamW + β2=0.95) on 2026-06-04
    # ([[project_v14_optimizer_default_b01_config_2026_06_04]]) and a main() launch
    # guard refuses constant-Adam/β2=0.999 on a real SSL/distill run — but these
    # function-level defaults stay at parity (constant / Adam / None betas) for the
    # tiny test experiments and nano smokes, which never go through the CLI guard.
    lr_schedule: tp.Literal["constant", "warmup_cosine"] = "constant",
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.0,
    weight_decay: float = 0.0,
    optimizer_name: tp.Literal["Adam", "AdamW"] = "Adam",
    # None → torch optimizer default betas (Adam (0.9, 0.999)). The CLI defaults
    # this to (0.9, 0.95) (the locked v14 SSL β2); this function-level default
    # stays None for behavior parity (see the split note above).
    adam_betas: tuple[float, float] | None = None,
    gradient_clip_val: float | None = None,
    # Lightning ``accumulate_grad_batches`` (effective-batch lever). 1 → no
    # accumulation (default; bit-for-bit prior behavior). >1 → effective batch =
    # batch_size * N at N× wall-clock/step. Prefer a larger physical batch on a
    # 48 GB+ card; this is the fallback for the 32 GB coganlab-gpu cards.
    accumulate_grad_batches: int = 1,
    # B36 WS-E (E3/E4) cross-phase checkpoint handoff, threaded onto the
    # Experiment so the multi-phase chain (``run_phase_pipeline``) or a manual
    # per-phase ``--resume-from`` warm-starts / snapshots the transferable
    # encoder (+PMA from P3). Both default off — inert for a standalone phase.
    pretrained_ckpt: str | None = None,
    snapshot_ckpt_to: str | None = None,
) -> Experiment:
    """Compose a v14 first-pass Experiment ready for ``.run()`` dispatch.

    The ``electrode_tokens_extractor`` arg is REQUIRED for real runs and must
    emit per-event ``(n_channels, n_time_bins, n_freq_bins)`` STFT tokens
    following the v14 preprocessing recipe (``N1 × R2 × I2 × F1`` post-5/25
    swap; ``I2L`` is now the F-log-amplitude sister via ``apply_log=True``).

    Word events are appended downstream of :class:`Wang2024Treebank` via
    :class:`BTWordEvents` (``ns.Chain``) so per-trial ``words_df`` /
    ``nonverbal_df`` only load when ``study.run()`` materialises the chain.
    """
    bt_root = bt_root or os.environ.get("ROOT_DIR_BRAINTREEBANK")
    if bt_root is None:
        raise RuntimeError(
            "ROOT_DIR_BRAINTREEBANK must be set or bt_root passed explicitly"
        )

    dkoleo_mode = _DKOLEO_MODE_ALIASES.get(dkoleo_mode, dkoleo_mode)
    _validate_choice("dkoleo_mode", dkoleo_mode, DKOLEO_MODES)
    _validate_choice("subtype_embed_vocab", subtype_embed_vocab, SUBTYPE_EMBED_VOCABS)
    subtype_vocab_size = 2 if subtype_embed_vocab == "binary" else 3
    _validate_choice("phase_mode", phase_mode, PHASE_MODES)
    _validate_choice("ffn_variant", ffn_variant, FFN_VARIANTS)
    _validate_choice("loss_variant", loss_variant, LOSS_VARIANTS)
    _validate_choice("jepa_phase", jepa_phase, JEPA_PHASES)
    optim_cfg = _build_optim_cfg(
        lr=lr, lr_schedule=lr_schedule, warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio, weight_decay=weight_decay,
        optimizer_name=optimizer_name, adam_betas=adam_betas,
    )

    if ffn_variant != "dense":
        # MoE-FFN audit 2026-05-28: ``soft_moe_4`` is reserved as a P2
        # if-budget sister and requires ``models/soft_moe.py``. Fail
        # closed until that lands.
        raise NotImplementedError(
            f"ffn_variant={ffn_variant!r} requires models/soft_moe.py "
            "(R-moe-ffn-soft-4 P2 if-budget sister; not yet built)."
        )

    # B29 corpus mix sum-to-1.0 assertion.
    corpus_mix = dict(corpus_mix) if corpus_mix is not None else dict(DEFAULT_CORPUS_MIX)
    if not include_ajile12 and "ajile12" in corpus_mix:
        # Re-normalize over remaining corpora when AJILE12 is excluded.
        del corpus_mix["ajile12"]
        total = sum(corpus_mix.values())
        if total <= 0:
            raise ValueError(
                "corpus_mix with AJILE12 removed sums to zero — supply an "
                "explicit ``corpus_mix`` dict."
            )
        corpus_mix = {k: v / total for k, v in corpus_mix.items()}
    mix_sum = sum(corpus_mix.values())
    if abs(mix_sum - 1.0) > 1e-4:
        raise ValueError(
            f"corpus_mix must sum to 1.0 ± 1e-4; got sum={mix_sum:.6f} "
            f"over {sorted(corpus_mix.keys())}"
        )

    # Every corpus in ``corpus_mix`` must have a
    # ``notch_filter_hz_by_corpus`` entry — closing the loop on
    # MASK-01 (60 Hz US sites vs 50 Hz SWEC CH). Default seeded from
    # ``MAINS_NOTCH_BY_CORPUS``; explicit dispatch can override.
    notch_filter_hz_by_corpus = (
        dict(notch_filter_hz_by_corpus)
        if notch_filter_hz_by_corpus is not None
        else dict(MAINS_NOTCH_BY_CORPUS)
    )
    missing_notch = sorted(set(corpus_mix) - set(notch_filter_hz_by_corpus))
    if missing_notch:
        raise ValueError(
            "notch_filter_hz_by_corpus is missing entries for "
            f"{missing_notch}; supply the per-corpus mains frequency "
            "(US sites = 60.0, SWEC CH = 50.0) so the extractor builds "
            "with the right notch for each corpus."
        )
    # Reconcile the legacy scalar with the per-corpus map: if the
    # default ``mains_notch_hz`` was kept, the BT extractor inherits
    # the map's BT entry; an explicit non-default scalar overrides
    # (preserves the SWEC-via-scalar dispatch path covered by
    # ``test_b28_dispatch_mains_notch_kwarg_overrides_default``).
    bt_notch_hz = notch_filter_hz_by_corpus.get("braintreebank", mains_notch_hz)
    effective_bt_notch_hz = (
        mains_notch_hz if mains_notch_hz != DEFAULT_MAINS_NOTCH_HZ else bt_notch_hz
    )
    notch_filter_hz_by_corpus["braintreebank"] = effective_bt_notch_hz

    if not 0.0 < ref_operator_alpha < 1.0:
        raise ValueError(
            f"ref_operator_alpha must lie in (0, 1); got {ref_operator_alpha}"
        )

    # Δlag (neural-response lag) must be causal and bounded: a cortical
    # response follows its stimulus (≥0) and the higher-order auditory /
    # associative response to passively-watched film is well under 1 s. Negative
    # would slice the neural window *before* the stimulus (acausal); > 1 s is
    # unphysical for this distill alignment.
    if not 0.0 <= neural_lag_s <= 1.0:
        raise ValueError(
            f"neural_lag_s (Δlag) must lie in [0.0, 1.0] s; got {neural_lag_s}"
        )

    # Resolve two-tier extractor cache root from env if not explicit.
    extractor_cache_folder = extractor_cache_folder or os.environ.get(
        "EXCA_EXTRACTOR_CACHE_FOLDER"
    )

    if electrode_tokens_extractor is None:
        # WS-C / C2 (B36) + FE-RAW-1 (2026-06-04): Multi-STFT front-end, RAW
        # |STFT| bins (F=50, front_end="raw" default), hop=128 → 16 Hz (8 Hz
        # latent), raw |X| via apply_log=False. C4: 0.5 Hz HPF removes DC + slow
        # drift before the STFT. C3: StandardScaler dropped (scaler=None) —
        # robust-z normalizes the front-end output downstream of the view (see
        # Nv14RobustZTransform / SessionRobustZNormalizer).
        mstft_kwargs: dict[str, tp.Any] = dict(
            event_types="Ieeg",
            car="shaft",
            notch_filter=effective_bt_notch_hz,
            filter=(0.5, None),
            scaler=None,
            apply_log=False,
            channel_order="original",
            c_max=c_max,
            session_robust_z=session_robust_z,
        )
        # #17: only attach the lof_* kwargs when LOF is ON. Off → none of them are
        # forwarded, so the view keeps its field defaults and the multi-TB STFT
        # cache uid is untouched. NOTE the load-bearing protection is this NOT-
        # passing, not exclude_defaults: a stray ``--lof-threshold 2.0`` without
        # ``--lof-bad-channels`` would, IF forwarded, be a non-default value that
        # exclude_defaults does NOT drop → it would perturb the cache. Not
        # forwarding it is what keeps the cache stable. On → drop_bads=True is
        # forced (the view validator requires it; default-False = the prior
        # no-drop behaviour) and threshold/n_neighbors/report_path flow through.
        if lof_bad_channels:
            mstft_kwargs.update(
                lof_bad_channels=True,
                drop_bads=True,
                lof_threshold=lof_threshold,
                lof_n_neighbors=lof_n_neighbors,
                lof_report_path=lof_report_path,
            )
        electrode_tokens_extractor = MultiStftView(**mstft_kwargs)
    _apply_extractor_cache(
        electrode_tokens_extractor, "electrode_tokens", extractor_cache_folder
    )

    # The encoder's ``ref_idx`` token must label the operator the
    # waveform actually saw. When the caller hands in a
    # :class:`RefAugMultiStftView` the operator varies per clip — lift
    # its ``ref_modes`` + ``seed`` so the draws stay aligned via the
    # shared ``(seed, event_key)`` key. Otherwise the operator is the
    # static CAR config baked into the electrode-tokens extractor:
    # collapse the label to that single mode (and reject any CAR that
    # does not map cleanly into ``REF_MODES`` rather than silently
    # mislabelling the operator).
    if isinstance(electrode_tokens_extractor, RefAugMultiStftView):
        ref_modes_for_label = tuple(electrode_tokens_extractor.ref_modes)
        ref_seed_for_label = int(electrode_tokens_extractor.seed)
    else:
        if not hasattr(electrode_tokens_extractor, "car"):
            raise ValueError(
                f"electrode_tokens_extractor {type(electrode_tokens_extractor).__name__} "
                "exposes no 'car' attribute, so the dispatch cannot label "
                "the upstream reference operator. Inherit from "
                "CARIeegExtractor, or use a RefAugMultiStftView for "
                "per-clip ref switching."
            )
        car = electrode_tokens_extractor.car
        reference = getattr(electrode_tokens_extractor, "reference", None)
        if car != "shaft" or reference == "bipolar":
            raise ValueError(
                "electrode_tokens_extractor uses a reference operator that "
                f"does not map into REF_MODES={REF_MODES!r} "
                f"(car={car!r}, reference={reference!r}). The default "
                "dispatch only supports a static shaft-CAR upstream; for "
                "per-clip operator switching use a RefAugMultiStftView. "
                "Either swap to LogStftView(car='shaft'), construct a "
                "RefAugMultiStftView, or pass an explicit ref_idx extractor."
            )
        ref_modes_for_label = ("shaft_car",)
        ref_seed_for_label = seed

    # WS-C / C1: size the encoder's n_time_bins (RoPE ceiling) from clip_len ×
    # the resolved front-end's frame geometry, unless the caller pinned it.
    # (Runs after extractor validation so a malformed extractor fails on the
    # more fundamental `car` check first.)
    if n_time_bins is None:
        if not hasattr(electrode_tokens_extractor, "n_time_bins_for_duration"):
            raise ValueError(
                "n_time_bins could not be derived: the supplied "
                f"electrode_tokens_extractor "
                f"{type(electrode_tokens_extractor).__name__} has no "
                "n_time_bins_for_duration(); pass n_time_bins explicitly."
            )
        n_time_bins = electrode_tokens_extractor.n_time_bins_for_duration(clip_len)

    study = Wang2024Treebank(
        path=Path(bt_root), mode=mode,
        infra_timelines={"cluster": None},
    )
    word_events = BTWordEvents(
        tasks=(task,),
        binary_tasks=binary_tasks,
        lite=(mode == "lite"),
        nano=(mode == "nano"),
        eval_mode=eval_mode,
        test_subject_id=test_subject_id,
        test_trial_id=test_trial_id,
        bt_root=bt_root,
    )
    chain = ns.Chain(steps=[study, word_events])

    dk_extractor = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=bt_root, unmapped_policy="zero",
        c_max=c_max,
    )
    _apply_extractor_cache(dk_extractor, "dk_support", extractor_cache_folder)
    valid_mask_extractor = ElectrodeValidMask(
        event_types="Ieeg", bt_root=bt_root, c_max=c_max,
        unmapped_policy="zero",
    )
    _apply_extractor_cache(valid_mask_extractor, "valid_mask", extractor_cache_folder)

    # B29 Item 11/12 per-clip metadata extractors. Each emits a
    # 1-element TimedArray that the Lightning collator stacks into a
    # ``(B,)`` tensor matching the encoder kwarg contract.
    ref_idx_extractor = RefIdxExtractor(
        event_types="Ieeg",
        seed=ref_seed_for_label,
        ref_modes=ref_modes_for_label,
    )
    _apply_extractor_cache(ref_idx_extractor, "ref_idx", extractor_cache_folder)
    subtype_extractor = SubjectSubtypeExtractor(
        event_types="Ieeg",
        vocab=subtype_embed_vocab,
        corpus="braintreebank",
    )
    _apply_extractor_cache(subtype_extractor, "subject_subtype", extractor_cache_folder)

    # B03 shaft-mask (5/27 PM final spec): student-only paradigm-B
    # electrode drop. Only constructed for the joint SSL phase — the
    # supervised Phase-4 dispatch never reads a shaft-mask (the BrainModule
    # does not exist on that path). The :class:`V14JointBrainModule`
    # routes ``batch.data["shaft_mask"]`` student-only per the B26
    # teacher full-input contract.
    segmenter_extractors: dict[str, tp.Any] = {
        "electrode_tokens": electrode_tokens_extractor,
        "support": dk_extractor,
        "valid_mask": valid_mask_extractor,
        "ref_idx": ref_idx_extractor,
        "subject_subtype": subtype_extractor,
    }
    if joint_phase:
        shaft_mask_extractor = BTShaftMaskExtractor(
            event_types="Ieeg",
            bt_root=bt_root,
            c_max=c_max,
            seed=seed,
        )
        _apply_extractor_cache(
            shaft_mask_extractor, "shaft_mask", extractor_cache_folder
        )
        segmenter_extractors["shaft_mask"] = shaft_mask_extractor

    # WS-H / T20: P3 Whisper teacher target. Built only when a cache dir is
    # passed (P3 dispatch); P1/P2/P4 leave it None so the batch never carries
    # the 1280-d stream. NO _apply_extractor_cache — the whole-movie teacher
    # cache IS the precompute; the extractor only mmap-slices it per clip.
    if whisper_target_cache_dir is not None:
        segmenter_extractors["whisper_target"] = WhisperTargetExtractor(
            cache_dir=whisper_target_cache_dir,
            layer_merge=whisper_layer_merge,
            clip_s=clip_len,
        )

    data = Data(
        study=chain,
        segmenter={
            "extractors": {
                **segmenter_extractors,
                "target": {
                    "name": "EventField",
                    "event_types": "Word",
                    "event_field": "label",
                    "aggregation": "trigger",
                },
            },
            "trigger_query": "type == 'Word'",
            # B36 Δlag: slide the neural clip start by +neural_lag_s so the
            # lagged cortical response aligns to the stimulus-time Whisper
            # teacher. 0.0 = 1:1 baseline. The teacher cache is unaffected
            # (audio-keyed), so a lag sweep needs no recache.
            #
            # WS-H / FLAG 9: this `start` is shared by EVERY extractor the
            # segmenter runs (the dataloader applies one (start, duration) to
            # all), and it is the NEURAL clock (est_idx/sample_rate). The
            # `whisper_target` extractor MUST NOT key off it: the teacher cache
            # is a whole-MOVIE stream indexed by audio movie-time, and the
            # neural vs movie clocks diverge 235-904 s within a single BT trial.
            # WhisperTargetExtractor sidesteps this by reading the trigger Word's
            # MOVIE-clock onset (`movie_onset_s`, threaded from words_df['start']
            # in word_events.py) off the event directly — it never touches this
            # `start`. So the teacher is anchored to movie-audio time while only
            # the NEURAL window slides with Δlag; no double-count, and Δlag=0
            # stays a true 1:1 baseline. (DO NOT "fix" this to est_idx/2048 — the
            # neural clock is exactly the wrong key; that inversion is the trap.)
            "start": neural_lag_s,
            # WS-C / C1: phase-conditional clip window (5 s SSL, 1 s P4).
            "duration": clip_len,
        },
        batch_size=batch_size,
        num_workers=num_workers,
    )

    exca_folder = exca_folder or os.environ.get("EXCA_CACHE_FOLDER")
    infra_cfg: dict[str, tp.Any] = {}
    if exca_folder is not None:
        infra_cfg["folder"] = exca_folder
    # #54 audit C1: exca cache mode. Not part of the config-derived uid, so it
    # does not perturb cache hits — it only governs how a stored FAILURE is
    # handled on relaunch (cached → re-raise; retry → recompute).
    infra_cfg["mode"] = exca_mode
    if cluster is not None:
        infra_cfg["cluster"] = cluster
    if slurm_partition is not None:
        infra_cfg["slurm_partition"] = slurm_partition
    if slurm_account is not None:
        infra_cfg["slurm_account"] = slurm_account
    if mem_gb is not None:
        infra_cfg["mem_gb"] = mem_gb
    if gpus_per_node is not None:
        infra_cfg["gpus_per_node"] = gpus_per_node
    if cpus_per_task is not None:
        infra_cfg["cpus_per_task"] = cpus_per_task
    if timeout_min is not None:
        infra_cfg["timeout_min"] = timeout_min
    # 4-GPU DDP (#33): one srun rank per GPU. exca raises if tasks_per_node>1
    # without slurm_use_srun, so always pair them.
    if tasks_per_node is not None:
        infra_cfg["tasks_per_node"] = tasks_per_node
    if slurm_use_srun:
        infra_cfg["slurm_use_srun"] = slurm_use_srun

    # #21: phase routing is exclusive — joint (P1/P2), P3 distill, and the
    # P4 frozen probe each replace the brain module differently and cannot
    # co-select. Fail at construction rather than silently letting the first
    # branch win.
    if sum((joint_phase, p3_distill, phase4_frozen_probe)) > 1:
        raise ValueError(
            "joint_phase / p3_distill / phase4_frozen_probe are mutually "
            f"exclusive phase selectors; got joint_phase={joint_phase}, "
            f"p3_distill={p3_distill}, phase4_frozen_probe={phase4_frozen_probe}."
        )

    experiment_cls: type[Experiment] = Experiment
    extra_experiment_kwargs: dict[str, tp.Any] = {}
    if joint_phase:
        # Imported here to avoid pulling V14JointExperiment into the
        # supervised Phase-4 path's import chain.
        from speech_decoding.experiments.v14_joint import (
            JOINT_PHASE_VALUE,
            V14JointExperiment,
        )
        experiment_cls = V14JointExperiment
        extra_experiment_kwargs = {
            "phase": JOINT_PHASE_VALUE,
            # B30-dispatch-sister-flags persisted onto the joint
            # experiment so the run-record YAML records the sister
            # choice; the field validators in V14JointExperiment refuse
            # non-default values until the runtime branch lands.
            "latent_valid_override": latent_valid_override,
            "sa_mask_mode": sa_mask_mode,
            # B31 loss-variant selector: 2-term default + 3 sister arms.
            "loss_variant": loss_variant,
            # B36 staged masked-JEPA sub-phase (H4): p1 front-end M2 / p2
            # parcel M4. The staged P1->P2 handoff is WS-E; this picks the stage.
            "jepa_phase": jepa_phase,
            # B36 WS-E E2: P2 front-end discriminative-LR scale.
            "frontend_lr_scale": frontend_lr_scale,
        }
    elif p3_distill:
        # WS-F P3 Whisper distillation. The teacher stream is mandatory — the
        # SmoothL1 loss consumes ``whisper_target``; without it the segmenter
        # emits no target and the P3 step crashes downstream. Fail early + loud.
        if whisper_target_cache_dir is None:
            raise ValueError(
                "p3_distill=True needs whisper_target_cache_dir (the whole-movie "
                "Whisper teacher cache); the P3 SmoothL1 loss has no target "
                "without it. Pass --whisper-target-cache-dir."
            )
        from speech_decoding.experiments.v14_phase3 import V14Phase3Experiment

        experiment_cls = V14Phase3Experiment
        extra_experiment_kwargs = {
            "phase": p3_stage,
            "projector_mode": projector_mode,
            "target_standardize": target_standardize,
            # Validated by V14Phase3Experiment.model_post_init (required when
            # target_standardize=True; the R-no-target-standardize sister sets
            # target_standardize=False and may leave this None).
            "channel_stats_path": channel_stats_path,
            # B33 §5 3b discriminative-LR scales (front-end base·scale, parcel
            # base·scale, connector base). No effect under 3a.
            "frontend_lr_scale": frontend_lr_scale,
            "parcel_lr_scale": parcel_lr_scale,
        }
    elif phase4_frozen_probe:
        # WS-G B35 frozen-encoder readout probe. Carries the transferable-state
        # protocol (encoder + PMA strict load, projector dropped) so it can
        # warm-start from a P3 snapshot; the base supervised Experiment cannot.
        # Gate-D fix: the neural_lag parity guard below lives in the *base*-P4
        # elif, which this branch short-circuits — so re-assert it here. The
        # frozen probe IS the canonical P4 readout (chain P4 + every
        # --resume-from / --frozen-probe run), exactly the path that must keep
        # the leaderboard-parity [onset, onset+1 s] window.
        if neural_lag_s != DEFAULT_NEURAL_LAG_S:
            raise ValueError(
                "neural_lag_s must stay 0.0 on the Phase-4 frozen-probe path: a "
                f"non-zero probe offset (got {neural_lag_s!r}) shifts the "
                "segmenter clip start off the leaderboard-parity "
                "[onset, onset+1 s] window."
            )
        from speech_decoding.experiments.v14_phase4 import (
            V14Phase4ReadoutExperiment,
        )

        experiment_cls = V14Phase4ReadoutExperiment
        extra_experiment_kwargs = {"phase": 4}
    elif (
        latent_valid_override != "support"
        or sa_mask_mode != "bidirectional"
        or loss_variant != DEFAULT_LOSS_VARIANT
        or jepa_phase != DEFAULT_JEPA_PHASE
        or frontend_lr_scale != DEFAULT_FRONTEND_LR_SCALE
        or neural_lag_s != DEFAULT_NEURAL_LAG_S
    ):
        # B30 + B31 + B36 joint-only flags have semantic effect under the
        # joint phase only. The supervised Phase-4 path doesn't run the SSL
        # aggregator, the bidirectional-mask latent-SA branch, a staged
        # masked-JEPA phase, or the P2 discriminative-LR split, so a non-default
        # flag here would silently mis-record the sister / stage. neural_lag_s
        # is blocked here for a different reason: it DOES shift the segmenter
        # window on this path, and a non-zero P4 probe offset breaks the
        # leaderboard-parity [onset, onset+1 s] window.
        raise ValueError(
            "latent_valid_override / sa_mask_mode / loss_variant / jepa_phase "
            "/ frontend_lr_scale / neural_lag_s are B30/B31/B36 joint-phase / "
            "distill selectors only; got "
            f"latent_valid_override={latent_valid_override!r}, "
            f"sa_mask_mode={sa_mask_mode!r}, "
            f"loss_variant={loss_variant!r}, "
            f"jepa_phase={jepa_phase!r}, "
            f"frontend_lr_scale={frontend_lr_scale!r}, "
            f"neural_lag_s={neural_lag_s!r} with joint_phase=False. "
            "Pass --phase 1 (joint) when setting these flags "
            "(neural_lag_s must stay 0.0 on the Phase-4 probe path)."
        )

    # 4-GPU DDP (#33 follow-up): a multi-rank run needs the find-unused DDP
    # strategy (see _resolve_ddp_strategy for the staged-phase rationale).
    # The strategy itself changes no numerics — AdamW skips grad=None params,
    # wd=0 (§7-locked, hard-guarded above). The only multi-rank delta is the
    # standard DDP mean-of-means reweighting of the per-element masked JEPA
    # loss (per-rank masked-cell counts differ): that makes the 4-GPU grad
    # equal to the validated single-GPU + grad-accum fallback (NOT a true
    # bs=N*batch single mean), and it is zero-mean over training. P4's
    # downstream metric is NOT computed under DDP — the chain forces the probe
    # single-GPU (see _build_v14_chain) so trainer.test() sees the full set.
    ddp_strategy = _resolve_ddp_strategy(tasks_per_node)

    return experiment_cls(
        data=data,
        infra=infra_cfg,
        target_field="label",
        brain_model_config={
            "name": "V14ParcelPerceiver",
            "n_freq_bins": n_freq_bins,
            "n_time_bins": n_time_bins,
            "k_parcels": DEFAULT_K_PARCELS,
            "d_model": d_model,
            "n_heads": n_heads,
            "depth_self_attn": depth,
            "m_sub_slots": m_sub_slots,
            "eps": eps,
            "time_last_input": True,
            # B28 cross-attn collapse (default ``[0]``; sister opt-in
            # ``[0, 3]`` for ``R-perceiver-original-2-cross-attns``).
            "cross_attn_positions": cross_attn_positions,
            # B29 Item 11 lock 2026-05-27 PM-late: subtype + ref embeds.
            "subtype_vocab": subtype_vocab_size,
            "subtype_embed_enabled": subtype_embed_enabled,
            "subtype_embed_reuse_kv": subtype_embed_reuse_kv,
            "ref_embed_enabled": ref_embed_enabled,
            "ref_embed_reuse_kv": ref_embed_reuse_kv,
            # B35 (2026-05-31): Phase-4 readout selector threaded onto the
            # model config. V14ParcelPerceiver.build() picks the frozen-PMA
            # readout (pma_*) or an attentive/meanpool sister.
            "readout": readout,
            # 2026-05-30 speedup audit (Tier-2, #119): default-off
            # activation checkpointing on the encoder block stacks.
            "gradient_checkpointing": gradient_checkpointing,
            # SSL-pretrain dispatch flags threaded onto the model config
            # so they ride along with the persisted run record. The
            # supervised downstream classifier path does not branch on
            # them; the SSL trainer reads them from this same snapshot.
            "dkoleo_mode": dkoleo_mode,
            "phase_mode": phase_mode,
            # NOTE: ``loss_variant`` (B31) lives on the V14JointExperiment
            # field via ``extra_experiment_kwargs`` below, NOT on the
            # brain-model config — the brain-model Pydantic schema is
            # ``extra='forbid'``. The run-record YAML still captures it
            # via the Experiment-level snapshot.
        },
        loss={"name": "CrossEntropyLoss"},
        # #37: optimizer/LR-schedule config built above from the optim flags.
        # NOT fused — fused sets _step_supports_amp_scaling=True, which makes
        # Lightning's bf16-mixed AMP plugin refuse the §7-locked grad_clip=1.0
        # (verified: job 47686013 crashed on this combo at the first opt step).
        # neuraltrain BaseTorchOptimizer forwards ``kwargs`` verbatim to
        # ``torch.optim.{Adam,AdamW}``. Applies to BOTH the supervised Phase-4
        # and the joint SSL optimizer.
        optim=optim_cfg,
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=n_epochs,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        limit_val_batches=limit_val_batches,
        collapse_guard=collapse_guard,
        # #66/#67: LR-warmup step count → the guard ignores soft criteria
        # while the LR is still ramping (warmup_steps reaches the optim above).
        guard_warmup_min_step=warmup_steps,
        early_stopping_patience=early_stopping_patience,
        # #37: Lightning gradient clipping. v14 §7 locks grad_clip=1.0 as a
        # non-swept universal; the CLI defaults --grad-clip to 1.0, but this
        # function default is None (the tiny test experiments don't clip).
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        seed=seed,
        # B36 WS-E (E3/E4) cross-phase handoff. On a standalone phase both are
        # None (no-op); the chain driver (run_phase_pipeline) rewrites them via
        # infra.clone_obj, and a manual incremental launch sets them from
        # --resume-from / --snapshot-ckpt-to.
        pretrained_ckpt=pretrained_ckpt,
        snapshot_ckpt_to=snapshot_ckpt_to,
        # Per-clip metadata reaches the encoder forward as additional
        # kwargs alongside (tokens, support, valid_mask).
        x_name=(
            "electrode_tokens", "support", "valid_mask",
            "subject_subtype", "ref_idx",
        ),
        accelerator="auto",
        devices="auto",
        ddp_strategy=ddp_strategy,
        precision=precision,
        fast_dev_run=fast_dev_run,
        **extra_experiment_kwargs,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V14 first-pass DCC dispatch (BT cohort, K=80 DK parcels)."
    )
    p.add_argument("--mode", choices=("nano", "lite", "full"), default="lite")
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="Neuroprobe task name (event field for the target).")
    p.add_argument("--eval-mode", choices=("CrossSession", "CrossSubject"),
                   default=DEFAULT_EVAL_MODE,
                   help="Split policy (CrossSession = submission gate, "
                        "CrossSubject = scientific generalization).")
    p.add_argument("--test-subject-id", type=int, default=DEFAULT_TEST_SUBJECT_ID)
    p.add_argument("--test-trial-id", type=int, default=DEFAULT_TEST_TRIAL_ID)
    p.add_argument("--binary-tasks", action="store_true", default=True,
                   help="(default) Binary label derivation per Neuroprobe leaderboard. "
                        "Pass --no-binary-tasks to switch to 3-class multiclass.")
    p.add_argument("--no-binary-tasks", dest="binary_tasks", action="store_false")
    p.add_argument("--eps", type=float, default=DEFAULT_SUPPORT_BIAS_EPS,
                   help="Vestigial under the B36 hard pool (reserved for the "
                        "gated R-bna-soft routing sister); ignored on the "
                        "default hard-pool path.")
    p.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--m-sub-slots", type=int, default=DEFAULT_M_SUB_SLOTS)
    p.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                   help="DataLoader worker processes (B1.4e, 2026-05-29). "
                        "0 starves the GPU on the per-sample CPU STFT; default "
                        "4 overlaps it. Set --cpus-per-task >= num_workers + 1.")
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS,
                   help="Epoch budget for the SSL/distill phases (P1/P2/P3a/P3b) "
                        "and the P4 cap. The SSL phases train to this fixed "
                        "budget; no early-stop.")
    p.add_argument("--ssl-max-steps", dest="ssl_max_steps", type=int, default=None,
                   help="Optional step budget for the SSL/distill phases "
                        "(P1/P2/P3a/P3b). When set it overrides --n-epochs for "
                        "those phases: training stops at this many optimizer "
                        "steps (max_epochs=-1). Real SSL runs over the "
                        "multi-hundred-hour joint corpus are budgeted in steps, "
                        "not epochs. P4 ignores this.")
    p.add_argument("--ssl-val-check-interval", dest="ssl_val_check_interval",
                   type=int, default=None,
                   help="Validation cadence (optimizer steps) for the SSL/"
                        "distill phases (#54). A step-budgeted phase ends mid-"
                        "epoch, so without this the collapse-guard soft panel "
                        "(RankMe/coverage/no-masking/loss-blowup) never fires. "
                        "Default when --ssl-max-steps is set: max(50, steps//10) "
                        "(~10 checks/phase). Ignored on an epoch budget. NOTE: "
                        "specified in OPTIMIZER steps; the Trainer converts to "
                        "Lightning's micro-batch unit via accumulate_grad_batches "
                        "(#66).")
    p.add_argument("--ssl-limit-val-batches", dest="ssl_limit_val_batches",
                   type=int, default=32,
                   help="Cap each SSL/distill validation to this many batches "
                        "(#66). The collapse-guard panel estimates RankMe/"
                        "coverage from a small sample, so an uncapped val set "
                        "(~875 batches, ~8 min/check) is pure wall-clock waste. "
                        "P4 is uncapped (its val_loss IS the downstream metric). "
                        "<=0 = uncapped.")
    p.add_argument("--no-collapse-guard", dest="collapse_guard",
                   action="store_false", default=True,
                   help="Disarm the collapse/divergence kill-switch (#54) on the "
                        "SSL/distill phases — DIAGNOSTIC ONLY. Lets a run record "
                        "the full RankMe trajectory past the soft-warn streak "
                        "instead of aborting (to confirm a born-low-rank floor "
                        "vs a true decline). P4 never carries the guard anyway.")
    p.add_argument("--p4-early-stop-patience", dest="p4_early_stop_patience",
                   type=int, default=DEFAULT_P4_EARLY_STOP_PATIENCE,
                   help="Early-stopping patience (epochs) on val_loss for the "
                        "P4 frozen probe ONLY — the one phase where val_loss-min "
                        "is the right signal (P4 IS the supervised task). The "
                        "SSL/distill phases never early-stop. Pass a value <= 0 "
                        "to disable and run P4 to the --n-epochs cap.")
    # #37 optimizer / LR-schedule flags (4-agent audit 2026-06-03). Production
    # default FLIPPED 2026-06-04 to the §7 / B01-locked config — AdamW + β2=0.95
    # + linear-warmup → cosine→0 + grad-clip 1.0 — because constant-Adam/β2=0.999
    # was the *unimplemented-default bug*, never a chosen config
    # ([[project_v14_optimizer_default_b01_config_2026_06_04]]). Peak --lr,
    # --warmup-steps, and --weight-decay stay M0-sweep params (reasonable
    # capstone defaults; consolidate/sweep later, #45). A launch guard in main()
    # refuses a real (non-fast-dev-run) SSL/distill run on constant-Adam/β2=0.999
    # so the §7 config can never be silently omitted.
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Peak/base LR (constant value, or warmup-cosine peak). "
                        "1e-3 default = prior behavior; the §7 M0 sweep centers "
                        "are 5e-4 (P1) / 3e-4 (P2) @ batch 1024/512 (√-rule-"
                        "rescale for the live bs=8 before a real run).")
    p.add_argument("--lr-schedule", dest="lr_schedule",
                   choices=("constant", "warmup_cosine"), default="warmup_cosine",
                   help="warmup_cosine (DEFAULT 2026-06-04, §7 locked shape: "
                        "linear warmup → cosine → min_lr_ratio·peak; reads its "
                        "horizon from estimated_stepping_batches, --ssl-max-steps "
                        "pins it) or constant (the prior unimplemented-default "
                        "behavior — refused on a real SSL/distill run by the "
                        "main() launch guard).")
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=0,
                   help="Linear-warmup optimizer steps for --lr-schedule "
                        "warmup_cosine (§7: 20k P1 / 5k P2 @ full corpus; scale "
                        "to the actual step budget). Clamped below total_steps.")
    p.add_argument("--min-lr-ratio", dest="min_lr_ratio", type=float, default=0.0,
                   help="Cosine floor as a fraction of peak LR (0.0 = →0, the "
                        "§7 lock). Per-group-proportional, so P2/P3 "
                        "discriminative-LR ratios survive a non-zero floor.")
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.0,
                   help="AdamW weight decay (0.0 default = prior plain-Adam). §7 "
                        "M0 sweep center is 0.05; only added to the optimizer "
                        "kwargs when > 0 (use --optimizer adamw for decoupled WD).")
    p.add_argument("--optimizer", dest="optimizer_name",
                   choices=("Adam", "AdamW"), default="AdamW",
                   help="AdamW (DEFAULT 2026-06-04, §7 locked family, decoupled "
                        "weight decay pinned to 0.0 until the no-WD groups land) "
                        "or Adam (the prior unimplemented-default — refused on a "
                        "real SSL/distill run by the main() launch guard).")
    p.add_argument("--adam-beta2", dest="adam_beta2", type=float, default=0.95,
                   help="optimizer betas = (0.9, beta2). DEFAULT 0.95 (2026-06-04, "
                        "§7 SSL lock). Pass 0.999 to recover the torch default — "
                        "but a real SSL/distill run refuses β2≥0.999 via the "
                        "main() launch guard.")
    p.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0,
                   help="Lightning gradient_clip_val. Defaults to 1.0 to restore "
                        "the §7 locked-universal grad_clip that was silently OFF "
                        "(top divergence-risk fix per the 6/03 run-health audit). "
                        "Pass <= 0 to disable clipping.")
    p.add_argument("--accumulate-grad-batches", dest="accumulate_grad_batches",
                   type=int, default=1,
                   help="Lightning accumulate_grad_batches. 1 (default) = no "
                        "accumulation. >1 sums grads over N micro-batches → "
                        "effective batch = --batch-size * N at N× wall-clock per "
                        "optimizer step. The effective-batch lever for the 32 GB "
                        "coganlab-gpu cards; a larger physical --batch-size on a "
                        "48 GB+ card is preferred.")
    p.add_argument("--clip-len", type=float, default=None,
                   help="Segmenter clip window (s): 5.0 for SSL P1/P2/P3 "
                        "(T_p=40), 1.0 for the P4 readout (T_p=8). Sizes the "
                        "encoder n_time_bins / RoPE ceiling. Unset → resolves "
                        "to 1.0 for --phase 4 (leaderboard parity, Gate-B "
                        "flag 3) and 5.0 otherwise.")
    p.add_argument("--neural-lag-s", dest="neural_lag_s", type=float,
                   default=DEFAULT_NEURAL_LAG_S,
                   help="B36 Δlag: neural-response lag (s) added to the clip "
                        "start so the lagged cortical response aligns to the "
                        "stimulus-time Whisper teacher. 0.0 (default) = 1:1 "
                        "baseline / falsifier null; P3-distill sweep sisters "
                        "R-distill-lag-{75,150,300}ms. Audio-keyed teacher ⇒ no "
                        "recache. Must stay 0.0 under --phase 4 (leaderboard "
                        "parity); no-op for P1/P2.")
    p.add_argument("--session-robust-z", dest="session_robust_z",
                   action="store_true", default=True,
                   help="C3 (B13): per-(electrode,freq,session) robust-z on the "
                        "default Multi-STFT front-end (fit per session over its "
                        "own full recording in prepare(), applied frozen per "
                        "clip). ON by default — required for any real run.")
    p.add_argument("--no-session-robust-z", dest="session_robust_z",
                   action="store_false",
                   help="Emit the RAW (un-normalized) filterbank. Only for "
                        "plumbing smokes / fast-dev-run — encoder LN absorbs "
                        "scale there; NOT valid for a science run.")
    p.add_argument("--lof-bad-channels", dest="lof_bad_channels",
                   action="store_true", default=False,
                   help="#17 (D2/T1.7): per-session MNE-LOF bad-channel drop on "
                        "the default Multi-STFT front-end (filtered, pre-CAR "
                        "voltage; flagged channels dropped before shaft-CAR). OFF "
                        "by default (cache uid untouched). ON forces drop_bads=True "
                        "and REQUIRES --lof-report-path: the per-subject drop "
                        "counts are written to JSON for Ben review before a scored "
                        "run ('report back how many channels are dropped'). "
                        "Changes the STFT cache uid → forces a rebuild.")
    p.add_argument("--lof-threshold", dest="lof_threshold", type=float,
                   default=1.5,
                   help="MNE-LOF z-score threshold (default 1.5, the L.0 recipe "
                        "amendment value). Lower = more aggressive. In the cache "
                        "uid. Inert unless --lof-bad-channels.")
    p.add_argument("--lof-n-neighbors", dest="lof_n_neighbors", type=int,
                   default=20,
                   help="MNE-LOF neighbor count (default 20). In the cache uid. "
                        "Inert unless --lof-bad-channels.")
    p.add_argument("--lof-report-path", dest="lof_report_path", default=None,
                   help="Where to write the per-session/per-subject LOF drop-count "
                        "JSON (the Ben review gate). REQUIRED when "
                        "--lof-bad-channels is set. Output-location only — NOT in "
                        "the cache uid.")
    p.add_argument("--c-max", dest="c_max", type=int, default=DEFAULT_C_MAX,
                   help=f"#35: padded electrode-slot count C (default "
                        f"{DEFAULT_C_MAX}, covers all 4 Phase-1 corpora / shared "
                        "multi-corpus cache). Pass 256 for BT-only runs (BT raw "
                        "max=256, the exact safe floor) to drop the 128 pure-pad "
                        "slots the per-electrode front-end would otherwise process "
                        "(~33%% wasted FLOPs, an OOM lever). In the extractor-cache "
                        "uid → a non-default value forces a fresh STFT cache "
                        "(recipe-amendment, HB02 re-cost).")
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--cluster", default=None,
                   help="Exca TaskInfra cluster ('slurm' or None for local).")
    p.add_argument("--exca-mode", dest="exca_mode", default=None,
                   choices=("cached", "retry", "force", "read-only"),
                   help="Exca TaskInfra cache mode (#54 audit C1). 'cached' "
                        "RE-RAISES a stored failure on a same-config relaunch "
                        "instead of recomputing — so a collapse-guard abort "
                        "followed by a code-level fix would never re-run. "
                        "'retry' recomputes a cached *error* while keeping "
                        "cached *successes*, so the abort → fix → relaunch loop "
                        "works. DEFAULT (unset): 'retry' for --chain, 'cached' "
                        "for a single phase — a chained capstone is C1-safe "
                        "without the operator remembering the flag. An explicit "
                        "value always wins. Not part of the uid.")
    # Slurm resource knobs (B1.4a, 2026-05-29). Only used when
    # --cluster=slurm; otherwise ignored. Defaults are conservative
    # (None = inherit submitit defaults) — pass explicit values for any
    # real BT run because submitit's ``common`` partition + 2GB-mem
    # fallback OOM-kills the LogStftView extractor prep.
    p.add_argument("--slurm-partition", default=None,
                   help="Slurm partition (e.g. 'scavenger-gpu', "
                        "'coganlab-gpu'). Required for real GPU runs.")
    p.add_argument("--slurm-account", default="coganlab",
                   help="Slurm account. Default 'coganlab' (the lab "
                        "account ht203 is associated with). Required by "
                        "scavenger partitions and matches the access "
                        "list for coganlab-gpu.")
    p.add_argument("--mem-gb", type=float, default=None,
                   help="Memory per task in GB. Lite BT data prep needs "
                        "≥64; Full BT needs ≥128.")
    p.add_argument("--gpus-per-node", type=int, default=None,
                   help="GPUs per node (1 for single-GPU Lite/Full).")
    p.add_argument("--cpus-per-task", type=int, default=None,
                   help="CPUs per task (≥8 recommended for data prep "
                        "parallelism).")
    p.add_argument("--timeout-min", type=int, default=None,
                   help="Slurm timeout in minutes (e.g. 720 = 12h).")
    # 4-GPU DDP (#33). One srun rank per GPU is required for Lightning DDP; a
    # bare --gpus-per-node N leaves tasks_per_node=1 (exca default) → N GPUs in
    # one rank → NCCL-init hang. Leave --tasks-per-node unset and main()
    # auto-sets it = --gpus-per-node (and --slurm-use-srun) for any slurm run
    # with >1 GPU. Pass --tasks-per-node 1 to force the legacy single-rank path.
    p.add_argument("--tasks-per-node", type=int, default=None,
                   help="srun ranks per node (DDP: =--gpus-per-node). Auto-set "
                        "for multi-GPU slurm runs. For a true single-GPU run use "
                        "--gpus-per-node 1 (NOT --tasks-per-node 1, which leaves "
                        "all GPUs in one rank → NCCL hang).")
    p.add_argument("--slurm-use-srun", action="store_true",
                   help="Launch under srun (exca requires it when "
                        "tasks_per_node>1). Auto-enabled for multi-GPU DDP.")
    # Lightning trainer precision. Default 'bf16-mixed' was chosen 2026-05-29
    # after the B31 Lite Phase-4 baseline OOM'd on RTX 5000 Ada (31 GiB) at
    # every batch size tried — factorized per-electrode SA over C=384 padded
    # electrodes at d=256 exhausts fp32 activations. bf16-mixed halves
    # activation memory at the standard transformer-training precision floor.
    # Pass '32-true' (or 'fp32') to restore the prior fp32 default.
    p.add_argument("--precision", default="bf16-mixed",
                   help="Lightning trainer precision. Default 'bf16-mixed' "
                        "per 2026-05-29 OOM diagnosis on RTX 5000 Ada. Pass "
                        "'32-true' or '16-mixed' to override.")
    p.add_argument("--extractor-cache-folder", default=None,
                   help="Two-tier extractor cache root (B1.5, 2026-05-29). "
                        "Each extractor's MapInfra folder gets pointed at "
                        "{root}/{extractor_name}/ so outputs survive across "
                        "Experiment-config changes (precision, batch_size, "
                        "model knobs). On DCC the path comes from the "
                        "EXCA_EXTRACTOR_CACHE_FOLDER env var (scripts/dcc/"
                        "dispatch injects the /work regenerate-cheap cache "
                        "tier; concrete path in CLAUDE.md storage tiering). "
                        "Locally pass --extractor-cache-folder /tmp/v14_cache "
                        "to reproduce the behavior, or leave unset for no "
                        "caching.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config without dispatching.")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Lightning fast-dev-run: 1 batch train+val+test, no checkpoints.")
    # Phase-2 shaft-mask 5/27 PM final spec. Default is
    # ``K = 1 if N_shafts >= 2 else 0`` with ``extent_blocks=("alpha",)``.
    # Supersedes the original ``K=3`` spec and the same-day AM
    # ``min(2, ceil(0.25 * N_shafts))`` fraction spec. Sisters reach K=2
    # / K=3 via ``--shaft-mask-k-override`` together with
    # ``--shaft-mask-extent-blocks``.
    p.add_argument(
        "--shaft-mask-k-override", type=int, default=None,
        help="Override the default ``K = 1 if N_shafts >= 2 else 0`` formula "
             "with a fixed K. Sister R-shaft-K1-explicit: 1 (matches default). "
             "Sister R-shaft-K2: 2. Sister R-shaft-K3-mixed-3block: 3.",
    )
    p.add_argument(
        "--shaft-mask-extent-blocks", default="alpha",
        help="Comma-separated list of active block extents. Default 'alpha'. "
             "Sister R-shaft-K2: 'alpha,beta'. "
             "Sister R-shaft-K3-mixed-3block: 'alpha,beta,gamma'.",
    )
    # B28 DKoleo demotion 2026-05-27 PM + B29 sister-set expansion
    # 2026-05-27 PM-late. ``batch_cls`` is accepted as a pre-B29 alias
    # of ``batch_cls_unit``.
    p.add_argument(
        "--dkoleo-mode",
        choices=(*DKOLEO_MODES, *_DKOLEO_MODE_ALIASES.keys()),
        default=DEFAULT_DKOLEO_MODE,
        help="DKoleo @ M4 unit (B28 2026-05-27 PM demotion + B29 expansion). "
             "'off' (default) drops the term. Sisters: 'intra_clip_slots' "
             "(B21 per-clip × 80 slots), 'batch_cls_unit' (DINOv2-faithful "
             "per-batch × CLS-analog utterance vectors), and "
             "'vicreg_slot_variance' (per-dim VICReg variance hinge gated "
             "on MON-SLOT-REDUNDANCY). 'batch_cls' is a pre-B29 alias of "
             "'batch_cls_unit'.",
    )
    # B29 Item 11 + 5/28 PM flip: subtype default OFF, so CLI flag enables.
    p.add_argument(
        "--subtype-embed", dest="subtype_embed_enabled",
        action="store_true", default=False,
        help="Enable the per-clip subject-subtype embedding "
             "(R-subtype-embed-on-with-kv-reuse P0 sister; default OFF per "
             "5/28 PM precedent-audit flip).",
    )
    p.add_argument(
        "--no-subtype-embed-reuse-kv", dest="subtype_embed_reuse_kv",
        action="store_false", default=True,
        help="Keep the subtype embed input-only (skip K/V reuse, M3AE-"
             "faithful). Sister R-subtype-embed-input-only P0.",
    )
    p.add_argument(
        "--subtype-embed-vocab",
        choices=SUBTYPE_EMBED_VOCABS, default=DEFAULT_SUBTYPE_EMBED_VOCAB,
        help="Subtype embed vocab: 'binary' (default, sEEG/iEEG vs ECoG) "
             "or 'three_way' (sEEG / iEEG-surface / ECoG).",
    )
    p.add_argument(
        "--ref-embed", dest="ref_embed_enabled",
        action="store_true", default=False,
        help="Enable the per-clip reference-operator embedding (B32 "
             "first-pass-no-input-aug default = OFF; pair with "
             "`--ref-aug` / RefAugMultiStftView to re-enable as the "
             "`R-ref-aug-3-cell` P1 sister).",
    )
    p.add_argument(
        "--no-ref-embed-reuse-kv", dest="ref_embed_reuse_kv",
        action="store_false", default=True,
        help="Keep the ref embed input-only (skip K/V reuse). "
             "Sister R-ref-embed-input-only.",
    )
    # B35 (2026-05-31): Phase-4 readout selector (reverts B34).
    # "pma_mean_linear" (default) = frozen P3-PMA collapse → mean-over-time
    # → Linear; "pma_flatten_linear"/"pma_timeattn_linear" vary the
    # temporal op; "attentive"/"meanpool" are the B34 sisters.
    p.add_argument(
        "--readout", dest="readout",
        choices=[
            "pma_mean_linear", "pma_flatten_linear", "pma_timeattn_linear",
            "attentive", "meanpool",
        ],
        default="pma_mean_linear",
        help="Phase-4 readout. 'pma_mean_linear' (default) = frozen P3-PMA "
             "collapses parcels → (B, T_p, d), then mean-over-time → Linear "
             "(only the linear trains). 'pma_flatten_linear' (R-p4-flatten) "
             "/ 'pma_timeattn_linear' (R-p4-time-attn-pool) vary the temporal "
             "op. 'attentive' (R-p4-attentive) = single-query attentive probe "
             "over (B, L, T, d); 'meanpool' (R-p4-meanpool-no-pma) = masked "
             "mean-pool over parcel×time, skipping the PMA.",
    )
    # 2026-05-30 speedup audit (Tier-2, #119): default-off activation
    # checkpointing on the encoder block stacks. Trades ~one extra encoder
    # forward for the activation memory of the deepest stack — insurance
    # for the bs=32 OOM, off by default since DeltaAI H100 80GB does not
    # bind on memory. Numerics-safe (bit-identical loss + grads).
    p.add_argument(
        "--gradient-checkpointing", dest="gradient_checkpointing",
        action="store_true", default=False,
        help="Enable activation checkpointing on the encoder token-block and "
             "latent-block stacks. Off by default; gated on training + grad "
             "so the no_grad EMA-teacher pass is never checkpointed.",
    )
    p.add_argument(
        "--phase-mode", choices=PHASE_MODES, default=DEFAULT_PHASE_MODE,
        help="B36 staged regime: split P1 (front-end M2) -> P2 (parcel M4) "
             "('split_p1_p2', default) vs the B29 single-joint-phase falsifier "
             "('joint_b29', sister R-joint-ssl). Recorded-only run-record "
             "metadata; the behavioral stage is selected via --jepa-phase, "
             "not this axis.",
    )
    p.add_argument(
        "--jepa-phase", choices=JEPA_PHASES, default=DEFAULT_JEPA_PHASE,
        help="B36 staged masked-JEPA sub-phase (joint SSL / --phase 1 only). "
             "'p1' (default) = front-end M2 masked prediction (all corpora); "
             "'p2' = parcel M4 masked prediction (anatomy corpora, front-end "
             "LR/10). The staged P1->P2 checkpoint handoff is WS-E; this flag "
             "selects which stage trains. A non-default raises under --phase 4.",
    )
    p.add_argument(
        "--p2-frontend-lr-scale", dest="frontend_lr_scale", type=float,
        default=DEFAULT_FRONTEND_LR_SCALE,
        help="B36 WS-E E2: P2 front-end discriminative-LR scale (joint SSL / "
             "--phase 1, jepa-phase p2 only). 0.1 (default) = base/10; 0.2 = "
             "R-p2-frontend-lr-5 (base/5); 0.0 = R-p2-freeze-frontend "
             "(front-end frozen, parcel side trains alone). A non-default "
             "raises under --phase 4.",
    )
    p.add_argument(
        "--latent-valid-override",
        choices=("support", "all_true", "parcels_supervised"),
        default="support",
        help="B30 sister selector for the latent-validity mask source. "
             "'support' (default) is the B30 lock; 'all_true' is "
             "R-item-12-all-true P0; 'parcels_supervised' is "
             "R-parcels-supervised-gating (retired-into-default falsifier). "
             "Sisters raise NotImplementedError until B2.2 lands.",
    )
    p.add_argument(
        "--sa-mask-mode",
        choices=("bidirectional", "key_only"),
        default="bidirectional",
        help="B30 sister selector for the latent self-attention mask "
             "shape. 'bidirectional' (default) is the B30 lock; "
             "'key_only' is R-sa-key-only P1. Sister raises "
             "NotImplementedError until the encoder branch lands.",
    )
    p.add_argument(
        "--loss-variant",
        choices=LOSS_VARIANTS, default=DEFAULT_LOSS_VARIANT,
        help="B31 V-JEPA-2-canonical 2-term joint SSL selector. "
             "'b31_default' (default) = L_pre_frame @ M2 + L_post_frame "
             "@ LN_frame(M4), both pure L1 per V-JEPA 2 §2.1 Eq 1. "
             "'b31_plus_m3' adds L_mid_slot (R-add-m3-loss P0); "
             "'b31_plus_utt' adds L_post_utterance (R-add-utterance-loss "
             "P0, EAT-faithful, ≥0.02 AUROC promotion gate); "
             "'b31_plus_both' adds both (R-add-both P0). Joint phase only.",
    )
    p.add_argument(
        "--ref-operator-alpha", type=float,
        default=DEFAULT_REF_OPERATOR_ALPHA,
        help="α-hierarchical corpus sampler weight (B29 Item 5). Default 0.3.",
    )
    p.add_argument(
        "--no-include-ajile12", dest="include_ajile12",
        action="store_false", default=DEFAULT_INCLUDE_AJILE12,
        help="Drop AJILE12 from the pretraining mix (sister "
             "R-no-ajile12). Default is to include it (B29 reversal of the "
             "5/27 PM-late same-day sensor-gap drop).",
    )
    p.add_argument(
        "--ffn-variant",
        choices=FFN_VARIANTS, default=DEFAULT_FFN_VARIANT,
        help="FFN variant. Default 'dense' (B29 Item 14 + MoE audit "
             "2026-05-28). 'soft_moe_4' raises NotImplementedError until "
             "models/soft_moe.py lands (R-moe-ffn-soft-4 P2 if-budget).",
    )
    # B28 cross-attn collapse 2026-05-27 PM.
    p.add_argument(
        "--cross-attn-positions", default=None,
        help="Comma-separated latent-stack positions for cross-attn blocks. "
             "Default (omitted) = [0] (Perceiver IO canonical). Sister "
             "R-perceiver-original-2-cross-attns: '0,3'. Position 0 is "
             "required; interior positions must satisfy p < depth_self_attn.",
    )
    # MASK-01 per-corpus mains-notch field.
    p.add_argument(
        "--mains-notch-hz", type=float, default=DEFAULT_MAINS_NOTCH_HZ,
        help="Mains-frequency notch (Hz). Default 60.0 (US — BT, D-cohort, "
             "AJILE12). Pass 50.0 for SWEC (Swiss site). Per-corpus map "
             "lives in MAINS_NOTCH_BY_CORPUS.",
    )
    p.add_argument("--phase", type=int, choices=(1, 2, 3, 4), default=4,
                   help="Training phase per docs/neuroprobe/plan.md §staged. "
                        "1 = masked-JEPA SSL (V14JointExperiment); B36 staged "
                        "P1->P2 selected via --jepa-phase (p1 front-end M2 / "
                        "p2 parcel M4). "
                        "2 = legacy split-P2 (raises — use --phase 1 --jepa-phase p2). "
                        "3 = Whisper all-layer-mean distillation (module wired; "
                        "raises the whisper_target data blocker until WS-H). "
                        "4 = downstream linear/finetune probe (current behavior).")
    p.add_argument("--p3-stage", choices=("3a", "3b"), default="3a",
                   help="Phase-3 sub-stage (only with --phase 3): '3a' freezes "
                        "the encoder for the PMA+projector connector warmup; "
                        "'3b' unfreezes it with a discriminative LR "
                        "(front-end base/10, parcel base/3, connector base). "
                        "Default 3a.")
    # WS-F P3 distillation data (#21). The teacher cache is the whole-movie
    # Whisper stream; channel stats are the train-only per-channel z-score.
    p.add_argument("--whisper-target-cache-dir", default=None,
                   help="Root of the whole-movie Whisper teacher cache "
                        "(scripts/neuroprobe/build_bt_teacher_cache.py). REQUIRED "
                        "for --phase 3 / --chain; the P3 SmoothL1 loss consumes "
                        "the per-clip whisper_target it slices.")
    p.add_argument("--whisper-layer-merge", choices=("mean_all", "8"),
                   default="mean_all",
                   help="Whisper teacher layer merge; must match the cache build. "
                        "'mean_all' (locked default) / '8' = R-whisper-single-layer-L8.")
    p.add_argument("--channel-stats-path", default=None,
                   help="Train-only per-channel target z-score stats "
                        "({'mean','inv_std'} from fit_channel_stats). REQUIRED for "
                        "--phase 3 / --chain unless --no-target-standardize.")
    p.add_argument("--no-target-standardize", dest="target_standardize",
                   action="store_false", default=True,
                   help="R-no-target-standardize sister: distill against the raw "
                        "1280-d Whisper target (no per-channel z-score).")
    # WS-G P4 frozen probe + cross-phase handoff (#21).
    p.add_argument("--frozen-probe", action="store_true", default=False,
                   help="--phase 4 only: build the B35 frozen-encoder readout "
                        "probe (V14Phase4ReadoutExperiment, transferable-state "
                        "protocol) instead of the from-scratch supervised CE "
                        "classifier. Implied when --resume-from is set on P4.")
    p.add_argument("--resume-from", default=None,
                   help="Warm-start this phase's encoder (+PMA from P3) from a "
                        "prior phase's transferable-state snapshot "
                        "(Experiment.pretrained_ckpt). Lets the chain run "
                        "incrementally, one verified sbatch per phase.")
    p.add_argument("--snapshot-ckpt-to", default=None,
                   help="After test, write this phase's transferable state to PATH "
                        "(Experiment.snapshot_ckpt_to) for the next phase to load.")
    # #21 full in-process chain driver.
    p.add_argument("--chain", action="store_true", default=False,
                   help="Run the full P1->P2->P3a->P3b->P4 pipeline in one "
                        "process with checkpoint handoff (run_phase_pipeline). "
                        "Overrides --phase. Needs --work-dir, "
                        "--whisper-target-cache-dir, and --channel-stats-path "
                        "(unless --no-target-standardize).")
    p.add_argument("--work-dir", default=None,
                   help="Directory for the --chain per-phase ckpt handoff "
                        "(phase_{i}.ckpt). Required with --chain.")
    return p


# B2.1 (#96) closed 2026-05-28: phase=1 routes to V14JointExperiment (B29
# Item 1). The surviving B2.x sister-gating fires at *construction* —
# ``V14JointExperiment.model_post_init`` (quarantined latent_valid /
# sa_mask_mode / loss_variant) and ``_build_brain_module`` (encoder-slot
# check) — NOT from a ``_train_and_test`` override, which the joint subclass
# does not define. The dispatch path is thus the construction gate only.
# (E5 2026-06-03: removed the dead ``_PHASE1_BLOCKERS`` tuple — never
# referenced from the dispatch.)
_PHASE2_BLOCKERS = (
    # B29 Item 1 collapsed P1 + P2 into a single joint phase; dispatch
    # via ``--phase 1`` (V14JointExperiment is pinned to ``phase=1``).
    # The R-keep-phase-split sister keeps explicit P1/P2 via the parent
    # V14Experiment — but is not exposed through this dispatch.
    "B29 Item 1 collapsed P1 + P2 into a single joint phase; "
    "dispatch via --phase 1 (V14JointExperiment). "
    "Sister R-keep-phase-split keeps explicit P1/P2 via the parent "
    "V14Experiment — see docs/neuroprobe/v14_blockers.md."
)
# E5/WS-F/#21 2026-06-03: ``_PHASE3_BLOCKERS`` removed. The whisper_target
# segmenter emission (WS-H, #20) landed, so --phase 3 now routes to
# V14Phase3Experiment via build_v14_experiment(p3_distill=True) and the chain
# driver runs P3a/P3b end-to-end. (``_PHASE1_BLOCKERS`` was removed in E5; only
# ``_PHASE2_BLOCKERS`` survives — phase 2 is the collapsed legacy split.)


def construct_v14_joint_callbacks(
    *,
    probe_dataloader: tp.Optional[tp.Iterable[tp.Any]] = None,
    probe_mode: tp.Literal["regression", "binary_classification"] = "binary_classification",
    co_save_ema_teacher: bool = True,
    seed: int = 0,
) -> list[tp.Any]:
    """Build the dispatch-time callbacks for V14JointBrainModule.

    Currently a single callback: the S06 best-val probe-r² / AUROC. The
    periodic MON-* canaries (slot-redundancy, sensor-type, ref-type,
    head-balance, MON-MASK-004) are stateless pure functions; the
    canary cadence + sustain-window state lives in the training loop's
    Lightning callbacks once the multi-corpus loader build supplies the
    probe-batch sources.

    When ``probe_dataloader`` is ``None`` the function returns an empty
    list — the joint module still runs MON-MASK-002 inline (B03d), but
    nothing else is registered. This keeps the dispatch path unblocked
    for the BT-Lite sister cell (which has no probe loader yet).

    Args:
        probe_dataloader: held-out probe DataLoader (one batch per
            validation epoch). Items yield ``{"electrode_tokens",
            "support", "probe_label"}``. S06 default: speech-onset
            binary on sub-1, M07 session held-out, 5-fold CV.
        probe_mode: ``"regression"`` (r²) or ``"binary_classification"``
            (AUROC). S06 default is binary (speech-onset).
        co_save_ema_teacher: S06 §3 lock — every checkpoint carries the
            EMA teacher state.
        seed: deterministic fold permutation seed.
    """
    from speech_decoding.experiments.best_val_probe import BestValProbeR2Callback

    callbacks: list[tp.Any] = []
    if probe_dataloader is not None:
        callbacks.append(
            BestValProbeR2Callback(
                probe_dataloader=probe_dataloader,
                mode=probe_mode,
                co_save_ema_teacher=co_save_ema_teacher,
                seed=seed,
            )
        )
    return callbacks


def _common_build_kwargs(
    args, *, cross_attn_positions: list[int] | None,
) -> dict[str, tp.Any]:
    """Knobs identical across every phase in BOTH the single-phase ``main()``
    build and the ``--chain`` builds.

    Centralized after the Gate-D audit: ``binary_tasks`` / ``latent_valid_override``
    / ``sa_mask_mode`` / ``loss_variant`` had drifted OUT of the chain's inline
    ``common`` dict while the single-phase call still passed them, so a
    ``--chain --loss-variant X`` (or any of those) silently ran the DEFAULT arm
    while the run summary printed the sister as applied. Both call sites consume
    this one dict, so a forgotten flag is now structurally impossible.

    Excludes the per-phase selectors (``joint_phase`` / ``p3_distill`` /
    ``phase4_frozen_probe`` / ``p3_stage`` / ``jepa_phase`` / ``clip_len`` /
    ``neural_lag_s`` / ``frontend_lr_scale`` / the whisper + handoff knobs),
    which each call site sets explicitly.
    """
    return dict(
        mode=args.mode, task=args.task, seed=args.seed,
        eval_mode=args.eval_mode,
        exca_mode=args.exca_mode,
        test_subject_id=args.test_subject_id,
        test_trial_id=args.test_trial_id,
        binary_tasks=args.binary_tasks,
        session_robust_z=args.session_robust_z,
        eps=args.eps, d_model=args.d_model, depth=args.depth,
        n_heads=args.n_heads, m_sub_slots=args.m_sub_slots,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_epochs=args.n_epochs,
        cluster=args.cluster, fast_dev_run=args.fast_dev_run,
        slurm_partition=args.slurm_partition,
        slurm_account=args.slurm_account,
        mem_gb=args.mem_gb,
        gpus_per_node=args.gpus_per_node,
        cpus_per_task=args.cpus_per_task,
        timeout_min=args.timeout_min,
        # 4-GPU DDP (#33): reaches every phase via this one dict so the chain +
        # single-phase builds stay in lock-step on the srun-rank topology.
        tasks_per_node=args.tasks_per_node,
        slurm_use_srun=args.slurm_use_srun,
        precision=args.precision,
        extractor_cache_folder=args.extractor_cache_folder,
        dkoleo_mode=args.dkoleo_mode,
        cross_attn_positions=cross_attn_positions,
        mains_notch_hz=args.mains_notch_hz,
        # #17 MNE-LOF bad-channel drop (default OFF). Reaches every phase via this
        # one dict so the chain + single-phase builds stay in lock-step.
        lof_bad_channels=args.lof_bad_channels,
        lof_threshold=args.lof_threshold,
        lof_n_neighbors=args.lof_n_neighbors,
        lof_report_path=args.lof_report_path,
        # #35: padded electrode-slot count. Reaches every phase via this one dict
        # so the chain's 4 c_max-padded extractors stay in lock-step.
        c_max=args.c_max,
        subtype_embed_enabled=args.subtype_embed_enabled,
        subtype_embed_reuse_kv=args.subtype_embed_reuse_kv,
        subtype_embed_vocab=args.subtype_embed_vocab,
        ref_embed_enabled=args.ref_embed_enabled,
        ref_embed_reuse_kv=args.ref_embed_reuse_kv,
        phase_mode=args.phase_mode,
        ref_operator_alpha=args.ref_operator_alpha,
        include_ajile12=args.include_ajile12,
        ffn_variant=args.ffn_variant,
        gradient_checkpointing=args.gradient_checkpointing,
        readout=args.readout,
        latent_valid_override=args.latent_valid_override,
        sa_mask_mode=args.sa_mask_mode,
        loss_variant=args.loss_variant,
        # #37 optim / LR-schedule (audit 2026-06-03). --adam-beta2 → (0.9, β2)
        # tuple; --grad-clip <= 0 → None (disable). Reaches every phase via this
        # one dict, so the chain and single-phase builds stay in lock-step.
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer_name,
        adam_betas=(0.9, args.adam_beta2) if args.adam_beta2 is not None else None,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )


def _validate_channel_stats_path(args) -> None:
    """Fast-fail a missing/non-file channel-stats PATH at dispatch.

    The stats are only ``torch.load``-ed in ``V14Phase3Experiment.
    _build_standardizer`` — i.e. at P3a, which on a --chain run is hours into
    P1+P2 — so a missing/typo'd path (or the stats *directory* passed instead of
    the .pt file) would otherwise burn the whole run before crashing (audit
    A4-HIGH#2b). ``.is_file()`` (not ``.exists()``) matches the codebase's
    loadable-file convention (experiment.py ckpt, whisper_target cache) and
    catches the directory mistake too. Only meaningful with target-std ON and a
    path supplied (the ``is None`` case is each caller's separate guard).
    """
    if (
        args.target_standardize
        and args.channel_stats_path is not None
        and not Path(args.channel_stats_path).is_file()
    ):
        raise ValueError(
            f"--channel-stats-path {args.channel_stats_path!r} is not a file. "
            "Target standardization (B33 default) loads it at P3a; a missing or "
            "directory path would crash hours into the chain. Build the .pt with "
            "the channel_stats fit helper, or pass --no-target-standardize."
        )


def _build_v14_chain(
    args, *, cross_attn_positions: list[int] | None,
) -> list[Experiment]:
    """Assemble the 5 staged experiments [P1, P2, P3a, P3b, P4] for the chain.

    Every phase shares the model/eval/cluster knobs from ``args``; only the
    phase selector, clip window, and (for P3) the teacher stream differ. The
    cross-phase ckpt handoff is NOT set here — ``run_phase_pipeline`` rewrites
    ``snapshot_ckpt_to`` / ``pretrained_ckpt`` per boundary via
    ``infra.clone_obj`` so each phase snapshots to / loads from its neighbour.

    Clip window: 5 s (T_p=40) for the SSL + distill phases so the student grid
    meets the Whisper teacher's 40-frame pool; 1 s (T_p=8) for the P4 readout
    (leaderboard parity). P4 forces ``neural_lag_s=0.0`` — a non-zero probe
    offset breaks the [onset, onset+1 s] parity window (Gate-B flag 3).
    """
    if args.work_dir is None:
        raise ValueError("--chain needs --work-dir for the per-phase ckpt handoff.")
    if args.whisper_target_cache_dir is None:
        raise ValueError(
            "--chain needs --whisper-target-cache-dir (the P3 stages distill "
            "against the Whisper teacher stream)."
        )
    if args.target_standardize and args.channel_stats_path is None:
        raise ValueError(
            "--chain with target standardization (B33 default) needs "
            "--channel-stats-path; pass --no-target-standardize to distill "
            "against the raw 1280-d target instead."
        )
    _validate_channel_stats_path(args)

    common = _common_build_kwargs(args, cross_attn_positions=cross_attn_positions)
    whisper = dict(
        whisper_target_cache_dir=args.whisper_target_cache_dir,
        whisper_layer_merge=args.whisper_layer_merge,
        channel_stats_path=args.channel_stats_path,
        target_standardize=args.target_standardize,
    )
    # SSL/distill phases (P1/P2/P3a/P3b): fixed budget, NEVER early-stop (their
    # val loss is a pretext objective, not the downstream metric).
    # ``--ssl-max-steps`` (when set) budgets them in steps instead of --n-epochs.
    # #54: a step-budgeted phase ends mid-epoch, so force a validation cadence
    # (≈10 checks across the phase) — else the collapse-guard soft panel, which
    # fires on validation, never evaluates. Resolved here so the run summary is
    # honest; --ssl-val-check-interval overrides. Stays None on an epoch budget
    # (epoch-boundary validation already runs).
    ssl_val_check = args.ssl_val_check_interval
    if ssl_val_check is None and args.ssl_max_steps is not None:
        ssl_val_check = max(50, args.ssl_max_steps // 10)
    # #66: cap each SSL validation (P4 stays uncapped — built separately below).
    ssl_limit_val = (
        args.ssl_limit_val_batches
        if args.ssl_limit_val_batches and args.ssl_limit_val_batches > 0
        else None
    )
    # Shared SSL/distill-phase kwargs (P1/P2/P3a/P3b — NOT P4). collapse_guard
    # rides here so --no-collapse-guard disarms all four at once; P4 keeps its
    # own explicit collapse_guard=False (frozen probe, #54 audit M1).
    ssl_budget: dict[str, tp.Any] = dict(
        max_steps=args.ssl_max_steps,
        val_check_interval=ssl_val_check,
        limit_val_batches=ssl_limit_val,
        collapse_guard=args.collapse_guard,
    )
    p1 = build_v14_experiment(
        **common, **ssl_budget, joint_phase=True, jepa_phase="p1", clip_len=5.0,
        neural_lag_s=args.neural_lag_s,
    )
    p2 = build_v14_experiment(
        **common, **ssl_budget, joint_phase=True, jepa_phase="p2", clip_len=5.0,
        frontend_lr_scale=args.frontend_lr_scale, neural_lag_s=args.neural_lag_s,
    )
    p3a = build_v14_experiment(
        **common, **ssl_budget, **whisper, p3_distill=True, p3_stage="3a",
        clip_len=5.0,
        frontend_lr_scale=args.frontend_lr_scale, neural_lag_s=args.neural_lag_s,
    )
    p3b = build_v14_experiment(
        **common, **ssl_budget, **whisper, p3_distill=True, p3_stage="3b",
        clip_len=5.0,
        frontend_lr_scale=args.frontend_lr_scale, neural_lag_s=args.neural_lag_s,
    )
    # binary_tasks now rides in `common` (reaches every phase, so the SSL /
    # distill clip population matches the P4 eval set); P4 only overrides the
    # parity window + zero lag. P4 is the one phase that early-stops on val_loss
    # (it IS the supervised task); <=0 disables and runs to the --n-epochs cap.
    p4_patience = (
        args.p4_early_stop_patience if args.p4_early_stop_patience > 0 else None
    )
    # 4-GPU DDP audit (2026-06-04, 4-agent): force the P4 probe single-GPU when
    # the SSL phases run multi-rank DDP. P4 is a frozen-encoder linear probe and
    # its AUROC/acc is the one number the chain produces — but under multi-rank
    # DDP, trainer.test() computes the metric over only rank-0's ~1/N shard (no
    # all_gather / sync_dist), and AUROC is non-decomposable, so the reported
    # value != the full-set metric. The probe is tiny, so multi-GPU buys nothing;
    # single-GPU gives the full test set (no DistributedSampler) and a correct
    # number. ``tasks_per_node=None`` makes _resolve_ddp_strategy return None
    # inside build. Only diverges from the SSL phases under real DDP — a
    # single-GPU / local chain leaves P4 identical (no behaviour change).
    p4_common = dict(common)
    if common.get("tasks_per_node") and common["tasks_per_node"] > 1:
        p4_common["gpus_per_node"] = 1
        p4_common["tasks_per_node"] = None
        p4_common["slurm_use_srun"] = False
    p4 = build_v14_experiment(
        **p4_common, phase4_frozen_probe=True,
        clip_len=1.0, neural_lag_s=0.0,
        early_stopping_patience=p4_patience,
        # #54 audit M1: no collapse guard on the supervised frozen probe — it
        # has EarlyStopping on the real downstream val_loss, and a frozen linear
        # head can't dimensionally collapse.
        collapse_guard=False,
    )
    return [p1, p2, p3a, p3b, p4]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # §7 launch guard (2026-06-04, project_v14_optimizer_default_b01_config). A
    # real SSL/distill run MUST use the locked optimizer config — constant-Adam /
    # β2=0.999 was the unimplemented-default *bug*, never a chosen config. A
    # non-fast-dev-run P1/P2/P3 (or any --chain) refuses to launch on it. This is
    # a backstop on top of the now-locked dispatch defaults (warmup_cosine +
    # AdamW + β2=0.95): it stops a stale launcher or copy-pasted flag from
    # silently shipping the §7-violating config on a multi-hour run.
    # --fast-dev-run is the smoke escape hatch (config shape is irrelevant to a
    # 1-step sanity run); --dry-run never trains (it prints the run summary and
    # short-circuits before any build), so it is a config preview, not a real
    # launch — both are exempt, matching Ben's "real (non-fast-dev-run) run"
    # framing. --phase 4 is the supervised probe, not SSL/distill.
    _is_ssl_distill = bool(args.chain) or args.phase in (1, 2, 3)
    if _is_ssl_distill and not args.fast_dev_run and not args.dry_run:
        _eff_beta2 = args.adam_beta2 if args.adam_beta2 is not None else 0.999
        _bad: list[str] = []
        if args.optimizer_name == "Adam":
            _bad.append("--optimizer Adam (use AdamW)")
        if args.lr_schedule == "constant":
            _bad.append("--lr-schedule constant (use warmup_cosine)")
        if _eff_beta2 >= 0.999:
            _bad.append(f"β2={_eff_beta2} (use --adam-beta2 0.95)")
        # warmup_cosine with too-short a warmup ramps to PEAK LR within the first
        # optimizer step of a cold-init SSL model — the §7 lock specifies a linear
        # warmup, and a near-instant ramp is the genuine divergence risk
        # (readiness-audit finding b). The ramp multiplier is (step+1)/warmup_steps,
        # so warmup_steps in {0,1} both put the model at peak on step 0 — a bare
        # `> 0` check is bypassable by --warmup-steps 1 (audit finding A1). Enforce
        # a real ramp: when the step budget is known, require >= 1% of it (the
        # locked recipe is 10% = 150/1500, so this admits the M0-sweep range and
        # rejects the step-1-peak pathology); else an absolute floor of 10 steps.
        # The length stays an M0-sweep param above the floor. --fast-dev-run /
        # --dry-run are already exempt (outer guard).
        if args.lr_schedule == "warmup_cosine":
            _warmup_floor = (
                max(10, args.ssl_max_steps // 100) if args.ssl_max_steps else 10
            )
            if args.warmup_steps < _warmup_floor:
                _bad.append(
                    f"warmup_cosine + --warmup-steps {args.warmup_steps} is too "
                    f"short to ramp (< {_warmup_floor}; a cold-init SSL model hits "
                    "peak LR within the first optimizer step → divergence risk; "
                    "pass ~10% of --ssl-max-steps)"
                )
        if _bad:
            raise SystemExit(
                "refusing a real SSL/distill run on the unimplemented-default "
                "optimizer config (the §7/B01 lock would be violated): "
                + "; ".join(_bad)
                + ". The dispatch defaults are AdamW + warmup_cosine + β2=0.95 — "
                "pass them explicitly, or --fast-dev-run for a smoke. See "
                "project_v14_optimizer_default_b01_config_2026_06_04."
            )
    # #54 audit C1: a chained run that aborts a phase (collapse-guard) and is
    # then relaunched with a code fix would, under the 'cached' default,
    # re-raise the stored failure instead of recomputing. Default an unset
    # --exca-mode to 'retry' for --chain (recompute cached errors, keep cached
    # successes) so the abort → fix → relaunch loop is safe without the
    # operator remembering the flag; a single phase stays 'cached'. An explicit
    # --exca-mode always wins.
    if args.exca_mode is None:
        args.exca_mode = "retry" if args.chain else "cached"
    # 4-GPU DDP enablement (#33). exca/submitit default tasks_per_node=1, so a
    # bare --gpus-per-node N allocates N GPUs to ONE srun task; Lightning's
    # SLURMEnvironment then waits for N ranks that never start and hangs at NCCL
    # init. Real DDP needs one srun rank per GPU: tasks_per_node = gpus_per_node
    # and slurm_use_srun=True (exca hard-requires the latter when
    # tasks_per_node>1). Only applies to a slurm run with >1 GPU; single-GPU and
    # local paths keep tasks_per_node=1 / no-srun. An explicit --tasks-per-node
    # wins. (The TRUE single-GPU escape hatch is --gpus-per-node 1, NOT
    # --tasks-per-node 1: the latter leaves N GPUs in one rank → the NCCL hang.)
    if args.cluster == "slurm" and args.gpus_per_node and args.gpus_per_node > 1:
        if args.tasks_per_node is None:
            args.tasks_per_node = args.gpus_per_node
        if args.tasks_per_node > 1:
            args.slurm_use_srun = True
    # Safety net (audit 2026-06-04): warn loudly if a multi-GPU run that submits
    # to slurm did NOT end up with real DDP topology (tasks_per_node>1 + srun).
    # The auto-resolve above fires only for --cluster slurm; --cluster auto and
    # an explicit --tasks-per-node 1 both leave N GPUs in ONE rank → Lightning
    # hangs at NCCL init. A loud line beats a silent multi-hour hang.
    if (args.cluster in ("slurm", "auto") and args.gpus_per_node
            and args.gpus_per_node > 1
            and not (args.tasks_per_node and args.tasks_per_node > 1
                     and args.slurm_use_srun)):
        print(
            f"WARNING: --gpus-per-node {args.gpus_per_node} but DDP topology is "
            f"NOT enabled (tasks_per_node={args.tasks_per_node}, "
            f"slurm_use_srun={args.slurm_use_srun}). All {args.gpus_per_node} "
            "GPUs would go to ONE rank → Lightning hangs at NCCL init. Use "
            "--cluster slurm (auto-enables DDP) or --gpus-per-node 1 for a true "
            "single-GPU run."
        )
    # Gate-B flag 3 / Gate-D fix: phase-couple the clip window when the operator
    # leaves --clip-len unset. A single --phase 4 run that defaulted to 5 s would
    # silently run the readout off the leaderboard-parity 1 s window. --chain
    # ignores this (it sets each phase's clip explicitly), but resolving here
    # keeps the run summary honest.
    if args.clip_len is None:
        args.clip_len = 1.0 if args.phase == 4 else DEFAULT_CLIP_LEN_S
    if args.phase == 2 and not args.chain:
        # Phase 2 is the legacy split-P2 entry-point, collapsed into the joint
        # phase by B29 Item 1; dispatch P1∪P2 via --phase 1 --jepa-phase p2
        # (V14JointExperiment). --phase 1 falls through to construct
        # V14JointExperiment; its B2.x sister-gating fires at construction in
        # model_post_init. (--chain builds its own P2 via --jepa-phase p2, so it
        # is exempt from this single-phase guard.) Phase 3 (#21, WS-F) is no
        # longer gated — --phase 3 routes to V14Phase3Experiment now that the
        # whisper_target emission (WS-H, #20) has landed.
        raise NotImplementedError(
            f"--phase 2 dispatch is gated on unresolved blockers: "
            f"{_PHASE2_BLOCKERS}. See docs/neuroprobe/v14_blockers.md."
        )
    print(f"V14 dispatch — cohort subject_ids = {V14_TRAIN_SUBJECT_IDS} (9 subjects, S5 excluded)")
    print(f"  mode={args.mode} task={args.task} binary_tasks={args.binary_tasks} seed={args.seed}")
    print(f"  eval_mode={args.eval_mode} test=({args.test_subject_id},{args.test_trial_id})")
    print(f"  d_model={args.d_model} depth={args.depth} n_heads={args.n_heads} "
          f"M={args.m_sub_slots} eps={args.eps}")
    print(f"  K=80 DK parcels, c_max={args.c_max}, batch_size={args.batch_size}, "
          f"n_epochs={args.n_epochs}")
    print(f"  dkoleo_mode={args.dkoleo_mode} cross_attn_positions={args.cross_attn_positions} "
          f"mains_notch_hz={args.mains_notch_hz}")
    print(f"  lof_bad_channels={args.lof_bad_channels} lof_threshold={args.lof_threshold} "
          f"lof_n_neighbors={args.lof_n_neighbors} lof_report_path={args.lof_report_path}")
    print(f"  phase_mode={args.phase_mode} jepa_phase={args.jepa_phase} "
          f"frontend_lr_scale={args.frontend_lr_scale} neural_lag_s={args.neural_lag_s} "
          f"include_ajile12={args.include_ajile12} ref_operator_alpha={args.ref_operator_alpha}")
    print(f"  subtype_embed=(enabled={args.subtype_embed_enabled},reuse_kv={args.subtype_embed_reuse_kv},"
          f"vocab={args.subtype_embed_vocab}) "
          f"ref_embed=(enabled={args.ref_embed_enabled},reuse_kv={args.ref_embed_reuse_kv}) "
          f"ffn_variant={args.ffn_variant} loss_variant={args.loss_variant} "
          f"readout={args.readout} "
          f"gradient_checkpointing={args.gradient_checkpointing}")
    print(f"  optim: name={args.optimizer_name} lr={args.lr} "
          f"lr_schedule={args.lr_schedule} warmup_steps={args.warmup_steps} "
          f"min_lr_ratio={args.min_lr_ratio} weight_decay={args.weight_decay} "
          f"adam_beta2={args.adam_beta2} grad_clip={args.grad_clip} "
          f"accumulate_grad_batches={args.accumulate_grad_batches}")
    print(f"  guard: collapse_guard={args.collapse_guard} (SSL phases; #68 "
          f"--no-collapse-guard disarms) "
          f"ssl_val_check_interval={args.ssl_val_check_interval} (opt-steps, "
          f"×accum at Trainer #66) ssl_limit_val_batches={args.ssl_limit_val_batches}")
    if args.cluster == "slurm":
        # Same source of truth build_v14_experiment uses (no drift). In --chain,
        # the SSL/distill phases run with this strategy; the P4 probe is forced
        # single-GPU (ddp_strategy=None) for a full-test-set metric.
        _ddp_strategy = _resolve_ddp_strategy(args.tasks_per_node)
        _p4_note = " (P4 probe forced single-GPU)" if args.chain and _ddp_strategy else ""
        print(f"  slurm: partition={args.slurm_partition} "
              f"account={args.slurm_account} mem_gb={args.mem_gb} "
              f"gpus_per_node={args.gpus_per_node} "
              f"tasks_per_node={args.tasks_per_node} "
              f"slurm_use_srun={args.slurm_use_srun} "
              f"ddp_strategy={_ddp_strategy}{_p4_note} "
              f"cpus_per_task={args.cpus_per_task} timeout_min={args.timeout_min}")
    print(f"  precision={args.precision}")
    # #21 phase routing + cross-phase handoff (recorded so the run YAML never
    # silently rides the wrong phase / checkpoint).
    print(f"  phase={args.phase} chain={args.chain} p3_stage={args.p3_stage} "
          f"frozen_probe={args.frozen_probe} clip_len={args.clip_len} "
          f"exca_mode={args.exca_mode}")
    print(f"  whisper_target_cache_dir={args.whisper_target_cache_dir!r} "
          f"whisper_layer_merge={args.whisper_layer_merge} "
          f"target_standardize={args.target_standardize} "
          f"channel_stats_path={args.channel_stats_path!r}")
    print(f"  resume_from={args.resume_from!r} "
          f"snapshot_ckpt_to={args.snapshot_ckpt_to!r} work_dir={args.work_dir!r}")
    _resolved_xc = args.extractor_cache_folder or os.environ.get(
        "EXCA_EXTRACTOR_CACHE_FOLDER"
    )
    print(f"  extractor_cache_folder={_resolved_xc!r}")

    cross_attn_positions: list[int] | None = None
    if args.cross_attn_positions is not None:
        cross_attn_positions = [
            int(x) for x in args.cross_attn_positions.split(",") if x.strip()
        ]

    if args.dry_run:
        print("  (dry-run: not building Experiment; "
              "default electrode-tokens extractor = MultiStftView)")
        return 0

    if args.chain:
        # #21: full P1->P2->P3a->P3b->P4 pipeline in one process with ckpt
        # handoff (run_phase_pipeline). Overrides --phase.
        from speech_decoding.experiments.v14_phase_pipeline import (
            run_phase_pipeline,
        )

        phases = _build_v14_chain(args, cross_attn_positions=cross_attn_positions)
        print(f"  chain: {len(phases)} phases (P1,P2,P3a,P3b,P4) "
              f"work_dir={args.work_dir}")
        results = run_phase_pipeline(phases, work_dir=args.work_dir)
        print(f"V14 chain results: {results}")
        return 0

    # WS-F/WS-G #21: single-phase routing. phase 1 -> joint (V14JointExperiment);
    # phase 3 -> V14Phase3Experiment (Whisper distill); phase 4 -> base
    # supervised Experiment, or V14Phase4ReadoutExperiment (frozen probe) when
    # --frozen-probe / --resume-from is set so it can warm-start from P3.
    if args.phase == 3 and args.whisper_target_cache_dir is None:
        # Operator-level guard, independent of BT-data presence: P3 distills
        # against the whisper_target stream, so the cache dir is mandatory.
        raise ValueError(
            "--phase 3 (Whisper distillation, V14Phase3Experiment) needs "
            "--whisper-target-cache-dir; the P3 SmoothL1 loss has no target "
            "without it. Add it (and --channel-stats-path unless "
            "--no-target-standardize)."
        )
    # Same dispatch-time channel-stats fast-fail as the chain (audit 2026-06-04):
    # a single --phase 3 with a missing/typo'd --channel-stats-path would crash at
    # _build_standardizer (build time) instead of at parse — cheap to catch here,
    # and keeps the single-phase M0/debug path consistent with --chain.
    if args.phase == 3:
        _validate_channel_stats_path(args)
    # --resume-from (warm-start) AND --snapshot-ckpt-to (hand off downstream)
    # both require the transferable-state protocol, which only the frozen-probe
    # V14Phase4ReadoutExperiment carries — the base supervised Experiment would
    # TypeError at runtime (after a full train) on either. So either flag, like
    # --frozen-probe, selects the readout experiment on P4.
    phase4_frozen_probe = args.phase == 4 and (
        args.frozen_probe
        or args.resume_from is not None
        or args.snapshot_ckpt_to is not None
    )
    # Fix-1 guard (#65, 2026-06-04): a bare --phase 4 with no --frozen-probe and
    # no checkpoint to resume from is the from-scratch supervised-CE path with the
    # ENCODER + PMA UNFROZEN (trained from random init). That is a smoke/debug
    # path only — it is NOT the B35 scientific readout (frozen P3 encoder + frozen
    # PMA → mean-over-time → linear), which B35 adopted precisely because a
    # trainable encoder is untrainable at Neuroprobe's ≤3500-sample/task budget.
    # Warn loudly (mirrors the DDP-topology WARNING above) so a real run can't
    # silently report a meaningless from-random-init number as a "P4 result".
    # --fast-dev-run (the supervised smoke) is exempt.
    if args.phase == 4 and not phase4_frozen_probe and not args.fast_dev_run:
        print(
            "WARNING: bare --phase 4 with no --frozen-probe and no "
            "--resume-from/--snapshot-ckpt-to → the FROM-SCRATCH supervised path "
            "with the encoder + PMA UNFROZEN (random init). This is a smoke/debug "
            "config, NOT the B35 readout (frozen encoder + frozen PMA → mean → "
            "linear). For the scientific P4, pass --frozen-probe with --resume-from "
            "<SSL/distill encoder ckpt>. See "
            "project_v14_b35_p4_frozen_pma_mean_linear_2026_05_31."
        )
    # #39 (audit 2026-06-03): mirror the chain's per-phase budget gating onto the
    # single-phase path (it previously reached only --chain). --ssl-max-steps is an
    # SSL/distill-phase step budget (P1/P2-via-jepa, P3a/3b); P4 ignores it and
    # uses --n-epochs + early-stop. --p4-early-stop-patience is P4-only (val_loss is
    # the right signal only there). Defaults stay None → prior epoch-budget behavior
    # is byte-identical; these fire only when the flags are passed. The single-phase
    # --phase 4 rerun is the next gate, so this closes the gap that would otherwise
    # silently ignore --p4-early-stop-patience there.
    single_max_steps = args.ssl_max_steps if args.phase in (1, 3) else None
    # #54: same step-budget-ends-mid-epoch fix as the chain, for a single SSL
    # phase. Only meaningful when this phase carries a step budget.
    single_val_check = args.ssl_val_check_interval
    if single_val_check is None and single_max_steps is not None:
        single_val_check = max(50, single_max_steps // 10)
    # #66: cap SSL/distill validation; the P4 probe stays uncapped (its
    # val_loss is the downstream metric).
    single_limit_val = (
        args.ssl_limit_val_batches
        if args.phase in (1, 3)
        and args.ssl_limit_val_batches
        and args.ssl_limit_val_batches > 0
        else None
    )
    single_p4_patience = (
        (args.p4_early_stop_patience if args.p4_early_stop_patience > 0 else None)
        if args.phase == 4
        else None
    )
    print(
        f"  single-phase budget: max_steps={single_max_steps} "
        f"val_check_interval={single_val_check} (opt-steps) "
        f"limit_val_batches={single_limit_val} "
        f"collapse_guard={(args.phase != 4) and args.collapse_guard} "
        f"early_stopping_patience={single_p4_patience}"
    )
    xp = build_v14_experiment(
        **_common_build_kwargs(args, cross_attn_positions=cross_attn_positions),
        clip_len=args.clip_len,
        neural_lag_s=args.neural_lag_s,
        max_steps=single_max_steps,
        val_check_interval=single_val_check,
        limit_val_batches=single_limit_val,
        early_stopping_patience=single_p4_patience,
        # #54 audit M1: guard is SSL/distill-only; a single --phase 4 probe run
        # disables it (EarlyStopping + real metric already cover it). #68:
        # --no-collapse-guard further disarms it for a diagnostic SSL run.
        collapse_guard=(args.phase != 4) and args.collapse_guard,
        joint_phase=(args.phase == 1),
        p3_distill=(args.phase == 3),
        p3_stage=args.p3_stage,
        whisper_target_cache_dir=args.whisper_target_cache_dir,
        whisper_layer_merge=args.whisper_layer_merge,
        channel_stats_path=args.channel_stats_path,
        target_standardize=args.target_standardize,
        phase4_frozen_probe=phase4_frozen_probe,
        pretrained_ckpt=args.resume_from,
        snapshot_ckpt_to=args.snapshot_ckpt_to,
        jepa_phase=args.jepa_phase,
        frontend_lr_scale=args.frontend_lr_scale,
    )
    result = xp.run()
    print(f"V14 dispatch result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
