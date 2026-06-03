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
B36): F=30 ⅓-octave filterbank, hop=128 → 16 Hz (8 Hz latent), ``apply_log=False`` (raw
|X|), 0.5 Hz HPF, and ``scaler=None`` (robust-z normalizes downstream).
:class:`LogStftView` is the demoted ``F-single-STFT`` sister; ``apply_log``
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
# F=30 ⅓-octave filterbank bins (≈1–813 Hz) → encoder F_p = (30−3)//3+1 = 10.
DEFAULT_N_FREQ_BINS = 30
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
    seed: int = 33,
    exca_folder: str | None = None,
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
        # WS-C / C2 (B36): Multi-STFT front-end (F=30 ⅓-octave filterbank,
        # hop=128 → 16 Hz (8 Hz latent), raw |X| via apply_log=False). C4: 0.5 Hz HPF
        # removes DC + slow drift before the STFT. C3: StandardScaler dropped
        # (scaler=None) — robust-z normalizes the filterbank output downstream
        # of the view (see Nv14RobustZTransform / SessionRobustZNormalizer).
        electrode_tokens_extractor = MultiStftView(
            event_types="Ieeg",
            car="shaft",
            notch_filter=effective_bt_notch_hz,
            filter=(0.5, None),
            scaler=None,
            apply_log=False,
            channel_order="original",
            c_max=DEFAULT_C_MAX,
            session_robust_z=session_robust_z,
        )
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
        c_max=DEFAULT_C_MAX,
    )
    _apply_extractor_cache(dk_extractor, "dk_support", extractor_cache_folder)
    valid_mask_extractor = ElectrodeValidMask(
        event_types="Ieeg", bt_root=bt_root, c_max=DEFAULT_C_MAX,
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
            c_max=DEFAULT_C_MAX,
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
        # ``fused=True`` Adam (2026-05-30 speedup audit Tier-1): one fused
        # kernel for the whole step instead of a per-tensor foreach loop.
        # neuraltrain BaseTorchOptimizer forwards ``kwargs`` verbatim to
        # ``torch.optim.Adam``; ``fused`` is valid on CPU + CUDA in the
        # pinned torch 2.10, so the laptop test path is unaffected. Applies
        # to BOTH the supervised Phase-4 and the joint SSL optimizer (same
        # ``self.optim``). Same Adam math as the foreach impl, equivalent
        # within fp tolerance — NOT bit-identical (the fused kernel uses a
        # different reduction order, esp. under bf16-mixed on CUDA).
        optim={"optimizer": {"name": "Adam", "lr": 1e-3, "kwargs": {"fused": True}}},
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=n_epochs,
        seed=seed,
        # Per-clip metadata reaches the encoder forward as additional
        # kwargs alongside (tokens, support, valid_mask).
        x_name=(
            "electrode_tokens", "support", "valid_mask",
            "subject_subtype", "ref_idx",
        ),
        accelerator="auto",
        devices="auto",
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
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    p.add_argument("--clip-len", type=float, default=DEFAULT_CLIP_LEN_S,
                   help="Segmenter clip window (s): 5.0 for SSL P1/P2/P3 "
                        "(T_p=40), 1.0 for the P4 readout (T_p=8). Sizes the "
                        "encoder n_time_bins / RoPE ceiling.")
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
    p.add_argument("--seed", type=int, default=33)
    p.add_argument("--cluster", default=None,
                   help="Exca TaskInfra cluster ('slurm' or None for local).")
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
_PHASE3_BLOCKERS = (
    # WS-F closed the Phase-3 readout/distillation wiring: the
    # V14Phase3DistillModule (frozen-or-discriminative encoder → PMA → 256→1280
    # projector → Smooth-L1 vs pooled+standardized Whisper target) and the
    # V14Phase3Experiment are built and unit-tested (module forward/loss/freeze
    # + the experiment's model_post_init guards + _build_standardizer in
    # test_v14_phase3.py). Full P3 end-to-end construction is the WS-I2 capstone
    # (pending). The one remaining LIVE-dispatch blocker is the whisper_target
    # data emission: the segmenter has no extractor for the cached 50 Hz
    # all-layer-mean Whisper feature ((B, 250, 1280)) yet — that is WS-H.
    "Phase-3 distillation (V14Phase3DistillModule / V14Phase3Experiment) is "
    "wired and unit-tested; the live dispatch path still needs the "
    "whisper_target segmenter extractor (cached 50 Hz Whisper all-layer-mean "
    "feature, (B, 250, 1280)) — the P3 data-layer emission is the remaining "
    "WS-H blocker. See ssl/distill.py + experiments/v14_phase3.py."
)


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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == 2:
        # Phase 2 is the legacy split-P2 entry-point, collapsed into the joint
        # phase by B29 Item 1; dispatch P1∪P2 via --phase 1 (V14JointExperiment).
        # Phase 3 is NO LONGER gated here — its module/experiment are wired
        # (WS-F: V14Phase3DistillModule / V14Phase3Experiment) and the live
        # dispatch hits the precise whisper_target data blocker below (after the
        # dry-run short-circuit) rather than a blanket gate. (phase=1 falls
        # through to construct V14JointExperiment; its B2.x sister-gating fires
        # at construction in model_post_init, see above.)
        raise NotImplementedError(
            f"--phase 2 dispatch is gated on unresolved blockers: "
            f"{_PHASE2_BLOCKERS}. See docs/neuroprobe/v14_blockers.md."
        )
    print(f"V14 dispatch — cohort subject_ids = {V14_TRAIN_SUBJECT_IDS} (9 subjects, S5 excluded)")
    print(f"  mode={args.mode} task={args.task} binary_tasks={args.binary_tasks} seed={args.seed}")
    print(f"  eval_mode={args.eval_mode} test=({args.test_subject_id},{args.test_trial_id})")
    print(f"  d_model={args.d_model} depth={args.depth} n_heads={args.n_heads} "
          f"M={args.m_sub_slots} eps={args.eps}")
    print(f"  K=80 DK parcels, batch_size={args.batch_size}, n_epochs={args.n_epochs}")
    print(f"  dkoleo_mode={args.dkoleo_mode} cross_attn_positions={args.cross_attn_positions} "
          f"mains_notch_hz={args.mains_notch_hz}")
    print(f"  phase_mode={args.phase_mode} jepa_phase={args.jepa_phase} "
          f"frontend_lr_scale={args.frontend_lr_scale} neural_lag_s={args.neural_lag_s} "
          f"include_ajile12={args.include_ajile12} ref_operator_alpha={args.ref_operator_alpha}")
    print(f"  subtype_embed=(enabled={args.subtype_embed_enabled},reuse_kv={args.subtype_embed_reuse_kv},"
          f"vocab={args.subtype_embed_vocab}) "
          f"ref_embed=(enabled={args.ref_embed_enabled},reuse_kv={args.ref_embed_reuse_kv}) "
          f"ffn_variant={args.ffn_variant} loss_variant={args.loss_variant} "
          f"readout={args.readout} "
          f"gradient_checkpointing={args.gradient_checkpointing}")
    if args.cluster == "slurm":
        print(f"  slurm: partition={args.slurm_partition} "
              f"account={args.slurm_account} mem_gb={args.mem_gb} "
              f"gpus_per_node={args.gpus_per_node} "
              f"cpus_per_task={args.cpus_per_task} timeout_min={args.timeout_min}")
    print(f"  precision={args.precision}")
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

    if args.phase == 3:
        # WS-F: the Phase-3 distillation module + experiment are wired
        # (V14Phase3DistillModule / V14Phase3Experiment) and unit-tested
        # (test_v14_phase3.py). Full P3 end-to-end construction is the WS-I2
        # capstone (pending). The LIVE dispatch path still needs the
        # whisper_target data emission (WS-H); fail fast + precise rather than
        # silently building a Phase-4 CE classifier.
        raise NotImplementedError(
            f"--phase 3 (stage {args.p3_stage}) dispatch: {_PHASE3_BLOCKERS} "
            "See docs/neuroprobe/v14_blockers.md."
        )

    xp = build_v14_experiment(
        mode=args.mode, task=args.task, seed=args.seed,
        eval_mode=args.eval_mode,
        test_subject_id=args.test_subject_id,
        test_trial_id=args.test_trial_id,
        binary_tasks=args.binary_tasks,
        session_robust_z=args.session_robust_z,
        eps=args.eps, d_model=args.d_model, depth=args.depth,
        n_heads=args.n_heads, m_sub_slots=args.m_sub_slots,
        clip_len=args.clip_len,
        neural_lag_s=args.neural_lag_s,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_epochs=args.n_epochs,
        cluster=args.cluster, fast_dev_run=args.fast_dev_run,
        slurm_partition=args.slurm_partition,
        slurm_account=args.slurm_account,
        mem_gb=args.mem_gb,
        gpus_per_node=args.gpus_per_node,
        cpus_per_task=args.cpus_per_task,
        timeout_min=args.timeout_min,
        precision=args.precision,
        extractor_cache_folder=args.extractor_cache_folder,
        dkoleo_mode=args.dkoleo_mode,
        cross_attn_positions=cross_attn_positions,
        mains_notch_hz=args.mains_notch_hz,
        subtype_embed_enabled=args.subtype_embed_enabled,
        subtype_embed_reuse_kv=args.subtype_embed_reuse_kv,
        subtype_embed_vocab=args.subtype_embed_vocab,
        ref_embed_enabled=args.ref_embed_enabled,
        ref_embed_reuse_kv=args.ref_embed_reuse_kv,
        phase_mode=args.phase_mode,
        ref_operator_alpha=args.ref_operator_alpha,
        joint_phase=(args.phase == 1),
        include_ajile12=args.include_ajile12,
        ffn_variant=args.ffn_variant,
        latent_valid_override=args.latent_valid_override,
        sa_mask_mode=args.sa_mask_mode,
        loss_variant=args.loss_variant,
        jepa_phase=args.jepa_phase,
        frontend_lr_scale=args.frontend_lr_scale,
        readout=args.readout,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    result = xp.run()
    print(f"V14 dispatch result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
