"""V14 first-pass DCC dispatch entrypoint.

Composes the v14 NeuralTrain Experiment: BT Wang2024Treebank study + DK-hard
support extractor + V14ParcelPerceiver + DETR readout, with first-pass defaults
locked in ``memory/project_v14_encoder_design_2026_05_13.md``.

DCC invocation (via ``scripts/dcc/dispatch``):

    scripts/dcc/dispatch -m speech_decoding.experiments.dispatch_v14 \\
        --mode lite --m-sub-slots 4 --d-model 128 --depth 6

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
from speech_decoding.extractors.view import (
    STFT_2BAND_HIGH,
    STFT_2BAND_LOW,
    STFT_3BAND_BETA,
    STFT_3BAND_HG,
    STFT_3BAND_SLOW,
    MultiStftView,
)
from speech_decoding.extractors.whisper_target import WhisperTargetExtractor
from speech_decoding.studies.braintreebank.anatomy import (
    V14_DK_PARCEL_LABELS,
    atlas_spec,
)
from speech_decoding.studies.braintreebank.manifest import V14_TRAIN_SUBJECT_IDS
from speech_decoding.studies.braintreebank.study import (
    _SESSIONS_BY_MODE,
    Wang2024Treebank,
)
from speech_decoding.studies.braintreebank.word_events import (
    BTWordEvents,
    DEFAULT_PRETRAIN_HOLDOUT_FRACTION,
)


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
# Per-GPU batch. 4 is the validated production value: it fits the 32 GB
# coganlab-gpu / RTX-5000-Ada cards WITH gradient_checkpointing ON, and the live
# BT chain (jobs 47811938/47829858) ran 6 h clean at bs=4 × accum=8 × 4-GPU =
# eff-128 ([[project_bt_chain_run_2026_06_07]]). Raise on a 48 GB+ card.
DEFAULT_BATCH_SIZE = 4
# DataLoader worker count (B1.4e, 2026-05-29). The per-sample extractor stack
# (CAR + torch.stft in LogStftView is recomputed per __getitem__; only the raw
# waveform load is MapInfra-cached) is CPU-bound, so num_workers=0 starves the
# GPU (1.48 it/s on the 5/29 Lite baseline). 15 workers (paired with the live
# run's cpus_per_task=16) keep the data path fed and did NOT fork-deadlock — the
# "nw=4 hang" was a launch-blocker hypothesis the production run disproved
# ([[project_bt_chain_run_2026_06_07]]). Pair with --cpus-per-task >= num_workers + 1.
DEFAULT_NUM_WORKERS = 15
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
DEFAULT_C_MAX = 256  # BT-only default (2026-06-07). BT max electrodes = 256
                     # (Wang2024Treebank raw) → lossless for the current BT-only
                     # chain, and drops 33% of the 384 padding slots (front-end
                     # FLOPs + activation memory are C-linear). The JOINT corpus
                     # (SWEC+AJILE12+D) needs --c-max 384: D-cohort max=366,
                     # AJILE12 max≈200, SWEC max=128. dk_support.py / view.py /
                     # valid_mask.py RAISE (fail loud, not lossy) if any
                     # subject's n_real > c_max — so 256 can never silently clip
                     # a >256-electrode subject; it errors at dispatch instead.

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

# B28 DKoleo demotion 2026-05-27 PM (B28 Item 1): DKoleo @ M4 is OFF by default.
# The ``--dkoleo-mode`` CLI sister-selector was culled 2026-06-13 (B28-demoted);
# the encoder keeps its own ``dkoleo_mode="off"`` internal default, so dispatch
# no longer threads a value.

# B29 Item 11 lock 2026-05-27 PM-late: subtype embed vocab choice.
SUBTYPE_EMBED_VOCABS: tuple[str, ...] = ("binary", "three_way")
DEFAULT_SUBTYPE_EMBED_VOCAB: str = "binary"

# MoE-FFN audit 2026-05-28: dense FFN preserved. The ``--ffn-variant`` CLI flag
# (and its ``soft_moe_4`` audit-rejected sister) was culled 2026-06-13 — the
# encoder is always dense; dispatch no longer threads a variant.

# B36 phase-mode relabel 2026-06-03: the ``--phase-mode`` CLI axis was
# recorded-only run-record metadata (joint_b29 / split_p1_p2 falsifier) and was
# culled 2026-06-13. The encoder keeps its own ``phase_mode`` internal default;
# the behavioral stage is selected via ``--jepa-phase``.

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

# B37 (2026-06-10) encoder + joint-SSL defaults. Single source for both the
# ``build_v14_experiment`` signature and the argparse layer so the two cannot
# drift. ``pool``/``latent_depth`` ride the brain-model config; the rest are
# joint-SSL (``ssl_mode``) + masking/predictor knobs (#75/#76). The mask
# ratios/types mirror the 6/03 masking lock (ssl/mask.py signature defaults);
# the predictor sizes mirror the V14JointExperiment pydantic field defaults.
# NOTE (2026-06-13): pool=mean + ssl_mode=joint are KEPT at the B37 values
# because they are interlocked with the committed B36-staged-SSL cull (auto now
# always resolves to joint, which requires the freq-preserving mean pool). The
# geometry/hyperparam knobs (d_model / latent_depth / predictor sizes /
# weight_decay) are left at their pre-B37 values — the in-flux capstone geometry
# is configured per-run via explicit CLI flags, not pinned here, so the code
# default can't drift back into "lying" about a specific production recipe.
DEFAULT_POOL: str = "mean"                 # B37 mean-pool (interlocked w/ joint SSL default + staged cull)
DEFAULT_LATENT_DEPTH: int = 2
DEFAULT_LATENT_MODE: str = "parcel"       # B37+ "joint" = parcel×time freq-batched
DEFAULT_MEAN_POOL_STD: bool = False       # B37+ RGB-style mean|std stem channel
DEFAULT_SSL_MODE: str = "joint"           # B37 single-forward joint SSL (interlocked w/ pool=mean + staged cull)
DEFAULT_LAMBDA_M4: float = 1.0            # D7 M4 term weight
# Predictor depth auto-rule (Ben 2026-06-12): when --m2/m4-predictor-depth is
# left unset, each predictor defaults to HALF the depth of the encoder it
# predicts from — M2 (front-end) tracks --depth, M4 (parcel) tracks
# --latent-depth (see main()). These constants are the fallback for direct
# build_v14_experiment() callers (tests); the CLI resolves None via the
# half-rule so a depth sweep auto-scales its predictor.
DEFAULT_M2_PREDICTOR_DEPTH: int = 3       # D9 per-tap (front-end M2) = depth//2 @ depth=6
DEFAULT_M4_PREDICTOR_DEPTH: int = 2       # D9 per-tap (parcel M4)
DEFAULT_JOINT_FRONTEND_LR_SCALE: float = 1.0   # D8 (no discrimination)
DEFAULT_JOINT_PARCEL_LR_SCALE: float = 1.0     # D8 (no discrimination)
DEFAULT_M2_MASK_TYPE: str = "bands"       # 6/03 masking lock
DEFAULT_M2_MASK_RATIO: float = 0.50       # 6/03 masking lock
DEFAULT_M4_MASK_TYPE: str = "tube"        # 6/03 masking lock
DEFAULT_M4_MASK_RATIO: float = 0.20       # 6/03 masking lock
DEFAULT_M2_TIME_BAND_FLOOR: int = 2       # 6/03 masking lock (M2 band width along T_p)
DEFAULT_M2_FREQ_BAND_FLOOR: int = 1       # 6/03 masking lock (M2 band width along F_p)
# EMA teacher momentum τ (B26 lock; mirrors ssl/ema.py P1_EMA_TAU == P2_EMA_TAU).
# The SSL-sweep CLI flags (--ema-tau / --m{2,4}-mask-ratio) default to None and
# resolve to these constants inside build_v14_experiment, so an UNSET flag is
# byte-identical to the prior hardcoded value (the ema.py / ssl.mask constants
# are NOT touched). A passed value overrides the τ / held-out ratio.
DEFAULT_EMA_TAU: float = 0.99925          # B26 EMA τ lock (V-JEPA 2 §2.4)
DEFAULT_PREDICTOR_HIDDEN: int = 128
DEFAULT_PREDICTOR_N_HEADS: int = 4

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

# AJILE12 inclusion. Default OFF (2026-06-07): the active chain is BT-only
# (swec_frac=0, no SWEC/AJILE12/D in the mix). B29 re-included AJILE12 for the
# full JOINT corpus; pass --include-ajile12 to restore it when the joint
# escalation runs ([[project_bt_chain_run_2026_06_07]]).
DEFAULT_INCLUDE_AJILE12: bool = False


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


def _assert_support_valid_config_agree(
    dk_extractor: V14DKHardSupportExtractor,
    valid_mask_extractor: ElectrodeValidMask,
) -> None:
    """DP9 fail-loud (data-integrity doctrine L4). ``support`` and ``valid_mask``
    are joined downstream by bare positional electrode index (``v14_encoder``),
    so they MUST be built over the SAME electrode-defining config — same parcel
    vocabulary, same unmapped policy, same padding, same corpus root, same event
    type. They are matched by hand at the construction site today; a silent edit
    to one would make row ``c`` of ``support`` and row ``c`` of ``valid_mask``
    describe different electrodes with no error. Enforce the convention here.
    See reports/bt_alignment/electrode_desync_damage_2026_06_09.md (DP9)."""
    shared = (
        "parcel_labels", "label_column", "exclude_single_electrode_parcels",
        "unmapped_policy", "c_max", "bt_root", "event_types", "electrode_set",
    )
    mismatched = {
        field: (getattr(dk_extractor, field), getattr(valid_mask_extractor, field))
        for field in shared
        if getattr(dk_extractor, field) != getattr(valid_mask_extractor, field)
    }
    if mismatched:
        raise ValueError(
            "support/valid_mask electrode-config disagree — rows would describe "
            f"different electrodes (DP9): {mismatched}. Build both extractors "
            "with identical parcel_labels / unmapped_policy / c_max / bt_root / "
            "event_types."
        )


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
    # §7/B01 no-WD param-group split is now implemented in the phase modules'
    # configure_optimizers (task #40, ``optim_param_groups.maybe_split_no_decay``):
    # a non-zero ``weight_decay`` is the top-level default and biases / LayerNorm
    # γβ / named embeds ride in a ``weight_decay: 0.0`` group (default
    # ``wd_exclude_norms=True``; ``--no-wd-exclude-norms`` decays them too). The
    # earlier guard that refused a non-zero ``weight_decay`` is therefore removed.
    # NOT fused — fused optimizers are incompatible with Lightning's bf16-mixed
    # AMP gradient clipping (see docstring; both are §7 locks). Empty default →
    # standard non-fused torch optimizer, which clears _step_supports_amp_scaling
    # so the AMP clip guard does not fire.
    optim_kwargs: dict[str, tp.Any] = {}
    if weight_decay:
        # Forwarded as the top-level optimizer weight_decay; the phase modules'
        # configure_optimizers carry the no-WD-group override for the exempt
        # subset (biases / LN γβ / embeds) when wd_exclude_norms is set.
        optim_kwargs["weight_decay"] = weight_decay
    elif optimizer_name == "AdamW":
        # AdamW's torch default is weight_decay=0.01 — pin it to 0.0 EXPLICITLY so
        # the intended weight_decay=0 default is not silently overridden by the
        # torch family default on the recommended ``--optimizer AdamW`` path
        # (without this pin AdamW would decay biases / LN γβ / freq_embed at 0.01
        # uniformly even though the user asked for no decay). Adam's torch default
        # is already 0.0, so its kwargs stay empty. (The wd VALUE is M0-swept, not
        # locked; only the no-WD param-group EXEMPTION is the §7/B01 convention.)
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


def _resolve_static_forward_cohesion(args: argparse.Namespace) -> None:
    """In-place: fill the AUTO defaults the static-shape throughput regime implies
    (Ben 2026-06-22: "make every confirmed throughput win default"). The V-JEPA
    drop-not-pad forward (``--converged-static-forward``) has two PROVEN companions;
    auto-enable them so a future launch can't silently leave a measured win off,
    while an explicit flag still overrides:

    * ``compile_dynamic``: AUTO (``None``) ⇒ static (``False``) under static-forward
      — the maskless kernel wants one compiled graph per session geometry; symbolic
      dynamic forfeits the specialization. Legacy ragged path stays ``True``
      (byte-identical to the prior default).
    * ``sdpa_backend``: ``"default"`` ⇒ ``"cudnn_latent"`` under static-forward —
      routes the large-L masked cross-electrode attention (~50% of GPU time) to the
      ~2.6x-faster cuDNN kernel, scoped so small-L calls keep the dispatcher.

    ``find_unused_parameters=False`` (``--ddp-static-graph``) is deliberately NOT
    auto-on: it is correctness-gated on the empty/static unused-param set, proven
    per-arch by a short run (DTAI A/B 2543181/2543183), never assumed."""
    if args.compile_dynamic is None:
        args.compile_dynamic = not args.converged_static_forward
    if args.converged_static_forward and args.sdpa_backend == "default":
        args.sdpa_backend = "cudnn_latent"


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


def _resolve_corpus_mode(
    *, joint_phase: bool, p3_distill: bool, mode: str, eval_mode: str,
) -> tuple[str, str]:
    """Leakage decouple (#82): pick the (study session-mode, BTWordEvents
    eval-mode) for a phase.

    The SSL/distill phases pretrain on the Neuroprobe-legal corpus + the
    ``"Pretrain"`` split. P1/P2 (joint) use study ``"pretrain"``
    (V14_PRETRAIN_SESSIONS — 13 sessions, no teacher needed). P3 (distill) uses
    study ``"p3_distill"`` (V14_P3_DISTILL_SESSIONS — those 12 minus (8,0), which
    has no Whisper teacher cache); routing P3 to ``"pretrain"`` would crash
    lazily on the first (8,0) clip. Both corpora are disjoint from the 12 eval
    sessions. ONLY the supervised P4 probe trains/tests on the eval split, so it
    runs the leaderboard protocol — its study universe is always BT_LITE (the 12
    Neuroprobe eval sessions; ``"nano"`` only for tiny smokes), NEVER BT_FULL.
    ``--mode`` selects only the SSL clip budget; routing P4 to ``"full"`` would
    (a) load BT_FULL incl excluded subject 5 (no data on disk → crash) and (b)
    turn CrossSession's train set into the test subject's *pretrain* trials — a
    silently wrong, non-leaderboard eval the (P4-exempt) leakage guard would NOT
    catch. The eval split TYPE (CrossSession/CrossSubject) is always preserved.
    This is the single safety-critical branch that prevents eval data from
    reaching pretraining AND keeps the P4 eval leaderboard-faithful; the runtime
    leakage guard is the fail-closed backstop. Pure + isolated so BOTH regression
    directions are unit-testable (SSL→eval = leakage caught by the guard;
    P4→pretrain/full = silent wrong eval, NOT caught — only this test)."""
    if p3_distill:
        return "p3_distill", "Pretrain"
    if joint_phase:
        return "pretrain", "Pretrain"
    return ("nano" if mode == "nano" else "lite"), eval_mode


def build_v14_experiment(
    *,
    bt_root: str | None = None,
    mode: tp.Literal["nano", "lite", "full"] = "lite",
    task: str = DEFAULT_TASK,
    eval_mode: tp.Literal[
        "WithinSession", "CrossSession", "CrossSubject", "AllCells"
    ] = DEFAULT_EVAL_MODE,
    test_subject_id: int = DEFAULT_TEST_SUBJECT_ID,
    test_trial_id: int = DEFAULT_TEST_TRIAL_ID,
    # WithinSession only: which KFold fold (0..n_folds-1, n_folds=2 for Lite)
    # this P4 eval cell scores. Forwarded to BTWordEvents.fold_index; ignored
    # for CrossSession/CrossSubject/Pretrain. Default 0 is the BTWordEvents
    # default, so the exca cache uid is unchanged for the non-WithinSession path.
    fold_index: int = 0,
    # Leakage decouple (#82): per-legal-session monitoring tail held out for the
    # SSL/distill "Pretrain" split (val AND test). Applies only on the SSL
    # phases (joint_phase / p3_distill), where the corpus is overridden to
    # V14_PRETRAIN_SESSIONS; ignored on the P4 eval path. NOT a leaderboard
    # quantity — it never partitions eval data.
    pretrain_holdout_fraction: float = DEFAULT_PRETRAIN_HOLDOUT_FRACTION,
    binary_tasks: bool = True,
    electrode_tokens_extractor: tp.Any | None = None,
    # Front-end family for the DEFAULT (no custom extractor) path.
    #   "raw"   — single F=50 raw |STFT| grid (FE-RAW-1, B36 default).
    #   "2stft" — dual-band 2STFT (Ben 2026-06-12): TWO single-band MultiStftViews
    #             (front_end="band") at DIFFERENT native hops, carried as separate
    #             batch keys electrode_tokens (LOW, N=512/hop256, 14 bins @ 8 Hz)
    #             + electrode_tokens_high (HIGH, N=128/hop64, 9 bins @ 16 Hz). Sets
    #             the encoder per_band_stem=True with the two Conv2d band stems;
    #             band freq-bin counts are DERIVED from the views (no drift) and
    #             time-bin counts from clip_len. Requires pool="mean".
    #   "3stft" — converged-arch per-electrode triple-band 3STFT 2/2/2 frontend
    #             (Ben 2026-06-17). THREE single-band MultiStftViews (front_end=
    #             "band") at the slow/beta/HG native hops, carried as separate
    #             batch keys electrode_tokens_slow/beta/hg. Routes the WHOLE
    #             dispatch to V14ConvergedExperiment + the V14Converged model
    #             config (self-contained M2/M4 SSL) instead of V14ParcelPerceiver;
    #             requires the converged_* shape params above. The band views are
    #             constructed identically to the --cache-band per-band builds
    #             (shared STFT_3BAND_* + common_fe_kwargs + band_<name> spec-cache
    #             subdirs) so a 3stft run HITs those caches.
    # Ignored when a custom ``electrode_tokens_extractor`` is supplied.
    frontend: tp.Literal["raw", "2stft", "3stft"] = "raw",
    # 3STFT per-band cache build (--cache-band, cache-only only). When set, ONE
    # named band of the locked 3STFT 2/2/2 ladder (slow | beta | hg,
    # ``reports/fe_3stft_2of2of2_spec_2026_06_17.md``) is built as a single-grid
    # ``MultiStftView(front_end="band")`` riding the ``electrode_tokens`` slot
    # (per_band_stem stays False). The band view is constructed identically to a
    # future ``--frontend 3stft`` training run (shared ``STFT_3BAND_*`` constants
    # + ``common_fe_kwargs``), so the spec-cache namespace matches → the run HITs
    # this cache. Overrides ``frontend``. None = no 3STFT band build.
    cache_band: tp.Literal["slow", "beta", "hg"] | None = None,
    # Cache-build parallelism (--cache-only): restrict the SSL/study corpus to a
    # single session by its index in ``_SESSIONS_BY_MODE[study_mode]`` so a SLURM
    # array builds one session's spec cache per task. None = full corpus. The
    # subset rides the study's timeline list ONLY — it does not change the
    # extractor uid or the per-session spec-cache key, so a cache built under one
    # index is byte-identical to what the full run reads back.
    cache_session_index: int | None = None,
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
    # Electrode MONTAGE (applied PRE-CAR, threaded study→loader + support +
    # valid_mask in lockstep). "all" (default) keeps the full montage minus STATIC
    # bad contacts — BT-FULL pretraining. "lite" subsets to the Neuroprobe-Lite
    # electrodes IN THE LOADER, before shaft-CAR, so the Lite result references
    # zero out-of-budget contacts (CAR over Lite electrodes only → reproducible
    # from the Lite budget; the honest leaderboard cell). This is NOT the old
    # pool-side ~valid_mask drop — the montage itself is Lite. The chain pins the
    # P4 eval phase to "lite" (enforced by the phase4_frozen_probe guard above)
    # while the SSL phases stay "all"; --electrode-set overrides a standalone
    # build. Distinct exca cache-uid when "lite" (timeline carries electrode_set +
    # the loader/support/valid_mask align to lite_voltage_order). BT-only.
    electrode_set: tp.Literal["all", "lite"] = "all",
    # Parcellation atlas. "dk" (default) = FreeSurfer Desikan-Killiany, K=80;
    # "dkt" = Desikan-Killiany-Tourville, K=74 (native depth-wm.csv DKT column,
    # 62 cortical = DK 34 bases minus {bankssts,frontalpole,temporalpole} per
    # hemi + 12 aseg subcortical). Selecting an atlas moves the depth-wm column
    # AND the parcel vocabulary together (anatomy.atlas_spec) — they can never
    # desync — and sets the encoder's k_parcels = len(vocabulary).
    atlas: tp.Literal["dk", "dkt"] = "dk",
    # Ben 2026-06-13: drop parcels covered by exactly ONE valid electrode — the
    # lone electrode is zeroed + marked invalid, the parcel left uncovered for
    # that subject (global K unchanged). A single-electrode parcel has a
    # degenerate within-parcel std (=0) that poisons the heteroscedastic M4
    # precision weight. Applied identically to the support + valid_mask extractors.
    exclude_single_electrode_parcels: bool = False,
    d_model: int = DEFAULT_D_MODEL,
    depth: int = DEFAULT_DEPTH,
    n_heads: int = DEFAULT_N_HEADS,
    # Converged-arch (--frontend 3stft) shape. REQUIRED-when-3stft (no silent
    # run defaults — the widths are Ben's to name at launch); raises a ValueError
    # below if any is left None on the 3stft path. Inert on raw/2stft (the
    # V14ParcelPerceiver path ignores them). The two predictor widths are the M2
    # (own-electrode) and M4 (parcel-query) paradigm-B predictor hidden dims +
    # depths from the converged FE spec §6.
    converged_frontend_layers: int | None = None,
    converged_latent_layers: int | None = None,
    converged_m2_pred_dim: int | None = None,
    converged_m2_pred_layers: int | None = None,
    converged_m4_pred_dim: int | None = None,
    converged_m4_pred_layers: int | None = None,
    # Converged M2/M4 loss-term weights (neutral 1.0 default; FE-spec §8.7
    # λ sister sweeps override). Inert on raw/2stft.
    converged_lambda_m2: float = 1.0,
    converged_lambda_m4: float = 1.0,
    # Converged MASK GEOMETRY (FE-spec §8 / Ben 2026-06-18 "tuned often"). None →
    # the V14ConvergedExperiment / M2MaskConfig / M4MaskConfig LOCKED defaults
    # (hg_start_rate 0.20, hg_span 3, beta_start_rate 0.30, beta_span 2,
    # parcel_mask_ratio 0.20); a value overrides. Inert on raw/2stft.
    converged_m2_hg_start_rate: float | None = None,
    converged_m2_hg_span: int | None = None,
    converged_m2_beta_start_rate: float | None = None,
    converged_m2_beta_span: int | None = None,
    converged_m2_slow_freq_tubes: int | None = None,
    converged_m4_parcel_mask_ratio: float | None = None,
    # Static-shape SSL (V-JEPA-2 throughput regime). tube_ratio set ⇒ static
    # tight-pack tube + rand_unmask masks (constant n_vis / N_mask per session);
    # static_forward then routes the forward through the fixed-K / maskless /
    # sync-free path (one compiled graph per session geometry). static_forward
    # REQUIRES tube_ratio AND group_by_session. None/False ⇒ legacy variable masks.
    converged_tube_ratio: float | None = None,
    converged_tube_p_fixed: int = 18,
    converged_static_forward: bool = False,
    # Converged M4 heteroscedastic down-weight (Ben 2026-06-18). ON by default
    # (α=1.0, n_ref=11 — the B37 downweight_dof lock); --converged-m4-precision-off
    # disables it (the R-precision-off sister). α/n_ref None → the config defaults.
    converged_m4_precision_off: bool = False,
    converged_m4_precision_alpha: float | None = None,
    converged_m4_precision_n_ref: float | None = None,
    # Online linear probe (diagnostic; spec reports/online_probe_spec_2026_06_18.md).
    # OFF by default — registering it is inert until turned on, so it can never
    # perturb a live run. Converged-path only; inert on raw/2stft. The probe DATASET
    # is built in-worker (DCC-only) when enabled; cadence tunes its firing period.
    online_probe_enabled: bool = False,
    online_probe_cadence: int = 1000,
    monitor_every_n_steps: int | None = None,
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
    # Cadence (opt-steps) of the keep-ALL ladder checkpoint. Default 500
    # (run-ops policy); a long run thins it via --ckpt-ladder-every.
    ckpt_ladder_every: int = 500,
    # Cap on validation batches per check (#66). An uncapped SSL val set
    # (~875 batches) made each collapse-guard validation ~8 min → ~80% of
    # wall-clock under the (now-fixed) over-frequent cadence. The guard panel
    # estimates RankMe/coverage from a small sample, so a cap is lossless for
    # it. None → full val set (correct for P4, where val_loss IS the metric).
    limit_val_batches: int | float | None = None,
    # Cap the final ``trainer.test()`` pass the same way ``limit_val_batches``
    # caps validation. The end-of-phase test computes the monitor panel
    # (RankMe/coverage) over the full holdout (~160 batches → ~2.5 min on a
    # nano run, dwarfing the training steps). For an intuition run a 2-batch
    # estimate is plenty; None → full test set (the prior, unchanged default).
    limit_test_batches: int | float | None = None,
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
    # probe over-fit). OFF by default everywhere now (Ben 2026-06-11: never
    # auto-kill); opt IN per-run via --collapse-guard.
    collapse_guard: bool = False,
    # --live nano learning-dynamics dashboard (near-free graphs: loss/val-loss,
    # RankMe, LR-curve). Defaults reproduce non-live behavior; at default values
    # they stay out of the exca cache uid. See
    # reports/nano_dynamics_dashboard_handoff_2026_06_07.md.
    wandb_config: tp.Any | None = None,
    lr_log_interval: str = "epoch",
    log_every_n_steps: int = 10,
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
    # #80 whole-movie raw-|STFT| feature cache. Default ON (derived from
    # ``extractor_cache_folder`` below) so the ~9-min-per-run session_robust_z
    # whole-movie STFT over the 13-session pretrain corpus is paid ONCE and every
    # later same-front-end-config run memmap-slices it instead of recomputing.
    # Location-only knob (excluded from the extractor cache uid), so arming it is
    # identity-transparent: same run uid, byte-identical features (parity-tested).
    # ``disable_spec_cache=True`` forces the per-run recompute. Only armed on the
    # default raw MultiStftView path (a custom electrode_tokens_extractor sets its
    # own spec_cache_dir).
    spec_cache_dir: str | None = None,
    disable_spec_cache: bool = False,
    # Lightweight-redeploy (Option B / DeltaAI): serve clips from the on-disk spec
    # cache ALONE — no extractor cache, no raw h5 on the target. Threads to every
    # default band view via ``common_fe_kwargs``; default off ⇒ DCC build path
    # byte-identical. Requires ``spec_cache_dir`` set + LOF off (view enforces).
    spec_only: bool = False,
    # h5-free trial-duration override (DeltaAI/spec_only). Path to a JSON
    # ``{"<subject_id>_<trial_id>": native_n_samples}``; when set, the study reads
    # each trial's duration from it instead of opening raw h5 (the only h5 touch in
    # timeline building). uid-invariant ⇒ byte-identical caches/clips. Default None
    # ⇒ duration read from h5 (DCC).
    trial_durations_path: str | None = None,
    # Layer-2 bad-electrode defense (#180): directory of per-session bad-time-window
    # sidecars from ``scripts/neuroprobe/precompute_bad_windows.py``. When set, SSL
    # clips overlapping a glitch span are dropped before sampling. SSL-ONLY: gated to
    # ``None`` for the P4 phase below so the Neuroprobe-Lite parity clip sets are never
    # altered. Removes events (rows), never electrodes — DP4 row-alignment untouched.
    bad_window_dir: str | None = None,
    # Throughput lever (science-neutral): session-homogeneous TRAIN batches so the
    # ragged forward pays each batch's true electrode count C, not the corpus max-C
    # (the latent attention is O(C²) in electrode count). SSL-ONLY: gated to False
    # for P4 so the Neuroprobe-Lite parity clip/batch sets are byte-identical. See
    # ``Data.group_by_session`` / ``_SessionGroupedBatchSampler``.
    group_by_session: bool = False,
    # #249 rank size-balancing (needs group_by_session): match the W DDP ranks'
    # simultaneous batches by electrode count C to remove straggler idle. SSL-only.
    balance_ranks: bool = False,
    # raw-GPU-util bubble removal (needs group_by_session): all W ranks run the
    # SAME session per micro-step ⇒ identical C ⇒ no all-reduce straggler. Stronger
    # than balance_ranks; per-step diversity → 1, grad-accum restores mixing.
    same_session_across_ranks: bool = False,
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
    # Warm 4-GPU worker (nano_worker_ddp): the process is ALREADY one of N srun
    # ranks (the worker launched ``srun --ntasks=N``), so exca must NOT submit —
    # it runs the experiment in-process (``cluster=None`` → local) while Lightning
    # picks up the live srun ranks via SLURMEnvironment. This flag only forces the
    # find-unused DDP strategy ON (a multi-rank staged-phase run needs it; see
    # _resolve_ddp_strategy); no slurm infra fields are set, so exca stays local.
    in_allocation_ddp: bool = False,
    # C5 resilience: emit ``#SBATCH --requeue`` so a SLURM-preempted job is
    # auto-resubmitted; combined with the within-phase ``last.ckpt`` resume
    # (Experiment._within_phase_resume_ckpt) the requeued job continues from the
    # last checkpoint instead of restarting the phase. Off by default (smokes /
    # single-shot runs); set for the long full chain.
    requeue: bool = False,
    # Lightning trainer precision. v14 first-pass default is bf16-mixed
    # per 2026-05-29 OOM diagnosis on RTX 5000 Ada (31 GiB): factorized
    # per-electrode SA over C=384 padded electrodes at d=256 exhausts
    # fp32 activations even at BS=8. bf16-mixed halves activation memory
    # at the standard transformer-training precision floor.
    precision: str | None = "bf16-mixed",
    fast_dev_run: bool | int = False,
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
    # B29 corpus mix lock 2026-05-27 PM-late.
    include_ajile12: bool = DEFAULT_INCLUDE_AJILE12,
    ref_operator_alpha: float = DEFAULT_REF_OPERATOR_ALPHA,
    corpus_mix: dict[str, float] | None = None,
    notch_filter_hz_by_corpus: dict[str, float] | None = None,
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
    # B37 (2026-06-10) encoder + joint-SSL knobs. ``pool`` picks the
    # electrode→parcel pooling: "cross_attn" (default, B36 learned pool) or
    # "mean" (B37 hard mean pool → freq-preserving thin parcel-SA latent).
    # ``latent_depth`` is the mean-pool latent's parcel-SA block count (D5,
    # inert under cross_attn). Both ride the brain-model config → every phase
    # shares one encoder. ``ssl_mode`` selects the SSL objective on the
    # ``joint_phase`` experiment: "staged" (default, B36 single-term-per-phase,
    # byte-identical) or "joint" (B37 D7 composite-mask M2+M4 single forward).
    # ``--pool mean`` defaults ``ssl_mode`` to "joint" in main(); the joint path
    # requires pool="mean". Joint-only knobs (inert on staged / P3 / P4):
    # ``lambda_m4`` (M4 term weight), ``m{2,4}_predictor_depth`` (D9 per-tap
    # predictor depths), ``joint_frontend_lr_scale`` / ``joint_parcel_lr_scale``
    # (D8 discriminative LR, both default 1.0 = no discrimination — distinct
    # from the B33 P3 ``parcel_lr_scale`` knob below). ``m{2,4}_mask_{type,ratio}``
    # + ``predictor_{depth,hidden,n_heads}`` expose the masking + STAGED-predictor
    # sizing (the joint M2/M4 predictors share hidden/n_heads, depths per-tap).
    pool: tp.Literal["cross_attn", "mean"] = DEFAULT_POOL,
    latent_depth: int = DEFAULT_LATENT_DEPTH,
    # B37+ (2026-06-11): mean-path latent cross-parcel mode ("parcel" default /
    # "joint" parcel×time freq-batched) + RGB-style mean|std stem channel. Both
    # ride the brain-model config (every phase shares the encoder); inert under
    # cross_attn. → [[project_v14_b37_meanpath_arch_temporal_gap_2026_06_11]].
    latent_mode: tp.Literal["parcel", "joint"] = DEFAULT_LATENT_MODE,
    mean_pool_std: bool = DEFAULT_MEAN_POOL_STD,
    ssl_mode: tp.Literal["staged", "joint"] = DEFAULT_SSL_MODE,
    lambda_m4: float = DEFAULT_LAMBDA_M4,
    m2_predictor_depth: int = DEFAULT_M2_PREDICTOR_DEPTH,
    m4_predictor_depth: int = DEFAULT_M4_PREDICTOR_DEPTH,
    joint_frontend_lr_scale: float = DEFAULT_JOINT_FRONTEND_LR_SCALE,
    joint_parcel_lr_scale: float = DEFAULT_JOINT_PARCEL_LR_SCALE,
    m2_mask_type: str = DEFAULT_M2_MASK_TYPE,
    # SSL-sweep override knobs (#sweep, 2026-06-11). ``None`` (the new default)
    # resolves to the 6/03 masking-lock constant inside the builder, so an unset
    # flag is byte-identical to the prior hardcoded held-out ratio; a float
    # overrides the M2 / M4 masked fraction for a sweep cell. The ssl/mask.py
    # signature defaults (0.50 / 0.20) are NOT touched — only what dispatch
    # forwards changes.
    m2_mask_ratio: float | None = None,
    m4_mask_type: str = DEFAULT_M4_MASK_TYPE,
    m4_mask_ratio: float | None = None,
    # M2 structured-band WIDTH along each axis (#sweep, 2026-06-11). The 6/03
    # mask holds the masked FRACTION fixed and varies granularity: a LARGER floor
    # = wider, fewer bands = harder reconstruction (the predictor can no longer
    # copy the immediate spectro-temporal neighbour). Tests Ben's "1D bands too
    # narrow → widen" hypothesis. Defaults reproduce the lock (time 2 / freq 1)
    # exactly. NB: the realized masked fraction rounds DOWN when the width does
    # not divide round(frac·n_valid) (see ssl/mask.py::_sample_axis_bands).
    m2_time_band_floor: int = DEFAULT_M2_TIME_BAND_FLOOR,
    m2_freq_band_floor: int = DEFAULT_M2_FREQ_BAND_FLOOR,
    # 2STFT dual-band M2 mask block geometry (FE-2STFT only; inert on the
    # single-band path). ``None`` → the dual-band sampler defaults (low: one
    # 3-wide freq block across all time; high: {3,3,2} time cols across all
    # freq). Each band overrides only when BOTH its width and nbands are set:
    # ``width`` = contiguous patches per block, ``nbands`` = #blocks. Low band
    # → ``low_freq_floor=width`` + ``low_freq_frac=width*nbands/F_p_low`` (the
    # sampler scales frac to n_bands); high band → ``high_time_widths`` =
    # ``(width,) * nbands`` (fixed absolute cols, does NOT scale with T). Lets
    # us soften the dual-band masking (Ben's "≤2-wide / ~40%" easier-mask run)
    # without touching the single-band locks above.
    m2_low_freq_width: int | None = None,
    m2_low_freq_nbands: int | None = None,
    m2_high_time_width: int | None = None,
    m2_high_time_nbands: int | None = None,
    # High-band ANCHOR-DILATE mode (Ben 2026-06-13 easier-mask regime; both → on).
    # Instead of the disjoint width/nbands multiset, the high band samples
    # ``frac`` of time positions and dilates each to ``width`` (mask the position
    # + the next width-1), OVERLAPS allowed → union. e.g. frac 0.30 + width 2 on
    # T_high_p=80 masks ~51% of high time-cols. Mutually exclusive with the
    # disjoint high knobs above (validated below).
    m2_high_anchor_frac: float | None = None,
    m2_high_anchor_width: int | None = None,
    # EMA teacher momentum τ override (#sweep). ``None`` resolves to
    # DEFAULT_EMA_TAU (== ssl/ema.py P1_EMA_TAU/P2_EMA_TAU, B26 lock) so an unset
    # flag reproduces the prior hardcoded 0.99925 exactly; a float in (0, 1)
    # overrides it for an EMA-τ sweep (R-ema-tau-{...}). Joint-only inside the
    # builder (P3/P4 use no EMA teacher), so passing it on every phase is inert
    # off the SSL path. The ema.py constants are NOT mutated.
    ema_tau: float | None = None,
    # Heteroscedastic / inverse-variance M4 (P2) loss weighting (joint +
    # mean_pool_std only; project_v14_heteroscedastic_ssl_loss). OFF by default;
    # ``--m4-precision-weight`` enables it and ``--m4-precision-alpha`` sets the
    # ``n^α`` exponent (α=1 raw count; α<1 damps high-n electrode redundancy).
    m4_precision_weight: bool = False,
    m4_precision_alpha: float = 1.0,
    m4_precision_floor_pct: float = 25.0,
    m4_precision_cap: float = 10.0,
    m4_precision_mode: str = "downweight_dof",
    m4_precision_nref: float = 11.0,
    predictor_hidden: int = DEFAULT_PREDICTOR_HIDDEN,
    predictor_n_heads: int = DEFAULT_PREDICTOR_N_HEADS,
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
    # 2026-06-08 ragged front-end (#91): DEFAULT-ON (flipped 2026-06-12 — Ben:
    # "all ragged optimizations default on"). When ON, the encoder's per-row
    # token blocks run only over valid rows (cross_attn: valid electrodes;
    # mean-pool: COVERED parcels) — pad/uncovered rows gathered out, scattered
    # back as zeros before the pool — and the P1 M2 loss drops pad rows.
    # Valid-row M2/M4 + P2 loss are bit-identical; only the P1 loss mean changes
    # (no longer dilutes on pad). Cuts the token-block + predictor FFN by the pad
    # fraction (~50% at BT-Lite c_max=256 ⇒ unblocks raw bs=8 + ~halves
    # front-end step time on padded batches). On the mean path with masking
    # active the uncovered-parcel drop is FOLDED into ragged_token's visible-token
    # gather (one combined gather, no double-drop). --no-ragged-frontend restores
    # the dense path.
    ragged_frontend: bool = True,
    # 2026-06-09 ragged parcel drop (#92): DEFAULT-ON (flipped after the
    # warm-4-GPU throughput matrix). When ON, the P2 parcel encoder gathers only
    # covered-&-visible parcels (pad-to-max per sample), runs the parcel/time SA
    # over them, and scatters back as zeros. Because the dense path already
    # key-masks uncovered/masked parcels out of attention, the kept-parcel M4 is
    # loss-neutral (bit-identical up to ~1e-6 matmul reassociation); the win is
    # cutting the parcel-SA + predictor FFN by the dropped fraction. Bit-identical
    # ⇒ safe to default-on. Escape: ``--no-ragged-parcel``. Independent of
    # ``ragged_frontend`` (front-end vs parcel).
    ragged_parcel: bool = True,
    # 2026-06-09 P1 token drop (#93, V-JEPA-2 visible-only front-end): DEFAULT-ON.
    # When ON, the per-electrode joint token blocks run ONLY over the VISIBLE
    # (non-masked, valid, freq-valid-kept) (t_p, f_p) tokens — gather → pad-to-max
    # → scatter back as zeros — with a per-row time-RoPE. Unlike ragged_frontend/
    # ragged_parcel this is a REPRESENTATION CHANGE: the dense path zeroes masked
    # tokens but still lets them attend (a zero LayerNorm-normalises to a constant
    # key), so dropping them makes the encoder truly visible-only — which IS the
    # canonical V-JEPA 2 §2.1 design (the zero-and-attend dense path was the
    # deviation, not the spec). On those grounds #93 is the default; the P4
    # transfer ablation is a CONFIRM (canonical ≥ legacy) run in parallel, NOT a
    # gate that blocks the default. Bit-identical to key-masking the masked tokens
    # out. Escape: ``--no-ragged-token``. Independent of ragged_frontend/
    # ragged_parcel (token vs row vs parcel axis). Dominant throughput lever in P1:
    # ~3× per-step compute (1.80→0.60 s/step, no-compile, warm 4-GPU matrix).
    ragged_token: bool = True,
    # 2026-06-09 predictor context+query drop (#94, V-JEPA-2 visible-only
    # predictor): DEFAULT-ON. When ON, the JEPA predictor gathers ONLY the
    # visible context cells + ONLY the real masked query slots (pad-to-max)
    # instead of feeding the full grid and key-padding the rest. BIT-identical to
    # the dense path (padded slots are key-masked out → exp(NEG_INF)=0), a pure
    # FLOP cut on the predictor attention/FFN at high mask ratio ⇒ safe to
    # default-on. Escape: ``--no-ragged-predictor``. Covers BOTH P1 and P2 (one
    # predictor per joint phase). Joint-only (no effect on P3/P4).
    ragged_predictor: bool = True,
    # 2026-06-09 freq-pos lock (A-JEPA, arXiv 2311.15830): positional encoding
    # for the ORDERED freq-patch axis. "sinusoidal" (default) = fixed MAE sincos
    # (absolute code + smoothness/ordering prior), matching how A-JEPA/AudioMAE
    # position the spectral axis; threaded to the encoder front-end AND mirrored
    # to the P1 predictor's freq id-tag. "learned" = R-freq-learned-embed sister
    # (pre-lock learned table). NOT bit-identical (a representation change) —
    # joint-phase only; time stays RoPE, parcel stays learned.
    freq_pos: str = "sinusoidal",
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
    # §7/B01 no-WD param-group split (#40). True (default) exempts biases +
    # LayerNorm/RMSNorm γβ (ndim<=1) + named embeds from weight decay — the
    # nanoGPT/timm convention. Inert when weight_decay<=0 (the wd=0 path stays
    # bit-identical). ``--no-wd-exclude-norms`` flips it False (uniform decay,
    # the uniform-decay falsifier). Threaded onto every V14 phase experiment
    # (joint P1/P2, P3, P4 frozen probe); the base supervised CE path ignores it
    # (it builds a neuraltrain BrainModule with no custom optimizer split).
    wd_exclude_norms: bool = True,
    # MON-TEACHER-FEATURE-RANK collapse thresholds (#74), joint-only (P1/P2 run
    # RankMe; P3/P4 don't). None → the canonical 0.5 warn / 0.25 alarm defaults
    # (single-sourced in monitors/teacher_rank.py). Exposed so the recalibration
    # sweep can lower them toward the measured ~0.31 BT floor without editing the
    # constants; the module validates 0 < alarm < warn <= 1. Run-gating: the
    # collapse-guard kills on the per-step alarm/warn flags these set.
    rankme_warn_threshold: float | None = None,
    rankme_alarm_threshold: float | None = None,
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

    # Resolve the atlas to its matched (depth-wm column, parcel vocabulary) pair —
    # the SINGLE source so the column and vocabulary can never desync — and derive
    # k_parcels from the vocabulary length (DK→80, DKT→74). Both the support and
    # the valid_mask extractor are built from THIS pair (verified by
    # _assert_support_valid_config_agree below).
    atlas_label_column, atlas_parcel_labels = atlas_spec(atlas)
    k_parcels = len(atlas_parcel_labels)

    _validate_choice("subtype_embed_vocab", subtype_embed_vocab, SUBTYPE_EMBED_VOCABS)
    subtype_vocab_size = 2 if subtype_embed_vocab == "binary" else 3
    _validate_choice("loss_variant", loss_variant, LOSS_VARIANTS)
    _validate_choice("jepa_phase", jepa_phase, JEPA_PHASES)

    # SSL-sweep overrides (#sweep, 2026-06-11): resolve the None sentinels to the
    # locked constants so an unset flag is byte-identical to the prior hardcoded
    # value, and range-check a passed override the same way the BrainModule /
    # mask samplers do (τ ∈ (0, 1) open; held-out ratios ∈ [0, 1)). Done here so
    # BOTH the single-phase main() build and the --chain builds (which share
    # _common_build_kwargs) get the same resolution + a fail-at-dispatch error.
    if ema_tau is None:
        ema_tau = DEFAULT_EMA_TAU
    elif not 0.0 < ema_tau < 1.0:
        raise ValueError(f"ema_tau must lie in (0.0, 1.0); got {ema_tau}")
    if m2_mask_ratio is None:
        m2_mask_ratio = DEFAULT_M2_MASK_RATIO
    elif not 0.0 <= m2_mask_ratio < 1.0:
        raise ValueError(f"m2_mask_ratio must lie in [0.0, 1.0); got {m2_mask_ratio}")
    if m4_mask_ratio is None:
        m4_mask_ratio = DEFAULT_M4_MASK_RATIO
    elif not 0.0 <= m4_mask_ratio < 1.0:
        raise ValueError(f"m4_mask_ratio must lie in [0.0, 1.0); got {m4_mask_ratio}")
    # Band WIDTHS are a positive count of grid cells; ssl/mask.py clamps width to
    # [1, n_valid] internally, but reject < 1 here so a typo fails at dispatch
    # rather than silently snapping to a 1-cell band.
    if m2_time_band_floor < 1:
        raise ValueError(f"m2_time_band_floor must be >= 1; got {m2_time_band_floor}")
    if m2_freq_band_floor < 1:
        raise ValueError(f"m2_freq_band_floor must be >= 1; got {m2_freq_band_floor}")
    # 2STFT dual-band block geometry: width/nbands override a band only as a PAIR
    # (one without the other is an ambiguous half-spec → fail fast rather than
    # silently fall back to the sampler default). Both must be >= 1 cells/blocks.
    for _w, _n, _name in (
        (m2_low_freq_width, m2_low_freq_nbands, "low_freq"),
        (m2_high_time_width, m2_high_time_nbands, "high_time"),
    ):
        if (_w is None) != (_n is None):
            raise ValueError(
                f"m2_{_name}_width and m2_{_name}_nbands must be set together "
                f"(got width={_w}, nbands={_n}); set both to override the "
                "dual-band default, or neither to keep it."
            )
        if _w is not None and _w < 1:
            raise ValueError(f"m2_{_name}_width must be >= 1; got {_w}")
        if _n is not None and _n < 1:
            raise ValueError(f"m2_{_name}_nbands must be >= 1; got {_n}")
    # High-band ANCHOR-DILATE: frac/width are a pair, and the anchor mode is
    # mutually exclusive with the disjoint high-time multiset (two different
    # high-band placements; setting both is ambiguous → fail loud).
    if (m2_high_anchor_frac is None) != (m2_high_anchor_width is None):
        raise ValueError(
            "m2_high_anchor_frac and m2_high_anchor_width must be set together "
            f"(got frac={m2_high_anchor_frac}, width={m2_high_anchor_width})."
        )
    if m2_high_anchor_frac is not None:
        if not 0.0 < m2_high_anchor_frac <= 1.0:
            raise ValueError(
                f"m2_high_anchor_frac must lie in (0.0, 1.0]; got {m2_high_anchor_frac}"
            )
        if m2_high_anchor_width is not None and m2_high_anchor_width < 1:
            raise ValueError(
                f"m2_high_anchor_width must be >= 1; got {m2_high_anchor_width}"
            )
        if m2_high_time_width is not None or m2_high_time_nbands is not None:
            raise ValueError(
                "high-band anchor mode (--m2-high-anchor-frac/-width) and the "
                "disjoint multiset (--m2-high-time-width/-nbands) are mutually "
                "exclusive high-band placements; set only one."
            )
    # Heteroscedastic M4 precision weighting (project_v14_heteroscedastic_ssl_loss):
    # fail fast at dispatch (the BrainModule re-validates) — it needs the σ source
    # (mean|std pool) and the joint B37 path. α >= 0 (α=1 raw count, α<1 damps high-n).
    if m4_precision_weight:
        if not mean_pool_std:
            raise ValueError(
                "--m4-precision-weight needs the mean|std pool as the σ source; "
                "pass --mean-pool-std (the B37 RGB-style pool)."
            )
        if ssl_mode != "joint":
            raise ValueError(
                "--m4-precision-weight is the heteroscedastic M4 (P2) loss weight "
                f"and only applies to the joint B37 path; got ssl_mode={ssl_mode!r}."
            )
    if m4_precision_alpha < 0.0:
        raise ValueError(f"m4_precision_alpha must be >= 0; got {m4_precision_alpha}")

    optim_cfg = _build_optim_cfg(
        lr=lr, lr_schedule=lr_schedule, warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio, weight_decay=weight_decay,
        optimizer_name=optimizer_name, adam_betas=adam_betas,
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
    # #80 whole-movie raw-|STFT| cache root. Default ON: when an extractor-cache
    # root exists (DCC) and the caller hasn't overridden, derive a sibling
    # ``v14_spec_cache`` dir so the per-run session_robust_z whole-movie STFT is
    # materialized to an fp32 memmap ONCE and every later same-front-end-config run
    # slices it (no 9-min recompute). OFF on laptop/tests (no cache root) and when
    # ``disable_spec_cache``. Armed only on the default raw MultiStftView built
    # below; a custom electrode_tokens_extractor manages its own spec_cache_dir.
    if disable_spec_cache:
        spec_cache_dir = None
    elif spec_cache_dir is None and extractor_cache_folder is not None:
        spec_cache_dir = str(Path(extractor_cache_folder) / "v14_spec_cache")

    # 2STFT dual-band front-end emits a SECOND batch key (electrode_tokens_high).
    # None on the single-grid raw/fbank path + when a custom extractor is supplied.
    high_band_extractor: tp.Any | None = None
    # 3STFT converged front-end emits THREE band keys (electrode_tokens_slow rides
    # the primary electrode_tokens_extractor; beta/hg here). None off the 3stft path.
    beta_band_extractor: tp.Any | None = None
    hg_band_extractor: tp.Any | None = None
    # True only for the dual-band path → threads to the encoder's per_band_stem.
    per_band_stem = False
    if electrode_tokens_extractor is None:
        # DP4 (2026-06-09): MNE-LOF bad-channel drop is GATED OFF at dispatch.
        # LOF (drop_bads) removes electrodes PER-TRIAL inside the front-end before
        # the scatter, packing the survivors into rows 0..k-1. But the DK
        # ``support`` / ``valid_mask`` extractors are per-SUBJECT static tensors
        # built from the full ``voltage_electrode_order`` — they cannot represent a
        # per-trial drop. So electrode_tokens (LOF-packed) would desync from
        # support/valid_mask (full voltage order) and silently route voltages into
        # the wrong parcels, even for a single subject. Re-enable only after
        # support/valid_mask are made per-event and drop the same LOF survivors in
        # lockstep (the removed kwargs-forwarding wiring lives in git history; the
        # view-level LOF mechanism stays covered by extractors/test_lof_wiring.py).
        # See reports/bt_alignment/electrode_desync_damage_2026_06_09.md.
        if lof_bad_channels:
            raise ValueError(
                "lof_bad_channels is gated OFF (DP4 row-alignment hazard): LOF "
                "drops electrodes per-trial before the front-end scatter, but the "
                "DK support / valid_mask extractors are per-subject static and "
                "cannot mirror a per-trial drop, so electrode_tokens would desync "
                "from support/valid_mask (voltages routed to the wrong parcels). "
                "Keep LOF off until per-event support/valid_mask plumbing lands. "
                "See reports/bt_alignment/electrode_desync_damage_2026_06_09.md."
            )
        # Front-end config shared by every default view (raw + both 2STFT bands):
        # WS-C / C2 (B36) + FE-RAW-1 — RAW |STFT| (apply_log=False), C4 0.5 Hz HPF
        # removes DC + slow drift, C3 StandardScaler dropped (robust-z downstream).
        common_fe_kwargs: dict[str, tp.Any] = dict(
            event_types="Ieeg",
            car="shaft",
            notch_filter=effective_bt_notch_hz,
            filter=(0.5, None),
            scaler=None,
            apply_log=False,
            channel_order="original",
            c_max=c_max,
            session_robust_z=session_robust_z,
            spec_only=spec_only,
        )
        if cache_band is not None:
            # 3STFT per-band cache build (--cache-band, cache-only). ONE named band
            # of the locked 2/2/2 ladder rides the single-grid electrode_tokens
            # slot (per_band_stem stays False → no encoder dual/triple-band path is
            # needed; --cache-only exits before the model is ever instantiated).
            # Constructed EXACTLY like the future --frontend 3stft training run
            # (same common_fe_kwargs + STFT_3BAND_* + band_<name> subdir naming),
            # so the spec-cache namespace matches and the run HITs this cache. The
            # slow band carries band_channelization="cartesian" (Re/Im); beta/HG
            # default to "mag". band_hop is the band's native hop = hop_length.
            band_const = {
                "slow": STFT_3BAND_SLOW,
                "beta": STFT_3BAND_BETA,
                "hg": STFT_3BAND_HG,
            }[cache_band]
            band_spec_cache = (
                str(Path(spec_cache_dir) / f"band_{cache_band}")
                if spec_cache_dir is not None else None
            )
            electrode_tokens_extractor = MultiStftView(
                **common_fe_kwargs, front_end="band", **band_const,
                hop_length=int(band_const["band_hop"]),
                spec_cache_dir=band_spec_cache,
            )
        elif frontend == "2stft":
            # The dual-band stems live on the B37 mean-pool path only (the encoder
            # raises the same check at build, but fail at dispatch for a clear msg).
            if pool != "mean":
                raise ValueError(
                    f"--frontend 2stft requires --pool mean (the 2STFT dual-band "
                    f"stem is the B37 mean-pool path); got pool={pool!r}."
                )
            # Dual-band 2STFT (Ben 2026-06-12): TWO single-band MultiStftViews at
            # DIFFERENT native hops (no common time grid). LOW (N=512/hop256, 4-56
            # Hz, 14 bins @ 8 Hz) → ``electrode_tokens``; HIGH (N=128/hop64, 64-192
            # Hz, 9 bins @ 16 Hz) → ``electrode_tokens_high``. Each band runs its
            # OWN session-robust-z fit (per-band display + cache). Separate spec
            # caches (distinct STFT config → distinct whole-movie memmap), kept in
            # sibling subdirs so the two never collide. The encoder's two Conv2d
            # band stems reconcile them onto one 62.5 ms token grid (per_band_stem).
            low_spec_cache = (
                str(Path(spec_cache_dir) / "band_low")
                if spec_cache_dir is not None else None
            )
            high_spec_cache = (
                str(Path(spec_cache_dir) / "band_high")
                if spec_cache_dir is not None else None
            )
            electrode_tokens_extractor = MultiStftView(
                **common_fe_kwargs, front_end="band", **STFT_2BAND_LOW,
                hop_length=int(STFT_2BAND_LOW["band_hop"]),
                spec_cache_dir=low_spec_cache,
            )
            high_band_extractor = MultiStftView(
                **common_fe_kwargs, front_end="band", **STFT_2BAND_HIGH,
                hop_length=int(STFT_2BAND_HIGH["band_hop"]),
                spec_cache_dir=high_spec_cache,
            )
            per_band_stem = True
        elif frontend == "3stft":
            # Converged-arch triple-band 3STFT 2/2/2 (Ben 2026-06-17): THREE
            # single-band MultiStftViews at the slow/beta/HG native hops, each its
            # OWN session-robust-z fit + spec cache. The slow band rides the
            # primary ``electrode_tokens_extractor`` (so the downstream ref_idx /
            # n_time_bins derivation runs on it like any single-grid view); beta/HG
            # are carried in the band locals. The three band views are built
            # IDENTICALLY to the ``--cache-band`` per-band builds (same
            # common_fe_kwargs + STFT_3BAND_* + ``band_<name>`` spec-cache subdir),
            # so a 3stft run HITs caches a prior --cache-band sweep wrote. Note:
            # ``per_band_stem`` stays False — the converged model owns its own
            # per-electrode 3-band frontend; the encoder's dual-band stem path is
            # NOT used here.
            band_spec = lambda name: (  # noqa: E731
                str(Path(spec_cache_dir) / f"band_{name}")
                if spec_cache_dir is not None else None
            )
            electrode_tokens_extractor = MultiStftView(
                **common_fe_kwargs, front_end="band", **STFT_3BAND_SLOW,
                hop_length=int(STFT_3BAND_SLOW["band_hop"]),
                spec_cache_dir=band_spec("slow"),
            )
            beta_band_extractor = MultiStftView(
                **common_fe_kwargs, front_end="band", **STFT_3BAND_BETA,
                hop_length=int(STFT_3BAND_BETA["band_hop"]),
                spec_cache_dir=band_spec("beta"),
            )
            hg_band_extractor = MultiStftView(
                **common_fe_kwargs, front_end="band", **STFT_3BAND_HG,
                hop_length=int(STFT_3BAND_HG["band_hop"]),
                spec_cache_dir=band_spec("hg"),
            )
        else:
            # FE-RAW-1 single F=50 raw |STFT| grid (hop=128 → 16 Hz, 8 Hz latent).
            electrode_tokens_extractor = MultiStftView(
                **common_fe_kwargs, spec_cache_dir=spec_cache_dir,
            )
    # 3STFT: the primary extractor is the SLOW band, cached under its own
    # ``electrode_tokens_slow`` key (NOT the single-grid ``electrode_tokens``) so
    # the converged segmenter's three band keys stay namespaced apart. Beta/HG
    # are cached below.
    primary_token_key = (
        "electrode_tokens_slow" if frontend == "3stft" else "electrode_tokens"
    )
    _apply_extractor_cache(
        electrode_tokens_extractor, primary_token_key, extractor_cache_folder
    )
    if high_band_extractor is not None:
        _apply_extractor_cache(
            high_band_extractor, "electrode_tokens_high", extractor_cache_folder
        )
    if beta_band_extractor is not None:
        _apply_extractor_cache(
            beta_band_extractor, "electrode_tokens_beta", extractor_cache_folder
        )
    if hg_band_extractor is not None:
        _apply_extractor_cache(
            hg_band_extractor, "electrode_tokens_hg", extractor_cache_folder
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

    # 2STFT band geometry → encoder per_band_stem (§3b). Derived from the band
    # VIEWS so the stem grid can never drift from the data: freq-bin counts from
    # each view's authoritative ``_band_bins()`` (physical-edge → rfft-k slice;
    # low 14, high 9), time-bin counts (STFT frame counts) from clip_len at each
    # band's native hop. The encoder floors these onto the patch grid (low fk2/tk1
    # → 7 patches @ 8 Hz; high fk3/tk2 → 3 patches @ 16 Hz; F_p=7+3=10). None on
    # the single-grid raw path. ``n_time_bins`` above already reflects the LOW
    # band (electrode_tokens) — the top-level RoPE ceiling is harmless-unused on
    # the dual-band path (the encoder sizes the dual-rate RoPE from these maxima).
    band_geometry: dict[str, int] = {}
    if per_band_stem:
        assert high_band_extractor is not None  # set together with per_band_stem
        k0_lo, k1_lo = electrode_tokens_extractor._band_bins()
        k0_hi, k1_hi = high_band_extractor._band_bins()
        band_geometry = {
            "band_low_n_freq_bins": k1_lo - k0_lo + 1,
            "band_high_n_freq_bins": k1_hi - k0_hi + 1,
            "band_low_n_time_bins": electrode_tokens_extractor.n_time_bins_for_duration(
                clip_len
            ),
            "band_high_n_time_bins": high_band_extractor.n_time_bins_for_duration(
                clip_len
            ),
        }

    # Leakage decouple (#82): the SSL/distill phases (P1/P2 joint, P3 distill)
    # pretrain on the Neuroprobe-legal corpus (V14_PRETRAIN_SESSIONS, disjoint
    # from the 12 eval sessions) under a "Pretrain" split that sends every legal
    # session to train minus a per-session monitoring tail. Only the supervised
    # P4 probe (phase4_frozen_probe) trains/tests on the eval split
    # (CrossSession/CrossSubject). Before this, all phases shared one chain, so
    # the CrossSession coupling pretrained SSL on the held-out eval subject's
    # other (also-eval) trial. ``--mode``/``--eval-mode`` now apply to the P4
    # eval ONLY; for SSL they are overridden here and ``mode`` instead controls
    # the clip-SAMPLING budget (lite=3500/class gate vs full=all). The runtime
    # leakage guard (V14Joint/Phase3Experiment.enforces_pretrain_leakage_guard)
    # is the fail-closed backstop on the realized loaders.
    study_mode, chain_eval_mode = _resolve_corpus_mode(
        joint_phase=joint_phase, p3_distill=p3_distill,
        mode=mode, eval_mode=eval_mode,
    )
    # Clip-sampling budget: SSL phases keep the raw --mode budget (lite=3500/class
    # gate vs full=all); the P4 probe uses its own leaderboard protocol, so the
    # cap follows the RESOLVED P4 universe (study_mode lite/nano) — a `--mode full`
    # run still evaluates P4 on the capped Lite eval, a consistent parity cell.
    budget_mode = mode if study_mode in ("pretrain", "p3_distill") else study_mode
    if study_mode in ("pretrain", "p3_distill"):
        corpus = (
            "V14_PRETRAIN_SESSIONS (13 sess)" if study_mode == "pretrain"
            else "V14_P3_DISTILL_SESSIONS (12 sess, S8 dropped: no teacher)"
        )
        print(
            f"[decouple #82] SSL phase → study mode={study_mode!r} "
            f"({corpus}, leakage-free), eval_mode={chain_eval_mode!r}, "
            f"holdout={pretrain_holdout_fraction}; clip-sampling budget from "
            f"--mode={mode!r}"
        )
    else:
        print(
            f"[decouple #82] P4 probe → study mode={study_mode!r} "
            f"(BT_LITE leaderboard eval, S5-free), eval_mode={chain_eval_mode!r}, "
            f"clip-budget lite={budget_mode == 'lite'}/nano={budget_mode == 'nano'} "
            f"(from resolved universe, NOT --mode={mode!r})"
        )
    session_subset: tuple[tuple[int, int], ...] | None = None
    if cache_session_index is not None:
        corpus_sessions = list(_SESSIONS_BY_MODE[study_mode])
        if not 0 <= cache_session_index < len(corpus_sessions):
            raise ValueError(
                f"--cache-session-index {cache_session_index} out of range for "
                f"study mode={study_mode!r} ({len(corpus_sessions)} sessions: "
                f"valid 0..{len(corpus_sessions) - 1})"
            )
        s_id, t_id = corpus_sessions[cache_session_index]
        session_subset = ((int(s_id), int(t_id)),)
        print(
            f"[cache-only] session-index {cache_session_index} → "
            f"({s_id}, {t_id}) (of {len(corpus_sessions)} in mode={study_mode!r})"
        )
    # Eval-always-lite invariant (Ben 2026-06-15): the P4 frozen-probe IS the
    # Neuroprobe-Lite leaderboard cell, so it MUST run the Lite montage — the Lite
    # subset is applied PRE-CAR (loader), so shaft-CAR references Lite electrodes
    # ONLY and the result depends on zero out-of-budget info (reproducible from the
    # Lite budget). The chain pins electrode_set="lite" for P4 and "all" for SSL;
    # --electrode-set auto resolves identically. Fail loud if a non-Lite montage
    # ever routes into eval.
    if phase4_frozen_probe and electrode_set != "lite":
        raise ValueError(
            "P4 frozen-probe eval must run electrode_set='lite' (Neuroprobe-Lite "
            f"parity + reproducibility), got {electrode_set!r}. The eval cell may "
            "never reference out-of-budget electrodes."
        )
    trial_durations: dict[str, int] | None = None
    if trial_durations_path is not None:
        import json

        with open(trial_durations_path) as fh:
            trial_durations = {str(k): int(v) for k, v in json.load(fh).items()}
    study = Wang2024Treebank(
        path=Path(bt_root), mode=study_mode,
        infra_timelines={"cluster": None},
        session_subset=session_subset,
        electrode_set=electrode_set,
        trial_durations=trial_durations,
    )
    # Class-balance only the P4 eval (Neuroprobe parity). SSL phases
    # (pretrain/p3_distill) are label-free → keep EVERY word + nonverbal anchor
    # (no minority-class bottleneck), lifting movie coverage from the balanced
    # subset to ~96-99%. See _resolve_corpus_mode + the BTWordEvents.balance
    # field. --mode still bounds the SSL anchor count (lite/nano cap per class).
    ssl_phase = study_mode in ("pretrain", "p3_distill")
    word_events = BTWordEvents(
        tasks=(task,),
        binary_tasks=binary_tasks,
        lite=(budget_mode == "lite"),
        nano=(budget_mode == "nano"),
        balance=(not ssl_phase),
        eval_mode=chain_eval_mode,
        fold_index=fold_index,
        test_subject_id=test_subject_id,
        test_trial_id=test_trial_id,
        bt_root=bt_root,
        pretrain_holdout_fraction=pretrain_holdout_fraction,
    )
    chain = ns.Chain(steps=[study, word_events])

    dk_extractor = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=bt_root, unmapped_policy="zero",
        c_max=c_max,
        parcel_labels=atlas_parcel_labels, label_column=atlas_label_column,
        exclude_single_electrode_parcels=exclude_single_electrode_parcels,
        electrode_set=electrode_set,
    )
    _apply_extractor_cache(dk_extractor, "dk_support", extractor_cache_folder)
    valid_mask_extractor = ElectrodeValidMask(
        event_types="Ieeg", bt_root=bt_root, c_max=c_max,
        unmapped_policy="zero", electrode_set=electrode_set,
        parcel_labels=atlas_parcel_labels, label_column=atlas_label_column,
        exclude_single_electrode_parcels=exclude_single_electrode_parcels,
    )
    _apply_extractor_cache(valid_mask_extractor, "valid_mask", extractor_cache_folder)
    _assert_support_valid_config_agree(dk_extractor, valid_mask_extractor)

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
    if frontend == "3stft":
        # Converged path: THREE band keys (no single-grid electrode_tokens). The
        # converged module reads electrode_tokens_slow/beta/hg + support +
        # valid_mask directly; ref_idx/subject_subtype are V14ParcelPerceiver
        # encoder tokens the converged model never consumes, so they are omitted.
        segmenter_extractors: dict[str, tp.Any] = {
            "electrode_tokens_slow": electrode_tokens_extractor,
            "electrode_tokens_beta": beta_band_extractor,
            "electrode_tokens_hg": hg_band_extractor,
            "support": dk_extractor,
            "valid_mask": valid_mask_extractor,
        }
    else:
        segmenter_extractors = {
            "electrode_tokens": electrode_tokens_extractor,
            "support": dk_extractor,
            "valid_mask": valid_mask_extractor,
            "ref_idx": ref_idx_extractor,
            "subject_subtype": subtype_extractor,
        }
    # 2STFT: the HIGH band is a SECOND batch key sharing electrodes/support/
    # valid_mask with the low band (electrode_tokens). The collate pads it to the
    # same C_max with its own (F_high, T_high) axes; the encoder per_band_stem
    # consumes both. Present only on the dual-band path.
    if high_band_extractor is not None:
        segmenter_extractors["electrode_tokens_high"] = high_band_extractor
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
        # Layer-2 clip filter: SSL phases only. P4 (frozen probe, ssl_phase=False)
        # keeps it None so the Neuroprobe-Lite parity clip set is never altered.
        bad_window_dir=(bad_window_dir if ssl_phase else None),
        # SSL-only: never group eval batches (P4 parity lock).
        group_by_session=(group_by_session if ssl_phase else False),
        # SSL-only rank size-balancing; inert without group_by_session.
        balance_ranks_by_size=(balance_ranks if ssl_phase else False),
        # SSL-only same-session-across-ranks bubble removal; inert w/o group_by_session.
        same_session_across_ranks=(same_session_across_ranks if ssl_phase else False),
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
    # C5: SLURM-level requeue on preemption (submitit renders bool True as a bare
    # ``#SBATCH --requeue``). The within-phase last.ckpt resume does the rest.
    if requeue:
        infra_cfg["slurm_additional_parameters"] = {"requeue": True}

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
        # B37 D8: in joint SSL the front-end + parcel side ride their own
        # discriminative-LR scales (both default 1.0 = no discrimination). The
        # staged ``frontend_lr_scale`` (B36 P2 base/10) does not apply, so swap
        # in the joint scale here; staged keeps the B36 value.
        _joint = ssl_mode == "joint"
        _frontend_lr_scale = joint_frontend_lr_scale if _joint else frontend_lr_scale
        extra_experiment_kwargs = {
            "phase": JOINT_PHASE_VALUE,
            # B30-dispatch-sister-flags: the ``--latent-valid-override`` /
            # ``--sa-mask-mode`` CLI selectors were culled 2026-06-13 (their
            # non-"support"/non-"bidirectional" choices were always
            # NotImplementedError sisters). Hardcode the B30-lock values so the
            # run-record YAML still records them; the V14JointExperiment field
            # validators keep raising on any non-default that reaches them.
            "latent_valid_override": "support",
            "sa_mask_mode": "bidirectional",
            # B31 loss-variant selector: 2-term default + 3 sister arms.
            "loss_variant": loss_variant,
            # B36 staged masked-JEPA sub-phase (H4): p1 front-end M2 / p2
            # parcel M4. The staged P1->P2 handoff is WS-E; this picks the stage.
            # Inert under ssl_mode="joint" (the joint forward trains BOTH taps).
            "jepa_phase": jepa_phase,
            # B36 WS-E E2 (staged) / B37 D8 (joint): front-end discriminative-LR
            # scale (resolved above per ssl_mode).
            "frontend_lr_scale": _frontend_lr_scale,
            # B37 D7/D9 joint SSL: objective mode + per-tap predictor depths + M4
            # term weight + D8 parcel-side LR scale. Inert under ssl_mode="staged".
            "ssl_mode": ssl_mode,
            "lambda_m4": lambda_m4,
            "m2_predictor_depth": m2_predictor_depth,
            "m4_predictor_depth": m4_predictor_depth,
            "joint_parcel_lr_scale": joint_parcel_lr_scale,
            # #75/#76 masking + predictor sizing. The joint M2/M4 predictors
            # share hidden/n_heads; their depths come from m{2,4}_predictor_depth
            # above. The staged single predictor's depth (V14JointExperiment
            # ``predictor_depth`` field) keeps its own default — dispatch no
            # longer threads a CLI value for it (staged CLI path culled).
            "m2_mask_type": m2_mask_type,
            "m2_mask_ratio": m2_mask_ratio,
            "m4_mask_type": m4_mask_type,
            "m4_mask_ratio": m4_mask_ratio,
            # M2 band WIDTHS (#sweep). Forwarded to V14JointExperiment, which
            # passes them to ssl/mask.py::sample_m2_mask (time/freq band floors).
            "m2_time_band_floor": m2_time_band_floor,
            "m2_freq_band_floor": m2_freq_band_floor,
            # 2STFT dual-band block geometry (#sweep; None → sampler default).
            # Forwarded to V14JointExperiment → ssl/mask.py::sample_m2_dual_band_mask.
            "m2_low_freq_width": m2_low_freq_width,
            "m2_low_freq_nbands": m2_low_freq_nbands,
            "m2_high_time_width": m2_high_time_width,
            "m2_high_time_nbands": m2_high_time_nbands,
            # High-band anchor-dilate (both → on; mutually exclusive w/ multiset).
            "m2_high_anchor_frac": m2_high_anchor_frac,
            "m2_high_anchor_width": m2_high_anchor_width,
            # SSL-sweep EMA τ override (#sweep). Resolved above (None →
            # DEFAULT_EMA_TAU == 0.99925, the B26 lock); threads onto
            # V14JointExperiment.ema_tau, which feeds the EmaTeacher schedule.
            "ema_tau": ema_tau,
            # Heteroscedastic / inverse-variance M4 (P2) loss weighting (joint +
            # mean_pool_std only; project_v14_heteroscedastic_ssl_loss). OFF by
            # default — opt-in via --m4-precision-weight on the run.
            "m4_precision_weight": m4_precision_weight,
            "m4_precision_alpha": m4_precision_alpha,
            "m4_precision_floor_pct": m4_precision_floor_pct,
            "m4_precision_cap": m4_precision_cap,
            "m4_precision_mode": m4_precision_mode,
            "m4_precision_nref": m4_precision_nref,
            "predictor_hidden": predictor_hidden,
            "predictor_n_heads": predictor_n_heads,
            # #94 V-JEPA-2 visible-only predictor (context+query drop). Joint-only.
            "ragged_predictor": ragged_predictor,
            # §7/B01 no-WD param-group split (#40).
            "wd_exclude_norms": wd_exclude_norms,
            # MON-TEACHER-FEATURE-RANK thresholds (#74), joint-only. None → the
            # canonical 0.5/0.25 defaults; lowered for the recalibration sweep.
            "rankme_warn_threshold": rankme_warn_threshold,
            "rankme_alarm_threshold": rankme_alarm_threshold,
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
            # CN-3 consumer self-check: lets _build_standardizer assert the loaded
            # channel-stats provenance matches the merge used for the teacher.
            "whisper_layer_merge": whisper_layer_merge,
            # B33 §5 3b discriminative-LR scales (front-end base·scale, parcel
            # base·scale, connector base). No effect under 3a.
            "frontend_lr_scale": frontend_lr_scale,
            "parcel_lr_scale": parcel_lr_scale,
            # §7/B01 no-WD param-group split (#40).
            "wd_exclude_norms": wd_exclude_norms,
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
        # §7/B01 no-WD param-group split (#40): the trainable Linear has its bias
        # exempted (ndim<=1) when weight_decay>0.
        extra_experiment_kwargs = {"phase": 4, "wd_exclude_norms": wd_exclude_norms}
    elif (
        loss_variant != DEFAULT_LOSS_VARIANT
        or jepa_phase != DEFAULT_JEPA_PHASE
        or frontend_lr_scale != DEFAULT_FRONTEND_LR_SCALE
        or neural_lag_s != DEFAULT_NEURAL_LAG_S
    ):
        # B31 + B36 joint-only flags have semantic effect under the joint phase
        # only. The supervised Phase-4 path doesn't run the SSL aggregator, a
        # staged masked-JEPA phase, or the P2 discriminative-LR split, so a
        # non-default flag here would silently mis-record the sister / stage.
        # neural_lag_s is blocked here for a different reason: it DOES shift the
        # segmenter window on this path, and a non-zero P4 probe offset breaks
        # the leaderboard-parity [onset, onset+1 s] window.
        raise ValueError(
            "loss_variant / jepa_phase / frontend_lr_scale / neural_lag_s are "
            "B31/B36 joint-phase / distill selectors only; got "
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
    # in_allocation_ddp: the worker is already an srun rank → force the
    # multi-rank strategy regardless of tasks_per_node (which stays None so exca
    # runs local, not via submitit). cluster must be None on this path.
    if in_allocation_ddp and cluster is not None:
        raise ValueError(
            "in_allocation_ddp runs the experiment in-process under the worker's "
            f"own srun; it cannot also submit via --cluster {cluster!r}."
        )
    ddp_strategy = (
        "ddp_find_unused_parameters_true"
        if in_allocation_ddp
        else _resolve_ddp_strategy(tasks_per_node)
    )

    if frontend == "3stft":
        # Converged-arch route (Ben 2026-06-17): the WHOLE experiment is the
        # self-contained V14ConvergedSSL (its own frontend EMA teacher + the two
        # paradigm-B M2/M4 predictors + the M2/M4 losses), wrapped by
        # V14ConvergedExperiment. The V14ParcelPerceiver return below is NOT
        # reached. The shape is REQUIRED here (no silent run defaults — the widths
        # are Ben's to name at launch); a missing converged_* param is a hard
        # ValueError, never a numerics-silent fallback.
        _converged_shape = {
            "frontend_layers": converged_frontend_layers,
            "latent_layers": converged_latent_layers,
            "m2_pred_dim": converged_m2_pred_dim,
            "m2_pred_layers": converged_m2_pred_layers,
            "m4_pred_dim": converged_m4_pred_dim,
            "m4_pred_layers": converged_m4_pred_layers,
        }
        _missing = [k for k, v in _converged_shape.items() if v is None]
        if _missing:
            raise ValueError(
                "--frontend 3stft requires the converged shape params (no silent "
                f"run defaults): missing {_missing}. Pass converged_* widths."
            )
        from speech_decoding.experiments.v14_converged_experiment import (
            V14ConvergedExperiment,
        )
        # Mask-geometry overrides: only forward the ones the caller named; None
        # falls through to the V14ConvergedExperiment LOCKED pydantic defaults
        # (single source of truth — no default duplicated here).
        _converged_mask = {
            k: v for k, v in {
                "m2_hg_start_rate": converged_m2_hg_start_rate,
                "m2_hg_span": converged_m2_hg_span,
                "m2_beta_start_rate": converged_m2_beta_start_rate,
                "m2_beta_span": converged_m2_beta_span,
                "m2_slow_freq_tubes": converged_m2_slow_freq_tubes,
                "m4_parcel_mask_ratio": converged_m4_parcel_mask_ratio,
                # Static tube: only forward when the caller named a ratio (None ⇒
                # legacy variable masks, the experiment's locked default). p_fixed
                # rides along only when the tube is active.
                "tube_ratio": converged_tube_ratio,
                "tube_p_fixed": (
                    converged_tube_p_fixed
                    if converged_tube_ratio is not None else None
                ),
            }.items() if v is not None
        }
        # M4 precision-weight model config: the off-switch is explicit; α/n_ref
        # overrides only when named (None → V14Converged config defaults).
        _converged_m4_prec = {
            k: v for k, v in {
                "m4_precision_alpha": converged_m4_precision_alpha,
                "m4_precision_n_ref": converged_m4_precision_n_ref,
            }.items() if v is not None
        }
        return V14ConvergedExperiment(
            data=data,
            infra=infra_cfg,
            target_field="label",
            brain_model_config={
                "name": "V14Converged",
                "d_model": d_model,
                # DK atlas vocabulary length (K=80) — the parcel-tag embedding
                # table; derived, never a run default.
                "n_parcels": k_parcels,
                "n_heads": n_heads,
                # freq_pos is the LOCKED converged-arch decision (1d learnable freq
                # tag), forced regardless of the V14ParcelPerceiver "sinusoidal"
                # dispatch default.
                "freq_pos": "learned",
                # Clip length retimes the 3STFT ladder's TIME axis: SSL pretrain
                # runs 5 s clips (slow 21 / beta 81 / HG 161 frames → 190 tokens),
                # Phase-4 eval 1 s (38 tokens). The model derives its bands from
                # this float (bands_for_clip_len) — must match the segmenter's
                # `duration` (= clip_len, line ~1733) that sized the cached STFT.
                "clip_len_s": clip_len,
                **_converged_shape,
                "lambda_m2": converged_lambda_m2,
                "lambda_m4": converged_lambda_m4,
                "m4_precision_weight": not converged_m4_precision_off,
                **_converged_m4_prec,
            },
            # EMA τ: caller override (--ema-tau) or the ema.py-locked 0.99925.
            ema_tau=(ema_tau if ema_tau is not None else 0.99925),
            # Per-step mask RNG = the run seed (reproducible with the run).
            mask_seed=seed,
            **_converged_mask,
            # Static-shape forward (fixed-K / maskless). Bool, default False =
            # legacy ragged forward; needs tube_ratio + group_by_session when ON.
            static_forward=converged_static_forward,
            wd_exclude_norms=wd_exclude_norms,
            # Online linear probe (diagnostic, OFF by default). seed/n_cap stay at
            # the V14ConvergedExperiment defaults (run seed feeds mask_seed above;
            # the probe build reseeds deterministically from its own default).
            online_probe_enabled=online_probe_enabled,
            online_probe_cadence=online_probe_cadence,
            monitor_every_n_steps=monitor_every_n_steps,
            # SSL computes its own loss; this satisfies the required field and is
            # never read by V14ConvergedBrainModule. No accuracy metric (no head).
            loss={"name": "CrossEntropyLoss"},
            optim=optim_cfg,
            metrics=[],
            n_epochs=n_epochs,
            log_every_n_steps=log_every_n_steps,
            lr_log_interval=lr_log_interval,
            wandb_config=wandb_config,
            max_steps=max_steps,
            val_check_interval=val_check_interval,
            ckpt_ladder_every=ckpt_ladder_every,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            collapse_guard=collapse_guard,
            guard_warmup_min_step=warmup_steps,
            early_stopping_patience=early_stopping_patience,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            seed=seed,
            pretrained_ckpt=pretrained_ckpt,
            snapshot_ckpt_to=snapshot_ckpt_to,
            # Only the C-peek key; the module reads all band keys from batch.data.
            x_name="electrode_tokens_slow",
            accelerator="auto",
            devices="auto",
            ddp_strategy=ddp_strategy,
            precision=precision,
            fast_dev_run=fast_dev_run,
        )

    return experiment_cls(
        data=data,
        infra=infra_cfg,
        target_field="label",
        brain_model_config={
            "name": "V14ParcelPerceiver",
            "n_freq_bins": n_freq_bins,
            "n_time_bins": n_time_bins,
            "k_parcels": k_parcels,
            "d_model": d_model,
            "n_heads": n_heads,
            "depth_self_attn": depth,
            "m_sub_slots": m_sub_slots,
            "time_last_input": True,
            # 2STFT dual-band front end (§3b, Ben 2026-06-12): two Conv2d band
            # stems instead of the single F=50 grid. ``band_geometry`` (freq/time
            # bin counts) is derived above from the band VIEWS so the stem grid
            # matches the cached data exactly. Empty on the single-grid raw path
            # → per_band_stem=False and the band_* kwargs fall to encoder defaults
            # (inert). The band Conv2d kernels stay at the encoder defaults
            # (low fk2/tk1, high fk3/tk2) — the §7 falsifier sisters re-point them.
            "per_band_stem": per_band_stem,
            **band_geometry,
            # B28 cross-attn collapse: ``--cross-attn-positions`` was culled
            # 2026-06-13 (Perceiver, thrown away). The encoder keeps its own
            # ``cross_attn_positions=None`` (→ [0]) and ``eps`` internal
            # defaults; dispatch no longer threads either.
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
            # 2026-06-08 ragged front-end (#91): default-off; skip pad
            # electrodes in the per-electrode token blocks (+ P1 loss).
            "ragged_frontend": ragged_frontend,
            # 2026-06-09 ragged parcel drop (#92): default-ON (dispatch threads the
            # real value, overriding the conservative encoder-library default);
            # gather covered-&-visible parcels only in the P2 encoder (pad-to-max,
            # scatter back). Bit-identical to the dense key-masked path.
            "ragged_parcel": ragged_parcel,
            # 2026-06-09 P1 token drop (#93): default-ON (canonical V-JEPA-2
            # visible-only front-end); run the front-end token blocks over visible
            # (t_p, f_p) tokens only (pad-to-max, scatter). P4-transfer confirm
            # runs in parallel as a regression check, not a gate.
            "ragged_token": ragged_token,
            # 2026-06-09 freq-pos lock (A-JEPA): "sinusoidal" (default) =
            # fixed MAE sincos on the ordered freq-patch axis; "learned" =
            # R-freq-learned-embed sister. Threaded to the encoder + mirrored
            # to the P1 predictor freq id-tag by the joint module.
            "freq_pos": freq_pos,
            # B37 D1/D5 (2026-06-10): electrode->parcel pooling + the
            # freq-preserving parcel-SA latent depth. "cross_attn" (default,
            # B36 learned pool) rides depth_self_attn; "mean" (B37 hard mean
            # pool) feeds a thin latent_parcel_depth-block parcel-SA latent
            # (parcel x freq x time preserved) and is the path ssl_mode="joint"
            # requires. latent_parcel_depth is inert under cross_attn. Both ride
            # this one config -> every phase shares the encoder.
            "pool": pool,
            "latent_parcel_depth": latent_depth,
            # B37+ (2026-06-11): "parcel" (default, byte-identical) vs "joint"
            # parcel×time freq-batched latent; RGB-style mean|std stem channel
            # (std-channel zero-init → no-op at init). Inert under cross_attn.
            "latent_mode": latent_mode,
            "mean_pool_std": mean_pool_std,
            # NOTE: ``dkoleo_mode`` (B28) + ``phase_mode`` (B36) were
            # recorded-only model-config metadata threaded from the now-culled
            # ``--dkoleo-mode`` / ``--phase-mode`` CLI flags (2026-06-13). The
            # encoder keeps its own internal defaults ("off" / "joint_b29"), so
            # dispatch no longer threads them.
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
        log_every_n_steps=log_every_n_steps,
        lr_log_interval=lr_log_interval,
        wandb_config=wandb_config,
        max_steps=max_steps,
        val_check_interval=val_check_interval,
        ckpt_ladder_every=ckpt_ladder_every,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
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
    # Default 'full' (2026-06-07): a no-flag SSL run (P1/P2/P3) should train on
    # ALL clips, not the 3500/class Lite gate — Lite is a leaderboard-parity
    # budget that has no reason to throttle label-free pretraining. P4 is exempt:
    # _resolve_corpus_mode forces the P4 universe to 'lite' for any --mode except
    # 'nano' (never 'full'), so the eval stays Neuroprobe-faithful regardless.
    # 'nano' = tiny smoke (applies to SSL budget AND P4 universe).
    p.add_argument("--mode", choices=("nano", "lite", "full"), default="full")
    p.add_argument("--task", default=DEFAULT_TASK,
                   help="Neuroprobe task name (event field for the target).")
    p.add_argument("--eval-mode",
                   choices=("WithinSession", "CrossSession", "CrossSubject",
                            "AllCells"),
                   default=DEFAULT_EVAL_MODE,
                   help="Split policy (WithinSession = KFold within one trial, "
                        "CrossSession = submission gate, "
                        "CrossSubject = scientific generalization, "
                        "AllCells = materialization-only: every BT_LITE eval cell "
                        "unsplit, for the lite-eval raw baseline (piece 4)).")
    p.add_argument("--test-subject-id", type=int, default=DEFAULT_TEST_SUBJECT_ID)
    p.add_argument("--test-trial-id", type=int, default=DEFAULT_TEST_TRIAL_ID)
    p.add_argument("--fold-index", dest="fold_index", type=int, default=0,
                   help="WithinSession only: which KFold fold (0 or 1 for Lite) "
                        "this P4 eval cell scores. Ignored for CrossSession/"
                        "CrossSubject.")
    p.add_argument(
        "--pretrain-holdout-fraction", dest="pretrain_holdout_fraction",
        type=float, default=DEFAULT_PRETRAIN_HOLDOUT_FRACTION,
        help="Leakage decouple (#82): per-legal-session tail fraction held out "
        "for SSL pretext-loss monitoring (val AND test) on the SSL phases. Only "
        "partitions legal pretraining data (V14_PRETRAIN_SESSIONS); never the "
        "leaderboard eval split.",
    )
    p.add_argument("--binary-tasks", action="store_true", default=True,
                   help="(default) Binary label derivation per Neuroprobe leaderboard. "
                        "Pass --no-binary-tasks to switch to 3-class multiclass.")
    p.add_argument("--no-binary-tasks", dest="binary_tasks", action="store_false")
    p.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--m-sub-slots", type=int, default=DEFAULT_M_SUB_SLOTS)
    p.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                   help="DataLoader worker processes (B1.4e, 2026-05-29). "
                        "0 starves the GPU on the per-sample CPU STFT; default "
                        "15 overlaps it (production value, cpus_per_task=16). "
                        "Set --cpus-per-task >= num_workers + 1.")
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS,
                   help="Epoch budget for the SSL/distill phases (P1/P2/P3a/P3b) "
                        "and the P4 cap. The SSL phases train to this fixed "
                        "budget; no early-stop.")
    p.add_argument("--ssl-max-steps", dest="ssl_max_steps", type=int, default=1500,
                   help="Step budget for the SSL/distill phases (P1/P2/P3a/P3b). "
                        "Default 1500/phase = the validated production budget "
                        "(P1 converged clean in 1500; val L1 plateaued ~step 900, "
                        "so #79 will transfer-score a cut). Overrides --n-epochs "
                        "for those phases (max_epochs=-1): training stops at this "
                        "many OPTIMIZER steps. Pass <=0 to fall back to the "
                        "--n-epochs budget. P4 ignores this.")
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
    p.add_argument("--ckpt-ladder-every", dest="ckpt_ladder_every",
                   type=int, default=500,
                   help="Cadence (optimizer steps) of the keep-ALL ladder "
                        "checkpoint (save_top_k=-1). Default 500 (Ben 2026-06-11 "
                        "run-ops policy: keep every rung for dense probe curves + "
                        "<=cadence-steps lost on preemption). A long run can thin "
                        "it (e.g. 1000 → half the checkpoints / disk). The best + "
                        "metric-independent last.ckpt still ride "
                        "--ssl-val-check-interval, unaffected.")
    p.add_argument("--ssl-limit-val-batches", dest="ssl_limit_val_batches",
                   type=int, default=32,
                   help="Cap each SSL/distill validation to this many batches "
                        "(#66). The collapse-guard panel estimates RankMe/"
                        "coverage from a small sample, so an uncapped val set "
                        "(~875 batches, ~8 min/check) is pure wall-clock waste. "
                        "P4 is uncapped (its val_loss IS the downstream metric). "
                        "<=0 = uncapped.")
    p.add_argument("--ssl-limit-test-batches", dest="ssl_limit_test_batches",
                   type=int, default=None,
                   help="Cap the final SSL/distill trainer.test() pass to this "
                        "many batches. The end-of-phase test recomputes the "
                        "monitor panel (RankMe/coverage) over the full holdout "
                        "(~160 batches → ~2.5 min on a nano run, dwarfing the "
                        "training steps). A nano intuition run wants the live "
                        "training dynamics, not a full held-out eval — pass 2. "
                        "None (default) = uncapped (unchanged for real runs). "
                        "P4 is left uncapped (its test metric is load-bearing).")
    # Collapse guard OFF by default (Ben 2026-06-11: NEVER auto-kill a run; he
    # stops diverging runs himself). RankMe/loss are still logged module-side
    # for post-hoc checkpoint selection — only the auto-TERMINATE is disarmed.
    p.add_argument("--collapse-guard", dest="collapse_guard",
                   action="store_true", default=False,
                   help="OPT IN to the collapse/divergence kill-switch (#54) on "
                        "the SSL/distill phases. OFF BY DEFAULT — a run is never "
                        "terminated for you; RankMe/loss monitoring still logs. "
                        "Only arm this for a specific run you WANT auto-aborted.")
    p.add_argument("--rankme-warn-threshold", dest="rankme_warn_threshold",
                   type=float, default=None,
                   help="MON-TEACHER-FEATURE-RANK soft-warn band on normalised "
                        "RankMe (rankme/d) for the SSL phases (#74). None (default) "
                        "= the canonical 0.5. Must satisfy 0 < alarm < warn <= 1. "
                        "The measured BT floor is ~0.31, so 0.5 sits ABOVE it — "
                        "recalibration is Ben-gated; this flag is the sweep lever.")
    p.add_argument("--rankme-alarm-threshold", dest="rankme_alarm_threshold",
                   type=float, default=None,
                   help="MON-TEACHER-FEATURE-RANK hard-alarm threshold on "
                        "normalised RankMe (#74); below this the collapse-guard "
                        "KILLS the run. None (default) = the canonical 0.25. The "
                        "~0.31 BT floor sits just above 0.25, so the default is "
                        "tight — recalibration Ben-gated; this is the sweep lever.")
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
    p.add_argument("--lr", type=float, default=None,
                   help="Peak/base LR (constant value, or warmup-cosine peak). "
                        "DEFAULT = AUTO √-rule: lr = 5e-4·√(eff/1024), eff = "
                        "batch_size × accumulate_grad_batches × gpus_per_node "
                        "(e.g. eff-128 → 1.76e-4, eff-32 → 8.8e-5). The anchor "
                        "5e-4 @ 1024 is the §7 SSL center; the rule is the "
                        "validated production LR (live chain ran 1.76e-4 @ "
                        "eff-128). Pass an explicit value to override (M0 #45 "
                        "sweeps around it). The resolved LR is printed at launch.")
    p.add_argument("--lr-schedule", dest="lr_schedule",
                   choices=("constant", "warmup_cosine"), default="warmup_cosine",
                   help="warmup_cosine (DEFAULT 2026-06-04, §7 locked shape: "
                        "linear warmup → cosine → min_lr_ratio·peak; reads its "
                        "horizon from estimated_stepping_batches, --ssl-max-steps "
                        "pins it) or constant (the prior unimplemented-default "
                        "behavior — refused on a real SSL/distill run by the "
                        "main() launch guard).")
    p.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=150,
                   help="Linear-warmup optimizer steps for --lr-schedule "
                        "warmup_cosine. Default 150 = 10%% of the 1500-step "
                        "production budget (the live chain's value; §7 anchor is "
                        "20k P1 / 5k P2 @ full corpus — scale to the step "
                        "budget). Clamped below total_steps.")
    p.add_argument("--min-lr-ratio", dest="min_lr_ratio", type=float, default=0.0,
                   help="Cosine floor as a fraction of peak LR (0.0 = →0, the "
                        "§7 lock). Per-group-proportional, so P2/P3 "
                        "discriminative-LR ratios survive a non-zero floor.")
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=0.05,
                   help="AdamW weight decay. Default 2.0 = the canonical B37 joint "
                        "capstone run (ojok37j3; the fixed-data regime argues high "
                        "WD; M0 #45 still sweeps it). Only added to the optimizer kwargs when "
                        "> 0 (use --optimizer AdamW for decoupled WD). When > 0 the "
                        "no-WD param-group split (#40) exempts biases / LayerNorm "
                        "γβ / embeds — see --no-wd-exclude-norms. Pass 0.0 for the "
                        "plain-Adam falsifier.")
    p.add_argument("--wd-exclude-norms", dest="wd_exclude_norms",
                   action="store_true", default=True,
                   help="§7/B01 no-WD split (DEFAULT ON): exempt biases + "
                        "LayerNorm/RMSNorm γβ (ndim<=1, nanoGPT) + named embed/"
                        "identity/query tokens (timm/ViT/V-JEPA-2) from weight "
                        "decay. Inert when --weight-decay<=0 (the wd=0 path is "
                        "bit-identical).")
    p.add_argument("--no-wd-exclude-norms", dest="wd_exclude_norms",
                   action="store_false",
                   help="Flip the no-WD split OFF: decay ALL params uniformly "
                        "(the uniform-decay falsifier). Only changes behavior "
                        "when --weight-decay>0.")
    p.add_argument("--optimizer", dest="optimizer_name",
                   choices=("Adam", "AdamW"), default="AdamW",
                   help="AdamW (DEFAULT 2026-06-04, §7 locked family, decoupled "
                        "weight decay; pinned to 0.0 when --weight-decay is unset, "
                        "else the no-WD param-group split (#40) applies) or Adam "
                        "(the prior unimplemented-default — refused on a real "
                        "SSL/distill run by the main() launch guard).")
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
                   type=int, default=8,
                   help="Lightning accumulate_grad_batches. Default 8 = the "
                        "validated production value (bs=4 × accum=8 × 4-GPU = "
                        "eff-128). Sums grads over N micro-batches → effective "
                        "batch = batch_size × N × gpus_per_node at N× wall-clock "
                        "per optimizer step. The effective-batch lever for the "
                        "32 GB coganlab-gpu cards (EMA is gated once-per-optimizer-"
                        "step so accum is correct, #46). 8-GPU recipe holds eff-128 "
                        "at accum=4 (faster); pass 1 for no accumulation.")
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
    p.add_argument("--electrode-set", dest="electrode_set",
                   choices=("auto", "all", "lite"), default="auto",
                   help="Electrode subset for a SINGLE-phase build (the --chain "
                        "path always pins P4 to 'lite' and the SSL phases to "
                        "'all'). 'auto' (default) → 'lite' for --phase 4 (the "
                        "leaderboard eval cell, Neuroprobe-Lite electrode count) "
                        "else 'all' (full montage = BT-FULL pretraining). 'all'/"
                        "'lite' force the subset. 'lite' is BT-only.")
    p.add_argument("--frontend", dest="frontend",
                   choices=("raw", "2stft", "3stft"), default="raw",
                   help="Front-end family. 'raw' (default) = single F=50 raw |STFT| "
                        "grid (FE-RAW-1). '2stft' = dual-band 2STFT (Ben 2026-06-12): "
                        "two single-band STFTs (low N=512/hop256 4-56Hz 14 bins @ "
                        "8Hz → electrode_tokens; high N=128/hop64 64-192Hz 9 bins @ "
                        "16Hz → electrode_tokens_high) reconciled by the encoder's "
                        "two Conv2d band stems (per_band_stem). Requires --pool mean. "
                        "'3stft' = converged-arch triple-band 3STFT 2/2/2 (Ben "
                        "2026-06-17): three single-band STFTs → electrode_tokens_"
                        "slow/beta/hg, routing the run to the self-contained "
                        "V14ConvergedSSL (M2/M4 SSL) instead of V14ParcelPerceiver. "
                        "Requires the --converged-* shape flags.")
    # --frontend 3stft converged-arch shape (REQUIRED on that path; no run
    # defaults — the widths are named at launch). Inert on raw/2stft.
    p.add_argument("--converged-frontend-layers", dest="converged_frontend_layers",
                   type=int, default=None,
                   help="3stft: per-electrode frontend joint freq×time SA depth.")
    p.add_argument("--converged-latent-layers", dest="converged_latent_layers",
                   type=int, default=None,
                   help="3stft: flat token-level latent global-SA depth.")
    p.add_argument("--converged-m2-pred-dim", dest="converged_m2_pred_dim",
                   type=int, default=None,
                   help="3stft: M2 (own-electrode) predictor hidden dim.")
    p.add_argument("--converged-m2-pred-layers", dest="converged_m2_pred_layers",
                   type=int, default=None,
                   help="3stft: M2 predictor depth.")
    p.add_argument("--converged-m4-pred-dim", dest="converged_m4_pred_dim",
                   type=int, default=None,
                   help="3stft: M4 (parcel-query) predictor hidden dim.")
    p.add_argument("--converged-m4-pred-layers", dest="converged_m4_pred_layers",
                   type=int, default=None,
                   help="3stft: M4 predictor depth.")
    p.add_argument("--converged-lambda-m2", dest="converged_lambda_m2",
                   type=float, default=1.0,
                   help="3stft: M2 loss-term weight (neutral 1.0).")
    p.add_argument("--converged-lambda-m4", dest="converged_lambda_m4",
                   type=float, default=1.0,
                   help="3stft: M4 loss-term weight (neutral 1.0).")
    # --- converged MASK GEOMETRY (Ben 2026-06-18: tuned often) -------------
    # None → the LOCKED defaults (hg_start_rate 0.20 / hg_span 3 /
    # beta_start_rate 0.30 / beta_span 2 / parcel_mask_ratio 0.20). The §8.7
    # sister sweeps set these.
    p.add_argument("--converged-m2-hg-start-rate", dest="converged_m2_hg_start_rate",
                   type=float, default=None,
                   help="3stft: HG M2 coverage dial — fraction of HG time positions "
                        "that start a width-hg_span span (default 0.20).")
    p.add_argument("--converged-m2-hg-span", dest="converged_m2_hg_span",
                   type=int, default=None,
                   help="3stft: HG M2 span width in time patches (default 3).")
    p.add_argument("--converged-m2-beta-start-rate", dest="converged_m2_beta_start_rate",
                   type=float, default=None,
                   help="3stft: beta M2 coverage dial — fraction of beta time positions "
                        "that start a width-beta_span freq-tube span (default 0.30 ≈ 50%%).")
    p.add_argument("--converged-m2-beta-span", dest="converged_m2_beta_span",
                   type=int, default=None,
                   help="3stft: beta M2 freq-tube span width in time patches (default 2 "
                        "on the coarse 250 ms grid).")
    p.add_argument("--converged-m2-slow-freq-tubes",
                   dest="converged_m2_slow_freq_tubes", type=int, default=None,
                   help="3stft: # of slow freq-patches held out across ALL time "
                        "(freq-tube). 0 (default) ⇒ slow exempt; 1 of 3 ⇒ ⅓ of slow masked.")
    p.add_argument("--converged-m4-parcel-mask-ratio",
                   dest="converged_m4_parcel_mask_ratio", type=float, default=None,
                   help="3stft: M4 whole-parcel tube ratio (default 0.20).")
    # --- static-shape SSL (V-JEPA-2 throughput regime; steps A/B/C) --------
    p.add_argument("--converged-tube-ratio", dest="converged_tube_ratio",
                   type=float, default=None,
                   help="3stft: electrode-budget tight-pack tube ratio. Set ⇒ STATIC "
                        "masks (constant n_vis / N_mask per session). None ⇒ legacy "
                        "variable-count masks.")
    p.add_argument("--converged-tube-p-fixed", dest="converged_tube_p_fixed",
                   type=int, default=18,
                   help="3stft: M4 query-parcel pad width for the static tube "
                        "(measured BT max 17 at ratio 0.25 → 18).")
    p.add_argument("--converged-static-forward", dest="converged_static_forward",
                   action="store_true",
                   help="3stft: route the forward through the fixed-K / maskless / "
                        "sync-free static path (one compiled graph per session "
                        "geometry). REQUIRES --converged-tube-ratio AND "
                        "--group-by-session.")
    # --- converged M4 heteroscedastic down-weight (Ben 2026-06-18) ---------
    p.add_argument("--converged-m4-precision-off", dest="converged_m4_precision_off",
                   action="store_true",
                   help="3stft: DISABLE the M4 electrode-dof down-weight (the "
                        "R-precision-off sister). Default ON with α=1.0, n_ref=11.")
    p.add_argument("--converged-m4-precision-alpha", dest="converged_m4_precision_alpha",
                   type=float, default=None,
                   help="3stft: M4 down-weight exponent α (default 1.0; >1 = "
                        "risk-averse overlay on low-n parcels).")
    p.add_argument("--converged-m4-precision-n-ref", dest="converged_m4_precision_n_ref",
                   type=float, default=None,
                   help="3stft: M4 down-weight full-trust electrode count n_ref "
                        "(default 11; the w=1 saturation point).")
    p.add_argument("--cache-band", dest="cache_band",
                   choices=("slow", "beta", "hg"), default=None,
                   help="3STFT per-band cache build (--cache-only only). Build ONE "
                        "named band of the locked 2/2/2 ladder (slow N=1024/hop512 "
                        "2-12Hz Cartesian Re/Im; beta N=256/hop128 16-56Hz |STFT|; "
                        "hg N=128/hop64 64-192Hz |STFT|) into <spec-cache-dir>/band_"
                        "<name>. Rides the single-grid electrode_tokens slot, so no "
                        "encoder change is needed; the band view matches a future "
                        "--frontend 3stft run → that run HITs this cache. Overrides "
                        "--frontend.")
    p.add_argument("--atlas", dest="atlas", choices=("dk", "dkt"), default="dk",
                   help="Parcellation atlas. 'dk' (default) = Desikan-Killiany "
                        "(K=80); 'dkt' = Desikan-Killiany-Tourville (K=74, native "
                        "depth-wm.csv DKT column). Moves the depth-wm column AND "
                        "the parcel vocabulary together (anatomy.atlas_spec) and "
                        "sets the encoder k_parcels = len(vocabulary).")
    p.add_argument("--exclude-single-electrode-parcels",
                   dest="exclude_single_electrode_parcels", action="store_true",
                   help="Drop parcels covered by exactly ONE valid electrode (the "
                        "lone electrode is zeroed + marked invalid, parcel left "
                        "uncovered for that subject; global K unchanged). Avoids "
                        "degenerate within-parcel std poisoning the heteroscedastic "
                        "M4 precision weight. Applied to support + valid_mask alike.")
    p.add_argument("--cache-only", dest="cache_only", action="store_true",
                   help="Build the front-end spec cache for the (subset of the) "
                        "SSL corpus, then exit 0 BEFORE constructing the trainer "
                        "(no GPU, no training). Drives the SAME study.run() → "
                        "segmenter.apply() → dataset.prepare() path the real run "
                        "uses, so the materialized cache is byte-identical to what "
                        "training reads back. Pair with --cache-session-index for a "
                        "massively-parallel per-session SLURM array on a CPU node.")
    p.add_argument("--cache-session-index", dest="cache_session_index", type=int,
                   default=None,
                   help="With --cache-only: build ONLY the session at this index in "
                        "the resolved study corpus (_SESSIONS_BY_MODE[study_mode]; "
                        "pretrain has 13). One array task per index. None (default) "
                        "= the whole corpus in a single job. Subsets the study's "
                        "timeline list only — never changes the spec-cache key.")
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
    p.add_argument("--in-allocation-ddp", action="store_true",
                   help="The process is ALREADY one of N srun ranks (warm "
                        "nano_worker_ddp). Run the experiment in-process (exca "
                        "local, no submitit) with DDP across the live srun "
                        "ranks. Forces ddp_find_unused; pass NO --cluster / "
                        "--gpus-per-node / --tasks-per-node with it.")
    p.add_argument("--slurm-requeue", action="store_true",
                   help="C5: emit '#SBATCH --requeue' so a preempted job is "
                        "auto-resubmitted; the within-phase last.ckpt resume "
                        "continues from the last checkpoint. Set for the long "
                        "full chain; leave off for smokes.")
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
    p.add_argument("--spec-cache-dir", default=None,
                   help="Whole-movie raw-|STFT| feature cache root (#80). "
                        "Default {extractor-cache-folder}/v14_spec_cache when an "
                        "extractor cache root is set, so the ~9-min-per-run "
                        "session_robust_z whole-movie STFT over the 13-session "
                        "pretrain corpus is paid ONCE and every later same-front-"
                        "end-config run memmap-slices it. Location-only knob "
                        "(excluded from the cache uid; features byte-identical to "
                        "the recompute path). Pass a path to override the location.")
    p.add_argument("--no-spec-cache", dest="no_spec_cache", action="store_true",
                   help="Disable the #80 whole-movie |STFT| cache (forces the "
                        "per-run recompute of the session_robust_z whole-movie "
                        "STFT). Default OFF (cache armed when a cache root exists).")
    p.add_argument("--spec-only", dest="spec_only", action="store_true",
                   help="Lightweight-redeploy (Option B): serve clips from the "
                        "on-disk spec cache ALONE — no extractor cache, no raw h5 on "
                        "the target. prepare() builds the index/channels/robust-z "
                        "stats from the .npy/.json/.stats.npz sidecars and skips the "
                        "extractor/raw layer. Requires --spec-cache-dir + LOF off; "
                        "model inputs byte-identical to the extractor-resident path. "
                        "Fails loud if any session's spec/stats sidecar is missing. "
                        "Default OFF.")
    p.add_argument("--trial-durations", dest="trial_durations", default=None,
                   help="Path to a JSON {\"<subject_id>_<trial_id>\": native_n_samples} "
                        "trial-duration override (scripts/neuroprobe/dump_trial_"
                        "durations.py). When set, the study reads each trial's duration "
                        "from this map instead of opening raw h5 — the only h5 touch in "
                        "timeline building — so --spec-only runs h5-free (DeltaAI). "
                        "Duration is uid-invariant ⇒ byte-identical caches/clips to the "
                        "h5 path. Default None = read duration from h5.")
    # Layer-2 bad-electrode clip filter (#180). SSL-only: clips overlapping a
    # precomputed glitch span are dropped before sampling. P4 eval untouched
    # (build gates this to the SSL phases). Default None = no filtering.
    p.add_argument("--bad-window-dir", default=None,
                   help="Directory of per-session bad-time-window sidecars "
                        "(scripts/neuroprobe/precompute_bad_windows.py). When set, "
                        "SSL clips whose neural window overlaps a glitch span are "
                        "dropped before sampling (Layer-2 of the bad-electrode "
                        "defense, #180). Pretrain/distill only — the P4 eval "
                        "datamodule never filters, so Neuroprobe parity is intact. "
                        "Removes events (rows), never electrodes. Default None = off.")
    p.add_argument("--group-by-session", action="store_true",
                   help="Throughput lever (#248, science-neutral): draw "
                        "session-homogeneous TRAIN batches so the ragged forward pays "
                        "each batch's true electrode count C instead of padding a "
                        "mixed batch up to the corpus max-C (the latent attention is "
                        "O(C^2) in electrode count). SSL phases only; P4 eval batches "
                        "are never grouped (Neuroprobe parity lock). Default off.")
    p.add_argument("--balance-ranks", action="store_true",
                   help="Throughput lever (#249, science-neutral; needs "
                        "--group-by-session): bucket sessions by electrode count C so "
                        "the W batches the W DDP ranks run simultaneously at each "
                        "gradient all-reduce have matched C — the small-C ranks stop "
                        "idling at the barrier while a big-C rank finishes its O(C^2) "
                        "forward (removes DDP straggler idle). Cost proxy = the exact "
                        "per-session post-static-drop electrode count; no-ops to the "
                        "plain stride if BraintreeBank data is absent. Default off.")
    p.add_argument("--same-session-ranks", action="store_true",
                   help="Throughput lever (raw-GPU-util bubble removal; needs "
                        "--group-by-session; stronger than --balance-ranks): all W "
                        "DDP ranks run the SAME session (different clips) at each "
                        "micro-step, so their electrode count C is identical and no "
                        "rank idles at the all-reduce barrier (flat raw GPU util) — "
                        "vs --balance-ranks which only cost-matches W distinct "
                        "sessions and still gates on the max-C rank. Per-step session "
                        "diversity drops to 1; grad-accum restores cross-session "
                        "mixing (each accum micro-step is a different session). "
                        "Science-neutral; needs no cost proxy; takes precedence over "
                        "--balance-ranks. SSL phases only. Default off.")
    # Layer-3 winsor cap (#180). Read-time per-cell |z| clamp on the session
    # robust-z front-end. Cache-NEUTRAL by design: implemented as the env knob
    # V14_SESSION_Z_WINSOR (NOT a serialized view field) so it never forks the
    # multi-TB spec cache. This flag SETS that env var in main(); default None
    # leaves the env untouched (no clamp).
    p.add_argument("--session-z-winsor", type=float, default=None,
                   help="Layer-3 winsor cap (#180): clamp the session robust-z "
                        "front-end to +/- this |z| per cell at read time (e.g. "
                        "2500). Caps spectral transients on flaky contacts that "
                        "survive LOF. Cache-neutral — sets V14_SESSION_Z_WINSOR, "
                        "not a cached view field, so it never re-forks the spec "
                        "cache. Default None = no clamp. The 3STFT path can OVERRIDE "
                        "per band with the flags below; this stays the fallback.")
    # Per-band winsor caps (#230): the slow/beta/HG bands have different |z|
    # distributions, so one scalar cap mis-clamps two of them. Each sets a
    # band-specific V14_SESSION_Z_WINSOR_<BAND> env knob that wins over the global
    # scalar for that band only — same cache-neutral env mechanism. Preliminary
    # caps are tuned per-band on the rebuilt 3STFT cache (Ben locks finals);
    # default None = fall back to --session-z-winsor for that band.
    for _b in ("slow", "beta", "hg"):
        p.add_argument(f"--session-z-winsor-{_b}", type=float, default=None,
                       help=f"Per-band winsor cap for the {_b} 3STFT band "
                            f"(sets V14_SESSION_Z_WINSOR_{_b.upper()}); overrides "
                            f"--session-z-winsor for {_b} only. Default None.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print resolved config without dispatching.")
    # Offline probe bench (pieces 1+3, Ben 2026-06-20): build the run exactly as a
    # real dispatch would, then — instead of training — load a checkpoint at 1 s
    # geometry and run the ridge-timing + logistic head-to-head bench
    # (offline_probe_bench.run_probe_bench). Read-only; no trainer, single GPU.
    p.add_argument("--probe-bench-ckpt", type=str, default=None,
                   help="Path to a converged-SSL checkpoint (.ckpt). When set, build "
                        "the experiment then run the offline probe bench at 1 s "
                        "geometry instead of training. Pair with --probe-bench-out.")
    p.add_argument("--probe-bench-out", type=str, default=None,
                   help="JSON path for the offline probe bench results "
                        "(default: <ckpt-dir>/probe_bench.json).")
    p.add_argument("--probe-bench-max-iter", type=int, default=2000,
                   help="lbfgs max_iter for the head-to-head logistic fits "
                        "(bounds the p>>n per-electrode d=256 fits; same cap per "
                        "tap so raw<->encoder stay comparable). Default 2000.")
    p.add_argument("--probe-bench-pieces", type=str, default="ridge,headtohead",
                   choices=["ridge,headtohead", "ridge", "headtohead"],
                   help="Which probe-bench pieces to run. 'ridge' = piece 1 only; "
                        "'headtohead' = piece 3 only; both (default) runs piece 1 then "
                        "piece 3. Use 'headtohead' to re-run only the logistic when "
                        "piece 1's ridge timing already landed.")
    p.add_argument("--probe-bench-taps", type=str, default="raw,frontend,latent",
                   help="Comma-separated subset of raw/frontend/latent to score in "
                        "the head-to-head (piece 3). Default all three. 'raw,frontend' "
                        "skips the expensive latent forward; 'raw' never touches the "
                        "model.")
    # Gradient-noise-scale → critical-batch diagnostic (gns_critical_batch.
    # run_gns_probe). Builds the real converged module + group_by_session loader,
    # runs accum'd single-session micro-batch grads, fits B_crit. No trainer, 1 GPU.
    p.add_argument("--gns-probe", action="store_true",
                   help="Measure the gradient noise scale / critical batch size of "
                        "the converged SSL step instead of training, then exit. Build "
                        "the experiment exactly as a real run (same arch/cache/sampler); "
                        "optionally warm-start with --gns-ckpt. Pair with --frontend 3stft "
                        "+ the converged_* shape flags + --group-by-session.")
    p.add_argument("--gns-ckpt", type=str, default=None,
                   help="Optional Lightning ckpt to warm-start the GNS probe (restores "
                        "student AND EMA teacher). None = measure from init.")
    p.add_argument("--gns-out", type=str, default=None,
                   help="JSON output path for --gns-probe (default: ./gns_critical_batch.json).")
    p.add_argument("--gns-rounds", type=int, default=64,
                   help="Number of accumulation rounds to average E‖g_B‖² over "
                        "(more = tighter B_crit; the intercept is noise-sensitive).")
    p.add_argument("--gns-accum", type=int, default=8,
                   help="Micro-batches accumulated per round; the batch-size axis is "
                        "b, 2b, ..., gns_accum*b (each a DIFFERENT session under "
                        "group_by_session, so it spans the cross-session variance).")
    p.add_argument("--lite-baseline-out", type=str, default=None,
                   help="JSON path for the raw |STFT| logistic baseline on the "
                        "Neuroprobe-Lite EVAL cells (piece 4). When set, build the "
                        "(lite) experiment then run the three eval-mode baselines "
                        "instead of training — no GPU. Pair with --mode lite "
                        "--electrode-set lite and a fresh --spec-cache-dir.")
    p.add_argument("--lite-baseline-max-iter", type=int, default=10000,
                   help="lbfgs max_iter for the lite-baseline logistic fits "
                        "(default 10000, the upstream-parity recipe).")
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Lightning fast-dev-run: 1 batch train+val+test, no checkpoints.")
    # --live nano learning-dynamics dashboard (Weights & Biases). Near-free
    # graphs only: train/val loss, RankMe, LR-schedule curve — no per-group
    # grad-norm / teacher-student gap / heatmap yet (those are the deferred
    # phase-2 of reports/nano_dynamics_dashboard_handoff_2026_06_07.md).
    p.add_argument("--live", action="store_true",
                   help="Stream live training metrics to Weights & Biases for the "
                        "nano learning-dynamics dashboard: per-step "
                        "LearningRateMonitor + log_every_n_steps=1 so the loss / "
                        "RankMe / LR curves update live and overlay across runs. "
                        "Needs `wandb` + a wandb login (or --wandb-offline). Pair "
                        "with --mode nano --ssl-max-steps ~150-300 for a ~30-60 s "
                        "loop. See reports/nano_dynamics_dashboard_handoff_2026_06_07.md.")
    p.add_argument("--wandb-project", dest="wandb_project",
                   default="v14-nano-dynamics",
                   help="W&B project for --live runs (default v14-nano-dynamics).")
    p.add_argument("--wandb-group", dest="wandb_group", default=None,
                   help="W&B group for --live runs (default <task>_<mode>).")
    p.add_argument("--wandb-run-name", dest="wandb_run_name", default=None,
                   help="W&B run name / overlay legend label for --live runs "
                        "(default <task>_<mode>_p<phase>). Set per predict-then-"
                        "check rung so the overlay legend reads clearly, e.g. "
                        "--wandb-run-name p1-lr1e-4.")
    p.add_argument("--wandb-run-id", dest="wandb_run_id", default=None,
                   help="Resume an EXISTING W&B run by id (e.g. pf5e66wp) "
                        "instead of minting a new one each launch. Threads into "
                        "WandbLoggerConfig.id (resume='allow'), so an auto/"
                        "--resume continuation appends to the ORIGINAL run line "
                        "rather than starting a fresh, disconnected one. "
                        "Excluded from the exca cell UID (wandb config is not "
                        "hashed), so it does NOT fork the cache cell.")
    p.add_argument("--wandb-offline", dest="wandb_offline", action="store_true",
                   help="Log --live runs offline (no wandb login/network); sync "
                        "later with `wandb sync`.")
    # NOTE: ``--shaft-mask-k-override`` / ``--shaft-mask-extent-blocks`` (shaft
    # mask) and ``--dkoleo-mode`` (B28-demoted) were culled 2026-06-13 — the
    # shaft mask is dropped (no downstream consumers) and DKoleo defaults OFF
    # via the encoder's own internal default.
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
        action="store_true", default=True,
        help="Enable activation checkpointing on the encoder token-block and "
             "latent-block stacks. ON by default (2026-06-07): it is what lets "
             "bs=4 fit the 32 GB coganlab-gpu cards (the live chain ran with it). "
             "Gated on training + grad so the no_grad EMA-teacher pass is never "
             "checkpointed. Bit-identical loss + grads. Pass "
             "--no-gradient-checkpointing on a 48 GB+ card to trade the recompute "
             "for speed.",
    )
    p.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing",
        action="store_false",
        help="Disable activation checkpointing (faster, more memory) — only safe "
             "on a card with headroom for full activations at the chosen "
             "--batch-size.",
    )
    # 2026-06-08 ragged front-end (#91): DEFAULT-ON (flipped 2026-06-12 — Ben:
    # "all ragged optimizations default on"). When ON, the encoder's per-row
    # token blocks run only over valid rows (cross_attn: valid electrodes;
    # mean-pool: covered parcels) — pad/uncovered rows gathered out, scattered
    # back as zeros before the pool — and the P1 M2 loss drops pad rows. On the
    # mean path with masking active the uncovered-parcel drop is folded into
    # ragged_token's visible-token gather (one combined gather).
    p.add_argument(
        "--ragged-frontend", "--no-ragged-frontend", dest="ragged_frontend",
        action=argparse.BooleanOptionalAction, default=True,
        help="Skip pad/uncovered rows in the per-row token blocks + P1 loss "
             "(needs valid_mask in the batch — i.e. a padded c_max). "
             "Valid-row M2/M4 and the P2 loss stay bit-identical; only the "
             "P1 loss mean changes (it no longer dilutes on zero-input pad "
             "rows). Cuts the token-block + predictor FFN by the pad "
             "fraction (~50%% at BT-Lite c_max=256) — unblocks raw bs=8 and "
             "~halves front-end step time on padded batches. ON by default; "
             "--no-ragged-frontend restores the dense path.",
    )
    # 2026-06-09 ragged parcel drop (#92): ON by default (dense path
    # bit-identical up to ~1e-6 matmul reassociation, so safe). When ON, the P2
    # parcel encoder gathers covered-&-visible parcels only (pad-to-max), runs SA
    # over them, and scatters back as zeros — cutting the parcel-SA + predictor
    # FFN by the dropped fraction. Independent of --ragged-frontend.
    p.add_argument(
        "--ragged-parcel", "--no-ragged-parcel", dest="ragged_parcel",
        action=argparse.BooleanOptionalAction, default=True,
        help="Gather covered-&-visible parcels only in the P2 encoder "
             "(pad-to-max per sample, scatter back as zeros). The dense path "
             "already key-masks uncovered/masked parcels out of attention, so "
             "the kept-parcel M4 is loss-neutral (bit-identical up to ~1e-6 "
             "matmul reassociation); the win is cutting the parcel-SA + "
             "predictor FFN by the dropped fraction. ON by default; "
             "--no-ragged-parcel restores the dense path.",
    )
    # 2026-06-09 P1 token drop (#93): ON by default. The canonical V-JEPA 2
    # visible-only front-end — drops masked (t_p, f_p) tokens from the token
    # blocks (vs zeroing-but-still-attending), so the encoder is truly
    # visible-only (V-JEPA 2 §2.1). The zero-and-attend dense path was the
    # deviation, not the spec, so #93 is the default. A REPRESENTATION CHANGE
    # (not loss-neutral vs the legacy dense path), so the P4 transfer ablation
    # runs in parallel as a CONFIRM (canonical ≥ legacy), NOT a gate. Dominant P1
    # throughput lever: ~3× per-step compute (warm 4-GPU matrix 2026-06-09).
    p.add_argument(
        "--ragged-token", "--no-ragged-token", dest="ragged_token",
        action=argparse.BooleanOptionalAction, default=True,
        help="Run the front-end token blocks over VISIBLE (t_p, f_p) tokens only "
             "(pad-to-max per electrode row, per-row time-RoPE, scatter back as "
             "zeros) — the canonical V-JEPA 2 visible-only encoder. REPRESENTATION "
             "CHANGE vs the legacy zero-and-attend path (bit-identical to "
             "key-masking masked tokens out). ON by default; --no-ragged-token "
             "restores the legacy dense path. Dominant P1 throughput lever (~3×).",
    )
    # 2026-06-09 predictor context+query drop (#94): ON by default (dense path
    # bit-identical up to ~1e-6 matmul reassociation, so safe). When ON, the JEPA
    # predictor consumes ONLY the visible context + ONLY the real masked query
    # (pad-to-max), instead of feeding the full grid and key-padding the rest.
    # Joint-only (P1 + P2). Independent of the encoder ragged flags.
    p.add_argument(
        "--ragged-predictor", "--no-ragged-predictor", dest="ragged_predictor",
        action=argparse.BooleanOptionalAction, default=True,
        help="Feed the JEPA predictor ONLY the visible context cells + ONLY the "
             "real masked query slots (pad-to-max per sample), instead of the "
             "full grid with the rest key-padded. The padded slots are masked "
             "out of attention either way (exp(NEG_INF)=0), so the prediction is "
             "bit-identical (up to ~1e-6 matmul reassociation); the win is a pure "
             "FLOP cut on the predictor attention/FFN at high mask ratio. Covers "
             "BOTH P1 and P2. ON by default; --no-ragged-predictor restores the "
             "dense path.",
    )
    # 2026-06-09 freq-pos lock (A-JEPA): positional encoding for the ordered
    # freq-patch axis (encoder front-end + P1 predictor freq id-tag). Default
    # "sinusoidal" (fixed MAE sincos); "learned" = R-freq-learned-embed sister.
    p.add_argument(
        "--freq-pos", dest="freq_pos",
        choices=["sinusoidal", "learned"], default="sinusoidal",
        help="Positional encoding for the ORDERED freq-patch axis. 'sinusoidal' "
             "(default) = fixed A-JEPA/MAE sincos (absolute code + smoothness/"
             "ordering prior), applied to the encoder front-end AND the P1 "
             "predictor freq id-tag. 'learned' = R-freq-learned-embed sister "
             "(the pre-lock learned table). Time stays RoPE; parcels stay "
             "learned. REPRESENTATION CHANGE (joint-phase only).",
    )
    # 2026-06-07 speedup-fanout C1: torch.compile the student/teacher encoder
    # forward. The toggle is the ``V14_COMPILE`` env var read inside
    # V14JointBrainModule.__init__ (an env var, not a pydantic field, so it never
    # forks the exca run uid — a compiled and an uncompiled run share a cache
    # namespace). This flag is the discoverable front-door: it sets that env var
    # in main() before exca submits, so it propagates to the slurm job (submitit
    # captures the driver env). The compile wraps only the forward callable in a
    # plain dict → no nn.Module re-registration, no ``_orig_mod.`` state_dict
    # prefix → EMA name-match intact.
    #
    # 2026-06-09: DEFAULT FLIPPED ON. Static torch.compile is loss-neutral
    # (P1 ±5% tripwire PASS: max 2.56%, mean 0.81% over 50 steps — compile only
    # reassociates fp ops) and net-POSITIVE on the long production passes
    # (P1 ~1500 steps + the full-encoder P2): it cuts steady per-step ~6× the
    # eager kernels (1.80→0.30 s/step dense, warm 4-GPU matrix 2026-06-09), and
    # the ~105s/job compile re-trace tax amortizes over a long run. Paired with
    # --compile-dynamic (now also default-ON): with ragged ON the per-batch shape
    # varies, and the matrix showed dynamic==static warm (no per-step penalty)
    # while dynamic eliminates the static recompile storm — so dynamic is the
    # better default here (see --compile-dynamic).
    # ESCAPE: ``--no-compile`` for runs where the cold-start does NOT amortize —
    # the M0 sweep's short BT-lite cells (#45), smoke/debug runs, and anything
    # under a few-hundred steps. P3/P4 are separate modules and ignore this flag.
    # Still GPU-validation-gated for the full P2 cold/gc-on path — smoke one cell
    # before a full relaunch (P1 warm-worker is the only path measured here).
    p.add_argument(
        "--compile", "--no-compile", dest="compile_encoder",
        action=argparse.BooleanOptionalAction, default=True,
        help="torch.compile the student/teacher encoder forward (sets "
             "V14_COMPILE). ON by default (static; ~35%% per-step cut on long "
             "P1/P2 runs, loss-neutral). Pass --no-compile for short runs "
             "(M0 sweep cells, smoke/debug) where cold-start won't amortize.",
    )
    p.add_argument(
        "--compile-mode", dest="compile_mode", default=None,
        choices=["default", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (sets V14_COMPILE_MODE). Default inductor mode "
             "if unset. 'max-autotune-no-cudagraphs' adds Triton/CUTLASS GEMM "
             "autotuning (same math, faster kernels) WITHOUT cudagraph capture. "
             "The cudagraph modes ('reduce-overhead', plain 'max-autotune') are "
             "deliberately NOT offered: DDP AccumulateGrad cross-stream comm + "
             "find_unused_parameters + the M2/M4 data-dependent early-returns break "
             "graph capture (shape-independent), so they would crash or corrupt "
             "grads here.",
    )
    p.add_argument(
        "--fast-backends", "--no-fast-backends", dest="fast_backends",
        action=argparse.BooleanOptionalAction, default=False,
        help="Enable the loss-neutral matmul/conv fast paths (sets "
             "V14_FAST_BACKENDS): TF32 for any residual fp32 matmul "
             "(set_float32_matmul_precision('high') + allow_tf32 on cuda+cudnn) and "
             "cudnn.benchmark autotuning for the static-shape conv stems. Off by "
             "default (bit-exact); on for throughput runs.",
    )
    # 2026-06-09: DEFAULT FLIPPED ON. The warm 4-GPU matrix measured dynamic ==
    # static warm (150-step submit→done 121s == 121s) — i.e. NO per-step penalty,
    # contradicting the earlier "marginally slower" worry. With ragged ON the
    # per-batch pad-to-max electrode count varies, so STATIC recompiles for every
    # distinct shape — a recompile storm that recurs across a long run as new
    # max-counts appear (and a 301s vs 121s cold storm on the first ragged job).
    # dynamic=True compiles ONCE with symbolic shapes and absorbs all of it at no
    # warm cost ⇒ strictly better here. ESCAPE: ``--no-compile-dynamic`` for
    # fixed-shape (non-ragged) runs where static specialization is marginally
    # tighter. No effect when --no-compile.
    p.add_argument(
        "--compile-dynamic", "--no-compile-dynamic", dest="compile_dynamic",
        action=argparse.BooleanOptionalAction, default=None,
        help="Compile ONCE with symbolic shapes (sets V14_COMPILE_DYNAMIC=1). "
             "With ragged ON the per-batch electrode count varies, so static "
             "compile recompiles per distinct shape (storm); dynamic absorbs the "
             "varying dim into one graph at no measured warm per-step cost "
             "(matrix: dynamic==static warm). DEFAULT IS AUTO (unset): dynamic ON "
             "for the legacy ragged forward, OFF for --converged-static-forward "
             "(its drop-not-pad kernel wants one static graph per session "
             "geometry, not a symbolic one). An explicit --compile-dynamic / "
             "--no-compile-dynamic overrides the auto-resolution. No effect when "
             "--no-compile. Loss-neutral (±5%% tripwire is the backstop).",
    )
    p.add_argument(
        "--sdpa-backend", dest="sdpa_backend",
        choices=["default", "cudnn", "cudnn_latent", "cudnn_m4", "flash", "efficient", "math"],
        default="default",
        help="Science-neutral SDPA kernel preference (sets V14_SDPA_BACKEND, read "
             "in V14ConvergedBrainModule). The masked latent/cross attention falls "
             "to the mem-efficient CUTLASS sm80 kernel on Hopper (profiled ~73%% of "
             "GPU time); 'cudnn' forces the Hopper-native mask-capable sm90 cuDNN "
             "kernel around the WHOLE forward; 'cudnn_latent' scopes that force to "
             "just the large-L cross-electrode latent blocks (where the 2.7x GPU win "
             "is, sparing small-L calls cuDNN's ~19ms/call host plan-building); "
             "'cudnn_m4' scopes it to just the M4 predictor block-loop (the A/B/C "
             "isolation control vs 'cudnn_latent'). "
             "Identical math (±5%% tripwire is the backstop); 'default' ⇒ stock "
             "selection, byte-identical, no cache blast (env, not a uid field).",
    )
    # 2026-06-09 throughput levers — front-doors for env vars read in data.py /
    # experiment.py (env, not pydantic fields → never fork the exca uid). All
    # loss-neutral: they touch startup/overhead, never the model/optimizer/data.
    p.add_argument(
        "--warm-dataset-cache", dest="warm_dataset_cache",
        action="store_true", default=False,
        help="In-process cache of the built split-DataLoaders (sets "
             "V14_WARM_DATASET_CACHE=1). A warm worker then skips dataset "
             "re-materialization on repeat runs of the same data config. "
             "Loss/FLOPs-neutral — loaders still draw a fresh batch each step.",
    )
    p.add_argument(
        "--no-sanity-val", dest="no_sanity_val",
        action="store_true", default=False,
        help="Skip Lightning's sanity-validation pass (sets V14_NO_SANITY_VAL=1). "
             "Startup-overhead lever; no training effect.",
    )
    p.add_argument(
        "--no-trainer-ckpt", dest="no_trainer_ckpt",
        action="store_true", default=False,
        help="Disable checkpoint writing (sets V14_NO_TRAINER_CKPT=1) for a "
             "throw-away throughput run.",
    )
    p.add_argument(
        "--no-test", dest="no_test",
        action="store_true", default=False,
        help="Skip the post-fit trainer.test() eval pass (sets V14_NO_TEST=1). "
             "test() is a diagnostic (test_loss + monitors), NOT part of training "
             "or the training loss curve, yet costs a fixed ~30 s DDP sampler "
             "setup every run. Throughput lever; the trained model + metrics.csv "
             "loss curve are unaffected.",
    )
    p.add_argument(
        "--ddp-static-graph", dest="ddp_static_graph",
        action="store_true", default=False,
        help="Despite the legacy name, this sets find_unused_parameters=False "
             "ONLY — it does NOT set torch static_graph (that was retired to a "
             "no-op: the ragged front-end varies the autograd graph step-to-step, "
             "tripping expect_autograd_hooks_). Sets V14_DDP_STATIC_GRAPH=1. Valid "
             "when the unused-param set is empty/static across steps (P1: "
             "pool/encoder/encoder_ln/parcel_embed never participate); DDP then "
             "skips the per-backward graph traversal. FLOPs/numerics-neutral; loss "
             "tripwire is the guard.",
    )
    p.add_argument(
        "--p1-freeze-parcel", dest="p1_freeze_parcel",
        action="store_true", default=False,
        help="P1-only throughput lever (sets V14_P1_FREEZE_PARCEL=1). Freezes "
             "the parcel side (pool/encoder/encoder_ln/parcel_embed), which is "
             "ALREADY excluded from the P1 optimizer and never in the m2_only "
             "forward, so it is bit-neutral for the trained P1 model. Removes "
             "those params from DDP's grad set so --ddp-static-graph no longer "
             "hits the expect_autograd_hooks_ reducer assert and "
             "find_unused_parameters can be False. Loss tripwire is the guard.",
    )
    p.add_argument(
        "--log-every-n-steps", dest="log_every_n_override", type=int, default=None,
        help="Override Trainer log_every_n_steps via V14_LOG_EVERY_N (env, "
             "uid-transparent). Use 1 to capture the per-step loss curve in "
             "metrics.csv without --live/wandb.",
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
        "--parcel-lr-scale", dest="parcel_lr_scale", type=float,
        default=1.0 / 3.0,
        help="B33 §5 P3-3b parcel-side discriminative-LR scale (--phase 3 "
             "p3-stage 3b only). 1/3 (default) = base/3 lock; the front-end "
             "rides its internal frontend_lr_scale and the connector trains "
             "at base. No effect under 3a / P1 / P2 / P4 (persisted onto the "
             "run record either way). Folded into the M0 optimizer sweep "
             "(#45/#78).",
    )
    # ----- B37 (2026-06-10) encoder + joint-SSL flags -------------------------
    p.add_argument(
        "--pool", dest="pool", choices=("cross_attn", "mean"),
        default=DEFAULT_POOL,
        help="B37 D1 electrode->parcel pooling. 'cross_attn' (default) = B36 "
             "learned per-parcel cross-attn pool. 'mean' = B37 hard mean pool "
             "feeding the freq-preserving thin parcel-SA latent; it DEFAULTS "
             "--ssl-mode to 'joint' (B37 D7) and is required by joint SSL.",
    )
    p.add_argument(
        "--latent-depth", dest="latent_depth", type=int,
        default=DEFAULT_LATENT_DEPTH,
        help="B37 D5 parcel-SA latent block count for the mean-pool path "
             "(default 6 = the canonical B37 joint run; was 2 thin). Inert "
             "under --pool cross_attn (that path rides --depth).",
    )
    p.add_argument(
        "--latent-mode", dest="latent_mode", choices=("parcel", "joint"),
        default=DEFAULT_LATENT_MODE,
        help="B37+ mean-path latent cross-parcel mode. 'parcel' = thin "
             "parcel-SA-only latent (freq+time batched). 'joint' (default, "
             "canonical B37 run) = JOINT parcel×time, freq batched — attends the K_c·T_p "
             "parcel×time tokens with RoPE-on-time + a learned global-parcel-id "
             "tag (cross-subject), filling the cross-region temporal gap. Inert "
             "under --pool cross_attn.",
    )
    p.add_argument(
        "--mean-pool-std", dest="mean_pool_std", action="store_true",
        default=DEFAULT_MEAN_POOL_STD,
        help="B37+ feed the masked per-parcel STD as a 2nd stem input channel "
             "(RGB-style mean|std), computed in the SAME masked reduction as the "
             "mean (no electrode-row desync). Conv std-channel is zero-init → "
             "no-op at init, learns from there. Inert under --pool cross_attn. "
             "DEFAULT ON (canonical B37 run); pass --no-mean-pool-std to disable.",
    )
    p.add_argument(
        "--no-mean-pool-std", dest="mean_pool_std", action="store_false",
        help="Disable the RGB-style mean|std 2nd stem channel (mean-only pool).",
    )
    p.add_argument(
        "--ssl-mode", dest="ssl_mode", choices=("auto", "joint"),
        default="auto",
        help="B37 D7 SSL objective (joint SSL / --phase 1). 'auto' (default) "
             "resolves to 'joint'. 'joint' = B37 composite-mask M2+M4 single "
             "forward (requires --pool mean). The B36 'staged' single-term "
             "per-phase CLI surface was culled 2026-06-13.",
    )
    p.add_argument(
        "--m4-loss-weight", dest="lambda_m4", type=float,
        default=DEFAULT_LAMBDA_M4,
        help="B37 D7 joint M4 term weight lambda in L = L_M2 + lambda*L_M4 "
             "(default 1.0). Joint-only.",
    )
    p.add_argument(
        "--m2-predictor-depth", dest="m2_predictor_depth", type=int,
        default=None,
        help="B37 D9 M2 (front-end) predictor depth. DEFAULT = AUTO = "
             "max(1, --depth // 2) (the half-rule: predictor tracks its "
             "front-end encoder; 3 @ canonical depth=6). Pass an int to pin. "
             "Joint-only; shares --predictor-hidden / --predictor-n-heads with "
             "the M4 predictor.",
    )
    p.add_argument(
        "--m4-predictor-depth", dest="m4_predictor_depth", type=int,
        default=None,
        help="B37 M4 (parcel) predictor depth. DEFAULT = AUTO = "
             "max(1, --latent-depth // 2) (the half-rule: predictor tracks its "
             "parcel-latent encoder; 3 @ canonical latent_depth=6). Pass an int "
             "to pin. Joint-only.",
    )
    p.add_argument(
        "--joint-frontend-lr-scale", dest="joint_frontend_lr_scale", type=float,
        default=DEFAULT_JOINT_FRONTEND_LR_SCALE,
        help="B37 D8 joint front-end discriminative-LR scale (default 1.0 = no "
             "discrimination; 0.0 freezes the front-end). Joint-only — distinct "
             "from the staged internal frontend_lr_scale.",
    )
    p.add_argument(
        "--joint-parcel-lr-scale", dest="joint_parcel_lr_scale", type=float,
        default=DEFAULT_JOINT_PARCEL_LR_SCALE,
        help="B37 D8 joint parcel-side (+ both predictors) discriminative-LR "
             "scale (default 1.0 = no discrimination; 0.0 = zero-LR, no update "
             "but optimizer state retained — unlike --joint-frontend-lr-scale 0.0 "
             "which hard-freezes the front-end). Joint-only — distinct from the "
             "B33 P3-3b --parcel-lr-scale.",
    )
    p.add_argument(
        "--m2-mask-ratio", dest="m2_mask_ratio", type=float,
        default=None,
        help="#75 M2 front-end held-out band ratio. Unset → 0.50 (the 6/03 "
             "masking lock); pass a float in [0, 1) to sweep the M2 masked "
             "fraction. None is byte-identical to the prior hardcoded default.",
    )
    p.add_argument(
        "--m2-mask-type", dest="m2_mask_type", choices=("bands", "random"),
        default=DEFAULT_M2_MASK_TYPE,
        help="#75 M2 mask shape. 'bands' (default, 6/03 lock) = structured 1D "
             "spectro-temporal bands; 'random' = R-m2-random must-beat sister.",
    )
    p.add_argument(
        "--m4-mask-ratio", dest="m4_mask_ratio", type=float,
        default=None,
        help="#75 M4 parcel tube ratio. Unset → 0.20 (the 6/03 masking lock); "
             "pass a float in [0, 1) to sweep the M4 masked fraction. None is "
             "byte-identical to the prior hardcoded default.",
    )
    p.add_argument(
        "--ema-tau", dest="ema_tau", type=float,
        default=None,
        help="EMA teacher momentum τ override. Unset → 0.99925 (the B26 lock, "
             "ssl/ema.py P1/P2_EMA_TAU); pass a float in (0, 1) to sweep τ "
             "(R-ema-tau-{...}). Joint-SSL only (P3/P4 use no EMA teacher). "
             "None is byte-identical to the prior hardcoded value.",
    )
    p.add_argument(
        "--m4-precision-weight", dest="m4_precision_weight", action="store_true",
        help="Heteroscedastic / inverse-variance weighting on the M4 (P2) masked-"
             "JEPA loss (project_v14_heteroscedastic_ssl_loss): multiply each "
             "masked M4 cell's L1 by a DETACHED, mean-1-normalized precision "
             "w = n_k^α / (σ²+ε) (n_k = parcel electrode count, σ = the mean|std "
             "pool's per-cell std) so the gradient down-weights shaky low-n/high-σ "
             "parcel targets. OFF by default (opt-in); REQUIRES --mean-pool-std "
             "(σ source) AND the joint B37 path. Uniform-weight limit = plain L1.",
    )
    p.add_argument(
        "--m4-precision-alpha", dest="m4_precision_alpha", type=float, default=1.0,
        help="The n^α exponent for --m4-precision-weight. 1.0 (default) = raw "
             "electrode count; α<1 damps high-n parcels (spatially-correlated "
             "electrodes give effective n_eff<n, so plain count over-trusts them).",
    )
    p.add_argument(
        "--m4-precision-floor-pct", dest="m4_precision_floor_pct", type=float,
        default=25.0,
        help="Empirical-Bayes shrinkage prior for --m4-precision-weight: σ²₀ = the "
             "p{this} percentile of in-batch scored σ², ADDED to σ² before "
             "inverting. THE load-bearing fix — without it the ~12.5%% degenerate "
             "(σ²≈0, single-/equal-electrode) cells swamp the loss and starve the "
             "informative cells (measured max/median ≈1e5 → ~4.7). Default p25.",
    )
    p.add_argument(
        "--m4-precision-cap", dest="m4_precision_cap", type=float, default=10.0,
        help="Max per-cell weight (after mean-1 normalization) for "
             "--m4-precision-weight; cheap insurance on top of the shrinkage "
             "floor. <=0 disables the cap. mean1_invvar only (inert under "
             "downweight_dof). Default 10.",
    )
    p.add_argument(
        "--m4-precision-mode", dest="m4_precision_mode",
        choices=("downweight_dof", "mean1_invvar"), default="downweight_dof",
        help="Precision-weight FORM for --m4-precision-weight "
             "(reports/m4_precision_downweight_handoff_2026_06_15.md). "
             "'downweight_dof' (default) = electrode-dof "
             "w=min(1,((n-1)/(n_ref-1))^α): n-only, NOT mean-1 (shaky low-n "
             "parcels genuinely contribute less, sub-1 mean ≈ a principled "
             "λ_m4≈0.58), 0 at n=1, no σ²₀ shrinkage needed. 'mean1_invvar' = "
             "the prior mean-1 inverse-variance w=n^α/(σ²+σ²₀) form (the "
             "R-precision-mean1 falsifier; floor-pct/cap apply only here).",
    )
    p.add_argument(
        "--m4-precision-nref", dest="m4_precision_nref", type=float, default=11.0,
        help="downweight_dof full-trust electrode count (the min(1,·) cap from "
             "electrode-correlation saturation, n_ref≈1+1/ρ; n_ref=11 ⇔ ρ≈0.1). "
             "n≥n_ref → weight 1.0. R-precision-nref-15 = gentler tail (ρ≈0.07). "
             "Must be > 1.0. Inert under mean1_invvar. Default 11.",
    )
    p.add_argument(
        "--m4-mask-type", dest="m4_mask_type", choices=("tube", "time_block"),
        default=DEFAULT_M4_MASK_TYPE,
        help="#75 M4 mask shape. 'tube' (default, 6/03 lock) = whole covered "
             "parcel; 'time_block' = R-time-block sister. In STAGED mode "
             "'time_block' couples to a co_temporal predictor (not yet landed → "
             "fails loud). In JOINT mode the freq-carrying M4 predictor has full "
             "visible-context attention, so 'time_block' is honored as a plain "
             "mask shape (the staged coupling gate does not apply).",
    )
    p.add_argument(
        "--m2-time-band-floor", dest="m2_time_band_floor", type=int,
        default=DEFAULT_M2_TIME_BAND_FLOOR,
        help="#127 M2 masked-band WIDTH along time (T_p). Default 2 (6/03 lock). "
             "LARGER = wider, fewer bands = harder reconstruction (predictor "
             "cannot copy the adjacent time bin) — tests the 'bands too narrow' "
             "hypothesis. The masked fraction is held fixed; realized fraction "
             "rounds DOWN if width does not divide round(frac·T_p).",
    )
    p.add_argument(
        "--m2-freq-band-floor", dest="m2_freq_band_floor", type=int,
        default=DEFAULT_M2_FREQ_BAND_FLOOR,
        help="#127 M2 masked-band WIDTH along frequency (F_p). Default 1 (6/03 "
             "lock). LARGER = wider spectral bands = harder reconstruction. Same "
             "fixed-fraction / round-down semantics as --m2-time-band-floor.",
    )
    # 2STFT dual-band M2 mask block geometry (FE-2STFT only; inert on single-band).
    # Softens the dual-band masking without touching the single-band locks above.
    # Each band overrides only when BOTH width and nbands are passed; unset → the
    # sampler default (low one 3-wide freq tube; high {3,3,2} time cols).
    p.add_argument(
        "--m2-low-freq-width", dest="m2_low_freq_width", type=int, default=None,
        help="2STFT dual-band: LOW-band masked freq-block WIDTH (contiguous "
             "F-patches per block). Default (unset) = sampler's 3-wide single "
             "tube. Requires --m2-low-freq-nbands. e.g. width 2 + nbands 1 → one "
             "2-of-7 freq tube across all time.",
    )
    p.add_argument(
        "--m2-low-freq-nbands", dest="m2_low_freq_nbands", type=int, default=None,
        help="2STFT dual-band: LOW-band number of freq blocks. The realized "
             "low masked frac = width*nbands/F_p_low (sampler scales to n_bands). "
             "Requires --m2-low-freq-width.",
    )
    p.add_argument(
        "--m2-high-time-width", dest="m2_high_time_width", type=int, default=None,
        help="2STFT dual-band: HIGH-band masked time-block WIDTH (contiguous "
             "T-patches per block, fixed ABSOLUTE cols — does NOT scale with T). "
             "Default (unset) = sampler's {3,3,2}. Requires --m2-high-time-nbands. "
             "e.g. width 2 + nbands 16 → 16 disjoint 2-wide time cols across all freq.",
    )
    p.add_argument(
        "--m2-high-time-nbands", dest="m2_high_time_nbands", type=int, default=None,
        help="2STFT dual-band: HIGH-band number of time blocks (each "
             "--m2-high-time-width wide). High masked cols = width*nbands; "
             "must be <= T_high_p. Requires --m2-high-time-width.",
    )
    p.add_argument(
        "--m2-high-anchor-frac", dest="m2_high_anchor_frac", type=float, default=None,
        help="2STFT dual-band HIGH band, ANCHOR-DILATE mode (Ben 2026-06-13 "
             "easier-mask regime). Fraction of high-band TIME positions sampled "
             "as anchors; each anchor masks the position + the next "
             "(--m2-high-anchor-width - 1), OVERLAPS allowed → union. e.g. 0.30 "
             "+ width 2 on T_high_p=80 ≈ 51%% of time-cols. Requires "
             "--m2-high-anchor-width; mutually exclusive with --m2-high-time-*.",
    )
    p.add_argument(
        "--m2-high-anchor-width", dest="m2_high_anchor_width", type=int, default=None,
        help="2STFT dual-band HIGH band anchor dilation WIDTH (mask the anchor + "
             "the next width-1 time-patches). Default regime uses 2. Requires "
             "--m2-high-anchor-frac.",
    )
    p.add_argument(
        "--predictor-hidden", dest="predictor_hidden", type=int,
        default=DEFAULT_PREDICTOR_HIDDEN,
        help="#76 predictor hidden width (default 192 = d/2 for the canonical "
             "B37 d=384 run; was 128). Shared by BOTH joint predictors.",
    )
    p.add_argument(
        "--predictor-n-heads", dest="predictor_n_heads", type=int,
        default=DEFAULT_PREDICTOR_N_HEADS,
        help="#76 predictor attention head count (default 4). Shared by BOTH "
             "joint predictors.",
    )
    # --------------------------------------------------------------------------
    # NOTE: ``--latent-valid-override`` / ``--sa-mask-mode`` (B30 sister
    # selectors) were culled 2026-06-13 — their only non-default choices were
    # always NotImplementedError sisters. The B30-lock values ("support" /
    # "bidirectional") are hardcoded at the joint-experiment build site.
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
        "--include-ajile12", dest="include_ajile12",
        action="store_true", default=DEFAULT_INCLUDE_AJILE12,
        help="Include AJILE12 in the pretraining mix. Default OFF (2026-06-07): "
             "the active chain is BT-only. Turn ON for the joint-corpus "
             "escalation (B29 mix).",
    )
    # NOTE: ``--no-include-ajile12`` (redundant store_false alias of the
    # default-OFF), ``--ffn-variant`` (MoE audit-rejected; always dense), and
    # ``--cross-attn-positions`` (Perceiver, thrown away) were culled
    # 2026-06-13. The encoder keeps its own ``cross_attn_positions`` (→ [0])
    # internal default.
    # MASK-01 per-corpus mains-notch field.
    p.add_argument(
        "--mains-notch-hz", type=float, default=DEFAULT_MAINS_NOTCH_HZ,
        help="Mains-frequency notch (Hz). Default 60.0 (US — BT, D-cohort, "
             "AJILE12). Pass 50.0 for SWEC (Swiss site). Per-corpus map "
             "lives in MAINS_NOTCH_BY_CORPUS.",
    )
    p.add_argument("--phase", type=int, choices=(1, 3, 4), default=1,
                   help="Training phase per docs/neuroprobe/plan.md §staged. "
                        "1 = masked-JEPA SSL (V14JointExperiment); B36 staged "
                        "P1->P2 selected via --jepa-phase (p1 front-end M2 / "
                        "p2 parcel M4). "
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
    # Online linear probe (diagnostic; spec reports/online_probe_spec_2026_06_18.md).
    p.add_argument("--online-probe", dest="online_probe", action="store_true",
                   default=False,
                   help="--frontend 3stft only: register the diagnostic frozen "
                        "online linear probe (frontend+latent taps, WS/CS AUROC "
                        "every --online-probe-cadence steps). OFF by default; the "
                        "probe dataset is built in-worker (DCC-only). Inert on "
                        "raw/2stft.")
    p.add_argument("--online-probe-cadence", type=int, default=1000,
                   help="Steps between online-probe fires (default 1000; spec §3 "
                        "keep if overhead <~3%%, else stretch).")
    p.add_argument("--monitor-every-n-steps", type=int, default=None,
                   help="--frontend 3stft only: cadence for the EXPENSIVE forward-"
                        "tap monitors (RankMe/coverage/input-stats — each re-runs a "
                        "no_grad extra forward; the post-latent RankMe tap is a "
                        "DENSE full-input latent pass). Default None ⇒ gate on "
                        "--log-every-n-steps (firing every step at cadence 1 ~2.1x'd "
                        "the step on 5000 Ada). Set e.g. 50 to keep loss curves "
                        "per-step while the step-doubling extra forward fires "
                        "sparsely.")
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
# E5/WS-F/#21 2026-06-03: ``_PHASE3_BLOCKERS`` removed. The whisper_target
# segmenter emission (WS-H, #20) landed, so --phase 3 now routes to
# V14Phase3Experiment via build_v14_experiment(p3_distill=True) and the chain
# driver runs P3a/P3b end-to-end. 2026-06-13: ``--phase 2`` (the legacy
# split-P2 entry) was dropped from the CLI ``--phase`` choices, so the
# ``_PHASE2_BLOCKERS`` tuple + its main() guard are gone too — dispatch P2 via
# ``--phase 1 --jepa-phase p2``.


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


def _resolve_lr(args) -> float:
    """Peak/base LR for every phase.

    ``--lr`` (default ``None``) resolves to the AUTO √-rule:
    ``lr = 5e-4 · √(eff / 1024)`` with ``eff = batch_size ×
    accumulate_grad_batches × gpus_per_node`` (the linear-scaling-anchor for
    SSL; 5e-4 @ 1024 is the §7 center, validated at eff-128 → 1.76e-4). An
    explicit ``--lr`` overrides. Idempotent + pure in ``args`` so the chain,
    the single-phase build, and the direct-call tests all resolve identically.
    """
    if args.lr is not None:
        return float(args.lr)
    n_gpu = args.gpus_per_node or 1
    eff = args.batch_size * args.accumulate_grad_batches * n_gpu
    return 5e-4 * (eff / 1024) ** 0.5


def _build_wandb_config(args) -> tp.Any | None:
    """W&B logger config for the --live nano learning-dynamics dashboard.

    None unless --live is set. Run name defaults to ``<task>_<mode>_p<phase>`` so a
    predict-then-check ladder overlays cleanly; override per rung with
    --wandb-run-name. See reports/nano_dynamics_dashboard_handoff_2026_06_07.md.
    """
    if not getattr(args, "live", False):
        return None
    from neuraltrain.utils import WandbLoggerConfig

    phase = getattr(args, "phase", "?")
    name = args.wandb_run_name or f"{args.task}_{args.mode}_p{phase}"
    group = args.wandb_group or f"{args.task}_{args.mode}"
    return WandbLoggerConfig(
        name=name, group=group, project=args.wandb_project,
        offline=args.wandb_offline,
        id=getattr(args, "wandb_run_id", None),
    )


def _common_build_kwargs(args) -> dict[str, tp.Any]:
    """Knobs identical across every phase in BOTH the single-phase ``main()``
    build and the ``--chain`` builds.

    Centralized after the Gate-D audit: ``binary_tasks`` / ``loss_variant`` had
    drifted OUT of the chain's inline ``common`` dict while the single-phase
    call still passed them, so a ``--chain --loss-variant X`` silently ran the
    DEFAULT arm while the run summary printed the sister as applied. Both call
    sites consume this one dict, so a forgotten flag is now structurally
    impossible.

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
        pretrain_holdout_fraction=args.pretrain_holdout_fraction,
        binary_tasks=args.binary_tasks,
        session_robust_z=args.session_robust_z,
        d_model=args.d_model, depth=args.depth,
        n_heads=args.n_heads, m_sub_slots=args.m_sub_slots,
        batch_size=args.batch_size, num_workers=args.num_workers,
        n_epochs=args.n_epochs,
        # --live nano learning-dynamics dashboard. Non-live → all three at their
        # Experiment defaults, so the cache uid + behavior are unchanged.
        # reports/nano_dynamics_dashboard_handoff_2026_06_07.md.
        wandb_config=_build_wandb_config(args),
        lr_log_interval="step" if getattr(args, "live", False) else "epoch",
        log_every_n_steps=1 if getattr(args, "live", False) else 10,
        ckpt_ladder_every=args.ckpt_ladder_every,
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
        in_allocation_ddp=args.in_allocation_ddp,
        requeue=args.slurm_requeue,
        precision=args.precision,
        extractor_cache_folder=args.extractor_cache_folder,
        # #80 whole-movie |STFT| cache (default ON; build derives the path from the
        # extractor cache root). Reaches every phase via this one dict.
        spec_cache_dir=args.spec_cache_dir,
        disable_spec_cache=args.no_spec_cache,
        spec_only=args.spec_only,
        trial_durations_path=args.trial_durations,
        # Layer-2 bad-electrode clip filter (#180). Reaches every phase via this one
        # dict; build_v14_experiment gates it to SSL phases (P4 self-zeroes to None).
        bad_window_dir=args.bad_window_dir,
        group_by_session=args.group_by_session,
        balance_ranks=args.balance_ranks,
        same_session_across_ranks=args.same_session_ranks,
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
        # Parcellation atlas + single-electrode-parcel exclusion ride this dict so
        # the chain's 4 phases share ONE vocabulary (k_parcels) and ONE support /
        # valid-mask config — a per-phase atlas mismatch would misalign the
        # cross-subject parcel embedding. Both reach the support + valid_mask
        # extractors inside build_v14_experiment.
        atlas=args.atlas,
        exclude_single_electrode_parcels=args.exclude_single_electrode_parcels,
        # Front-end family (raw single-grid vs 2STFT dual-band). Rides this dict
        # so every chain phase shares ONE front end — a per-phase mismatch would
        # feed the P4 probe a different token geometry than the SSL encoder saw.
        frontend=args.frontend,
        converged_frontend_layers=args.converged_frontend_layers,
        converged_latent_layers=args.converged_latent_layers,
        converged_m2_pred_dim=args.converged_m2_pred_dim,
        converged_m2_pred_layers=args.converged_m2_pred_layers,
        converged_m4_pred_dim=args.converged_m4_pred_dim,
        converged_m4_pred_layers=args.converged_m4_pred_layers,
        converged_lambda_m2=args.converged_lambda_m2,
        converged_lambda_m4=args.converged_lambda_m4,
        converged_m2_hg_start_rate=args.converged_m2_hg_start_rate,
        converged_m2_hg_span=args.converged_m2_hg_span,
        converged_m2_beta_start_rate=args.converged_m2_beta_start_rate,
        converged_m2_beta_span=args.converged_m2_beta_span,
        converged_m2_slow_freq_tubes=args.converged_m2_slow_freq_tubes,
        converged_m4_parcel_mask_ratio=args.converged_m4_parcel_mask_ratio,
        converged_tube_ratio=args.converged_tube_ratio,
        converged_tube_p_fixed=args.converged_tube_p_fixed,
        converged_static_forward=args.converged_static_forward,
        converged_m4_precision_off=args.converged_m4_precision_off,
        converged_m4_precision_alpha=args.converged_m4_precision_alpha,
        converged_m4_precision_n_ref=args.converged_m4_precision_n_ref,
        online_probe_enabled=args.online_probe,
        online_probe_cadence=args.online_probe_cadence,
        monitor_every_n_steps=args.monitor_every_n_steps,
        cache_band=args.cache_band,
        subtype_embed_enabled=args.subtype_embed_enabled,
        subtype_embed_reuse_kv=args.subtype_embed_reuse_kv,
        subtype_embed_vocab=args.subtype_embed_vocab,
        ref_embed_enabled=args.ref_embed_enabled,
        ref_embed_reuse_kv=args.ref_embed_reuse_kv,
        ref_operator_alpha=args.ref_operator_alpha,
        include_ajile12=args.include_ajile12,
        gradient_checkpointing=args.gradient_checkpointing,
        cache_session_index=args.cache_session_index,
        ragged_frontend=args.ragged_frontend,
        ragged_parcel=args.ragged_parcel,
        ragged_token=args.ragged_token,
        ragged_predictor=args.ragged_predictor,
        freq_pos=args.freq_pos,
        readout=args.readout,
        # B37 (2026-06-10) encoder + joint-SSL knobs. pool/latent_depth ride the
        # brain-model config (shared encoder, every phase); the rest are
        # joint-only inside build_v14_experiment (inert on P3/P4). ssl_mode is
        # resolved from "auto" in main() before this dict is built.
        pool=args.pool,
        latent_depth=args.latent_depth,
        latent_mode=args.latent_mode,
        mean_pool_std=args.mean_pool_std,
        ssl_mode=args.ssl_mode,
        lambda_m4=args.lambda_m4,
        m2_predictor_depth=args.m2_predictor_depth,
        m4_predictor_depth=args.m4_predictor_depth,
        joint_frontend_lr_scale=args.joint_frontend_lr_scale,
        joint_parcel_lr_scale=args.joint_parcel_lr_scale,
        m2_mask_type=args.m2_mask_type,
        m2_mask_ratio=args.m2_mask_ratio,
        m4_mask_type=args.m4_mask_type,
        m4_mask_ratio=args.m4_mask_ratio,
        m2_time_band_floor=args.m2_time_band_floor,
        m2_freq_band_floor=args.m2_freq_band_floor,
        # 2STFT dual-band M2 block geometry (None → sampler default; inert single-band).
        m2_low_freq_width=args.m2_low_freq_width,
        m2_low_freq_nbands=args.m2_low_freq_nbands,
        m2_high_time_width=args.m2_high_time_width,
        m2_high_time_nbands=args.m2_high_time_nbands,
        m2_high_anchor_frac=args.m2_high_anchor_frac,
        m2_high_anchor_width=args.m2_high_anchor_width,
        # SSL-sweep EMA τ override (#sweep). None → build_v14_experiment resolves
        # it to 0.99925 (the B26 lock); joint-only inside the builder, so passing
        # it on every phase (incl. P3/P4) is inert off the SSL path.
        ema_tau=args.ema_tau,
        m4_precision_weight=args.m4_precision_weight,
        m4_precision_alpha=args.m4_precision_alpha,
        m4_precision_floor_pct=args.m4_precision_floor_pct,
        m4_precision_cap=args.m4_precision_cap,
        m4_precision_mode=args.m4_precision_mode,
        m4_precision_nref=args.m4_precision_nref,
        predictor_hidden=args.predictor_hidden,
        predictor_n_heads=args.predictor_n_heads,
        loss_variant=args.loss_variant,
        # #37 optim / LR-schedule (audit 2026-06-03). --adam-beta2 → (0.9, β2)
        # tuple; --grad-clip <= 0 → None (disable). Reaches every phase via this
        # one dict, so the chain and single-phase builds stay in lock-step.
        lr=_resolve_lr(args),
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer_name,
        adam_betas=(0.9, args.adam_beta2) if args.adam_beta2 is not None else None,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        # §7/B01 no-WD param-group split (#40). Uniform across every phase
        # (joint P1/P2, P3, P4 frozen probe) — reaches both the single-phase
        # main() build and the --chain builds via this one dict.
        wd_exclude_norms=args.wd_exclude_norms,
        # MON-TEACHER-FEATURE-RANK thresholds (#74). Joint-only inside
        # build_v14_experiment; passing them on every build is harmless (P3/P4
        # branches don't forward them). None preserves the canonical 0.5/0.25.
        rankme_warn_threshold=args.rankme_warn_threshold,
        rankme_alarm_threshold=args.rankme_alarm_threshold,
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
    if not (args.target_standardize and args.channel_stats_path is not None):
        return
    if not Path(args.channel_stats_path).is_file():
        raise ValueError(
            f"--channel-stats-path {args.channel_stats_path!r} is not a file. "
            "Target standardization (B33 default) loads it at P3a; a missing or "
            "directory path would crash hours into the chain. Build the .pt with "
            "the channel_stats fit helper, or pass --no-target-standardize."
        )
    # CN-3: the per-channel affine is the SAME width (1280-d) across layer-merges,
    # so a stats file fit on one merge applied to a teacher cache built for another
    # z-scores the distillation target with the WRONG frozen affine SILENTLY (no
    # shape error). channel_stats_path / whisper_layer_merge are independent free
    # args — guard their coupling here, at the one junction both are known.
    # Assert-if-present: legacy stats files (no provenance) and unreadable metadata
    # are skipped (no new failure mode), so this only fires on a true mismatch.
    stats_merge = None
    try:
        import torch

        rec = torch.load(
            args.channel_stats_path, map_location="cpu", weights_only=True,
        )
        stats_merge = rec.get("layer_merge") if isinstance(rec, dict) else None
    except Exception:
        stats_merge = None
    if stats_merge is not None:
        from speech_decoding.bt_alignment.teacher_cache import merge_slug

        want = merge_slug(args.whisper_layer_merge)
        got = merge_slug(stats_merge)
        if got != want:
            raise ValueError(
                f"--channel-stats-path was fit on layer_merge={stats_merge!r} "
                f"({got}) but --whisper-layer-merge={args.whisper_layer_merge!r} "
                f"({want}). The 1280-d per-channel affine would silently z-score "
                "the distillation target with the wrong frozen stats (CN-3). Re-fit "
                "channel stats for this layer_merge, or pass the matching cache."
            )


def _build_v14_chain(args) -> list[Experiment]:
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

    common = _common_build_kwargs(args)
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
    # <=0 → None = fall back to the --n-epochs budget (the default 1500 is the
    # production step budget; 0/negative is the documented epoch-budget escape).
    ssl_max_steps = (
        args.ssl_max_steps if (args.ssl_max_steps and args.ssl_max_steps > 0)
        else None
    )
    ssl_val_check = args.ssl_val_check_interval
    if ssl_val_check is None and ssl_max_steps is not None:
        ssl_val_check = max(50, ssl_max_steps // 10)
    # #66: cap each SSL validation (P4 stays uncapped — built separately below).
    ssl_limit_val = (
        args.ssl_limit_val_batches
        if args.ssl_limit_val_batches and args.ssl_limit_val_batches > 0
        else None
    )
    # Shared SSL/distill-phase budget (P1/P2/P3a/P3b — NOT P4). collapse_guard is
    # set PER PHASE below, NOT here: P1's front-end RankMe floor (~0.31 on the M2
    # |STFT| representation, measured by the guard-OFF diagnostic 47723576) sits
    # above the 0.25 hard alarm and its soft warn is already advisory, so P1 can
    # never false-positive — it ALWAYS keeps the guard. That also pins P1's exca
    # UID, so a `--no-collapse-guard` relaunch reuses the cached P1 instead of
    # re-running it (~6 h). The parcel/distill phases operate on the M4 parcel-
    # token representation, whose floor is ~0.05 (raw effective rank ~ active-
    # parcel count ~16, normalised by d=256) — that is what false-positived the
    # M2-calibrated alarm and killed chain 47725245 at P2, so `--no-collapse-
    # guard` disarms ONLY P2/P3a/P3b. P4 keeps its own explicit
    # collapse_guard=False (frozen probe, #54 audit M1). The per-monitor (M2 vs
    # M4) threshold recalibration that this comment used to call the long-term
    # fix has LANDED: the M4 probe now defaults to its own empirical band
    # (0.04/0.02) phase-keyed inside V14JointBrainModule, so P2 stays ARMED at a
    # floor-anchored alarm without needing `--no-collapse-guard`. The flag and
    # this per-phase split are retained as the diagnostic escape hatch.
    # See [[project_v14_gate_cadence_guard_response_lock_2026_06_05]].
    ssl_budget: dict[str, tp.Any] = dict(
        max_steps=ssl_max_steps,
        val_check_interval=ssl_val_check,
        limit_val_batches=ssl_limit_val,
    )
    p1 = build_v14_experiment(
        **common, **ssl_budget, collapse_guard=args.collapse_guard,
        joint_phase=True,
        jepa_phase="p1", clip_len=5.0, neural_lag_s=args.neural_lag_s,
    )
    p2 = build_v14_experiment(
        **common, **ssl_budget, collapse_guard=args.collapse_guard,
        joint_phase=True, jepa_phase="p2", clip_len=5.0,
        neural_lag_s=args.neural_lag_s,
    )
    p3a = build_v14_experiment(
        **common, **ssl_budget, collapse_guard=args.collapse_guard, **whisper,
        p3_distill=True, p3_stage="3a",
        clip_len=5.0,
        parcel_lr_scale=args.parcel_lr_scale, neural_lag_s=args.neural_lag_s,
    )
    p3b = build_v14_experiment(
        **common, **ssl_budget, collapse_guard=args.collapse_guard, **whisper,
        p3_distill=True, p3_stage="3b",
        clip_len=5.0,
        parcel_lr_scale=args.parcel_lr_scale, neural_lag_s=args.neural_lag_s,
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
        # Eval = BT-Lite: the P4 probe is the leaderboard cell, so it pools over
        # the Neuroprobe-Lite electrode subset (parity on electrode count). The
        # SSL phases above pretrain on the full montage (BT-FULL). The frozen
        # encoder is permutation/montage-invariant by construction (zero-per-
        # subject per-parcel pool), so dropping to the Lite subset at eval is the
        # intended generalization, not a distribution break.
        electrode_set="lite",
    )
    return [p1, p2, p3a, p3b, p4]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # --cache-band is a cache-build-only lever (it swaps the electrode_tokens
    # extractor for one named 3STFT band and rides --cache-only's exit-before-
    # trainer path). Running it without --cache-only would build a single-band
    # model and train it — never intended; fail loudly instead.
    if args.cache_band is not None and not args.cache_only:
        raise ValueError(
            f"--cache-band {args.cache_band!r} is a cache-build lever and requires "
            "--cache-only (it builds one 3STFT band's spec cache, then exits before "
            "the trainer). Add --cache-only, or drop --cache-band for a real run."
        )
    # Predictor depth half-rule (Ben 2026-06-12). Each JEPA predictor defaults to
    # HALF the depth of the encoder stack it predicts from: M2 (front-end) tracks
    # --depth, M4 (parcel/latent) tracks --latent-depth. At the canonical B37
    # depth-6/latent_depth-6 this resolves to 3/3 — byte-identical to the old
    # hard 3/3 default — but a depth sweep (e.g. --latent-depth 2) now auto-scales
    # its predictor (-> M4 depth 1) instead of leaving an over-deep predictor on a
    # shrunk encoder. An explicit --m2/m4-predictor-depth overrides the rule.
    if args.m2_predictor_depth is None:
        args.m2_predictor_depth = max(1, args.depth // 2)
    if args.m4_predictor_depth is None:
        args.m4_predictor_depth = max(1, args.latent_depth // 2)
    _resolve_static_forward_cohesion(args)
    # speedup-fanout C1: --compile/--no-compile is a front-door for the
    # V14_COMPILE env var the brain module reads at construction. Set it here
    # (before exca submits) so submitit captures it into the slurm job env. Set
    # EXPLICITLY to "0"/"1" so --no-compile is authoritative and a prior run's
    # value in a long-lived warm worker can never leak into this run's read. The
    # old --in-allocation-ddp force-disable was removed 2026-06-21 (compile is now
    # cloudpickle-safe via the module __getstate__), so the requested flag is
    # honored verbatim — the 4-GPU in-allocation run trains WITH compile on.
    os.environ["V14_COMPILE"] = "1" if args.compile_encoder else "0"
    if args.compile_mode:
        os.environ["V14_COMPILE_MODE"] = args.compile_mode
    # Set EXPLICITLY "1"/"0" (not on-true only) so --no-compile-dynamic reaches
    # the module as genuine dynamic=False (FULLY static, no symbolic reasoning) —
    # not the unset→None automatic-dynamic that still storms on torch 2.10/GH200.
    # Same no-stale-leak rationale as the throughput levers below.
    os.environ["V14_COMPILE_DYNAMIC"] = "1" if args.compile_dynamic else "0"
    # Science-neutral SDPA kernel preference (front-door for V14_SDPA_BACKEND, read
    # in V14ConvergedBrainModule). Set EXPLICITLY (set/pop) so a prior run's value
    # in a long-lived warm worker never leaks; 'default' pops it (stock selection).
    if args.sdpa_backend and args.sdpa_backend != "default":
        os.environ["V14_SDPA_BACKEND"] = args.sdpa_backend
    else:
        os.environ.pop("V14_SDPA_BACKEND", None)
    # Loss-neutral matmul/conv fast paths. --in-allocation-ddp runs main() in EACH
    # rank (srun launches ntasks copies), so setting the process-global torch
    # backend state here applies on every GPU before the model/trainer is built.
    # TF32 only touches RESIDUAL fp32 matmuls (the bf16-mixed hot path is already
    # bf16); cudnn.benchmark autotunes the static-shape conv stems. Env-gated +
    # default OFF so the bit-exact path is unchanged and the exca uid is untouched.
    os.environ["V14_FAST_BACKENDS"] = "1" if args.fast_backends else "0"
    if args.fast_backends:
        import torch
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(
            "[dispatch] --fast-backends: matmul_precision=high, cuda/cudnn "
            "allow_tf32=True, cudnn.benchmark=True (loss-neutral fast paths).",
            flush=True,
        )
    # 2026-06-09 throughput levers — same front-door pattern. Set EXPLICITLY to
    # "0"/"1" (not just on-true) so a prior run's value in a long-lived warm
    # worker process never leaks into this run's data.py / experiment.py reads.
    os.environ["V14_WARM_DATASET_CACHE"] = "1" if args.warm_dataset_cache else "0"
    os.environ["V14_NO_SANITY_VAL"] = "1" if args.no_sanity_val else "0"
    os.environ["V14_NO_TRAINER_CKPT"] = "1" if args.no_trainer_ckpt else "0"
    os.environ["V14_NO_TEST"] = "1" if args.no_test else "0"
    os.environ["V14_DDP_STATIC_GRAPH"] = "1" if args.ddp_static_graph else "0"
    os.environ["V14_P1_FREEZE_PARCEL"] = "1" if args.p1_freeze_parcel else "0"
    if args.log_every_n_override is not None:
        os.environ["V14_LOG_EVERY_N"] = str(args.log_every_n_override)
    else:
        os.environ.pop("V14_LOG_EVERY_N", None)  # no stale leak in warm worker
    # Layer-3 winsor cap (#180): front-door for V14_SESSION_Z_WINSOR, which
    # extractors.normalize.SessionRobustZNormalizer reads at construction. Kept as
    # an env knob (not a build_v14_experiment / view field) on purpose — it is a
    # read-time clamp that must NOT enter the exca cache uid (a serialized field
    # would re-fork the multi-TB spec cache). Set/pop EXPLICITLY (same warm-worker
    # no-leak pattern as the levers above) so --session-z-winsor is authoritative
    # and a prior run's value never leaks. Default None = unset = no clamp.
    if args.session_z_winsor is not None:
        os.environ["V14_SESSION_Z_WINSOR"] = str(args.session_z_winsor)
    else:
        os.environ.pop("V14_SESSION_Z_WINSOR", None)
    # Per-band winsor caps (#230): same front-door pattern, one env knob per band.
    # A band-specific cap wins over the global scalar for that band only; set/pop
    # explicitly so a warm worker never inherits a prior run's per-band value.
    for _b in ("slow", "beta", "hg"):
        _cap = getattr(args, f"session_z_winsor_{_b}")
        _env = f"V14_SESSION_Z_WINSOR_{_b.upper()}"
        if _cap is not None:
            os.environ[_env] = str(_cap)
        else:
            os.environ.pop(_env, None)
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
    _is_ssl_distill = bool(args.chain) or args.phase in (1, 3)
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
    # B37 D7 (2026-06-10): resolve --ssl-mode auto → 'joint' when --pool mean
    # else 'staged', then enforce the pool↔ssl-mode coupling. The B37 joint
    # objective is the freq-preserving mean-pool path: a cross-attn encoder
    # cannot run it (the module rejects it too), and on an SSL phase the
    # mean-pool freq-preserving latent has no staged single-term path — so the
    # two must move together. Scoped: the 'mean ⇒ joint' direction is enforced
    # only on the SSL phases (--phase 1 / --chain); --phase 4 may build a
    # mean-pool encoder for a downstream probe where ssl_mode is irrelevant.
    # 2026-06-13: the B36 'staged' SSL CLI surface was culled. 'auto' now always
    # resolves to 'joint' (the B37 D7 single-forward composite-mask objective);
    # the joint objective is the freq-preserving mean-pool path, so an SSL phase
    # requires --pool mean. (P4 may build a cross_attn encoder for a downstream
    # probe where ssl_mode is irrelevant — the joint⇒mean coupling is scoped to
    # the SSL phases.)
    if args.ssl_mode == "auto":
        args.ssl_mode = "joint"
    _ssl_phase = bool(args.chain) or args.phase == 1
    # Static-shape forward needs BOTH the static tube masks and the session-
    # homogeneous batch sampler (compute_static_shapes fails loud on a hetero
    # batch). Catch a misconfigured launch here rather than mid-run.
    if args.converged_static_forward:
        if args.converged_tube_ratio is None:
            raise SystemExit(
                "--converged-static-forward requires --converged-tube-ratio "
                "(the static forward needs the tight-pack tube's constant n_vis).")
        if not args.group_by_session:
            raise SystemExit(
                "--converged-static-forward requires --group-by-session (the "
                "forward needs a session-homogeneous batch; it fails loud "
                "otherwise).")
    if args.same_session_ranks and not args.group_by_session:
        raise SystemExit(
            "--same-session-ranks requires --group-by-session (it reorganizes "
            "the session-homogeneous batches so all ranks share one session; "
            "without grouping there are no per-session batches to align).")
    if _ssl_phase and args.ssl_mode == "joint" and args.pool != "mean":
        raise SystemExit(
            "--ssl-mode joint (B37 D7) requires --pool mean (the "
            f"freq-preserving mean-pool encoder); got --pool {args.pool}. The "
            "B36 staged cross_attn SSL path was culled — run the joint SSL "
            "phase with --pool mean."
        )
    # B37: the mean-pool encoder emits a 5-D parcel×freq×time latent. P4 has a
    # freq-preserving readout (V14FreqPreservingPmaReadout, wired in build());
    # P3 distill does NOT yet — V14Phase3DistillModule._student_readout feeds the
    # M4 tap straight to a V14ParcelCollapsePMA that unpacks a 4-D (B,K,T,d)
    # latent, so --phase 3 --pool mean would crash one batch into the forward.
    # Fail loud at config time until the P3 freq-preserving distill readout lands
    # (deferred — same status as the non-default P4 readouts the build() guard
    # rejects). The full B37 P1(joint)→P3→P4 pipeline is blocked on that readout.
    if args.phase == 3 and args.pool == "mean":
        raise SystemExit(
            "--pool mean (B37) on --phase 3 is not wired: the P3 distill student "
            "readout consumes a 4-D cross_attn latent and has no freq-preserving "
            "variant for the 5-D mean-pool latent yet (it would crash one batch "
            "into the forward). Run P3 with the default --pool cross_attn, or "
            "launch the joint SSL phase with --phase 1 --pool mean."
        )
    # The B37 joint chain (one joint-SSL phase → P3 → P4) is NOT yet wired:
    # _build_v14_chain hardcodes the B36 staged p1→p2 split, which would run the
    # full joint M2+M4 objective TWICE (once per staged sub-phase). Run the
    # joint SSL phase on its own with --phase 1 --pool mean for now.
    if args.chain and args.ssl_mode == "joint":
        raise SystemExit(
            "--chain with --ssl-mode joint (B37 D7) is not wired yet — the chain "
            "builder still uses the B36 staged p1→p2 split and would run the "
            "joint objective twice. Launch the joint SSL phase standalone with "
            "--phase 1 --pool mean."
        )
    # --frontend 3stft routes the WHOLE run to the self-contained V14ConvergedSSL
    # (its own M2/M4 SSL), which replaces the staged V14ParcelPerceiver pipeline.
    # _build_v14_chain does not thread --frontend, so --chain would SILENTLY build
    # raw V14ParcelPerceiver phases. Reject the combo loudly.
    if args.chain and args.frontend == "3stft":
        raise SystemExit(
            "--chain with --frontend 3stft is not supported: the chain builder "
            "assembles the V14ParcelPerceiver staged P1→P2→P3→P4 pipeline, which "
            "the self-contained V14ConvergedSSL replaces. Launch the converged "
            "SSL phase standalone (no --chain)."
        )
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
    # 2026-06-13: --phase 2 (legacy split-P2) was dropped from the --phase
    # choices; dispatch P2 via --phase 1 --jepa-phase p2 (V14JointExperiment).
    print(f"V14 dispatch — cohort subject_ids = {V14_TRAIN_SUBJECT_IDS} (9 subjects, S5 excluded)")
    print(f"  mode={args.mode} task={args.task} binary_tasks={args.binary_tasks} seed={args.seed}")
    print(f"  eval_mode={args.eval_mode} test=({args.test_subject_id},{args.test_trial_id})")
    print(f"  d_model={args.d_model} depth={args.depth} n_heads={args.n_heads} "
          f"M={args.m_sub_slots}")
    # Resolve K from the SELECTED atlas (not a hard-coded "K=80 DK") so the run
    # record can never silently claim DK/K=80 while a --atlas dkt (K=74) run is
    # actually building — atlas_spec is the same source build_v14_experiment uses.
    _atlas_k = len(atlas_spec(args.atlas)[1])
    _excl_note = (
        ", single-electrode parcels excluded"
        if args.exclude_single_electrode_parcels
        else ""
    )
    print(f"  atlas={args.atlas.upper()} (K={_atlas_k} parcels{_excl_note}), "
          f"c_max={args.c_max}, batch_size={args.batch_size}, "
          f"n_epochs={args.n_epochs}")
    print(f"  mains_notch_hz={args.mains_notch_hz}")
    print(f"  lof_bad_channels={args.lof_bad_channels} lof_threshold={args.lof_threshold} "
          f"lof_n_neighbors={args.lof_n_neighbors} lof_report_path={args.lof_report_path}")
    print(f"  jepa_phase={args.jepa_phase} "
          f"neural_lag_s={args.neural_lag_s} "
          f"include_ajile12={args.include_ajile12} ref_operator_alpha={args.ref_operator_alpha}")
    # B37 (2026-06-10) encoder + joint-SSL config (ssl_mode already resolved from
    # 'auto'). Surfaced so the persisted run record never silently rides the
    # wrong pool / SSL objective / discriminative-LR split.
    print(f"  pool={args.pool} latent_depth={args.latent_depth} ssl_mode={args.ssl_mode} "
          f"lambda_m4={args.lambda_m4} predictor_depth=(m2={args.m2_predictor_depth},"
          f"m4={args.m4_predictor_depth},"
          f"hidden={args.predictor_hidden},heads={args.predictor_n_heads}) "
          f"joint_lr_scale=(frontend={args.joint_frontend_lr_scale},"
          f"parcel={args.joint_parcel_lr_scale}) "
          # SSL-sweep overrides: show the EFFECTIVE value (None → the locked
          # constant build_v14_experiment resolves to) so the summary never
          # prints a bare "None" for an unset sweep knob. Explicit None checks
          # (not ``or``) so a legitimately-passed 0.0 ratio still prints 0.0.
          f"mask=(m2={args.m2_mask_type}@"
          f"{DEFAULT_M2_MASK_RATIO if args.m2_mask_ratio is None else args.m2_mask_ratio},"
          f"m4={args.m4_mask_type}@"
          f"{DEFAULT_M4_MASK_RATIO if args.m4_mask_ratio is None else args.m4_mask_ratio}) "
          f"m2_band_floor=(t{args.m2_time_band_floor},f{args.m2_freq_band_floor}) "
          f"ema_tau={DEFAULT_EMA_TAU if args.ema_tau is None else args.ema_tau}")
    # 2STFT dual-band M2 block geometry (only meaningful on the FE-2STFT path).
    # Print the resolved override or "default" per band so the persisted summary
    # never implies the locked single-band geometry on a dual-band run.
    _low_dual = (
        f"{args.m2_low_freq_width}x{args.m2_low_freq_nbands}"
        if args.m2_low_freq_width is not None else "default(3x1)"
    )
    if args.m2_high_anchor_frac is not None:
        _high_dual = (
            f"anchor(frac={args.m2_high_anchor_frac},width={args.m2_high_anchor_width},"
            "overlaps)"
        )
    elif args.m2_high_time_width is not None:
        _high_dual = f"multiset({args.m2_high_time_width}x{args.m2_high_time_nbands})"
    else:
        _high_dual = "default({3,3,2})"
    print(f"  m2_dual_band(2STFT only): low_freq={_low_dual} high_time={_high_dual}")
    print(f"  subtype_embed=(enabled={args.subtype_embed_enabled},reuse_kv={args.subtype_embed_reuse_kv},"
          f"vocab={args.subtype_embed_vocab}) "
          f"ref_embed=(enabled={args.ref_embed_enabled},reuse_kv={args.ref_embed_reuse_kv}) "
          f"loss_variant={args.loss_variant} "
          f"readout={args.readout} "
          f"gradient_checkpointing={args.gradient_checkpointing} "
          f"ragged_frontend={args.ragged_frontend} "
          f"ragged_parcel={args.ragged_parcel} "
          f"ragged_token={args.ragged_token} "
          f"ragged_predictor={args.ragged_predictor} "
          f"freq_pos={args.freq_pos}")
    _resolved_lr = _resolve_lr(args)
    _lr_disp = (
        f"{_resolved_lr:.3e} (AUTO √-rule: 5e-4·√(eff/1024), "
        f"eff={args.batch_size}×{args.accumulate_grad_batches}×"
        f"{args.gpus_per_node or 1})"
        if args.lr is None else f"{_resolved_lr:.3e} (explicit)"
    )
    print(f"  optim: name={args.optimizer_name} lr={_lr_disp} "
          f"lr_schedule={args.lr_schedule} warmup_steps={args.warmup_steps} "
          f"min_lr_ratio={args.min_lr_ratio} weight_decay={args.weight_decay} "
          f"wd_exclude_norms={args.wd_exclude_norms} "
          f"adam_beta2={args.adam_beta2} grad_clip={args.grad_clip} "
          f"accumulate_grad_batches={args.accumulate_grad_batches}")
    print(f"  disc-lr: parcel_lr_scale={args.parcel_lr_scale} "
          f"(P2/P3 discriminative-LR)")
    print(f"  guard: collapse_guard={args.collapse_guard} (OFF by default — "
          f"never auto-kills; --collapse-guard opts in) "
          f"ssl_val_check_interval={args.ssl_val_check_interval} (opt-steps, "
          f"×accum at Trainer #66) ssl_limit_val_batches={args.ssl_limit_val_batches}")
    _joint = args.ssl_mode == "joint"
    _warn_disp = (
        ("joint-default (M2=M4 0.020)" if _joint
         else "phase-default (P1/M2 0.5, P2/M4 0.04)")
        if args.rankme_warn_threshold is None
        else args.rankme_warn_threshold)
    _alarm_disp = (
        ("joint-default (M2=M4 0.010)" if _joint
         else "phase-default (P1/M2 0.25, P2/M4 0.02)")
        if args.rankme_alarm_threshold is None
        else args.rankme_alarm_threshold)
    print(f"  rankme: warn={_warn_disp} alarm={_alarm_disp} "
          f"(joint P1/P2; normalised RankMe; alarm kills, #74)")
    # Heteroscedastic (precision-weighted) M4 SSL loss (#140; OFF by default).
    # Surfaced so the run record shows whether the M4 L1 was inverse-variance
    # weighted (and with what α / floor-pct / cap) instead of silently riding
    # the uniform-weight default.
    if args.m4_precision_weight:
        if args.m4_precision_mode == "downweight_dof":
            note = ("downweight-only electrode-dof w=min(1,((n-1)/(nref-1))^α); "
                    "floor_pct/cap inert; sub-1 mean ≈ a principled λ_m4")
        else:
            note = ("mean-1 inverse-variance w=n^α/(σ²+σ²₀); nref inert; "
                    "R-precision-mean1")
        # alpha stays immediately after ON (run-record greppability); every knob's
        # value is printed so the config is reconstructable, the note flags active.
        print(f"  m4_precision: ON alpha={args.m4_precision_alpha} "
              f"mode={args.m4_precision_mode} nref={args.m4_precision_nref} "
              f"floor_pct={args.m4_precision_floor_pct} "
              f"cap={args.m4_precision_cap} "
              f"({note}, #140)")
    else:
        print("  m4_precision: OFF (uniform-weight M4 L1)")
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
    elif args.in_allocation_ddp:
        # Warm 4-GPU worker: exca local, DDP across the live srun ranks.
        print(f"  in-allocation DDP: WORLD_SIZE={os.environ.get('SLURM_NTASKS', '?')} "
              f"ddp_strategy=ddp_find_unused_parameters_true (exca local, no submitit)")
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
    _resolved_spec = (
        None
        if args.no_spec_cache
        else (
            args.spec_cache_dir
            or (f"{_resolved_xc}/v14_spec_cache" if _resolved_xc else None)
        )
    )
    print(f"  spec_cache_dir={_resolved_spec!r} (#80 whole-movie |STFT| cache; "
          f"paid once, then memmap-sliced)")
    print(f"  spec_only={args.spec_only} trial_durations={args.trial_durations!r} "
          f"(h5-free deploy: serve clips + durations off the spec cache, no raw h5)")

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

        phases = _build_v14_chain(args)
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
    # <=0 → None = epoch-budget escape (see --ssl-max-steps help / chain path).
    _ssl_steps = (
        args.ssl_max_steps if (args.ssl_max_steps and args.ssl_max_steps > 0)
        else None
    )
    single_max_steps = _ssl_steps if args.phase in (1, 3) else None
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
    # Cap the final test pass on SSL/distill phases only; P4's test metric is
    # the downstream leaderboard number, so it stays uncapped.
    single_limit_test = (
        args.ssl_limit_test_batches
        if args.phase in (1, 3)
        and args.ssl_limit_test_batches
        and args.ssl_limit_test_batches > 0
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
        f"limit_test_batches={single_limit_test} "
        f"collapse_guard={(args.phase != 4) and args.collapse_guard} "
        f"early_stopping_patience={single_p4_patience}"
    )
    # --electrode-set auto → 'lite' for the P4 eval cell, 'all' otherwise. The
    # --chain path ignores this (P4 hardcoded 'lite', SSL 'all'); only standalone
    # single-phase builds read it (e.g. a --phase 4 --resume-from leaderboard cell).
    single_electrode_set = (
        ("lite" if args.phase == 4 else "all")
        if args.electrode_set == "auto"
        else args.electrode_set
    )
    print(f"  electrode_set={single_electrode_set} (--electrode-set {args.electrode_set})")
    xp = build_v14_experiment(
        **_common_build_kwargs(args),
        clip_len=args.clip_len,
        electrode_set=single_electrode_set,
        neural_lag_s=args.neural_lag_s,
        max_steps=single_max_steps,
        val_check_interval=single_val_check,
        limit_val_batches=single_limit_val,
        limit_test_batches=single_limit_test,
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
        fold_index=args.fold_index,
        pretrained_ckpt=args.resume_from,
        snapshot_ckpt_to=args.snapshot_ckpt_to,
        jepa_phase=args.jepa_phase,
        parcel_lr_scale=args.parcel_lr_scale,
    )
    if args.cache_only:
        # Build the front-end spec cache then EXIT before the trainer (no GPU).
        # Drives the exact study.run() -> segmenter.apply() -> dataset.prepare()
        # path Data.build() uses, so the materialized whole-movie |STFT| memmap
        # is byte-identical to what the real run memmap-slices. dataset.prepare()
        # fans out per session present in the (subset) corpus and, for the 2STFT
        # frontend, builds BOTH band caches (low + high spec_cache_dir subdirs).
        import time as _time

        data = xp.data
        t0 = _time.time()
        print("[cache-only] building spec cache "
              f"(session_index={args.cache_session_index}) ...")
        events = data.study.run()
        dataset = data.segmenter.apply(events)
        dataset.prepare()
        print(f"[cache-only] DONE in {_time.time() - t0:.1f}s — "
              "spec cache materialized; exiting before trainer (no GPU used).")
        return 0
    if args.probe_bench_ckpt is not None:
        # Offline probe bench (pieces 1+3): no trainer. The experiment is built the
        # same way a real run is (same data chain / model config), so the 1 s probe
        # dataset and the 5s->1s model load are byte-faithful to production.
        import os as _os

        from speech_decoding.experiments.offline_probe_bench import run_probe_bench

        out = args.probe_bench_out or _os.path.join(
            _os.path.dirname(args.probe_bench_ckpt), "probe_bench.json"
        )
        _pieces = set(args.probe_bench_pieces.split(","))
        _taps = tuple(t.strip() for t in args.probe_bench_taps.split(",") if t.strip())
        run_probe_bench(
            xp, ckpt_path=args.probe_bench_ckpt, out_path=out,
            clip_len_s=1.0, max_iter=args.probe_bench_max_iter,
            do_ridge="ridge" in _pieces, do_headtohead="headtohead" in _pieces,
            taps=_taps,
        )
        return 0
    if args.gns_probe:
        # Gradient-noise-scale / critical-batch diagnostic. xp is built exactly as
        # a real run (same arch/cache/group_by_session sampler), so the measured
        # B_crit governs the production step. No trainer; 1 GPU.
        import os as _os

        from speech_decoding.experiments.gns_critical_batch import run_gns_probe

        out = args.gns_out or _os.path.join(_os.getcwd(), "gns_critical_batch.json")
        run_gns_probe(
            xp, out_path=out, ckpt_path=args.gns_ckpt,
            n_accum=args.gns_accum, rounds=args.gns_rounds,
        )
        return 0
    if args.lite_baseline_out is not None:
        # Piece 4: raw |STFT| logistic baseline on the Neuroprobe-Lite eval cells.
        # Model-free (no forward) — runs on a CPU scavenger node. Reuses the lite
        # Data's study + 3STFT segmenter; the inverse firewall in build_lite_eval_cells
        # asserts every materialized cell is a BT_LITE_SESSIONS member.
        import json as _json
        import os as _os

        from speech_decoding.experiments.lite_eval_raw_baseline import (
            build_lite_eval_cells,
            run_lite_eval_raw_baseline,
        )
        from speech_decoding.experiments.online_probe_dataset import N_CAP, PROBE_TASKS

        print("[lite-baseline] materializing Neuroprobe-Lite eval cells ...")
        cells = build_lite_eval_cells(xp.data, n_cap=N_CAP, tasks=PROBE_TASKS)
        print(f"[lite-baseline] materialized {len(cells)} cells: {sorted(cells)}")
        result = run_lite_eval_raw_baseline(
            cells, tasks=PROBE_TASKS, max_iter=args.lite_baseline_max_iter
        )
        out_dir = _os.path.dirname(args.lite_baseline_out)
        if out_dir:
            _os.makedirs(out_dir, exist_ok=True)
        with open(args.lite_baseline_out, "w") as f:
            _json.dump(result, f, indent=2)
        print(f"[lite-baseline] wrote {args.lite_baseline_out}")
        print(f"[lite-baseline] metrics: {result['metrics']}")
        return 0
    result = xp.run()
    print(f"V14 dispatch result: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
