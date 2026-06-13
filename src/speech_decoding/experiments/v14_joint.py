"""B29 Item 1: single joint SSL phase (replaces split P1 + P2).

Lock memos: ``memory/project_v14_b29_joint_default_2026_05_27.md`` §Item 1
(joint phase), as amended by
``memory/project_v14_b31_vjepa2_canonical_loss_2026_05_28.md`` (B31, the
current loss-surface winner).

Per the 5/27 PM-late lock, v14's pretraining collapses to a single SSL
phase. B31 (5/28 PM) then trimmed the loss surface to the V-JEPA-2-canonical
2-term default. (B36 2026-06-01 removed the soft ``λ_anat`` anatomy-bias gate
entirely — the hard per-parcel pool needs none; the B29-Item-12 bullet below
is retained as lineage. B36 also re-stages P1/P2; that orchestration change is
WS-E, not yet wired here.) The old split P1/P2 path is preserved as the
``R-keep-phase-split`` sister via ``V14Experiment``; this module is the
*joint-by-default* surface.

Lineage of the loss form:

  * B19 (5/24): 5-term lock at ``W_PRE_FRAME / W_MID_SLOT / W_POST_FRAME /
    W_POST_UTTERANCE / W_DKOLEO_M4 = 1, 1, 1, 1, 0.1``.
  * B25 (5/27 AM): Smooth-L1 β=1.0 across SSL prediction terms.
  * B26 (5/27 PM): pure L1 (V-JEPA-2 §2.1 Eq 1 canonical) — Smooth-L1
    citation was wrong. EMA τ=0.99925 fixed. Teacher full-input contract.
  * B27 (5/27 PM-late): DROP context loss + λ_ctx warmup; keep B26's
    pure L1 + fixed EMA + full-input teacher.
  * B28 (5/27 PM-late): DKoleo @ M4 demoted from default → 3 sisters;
    cross-attn count 2 @ {0, 3} → 1 @ {0}; anatomy-bias linear warmup
    over last 25% of P1 ∪ first 25% of P2.
  * B29 Item 1 (5/27 PM-late): P1 + P2 collapse → single joint phase.
  * B29 Item 12 (5/27 PM-late): replace step-time anatomy-bias warmup
    with per-clip ``λ_anat`` gate from metadata.
  * B31 (5/28 PM): V-JEPA-2-canonical 2-term default. DROP ``L_mid_slot``
    @ M3 and ``L_post_utterance`` @ M4-PMA from the default surface; they
    return only via the ``loss_variant`` sisters (``b31_plus_m3`` /
    ``b31_plus_utt`` / ``b31_plus_both``). PMA is trained only at P3
    (Whisper distillation), frozen at P4.

  * B36 (6/01, [[project_v14_b36_perparcel_pool_structured_jepa_2026_06_01]]):
    the inert B31 2-term self-distill (full input both sides, no mask, no
    predictor) is REPLACED by real paradigm-B masked JEPA, staged into two
    phases with exactly one term each (B9):

      P1  L = L1(predictor(masked freq×time), sg teacher M2[masked])    (B)
      P2  L = L1(predictor(masked parcel×time), sg teacher M4[masked])  (B)

    both **pure L1** (V-JEPA-2 §2.1 Eq 1); the target is the EMA teacher's
    own terminal LayerNorm (``frontend_ln`` at M2 / ``encoder_ln`` at M4),
    NO separate LN head (B6). EMA τ fixed at 0.99925; teacher always sees
    full input (B7). The live composition runs through
    :mod:`speech_decoding.ssl.masked_jepa`; the aggregator multi-term path
    is retired from the default (its ``b31_plus_*`` sisters are quarantined).
"""

from __future__ import annotations

from typing import ClassVar, Literal

import pydantic
import torch
from torch import Tensor

from speech_decoding.experiments.v14_experiment import V14Experiment
from speech_decoding.ssl.aggregator import LossVariant


JOINT_PHASE: Literal["joint_b29"] = "joint_b29"
"""Dispatch-side tag for the joint mode (B29 Item 1).

This is the ``phase_mode`` value the CLI emits **for the B29-joint
falsifier sister (R-joint-ssl)**, NOT the ``phase`` field on
:class:`V14Experiment`. As of the B36 relabel (2026-06-03) the default
recorded regime is ``split_p1_p2`` (see ``dispatch_v14.DEFAULT_PHASE_MODE``);
``joint_b29`` is no longer the dispatch default. The joint experiment
still runs at the parent class's canonical first phase (``phase == 1``)
and the dispatch wraps that single phase under the ``JOINT_PHASE`` label
so the trainer's checkpoint cadence and logging match the P1 ∪ P2
collapse contract.
"""

JOINT_PHASE_VALUE: Literal[1] = 1
"""Single-value ``phase`` that :class:`V14JointExperiment` is pinned to.

B29 Item 1 collapses split P1+P2 into one phase; the parent's pydantic
``phase`` field stays integer-typed (so the broader Experiment hierarchy
doesn't need a schema change), but the joint subclass refuses every
value except this one.
"""


def v14_joint_l1_loss(
    pred: Tensor, target: Tensor, *, reduction: str = "mean",
) -> Tensor:
    """Per-term L1 loss matching V-JEPA-2 §2.1 Eq 1.

    B26 (5/27 PM) corrected the Smooth-L1 citation: V-JEPA 2 / 2.1 /
    data2vec-2.0 §3.1 are pure L1, not Smooth-L1. B27 (same day, late)
    kept B26's loss form decision.

    Reductions: ``"mean"`` (default, term-level averaging) or ``"none"``
    (per-element). ``"sum"`` is also accepted to match ``F.l1_loss``.
    """
    if reduction not in ("mean", "none", "sum"):
        raise ValueError(
            f"reduction must be one of 'mean', 'none', 'sum'; got {reduction!r}"
        )
    return torch.nn.functional.l1_loss(pred, target, reduction=reduction)


LatentValidOverride = Literal["support", "all_true", "parcels_supervised"]
"""B30 sister selector for the latent-validity mask source.

``support`` (default): B30 single source of truth, ``support.sum(over
electrodes) > 0`` expanded across the M sub-slots — the same tensor the
encoder's latent-SA ``attn_mask`` consumes and the aggregator threads
through ``L_mid_slot`` / ``L_post_frame`` / ``L_post_utterance``.

Sister falsifiers (drift row ``B30-dispatch-sister-flags``):

* ``all_true`` (``R-item-12-all-true`` P0) — every slot active for every
  clip; falsifies the per-subject anatomy gating.
* ``parcels_supervised`` (``R-parcels-supervised-gating`` P0-retired-into-default)
  — pre-B30 per-subject ``parcels_supervised[subject]`` override; kept
  as falsifier so the retire-into-default move is empirically defended.

Only the ``support`` value is wired into the SSL trainer today; the
sister branches raise :class:`NotImplementedError` at construction
until the joint-phase aggregator-call path (#97 B2.2) lands.
"""

SaMaskMode = Literal["bidirectional", "key_only"]
"""B30 sister selector for the latent self-attention mask shape.

``bidirectional`` (default): inactive slots fully bypass latent SA —
neither keys nor queries. Encoder applies an ``attn_mask (L, L)`` over
the latent token sequence.

``key_only`` (``R-sa-key-only`` P1 falsifier): pre-B30 key-only
``key_padding_mask`` path; queries from inactive slots still emit
attention but receive zeroed contributions. Falsifies the move from
key-only to bidirectional masking.

Only ``bidirectional`` is wired into the encoder today; ``key_only``
raises :class:`NotImplementedError` at construction until the
encoder-side branch lands.
"""

SslMode = Literal["staged", "joint"]
"""B37 D7 SSL-objective mode selector.

``staged`` (default): the B36 single-term-per-phase path — one masked-JEPA
term (M2 front-end OR M4 parcel, picked by ``jepa_phase``) per experiment,
byte-identical to before B37 (the ``R-staged-ssl`` falsifier).

``joint`` (B37 D7/D9): ONE masked student forward produces BOTH the M2
front-end and M4 parcel taps under a single composite (band ∧ tube) mask,
scored by TWO predictors (M2 @ ``m2_predictor_depth``, M4 @
``m4_predictor_depth``) against one EMA full-input teacher pass —
``L = L_M2 + λ·L_M4``. Requires the freq-preserving mean-pool encoder
(``pool="mean"``); the module rejects ``joint`` on a cross-attn encoder.
"""


class V14JointExperiment(V14Experiment):
    """Joint-by-default v14 SSL experiment (B29 Item 1).

    Pinned to a single ``phase`` value (``"joint_b29"``-shaped). The
    sister ``R-keep-phase-split`` uses the parent :class:`V14Experiment`
    with explicit ``phase=1`` then ``phase=2``.

    The per-step loss runs through the staged B36 masked-JEPA objective
    (:mod:`speech_decoding.ssl.masked_jepa` — ``p1_frontend_m2_loss`` /
    ``p2_parcel_m4_loss``) inside
    :meth:`speech_decoding.experiments.v14_joint_module.V14JointBrainModule._step`,
    NOT the aggregator. ``loss_variant`` (default ``"b31_default"``) is
    retained only as a dispatch config-record + the quarantined
    ``b31_plus_*`` falsifier selector — the
    :func:`speech_decoding.ssl.aggregator.compute_v14_ssl_losses` multi-term
    path is off the default. (B36 2026-06-01 also removed the soft
    ``λ_anat`` routing bias — the hard per-parcel pool needs no per-clip
    gate, and the ``LambdaAnatExtractor`` it flowed through is gone.)

    B30 sister selectors (drift row ``B30-dispatch-sister-flags``,
    surfaced 2026-05-28 by the R12 wiring audit):

    * ``latent_valid_override`` picks the source of the ``latent_valid``
      gate threaded into
      :class:`speech_decoding.experiments.v14_joint_module.V14JointBrainModule`.
      Default ``"support"`` matches the B30 lock; ``"all_true"`` /
      ``"parcels_supervised"`` are :class:`NotImplementedError`-gated sister
      falsifiers.
    * ``sa_mask_mode`` picks the encoder's latent-SA mask shape.
      Default ``"bidirectional"`` matches the B30 lock; ``"key_only"``
      is a :class:`NotImplementedError`-gated sister falsifier.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # #82: P1/P2 SSL must never pretrain on a Neuroprobe off-limits eval
    # session. Enforced fail-closed in Experiment._train_and_test.
    enforces_pretrain_leakage_guard: ClassVar[bool] = True
    # Anti-starvation floor: P1/P2 span the full V14_PRETRAIN_SESSIONS cohort
    # (7 subjects {1,2,3,4,6,8,9} / 13 sessions).
    pretrain_cohort_floor: ClassVar[tuple[int, int] | None] = (7, 13)

    # B30 sister selectors. Pinned to the locked defaults; sister values
    # are accepted by the field but rejected at construction until the
    # respective runtime branch lands (B2.2 aggregator-call, encoder
    # latent-SA key-only path).
    latent_valid_override: LatentValidOverride = "support"
    sa_mask_mode: SaMaskMode = "bidirectional"

    # B36 WS-B masked-JEPA phase + mask config. ``jepa_phase`` selects which
    # single term is active (B9): ``"p1"`` = front-end M2 (paradigm B,
    # unconstrained-scope predictor), ``"p2"`` = parcel M4 (paradigm B,
    # cross-time predictor). The staged P1→P2 handoff is WS-E
    # (phase orchestration); this field is the surface WS-E threads.
    #
    # Mask config carries the 6/03 masking-lock defaults
    # (reports/b36_masking_handoff_2026_06_03.md): M2 ``bands`` held-out 0.50,
    # M4 ``tube`` 0.20 of covered parcels. ``predictor_scope`` couples with
    # ``m4_mask_type`` (tube↔cross_time, time_block↔co_temporal — the module
    # enforces it). ``m2_mask_type="random"`` is R-m2-random; ``m4_mask_type
    # ="time_block"`` is R-time-block; ``m{2,4}_mask_ratio`` back the
    # R-m2-ratio / R-m4-ratio sisters.
    jepa_phase: Literal["p1", "p2"] = "p1"
    m2_mask_type: Literal["bands", "random"] = "bands"
    m2_mask_ratio: float = 0.50
    m2_time_band_floor: int = 2
    m2_freq_band_floor: int = 1
    # 2STFT dual-band M2 mask block geometry (FE-2STFT; None → sampler default:
    # low one 3-wide freq block, high {3,3,2} time cols). width = contiguous
    # patches per block, nbands = #blocks; a band overrides only when BOTH set.
    m2_low_freq_width: int | None = None
    m2_low_freq_nbands: int | None = None
    m2_high_time_width: int | None = None
    m2_high_time_nbands: int | None = None
    m4_mask_type: Literal["tube", "time_block"] = "tube"
    m4_mask_ratio: float = 0.20
    m4_n_min_visible: int = 3
    predictor_scope: Literal["cross_time", "co_temporal"] = "cross_time"
    mask_seed: int = 0

    # EMA teacher momentum τ (B26 lock; V-JEPA 2 §2.4 fixed/no-ramp). Default
    # 0.99925 == ssl/ema.py P1_EMA_TAU/P2_EMA_TAU — so the field default is
    # byte-identical to the prior hardcoded ``_build_brain_module`` value. The
    # dispatch ``--ema-tau`` sweep flag threads an override here (R-ema-tau-{...});
    # the BrainModule re-validates 0 < τ < 1 at construction. The ema.py constants
    # are NOT mutated.
    ema_tau: float = pydantic.Field(default=0.99925, gt=0.0, lt=1.0)

    # B36 §5/§14 paradigm-B predictor sizing. Default 3 blocks @ hidden=128,
    # 4 heads (the locked P0 center). These knobs make the mandated P0 depth
    # sweep {2,3,4} AND the ``R-p1-predictor-large`` (16@512) underfit-recovery
    # sister launchable from dispatch
    # ([[project_v14_predictor_design_rope_lock_2026_06_04]]). The predictor is
    # discarded after SSL, so its size does not count toward the ≤30M readout.
    predictor_depth: int = pydantic.Field(default=3, ge=1)
    predictor_hidden: int = pydantic.Field(default=128, ge=1)
    predictor_n_heads: int = pydantic.Field(default=4, ge=1)

    # B37 D7/D9 joint SSL (2026-06-10). ``ssl_mode="joint"`` runs ONE masked
    # student forward producing BOTH the M2 + M4 taps under a composite
    # (band ∧ tube) mask, scored by TWO predictors against one EMA teacher pass
    # (``L = L_M2 + λ·L_M4``). It requires the freq-preserving mean-pool encoder
    # (``pool="mean"``) — validated below + re-checked in the BrainModule.
    # ``"staged"`` (default) preserves the B36 single-term-per-phase path
    # byte-identical (``R-staged-ssl``). ``m{2,4}_predictor_depth`` are the
    # per-tap predictor depths (D9: M2 @ 3, M4 @ 2; the shared
    # ``predictor_hidden`` / ``predictor_n_heads`` size both). ``lambda_m4`` is
    # the M4 term weight (D7, default 1.0). ``predictor_depth`` (above) stays
    # the STAGED single-predictor depth and is ignored under ``joint``.
    ssl_mode: SslMode = "staged"
    lambda_m4: float = pydantic.Field(default=1.0, ge=0.0)
    m2_predictor_depth: int = pydantic.Field(default=3, ge=1)
    m4_predictor_depth: int = pydantic.Field(default=2, ge=1)

    # Heteroscedastic / inverse-variance M4 (P2) loss weighting (joint only;
    # project_v14_heteroscedastic_ssl_loss). ``m4_precision_weight`` multiplies
    # each masked M4 cell's L1 by a DETACHED, mean-1-normalized precision
    # ``w = n_k^α / (σ²+ε)`` (n_k = parcel electrode count, σ = the mean|std
    # pool's per-cell std) so the gradient down-weights shaky low-n/high-σ parcel
    # targets. OFF by default (opt-in → the landed baseline is byte-identical);
    # requires ssl_mode="joint" + mean_pool_std (the σ source), re-validated in
    # the BrainModule. ``m4_precision_alpha`` is the ``n^α`` exponent (α=1 = raw
    # count; α<1 damps high-n since spatially-correlated electrodes give n_eff<n).
    # ``m4_precision_floor_pct`` sets the empirical-Bayes shrinkage prior σ²₀ = the
    # p{floor_pct} percentile of in-batch scored σ² (the load-bearing fix — without
    # it, the ~12.5% degenerate σ²≈0 cells swamp the loss and starve the
    # informative cells; measured 2026-06-13). ``m4_precision_cap`` bounds the
    # post-normalization weight (cheap insurance on top of the shrinkage floor).
    m4_precision_weight: bool = False
    m4_precision_alpha: float = pydantic.Field(default=1.0, ge=0.0)
    m4_precision_floor_pct: float = pydantic.Field(default=25.0, ge=0.0, le=100.0)
    m4_precision_cap: float = pydantic.Field(default=10.0, ge=0.0)

    # B37 D8 parcel-side discriminative-LR scale (joint only). The parcel side
    # (+ both predictors) rides at ``base_lr · joint_parcel_lr_scale``; the
    # front-end rides at ``base_lr · frontend_lr_scale``. Both default 1.0 (no
    # discrimination). NOTE the two zeros are NOT symmetric: a 0.0 *front-end*
    # scale HARD-freezes the front-end (``requires_grad_(False)`` + dropped from
    # the optimizer), whereas a 0.0 *parcel* scale only sets the parcel group's
    # LR to 0 (no update, but the params + Adam state stay allocated). The hard
    # freeze is the R-p2-freeze-frontend ablation; a frozen parcel side has no
    # such falsifier, so it is left as a zero-LR group. Named ``joint_*`` to stay
    # distinct from the B33 P3-3b ``V14Phase3Experiment.parcel_lr_scale`` (a
    # different phase, same bare name) — dispatch wires this one only on the
    # joint phase via ``--joint-parcel-lr-scale``.
    joint_parcel_lr_scale: float = pydantic.Field(default=1.0, ge=0.0, le=1.0)

    # #94 predictor context+query drop (V-JEPA-2 visible-only predictor). When
    # True the predictor gathers ONLY the visible context cells + ONLY the real
    # masked query slots (pad-to-per-batch-max) instead of feeding the full grid
    # and key-padding the rest. BIT-identical to the dense path (padded slots are
    # key-masked out → exp(NEG_INF)=0), a pure FLOP cut on the predictor's
    # attention/FFN at high mask ratio. Default OFF (dense path). Covers BOTH P1
    # and P2 (one predictor per joint phase).
    ragged_predictor: bool = pydantic.Field(default=False)

    # B36 WS-E E2 LR-staging hyperparameter (P2 only). The front-end was
    # pretrained in P1; in P2 it rides at ``base_lr · frontend_lr_scale`` while
    # the fresh parcel side + predictor get full base LR. Default 0.1 (base/10).
    # Falsifiers: 0.0 = R-p2-freeze-frontend (front-end frozen, parcel-only);
    # 0.2 = R-p2-frontend-lr-5 (base/5). No effect under jepa_phase="p1" (the
    # front-end is the only trained group there).
    frontend_lr_scale: float = pydantic.Field(default=0.1, ge=0.0, le=1.0)

    # MON-TEACHER-FEATURE-RANK thresholds (#74), joint-only (P1/P2 run RankMe;
    # P3/P4 do not). None → the canonical teacher_rank defaults (0.5 warn / 0.25
    # alarm); the module resolves + validates 0 < alarm < warn <= 1. Exposed so
    # the recalibration sweep can lower them toward the measured ~0.31 BT floor
    # without editing the source constants. The collapse-guard kills on the
    # per-step alarm/warn flags these set, so they are run-gating.
    rankme_warn_threshold: float | None = pydantic.Field(default=None, gt=0.0, le=1.0)
    rankme_alarm_threshold: float | None = pydantic.Field(default=None, gt=0.0, le=1.0)

    # B31 ``loss_variant`` field RETAINED for dispatch config-record
    # compatibility (``dispatch_v14`` passes it through), but B36 WS-B
    # replaced the multi-term aggregator path with the single-term masked
    # JEPA: only ``"b31_default"`` is wired. The ``b31_plus_*`` sisters
    # (R-add-m3-loss / R-add-utterance-loss / R-add-both) are quarantined —
    # they raise at construction (B9) until re-added on the masked-JEPA path.
    loss_variant: LossVariant = "b31_default"

    # NOTE: ``phase`` inherits from ``V14Experiment``; restricting to a
    # single-value ``Literal`` here would require redeclaring the field
    # on the subclass. Use a runtime check in ``model_post_init`` so the
    # field type stays consistent across the hierarchy.

    def model_post_init(self, _ctx) -> None:  # type: ignore[override]
        super().model_post_init(_ctx)
        # B30 sister selectors: default values match the B30 lock and are
        # always accepted. Non-default values flag the drift-table
        # B30-dispatch-sister-flags row as runtime-gated — they parse OK
        # at config time so the run-record YAML records the choice, but
        # the trainer / encoder branch hasn't landed yet.
        if self.latent_valid_override != "support":
            raise NotImplementedError(
                f"latent_valid_override={self.latent_valid_override!r} is a "
                "B30 sister falsifier (R-item-12-all-true / "
                "R-parcels-supervised-gating). Wiring blocked on #97 B2.2 "
                "(joint SSL aggregator call) — see "
                "docs/neuroprobe/v14_blockers.md row B30-dispatch-sister-flags."
            )
        if self.sa_mask_mode != "bidirectional":
            raise NotImplementedError(
                f"sa_mask_mode={self.sa_mask_mode!r} is the B30 sister "
                "falsifier R-sa-key-only. Wiring blocked on the encoder "
                "latent-SA key-only branch — see "
                "docs/neuroprobe/v14_blockers.md row B30-dispatch-sister-flags."
            )
        # 6/03 masking lock (STAGED only): M4 mask shape ↔ predictor scope must
        # couple, or the SSL task leaks (time_block+cross_time = H1). Fail at
        # config time (before dispatch), not just when the BrainModule is built.
        # The B37 JOINT M4 predictor is freq-carrying with full visible-context
        # attention (no cross-time restriction), so the coupling gate does NOT
        # apply there — this MUST match the module's ``ssl_mode == "staged"``
        # gate (v14_joint_module.py), or joint + ``--m4-mask-type time_block``
        # (a valid R-time-block joint sister) is wrongly rejected at config time.
        from speech_decoding.ssl.mask import validate_m4_coupling

        if self.ssl_mode == "staged":
            validate_m4_coupling(self.m4_mask_type, self.predictor_scope)
        # B9 (B36 WS-B): the masked-JEPA default is single-term per phase.
        # The B31 multi-term sisters (``b31_plus_m3`` / ``b31_plus_utt`` /
        # ``b31_plus_both``) are quarantined — re-adding the M3 / utterance
        # terms onto the masked path is future sister work (R-add-m3-loss /
        # R-add-utterance-loss). Until then, only ``b31_default`` is wired.
        if self.loss_variant != "b31_default":
            raise NotImplementedError(
                f"loss_variant={self.loss_variant!r} is a quarantined B36 "
                "WS-B sister (B9 single-term masked-JEPA default). The "
                "b31_plus_* multi-term aggregator path was retired with the "
                "masked-JEPA rewrite — see docs/neuroprobe/v14_blockers.md."
            )
        # B37 D7: joint SSL ⟺ the freq-preserving mean-pool encoder. The
        # coupling is BIDIRECTIONAL — enforce BOTH directions here so the
        # contract holds regardless of entry point (CLI or a programmatic
        # ``build_v14_experiment(pool="mean")`` that takes the staged default):
        #   * joint  ⇒ pool must be "mean"   (the 5-D M2/M4 taps need it), and
        #   * mean   ⇒ ssl_mode must be "joint" (the staged single-predictor
        #     path unpacks a 4-D cross_attn latent and would crash one batch
        #     into the forward on the 5-D mean-pool latent).
        # The BrainModule re-checks the joint⇒mean half; catching both here
        # gives the dispatch a clear pre-build error instead of a late crash.
        # ``brain_model_config`` may be absent on bare test fixtures (default
        # ssl_mode='staged' on a cross_attn encoder needs no pool check) — guard
        # the attribute access so those constructions stay valid.
        _brain_cfg = getattr(self, "brain_model_config", None)
        pool = getattr(_brain_cfg, "pool", None) if _brain_cfg is not None else None
        if self.ssl_mode == "joint" and pool is not None and pool != "mean":
            raise ValueError(
                "ssl_mode='joint' (B37 D7) is the freq-preserving mean-pool "
                f"path; it requires the encoder pool='mean', got pool={pool!r}. "
                "Launch with --pool mean (which defaults --ssl-mode to joint)."
            )
        if pool == "mean" and self.ssl_mode != "joint":
            raise ValueError(
                f"pool='mean' (B37) emits a freq-preserving 5-D parcel×freq×time "
                f"latent that only the joint SSL path consumes; got "
                f"ssl_mode={self.ssl_mode!r}. The staged single-predictor path "
                "unpacks a 4-D cross_attn latent and would crash one batch in. "
                "Use ssl_mode='joint' with pool='mean' (--pool mean defaults "
                "--ssl-mode to joint)."
            )
        # B29 Item 1: ``V14JointExperiment`` is *only* the joint phase,
        # which is pinned to ``phase == 1`` (the canonical collapsed
        # P1 ∪ P2). Split P1 / P2 must instantiate ``V14Experiment``
        # directly via the ``R-keep-phase-split`` sister.
        if self.phase != JOINT_PHASE_VALUE:
            raise ValueError(
                f"V14JointExperiment models the B29 joint phase only "
                f"(phase={JOINT_PHASE_VALUE}); got phase={self.phase!r}. "
                "Use V14Experiment directly for split P1/P2 "
                "(R-keep-phase-split sister)."
            )

    def _build_brain_module(self, train_loader):  # type: ignore[override]
        """B36 WS-B: build the masked-JEPA SSL Lightning module.

        Replaces the parent's CE-classifier ``BrainModule`` with
        :class:`speech_decoding.experiments.v14_joint_module.V14JointBrainModule`,
        which owns the EMA teacher + the student-only ``JepaPredictor`` and
        composes the single active masked-JEPA term for ``jepa_phase`` (B9):
        P1 front-end M2 or P2 parcel M4 — both paradigm B (exact parity:
        visible-only student + own predictor + EMA full-input teacher + L1;
        only the predictor's identity axis + scope differ). Targets are the
        EMA teacher's own terminal-LN taps (``frontend_ln`` / ``encoder_ln``)
        — there is no separate LN head (B6).

        The parent's ``brain_model_config.build(n_in_channels, n_outputs)``
        returns a :class:`V14ParcelPerceiverWithHead` (encoder + frozen
        parcel-PMA + flat head) — meant for Phase-4 supervised. The SSL
        trainer only needs the bare encoder; we extract ``.encoder`` and
        let the BrainModule construct its own predictor + EMA teacher mirror.

        ``n_outputs`` is informational here (SSL has no classification
        head); we pass ``1`` so the build call succeeds, then ignore the
        head.
        """
        from speech_decoding.experiments.v14_joint_module import V14JointBrainModule
        from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel

        batch = next(iter(train_loader))
        input_name = self._input_tensor_name()
        x = batch.data[input_name]
        head_model = self.brain_model_config.build(
            n_in_channels=int(x.shape[1]),
            n_outputs=1,
        )
        encoder = getattr(head_model, "encoder", None)
        if not isinstance(encoder, V14ParcelPerceiverModel):
            raise RuntimeError(
                "V14JointExperiment expected brain_model_config.build to "
                "return a model with an ``.encoder`` attribute of type "
                "V14ParcelPerceiverModel (per V14ParcelPerceiverWithHead); "
                f"got {type(head_model).__name__} without a recognized "
                "encoder slot."
            )
        return V14JointBrainModule(
            encoder=encoder,
            optim_config=self.optim,
            phase=self.jepa_phase,
            ssl_mode=self.ssl_mode,
            lambda_m4=self.lambda_m4,
            m2_predictor_depth=self.m2_predictor_depth,
            m4_predictor_depth=self.m4_predictor_depth,
            parcel_lr_scale=self.joint_parcel_lr_scale,
            m2_mask_type=self.m2_mask_type,
            m2_mask_ratio=self.m2_mask_ratio,
            m2_time_band_floor=self.m2_time_band_floor,
            m2_freq_band_floor=self.m2_freq_band_floor,
            m2_low_freq_width=self.m2_low_freq_width,
            m2_low_freq_nbands=self.m2_low_freq_nbands,
            m2_high_time_width=self.m2_high_time_width,
            m2_high_time_nbands=self.m2_high_time_nbands,
            m4_mask_type=self.m4_mask_type,
            m4_mask_ratio=self.m4_mask_ratio,
            m4_n_min_visible=self.m4_n_min_visible,
            predictor_scope=self.predictor_scope,
            predictor_depth=self.predictor_depth,
            predictor_hidden=self.predictor_hidden,
            predictor_n_heads=self.predictor_n_heads,
            ragged_predictor=self.ragged_predictor,
            mask_seed=self.mask_seed,
            ema_tau=self.ema_tau,
            loss_form="l1",
            m4_precision_weight=self.m4_precision_weight,
            m4_precision_alpha=self.m4_precision_alpha,
            m4_precision_floor_pct=self.m4_precision_floor_pct,
            m4_precision_cap=self.m4_precision_cap,
            latent_valid_override=self.latent_valid_override,
            sa_mask_mode=self.sa_mask_mode,
            frontend_lr_scale=self.frontend_lr_scale,
            wd_exclude_norms=self.wd_exclude_norms,
            rankme_warn_threshold=self.rankme_warn_threshold,
            rankme_alarm_threshold=self.rankme_alarm_threshold,
        )


__all__ = [
    "JOINT_PHASE",
    "JOINT_PHASE_VALUE",
    "LatentValidOverride",
    "SaMaskMode",
    "V14JointExperiment",
    "v14_joint_l1_loss",
]
