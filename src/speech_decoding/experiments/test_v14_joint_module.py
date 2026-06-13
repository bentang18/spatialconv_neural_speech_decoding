"""B36 WS-B tests for :class:`V14JointBrainModule` (masked-JEPA SSL).

Covers the module-level masked-JEPA contract (the encoder-level B1/B2/B5/B8
unit tests live in ``models/test_v14_encoder.py``):

* Construction: EMA teacher is a frozen mirror; the student bundle is
  encoder-only (no ``ln_frame`` / ``ln_mid`` / ``ln_utt`` / PMA heads); a
  student-only :class:`JepaPredictor` is built and is NOT EMA-mirrored.
* B30 sister-flag runtime gates + invalid ``phase`` raise at construction.
* P1 (``phase="p1"``, paradigm B): ``_step`` returns a single-term
  ``MaskedJepaBreakdown(phase="p1")``; gradient reaches the front-end
  (``frontend_ln``) AND the predictor, but NOT the terminal ``encoder_ln``.
* P2 (``phase="p2"``, paradigm B): single-term ``MaskedJepaBreakdown(
  phase="p2")``; gradient reaches both the encoder and the predictor.
  P1/P2 are EXACT-PARITY paradigm B — only the predictor's attention scope
  differs ([[project_v14_predictor_design_rope_lock_2026_06_04]]).
* B6: empty mask → exact-0 total (no NaN); target is detached (teacher
  accumulates no grad); the loss is L1, not MSE.
* B7: the EMA teacher always encodes the FULL input — the guard fires if a
  False-containing visibility mask reaches it.
* B9: exactly ONE active loss term per phase; the retired multi-term
  aggregator helpers are not imported by the module.
* B26 EMA step τ=0.99925 fixed.
* 5/28 P0 monitors (coverage / RankMe / grad-spike) still wired.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_joint_module import (
    V14JointBrainModule,
    _V14StudentBundle,
)
from speech_decoding.models.v14_encoder import JepaPredictor, V14ParcelPerceiverModel
from speech_decoding.ssl.masked_jepa import MaskedJepaBreakdown


def _optim_config() -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3})


def _make_tiny_encoder(
    *,
    n_freq_bins: int = 4,
    n_time_bins: int = 8,
    k_parcels: int = 5,
    m_sub_slots: int = 1,
    d_model: int = 16,
    n_heads: int = 4,
    depth_self_attn: int = 1,
    n_token_blocks: int = 1,
    patch_kernel_freq: int = 2,
    patch_kernel_time: int = 2,
    cross_attn_positions=None,
) -> V14ParcelPerceiverModel:
    return V14ParcelPerceiverModel(
        n_freq_bins=n_freq_bins,
        n_time_bins=n_time_bins,
        k_parcels=k_parcels,
        m_sub_slots=m_sub_slots,
        d_model=d_model,
        n_heads=n_heads,
        depth_self_attn=depth_self_attn,
        n_token_blocks=n_token_blocks,
        patch_kernel_freq=patch_kernel_freq,
        patch_kernel_time=patch_kernel_time,
        cross_attn_positions=cross_attn_positions,
    )


def _make_synthetic_batch(
    *,
    B: int = 2,
    C: int = 5,
    T_bins: int = 8,
    F_bins: int = 4,
    K: int = 5,
) -> SimpleNamespace:
    torch.manual_seed(0)
    electrode_tokens = torch.randn(B, C, T_bins, F_bins)
    # One covered electrode per parcel (diagonal support) so all K parcels are
    # covered → latent_valid is non-empty everywhere AND the locked M4 tube
    # default (0.20 of covered, n_min_visible=3) masks exactly 1 of 5 parcels
    # while keeping ≥3 visible — exercising the real default in P2.
    support = torch.zeros(B, C, K)
    for i in range(min(C, K)):
        support[:, i, i] = 1.0
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    data = {
        "electrode_tokens": electrode_tokens,
        "support": support,
        "valid_mask": valid_mask,
    }
    return SimpleNamespace(data=data)


def _make_module(
    encoder=None, *, phase: str = "p1", **kwargs,
) -> V14JointBrainModule:
    if encoder is None:
        encoder = _make_tiny_encoder()
    return V14JointBrainModule(
        encoder=encoder,
        optim_config=_optim_config(),
        phase=phase,  # type: ignore[arg-type]
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_frozen_teacher_and_encoder_only_student() -> None:
    module = _make_module()
    for p in module.teacher.parameters():
        assert p.requires_grad is False
    # B6/B36 §4: the student bundle is encoder-only — no LN/PMA heads.
    assert isinstance(module.student, _V14StudentBundle)
    assert hasattr(module.student, "encoder")
    for dead in ("ln_frame", "ln_mid", "ln_utt", "pma"):
        assert not hasattr(module.student, dead), dead


def test_predictor_is_jepa_predictor_and_not_ema_mirrored() -> None:
    """The predictor is student-only — V-JEPA predictors are never part of
    the teacher. The EMA mirror deepcopies only the student BUNDLE (encoder)."""
    module = _make_module()
    assert isinstance(module.predictor, JepaPredictor)
    # The teacher mirrors the bundle (which holds only the encoder); it must
    # NOT carry a copy of the predictor.
    assert not hasattr(module.teacher.model, "predictor")


def test_predictor_sizing_is_a_dispatch_reachable_knob() -> None:
    """B36 §5/§14: the predictor's depth/hidden/heads are config knobs so the P0
    depth sweep {2, 3, 4} and ``R-p1-predictor-large`` (16@512) are launchable.
    The default is the locked 3@128/4-head center; overrides flow through to the
    built :class:`JepaPredictor`. (The defaults are bit-identical to the prior
    hard-coded build — see the default-case asserts.)"""
    default = _make_module(phase="p1")
    assert default.predictor.depth == 3
    assert default.predictor.hidden == 128
    assert len(default.predictor.blocks) == 3

    large = _make_module(
        phase="p1", predictor_depth=16, predictor_hidden=512, predictor_n_heads=8,
    )
    assert large.predictor.depth == 16
    assert large.predictor.hidden == 512
    assert len(large.predictor.blocks) == 16

    # P0 depth-sweep endpoints construct.
    for depth in (2, 4):
        m = _make_module(phase="p1", predictor_depth=depth)
        assert len(m.predictor.blocks) == depth


def test_experiment_exposes_predictor_sizing_fields() -> None:
    """The sizing knobs are reachable from the dispatch surface
    (:class:`V14JointExperiment`), with the locked 3@128/4-head defaults — so
    the depth sweep / R-p1-predictor-large are launchable without a code edit."""
    from speech_decoding.experiments.v14_joint import V14JointExperiment

    fields = V14JointExperiment.model_fields
    assert fields["predictor_depth"].default == 3
    assert fields["predictor_hidden"].default == 128
    assert fields["predictor_n_heads"].default == 4


# ---------------------------------------------------------------------------
# MON-TEACHER-FEATURE-RANK thresholds (task #74) — CLI-exposed, defaults locked
# ---------------------------------------------------------------------------
def test_rankme_thresholds_default_resolve_and_validate() -> None:
    """P1 (M2 front-end probe): None → the canonical 0.5/0.25 (single-sourced
    in teacher_rank.py); overrides are stored; a bad order or out-of-range
    value raises at construction (not at step 0)."""
    from speech_decoding.experiments.monitors import (
        RANKME_NORMALISED_ALARM,
        RANKME_NORMALISED_WARN,
    )

    m = _make_module()  # defaults (phase="p1")
    assert m._rankme_warn_threshold == RANKME_NORMALISED_WARN
    assert m._rankme_alarm_threshold == RANKME_NORMALISED_ALARM

    m2 = _make_module(rankme_warn_threshold=0.4, rankme_alarm_threshold=0.3)
    assert m2._rankme_warn_threshold == 0.4
    assert m2._rankme_alarm_threshold == 0.3

    # alarm >= warn is rejected.
    with pytest.raises(ValueError, match="rankme_alarm"):
        _make_module(rankme_warn_threshold=0.3, rankme_alarm_threshold=0.4)
    # alarm must be > 0.
    with pytest.raises(ValueError, match="rankme_alarm"):
        _make_module(rankme_warn_threshold=0.5, rankme_alarm_threshold=0.0)


def test_rankme_thresholds_partial_override() -> None:
    """Only one flag set → the other resolves to its canonical default, and the
    0<alarm<warn<=1 check runs AFTER resolution (the real sweep surface: a sweep
    lowers warn alone toward the ~0.31 floor while alarm stays 0.25)."""
    from speech_decoding.experiments.monitors import (
        RANKME_NORMALISED_ALARM,
        RANKME_NORMALISED_WARN,
    )

    # warn-only: alarm falls back to 0.25; 0.25 < 0.4 holds → constructs.
    m = _make_module(rankme_warn_threshold=0.4)
    assert m._rankme_warn_threshold == 0.4
    assert m._rankme_alarm_threshold == RANKME_NORMALISED_ALARM
    # alarm-only: warn falls back to 0.5; 0.2 < 0.5 holds → constructs.
    m2 = _make_module(rankme_alarm_threshold=0.2)
    assert m2._rankme_alarm_threshold == 0.2
    assert m2._rankme_warn_threshold == RANKME_NORMALISED_WARN
    # warn-only BELOW the default alarm → post-resolution order is violated
    # (warn=0.2 < alarm=0.25) and construction must raise, not silently pass.
    with pytest.raises(ValueError, match="rankme_alarm"):
        _make_module(rankme_warn_threshold=0.2)


def test_rankme_thresholds_phase_keyed_default() -> None:
    """The canonical default is phase-keyed: P2 runs the M4 parcel-token probe,
    whose effective rank ≈ active-parcel count → floor ~0.05, so an unset P2
    resolves to the empirical M4 band (0.04/0.02), NOT the M2 |STFT| band
    (0.5/0.25) that false-positives on M4. P1/P3/P4 keep the M2 band. An
    explicit override still wins for either phase."""
    from speech_decoding.experiments.monitors import (
        RANKME_M4_NORMALISED_ALARM,
        RANKME_M4_NORMALISED_WARN,
        RANKME_NORMALISED_ALARM,
        RANKME_NORMALISED_WARN,
    )

    # P2 unset → M4 band, well below the M4 ~0.05 floor.
    p2 = _make_module(phase="p2")
    assert p2._rankme_warn_threshold == RANKME_M4_NORMALISED_WARN == 0.04
    assert p2._rankme_alarm_threshold == RANKME_M4_NORMALISED_ALARM == 0.02
    # P1 unset → M2 band (regression guard: phase keying didn't leak to P1).
    p1 = _make_module(phase="p1")
    assert p1._rankme_warn_threshold == RANKME_NORMALISED_WARN
    assert p1._rankme_alarm_threshold == RANKME_NORMALISED_ALARM
    # Explicit override beats the phase default on P2 (the manual sweep lever).
    p2o = _make_module(phase="p2", rankme_warn_threshold=0.3,
                       rankme_alarm_threshold=0.1)
    assert p2o._rankme_warn_threshold == 0.3
    assert p2o._rankme_alarm_threshold == 0.1


def test_rankme_thresholds_forwarded_to_both_monitor_call_sites(monkeypatch) -> None:
    """The stored thresholds reach BOTH RankMe probes: the M4 teacher-rank path
    (P2 health) and the M2 front-end path (P1 health). Spying on the monitor is
    the strongest proof the call sites pass them (not the module defaults)."""
    import speech_decoding.experiments.v14_joint_module as jm

    captured: list[tuple] = []

    def _spy(features, *, valid_mask=None, warn_threshold=None, alarm_threshold=None):
        captured.append((warn_threshold, alarm_threshold))
        return SimpleNamespace(
            rankme=8.0, rankme_normalised=0.5, n_samples=10, d_feature=8,
            is_warn=False, is_alarm=False,
        )

    monkeypatch.setattr(jm, "teacher_rank_monitor", _spy)

    m = _make_module(rankme_warn_threshold=0.45, rankme_alarm_threshold=0.35)
    m.log = lambda *a, **k: None  # type: ignore[method-assign]  # _log_rankme uses self.log

    # M4 teacher-rank path: (B, L, T, d) + (B, L) latent_valid.
    m._run_teacher_rank_monitor(
        teacher_m4=torch.randn(2, 5, 3, 8),
        latent_valid=torch.ones(2, 5, dtype=torch.bool),
        step_name="train",
    )
    # M2 front-end path: (B, C, F_p, T_p, d) + (B, C) valid_mask.
    m._run_frontend_rank_monitor(
        teacher_m2=torch.randn(2, 5, 2, 3, 8),
        valid_mask=torch.ones(2, 5, dtype=torch.bool),
        step_name="train",
    )

    assert captured == [(0.45, 0.35), (0.45, 0.35)]


@pytest.mark.parametrize("dual_band", [True, False])
def test_frontend_rank_monitor_logs_and_masks_parcels_in_joint_mode(
    dual_band, monkeypatch
) -> None:
    """Front-end RankMe must (a) reach the monitor for BOTH the 2STFT dual-band
    4-D tap ``(B, K, S, d)`` — the prior ``dim() != 5`` guard skipped it
    wholesale, so 2STFT never logged a front-end rank — and the B37 joint 5-D
    tap ``(B, K, F_p, T_p, d)``, and (b) gate axis 1 by parcel coverage
    ``latent_valid (B, K)``, dropping invalid parcels' rows. The mask keys off
    the actual axis-1 length, so it applies in joint mode (parcel coverage)
    instead of silently requiring the electrode shape ``(B, C)``."""
    import speech_decoding.experiments.v14_joint_module as jm

    captured: list = []

    def _spy(features, **_kw):
        captured.append(features)
        return SimpleNamespace(
            rankme=8.0, rankme_normalised=0.5, n_samples=int(features.shape[0]),
            d_feature=int(features.shape[1]), is_warn=False, is_alarm=False,
        )

    monkeypatch.setattr(jm, "teacher_rank_monitor", _spy)

    m = _make_module()
    logged: dict[str, float] = {}
    m.log = lambda key, value, **_kw: logged.update(  # type: ignore[method-assign]
        {key: float(value.detach() if hasattr(value, "detach") else value)})

    B, K, d = 2, 4, 8
    valid = torch.ones(B, K, dtype=torch.bool)
    valid[0, 0] = False  # one invalid parcel in batch 0 → its rows must drop
    if dual_band:
        cells_per_parcel = 6  # S
        tap = torch.randn(B, K, cells_per_parcel, d)
    else:
        F_p, T_p = 2, 3
        cells_per_parcel = F_p * T_p
        tap = torch.randn(B, K, F_p, T_p, d)

    m._run_frontend_rank_monitor(
        teacher_m2=tap, valid_mask=valid, step_name="train"
    )

    # (a) it logged for BOTH tap ranks (the 4-D dual-band tap no longer skipped).
    assert any(k.startswith("train_mon_frontend_rankme") for k in logged)
    assert len(captured) == 1
    # (b) exactly one parcel-worth of cells was removed by the parcel mask
    # (B*K cells total, one invalid → that parcel's cells_per_parcel rows gone).
    assert captured[0].shape[0] == (B * K - 1) * cells_per_parcel
    assert captured[0].shape[1] == d


def _stub_dual_band_samplers(monkeypatch, *, grid):
    """Stub the two dual-band mask samplers and capture the M2 sampler's kwargs.

    Returns the ``captured`` dict (filled when ``_sample_dual_band_mask`` runs).
    ``grid = (F_p_low, T_low_p, F_p_high, T_high_p)``.
    """
    import speech_decoding.experiments.v14_joint_module as jm

    F_p_low, T_low_p, F_p_high, T_high_p = grid
    captured: dict = {}

    def _spy_m2(B, K, *, F_p_low, T_low_p, F_p_high, T_high_p,
                generator, device=None, **kw):
        captured.update(kw)
        S = F_p_low * T_low_p + F_p_high * T_high_p
        return torch.zeros(B, K, S, dtype=torch.bool, device=device)

    def _spy_tube(support, *, n_time_patches, mask_ratio, n_min_visible, generator):
        B, _, K = support.shape
        return torch.zeros(B, K, n_time_patches, dtype=torch.bool), None

    monkeypatch.setattr(jm, "sample_m2_dual_band_mask", _spy_m2)
    monkeypatch.setattr(jm, "sample_parcel_tube_mask", _spy_tube)
    return captured


def test_dual_band_mask_knobs_forward_to_sampler(monkeypatch) -> None:
    """The 4 dual-band M2 block-geometry knobs map onto
    ``sample_m2_dual_band_mask``: low ``(width, nbands)`` →
    ``low_freq_floor=width`` + ``low_freq_frac=width*nbands/F_p_low`` (so the
    sampler's ``round(frac·F_p_low)//width`` lands exactly ``nbands`` blocks of
    ``width``); high ``(width, nbands)`` → ``high_time_widths=(width,)*nbands``
    (a fixed uniform width multiset). This is the only behavioral coupling the
    easier-masking 6e-4 run depends on."""
    grid = (7, 40, 3, 80)  # F_p_low, T_low_p, F_p_high, T_high_p (lr6e4 5s clip)
    captured = _stub_dual_band_samplers(monkeypatch, grid=grid)

    m = _make_module(
        m2_low_freq_width=2, m2_low_freq_nbands=1,
        m2_high_time_width=2, m2_high_time_nbands=21,
    )
    m.student.encoder.dual_band_grid_shape = lambda _x: grid  # type: ignore[assignment]

    B, C, K = 2, 5, m.student.encoder.k_parcels
    et_high = torch.randn(B, 1, 1, 1)
    support = torch.zeros(B, C, K)
    m._sample_dual_band_mask(electrode_tokens_high=et_high, support=support)

    assert captured["low_freq_floor"] == 2
    assert captured["low_freq_frac"] == pytest.approx(2 * 1 / grid[0])
    assert captured["high_time_widths"] == (2,) * 21


def test_dual_band_high_anchor_knobs_forward_to_sampler(monkeypatch) -> None:
    """The high-band anchor-dilate knobs map onto ``sample_m2_dual_band_mask``'s
    ``high_time_anchor_frac`` / ``high_time_anchor_width`` (the Ben 2026-06-13
    overlap-allowed high regime), and do NOT emit the disjoint multiset's
    ``high_time_widths``."""
    grid = (7, 40, 3, 80)
    captured = _stub_dual_band_samplers(monkeypatch, grid=grid)

    m = _make_module(
        m2_low_freq_width=2, m2_low_freq_nbands=1,
        m2_high_anchor_frac=0.30, m2_high_anchor_width=2,
    )
    m.student.encoder.dual_band_grid_shape = lambda _x: grid  # type: ignore[assignment]

    B, C, K = 2, 5, m.student.encoder.k_parcels
    m._sample_dual_band_mask(
        electrode_tokens_high=torch.randn(B, 1, 1, 1),
        support=torch.zeros(B, C, K),
    )
    assert captured["high_time_anchor_frac"] == 0.30
    assert captured["high_time_anchor_width"] == 2
    assert "high_time_widths" not in captured  # not the disjoint-multiset path
    # low band still forwards its freq tube
    assert captured["low_freq_floor"] == 2


def test_dual_band_mask_knobs_unset_keeps_sampler_defaults(monkeypatch) -> None:
    """Unset knobs (the default) pass NO geometry override to the sampler, so the
    dual-band path is byte-identical to the un-flagged version (sampler defaults:
    low one 3-wide freq tube; high {3,3,2} time cols)."""
    grid = (7, 40, 3, 80)
    captured = _stub_dual_band_samplers(monkeypatch, grid=grid)

    m = _make_module()  # no knobs
    m.student.encoder.dual_band_grid_shape = lambda _x: grid  # type: ignore[assignment]

    B, C, K = 2, 5, m.student.encoder.k_parcels
    m._sample_dual_band_mask(
        electrode_tokens_high=torch.randn(B, 1, 1, 1),
        support=torch.zeros(B, C, K),
    )
    for k in ("low_freq_floor", "low_freq_frac", "high_time_widths"):
        assert k not in captured, f"{k} should not be forwarded when unset"


def test_dual_band_mask_half_spec_keeps_sampler_default(monkeypatch) -> None:
    """A half-spec at the MODULE layer (width without nbands) is treated as
    'unset' for that band — the override only applies when BOTH are set, so a
    stray single knob can never silently half-configure the sampler. (Dispatch
    rejects the half-spec loudly before this; the module is the last-line
    defense if a module is built directly.)"""
    grid = (7, 40, 3, 80)
    captured = _stub_dual_band_samplers(monkeypatch, grid=grid)

    m = _make_module(m2_low_freq_width=2)  # nbands missing
    m.student.encoder.dual_band_grid_shape = lambda _x: grid  # type: ignore[assignment]

    B, C, K = 2, 5, m.student.encoder.k_parcels
    m._sample_dual_band_mask(
        electrode_tokens_high=torch.randn(B, 1, 1, 1),
        support=torch.zeros(B, C, K),
    )
    assert "low_freq_floor" not in captured
    assert "low_freq_frac" not in captured


def test_unset_p2_m4_probe_receives_empirical_band(monkeypatch) -> None:
    """End-to-end: an UNSET phase="p2" module forwards the empirical M4 band
    (0.04/0.02) all the way to the M4 probe call site — not the M2 0.5/0.25
    band that false-positived on M4. Closes the composition gap between the
    phase-keyed-default test and the forwarding test."""
    import speech_decoding.experiments.v14_joint_module as jm

    captured: list[tuple] = []

    def _spy(features, *, valid_mask=None, warn_threshold=None, alarm_threshold=None):
        captured.append((warn_threshold, alarm_threshold))
        return SimpleNamespace(
            rankme=12.0, rankme_normalised=0.047, n_samples=100, d_feature=256,
            is_warn=True, is_alarm=False,
        )

    monkeypatch.setattr(jm, "teacher_rank_monitor", _spy)

    m = _make_module(phase="p2")  # unset thresholds → M4 phase default
    m.log = lambda *a, **k: None  # type: ignore[method-assign]
    m._run_teacher_rank_monitor(
        teacher_m4=torch.randn(2, 5, 3, 8),
        latent_valid=torch.ones(2, 5, dtype=torch.bool),
        step_name="val",
    )
    assert captured == [(0.04, 0.02)]


# ---------------------------------------------------------------------------
# §7/B01 no-WD param-group split (task #40) — configure_optimizers integration
# ---------------------------------------------------------------------------
def _adamw_optim(weight_decay: float) -> LightningOptimizer:
    return LightningOptimizer(
        optimizer={"name": "AdamW", "lr": 1e-3,
                   "kwargs": {"weight_decay": weight_decay}}
    )


def _built_param_groups(module: V14JointBrainModule):
    out = module.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    return opt.param_groups


def test_configure_optimizers_no_wd_split_applied_at_positive_wd() -> None:
    """P1 wd>0 + wd_exclude_norms (default) → the built optimizer carries a
    weight_decay==0.0 group for the exempt params (biases / LN γβ / embeds) and
    a weight_decay==wd group for the matmul weights."""
    module = V14JointBrainModule(
        encoder=_make_tiny_encoder(), optim_config=_adamw_optim(0.1),
        phase="p1",  # type: ignore[arg-type]
    )
    groups = _built_param_groups(module)
    wds = {g["weight_decay"] for g in groups}
    assert 0.0 in wds and 0.1 in wds
    # Every exempt-by-id param sits in a wd==0.0 group; no decay param does.
    from speech_decoding.experiments.optim_param_groups import no_decay_param_ids
    exempt = no_decay_param_ids(module)
    optimized = {id(p) for g in groups for p in g["params"]}
    for g in groups:
        for p in g["params"]:
            if id(p) in exempt:
                assert g["weight_decay"] == 0.0
            else:
                assert g["weight_decay"] == 0.1
    # The split partitions, never drops: optimized set == the P1 param ids.
    assert optimized  # non-empty


def test_configure_optimizers_bit_identical_at_zero_wd() -> None:
    """wd=0 → no split (single inherited group), bit-identical to pre-#40."""
    module = V14JointBrainModule(
        encoder=_make_tiny_encoder(), optim_config=_adamw_optim(0.0),
        phase="p1",  # type: ignore[arg-type]
    )
    groups = _built_param_groups(module)
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.0


def test_configure_optimizers_no_split_when_exclude_off() -> None:
    """--no-wd-exclude-norms (wd_exclude_norms=False) decays ALL params: one
    group at the swept wd, even with wd>0 (the R-uniform-wd falsifier)."""
    module = V14JointBrainModule(
        encoder=_make_tiny_encoder(), optim_config=_adamw_optim(0.1),
        phase="p1", wd_exclude_norms=False,  # type: ignore[arg-type]
    )
    groups = _built_param_groups(module)
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.1


def test_configure_optimizers_p2_discriminative_lr_survives_split() -> None:
    """P2 returns two discriminative-LR groups (front-end @ base/10, parcel +
    predictor @ base). With wd>0 each splits into decay/no-decay, and BOTH halves
    keep their group's lr — the no-WD split must not flatten the discriminative
    schedule."""
    base_lr = 1e-3
    module = V14JointBrainModule(
        encoder=_make_tiny_encoder(), optim_config=_adamw_optim(0.1),
        phase="p2", frontend_lr_scale=0.1,  # type: ignore[arg-type]
    )
    groups = _built_param_groups(module)
    lrs = {round(g["lr"], 8) for g in groups}
    # Front-end group at base/10 and the parcel/predictor group at base both
    # present (each possibly split into a wd / wd=0 pair sharing the lr).
    assert round(base_lr * 0.1, 8) in lrs
    assert round(base_lr, 8) in lrs
    # Within each lr tier, an exempt param only ever lands in a wd==0.0 group.
    from speech_decoding.experiments.optim_param_groups import no_decay_param_ids
    exempt = no_decay_param_ids(module)
    for g in groups:
        for p in g["params"]:
            if id(p) in exempt:
                assert g["weight_decay"] == 0.0


def test_rejects_b30_sister_latent_valid_override() -> None:
    with pytest.raises(NotImplementedError, match="B30"):
        _make_module(latent_valid_override="all_true")


def test_rejects_b30_sister_sa_mask_mode() -> None:
    with pytest.raises(NotImplementedError, match="key-only"):
        _make_module(sa_mask_mode="key_only")


def test_rejects_unknown_phase() -> None:
    with pytest.raises(ValueError, match="phase"):
        _make_module(phase="p3")


# ---------------------------------------------------------------------------
# B5/B6/B9: P1 paradigm-B front-end masked JEPA
# ---------------------------------------------------------------------------


def test_p1_step_returns_single_term_p1_breakdown() -> None:
    module = _make_module(phase="p1")
    breakdown = module._step(_make_synthetic_batch().data)
    assert isinstance(breakdown, MaskedJepaBreakdown)
    assert breakdown.phase == "p1"
    assert breakdown.n_masked > 0
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total)
    assert float(breakdown.total.detach()) >= 0.0  # L1 is non-negative


def test_p1_grad_reaches_frontend_and_predictor_not_terminal_ln() -> None:
    """B36 §7 P1 grad-scope (paradigm B): the visible-only front-end produces
    the M2 context that feeds the separate predictor, so gradient reaches
    ``frontend_ln`` (front-end terminal LN) AND the predictor (``output_proj``)
    — exact parity with P2. The downstream pool / inter-parcel encoder
    (``encoder_ln``) is off the M2 loss path → NO grad (the loss is computed
    entirely at M2). The predictor-free paradigm-A self-distill here was the
    P1-collapse regression ([[project_v14_p1_predictor_paradigm_b_regression_2026_06_04]])."""
    module = _make_module(phase="p1")
    breakdown = module._step(_make_synthetic_batch().data)
    breakdown.total.backward()
    enc = module.student.encoder
    assert enc.frontend_ln.weight.grad is not None
    assert torch.isfinite(enc.frontend_ln.weight.grad).all()
    # Downstream of M2 → off the loss path → no grad.
    assert enc.encoder_ln.weight.grad is None
    # Paradigm B: the predictor IS on the P1 loss path.
    assert module.predictor.output_proj.weight.grad is not None
    assert torch.isfinite(module.predictor.output_proj.weight.grad).all()


# ---------------------------------------------------------------------------
# B6/B8/B9: P2 paradigm-B parcel masked JEPA
# ---------------------------------------------------------------------------


def test_p2_step_returns_single_term_p2_breakdown() -> None:
    module = _make_module(phase="p2")
    breakdown = module._step(_make_synthetic_batch().data)
    assert isinstance(breakdown, MaskedJepaBreakdown)
    assert breakdown.phase == "p2"
    assert breakdown.n_masked > 0
    assert breakdown.total.ndim == 0
    assert torch.isfinite(breakdown.total)
    assert float(breakdown.total.detach()) >= 0.0


def test_p2_grad_reaches_encoder_and_predictor() -> None:
    """P2 paradigm B: the visible-only encoder feeds the separate predictor,
    so gradient reaches both the encoder (``encoder_ln``) and the predictor
    (``output_proj``)."""
    module = _make_module(phase="p2")
    breakdown = module._step(_make_synthetic_batch().data)
    breakdown.total.backward()
    enc = module.student.encoder
    assert enc.encoder_ln.weight.grad is not None
    assert module.predictor.output_proj.weight.grad is not None
    assert torch.isfinite(module.predictor.output_proj.weight.grad).all()


# ---------------------------------------------------------------------------
# #94: predictor context+query drop (V-JEPA-2 visible-only predictor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("phase", ["p1", "p2"])
def test_ragged_predictor_step_loss_bit_identical(phase: str) -> None:
    """End-to-end: toggling ``ragged_predictor`` on the LIVE ``_step`` (real
    ``p1_frontend_m2_loss`` / ``p2_parcel_m4_loss`` call sites that pass
    ``query_valid``) yields a BIT-identical loss + masked count for BOTH phases.

    The mask is seeded by ``mask_seed + global_step`` (=0 with no trainer), and
    ``_step`` is forward-only (no weight mutation), so the two passes differ ONLY
    in whether the predictor gathers visible-context/real-query (ragged) or feeds
    the full grid key-padded (dense). The padded slots are masked out of
    attention either way (``exp(NEG_INF)`` → 0), so the prediction — and the L1
    loss — match up to ~1e-6 matmul reassociation (the #91/#93 standard)."""
    module = _make_module(phase=phase)
    batch = _make_synthetic_batch().data

    module.predictor.ragged_predictor = False
    dense = module._step(batch)
    module.predictor.ragged_predictor = True
    ragged = module._step(batch)

    assert ragged.n_masked == dense.n_masked > 0
    torch.testing.assert_close(ragged.total, dense.total, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("phase", ["p1", "p2"])
def test_ragged_predictor_step_grad_bit_identical(phase: str) -> None:
    """The ragged predictor must also produce a bit-identical GRADIENT into the
    predictor's terminal ``output_proj`` (the masked-position prediction is what
    backprops), confirming the drop is loss- AND grad-neutral. One module, one
    batch, one mask (seed+step=0): toggling the flag is the ONLY difference."""
    module = _make_module(phase=phase)
    batch = _make_synthetic_batch().data

    module.predictor.ragged_predictor = False
    module._step(batch).total.backward()
    g_dense = module.predictor.output_proj.weight.grad.clone()

    module.zero_grad(set_to_none=True)
    module.predictor.ragged_predictor = True
    module._step(batch).total.backward()
    g_ragged = module.predictor.output_proj.weight.grad

    torch.testing.assert_close(g_ragged, g_dense, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# B6: masked-empty exact 0, detached target, L1 form
# ---------------------------------------------------------------------------


@pytest.mark.must_pass_before_dispatch
def test_p1_empty_mask_gives_exact_zero_no_nan() -> None:
    """B6 masked-empty contract: ratio 0 → no masked cell → total is an
    exact 0 (graph-connected, no NaN)."""
    module = _make_module(phase="p1", m2_mask_ratio=0.0)
    breakdown = module._step(_make_synthetic_batch().data)
    assert breakdown.n_masked == 0
    assert float(breakdown.total.detach()) == 0.0
    assert torch.isfinite(breakdown.total)


@pytest.mark.must_pass_before_dispatch
def test_teacher_accumulates_no_grad_target_is_detached() -> None:
    """B6/B26: the teacher target is ``detach()``ed and the teacher forward
    runs under ``no_grad`` — no teacher parameter accumulates gradient."""
    module = _make_module(phase="p2")
    breakdown = module._step(_make_synthetic_batch().data)
    breakdown.total.backward()
    for p in module.teacher.parameters():
        assert p.grad is None


def test_loss_is_l1_not_mse() -> None:
    """B6: ``loss_form='mse'`` produces a strictly different scalar than the
    default L1 on the same (seeded) masked set — proves the default is L1."""
    batch = _make_synthetic_batch()
    enc = _make_tiny_encoder()
    l1 = _make_module(enc, phase="p1", loss_form="l1")._step(batch.data)
    # A fresh module with an identical encoder + the same mask seed, MSE form.
    enc2 = _make_tiny_encoder()
    enc2.load_state_dict(enc.state_dict())
    mse = _make_module(enc2, phase="p1", loss_form="mse")._step(batch.data)
    assert l1.n_masked == mse.n_masked > 0
    assert not torch.allclose(l1.total, mse.total)


def test_b6_l1_gradient_magnitude_constant_in_error() -> None:
    """B6 (canonical V-JEPA target-norm) — the masked loss *kernel* is pure L1,
    so ``d|p-t|/dp = sign(p-t)``: the per-element gradient magnitude is a
    constant ``1/numel`` regardless of the error scale. (MSE / Smooth-L1 grads
    scale with the error and would NOT be constant.)

    Under paradigm B (both phases) the loss is ``L1(predictor_output,
    sg teacher_target)`` — the masked STUDENT cells are padding-masked out of
    the predictor context, so the L1-form property lives on the gradient w.r.t.
    the predictor OUTPUT, scored by the shared ``_l1_or_zero`` kernel both
    ``p1_frontend_m2_loss`` and ``p2_parcel_m4_loss`` call. Probing that kernel
    is the phase-agnostic "gradient magnitude constant in error" check the B6
    TEST clause demands, stronger than the L1≠MSE scalar comparison above."""
    from speech_decoding.ssl.masked_jepa import _l1_or_zero

    n, d = 5, 4
    target = torch.zeros(n, d)  # detached teacher target stand-in

    grads = []
    for scale in (0.1, 1.0, 5.0):
        pred = torch.full((n, d), float(scale), requires_grad=True)
        _l1_or_zero(pred, target, "l1").backward()
        g = pred.grad.abs()
        # pred > target ⇒ sign = +1 ⇒ |grad| == 1/numel everywhere.
        expected = 1.0 / pred.numel()
        torch.testing.assert_close(g, torch.full_like(g, expected))
        grads.append(g)
    # The discriminator: L1's grad is identical across error scales (MSE's
    # would be 0.1× vs 5×). Constant ⇒ pure L1.
    torch.testing.assert_close(grads[0], grads[-1])


# ---------------------------------------------------------------------------
# B7: teacher full-input guard
# ---------------------------------------------------------------------------


def test_b7_teacher_full_input_guard_is_wired() -> None:
    """B7: the teacher forward must see full input. The module never threads
    a JEPA mask into the teacher; the guard fires if a False-containing
    visibility mask is passed (simulating a leak)."""
    from speech_decoding.ssl.ema import assert_teacher_full_input

    # The wired call: both masks None ⇒ vacuously passes (teacher full-input).
    assert_teacher_full_input(patch_mask=None, shaft_mask=None)
    # A leaked student mask → its visibility (~mask) has False entries → raise.
    token_mask = torch.zeros(2, 3, 2, 4, dtype=torch.bool)
    token_mask[0, 0, 0, 0] = True
    with pytest.raises(AssertionError, match="full-input"):
        assert_teacher_full_input(patch_mask=~token_mask)


def test_b7_step_does_not_pass_mask_to_teacher() -> None:
    """B7 integration: running ``_step`` (which calls the guard at the teacher
    call site) never raises — the teacher truly gets full input."""
    for phase in ("p1", "p2"):
        module = _make_module(phase=phase)
        breakdown = module._step(_make_synthetic_batch().data)
        assert torch.isfinite(breakdown.total)


def test_b7_teacher_forward_call_site_raises_on_leaked_mask() -> None:
    """B7 call-site wiring: ``_teacher_forward`` runs
    ``assert_teacher_full_input`` on the EXACT kwargs the teacher receives.
    Inject a partial ``token_mask`` (simulating a refactor that leaks the
    student mask into the teacher pass) and confirm the guard fires — this
    exercises the live ``_step`` call site, not just the helper in isolation,
    closing the 'guard is vacuously wired' gap."""
    module = _make_module(phase="p1")
    batch = _make_synthetic_batch()
    student_kwargs = module._extract_student_kwargs(batch.data)
    C, F_p, T_p = module.student.encoder.patch_grid_shape(
        student_kwargs["electrode_tokens"],
    )
    B = batch.data["electrode_tokens"].shape[0]
    leaked = torch.zeros(B, C, F_p, T_p, dtype=torch.bool)
    leaked[0, 0, 0, 0] = True  # one masked cell ⇒ ~leaked has a False entry
    teacher_kwargs = dict(student_kwargs, token_mask=leaked)
    with pytest.raises(AssertionError, match="full-input"):
        module._teacher_forward(teacher_kwargs)

    # A parcel-time leak fires the same tripwire.
    K = student_kwargs["support"].shape[-1]
    leaked_ptm = torch.zeros(B, K, T_p, dtype=torch.bool)
    leaked_ptm[0, 0, 0] = True
    with pytest.raises(AssertionError, match="full-input"):
        module._teacher_forward(dict(student_kwargs, parcel_time_mask=leaked_ptm))


def test_b7_teacher_forward_full_input_passes_and_returns_taps() -> None:
    """B7 live path: with no mask key in ``teacher_kwargs`` the guard passes
    and the teacher returns its tap dict. ``m2_only=True`` (the P1 path)
    returns just the M2 tap."""
    module = _make_module(phase="p1")
    student_kwargs = module._extract_student_kwargs(_make_synthetic_batch().data)
    taps = module._teacher_forward(dict(student_kwargs), m2_only=True)
    assert set(taps.keys()) == {"M2"}


# ---------------------------------------------------------------------------
# B9: exactly one term; retired multi-term path not imported
# ---------------------------------------------------------------------------


def test_b9_module_does_not_import_retired_aggregator_helpers() -> None:
    """B9: the retired multi-term aggregator surface is gone from the joint
    module's namespace (the masked-JEPA default is single-term)."""
    mod = importlib.import_module(
        "speech_decoding.experiments.v14_joint_module"
    )
    for dead in (
        "compute_v14_ssl_losses",
        "V14TotalLossBreakdown",
        "LossVariant",
        "_variant_wants_m3",
        "_variant_wants_utt",
        "_compose_l_pre_frame",
    ):
        assert not hasattr(mod, dead), dead


def test_b9_breakdown_exposes_exactly_one_scalar_term() -> None:
    """B9: the breakdown carries a single ``total`` scalar + its phase tag —
    there are no per-term sub-fields (l_mid_slot / l_post_utterance / ...)."""
    breakdown = _make_module(phase="p1")._step(_make_synthetic_batch().data)
    fields = set(vars(breakdown).keys())
    assert fields == {"total", "phase", "n_masked"}


def test_b9_layer_avg_with_instance_norm_retired_from_runtime() -> None:
    """B9: ``layer_avg_with_instance_norm`` (data2vec-2.0 / EAT layer-averaging)
    is explicitly named for quarantine. Under the canonical V-JEPA target-norm
    the target is the encoder's own terminal LN, so this helper builds NO live
    target — neither the joint module nor the masked-JEPA loss module imports
    or calls it (it survives only as the ``R-layer-avg-target`` sister + its
    own unit test)."""
    import inspect

    import speech_decoding.experiments.v14_joint_module as jm
    import speech_decoding.ssl.masked_jepa as mj

    for mod in (jm, mj):
        assert not hasattr(mod, "layer_avg_with_instance_norm"), mod.__name__
        assert "layer_avg_with_instance_norm(" not in inspect.getsource(mod), (
            f"{mod.__name__} must not CALL the retired data2vec helper"
        )


# ---------------------------------------------------------------------------
# B26 EMA + optimizer scope
# ---------------------------------------------------------------------------


def test_ema_step_updates_teacher_fixed_tau() -> None:
    """B26 lock: τ=0.99925 fixed; ``update_from`` pulls the teacher toward
    the (post-step) student. Exercised on the encoder's ``encoder_ln``."""
    module = _make_module()
    with torch.no_grad():
        module.student.encoder.encoder_ln.weight.fill_(2.0)
    pre = module.teacher.model.encoder.encoder_ln.weight.detach().clone()
    coeff = module.teacher.update_from(module.student)
    post = module.teacher.model.encoder.encoder_ln.weight.detach().clone()
    assert coeff == pytest.approx(0.99925)
    torch.testing.assert_close(post, 0.99925 * pre + 0.00075 * 2.0)


def test_ema_tau_override_changes_teacher_coefficient() -> None:
    """SSL-sweep knob: a non-default ``ema_tau`` reaches the EmaTeacher schedule
    so ``update_from`` blends at the swept τ — and the default reproduces the
    B26-locked 0.99925 byte-for-byte (the same EMA-step contract as
    ``test_ema_step_updates_teacher_fixed_tau``, parametrized by the new flag)."""
    # default: unchanged locked behaviour
    default_module = _make_module()
    assert default_module.teacher.update_from(default_module.student) == pytest.approx(
        0.99925
    )

    # override: the coefficient + the realized blend both move to the swept τ
    tau = 0.9999
    module = _make_module(ema_tau=tau)
    with torch.no_grad():
        module.student.encoder.encoder_ln.weight.fill_(2.0)
    pre = module.teacher.model.encoder.encoder_ln.weight.detach().clone()
    coeff = module.teacher.update_from(module.student)
    post = module.teacher.model.encoder.encoder_ln.weight.detach().clone()
    assert coeff == pytest.approx(tau)
    torch.testing.assert_close(post, tau * pre + (1.0 - tau) * 2.0)


def test_ema_tau_out_of_range_raises() -> None:
    """The module re-validates τ ∈ (0, 1) at construction (open interval), so a
    bad swept value fails loud rather than freezing/decaying the teacher."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ema_tau"):
            _make_module(ema_tau=bad)


def test_mask_ratio_overrides_reach_samplers(monkeypatch) -> None:
    """SSL-sweep knobs: ``m2_mask_ratio`` / ``m4_mask_ratio`` are stored on the
    module and forwarded as the held-out/mask ratio at the actual sampler call
    site in ``_sample_phase_mask``. Defaults reproduce the 6/03 masking lock
    (0.50 / 0.20); a passed value overrides it. Spying the sampler ratio kwarg
    is grid-independent (a count inequality saturates on the tiny test grid)."""
    import speech_decoding.experiments.v14_joint_module as jm

    # defaults: the locked held-out ratios are stored verbatim
    default_module = _make_module()
    assert default_module._m2_mask_ratio == 0.50
    assert default_module._m4_mask_ratio == 0.20

    batch = _make_synthetic_batch()
    et = batch.data["electrode_tokens"]
    support = batch.data["support"]

    # P1 (M2): the configured held_out_ratio reaches sample_m2_mask. Spy the
    # call, returning the real mask so the rest of the path is untouched.
    seen_m2: dict[str, float] = {}
    real_m2 = jm.sample_m2_mask

    def _spy_m2(shape, **kw):
        seen_m2["ratio"] = kw["held_out_ratio"]
        return real_m2(shape, **kw)

    monkeypatch.setattr(jm, "sample_m2_mask", _spy_m2)
    for ratio in (0.50, 0.80):
        m = _make_module(_make_tiny_encoder(), phase="p1", m2_mask_ratio=ratio)
        assert m._m2_mask_ratio == ratio
        m._sample_phase_mask(electrode_tokens=et, support=support)
        assert seen_m2["ratio"] == ratio, (
            f"m2_mask_ratio={ratio} must reach sample_m2_mask.held_out_ratio"
        )

    # P2 (M4): the configured mask_ratio reaches sample_m4_mask.
    seen_m4: dict[str, float] = {}
    real_m4 = jm.sample_m4_mask

    def _spy_m4(support_arg, **kw):
        seen_m4["ratio"] = kw["mask_ratio"]
        return real_m4(support_arg, **kw)

    monkeypatch.setattr(jm, "sample_m4_mask", _spy_m4)
    for ratio in (0.20, 0.40):
        m = _make_module(_make_tiny_encoder(), phase="p2", m4_mask_ratio=ratio)
        assert m._m4_mask_ratio == ratio
        m._sample_phase_mask(electrode_tokens=et, support=support)
        assert seen_m4["ratio"] == ratio, (
            f"m4_mask_ratio={ratio} must reach sample_m4_mask.mask_ratio"
        )


def test_ema_fires_once_per_optimizer_step_not_per_microbatch() -> None:
    """#46: the EMA update must live on ``on_before_zero_grad`` — Lightning's
    once-per-optimiser-step hook (after ``optimizer.step()``, before grads are
    zeroed) — NOT ``on_train_batch_end``, which fires once per micro-batch.

    Under ``accumulate_grad_batches=K`` the per-micro-batch placement applied K
    EMA updates per optimiser step, so the effective momentum became τ^K and the
    teacher trailed K× too fast — silently changing the SSL dynamics. This guards
    against a revert to the per-micro-batch hook."""
    module = _make_module()
    calls = {"n": 0}
    orig = module.teacher.update_from

    def _counting_update_from(student, **kw):  # instance attr: no self-binding
        calls["n"] += 1
        return orig(student, **kw)

    module.teacher.update_from = _counting_update_from  # type: ignore[method-assign]

    # The EMA lives on the once-per-optimiser-step hook.
    module.on_before_zero_grad(optimizer=None)
    assert calls["n"] == 1, "on_before_zero_grad must apply exactly one EMA step"

    # It must NOT also fire per micro-batch: the base-class ``on_train_batch_end``
    # is a no-op, so driving it (as Lightning does every micro-batch) leaves the
    # teacher untouched. A revert that re-adds the EMA call here would trip this.
    before = calls["n"]
    module.on_train_batch_end(outputs=None, batch=None, batch_idx=0)
    assert calls["n"] == before, (
        "on_train_batch_end must not apply an EMA step — per-micro-batch updates "
        "break gradient accumulation (#46): K updates/step ⇒ effective τ^K"
    )


def test_trainable_parameters_include_predictor() -> None:
    """The optimizer scope (``_trainable_parameters``) covers the student
    encoder + the predictor, and excludes the frozen teacher."""
    module = _make_module()
    trainable = {id(p) for p in module._trainable_parameters()}
    assert all(id(p) in trainable for p in module.predictor.parameters())
    assert all(id(p) in trainable for p in module.student.parameters())
    assert not any(id(p) in trainable for p in module.teacher.parameters())


# ---------------------------------------------------------------------------
# 5/28 P0 monitors — coverage / RankMe / grad-spike still wired
# ---------------------------------------------------------------------------


def test_monitor_logs_parcel_coverage_on_every_step() -> None:
    module = _make_module()
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(_make_synthetic_batch().data, step_name="train")
    for key in (
        "train_mon_coverage_active_mean",
        "train_mon_coverage_active_cv",
        "train_mon_coverage_slot_var",
        "train_mon_coverage_alarm",
    ):
        assert key in logged
    assert logged["train_mon_coverage_alarm"] in (0.0, 1.0)


def test_monitor_logs_teacher_rank_on_train_from_step0_and_val() -> None:
    """I1 (B36 WS-I): the M4 RankMe fires on the TRAIN loop from step 0 (the
    val/test gate was dropped) so a teacher-feature collapse is caught at the
    start of pretraining, not only at the first val epoch. M4 is the P2 target,
    so this is the P2-phase probe (2026-06-03 phase-scope fix moved M4 RankMe
    out of P1, where M4 is untrained random-init)."""
    module = _make_module(phase="p2")
    train_logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: train_logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(_make_synthetic_batch().data, step_name="train")
    for key in (
        "train_mon_rankme",
        "train_mon_rankme_normalised",
        "train_mon_rankme_warn",
        "train_mon_rankme_alarm",
    ):
        assert key in train_logged, key

    val_logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: val_logged.update({key: float(value)})  # type: ignore[method-assign]
    module._monitor_from_step(_make_synthetic_batch().data, step_name="val")
    for key in (
        "val_mon_rankme",
        "val_mon_rankme_normalised",
        "val_mon_rankme_warn",
        "val_mon_rankme_alarm",
    ):
        assert key in val_logged


def test_rankme_reads_post_encoder_ln_tap_not_ln_frame() -> None:
    """I1: the RankMe monitor reads the EMA teacher's post-``encoder_ln`` M4 tap
    (the canonical terminal LN, B6) — there is no separate ``ln_frame`` head any
    more. Guards the doc/code claim against an ``ln_frame`` revival."""
    module = _make_module()
    student_enc = module.student.encoder
    teacher_enc = module.teacher.model.encoder
    assert hasattr(student_enc, "encoder_ln")
    assert hasattr(teacher_enc, "encoder_ln")
    assert not hasattr(student_enc, "ln_frame")
    assert not hasattr(teacher_enc, "ln_frame")


def test_training_step_at_batch_idx_0_logs_rankme() -> None:
    """I1 end-to-end: driving ``training_step`` at global step 0 (monitor-due
    with no trainer attached → every-step cadence) emits the PHASE-APPROPRIATE
    feature-rank probe — ``train_mon_frontend_rankme`` in P1 (M2), and
    ``train_mon_rankme`` in P2 (M4)."""
    p1 = _make_module(phase="p1")
    p1_logged: dict[str, float] = {}
    p1.log = lambda key, value, **_kw: p1_logged.update({key: float(value)})  # type: ignore[method-assign]
    p1.training_step(_make_synthetic_batch(), 0)
    assert "train_mon_frontend_rankme" in p1_logged
    assert "train_mon_rankme" not in p1_logged

    p2 = _make_module(phase="p2")
    p2_logged: dict[str, float] = {}
    p2.log = lambda key, value, **_kw: p2_logged.update({key: float(value)})  # type: ignore[method-assign]
    p2.training_step(_make_synthetic_batch(), 0)
    assert "train_mon_rankme" in p2_logged
    assert "train_mon_frontend_rankme" not in p2_logged


def test_monitor_rank_probe_is_phase_scoped() -> None:
    """2026-06-03 mis-scope fix: each phase probes ONLY the representation it
    trains. P1 trains M2 (front-end) and never gradients the hard pool /
    inter-parcel stack that build M4, so probing M4 RankMe in P1 reads
    random-init layers and fires a false collapse alarm from step 0. So P1 ->
    front-end (M2) rank only; P2 -> M4 rank only. The two never alias on the
    same metric key."""
    p1 = _make_module(phase="p1")
    p1_logged: dict[str, float] = {}
    p1.log = lambda key, value, **_kw: p1_logged.update({key: float(value)})  # type: ignore[method-assign]
    p1._monitor_from_step(_make_synthetic_batch().data, step_name="train")
    for key in (
        "train_mon_frontend_rankme",
        "train_mon_frontend_rankme_normalised",
        "train_mon_frontend_rankme_warn",
        "train_mon_frontend_rankme_alarm",
    ):
        assert key in p1_logged, key
    # P1 must NOT emit the M4 probes (random-init M4 -> false alarm) ...
    assert "train_mon_rankme" not in p1_logged
    assert "train_mon_rankme_alarm" not in p1_logged
    # ... nor the M4-based orphan ratio.
    assert "train_mon_mask_002_ratio" not in p1_logged

    p2 = _make_module(phase="p2")
    p2_logged: dict[str, float] = {}
    p2.log = lambda key, value, **_kw: p2_logged.update({key: float(value)})  # type: ignore[method-assign]
    p2._monitor_from_step(_make_synthetic_batch().data, step_name="train")
    assert "train_mon_rankme_alarm" in p2_logged
    assert "train_mon_frontend_rankme" not in p2_logged


def _perturb_student_for_nonzero_grad(module: V14JointBrainModule) -> None:
    """Perturb the student so the deepcopy-identical EMA teacher no longer
    matches; otherwise all L1 terms could be ~0. ``encoder_ln`` is on the
    M4 path (P2) and downstream of M2 (P1)."""
    with torch.no_grad():
        module.student.encoder.frontend_ln.weight.add_(0.3)


def test_on_before_optimizer_step_logs_grad_spike() -> None:
    module = _make_module(phase="p1")
    _perturb_student_for_nonzero_grad(module)
    breakdown = module._step(_make_synthetic_batch().data)
    breakdown.total.backward()
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module.on_before_optimizer_step(optimizer=None)
    for key in (
        "train_mon_grad_l2",
        "train_mon_grad_ema_l2",
        "train_mon_grad_spike_ratio",
        "train_mon_grad_spike",
        "train_mon_grad_diverged",
    ):
        assert key in logged
    assert logged["train_mon_grad_ema_l2"] == pytest.approx(0.0)
    assert logged["train_mon_grad_l2"] > 0.0
    assert float(module._grad_ema_l2.item()) > 0.0


def test_on_before_optimizer_step_logs_grad_routing_and_clip() -> None:
    """#119 GRAD-ROUTING + CLIP — per-group grad/weight norms decompose the
    global, and the clip metrics fire when a trainer clip ceiling is set."""
    module = _make_module(phase="p1")
    _perturb_student_for_nonzero_grad(module)
    module._step(_make_synthetic_batch().data).total.backward()
    # A trainer with a clip ceiling so the clip metrics are emitted.
    module.trainer = SimpleNamespace(gradient_clip_val=1.0)  # type: ignore[assignment]
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module.on_before_optimizer_step(optimizer=None)

    for group in ("frontend", "latent", "predictor"):
        assert f"train_mon_grad_l2_{group}" in logged
        assert f"train_mon_wnorm_{group}" in logged
        assert logged[f"train_mon_wnorm_{group}"] > 0.0
    # The three group grad-L2² must sum to the global grad-L2² (exact tiling).
    per_group_sq = sum(
        logged[f"train_mon_grad_l2_{g}"] ** 2
        for g in ("frontend", "latent", "predictor")
    )
    assert per_group_sq == pytest.approx(logged["train_mon_grad_l2"] ** 2, rel=1e-4)
    # P1: the latent (parcel-SA) side is grad-free → ~zero; the front-end +
    # predictor carry the gradient.
    assert logged["train_mon_grad_l2_latent"] == pytest.approx(0.0, abs=1e-6)
    assert logged["train_mon_grad_l2_frontend"] > 0.0
    # Clip metrics present + self-consistent with the global grad-L2.
    assert "train_mon_grad_clipped" in logged
    assert "train_mon_grad_clip_scale" in logged
    expected_scale = min(1.0, 1.0 / (logged["train_mon_grad_l2"] + 1e-12))
    assert logged["train_mon_grad_clip_scale"] == pytest.approx(expected_scale, rel=1e-4)
    assert logged["train_mon_grad_clipped"] == (
        1.0 if logged["train_mon_grad_l2"] > 1.0 else 0.0
    )


def test_on_before_optimizer_step_skips_clip_metrics_without_trainer() -> None:
    """No trainer attached (unit-test path) → clip metrics are silently
    skipped, not a crash."""
    module = _make_module(phase="p1")
    _perturb_student_for_nonzero_grad(module)
    module._step(_make_synthetic_batch().data).total.backward()
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module.on_before_optimizer_step(optimizer=None)
    assert "train_mon_grad_clipped" not in logged
    assert "train_mon_grad_clip_scale" not in logged
    # Routing metrics do NOT need a trainer.
    assert "train_mon_grad_l2_frontend" in logged


def test_grad_ema_buffer_persists_across_calls() -> None:
    module = _make_module(phase="p1")
    _perturb_student_for_nonzero_grad(module)
    batch = _make_synthetic_batch()
    module._step(batch.data).total.backward()
    module.on_before_optimizer_step(optimizer=None)
    first_ema = float(module._grad_ema_l2.item())
    assert first_ema > 0.0

    for p in module._trainable_parameters():
        if p.grad is not None:
            p.grad.zero_()
    module._step(batch.data).total.backward()
    logged: dict[str, float] = {}
    module.log = lambda key, value, **_kw: logged.update({key: float(value)})  # type: ignore[method-assign]
    module.on_before_optimizer_step(optimizer=None)
    assert logged["train_mon_grad_ema_l2"] == pytest.approx(first_ema)


def test_train_monitor_due_fires_on_log_cadence() -> None:
    module = _make_module()
    module.trainer = SimpleNamespace(log_every_n_steps=10)  # type: ignore[assignment]
    for due_idx in (0, 10, 20, 100):
        assert module._train_monitor_due(due_idx) is True, due_idx
    for skip_idx in (1, 5, 9, 11, 19, 99):
        assert module._train_monitor_due(skip_idx) is False, skip_idx


def test_train_monitor_due_falls_back_to_every_step() -> None:
    module = _make_module()  # no trainer attached → property raises
    for idx in (0, 1, 2, 3, 7, 50):
        assert module._train_monitor_due(idx) is True, idx


# ---------------------------------------------------------------------------
# Speedup C1/C2: torch.compile forward override (default OFF, env-gated)
# ---------------------------------------------------------------------------


def test_compile_flag_off_default_keeps_eager_forward(monkeypatch) -> None:
    """``V14_COMPILE`` unset (default) → no compile, the eager modules are
    called → byte-identical → zero blast radius on the running config."""
    monkeypatch.delenv("V14_COMPILE", raising=False)
    m = _make_module(phase="p1")
    assert m._compiled_fwd == {}
    # the routing helpers fall through to the uncompiled modules
    assert m._compiled_fwd.get("student", m.student) is m.student
    assert m._compiled_fwd.get("teacher", m.teacher.model) is m.teacher.model


def test_compile_flag_does_not_double_register_params(monkeypatch) -> None:
    """EMA-safety invariant: the compiled wrappers SHARE parameters with
    ``self.student`` / ``self.teacher.model``. They are stored in a plain dict
    (not a submodule) so nn.Module never double-registers those tensors — else
    the optimizer + EMA param zip would be corrupted and checkpoint keys would
    gain the ``_orig_mod.`` prefix that breaks the EMA name-match."""
    monkeypatch.delenv("V14_COMPILE", raising=False)
    eager = _make_module(phase="p1")

    monkeypatch.setenv("V14_COMPILE", "1")
    compiled = _make_module(phase="p1")
    assert set(compiled._compiled_fwd) == {"student", "teacher"}

    # Same architecture → identical param COUNT iff compile added nothing.
    assert len(list(compiled.parameters())) == len(list(eager.parameters()))
    # state_dict keys are unchanged (no ``_orig_mod.`` prefix anywhere).
    assert set(compiled.state_dict()) == set(eager.state_dict())
    assert not any("_orig_mod" in k for k in compiled.state_dict())
    # the compiled student wrapper exposes the SAME tensors as self.student.
    assert {id(p) for p in compiled._compiled_fwd["student"].parameters()} == {
        id(p) for p in compiled.student.parameters()
    }


def test_compiled_step_matches_eager_within_tol(monkeypatch) -> None:
    """C2 correctness: compiled ``_step`` loss ≈ eager within tolerance on a
    synthetic CPU batch (identical weights via load_state_dict; the per-step
    mask is sampled OUTSIDE the compiled region → identical for both)."""
    batch = _make_synthetic_batch().data

    monkeypatch.delenv("V14_COMPILE", raising=False)
    eager = _make_module(_make_tiny_encoder(), phase="p1")
    eager.eval()
    with torch.no_grad():
        eager_loss = float(eager._step(batch).total.detach())

    monkeypatch.setenv("V14_COMPILE", "1")
    compiled = _make_module(_make_tiny_encoder(), phase="p1")
    compiled.load_state_dict(eager.state_dict())  # identical weights
    compiled.eval()
    try:
        with torch.no_grad():
            compiled_loss = float(compiled._step(batch).total.detach())
    except Exception as exc:  # pragma: no cover - backend-dependent
        pytest.skip(f"torch.compile backend unavailable on this host: {exc}")

    assert compiled_loss == pytest.approx(eager_loss, abs=1e-3, rel=1e-3)
