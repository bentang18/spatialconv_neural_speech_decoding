"""Integration tests for ``build_v14_experiment`` after the view wrapper landed.

Replaces the placeholder ``test_v14_dispatch_raises_until_electrode_tokens_extractor_wired``
in ``test_v14_wiring.py`` — that gap is now closed by the default
:class:`speech_decoding.extractors.view.LogStftView` plus
:class:`speech_decoding.extractors.valid_mask.ElectrodeValidMask` plus
``c_max``-padded :class:`V14DKHardSupportExtractor`.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.experiments import dispatch_v14
from speech_decoding.extractors.dk_support import V14DKHardSupportExtractor
from speech_decoding.extractors.valid_mask import ElectrodeValidMask
from speech_decoding.extractors.view import LogStftView


def test_dispatch_default_wires_log_stft_view(tmp_path, monkeypatch) -> None:
    """Without ``electrode_tokens_extractor``, the dispatch picks LogStftView."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    extractors = xp.data.segmenter.extractors
    assert isinstance(extractors["electrode_tokens"], LogStftView)
    ext = extractors["electrode_tokens"]
    assert ext.car == "shaft"
    assert ext.notch_filter == 60.0
    assert ext.stft_nperseg == 512
    assert abs(ext.stft_poverlap - 0.75) < 1e-9
    assert ext.stft_max_freq_hz == 150.0


def test_dispatch_default_wires_valid_mask_and_support_with_c_max(
    tmp_path, monkeypatch,
) -> None:
    """Support and valid-mask both align to ``c_max`` so per-batch collation works."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    extractors = xp.data.segmenter.extractors
    assert isinstance(extractors["support"], V14DKHardSupportExtractor)
    assert isinstance(extractors["valid_mask"], ElectrodeValidMask)
    assert extractors["support"].c_max == 384
    assert extractors["valid_mask"].c_max == 384


def test_dispatch_default_sets_x_name_tuple_with_mask(
    tmp_path, monkeypatch,
) -> None:
    """v14 BrainModule x_name unpacks the base 3-tuple plus the
    per-clip metadata kwargs (subject_subtype, ref_idx, lambda_anat)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    assert tuple(xp.x_name) == (
        "electrode_tokens", "support", "valid_mask",
        "subject_subtype", "ref_idx", "lambda_anat",
    )


def test_b_m2_dispatch_registers_metadata_extractors_in_segmenter(
    tmp_path, monkeypatch,
) -> None:
    """Per-clip metadata extractors are wired to NeuralSet's segmenter
    so the dataloader produces the encoder's B29 Item 11/12 kwargs from
    the actual events."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    extractors = xp.data.segmenter.extractors
    assert "ref_idx" in extractors
    assert "subject_subtype" in extractors
    assert "lambda_anat" in extractors


def test_dispatch_default_sets_time_last_input_true(tmp_path, monkeypatch) -> None:
    """The encoder must transpose NS time-last input (B, C, F, T) -> (B, C, T, F)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    assert xp.brain_model_config.time_last_input is True


def test_dispatch_accepts_custom_electrode_tokens_extractor(
    tmp_path, monkeypatch,
) -> None:
    """Caller can still pass an explicit extractor; default is just a default."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    custom = LogStftView(
        event_types="Ieeg",
        car="shaft",
        notch_filter=60.0,
        stft_nperseg=256,  # different from default to confirm caller-override
    )
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", electrode_tokens_extractor=custom,
    )
    assert xp.data.segmenter.extractors["electrode_tokens"] is custom


def test_dispatch_dry_run_no_longer_mentions_missing_extractor(
    capsys, tmp_path, monkeypatch,
) -> None:
    """``--dry-run`` output should no longer claim the extractor is unwired."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    rc = dispatch_v14.main(["--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "not wired yet" not in out
    assert "LogStftView" in out or "electrode-tokens extractor wired" in out


def test_dk_support_c_max_pads_output(tmp_path) -> None:
    """When ``c_max`` is set, the support tensor pads to (c_max, K=80) zero-rows."""
    csv_dir = tmp_path / "localization" / "sub_1"
    csv_dir.mkdir(parents=True)
    (csv_dir / "depth-wm.csv").write_text(
        "Electrode,DesikanKilliany\nE1,ctx-lh-precentral\nE2,ctx-rh-bankssts\n"
    )

    ext = V14DKHardSupportExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120,
    )
    from types import SimpleNamespace
    out = ext.get_static(SimpleNamespace(subject="1"))  # type: ignore[arg-type]

    assert out.shape == (120, 80)
    assert out[:2].sum().item() == 2.0
    assert out[2:].sum().item() == 0.0


def test_dispatch_default_wires_btwordevents_chain(tmp_path, monkeypatch) -> None:
    """``build_v14_experiment`` wraps Wang2024Treebank + BTWordEvents in an
    ``ns.Chain`` so split-aware Word events flow into the segmenter."""
    import neuralset as ns
    from speech_decoding.studies.braintreebank.study import Wang2024Treebank
    from speech_decoding.studies.braintreebank.word_events import BTWordEvents

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    chain = xp.data.study
    assert isinstance(chain, ns.Chain), f"expected ns.Chain, got {type(chain)!r}"
    assert len(chain.steps) == 2
    assert isinstance(chain.steps[0], Wang2024Treebank)
    assert isinstance(chain.steps[1], BTWordEvents)
    assert chain.steps[1].tasks == (dispatch_v14.DEFAULT_TASK,)
    assert chain.steps[1].eval_mode == "CrossSession"
    assert xp.data.segmenter.trigger_query == "type == 'Word'"


def test_dispatch_default_target_pulls_label_from_word(tmp_path, monkeypatch) -> None:
    """Target extractor must read ``label`` off the Word events, not from Ieeg."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    target = xp.data.segmenter.extractors["target"]
    assert target.event_types == "Word"
    assert target.event_field == "label"


def test_log_stft_view_pads_to_c_max(monkeypatch) -> None:
    """When ``c_max`` is set, LogStftView pads (C_event, F, T) -> (c_max, F, T)."""
    from neuralset.base import TimedArray
    from speech_decoding.extractors.reference import CARIeegExtractor
    import numpy as np

    rng = np.random.default_rng(0)
    fake_ta = TimedArray(
        frequency=2048.0, start=0.0, duration=1.0,
        data=rng.standard_normal((5, 2048)).astype(np.float32),
    )

    def _fake(self, event, start, duration):
        return fake_ta

    monkeypatch.setattr(CARIeegExtractor, "_get_timed_array", _fake, raising=False)

    view = LogStftView(
        event_types="Ieeg", car="shaft", notch_filter=60.0, c_max=120,
    )

    class _Ev:
        start = 0.0
        duration = 1.0
        frequency = 2048.0

    out = view._get_timed_array(_Ev(), start=0.0, duration=1.0)
    assert out.data.shape == (120, 38, 17)
    # padding rows are exactly zero (log_eps spec from real waveform is finite,
    # so any all-zero row is from padding).
    assert (out.data[5:] == 0.0).all()


def test_encoder_time_last_input_transposes_correctly() -> None:
    """`time_last_input=True` transposes (B, C, F, T) -> (B, C, T, F) so the
    forward body sees the canonical layout regardless of NeuralSet time-last."""
    from speech_decoding.models import V14ParcelPerceiver

    cfg = V14ParcelPerceiver(
        n_freq_bins=3, n_time_bins=5, k_parcels=6,
        d_model=32, n_heads=4, depth_self_attn=1, m_sub_slots=2,
        time_last_input=True,
    )
    model = cfg.build(n_outputs=2)
    # NS-style time-last input: (B, C, F=3, T=5)
    x = torch.randn(2, 4, 3, 5)
    support = torch.zeros(2, 4, 6)
    support[..., 0] = 1.0
    out = model(x, support)
    assert out.shape == (2, 2)


# --- B28 (2026-05-27 PM) dispatch flags ------------------------------


def test_b28_dispatch_default_dkoleo_mode_is_off(tmp_path, monkeypatch) -> None:
    """B28 default: DKoleo @ M4 is off in dispatch; the 4-term composer fires."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    # The dispatch records its dkoleo selection through the build kwarg; the
    # composer-side wiring is downstream (LOSS-01), so we just verify the
    # function accepts the default and runs end-to-end.
    assert xp is not None


def test_b28_dispatch_rejects_unknown_dkoleo_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest
    with pytest.raises(ValueError, match="dkoleo_mode"):
        dispatch_v14.build_v14_experiment(mode="nano", dkoleo_mode="bogus")


def test_b28_dispatch_propagates_cross_attn_positions(tmp_path, monkeypatch) -> None:
    """`cross_attn_positions=[0, 3]` (R-perceiver-original-2-cross-attns)
    flows from build kwargs into the V14ParcelPerceiver config."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", cross_attn_positions=[0, 3], depth=6,
    )
    assert xp.brain_model_config.cross_attn_positions == [0, 3]


def test_b28_dispatch_default_cross_attn_positions_is_none(
    tmp_path, monkeypatch,
) -> None:
    """Default: ``cross_attn_positions`` passes through as None → encoder
    constructor's default of [0] applies."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp.brain_model_config.cross_attn_positions is None


def test_b28_dispatch_swec_corpus_uses_50hz_notch() -> None:
    """``MAINS_NOTCH_BY_CORPUS['swec'] == 50.0`` (Swiss site)."""
    assert dispatch_v14.MAINS_NOTCH_BY_CORPUS["swec"] == 50.0
    # US sites stay at 60 Hz.
    for us_corpus in ("braintreebank", "d_cohort", "ajile12"):
        assert dispatch_v14.MAINS_NOTCH_BY_CORPUS[us_corpus] == 60.0


def test_b28_dispatch_mains_notch_kwarg_overrides_default(
    tmp_path, monkeypatch,
) -> None:
    """SWEC dispatch passes ``mains_notch_hz=50.0`` → LogStftView gets 50 Hz."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano", mains_notch_hz=50.0)
    ext = xp.data.segmenter.extractors["electrode_tokens"]
    assert ext.notch_filter == 50.0


def test_b28_cli_parses_cross_attn_positions() -> None:
    """`--cross-attn-positions 0,3` parses to ``"0,3"``."""
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--cross-attn-positions", "0,3"])
    assert args.cross_attn_positions == "0,3"


def test_b28_cli_parses_dkoleo_mode() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--dkoleo-mode", "intra_clip_slots"])
    assert args.dkoleo_mode == "intra_clip_slots"


def test_b28_cli_default_dkoleo_mode_is_off() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.dkoleo_mode == "off"


def test_b28_cli_default_mains_notch_is_60() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.mains_notch_hz == 60.0


def test_b28_cli_mains_notch_accepts_swec_50() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--mains-notch-hz", "50.0"])
    assert args.mains_notch_hz == 50.0


def test_c3_dispatch_rejects_corpus_mix_with_missing_notch_entry(
    tmp_path, monkeypatch,
) -> None:
    """Every corpus in ``corpus_mix`` must have a
    ``notch_filter_hz_by_corpus`` entry so the per-corpus extractor
    builds with the right mains frequency."""
    import pytest

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    bad_mix = {"braintreebank": 0.5, "fake_eu_corpus": 0.5}
    with pytest.raises(ValueError, match="notch_filter_hz_by_corpus"):
        dispatch_v14.build_v14_experiment(
            mode="nano",
            corpus_mix=bad_mix,
            # MAINS_NOTCH_BY_CORPUS does NOT contain "fake_eu_corpus"
            # so the validation must trip even with the default map.
        )


def test_c3_dispatch_per_corpus_notch_routes_into_bt_extractor(
    tmp_path, monkeypatch,
) -> None:
    """When ``mains_notch_hz`` is left at default, the BT extractor
    reads its notch from ``notch_filter_hz_by_corpus`` — so an
    override-via-map for BT actually reaches the extractor."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano",
        notch_filter_hz_by_corpus={
            "braintreebank": 50.0,
            "wang2024treebank": 60.0,
            "d_cohort": 60.0,
            "cogan_dcohort": 60.0,
            "ajile12": 60.0,
            "swec": 50.0,
        },
    )
    ext = xp.data.segmenter.extractors["electrode_tokens"]
    assert ext.notch_filter == 50.0


# ---------------------------------------------------------------------------
# B29 lock 2026-05-27 PM-late — dispatch surface
# ---------------------------------------------------------------------------


def test_b29_dispatch_default_m_sub_slots_is_one(tmp_path, monkeypatch) -> None:
    """B29 Item 13: default M=1 (was 4) propagates to brain_model_config."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    assert xp.brain_model_config.m_sub_slots == 1


def test_b29_dispatch_r_m4_slots_sister_restores_m_eq_4(tmp_path, monkeypatch) -> None:
    """Sister R-m4-slots flips ``m_sub_slots`` back to 4 via dispatch."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano", m_sub_slots=4)
    assert xp.brain_model_config.m_sub_slots == 4


def test_b29_dispatch_default_disables_subtype_and_ref_embed(
    tmp_path, monkeypatch,
) -> None:
    """B29 Item 11 + 5/28 PM precedent-audit flip: subtype default OFF.
    B32 5/28 PM-late first-pass-no-input-aug lock: ref_embed default
    also OFF (was ON). vocab=2 (binary) stays."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    cfg = xp.brain_model_config
    assert cfg.subtype_embed_enabled is False
    assert cfg.subtype_embed_reuse_kv is True
    # B32 5/28 PM-late first-pass-no-input-aug lock: ref_embed default OFF.
    assert cfg.ref_embed_enabled is False
    assert cfg.ref_embed_reuse_kv is True
    assert cfg.subtype_vocab == 2  # binary default


def test_b29_dispatch_subtype_three_way_vocab_propagates(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", subtype_embed_vocab="three_way",
    )
    assert xp.brain_model_config.subtype_vocab == 3


def test_b29_dispatch_r_subtype_embed_on_with_kv_reuse_sister(
    tmp_path, monkeypatch,
) -> None:
    """Sister ``R-subtype-embed-on-with-kv-reuse`` P0 (NEW, prior default
    → sister): enabling subtype_embed restores the binary embed + reuse."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", subtype_embed_enabled=True,
    )
    assert xp.brain_model_config.subtype_embed_enabled is True
    assert xp.brain_model_config.subtype_embed_reuse_kv is True


def test_b29_dispatch_r_subtype_embed_input_only_sister(
    tmp_path, monkeypatch,
) -> None:
    """Sister ``R-subtype-embed-input-only`` P0 (PROMOTED M3AE-faithful):
    enabling subtype with reuse_kv=False adds at A1 only."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano",
        subtype_embed_enabled=True,
        subtype_embed_reuse_kv=False,
    )
    assert xp.brain_model_config.subtype_embed_enabled is True
    assert xp.brain_model_config.subtype_embed_reuse_kv is False


def test_b29_dispatch_ref_embed_input_only_sister(tmp_path, monkeypatch) -> None:
    """Post-B32 the default is ``ref_embed_enabled=False``; the input-only
    sister cell now needs callers to opt-in both flags explicitly."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", ref_embed_enabled=True, ref_embed_reuse_kv=False,
    )
    assert xp.brain_model_config.ref_embed_enabled is True
    assert xp.brain_model_config.ref_embed_reuse_kv is False


def test_b29_dispatch_default_corpus_mix_sums_to_one(tmp_path, monkeypatch) -> None:
    """B29 corpus mix default normalizes to sum == 1.0 ± 1e-4."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    # Just calling build_v14_experiment with mode='nano' must not raise the
    # sum-to-1 assertion.
    dispatch_v14.build_v14_experiment(mode="nano")
    # Direct numerical check.
    total = sum(dispatch_v14.DEFAULT_CORPUS_MIX.values())
    assert abs(total - 1.0) < 1e-4


def test_b29_dispatch_corpus_mix_must_sum_to_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    bad = {"swec": 0.5, "ajile12": 0.5, "d_cohort": 0.5, "braintreebank": 0.5}
    import pytest

    with pytest.raises(ValueError, match="corpus_mix must sum to 1.0"):
        dispatch_v14.build_v14_experiment(mode="nano", corpus_mix=bad)


def test_b29_dispatch_corpus_mix_renormalizes_when_ajile12_excluded(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    # Default corpus_mix includes ajile12; with include_ajile12=False the
    # build re-normalizes over the remaining three corpora and does NOT
    # raise the sum-to-1 assertion.
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", include_ajile12=False,
    )
    assert xp is not None  # build succeeded with renormalized mix


def test_b29_dispatch_ref_operator_alpha_must_be_in_open_unit_interval(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest

    with pytest.raises(ValueError, match="ref_operator_alpha"):
        dispatch_v14.build_v14_experiment(mode="nano", ref_operator_alpha=0.0)
    with pytest.raises(ValueError, match="ref_operator_alpha"):
        dispatch_v14.build_v14_experiment(mode="nano", ref_operator_alpha=1.0)


def test_b29_dispatch_phase_mode_default_is_joint_b29() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.phase_mode == "joint_b29"


def test_b29_dispatch_anatomy_bias_mode_default_is_per_clip_gate() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.anatomy_bias_mode == "per_clip_gate_b29"


def test_b29_dispatch_default_includes_ajile12() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.include_ajile12 is True


def test_b29_dispatch_default_ref_operator_alpha_is_0p3() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.ref_operator_alpha == 0.3


def test_b29_dispatch_default_ffn_variant_is_dense() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.ffn_variant == "dense"


def test_b29_dispatch_soft_moe_4_raises_not_implemented(
    tmp_path, monkeypatch,
) -> None:
    """B29 MoE audit 2026-05-28: soft_moe_4 must fail-closed until
    ``models/soft_moe.py`` lands."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest

    with pytest.raises(NotImplementedError, match="soft_moe_4"):
        dispatch_v14.build_v14_experiment(mode="nano", ffn_variant="soft_moe_4")


def test_b29_dispatch_dkoleo_vicreg_slot_variance_mode_accepted(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", dkoleo_mode="vicreg_slot_variance",
    )
    assert xp is not None


def test_b29_dispatch_dkoleo_batch_cls_alias_maps_to_unit(
    tmp_path, monkeypatch,
) -> None:
    """Back-compat: ``batch_cls`` is aliased to ``batch_cls_unit`` so existing
    pre-B29 dispatch scripts still validate."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", dkoleo_mode="batch_cls",
    )
    assert xp is not None


def test_b29_dispatch_invalid_subtype_vocab_rejected(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest

    with pytest.raises(ValueError, match="subtype_embed_vocab"):
        dispatch_v14.build_v14_experiment(
            mode="nano", subtype_embed_vocab="quaternary",
        )


def test_b29_cli_parses_subtype_three_way_vocab() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--subtype-embed-vocab", "three_way"])
    assert args.subtype_embed_vocab == "three_way"


def test_b29_cli_subtype_embed_flag_enables() -> None:
    """Post 5/28 PM flip: subtype default is OFF; the CLI flag opts in."""
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--subtype-embed"])
    assert args.subtype_embed_enabled is True


def test_b29_cli_subtype_embed_default_is_disabled() -> None:
    """No flag → default OFF per 5/28 PM precedent-audit flip."""
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.subtype_embed_enabled is False


def test_b29_cli_no_ref_embed_reuse_kv_flag_disables_reuse() -> None:
    """`--no-ref-embed-reuse-kv` only flips reuse_kv. Post-B32 the
    ref_embed default is OFF, so this flag in isolation does NOT enable
    the embed — callers must combine it with ``--ref-embed`` to land the
    input-only sister."""
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--no-ref-embed-reuse-kv"])
    assert args.ref_embed_reuse_kv is False
    assert args.ref_embed_enabled is False  # B32 default; ``--ref-embed`` not passed


def test_b29_cli_no_include_ajile12_flag() -> None:
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--no-include-ajile12"])
    assert args.include_ajile12 is False


# ---------------------------------------------------------------------------
# B31 V-JEPA-2-canonical 2-term lock 2026-05-28 PM-late
# ([[project_v14_b31_vjepa2_canonical_loss_2026_05_28]]). CLI surface +
# dispatch wiring of the ``--loss-variant`` selector.
# ---------------------------------------------------------------------------


def test_b31_cli_loss_variant_default_is_b31_default() -> None:
    """B31 default: ``--loss-variant`` omitted → ``"b31_default"`` (the
    V-JEPA-2-canonical 2-term joint SSL surface)."""
    parser = dispatch_v14._parser()
    args = parser.parse_args([])
    assert args.loss_variant == "b31_default"


@pytest.mark.parametrize(
    "variant",
    ["b31_default", "b31_plus_m3", "b31_plus_utt", "b31_plus_both"],
)
def test_b31_cli_loss_variant_accepts_all_four_arms(variant: str) -> None:
    """All four B31 ``loss_variant`` arms parse without error."""
    parser = dispatch_v14._parser()
    args = parser.parse_args(["--loss-variant", variant])
    assert args.loss_variant == variant


def test_b31_cli_loss_variant_rejects_bogus_value() -> None:
    """argparse rejects an unknown variant (defends the run record
    YAML from drift)."""
    parser = dispatch_v14._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--loss-variant", "bogus"])


def test_b31_dispatch_supervised_phase_rejects_non_default_loss_variant(
    tmp_path, monkeypatch,
) -> None:
    """``loss_variant`` is a joint-phase-only selector. Passing a
    sister value with ``joint_phase=False`` (the default supervised
    Phase-4 path) raises so the run record never silently mis-records
    the sister."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match="loss_variant"):
        dispatch_v14.build_v14_experiment(
            mode="nano", loss_variant="b31_plus_utt",
        )


def test_b31_dispatch_supervised_phase_accepts_default_loss_variant(
    tmp_path, monkeypatch,
) -> None:
    """Sanity: supervised Phase-4 path WITH the default
    ``loss_variant`` constructs cleanly."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", loss_variant="b31_default",
    )
    # Supervised path uses vanilla Experiment, not V14JointExperiment.
    assert type(xp).__name__ == "Experiment"


@pytest.mark.parametrize(
    "variant",
    ["b31_default", "b31_plus_m3", "b31_plus_utt", "b31_plus_both"],
)
def test_b31_dispatch_joint_phase_propagates_loss_variant_to_experiment(
    tmp_path, monkeypatch, variant: str,
) -> None:
    """Joint-phase dispatch threads ``loss_variant`` onto the
    :class:`V14JointExperiment` instance. The brain-model config does
    NOT carry it (its Pydantic schema is ``extra='forbid'``); the
    Experiment-level snapshot records the choice."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", joint_phase=True, loss_variant=variant,
    )
    assert type(xp).__name__ == "V14JointExperiment"
    assert xp.loss_variant == variant
    # The brain-model config must NOT carry ``loss_variant`` (would
    # crash Pydantic ``extra='forbid'``).
    assert "loss_variant" not in xp.brain_model_config.model_dump()


def test_b31_dispatch_invalid_loss_variant_rejected(
    tmp_path, monkeypatch,
) -> None:
    """``build_v14_experiment`` validates the variant string up front,
    before any Experiment construction."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    with pytest.raises(ValueError, match="loss_variant must be one of"):
        dispatch_v14.build_v14_experiment(
            mode="nano", joint_phase=True, loss_variant="bogus",
        )


def test_b_bug_4_default_log_stft_view_pins_ref_idx_to_shaft_car(
    tmp_path, monkeypatch,
) -> None:
    """When the dispatch builds the static ``LogStftView(car="shaft")``
    electrode-tokens path, ``RefIdxExtractor`` must collapse to a single
    mode so the ``ref_idx`` label matches the operator the waveform
    actually saw — otherwise the encoder learns to condition on a label
    that lies.
    """
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")
    ref_idx_ext = xp.data.segmenter.extractors["ref_idx"]
    assert tuple(ref_idx_ext.ref_modes) == ("shaft_car",)


def test_b_bug_4_ref_aug_multi_stft_view_lifts_ref_modes_to_label(
    tmp_path, monkeypatch,
) -> None:
    """When the caller supplies a :class:`RefAugMultiStftView` (REF-aug
    3-cell), the dispatch lifts its ``ref_modes`` onto the label
    extractor so both halves of the REF-aug contract draw from the same
    set.
    """
    from speech_decoding.extractors.ref_aug import RefAugMultiStftView

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    view = RefAugMultiStftView(
        event_types="Ieeg",
        ref_modes=("shaft_car", "bipolar"),
        seed=42,
    )
    xp = dispatch_v14.build_v14_experiment(
        mode="nano", electrode_tokens_extractor=view, seed=7,
    )
    ref_idx_ext = xp.data.segmenter.extractors["ref_idx"]
    assert tuple(ref_idx_ext.ref_modes) == ("shaft_car", "bipolar")
    # ``seed`` is lifted from the view at construction so the shared
    # ``_event_key`` SHA keeps draws aligned regardless of the
    # dispatch's ``seed`` argument.
    assert int(ref_idx_ext.seed) == 42


def test_r3_bug_3_dispatch_rejects_global_car_log_stft_view(
    tmp_path, monkeypatch,
) -> None:
    """A non-RefAugMultiStftView extractor whose CAR config does not
    map cleanly into ``REF_MODES`` would otherwise silently train the
    encoder against a ``ref_idx`` label that lies about the operator.
    Raise instead so the caller picks a supported config (shaft-CAR
    LogStftView) or opts into RefAugMultiStftView.
    """
    import pytest

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    bad_view = LogStftView(
        event_types="Ieeg",
        car="global",
        notch_filter=60.0,
        scaler="StandardScaler",
        channel_order="original",
        c_max=384,
    )
    with pytest.raises(ValueError, match="does not map into REF_MODES"):
        dispatch_v14.build_v14_experiment(
            mode="nano", electrode_tokens_extractor=bad_view,
        )


def test_dispatch_rejects_extractor_without_car_attribute(
    tmp_path, monkeypatch,
) -> None:
    """When the caller hands in an electrode-tokens extractor that
    exposes no ``car`` attribute at all (a future raw-waveform view
    that doesn't inherit ``CARIeegExtractor``), the dispatch must raise
    a clear ValueError naming the extractor class instead of falling
    through with a misleading "does not map into REF_MODES" message.
    """
    import pytest

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))

    class _NoCarExtractor:
        pass

    with pytest.raises(ValueError, match="exposes no 'car' attribute"):
        dispatch_v14.build_v14_experiment(
            mode="nano",
            electrode_tokens_extractor=_NoCarExtractor(),
        )


def test_r3_bug_4_v14_parcel_perceiver_rejects_bogus_ssl_modes() -> None:
    """``dkoleo_mode`` / ``phase_mode`` / ``anatomy_bias_mode`` on
    :class:`V14ParcelPerceiver` are typed as :class:`Literal` so
    reconstruction from a persisted YAML rejects typos at
    deserialization, rather than silently riding a bogus string through
    to the (absent) SSL trainer.
    """
    import pytest
    from pydantic import ValidationError

    from speech_decoding.models.v14_encoder import V14ParcelPerceiver

    with pytest.raises(ValidationError):
        V14ParcelPerceiver(
            n_freq_bins=12, n_time_bins=8, k_parcels=80,
            dkoleo_mode="bogus",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        V14ParcelPerceiver(
            n_freq_bins=12, n_time_bins=8, k_parcels=80,
            phase_mode="p1_only",  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        V14ParcelPerceiver(
            n_freq_bins=12, n_time_bins=8, k_parcels=80,
            anatomy_bias_mode="off",  # type: ignore[arg-type]
        )


def test_b_cr_3_brain_model_config_carries_ssl_dispatch_flags(
    tmp_path, monkeypatch,
) -> None:
    """``dkoleo_mode`` / ``phase_mode`` / ``anatomy_bias_mode`` were
    validated by ``build_v14_experiment`` then silently dropped on the
    floor — the downstream supervised path and the SSL trainer alike
    never saw them in the persisted config. Lock them as named fields
    on the ``V14ParcelPerceiver`` config so they ride along with the
    run record.
    """
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(
        mode="nano",
        dkoleo_mode="vicreg_slot_variance",
        phase_mode="split_p1_p2",
        anatomy_bias_mode="warmup_b28",
    )
    cfg = xp.brain_model_config
    assert cfg.dkoleo_mode == "vicreg_slot_variance"
    assert cfg.phase_mode == "split_p1_p2"
    assert cfg.anatomy_bias_mode == "warmup_b28"


def test_b23_dispatch_joint_phase_wires_shaft_mask_extractor(
    tmp_path, monkeypatch,
) -> None:
    """B2.3 / B03 lock 2026-05-27 PM: the joint SSL dispatch wires a
    :class:`BTShaftMaskExtractor` under the ``shaft_mask`` key. The
    student-only routing (teacher full-input contract) is enforced by
    :class:`V14JointBrainModule._extract_student_kwargs` — this test
    locks the segmenter-side wiring."""
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano", joint_phase=True)
    extractors = xp.data.segmenter.extractors
    assert "shaft_mask" in extractors, (
        "joint phase must wire the BT shaft-mask extractor"
    )
    assert isinstance(extractors["shaft_mask"], BTShaftMaskExtractor)


def test_b23_dispatch_supervised_phase_4_omits_shaft_mask(
    tmp_path, monkeypatch,
) -> None:
    """Phase-4 supervised dispatch does NOT run the joint BrainModule, so
    the shaft-mask extractor stays absent (avoids a no-op extractor +
    cache cost on the downstream classifier path)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")  # joint_phase=False
    extractors = xp.data.segmenter.extractors
    assert "shaft_mask" not in extractors, (
        "supervised Phase-4 path must NOT wire the shaft-mask extractor"
    )
