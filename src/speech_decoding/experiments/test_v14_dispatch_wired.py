"""Integration tests for ``build_v14_experiment`` after the view wrapper landed.

Replaces the placeholder ``test_v14_dispatch_raises_until_electrode_tokens_extractor_wired``
in ``test_v14_wiring.py`` — that gap is now closed by the default
:class:`speech_decoding.extractors.view.LogStftView` plus
:class:`speech_decoding.extractors.valid_mask.ElectrodeValidMask` plus
``c_max``-padded :class:`V14DKHardSupportExtractor`.
"""

from __future__ import annotations

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
    assert extractors["support"].c_max == 120
    assert extractors["valid_mask"].c_max == 120


def test_dispatch_default_sets_x_name_tuple_with_mask(
    tmp_path, monkeypatch,
) -> None:
    """v14 BrainModule x_name unpacks (electrode_tokens, support, valid_mask)."""
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    xp = dispatch_v14.build_v14_experiment(mode="nano")

    assert tuple(xp.x_name) == ("electrode_tokens", "support", "valid_mask")


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
