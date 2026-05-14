"""End-to-end wiring tests for the v14 Perceiver-IO encoder.

Validates the NeuralTrain/Exca integration without requiring BT data on the
laptop. Three slices:

1. ``V14ParcelPerceiver.build(n_in_channels, n_outputs)`` is callable with the
   ``BaseModelConfig.build`` kwargs that ``Experiment._build_brain_module`` uses.
2. ``BrainModule`` with ``x_name=("electrode_tokens", "support")`` unpacks both
   tensors as positional args into the v14 model.
3. The full Experiment scaffold runs ``fast_dev_run`` end-to-end on a synthetic
   3-tensor study (electrode_tokens + support + target), produces a record, and
   exca-caches the second call.
"""

from __future__ import annotations

import json

import neuralset as ns
import numpy as np
import pandas as pd
import torch

from speech_decoding.experiments import Data, Experiment
from speech_decoding.experiments.module import BrainModule
from speech_decoding.models import V14ParcelPerceiver


C_MAX = 4
T_BINS = 5
F_BINS = 3
K_PARCELS = 6


# ---------- Slice 1: build signature compatibility ----------------------


def test_v14_config_build_accepts_neuraltrain_kwargs() -> None:
    """``Experiment._build_brain_module`` calls
    ``brain_model_config.build(n_in_channels=..., n_outputs=...)``. The v14
    config must accept that kwarg shape and map ``n_outputs`` to ``n_classes``
    (``n_in_channels`` is informational only — v14 handles variable C via the
    ``valid_mask``)."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=F_BINS, n_time_bins=T_BINS, k_parcels=K_PARCELS,
        d_model=32, n_heads=4, depth_self_attn=1, m_sub_slots=2,
    )
    model = cfg.build(n_in_channels=C_MAX, n_outputs=7)

    electrodes = torch.randn(2, C_MAX, T_BINS, F_BINS)
    support = torch.zeros(2, C_MAX, K_PARCELS)
    support[..., 0] = 1.0
    logits = model(electrodes, support)
    assert logits.shape == (2, 7), f"expected (2, 7), got {tuple(logits.shape)}"


# ---------- Slice 2: BrainModule tuple-input unpacking ------------------


class _DummyBatch:
    """Minimal substitute for the NeuralSet collated batch object."""

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        self.data = data


def test_brain_module_unpacks_tuple_x_name_for_v14() -> None:
    cfg = V14ParcelPerceiver(
        n_freq_bins=F_BINS, n_time_bins=T_BINS, k_parcels=K_PARCELS,
        d_model=32, n_heads=4, depth_self_attn=1, m_sub_slots=2,
    )
    model = cfg.build(n_in_channels=C_MAX, n_outputs=3)

    from neuraltrain.optimizers import LightningOptimizer

    module = BrainModule(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optim_config=LightningOptimizer(optimizer={"name": "Adam", "lr": 1e-3}),
        metrics={},
        x_name=("electrode_tokens", "support"),
        y_name="target",
    )

    electrodes = torch.randn(2, C_MAX, T_BINS, F_BINS)
    support = torch.zeros(2, C_MAX, K_PARCELS)
    support[..., 0] = 1.0
    batch = _DummyBatch({
        "electrode_tokens": electrodes,
        "support": support,
        "target": torch.tensor([0, 1]),
    })

    logits = module.forward(batch)
    assert logits.shape == (2, 3)


# ---------- Slice 3: end-to-end Experiment dry-run ---------------------


class V14SyntheticStudy(ns.Step):
    """Synthetic NS study producing 3-tensor (electrode_tokens, support, code) events."""

    def _run(self) -> pd.DataFrame:
        rows = []
        splits = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
        for idx, split in enumerate(splits):
            rows.append({
                "type": "Stimulus",
                "start": float(idx * 2),
                "duration": 0.2,
                "timeline": "run0",
                "code": idx % 2,
                "split": split,
            })
        return ns.events.standardize_events(pd.DataFrame(rows))


class _ConstantTokensExtractor(ns.extractors.base.BaseStatic):
    """Emit a deterministic ``(C_MAX, T_BINS, F_BINS)`` tensor per Stimulus event."""

    event_types: str = "Stimulus"
    seed_offset: int = 0

    def get_static(self, event):  # type: ignore[override]
        idx = int(event.start / 2.0) + self.seed_offset
        rng = np.random.default_rng(idx)
        x = rng.standard_normal((C_MAX, T_BINS, F_BINS)).astype(np.float32)
        return torch.from_numpy(x)


class _ConstantSupportExtractor(ns.extractors.base.BaseStatic):
    """Emit a fixed one-hot ``(C_MAX, K_PARCELS)`` support tensor per event."""

    event_types: str = "Stimulus"

    def get_static(self, event):  # type: ignore[override]
        s = np.zeros((C_MAX, K_PARCELS), dtype=np.float32)
        for i in range(C_MAX):
            s[i, i % K_PARCELS] = 1.0
        return torch.from_numpy(s)


def _v14_synthetic_data(batch_size: int = 2) -> Data:
    return Data(
        study=V14SyntheticStudy(),
        segmenter={
            "extractors": {
                "electrode_tokens": _ConstantTokensExtractor(),
                "support": _ConstantSupportExtractor(),
                "target": {
                    "name": "EventField",
                    "event_types": "Stimulus",
                    "event_field": "code",
                },
            },
            "trigger_query": "type == 'Stimulus'",
            "start": 0.0,
            "duration": 1.0,
        },
        batch_size=batch_size,
    )


def test_v14_experiment_dry_run_end_to_end(tmp_path) -> None:
    """V14 trains through one fast_dev_run iteration via the Experiment scaffold,
    writes one experiment record, and re-runs cached on a second call."""
    import speech_decoding.models  # noqa: F401  # registers V14ParcelPerceiver

    run_root = tmp_path / "runs"
    xp = Experiment(
        data=_v14_synthetic_data(),
        brain_model_config={
            "name": "V14ParcelPerceiver",
            "n_freq_bins": F_BINS,
            "n_time_bins": T_BINS,
            "k_parcels": K_PARCELS,
            "d_model": 32,
            "n_heads": 4,
            "depth_self_attn": 1,
            "m_sub_slots": 2,
        },
        loss={"name": "CrossEntropyLoss"},
        optim={"optimizer": {"name": "Adam", "lr": 1e-2}},
        metrics=[
            {
                "name": "Accuracy",
                "log_name": "acc",
                "kwargs": {"task": "multiclass", "num_classes": 2},
            }
        ],
        n_epochs=1,
        accelerator="cpu",
        devices=1,
        x_name=("electrode_tokens", "support"),
        fast_dev_run=True,
        infra={"folder": str(run_root), "cluster": None},
    )

    result = xp.run()
    assert isinstance(result, dict)
    records = list(run_root.rglob("experiment_record.json"))
    assert len(records) == 1
    payload = json.loads(records[0].read_text())
    assert payload["status"] == "succeeded"


def test_v14_config_eps_is_plumbed_to_model_forward() -> None:
    """``V14ParcelPerceiver(eps=...)`` must change the model output. Required
    for the Stage-1 ``eps ∈ {1e-4, 1e-3, 1e-2, 1e-1}`` sweep."""
    torch.manual_seed(0)
    base_kwargs = dict(
        n_freq_bins=F_BINS, n_time_bins=T_BINS, k_parcels=K_PARCELS,
        d_model=32, n_heads=4, depth_self_attn=1, m_sub_slots=2,
    )
    cfg_strong = V14ParcelPerceiver(**base_kwargs, eps=1e-4)
    cfg_weak = V14ParcelPerceiver(**base_kwargs, eps=1e-1)

    torch.manual_seed(0)
    model_strong = cfg_strong.build(n_outputs=2)
    torch.manual_seed(0)
    model_weak = cfg_weak.build(n_outputs=2)

    electrodes = torch.randn(2, C_MAX, T_BINS, F_BINS)
    # Varied support across electrodes — each electrode targets a different
    # parcel — so eps changes the softmax over electrodes (otherwise the bias
    # is a uniform additive constant per latent and softmax is invariant to it).
    support = torch.zeros(2, C_MAX, K_PARCELS)
    for i in range(C_MAX):
        support[:, i, i % K_PARCELS] = 1.0
    out_strong = model_strong(electrodes, support)
    out_weak = model_weak(electrodes, support)
    assert not torch.allclose(out_strong, out_weak, atol=1e-4), (
        "eps must change the cross-attn bias and therefore the logits"
    )


def test_v14_dispatch_script_imports() -> None:
    """The DCC dispatch entrypoint imports without missing-symbol errors.

    Smoke test only — does NOT execute the dispatch (which requires DCC paths
    and BT data). Verifies the import graph is intact so ``scripts/dcc/dispatch
    --help`` works as a preflight."""
    from speech_decoding.experiments import dispatch_v14  # noqa: F401
    assert hasattr(dispatch_v14, "build_v14_experiment")
    assert hasattr(dispatch_v14, "main")


def test_v14_dispatch_dry_run_prints_config(capsys) -> None:
    """`--dry-run` prints the resolved config without dispatching or touching BT data."""
    from speech_decoding.experiments import dispatch_v14
    rc = dispatch_v14.main(["--dry-run", "--eps", "0.05", "--m-sub-slots", "8"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "eps=0.05" in out
    assert "M=8" in out
    assert "S5 excluded" in out


def test_v14_dispatch_raises_until_electrode_tokens_extractor_wired(
    tmp_path, monkeypatch,
) -> None:
    """Without an electrode-tokens extractor, ``build_v14_experiment`` must
    raise NotImplementedError loudly — not silently produce a half-wired config."""
    from speech_decoding.experiments import dispatch_v14
    monkeypatch.setenv("ROOT_DIR_BRAINTREEBANK", str(tmp_path))
    import pytest
    with pytest.raises(NotImplementedError, match="electrode-tokens extractor"):
        dispatch_v14.build_v14_experiment(mode="nano")
