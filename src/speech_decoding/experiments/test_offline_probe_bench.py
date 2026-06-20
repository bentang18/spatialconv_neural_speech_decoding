"""Tests for the offline probe bench's pure logic (offline_probe_bench).

Only the token-source-generic WS/CS logistic loop is laptop-testable (the forward,
ridge-timing, ckpt-load, and driver need a real model + GPU + BT voltage). This
checks it on a synthetic ProbeDataset: the d=1 raw path AND a d=256 encoder-shaped
path go through the same loop (the comparison is only meaningful if both work), the
emitted keys/ranges match online_probe naming, and the result is deterministic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

import pytest

from speech_decoding.experiments.offline_probe_bench import (
    bench_logistic_head_to_head,
    logistic_ws_cs_from_tokens,
    logistic_ws_cs_streaming,
)
from speech_decoding.experiments.online_probe import SubjectProbeData
from speech_decoding.experiments.online_probe_raw_baseline import raw_tokens_from_bands
from speech_decoding.models.v14_converged import BANDS, N_TOKENS


@dataclasses.dataclass
class _FakeDataset:
    ws_subjects: list[int]
    cs_anchor: int
    cs_test_subjects: list[int]
    tasks: list[str]
    n_parcels: int
    _data: dict[int, SubjectProbeData]

    def subject_data(self, subject_id: int) -> SubjectProbeData:
        return self._data[subject_id]


def _subject(subject_id: int, n: int, c: int, n_parcels: int, *, seed: int):
    g = torch.Generator().manual_seed(seed)
    half = np.array([1.0, -1.0] * (n // 2), dtype=np.float64)[:n]
    labels = {"delta_volume": half.copy()}
    slow = torch.randn(n, c, 2, BANDS[0].n_freq_bins, BANDS[0].n_time_frames, generator=g)
    slow[:, :, 0] += torch.tensor(half, dtype=torch.float32)[:, None, None, None]
    beta = torch.randn(n, c, BANDS[1].n_freq_bins, BANDS[1].n_time_frames, generator=g).abs()
    hg = torch.randn(n, c, BANDS[2].n_freq_bins, BANDS[2].n_time_frames, generator=g).abs()
    ppe = torch.arange(c) % n_parcels
    mask = torch.ones(c, dtype=torch.bool)
    return SubjectProbeData(
        subject_id=subject_id, slow=slow, beta=beta, hg=hg,
        parcel_per_electrode=ppe, electrode_mask=mask, labels=labels,
        sessions=[(subject_id, 0)],
    )


def _fake():
    n_parcels = 4
    data = {
        2: _subject(2, 8, 6, n_parcels, seed=0),
        10: _subject(10, 8, 5, n_parcels, seed=1),
        11: _subject(11, 8, 5, n_parcels, seed=2),
    }
    ds = _FakeDataset(
        ws_subjects=[10, 11], cs_anchor=2, cs_test_subjects=[10],
        tasks=["delta_volume"], n_parcels=n_parcels, _data=data,
    )
    sd = {s: ds.subject_data(s) for s in (2, 10, 11)}
    return ds, sd


def _raw_tokens(sd):
    return {s: raw_tokens_from_bands(d.slow, d.beta, d.hg) for s, d in sd.items()}


def test_keys_and_ranges_raw_d1():
    ds, sd = _fake()
    out = logistic_ws_cs_from_tokens(ds, _raw_tokens(sd), sd, n_parcels=4, tap="raw")
    assert set(out) == {
        "val_probe/raw/ws/delta_volume",
        "val_probe/raw/cs/delta_volume",
        "val_probe/raw/gap/delta_volume",
    }
    ws, cs = out["val_probe/raw/ws/delta_volume"], out["val_probe/raw/cs/delta_volume"]
    assert 0.0 <= ws <= 1.0 and 0.0 <= cs <= 1.0
    assert out["val_probe/raw/gap/delta_volume"] == ws - cs


def test_deterministic():
    ds, sd = _fake()
    a = logistic_ws_cs_from_tokens(ds, _raw_tokens(sd), sd, n_parcels=4, tap="raw")
    b = logistic_ws_cs_from_tokens(ds, _raw_tokens(sd), sd, n_parcels=4, tap="raw")
    assert a == b


def test_encoder_shaped_d256_path_runs():
    # The loop must be shape-generic: a (N,C,N_TOKENS,d>1) "encoder" token tensor
    # (label signal injected into a few dims) goes through the same WS/CS path and
    # yields finite, in-range AUROCs under the 'latent' tap name.
    ds, sd = _fake()
    d = 8
    tokens = {}
    for s, dd in sd.items():
        n, c = dd.slow.shape[0], dd.slow.shape[1]
        g = torch.Generator().manual_seed(100 + s)
        t = torch.randn(n, c, N_TOKENS, d, generator=g)
        t[..., 0] += torch.tensor(dd.labels["delta_volume"], dtype=torch.float32)[
            :, None, None
        ]
        tokens[s] = t
    out = logistic_ws_cs_from_tokens(ds, tokens, sd, n_parcels=4, tap="latent", max_iter=200)
    assert set(out) == {
        "val_probe/latent/ws/delta_volume",
        "val_probe/latent/cs/delta_volume",
        "val_probe/latent/gap/delta_volume",
    }
    assert np.isfinite(out["val_probe/latent/ws/delta_volume"])
    assert 0.0 <= out["val_probe/latent/ws/delta_volume"] <= 1.0


def test_streaming_matches_all_at_once():
    # logistic_ws_cs_streaming holds one subject's tokens at a time (the memory fix
    # for the d=256 encoder OOM) but must produce identical numbers to the
    # all-at-once logistic_ws_cs_from_tokens. A builder that returns precomputed
    # tokens isolates the streaming control flow from any token difference.
    ds, sd = _fake()
    tokens = _raw_tokens(sd)
    ref = logistic_ws_cs_from_tokens(ds, tokens, sd, n_parcels=4, tap="raw")
    streamed = logistic_ws_cs_streaming(
        ds, lambda s: tokens[s], sd, n_parcels=4, tap="raw"
    )
    assert streamed == ref


def test_head_to_head_taps_filter_raw_only():
    # taps=("raw",) scores only the raw |STFT| baseline and never touches the model
    # (the encoder taps are the expensive forwards). device is passed so the
    # `next(model.parameters())` device probe is short-circuited → model=None is safe.
    ds, _ = _fake()
    out = bench_logistic_head_to_head(
        None, ds, device=torch.device("cpu"), taps=("raw",)
    )
    assert set(out["timings"]) == {"raw"}
    assert out["metrics"]
    assert all(k.split("/")[1] == "raw" for k in out["metrics"])


def test_head_to_head_rejects_unknown_tap():
    ds, _ = _fake()
    with pytest.raises(ValueError, match="unknown tap"):
        bench_logistic_head_to_head(
            None, ds, device=torch.device("cpu"), taps=("raw", "bogus")
        )
