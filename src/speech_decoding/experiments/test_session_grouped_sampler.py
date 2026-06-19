"""Unit tests for the session-homogeneous batch sampler (throughput lever S1).

The sampler must (a) never mix two sessions in one batch, (b) cover every sample
exactly once per epoch with drop_last=False, (c) be deterministic given a seeded
generator, and (d) reshuffle across epochs."""
from __future__ import annotations

import torch

from speech_decoding.experiments.data import _SessionGroupedBatchSampler


def _keys(n_per_session: dict) -> list[tuple]:
    keys: list[tuple] = []
    for sess, n in n_per_session.items():
        keys.extend([sess] * n)
    return keys


def test_batches_are_session_homogeneous():
    keys = _keys({(1, 0): 10, (2, 1): 7, (3, 0): 5})
    g = torch.Generator().manual_seed(0)
    s = _SessionGroupedBatchSampler(keys, 4, shuffle=True, drop_last=False, generator=g)
    for batch in s:
        sessions = {keys[i] for i in batch}
        assert len(sessions) == 1, f"batch spans sessions {sessions}"


def test_covers_every_sample_once_no_drop():
    keys = _keys({(1, 0): 10, (2, 1): 7, (3, 0): 5})
    g = torch.Generator().manual_seed(1)
    s = _SessionGroupedBatchSampler(keys, 4, shuffle=True, drop_last=False, generator=g)
    seen = [i for batch in s for i in batch]
    assert sorted(seen) == list(range(len(keys)))
    assert len(s) == sum(1 for _ in s)


def test_drop_last_drops_only_partial_tail():
    keys = _keys({(1, 0): 10, (2, 1): 7})  # 10->2 full+1 partial; 7->1 full+1 partial
    g = torch.Generator().manual_seed(2)
    s = _SessionGroupedBatchSampler(keys, 4, shuffle=False, drop_last=True, generator=g)
    batches = list(s)
    assert all(len(b) == 4 for b in batches)
    assert len(batches) == 2 + 1  # floor(10/4)=2, floor(7/4)=1
    assert len(s) == len(batches)


def test_deterministic_under_same_seed():
    keys = _keys({(1, 0): 9, (2, 1): 9})
    a = list(_SessionGroupedBatchSampler(
        keys, 4, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(7)))
    b = list(_SessionGroupedBatchSampler(
        keys, 4, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(7)))
    assert a == b


def test_reshuffles_across_epochs():
    keys = _keys({(1, 0): 12, (2, 1): 12, (3, 0): 12})
    g = torch.Generator().manual_seed(3)
    s = _SessionGroupedBatchSampler(keys, 4, shuffle=True, drop_last=False, generator=g)
    epoch1 = list(s)
    epoch2 = list(s)
    # same coverage, different order (batch order + within-session order reshuffle)
    assert sorted(i for b in epoch1 for i in b) == sorted(i for b in epoch2 for i in b)
    assert epoch1 != epoch2
