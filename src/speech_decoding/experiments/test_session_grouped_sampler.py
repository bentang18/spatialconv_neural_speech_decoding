"""Unit tests for the session-homogeneous batch sampler (throughput lever S1).

The sampler must (a) never mix two sessions in one batch, (b) cover every sample
exactly once per epoch with drop_last=False, (c) be deterministic given a seed,
(d) reshuffle across epochs via set_epoch, and (e) self-shard across DDP ranks
(equal batch count per rank, disjoint partition that covers every sample)."""
from __future__ import annotations

from speech_decoding.experiments.data import _SessionGroupedBatchSampler


def _keys(n_per_session: dict) -> list[tuple]:
    keys: list[tuple] = []
    for sess, n in n_per_session.items():
        keys.extend([sess] * n)
    return keys


def test_batches_are_session_homogeneous():
    keys = _keys({(1, 0): 10, (2, 1): 7, (3, 0): 5})
    s = _SessionGroupedBatchSampler(keys, 4, shuffle=True, drop_last=False, seed=0)
    for batch in s:
        sessions = {keys[i] for i in batch}
        assert len(sessions) == 1, f"batch spans sessions {sessions}"


def test_covers_every_sample_once_no_drop():
    keys = _keys({(1, 0): 10, (2, 1): 7, (3, 0): 5})
    s = _SessionGroupedBatchSampler(keys, 4, shuffle=True, drop_last=False, seed=1)
    seen = [i for batch in s for i in batch]
    assert sorted(seen) == list(range(len(keys)))
    assert len(s) == sum(1 for _ in s)


def test_drop_last_drops_only_partial_tail():
    keys = _keys({(1, 0): 10, (2, 1): 7})  # 10->2 full+1 partial; 7->1 full+1 partial
    s = _SessionGroupedBatchSampler(keys, 4, shuffle=False, drop_last=True, seed=2)
    batches = list(s)
    assert all(len(b) == 4 for b in batches)
    assert len(batches) == 2 + 1  # floor(10/4)=2, floor(7/4)=1
    assert len(s) == len(batches)


def test_deterministic_under_same_seed():
    keys = _keys({(1, 0): 9, (2, 1): 9})
    a = list(_SessionGroupedBatchSampler(
        keys, 4, shuffle=True, drop_last=False, seed=7))
    b = list(_SessionGroupedBatchSampler(
        keys, 4, shuffle=True, drop_last=False, seed=7))
    assert a == b


def test_reshuffles_across_epochs():
    keys = _keys({(1, 0): 12, (2, 1): 12, (3, 0): 12})
    s = _SessionGroupedBatchSampler(keys, 4, shuffle=True, drop_last=False, seed=3)
    epoch1 = list(s)
    s.set_epoch(1)
    epoch2 = list(s)
    # same coverage, different order (batch order + within-session order reshuffle)
    assert sorted(i for b in epoch1 for i in b) == sorted(i for b in epoch2 for i in b)
    assert epoch1 != epoch2


# ---- DDP self-sharding ----------------------------------------------------


def test_single_process_default_is_unsharded():
    """world_size=1 (the default, no explicit num_replicas) == legacy behavior."""
    keys = _keys({(1, 0): 10, (2, 1): 7, (3, 0): 5})
    full = list(_SessionGroupedBatchSampler(
        keys, 4, shuffle=True, drop_last=False, seed=5))
    w1 = list(_SessionGroupedBatchSampler(
        keys, 4, shuffle=True, drop_last=False, seed=5,
        num_replicas=1, rank=0))
    assert full == w1


def test_ddp_ranks_build_identical_full_list():
    """Every rank's underlying full batch list is rank-independent (seed only)."""
    keys = _keys({(1, 0): 10, (2, 1): 7, (3, 0): 5})
    full = [
        _SessionGroupedBatchSampler(
            keys, 4, shuffle=True, drop_last=False, seed=9,
            num_replicas=4, rank=r,
        )._all_batches()
        for r in range(4)
    ]
    assert all(fb == full[0] for fb in full)


def test_ddp_equal_batches_per_rank_no_uneven_hang():
    """Each rank yields the SAME number of batches (DDP uneven-input backstop)."""
    keys = _keys({(1, 0): 10, (2, 1): 7, (3, 0): 5})
    world = 4
    counts = [
        len(list(_SessionGroupedBatchSampler(
            keys, 4, shuffle=True, drop_last=False, seed=11,
            num_replicas=world, rank=r)))
        for r in range(world)
    ]
    assert len(set(counts)) == 1, f"unequal per-rank batch counts {counts}"
    # __len__ matches the per-rank yield.
    one = _SessionGroupedBatchSampler(
        keys, 4, shuffle=True, drop_last=False, seed=11,
        num_replicas=world, rank=0)
    assert len(one) == counts[0]


def test_ddp_shards_cover_every_sample_and_stay_homogeneous():
    keys = _keys({(1, 0): 10, (2, 1): 7, (3, 0): 5})
    world = 4
    all_samples: set[int] = set()
    for r in range(world):
        s = _SessionGroupedBatchSampler(
            keys, 4, shuffle=True, drop_last=False, seed=13,
            num_replicas=world, rank=r)
        for batch in s:
            assert len({keys[i] for i in batch}) == 1  # still homogeneous
            all_samples.update(batch)
    # union over ranks covers every sample (wrap-pad may duplicate a few).
    assert all_samples == set(range(len(keys)))


def test_ddp_shards_are_a_disjoint_partition_of_padded_list():
    """The real (pre-pad) batches partition disjointly across ranks by position."""
    keys = _keys({(1, 0): 12, (2, 1): 8, (3, 0): 4})  # 3+2+1 = 6 batches, %4 -> pad
    world = 4
    ref = _SessionGroupedBatchSampler(
        keys, 4, shuffle=True, drop_last=False, seed=17,
        num_replicas=world, rank=0)._all_batches()
    n = len(ref)
    # collect (rank -> list of batches), then verify the first n stride positions
    # reconstruct the full list exactly (positions 0..n-1, no gaps, no overlap).
    reconstructed: dict[int, list] = {}
    per_rank = (n + world - 1) // world
    for r in range(world):
        batches = list(_SessionGroupedBatchSampler(
            keys, 4, shuffle=True, drop_last=False, seed=17,
            num_replicas=world, rank=r))
        for j, b in enumerate(batches):
            pos = j * world + r
            reconstructed[pos] = b
    for pos in range(n):
        assert reconstructed[pos] == ref[pos]
    assert per_rank * world == len(reconstructed)  # every padded slot filled once
