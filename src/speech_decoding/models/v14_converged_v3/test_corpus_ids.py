"""Tests for cross-corpus subject-id allocation (corpus_ids)."""

from __future__ import annotations

import pytest

from speech_decoding.models.v14_converged_v3 import corpus_ids as cid


def test_blocks_disjoint_and_bt_preserved():
    # BT's historical ids 1..10 must still resolve to "bt" (caches on disk).
    for sid in range(1, 11):
        assert cid.corpus_of(sid) == "bt"
    # No import-time overlap (constructor asserts, but pin it explicitly too).
    cid._assert_blocks_disjoint()


def test_cogan_ids_human_readable_and_disjoint():
    assert cid.cogan_global_id(23) == 1023
    assert cid.cogan_global_id(24) == 1024
    assert cid.corpus_of(cid.cogan_global_id(140)) == "cogan"
    # never collides with BT
    assert cid.cogan_global_id(1) == 1001 != 1


def test_cogan_reimplant_folds_into_top_half():
    # The real corpus has both bare D107 and re-implant D107A — distinct brains,
    # distinct electrode sets, so distinct ids in the same cogan block.
    assert cid.cogan_global_id(107) == 1107
    assert cid.cogan_global_id(107, "A") == 1607
    assert cid.cogan_global_id(107, "a") == 1607  # case-insensitive
    assert cid.corpus_of(1607) == "cogan"
    # a second re-implant letter has no room in the 1000-wide block → loud
    with pytest.raises(ValueError, match="unsupported"):
        cid.cogan_global_id(107, "B")
    # bare d_num must clear the re-implant band
    with pytest.raises(ValueError, match="bare band"):
        cid.cogan_global_id(500)


def test_cogitate_ids():
    assert cid.cogitate_global_id(1) == 2001
    assert cid.corpus_of(cid.cogitate_global_id(38)) == "cogitate"


def test_ram_allocation_is_pure_and_ordered():
    rids = ["R1001P", "R1002P", "R1060M", "R1467M"]
    m = cid.allocate_ram_ids(rids)
    assert m == {"R1001P": 3000, "R1002P": 3001, "R1060M": 3002, "R1467M": 3003}
    assert all(cid.corpus_of(g) == "ram" for g in m.values())


def test_ram_requires_sorted_unique():
    with pytest.raises(ValueError, match="de-duplicated"):
        cid.allocate_ram_ids(["R1001P", "R1001P"])
    with pytest.raises(ValueError, match="sorted"):
        cid.allocate_ram_ids(["R1002P", "R1001P"])


def test_out_of_range_and_capacity():
    with pytest.raises(ValueError):
        cid.cogan_global_id(0)
    with pytest.raises(ValueError):
        cid.cogan_global_id(1000)  # would spill into cogitate block
    with pytest.raises(ValueError, match="exceed block"):
        cid.allocate_ram_ids([f"R{i:04d}" for i in range(1001)])


def test_corpus_of_rejects_gap():
    with pytest.raises(ValueError, match="no corpus block"):
        cid.corpus_of(10_000)
    with pytest.raises(ValueError):
        cid.corpus_of(0)


def test_unknown_corpus():
    with pytest.raises(ValueError, match="unknown corpus"):
        cid._block("nope")
