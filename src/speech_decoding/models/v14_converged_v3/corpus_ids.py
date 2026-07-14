"""Cross-corpus subject-id allocation for v3 multi-corpus pretraining.

The v3 spec-cache key embeds an integer ``subject_id`` that the consumer
regex-parses back out (``cache_index.parse_key_session``: ``"subject_id":\\s*(\\d+)``).
With more than one corpus feeding a single training run, those ints MUST be
globally disjoint — otherwise Cogan ``D23`` and some BT/RAM session could share
a ``subject_id`` and collide in the key namespace (and in every per-subject map
keyed on it: parcel_fn, guard-1, localization).

This module owns the disjoint blocks + the deterministic per-corpus
native-id → global-int maps. BrainTreebank keeps its historical ``1..10``
untouched (it predates multi-corpus and its caches are already on disk).

Native sampling rate is deliberately NOT tracked here. Every loader reads
``sfreq`` from its own file header — EDF/BIDS/NWB all carry it authoritatively,
and ``mne.io.read_raw_edf(preload=False)`` reads only the header — which is
cheaper than a hardcoded ``{id: rate}`` table and cannot drift from it. The
loader then resamples to the v3 canonical 2048 Hz. So a "rate registry" is a
non-goal; the ID scheme below is the whole contract.

PROPOSED CONVENTION (Ben may veto): 1000-wide blocks, human-readable where the
native id is a small int (Cogan ``D<n>`` → ``1000+n``; Cogitate ``sub-<n>`` →
``2000+n``), enumerated where it is not (RAM ``R1001P``-style → ``3000+index``,
the index→R-id map persisted alongside the RAM manifest so it is reproducible).
"""

from __future__ import annotations

from collections.abc import Sequence

# Inclusive (lo, hi) global-id block per corpus. Disjoint by construction;
# checked at import. Each corpus's usable count sits well inside its 1000 span
# (BT 10, Cogan ~86/D#≤140, Cogitate 38, RAM 383).
CORPUS_BLOCKS: dict[str, tuple[int, int]] = {
    "bt": (1, 999),
    "cogan": (1000, 1999),
    "cogitate": (2000, 2999),
    "ram": (3000, 3999),
}


def _block(corpus: str) -> tuple[int, int]:
    try:
        return CORPUS_BLOCKS[corpus]
    except KeyError:
        raise ValueError(
            f"unknown corpus {corpus!r}; known: {sorted(CORPUS_BLOCKS)}"
        ) from None


def cogan_global_id(d_num: int, implant: str = "") -> int:
    """Cogan ``D<n>`` patient (+ optional re-implant letter) → global ``subject_id``.

    Bare ``D23`` → ``1023``. A re-implant (the corpus has exactly one, ``D107A``)
    is a LATER surgery with a new electrode set — distinct coordinates, hence a
    distinct parcel_fn / guard-1 map / localization, all keyed on ``subject_id``
    — so it must NOT share ``D107``'s id. It folds into the top half of the block:
    ``D107A`` → ``1500 + 107 = 1607``, disjoint from bare ``D107`` → ``1107``.

    Bare ids occupy ``[1001, 1499]`` (d_num ≤ 499), re-implants ``[1501, 1999]``.
    The 1000-wide block fits bare + ONE re-implant letter per d_num; a second
    letter (``B``) raises — widen the Cogan block if Cogan ever adds one.
    """
    lo, _ = _block("cogan")  # (1000, 1999)
    n = int(d_num)
    if not (1 <= n <= 499):
        raise ValueError(f"cogan d_num {d_num} out of bare band [1, 499]")
    implant = (implant or "").strip().upper()
    if implant == "":
        return lo + n            # 1000 + n ∈ [1001, 1499]
    if implant == "A":
        return lo + 500 + n      # 1500 + n ∈ [1501, 1999]
    raise ValueError(
        f"cogan implant suffix {implant!r} unsupported: the 1000-wide block fits "
        "only bare + one re-implant ('A'); widen the block to add more"
    )


def cogitate_global_id(sub_num: int) -> int:
    """Cogitate ``sub-<n>`` number → global ``subject_id`` (``sub-1`` → ``2001``)."""
    lo, hi = _block("cogitate")
    gid = lo + int(sub_num)
    if not (1 <= sub_num <= hi - lo):
        raise ValueError(f"cogitate sub_num {sub_num} out of block {(lo, hi)}")
    return gid


def allocate_ram_ids(sorted_unique_rids: Sequence[str]) -> dict[str, int]:
    """RAM native ids (``R1001P`` …) → global ints, ``3000 + index``.

    RAM ids have no small-int core we can reuse cleanly across sites, so we
    enumerate. Requires an already-sorted, de-duplicated list (the RAM manifest
    step owns dedup across ds004789/ds004809 and the sort) so the mapping is a
    pure function of the input order — persist this dict beside the manifest to
    keep it reproducible.
    """
    rids = list(sorted_unique_rids)
    if len(rids) != len(set(rids)):
        raise ValueError("ram rids must be de-duplicated before allocation")
    if rids != sorted(rids):
        raise ValueError("ram rids must be sorted before allocation")
    lo, hi = _block("ram")
    if len(rids) > hi - lo + 1:
        raise ValueError(
            f"{len(rids)} RAM subjects exceed block {(lo, hi)} capacity {hi - lo + 1}"
        )
    return {rid: lo + i for i, rid in enumerate(rids)}


def corpus_of(global_id: int) -> str:
    """Which corpus a global ``subject_id`` belongs to (inverse of the blocks)."""
    for corpus, (lo, hi) in CORPUS_BLOCKS.items():
        if lo <= global_id <= hi:
            return corpus
    raise ValueError(f"global_id {global_id} in no corpus block")


def _assert_blocks_disjoint() -> None:
    spans = sorted(CORPUS_BLOCKS.items(), key=lambda kv: kv[1])
    for (a_name, (_, a_hi)), (b_name, (b_lo, _)) in zip(spans, spans[1:]):
        if a_hi >= b_lo:
            raise RuntimeError(
                f"corpus blocks overlap: {a_name} {CORPUS_BLOCKS[a_name]} vs "
                f"{b_name} {CORPUS_BLOCKS[b_name]}"
            )


_assert_blocks_disjoint()
