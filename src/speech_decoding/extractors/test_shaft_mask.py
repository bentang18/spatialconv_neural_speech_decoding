"""Tests for ShaftMaskExtractor (Phase-2 shaft-mask 5/27 PM final spec).

Default ``K = 1 if N_shafts >= 2 else 0``; default ``extent_blocks =
("alpha",)``. Block α only; β / γ live in the module for sister wiring.
Output is a ``(C,)`` bool tensor: ``True`` at shaft-blocked electrode
positions.

Sister cells:

* ``R-shaft-K2`` — ``k_override=2, extent_blocks=("alpha", "beta")``.
* ``R-shaft-K3-mixed-3block`` — ``k_override=3,
  extent_blocks=("alpha", "beta", "gamma")``.
* ``R-shaft-K1-explicit`` — ``k_override=1``.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.extractors.shaft_mask import (
    BLOCK_ALPHA,
    BLOCK_BETA,
    BLOCK_GAMMA,
    ShaftMaskExtractor,
    SubjectAnatomy,
    default_k,
)


def _make_anatomy(shaft_sizes: list[int]) -> SubjectAnatomy:
    """Build a SubjectAnatomy from a list of per-shaft electrode counts."""
    n_total = sum(shaft_sizes)
    shaft_ids = torch.zeros(n_total, dtype=torch.long)
    contact_pos = torch.zeros(n_total, dtype=torch.float32)
    i = 0
    for sid, size in enumerate(shaft_sizes):
        for j in range(size):
            shaft_ids[i] = sid
            contact_pos[i] = j / max(size - 1, 1)  # 0..1 along the shaft
            i += 1
    return SubjectAnatomy(shaft_ids=shaft_ids, contact_pos=contact_pos)


# ---------- default_k contract ----------------------------------------


def test_default_k_zero_at_n_shafts_zero_or_one() -> None:
    """5/27 PM lock: ``K = 0`` when ``N_shafts < 2`` — single guard
    closes the catastrophic-drop hole at N_shafts == 1."""
    assert default_k(0) == 0
    assert default_k(1) == 0


def test_default_k_one_at_n_shafts_ge_2() -> None:
    """5/27 PM lock: ``K = 1`` for every ``N_shafts >= 2`` — no
    fraction, no ceiling. The previous AM ``min(2, ceil(0.25 * N))``
    formula and the original K=3 spec are both retired."""
    for n in (2, 3, 4, 5, 7, 8, 10, 15, 20, 30, 50):
        assert default_k(n) == 1, f"N_shafts={n}: expected K=1, got {default_k(n)}"


# ---------- ShaftMaskExtractor default --------------------------------


def test_extractor_default_extent_blocks_are_alpha_only() -> None:
    """5/27 PM lock: default ``extent_blocks=("alpha",)``. β and γ are
    available in the module but NOT in the default active list."""
    ext = ShaftMaskExtractor(seed=0)
    assert ext.block_extents == [BLOCK_ALPHA]
    assert BLOCK_BETA not in ext.block_extents
    assert BLOCK_GAMMA not in ext.block_extents


def test_extractor_k1_for_bt_shaft_range_7_to_10() -> None:
    """BT subjects (7–10 shafts) now get K=1 (was K=2 under the AM
    fraction spec). Block α only is exercised."""
    for n_shafts in range(7, 11):
        anatomy = _make_anatomy([8] * n_shafts)
        ext = ShaftMaskExtractor(seed=0)
        assert ext._k_for_anatomy(anatomy) == 1, (
            f"N_shafts={n_shafts}: expected K=1 under the 5/27 PM lock"
        )


def test_extractor_k1_for_dcohort_large_shaft_range() -> None:
    """D-cohort subjects (≫ 8 shafts) still get K=1 — same default
    everywhere; the formula doesn't grow with N_shafts anymore."""
    for n_shafts in (10, 15, 20, 30):
        anatomy = _make_anatomy([8] * n_shafts)
        ext = ShaftMaskExtractor(seed=0)
        assert ext._k_for_anatomy(anatomy) == 1


def test_extractor_k0_when_n_shafts_is_1() -> None:
    """5/27 PM guard: a single-shaft subject yields K=0 (the prior AM
    fraction spec would have applied K=1 here, dropping the whole
    subject's electrodes into the shaft block)."""
    anatomy = _make_anatomy([10])  # 1 shaft of 10 contacts
    ext = ShaftMaskExtractor(seed=0)
    assert ext._k_for_anatomy(anatomy) == 0
    mask = ext(anatomy)
    assert not mask.any(), "K=0 must produce an all-False mask"


def test_extractor_returns_c_shaped_bool_tensor() -> None:
    """Output shape matches the total electrode count and is bool."""
    anatomy = _make_anatomy([8, 7, 6, 9, 10, 8, 7])  # 7 shafts, 55 electrodes
    ext = ShaftMaskExtractor(seed=42)
    mask = ext(anatomy)
    assert mask.shape == (55,)
    assert mask.dtype == torch.bool


def test_extractor_deterministic_with_seed() -> None:
    anatomy = _make_anatomy([8] * 8)
    a = ShaftMaskExtractor(seed=0)(anatomy)
    b = ShaftMaskExtractor(seed=0)(anatomy)
    assert torch.equal(a, b)


def test_extractor_different_seeds_can_differ() -> None:
    anatomy = _make_anatomy([8] * 8)
    a = ShaftMaskExtractor(seed=0)(anatomy)
    b = ShaftMaskExtractor(seed=1)(anatomy)
    assert not torch.equal(a, b), (
        "different seeds should produce different masks (probabilistic)"
    )


def test_extractor_block_alpha_present_when_k_ge_1() -> None:
    """When K=1, block α fires → mask has ≥ 1 True position on
    exactly one shaft."""
    anatomy = _make_anatomy([10] * 3)  # 3 shafts → K=1
    ext = ShaftMaskExtractor(seed=0)
    mask = ext(anatomy)
    assert mask.any(), "K=1 must produce at least one shaft-blocked electrode"
    # All masked positions belong to a single shaft (K=1 = one block on
    # one shaft).
    masked_idx = torch.nonzero(mask).flatten()
    shafts_touched = torch.unique(anatomy.shaft_ids[masked_idx])
    assert shafts_touched.numel() == 1, (
        f"K=1 expected to touch exactly 1 shaft, got "
        f"{shafts_touched.numel()} ({shafts_touched.tolist()})"
    )


def test_extractor_block_extent_is_contiguous_on_one_shaft() -> None:
    """The K=1 block is a contiguous span of electrodes on a single
    shaft (paradigm B drop, not scattered)."""
    anatomy = _make_anatomy([10] * 8)
    ext = ShaftMaskExtractor(seed=0)
    mask = ext(anatomy)
    masked_idx = torch.nonzero(mask).flatten()
    assert masked_idx.numel() >= 1
    # Contiguity check: the masked indices form a single run within
    # their shaft.
    if masked_idx.numel() > 1:
        diffs = (masked_idx[1:] - masked_idx[:-1]).tolist()
        assert all(d == 1 for d in diffs), (
            f"K=1 block expected to be a contiguous span; got idx={masked_idx.tolist()}"
        )


# ---------- Sister wiring ---------------------------------------------


def test_extractor_r_shaft_k2_sister_via_k_override_plus_alpha_beta() -> None:
    """``R-shaft-K2`` sister: ``k_override=2,
    extent_blocks=("alpha", "beta")``. Picks 2 blocks across 2 shafts."""
    anatomy = _make_anatomy([8] * 8)
    ext = ShaftMaskExtractor(
        seed=0, k_override=2, extent_blocks=("alpha", "beta"),
    )
    assert ext._k_for_anatomy(anatomy) == 2
    assert ext.block_extents == [BLOCK_ALPHA, BLOCK_BETA]
    mask = ext(anatomy)
    # 2 blocks → masked electrodes span 2 distinct shafts.
    masked_idx = torch.nonzero(mask).flatten()
    shafts_touched = torch.unique(anatomy.shaft_ids[masked_idx])
    assert shafts_touched.numel() == 2, (
        f"K=2 expected to touch 2 shafts, got {shafts_touched.tolist()}"
    )


def test_extractor_r_shaft_k3_mixed_3block_sister() -> None:
    """``R-shaft-K3-mixed-3block`` sister: ``k_override=3,
    extent_blocks=("alpha", "beta", "gamma")``."""
    anatomy = _make_anatomy([8] * 8)
    ext = ShaftMaskExtractor(
        seed=0, k_override=3, extent_blocks=("alpha", "beta", "gamma"),
    )
    assert ext._k_for_anatomy(anatomy) == 3
    assert ext.block_extents == [BLOCK_ALPHA, BLOCK_BETA, BLOCK_GAMMA]


def test_extractor_r_shaft_k1_explicit_sister_matches_default() -> None:
    """``R-shaft-K1-explicit`` sister: ``k_override=1`` matches the
    default but documents the lock in the dispatch log."""
    anatomy = _make_anatomy([8] * 8)
    ext_default = ShaftMaskExtractor(seed=0)
    ext_explicit = ShaftMaskExtractor(seed=0, k_override=1)
    assert ext_default._k_for_anatomy(anatomy) == ext_explicit._k_for_anatomy(anatomy)
    a = ext_default(anatomy)
    b = ext_explicit(anatomy)
    assert torch.equal(a, b), "explicit k_override=1 must match default"


def test_extractor_k_override_caps_at_n_shafts() -> None:
    """``k_override=3`` on a 2-shaft subject collapses to K=2 — sisters
    are upper-bounded by the available shafts."""
    anatomy = _make_anatomy([8, 8])  # 2 shafts
    ext = ShaftMaskExtractor(
        seed=0, k_override=3, extent_blocks=("alpha", "beta", "gamma"),
    )
    assert ext._k_for_anatomy(anatomy) == 2


def test_extractor_k_override_zero_is_legal_no_op() -> None:
    """``k_override=0`` is a legal config: emits an all-False mask.
    Useful as a Phase-1 control where shaft-mask is OFF but the
    extractor is still constructed."""
    anatomy = _make_anatomy([8] * 8)
    ext = ShaftMaskExtractor(seed=0, k_override=0)
    assert ext._k_for_anatomy(anatomy) == 0
    assert not ext(anatomy).any()


def test_extractor_rejects_negative_k_override() -> None:
    with pytest.raises(ValueError, match="k_override"):
        ShaftMaskExtractor(seed=0, k_override=-1)


def test_extractor_rejects_unknown_block_extent_name() -> None:
    """Unknown extent name fails loudly at construction time, not
    silently at sampling."""
    with pytest.raises(ValueError, match="unknown block extent"):
        ShaftMaskExtractor(seed=0, extent_blocks=("alpha", "delta"))


def test_extractor_rejects_empty_extent_blocks() -> None:
    """An empty ``extent_blocks`` list is a config error — at least one
    block must be available."""
    with pytest.raises(ValueError, match="extent_blocks"):
        ShaftMaskExtractor(seed=0, extent_blocks=())


def test_extractor_accepts_block_extent_object_directly() -> None:
    """``extent_blocks`` can also take ``BlockExtent`` instances directly
    (custom shaft sweeps)."""
    ext = ShaftMaskExtractor(seed=0, extent_blocks=(BLOCK_ALPHA,))
    assert ext.block_extents == [BLOCK_ALPHA]


# ---------- MON-MASK-001 fraction expectation -------------------------


def test_extractor_shaft_mask_fraction_in_expected_band_under_k1_default() -> None:
    """MON-MASK-001 (revised 2026-05-27 PM): with the K=1 default, the
    average shaft mask fraction is roughly ``1 / N_shafts × extent_α``.

    For ``N_shafts = 8`` and block α (shaft-extent 0.45–0.60), one shaft
    contributes about 1/8 × 0.525 ≈ 6.6% of the total electrode count
    on average. Allow a wide [0.02, 0.15] band over 16 seeds to absorb
    the per-seed variance while still catching a K=2+ regression (which
    would land ~12–15%) or a K=0 regression (which would land at 0)."""
    anatomy = _make_anatomy([8] * 8)  # 64 electrodes total, 8 shafts
    frac_sum = 0.0
    n_seeds = 16
    for seed in range(n_seeds):
        ext = ShaftMaskExtractor(seed=seed)
        mask = ext(anatomy)
        frac_sum += mask.float().mean().item()
    mean_frac = frac_sum / n_seeds
    assert 0.02 <= mean_frac <= 0.15, (
        f"K=1 shaft-mask fraction {mean_frac:.3f} outside the [0.02, 0.15] "
        f"band expected under the 5/27 PM lock"
    )


# ---------- BTShaftMaskExtractor (NeuralFetch-style per-event wrapper) -


from pathlib import Path
from types import SimpleNamespace


def _write_depth_wm(
    bt_root: Path, subject_id: int, rows: list[tuple[str, str]],
) -> Path:
    path = bt_root / "localization" / f"sub_{subject_id}" / "depth-wm.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("Electrode,DesikanKilliany\n")
        for electrode, label in rows:
            f.write(f"{electrode},{label}\n")
    return path


def _bt_event(
    subject_id: int = 1, timeline: str = "t0", start: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        subject=str(subject_id), timeline=timeline, start=start, duration=1.0,
    )


def _bt_rows_two_shafts() -> list[tuple[str, str]]:
    # 2 shafts × 5 contacts = 10 real electrodes. Each row uses a DK label
    # that's in V14_DK_PARCEL_LABELS so unknown_label_policy="raise"
    # passes too.
    rows = []
    for i in range(1, 6):
        rows.append((f"OFa{i}", "ctx-lh-superiortemporal"))
    for i in range(1, 6):
        rows.append((f"OFb{i}", "ctx-rh-bankssts"))
    return rows


def test_bt_shaft_mask_extractor_shape_and_dtype(tmp_path: Path) -> None:
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    _write_depth_wm(tmp_path, subject_id=1, rows=_bt_rows_two_shafts())
    ext = BTShaftMaskExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120, seed=0,
    )
    out = ext.get_static(_bt_event(subject_id=1))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (120,)
    assert out.dtype == torch.bool


def test_bt_shaft_mask_extractor_padding_slots_are_false(tmp_path: Path) -> None:
    """Slots past ``n_electrodes`` must stay False — they must NOT count as
    shaft-blocked or they'd silently corrupt the cross-attn keypad path."""
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    _write_depth_wm(tmp_path, subject_id=1, rows=_bt_rows_two_shafts())
    ext = BTShaftMaskExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120, seed=0,
    )
    out = ext.get_static(_bt_event(subject_id=1))
    n_real = len(_bt_rows_two_shafts())  # 10
    # All padding slots must be False.
    assert not out[n_real:].any(), "padding slots must remain False"


def test_bt_shaft_mask_extractor_k1_at_n_shafts_2(tmp_path: Path) -> None:
    """N_shafts=2 → K=1 → exactly one shaft block in the first ``n_real``
    slots; the mask must have at least one True."""
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    _write_depth_wm(tmp_path, subject_id=1, rows=_bt_rows_two_shafts())
    ext = BTShaftMaskExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120, seed=0,
    )
    out = ext.get_static(_bt_event(subject_id=1))
    assert out.any(), "K=1 must produce >=1 True at the real-electrode region"


def test_bt_shaft_mask_extractor_per_event_seed_varies(tmp_path: Path) -> None:
    """Two events on the same subject (distinct ``event.start``) get
    different per-event seeds → likely different masks."""
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    _write_depth_wm(tmp_path, subject_id=1, rows=_bt_rows_two_shafts())
    ext = BTShaftMaskExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120, seed=0,
    )
    a = ext.get_static(_bt_event(subject_id=1, start=0.0))
    b = ext.get_static(_bt_event(subject_id=1, start=10.0))
    # K=1 on 2 shafts could collide; check across many event-start values.
    seen = {tuple(ext.get_static(_bt_event(subject_id=1, start=float(s))).tolist())
            for s in range(8)}
    assert len(seen) >= 2, "per-event seed should produce at least 2 distinct masks"
    # And the dispatch contract uses (a, b) directly.
    _ = a, b


def test_bt_shaft_mask_extractor_reproducible_under_fixed_event(tmp_path: Path) -> None:
    """Same (subject, timeline, start) → identical mask. Required so the
    NeuralSet collator + caching see the same bytes per event."""
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    _write_depth_wm(tmp_path, subject_id=1, rows=_bt_rows_two_shafts())
    ext = BTShaftMaskExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120, seed=42,
    )
    e = _bt_event(subject_id=1, timeline="trial0", start=3.14)
    a = ext.get_static(e)
    b = ext.get_static(e)
    assert torch.equal(a, b)


def test_bt_shaft_mask_extractor_rejects_subject_exceeding_c_max(tmp_path: Path) -> None:
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    _write_depth_wm(tmp_path, subject_id=1, rows=_bt_rows_two_shafts())
    ext = BTShaftMaskExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=4, seed=0,
    )
    with pytest.raises(ValueError, match="c_max"):
        ext.get_static(_bt_event(subject_id=1))


def test_bt_shaft_mask_extractor_unknown_label_policy_skip(tmp_path: Path) -> None:
    """``unknown_label_policy="skip"`` drops out-of-vocab DK rows. Must
    match the DK/valid-mask filter so the C dimension stays aligned."""
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    rows = _bt_rows_two_shafts() + [("BAD1", "not-a-real-dk-label")]
    _write_depth_wm(tmp_path, subject_id=1, rows=rows)
    ext = BTShaftMaskExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120, seed=0,
        unknown_label_policy="skip",
    )
    out = ext.get_static(_bt_event(subject_id=1))
    # Real electrodes after skip = 10 (the 2 shafts), so positions 10+ are
    # all padding-False.
    assert not out[10:].any()


def test_bt_shaft_mask_extractor_k0_at_single_shaft(tmp_path: Path) -> None:
    """Single-shaft subject → K=0 → all-False (5/27 PM guard)."""
    from speech_decoding.extractors.shaft_mask import BTShaftMaskExtractor

    rows = [(f"OFa{i}", "ctx-lh-superiortemporal") for i in range(1, 6)]
    _write_depth_wm(tmp_path, subject_id=1, rows=rows)
    ext = BTShaftMaskExtractor(
        event_types="Ieeg", bt_root=str(tmp_path), c_max=120, seed=0,
    )
    out = ext.get_static(_bt_event(subject_id=1))
    assert not out.any(), "N_shafts=1 must produce K=0 (all-False mask)"
