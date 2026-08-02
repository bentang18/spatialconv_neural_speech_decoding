"""`fm:` frontend-ablation arms — the contract that makes the ablation mean something.

The failure mode is not a crash. A column mask that slices the wrong axis, or reads a band
layout that does not describe the cache, produces a perfectly plausible AUROC that answers a
different question than the one asked. So these tests pin the three things that could silently
make the verdict wrong:

  * the published path is UNTOUCHED — `fm:full:` reproduces the plain tap bit-for-bit, end to
    end through `_ws_cell`, so an arm can never be blamed for (or credited with) a board number;
  * the mask slices the (time, freq) axes the layout claims, verified against columns whose
    VALUE encodes their own (band, t, f) coordinate — a transposed or off-by-one slice moves
    values that the test can name;
  * a layout that does not describe the cache FAILS LOUD instead of masking wrong columns.
"""
from __future__ import annotations

import numpy as np
import pytest


from scripts.neuroprobe.test_v3_board_readout import _rec
from scripts.neuroprobe.v3_board_readout import (
    BOARD_TASKS,
    FEAT_ARMS,
    _ARM_BASE,
    _base_tap,
    _feat,
    _fm_apply,
    _fm_spec,
    _fm_width,
    _have,
    _is_elec,
    _validate_taps,
    _ws_cell,
)

BL = (4, 16, 32)     # SLOW / MID / HGA frames at 1 s
FD = (7, 6, 7)       # bins per band
WIDTH = 348          # 4*7 + 16*6 + 32*7


def _coded(r=3, u=2):
    """(r, U, 348) whose every column VALUE encodes its own (band, t, f) coordinate.

    band*10000 + t*100 + f is unique per column, so a wrong slice cannot coincidentally
    produce the right numbers — the test can name exactly which coordinates survived.
    """
    cols, off = np.zeros(WIDTH, dtype=np.float32), 0
    for b, (t_b, f_b) in enumerate(zip(BL, FD)):
        for t in range(t_b):
            for f in range(f_b):
                cols[off] = b * 10000 + t * 100 + f
                off += 1
    assert off == WIDTH
    return np.broadcast_to(cols, (r, u, WIDTH)).copy()


def _decode(x):
    """Flat masked block → the set of (band, t, f) coordinates it kept."""
    return {(int(v) // 10000, (int(v) % 10000) // 100, int(v) % 100) for v in x[0, 0].ravel()}


# ── the published path is untouched ────────────────────────────────────────────────

def test_fm_full_is_bit_identical_to_the_plain_tap() -> None:
    """`fm:full:` must be the plain tap to the bit — the guard that an arm cannot move a
    board number. Everything else in this file only matters if this holds."""
    rec = _rec(n=32, n_parcels=3, feat=WIDTH)
    rec["band_lengths"], rec["band_fdims"] = np.asarray(BL), np.asarray(FD)
    rows = np.arange(32)
    plain = _feat(rec, "enc0", rows)
    armed = _feat(rec, "fm:full:enc0", rows)
    assert armed.shape == plain.shape
    assert np.array_equal(armed, plain), "fm:full: perturbed the published feature matrix"


def test_fm_full_reproduces_the_published_ws_cell() -> None:
    """End to end: same splits, same standardize, same λ grid ⇒ the same AUROC."""
    rec = _rec(n=64, n_parcels=3, feat=WIDTH)
    rec["band_lengths"], rec["band_fdims"] = np.asarray(BL), np.asarray(FD)
    task = BOARD_TASKS[0]
    cells = _ws_cell(rec, task, ("enc0", "fm:full:enc0"))["cells"]
    plain, armed = cells["enc0|std"], cells["fm:full:enc0|std"]
    assert plain["test"] == pytest.approx(armed["test"], abs=0.0), (plain, armed)
    # same λ too — an identical AUROC off a different λ would be a coincidence, not identity
    assert plain["lam_mult"] == armed["lam_mult"], (plain, armed)


# ── the mask slices the axes the layout claims ─────────────────────────────────────

def test_band_drop_keeps_exactly_the_named_bands() -> None:
    x = _coded()
    assert {b for b, _, _ in _decode(_fm_apply(x, "slow", BL, FD))} == {0}
    assert {b for b, _, _ in _decode(_fm_apply(x, "mid", BL, FD))} == {1}
    assert {b for b, _, _ in _decode(_fm_apply(x, "hga", BL, FD))} == {2}
    assert {b for b, _, _ in _decode(_fm_apply(x, "nohga", BL, FD))} == {0, 1}
    assert {b for b, _, _ in _decode(_fm_apply(x, "noslow", BL, FD))} == {1, 2}
    assert {b for b, _, _ in _decode(_fm_apply(x, "nomid", BL, FD))} == {0, 2}


def test_hga_time_ladder_decimates_the_TIME_axis_only() -> None:
    """The whole multirate claim rides on this: hgat16 must drop FRAMES, never bins."""
    x = _coded()
    for arm, keep in (("hgat16", 2), ("hgat8", 4), ("hgat4", 8), ("hgat1", 32)):
        got = _decode(_fm_apply(x, arm, BL, FD))
        hga = {(t, f) for b, t, f in got if b == 2}
        assert {t for t, _ in hga} == set(range(0, 32, keep)), arm
        assert {f for _, f in hga} == set(range(7)), f"{arm} touched the frequency axis"
        # the other two bands are untouched
        assert {(t, f) for b, t, f in got if b == 0} == {(t, f) for t in range(4) for f in range(7)}
        assert {(t, f) for b, t, f in got if b == 1} == {(t, f) for t in range(16) for f in range(6)}


def test_hgaf1_averages_the_bins_and_keeps_every_frame() -> None:
    """hgaf1 is a MEAN over HGA's 7 bins, not a pick of one — a pick would also shed the
    noise averaging, which would confound the frequency-resolution question with SNR."""
    x = _coded()
    out = _fm_apply(x, "hgaf1", BL, FD)
    assert out.shape[-1] == 4 * 7 + 16 * 6 + 32 * 1
    hga = out[0, 0, 4 * 7 + 16 * 6:]
    assert hga.shape == (32,)
    for t in range(32):
        assert hga[t] == pytest.approx(20000 + t * 100 + np.mean(np.arange(7)))


def test_widths_match_the_arms_actual_output() -> None:
    x = _coded()
    for arm in FEAT_ARMS:
        if _ARM_BASE.get(arm, BL) != BL:
            continue                      # designed on another bake; covered by the 64 Hz tests
        assert _fm_apply(x, arm, BL, FD).shape[-1] == _fm_width(arm, BL, FD), arm
    assert _fm_width("full", BL, FD) == WIDTH


# ── a layout that does not describe the cache fails loud ───────────────────────────

def test_wrong_layout_raises_instead_of_masking_wrong_columns() -> None:
    x = _coded()
    with pytest.raises(SystemExit, match="do not describe this cache"):
        _fm_apply(x, "hga", BL, (7, 7, 7))       # sums to 4*7+16*7+32*7 != 348


def test_stride_that_does_not_divide_the_band_raises() -> None:
    x = _coded()
    bad = dict(FEAT_ARMS)
    bad["odd"] = ((1, "all"), (1, "all"), (5, "all"))   # 5 does not divide 32
    FEAT_ARMS["_odd_"] = bad["odd"]
    try:
        with pytest.raises(SystemExit, match="does not divide"):
            _fm_apply(x, "_odd_", BL, FD)
    finally:
        del FEAT_ARMS["_odd_"]


# ── parse-time refusals ────────────────────────────────────────────────────────────

def test_unknown_arm_is_refused_at_parse_time() -> None:
    with pytest.raises(SystemExit, match="unknown arm"):
        _validate_taps(("fm:nosuch:enc0",))


def test_fm_on_an_encoder_tap_is_refused() -> None:
    """enc12 is (k tokens x d) — it has no frequency axis, so an arm name would silently
    mean a different operation there."""
    with pytest.raises(SystemExit, match="spectrogram layout only"):
        _validate_taps(("fm:hga:enc12",))


def test_fm_without_a_tap_is_refused() -> None:
    with pytest.raises(SystemExit, match="names no tap"):
        _validate_taps(("fm:hga",))


def test_valid_arms_pass_validation() -> None:
    _validate_taps(tuple(f"fm:{a}:enc0_elec" for a in FEAT_ARMS) + ("enc0", "enc0_elec"))


# ── an arm keeps its base tap's unit and availability ──────────────────────────────

def test_arm_inherits_the_base_taps_unit_and_availability() -> None:
    assert _is_elec("fm:hga:enc0_elec") is True
    assert _is_elec("fm:hga:enc0") is False
    assert _base_tap("fm:hgat16:enc0_elec") == "enc0_elec"
    assert _fm_spec("enc0") == ("", "enc0")
    rec = _rec(n=8, n_parcels=2, feat=WIDTH)
    rec["band_lengths"], rec["band_fdims"] = np.asarray(BL), np.asarray(FD)
    assert _have(rec, "fm:hga:enc0") is True
    assert _have(rec, "fm:hga:enc0_elec") is False      # not in this synthetic record


# ── the 64 Hz HGA bake (band_v3hga): a stride is a RATE only against a stated base ──

BL64 = (4, 16, 64)   # SLOW 32 Hz/::8 | MID 32 Hz/::2 | HGA 64 Hz/::1
FD64 = (7, 6, 4)     # band_v3hga is 4 bins over 64-160 Hz
WIDTH64 = 380        # 4*7 + 16*6 + 64*4


def _coded64(r=3, u=2):
    cols, off = np.zeros(WIDTH64, dtype=np.float32), 0
    for b, (t_b, f_b) in enumerate(zip(BL64, FD64)):
        for t in range(t_b):
            for f in range(f_b):
                cols[off] = b * 10000 + t * 100 + f
                off += 1
    assert off == WIDTH64
    return np.broadcast_to(cols, (r, u, WIDTH64)).copy()


def test_a_rate_arm_is_refused_on_a_bake_it_was_not_designed_on() -> None:
    """The whole point of _ARM_BASE: `hgat16` is stride 2, which is 16 Hz on the 32 Hz bake but
    32 Hz on the 64 Hz one. Both produce a valid-looking result, so only a name check catches it."""
    with pytest.raises(SystemExit, match="designed on band_lengths"):
        _fm_apply(_coded64(), "hgat16", BL64, FD64)
    with pytest.raises(SystemExit, match="designed on band_lengths"):
        _fm_apply(_coded(), "hga64t32", BL, FD)
    # ...and the width helper must refuse identically, or a run could print a width for an arm
    # that then raises mid-fit.
    with pytest.raises(SystemExit, match="designed on band_lengths"):
        _fm_width("hgat16", BL64, FD64)


def test_layout_agnostic_arms_still_work_on_the_64hz_bake() -> None:
    """`full` and the band-drop arms mean the same thing at any bake, so they carry no base."""
    x = _coded64()
    assert _fm_apply(x, "full", BL64, FD64).shape[-1] == WIDTH64
    assert np.array_equal(_fm_apply(x, "full", BL64, FD64), x), "full must be the identity"
    assert _fm_apply(x, "nohga", BL64, FD64).shape[-1] == 4 * 7 + 16 * 6
    assert _fm_width("hga", BL64, FD64) == 64 * 4


def test_hga64t32_halves_the_HGA_rate_and_touches_nothing_else() -> None:
    """The rate control: 64 -> 32 Hz at a FIXED window and bin count, so `full` vs this arm
    isolates temporal rate from the window and bin changes that come with the v3hga bake."""
    got = _decode(_fm_apply(_coded64(), "hga64t32", BL64, FD64))
    hga = {(t, f) for b, t, f in got if b == 2}
    assert {t for t, _ in hga} == set(range(0, 64, 2)), "did not land on 32 frames"
    assert {f for _, f in hga} == set(range(4)), "touched the frequency axis"
    assert {(t, f) for b, t, f in got if b == 0} == {(t, f) for t in range(4) for f in range(7)}
    assert {(t, f) for b, t, f in got if b == 1} == {(t, f) for t in range(16) for f in range(6)}


def test_64hz_arm_widths_match_their_actual_output() -> None:
    x = _coded64()
    for arm in FEAT_ARMS:
        if _ARM_BASE.get(arm, BL64) != BL64:
            continue
        assert _fm_apply(x, arm, BL64, FD64).shape[-1] == _fm_width(arm, BL64, FD64), arm
    assert _fm_width("hga64t32", BL64, FD64) == 4 * 7 + 16 * 6 + 32 * 4
    assert _fm_width("hga64f1", BL64, FD64) == 4 * 7 + 16 * 6 + 64 * 1
