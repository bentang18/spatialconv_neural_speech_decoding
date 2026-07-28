"""_encode_taps writes batch slices into preallocated buffers instead of cat-ing a list.

The list+cat version held both the per-batch list and its concatenation alive at once — a 2x
peak on the ~40 GB enc12_elec tap, which is what forced --mem=300G on the board encode. The
rewrite must be BIT-identical: same values, same dtype, same row order, for any batching that
does or does not divide n evenly. These tests pin that against a literal reimplementation of
the old path.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.neuroprobe.v3_probe_encode_r4 import GPU_TAPS, _encode_taps, _pool_parcels

K_FULL = 3
D = 4
N_CONTACTS = 5
PARCEL_CANON = np.array([10, 10, 20, 20, 30])
PRESENT = [10, 20, 30]


class _Grid:
    k_full = K_FULL


class _Teacher:
    """Returns a deterministic per-window slice, so any batching bug shows up as reordering."""

    def __init__(self, n: int) -> None:
        torch.manual_seed(0)
        self.full = {t: torch.randn(n, N_CONTACTS * K_FULL, D) * (t + 1) for t in GPU_TAPS}
        self.cursor = 0

    def forward(self, bb, grid, parcel_packed, *, tap_blocks):
        b = bb[0].shape[0]
        lo, self.cursor = self.cursor, self.cursor + b
        return None, {t: self.full[t][lo:lo + b] for t in tap_blocks}


def _old_path(teacher, bands, grid, parcel_canon, present, *, batch_size, elec_taps):
    """The pre-rewrite implementation, verbatim in behaviour: list append + torch.cat."""
    n = bands[0].shape[0]
    k = grid.k_full
    acc: dict = {t: [] for t in GPU_TAPS}
    for t in elec_taps:
        acc[f"elec{t}"] = []
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        bb = [b[s:e] for b in bands]
        Bb = e - s
        _z, taps = teacher.forward(bb, grid, None, tap_blocks=GPU_TAPS)
        for t in GPU_TAPS:
            enc = taps[t].float().reshape(Bb, -1, k, taps[t].shape[-1])
            acc[t].append(_pool_parcels(enc, parcel_canon, present))
            if t in elec_taps:
                acc[f"elec{t}"].append(enc.reshape(Bb, enc.shape[1], -1).to(torch.float16))
    return {t: {"raw": torch.cat(v, 0)} for t, v in acc.items()}


def _run(n: int, batch_size: int, elec_taps=(12,)):
    bands = [torch.zeros(n, 1, 1)]
    kw = dict(batch_size=batch_size, elec_taps=elec_taps)
    new = _encode_taps(_Teacher(n), bands, _Grid(), None, PARCEL_CANON, PRESENT,
                       device=torch.device("cpu"), **kw)
    old = _old_path(_Teacher(n), bands, _Grid(), PARCEL_CANON, PRESENT, **kw)
    return new, old


@pytest.mark.parametrize("n,batch_size", [
    (12, 4),    # batch divides n evenly
    (13, 4),    # ragged final batch — the slice-assignment edge case
    (7, 16),    # single batch larger than n
    (5, 1),     # every window its own batch
])
def test_preallocated_write_is_bit_identical_to_cat(n, batch_size):
    new, old = _run(n, batch_size)
    assert set(new) == set(old)
    for key in old:
        a, b = new[key]["raw"], old[key]["raw"]
        assert a.dtype == b.dtype, key
        assert a.shape == b.shape, key
        assert torch.equal(a, b), key


def test_row_order_follows_window_order():
    """A buffer written out of order would still match on shape/dtype — pin the ordering."""
    n, batch_size = 12, 5
    new, _ = _run(n, batch_size)
    teacher = _Teacher(n)
    elec = new["elec12"]["raw"]
    expected = teacher.full[12].float().reshape(n, N_CONTACTS, K_FULL, D)
    expected = expected.reshape(n, N_CONTACTS, -1).to(torch.float16)
    assert torch.equal(elec, expected)


def test_length_matches_n_not_batch_multiple():
    n = 13
    new, _ = _run(n, 4)
    for key, val in new.items():
        assert val["raw"].shape[0] == n, key


def test_no_elec_taps_yields_only_parcel_keys():
    new, old = _run(8, 3, elec_taps=())
    assert set(new) == set(GPU_TAPS)
    assert set(new) == set(old)


def test_parcel_tap_shape_is_present_parcels():
    new, _ = _run(8, 3)
    for t in GPU_TAPS:
        assert new[t]["raw"].shape[:2] == (8, len(PRESENT))
