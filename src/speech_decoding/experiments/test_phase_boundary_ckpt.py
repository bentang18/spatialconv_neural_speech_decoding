"""SCAFFOLD-08 (TST03 + RT10): P1↔P2 checkpoint compatibility.

At the Phase-1 → Phase-2 boundary, the encoder weights are loaded with
``strict=True``. ``strict=False`` would silently random-initialize
missing keys — which would mask stale-cascade bugs (e.g., a broken
Predictor2Block warm-start, an EMA-teacher key shape that drifted
between phases).

This test pins the contract: a Phase-1 encoder state-dict must load
into a Phase-2 encoder ``strict=True`` cleanly. Both models are
instantiated with identical hyper-parameters here — Phase-2 differs
from Phase-1 only in trainer-side switches (anatomy bias ON, shaft
electrode mask, Predictor2Block warm-start), not in the underlying
encoder topology, so the encoder state-dict must round-trip exactly.

If a future encoder change adds a new parameter that should be loaded
fresh in Phase-2, the failing assertion here is the early warning to
either (a) add the new parameter to Phase-1 too (so the encoder
topology is phase-invariant), or (b) document the asymmetry and
explicitly allow it via a key-strict loader.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel


_TINY = dict(
    n_freq_bins=6, n_time_bins=4, k_parcels=6,
    d_model=32, n_heads=4, depth_self_attn=1, m_sub_slots=2,
    patch_kernel_freq=3,  # FE-RAW-1: kernel-3 path for the tiny F=6 config
)


def test_p1_to_p2_state_dict_loads_strict_true() -> None:
    """The Phase-1 encoder state-dict loads into a freshly-instantiated
    Phase-2 encoder with ``strict=True``."""
    torch.manual_seed(0)
    p1 = V14ParcelPerceiverModel(**_TINY)
    p2 = V14ParcelPerceiverModel(**_TINY)
    sd = p1.state_dict()
    missing, unexpected = p2.load_state_dict(sd, strict=True)
    assert not missing, f"missing keys at P1→P2 reload: {missing}"
    assert not unexpected, f"unexpected keys at P1→P2 reload: {unexpected}"


def test_p1_to_p2_state_dict_actually_transfers_values() -> None:
    """Sanity check the load actually moves parameter values (catches a
    silent ``strict=False`` fallback)."""
    torch.manual_seed(0)
    p1 = V14ParcelPerceiverModel(**_TINY)
    torch.manual_seed(1)
    p2 = V14ParcelPerceiverModel(**_TINY)
    # Pre-load: parameters differ (different seeds).
    assert not torch.allclose(
        next(p1.parameters()), next(p2.parameters())
    )
    p2.load_state_dict(p1.state_dict(), strict=True)
    # Post-load: every parameter matches.
    for k, v_p1 in p1.state_dict().items():
        v_p2 = p2.state_dict()[k]
        assert torch.equal(v_p1, v_p2), f"value mismatch at key {k!r}"


def test_p1_p2_have_identical_keys() -> None:
    """The state-dict key sets are identical — the canonical detector
    for accidental Phase-2-only parameter additions."""
    p1 = V14ParcelPerceiverModel(**_TINY)
    p2 = V14ParcelPerceiverModel(**_TINY)
    assert set(p1.state_dict().keys()) == set(p2.state_dict().keys())


def test_strict_load_rejects_extra_p2_key() -> None:
    """If a future Phase-2 encoder grows an extra parameter, the
    ``strict=True`` load surfaces it as a 'missing' key — this test
    pins that pytorch's contract still holds (regression guard)."""
    torch.manual_seed(0)
    p1 = V14ParcelPerceiverModel(**_TINY)
    p2 = V14ParcelPerceiverModel(**_TINY)
    sd = p1.state_dict()
    # Manually drop a key from the source sd; the target P2 still expects it.
    first_key = next(iter(sd))
    del sd[first_key]
    with pytest.raises(RuntimeError, match=r"[Mm]issing"):
        p2.load_state_dict(sd, strict=True)
