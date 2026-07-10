"""v14_converged_v3 Phase 4b — L1 / L2 attention blocks (TDD).

Memo project-v14-converged-v3-sensor-architecture, the two disjoint mixers:

  L1  within-sensor JOINT spatiotemporal. Block-diagonal by shaft: a token only
      attends to (contact, time) pairs of its OWN shaft. JOINT ⇒ time and
      contact mix together (not factorized). RoPE (index+time), relative.
  L2  cross-sensor FACTORIZED at same-t. Mixes contacts ACROSS shafts, but only
      within a single time slice (temporal lag is left to a 2-hop L1∘L2 compose).
      Plain same-t cross-shaft attention; no RoPE. Parcel/DKT identity is added at
      the TOWER input (V-JEPA 2.1 modality-embed style), not injected in L2 — so
      that behaviour is tested in test_towers, not here.

The load-bearing behavioural contrasts, asserted below:
  - L1 is block-diagonal (perturbing shaft B leaves shaft A untouched); L2 is NOT
    (it mixes across shafts).
  - L1 is joint in time (perturbing t2 can change t1's output within a shaft); L2
    is time-factorized (perturbing t2 never changes t1's output).
  - The padded varlen gather is exact: running L1 on a lone shaft matches that
    shaft's rows from a multi-shaft run.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.attention import L1Block, L2Block
from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar

D, H, T = 32, 4, 5

# A non-uniform bump: LayerNorm is invariant to a constant added across the
# d-axis, so a scalar `+= c` would be silently erased before attention (and make
# the "unchanged" assertions false-pass). Perturb with a fixed random vector.
torch.manual_seed(1)
_BUMP_D = torch.randn(D) * 3.0
_BUMP_TD = torch.randn(T, D) * 3.0


def _sc(labels, parcels):
    return build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))


def _two_shaft():
    # shaft A: 3 contacts, shaft B: 2 contacts.
    sc = _sc(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    return sc, build_l1_geometry(sc)


# ---------------------------------------------------------------- L1


def test_l1_preserves_shape() -> None:
    sc, geom = _two_shaft()
    blk = L1Block(D, H).eval()
    x = torch.randn(2, 5, T, D)
    assert blk(x, geom).shape == x.shape


def test_l1_is_block_diagonal_across_shafts() -> None:
    sc, geom = _two_shaft()
    blk = L1Block(D, H).eval()
    x = torch.randn(1, 5, T, D)
    out = blk(x, geom)
    x2 = x.clone()
    x2[0, 3] += _BUMP_TD  # perturb a shaft-B contact
    out2 = blk(x2, geom)
    # shaft A (contacts 0,1,2) unchanged; shaft B (3,4) changed.
    assert torch.allclose(out[0, :3], out2[0, :3], atol=1e-5)
    assert not torch.allclose(out[0, 3:], out2[0, 3:], atol=1e-4)


def test_l1_is_joint_in_time() -> None:
    # Within a shaft, perturbing a contact at time t=2 can change the SAME shaft's
    # output at t=0 — L1 attends jointly over (contact, time).
    sc, geom = _two_shaft()
    blk = L1Block(D, H).eval()
    x = torch.randn(1, 5, T, D)
    out = blk(x, geom)
    x2 = x.clone()
    x2[0, 1, 2] += _BUMP_D  # shaft-A contact 1, time 2
    out2 = blk(x2, geom)
    assert not torch.allclose(out[0, 0, 0], out2[0, 0, 0], atol=1e-4)


def test_l1_padded_varlen_matches_lone_shaft() -> None:
    # Definitive varlen-correctness: the short shaft's rows in a 2-shaft run must
    # equal running the block on that shaft alone (padding/masking is exact).
    sc, geom = _two_shaft()
    blk = L1Block(D, H).eval()
    x = torch.randn(1, 5, T, D)
    out = blk(x, geom)  # full run

    sc_b = _sc(["LB1", "LB2"], [1, 1])
    geom_b = build_l1_geometry(sc_b)
    out_b = blk(x[:, 3:5], geom_b)  # lone shaft B
    assert torch.allclose(out[0, 3:5], out_b[0], atol=1e-5)


# ---------------------------------------------------------------- L2


def test_l2_preserves_shape() -> None:
    blk = L2Block(D, H).eval()
    x = torch.randn(2, 5, T, D)
    assert blk(x).shape == x.shape


def test_l2_mixes_across_sensors() -> None:
    # Opposite of L1: perturbing one contact changes OTHER contacts' outputs
    # (cross-shaft mixing) at the same time.
    blk = L2Block(D, H).eval()
    x = torch.randn(1, 5, T, D)
    out = blk(x)
    x2 = x.clone()
    x2[0, 3, 0] += _BUMP_D  # contact 3 (shaft B), time 0
    out2 = blk(x2)
    # a DIFFERENT-shaft contact (0, shaft A) at the SAME time is affected.
    assert not torch.allclose(out[0, 0, 0], out2[0, 0, 0], atol=1e-4)


def test_l2_is_time_factorized() -> None:
    # Perturbing any contact at time t=2 must leave EVERY output at t=0 unchanged
    # (L2 mixes space only within a time slice).
    blk = L2Block(D, H).eval()
    x = torch.randn(1, 5, T, D)
    out = blk(x)
    x2 = x.clone()
    x2[0, 3, 2] += _BUMP_D  # contact 3, time 2
    out2 = blk(x2)
    assert torch.allclose(out[0, :, 0], out2[0, :, 0], atol=1e-5)


def test_both_blocks_run_under_bf16_autocast() -> None:
    # Regression: v3 is documented to train bf16-mixed (params fp32). L1Block's
    # visible-gather scatter (out[...] = ctx[...]) must allocate `out` in ctx's
    # dtype — under autocast x=norm1(x) is fp32 but ctx=out(ctx) is bf16, and the
    # advanced-index assignment does NOT promote (unlike `+`). Every other test runs
    # fp32, so nothing else catches this. The residual add must stay fp32.
    sc, geom = _two_shaft()
    x = torch.randn(2, 5, T, D)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        l1_out = L1Block(D, H)(x, geom)
        l2_out = L2Block(D, H)(x)
    assert l1_out.shape == x.shape and l2_out.shape == x.shape
    # forward() adds these to the fp32 residual stream → promotes back to fp32.
    assert (x + l1_out).dtype == torch.float32
    assert (x + l2_out).dtype == torch.float32
