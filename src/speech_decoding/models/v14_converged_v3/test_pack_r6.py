"""r6 flat-grid mask flags — ``token_flags_r6``.

r6 reuses ``build_r4_grid`` (identical 3-band 32 Hz-lattice grid) and adds ``token_flags_r6``,
which differs from r4's ``token_flags`` in exactly two ways:

  * band masks are PER-SENSOR ``(B,N,T_b)``, not global ``(B,T_b)`` — each contact hides its own
    frames (Ben 2026-07-23: r4's global time mask bought nothing, the encoder is L1-within-shaft
    only, so per-sensor is free extra diversity);
  * NO margin gate — ``in_loss == masked``, every masked token is scored.

THE correctness test: with a mask that is constant across sensors, r6's ``masked`` set is
byte-identical to r4's, and its ``in_loss`` is r4's ``in_loss`` PLUS exactly the tokens r4's M14
margin gate excluded. Invariants named + asserted + printed.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3Masks,
    V3MasksR6,
    sample_masks,
    sample_masks_r6,
)
from speech_decoding.models.v14_converged_v3.pack_r4 import (
    build_r4_grid,
    token_flags,
    token_flags_r6,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _session(shaft_sizes: list[int]):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    sc = build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))
    return sc, build_l1_geometry(sc)


def test_shapes_and_in_loss_equals_masked() -> None:
    sc, geom = _session([3, 4, 4])
    n = int(geom.valid.sum())
    grid = build_r4_grid(geom, n_time=32)
    masks = sample_masks_r6(geom, n, n_time=32, n_rows=6, generator=_gen())
    masked, in_loss = token_flags_r6(grid, masks)
    assert masked.shape == (6, grid.total)
    assert in_loss.shape == (6, grid.total)
    assert torch.equal(in_loss, masked)  # NO margin gate — every masked token is scored
    print("[check] OK token_flags_r6 shapes (B,total) + in_loss == masked (no gate)")


def _broadcast_global_to_persensor(m: V3Masks, n_contacts: int) -> V3MasksR6:
    """Lift a GLOBAL-band r4 V3Masks to a per-sensor V3MasksR6 constant across contacts (every
    contact gets the SAME band mask) — the degenerate case where r6's masked set must equal r4's."""
    def bc(t: torch.Tensor) -> torch.Tensor:  # (R, T_b) → (R, N, T_b)
        return t[:, None, :].expand(t.shape[0], n_contacts, t.shape[1]).contiguous()

    return V3MasksR6(
        contact_mask=m.contact_mask,
        hga_mask=bc(m.hga_mask),
        mid_mask=bc(m.mid_mask),
        slow_mask=bc(m.slow_mask),
    )


def test_masked_reduces_to_r4_and_in_loss_is_r4_plus_the_gated_edges() -> None:
    # THE correctness invariant. Constant-across-sensors masks ⇒ r6 must select the SAME masked
    # token set as r4. The scored set is where the two arms deliberately part: r4 drops the block
    # edges (M14 margin gate), r6 keeps them, so r6's in_loss ⊋ r4's in_loss and the difference is
    # entirely inside r4's masked set (no token becomes scored that was not masked).
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    grid = build_r4_grid(geom, n_time=32)
    r4 = sample_masks(geom, n, n_time=32, n_rows=8, generator=_gen(7))
    r6 = _broadcast_global_to_persensor(r4, n)
    m4, l4 = token_flags(grid, r4)
    m6, l6 = token_flags_r6(grid, r6)
    assert torch.equal(m4, m6), "masked differs from r4 under constant-across-sensor masks"
    assert torch.equal(l6, m6), "r6 scores every masked token"
    assert torch.all(l4 <= l6), "r6 must score a superset of r4"
    extra = int((l6 & ~l4).sum())
    assert extra > 0, "r4's margin gate excluded nothing — test session is degenerate"
    assert torch.all((l6 & ~l4) <= m4), "the extra scored tokens must all have been masked"
    print(f"[check] OK masked == r4 exactly; in_loss = r4's + {extra} margin-gated edge tokens")


def test_temporal_is_read_per_sensor() -> None:
    # Two contacts of the SAME shaft, contradictory HGA masks: contact 0 fully HGA-masked, contact
    # 1 fully HGA-visible. Under r4 (global) or the old per-shaft r6 this state is unrepresentable;
    # here it must be honoured token-for-token, proving the per-SENSOR read.
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    t = 32
    grid = build_r4_grid(geom, n_time=t)
    B = 1
    hga = torch.zeros(B, n, t, dtype=torch.bool)
    hga[:, 0, :] = True  # contact 0 fully HGA-masked; every other contact all visible
    mid = torch.zeros(B, n, t // 2, dtype=torch.bool)
    slow = torch.zeros(B, n, t // 8, dtype=torch.bool)
    contact = torch.zeros(B, n, dtype=torch.bool)
    masks = V3MasksR6(contact_mask=contact, hga_mask=hga, mid_mask=mid, slow_mask=slow)
    masked, _ = token_flags_r6(grid, masks)
    is_hga = grid.band == 2
    c0 = is_hga & (grid.contact == 0)
    c1 = is_hga & (grid.contact == 1)
    same_shaft = int(grid.shaft[grid.contact == 0][0]) == int(grid.shaft[grid.contact == 1][0])
    assert same_shaft, "contacts 0 and 1 must share a shaft for this to test SENSOR granularity"
    assert torch.all(masked[0, c0]) and not torch.any(masked[0, c1])
    print("[check] OK band mask is read PER-SENSOR (contact0 HGA all masked, contact1 none, "
          "same shaft)")


def test_block_edges_are_scored_no_margin_gate() -> None:
    # The inverse of r4's margin-gate test: a single contiguous HGA block is masked, and ALL of it
    # is scored — including the 2 edge tokens r4 excludes for sharing raw samples with a visible
    # same-band neighbour. This is the accepted M14 trade (Ben 2026-07-23).
    sc, geom = _session([4])
    n = int(geom.valid.sum())
    t = 32
    grid = build_r4_grid(geom, n_time=t)
    B = 1
    hga = torch.zeros(B, n, t, dtype=torch.bool)
    hga[:, :, 10:20] = True  # width-10 masked block, positions 10..19 (rest visible)
    mid = torch.zeros(B, n, t // 2, dtype=torch.bool)
    slow = torch.zeros(B, n, t // 8, dtype=torch.bool)
    contact = torch.zeros(B, n, dtype=torch.bool)  # no spatial mask ⇒ the temporal set alone
    masks = V3MasksR6(contact_mask=contact, hga_mask=hga, mid_mask=mid, slow_mask=slow)
    masked, in_loss = token_flags_r6(grid, masks)
    is_hga = grid.band == 2
    sel = is_hga & (grid.contact == 0)
    pos = grid.bandpos[sel]
    order = torch.argsort(pos)
    pos, mk, il = pos[order], masked[0, sel][order], in_loss[0, sel][order]
    assert set(pos[mk].tolist()) == set(range(10, 20))  # exactly the block is masked
    assert set(pos[il].tolist()) == set(range(10, 20))  # …and ALL of it is scored, edges included
    print("[check] OK no margin gate: masked == in_loss == block 10..19 (edges 10 and 19 scored)")


def test_contact_masked_token_is_in_loss() -> None:
    # A spatially-masked contact is scored regardless of temporal masking (it was always scored —
    # r4 gated only the temporal term — so this invariant is unchanged by dropping the gate).
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    t = 32
    grid = build_r4_grid(geom, n_time=t)
    B = 1
    contact = torch.zeros(B, n, dtype=torch.bool)
    contact[:, 0] = True  # mask contact 0 entirely
    z_hga = torch.zeros(B, n, t, dtype=torch.bool)
    z_mid = torch.zeros(B, n, t // 2, dtype=torch.bool)
    z_slow = torch.zeros(B, n, t // 8, dtype=torch.bool)
    masks = V3MasksR6(contact_mask=contact, hga_mask=z_hga, mid_mask=z_mid, slow_mask=z_slow)
    masked, in_loss = token_flags_r6(grid, masks)
    c0 = grid.contact == 0
    assert torch.all(masked[0, c0]) and torch.all(in_loss[0, c0])
    print("[check] OK spatially-masked contact ⇒ all its tokens in_loss")
