"""r6 flat-grid mask flags — ``token_flags_r6``.

r6 reuses ``build_r4_grid`` (identical 3-band 32 Hz-lattice grid) and adds ``token_flags_r6``:
r4's margin-gated leak-safety (``in_loss`` ⊆ ``masked``, M14 margin 2 — KEPT) on r5-style
PER-SHAFT band masks ``(B,S,T_b)``. THE correctness test: it reduces EXACTLY to r4's
``token_flags`` when every shaft shares one mask. Invariants named + asserted + printed.
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


def test_shapes_and_in_loss_subset_of_masked() -> None:
    sc, geom = _session([3, 4, 4])
    n = int(geom.valid.sum())
    grid = build_r4_grid(geom, n_time=32)
    masks = sample_masks_r6(geom, n, n_time=32, n_rows=6, generator=_gen())
    masked, in_loss = token_flags_r6(grid, masks)
    assert masked.shape == (6, grid.total)
    assert in_loss.shape == (6, grid.total)
    assert torch.all(in_loss <= masked)  # in_loss ⊆ masked (bool ≤)
    print("[check] OK token_flags_r6 shapes (B,total) + in_loss ⊆ masked")


def _broadcast_global_to_pershaft(m: V3Masks, n_shafts: int) -> V3MasksR6:
    """Lift a GLOBAL-band r4 V3Masks to a per-shaft V3MasksR6 that is constant across shafts
    (every shaft gets the SAME band mask) — the degenerate case where r6 must equal r4."""
    def bc(t: torch.Tensor) -> torch.Tensor:  # (R, T_b) → (R, S, T_b)
        return t[:, None, :].expand(t.shape[0], n_shafts, t.shape[1]).contiguous()

    return V3MasksR6(
        contact_mask=m.contact_mask,
        hga_mask=bc(m.hga_mask),
        mid_mask=bc(m.mid_mask),
        slow_mask=bc(m.slow_mask),
    )


def test_reduces_to_r4_when_masks_are_global() -> None:
    # THE correctness invariant: with the same per-shaft mask on every shaft, r6's per-shaft
    # margin gate must produce byte-identical (masked, in_loss) to r4's global token_flags.
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    grid = build_r4_grid(geom, n_time=32)
    r4 = sample_masks(geom, n, n_time=32, n_rows=8, generator=_gen(7))
    r6 = _broadcast_global_to_pershaft(r4, geom.n_shafts)
    m4, l4 = token_flags(grid, r4)
    m6, l6 = token_flags_r6(grid, r6)
    assert torch.equal(m4, m6), "masked differs from r4 under global masks"
    assert torch.equal(l4, l6), "in_loss differs from r4 under global masks"
    print("[check] OK r6 == r4 token_flags exactly when band masks are global (per-shaft reduces)")


def test_temporal_is_read_per_shaft() -> None:
    # Two shafts, contradictory HGA masks: shaft 0 fully HGA-masked, shaft 1 fully HGA-visible.
    # A shaft-0 HGA token must be masked; a shaft-1 HGA token must not — proving the per-shaft read.
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    t = 32
    grid = build_r4_grid(geom, n_time=t)
    S = geom.n_shafts
    B = 1
    hga = torch.zeros(B, S, t, dtype=torch.bool)
    hga[:, 0, :] = True  # shaft 0 fully HGA-masked; shaft 1 all visible
    mid = torch.zeros(B, S, t // 2, dtype=torch.bool)
    slow = torch.zeros(B, S, t // 8, dtype=torch.bool)
    contact = torch.zeros(B, n, dtype=torch.bool)
    masks = V3MasksR6(contact_mask=contact, hga_mask=hga, mid_mask=mid, slow_mask=slow)
    masked, _ = token_flags_r6(grid, masks)
    is_hga = grid.band == 2
    s0 = is_hga & (grid.shaft == 0)
    s1 = is_hga & (grid.shaft == 1)
    assert torch.all(masked[0, s0]) and not torch.any(masked[0, s1])
    print("[check] OK band mask is read PER-SHAFT (shaft0 HGA all masked, shaft1 none)")


def test_margin_gate_excludes_block_edges() -> None:
    # One shaft, a single contiguous HGA block masked on all shafts; the block's INTERIOR
    # (dist ≥ MARGIN=2 from a visible frame) is in_loss, its 2 EDGE tokens are masked-not-in_loss.
    sc, geom = _session([4])
    n = int(geom.valid.sum())
    t = 32
    grid = build_r4_grid(geom, n_time=t)
    S = geom.n_shafts
    B = 1
    hga = torch.zeros(B, S, t, dtype=torch.bool)
    hga[:, :, 10:20] = True  # width-10 masked block, positions 10..19 (rest visible)
    mid = torch.zeros(B, S, t // 2, dtype=torch.bool)
    slow = torch.zeros(B, S, t // 8, dtype=torch.bool)
    contact = torch.zeros(B, n, dtype=torch.bool)  # no spatial mask ⇒ margin gate alone decides
    masks = V3MasksR6(contact_mask=contact, hga_mask=hga, mid_mask=mid, slow_mask=slow)
    masked, in_loss = token_flags_r6(grid, masks)
    is_hga = grid.band == 2
    # gather (bandpos → masked/in_loss) for one shaft's HGA tokens
    sel = is_hga & (grid.shaft == 0)
    pos = grid.bandpos[sel]
    mk = masked[0, sel]
    il = in_loss[0, sel]
    order = torch.argsort(pos)
    pos, mk, il = pos[order], mk[order], il[order]
    masked_pos = set(pos[mk].tolist())
    in_loss_pos = set(pos[il].tolist())
    assert masked_pos == set(range(10, 20))  # exactly the block is masked
    # margin 2 = dist ≥ 2 from a visible frame (r4 semantics): only the outermost edges 10 (dist 1)
    # and 19 (dist 1) are excluded; 11 and 18 sit at dist exactly 2 ⇒ in_loss. So 11..18 scored.
    assert in_loss_pos == set(range(11, 19)), sorted(in_loss_pos)
    print(f"[check] OK margin gate: masked={sorted(masked_pos)} in_loss={sorted(in_loss_pos)}")


def test_contact_masked_token_is_in_loss() -> None:
    # A spatially-masked contact is leak-proof (no own visible same-band frame) ⇒ ALWAYS in_loss,
    # regardless of temporal masking.
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    t = 32
    grid = build_r4_grid(geom, n_time=t)
    S = geom.n_shafts
    B = 1
    contact = torch.zeros(B, n, dtype=torch.bool)
    contact[:, 0] = True  # mask contact 0 entirely
    z_hga = torch.zeros(B, S, t, dtype=torch.bool)
    z_mid = torch.zeros(B, S, t // 2, dtype=torch.bool)
    z_slow = torch.zeros(B, S, t // 8, dtype=torch.bool)
    masks = V3MasksR6(contact_mask=contact, hga_mask=z_hga, mid_mask=z_mid, slow_mask=z_slow)
    masked, in_loss = token_flags_r6(grid, masks)
    c0 = grid.contact == 0
    assert torch.all(masked[0, c0]) and torch.all(in_loss[0, c0])
    print("[check] OK spatially-masked contact ⇒ all its tokens in_loss (leak-proof)")
