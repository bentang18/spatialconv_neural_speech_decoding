"""v14_converged_v3 — CROSS-SESSION shaft-pack isolation (shaft-level batching gate).

The linchpin of shaft-level (cross-patient) batching: the L1-only towers carry NO
session identity — a token is defined solely by its (depth, time_pos, parcel_id) coords
and its cu_seqlens block. So packing shafts from DIFFERENT sessions into one flat grid
must be CORRECTNESS-NEUTRAL: each shaft's per-token output is bit-identical whether it is
run alone or packed alongside shafts of a different patient (different size, different
parcels, different depth-gaps).

test_towers_flat already proves block-diagonality WITHIN one session's montage. This file
proves the cross-session case that the batching change actually relies on: a HETEROGENEOUS
multi-shaft grid (built here to stand in for a mixed-patient pack) is decomposable, token
for token, into the single-shaft runs. If this holds, the whole shaft-batching premise is
sound and the model forward needs no change — only the data layer that assembles the pack.

Two distinct guarantees, tested at the right tolerance for each:
  * SAME-grid stability (perturb one shaft, hold the pack shape): BIT-exact (torch.equal) —
    identical matmul shapes, so a coord bleed is the ONLY thing that can move a flanking token.
  * ALONE vs PACKED (different pack shapes): tight allclose (~1e-6, atol=1e-5 as test_towers_flat
    uses). The CPU reference runs ONE matmul over the whole padded sequence and masks cross-block
    scores to -1e4 (underflows to exactly 0 ⇒ NO leak), but changing the K-dimension (tA vs tA+tB)
    reassociates the real terms' accumulation → fp noise. A real leak is O(1), not 1e-6 — and the
    same-grid test pins that O(1) channel shut at bit-exactness. So 1e-6 here is arithmetic
    reassociation, not a cross-patient coordinate bleed.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.towers import build_encoder, build_predictor

T = 16  # SLOW 2, MID 8, HGA 16 tokens per contact ⇒ k_full 26.
D_ENC, D_PRED = 256, 128
N_PARCELS = 32


def _grid(labels, parcels):
    """One shaft-set → (sidecar, grid, per-token parcel id)."""
    sc = build_sidecar(list(labels), parcel_id=torch.tensor(parcels))
    grid = build_r4_grid(build_l1_geometry(sc), n_time=T)
    return sc, grid, sc.parcel_id[grid.contact]  # (total,)


# Two stand-in "sessions", each a single shaft — deliberately UNLIKE each other:
#   A: shaft LA, 3 contacts, parcel 5, depth-gap at 3 (1,2,4).
#   B: shaft RB, 5 contacts, parcel 12, depth-gap at 4 (1,2,3,5,6).
# Different size, parcel, and gap pattern ⇒ nothing accidentally symmetric.
_A_LABELS, _A_PARCELS = ["LA1", "LA2", "LA4"], [5, 5, 5]
_B_LABELS, _B_PARCELS = ["RB1", "RB2", "RB3", "RB5", "RB6"], [12, 12, 12, 12, 12]


ATOL = 1e-5  # test_towers_flat's block-diagonal convention; a real leak is O(1), not this.


def _pack_two():
    """A and B run alone, plus the combined [A, B] grid (shaft-major ⇒ A then B)."""
    _, gA, pidA = _grid(_A_LABELS, _A_PARCELS)
    _, gB, pidB = _grid(_B_LABELS, _B_PARCELS)
    _, gAB, pidAB = _grid(_A_LABELS + _B_LABELS, _A_PARCELS + _B_PARCELS)
    return (gA, pidA), (gB, pidB), (gAB, pidAB)


def test_cross_session_two_shaft_pack_matches_alone() -> None:
    """Pack two different-patient shafts; each shaft's encoder AND predictor output matches
    running that shaft alone to fp-matmul-reassociation tolerance (~1e-6 ≪ any real leak)."""
    torch.manual_seed(0)
    (gA, pidA), (gB, pidB), (gAB, pidAB) = _pack_two()
    tA, tB = gA.total, gB.total
    assert gAB.total == tA + tB  # shaft-major concat: A's tokens first, then B's.

    for name, build, d in (("encoder", build_encoder, D_ENC), ("predictor", build_predictor, D_PRED)):
        tower = build(n_parcels=N_PARCELS).eval()
        xA = torch.randn(1, tA, d)
        xB = torch.randn(1, tB, d)
        xAB = torch.cat([xA, xB], dim=1)  # matches gAB token order (A then B)

        oA = tower.forward_flat(xA, gA, pidA)
        oB = tower.forward_flat(xB, gB, pidB)
        oAB = tower.forward_flat(xAB, gAB, pidAB)

        dA = (oAB[:, :tA] - oA).abs().max().item()
        dB = (oAB[:, tA:] - oB).abs().max().item()
        okA, okB = dA <= ATOL, dB <= ATOL
        print(f"[check] {name}: A-in-pack≈A-alone (max|Δ|={dA:.1e}); "
              f"B-in-pack≈B-alone (max|Δ|={dB:.1e}) [≪ O(1) leak] "
              f"{'OK' if okA and okB else 'VIOLATED'}")
        assert okA and okB, f"{name}: cross-session pack leaked (A={dA:.2e}, B={dB:.2e})"


def test_perturbing_one_patient_shaft_leaves_the_others_bit_stable() -> None:
    """Three heterogeneous single-shaft 'patients' in one pack; perturbing the middle
    shaft's input tokens moves ONLY its own output — the flanking patients are bit-stable."""
    torch.manual_seed(1)
    labels = _A_LABELS + _B_LABELS + ["MC1", "MC2"]
    parcels = _A_PARCELS + _B_PARCELS + [20, 20]
    _, grid, pid = _grid(labels, parcels)
    enc = build_encoder(n_parcels=N_PARCELS).eval()

    x = torch.randn(1, grid.total, D_ENC)
    out = enc.forward_flat(x, grid, pid)

    mid = grid.shaft == 1  # the RB 'patient'
    others = grid.shaft != 1
    x2 = x.clone()
    x2[0, mid] += torch.randn_like(x2[0, mid]) * 3.0
    out2 = enc.forward_flat(x2, grid, pid)

    others_stable = torch.equal(out[0, others], out2[0, others])
    mid_moved = not torch.allclose(out[0, mid], out2[0, mid], atol=1e-4)
    print(f"[check] perturb patient-shaft 1: flanking patients bit-stable ({others_stable}); "
          f"perturbed shaft moved ({mid_moved}) {'OK' if others_stable and mid_moved else 'VIOLATED'}")
    assert others_stable and mid_moved
