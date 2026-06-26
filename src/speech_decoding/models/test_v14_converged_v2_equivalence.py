"""P2.8 dense==static equivalence oracle for converged-v2 packing.

The static (drop-not-pad) hot path packs visible cells (frontend) and untubed
parcels (latent) into dense tensors; the dense oracle keeps full shapes and uses
key-masks. They must produce identical outputs at the compared positions. This is
premortem landmine #3 — toy tests pass while real session-homogeneous batches
break — so each stage is checked over realistic ``(C, P, n_p)`` incl. an isolated
``n_p=1`` parcel and per-parcel-DIFFERING M2 (the case forcing per-row gather +
per-row RoPE).

Decomposition by where mixing happens:
  * frontend MIXES cells (cross-cell SA) ⇒ cell-packing is non-trivial (THE test).
  * latent MIXES parcels ⇒ parcel-packing via drop vs key-mask (cost-center).
  * pool is cell-INDEPENDENT ⇒ packing is gather-commutes-with-pool.
"""

from __future__ import annotations

import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    FrontendEncoderV2,
    LatentEncoderV2,
    SetPoolV2,
    active_parcels,
    cell_operator_index,
    n_operators,
    sample_m2_masks_v2,
)

N_PARCELS = 62
D = 32


def _gen(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _vis_index(visible: torch.Tensor, s_vis: int) -> torch.Tensor:
    """``(B,C,S)`` bool (True=visible, constant count s_vis/row) → ``(B,C,s_vis)``
    long of the visible cell positions in ascending order."""
    order = visible.int().argsort(dim=-1, descending=True, stable=True)
    return order[..., :s_vis].sort(dim=-1).values


def test_frontend_cell_packing_equals_keymask():
    """Per-parcel-differing M2: packing the visible cells (per-row gather + per-row
    RoPE) == running all S cells with the masked ones key-masked. n_p=1 included."""
    torch.manual_seed(0)
    bands = v2.bands_for_clip_len(5.0)
    # 6 electrodes → parcels {5×3, 9×1 (isolated), 20×2}
    poe = torch.tensor([5, 5, 5, 9, 20, 20])
    labels, membership = active_parcels(poe)
    P, C = membership.shape
    B = 3
    m2 = sample_m2_masks_v2(B, P, bands, _gen(1))            # (B,P,S) parcel-uniform
    # broadcast parcel mask → per-electrode (B,C,S)
    parcel_idx = torch.tensor([int((labels == p).nonzero()) for p in poe])  # (C,)
    elec_mask = m2[:, parcel_idx, :]                          # (B,C,S) True=masked
    visible = ~elec_mask
    s_vis = int(visible[0, 0].sum())
    assert (visible.sum(-1) == s_vis).all()                  # constant per row

    enc = FrontendEncoderV2(D, n_heads=4, n_layers=3, bands=bands)
    enc.eval()
    lfs = torch.randn(B, C, bands[0].n_freq_bins, bands[0].n_time_frames)
    hga = torch.randn(B, C, bands[1].n_freq_bins, bands[1].n_time_frames)

    with torch.no_grad():
        dense = enc(lfs, hga, key_mask=visible)              # (B,C,S,d)
        # static: tokenize, gather visible cells + their freq/slot tags, encode.
        tok = enc.tokenizer(lfs, hga)                        # (B,C,S,d)
        vis_idx = _vis_index(visible, s_vis)                 # (B,C,s_vis)
        gtok = torch.gather(tok, 2, vis_idx.unsqueeze(-1).expand(-1, -1, -1, D))
        gfreq = enc.tokenizer.freq_patch_id[vis_idx]         # (B,C,s_vis)
        gslot = enc.tokenizer.time_slot[vis_idx]             # (B,C,s_vis)
        static = enc.encode_tokens(gtok, gfreq, gslot)       # (B,C,s_vis,d)
        dense_vis = torch.gather(dense, 2, vis_idx.unsqueeze(-1).expand(-1, -1, -1, D))

    assert torch.allclose(dense_vis, static, atol=1e-5)


def test_pool_cell_packing_equals_full_then_select():
    """Pool is cell-independent: pooling a visible cell-subset == pooling all cells
    then selecting (with the cell_patch gathered consistently)."""
    torch.manual_seed(0)
    bands = v2.bands_for_clip_len(5.0)
    S = sum(b.n_tokens for b in bands)
    poe = torch.tensor([5, 5, 9, 9])
    labels, membership = active_parcels(poe)
    cell_patch = cell_operator_index(bands, tie_lfs=True)     # (S,)
    pool = SetPoolV2(D, 4, k=2, n_parcels=N_PARCELS, n_op=n_operators(True))
    pool.eval()
    x = torch.randn(2, 4, S, D)
    # a shared visible subset (cell-independence ⇒ any subset works)
    sub = torch.tensor([0, 1, 5, 10, 20, 33, 60, 90, 109])
    with torch.no_grad():
        full = pool(x, membership, labels, cell_patch)       # (B,P,k,S,d)
        packed = pool(x[:, :, sub], membership, labels, cell_patch[sub])
    assert torch.allclose(full[:, :, :, sub], packed, atol=1e-5)


def test_latent_parcel_packing_equals_keymask():
    """Latent cost-center: dropping tubed parcels (static) == key-masking them
    (dense). Untubed parcels' deep latents must match."""
    torch.manual_seed(0)
    bands = v2.bands_for_clip_len(5.0)
    S = sum(b.n_tokens for b in bands)
    _, _, slot = v2.token_metadata(bands)
    B, P, k = 3, 5, 2
    labels = torch.tensor([3, 7, 12, 20, 41])[:P]
    seeds = torch.randn(B, P, k, S, D)
    # tube the SAME 2 parcels for every clip (shared P_vis ⇒ shared labels gather)
    tubed = torch.tensor([1, 3])
    untubed = torch.tensor([0, 2, 4])
    key_mask = torch.ones(B, P, k, S, dtype=torch.bool)
    key_mask[:, tubed] = False                               # hide tubed parcels

    enc = LatentEncoderV2(D, 4, n_layers=3, n_parcels=N_PARCELS)
    enc.eval()
    with torch.no_grad():
        dense = enc(seeds, labels, slot, key_mask=key_mask)  # (B,P,k,S,d)
        static = enc(seeds[:, untubed], labels[untubed], slot)  # (B,P_vis,k,S,d)
    assert torch.allclose(dense[:, untubed], static, atol=1e-5)


def test_latent_tubed_leak_free_under_packing():
    """Sanity: the dropped tubed parcels do NOT influence untubed latents (their
    seed values are irrelevant) — the property that makes drop==mask hold."""
    torch.manual_seed(0)
    bands = v2.bands_for_clip_len(1.0)
    S = sum(b.n_tokens for b in bands)
    _, _, slot = v2.token_metadata(bands)
    B, P, k = 2, 4, 2
    labels = torch.tensor([3, 7, 12, 20])
    seeds = torch.randn(B, P, k, S, D)
    tubed = torch.tensor([1])
    untubed = torch.tensor([0, 2, 3])
    key_mask = torch.ones(B, P, k, S, dtype=torch.bool)
    key_mask[:, tubed] = False
    enc = LatentEncoderV2(D, 4, n_layers=2, n_parcels=N_PARCELS)
    enc.eval()
    with torch.no_grad():
        out1 = enc(seeds, labels, slot, key_mask=key_mask)
        s2 = seeds.clone()
        s2[:, tubed] += 11.0                                 # corrupt tubed seeds
        out2 = enc(s2, labels, slot, key_mask=key_mask)
    assert torch.allclose(out1[:, untubed], out2[:, untubed], atol=1e-6)
