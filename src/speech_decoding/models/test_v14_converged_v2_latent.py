"""P2.4 tests for the converged-v2 latent encoder + EMA teacher update."""

from __future__ import annotations

import copy

import pytest
import torch

from speech_decoding.models import v14_converged_v2 as v2
from speech_decoding.models.v14_converged_v2 import (
    LatentEncoderV2,
    SetPoolV2,
    ema_update,
)

N_PARCELS = 62


def _seeds(clip_s=1.0, b=2, P=3, k=2, d=32):
    bands = v2.bands_for_clip_len(clip_s)
    S = sum(bd.n_tokens for bd in bands)
    _, _, cell_time_slot = v2.token_metadata(bands)
    torch.manual_seed(0)
    seeds = torch.randn(b, P, k, S, d)
    parcel_labels = torch.tensor([5, 9, 20])[:P]
    return seeds, parcel_labels, cell_time_slot, S


@pytest.mark.parametrize("clip_s, total", [(1.0, 22), (5.0, 110)])
def test_latent_output_shape(clip_s, total):
    seeds, labels, slot, S = _seeds(clip_s)
    enc = LatentEncoderV2(d_model=32, n_heads=4, n_layers=2, n_parcels=N_PARCELS)
    out = enc(seeds, labels, slot)
    assert out.shape == seeds.shape == (2, 3, 2, total, 32)


def test_latent_rope_table_sized_for_5s():
    enc = LatentEncoderV2(d_model=32, n_heads=4, n_layers=1, n_parcels=N_PARCELS)
    # 5 s HGA reaches time_slot 79 ⇒ table ≥ 80 slots.
    bands5 = v2.bands_for_clip_len(5.0)
    assert enc.key_rope.shape[1] >= v2._max_time_slot(bands5) + 1


def test_latent_rejects_wrong_slot_length():
    seeds, labels, slot, S = _seeds()
    enc = LatentEncoderV2(d_model=32, n_heads=4, n_layers=1, n_parcels=N_PARCELS)
    with pytest.raises(ValueError, match="cell_time_slot must be"):
        enc(seeds, labels, slot[:-1])


def test_latent_key_mask_leak_free():
    """Masked seeds never key ⇒ visible seeds' latent outputs are value-invariant."""
    seeds, labels, slot, S = _seeds(P=2)
    enc = LatentEncoderV2(d_model=32, n_heads=4, n_layers=2, n_parcels=N_PARCELS)
    enc.eval()
    B, P, k, _, d = seeds.shape
    key_mask = torch.ones(B, P, k, S, dtype=torch.bool)
    key_mask[:, 1] = False                       # hide all of parcel 1
    with torch.no_grad():
        out1 = enc(seeds, labels, slot, key_mask=key_mask)
        s2 = seeds.clone()
        s2[:, 1] += 5.0                          # corrupt the masked parcel
        out2 = enc(s2, labels, slot, key_mask=key_mask)
    assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-6)   # visible unchanged
    assert not torch.allclose(out1[:, 1], out2[:, 1])          # masked free


def test_latent_gradient_flows():
    seeds, labels, slot, S = _seeds()
    seeds = seeds.requires_grad_(True)
    enc = LatentEncoderV2(d_model=32, n_heads=4, n_layers=2, n_parcels=N_PARCELS)
    enc(seeds, labels, slot).sum().backward()
    assert seeds.grad is not None and torch.isfinite(seeds.grad).all()


def test_ema_update_moves_teacher_toward_student():
    student = SetPoolV2(32, 4, k=2, n_parcels=N_PARCELS, n_op=2)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)                          # student now teacher + 1
    before = teacher.base_seed.clone()
    ema_update(teacher, student, tau=0.9)
    # θ_t ← 0.9·θ_t + 0.1·θ_s = θ_t + 0.1·(θ_s − θ_t) = before + 0.1·1
    assert torch.allclose(teacher.base_seed, before + 0.1, atol=1e-6)


def test_ema_tau_zero_copies_student():
    student = SetPoolV2(16, 2, k=2, n_parcels=N_PARCELS, n_op=2)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        for p in student.parameters():
            p.mul_(3.0)
    ema_update(teacher, student, tau=0.0)
    for t, s in zip(teacher.parameters(), student.parameters()):
        assert torch.allclose(t, s)


def test_ema_buffers_copied():
    """RoPE/non-learned buffers track the student exactly (hard copy, not EMA)."""
    student = LatentEncoderV2(32, 4, 1, N_PARCELS)
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        student.key_rope.add_(2.0)
    ema_update(teacher, student, tau=0.99)
    assert torch.allclose(teacher.key_rope, student.key_rope)


def test_ema_rejects_bad_tau():
    student = SetPoolV2(16, 2, k=1, n_parcels=N_PARCELS, n_op=2)
    teacher = copy.deepcopy(student)
    with pytest.raises(ValueError, match="tau must be"):
        ema_update(teacher, student, tau=1.5)
