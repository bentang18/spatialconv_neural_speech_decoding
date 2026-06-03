"""Tests for the B36 masked-JEPA loss terms.

Focus: the C5 freq-patch exclusion on the P1 front-end M2 loss. (The core P1/P2
masked-prediction behavior is exercised in ``experiments/test_v14_joint_module.py``;
here we pin only the per-corpus valid-bin contract.)
"""

from __future__ import annotations

import torch

from speech_decoding.ssl.masked_jepa import p1_frontend_m2_loss


def test_c5_p1_loss_excludes_invalid_freq_patches_from_target() -> None:
    """With freq_patch_valid masking F-patches 7–9, masked cells on those
    patches are dropped from the L1 target: ``n_masked`` counts only
    valid-patch masked cells, the loss equals the loss on the pre-ANDed
    token_mask, and the values at the invalid patches cannot move the loss."""
    torch.manual_seed(0)
    B, C, F_p, T_p, d = 2, 3, 10, 4, 8
    student = torch.randn(B, C, F_p, T_p, d)
    teacher = torch.randn(B, C, F_p, T_p, d)
    token_mask = torch.rand(B, C, F_p, T_p) < 0.5  # ~half masked

    fpv = torch.tensor([True] * 7 + [False] * 3)
    bd = p1_frontend_m2_loss(
        student_m2=student, teacher_m2=teacher, token_mask=token_mask,
        freq_patch_valid=fpv,
    )

    # Reference: AND the freq mask into token_mask up front, no freq arg.
    ref_mask = token_mask & fpv.view(1, 1, F_p, 1)
    ref = p1_frontend_m2_loss(
        student_m2=student, teacher_m2=teacher, token_mask=ref_mask,
    )
    assert bd.n_masked == int(ref_mask.sum())
    assert bd.n_masked < int(token_mask.sum()), "some invalid-patch cells dropped"
    torch.testing.assert_close(bd.total, ref.total, atol=0, rtol=0)

    # No masked cell on F-patches 7–9 contributes: garbaging those student /
    # teacher values must NOT change the loss.
    s2 = student.clone()
    s2[:, :, 7:] = 123.0
    t2 = teacher.clone()
    t2[:, :, 7:] = -77.0
    bd2 = p1_frontend_m2_loss(
        student_m2=s2, teacher_m2=t2, token_mask=token_mask, freq_patch_valid=fpv,
    )
    torch.testing.assert_close(bd2.total, bd.total, atol=0, rtol=0)


def test_c5_p1_loss_none_is_byte_identical() -> None:
    """freq_patch_valid=None and an all-True mask both reduce to the pre-C5
    loss — the BT no-op (all freq patches valid)."""
    torch.manual_seed(1)
    B, C, F_p, T_p, d = 2, 3, 10, 4, 8
    student = torch.randn(B, C, F_p, T_p, d)
    teacher = torch.randn(B, C, F_p, T_p, d)
    token_mask = torch.rand(B, C, F_p, T_p) < 0.5

    base = p1_frontend_m2_loss(
        student_m2=student, teacher_m2=teacher, token_mask=token_mask,
    )
    allt = p1_frontend_m2_loss(
        student_m2=student, teacher_m2=teacher, token_mask=token_mask,
        freq_patch_valid=torch.ones(F_p, dtype=torch.bool),
    )
    assert base.n_masked == allt.n_masked
    torch.testing.assert_close(base.total, allt.total, atol=0, rtol=0)


def test_c5_p1_loss_accepts_batched_freq_patch_valid() -> None:
    """A per-sample (B, F_p) freq_patch_valid is honored independently per
    clip (so a future mixed-corpus batch masks each clip by its own bins)."""
    torch.manual_seed(2)
    B, C, F_p, T_p, d = 2, 3, 10, 4, 8
    student = torch.randn(B, C, F_p, T_p, d)
    teacher = torch.randn(B, C, F_p, T_p, d)
    token_mask = torch.ones(B, C, F_p, T_p, dtype=torch.bool)

    fpv = torch.ones(B, F_p, dtype=torch.bool)
    fpv[0, 7:] = False  # clip 0 = SWEC (7–9 invalid); clip 1 = all valid
    bd = p1_frontend_m2_loss(
        student_m2=student, teacher_m2=teacher, token_mask=token_mask,
        freq_patch_valid=fpv,
    )
    expected = C * 7 * T_p + C * F_p * T_p  # clip0 valid patches + clip1 all
    assert bd.n_masked == expected
