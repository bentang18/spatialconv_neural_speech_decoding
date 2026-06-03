"""P3 geometry lock: student T_p(5 s) == teacher pooled frames == 40 @ 8 Hz.

Gate-B adversarial audit (2026-06-03) confirmed the geometry is mechanically
correct but flagged that the load-bearing equality is reached by a *one-frame*
floor-absorption: the front-end emits 81 STFT frames for a 5 s clip (an odd
center-pad frame on top of the nominal 80), and the (3, 2) patch stem floors
that to T_p=40. That 40 is one frame from flipping to 41 (T_bins=82, i.e.
clip_len >= 5.0625 s, or a smaller hop). The P3 distill step asserts
``student.shape[-2] == teacher.shape[-2]`` at *runtime*, so a drift crashes
loudly rather than silently — but nothing pinned the invariant at CI time.
These tests do, across every STFT front-end the dispatch can select (the
config-drift audit verified all four are hop=128-equivalent).
"""

from __future__ import annotations

import pytest

from speech_decoding.extractors.view import LogStftView, MultiStftView
from speech_decoding.extractors.ref_aug import RefAugMultiStftView
from speech_decoding.extractors.whisper_teacher_pool import (
    _EXPECTED_N_OUT,
    triangular_pool_50_to_8_hz,
)
from speech_decoding.models.v14_encoder import _PatchStem, V14ParcelPerceiverModel

import torch

# Whole-pipeline lock: 5 s SSL/distill clips → T_p=40; 1 s P4 readout clips → 8.
_FIVE_S_BINS = 81
_ONE_S_BINS = 17
_FIVE_S_TP = 40
_ONE_S_TP = 8


def _stem() -> _PatchStem:
    # The dispatch builds the stem with the FE-02 default (3, 2) kernel/stride.
    return _PatchStem(d_model=8, kernel_freq=3, kernel_time=2)


@pytest.mark.parametrize(
    "view",
    [
        MultiStftView(event_types="Ieeg"),               # dispatch default (C2)
        RefAugMultiStftView(event_types="Ieeg"),         # ref-aug sister
        LogStftView(event_types="Ieeg"),                 # F-single-STFT sister
    ],
    ids=["multi_stft", "ref_aug_multi_stft", "log_stft_single"],
)
def test_all_frontends_share_hop128_frame_geometry(view) -> None:
    # Every selectable STFT front-end must yield the SAME frame count, so
    # swapping the front-end can never silently move T_p off the lock.
    assert view.n_time_bins_for_duration(5.0) == _FIVE_S_BINS
    assert view.n_time_bins_for_duration(1.0) == _ONE_S_BINS


def test_student_tp_equals_teacher_pool_at_5s() -> None:
    # THE LOCK: 5 s clip → 81 STFT frames → (3,2) stem → T_p=40, and the
    # Whisper teacher pools 250 → 40. Both must be 40 or the P3 SmoothL1 is
    # frame-misaligned (the runtime assert would fire).
    stem = _stem()
    student_tp = stem.n_time_patches(_FIVE_S_BINS)
    assert student_tp == _FIVE_S_TP

    pooled = triangular_pool_50_to_8_hz(torch.zeros(1, 250, 1280))
    teacher_frames = pooled.shape[-2]
    assert teacher_frames == _EXPECTED_N_OUT == _FIVE_S_TP
    assert student_tp == teacher_frames  # the equality P3 depends on


def test_p4_readout_tp_is_8_at_1s() -> None:
    assert _stem().n_time_patches(_ONE_S_BINS) == _ONE_S_TP


def test_rope_prefix_property_across_phases() -> None:
    """Cross-phase invariant the whole P1→P4 story rests on: an encoder built
    with a LARGER time ceiling (a 5 s pretrain) and one built with a SMALLER
    ceiling (a 1 s P4 readout) must produce bit-identical output on the SAME
    short clip — the RoPE table is absolute-position and read as a strict
    prefix, so positions [0..T_p) get identical rotations regardless of ceiling.
    Gate-B's phase/RoPE audit proved maxdiff=0.0 but no test pinned it.
    """
    # n_time_bins is only the RoPE ceiling; the forward derives T_p from the
    # input. Big ceiling=8 (T_p<=4); small ceiling=4 (T_p<=2). Both see a
    # 4-frame input → T_p=2, exercising the prefix slice on the big one.
    common = dict(
        d_model=8, n_freq_bins=6, k_parcels=6,
        n_heads=2, depth_self_attn=1,
    )
    big = V14ParcelPerceiverModel(n_time_bins=8, **common).eval()
    small = V14ParcelPerceiverModel(n_time_bins=4, **common).eval()
    # Share learned weights; RoPE buffers are persistent=False so they are
    # absent from the state_dict and each instance keeps its own correct-size
    # table — load_state_dict(strict=True) must report no missing/unexpected.
    small.load_state_dict(big.state_dict())

    B, C = 2, 4
    electrodes = torch.randn(B, C, 4, common["n_freq_bins"])  # time axis = 4
    support = torch.eye(common["k_parcels"])[None, :C, :].float().expand(B, C, -1).contiguous()
    shaft_mask = torch.ones(B, C, dtype=torch.bool)

    with torch.no_grad():
        out_big = big(electrodes, support, shaft_mask=shaft_mask)
        out_small = small(electrodes, support, shaft_mask=shaft_mask)
    assert out_big.shape == out_small.shape
    assert torch.equal(out_big, out_small), (
        f"RoPE prefix property broken: maxdiff={(out_big - out_small).abs().max()}"
    )


def test_one_frame_cliff_is_where_the_audit_said() -> None:
    # Sensitivity guard: document/pin exactly where the floor flips off 40.
    # T_bins 79->39, 80->40, 81->40 (live), 82->41. If a future change pushes
    # the 5 s frame count to 82 (e.g. clip_len >= 5.0625 s or a smaller hop)
    # this test flips first, before the runtime P3 assert ever runs.
    stem = _stem()
    assert [stem.n_time_patches(b) for b in (79, 80, 81, 82, 83)] == [39, 40, 40, 41, 41]
    # The live 5 s frame count sits on the UPPER edge of the 40-plateau.
    assert MultiStftView(event_types="Ieeg").n_time_bins_for_duration(5.0) == _FIVE_S_BINS
    assert stem.n_time_patches(_FIVE_S_BINS) == _FIVE_S_TP
