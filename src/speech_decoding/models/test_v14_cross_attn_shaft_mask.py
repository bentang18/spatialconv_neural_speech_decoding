"""TST-MASK-001 (Phase-2 shaft-mask 5/27 PM final spec):
end-to-end check of the shaft-mask path through the v14 encoder.

Three things must hold pre-Phase-2 dispatch:

1. Shaft-masked electrode K/V positions contribute < 1e-6 attention
   weight in cross-attn ❺ (verified by perturbing the masked
   electrode's tokens and confirming the output is bit-identical).
2. The teacher forward never sees ``shaft_mask`` (B03d asymmetry — the
   *caller* enforces this by passing ``shaft_mask=None`` on the teacher
   path; this test pins that the encoder doesn't sneak a shaft-mask
   in via shared state).
3. ShaftMaskExtractor produces K=1 for every ``N_shafts >= 2``
   (BT 7–10, D-cohort ≫ 8, single-shaft AJILE12 collapses to K=0).
   The 5/27 PM spec drops the AM K ≤ 2 fraction formula entirely.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.extractors.shaft_mask import (
    ShaftMaskExtractor,
    SubjectAnatomy,
    default_k,
)
from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel


C_MAX = 8
K_PARCELS = 6
F_BINS = 6
T_BINS = 4


def _build_model() -> torch.nn.Module:
    """Build the bare ``V14ParcelPerceiverModel`` (no classification head).
    The head wrapper does not pipe ``shaft_mask`` — and shouldn't, since
    the shaft-mask path is SSL-only — so the test exercises the encoder
    directly."""
    return V14ParcelPerceiverModel(
        n_freq_bins=F_BINS, n_time_bins=T_BINS, k_parcels=K_PARCELS,
        d_model=32, n_heads=4, depth_self_attn=1, m_sub_slots=2,
    )


def _make_anatomy_from_shafts(shaft_sizes: list[int]) -> SubjectAnatomy:
    n_total = sum(shaft_sizes)
    shaft_ids = torch.zeros(n_total, dtype=torch.long)
    i = 0
    for sid, size in enumerate(shaft_sizes):
        for _ in range(size):
            shaft_ids[i] = sid
            i += 1
    contact_pos = torch.zeros(n_total, dtype=torch.float32)
    return SubjectAnatomy(shaft_ids=shaft_ids, contact_pos=contact_pos)


# --- (1) shaft-masked positions contribute zero attention -------------


def test_shaft_masked_positions_contribute_zero_attention() -> None:
    """Perturb tokens at shaft-masked electrodes; the model output must
    be bit-identical (within numerical noise) — the only way that holds
    is if the cross-attn weight on those positions is < 1e-6 ≈ exp(-1e4)."""
    torch.manual_seed(0)
    model = _build_model().eval()
    electrodes_a = torch.randn(1, C_MAX, T_BINS, F_BINS)
    electrodes_b = electrodes_a.clone()
    support = torch.zeros(1, C_MAX, K_PARCELS)
    for c in range(C_MAX):
        support[0, c, c % K_PARCELS] = 1.0

    # Mask electrodes 0, 2, 5; perturb their tokens to garbage in (b).
    shaft_mask = torch.zeros(1, C_MAX, dtype=torch.bool)
    shaft_mask[0, [0, 2, 5]] = True
    electrodes_b[0, [0, 2, 5]] = 999.0

    with torch.no_grad():
        out_a = model(electrodes_a, support, shaft_mask=shaft_mask)
        out_b = model(electrodes_b, support, shaft_mask=shaft_mask)
    assert torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5), (
        "shaft-masked electrodes leaked into cross-attn; "
        "max |Δ| = "
        f"{(out_a - out_b).abs().max().item():.2e}"
    )


# --- (2) teacher path is uncontaminated --------------------------------


def test_teacher_forward_unaffected_by_student_shaft_mask() -> None:
    """B03d asymmetry: the teacher forward must NOT receive a shaft
    mask. Smoke test: same model instance called twice — once with
    ``shaft_mask=mask`` (student path), once with ``shaft_mask=None``
    (teacher path) — and the second call's output is identical to the
    no-mask baseline, proving no shared state contaminates the call."""
    torch.manual_seed(0)
    model = _build_model().eval()
    electrodes = torch.randn(1, C_MAX, T_BINS, F_BINS)
    support = torch.zeros(1, C_MAX, K_PARCELS)
    for c in range(C_MAX):
        support[0, c, c % K_PARCELS] = 1.0
    shaft_mask = torch.zeros(1, C_MAX, dtype=torch.bool)
    shaft_mask[0, [1, 4]] = True

    with torch.no_grad():
        out_baseline = model(electrodes, support, shaft_mask=None)
        _ = model(electrodes, support, shaft_mask=shaft_mask)
        out_teacher = model(electrodes, support, shaft_mask=None)
    torch.testing.assert_close(out_baseline, out_teacher, atol=1e-7, rtol=1e-7)


# --- (3) K cardinality across cohort ranges (5/27 PM final spec) ------


@pytest.mark.parametrize("n_shafts", [2, 3, 4, 5, 7, 8, 9, 10, 15, 20, 30])
def test_k_equals_1_for_every_n_shafts_ge_2(n_shafts: int) -> None:
    """5/27 PM final spec: ``K = 1`` for every ``N_shafts >= 2``.

    Covers the BT range (7–10), the D-cohort range (≫ 8), and small
    multi-shaft subjects (2–4). The AM ``min(2, ceil(0.25 * N))``
    fraction formula and the original ``K=3`` spec are both retired."""
    assert default_k(n_shafts) == 1
    anatomy = _make_anatomy_from_shafts([8] * n_shafts)
    ext = ShaftMaskExtractor(seed=0)
    assert ext._k_for_anatomy(anatomy) == 1


def test_k_equals_0_for_single_shaft_subject() -> None:
    """5/27 PM guard: ``N_shafts == 1`` → ``K = 0`` (no shaft drop).
    Closes the catastrophic-drop hole the AM fraction formula had
    (it would emit K=1 here, dropping the entire subject's electrodes)."""
    assert default_k(1) == 0
    anatomy = _make_anatomy_from_shafts([8])  # one shaft of 8 contacts
    ext = ShaftMaskExtractor(seed=0)
    assert ext._k_for_anatomy(anatomy) == 0
    # No mask emitted.
    assert not ext(anatomy).any()
