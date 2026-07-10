"""v14_converged_v3 — varlen pack plan (#24) TDD.

``build_pack_plan`` lays a clip's SELECTED contacts (visible for the online
encoder, all-valid for teacher/predictor) end to end in shaft-grouped order and
emits the ``cu_seqlens`` the ragged L1 attention consumes. The invariants that
keep it correct AND compile-safe:

  * shaft-grouped ``order`` (contiguous by shaft) so ``cu_seqlens`` marks real blocks;
  * ``depth`` carries the RAW clinical index (gaps preserved, NEVER condensed) —
    the RoPE coordinate, Ben's hard catch;
  * per-clip ``cu_seqlens`` of constant length B·S+1 (zero-length whole-masked
    shafts KEPT, not filtered) — the per-session-constant that makes the plan
    recompile-free;
  * fixed-shape selection (``n_selected`` supplied) — no data-dependent nonzero.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.packing import (
    build_pack_plan,
    gather_tokens,
    scatter_tokens,
)
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar


def _sidecar(labels, parcels):
    return build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))


def _geom(labels, parcels):
    return build_l1_geometry(_sidecar(labels, parcels))


# ── unmasked (teacher / predictor): all valid contacts ───────────────────────


def test_unmasked_selects_every_contact_shaft_grouped() -> None:
    # LA(3) LB(2), N=5. visible=None ⇒ P=N, order = the canonical shaft order.
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    plan = build_pack_plan(geom, n_time=4, batch=1, n_selected=5, visible=None)
    assert plan.n_selected == 5
    assert plan.order.shape == (1, 5)
    assert plan.order[0].tolist() == [0, 1, 2, 3, 4]


def test_unmasked_depth_carries_raw_clinical_index_with_gaps() -> None:
    # LA has a drop-gap: LA1, LA2, LA5 (clinical 1,2,5 — LA3/LA4 dropped upstream).
    # depth MUST stay [1,2,5], never re-densified to [1,2,3].
    geom = _geom(["LA1", "LA2", "LA5", "LB2", "LB3"], [0, 0, 0, 1, 1])
    plan = build_pack_plan(geom, n_time=4, batch=1, n_selected=5, visible=None)
    assert plan.depth[0].tolist() == [1, 2, 5, 2, 3]


def test_unmasked_cu_seqlens_marks_each_shaft_block_in_tokens() -> None:
    # LA(3) LB(2), T=4 ⇒ blocks of 3*4=12 and 2*4=8 tokens.
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    plan = build_pack_plan(geom, n_time=4, batch=1, n_selected=5, visible=None)
    assert plan.cu_seqlens.dtype == torch.int32
    assert plan.cu_seqlens.tolist() == [0, 12, 20]
    assert plan.max_seqlen == geom.max_c * 4  # 3 * 4
    assert plan.n_tokens == 5 * 4


def test_unmasked_batch_replicates_the_plan_per_clip() -> None:
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    plan = build_pack_plan(geom, n_time=4, batch=3, n_selected=5, visible=None)
    assert plan.order.shape == (3, 5)
    # B*S+1 = 3*2+1 = 7 bounds; each clip contributes its two blocks 12,8.
    assert plan.cu_seqlens.tolist() == [0, 12, 20, 32, 40, 52, 60]


# ── masked (online encoder): per-clip visible set ────────────────────────────


def test_masked_selects_only_visible_shaft_grouped() -> None:
    # LA(3) LB(2). Mask contact 1 (LA2) and 3 (LB1). visible = [T,F,T,F,T].
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    visible = torch.tensor([[True, False, True, False, True]])
    plan = build_pack_plan(geom, n_time=4, batch=1, n_selected=3, visible=visible)
    # visible contacts in canonical order: 0 (LA1), 2 (LA3), 4 (LB2).
    assert plan.order[0].tolist() == [0, 2, 4]
    assert plan.depth[0].tolist() == [1, 3, 2]  # LA1=1, LA3=3, LB2=2
    # LA keeps 2 visible (0,2) ⇒ 2*4=8 tok; LB keeps 1 (4) ⇒ 4 tok.
    assert plan.cu_seqlens.tolist() == [0, 8, 12]


def test_masked_whole_shaft_gives_zero_length_block_not_filtered() -> None:
    # Mask ALL of LB (contacts 3,4). LB block must survive as a 0-length segment
    # so cu_seqlens length stays B*S+1 (compile-safe), not shrink to the live shafts.
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    visible = torch.tensor([[True, True, True, False, False]])
    plan = build_pack_plan(geom, n_time=4, batch=1, n_selected=3, visible=visible)
    assert plan.order[0].tolist() == [0, 1, 2]
    # LA=3*4=12, LB=0 ⇒ bounds [0,12,12]; length still S+1=3.
    assert plan.cu_seqlens.tolist() == [0, 12, 12]
    assert plan.cu_seqlens.shape == (geom.n_shafts + 1,)


def test_masked_cu_seqlens_length_is_constant_across_masks() -> None:
    # Two DIFFERENT masks with the SAME visible count P must yield the SAME
    # cu_seqlens length (B*S+1) — the invariant flash/compile rely on.
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    m1 = torch.tensor([[True, False, True, True, False]])  # LA:2 LB:1
    m2 = torch.tensor([[False, True, True, False, True]])  # LA:2 LB:1
    p1 = build_pack_plan(geom, n_time=4, batch=1, n_selected=3, visible=m1)
    p2 = build_pack_plan(geom, n_time=4, batch=1, n_selected=3, visible=m2)
    assert p1.cu_seqlens.shape == p2.cu_seqlens.shape == (3,)
    assert p1.n_selected == p2.n_selected == 3
    assert p1.n_tokens == p2.n_tokens


def test_masked_per_clip_independent_masks_in_one_batch() -> None:
    # Two clips, different masks, same P=3 (each masks 2 of 5). cu_seqlens spans
    # both clips: B*S+1 = 2*2+1 = 5 bounds.
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    visible = torch.tensor([
        [True, True, False, True, False],   # clip0: LA:2 (0,1) LB:1 (3)
        [False, True, True, False, True],   # clip1: LA:2 (1,2) LB:1 (4)
    ])
    plan = build_pack_plan(geom, n_time=4, batch=2, n_selected=3, visible=visible)
    assert plan.order[0].tolist() == [0, 1, 3]
    assert plan.order[1].tolist() == [1, 2, 4]
    # clip0: LA 8, LB 4 → 8,12; clip1: LA 8, LB 4 → +8,+4.
    assert plan.cu_seqlens.tolist() == [0, 8, 12, 20, 24]


def test_masked_depth_still_raw_after_selection() -> None:
    # Gapped LA (1,2,5); mask LA2 (contact 1). Surviving depths keep the gap.
    geom = _geom(["LA1", "LA2", "LA5", "LB2", "LB3"], [0, 0, 0, 1, 1])
    visible = torch.tensor([[True, False, True, True, True]])
    plan = build_pack_plan(geom, n_time=4, batch=1, n_selected=4, visible=visible)
    assert plan.order[0].tolist() == [0, 2, 3, 4]
    assert plan.depth[0].tolist() == [1, 5, 2, 3]  # LA1=1, LA5=5, LB2=2, LB3=3


def test_gather_scatter_round_trip_unmasked() -> None:
    # gather_tokens ∘ scatter_tokens = identity when all contacts are selected.
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    plan = build_pack_plan(geom, n_time=4, batch=2, n_selected=5, visible=None)
    x = torch.randn(2, 5, 4, 8)
    packed = gather_tokens(x, plan.order)
    full = scatter_tokens(packed, plan.order, n_full=5)
    assert torch.equal(full, x)


def test_gather_scatter_masked_leaves_masked_rows_zero() -> None:
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    visible = torch.tensor([[True, False, True, False, True]])
    plan = build_pack_plan(geom, n_time=4, batch=1, n_selected=3, visible=visible)
    x = torch.randn(1, 5, 4, 8)
    full = scatter_tokens(gather_tokens(x, plan.order), plan.order, n_full=5)
    assert torch.equal(full[0, [0, 2, 4]], x[0, [0, 2, 4]])  # visible preserved
    assert torch.count_nonzero(full[0, 1]) == 0 and torch.count_nonzero(full[0, 3]) == 0


def test_order_scatters_back_to_full_n_buffer() -> None:
    # The predictor re-inserts encoder-visible tokens into an all-N buffer via
    # ``order``. Verify a scatter round-trips: buffer[order] = payload lands the
    # payload at the true contact rows.
    geom = _geom(["LA1", "LA2", "LA3", "LB1", "LB2"], [0, 0, 0, 1, 1])
    visible = torch.tensor([[True, False, True, False, True]])
    plan = build_pack_plan(geom, n_time=4, batch=1, n_selected=3, visible=visible)
    T, d = 4, 2
    full = torch.zeros(1, 5, T, d)
    payload = torch.arange(1, 3 * T * d + 1, dtype=torch.float32).reshape(1, 3, T, d)
    idx = plan.order[:, :, None, None].expand(-1, -1, T, d)
    full.scatter_(1, idx, payload)
    # Visible rows 0,2,4 got payload rows 0,1,2; masked rows 1,3 stay zero.
    assert torch.equal(full[0, 0], payload[0, 0])
    assert torch.equal(full[0, 2], payload[0, 1])
    assert torch.equal(full[0, 4], payload[0, 2])
    assert torch.count_nonzero(full[0, 1]) == 0
    assert torch.count_nonzero(full[0, 3]) == 0
