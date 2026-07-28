"""R2 ablation (#19) — build the towers with NO parcel identity embed.

The embed is purely ADDITIVE at every call site (towers.py:166/197/280: ``x = x + embed(pid)``),
never concatenated, so "zero it out" and "don't add it" are the SAME edit. We take the stronger
form: with ``parcel_embed=False`` the table is NOT CONSTRUCTED at all — no dead parameter for
weight decay to touch, and the checkpoint's param count is honest. Geometry then reaches the model
ONLY through L1 RoPE (contact index + time); there is no anatomy channel left.

Invariants asserted + printed: (1) the table is gone, and by exactly n_parcels·d_model params;
(2) the disabled tower is INVARIANT to parcel relabelling; (3) disabled == enabled-with-a-zeroed-
table, bit-for-bit (this is what proves "off == adding zero"); (4) the r6 objective drops exactly
the two tables (online encoder + predictor) and still produces a finite loss.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks_r6
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import PER_BAND_SPECS
from speech_decoding.models.v14_converged_v3.towers import (
    ENC_D_MODEL,
    PRED_D_MODEL,
    build_encoder,
    build_predictor,
)

T = 32
B = 2
N = 5
N_PARCELS = 8
BINS = tuple(nb for nb, _ in PER_BAND_SPECS)


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _bands(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, N, BINS[b], T, generator=g) for b in range(3)]


# --------------------------------------------------------------------------- #
def test_disabled_tower_has_no_parcel_table() -> None:
    # The table is absent, not zeroed: forward returns zeros from a None submodule table.
    for name, build, d in (
        ("encoder", build_encoder, ENC_D_MODEL),
        ("predictor", build_predictor, PRED_D_MODEL),
    ):
        on = build(n_parcels=N_PARCELS, parcel_embed=True)
        off = build(n_parcels=N_PARCELS, parcel_embed=False)
        n_tab = sum(p.numel() for p in off.parcel_embed.parameters())
        drop = sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters())
        ok = off.parcel_embed.embed is None and n_tab == 0 and drop == N_PARCELS * d
        print(f"[check] {name}: table=None ({off.parcel_embed.embed is None}), "
              f"embed params {n_tab}, total param drop {drop} == {N_PARCELS}·{d} "
              f"{'OK' if ok else 'VIOLATED'}")
        assert ok


def test_disabled_tower_is_invariant_to_parcel_relabelling() -> None:
    # The mirror of test_parcel_embed_added_once_at_tower_input: with the embed ON, relabelling a
    # contact's parcel CHANGES the output; with it OFF the output must be bit-identical, i.e. the
    # anatomy tag is fully disconnected from the forward pass.
    sc, geom = _session()
    x = torch.randn(1, N, T, ENC_D_MODEL)
    pid_b = sc.parcel_id.clone()
    pid_b[0] = 5

    off = build_encoder(n_parcels=N_PARCELS, parcel_embed=False).eval()
    with torch.no_grad():
        a, b = off(x, geom, sc.parcel_id), off(x, geom, pid_b)
    invariant = torch.equal(a, b)

    on = build_encoder(n_parcels=N_PARCELS, parcel_embed=True).eval()
    with torch.no_grad():
        sensitive = not torch.allclose(
            on(x, geom, sc.parcel_id), on(x, geom, pid_b), atol=0.0, rtol=0.0
        )

    print(f"[check] OFF invariant to relabel ({invariant}), ON sensitive ({sensitive}) "
          f"{'OK' if invariant and sensitive else 'VIOLATED'}")
    assert invariant and sensitive


def test_disabled_equals_enabled_with_zeroed_table() -> None:
    # "Don't add it" == "add zero". Give the two towers identical params everywhere else, then
    # zero the enabled tower's table: it must reproduce the disabled tower BIT-FOR-BIT.
    sc, geom = _session()
    x = torch.randn(1, N, T, ENC_D_MODEL)
    on = build_encoder(n_parcels=N_PARCELS, parcel_embed=True).eval()
    off = build_encoder(n_parcels=N_PARCELS, parcel_embed=False).eval()
    with torch.no_grad():
        assert on.parcel_embed.embed is not None
        on.parcel_embed.embed.weight.zero_()
        # Copy every OTHER param across (a shared seed would NOT suffice — building the table
        # consumes RNG draws, shifting every downstream init). strict=True is the check that the
        # two towers differ by exactly the one missing key.
        off.load_state_dict({k: v for k, v in on.state_dict().items()
                             if "parcel_embed.embed" not in k})
        ok = torch.equal(on(x, geom, sc.parcel_id), off(x, geom, sc.parcel_id))
    print(f"[check] zeroed-table ON == OFF bit-for-bit {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_r6_objective_drops_both_tables_and_still_trains() -> None:
    # Objective level: the flag reaches the online encoder AND the predictor (MAE ⇒ no EMA
    # teacher; on the JEPA arm the teacher is a deep copy of `online`, so it follows for free).
    sc, geom = _session()
    on = V3JepaObjective(n_parcels=N_PARCELS, r6=True, mae=True)
    off = V3JepaObjective(n_parcels=N_PARCELS, r6=True, mae=True, parcel_embed=False)
    tables_on = [k for k in on.state_dict() if "parcel_embed.embed" in k]
    tables_off = [k for k in off.state_dict() if "parcel_embed.embed" in k]
    ok_tables = len(tables_on) == 2 and tables_off == []

    masks = sample_masks_r6(geom, N, n_time=T, n_rows=B, generator=torch.Generator().manual_seed(1))
    out = off(_bands(), geom, sc.parcel_id, masks)
    loss = out.loss
    loss.backward()
    grads = [p.grad for p in off.parameters() if p.requires_grad and p.grad is not None]
    ok_step = bool(torch.isfinite(loss)) and len(grads) > 0 and all(
        bool(torch.isfinite(g).all()) for g in grads
    )
    print(f"[check] tables ON {tables_on} → OFF {tables_off} ({ok_tables}); "
          f"loss {loss.item():.4f} finite + {len(grads)} finite grads ({ok_step}) "
          f"{'OK' if ok_tables and ok_step else 'VIOLATED'}")
    assert ok_tables and ok_step
