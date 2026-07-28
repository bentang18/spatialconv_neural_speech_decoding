"""v3r6 model wiring — end-to-end + session_plan.

r6 is r4's arm0 frontend VERBATIM (same 32 Hz caches, same ``PerBandStem``) with exactly four
deltas; three of them live in the model/objective and are asserted here:

  * NO ``norm_pix`` on the MAE target;
  * NO M14 margin gate — ``in_loss == masked``, every masked token is scored;
  * PER-SENSOR band time masks + a predictor BAND embed (without it, the three co-located tokens
    of one contact at a lattice position divisible by 8 are byte-identical predictor inputs asked
    to reconstruct three different bands).

The fourth (shaft-level batching) is a datamodule concern, tested in ``test_dispatch_v3``.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.model import V3ConvergedModel
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import PER_BAND_SPECS, PerBandStem

N_PARCELS = 8
T32 = 32  # 32 Hz clock; native bands SLOW T/8=4, MID T/2=16, HGA T=32.
BINS = tuple(nb for nb, _ in PER_BAND_SPECS)  # (7, 6, 7)
STRIDES = tuple(st for _, st in PER_BAND_SPECS)  # (8, 2, 1)


def _session(shaft_sizes=(5, 4, 4)):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s % N_PARCELS)
    sc = build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))
    return sc, build_l1_geometry(sc)


def _bands(n, B=1):
    """All 3 bands on the SHARED 32 Hz clock ``(B,n,F_b,T32)`` — r4's cache convention exactly.
    PerBandStem decimates internally (stride 8/2/1); r6 reads the same cache with no rescale."""
    return [torch.randn(B, n, BINS[b], T32) for b in range(3)]


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_r6_end_to_end_mae() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS, r6=True, mae=True)
    assert model.r6 is True and model.objective.r6 is True
    # r6 uses r4's stem EXACTLY — no native-per-band variant (that bake never existed).
    assert type(model.objective.online.stem) is PerBandStem
    out = model(_bands(n, B=2), geom, sc.parcel_id, generator=_gen())
    assert out.loss.ndim == 0 and torch.isfinite(out.loss) and out.loss.requires_grad
    out.loss.backward()
    stem = model.objective.online.stem
    # grads flow through all three per-band projections + the band-type embedding.
    for proj in stem.projs:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in proj.parameters())
    assert stem.band_type_emb.grad is not None
    print("[check] OK r6 end-to-end MAE: PerBandStem, finite loss, grads in all 3 band projs")


def test_r6_predictor_band_embed_exists_and_trains() -> None:
    # THE degeneracy fix. SLOW token j sits at time_pos 8j, MID at 2j, HGA at j — at every lattice
    # position divisible by 8 the three co-located tokens of one contact share depth, time_pos AND
    # parcel. Without a band embed they are byte-identical predictor queries for three different
    # reconstruction targets. Assert the parameter exists, is r6-only, and receives gradient.
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS, r6=True, mae=True)
    emb = model.objective.pred_band_emb
    assert emb is not None and emb.shape[0] == len(PER_BAND_SPECS)
    out = model(_bands(n, B=2), geom, sc.parcel_id, generator=_gen())
    out.loss.backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0
    arm0 = V3ConvergedModel(n_parcels=N_PARCELS, mae=True)
    assert arm0.objective.pred_band_emb is None  # r6-only; r4 is untouched
    print(f"[check] OK pred_band_emb ({tuple(emb.shape)}) is r6-only and trains")


def test_r6_scores_every_masked_token() -> None:
    # No margin gate: the loss token count equals the masked token count exactly. Read the flags
    # the forward would build via session_plan's own path, then compare against n_masked.
    from speech_decoding.models.v14_converged_v3.masking import sample_masks_r6
    from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid, token_flags_r6

    sc, geom = _session()
    n = len(sc.labels)
    grid = build_r4_grid(geom, n_time=T32)
    masks = sample_masks_r6(geom, n, n_time=T32, n_rows=2, generator=_gen())
    masked, in_loss = token_flags_r6(grid, masks)
    assert torch.equal(masked, in_loss)
    model = V3ConvergedModel(n_parcels=N_PARCELS, r6=True, mae=True)
    out = model(_bands(n, B=2), geom, sc.parcel_id, generator=_gen())
    assert int(out.n_masked.item()) == int(masked.sum())
    print(f"[check] OK n_masked {int(out.n_masked.item())} == |masked| (no margin gate)")


def test_r6_target_is_not_norm_pix() -> None:
    # norm_pix is a per-token whitening of the RECONSTRUCTION TARGET. Deterministic test: poison
    # V3JepaObjective._norm_pix so any call raises. r6 must run clean (target = raw |STFT| bins);
    # arm0 must trip it (same code path, gate flipped) — proving the flag reaches the target and
    # is the ONLY difference, not that some other branch happens to skip normalisation.
    from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective

    sc, geom = _session()
    n = len(sc.labels)
    bands = _bands(n, B=2)
    # Capture the raw DESCRIPTOR, not getattr's unwrapped function: _norm_pix is a staticmethod,
    # so restoring the plain function would rebind it as an instance method (self lands in x) and
    # silently break every later test in the session.
    orig = V3JepaObjective.__dict__["_norm_pix"]

    def poisoned(*a, **kw):
        raise AssertionError("_norm_pix called")

    V3JepaObjective._norm_pix = staticmethod(poisoned)
    try:
        r6 = V3ConvergedModel(n_parcels=N_PARCELS, r6=True, mae=True)
        out = r6(bands, geom, sc.parcel_id, generator=_gen())
        assert torch.isfinite(out.loss)  # r6 never touches norm_pix
        arm0 = V3ConvergedModel(n_parcels=N_PARCELS, mae=True)
        tripped = False
        try:
            arm0(bands, geom, sc.parcel_id, generator=_gen())
        except AssertionError as e:
            tripped = "_norm_pix called" in str(e)
        assert tripped, "arm0 did NOT call _norm_pix — the poison test proves nothing"
    finally:
        V3JepaObjective._norm_pix = orig
    print("[check] OK norm_pix OFF for r6 (never called) and ON for arm0 (called)")


def test_r6_force_norm_pix_override_turns_it_on() -> None:
    # Isolation OFAT: --mae-norm-pix (mae_force_norm_pix=True) forces norm_pix ON under r6,
    # decoupled from `norm_pix=not r6`, so the delta can be measured single-handed. Same poison
    # pattern: default r6 must NOT call _norm_pix; r6 + override MUST. Proves the override is the
    # ONLY thing that flips, and that it reaches the same target path arm0 uses.
    from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective

    sc, geom = _session()
    n = len(sc.labels)
    bands = _bands(n, B=2)
    orig = V3JepaObjective.__dict__["_norm_pix"]

    def poisoned(*a, **kw):
        raise AssertionError("_norm_pix called")

    V3JepaObjective._norm_pix = staticmethod(poisoned)
    try:
        # default r6: override False ⇒ still OFF (byte-identical to contract)
        r6_off = V3ConvergedModel(n_parcels=N_PARCELS, r6=True, mae=True)
        assert r6_off.objective.force_norm_pix is False
        out = r6_off(bands, geom, sc.parcel_id, generator=_gen())
        assert torch.isfinite(out.loss)  # untouched
        # r6 + override: norm_pix ON ⇒ MUST trip the poison
        r6_on = V3ConvergedModel(
            n_parcels=N_PARCELS, r6=True, mae=True, mae_force_norm_pix=True
        )
        assert r6_on.objective.force_norm_pix is True
        tripped = False
        try:
            r6_on(bands, geom, sc.parcel_id, generator=_gen())
        except AssertionError as e:
            tripped = "_norm_pix called" in str(e)
        assert tripped, "r6 + mae_force_norm_pix did NOT call _norm_pix — override is dead"
    finally:
        V3JepaObjective._norm_pix = orig
    print("[check] OK force_norm_pix: OFF by default under r6, ON with the override")


def test_r6_session_plan_matches_forward_shapes() -> None:
    sc, geom = _session()
    n = len(sc.labels)
    model = V3ConvergedModel(n_parcels=N_PARCELS, r6=True, mae=True)
    gms, m_vis, pms = model.session_plan(geom, sc.parcel_id, T32)
    out = model(
        _bands(n, B=2), geom, sc.parcel_id, generator=_gen(),
        grid_max_seqlen=gms, m_vis=m_vis, pack_max_seqlen=pms,
    )
    assert torch.isfinite(out.loss)
    assert gms > 0 and m_vis > 0 and pms > 0
    print(f"[check] OK session_plan ({gms}, {m_vis}, {pms}) matches the forward under per-sensor "
          "masks (count invariance holds)")
