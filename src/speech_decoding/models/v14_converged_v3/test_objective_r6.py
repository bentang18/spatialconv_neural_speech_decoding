"""r6 objective arm — MAE-only, r4's 3-band |STFT|-bin reconstruction with a RAW target.

r6 is r4-MAE's objective VERBATIM with exactly two loss-side deltas: ``norm_pix`` off, and no
M14 margin gate (``in_loss == masked``). It reads the SAME uniform 32 Hz caches and uses the SAME
``PerBandStem`` and the SAME ``_mae_gather_target`` — there is no r6 gather, no r6 output path and
no native-rate stem (the 4/16/32 Hz bake those assumed was never made; see
project-r6-band-rates-cache-rate-bug-2026-07-23). The one ADDED parameter is ``pred_band_emb``.
Invariants named + asserted + printed.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import sample_masks_r6
from speech_decoding.models.v14_converged_v3.objective import V3JepaObjective
from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid, token_flags_r6
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar
from speech_decoding.models.v14_converged_v3.stem import PER_BAND_SPECS, PerBandStem
from speech_decoding.models.v14_converged_v3.towers import PRED_D_MODEL

T = 32  # 32 Hz clock; the stem decimates to SLOW T/8=4, MID T/2=16, HGA T=32.
B = 2
N = 5
F_MAX = max(nb for nb, _ in PER_BAND_SPECS)  # 7
BINS = tuple(nb for nb, _ in PER_BAND_SPECS)  # (7, 6, 7)
STRIDES = tuple(st for _, st in PER_BAND_SPECS)  # (8, 2, 1)


def _session():
    sc = build_sidecar(
        ["LA1", "LA2", "LA3", "LB1", "LB2"],
        parcel_id=torch.tensor([0, 0, 0, 1, 1]),
    )
    return sc, build_l1_geometry(sc)


def _bands(seed: int = 0):
    """All 3 bands on the SHARED 32 Hz clock (B,N,F_b,T) — r4's cache convention exactly."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, N, BINS[b], T, generator=g) for b in range(3)]


def _masks(geom, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return sample_masks_r6(geom, N, n_time=T, n_rows=B, generator=g)


def _r6(**kw) -> V3JepaObjective:
    return V3JepaObjective(n_parcels=8, r6=True, mae=True, **kw)


# --------------------------------------------------------------------------- #
def test_r6_is_mae_only() -> None:
    with pytest.raises(ValueError, match="MAE-only"):
        V3JepaObjective(n_parcels=8, r6=True, mae=False)
    print("[check] OK r6 + not mae raises 'r6 is MAE-only'")


def test_r6_mutually_exclusive_with_other_frontends() -> None:
    for other in ("early_fusion", "no_fusion", "native_fine_hga"):
        with pytest.raises(ValueError, match="mutually exclusive"):
            V3JepaObjective(n_parcels=8, r6=True, mae=True, **{other: True})
    print("[check] OK r6 ⊥ early_fusion / no_fusion / native_fine_hga")


def test_r6_uses_arm0_stem_and_heads_no_teacher() -> None:
    obj = _r6()
    # EXACT type, not isinstance: a subclass would mean a second frontend crept back in.
    is_arm0_stem = type(obj.online.stem) is PerBandStem
    no_teacher = obj.teacher is None
    heads_ok = (
        obj.mae_heads is not None
        and [h.out_features for h in obj.mae_heads] == list(BINS)  # 7/6/7 per-band
        and obj.mae_head_r5 is None
    )
    ok = is_arm0_stem and no_teacher and heads_ok
    print(f"[check] PerBandStem exactly ({is_arm0_stem}), teacher=None ({no_teacher}), "
          f"heads={[h.out_features for h in obj.mae_heads]} ({heads_ok}) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok
    assert obj.update_teacher() == 0.0  # no-op every opt-step (no EMA teacher on MAE)


def test_r6_gather_is_r4_gather_bit_for_bit() -> None:
    # r6 has NO gather of its own. Same bands, same grid ⇒ the r4 and r6 objectives must produce
    # a bit-identical target/valid/count triple. This is the assertion the deleted
    # _mae_gather_target_r6 used to make approximately (and on a bake that did not exist).
    sc, geom = _session()
    r4 = V3JepaObjective(n_parcels=8, mae=True)
    r6 = _r6()
    grid = build_r4_grid(geom, n_time=T)
    bands = _bands(seed=9)
    t4, v4, c4 = r4._mae_gather_target(bands, grid)
    t6, v6, c6 = r6._mae_gather_target(bands, grid)
    ok = torch.equal(t4, t6) and torch.equal(v4, v6) and torch.equal(c4, c6)
    assert t4.shape == (B, grid.total, F_MAX)
    print(f"[check] r6 target == r4 target BIT-IDENTICAL {tuple(t4.shape)} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_target_is_the_tokens_own_decimated_frame() -> None:
    # THE alignment invariant, restated on the real cache: token (band b, contact c, bandpos p)
    # targets 32 Hz frame p·stride_b — the frame its own stem token was built from. Not p, and not
    # a rescaled index (that rescale, on a 32 Hz cache, was the 07-23 misalignment bug).
    sc, geom = _session()
    obj = _r6()
    grid = build_r4_grid(geom, n_time=T)
    bands = _bands(seed=3)
    target, feat_valid, feat_count = obj._mae_gather_target(bands, grid)
    max_dev = 0.0
    for t in range(grid.total):
        b = int(grid.band[t]); c = int(grid.contact[t]); p = int(grid.bandpos[t])
        assert int(grid.time_pos[t]) == p * STRIDES[b]  # the lattice identity itself
        ref = bands[b][:, c, :, p * STRIDES[b]]  # (B, F_b) raw 32 Hz frame at the lattice slot
        nb = BINS[b]
        assert int(feat_count[t]) == nb
        assert feat_valid[t][:nb].all() and not feat_valid[t][nb:].any()
        max_dev = max(max_dev, (target[:, t, :nb] - ref).abs().max().item())
        if nb < F_MAX:  # pad slots zeroed
            max_dev = max(max_dev, target[:, t, nb:].abs().max().item())
    exact = max_dev < 1e-6
    print(f"[check] target == own frame p·stride for all {grid.total} tokens: "
          f"max|Δ|={max_dev:.2e}, feat_count 7/6/7, pad zeroed {'OK' if exact else 'VIOLATED'}")
    assert exact


def test_r6_skips_norm_pix_raw_target() -> None:
    # NO norm_pix: fill every bin with a CONSTANT c ⇒ per-token var=0 ⇒ norm_pix would map the
    # target to 0 (loss 0); the RAW target keeps c (loss c² when pred=0). Distinguishes the two.
    sc, geom = _session()
    obj = _r6()
    c = 2.0
    with torch.no_grad():
        for head in obj.mae_heads:  # zero heads ⇒ pred == 0 everywhere
            head.weight.zero_(); head.bias.zero_()
    grid = build_r4_grid(geom, n_time=T)
    bands = [torch.full((B, N, BINS[b], T), c) for b in range(3)]
    h = torch.zeros(B, grid.total, PRED_D_MODEL)  # irrelevant (heads zeroed)
    in_loss = torch.ones(B, grid.total, dtype=torch.bool)  # score all tokens
    out = obj._mae_output(bands, grid, h, in_loss, enc_taps=None, collect_taps=False,
                          norm_pix=False)
    loss = float(out.loss.detach())
    ok = abs(loss - c * c) < 1e-6  # raw ⇒ c²=4 ; norm_pix would give 0
    print(f"[check] no norm_pix: constant bins c={c} ⇒ loss={loss:.6f} (raw c²={c * c}, "
          f"norm_pix would be 0) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_r6_has_no_margin_gate() -> None:
    # r6 DROPS the M14 margin gate (Ben 2026-07-23: "No ML SSL does a margin gate for masked
    # tokens — score ALL masked tokens"). in_loss == masked exactly, not a strict subset.
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    masked, in_loss = token_flags_r6(grid, _masks(geom, seed=5))
    ok = torch.equal(in_loss, masked)
    print(f"[check] no margin gate: in_loss {int(in_loss.sum())} == masked {int(masked.sum())} "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_count_proportional_band_weighting_hga_dominates() -> None:
    # count-proportional (verbatim r4): the flat masked-mean gives HGA (T tokens) ~8:1 dominance
    # over SLOW (T/8) by SCORED-TOKEN COUNT — no per-band reweight. Verify HGA scored ≫ SLOW scored.
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    _, in_loss = token_flags_r6(grid, _masks(geom, seed=2))
    per_band = {b: int((in_loss[0] & (grid.band == b)).sum()) for b in range(3)}
    ratio = per_band[2] / max(per_band[0], 1)  # HGA / SLOW scored count
    ok = per_band[2] > per_band[1] > per_band[0] and ratio > 3.0
    print(f"[check] count-proportional: scored SLOW={per_band[0]} MID={per_band[1]} "
          f"HGA={per_band[2]} (HGA/SLOW={ratio:.1f}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_pred_band_emb_breaks_the_colocated_query_degeneracy() -> None:
    # Ben 2026-07-23: "ALL tokens in predictor get +parcel embed and +freq embed." The parcel half
    # was already there (towers._run_flat). WITHOUT the band half, the SLOW/MID/HGA tokens of one
    # contact at a lattice position divisible by 8 share depth, time_pos AND parcel — three
    # byte-identical masked queries for three different reconstruction targets. Assert (a) the
    # degenerate triples exist in the grid, (b) the embed is r6-only, (c) it separates them.
    sc, geom = _session()
    obj = _r6()
    grid = build_r4_grid(geom, n_time=T)
    # (a) co-located triples really occur: same contact, same 32 Hz lattice slot, 3 distinct bands.
    key = {}
    for t in range(grid.total):
        key.setdefault((int(grid.contact[t]), int(grid.time_pos[t])), set()).add(int(grid.band[t]))
    triples = [k for k, v in key.items() if len(v) == 3]
    assert triples, "no co-located 3-band slot in this grid — the test proves nothing"
    # (b) r6-only.
    emb = obj.pred_band_emb
    assert emb is not None and emb.shape == (len(PER_BAND_SPECS), PRED_D_MODEL)
    assert V3JepaObjective(n_parcels=8, mae=True).pred_band_emb is None
    # (c) the three additive vectors are distinct ⇒ the queries separate.
    pairs = [(0, 1), (0, 2), (1, 2)]
    seps = [float((emb[i] - emb[j]).abs().max()) for i, j in pairs]
    assert all(s > 0 for s in seps)
    print(f"[check] OK {len(triples)} co-located 3-band slots; pred_band_emb "
          f"{tuple(emb.shape)} r6-only, pairwise max|Δ| {['%.3f' % s for s in seps]}")


def test_forward_finite_and_grads_reach_stem_heads_predictor_and_band_emb() -> None:
    sc, geom = _session()
    obj = _r6()
    obj.train()
    out = obj(_bands(), geom, sc.parcel_id, _masks(geom))
    loss_val = out.loss.detach()
    out.loss.backward()
    finite = bool(torch.isfinite(loss_val)) and loss_val.ndim == 0 and float(loss_val) >= 0.0
    checks = {
        "stem": obj.online.stem,
        "encoder": obj.online.encoder,
        "predictor": obj.predictor,
        "enc_to_pred": obj.enc_to_pred,
        "mae_heads": obj.mae_heads,
    }
    grad_ok = {
        name: all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        for name, m in checks.items()
    }
    bt_grad = obj.online.stem.band_type_emb.grad is not None
    mask_grad = obj.mask_token.grad is not None and torch.isfinite(obj.mask_token.grad).all()
    pb = obj.pred_band_emb
    pb_grad = pb.grad is not None and torch.isfinite(pb.grad).all() and bool(pb.grad.abs().sum() > 0)
    ok = finite and all(grad_ok.values()) and bt_grad and mask_grad and pb_grad
    print(f"[check] r6 loss={float(loss_val):.4f} finite={finite}; grads {grad_ok}; "
          f"band_type_emb grad={bt_grad}; mask_token grad={mask_grad}; "
          f"pred_band_emb grad={pb_grad} {'OK' if ok else 'VIOLATED'}")
    assert ok


# --------------------------------------------------------------------------- #
# HGA-envelope OFAT (2026-07-28). Single swap: HGA's target becomes the MEAN of its 7 |STFT|
# bins; SLOW and MID keep per-bin targets. Rationale is measured, not assumed — the orthogonal
# y = y_env + y_res decomposition on the real recon dump gives HGA env .154 EV vs shape .043
# (average it), MID .119 vs .113 (a wash), SLOW .109 vs .181 (per-bin EARNS its place). Input is
# UNCHANGED and no head is resized: HGA reuses the MID pad machinery — feat_count 1,
# feat_valid[:, 0] only, target[:, 0] = the bin mean. The HGA head's other 6 columns then receive
# exactly zero gradient, which for training dynamics IS a 1-wide head while keeping ckpt shape.
HGA = 2  # PER_BAND_SPECS band-axis order is (SLOW, MID, HGA); see pack_r4.py:42


def test_hga_envelope_target_is_the_bin_mean_and_only_column_zero_is_valid() -> None:
    sc, geom = _session()
    obj = _r6(mae_hga_envelope=True)
    grid = build_r4_grid(geom, n_time=T)
    bands = _bands(seed=7)
    target, feat_valid, feat_count = obj._mae_gather_target(bands, grid)
    target, feat_valid, feat_count = obj._collapse_hga_to_envelope(
        target, feat_valid, feat_count, grid)
    max_dev = 0.0
    n_hga = 0
    for t in range(grid.total):
        if int(grid.band[t]) != HGA:
            continue
        n_hga += 1
        c, p = int(grid.contact[t]), int(grid.bandpos[t])
        ref = bands[HGA][:, c, :, p * STRIDES[HGA]].mean(-1)  # (B,) mean of the token's 7 bins
        max_dev = max(max_dev, (target[:, t, 0] - ref).abs().max().item())
        max_dev = max(max_dev, target[:, t, 1:].abs().max().item())  # columns 1..6 zeroed
        assert int(feat_count[t]) == 1
        assert bool(feat_valid[t][0]) and not feat_valid[t][1:].any()
    exact = max_dev < 1e-6 and n_hga > 0
    print(f"[check] HGA envelope: {n_hga} HGA tokens, target[:,0]==mean(7 bins) "
          f"max|Δ|={max_dev:.2e}, cols 1-6 zeroed, feat_count=1 {'OK' if exact else 'VIOLATED'}")
    assert exact


def test_hga_envelope_leaves_slow_and_mid_bit_identical() -> None:
    # SINGLE swap. SLOW's bin shape earns MORE EV than its envelope (.181 vs .109) and MID is a
    # coin flip, so both MUST come through untouched — bit-for-bit, not approximately.
    sc, geom = _session()
    obj = _r6(mae_hga_envelope=True)
    grid = build_r4_grid(geom, n_time=T)
    bands = _bands(seed=11)
    t0, v0, c0 = obj._mae_gather_target(bands, grid)
    t1, v1, c1 = obj._collapse_hga_to_envelope(t0.clone(), v0.clone(), c0.clone(), grid)
    keep = (grid.band != HGA)
    ok = (
        torch.equal(t1[:, keep], t0[:, keep])
        and torch.equal(v1[keep], v0[keep])
        and torch.equal(c1[keep], c0[keep])
    )
    print(f"[check] SLOW+MID untouched by the HGA swap: {int(keep.sum())} tokens bit-identical "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_hga_envelope_off_by_default_is_the_r6_contract() -> None:
    # The flag OFF must be byte-identical to the shipped r6 arm — the isolation-override
    # discipline (cf. mae_force_norm_pix). A default that shifts the contract voids the keeper.
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    bands = _bands(seed=13)
    base, flag = _r6(), _r6(mae_hga_envelope=True)
    assert base.mae_hga_envelope is False and flag.mae_hga_envelope is True
    torch.manual_seed(0)
    l_base = float(base(_bands(seed=2), geom, sc.parcel_id, _masks(geom)).loss.detach())
    t0, v0, c0 = base._mae_gather_target(bands, grid)
    t1, v1, c1 = flag._mae_gather_target(bands, grid)  # gather itself is band-agnostic, always r4
    ok = torch.equal(t0, t1) and torch.equal(v0, v1) and torch.equal(c0, c1) and l_base > 0
    print(f"[check] flag OFF == r6 contract (gather bit-identical, loss={l_base:.4f} finite) "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok


def test_hga_envelope_is_incompatible_with_norm_pix() -> None:
    # norm_pix divides by the UNBIASED variance over a token's valid bins. With one bin that
    # variance is 0/0 — degenerate. Refuse the combination at construction rather than emit NaN.
    with pytest.raises(ValueError, match="norm_pix"):
        _r6(mae_hga_envelope=True, mae_force_norm_pix=True)
    with pytest.raises(ValueError, match="r6"):  # r4-MAE norm_pix's ON by default ⇒ same trap
        V3JepaObjective(n_parcels=8, mae=True, mae_hga_envelope=True)
    print("[check] OK mae_hga_envelope ⊥ norm_pix (1-bin unbiased var is 0/0), r6-only")


def test_hga_envelope_loss_ignores_the_dead_head_columns() -> None:
    # The 6 unsupervised HGA columns must contribute NOTHING: perturbing them cannot move the
    # loss, and only column 0's weights may receive gradient. That is what makes keeping the
    # 7-wide head equivalent to a 1-wide one.
    sc, geom = _session()
    obj = _r6(mae_hga_envelope=True)
    grid = build_r4_grid(geom, n_time=T)
    bands = _bands(seed=17)
    h = torch.randn(B, grid.total, PRED_D_MODEL)
    in_loss = torch.ones(B, grid.total, dtype=torch.bool)
    def _loss(w):
        return obj._mae_output(bands, grid, h, w, enc_taps=None, collect_taps=False,
                               norm_pix=False).loss
    l0 = float(_loss(in_loss).detach())
    with torch.no_grad():  # blow up rows 1..6 of the HGA head
        obj.mae_heads[HGA].weight[1:].add_(50.0)
        obj.mae_heads[HGA].bias[1:].add_(50.0)
    l1 = float(_loss(in_loss).detach())
    obj.zero_grad()
    _loss(in_loss).backward()
    g = obj.mae_heads[HGA].weight.grad
    dead = float(g[1:].abs().max())
    live = float(g[0].abs().max())
    ok = abs(l1 - l0) < 1e-6 and dead == 0.0 and live > 0.0
    print(f"[check] dead HGA columns: loss {l0:.6f} -> {l1:.6f} under a +50 perturbation, "
          f"grad row0 max={live:.3e} rows1-6 max={dead:.1e} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_hga_envelope_shrinks_hga_error_without_touching_slow_mid_error() -> None:
    # WHY the OFAT exists, restated as an invariant on the loss itself: with a zeroed head
    # (pred == 0) the per-token SE is the target's own mean-square, so the HGA term falls to
    # mean(bins)² ≤ mean(bins²) — Jensen, exactly the unpredictable-residual removal — while
    # SLOW/MID terms are unchanged. Assert both directions on the real gather.
    sc, geom = _session()
    grid = build_r4_grid(geom, n_time=T)
    bands = _bands(seed=19)
    h = torch.zeros(B, grid.total, PRED_D_MODEL)
    per_band = {}
    for name, obj in (("perbin", _r6()), ("env", _r6(mae_hga_envelope=True))):
        with torch.no_grad():
            for head in obj.mae_heads:
                head.weight.zero_(); head.bias.zero_()
        for b in range(3):
            sel = (grid.band == b)
            in_loss = torch.zeros(B, grid.total, dtype=torch.bool)
            in_loss[:, sel] = True  # score ONE band at a time ⇒ per-band mean SE
            per_band[(name, b)] = float(obj._mae_output(
                bands, grid, h, in_loss, enc_taps=None, collect_taps=False,
                norm_pix=False).loss.detach())
    same_lo = all(abs(per_band[("env", b)] - per_band[("perbin", b)]) < 1e-6 for b in (0, 1))
    hga_drops = per_band[("env", HGA)] < per_band[("perbin", HGA)]
    ratio = per_band[("perbin", HGA)] / max(per_band[("env", HGA)], 1e-9)
    ok = same_lo and hga_drops
    print(f"[check] zero-pred SE per band: SLOW {per_band[('perbin', 0)]:.4f}=="
          f"{per_band[('env', 0)]:.4f}, MID {per_band[('perbin', 1)]:.4f}=="
          f"{per_band[('env', 1)]:.4f}, HGA {per_band[('perbin', HGA)]:.4f}->"
          f"{per_band[('env', HGA)]:.4f} ({ratio:.1f}x) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_r6_stem_has_no_conv() -> None:
    # r6 keeps arm0's LINEAR per-band projections — the conv stem is OFAT #2, not shipped in r6.
    obj = _r6()
    stem = obj.online.stem
    assert isinstance(stem, PerBandStem)
    has_conv = any(isinstance(m, torch.nn.Conv1d) for m in stem.modules())
    assert not has_conv
    print("[check] OK r6 stem is arm0's linear PerBandStem, no Conv1d (conv stem is OFAT #2)")
