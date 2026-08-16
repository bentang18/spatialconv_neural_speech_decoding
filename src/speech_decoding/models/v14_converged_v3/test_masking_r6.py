"""r6 masking — ``sample_masks_r6``.

r4's 3-band leak-safe STRUCTURE (one per-shaft-balanced SPACE mask shared across bands +
three width-``block_w_band`` band TIME masks on the SLOW/MID/HGA native grids) crossed with
PER-SENSOR INDEPENDENCE (each band's time mask is ``(R,N,T_b)``, drawn per contact, not the
r4 global ``(R,T_b)``) — Ben 2026-07-23: the encoder is L1-within-shaft only, so per-sensor
masking is free extra diversity. The M14 margin gate is GONE downstream (``in_loss == masked``);
this sampler only lays the width-4 blocks. Invariants named + asserted + printed.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.geometry import build_l1_geometry
from speech_decoding.models.v14_converged_v3.masking import (
    V3MaskConfig,
    V3MasksR6,
    assert_mask_feasible,
    sample_masks,
    sample_masks_r6,
)
from speech_decoding.models.v14_converged_v3.pack_r4 import build_r4_grid, token_flags_r6
from speech_decoding.models.v14_converged_v3.sidecar import build_sidecar


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _session(shaft_sizes: list[int]):
    labels, parcels = [], []
    for s, n in enumerate(shaft_sizes):
        for c in range(1, n + 1):
            labels.append(f"L{chr(65 + s)}{c}")
            parcels.append(s)
    sc = build_sidecar(labels, parcel_id=torch.tensor(parcels, dtype=torch.long))
    return sc, build_l1_geometry(sc)


def test_shape_and_four_fields() -> None:
    # T=32 ⇒ HGA 32 (32 Hz), MID 16 (16 Hz), SLOW 4 (4 Hz); SLOW grid 4 == block_w_band (leak-safe).
    sc, geom = _session([3, 3, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=5, generator=_gen())
    assert isinstance(m, V3MasksR6)
    assert m.contact_mask.shape == (5, n, 3)  # (R, N, band) — band axis (SLOW, MID, HGA)
    assert m.hga_mask.shape == (5, n, 32)
    assert m.mid_mask.shape == (5, n, 16)
    assert m.slow_mask.shape == (5, n, 4)
    print("[check] OK V3MasksR6: shared contact (R,N) + per-sensor SLOW/MID/HGA (R,N,T_b)")


def test_each_band_temporal_count_is_exact_per_sensor() -> None:
    # per-band PER-SENSOR count == round(frac · band-grid length) EXACTLY. This is what keeps the
    # visible-token count a per-session constant under per-sensor masking: every contact hides the
    # same number of frames ⇒ per-shaft total = n_contacts · cnt ⇒ m_vis/cu_seqlens unchanged.
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    for frac in (0.25, 0.5, 0.75):
        cfg = V3MaskConfig(
            hga_mask_frac=frac, mid_mask_frac=frac, slow_mask_frac=frac
        )
        m = sample_masks_r6(geom, n, n_time=32, n_rows=16, generator=_gen(1), cfg=cfg)
        for name, tm, length in (
            ("hga", m.hga_mask, 32), ("mid", m.mid_mask, 16), ("slow", m.slow_mask, 4)
        ):
            want = round(frac * length)
            counts = tm.sum(-1)  # (R, N) per contact
            assert torch.all(counts == want), (name, frac, counts.unique().tolist())
    print("[check] OK per-band per-sensor count == round(frac·length) exactly, all 3 bands")


def test_space_balanced_shared_and_keep_alive() -> None:
    # SPACE is ONE mask shared across bands (r4 outer-product structure), per-shaft balanced.
    sc, geom = _session([4, 4, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=64, generator=_gen(3))
    tube = m.contact_mask[..., 0]  # default per_band_space=False ⇒ all 3 slices identical
    csum = tube.long().sum(1)  # (R,) total masked per row
    assert torch.all(csum == 6), csum.unique().tolist()  # 3 shafts × round(0.5·4)=2
    assert not tube.all(1).any()  # keep-alive: never a fully-masked row
    print("[check] OK shared per-shaft-balanced space (6/12), keep-alive holds")


def test_masked_counts_constant_across_rows_static_shapes() -> None:
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=8, generator=_gen(2))
    per_row = m.contact_mask[..., 0].long().sum(1)
    assert torch.all(per_row == per_row[0])
    for tm in (m.hga_mask, m.mid_mask, m.slow_mask):
        per_sensor = tm.long().sum(-1)  # (R, N)
        assert torch.all(per_sensor == per_sensor[0, 0])
    print("[check] OK contact + per-band counts row-constant ⇒ static compiled shapes")


def test_bands_are_independent() -> None:
    # r4 structure: the three band masks are drawn INDEPENDENTLY on their own grids. Compare the
    # bands on their shared coarse (SLOW) lattice — a SLOW-masked slot need not be HGA/MID-masked.
    sc, geom = _session([4, 4, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=8, generator=_gen(5))
    hga_on_slow = m.hga_mask[..., ::8]  # (R,S,4) HGA sampled at SLOW lattice positions
    mid_on_slow = m.mid_mask[..., ::4]  # (R,S,4) MID sampled at SLOW lattice positions
    assert not torch.equal(hga_on_slow, m.slow_mask)
    assert not torch.equal(mid_on_slow, m.slow_mask)
    assert not torch.equal(hga_on_slow, mid_on_slow)
    print("[check] OK SLOW/MID/HGA masks independent (differ at shared lattice positions)")


def test_temporal_is_tubed_across_contacts_within_a_shaft() -> None:
    # THE contract (Ben 2026-08-14): a band's time blocks are a TUBE across the contacts of a
    # shaft. The encoder is L1-within-shaft only, so if contacts drew independently a masked
    # (c,t) would keep a visible same-shaft neighbour at the same t to copy from and the pretext
    # would never force temporal modelling. Within a shaft the masks must be IDENTICAL; across
    # shafts they must differ, or the draw has collapsed to r4's single global mask.
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(2))
    for tm in (m.hga_mask, m.mid_mask, m.slow_mask):
        for s in range(geom.n_shafts):
            members = [i for i in range(n) if int(geom.shaft_of_contact[i]) == s]
            for r in range(tm.shape[0]):
                for i in members[1:]:
                    assert torch.equal(tm[r, members[0]], tm[r, i]), (
                        f"shaft {s} contact {i} does not share its shaft's time mask — "
                        "the tube is broken and the same-t neighbour shortcut is open"
                    )
        heads = [
            [i for i in range(n) if int(geom.shaft_of_contact[i]) == s][0]
            for s in range(geom.n_shafts)
        ]
        assert any(
            not torch.equal(tm[r, heads[a]], tm[r, b])
            for r in range(tm.shape[0])
            for a in range(len(heads))
            for b in heads[a + 1:]
        ), "every shaft shares one mask — this is r4's global draw, not a per-shaft tube"
    print("[check] OK band masks tube across contacts within a shaft, shafts independent")


def test_band_time_unit_contact_reproduces_the_old_per_sensor_draw() -> None:
    # The pre-2026-08-14 behaviour stays reachable so those checkpoints remain reproducible.
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    cfg = V3MaskConfig(band_time_unit="contact")
    m = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(2), cfg=cfg)
    same_shaft = [i for i in range(n) if int(geom.shaft_of_contact[i]) == 1]
    for tm in (m.hga_mask, m.mid_mask, m.slow_mask):
        assert any(
            not torch.equal(tm[r, i], tm[r, j])
            for r in range(tm.shape[0])
            for a, i in enumerate(same_shaft)
            for j in same_shaft[a + 1:]
        ), "band_time_unit='contact' did not restore the per-sensor draw"
    print("[check] OK band_time_unit='contact' restores the 07-23 per-sensor draw")


def test_block_width_controls_masked_run_length() -> None:
    # block_w_band lays contiguous width-blocks: wider ⇒ longer contiguous masked runs at the SAME
    # frac (the M14 leak-safe knob). Assert the WIDTH drives it on the HGA grid (narrow vs wide).
    sc, geom = _session([6, 6])
    n = int(geom.valid.sum())
    t = 32

    def mean_max_run(tm: torch.Tensor) -> float:
        runs = []
        for row in tm.reshape(-1, t):
            best = cur = 0
            for v in row.tolist():
                cur = cur + 1 if v else 0
                best = max(best, cur)
            runs.append(best)
        return sum(runs) / len(runs)

    narrow = sample_masks_r6(
        geom, n, n_time=t, n_rows=64, generator=_gen(11),
        cfg=V3MaskConfig(block_w_band=1),
    )
    wide = sample_masks_r6(
        geom, n, n_time=t, n_rows=64, generator=_gen(11),
        cfg=V3MaskConfig(block_w_band=8),
    )
    r_narrow, r_wide = mean_max_run(narrow.hga_mask), mean_max_run(wide.hga_mask)
    assert r_wide > r_narrow + 1.0, (r_narrow, r_wide)
    print(f"[check] OK block_w_band drives run length: w=1 {r_narrow:.2f} < w=8 {r_wide:.2f}")


def test_band_block_w_1_is_uniform_random_masking() -> None:
    """A1's premise: at ``block_w_band=1`` the cover degenerates to an independent uniform draw.

    "Shorter runs" (the test above) does NOT prove randomness on its own, so assert the two
    properties that actually define it:
      1. MARGINAL uniformity — every grid position is masked at the same rate.
      2. NO CONTIGUITY — P(i+1 masked | i masked) ≈ the without-replacement rate 15/31 ≈ .484,
         whereas the leak-safe width-4 block drives that conditional far above it by construction.
    Without this, the random-masking ablation would not be testing what it claims to test.
    """
    sc, geom = _session([6, 6])
    n = int(geom.valid.sum())
    t, rows = 32, 512

    def hga(width: int) -> torch.Tensor:
        m = sample_masks_r6(
            geom, n, n_time=t, n_rows=rows, generator=_gen(7),
            cfg=V3MaskConfig(block_w_band=width),
        )
        return m.hga_mask.reshape(-1, t).float()

    w1, w4 = hga(1), hga(4)

    rate = w1.mean(0)  # (T,) per-position marginal
    dev = float((rate - 0.5).abs().max())
    assert dev < 0.08, (dev, rate.tolist())

    def adjacency(m: torch.Tensor) -> float:
        both = (m[:, :-1] * m[:, 1:]).sum()
        return float(both / m[:, :-1].sum())

    a1, a4 = adjacency(w1), adjacency(w4)
    assert a1 < 0.55, a1
    assert a4 > 0.65, a4
    print(
        f"[check] OK block_w_band=1 is UNIFORM RANDOM: max |rate-0.5| {dev:.4f} < .08; "
        f"P(i+1|i) w=1 {a1:.3f} (chance 15/31={15 / 31:.3f}) vs w=4 {a4:.3f}"
    )


def test_ntime_not_multiple_of_slow_stride_raises() -> None:
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    for bad in (20, 12):  # not divisible by SLOW_STRIDE=8
        try:
            sample_masks_r6(geom, n, n_time=bad, n_rows=2, generator=_gen())
        except ValueError:
            continue
        raise AssertionError(f"n_time={bad} should raise (not a multiple of SLOW_STRIDE)")
    print("[check] OK n_time not divisible by SLOW_STRIDE=8 raises")


def test_deterministic_in_generator_seed() -> None:
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    a = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(7))
    b = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(7))
    c = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(8))
    same = (
        torch.equal(a.contact_mask, b.contact_mask)
        and torch.equal(a.hga_mask, b.hga_mask)
        and torch.equal(a.mid_mask, b.mid_mask)
        and torch.equal(a.slow_mask, b.slow_mask)
    )
    diff = not torch.equal(a.hga_mask, c.hga_mask)
    assert same and diff
    print(f"[check] OK same seed → identical ({same}); different seed → differs ({diff})")


# ── space_frac == 0.0: the no-spatial-masking arm ────────────────────────────
# Ben 2026-07-28: ms70 → keeper → ms25 was monotone in the probe as spatial masking fell,
# so the endpoint (no spatial masking at all, time held at the ASR-convention 0.50) has to
# be reachable. It was not: assert_mask_feasible rejected Σd_s == 0. These pin down that the
# arm is WELL-DEFINED, not merely permitted — the loss must still have targets.


def test_space_frac_zero_is_feasible_and_masks_no_contacts() -> None:
    sc, geom = _session([3, 3, 4])
    cfg = V3MaskConfig(space_frac=0.0)
    assert_mask_feasible(geom, cfg)  # must not raise
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=5, generator=_gen(), cfg=cfg)
    assert not m.contact_mask.any(), "space_frac=0 must mask zero contacts"
    # the time masks are untouched — each band still hides round(0.50 * T_b) per sensor
    for name, band in (("slow", m.slow_mask), ("mid", m.mid_mask), ("hga", m.hga_mask)):
        assert band.any(), f"{name} time mask empty — the arm would have no loss targets"
    print(f"[check] space_frac=0 OK: contacts masked=0, time masks non-empty (N={n})")


def test_space_frac_zero_still_leaves_loss_targets() -> None:
    """The guard that matters: masked == time mask alone, and it is NOT empty."""
    sc, geom = _session([3, 3, 4])
    n = int(geom.valid.sum())
    grid = build_r4_grid(geom, n_time=32)
    m = sample_masks_r6(
        geom, n, n_time=32, n_rows=4, generator=_gen(), cfg=V3MaskConfig(space_frac=0.0)
    )
    masked, in_loss = token_flags_r6(grid, m)
    assert masked.any(), "no masked tokens ⇒ _masked_mean_l1 would divide by zero"
    assert bool((in_loss == masked).all()), "r6 contract: in_loss == masked"
    frac = float(masked.float().mean())
    # space ∪ time collapses to time alone; the three bands sit at 0.50 each.
    assert 0.3 < frac < 0.7, f"masked fraction {frac:.3f} implausible for time-only masking"
    print(f"[check] space_frac=0 loss targets OK: masked fraction={frac:.3f}, in_loss==masked")


def test_degenerate_montage_still_raises_when_masking_was_requested() -> None:
    """The guard keeps its real job: space_frac > 0 that silently masks nothing is a bug."""
    sc, geom = _session([1, 1, 1])  # every shaft size 1 ⇒ keep_alive forces d_s = 0
    try:
        assert_mask_feasible(geom, V3MaskConfig(space_frac=0.5))
    except ValueError as e:
        assert "not in (0, N=" in str(e)
        print(f"[check] degenerate montage still raises: {e}")
        return
    raise AssertionError("space_frac=0.5 on all-size-1 shafts must still raise")


# ── R19+R20: per-band SPACE masks + per-band spatial block width ──────────────
# Ben 2026-07-29: "I like the symmetry for independent band mask for both space and time."
# The three TIME masks were already independent per band; SPACE was the lone shared axis. These
# pin the two properties that make it a legitimate single-knob arm: the DEFAULT is byte-identical
# to the locked tube, and turning it on changes ARRANGEMENT ONLY — never the masked token count.


def test_default_is_the_shared_tube_all_three_band_slices_identical() -> None:
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=8, generator=_gen(11))
    assert torch.equal(m.contact_mask[..., 0], m.contact_mask[..., 1])
    assert torch.equal(m.contact_mask[..., 1], m.contact_mask[..., 2])
    print("[check] OK per_band_space=False ⇒ one draw broadcast; band slices identical (the tube)")


def test_default_generator_consumption_matches_the_r4_space_prologue() -> None:
    """THE REGRESSION GUARD for every existing r6 run.

    ``sample_masks`` (r4) and ``sample_masks_r6`` share a verbatim space prologue drawing from the
    generator in the same order. If R19 had added or reordered a draw under the default, the two
    would diverge. This pins generator consumption without needing the pre-R19 code."""
    sc, geom = _session([3, 5, 4])
    n = int(geom.valid.sum())
    r4 = sample_masks(geom, n, n_time=32, n_rows=8, generator=_gen(21))
    r6 = sample_masks_r6(geom, n, n_time=32, n_rows=8, generator=_gen(21))
    assert torch.equal(r4.contact_mask, r6.contact_mask[..., 0]), "r6 space draw diverged from r4"
    print("[check] OK default space draw bit-matches sample_masks ⇒ generator order unchanged")


def test_per_band_space_bands_differ_but_counts_are_matched_visible() -> None:
    sc, geom = _session([4, 6, 8])
    n = int(geom.valid.sum())
    cfg = V3MaskConfig(per_band_space=True)
    m = sample_masks_r6(geom, n, n_time=32, n_rows=32, generator=_gen(12), cfg=cfg)
    assert not torch.equal(m.contact_mask[..., 0], m.contact_mask[..., 2]), "bands must differ"
    # MATCHED-VISIBLE: exact-count snapping runs per band ⇒ identical masked count in every band.
    per_band = m.contact_mask.long().sum(1)  # (R, 3)
    assert torch.all(per_band == per_band[:, :1]), per_band.unique().tolist()
    tube = sample_masks_r6(geom, n, n_time=32, n_rows=32, generator=_gen(12)).contact_mask
    assert int(per_band[0, 0]) == int(tube[..., 0].long().sum(1)[0]), "count must match the tube"
    print(f"[check] OK per-band space differs across bands, count identical ({int(per_band[0,0])}/{n} every band)")


def test_per_band_space_creates_the_cross_band_bridge() -> None:
    """The mechanism under test: a contact spatially masked in HGA but VISIBLE in SLOW/MID.
    Impossible under the tube by construction — that is the whole point of the arm."""
    sc, geom = _session([4, 6, 8])
    n = int(geom.valid.sum())
    cfg = V3MaskConfig(per_band_space=True)
    m = sample_masks_r6(geom, n, n_time=32, n_rows=32, generator=_gen(13), cfg=cfg)
    hga_only = m.contact_mask[..., 2] & ~m.contact_mask[..., 0] & ~m.contact_mask[..., 1]
    assert hga_only.any(), "no contact masked in HGA alone ⇒ no cross-band bridge"
    tube = sample_masks_r6(geom, n, n_time=32, n_rows=32, generator=_gen(13)).contact_mask
    assert not (tube[..., 2] & ~tube[..., 0]).any(), "the tube must never produce a bridge"
    print(f"[check] OK cross-band bridge exists ({int(hga_only.sum())} HGA-only contacts); tube has none")


def test_per_band_widths_change_arrangement_not_count() -> None:
    sc, geom = _session([8, 8, 16])
    n = int(geom.valid.sum())
    base = V3MaskConfig(per_band_space=True)
    wide = V3MaskConfig(per_band_space=True, block_w_space_bands=(6, 4, 1))
    a = sample_masks_r6(geom, n, n_time=32, n_rows=32, generator=_gen(14), cfg=base)
    b = sample_masks_r6(geom, n, n_time=32, n_rows=32, generator=_gen(14), cfg=wide)
    assert not torch.equal(a.contact_mask, b.contact_mask), "widths must change the arrangement"
    assert torch.equal(a.contact_mask.long().sum(1), b.contact_mask.long().sum(1)), \
        "widths must NOT change the masked count — that would break matched-visible"
    # a wider block must produce longer contiguous runs; a width-1 block, shorter.
    def longest_run(v: torch.Tensor) -> int:
        best = cur = 0
        for x in v.tolist():
            cur = cur + 1 if x else 0
            best = max(best, cur)
        return best
    shaft16 = geom.gather_idx[2][geom.valid[2]]  # the 16-contact shaft, in depth order
    slow_runs = max(longest_run(b.contact_mask[r][shaft16, 0]) for r in range(32))
    hga_runs = max(longest_run(b.contact_mask[r][shaft16, 2]) for r in range(32))
    assert slow_runs > hga_runs, (slow_runs, hga_runs)
    print(f"[check] OK per-band widths: count fixed, SLOW(w6) run {slow_runs} > HGA(w1) run {hga_runs}")


def test_per_band_widths_without_per_band_space_raises() -> None:
    """A per-band width under ONE shared draw would silently apply only the LAST band's width."""
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    cfg = V3MaskConfig(block_w_space_bands=(6, 4, 1))
    try:
        sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(), cfg=cfg)
    except ValueError as e:
        assert "per_band_space" in str(e)
        print("[check] OK block_w_space_bands without per_band_space raises")
        return
    raise AssertionError("expected ValueError")


def test_token_flags_reads_space_per_band() -> None:
    """End-to-end: a contact masked ONLY in HGA must flag its HGA tokens and no others."""
    sc, geom = _session([4, 6])
    n = int(geom.valid.sum())
    grid = build_r4_grid(geom, n_time=32)
    zeros = lambda t: torch.zeros(1, n, t, dtype=torch.bool)  # noqa: E731
    contact = torch.zeros(1, n, 3, dtype=torch.bool)
    contact[0, 2, 2] = True  # contact 2, HGA only
    masks = V3MasksR6(
        contact_mask=contact, hga_mask=zeros(32), mid_mask=zeros(16), slow_mask=zeros(4)
    )
    masked, in_loss = token_flags_r6(grid, masks)
    hit = masked[0]
    assert bool(hit[(grid.contact == 2) & (grid.band == 2)].all()), "all HGA tokens must be masked"
    assert not bool(hit[(grid.contact == 2) & (grid.band != 2)].any()), "SLOW/MID must stay visible"
    assert not bool(hit[grid.contact != 2].any()), "other contacts untouched"
    assert bool((in_loss == masked).all())
    print(f"[check] OK token_flags_r6 space is per band ({int(hit.sum())} tokens, HGA of contact 2 only)")


def _masked_fraction(m: V3MasksR6, *, n_time: int) -> float:
    """Fraction of (row, contact, frame) tokens that are masked, pooled over all 3 bands.

    A token is masked if its contact is spatially masked in that band OR its frame is in a
    temporal block. This is the union the loss actually scores, so it is the quantity two
    arms have to match on for the comparison between them to be about ARRANGEMENT."""
    tot = hit = 0
    for bi, (tm, length) in enumerate(
        ((m.slow_mask, n_time // 8), (m.mid_mask, n_time // 2), (m.hga_mask, n_time))
    ):
        space = m.contact_mask[..., bi].unsqueeze(-1)  # (R, N, 1)
        masked = space | tm.bool()
        hit += int(masked.sum())
        tot += masked.numel()
        assert tm.shape[-1] == length
    return hit / tot


def test_no_time_mask_and_no_space_mask_mask_the_same_fraction() -> None:
    # THE PRECONDITION FOR ARM B (no temporal mask, 2026-08-04). The arm is only interpretable
    # against pbs00 if the two mask the SAME fraction and differ only in which axis does it.
    # Measured on the real sampler rather than argued from the fractions: exact-count snapping,
    # keep_alive and the block cover all sit between cfg and the realized count.
    sc, geom = _session([4, 4, 4])  # round(0.5*4)=2 per shaft ⇒ the spatial half is exact
    n = int(geom.valid.sum())
    no_time = V3MaskConfig(hga_mask_frac=0.0, mid_mask_frac=0.0, slow_mask_frac=0.0)
    pbs00 = V3MaskConfig(space_frac=0.0)
    f_no_time = _masked_fraction(
        sample_masks_r6(geom, n, n_time=32, n_rows=64, generator=_gen(7), cfg=no_time),
        n_time=32,
    )
    f_pbs00 = _masked_fraction(
        sample_masks_r6(geom, n, n_time=32, n_rows=64, generator=_gen(7), cfg=pbs00),
        n_time=32,
    )
    assert abs(f_no_time - 0.50) < 1e-9, f_no_time
    assert abs(f_pbs00 - 0.50) < 1e-9, f_pbs00
    # and both are strictly LESS masked than the locked 0.50-space x 0.50-time config
    f_locked = _masked_fraction(
        sample_masks_r6(geom, n, n_time=32, n_rows=64, generator=_gen(7)), n_time=32
    )
    assert f_locked > 0.70, f_locked
    print(f"[check] OK masked fraction — no-time {f_no_time:.4f}, pbs00 {f_pbs00:.4f} "
          f"(matched pair), locked space.50xtime.50 {f_locked:.4f}")


def test_shaft_budget_total_is_identical_to_the_per_contact_draw() -> None:
    # THE SHAPE CLAIM (Ben 2026-08-15). shaft_budget frees the per-CONTACT count but must leave the
    # per-SHAFT total byte-identical to the per-contact draw, because that total (n_s·cnt) is what
    # every downstream m_vis / cu_seqlens / compiled shape is derived from. If this ever fails, the
    # arm is not a masking ablation, it is a silent shape change.
    sc, geom = _session([3, 6, 8, 5])
    n = int(geom.valid.sum())
    cfg_c = V3MaskConfig(band_time_unit="contact", hga_mask_frac=0.75,
                         mid_mask_frac=0.75, slow_mask_frac=0.75)
    cfg_b = V3MaskConfig(band_time_unit="shaft_budget", hga_mask_frac=0.75,
                         mid_mask_frac=0.75, slow_mask_frac=0.75)
    mc = sample_masks_r6(geom, n, n_time=32, n_rows=8, generator=_gen(11), cfg=cfg_c)
    mb = sample_masks_r6(geom, n, n_time=32, n_rows=8, generator=_gen(11), cfg=cfg_b)

    for name, tc, tb in (("hga", mc.hga_mask, mb.hga_mask),
                         ("mid", mc.mid_mask, mb.mid_mask),
                         ("slow", mc.slow_mask, mb.slow_mask)):
        assert tc.shape == tb.shape, (name, tc.shape, tb.shape)
        for s in range(geom.n_shafts):
            idx = [i for i in range(n) if int(geom.shaft_of_contact[i]) == s]
            got = tb[:, idx].sum(dim=(1, 2))
            want = tc[:, idx].sum(dim=(1, 2))
            assert torch.equal(got, want), (name, s, got.tolist(), want.tolist())
    print("[check] OK shaft_budget per-SHAFT total == per-contact total, all 3 bands, every shaft")


def test_shaft_budget_frees_the_per_contact_fraction() -> None:
    # THE POINT OF THE ARM. Under 'contact' every contact is snapped to exactly round(frac·L), so
    # every sensor keeps exactly (1−frac) of its frames and none is ever close to fully hidden —
    # which is the regime the SPACE tier exists to create. Under 'shaft_budget' the per-contact
    # fraction must actually SPREAD, otherwise the arm cannot subsume the space tier and running it
    # with space_frac=0 would test nothing.
    sc, geom = _session([8, 8, 8])
    n = int(geom.valid.sum())
    cfg = V3MaskConfig(band_time_unit="shaft_budget", space_frac=0.0,
                       hga_mask_frac=0.75, mid_mask_frac=0.75, slow_mask_frac=0.75)
    m = sample_masks_r6(geom, n, n_time=64, n_rows=32, generator=_gen(12), cfg=cfg)

    per_contact = m.hga_mask.float().mean(-1)  # (R, N) masked fraction per (row, contact)
    lo, hi = float(per_contact.min()), float(per_contact.max())
    assert per_contact.std() > 0.02, f"per-contact fraction did not spread (std {per_contact.std():.4f})"
    assert hi > 0.85, f"no contact was heavily masked (max {hi:.3f}) — space tier has nothing to do"
    assert lo < 0.65, f"no contact was lightly masked (min {lo:.3f})"
    # and the per-contact draw is the flat control: exactly one value, no spread at all.
    cfg_c = V3MaskConfig(band_time_unit="contact", space_frac=0.0, hga_mask_frac=0.75,
                         mid_mask_frac=0.75, slow_mask_frac=0.75)
    flat = sample_masks_r6(geom, n, n_time=64, n_rows=32, generator=_gen(12),
                           cfg=cfg_c).hga_mask.float().mean(-1)
    assert float(flat.std()) == 0.0, f"per-contact draw should be flat, std {float(flat.std())}"
    print(f"[check] OK shaft_budget per-contact masked fraction spreads {lo:.3f}..{hi:.3f} "
          f"(std {float(per_contact.std()):.4f}); 'contact' is flat at {float(flat[0, 0]):.3f}")


def test_shaft_budget_is_not_a_tube_and_shafts_stay_independent() -> None:
    # shaft_budget shares a BUDGET within a shaft, never a MASK. Contacts of a shaft must still
    # differ (else it collapsed into the 08-14 tube), and shafts must not be drawn together.
    sc, geom = _session([5, 5, 5])
    n = int(geom.valid.sum())
    cfg = V3MaskConfig(band_time_unit="shaft_budget", hga_mask_frac=0.75,
                       mid_mask_frac=0.75, slow_mask_frac=0.75)
    m = sample_masks_r6(geom, n, n_time=32, n_rows=4, generator=_gen(13), cfg=cfg)
    same = [i for i in range(n) if int(geom.shaft_of_contact[i]) == 1]
    assert any(
        not torch.equal(m.hga_mask[r, i], m.hga_mask[r, j])
        for r in range(m.hga_mask.shape[0])
        for a, i in enumerate(same)
        for j in same[a + 1:]
    ), "shaft_budget collapsed into the shaft tube — contacts share a mask"
    heads = [
        next(i for i in range(n) if int(geom.shaft_of_contact[i]) == s)
        for s in range(geom.n_shafts)
    ]
    assert any(
        not torch.equal(m.hga_mask[r, a], m.hga_mask[r, b])
        for r in range(m.hga_mask.shape[0])
        for k, a in enumerate(heads)
        for b in heads[k + 1:]
    ), "shafts are not independent under shaft_budget"
    print("[check] OK shaft_budget shares a BUDGET within a shaft, not a mask; shafts independent")


def test_unknown_band_time_unit_is_a_hard_error() -> None:
    # The whole band_time_unit family is a NO-TRACE flag: nothing lands in the state_dict, so a
    # typo used to fall through to the per-contact draw and run the wrong arm with no error.
    sc, geom = _session([3, 4])
    n = int(geom.valid.sum())
    cfg = V3MaskConfig(band_time_unit="shaftbudget")  # missing underscore
    try:
        sample_masks_r6(geom, n, n_time=32, n_rows=2, generator=_gen(14), cfg=cfg)
    except ValueError as e:
        assert "shaft_budget" in str(e)
        print("[check] OK unknown band_time_unit raises instead of silently running 'contact'")
        return
    raise AssertionError("unknown band_time_unit did not raise")


# ── mask_iid (Ben 2026-08-15): no tiers, ONE global permutation over the whole token grid ──


def _iid_cfg(frac: float = 0.75) -> V3MaskConfig:
    return V3MaskConfig(mask_iid=True, iid_mask_frac=frac, space_frac=0.0, whole_shaft_frac=0.0)


def test_iid_masks_exactly_frac_of_the_WHOLE_grid_per_row() -> None:
    # THE invariant. M_vis is a compile-time constant for the encoder's visible-token gather
    # (objective.py:190), so the GLOBAL count per row must be exact even though every per-contact
    # and per-band count is free. A Bernoulli draw would pass the marginal and fail this.
    # Since the draw became per-attention-unit (Ben 2026-08-16) the global count is the SUM of the
    # per-unit rounds, which need not equal round(frac·total): at frac=.90 on this montage it is
    # 514 against 515. Constancy is the requirement, not that particular arithmetic, so assert the
    # requirement -- and separately that the realised rate still tracks frac to within a token per
    # unit, so a rounding drift can never quietly become a rate mismatch against the two-tier arms.
    sc, geom = _session([4, 4, 3])
    n = int(geom.valid.sum())
    t = 32
    total = n * (t + t // 2 + t // 8)  # HGA 32 + MID 16 + SLOW 4 per contact
    grid = build_r4_grid(geom, n_time=t)
    n_unit = int(grid.cu_seqlens.numel() - 1)
    cells = torch.bincount(grid.shaft, minlength=n_unit)
    for frac in (0.75, 0.50, 0.90):
        m = sample_masks_r6(geom, n, n_time=t, n_rows=7, generator=_gen(1), cfg=_iid_cfg(frac))
        per_row = (
            m.hga_mask.flatten(1).sum(1) + m.mid_mask.flatten(1).sum(1)
            + m.slow_mask.flatten(1).sum(1)
        )
        want = int(torch.round(frac * cells.float()).long().sum())
        assert per_row.tolist() == [want] * 7, f"frac={frac}: {per_row.tolist()} != {want}"
        assert abs(want - round(frac * total)) <= n_unit, \
            f"frac={frac}: per-unit rounding drifted {want} vs {round(frac * total)} over {n_unit} units"
        print(f"[check] OK iid frac={frac}: exactly {want}/{total} masked in EVERY row "
              f"(sum of {n_unit} per-unit rounds; round(frac*total)={round(frac * total)})")


def test_iid_leaves_contact_mask_empty_so_the_space_tier_is_truly_off() -> None:
    # The mask is a space (x) time outer product downstream (token_flags_r6). The i.i.d. arm carries
    # its whole pattern in the three band tensors, so contact_mask must contribute NOTHING — if it
    # did, the realised rate would exceed iid_mask_frac.
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    m = sample_masks_r6(geom, n, n_time=32, n_rows=5, generator=_gen(2), cfg=_iid_cfg())
    assert m.contact_mask.shape == (5, n, 3)
    assert not bool(m.contact_mask.any()), "contact_mask must be all-False under mask_iid"

    grid = build_r4_grid(geom, n_time=32)
    masked, in_loss = token_flags_r6(grid, m)
    frac = masked.float().mean(1)
    assert torch.allclose(frac, torch.full_like(frac, 0.75), atol=1.0 / grid.total), \
        f"realised token-grid rate {frac.tolist()} != 0.75"
    assert bool((masked == in_loss).all()), "r6 scores every masked token (no margin gate)"
    print(f"[check] OK contact_mask empty; realised grid rate {float(frac.mean()):.4f} == 0.75")


def test_iid_frees_the_per_contact_band_fraction_that_the_per_contact_draw_snaps() -> None:
    # This is the ONLY thing that separates the i.i.d. arm from the sf0/w1 arm. The per-contact draw
    # snaps every (contact, band) to exactly round(frac*length); i.i.d. must NOT. The freedom is
    # largest on the shortest grid (SLOW = 8 tokens at 2 s), which is where a contact can go fully
    # masked -- impossible under the snapped draw.
    sc, geom = _session([5, 5, 5, 5])
    n = int(geom.valid.sum())
    t = 64  # 2 s at 32 Hz => SLOW 8, MID 32, HGA 64
    m = sample_masks_r6(geom, n, n_time=t, n_rows=64, generator=_gen(3), cfg=_iid_cfg())
    snapped = sample_masks_r6(
        geom, n, n_time=t, n_rows=64, generator=_gen(3),
        cfg=V3MaskConfig(space_frac=0.0, whole_shaft_frac=0.0, block_w_band=1,
                         band_time_unit="contact", hga_mask_frac=0.75,
                         mid_mask_frac=0.75, slow_mask_frac=0.75),
    )
    for name, iid_b, snap_b, length in (
        ("SLOW", m.slow_mask, snapped.slow_mask, t // 8),
        ("MID", m.mid_mask, snapped.mid_mask, t // 2),
        ("HGA", m.hga_mask, snapped.hga_mask, t),
    ):
        snap_frac = snap_b.float().mean(-1)
        assert snap_frac.unique().numel() == 1, f"{name}: per-contact draw is not snapped"
        iid_frac = iid_b.float().mean(-1)
        assert iid_frac.unique().numel() > 1, f"{name}: iid per-contact fraction is not free"
        full = float((iid_frac == 1.0).float().mean())
        print(f"[check] OK {name} (len {length}): snapped sd {float(snap_frac.std()):.5f} -> "
              f"iid sd {float(iid_frac.std()):.5f}, fully-masked cells {100 * full:.1f}%")
    slow_full = float((m.slow_mask.float().mean(-1) == 1.0).float().mean())
    assert slow_full > 0.0, "SLOW should sometimes go fully masked under iid (p ~ .75**8 = .10)"


def test_iid_raises_rather_than_silently_composing_with_a_space_tier() -> None:
    # The two tiers compose multiplicatively. A leftover space_frac would push the realised rate
    # past iid_mask_frac and silently break the rate match against every two-tier arm, with nothing
    # in the state_dict to catch it later -- the no-trace-flag failure mode.
    sc, geom = _session([4, 4])
    n = int(geom.valid.sum())
    for bad in (V3MaskConfig(mask_iid=True, space_frac=0.50, whole_shaft_frac=0.0),
                V3MaskConfig(mask_iid=True, space_frac=0.0, whole_shaft_frac=0.25)):
        try:
            sample_masks_r6(geom, n, n_time=32, n_rows=3, generator=_gen(), cfg=bad)
        except ValueError as e:
            assert "mask_iid" in str(e)
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    print("[check] OK mask_iid + a live space tier raises instead of composing")


def _per_unit_visible(grid, masks) -> "torch.Tensor":
    """(R, S) visible-token count per attention unit — exactly what build_visible_pack segments on."""
    masked, _ = token_flags_r6(grid, masks)
    n_unit = int(grid.cu_seqlens.numel() - 1)
    out = torch.zeros(masked.shape[0], n_unit, dtype=torch.long)
    out.scatter_add_(1, grid.shaft[None, :].expand(masked.shape[0], -1), (~masked).long())
    return out


def test_every_arm_holds_the_per_unit_visible_count_that_the_pack_copies_from_clip_0() -> None:
    # 🔴 THE INVARIANT THAT KILLED JOB 2955569. build_visible_pack takes the per-unit visible counts
    # from CLIP 0 and applies them to the entire batch (pack_r4.py:212), because
    # towers.forward_flat_pack documents cu_seqlens as "clip-shared (per-shaft visible count is a
    # per-session constant)". Nothing enforced it. A global-permutation mask holds M_vis per clip --
    # so it compiles, launches, and trains -- while breaking this one, which mis-groups attention on
    # most clips and finally runs the varlen kernel off the end of a block: all 247 tensors NaN.
    # Every arm we can run must satisfy this, so assert it for ALL of them, not just the i.i.d. one.
    sc, geom = _session([10, 12, 8, 14, 10, 12])
    n = int(geom.valid.sum())
    t = 32
    grid = build_r4_grid(geom, n_time=t)
    arms = {
        "two-tier sf0 t.75 w1 unit=contact": V3MaskConfig(
            space_frac=0.0, whole_shaft_frac=0.0, block_w_band=1, band_time_unit="contact",
            hga_mask_frac=0.75, mid_mask_frac=0.75, slow_mask_frac=0.75),
        "two-tier sf0 t.75 w1 unit=shaft": V3MaskConfig(
            space_frac=0.0, whole_shaft_frac=0.0, block_w_band=1, band_time_unit="shaft",
            hga_mask_frac=0.75, mid_mask_frac=0.75, slow_mask_frac=0.75),
        "two-tier sf.50 t.50 w4 (canon)": V3MaskConfig(
            space_frac=0.50, whole_shaft_frac=0.0, block_w_band=4, band_time_unit="shaft"),
        "iid per-attention-unit": _iid_cfg(0.75),
    }
    for name, cfg in arms.items():
        m = sample_masks_r6(geom, n, n_time=t, n_rows=64, generator=_gen(33), cfg=cfg)
        per_unit = _per_unit_visible(grid, m)
        assert bool((per_unit == per_unit[0][None, :]).all()), (
            f"{name}: per-unit visible count VARIES across clips, so build_visible_pack would "
            f"segment every clip with clip 0's boundaries. clip0={per_unit[0].tolist()} "
            f"worst={per_unit[(per_unit - per_unit[0]).abs().sum(1).argmax()].tolist()}"
        )
        m_vis = (~token_flags_r6(grid, m)[0]).sum(1)
        assert int(m_vis.min()) == int(m_vis.max()), f"{name}: M_vis varies across clips"
        print(f"[check] OK {name}: per-unit counts constant, max_seqlen "
              f"{int(per_unit[0].max())}, M_vis {int(m_vis[0])}")


def test_iid_masks_exactly_the_frac_within_every_attention_unit() -> None:
    # The positive statement of the rule (Ben 2026-08-16): "drop 75% random per sensor unit --
    # whether that is shaft rn, or ECoG in the future -- per full joint attention unit."
    sc, geom = _session([10, 12, 8, 14])
    n = int(geom.valid.sum())
    t = 32
    grid = build_r4_grid(geom, n_time=t)
    for frac in (0.75, 0.50):
        m = sample_masks_r6(geom, n, n_time=t, n_rows=32, generator=_gen(7), cfg=_iid_cfg(frac))
        masked, _ = token_flags_r6(grid, m)
        n_unit = int(grid.cu_seqlens.numel() - 1)
        got = torch.zeros(masked.shape[0], n_unit, dtype=torch.long)
        got.scatter_add_(1, grid.shaft[None, :].expand(masked.shape[0], -1), masked.long())
        cells = torch.bincount(grid.shaft, minlength=n_unit)
        want = torch.round(frac * cells.float()).long()
        assert bool((got == want[None, :]).all()), \
            f"frac={frac}: per-unit masked counts {got[0].tolist()} != {want.tolist()}"
        print(f"[check] OK frac={frac}: EXACTLY {want.tolist()} masked per attention unit, every clip")
