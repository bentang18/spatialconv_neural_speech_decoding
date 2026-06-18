"""TDD for the v14 converged architecture (``v14_converged``).

Component 1: the 3STFT per-electrode frontend tokenizer. Tests pin the LOCKED
ladder geometry (FE spec §2/§5), the lossless-lift patch dims, electrode
ISOLATION (Stage-1 contract), and the shared 8:2:1 RoPE clock.
"""

from __future__ import annotations

import pytest
import torch

from speech_decoding.models import v14_converged as vc
from speech_decoding.models.v14_encoder import _apply_rope


# ----------------------------------------------------------- band geometry
def test_band_geometry_matches_locked_table() -> None:
    # FE spec §2: slow 3×2=6, beta 2×8=16, HG 1×16=16 → 38 tokens; F_p 3+2+1=6.
    assert (vc.SLOW.n_freq_patches, vc.SLOW.n_time_patches, vc.SLOW.n_tokens) == (3, 2, 6)
    assert (vc.BETA.n_freq_patches, vc.BETA.n_time_patches, vc.BETA.n_tokens) == (2, 8, 16)
    assert (vc.HG.n_freq_patches, vc.HG.n_time_patches, vc.HG.n_tokens) == (1, 16, 16)
    assert vc.N_TOKENS == 38
    assert vc.N_FREQ_PATCHES == 6


def test_patch_input_dims_are_lossless_lift() -> None:
    # FE §5: patch input dim = fk·tk·channels = slow 8, beta 6, HG 18.
    assert vc.SLOW.patch_input_dim == 8
    assert vc.BETA.patch_input_dim == 6
    assert vc.HG.patch_input_dim == 18


def test_time_patch_strides_are_8_2_1() -> None:
    # FE §6: tk=2 everywhere → strides tk·hop = N → slow 1024 / beta 256 / HG 128.
    assert vc.SLOW.time_patch_stride_samples == 1024
    assert vc.BETA.time_patch_stride_samples == 256
    assert vc.HG.time_patch_stride_samples == 128
    assert vc.SLOW.kernel_time == vc.BETA.kernel_time == vc.HG.kernel_time == 2


# ------------------------------------------------------------- tokenizer I/O
def _fake_bands(B: int, C: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slow = torch.randn(B, C, 2, 6, 5)   # (Re, Im)
    beta = torch.randn(B, C, 6, 17)
    hg = torch.randn(B, C, 9, 33)
    return slow, beta, hg


def test_tokenizer_output_shape() -> None:
    tok = vc.ThreeBandTokenizer(d_model=32)
    out = tok(*_fake_bands(2, 7))
    assert out.shape == (2, 7, 38, 32)


def test_tokenizer_conv_fan_in_matches_patch_dims() -> None:
    # The Conv2d IS the lossless lift: its per-output fan-in = patch_input_dim.
    tok = vc.ThreeBandTokenizer(d_model=16)
    for band, stem in zip(vc.BANDS, tok.stems):
        w = stem.conv.weight  # (d, Cin, kf, kt)
        assert w.shape[1] * w.shape[2] * w.shape[3] == band.patch_input_dim


def test_tokenizer_metadata_band_layout() -> None:
    tok = vc.ThreeBandTokenizer(d_model=8)
    assert tok.band_id.tolist() == [0] * 6 + [1] * 16 + [2] * 16
    # freq_global_id spans [0,6); slow 0..2, beta 3..4, HG 5.
    assert set(tok.freq_global_id.tolist()) == set(range(6))
    assert tok.freq_global_id[:6].max().item() == 2          # slow uses 0..2
    assert set(tok.freq_global_id[6:22].tolist()) == {3, 4}  # beta uses 3,4
    assert set(tok.freq_global_id[22:].tolist()) == {5}      # HG uses 5


def test_tokenizer_time_slot_8_2_1_clock() -> None:
    tok = vc.ThreeBandTokenizer(d_model=8)
    slow_slots = tok.time_slot[:6]                  # ×8, tp∈{0,1}
    beta_slots = tok.time_slot[6:22]                # ×2, tp∈0..7
    hg_slots = tok.time_slot[22:]                   # ×1, tp∈0..15
    assert set(slow_slots.tolist()) == {0, 8}
    assert set(beta_slots.tolist()) == {0, 2, 4, 6, 8, 10, 12, 14}
    assert set(hg_slots.tolist()) == set(range(16))
    # HG defines the finest grid → max slot 15 spans the full 1 s clip.
    assert tok.time_slot.max().item() == 15


# -------------------------------------------------------- STAGE-1 ISOLATION
def test_electrode_isolation_permutation_equivariance() -> None:
    # The load-bearing Stage-1 contract: no cross-electrode pathway. Permuting
    # the electrode axis must permute the output tokens identically (each
    # electrode is tokenized independently in the batch dim).
    torch.manual_seed(0)
    tok = vc.ThreeBandTokenizer(d_model=24).eval()
    slow, beta, hg = _fake_bands(2, 6)
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    with torch.no_grad():
        base = tok(slow, beta, hg)
        permed = tok(slow[:, perm], beta[:, perm], hg[:, perm])
    assert torch.allclose(permed, base[:, perm], atol=1e-6)


def test_electrode_isolation_no_cross_leak() -> None:
    # Changing ONE electrode's input must not change any OTHER electrode's tokens.
    torch.manual_seed(1)
    tok = vc.ThreeBandTokenizer(d_model=24).eval()
    slow, beta, hg = _fake_bands(1, 4)
    with torch.no_grad():
        base = tok(slow, beta, hg)
        slow2 = slow.clone()
        slow2[:, 2] += 5.0  # perturb only electrode 2
        out = tok(slow2, beta, hg)
    untouched = [0, 1, 3]
    assert torch.allclose(out[:, untouched], base[:, untouched], atol=1e-6)
    assert not torch.allclose(out[:, 2], base[:, 2])  # electrode 2 did change


# ------------------------------------------------------ construction guards
def test_non_integer_multiple_hops_rejected() -> None:
    # A band whose time-patch stride is not an integer multiple of the finest
    # stride breaks the shared clock → must raise at construction.
    bad = vc.BandSpec("bad", 100, 50, 6, 17, kernel_freq=3, kernel_time=3, in_channels=1)
    with pytest.raises(ValueError, match="integer multiple"):
        vc.ThreeBandTokenizer(d_model=8, bands=(vc.HG, bad))


def test_d_model_must_be_positive() -> None:
    with pytest.raises(ValueError, match="d_model"):
        vc.ThreeBandTokenizer(d_model=0)


# ============================================================ M2 masking (§8)
def _gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_m2_slow_always_exempt() -> None:
    # FE §8.2: slow is EXEMPT — never an M2 target, under any draw.
    for seed in range(25):
        m = vc.sample_m2_mask(_gen(seed))
        assert m.shape == (38,)
        assert not m[:6].any(), f"slow masked at seed {seed} — must be exempt"


def test_m2_beta_freq_tube_is_50pct_both_subbands() -> None:
    # FE §8.5: beta = 1 start × width 4, BOTH freq-patches co-masked = 8/16 = 50%.
    for seed in range(25):
        beta = vc.sample_m2_mask(_gen(seed))[6:22].reshape(2, 8)  # (F_p, T_p)
        assert beta.sum().item() == 8, "beta tube = 4 time × 2 freq = 8"
        assert torch.equal(beta[0], beta[1]), "freq-tube: sub-bands co-masked"
        cols = beta[0].nonzero().flatten()
        assert cols.numel() == 4 and (cols[-1] - cols[0]).item() == 3, "contiguous w=4"


def test_m2_hg_span_mask_coverage() -> None:
    # FE §8.5 (Ben 2026-06-18 → 20% default): HG = round(0.20·16)=3 distinct starts
    # × width 3, overlaps allowed. Distinct starts ⇒ union ∈ [5,9] (5 when the 3
    # starts are consecutive → one run of 5; 9 when disjoint).
    for seed in range(25):
        hg = vc.sample_m2_mask(_gen(seed))[22:38]
        assert 5 <= hg.sum().item() <= 9, f"HG coverage {hg.sum()} out of [5,9]"


def test_m2_total_in_maskable_pool_range() -> None:
    # FE §8.6: pool = 32 (beta+HG); realized ~beta 8 + HG 5–9 ≈ 13–17 (20% default).
    counts = [vc.sample_m2_mask(_gen(s)).sum().item() for s in range(60)]
    assert all(13 <= c <= 17 for c in counts)
    assert 13 <= sum(counts) / len(counts) <= 17


def test_m2_determinism() -> None:
    assert torch.equal(vc.sample_m2_mask(_gen(7)), vc.sample_m2_mask(_gen(7)))
    assert not torch.equal(vc.sample_m2_mask(_gen(7)), vc.sample_m2_mask(_gen(8)))


def test_m2_beta_width_sister() -> None:
    # §8.3 bleed floor: min width 3 (the redundancy sister sets beta_span).
    beta = vc.sample_m2_mask(_gen(0), cfg=vc.M2MaskConfig(beta_span=3))[6:22].reshape(2, 8)
    assert beta.sum().item() == 6  # width 3 × 2 freq-patches


def test_m2_hg_start_rate_is_coverage_dial() -> None:
    # §8.6: p (hg_start_rate) is THE coverage knob — higher p, more HG masked.
    lo = [vc.sample_m2_mask(_gen(s), cfg=vc.M2MaskConfig(hg_start_rate=0.15)).sum().item()
          for s in range(50)]
    hi = [vc.sample_m2_mask(_gen(s), cfg=vc.M2MaskConfig(hg_start_rate=0.30)).sum().item()
          for s in range(50)]
    assert sum(hi) / 50 > sum(lo) / 50


# ===================================================== M4 parcel-tube (latent)
def test_m4_tube_ratio_and_subset() -> None:
    present = torch.arange(50)
    for seed in range(20):
        tubed = vc.sample_parcel_tube(present, _gen(seed))
        assert tubed.numel() == 10                       # round(0.20·50)
        assert torch.isin(tubed, present).all()          # subset
        assert tubed.unique().numel() == tubed.numel()   # distinct


def test_m4_tube_keeps_context_and_target() -> None:
    # ≥1 tubed (M4 has a target) AND ≥1 un-tubed (M4 has visible context).
    for n in (2, 3, 5, 17, 100):
        present = torch.arange(n)
        tubed = vc.sample_parcel_tube(present, _gen(n))
        assert 1 <= tubed.numel() <= n - 1


def test_m4_tube_single_parcel_inert() -> None:
    # 1-parcel clip → nothing to infer from → tube nothing (M4 inert).
    assert vc.sample_parcel_tube(torch.tensor([7]), _gen(0)).numel() == 0


def test_m4_tube_determinism_and_dial() -> None:
    present = torch.arange(40)
    assert torch.equal(vc.sample_parcel_tube(present, _gen(3)),
                       vc.sample_parcel_tube(present, _gen(3)))
    big = vc.sample_parcel_tube(present, _gen(3), cfg=vc.M4MaskConfig(parcel_mask_ratio=0.40))
    small = vc.sample_parcel_tube(present, _gen(3), cfg=vc.M4MaskConfig(parcel_mask_ratio=0.10))
    assert big.numel() > small.numel()


def test_electrode_tube_mask_whole_parcel_together() -> None:
    # Whole-parcel: all electrodes of a tubed parcel are masked together.
    parcel_per_electrode = torch.tensor([0, 0, 1, 1, 1, 2, 3])
    mask = vc.electrode_tube_mask(parcel_per_electrode, torch.tensor([1, 3]))
    assert mask.tolist() == [False, False, True, True, True, False, True]


def test_electrode_tube_mask_empty_tube() -> None:
    pe = torch.tensor([0, 1, 2])
    assert not vc.electrode_tube_mask(pe, torch.tensor([], dtype=torch.long)).any()


# ====================================================== FrontendEncoder (Stage 1)
def _rotate_at_slot(key_rope: torch.Tensor, slot: int, vec: torch.Tensor) -> torch.Tensor:
    """Rotate one head-dim vector by the encoder's RoPE at ``slot`` (T=1 gather)."""
    rope_slot = key_rope[:, slot : slot + 1, :]          # (2, 1, head_dim)
    return _apply_rope(vec.reshape(1, -1), rope_slot).reshape(-1)


def test_frontend_encoder_output_shape() -> None:
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=2).eval()
    out = fe(*_fake_bands(2, 5))
    assert out.shape == (2, 5, 38, 32)


def test_frontend_encoder_rope_table_spans_the_clip() -> None:
    # The shared clock is the HG stride; 16 slots (0..15) span the full 1 s clip.
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=1)
    assert fe.key_rope.shape == (2, 16, 8)               # (cos/sin, n_slots, head_dim)
    assert int(fe.tokenizer.time_slot.max()) == 15


def test_frontend_rope_relative_position_is_band_agnostic() -> None:
    # THE multirate rigor proof. RoPE's relative-position identity:
    # ⟨R_a q, R_b k⟩ depends ONLY on the slot difference (b−a). The slots are
    # physical-time indices on the ONE shared 62.5 ms (HG-stride) grid, so a
    # token-pair separated by the same real time gets the same rotated inner
    # product NO MATTER which band each token lives in. Verify a fixed q,k pair
    # at three different absolute slot positions but the same gap of 8:
    #   gap 8 = 8 × 62.5 ms = 0.5 s — e.g. slow tp0→tp1, beta tp0→tp4, HG tp0→tp8.
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=1)
    kr = fe.key_rope
    torch.manual_seed(0)
    q = torch.randn(8)
    k = torch.randn(8)

    def ip(a: int, b: int) -> float:
        return torch.dot(_rotate_at_slot(kr, a, q), _rotate_at_slot(kr, b, k)).item()

    base = ip(0, 8)
    assert abs(ip(4, 12) - base) < 1e-5    # same gap, shifted absolute position
    assert abs(ip(7, 15) - base) < 1e-5
    # A different gap gives a different inner product (rotation is real, not a no-op).
    assert abs(ip(0, 4) - base) > 1e-3


def test_frontend_same_physical_time_tokens_share_rope() -> None:
    # The cross-band consequence: every token landing on physical-time slot 8
    # (slow tp1, beta tp4, HG tp8) gets the IDENTICAL rope row — so the encoder
    # treats them as the same time, which is the whole point of the shared clock.
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=1)
    rope = fe.key_rope[:, fe.tokenizer.time_slot, :]     # (2, 38, head_dim)
    slot8 = (fe.tokenizer.time_slot == 8).nonzero().flatten()
    bands_at_slot8 = set(fe.tokenizer.band_id[slot8].tolist())
    assert bands_at_slot8 == {0, 1, 2}, "all three bands must reach physical slot 8"
    ref = rope[:, slot8[0]]
    for i in slot8[1:]:
        assert torch.allclose(rope[:, i], ref), "same physical time ⇒ same rotation"


def test_frontend_electrode_isolation_through_transformer() -> None:
    # Stage-1 contract survives the self-attention stack: permuting electrodes
    # permutes outputs (electrodes never attend to each other).
    torch.manual_seed(0)
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=2).eval()
    slow, beta, hg = _fake_bands(2, 6)
    perm = torch.tensor([5, 2, 0, 4, 1, 3])
    with torch.no_grad():
        base = fe(slow, beta, hg)
        permed = fe(slow[:, perm], beta[:, perm], hg[:, perm])
    assert torch.allclose(permed, base[:, perm], atol=1e-5)


def test_frontend_learned_freq_tag_is_a_used_parameter() -> None:
    # freq_pos="learned" ⇒ a trainable (6, d) table that actually moves the output.
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=1).eval()
    assert isinstance(fe.freq_embed, torch.nn.Parameter)
    assert fe.freq_embed.shape == (6, 32)
    slow, beta, hg = _fake_bands(1, 3)
    with torch.no_grad():
        base = fe(slow, beta, hg)
        fe.freq_embed.add_(1.0)                          # perturb the freq tag
        moved = fe(slow, beta, hg)
    assert not torch.allclose(moved, base)


def test_frontend_sinusoidal_freq_tag_is_a_buffer() -> None:
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=1, freq_pos="sinusoidal")
    assert not isinstance(fe.freq_embed, torch.nn.Parameter)
    assert "freq_embed" in dict(fe.named_buffers())
    assert fe.freq_embed.shape == (6, 32)


def test_frontend_construction_guards() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        vc.FrontendEncoder(d_model=30, n_heads=4, n_layers=1)
    with pytest.raises(ValueError, match="even head_dim"):
        vc.FrontendEncoder(d_model=12, n_heads=4, n_layers=1)  # head_dim 3 = odd
    with pytest.raises(ValueError, match="freq_pos"):
        vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=1, freq_pos="bogus")


# ------------------------------------------------ Stage 1 RAGGED frontend (Ben 06-18)
def _keep_visible(B: int, C: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random kept-electrode + visible-token masks for ragged-frontend tests.

    ``keep`` drops ~1/3 of electrodes (the student's tubed + padded set);
    ``visible`` is the per-electrode un-M2-masked set. Every KEPT electrode keeps
    ≥1 token (token 0 forced visible) — the slow-exempt guarantee in miniature —
    so no ragged row is all-pad. Dropped electrodes are left all-visible so the
    DENSE oracle never hits an empty-softmax NaN at rows we won't compare."""
    g = _gen(seed)
    keep = torch.rand(B, C, generator=g) > 0.33
    keep[:, 0] = True                                    # ≥1 kept electrode / sample
    visible = torch.ones(B, C, 38, dtype=torch.bool)
    for b in range(B):
        for c in range(C):
            if bool(keep[b, c]):
                m = vc.sample_m2_mask(_gen(seed * 131 + b * 17 + c))  # True = masked
                vis = ~m
                vis[0] = True                            # guarantee ≥1 visible
                visible[b, c] = vis
    return keep, visible


def test_frontend_ragged_equals_dense_on_visible() -> None:
    # TDD veracity: the ragged gather (production path) is OUTPUT-IDENTICAL to the
    # dense key-masked oracle on the kept-electrode visible tokens. A masked key
    # contributes 0 to the softmax and masked queries are never read, so dropping
    # them physically cannot change a visible token's features.
    torch.manual_seed(0)
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=3).eval()
    B, C = 3, 6
    slow, beta, hg = _fake_bands(B, C)
    keep, visible = _keep_visible(B, C, seed=1)
    with torch.no_grad():
        dense = fe(slow, beta, hg, key_mask=visible)            # (B,C,38,d)
        ragged = fe.forward_ragged(slow, beta, hg, keep, visible)
    sel = keep[:, :, None] & visible                            # kept-visible tokens
    assert torch.allclose(ragged[sel], dense[sel], atol=1e-5)


def test_frontend_ragged_drops_tubed_padded_and_masked_to_zero() -> None:
    # Structural: nothing dense is processed. Dropped electrodes (every token) and
    # kept-but-M2-masked tokens come back EXACTLY zero — they were never encoded.
    torch.manual_seed(0)
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=2).eval()
    B, C = 2, 7
    slow, beta, hg = _fake_bands(B, C)
    keep, visible = _keep_visible(B, C, seed=2)
    with torch.no_grad():
        ragged = fe.forward_ragged(slow, beta, hg, keep, visible)
    dropped = ~keep                                             # tubed/padded electrodes
    assert torch.count_nonzero(ragged[dropped]) == 0
    masked = keep[:, :, None] & ~visible                        # kept but M2-masked
    assert torch.count_nonzero(ragged[masked]) == 0
    kept_visible = keep[:, :, None] & visible
    assert torch.count_nonzero(ragged[kept_visible]) > 0        # the rest IS encoded


def test_frontend_ragged_is_per_electrode_never_dense_over_electrodes() -> None:
    # Ben 06-18: "the SA is per each electrode's 38 token sequence — never dense
    # over all electrodes — 38 tokens at a time only." Proof: encoding ONE kept
    # electrode alone reproduces its slice of the full ragged pass bit-for-bit, so
    # no electrode's output depends on any other (no cross-electrode K/V pathway).
    torch.manual_seed(0)
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=2).eval()
    B, C = 2, 5
    slow, beta, hg = _fake_bands(B, C)
    keep, visible = _keep_visible(B, C, seed=3)
    with torch.no_grad():
        full = fe.forward_ragged(slow, beta, hg, keep, visible)
        for b in range(B):
            for c in range(C):
                if not bool(keep[b, c]):
                    continue
                solo = fe.forward_ragged(
                    slow[b : b + 1, c : c + 1],
                    beta[b : b + 1, c : c + 1],
                    hg[b : b + 1, c : c + 1],
                    keep[b : b + 1, c : c + 1],
                    visible[b : b + 1, c : c + 1],
                )                                              # (1,1,38,d)
                assert torch.allclose(solo[0, 0], full[b, c], atol=1e-6)


def test_frontend_ragged_teacher_full_set_equals_dense() -> None:
    # The EMA teacher: full electrode set, all tokens visible, RAGGED not padded
    # (here C electrodes, no padding) — must equal the dense unmasked frontend.
    torch.manual_seed(0)
    fe = vc.FrontendEncoder(d_model=32, n_heads=4, n_layers=2).eval()
    B, C = 2, 5
    slow, beta, hg = _fake_bands(B, C)
    keep = torch.ones(B, C, dtype=torch.bool)
    visible = torch.ones(B, C, 38, dtype=torch.bool)
    with torch.no_grad():
        dense = fe(slow, beta, hg)                              # no mask = teacher
        ragged = fe.forward_ragged(slow, beta, hg, keep, visible)
    assert torch.allclose(ragged, dense, atol=1e-5)


# ===================================================== LatentEncoder (Stage 2)
def _fake_feats(B: int, C: int, d: int = 32) -> torch.Tensor:
    return torch.randn(B, C, 38, d)


def test_latent_output_shape_preserves_electrode_tokens() -> None:
    # No pooling bottleneck: electrode-token granularity survives the latent.
    lat = vc.LatentEncoder(d_model=32, n_heads=4, n_layers=2, n_parcels=74).eval()
    feats = _fake_feats(2, 7)
    pe = torch.randint(0, 74, (2, 7))
    out = lat(feats, pe)
    assert out.shape == (2, 7, 38, 32)


def test_latent_does_cross_electrode_mixing() -> None:
    # THE point of Stage 2 (and the opposite of the frontend's isolation):
    # perturbing one electrode's input MUST change other electrodes' outputs.
    torch.manual_seed(0)
    lat = vc.LatentEncoder(d_model=32, n_heads=4, n_layers=2, n_parcels=74).eval()
    feats = _fake_feats(1, 4)
    pe = torch.tensor([[0, 1, 2, 3]])
    with torch.no_grad():
        base = lat(feats, pe)
        feats2 = feats.clone()
        feats2[:, 2] += 5.0                       # perturb only electrode 2
        out = lat(feats2, pe)
    # every OTHER electrode's output changed → information crossed electrodes.
    for e in (0, 1, 3):
        assert not torch.allclose(out[:, e], base[:, e]), f"electrode {e} did not mix"


def test_latent_permutation_equivariance() -> None:
    # All-pairs SA is set-equivariant; RoPE-time is identical per electrode, so
    # permuting electrodes WITH their parcel ids permutes the outputs.
    torch.manual_seed(1)
    lat = vc.LatentEncoder(d_model=32, n_heads=4, n_layers=2, n_parcels=74).eval()
    feats = _fake_feats(1, 5)
    pe = torch.tensor([[10, 11, 12, 13, 14]])
    perm = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        base = lat(feats, pe)
        permed = lat(feats[:, perm], pe[:, perm])
    assert torch.allclose(permed, base[:, perm], atol=1e-5)


def test_latent_parcel_tag_is_the_only_added_pe() -> None:
    # Learned parcel embedding; perturbing it moves the output (it is used). And
    # the bridge adds it per electrode (two electrodes sharing a parcel id get
    # the SAME additive tag) — verified via the embedding table directly.
    lat = vc.LatentEncoder(d_model=32, n_heads=4, n_layers=1, n_parcels=74).eval()
    assert isinstance(lat.parcel_embed, torch.nn.Embedding)
    feats = _fake_feats(1, 3)
    pe = torch.tensor([[5, 5, 9]])               # electrodes 0,1 share parcel 5
    with torch.no_grad():
        base = lat(feats, pe)
        lat.parcel_embed.weight[5].add_(3.0)     # move parcel-5 tag only
        moved = lat(feats, pe)
    # electrodes 0 and 1 (parcel 5) are affected; the math added one shared tag.
    assert not torch.allclose(moved, base)


def test_latent_no_distance_or_membership_bias() -> None:
    # MNI banned + no same-parcel boost: the latent must hold NO coordinate
    # buffer and NO learned pairwise/parcel-membership attention bias. Only the
    # parcel-tag embedding, the RoPE clock, and plain SA blocks may carry params.
    lat = vc.LatentEncoder(d_model=16, n_heads=2, n_layers=1, n_parcels=8)
    buf_names = {n for n, _ in lat.named_buffers()}
    assert not any("mni" in n.lower() or "coord" in n.lower() or "dist" in n.lower()
                   for n in buf_names), f"spatial-distance buffer present: {buf_names}"
    param_names = {n for n, _ in lat.named_parameters()}
    assert not any("bias" in n.lower() and ("parcel" in n.lower() or "dist" in n.lower()
                   or "support" in n.lower()) for n in param_names), param_names


def test_latent_ragged_key_mask_isolates_padding() -> None:
    # Ragged contract: a padded (masked) electrode must NOT change the real
    # electrodes' outputs — no pad-to-max contamination. Compare a 3-electrode
    # clip against the same 3 + a 4th padded electrode marked invalid.
    torch.manual_seed(2)
    lat = vc.LatentEncoder(d_model=32, n_heads=4, n_layers=2, n_parcels=74).eval()
    feats3 = _fake_feats(1, 3)
    pe3 = torch.tensor([[1, 2, 3]])
    pad = torch.randn(1, 1, 38, 32) * 9.0        # loud junk in the pad slot
    feats4 = torch.cat([feats3, pad], dim=1)
    pe4 = torch.tensor([[1, 2, 3, 0]])
    mask4 = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        out3 = lat(feats3, pe3)                                  # no mask, all real
        out4 = lat(feats4, pe4, electrode_mask=mask4)            # 4th padded-out
    assert torch.allclose(out4[:, :3], out3, atol=1e-5), "padding leaked into reals"


# ------------------------------------------------- Stage 2 RAGGED latent (Ben 06-18)
def _latent_vis(B: int, C: int, seed: int) -> torch.Tensor:
    """A student SEE-set mask: ~1/3 electrodes fully tubed; the rest keep their
    un-M2-masked tokens (token 0 forced visible). ≥1 visible token per sample."""
    g = _gen(seed)
    untubed = torch.rand(B, C, generator=g) > 0.33
    untubed[:, 0] = True
    vis = torch.zeros(B, C, 38, dtype=torch.bool)
    for b in range(B):
        for c in range(C):
            if bool(untubed[b, c]):
                v = ~vc.sample_m2_mask(_gen(seed * 97 + b * 13 + c))
                v[0] = True
                vis[b, c] = v
    return vis


def test_latent_ragged_equals_dense_on_visible() -> None:
    # TDD veracity: the ragged cross-electrode gather is output-identical to the
    # dense all-pairs key-masked oracle on every visible token. Masked keys add 0
    # to the softmax and masked tokens are never read, so physically dropping them
    # leaves the visible tokens' cross-electrode features unchanged.
    torch.manual_seed(0)
    lat = vc.LatentEncoder(d_model=32, n_heads=4, n_layers=3, n_parcels=74).eval()
    B, C = 3, 6
    feats = _fake_feats(B, C)
    pe = torch.randint(0, 74, (B, C))
    vis = _latent_vis(B, C, seed=5)
    with torch.no_grad():
        dense = lat(feats, pe, token_mask=vis)
        ragged = lat.forward_ragged(feats, pe, vis)
    assert torch.allclose(ragged[vis], dense[vis], atol=1e-5)
    assert torch.count_nonzero(ragged[~vis]) == 0          # non-visible never built


def test_latent_ragged_padding_independence() -> None:
    # Ragged contract on the gather path: appending a fully-masked (padded/tubed)
    # electrode cannot change the visible outputs — the gather skips it, so it is
    # never built into the all-pairs set (no pad-to-max contamination).
    torch.manual_seed(2)
    lat = vc.LatentEncoder(d_model=32, n_heads=4, n_layers=2, n_parcels=74).eval()
    feats3 = _fake_feats(1, 3)
    pe3 = torch.tensor([[1, 2, 3]])
    vis3 = torch.ones(1, 3, 38, dtype=torch.bool)
    pad = torch.randn(1, 1, 38, 32) * 9.0
    feats4 = torch.cat([feats3, pad], dim=1)
    pe4 = torch.tensor([[1, 2, 3, 0]])
    vis4 = torch.cat([vis3, torch.zeros(1, 1, 38, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        out3 = lat.forward_ragged(feats3, pe3, vis3)
        out4 = lat.forward_ragged(feats4, pe4, vis4)
    assert torch.allclose(out4[:, :3], out3, atol=1e-5)


def test_latent_construction_guards() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        vc.LatentEncoder(d_model=30, n_heads=4, n_layers=1, n_parcels=8)
    with pytest.raises(ValueError, match="even head_dim"):
        vc.LatentEncoder(d_model=12, n_heads=4, n_layers=1, n_parcels=8)


# ----------------------------------------------- shared token metadata helper
def test_token_metadata_single_source_matches_tokenizer() -> None:
    # The latent and the tokenizer must agree on time_slot (one source).
    tok = vc.ThreeBandTokenizer(d_model=8)
    _, _, time_slot = vc.token_metadata()
    assert torch.equal(time_slot, tok.time_slot)
    lat = vc.LatentEncoder(d_model=16, n_heads=2, n_layers=1, n_parcels=8)
    assert torch.equal(lat.time_slot, tok.time_slot)


def test_band_slot_mults_are_8_2_1() -> None:
    assert vc.band_slot_mults() == [8, 2, 1]


# ============================================ M4 teacher target (electrode-mean)
def test_parcel_electrode_mean_is_the_mean_over_a_parcels_electrodes() -> None:
    # parcel 1 has electrodes {0,2}; target = their elementwise mean (keep 38×d).
    feats = torch.zeros(3, 38, 4)
    feats[0] = 1.0
    feats[2] = 3.0          # parcel-1 electrodes 0,2 → mean 2.0
    feats[1] = 99.0         # parcel 0 (ignored)
    pe = torch.tensor([1, 0, 1])
    out = vc.parcel_electrode_mean(feats, pe, torch.tensor([1]))
    assert out.shape == (1, 38, 4)
    assert torch.allclose(out[0], torch.full((38, 4), 2.0))


def test_parcel_electrode_mean_single_electrode_parcel() -> None:
    feats = torch.randn(4, 38, 6)
    pe = torch.tensor([5, 6, 7, 8])           # all singletons
    out = vc.parcel_electrode_mean(feats, pe, torch.tensor([7]))
    assert torch.allclose(out[0], feats[2])   # mean of one = itself


def test_parcel_electrode_mean_drops_std_not_just_smooths() -> None:
    # Two electrodes ±5 around a mean of 10: target is exactly 10 everywhere —
    # the spread (std) is DROPPED, only the mean survives (converged M4 lock).
    feats = torch.empty(2, 38, 3)
    feats[0] = 15.0
    feats[1] = 5.0
    out = vc.parcel_electrode_mean(feats, torch.tensor([2, 2]), torch.tensor([2]))
    assert torch.allclose(out[0], torch.full((38, 3), 10.0))


def test_parcel_electrode_mean_multiple_targets() -> None:
    feats = torch.randn(6, 38, 8)
    pe = torch.tensor([0, 0, 1, 1, 1, 2])
    out = vc.parcel_electrode_mean(feats, pe, torch.tensor([0, 2]))
    assert out.shape == (2, 38, 8)
    assert torch.allclose(out[0], feats[:2].mean(0))    # parcel 0 = mean{0,1}
    assert torch.allclose(out[1], feats[5])             # parcel 2 = singleton


# ==================================================== M4Predictor (parcel JEPA)
def _ctx(B: int, N: int, d: int = 32) -> torch.Tensor:
    return torch.randn(B, N, d)


def test_m4_predictor_output_shape() -> None:
    pred = vc.M4Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2,
                          n_parcels=74).eval()
    ctx = _ctx(2, 10)
    ctx_slot = torch.randint(0, 16, (10,))
    qp = torch.tensor([[3, 7], [1, 5]])
    out = pred(ctx, ctx_slot, qp)
    assert out.shape == (2, 2, 38, 32)        # (B, P, 38, d_model teacher dim)


def test_m4_predictor_predicts_from_context() -> None:
    # M4 SEES the visible context: perturbing it must move the prediction.
    torch.manual_seed(0)
    pred = vc.M4Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2,
                          n_parcels=74).eval()
    ctx = _ctx(1, 8)
    ctx_slot = torch.randint(0, 16, (8,))
    qp = torch.tensor([[4]])
    with torch.no_grad():
        base = pred(ctx, ctx_slot, qp)
        ctx2 = ctx.clone()
        ctx2[:, 3] += 5.0
        out = pred(ctx2, ctx_slot, qp)
    assert not torch.allclose(out, base)      # context informs the prediction


def test_m4_predictor_parcel_tag_addresses_the_target() -> None:
    # Different tubed parcel ⇒ different query tag ⇒ different prediction.
    torch.manual_seed(1)
    pred = vc.M4Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=1,
                          n_parcels=74).eval()
    ctx = _ctx(1, 6)
    ctx_slot = torch.randint(0, 16, (6,))
    with torch.no_grad():
        p3 = pred(ctx, ctx_slot, torch.tensor([[3]]))
        p7 = pred(ctx, ctx_slot, torch.tensor([[7]]))
    assert not torch.allclose(p3, p7)


def test_m4_predictor_freq_time_cells_are_distinguished() -> None:
    # The 38 predicted cells of one parcel must NOT collapse to one value —
    # the freq-tag + RoPE-time give each cell a distinct query.
    torch.manual_seed(2)
    pred = vc.M4Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2,
                          n_parcels=74).eval()
    ctx = _ctx(1, 8)
    ctx_slot = torch.randint(0, 16, (8,))
    with torch.no_grad():
        out = pred(ctx, ctx_slot, torch.tensor([[5]]))[0, 0]   # (38, d)
    spread = out.std(dim=0).mean().item()
    assert spread > 1e-4, "all 38 cells collapsed — freq/time tag not addressing"


def test_m4_predictor_ragged_context_key_mask_isolates_padding() -> None:
    # A padded (masked) context token must not change the prediction.
    torch.manual_seed(3)
    pred = vc.M4Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2,
                          n_parcels=74).eval()
    ctx = _ctx(1, 5)
    ctx_slot = torch.randint(0, 16, (5,))
    qp = torch.tensor([[2]])
    pad = torch.randn(1, 1, 32) * 9.0
    ctx_pad = torch.cat([ctx, pad], dim=1)
    slot_pad = torch.cat([ctx_slot, torch.tensor([7])])
    mask = torch.tensor([[True, True, True, True, True, False]])
    with torch.no_grad():
        base = pred(ctx, ctx_slot, qp)                              # all real, no mask
        out = pred(ctx_pad, slot_pad, qp, key_mask=mask)           # 6th padded out
    assert torch.allclose(out, base, atol=1e-5)


def test_m4_predictor_independent_learned_params() -> None:
    pred = vc.M4Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=1,
                          n_parcels=74)
    assert isinstance(pred.mask_token, torch.nn.Parameter)
    assert isinstance(pred.parcel_embed, torch.nn.Embedding)
    assert isinstance(pred.freq_embed, torch.nn.Parameter)        # learned default
    assert pred.freq_embed.shape == (6, 16)
    assert pred.head.out_features == 32                           # projects to teacher d


def test_m4_predictor_construction_guards() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        vc.M4Predictor(d_model=32, pred_dim=30, n_heads=4, n_layers=1, n_parcels=8)
    with pytest.raises(ValueError, match="even head_dim"):
        vc.M4Predictor(d_model=32, pred_dim=12, n_heads=4, n_layers=1, n_parcels=8)
    with pytest.raises(ValueError, match="freq_pos"):
        vc.M4Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=1,
                       n_parcels=8, freq_pos="bogus")


# ============================================ M2Predictor (frontend per-elec JEPA)
def _m2_inputs(Bp: int, N: int, M: int, d: int = 32):
    ctx = torch.randn(Bp, N, d)
    ctx_slot = torch.randint(0, 16, (Bp, N))
    qfreq = torch.randint(0, 6, (Bp, M))
    qslot = torch.randint(0, 16, (Bp, M))
    return ctx, ctx_slot, qfreq, qslot


def test_m2_predictor_output_shape() -> None:
    pred = vc.M2Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2).eval()
    ctx, cs, qf, qs = _m2_inputs(5, 20, 13)         # 5 electrodes, 20 vis, 13 masked
    out = pred(ctx, cs, qf, qs)
    assert out.shape == (5, 13, 32)                 # (B'=electrodes, M, teacher d)


def test_m2_predictor_is_electrode_isolated() -> None:
    # THE M2 contract: each electrode rides the batch dim, so perturbing one
    # electrode's inputs CANNOT change another electrode's predictions.
    torch.manual_seed(0)
    pred = vc.M2Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2).eval()
    ctx, cs, qf, qs = _m2_inputs(3, 10, 8)
    with torch.no_grad():
        base = pred(ctx, cs, qf, qs)
        ctx2 = ctx.clone()
        ctx2[1] += 5.0                              # perturb only electrode 1
        out = pred(ctx2, cs, qf, qs)
    assert not torch.allclose(out[1], base[1])      # electrode 1 changed
    assert torch.allclose(out[0], base[0], atol=1e-6)   # 0 and 2 untouched (isolated)
    assert torch.allclose(out[2], base[2], atol=1e-6)


def test_m2_predictor_predicts_from_own_visible_context() -> None:
    torch.manual_seed(1)
    pred = vc.M2Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2).eval()
    ctx, cs, qf, qs = _m2_inputs(2, 12, 8)
    with torch.no_grad():
        base = pred(ctx, cs, qf, qs)
        ctx2 = ctx.clone()
        ctx2[:, 4] += 4.0
        out = pred(ctx2, cs, qf, qs)
    assert not torch.allclose(out, base)


def test_m2_predictor_freq_tag_addresses_masked_position() -> None:
    # Two queries differing ONLY in freq-id must get different predictions.
    torch.manual_seed(2)
    pred = vc.M2Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=1).eval()
    ctx = torch.randn(1, 8, 32)
    cs = torch.randint(0, 16, (1, 8))
    qs = torch.tensor([[4]])
    with torch.no_grad():
        a = pred(ctx, cs, torch.tensor([[1]]), qs)
        b = pred(ctx, cs, torch.tensor([[4]]), qs)
    assert not torch.allclose(a, b)


def test_m2_predictor_rope_time_is_active() -> None:
    # Same query freq, different time-slot ⇒ different prediction (RoPE-time on).
    torch.manual_seed(3)
    pred = vc.M2Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2).eval()
    ctx = torch.randn(1, 8, 32)
    cs = torch.randint(0, 16, (1, 8))
    qf = torch.tensor([[2]])
    with torch.no_grad():
        a = pred(ctx, cs, qf, torch.tensor([[0]]))
        b = pred(ctx, cs, qf, torch.tensor([[15]]))
    assert not torch.allclose(a, b)


def test_m2_predictor_ragged_masks_isolate_padding() -> None:
    # A padded context token (ctx_mask False) must not change the predictions.
    torch.manual_seed(4)
    pred = vc.M2Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=2).eval()
    ctx = torch.randn(1, 6, 32)
    cs = torch.randint(0, 16, (1, 6))
    qf = torch.tensor([[1, 3]])
    qs = torch.tensor([[5, 9]])
    pad = torch.randn(1, 1, 32) * 9.0
    ctx_pad = torch.cat([ctx, pad], dim=1)
    cs_pad = torch.cat([cs, torch.tensor([[7]])], dim=1)
    cmask = torch.tensor([[True, True, True, True, True, True, False]])
    with torch.no_grad():
        base = pred(ctx, cs, qf, qs)
        out = pred(ctx_pad, cs_pad, qf, qs, ctx_mask=cmask)
    assert torch.allclose(out, base, atol=1e-5)


def test_m2_predictor_has_no_parcel_embed() -> None:
    # M2 query is freq+time only — no parcel tag exists at the frontend stage.
    pred = vc.M2Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=1)
    names = {n for n, _ in pred.named_parameters()}
    assert not any("parcel" in n.lower() for n in names), names
    assert isinstance(pred.mask_token, torch.nn.Parameter)
    assert isinstance(pred.freq_embed, torch.nn.Parameter)


def test_m2_predictor_construction_guards() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        vc.M2Predictor(d_model=32, pred_dim=30, n_heads=4, n_layers=1)
    with pytest.raises(ValueError, match="even head_dim"):
        vc.M2Predictor(d_model=32, pred_dim=12, n_heads=4, n_layers=1)
    with pytest.raises(ValueError, match="freq_pos"):
        vc.M2Predictor(d_model=32, pred_dim=16, n_heads=2, n_layers=1, freq_pos="x")


# ==================================================== ParcelReadout (Stage 3)
def test_parcel_readout_output_shape() -> None:
    ro = vc.ParcelReadout(d_model=32, n_parcels=10, n_classes=9, n_heads=4).eval()
    feats = _fake_feats(2, 6)
    pe = torch.randint(0, 10, (2, 6))
    assert ro(feats, pe).shape == (2, 9)


def test_parcel_readout_empty_parcel_is_zero_slot() -> None:
    # A parcel with no electrodes pools to an exactly-zero slot (missing token).
    ro = vc.ParcelReadout(d_model=32, n_parcels=8, n_classes=9, n_heads=4).eval()
    feats = _fake_feats(1, 4)
    pe = torch.tensor([[1, 2, 1, 3]])           # parcels {1,2,3} present; 0,4..7 empty
    with torch.no_grad():
        pooled = ro.pool_parcels(feats, pe)     # (1, 8, 32)
    for empty in (0, 4, 5, 6, 7):
        assert torch.count_nonzero(pooled[0, empty]) == 0, f"parcel {empty} not zero"
    assert torch.count_nonzero(pooled[0, 1]) > 0    # present parcel is non-zero


def test_parcel_readout_hard_grouping_no_cross_parcel_leak() -> None:
    # Parcel p's slot pools ONLY parcel-p tokens: perturbing a parcel-q electrode
    # must not change parcel-p's slot (hard one-hot grouping, no soft bias).
    torch.manual_seed(0)
    ro = vc.ParcelReadout(d_model=32, n_parcels=8, n_classes=9, n_heads=4).eval()
    feats = _fake_feats(1, 4)
    pe = torch.tensor([[1, 2, 1, 2]])           # elec 0,2 → parcel 1; 1,3 → parcel 2
    with torch.no_grad():
        base = ro.pool_parcels(feats, pe)
        feats2 = feats.clone()
        feats2[:, 1] += 9.0                      # perturb a parcel-2 electrode
        out = ro.pool_parcels(feats2, pe)
    assert torch.allclose(out[0, 1], base[0, 1], atol=1e-6)   # parcel-1 slot untouched
    assert not torch.allclose(out[0, 2], base[0, 2])          # parcel-2 slot moved


def test_parcel_readout_is_montage_invariant() -> None:
    # Permuting electrodes (with their parcel ids) leaves the K-parcel rep AND
    # the logits unchanged — the readout keys off parcel membership, not montage.
    torch.manual_seed(1)
    ro = vc.ParcelReadout(d_model=32, n_parcels=10, n_classes=9, n_heads=4).eval()
    feats = _fake_feats(1, 6)
    pe = torch.tensor([[3, 3, 5, 1, 5, 1]])
    perm = torch.tensor([4, 0, 5, 2, 1, 3])
    with torch.no_grad():
        base = ro(feats, pe)
        permed = ro(feats[:, perm], pe[:, perm])
    assert torch.allclose(permed, base, atol=1e-5)


def test_parcel_readout_no_support_bias_param() -> None:
    # No support / distance / anatomy bias anywhere — only seed, pool, head.
    ro = vc.ParcelReadout(d_model=16, n_parcels=8, n_classes=4, n_heads=2)
    names = {n for n, _ in ro.named_parameters()}
    assert not any("support" in n.lower() or "bias_anat" in n.lower()
                   or "dist" in n.lower() for n in names), names


# ============================================ V14ConvergedSSL (integration)
def _ssl_model(**kw):
    base = dict(d_model=32, n_parcels=12, n_heads=4, frontend_layers=2,
                latent_layers=2, m2_pred_dim=16, m2_pred_layers=2,
                m4_pred_dim=16, m4_pred_layers=2)
    base.update(kw)
    return vc.V14ConvergedSSL(**base)


def _ssl_batch(B: int = 2, C: int = 6, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    slow = torch.randn(B, C, 2, 6, 5, generator=g)
    beta = torch.randn(B, C, 6, 17, generator=g)
    hg = torch.randn(B, C, 9, 33, generator=g)
    # parcels chosen so each clip has several parcels (M4 has a target + context)
    pe = torch.randint(0, 5, (B, C), generator=g)
    emask = torch.ones(B, C, dtype=torch.bool)
    masks = vc.sample_ssl_masks(pe, emask, g)
    return slow, beta, hg, pe, emask, masks


def _run(model, batch):
    slow, beta, hg, pe, emask, m = batch
    return model(slow, beta, hg, pe, emask, m["m2_mask"], m["tube_mask"],
                 m["tubed_parcels"], m["tubed_parcel_mask"])


def test_forward_ragged_equals_dense() -> None:
    # Stage 5: the production forward (fully ragged: teacher/student/latent/M2/M4)
    # is output-identical to the dense oracle _forward_dense across all 3 losses.
    torch.manual_seed(0)
    model = _ssl_model().eval()
    slow, beta, hg, pe, emask, m = _ssl_batch(B=3, C=8, seed=2)
    args = (slow, beta, hg, pe, emask, m["m2_mask"], m["tube_mask"],
            m["tubed_parcels"], m["tubed_parcel_mask"])
    with torch.no_grad():
        ragged = model(*args)
        dense = model._forward_dense(*args)
    # equivalence across ALL emitted scalars, including the per-band diagnostics
    # (same valid cell set in either layout ⇒ same per-band means).
    for k in ("loss", "l_m2", "l_m4", "l_m2_beta", "l_m2_hg",
              "l_m4_slow", "l_m4_beta", "l_m4_hg"):
        assert torch.allclose(ragged[k], dense[k], atol=1e-5), k
    assert float(dense["l_m2"]) > 0.0 and float(dense["l_m4"]) > 0.0   # non-trivial


def test_forward_ragged_equals_dense_with_padding() -> None:
    # With padded electrodes (emask False), the ragged forward still matches the
    # dense oracle — padding is gathered away, never built into any stage.
    torch.manual_seed(0)
    model = _ssl_model().eval()
    slow, beta, hg, pe, emask, _ = _ssl_batch(B=2, C=6, seed=5)
    emask = emask.clone()
    emask[:, -1] = False                                   # last electrode padded
    g = torch.Generator().manual_seed(11)
    m = vc.sample_ssl_masks(pe, emask, g)                  # masks respect emask
    args = (slow, beta, hg, pe, emask, m["m2_mask"], m["tube_mask"],
            m["tubed_parcels"], m["tubed_parcel_mask"])
    with torch.no_grad():
        ragged = model(*args)
        dense = model._forward_dense(*args)
    for k in ("loss", "l_m2", "l_m4"):
        assert torch.allclose(ragged[k], dense[k], atol=1e-5), k


def test_m2_loss_ragged_equals_dense() -> None:
    # Stage 3 (Ben 06-18): the ragged M2 gathers only un-tubed target electrodes,
    # each electrode's visible context + masked queries — output-identical to the
    # dense oracle that runs all B*C rows with 38-token key-masked context.
    torch.manual_seed(0)
    model = _ssl_model().eval()
    slow, beta, hg, pe, emask, m = _ssl_batch(B=3, C=7, seed=4)
    with torch.no_grad():
        t_f = model.teacher_frontend(slow, beta, hg).detach()
        student_vis = ~m["m2_mask"]
        s_f = model.student_frontend(slow, beta, hg, key_mask=student_vis)
        dense, _ = model._m2_loss(
            s_f, t_f, m["m2_mask"], student_vis, emask, m["tube_mask"])
        ragged, _ = model._m2_loss_ragged(
            s_f, t_f, m["m2_mask"], student_vis, emask, m["tube_mask"])
    assert float(dense) > 0.0                      # the batch carries real targets
    assert torch.allclose(ragged, dense, atol=1e-5)


def test_m2_loss_ragged_ignores_padded_electrodes() -> None:
    # Appending a padded electrode (emask False, loud junk) must not move the
    # ragged M2 loss — the electrode gather drops it; it is never built.
    torch.manual_seed(0)
    model = _ssl_model().eval()
    slow, beta, hg, pe, emask, m = _ssl_batch(B=2, C=6, seed=7)
    with torch.no_grad():
        t_f = model.teacher_frontend(slow, beta, hg).detach()
        student_vis = ~m["m2_mask"]
        s_f = model.student_frontend(slow, beta, hg, key_mask=student_vis)
        base, _ = model._m2_loss_ragged(
            s_f, t_f, m["m2_mask"], student_vis, emask, m["tube_mask"])
        # pad with a junk electrode marked unreal + (defensively) M2-masked nowhere
        B, C = emask.shape
        s_f2 = torch.cat([s_f, torch.randn(B, 1, 38, s_f.shape[-1]) * 9.0], dim=1)
        t_f2 = torch.cat([t_f, torch.randn(B, 1, 38, t_f.shape[-1]) * 9.0], dim=1)
        m2_2 = torch.cat([m["m2_mask"], torch.zeros(B, 1, 38, dtype=torch.bool)], dim=1)
        vis2 = ~m2_2
        emask2 = torch.cat([emask, torch.zeros(B, 1, dtype=torch.bool)], dim=1)
        tube2 = torch.cat([m["tube_mask"], torch.zeros(B, 1, dtype=torch.bool)], dim=1)
        padded, _ = model._m2_loss_ragged(s_f2, t_f2, m2_2, vis2, emask2, tube2)
    assert torch.allclose(padded, base, atol=1e-6)


def _m4_pieces(model, batch):
    """Compute (s_l, t_f, latent_vis) the way forward does, for M4-loss tests."""
    slow, beta, hg, pe, emask, m = batch
    t_f = model.teacher_frontend(slow, beta, hg).detach()
    student_vis = ~m["m2_mask"]
    s_f = model.student_frontend(slow, beta, hg, key_mask=student_vis)
    latent_vis = emask[:, :, None] & (~m["tube_mask"])[:, :, None] & student_vis
    s_l = model.latent(s_f, pe, token_mask=latent_vis)
    return s_l, t_f, latent_vis


def test_m4_loss_ragged_equals_dense() -> None:
    # Stage 4 (Ben 06-18): the ragged M4 physically gathers the visible un-tubed
    # latent SEE-set across all electrodes, predicts only the tubed parcels —
    # output-identical to the dense oracle that key-masks the full C*38 context.
    torch.manual_seed(0)
    model = _ssl_model().eval()
    batch = _ssl_batch(B=3, C=8, seed=2)
    _, _, _, pe, emask, m = batch
    with torch.no_grad():
        s_l, t_f, latent_vis = _m4_pieces(model, batch)
        dense, _ = model._m4_loss(
            s_l, t_f, pe, emask, latent_vis,
            m["tubed_parcels"], m["tubed_parcel_mask"])
        ragged, _ = model._m4_loss_ragged(
            s_l, t_f, pe, emask, latent_vis,
            m["tubed_parcels"], m["tubed_parcel_mask"])
    assert float(dense) > 0.0                      # the batch tubes real parcels
    assert torch.allclose(ragged, dense, atol=1e-5)


def test_m4_loss_ragged_ignores_padded_electrodes() -> None:
    # Appending a padded electrode (emask False, latent_vis False, junk feats)
    # cannot move the ragged M4 loss — the context gather skips it and pe=-1
    # excludes it from every tubed parcel's electrode-mean target.
    torch.manual_seed(0)
    model = _ssl_model().eval()
    batch = _ssl_batch(B=2, C=6, seed=3)
    _, _, _, pe, emask, m = batch
    with torch.no_grad():
        s_l, t_f, latent_vis = _m4_pieces(model, batch)
        base, _ = model._m4_loss_ragged(
            s_l, t_f, pe, emask, latent_vis,
            m["tubed_parcels"], m["tubed_parcel_mask"])
        B, C, S, d = s_l.shape
        s_l2 = torch.cat([s_l, torch.randn(B, 1, S, d) * 9.0], dim=1)
        t_f2 = torch.cat([t_f, torch.randn(B, 1, S, t_f.shape[-1]) * 9.0], dim=1)
        pe2 = torch.cat([pe, torch.zeros(B, 1, dtype=torch.long)], dim=1)
        emask2 = torch.cat([emask, torch.zeros(B, 1, dtype=torch.bool)], dim=1)
        lv2 = torch.cat([latent_vis, torch.zeros(B, 1, S, dtype=torch.bool)], dim=1)
        padded, _ = model._m4_loss_ragged(
            s_l2, t_f2, pe2, emask2, lv2,
            m["tubed_parcels"], m["tubed_parcel_mask"])
    assert torch.allclose(padded, base, atol=1e-6)


def test_ssl_forward_returns_finite_nonneg_losses() -> None:
    model = _ssl_model().eval()
    out = _run(model, _ssl_batch())
    for k in ("loss", "l_m2", "l_m4"):
        assert torch.isfinite(out[k]).all(), f"{k} not finite"
        assert out[k].item() >= 0.0
    assert out["loss"].item() > 0.0       # both heads active on this batch


def test_ssl_teacher_is_frontend_only() -> None:
    # The teacher halts after the frontend — no latent, no predictors, no parcel
    # embed. Its param count equals a standalone FrontendEncoder's.
    model = _ssl_model()
    assert not hasattr(model.teacher_frontend, "latent")
    fe = vc.FrontendEncoder(32, 4, 2)
    n_teacher = sum(p.numel() for p in model.teacher_frontend.parameters())
    n_fe = sum(p.numel() for p in fe.parameters())
    assert n_teacher == n_fe


def test_ssl_gradient_flows_to_student_not_teacher() -> None:
    model = _ssl_model().train()
    out = _run(model, _ssl_batch())
    out["loss"].backward()
    s_grad = [p.grad is not None for p in model.student_frontend.parameters()]
    assert any(s_grad), "student frontend got no gradient"
    assert all(p.grad is None for p in model.teacher_frontend.parameters()), \
        "teacher must be grad-free (EMA only)"
    # both predictors and the latent train too
    assert any(p.grad is not None for p in model.m2_predictor.parameters())
    assert any(p.grad is not None for p in model.m4_predictor.parameters())
    assert any(p.grad is not None for p in model.latent.parameters())


def test_ssl_ema_update_moves_teacher_toward_student() -> None:
    model = _ssl_model()
    # perturb the student so they differ
    with torch.no_grad():
        for p in model.student_frontend.parameters():
            p.add_(torch.randn_like(p))
    before = [t.clone() for t in model.teacher_frontend.parameters()]
    model.update_teacher(tau=0.9)
    moved = False
    for t0, t1, s in zip(before, model.teacher_frontend.parameters(),
                         model.student_frontend.parameters()):
        # τ=0.9 EMA: teacher must move 10% toward student
        expected = 0.9 * t0 + 0.1 * s
        assert torch.allclose(t1, expected, atol=1e-6)
        if not torch.allclose(t1, t0):
            moved = True
    assert moved


def test_ssl_lambda_gates_each_head() -> None:
    batch = _ssl_batch()
    m4_only = _ssl_model(lambda_m2=0.0, lambda_m4=1.0).eval()
    o = _run(m4_only, batch)
    assert torch.allclose(o["loss"], o["l_m4"])           # M2 gated out
    m2_only = _ssl_model(lambda_m2=1.0, lambda_m4=0.0).eval()
    o2 = _run(m2_only, batch)
    assert torch.allclose(o2["loss"], o2["l_m2"])         # M4 gated out


def test_ssl_no_tube_zeroes_m4_no_mask_zeroes_m2() -> None:
    slow, beta, hg, pe, emask, m = _ssl_batch()
    # no tubed parcels → M4 has nothing to predict
    no_tube = dict(m)
    no_tube["tube_mask"] = torch.zeros_like(m["tube_mask"])
    no_tube["tubed_parcel_mask"] = torch.zeros_like(m["tubed_parcel_mask"])
    model = _ssl_model().eval()
    o = model(slow, beta, hg, pe, emask, m["m2_mask"], no_tube["tube_mask"],
              m["tubed_parcels"], no_tube["tubed_parcel_mask"])
    assert o["l_m4"].item() == 0.0
    # no M2 mask → M2 has nothing to predict
    o2 = model(slow, beta, hg, pe, emask, torch.zeros_like(m["m2_mask"]),
               m["tube_mask"], m["tubed_parcels"], m["tubed_parcel_mask"])
    assert o2["l_m2"].item() == 0.0


def test_ssl_overfits_one_batch() -> None:
    # The whole pipeline trains: a few steps on ONE batch drive the loss down
    # (CLAUDE.md overfit-one-batch sanity — proves grads connect end to end).
    torch.manual_seed(0)
    model = _ssl_model().train()
    batch = _ssl_batch(seed=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = _run(model, batch)["loss"].item()
    for _ in range(40):
        opt.zero_grad()
        loss = _run(model, batch)["loss"]
        loss.backward()
        opt.step()
        model.update_teacher(tau=0.99)
    last = _run(model, batch)["loss"].item()
    assert last < 0.6 * first, f"no overfit: {first:.4f} -> {last:.4f}"


# ----------------------------------------------------- sample_ssl_masks helper
def test_sample_ssl_masks_shapes_and_invariants() -> None:
    g = torch.Generator().manual_seed(3)
    pe = torch.randint(0, 6, (2, 8), generator=g)
    emask = torch.ones(2, 8, dtype=torch.bool)
    m = vc.sample_ssl_masks(pe, emask, g)
    assert m["m2_mask"].shape == (2, 8, 38)
    assert m["tube_mask"].shape == (2, 8)
    # slow tokens (0..5) are never M2-masked (exempt)
    assert not m["m2_mask"][:, :, :6].any()
    # tubed electrodes carry no M2 target
    assert not m["m2_mask"][m["tube_mask"]].any()
    # tubed parcels are present in the montage
    for b in range(2):
        for p, ok in zip(m["tubed_parcels"][b], m["tubed_parcel_mask"][b]):
            if ok:
                assert (pe[b] == p).any()


def test_sample_ssl_masks_padded_electrodes_get_no_mask() -> None:
    g = torch.Generator().manual_seed(4)
    pe = torch.randint(0, 6, (1, 6), generator=g)
    emask = torch.tensor([[True, True, True, True, False, False]])
    m = vc.sample_ssl_masks(pe, emask, g)
    assert not m["m2_mask"][0, 4:].any()      # padded electrodes never masked
    assert not m["tube_mask"][0, 4:].any()


# ============================ eval feature taps + probe pool ==================
# The converged success criterion: a linear probe on the post-LATENT tap must
# beat one on the post-FRONTEND tap. These check the two taps exist, differ in
# the right way (frontend isolated, latent mixed), and that the parameter-free
# probe pool is an honest masked mean.
def test_encode_frontend_is_electrode_isolated() -> None:
    m = _ssl_model().eval()
    slow, beta, hg, _, _, _ = _ssl_batch()
    f0 = m.encode_frontend(slow, beta, hg)
    assert f0.shape == (2, 6, 38, 32)
    slow2 = slow.clone()
    slow2[:, 0] += 5.0                                   # perturb electrode 0 only
    f1 = m.encode_frontend(slow2, beta, hg)
    assert not torch.allclose(f0[:, 0], f1[:, 0])        # its own output changed
    assert torch.allclose(f0[:, 1:], f1[:, 1:])          # others untouched (isolated)


def test_encode_latent_mixes_across_electrodes() -> None:
    m = _ssl_model().eval()
    slow, beta, hg, pe, emask, _ = _ssl_batch()
    l0 = m.encode_latent(slow, beta, hg, pe, electrode_mask=emask)
    assert l0.shape == (2, 6, 38, 32)
    slow2 = slow.clone()
    slow2[:, 0] += 5.0                                   # perturb electrode 0 only
    l1 = m.encode_latent(slow2, beta, hg, pe, electrode_mask=emask)
    # the latent's global SA propagates electrode 0's change to its parcel peers
    assert not torch.allclose(l0[:, 1:], l1[:, 1:])


def test_probe_taps_differ_frontend_vs_latent() -> None:
    m = _ssl_model().eval()
    slow, beta, hg, pe, emask, _ = _ssl_batch()
    pf = vc.probe_pool(m.encode_frontend(slow, beta, hg), emask)
    pl = vc.probe_pool(m.encode_latent(slow, beta, hg, pe, electrode_mask=emask), emask)
    assert pf.shape == (2, 32) and pl.shape == (2, 32)
    assert not torch.allclose(pf, pl), "the two probe taps must carry distinct signal"


def test_probe_pool_masked_mean_excludes_padding() -> None:
    feats = torch.randn(1, 4, 38, 8)
    emask = torch.tensor([[True, True, False, False]])
    got = vc.probe_pool(feats, emask)
    want = feats[:, :2].mean(dim=(1, 2))                 # mean over the 2 real elecs
    assert torch.allclose(got, want, atol=1e-6)
    # no mask ⇒ plain mean over all electrodes+tokens
    assert torch.allclose(vc.probe_pool(feats), feats.mean(dim=(1, 2)), atol=1e-6)


# ============================ M4 heteroscedastic down-weight (Ben 2026-06-18) ===
def test_downweight_dof_formula() -> None:
    # w = min(1, (n-1)/(n_ref-1)) at α=1 — the B37 downweight_dof ported to the
    # converged electrode-MEAN target (n = real electrodes per parcel).
    n = torch.tensor([1.0, 2.0, 3.0, 6.0, 11.0, 20.0])
    w = vc._downweight_dof(n, n_ref=11.0, alpha=1.0)
    assert torch.allclose(w, torch.tensor([0.0, 0.1, 0.2, 0.5, 1.0, 1.0]), atol=1e-6)
    assert not w.requires_grad        # a fixed, detached weight (never learned)


def test_downweight_dof_alpha_sharpens() -> None:
    # α>1 is the risk-averse overlay: it pushes sub-n_ref parcels further down.
    n = torch.tensor([6.0])
    assert vc._downweight_dof(n, n_ref=11.0, alpha=1.0).item() == pytest.approx(0.5)
    assert vc._downweight_dof(n, n_ref=11.0, alpha=2.0).item() == pytest.approx(0.25)


def test_m4_precision_defaults_locked() -> None:
    m = _ssl_model()
    assert m.m4_precision_weight is True       # ON by default (Ben 2026-06-18)
    assert m.m4_precision_alpha == 1.0
    assert m.m4_precision_n_ref == 11.0


def test_m4_precision_downweights_loss_vs_off() -> None:
    # Same weights, only the M4 precision toggle differs. Every parcel weight ∈
    # [0,1] ⇒ weighted M4 ≤ unweighted; the small-n parcels of a C=8 montage
    # (most parcels carry 1–2 electrodes) make it STRICTLY smaller. M2 untouched.
    torch.manual_seed(0)
    batch = _ssl_batch(B=3, C=8, seed=2)
    on = _ssl_model().eval()
    off = _ssl_model(m4_precision_weight=False).eval()
    off.load_state_dict(on.state_dict())       # identical params ⇒ isolate the toggle
    assert on.m4_precision_weight is True and off.m4_precision_weight is False
    o_on, o_off = _run(on, batch), _run(off, batch)
    assert o_on["l_m4"].item() <= o_off["l_m4"].item() + 1e-6
    assert o_on["l_m4"].item() < o_off["l_m4"].item()
    assert torch.allclose(o_on["l_m2"], o_off["l_m2"])   # M4 weight never touches M2


def test_m4_precision_n_ref_controls_strictness() -> None:
    # n_ref is the saturation point: a LARGER n_ref down-weights MORE parcels
    # (every n in 2..n_ref-1 sits below 1), so n_ref=11 ≤ n_ref=2 ≤ OFF. (n=1
    # parcels carry w=0 for ANY n_ref — dof=n-1=0 — so n_ref never recovers the
    # OFF loss; it only sets where 2≤n saturates to 1.)
    torch.manual_seed(0)
    batch = _ssl_batch(B=3, C=8, seed=2)
    off = _ssl_model(m4_precision_weight=False).eval()
    strict = _ssl_model(m4_precision_n_ref=11.0).eval()
    loose = _ssl_model(m4_precision_n_ref=2.0).eval()
    strict.load_state_dict(off.state_dict())
    loose.load_state_dict(off.state_dict())
    l_off = _run(off, batch)["l_m4"].item()
    l_strict = _run(strict, batch)["l_m4"].item()
    l_loose = _run(loose, batch)["l_m4"].item()
    assert l_strict <= l_loose + 1e-6
    assert l_loose <= l_off + 1e-6
    assert l_strict < l_off          # n_ref=11 genuinely down-weights this montage


# ============= per-band M2/M4 loss + explained_var/target_var monitor (Ben 06-18)
def test_jepa_stats_on_cells_matches_hand_computed() -> None:
    # The monitor stat (loss / explained_var / target_var) must match the canonical
    # masked_jepa L1 definition to the digit. d=1, 4 valid cells, pred=0:
    #   loss = mean|0-t| = (0+1+2+3)/4 = 1.5
    #   target_var = Var(t)|unbiased=False = mean((t-1.5)^2) = 1.25
    #   explained_var = 1 - loss/Var(t) = 1 - 1.5/1.25 = -0.2
    pred = torch.zeros(4, 1)
    tgt = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    valid = torch.ones(4, dtype=torch.bool)
    loss, ev, tv = vc._jepa_stats_on_cells(pred, tgt, valid)
    assert loss.item() == pytest.approx(1.5)
    assert tv.item() == pytest.approx(1.25)
    assert ev.item() == pytest.approx(-0.2, abs=1e-5)


def test_jepa_stats_target_var_is_mean_over_feature_dims() -> None:
    # d=2: target_var is the MEAN over feature dims of each dim's variance, NOT the
    # sum. col0=[0,2,4] Var=8/3; col1=[0,0,0] Var=0 ⇒ tv=(8/3+0)/2=4/3.
    # explained_var uses Var over ALL elements: t=[0,0,2,0,4,0] mean=1 Var=14/6;
    # loss=(0+0+2+0+4+0)/6=1 ⇒ ev = 1 - 1/(14/6) = 1 - 3/7 = 4/7.
    pred = torch.zeros(3, 2)
    tgt = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    valid = torch.ones(3, dtype=torch.bool)
    loss, ev, tv = vc._jepa_stats_on_cells(pred, tgt, valid)
    assert loss.item() == pytest.approx(1.0)
    assert tv.item() == pytest.approx(4.0 / 3.0)
    assert ev.item() == pytest.approx(4.0 / 7.0, abs=1e-5)


def test_jepa_stats_nan_below_two_cells() -> None:
    # variance is undefined with < 2 scored cells → ev/tv NaN (logger skips them);
    # with 0 cells the loss is a graph-free 0 (no targets this step).
    pred = torch.zeros(4, 1)
    tgt = torch.ones(4, 1)
    one = torch.tensor([True, False, False, False])
    loss, ev, tv = vc._jepa_stats_on_cells(pred, tgt, one)
    assert loss.item() == pytest.approx(1.0)           # 1 cell → loss defined
    assert torch.isnan(ev) and torch.isnan(tv)         # variance undefined
    zero = torch.zeros(4, dtype=torch.bool)
    loss0, ev0, tv0 = vc._jepa_stats_on_cells(pred, tgt, zero)
    assert loss0.item() == 0.0 and not loss0.requires_grad
    assert torch.isnan(ev0) and torch.isnan(tv0)


def test_band_diagnostics_isolates_each_band_loss() -> None:
    # Craft a (rows, queries, d) set where beta cells are perfect and hg cells err
    # by 1 ⇒ the per-band raw L1 separates them exactly (regression: the per-band
    # loss is unchanged by the ev/tv refactor).
    d = 3
    pred = torch.zeros(2, 4, d)
    tgt = torch.zeros(2, 4, d)
    band = torch.tensor([[1, 1, 2, 2], [1, 2, 1, 2]])     # 1=beta, 2=hg
    valid = torch.ones(2, 4, dtype=torch.bool)
    tgt[band == 2] = 1.0                                   # hg cells |err|=1
    out = vc._band_diagnostics(pred, tgt, valid, band, vc._M2_BAND_IDS, "m2")
    assert out["l_m2_beta"].item() == 0.0
    assert out["l_m2_hg"].item() == pytest.approx(1.0)
    assert not out["l_m2_hg"].requires_grad               # detached diagnostic
    # the ev/tv keys ride alongside the per-band loss (aggregate + per band)
    for k in ("ev_m2", "tv_m2", "ev_m2_beta", "tv_m2_beta", "ev_m2_hg", "tv_m2_hg"):
        assert k in out, f"missing {k}"
        assert not out[k].requires_grad


def test_band_diagnostics_per_band_ev_tv_match_subset_stats() -> None:
    # Each per-band ev/tv must equal _jepa_stats_on_cells restricted to that band's
    # cells — i.e. the band split is an exact restriction of the aggregate stat.
    torch.manual_seed(3)
    d = 4
    pred = torch.randn(2, 5, d)
    tgt = torch.randn(2, 5, d)
    band = torch.tensor([[0, 1, 2, 1, 2], [2, 2, 1, 0, 1]])
    valid = torch.ones(2, 5, dtype=torch.bool)
    out = vc._band_diagnostics(pred, tgt, valid, band, vc._M4_BAND_IDS, "m4")
    for name, bid in vc._M4_BAND_IDS:
        bl, bev, btv = vc._jepa_stats_on_cells(pred, tgt, valid & (band == bid))
        assert out[f"l_m4_{name}"].item() == pytest.approx(bl.item())
        assert out[f"ev_m4_{name}"].item() == pytest.approx(bev.item())
        assert out[f"tv_m4_{name}"].item() == pytest.approx(btv.item())


def test_zero_bands_key_set_matches_band_diagnostics() -> None:
    # The no-target early-return must emit the SAME key set as the live path so the
    # forward dict keys are batch-stable (epoch aggregation needs stable keys).
    ref = torch.zeros(())
    for pairs, head in ((vc._M2_BAND_IDS, "m2"), (vc._M4_BAND_IDS, "m4")):
        pred = torch.zeros(2, 3, 4)
        tgt = torch.zeros(2, 3, 4)
        band = torch.zeros(2, 3, dtype=torch.long)
        valid = torch.ones(2, 3, dtype=torch.bool)
        live = set(vc._band_diagnostics(pred, tgt, valid, band, pairs, head))
        zeroed = set(vc._zero_bands(ref, pairs, head))
        assert live == zeroed, (head, live ^ zeroed)


def test_forward_emits_aggregate_and_per_band_ev_tv() -> None:
    model = _ssl_model().eval()
    out = _run(model, _ssl_batch(B=3, C=8, seed=2))
    # aggregate ev/tv pool all bands → well-defined (finite); detached monitors.
    for k in ("ev_m2", "tv_m2", "ev_m4", "tv_m4"):
        assert k in out and not out[k].requires_grad
        assert torch.isfinite(out[k])
    assert out["tv_m2"].item() >= 0.0 and out["tv_m4"].item() >= 0.0
    # per-band ev/tv keys are always present (NaN-or-finite, detached). A band with
    # < 2 scored cells this step is NaN — that is valid, _log_losses drops it.
    per_band = (
        "ev_m2_beta", "tv_m2_beta", "ev_m2_hg", "tv_m2_hg",
        "ev_m4_slow", "tv_m4_slow", "ev_m4_beta", "tv_m4_beta", "ev_m4_hg", "tv_m4_hg",
    )
    for k in per_band:
        assert k in out, f"missing {k}"
        assert not out[k].requires_grad
        v = out[k]
        assert torch.isnan(v) or torch.isfinite(v)


def test_forward_emits_per_band_stem_norms() -> None:
    model = _ssl_model().eval()
    out = _run(model, _ssl_batch(B=3, C=8, seed=2))
    for k in ("stem_norm_slow", "stem_norm_beta", "stem_norm_hg"):
        assert k in out, f"missing {k}"
        assert out[k].dim() == 0 and torch.isfinite(out[k]) and out[k].item() > 0.0
        assert not out[k].requires_grad                   # detached monitor


def test_stem_norm_matches_true_stem_output() -> None:
    # Veracity: the reported stem_norm_{band} EQUALS the band stem's true mean
    # per-token L2 norm — it is the magnitude that would dominate the latent.
    torch.manual_seed(0)
    tok = vc.ThreeBandTokenizer(d_model=16).eval()
    slow, beta, hg = _fake_bands(2, 5)
    with torch.no_grad():
        tok(slow, beta, hg)
        norms = tok.last_band_token_norm
        for b, stem, x in zip(vc.BANDS, tok.stems, (slow, beta, hg)):
            expected = stem(x).norm(dim=-1).mean()
            assert torch.allclose(norms[b.name], expected, atol=1e-6)


def test_stem_norm_catches_hot_stem() -> None:
    # The contract Ben asked for: one stem running 10× hotter must show up as a 10×
    # larger stem_norm for THAT band, with the others unchanged — nothing else
    # currently catches a single stem dominating the additive latent.
    class _Scale(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module, k: float) -> None:
            super().__init__()
            self.inner = inner
            self.k = k

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x) * self.k

    torch.manual_seed(0)
    tok = vc.ThreeBandTokenizer(d_model=16).eval()
    slow, beta, hg = _fake_bands(2, 5)
    with torch.no_grad():
        tok(slow, beta, hg)
        base = {k: v.clone() for k, v in tok.last_band_token_norm.items()}
        tok.stems[2] = _Scale(tok.stems[2], 10.0)          # hg = band index 2
        tok(slow, beta, hg)
        hot = tok.last_band_token_norm
    assert hot["hg"].item() == pytest.approx(10.0 * base["hg"].item(), rel=1e-5)
    assert hot["slow"].item() == pytest.approx(base["slow"].item(), rel=1e-6)
    assert hot["beta"].item() == pytest.approx(base["beta"].item(), rel=1e-6)


def test_forward_emits_per_band_diagnostics() -> None:
    model = _ssl_model().eval()
    out = _run(model, _ssl_batch(B=3, C=8, seed=2))
    for k in ("l_m2_beta", "l_m2_hg", "l_m4_slow", "l_m4_beta", "l_m4_hg"):
        assert k in out, f"missing per-band diagnostic {k}"
        assert torch.isfinite(out[k]).all() and out[k].item() >= 0.0
    # slow is M2-EXEMPT ⇒ there is no M2 slow target (and no l_m2_slow key)
    assert "l_m2_slow" not in out
    # the per-band diagnostics are detached (never folded into the loss graph)
    assert not out["l_m4_slow"].requires_grad


def test_m2mask_default_hg_start_rate_is_20pct() -> None:
    # Ben 2026-06-18: 20% is the HG coverage default (was 15%).
    assert vc.M2MaskConfig().hg_start_rate == 0.20
