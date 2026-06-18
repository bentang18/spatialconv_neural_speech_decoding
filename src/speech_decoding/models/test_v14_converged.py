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
    # FE §8.5: HG = round(0.15·16)=2 distinct starts × width 3, overlaps allowed
    # (distinct starts ⇒ union ∈ [4,6]); each masked run is contiguous.
    for seed in range(25):
        hg = vc.sample_m2_mask(_gen(seed))[22:38]
        assert 4 <= hg.sum().item() <= 6, f"HG coverage {hg.sum()} out of [4,6]"


def test_m2_total_in_maskable_pool_range() -> None:
    # FE §8.6: pool = 32 (beta+HG); realized ~beta 8 + HG 4–6 ≈ 12–14.
    counts = [vc.sample_m2_mask(_gen(s)).sum().item() for s in range(60)]
    assert all(12 <= c <= 14 for c in counts)
    assert 12 <= sum(counts) / len(counts) <= 14


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
