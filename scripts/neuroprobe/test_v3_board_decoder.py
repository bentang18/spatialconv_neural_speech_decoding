"""Shape/layout contract tests for the CS decoder sweep.

These guard the two things that fail SILENTLY at scale: (a) the flat->structured unflatten of the
cache tap, which if wrong just trains on scrambled features and returns a plausible-looking AUROC,
and (b) the harness CNN's hand-computed `conv_output_size`, which is an integer-division formula
(`dim // 8`) that agrees with three MaxPool halvings only for some dims -- a mismatch is a shape
error at fit time, i.e. 150 dead cluster cells.
"""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from v3_board_decoder import (  # noqa: E402
    ATTN_TAPS,
    BAND_F0,
    BAND_T,
    D_MODEL,
    ENC0_DIM,
    ENC12_DIM,
    AttnPoolDecoder,
    HarnessCNN,
    PerBandCNN,
    build,
    shape_for,
    tap_bands,
    tap_dim,
)


def test_tap_dims_match_the_measured_cache():
    # measured on v3_board_cache_board_r6_40k/enc_s1_t1: enc0 348, enc12 13312
    assert ENC0_DIM == 348
    assert ENC12_DIM == 13312
    assert tap_dim("enc0") == 348
    assert tap_dim("enc12") == 13312


def test_band_blocks_tile_the_flat_axis_exactly():
    for enc, total in (("enc0", 348), ("enc12", 13312)):
        blocks = tap_bands(enc)
        assert len(blocks) == 3, "SLOW, MID, HGA"
        assert sum(t * d for t, d in blocks) == total
        # time axis is the shared 4/16/32 token grid in both taps
        assert tuple(t for t, _ in blocks) == BAND_T


def test_enc0_blocks_carry_per_band_freq_bins():
    # enc0's feature axis is per-band F_b, NOT a uniform d_model -- this is exactly why enc0
    # takes the Conv1d branch in the naive arm.
    assert tuple(d for _, d in tap_bands("enc0")) == BAND_F0
    assert tuple(d for _, d in tap_bands("enc12")) == (D_MODEL,) * 3


def test_unsupported_tap_rejected():
    with pytest.raises(ValueError):
        tap_bands("enc7")


def test_intermediate_taps_share_the_enc12_layout():
    # attnpool reads enc3/enc6 too; they are encoder outputs like enc12, so same (T_b, d_model)
    for enc in ("enc3", "enc6"):
        assert tap_bands(enc) == tap_bands("enc12")
        assert tap_dim(enc) == ENC12_DIM


def test_attn_taps_are_the_four_cached_depths():
    assert ATTN_TAPS == ("enc0", "enc3", "enc6", "enc12")


def _attn_inputs(b, n_parcels):
    return tuple(torch.randn(b, n_parcels, tap_dim(t)) for t in ATTN_TAPS)


@pytest.mark.parametrize("n_parcels", [3, 16])
def test_attnpool_forwards_to_two_logits(n_parcels):
    model = build("attnpool", ATTN_TAPS, n_parcels, 2)
    out = model(_attn_inputs(5, n_parcels))
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()


def test_attnpool_weights_are_convex_over_the_four_taps():
    """The learned query must produce a genuine convex combination -- non-negative, summing to 1,
    one weight per TAP (not per parcel). A softmax over the wrong axis silently still returns
    finite logits, so assert the axis, not just the values."""
    torch.manual_seed(0)
    n_parcels = 16
    model = AttnPoolDecoder(n_parcels, tuple(tap_dim(t) for t in ATTN_TAPS), 2).eval()
    with torch.no_grad():
        w = model.tap_weights(_attn_inputs(7, n_parcels))
    assert w.shape == (7, len(ATTN_TAPS))
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(1), torch.ones(7), atol=1e-5)


def test_attnpool_global_token_reads_every_tap():
    """Perturbing ANY ONE tap must move the output. If a tap is silently dropped (wrong zip,
    wrong index) the arm degenerates to single-tap and the whole experiment is a no-op."""
    torch.manual_seed(0)
    n_parcels = 8
    model = AttnPoolDecoder(n_parcels, tuple(tap_dim(t) for t in ATTN_TAPS), 2).eval()
    xs = _attn_inputs(4, n_parcels)
    with torch.no_grad():
        base = model(xs)
    for i in range(len(ATTN_TAPS)):
        bumped = list(xs)
        bumped[i] = bumped[i] + 5.0
        with torch.no_grad():
            got = model(tuple(bumped))
        assert not torch.allclose(base, got, atol=1e-6), f"tap {ATTN_TAPS[i]} is not read"


def test_attnpool_parcel_tokens_come_from_the_main_tap():
    """`main_tap=-1` means enc12 supplies the per-parcel tokens. The head consumes (P+1) tokens:
    one pooled global + P parcels. A miscount here is a silent shape-time death on the cluster."""
    n_parcels = 11
    model = AttnPoolDecoder(n_parcels, tuple(tap_dim(t) for t in ATTN_TAPS), 2)
    assert model.fc1.in_features == (n_parcels + 1) * D_MODEL
    assert model.main_tap == -1
    assert model.proj[-1].in_features == tap_dim("enc12")


@pytest.mark.parametrize("enc", ["enc0", "enc12"])
@pytest.mark.parametrize("arm", ["mlp", "naive", "perband"])
@pytest.mark.parametrize("n_parcels", [3, 16])
def test_every_arm_forwards_to_two_logits(arm, enc, n_parcels):
    b = 5
    model = build(arm, enc, n_parcels, 2)
    x = torch.randn(b, *shape_for(arm, enc, n_parcels))
    out = model(x)
    assert out.shape == (b, 2)
    assert torch.isfinite(out).all()


def test_harness_cnn_flat_size_matches_actual_pooling_enc12():
    """The `//8` formula must equal three real MaxPool2d(2) halvings on (52, 256)."""
    n_parcels = 16
    m = HarnessCNN((n_parcels, sum(BAND_T), D_MODEL), 2)
    x = torch.randn(2, n_parcels, sum(BAND_T), D_MODEL)
    z = m.pool(m.relu(m.conv1(x)))
    z = m.pool(m.relu(m.conv2(z)))
    z = m.pool(m.relu(m.conv3(z)))
    assert z.flatten(1).shape[1] == m.fc1.in_features
    assert z.shape[-2:] == (52 // 8, 256 // 8) == (6, 32)


def test_harness_cnn_flat_size_matches_actual_pooling_enc0():
    """Same check on the Conv1d branch: 348 -> 174 -> 87 -> 43, and 348 // 8 == 43."""
    n_parcels = 16
    m = HarnessCNN((n_parcels, 348), 2)
    x = torch.randn(2, n_parcels, 348)
    z = m.pool(m.relu(m.conv1(x)))
    z = m.pool(m.relu(m.conv2(z)))
    z = m.pool(m.relu(m.conv3(z)))
    assert z.flatten(1).shape[1] == m.fc1.in_features
    assert z.shape[-1] == 348 // 8 == 43


@pytest.mark.parametrize("enc", ["enc0", "enc12"])
def test_perband_split_recovers_the_cache_flatten_order(enc):
    """A flat row built by CONCATENATING per-band (T_b, D_b) blocks must be split back into
    exactly those blocks. This is the unflatten that, if transposed, silently scrambles features."""
    blocks = tap_bands(enc)
    n_parcels = 4
    # build a flat feature axis whose every entry encodes (band, t, d) so a wrong split shows up
    parts, tag = [], 0.0
    for bi, (t_b, d_b) in enumerate(blocks):
        blk = torch.full((1, n_parcels, t_b, d_b), float(bi))
        parts.append(blk.reshape(1, n_parcels, t_b * d_b))
        tag += 1
    flat = torch.cat(parts, dim=2)
    assert flat.shape[-1] == tap_dim(enc)

    off = 0
    for bi, (t_b, d_b) in enumerate(blocks):
        span = t_b * d_b
        got = flat[:, :, off : off + span].reshape(1, n_parcels, t_b, d_b)
        assert torch.all(got == float(bi)), f"band {bi} split landed on the wrong span"
        off += span
    assert off == tap_dim(enc)


@pytest.mark.parametrize("enc", ["enc0", "enc12"])
def test_perband_never_mixes_bands(enc):
    """Perturbing ONLY the HGA block must leave the SLOW and MID stack outputs bit-identical --
    the property `naive` deliberately lacks (its kernels straddle the seams)."""
    torch.manual_seed(0)
    blocks = tap_bands(enc)
    n_parcels = 4
    m = PerBandCNN(n_parcels, blocks, 2).eval()
    x = torch.randn(2, n_parcels, tap_dim(enc))
    y = x.clone()
    hga_span = blocks[-1][0] * blocks[-1][1]
    y[:, :, -hga_span:] += 10.0  # clobber HGA only

    def band_feats(t):
        outs, off = [], 0
        for stack, (t_b, d_b) in zip(m.stacks, blocks):
            span = t_b * d_b
            blk = t[:, :, off : off + span].reshape(t.size(0), t.size(1), t_b, d_b)
            outs.append(stack(blk).flatten(1))
            off += span
        return outs

    with torch.no_grad():
        a, b = band_feats(x), band_feats(y)
    assert torch.equal(a[0], b[0]), "SLOW band changed when only HGA was perturbed"
    assert torch.equal(a[1], b[1]), "MID band changed when only HGA was perturbed"
    assert not torch.equal(a[2], b[2]), "HGA band did NOT change -- the split is not wired"
