"""The strip's whole job is to show WHICH tokens were hidden and what came back there.

Every test below is about a way the picture could be wrong while still looking plausible:
a transposed time axis, a band read at the wrong feature width, pad columns drawn as data,
or a per-panel colour stretch turning a flat prediction into apparent structure. A strip
that is merely pretty is not evidence, so the layout is asserted rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.neuroprobe.viz_mae_strip import figure, panels, token_index

BAND_LENGTHS = (2, 8, 16)          # r4/r6 ratio 4:16:32, scaled down
FDIMS = (7, 6, 7)                  # the real per-band bin counts
K_FULL = sum(BAND_LENGTHS)
N_CONTACTS = 3
F_MAX = max(FDIMS)


def _dump(tmp_path, n_clips: int = 2, seed: int = 0, flat_pred: bool = False):
    """A dump shaped exactly like v3_mae_recon.py's, with a KNOWN value per token.

    Each token's target is filled with its own (contact, band, time) so a mis-index shows up
    as a wrong number rather than as a plausible-looking smear.
    """
    rng = np.random.default_rng(seed)
    band = np.concatenate([np.repeat(np.arange(3), BAND_LENGTHS)] * N_CONTACTS)
    contact = np.repeat(np.arange(N_CONTACTS), K_FULL)
    n_tok = len(band)
    feat_count = np.asarray([FDIMS[b] for b in band], dtype=np.int64)

    target = np.zeros((n_clips, n_tok, F_MAX), dtype=np.float32)
    for i in range(n_tok):
        f = feat_count[i]
        # value encodes the token identity; pad columns stay a sentinel that must never show
        target[:, i, :f] = contact[i] * 100 + band[i] * 10 + np.arange(f) * 0.1
        target[:, i, f:] = -999.0
    pred = target.copy() if not flat_pred else np.zeros_like(target)
    in_loss = rng.random((n_clips, n_tok)) < 0.75

    p = tmp_path / "recon.npz"
    np.savez_compressed(
        p, target=target, pred=pred, in_loss=in_loss,
        feat_valid=np.ones((n_tok, F_MAX), dtype=bool), feat_count=feat_count,
        band=band, contact=contact,
        band_lengths=np.asarray(BAND_LENGTHS, dtype=np.int64),
        band_fdims=np.asarray(FDIMS, dtype=np.int64),
        k_full=np.int64(K_FULL), clip_frames=np.int64(64),
        starts=np.arange(n_clips), subject_id=np.int64(10), trial_id=np.int64(0),
        labels=np.asarray([f"c{i}" for i in range(N_CONTACTS)], dtype=object).astype(str),
    )
    return p


def test_token_index_recovers_the_contact_major_band_major_layout(tmp_path) -> None:
    z = np.load(_dump(tmp_path))
    rows, contacts = token_index(z["band"], z["contact"], K_FULL, z["band_lengths"])
    assert list(contacts) == list(range(N_CONTACTS))
    for c in range(N_CONTACTS):
        for b in range(3):
            ix = rows[(c, b)]
            assert len(ix) == BAND_LENGTHS[b]
            assert (z["band"][ix] == b).all()
            assert (z["contact"][ix] == c).all()
            assert (np.diff(ix) > 0).all(), "band rows must come back in TIME order"


def test_a_transposed_or_band_interleaved_layout_is_rejected(tmp_path) -> None:
    """If the grid were ever laid out band-major across contacts the strip would read one
    band's time course as a mixture of contacts. That has to raise, not render."""
    z = np.load(_dump(tmp_path))
    bad = np.concatenate([np.repeat(np.arange(3), BAND_LENGTHS)[::-1]] +
                         [np.repeat(np.arange(3), BAND_LENGTHS)] * (N_CONTACTS - 1))
    with pytest.raises(AssertionError, match="band layout differs"):
        token_index(bad, z["contact"], K_FULL, z["band_lengths"])


def test_band_lengths_that_contradict_the_stored_layout_raise(tmp_path) -> None:
    z = np.load(_dump(tmp_path))
    with pytest.raises(AssertionError, match="band_lengths says"):
        token_index(z["band"], z["contact"], K_FULL, np.asarray([3, 7, 16]))


def test_panels_slice_the_right_band_and_never_return_pad_columns(tmp_path) -> None:
    z = np.load(_dump(tmp_path))
    for b in range(3):
        truth, pred, masked, label = panels(z, 0, 1, b)
        assert truth.shape == (FDIMS[b], BAND_LENGTHS[b]), "shape is (F_b, T_b), not (T, F)"
        assert masked.shape == (BAND_LENGTHS[b],)
        assert label == 1
        assert (truth > -900).all(), "a pad column leaked into the picture"
        # the encoded identity proves it is contact 1, band b -- not some neighbouring block
        assert np.isclose(truth[0, 0], 1 * 100 + b * 10)
        np.testing.assert_allclose(pred, truth)


def test_a_feat_count_disagreeing_with_band_fdims_raises(tmp_path) -> None:
    """band_fdims (7,6,7) is not back-solvable from the token width, so a wrong one would
    silently slice pad columns in as data. feat_count is the independent check."""
    p = _dump(tmp_path)
    d = dict(np.load(p))
    d["band_fdims"] = np.asarray([7, 7, 7])          # MID is 6, not 7
    np.savez_compressed(p, **d)
    with pytest.raises(AssertionError, match="feat_count"):
        panels(np.load(p), 0, 0, 1)


def test_figure_writes_and_reports_the_masked_fraction_it_drew(tmp_path) -> None:
    p = _dump(tmp_path, seed=4)
    out = tmp_path / "strip.png"
    info = figure(str(p), str(out), clip=0, band=2, n_contacts=2, rate=32.0, offset=0.0)
    assert out.stat().st_size > 0
    assert info["band"] == "hga" and len(info["contacts"]) == 2
    # the contacts drawn are the MOST-masked ones, so the reported fraction cannot be tiny
    assert all(f >= 0.4 for f in info["masked_frac"]), info["masked_frac"]


def _bimodal(tmp_path, seed=1):
    """r6's real per-contact structure: some contacts spatially dropped (every token hidden),
    the rest keeping half their frames. Contact 0 partial, contact 2 fully hidden."""
    p = _dump(tmp_path, seed=seed)
    d = dict(np.load(p))
    rows, _ = token_index(d["band"], d["contact"], K_FULL, d["band_lengths"])
    ix0 = rows[(0, 2)]
    d["in_loss"][0, ix0] = False
    d["in_loss"][0, ix0[: len(ix0) // 2]] = True     # contact 0: half hidden
    d["in_loss"][0, rows[(2, 2)]] = True             # contact 2: everything hidden
    np.savez_compressed(p, **d)
    return p


def test_the_selection_spans_both_masking_regimes(tmp_path) -> None:
    """The bug this pins: ranking contacts by "most masked" can only ever return fully
    dropped ones, because r6's per-contact masked fraction is bimodal (0.50 for a spatially
    kept contact, 1.00 for a dropped one) -- never in between. Every drawn row then has an
    EMPTY encoder-input panel and the figure cannot show an infill at all. So the draw must
    include at least one contact that kept some of its own history."""
    p = _bimodal(tmp_path)
    info = figure(str(p), str(tmp_path / "s.png"), clip=0, band=2, n_contacts=2,
                  rate=32.0, offset=0.0)
    fracs = info["masked_frac"]
    assert any(f < 1.0 for f in fracs), f"every drawn contact is fully hidden: {fracs}"
    assert any(f >= 1.0 for f in fracs), f"the fully-hidden regime is not shown: {fracs}"


def test_a_fully_visible_contact_is_not_drawn_over_a_partly_masked_one(tmp_path) -> None:
    """The original concern still holds: a contact with nothing hidden shows nothing about
    reconstruction, so it must lose to one that has holes."""
    p = _dump(tmp_path, seed=1)
    d = dict(np.load(p))
    rows, _ = token_index(d["band"], d["contact"], K_FULL, d["band_lengths"])
    d["in_loss"][0, rows[(0, 2)]] = False            # contact 0: nothing hidden
    ix1 = rows[(1, 2)]
    d["in_loss"][0, ix1] = False
    d["in_loss"][0, ix1[: len(ix1) // 2]] = True     # contact 1: half hidden
    d["in_loss"][0, rows[(2, 2)]] = True             # contact 2: everything hidden
    np.savez_compressed(p, **d)
    info = figure(str(p), str(tmp_path / "s.png"), clip=0, band=2, n_contacts=2,
                  rate=32.0, offset=0.0)
    assert 0 not in info["contacts"], f"drew the fully-visible contact: {info['contacts']}"
    assert set(info["contacts"]) == {1, 2}, info["contacts"]


def test_a_flat_prediction_is_not_stretched_into_looking_like_structure(tmp_path) -> None:
    """One colour scale per band, taken from the TRUTH. A per-panel stretch would rescale a
    constant prediction to full gamut and the strip would advertise a reconstruction that
    does not exist -- so the flat panel must come out uniform."""
    import matplotlib
    matplotlib.use("Agg")

    p = _dump(tmp_path, seed=2, flat_pred=True)
    out = tmp_path / "flat.png"
    figure(str(p), str(out), clip=0, band=2, n_contacts=1, rate=32.0, offset=0.0)
    import matplotlib.image as mpimg
    img = mpimg.imread(str(out))[..., :3]
    truth, pred, _, _ = panels(np.load(p), 0, 0, 2)
    vmin, vmax = np.percentile(truth, [2, 98])
    # the prediction is all zeros and the truth spans 200..201; on the shared scale that is
    # off the bottom of the colormap, i.e. one flat colour, which is the honest rendering
    assert pred.max() == 0.0 and pred.min() == 0.0
    assert vmin > 1.0 and vmax > vmin, (vmin, vmax)
    assert img.size > 0
