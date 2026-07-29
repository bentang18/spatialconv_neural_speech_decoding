"""What has to be true for the per-electrode AUROC map to mean anything.

``--self-test`` inside the script covers the statistic itself (null calibration, planted-effect
recovery, the drift guard) because that check has to be runnable on a login node before the
array is submitted. These tests cover what it does not: the plumbing that turns a shard into a
region table, and the two guards whose failure mode is SILENT rather than loud.
"""
from __future__ import annotations

import numpy as np

from scripts.neuroprobe.viz_elec_auroc import (
    _halves,
    aggregate,
    auroc_cols,
    bh_fdr,
    cv_auroc,
    maxstat_threshold,
    pooled_p,
    region_table,
    self_test,
    session_shard,
)


def test_self_test_properties_hold() -> None:
    """The pre-flight check is itself under test, so it cannot rot into a no-op."""
    self_test(0)


def test_halves_keep_both_classes_in_both_folds() -> None:
    """An alternating label used to put one class entirely in each fold -- AUROC then pinned
    at exactly 0.5 everywhere, which reads as "no effect" rather than as a broken split."""
    for y in (np.tile([0, 1], 60),                       # the pathological alternating case
              np.repeat([0, 1], 60),                     # blocked
              np.random.default_rng(0).integers(0, 2, 121)):
        h0, h1 = _halves(y)
        assert len(set(h0) & set(h1)) == 0
        assert sorted(np.concatenate([h0, h1])) == list(range(len(y)))
        for h in (h0, h1):
            assert y[h].sum() > 0 and y[h].sum() < len(h), y[h].sum()


def test_halves_split_drift_evenly() -> None:
    """The reason for interleaving rather than a random split: mean trial index must match."""
    y = np.repeat([0, 1], 100)
    h0, h1 = _halves(y)
    assert abs(h0.mean() - h1.mean()) < 2.0, (h0.mean(), h1.mean())


def test_flat_column_is_exactly_half_not_index_order() -> None:
    """A dead contact must score 0.5, not inherit the label's own drift through tie-breaking.

    Without the guard, a constant score column gets ranked by trial index, so a label that
    increases with trial index (word_index does, by construction) reads as AUROC 1.0 on a
    channel carrying no signal at all.
    """
    n = 80
    y = (np.arange(n) >= n // 2).astype(np.int64)         # label == trial order
    s = np.zeros((n, 3))
    s[:, 1] = 7.5                                        # constant, different value
    au = auroc_cols(s, y)
    assert np.allclose(au, 0.5), au


def test_cv_auroc_is_not_circular() -> None:
    """Fitting and scoring on the same trials would find structure in pure noise at this
    K; the held-out fold is what keeps the map honest."""
    rng = np.random.default_rng(1)
    n, P, T, K = 60, 4, 5, 32                            # K comparable to n/2 on purpose
    Z = rng.normal(size=(n, P, T, K))
    y = np.tile([0, 1], n // 2)
    au = cv_auroc(Z, y)
    assert abs(au.mean() - 0.5) < 0.06, au.mean()

    # the circular version, for contrast: fit and score the same trials
    h = np.arange(n)
    w = Z[h][y == 1].mean(axis=0) - Z[h][y == 0].mean(axis=0)
    s = np.einsum("nptk,ptk->npt", Z, w)
    circ = auroc_cols(s.reshape(n, P * T), y)
    assert circ.mean() > 0.75, circ.mean()


def test_bh_fdr_matches_a_hand_worked_case() -> None:
    # m=6, thresholds q*k/m = .00833 .01667 .025 .03333 .04167 .05
    p = np.array([0.001, 0.009, 0.024, 0.60, 0.70, 0.80])
    assert bh_fdr(p, 0.05).tolist() == [True, True, True, False, False, False]
    # .039 sits ABOVE its own k=3 threshold of .025, so nothing past k=2 is rejected
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.60])
    assert bh_fdr(p, 0.05).tolist() == [True, True, False, False, False, False]
    assert bh_fdr(np.array([0.9, 0.8]), 0.05).sum() == 0

    # the prefix rule: m=4, thresholds .0125 .025 .0375 .05. p=.030 fails its own k=2
    # threshold but lies below the largest passing p (.035 at k=3), so BH rejects it anyway.
    # Testing only the individually-passing p's would be Bonferroni-ish and lose power.
    p = np.array([0.010, 0.030, 0.035, 0.9])
    assert bh_fdr(p, 0.05).tolist() == [True, True, True, False], bh_fdr(p, 0.05)


def test_maxstat_uses_the_map_maximum_not_the_cell() -> None:
    """The correction has to grow with map size, or it is not a correction at all."""
    rng = np.random.default_rng(2)
    small = rng.normal(0.5, 0.05, size=(500, 2, 2))
    big = rng.normal(0.5, 0.05, size=(500, 40, 40))
    assert maxstat_threshold(big) > maxstat_threshold(small)
    # and it is a threshold on |AUROC-.5|, so it is positive and scale-following
    assert maxstat_threshold(small) > 0
    assert maxstat_threshold(0.5 + 2 * (small - 0.5)) > 1.9 * maxstat_threshold(small)


def test_pooled_p_beats_the_per_cell_floor() -> None:
    """Its whole reason to exist: resolution below 1/(n_perm+1)."""
    null = np.full((100, 8, 8), 0.5) + np.random.default_rng(3).normal(0, .01, (100, 8, 8))
    obs = np.full((8, 8), 0.5)
    obs[0, 0] = 0.9                                       # far outside the null
    p = pooled_p(obs, null)
    assert p[0, 0] < 1.0 / 101.0, p[0, 0]
    assert p[0, 0] >= 1.0 / (1 + 100 * 64)
    assert p[1, 1] > 0.1, p[1, 1]                          # a null cell stays unremarkable


def _fake_cache(tmp_path, *, plant_contact: int, seed: int = 0):
    """A record with the encode cache's exact schema, plus a planted effect in one contact.

    The band layout is deliberately NOT square: band_lengths (4,8,16) and band_fdims (3,4,5)
    mean the HGA block sits at a non-zero offset with a width unique to it, so an off-by-one
    in ``_band_slice`` shows up as a wrong answer instead of quietly reading the wrong band.
    """
    import torch

    from scripts.neuroprobe.viz_anatomy import dkt_tables

    base_of, _ = dkt_tables()
    id_of = {b: i for i, b in base_of.items()}
    rng = np.random.default_rng(seed)
    n, P = 240, 6
    bl, bf, d = (4, 8, 16), (3, 4, 5), 8
    k_full, T_hga, C0, C12 = sum(bl), bl[2], bf[2], d
    canon = np.array([id_of["superiortemporal"]] * 3 + [id_of["middletemporal"]] * 3)
    y = np.tile([0.0, 1.0], n // 2)

    feats = {}
    for tap, width in (("enc0_elec", sum(t * f for t, f in zip(bl, bf))),
                       ("enc12_elec", k_full * d)):
        x = rng.normal(size=(n, P, width)).astype(np.float32)
        if tap == "enc0_elec":
            off = bl[0] * bf[0] + bl[1] * bf[1]
            blk = x[:, :, off:off + T_hga * C0].reshape(n, P, T_hga, C0)
        else:
            blk = x[:, :, bl[0] * d + bl[1] * d:].reshape(n, P, T_hga, C12)
        blk[y == 1, plant_contact, 5:8, 0] += 2.0          # HGA band only, 3 bins, one contact
        feats[tap] = {"raw": torch.from_numpy(x)}

    rec = {
        "subject_id": 1, "trial_id": 1,
        "present_parcels": np.unique(canon), "parcel_canon": canon,
        "band_lengths": bl, "band_fdims": bf,
        "feats": feats,
        "labels": {"onset": y, "frame_brightness": rng.integers(0, 2, n).astype(float)},
        "n_windows": n,
    }
    path = tmp_path / "enc_s1_t1.pt"
    torch.save(rec, path)
    return str(path), T_hga


def test_session_shard_end_to_end_on_a_real_schema_record(tmp_path) -> None:
    """The cache-reading path, which the in-script self-test cannot reach.

    Exercises band slicing, the mmap load, the ``*_elec`` row-count assert, the label
    selection, the PC reduction and the shard write in one go -- then checks that the planted
    contact is the one the region table names. Every one of those is a place where a silent
    error produces a plausible-looking map.
    """
    path, T = _fake_cache(tmp_path, plant_contact=1)     # contact 1 is superiortemporal
    out = session_shard(path, taps=("enc0_elec", "enc12_elec"),
                        tasks=("onset", "frame_brightness"), band="hga", n_pc=4, n_perm=60,
                        perm_block=0, chunk=64, seed=0, verbose=False)

    for tap in ("enc0_elec", "enc12_elec"):
        au = out[f"auroc/{tap}/onset"]
        assert au.shape == (6, T), (tap, au.shape)
        thr = maxstat_threshold(out[f"null_max/{tap}/onset"], 0.05)
        sig = np.abs(au - 0.5) > thr
        assert sig[1, 5:8].all(), (tap, au[1], thr)      # planted cells found
        assert not sig[np.arange(6) != 1].any(), (tap, sig)   # and no other contact
        # the planted effect is in the HGA band only, so reading another band would miss it
        assert au[1, 5:8].min() > 0.75, (tap, au[1])
        # a label with no relation to the features stays inside its own null
        au_b = out[f"auroc/{tap}/frame_brightness"]
        thr_b = maxstat_threshold(out[f"null_max/{tap}/frame_brightness"], 0.05)
        assert not (np.abs(au_b - 0.5) > thr_b).any(), (tap, np.abs(au_b - .5).max(), thr_b)

    # enc0 has C=5 so the PC basis is capped there, and the retained variance must be reported
    assert out["pc_var/enc0_elec"].shape == (6,)
    assert (out["pc_var/enc0_elec"] > 0.5).all()

    # ... and the shard survives a real npz round trip into the region table + report
    np.savez_compressed(tmp_path / "elec_s1_t1_hga.npz", **out)
    agg = aggregate(str(tmp_path), taps=("enc12_elec",), tasks=("onset",))
    rows = region_table(agg, "enc12_elec", "onset")
    assert rows[0][0] == "superiortemporal", rows
    # the count is CONTACTS, not cells: exactly one of the three STG contacts was planted
    assert (rows[0][2], rows[0][3]) == (1, 3), rows
    assert rows[1][2] == 0, rows

    # and the printed report runs -- a crash here would only surface after the array had run
    from scripts.neuroprobe.viz_elec_auroc import report
    r = report(agg, taps=("enc12_elec",), tasks=("onset",))
    assert r[("enc12_elec", "onset")]["st_rank"] == 1, r
    assert r[("enc12_elec", "onset")]["st_sig"] == 1, r

    # the FDR route reaches the same contact and, being the more sensitive one, never fewer
    agg_f = aggregate(str(tmp_path), taps=("enc12_elec",), tasks=("onset",), inference="fdr")
    rows_f = region_table(agg_f, "enc12_elec", "onset")
    assert rows_f[0][0] == "superiortemporal", rows_f
    assert rows_f[0][2] >= rows[0][2], (rows_f, rows)


def test_shard_round_trip_to_region_table(tmp_path) -> None:
    """Key naming is the fragile part: a renamed shard key would make every region table
    empty, and an empty table reads as "no result" rather than as a bug."""
    from scripts.neuroprobe.viz_anatomy import dkt_tables

    base_of, _ = dkt_tables()
    id_of = {b: i for i, b in base_of.items()}
    st, mt = id_of["superiortemporal"], id_of["middletemporal"]

    P, T = 6, 8
    rng = np.random.default_rng(0)
    for sub, trial in ((1, 1), (2, 0)):
        canon = np.array([st, st, st, mt, mt, mt])
        au = np.full((P, T), 0.5)
        au[:3, 4] = 0.80                                  # STG carries it, far above the null
        au[3:, 4] = 0.52                                  # middletemporal, inside the null
        # a null whose per-map max deviation is ~.03, so .30 survives FWER and .02 does not
        null = 0.5 + rng.uniform(-0.03, 0.03, size=(50, P, T)).astype(np.float32)
        np.savez_compressed(
            tmp_path / f"elec_s{sub}_t{trial}_hga.npz",
            subject_id=np.int64(sub), trial_id=np.int64(trial), parcel_canon=canon,
            n_windows=np.int64(100), perm_block=np.int64(200), n_perm=np.int64(50),
            **{"auroc/enc12_elec/onset": au.astype(np.float32),
               "null/enc12_elec/onset": null})

    agg = aggregate(str(tmp_path), taps=("enc12_elec",), tasks=("onset",))
    rows = region_table(agg, "enc12_elec", "onset")
    assert rows, "region table is empty -- the shard keys did not survive the round trip"
    assert rows[0][0] == "superiortemporal", rows
    assert rows[0][1] > rows[1][1], rows                  # STG above middletemporal
    assert rows[0][2] == 6 and rows[0][3] == 6, rows      # 3 contacts x 2 sessions, all sig
    assert rows[0][4] == 2, rows                          # pooled over both subjects
    assert rows[1][2] == 0, rows                          # middletemporal survives nothing
