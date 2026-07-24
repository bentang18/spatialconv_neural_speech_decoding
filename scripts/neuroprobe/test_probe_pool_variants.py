"""Ground-truth tests for the electrode-pooling variants.

The load-bearing claims are algebraic, so they are tested as identities rather than smoke:
β=0 IS the arithmetic mean (so every Δ in the report is a real Δ), weights are per-BAND (so a
different electrode can dominate HGA than the slow envelope), and large β selects the strongest
electrode. Plus the parity check itself must FAIL on a corrupted segment map — a parity check that
cannot fail is worse than none, since the whole point is to catch a wrong reshape before any AUROC
is read.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probe_pool_variants as ppv  # noqa: E402
from v3_probe_readout_r4 import PROBE_TASKS  # noqa: E402

D = ppv.ENC_D_MODEL
K = 3                                   # 3 time bins, one per band → band_lengths (1,1,1)
F = K * D
N_CONTACTS = 6
PARCELS = np.array([10, 10, 10, 20, 20, 30], dtype=np.int64)   # sizes 3 / 2 / 1


def _make_session(subject, trial, tag, out_dir, rng, n=40, elec=None):
    """enc12 is built as the TRUE mean of enc12_elec, so β=0 parity is a real check."""
    if elec is None:
        elec = rng.standard_normal((n, N_CONTACTS, F)).astype(np.float32)
    present = np.unique(PARCELS)
    pooled = np.stack([elec.reshape(n, N_CONTACTS, F)[:, PARCELS == p].mean(1) for p in present], 1)

    y = rng.integers(0, 2, size=n).astype(np.float64) * 2.0 - 1.0
    half = n // 2
    tr, te = np.arange(half), np.arange(half, n)
    labels = {t: y.copy() for t in PROBE_TASKS}
    payload = {
        "subject_id": subject, "trial_id": trial, "ckpt_tag": tag,
        "present_parcels": present,
        "parcel_canon": PARCELS.copy(),
        "band_lengths": (1, 1, 1),
        "feats": {
            "enc12": {"raw": torch.from_numpy(pooled).to(torch.float16)},
            "enc12_elec": {"raw": torch.from_numpy(elec).to(torch.float16)},
        },
        "clip_starts": np.arange(n), "labels": labels,
        "ws_split": {t: {"fold0": {"train": tr, "test": te}} for t in PROBE_TASKS},
        "cs_split": {t: {"test": te} for t in PROBE_TASKS},
        "n_windows": n,
    }
    torch.save(payload, os.path.join(out_dir, f"enc_s{subject}_t{trial}_{tag}.pt"))
    return payload


def test_beta_zero_is_exactly_the_mean() -> None:
    """β=0 must be the arithmetic mean, not merely close to it — every reported Δ is measured
    against it, so a biased baseline would silently offset the whole β ladder."""
    rng = np.random.default_rng(0)
    blk = torch.from_numpy(rng.standard_normal((5, 4, 2, D)).astype(np.float32))
    w = ppv._band_weights(blk, 0.0)
    assert torch.allclose(w, torch.full((5, 4), 0.25))
    pooled = (w[:, :, None, None] * blk).sum(1)
    assert torch.allclose(pooled, blk.mean(1), atol=1e-6)


def test_weights_are_convex_per_segment_and_band() -> None:
    rng = np.random.default_rng(1)
    blk = torch.from_numpy(rng.standard_normal((7, 5, 3, D)).astype(np.float32))
    for beta in (0.0, 1.0, 4.0):
        w = ppv._band_weights(blk, beta)
        assert torch.allclose(w.sum(1), torch.ones(7), atol=1e-5)
        assert bool((w >= 0).all())


def test_large_beta_selects_the_strongest_electrode() -> None:
    rng = np.random.default_rng(2)
    blk = torch.from_numpy(rng.standard_normal((3, 4, 2, D)).astype(np.float32))
    blk[:, 2] *= 50.0                                          # electrode 2 dominates
    w = ppv._band_weights(blk, 40.0)
    assert bool((w.argmax(1) == 2).all())
    assert float(w[:, 2].min()) > 0.99


def test_single_electrode_segment_does_not_divide_by_zero() -> None:
    blk = torch.ones((4, 1, 2, D))
    for beta in (0.0, 4.0):
        w = ppv._band_weights(blk, beta)
        assert torch.allclose(w, torch.ones((4, 1)))


def test_weights_are_per_band_not_shared_across_time() -> None:
    """The design claim: the contact carrying HGA need not carry the slow envelope. Plant a
    different dominant electrode in each band and require the pooled output to follow it."""
    rng = np.random.default_rng(3)
    elec = rng.standard_normal((12, N_CONTACTS, K, D)).astype(np.float32) * 0.01
    elec[:, 0, 0, :] = 5.0        # band 0 dominated by contact 0   (parcel 10)
    elec[:, 1, 1, :] = 7.0        # band 1 dominated by contact 1   (parcel 10)
    elec[:, 2, 2, :] = 9.0        # band 2 dominated by contact 2   (parcel 10)
    rec = {"feats": {"enc12_elec": {"raw": torch.from_numpy(elec.reshape(12, N_CONTACTS, F))}},
           "present_parcels": np.unique(PARCELS), "parcel_canon": PARCELS.copy(),
           "band_lengths": (1, 1, 1)}
    seg_cols, seg_atlas = ppv._segments(rec, "A")
    z = ppv._pool(rec, np.arange(12), seg_cols, ppv._band_slices(rec, K), 40.0, chunk=8)
    p10 = int(np.where(seg_atlas == 10)[0][0])                 # the 3-contact parcel
    got = z[:, p10].reshape(12, K, D)
    assert got[:, 0].mean() == pytest.approx(5.0, rel=0.02)
    assert got[:, 1].mean() == pytest.approx(7.0, rel=0.02)
    assert got[:, 2].mean() == pytest.approx(9.0, rel=0.02)


def test_pooling_preserves_time_and_band_dimensions() -> None:
    rng = np.random.default_rng(4)
    elec = rng.standard_normal((9, N_CONTACTS, F)).astype(np.float32)
    rec = {"feats": {"enc12_elec": {"raw": torch.from_numpy(elec)}},
           "present_parcels": np.unique(PARCELS), "parcel_canon": PARCELS.copy(),
           "band_lengths": (1, 1, 1)}
    for variant, n_seg in (("A", 3), ("B", 1)):
        seg_cols, _ = ppv._segments(rec, variant)
        z = ppv._pool(rec, np.arange(9), seg_cols, ppv._band_slices(rec, K), 1.0, chunk=4)
        assert z.shape == (9, n_seg, F), "electrode axis pooled; time·band·feature untouched"


def test_variant_b_needs_no_anatomy() -> None:
    rng = np.random.default_rng(5)
    elec = rng.standard_normal((5, N_CONTACTS, F)).astype(np.float32)
    rec = {"feats": {"enc12_elec": {"raw": torch.from_numpy(elec)}}, "band_lengths": (1, 1, 1)}
    seg_cols, seg_atlas = ppv._segments(rec, "B")               # no parcel_canon, no present_parcels
    assert len(seg_cols) == 1 and seg_cols[0].size == N_CONTACTS
    assert seg_atlas.tolist() == [ppv.GLOBAL_ATLAS_ID]
    # the sentinel makes the CS intersection work with no special case
    assert np.intersect1d(seg_atlas, seg_atlas).size == 1


def test_variant_a_without_parcel_canon_is_skipped_not_crashed(capsys) -> None:
    rec = {"feats": {"enc12_elec": {"raw": torch.zeros((3, N_CONTACTS, F))}},
           "band_lengths": (1, 1, 1)}
    assert ppv._prep(rec, "A", 1, 10**9) is None
    assert "parcel_canon" in capsys.readouterr().out


def test_parity_passes_and_catches_a_corrupted_segment_map(tmp_path, capsys) -> None:
    """A parity check that cannot fail is worse than none."""
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(6)
    _make_session(2, 1, "t", str(cache), rng)
    assert ppv._parity(str(cache), "t", (2, 1), chunk=8, n_rows=16) is True
    assert "OK" in capsys.readouterr().out

    bad = torch.load(str(cache / "enc_s2_t1_t.pt"), map_location="cpu", weights_only=False)
    bad["parcel_canon"] = PARCELS[::-1].copy()                 # membership scrambled, sizes kept
    torch.save(bad, str(cache / "enc_s2_t1_bad.pt"))
    assert ppv._parity(str(cache), "bad", (2, 1), chunk=8, n_rows=16) is False
    assert "VIOLATED" in capsys.readouterr().out


def test_parcel_stratification_drops_small_parcels() -> None:
    rec = {"feats": {"enc12_elec": {"raw": torch.zeros((3, N_CONTACTS, F))}},
           "present_parcels": np.unique(PARCELS), "parcel_canon": PARCELS.copy(),
           "band_lengths": (1, 1, 1)}
    _cols, atlas = ppv._segments(rec, "A", min_elecs=2)
    assert atlas.tolist() == [10, 20], "the 1-electrode parcel 30 must be dropped"


def test_end_to_end_all_mode_writes_both_variants(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(7)
    cohort = ((2, 1), (1, 0), (3, 2))
    for subj, trial in cohort:
        _make_session(subj, trial, "t", str(cache), rng)
    monkeypatch.setattr(ppv, "PROBE_COHORT_7", cohort)
    monkeypatch.setattr(ppv, "CS_TEST_SUBJECTS", (1, 3))
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "probe_pool_variants.py", "--mode", "all", "--cache-dir", str(cache), "--tag", "t",
        "--variants", "A,B", "--betas", "0,2", "--chunk", "16", "--out", str(out)])
    ppv.main()

    res = json.loads(out.read_text())
    for variant in ("A", "B"):
        for beta in ("0", "2"):
            key = f"{variant}|b{beta}|std|onset"
            assert key in res["cells"]
            assert np.isfinite(res["cells"][key]["cs_mean"])
            assert np.isfinite(res["cells"][key]["ws_cohort"])
    assert res["parcel_elec_counts"]["S2T1"] == {"10": 3, "20": 2, "30": 1}


def test_betas_must_include_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "probe_pool_variants.py", "--cache-dir", str(tmp_path), "--tag", "t",
        "--betas", "1,2", "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="must include 0"):
        ppv.main()


def test_relative_paths_are_rejected(monkeypatch) -> None:
    """Standing rule: readouts always take ABSOLUTE cache/shard/out paths."""
    monkeypatch.setattr(sys, "argv", [
        "probe_pool_variants.py", "--cache-dir", "relative/dir", "--tag", "t"])
    with pytest.raises(SystemExit, match="ABSOLUTE"):
        ppv.main()
