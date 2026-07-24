"""Ground-truth tests for the enc12 surplus-rank subspace probe.

The probe's whole job is to say WHERE in the spectrum information lives, so the tests plant
known structure at known ranks and check it is recovered there and nowhere else: task signal
on the TOP component, parcel-identity nuisance on a TAIL component. A probe that reported the
same answer for every band would pass a smoke test and fail these.

Also pins the one load-bearing algebraic claim in the module — that projecting the
parcel-POOLED cache equals projecting per-contact and then pooling — since that identity is
why the downstream leg needs no contact->parcel assignment.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probe_subspace_surplus as psp  # noqa: E402
from v3_probe_readout_r4 import PROBE_TASKS  # noqa: E402

D = 256
TOP_COMPONENT = 0                          # task signal here      -> band [0, 64)
TAIL_COMPONENTS = np.arange(200, 208)      # parcel identity here  -> band [192, 256)
N_PARCELS, K_TIME = 5, 2


def _orthonormal(d: int, rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return q


def _make_session(subject: int, trial: int, tag: str, out_dir: str, rng,
                  basis: np.ndarray, n: int = 140) -> None:
    """Cache with task info on a TOP component and parcel identity on a TAIL component.

    Amplitudes are planted as three well-SEPARATED tiers rather than a smooth power law. With a
    smooth decay the tail amplitudes differ by ~20% while finite-sample SVD noise at these row
    counts is larger than that, so tail ordering scrambles and band membership stops being known
    a priori — the fixture, not the probe, would be the thing under test. Tier gaps of 4-5x sit
    far above sampling noise, so component -> band is deterministic. Within a tier the
    amplitudes are degenerate and the SVD returns an arbitrary rotation, which is fine: the
    planted direction still lies in that tier's SPAN, which is what a band probe reads.

    The tail tier is ~12% of the top tier, comfortably above the fp16 storage floor (~5e-4
    relative) that the real caches also impose.

    `basis` is SHARED across sessions on purpose. Real enc12 features come from one set of
    encoder weights, so feature dimension j means the same thing in every session — which is
    exactly what lets the probe estimate V on one session and apply it to the rest. Giving each
    session its own rotation would break that premise and make the cross-session average
    meaningless.
    """
    y = rng.integers(0, 2, size=n).astype(np.float64) * 2.0 - 1.0     # per-window task label
    rows = n * N_PARCELS * K_TIME
    a = rng.standard_normal((rows, D))

    win_of_row = np.repeat(np.arange(n), N_PARCELS * K_TIME)
    parcel_of_row = np.tile(np.repeat(np.arange(N_PARCELS), K_TIME), n)
    a[:, TOP_COMPONENT] = y[win_of_row]
    # Parcel identity as a MULTI-dimensional embedding, not a scalar code: one-vs-rest AUROC on
    # a single monotone scalar can only separate the extreme classes, so a scalar plant would
    # cap the recoverable macro-AUROC near chance for the middle parcels regardless of probe.
    embed = rng.standard_normal((N_PARCELS, len(TAIL_COMPONENTS)))
    embed = (embed - embed.mean(0)) / embed.std(0)
    a[:, TAIL_COMPONENTS] = embed[parcel_of_row]

    amp = np.full(D, 0.05)              # tier 3 (tail)  dims 192..255 -> band [192,256)
    amp[:64] = 1.0                      # tier 1 (top)   dims   0.. 63 -> band [0,64)
    amp[64:192] = 0.2                   # tier 2 (middle)
    amp[TAIL_COMPONENTS] = 0.12         # rank above their tier-3 mates, still below tier 2
    z = (a * amp) @ basis.T
    feat = torch.from_numpy(
        z.reshape(n, N_PARCELS, K_TIME * D).astype(np.float32)).to(torch.float16)

    half = n // 2
    tr, te = np.arange(half), np.arange(half, n)
    labels, ws_split, cs_split = {}, {}, {}
    for t in PROBE_TASKS:
        labels[t] = y.copy()
        ws_split[t] = {"fold0": {"train": tr, "test": te}}
        cs_split[t] = {"test": te}

    payload = {
        "subject_id": subject, "trial_id": trial, "ckpt_tag": tag,
        "present_parcels": np.arange(N_PARCELS, dtype=np.int64),
        "band_lengths": (K_TIME,),
        "feats": {"enc12": {"raw": feat}},
        "clip_starts": np.arange(n), "labels": labels,
        "ws_split": ws_split, "cs_split": cs_split, "n_windows": n,
    }
    torch.save(payload, os.path.join(out_dir, f"enc_s{subject}_t{trial}_{tag}.pt"))


def test_pooling_commutes_with_band_projection() -> None:
    """pool(x @ V) == pool(x) @ V — the identity that lets the downstream leg use the pooled
    cache and skip the contact->parcel assignment the caches never stored."""
    rng = np.random.default_rng(0)
    n, n_contacts, k = 6, 9, 3
    x = torch.from_numpy(rng.standard_normal((n, n_contacts, k, D)).astype(np.float32))
    v = torch.from_numpy(rng.standard_normal((D, 16)).astype(np.float32))
    assign = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])

    proj_then_pool, pool_then_proj = [], []
    for p in np.unique(assign):
        sub = x[:, assign == p]
        proj_then_pool.append(torch.einsum("nckd,de->ncke", sub, v).mean(1))
        pool_then_proj.append(torch.einsum("nkd,de->nke", sub.mean(1), v))
    assert torch.allclose(torch.stack(proj_then_pool), torch.stack(pool_then_proj), atol=1e-4)


def test_subspace_overlap_endpoints() -> None:
    rng = np.random.default_rng(1)
    q = _orthonormal(D, rng)
    assert psp._subspace_overlap(q, q, 32) == pytest.approx(1.0, abs=1e-6)
    # a subspace vs a disjoint block of the same basis is exactly orthogonal
    assert psp._subspace_overlap(q, np.roll(q, -32, axis=1), 32) == pytest.approx(0.0, abs=1e-6)


def test_spectrum_recovers_planted_decay() -> None:
    rng = np.random.default_rng(2)
    rows = 4000
    a = rng.standard_normal((rows, D))
    z = (a * ((1.0 + np.arange(D)) ** -0.7)) @ _orthonormal(D, rng).T
    sp = psp._spectrum(z)
    assert sp["dims_50pct_energy"] < sp["dims_90pct_energy"] < sp["dims_99pct_energy"]
    assert 1.0 < sp["rankme_centered"] <= D


def test_bands_separate_planted_task_signal_from_tail_nuisance(tmp_path, monkeypatch,
                                                               capsys) -> None:
    """Top band must carry the task and NOT parcel identity; the tail band the reverse."""
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(7)
    basis = _orthonormal(D, np.random.default_rng(99))
    cohort = ((2, 1), (1, 0), (3, 2))
    for subj, trial in cohort:
        _make_session(subj, trial, "arm", str(cache), rng, basis)

    monkeypatch.setattr(psp, "PROBE_COHORT_7", cohort)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "probe_subspace_surplus.py",
        "--arm", f"a={cache}:arm",
        "--band-width", "64", "--spectrum-session", "2,1",
        "--skip-cs", "--out", str(out),
    ])
    psp.main()
    capsys.readouterr()

    import json
    res = json.loads(out.read_text())
    bands = res["arms"]["a"]["bands"]
    top, tail = bands["[0,64)"], bands["[192,256)"]

    top_task = float(np.nanmean(list(top["ws"].values())))
    tail_task = float(np.nanmean(list(tail["ws"].values())))
    assert top_task > 0.80, f"planted task signal not recovered in top band: {top_task}"
    assert top_task > tail_task + 0.15, f"bands not separated: top={top_task} tail={tail_task}"

    assert tail["nuisance"]["parcel_id"] > 0.75, "planted tail nuisance not recovered"
    assert tail["nuisance"]["parcel_id"] > top["nuisance"]["parcel_id"], \
        "parcel identity should localize to the tail band where it was planted"


def test_json_carries_geometry_and_config(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(11)
    basis = _orthonormal(D, np.random.default_rng(99))
    cohort = ((2, 1), (1, 0))
    for subj, trial in cohort:
        for tag in ("armA", "armB"):
            _make_session(subj, trial, tag, str(cache), rng, basis)

    monkeypatch.setattr(psp, "PROBE_COHORT_7", cohort)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "probe_subspace_surplus.py",
        "--arm", f"A={cache}:armA", "--arm", f"B={cache}:armB",
        "--band-width", "128", "--spectrum-session", "2,1",
        "--skip-cs", "--out", str(out),
    ])
    psp.main()

    import json
    res = json.loads(out.read_text())
    assert set(res["arms"]) == {"A", "B"}
    assert "A|B" in res["geometry"]
    g = res["geometry"]["A|B"]
    assert 0.0 <= g["overlap"]["top32"] <= 1.0
    assert 0.0 <= g["cka"] <= 1.0
    for arm in ("A", "B"):
        assert set(res["arms"][arm]["bands"]) == {"[0,128)", "[128,256)"}


def test_relative_cache_dir_is_rejected(tmp_path, monkeypatch) -> None:
    """Standing rule: readouts always take ABSOLUTE cache/shard/out paths."""
    monkeypatch.setattr(sys, "argv", [
        "probe_subspace_surplus.py", "--arm", "a=relative/dir:tag",
        "--out", str(tmp_path / "x.json"),
    ])
    with pytest.raises(SystemExit):
        psp.main()
