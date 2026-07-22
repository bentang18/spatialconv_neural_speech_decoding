"""v14_converged_v3 — ShaftPackDataset + V3DataModule shaft-branch tests.

The cross-patient pack stream must: (1) yield exact-budget packs (overfill-and-trim, no
pad) that collate to a B=1 super-montage with a PINNED grid.total; (2) read the drawn
shaft's real contacts (depth/parcel from the geom, trim = the tail); (3) robust-z with the
frozen stats SLICED to those rows (byte-identical to the full-session transform); and the
datamodule must branch to it under ``batch_unit="shaft"`` (requiring a contact budget).

Reuses ``test_dataset._spec`` (tiny synthetic .npy caches); the real braintreebank adapter
is the DeltaAI-F2 layer, out of scope here.
"""

from __future__ import annotations

import copy

import pytest
import torch

from speech_decoding.models.v14_converged_v3.datamodule import V3DataModule
from speech_decoding.models.v14_converged_v3.dataset import UNIFORM_BAND_RATES
from speech_decoding.models.v14_converged_v3.pack_r4 import band_token_counts, build_r4_grid
from speech_decoding.models.v14_converged_v3.shaft_batch import (
    ShaftClipSample,
    collate_shaft_pack,
)
from speech_decoding.models.v14_converged_v3.shaft_dataset import ShaftPackDataset
from speech_decoding.models.v14_converged_v3.test_dataset import FPS, T_CLIP, _spec


def _two_subjects(tmp):
    # subject 1: 3 shafts (4,3,3); subject 2: 2 shafts (5,5). 17 contacts, 5 shafts.
    return [
        _spec(tmp, key=(1, 0), shaft_sizes=(4, 3, 3)),
        _spec(tmp, key=(2, 0), shaft_sizes=(5, 5)),
    ]


def _pack_ds(tmp, budget, sessions=None, *, alpha=0.5, seed=0, packs=4):
    return ShaftPackDataset(
        sessions if sessions is not None else _two_subjects(tmp),
        contact_budget=budget, clip_frames=T_CLIP, fps=FPS,
        band_rates=UNIFORM_BAND_RATES, packs_per_epoch=packs, alpha=alpha, seed=seed,
    )


# ── pack stream ───────────────────────────────────────────────────────────────────
def test_pack_stream_yields_exact_budget_super_montage(tmp_path) -> None:
    ds = _pack_ds(tmp_path, budget=10)
    pack = next(iter(ds))
    assert all(isinstance(x, ShaftClipSample) for x in pack)
    tot = sum(x.bands[0].shape[0] for x in pack)
    assert tot == 10                                       # overfill-and-trim ⇒ exact budget
    batch = collate_shaft_pack(pack)
    assert batch.bands[0].shape == (1, 10, 7, T_CLIP)      # B=1 super-montage (F_SLOW=7)
    k_full = sum(band_token_counts(T_CLIP))
    assert build_r4_grid(batch.geom, n_time=T_CLIP).total == 10 * k_full  # grid.total pinned
    assert batch.session_key == ("shaft_pack", 10)
    print(f"[check] pack stream: ΣN=10 super-montage, grid.total=10×{k_full} OK")


def test_stream_pins_grid_total_across_every_step(tmp_path) -> None:
    ds = _pack_ds(tmp_path, budget=12, packs=5)
    k_full = sum(band_token_counts(T_CLIP))
    seen = 0
    for pack in ds:
        batch = collate_shaft_pack(pack)
        assert build_r4_grid(batch.geom, n_time=T_CLIP).total == 12 * k_full
        assert batch.session_key == ("shaft_pack", 12)
        seen += 1
    assert seen == 5
    print(f"[check] stream: {seen} steps all grid.total==12×{k_full} (one compiled shape) OK")


def test_shaft_read_pulls_the_drawn_shafts_contacts(tmp_path) -> None:
    # ONE session, ONE shaft (depths 1..4, parcel 0) ⇒ every draw is that shaft; budget 6 ⇒
    # pack = full shaft (4) + trimmed (2). Depth (index-RoPE) + parcel come from the geom, and
    # the trim keeps the depth-order PREFIX (LA1,LA2), never fabricates contacts.
    s = _spec(tmp_path, key=(3, 0), shaft_sizes=(4,))
    ds = ShaftPackDataset([s], contact_budget=6, clip_frames=T_CLIP, fps=FPS,
                          band_rates=UNIFORM_BAND_RATES, packs_per_epoch=1, seed=1)
    pack = next(iter(ds))
    assert sum(x.bands[0].shape[0] for x in pack) == 6
    depths = sorted(tuple(x.depth.tolist()) for x in pack)
    assert depths == [(1, 2), (1, 2, 3, 4)]                # full shaft + depth-order trim
    for x in pack:
        n = x.bands[0].shape[0]
        assert x.depth.shape[0] == n and x.parcel_id.shape[0] == n
        assert x.parcel_id.tolist() == [0] * n             # shaft 0 → parcel 0
    print(f"[check] shaft read: depths {depths}, trim = depth-order prefix OK")


def test_sliced_normalizer_matches_full_on_the_same_rows(tmp_path) -> None:
    # the per-shaft robust-z must equal the session normalizer's transform on those rows —
    # the slicing the dataset relies on (median/sigma indexed by contact id).
    s = _spec(tmp_path, key=(9, 0), shaft_sizes=(5, 4))
    norm = s.band_norms[0]
    n, f = norm.median.shape[0], norm.median.shape[1]
    x = torch.randn(n, f, 20)
    z_full = norm.transform(x)
    cids = torch.tensor([0, 3, 6])
    sn = copy.copy(norm)
    sn.median, sn.sigma = norm.median[cids], norm.sigma[cids]
    assert torch.allclose(sn.transform(x[cids]), z_full[cids], atol=1e-6)
    print("[check] sliced normalizer == full transform on the same rows OK")


# ── datamodule branch ───────────────────────────────────────────────────────────────
def test_datamodule_shaft_requires_a_contact_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="contact_budget"):
        V3DataModule(
            _two_subjects(tmp_path), batch_size=4, clips_per_session=2,
            clip_frames=T_CLIP, fps=FPS, batch_unit="shaft",
        )


def test_datamodule_shaft_loader_yields_pinned_super_montages(tmp_path) -> None:
    dm = V3DataModule(
        _two_subjects(tmp_path), batch_size=4, clips_per_session=2, clip_frames=T_CLIP,
        fps=FPS, batch_unit="shaft", contact_budget=8, num_workers=0,
    )
    k_full = sum(band_token_counts(T_CLIP))
    seen = 0
    for batch in dm.train_dataloader():
        assert build_r4_grid(batch.geom, n_time=T_CLIP).total == 8 * k_full
        assert not hasattr(batch, "stat_mean") and not hasattr(batch, "stat_std")  # secondary PURGED
        seen += 1
        if seen >= 3:
            break
    assert seen >= 1
    print(f"[check] datamodule shaft loader: {seen} pinned super-montages (8×{k_full}) OK")
