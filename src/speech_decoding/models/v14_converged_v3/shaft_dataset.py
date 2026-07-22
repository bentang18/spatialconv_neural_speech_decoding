"""v14_converged_v3 — ShaftPackDataset: the shaft-level (cross-patient) pack stream.

Training-side counterpart to ``shaft_batch.py`` (pool / temperature sampler / collate).
Where ``V3SessionDataset`` yields one whole-session clip (session-homogeneous batching),
this ``IterableDataset`` yields one CROSS-PATIENT PACK per step: a temperature-drawn set
of shafts (``draw_pack_to_budget``, overfill-and-trim to an EXACT contact budget), each
read as its own shaft-clip and unioned by ``collate_shaft_pack`` into a B=1 super-montage.
The model forward is untouched — L1-only shaft-local towers make this correctness-neutral
(``test_shaft_pack_isolation``).

Per-shaft read (only the drawn shaft's rows are touched — never the whole session): take
the first ``n_keep`` valid contacts of the shaft (depth order; the trim drops the tail),
map them to their C_full memmap rows via ``keep_idx``, mmap-read just those rows over a
fresh random-continuous clip window, and robust-z with the session normalizer SLICED to
those rows (exact reuse of the frozen per-(elec,bin) stats).

Each ``__next__`` yields a ``list[ShaftClipSample]``; the loader runs ``batch_size=None``
with ``collate_fn=collate_shaft_pack`` so one pack → one ``V3Batch``. DDP/worker streams
are seeded disjointly (rank × workers) so every rank draws the SAME shape (fixed budget),
DIFFERENT data — no straggler, the G× diversity the shaft pool exists for.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from speech_decoding.models.v14_converged_v3.clip_sampler import sample_clip_start
from speech_decoding.models.v14_converged_v3.dataset import V3SessionSpec, _start_align
from speech_decoding.models.v14_converged_v3.shaft_batch import (
    ShaftClipSample,
    ShaftRef,
    TemperatureShaftSampler,
    build_shaft_pool,
)


class ShaftPackDataset(IterableDataset):
    """Stream of cross-patient shaft packs (one ``list[ShaftClipSample]`` per step)."""

    def __init__(
        self,
        sessions: Sequence[V3SessionSpec],
        *,
        contact_budget: int,
        clip_frames: int,
        fps: float,
        band_rates: Sequence[tuple[int, int]],
        packs_per_epoch: int,
        alpha: float = 0.5,
        seed: int = 0,
        subject_of: Callable[[object], object] | None = None,
    ) -> None:
        self.sessions = list(sessions)
        self.pool = build_shaft_pool(self.sessions, subject_of=subject_of)
        self.sampler = TemperatureShaftSampler(self.pool, alpha=alpha)
        self.contact_budget = int(contact_budget)
        self.clip_frames = int(clip_frames)
        self.fps = float(fps)
        self.band_rates = tuple((int(n), int(d)) for n, d in band_rates)
        self.start_align = _start_align(self.band_rates)
        self.packs_per_epoch = int(packs_per_epoch)
        self._seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _stream(self) -> tuple[int, int]:
        """This (rank, worker) stream's disjoint id + the total stream count, so every
        stream seeds differently ⇒ same shapes, different packs (the DDP no-straggler win)."""
        wi = get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi is not None else (0, 1)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        else:
            rank, world = 0, 1
        return rank * nw + wid, world * nw

    def _load_shaft_clip(
        self, ref: ShaftRef, n_keep: int, g: torch.Generator
    ) -> ShaftClipSample:
        s = self.sessions[ref.session_idx]
        geom = s.setup.geom
        cols = geom.valid[ref.shaft_id].nonzero(as_tuple=True)[0][:n_keep]  # depth-order slots
        cids = geom.gather_idx[ref.shaft_id, cols]  # (n_keep,) contact index into N-order
        depth = geom.depth[ref.shaft_id, cols].clone()  # clinical depths (index-RoPE coord)
        parcel = s.setup.parcel_id[cids].clone()  # (n_keep,) DKT tags
        rows = s.keep_idx[cids].numpy()  # (n_keep,) rows into the .npy C_full dim

        t0 = sample_clip_start(
            n_frames=s.n_frames, clip_frames=self.clip_frames, bad_spans_s=s.bad_spans_s,
            fps=self.fps, generator=g, start_align=self.start_align,
        )
        end = t0 + self.clip_frames
        bands: list[torch.Tensor] = []
        for path, norm, (num, den) in zip(s.band_paths, s.band_norms, self.band_rates):
            lo, hi = t0 * num // den, end * num // den  # native-rate band window
            mm = np.load(path, mmap_mode="r")  # (C_full, F_band, T_band)
            clip = np.asarray(mm[rows, :, lo:hi], dtype=np.float32)  # (n_keep, F, T)
            del mm
            # robust-z with the frozen per-(elec,bin) stats SLICED to this shaft's rows —
            # byte-identical to V3SessionDataset's transform on the same rows (the caches
            # carry no valid_bin_mask, so a shallow copy + median/sigma slice is exact).
            assert norm._valid_bin_mask is None, "sliced-norm path assumes no valid_bin_mask"
            sn = copy.copy(norm)
            sn.median = norm.median[cids]
            sn.sigma = norm.sigma[cids]
            bands.append(sn.transform(torch.from_numpy(clip)))
        return ShaftClipSample(
            bands=tuple(bands), depth=depth, parcel_id=parcel,
            session_key=s.session_key, shaft_id=int(ref.shaft_id),
        )

    def __iter__(self):
        stream, n_streams = self._stream()
        g = torch.Generator().manual_seed(
            (self._seed * 1_000_003 + self._epoch * 10_000_019 + stream * 2_654_435_761)
            & 0x7FFFFFFF
        )
        my_packs = max(1, self.packs_per_epoch // n_streams)
        for _ in range(my_packs):
            draws = self.sampler.draw_pack_to_budget(g, contact_budget=self.contact_budget)
            yield [self._load_shaft_clip(ref, nk, g) for ref, nk in draws]


__all__ = ["ShaftPackDataset"]
