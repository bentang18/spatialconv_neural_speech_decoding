"""Streaming |robust-z| histogram for per-band WINSOR / CLIP threshold profiling.

The bad-electrode WINSOR cap and the CLIP bad-window thresholds are both set from
the tail of the post-STATIC ``|robust-z|`` distribution per band (slow/beta/hg).
That distribution has O(10^9) cells per band across the corpus — too many to hold
in memory — so we accumulate a fixed-edge histogram as we stream each session-band
memmap chunk, then read percentiles off the cumulative counts.

Edge design: 0.1-wide linear bins up to 100 (real high-gamma/beta power lives here,
so the real-signal ceiling needs fine resolution) and ~1.15x-geometric bins from 100
to 30000 (the artifact tail, where relative resolution is what matters). The top edge
(30000) sits well above the largest observed |z| (~1.5e4 on the static_fixed cache),
so the overflow bin is empty in practice; ``overflow``/``max_val`` are exposed so a
caller can assert that.

Percentiles are exact to within one bin width. Used by
``scripts/neuroprobe/profile_band_robustz.py`` (the cache-read profiler) and
reusable by the CLIP re-tune (#231).
"""
from __future__ import annotations

import numpy as np


def make_edges() -> np.ndarray:
    """Linear 0.1-wide bins on [0, 100], geometric ~1.15x bins on [100, 30000]."""
    lin = np.linspace(0.0, 100.0, 1001)            # 0.1 width, real-signal region
    logp = np.geomspace(100.0, 30000.0, 500)       # log tail, artifact region
    return np.unique(np.concatenate([lin, logp]))


class RobustZHistogram:
    """Memory-bounded |z| distribution accumulated over streamed chunks.

    ``update`` takes the abs already or raw values (abs is applied internally), so
    a caller can pass signed robust-z. Percentiles read off the cumulative counts.
    """

    def __init__(self, edges: np.ndarray | None = None) -> None:
        self.edges = make_edges() if edges is None else np.asarray(edges, dtype=np.float64)
        if self.edges.ndim != 1 or self.edges.size < 2:
            raise ValueError("edges must be a 1-D array with >= 2 entries")
        if not np.all(np.diff(self.edges) > 0):
            raise ValueError("edges must be strictly increasing")
        self.counts = np.zeros(self.edges.size - 1, dtype=np.int64)
        self.overflow = 0  # count of |z| >= edges[-1] (above the top edge)
        self.total = 0
        self.max_val = 0.0

    def update(self, values: np.ndarray) -> None:
        v = np.abs(np.asarray(values, dtype=np.float64)).ravel()
        if v.size == 0:
            return
        self.total += int(v.size)
        self.max_val = max(self.max_val, float(v.max()))
        over = v >= self.edges[-1]
        n_over = int(over.sum())
        self.overflow += n_over
        if n_over < v.size:
            c, _ = np.histogram(v[~over] if n_over else v, bins=self.edges)
            self.counts += c

    def percentile(self, q: float) -> float:
        """The q-th percentile (q in [0, 100]) of |z|, linear-interpolated within
        the crossing bin. Returns the top edge as a floor if q falls in the
        (normally empty) overflow region."""
        if self.total == 0:
            return float("nan")
        target = (q / 100.0) * self.total
        cum = np.cumsum(self.counts)
        total_in_bins = int(cum[-1]) if cum.size else 0
        if target > total_in_bins:
            # q lands in the overflow region (|z| >= top edge). No sub-bin info;
            # the top edge is a conservative lower bound. Callers should check
            # ``overflow`` and fall back to ``max_val`` when this fires.
            return float(self.edges[-1])
        i = int(np.searchsorted(cum, target, side="left"))
        i = min(i, self.counts.size - 1)
        below = int(cum[i - 1]) if i > 0 else 0
        c_i = int(self.counts[i])
        if c_i == 0:
            return float(self.edges[i])
        frac = (target - below) / c_i
        return float(self.edges[i] + frac * (self.edges[i + 1] - self.edges[i]))

    def percentiles(self, qs: tuple[float, ...] = (50, 90, 99, 99.9, 99.99)) -> dict[float, float]:
        return {q: self.percentile(q) for q in qs}

    def summary(self) -> dict:
        d: dict = {f"p{q}": v for q, v in self.percentiles().items()}
        d["max"] = self.max_val
        d["total"] = self.total
        d["overflow"] = self.overflow
        return d
