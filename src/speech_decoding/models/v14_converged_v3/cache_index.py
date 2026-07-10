"""v14_converged_v3 — cache/label file parsers (Phase D1/D2 adapter).

Thin disk-facing helpers the ``load_v3_sessions`` adapter composes into per-session
``V3SessionSpec``s. Pure file-parsing (no braintreebank anatomy calls — those live
in the orchestrator), so unit-testable on synthetic JSON.

  * ``load_bad_window_spans`` / ``index_bad_windows`` — the guard-2 spans from the
    DCC scan output (``bad_windows_s`` = ``[[start_s, end_s], ...]``), keyed by
    ``(subject_id, trial_id)``.
  * ``index_band_cache`` — scan a band's spec-cache dir, mapping each session's
    ``key`` (from the ``{stem}.json`` sidecar) to its ``.npy`` / ``.stats.npz``
    paths + ``ch_names`` (the npy channel/voltage order) + ``total_frames``.
  * ``parse_lof_report`` — the LOF bad-channel report → guard-1 drop set per
    ``(subject_id, trial_id)`` (optional; absent → no drops).
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass


def load_bad_window_spans(path: str) -> list[tuple[float, float]]:
    """A guard-2 span file's ``bad_windows_s`` as ``[(start_s, end_s), ...]``."""
    with open(path) as fh:
        d = json.load(fh)
    return [(float(a), float(b)) for a, b in d.get("bad_windows_s", [])]


def index_bad_windows(span_dir: str) -> dict[tuple[int, int], list[tuple[float, float]]]:
    """Scan a dir of guard-2 span files → ``(subject_id, trial_id)`` → spans."""
    out: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for path in sorted(glob.glob(os.path.join(span_dir, "*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        key = (int(d["subject_id"]), int(d["trial_id"]))
        out[key] = [(float(a), float(b)) for a, b in d.get("bad_windows_s", [])]
    return out


def parse_session_name(name: str) -> tuple[int, int]:
    """``"btbank6_t4"`` → ``(6, 4)`` (subject_id, trial_id)."""
    m = re.fullmatch(r"btbank(\d+)_t(\d+)", name)
    if not m:
        raise ValueError(f"unrecognised session name {name!r}")
    return int(m.group(1)), int(m.group(2))


@dataclass(frozen=True)
class BandCacheEntry:
    npy_path: str
    stats_path: str
    ch_names: tuple[str, ...]
    total_frames: int
    sample_rate: int


def index_band_cache(band_dir: str) -> dict[str, BandCacheEntry]:
    """Map each session ``key`` → its ``BandCacheEntry`` for one band's cache dir.

    Scans ``{stem}.json`` sidecars; a sidecar without a matching ``.npy`` (a
    partial/aborted write) is skipped, never half-indexed. ``ch_names`` is the npy's
    channel (voltage) order, which the sidecar/geometry build treats as the FULL
    session order.
    """
    out: dict[str, BandCacheEntry] = {}
    for meta_path in sorted(glob.glob(os.path.join(band_dir, "*.json"))):
        if meta_path.endswith(".stats.npz"):  # defensive; .npz never matches *.json
            continue
        stem = meta_path[: -len(".json")]
        npy_path = stem + ".npy"
        if not os.path.exists(npy_path):
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        out[str(meta["key"])] = BandCacheEntry(
            npy_path=npy_path,
            stats_path=stem + ".stats.npz",
            ch_names=tuple(meta["ch_names"]),
            total_frames=int(meta["total_frames"]),
            sample_rate=int(meta.get("sample_rate", 32)),
        )
    return out


def parse_lof_report(path: str | None) -> dict[tuple[int, int], set[str]]:
    """LOF report → ``(subject_id, trial_id)`` → bad-channel set (guard-1 drop).

    ``path=None`` (no report fitted) → empty map (no drops). Each ``sessions[]``
    entry carries a ``session`` name (``btbankS_tT``) and ``bad_channels``.
    """
    if path is None:
        return {}
    with open(path) as fh:
        report = json.load(fh)
    out: dict[tuple[int, int], set[str]] = {}
    for s in report.get("sessions", []):
        key = parse_session_name(str(s["session"]))
        out[key] = set(s.get("bad_channels", []))
    return out
