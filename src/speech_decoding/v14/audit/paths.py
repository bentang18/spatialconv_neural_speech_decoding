"""BIDS path helpers for the `#34` phoneme-loading audit.

`ps_bids_root` is loaded from `configs/paths.yaml` so this module is
machine-agnostic. The repo root is resolved from this file's location.
"""
from __future__ import annotations

from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def ps_bids_root() -> Path:
    cfg = yaml.safe_load((repo_root() / "configs" / "paths.yaml").read_text())
    return Path(cfg["ps_bids_root"])


def ps_tokens_csv() -> Path:
    return repo_root() / "data" / "ps_tokens.csv"


def reports_dir() -> Path:
    return repo_root() / "reports" / "phoneme_audit_2026_04_16"


def trial_fif(patient: str) -> Path:
    root = ps_bids_root()
    return (
        root
        / "derivatives"
        / "epoch(CAR)"
        / f"sub-{patient}"
        / "epoch(band)(power)"
        / f"sub-{patient}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"
    )


def phoneme_fif(patient: str) -> Path:
    """Phoneme-level `.fif` (audit-only cross-check per `#18`)."""
    root = ps_bids_root()
    return (
        root
        / "derivatives"
        / "epoch(phonemeLevel)(CAR)"
        / f"sub-{patient}"
        / "epoch(band)(power)"
        / f"sub-{patient}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"
    )


def events_authoritative(patient: str) -> Path:
    """Authoritative events TSV (named `eventsOLD` on disk; confirmed via
    `.fif` alignment + audio-video verification 2026-04-16)."""
    root = ps_bids_root()
    return (
        root
        / f"sub-{patient}"
        / "ieeg"
        / f"sub-{patient}_task-phoneme_acq-01_run-01_eventsOLD.tsv"
    )


def events_stale(patient: str) -> Path:
    """Known-stale events TSV (named `events.tsv` on disk). Used only to
    assert divergence from the authoritative file."""
    root = ps_bids_root()
    return (
        root
        / f"sub-{patient}"
        / "ieeg"
        / f"sub-{patient}_task-phoneme_acq-01_run-01_events.tsv"
    )


def raw_microphone_wav(patient: str) -> Path:
    root = ps_bids_root()
    return (
        root
        / "derivatives"
        / "audio"
        / f"sub-{patient}"
        / "microphone"
        / f"sub-{patient}_task-phoneme_acq-01_run-01_desc-raw_microphone.wav"
    )


def production_events(patient: str) -> Path:
    """Audio-clock production events TSV. `onset` is audio-clock time, `sample`
    is ECoG-clock sample index — the two columns together pin the clock
    relationship. Trial identities in this file are known-buggy (agrees with
    the bad `events.tsv`), but per-row TIMING is trustworthy."""
    root = ps_bids_root()
    return (
        root
        / "derivatives"
        / "audio"
        / f"sub-{patient}"
        / "events"
        / f"sub-{patient}_task-phoneme_acq-01_run-01_desc-production_events.tsv"
    )
