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
    """Trial-level `.fif`. Only S14 has this upstream as of 2026-04-16."""
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
    """Phoneme-level `.fif`. Position-0 (every 3rd starting at 0) epochs lock
    to the same raw sample as the trial-level response onset (verified on S14:
    `phon.events[0::3, 0] == trial.events[:, 0]` exact). Used as the Phase-1
    stand-in for trial-level epochs everywhere trial-level is unavailable."""
    root = ps_bids_root()
    return (
        root
        / "derivatives"
        / "epoch(phonemeLevel)(CAR)"
        / f"sub-{patient}"
        / "epoch(band)(power)"
        / f"sub-{patient}_task-PhonemeSequence_desc-productionZscore_highgamma.fif"
    )


def _events_old(patient: str) -> Path:
    root = ps_bids_root()
    return (
        root
        / f"sub-{patient}"
        / "ieeg"
        / f"sub-{patient}_task-phoneme_acq-01_run-01_eventsOLD.tsv"
    )


def _events_plain(patient: str) -> Path:
    root = ps_bids_root()
    return (
        root
        / f"sub-{patient}"
        / "ieeg"
        / f"sub-{patient}_task-phoneme_acq-01_run-01_events.tsv"
    )


def events_authoritative(patient: str) -> Path:
    """Authoritative events TSV. Prefer `eventsOLD.tsv` when present (S14:
    upstream replaced the original with a broken `events.tsv` but kept the
    original as `eventsOLD`). Fall back to `events.tsv` for the ~10 patients
    where no OLD backup exists — verified 2026-04-16 that its `sample` column
    aligns with phoneme-level `.fif` at ≤5 ms for every core patient except
    S26 (which is a separate data blocker)."""
    old = _events_old(patient)
    return old if old.exists() else _events_plain(patient)


def events_stale(patient: str) -> Path | None:
    """Known-stale events TSV if one exists. Returns `None` when the patient
    has no supersession history (only `events.tsv`, which is authoritative
    for them)."""
    old = _events_old(patient)
    return _events_plain(patient) if old.exists() else None


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
