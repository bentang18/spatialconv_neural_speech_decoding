"""Per-patient audit orchestration. Loads artifacts, runs every predicate,
assembles the JSON result, and writes diagnostics.

Data source is phoneme-level `.fif` (trial-level is S14-only upstream as of
2026-04-16). Pos-0 phoneme epochs stand in for trial-level epochs everywhere
a response-locked epoch is needed — verified identical at raw-sample
resolution on S14.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from speech_decoding.v14.audit import paths
from speech_decoding.v14.audit.audio_checks import run_audio_checks
from speech_decoding.v14.audit.io import (
    load_events,
    load_phoneme_epochs,
    load_ps_tokens,
    load_raw_audio,
    response_rows,
)
from speech_decoding.v14.audit.schema import AuditResult, Check
from speech_decoding.v14.audit.structural_checks import (
    check_event_id_is_9_ps_phonemes,
    check_events_stale_is_divergent,
    check_fif_labels_match_authoritative,
    check_fif_path_exists,
    check_fif_window_contains_target,
    check_no_leaked_tokens,
    check_per_trial_token_reconstruction,
    check_signal_finite,
)
from speech_decoding.v14.audit.timing_checks import (
    check_epoch_t0_equals_response_onset,
    check_fif_samples_match_authoritative,
)
from speech_decoding.v14.audit.verdict import assemble_result


def run_patient_audit(patient: str, verbose: bool = True) -> AuditResult:
    out_dir = paths.reports_dir()
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fif_path = paths.phoneme_fif(patient)
    ev_auth_path = paths.events_authoritative(patient)
    ev_stale_path = paths.events_stale(patient)
    wav_path = paths.raw_microphone_wav(patient)
    ps_csv = paths.ps_tokens_csv()

    checks: list[Check] = []
    workarounds: list[str] = []
    metadata: dict = {
        "patient": patient,
        "fif_path": str(fif_path),
        "fif_level": "phoneme (pos-0 used as trial stand-in)",
        "events_authoritative_path": str(ev_auth_path),
        "events_authoritative_kind": (
            "eventsOLD" if ev_auth_path.name.endswith("_eventsOLD.tsv") else "events"
        ),
        "events_stale_path": str(ev_stale_path) if ev_stale_path else None,
        "raw_microphone_wav": str(wav_path),
    }

    # 1. path
    path_check = check_fif_path_exists(fif_path)
    checks.append(path_check)
    if not path_check["passed"]:
        return assemble_result(patient, checks, metadata, workarounds)

    # 2. load everything
    if verbose:
        print(f"[{patient}] loading artifacts…")
    epochs = load_phoneme_epochs(fif_path)
    if epochs is None:
        checks.append(
            {
                "name": "fif_load",
                "level": "must",
                "passed": False,
                "detail": f"load_phoneme_epochs returned None for {fif_path}",
            }
        )
        return assemble_result(patient, checks, metadata, workarounds)
    ps_tokens = load_ps_tokens(ps_csv)
    ev_auth = load_events(ev_auth_path)
    auth_resp = response_rows(ev_auth)
    stale_resp = (
        response_rows(load_events(ev_stale_path)) if ev_stale_path else None
    )
    audio_sr, audio = load_raw_audio(wav_path)
    prod_events_path = paths.production_events(patient)
    prod_events = None
    if prod_events_path.exists() and prod_events_path.stat().st_size > 16:
        # Some patients (e.g. S22) have a 1-byte placeholder — treat as missing.
        try:
            prod_events = load_events(prod_events_path)
        except Exception:
            prod_events = None

    n_phon_epochs = len(epochs)
    n_trials = n_phon_epochs // 3
    metadata.update(
        {
            "n_phoneme_epochs": n_phon_epochs,
            "n_trials": n_trials,
            "tmin_s": float(epochs.tmin),
            "tmax_s": float(epochs.tmax),
            "sfreq_hz": float(epochs.info["sfreq"]),
            "T_samples": int(epochs.times.shape[0]),
            "n_events_auth_response": int(len(auth_resp)),
            "n_events_stale_response": int(len(stale_resp)) if stale_resp is not None else None,
            "n_event_id_keys": int(len(epochs.event_id)),
            "audio_sr_hz": int(audio_sr),
            "audio_duration_s": float(len(audio) / audio_sr),
        }
    )

    # 3. structural checks
    if verbose:
        print(f"[{patient}] structural checks…")
    checks.append(check_fif_window_contains_target(epochs))
    checks.append(check_event_id_is_9_ps_phonemes(epochs))
    checks.append(check_per_trial_token_reconstruction(epochs, ps_tokens))
    checks.append(check_fif_labels_match_authoritative(epochs, auth_resp, ps_tokens))
    checks.append(check_no_leaked_tokens(auth_resp, ps_tokens))
    checks.append(check_signal_finite(epochs))
    if stale_resp is not None:
        checks.append(check_events_stale_is_divergent(epochs, stale_resp, ps_tokens))

    # 4. timing checks
    if verbose:
        print(f"[{patient}] timing checks…")
    checks.append(check_fif_samples_match_authoritative(epochs, auth_resp))
    checks.append(check_epoch_t0_equals_response_onset(epochs, auth_resp))

    # 5. audio + audio-neural alignment
    if verbose:
        print(f"[{patient}] audio-neural alignment…")
    audio_checks, audio_meta = run_audio_checks(
        patient=patient,
        audio=audio,
        audio_sr=audio_sr,
        authoritative_resp=auth_resp,
        production_events=prod_events,
        plots_dir=plots_dir,
        exclusions_csv=out_dir / f"{patient}_exclusion_candidates.csv",
    )
    checks.extend(audio_checks)
    metadata["audio"] = audio_meta

    # 6. workarounds list (populated from soft-warn checks that have a known fix)
    for c in checks:
        if c["level"] == "soft" and not c["passed"]:
            if c["name"] == "silent_trial_fraction_low":
                csv = audio_meta.get("exclusion_candidates_csv")
                if csv:
                    workarounds.append(
                        f"exclude silent-suspect trials at loader time via {csv}"
                    )
            elif c["name"] == "events_stale_is_divergent":
                workarounds.append("ignore events.tsv; use eventsOLD.tsv as authoritative")

    return assemble_result(patient, checks, metadata, workarounds)


def write_result(result: AuditResult) -> Path:
    out_path = paths.reports_dir() / f"{result['patient']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_jsonify(result), indent=2))
    return out_path


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj
