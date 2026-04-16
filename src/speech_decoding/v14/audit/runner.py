"""Per-patient audit orchestration. Loads artifacts, runs every predicate,
assembles the JSON result, and writes diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from speech_decoding.v14.audit import paths
from speech_decoding.v14.audit.audio_checks import run_audio_checks
from speech_decoding.v14.audit.io import (
    load_events,
    load_ps_tokens,
    load_raw_audio,
    load_trial_epochs,
    response_rows,
)
from speech_decoding.v14.audit.schema import AuditResult, Check
from speech_decoding.v14.audit.structural_checks import (
    check_event_id_is_52_ps_tokens,
    check_events_stale_is_divergent,
    check_fif_labels_match_authoritative,
    check_fif_path_exists,
    check_fif_window_contains_target,
    check_no_leaked_tokens,
    check_per_epoch_token_decomposition,
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

    fif_path = paths.trial_fif(patient)
    ev_auth_path = paths.events_authoritative(patient)
    ev_stale_path = paths.events_stale(patient)
    wav_path = paths.raw_microphone_wav(patient)
    ps_csv = paths.ps_tokens_csv()

    checks: list[Check] = []
    workarounds: list[str] = []
    metadata: dict = {
        "patient": patient,
        "fif_path": str(fif_path),
        "events_authoritative_path": str(ev_auth_path),
        "events_stale_path": str(ev_stale_path),
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
    epochs = load_trial_epochs(fif_path)
    ps_tokens = load_ps_tokens(ps_csv)
    ev_auth = load_events(ev_auth_path)
    auth_resp = response_rows(ev_auth)
    ev_stale = load_events(ev_stale_path)
    stale_resp = response_rows(ev_stale)
    audio_sr, audio = load_raw_audio(wav_path)
    prod_events = load_events(paths.production_events(patient))

    metadata.update(
        {
            "n_epochs": len(epochs),
            "tmin_s": float(epochs.tmin),
            "tmax_s": float(epochs.tmax),
            "sfreq_hz": float(epochs.info["sfreq"]),
            "T_samples": int(epochs.times.shape[0]),
            "n_events_auth_response": int(len(auth_resp)),
            "n_events_stale_response": int(len(stale_resp)),
            "n_event_id_keys": int(len(epochs.event_id)),
            "audio_sr_hz": int(audio_sr),
            "audio_duration_s": float(len(audio) / audio_sr),
        }
    )

    # 3. structural checks
    if verbose:
        print(f"[{patient}] structural checks…")
    checks.append(check_fif_window_contains_target(epochs))
    checks.append(check_event_id_is_52_ps_tokens(epochs, ps_tokens))
    checks.append(check_per_epoch_token_decomposition(epochs, ps_tokens))
    checks.append(check_fif_labels_match_authoritative(epochs, auth_resp))
    checks.append(check_no_leaked_tokens(auth_resp, ps_tokens))
    checks.append(check_signal_finite(epochs))
    checks.append(check_events_stale_is_divergent(epochs, stale_resp))

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
                workarounds.append(
                    f"exclude silent-suspect trials at loader time via {audio_meta['exclusion_candidates_csv']}"
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
