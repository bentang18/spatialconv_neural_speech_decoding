"""Structural predicates for the `#34` audit.

## Authoritativeness

The `.fif` is the authoritative source of signal, labels, and per-trial
response onsets for the Phase-1 loader. Zac's preprocessing writes labels
into `event_id` + per-epoch codes and response onsets into `events[:, 0]`;
these are self-contained and self-validating (per-trial 3 phonemes → unique
PS token).

The BIDS events TSV (`eventsOLD.tsv` or `events.tsv`) is a **cross-check**
that was originally derived from the same trial table but can drift
independently from the `.fif` upstream (regeneration bugs, accidental
row drops). S14 and S26 are concrete examples — in both cases the `.fif`
is correct and internally consistent but the events TSV is broken.

Therefore:
- **must-pass**: invariants the `.fif` must satisfy on its own.
- **soft**: cross-checks between the `.fif` and the events TSV. A mismatch
  means the events TSV is out-of-sync; the loader can still use the `.fif`.

Data shape: phoneme-level `.fif` (one epoch per phoneme; shape
`(3*n_trials, n_ch, T)`). Position-0 phoneme epochs — `epochs.events[0::3]` —
lock to the same raw sample as the trial response onset (verified on S14:
exact equality with trial-level `.fif` at raw-sample resolution).

Covers:
- must: file existence; window/sfreq/T; `event_id` = 9 PS phonemes;
  per-trial 3-phoneme → unique PS token; signal finite.
- soft: fif labels vs events TSV `value`; fif pos-0 sample vs events TSV
  `sample`; fif t=0 vs events TSV `onset`; events TSV token set ⊆ PS;
  events.tsv trial identities diverge from `.fif` (when eventsOLD exists).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from speech_decoding.data.phoneme_map import normalize_label
from speech_decoding.v14.audit.schema import Check, make_check

# #29: target loader window (seconds). Relaxed 2026-04-16 during S14 pilot:
# trial-level `.fif` stores `tmax=0.995` (one sample short of 1.0 at 200 Hz
# because tmax is the LAST-sample time, not the right edge). Phoneme-level
# has a wider window `[-1.0, +1.495]` which trivially contains this.
TARGET_TMIN = -0.5
TARGET_TMAX = 0.995
TARGET_T_AT_200HZ = 300
TARGET_SFREQ = 200.0

# PS ARPABET phoneme inventory (per `#17`, alphabetical per `#16`).
PS_PHONEMES_9 = ("AA", "AE", "B", "G", "IY", "K", "P", "UW", "V")


def check_fif_path_exists(fif_path, label: str = "fif_path_exists") -> Check:
    return make_check(
        label,
        "must",
        fif_path.exists(),
        f"{fif_path}",
    )


def check_fif_window_contains_target(epochs) -> Check:
    """Predicate #2: file window must ⊇ `#29` target `[-0.5, 0.995]`."""
    tmin, tmax, sfreq = epochs.tmin, epochs.tmax, epochs.info["sfreq"]
    ok = (
        tmin <= TARGET_TMIN + 1e-6
        and tmax + 1e-6 >= TARGET_TMAX
        and abs(sfreq - TARGET_SFREQ) < 1e-6
    )
    return make_check(
        "fif_window_contains_target",
        "must",
        ok,
        f"tmin={tmin:.3f} tmax={tmax:.3f} sfreq={sfreq} — "
        f"target ⊆ [{TARGET_TMIN}, {TARGET_TMAX}] @ {TARGET_SFREQ}Hz",
    )


def _id_to_phoneme_map(epochs) -> dict[int, str]:
    return {int(v): str(k) for k, v in epochs.event_id.items()}


def _reconstruct_trial_tokens(
    epochs, ps_tokens: pd.DataFrame
) -> tuple[list[str | None], list[tuple[int, list[str], str]]]:
    """For each trial, look up the PS token whose `arpabet_sequence` matches
    the 3 consecutive phoneme labels. Returns (tokens_or_none, failures).
    Failures is a list of `(trial_idx, raw_labels, reason)` tuples.
    """
    id2phon = _id_to_phoneme_map(epochs)
    codes = epochs.events[:, 2]
    if len(codes) % 3 != 0:
        return [], [(0, [], f"n_epochs={len(codes)} not divisible by 3")]
    n_trials = len(codes) // 3
    seq_to_tok = dict(zip(ps_tokens["arpabet_sequence"], ps_tokens["ps_notation"]))
    tokens: list[str | None] = []
    failures: list[tuple[int, list[str], str]] = []
    for i in range(n_trials):
        raw = [id2phon[int(codes[i * 3 + j])] for j in range(3)]
        try:
            arpa = [normalize_label(x) for x in raw]
        except ValueError as e:
            tokens.append(None)
            failures.append((i, raw, f"normalize_label failed: {e}"))
            continue
        seq = " ".join(arpa)
        if seq not in seq_to_tok:
            tokens.append(None)
            failures.append((i, raw, f"'{seq}' not in ps_tokens.arpabet_sequence"))
            continue
        tokens.append(seq_to_tok[seq])
    return tokens, failures


def check_event_id_is_9_ps_phonemes(epochs) -> Check:
    """Predicate #3: `event_id.keys()` normalizes to the 9 PS ARPABET phonemes."""
    try:
        fif_keys = {normalize_label(k) for k in epochs.event_id.keys()}
    except ValueError as e:
        return make_check(
            "event_id_is_9_ps_phonemes",
            "must",
            False,
            f"normalize_label failed on event_id: {e}",
        )
    expected: set[str] = set(PS_PHONEMES_9)
    missing = expected - fif_keys
    extra = fif_keys - expected
    ok = len(missing) == 0 and len(extra) == 0 and len(fif_keys) == 9
    detail = (
        f"n_keys={len(fif_keys)}"
        + (f" missing={sorted(missing)}" if missing else "")
        + (f" extra={sorted(extra)}" if extra else "")
    )
    return make_check("event_id_is_9_ps_phonemes", "must", ok, detail)


def check_per_trial_token_reconstruction(epochs, ps_tokens: pd.DataFrame) -> Check:
    """Predicate #4: each trial's 3 consecutive phoneme labels resolve to
    exactly one `arpabet_sequence` in `ps_tokens.csv`."""
    _, failures = _reconstruct_trial_tokens(epochs, ps_tokens)
    ok = len(failures) == 0
    n_trials = len(epochs.events) // 3
    return make_check(
        "per_trial_token_reconstruction",
        "must",
        ok,
        f"all {n_trials} trial-reconstructions resolve uniquely to a PS token"
        if ok
        else f"{len(failures)}/{n_trials} failed, first: {failures[:3]}",
    )


def check_fif_labels_match_events_tsv(
    epochs, authoritative_resp: pd.DataFrame, ps_tokens: pd.DataFrame
) -> Check:
    """Soft cross-check: for every trial, the reconstructed token from the
    3 consecutive phoneme epochs equals `events[OLD].tsv[i].value`. A
    mismatch indicates the events TSV is out of sync with the `.fif` (e.g.
    dropped row for S26 at trial 71, or wholesale regeneration for S14);
    the loader falls back to the `.fif`'s own labels."""
    tokens, _ = _reconstruct_trial_tokens(epochs, ps_tokens)
    n_trials = len(tokens)
    n_ev = len(authoritative_resp)
    if n_trials != n_ev:
        return make_check(
            "fif_labels_match_events_tsv",
            "soft",
            False,
            f"n_trials={n_trials} but events TSV responses={n_ev} — events TSV likely has a dropped/extra row",
        )
    mismatches = []
    for i, tok in enumerate(tokens):
        old_tok = authoritative_resp.iloc[i]["value"]
        if tok is None or tok != old_tok:
            mismatches.append((i, tok, old_tok))
    ok = len(mismatches) == 0
    return make_check(
        "fif_labels_match_events_tsv",
        "soft",
        ok,
        f"{n_trials} reconstructed trial tokens align with events TSV"
        if ok
        else f"{len(mismatches)}/{n_trials} mismatched, first: {mismatches[:3]} — events TSV out of sync, `.fif` labels are used",
    )


def check_events_tsv_tokens_in_ps(
    authoritative_resp: pd.DataFrame, ps_tokens: pd.DataFrame
) -> Check:
    """Soft cross-check: no `rest` / `task-start` / non-PS tokens in the events
    TSV's response rows. A failure means the events TSV itself contains
    non-PS tokens — still doesn't block loading since labels come from the
    `.fif`, but worth surfacing."""
    values = set(authoritative_resp["value"])
    ps_keys = set(ps_tokens["ps_notation"])
    leaks = values - ps_keys
    ok = len(leaks) == 0
    return make_check(
        "events_tsv_tokens_in_ps",
        "soft",
        ok,
        f"all {len(values)} distinct tokens ⊆ PS inventory"
        if ok
        else f"leaked tokens: {sorted(leaks)}",
    )


def check_signal_finite(epochs) -> Check:
    """Predicate #8: signal finite across the full file window."""
    data = epochs.get_data()  # (n_phon_epochs, n_ch, T)
    n_bad = int(np.sum(~np.isfinite(data)))
    ok = n_bad == 0
    return make_check(
        "signal_finite",
        "must",
        ok,
        f"{n_bad} non-finite samples across {data.size} total",
    )


def check_events_stale_is_divergent(
    epochs, stale_resp: pd.DataFrame, ps_tokens: pd.DataFrame
) -> Check:
    """Soft sanity: `events.tsv` should NOT match the reconstructed trial
    tokens from the `.fif` (it is the known-bad file). A high match count
    means we've mis-identified which TSV is authoritative."""
    tokens, _ = _reconstruct_trial_tokens(epochs, ps_tokens)
    n_trials = len(tokens)
    n_ev = len(stale_resp)
    n = min(n_trials, n_ev)
    matches = sum(tokens[i] == stale_resp.iloc[i]["value"] for i in range(n))
    ok = matches < 0.5 * n
    return make_check(
        "events_stale_is_divergent",
        "soft",
        ok,
        f"events.tsv matches {matches}/{n} reconstructed trial tokens "
        f"(expect << 0.5N since events.tsv is known-bad)",
    )
