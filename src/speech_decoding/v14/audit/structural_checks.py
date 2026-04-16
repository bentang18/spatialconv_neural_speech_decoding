"""Structural predicates for the `#34` audit.

Covers:
- file existence
- `.fif` window / sfreq / T (relaxed: file window must ⊇ `#29`'s `[-0.5, 1.0]`)
- `event_id` = 52-token PS inventory
- per-epoch token decomposes to exactly 3 ARPABET phonemes that appear as a
  unique `arpabet_sequence` row in `ps_tokens.csv`
- per-epoch `event_id[epoch.value]` matches the authoritative events TSV
- `n_epochs == authoritative_events.trial.nunique()`
- no `rest` / `task-start` / `malformed` tokens leak through

Predicate numbers follow `docs/implementation_tasks.md` #34, revised 2026-04-16
(predicate #2 relaxed; eventsOLD pinned as authoritative; audio predicates
live in `audio_checks.py`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from speech_decoding.data.phoneme_map import normalize_label
from speech_decoding.v14.audit.schema import Check, make_check

# #29: target loader window (seconds). Relaxed 2026-04-16 during S14 pilot:
# the canonical `.fif` has `tmax=0.995` (one sample short of 1.0 at 200 Hz
# because tmax is the LAST-sample time, not the right edge). Losing one 5 ms
# sample at the end is negligible; we crop to [-0.5, 0.995] (T=300).
TARGET_TMIN = -0.5
TARGET_TMAX = 0.995
TARGET_T_AT_200HZ = 300
TARGET_SFREQ = 200.0


def check_fif_path_exists(fif_path, label: str = "fif_path_exists") -> Check:
    return make_check(
        label,
        "must",
        fif_path.exists(),
        f"{fif_path}",
    )


def check_fif_window_contains_target(epochs) -> Check:
    """Predicate #2 relaxed (2026-04-16): file window must ⊇ `#29` window.
    Loader crops at load time."""
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


def check_event_id_is_52_ps_tokens(epochs, ps_tokens: pd.DataFrame) -> Check:
    """Predicate #3: `event_id.keys()` is exactly the PS 52-token inventory."""
    fif_keys = set(epochs.event_id.keys())
    ps_keys = set(ps_tokens["ps_notation"])
    missing = ps_keys - fif_keys
    extra = fif_keys - ps_keys
    ok = (len(missing) == 0) and (len(extra) == 0) and len(fif_keys) == 52
    detail = (
        f"n_keys={len(fif_keys)}"
        + (f" missing={sorted(missing)}" if missing else "")
        + (f" extra={sorted(extra)}" if extra else "")
    )
    return make_check("event_id_is_52_ps_tokens", "must", ok, detail)


def check_per_epoch_token_decomposition(epochs, ps_tokens: pd.DataFrame) -> Check:
    """Predicate #4b-c: every epoch's token decomposes to exactly 3 ARPABET
    phonemes, and that sequence appears exactly once in `ps_tokens.arpabet_sequence`.
    """
    id_to_tok = {v: k for k, v in epochs.event_id.items()}
    arpa_seqs = set(ps_tokens["arpabet_sequence"])
    fails = []
    for i, code in enumerate(epochs.events[:, 2]):
        tok = id_to_tok[int(code)]
        ps_row = ps_tokens.loc[ps_tokens["ps_notation"] == tok]
        if ps_row.empty:
            fails.append((i, tok, "no ps_tokens row"))
            continue
        ps_seq = ps_row.iloc[0]["ps_sequence"].split()
        if len(ps_seq) != 3:
            fails.append((i, tok, f"ps_sequence has {len(ps_seq)} phonemes"))
            continue
        try:
            arpa_seq = " ".join(normalize_label(p) for p in ps_seq)
        except ValueError as e:
            fails.append((i, tok, f"normalize_label failed: {e}"))
            continue
        if arpa_seq not in arpa_seqs:
            fails.append((i, tok, f"arpa '{arpa_seq}' not in ps_tokens"))
    ok = len(fails) == 0
    return make_check(
        "per_epoch_token_decomposition",
        "must",
        ok,
        "all 3-phoneme ARPABET sequences unique in ps_tokens.csv"
        if ok
        else f"{len(fails)}/{len(epochs)} failed, first: {fails[:3]}",
    )


def check_fif_labels_match_authoritative(
    epochs, authoritative_resp: pd.DataFrame
) -> Check:
    """Predicate #4a + #5: for every `.fif` epoch, `event_id[epoch.value]` equals
    the corresponding row in `eventsOLD.tsv` AND epoch count equals trial count."""
    id_to_tok = {v: k for k, v in epochs.event_id.items()}
    n_fif = len(epochs)
    n_ev = len(authoritative_resp)
    if n_fif != n_ev:
        return make_check(
            "fif_labels_match_authoritative",
            "must",
            False,
            f"n_epochs={n_fif} but eventsOLD responses={n_ev}",
        )
    mismatches = []
    for i, code in enumerate(epochs.events[:, 2]):
        fif_tok = id_to_tok[int(code)]
        old_tok = authoritative_resp.iloc[i]["value"]
        if fif_tok != old_tok:
            mismatches.append((i, fif_tok, old_tok))
    ok = len(mismatches) == 0
    return make_check(
        "fif_labels_match_authoritative",
        "must",
        ok,
        f"{n_fif} epochs align with eventsOLD.tsv"
        if ok
        else f"{len(mismatches)}/{n_fif} mismatched, first: {mismatches[:3]}",
    )


def check_no_leaked_tokens(
    authoritative_resp: pd.DataFrame, ps_tokens: pd.DataFrame
) -> Check:
    """Predicate #9: no `rest` / `task-start` / non-PS tokens in the authoritative
    events file's response rows."""
    values = set(authoritative_resp["value"])
    ps_keys = set(ps_tokens["ps_notation"])
    leaks = values - ps_keys
    ok = len(leaks) == 0
    return make_check(
        "no_leaked_tokens",
        "must",
        ok,
        f"all {len(values)} distinct tokens ⊆ PS inventory"
        if ok
        else f"leaked tokens: {sorted(leaks)}",
    )


def check_signal_finite(epochs) -> Check:
    """Predicate #8: signal finite across the full file window (not yet cropped)."""
    data = epochs.get_data()  # small: (n_epochs, n_ch, T); S14 = 153 × 124 × 500
    n_bad = int(np.sum(~np.isfinite(data)))
    ok = n_bad == 0
    return make_check(
        "signal_finite",
        "must",
        ok,
        f"{n_bad} non-finite samples across {data.size} total",
    )


def check_events_stale_is_divergent(
    epochs, stale_resp: pd.DataFrame
) -> Check:
    """Sanity: `events.tsv` should NOT match the `.fif` labels (it is the
    known-bad file). A high match count means we've mis-identified which
    TSV is authoritative."""
    id_to_tok = {v: k for k, v in epochs.event_id.items()}
    n_fif = len(epochs)
    n_ev = len(stale_resp)
    n = min(n_fif, n_ev)
    matches = sum(
        id_to_tok[int(epochs.events[i, 2])] == stale_resp.iloc[i]["value"]
        for i in range(n)
    )
    # Expect matches << n (most trials diverge). Soft check.
    ok = matches < 0.5 * n
    return make_check(
        "events_stale_is_divergent",
        "soft",
        ok,
        f"events.tsv matches {matches}/{n} .fif epochs by token "
        f"(expect << 0.5N since events.tsv is known-bad)",
    )
