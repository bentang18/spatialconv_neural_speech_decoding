"""Voltage reader for RAM (Kahana/UPenn) OpenNeuro BIDS iEEG runs, v3-ready.

RAM analogue of ``cogan_dcohort.loader.cogan_load_raw`` / ``braintreebank`` — returns
``(data (n_ch, n_samples) float32, ch_names, sfreq)`` at the v3 canonical rate, with
non-neural channels + guard-1 static-bad + micro-electrodes dropped, and NO
re-reference (group-CAR lives in the ``MultiStftView`` front-end, over exactly the
rows returned here).

RAM specifics verified against the public OpenNeuro S3 mirror (2026-07-14,
``ds004789/sub-R1001P`` FR1; recon #100):

  * The MONOPOLAR acquisition is the one we consume (``acq-monopolar_ieeg.edf``); the
    dataset also ships a pre-referenced ``acq-bipolar`` — we take monopolar and apply
    our own group-CAR, for front-end parity with Cogan/BT.
  * ``channels.tsv`` carries ``type`` (ECOG for grids/strips, SEEG for depths — the
    SAME inconsistent-but-both-neural mix as Cogan), plus ``group`` (the researcher's
    CAR grouping) AND ``sampling_frequency`` in one file. So neural selection uses the
    ECOG/SEEG whitelist (never ``type == SEEG``), exactly as Cogan.
  * Native ``sampling_frequency`` varies per subject; we read it from the EDF header
    (authoritative) and polyphase-resample UP to ``TARGET_RATE_HZ``.

Group-CAR note (recon #100 correction): 100% of macro contact names parse
``<stem><int>``, but ~2% are compound ``_``-names (``LP1_1``) whose greedy trailing-digit
parse yields stem ``LP1_`` and FRAGMENTS the true ``group`` ``LP``. The existing
front-end name-parse CAR is therefore correct for ~98% of contacts; ``ram_car_groups``
exposes the AUTHORITATIVE channels.tsv ``group`` per kept channel for the #94
explicit-group-CAR path that fixes the remaining ~2%. Micro-electrodes (electrodes.tsv
``type == micro``, 80 corpus-wide) are ECOG/SEEG-typed with a trailing index, so the
whitelist alone would keep them — they are supplied via ``extra_bad`` from the manifest.

``mne`` EDF reads only fire where the raw tree is staged; the channel-select + resample
core unit-tests on a synthetic ``RawArray`` with no EDF on disk.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from math import gcd

import numpy as np
from scipy.signal import resample_poly

# BIDS iEEG channel types kept as neural voltage. RAM (like Cogan) types depth
# contacts SEEG and grid/strip contacts ECOG; both are neural for us.
NEURAL_TYPES: frozenset[str] = frozenset({"ECOG", "SEEG"})

# v3 canonical rate (matches the STFT front-end's hard-coded 2048 Hz).
TARGET_RATE_HZ: float = 2048.0

# contact name = group/shaft label + trailing contact index; lazy stem + end-anchored
# digits ⇒ exactly the maximal trailing run (``LAF12`` → (``LAF``, 12)).
_CONTACT_RE = re.compile(r"^(?P<shaft>.*?)(?P<depth>\d+)$")


def parse_contact(name: str) -> tuple[str, int] | None:
    """``ch_name`` → ``(stem, contact_index)`` or ``None`` if no trailing index.

    A missing trailing index marks a physio/scalp ref mistyped as ECOG/SEEG, and is
    also disqualifying for the model (within-shaft RoPE positions each contact by this
    index, so a contact-less channel cannot be placed). NOTE the parsed ``stem`` is NOT
    always the CAR group — see ``ram_car_groups`` for the authoritative grouping.
    """
    m = _CONTACT_RE.match(name.strip())
    if not m:
        return None
    return m.group("shaft"), int(m.group("depth"))


def read_channels_tsv(channels_tsv_path: str) -> dict[str, dict[str, str]]:
    """RAM BIDS ``channels.tsv`` → ``{name: {type, group, sampling_frequency}}``.

    ``utf-8-sig`` strips a leading BOM if present (defensive, matching Cogan). Values
    are raw strings; ``type`` is upper-cased. Missing columns yield empty strings.
    """
    out: dict[str, dict[str, str]] = {}
    with open(channels_tsv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            out[row["name"]] = {
                "type": (row.get("type") or "").strip().upper(),
                "group": (row.get("group") or "").strip(),
                "sampling_frequency": (row.get("sampling_frequency") or "").strip(),
            }
    return out


def select_neural(
    ch_names: list[str],
    data: np.ndarray,
    channels_meta: dict[str, dict[str, str]],
    extra_bad: Iterable[str] = (),
) -> tuple[np.ndarray, list[str]]:
    """Keep neural, non-bad, indexed channels in the given (voltage) order.

    ``data`` is ``(n_ch, n_samples)`` row-aligned with ``ch_names``. A channel is kept
    iff its ``channels.tsv`` type is in ``NEURAL_TYPES``, it is not in ``extra_bad``
    (guard-1 static-bad + micro-electrode names supplied by the manifest), AND it parses
    as a contact (trailing index). A channel with no ``channels.tsv`` row is treated as
    non-neural (dropped) — fail-safe, never fail-open.
    """
    if data.shape[0] != len(ch_names):
        raise ValueError(f"data rows {data.shape[0]} != n ch_names {len(ch_names)}")
    bad = set(extra_bad)
    keep_idx = [
        i
        for i, nm in enumerate(ch_names)
        if channels_meta.get(nm, {}).get("type", "") in NEURAL_TYPES
        and nm not in bad
        and parse_contact(nm) is not None
    ]
    if not keep_idx:
        raise ValueError(
            "no neural channels survived selection "
            f"(types seen: {sorted({m.get('type','') for m in channels_meta.values()})}, "
            f"n extra_bad={len(bad)})"
        )
    kept = [ch_names[i] for i in keep_idx]
    return np.ascontiguousarray(data[keep_idx], dtype=np.float32), kept


def ram_car_groups(
    kept_names: list[str], channels_meta: dict[str, dict[str, str]]
) -> list[str]:
    """Authoritative CAR group per kept channel, from the channels.tsv ``group`` column.

    Falls back to the name-parse stem when ``group`` is absent (older exports). This is
    the input to the #94 explicit-group-CAR front-end path; it agrees with the current
    name-parse CAR for ~98% of RAM contacts and CORRECTS the ~2% compound ``_``-names.
    """
    groups: list[str] = []
    for nm in kept_names:
        g = channels_meta.get(nm, {}).get("group", "")
        if not g:
            parsed = parse_contact(nm)
            g = parsed[0] if parsed else nm
        groups.append(g)
    return groups


def resample_to(
    data: np.ndarray, native_rate: float, target_rate: float = TARGET_RATE_HZ
) -> np.ndarray:
    """Polyphase-resample ``(n_ch, n_samples)`` from ``native_rate`` → ``target_rate``.

    Identity when rates match; else the reduced integer ratio (e.g. 500→2048 up 512 /
    down 125; 1000→2048 up 256/125; 1600→2048 up 64/50). Matches ``bt_load_raw``.
    """
    nr, tr = int(round(native_rate)), int(round(target_rate))
    if nr == tr:
        return np.ascontiguousarray(data, dtype=np.float32)
    if nr <= 0:
        raise ValueError(f"non-positive native_rate {native_rate}")
    g = gcd(nr, tr)
    out = resample_poly(data, tr // g, nr // g, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def ram_load_raw(
    edf_path: str,
    channels_tsv_path: str,
    *,
    extra_bad: Iterable[str] = (),
    target_rate: float = TARGET_RATE_HZ,
) -> tuple[np.ndarray, list[str], float]:
    """``(data, ch_names, sfreq)`` for one RAM monopolar BIDS run, v3-ready.

    Reads the EDF via ``mne`` (header gives native ``sfreq``), keeps neural non-bad
    indexed channels in EDF/voltage order, resamples to ``target_rate``.
    """
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    meta = read_channels_tsv(channels_tsv_path)
    data, ch_names = select_neural(
        list(raw.ch_names), np.asarray(raw.get_data()), meta, extra_bad
    )
    data = resample_to(data, float(raw.info["sfreq"]), target_rate)
    return data, ch_names, float(target_rate)
