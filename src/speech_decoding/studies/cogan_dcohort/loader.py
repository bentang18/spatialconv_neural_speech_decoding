"""Voltage reader for Cogan D-cohort BIDS EDF runs, resampled to the v3 rate.

Cogan analogue of ``braintreebank.loader.bt_load_raw``: returns voltage in the
shape NeuralSet's ``Ieeg._read()`` wants —
``(data: np.ndarray (n_ch, n_samples) float32, ch_names: list[str], sfreq: float)``
— at the v3 canonical rate, with non-neural channels and guard-1 static-bad
contacts already dropped, and NO re-reference applied (shaft-CAR lives in the
front-end ``MultiStftView``, over exactly the rows returned here).

Two Cogan-specific facts drove the channel-select design (verified on DCC,
``BIDS-1.4_Phoneme_sequencing/sub-D0019`` 2026-07-13):

  * The BIDS ``channels.tsv`` ``type`` column labels the DIXI depth contacts
    ``ECOG`` (not ``SEEG``) on at least some runs, alongside a ``TRIG`` channel.
    So we select by a NEURAL-TYPE WHITELIST and drop everything else (TRIG, EKG,
    …) rather than filter on ``type == SEEG`` — the latter would silently drop
    every neural channel on an ECOG-typed run.
  * Native ``SamplingFrequency`` is mixed across the corpus (2048 / 2000 / 1024
    / 1000). We read it from the EDF header (authoritative, cheap) and
    polyphase-resample UP to ``TARGET_RATE_HZ`` here, at the one place voltage
    enters — so returned ``sfreq`` is always ``TARGET_RATE_HZ``.

Guard-1 static-bad drop: passed in via ``extra_bad`` (from the per-run manifest,
populated by the guard-1 scan) and applied here BEFORE any re-reference — a bad
contact must not reach its shaft's CAR reference. Empty until the scan lands.

``mne`` EDF reads only fire where the Cogan BIDS tree is mounted (DCC). The pure
channel-select + resample core is factored out so it unit-tests on a synthetic
``RawArray`` with no EDF on disk.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from math import gcd

import numpy as np
from scipy.signal import resample_poly

# BIDS iEEG channel types we keep as neural voltage. Cogan depth contacts appear
# as ECOG on some runs and SEEG on others (BIDS-typing is inconsistent across the
# corpus); both are the same physical depth signal for us. Everything else
# (TRIG, EKG/ECG, EOG, EMG, DC, MISC, …) is dropped.
NEURAL_TYPES: frozenset[str] = frozenset({"ECOG", "SEEG"})

# v3 canonical rate (matches the STFT front-end's hard-coded 2048 Hz).
TARGET_RATE_HZ: float = 2048.0

# A depth/grid contact name = shaft label + trailing contact index. The shaft is
# everything up to the maximal trailing digit run; it MAY itself contain digits
# (``L1IF10`` → shaft ``L1IF``, depth 10 — the ``L1`` implant-grid prefix stays).
# Lazy ``.*?`` + end-anchored ``\d+`` ⇒ the digits captured are exactly the
# maximal trailing run, so ``ROG13`` → (``ROG``, 13) and ``R2SF3`` → (``R2SF``, 3).
_CONTACT_RE = re.compile(r"^(?P<shaft>.*?)(?P<depth>\d+)$")


def parse_contact(name: str) -> tuple[str, int] | None:
    """``ch_name`` → ``(shaft_label, contact_index)`` or ``None`` if it has no
    trailing contact number.

    A missing contact number is the empirical signature (verified over the whole
    Cogan corpus, 4003 distinct neural-typed names, 2026-07-13) of a physiological
    /scalp reference mistyped as ``ECOG``/``SEEG`` — ``EKGL``, ``EKGR``, ``LEMG``,
    ``REMG``, ``Cz``/``Fz``/``Pz``, or a blank name. Every one of the 3992 real
    depth/grid contacts carries a trailing index; none of the 11 refs do. It is
    also a HARD requirement of the model: within-shaft RoPE positions each contact
    by this index, so a contact-less channel cannot be placed.
    """
    m = _CONTACT_RE.match(name.strip())
    if not m:
        return None
    return m.group("shaft"), int(m.group("depth"))


def read_channel_types(channels_tsv_path: str) -> dict[str, str]:
    """BIDS ``channels.tsv`` → ``{channel_name: TYPE}`` (upper-cased type).

    ``utf-8-sig`` strips a leading UTF-8 BOM if present (13 Cogan subjects — e.g.
    sub-D0057 — carry a ``EF BB BF`` BOM + CRLF from a Windows export, which under a
    plain-utf-8 DictReader mangles the first header into ``\\ufeffname`` and KeyErrors
    on ``row['name']``); it behaves as utf-8 for the BOM-free majority. csv handles
    the CRLF itself under ``newline=''``.
    """
    types: dict[str, str] = {}
    with open(channels_tsv_path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            types[row["name"]] = (row.get("type") or "").strip().upper()
    return types


def select_neural(
    ch_names: list[str],
    data: np.ndarray,
    ch_type_by_name: dict[str, str],
    extra_bad: Iterable[str] = (),
) -> tuple[np.ndarray, list[str]]:
    """Keep neural, non-bad DEPTH/GRID channels in the given (voltage) order.

    ``data`` is ``(n_ch, n_samples)`` row-aligned with ``ch_names``. A channel is
    kept iff its ``channels.tsv`` type is in ``NEURAL_TYPES``, it is not in
    ``extra_bad``, AND it parses as a contact (has a trailing index). The last
    clause drops physio/scalp references mistyped as ``ECOG``/``SEEG``
    (``EKGL``/``LEMG``/``Cz``…) — the type column alone doesn't catch them. A
    channel with no ``channels.tsv`` type is treated as non-neural (dropped) —
    fail-safe, never fail-open.
    """
    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"data rows {data.shape[0]} != n ch_names {len(ch_names)}"
        )
    bad = set(extra_bad)
    keep_idx = [
        i
        for i, nm in enumerate(ch_names)
        if ch_type_by_name.get(nm, "") in NEURAL_TYPES
        and nm not in bad
        and parse_contact(nm) is not None
    ]
    if not keep_idx:
        raise ValueError(
            "no neural channels survived selection "
            f"(types seen: {sorted(set(ch_type_by_name.values()))}, "
            f"n extra_bad={len(bad)})"
        )
    kept_names = [ch_names[i] for i in keep_idx]
    return np.ascontiguousarray(data[keep_idx], dtype=np.float32), kept_names


def resample_to(
    data: np.ndarray, native_rate: float, target_rate: float = TARGET_RATE_HZ
) -> np.ndarray:
    """Polyphase-resample ``(n_ch, n_samples)`` from ``native_rate`` → ``target_rate``.

    Identity when the rates match. Uses the reduced integer ratio (e.g. 2000→2048
    = up 128 / down 125; 1024→2048 = up 2 / down 1), matching ``bt_load_raw``.
    """
    nr, tr = int(round(native_rate)), int(round(target_rate))
    if nr == tr:
        return np.ascontiguousarray(data, dtype=np.float32)
    if nr <= 0:
        raise ValueError(f"non-positive native_rate {native_rate}")
    g = gcd(nr, tr)
    up, down = tr // g, nr // g
    out = resample_poly(data, up, down, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def cogan_load_raw(
    edf_path: str,
    channels_tsv_path: str,
    *,
    extra_bad: Iterable[str] = (),
    target_rate: float = TARGET_RATE_HZ,
) -> tuple[np.ndarray, list[str], float]:
    """``(data, ch_names, sfreq)`` for one Cogan BIDS run, v3-ready.

    Reads the EDF via ``mne`` (header gives native ``sfreq``), keeps neural
    non-bad channels in EDF/voltage order, resamples to ``target_rate``.
    """
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    types = read_channel_types(channels_tsv_path)
    data, ch_names = select_neural(
        list(raw.ch_names), np.asarray(raw.get_data()), types, extra_bad
    )
    data = resample_to(data, float(raw.info["sfreq"]), target_rate)
    return data, ch_names, float(target_rate)
