"""Authenticity check for a built 2-band (LFS + HGA) converged-v2 exca spec cache.

The 2-band cache (P1.4/P1.5) stores ONE whole-movie spec memmap per session per
band under

    <spec-cache-dir>/band_<lfs|hga>/<MultiStftView._get_data,1>/<uid>/<hash>.{json,npy}

The cache build is the byte-bridge between the raw h5 + the per-session guard1ps
STATIC drop and what the v2 model ingests, so before any run we assert it is what
we think it is. For the named cache root this checks, per band:

  * exactly the expected session set is present (and, with ``--lite``, that it
    equals ``BT_LITE_SESSIONS`` — the byte-exact eval cohort);
  * the freq-bin geometry is the band's locked count (LFS 28, HGA 7);
  * ``npy.shape[0] == len(ch_names)`` (the electrode axis is consistent);

and CROSS-band, that the two bands agree on the session set, the per-session
electrode count, and the per-session ``extra_bad`` STATIC drop (a desync there
would mean LFS and HGA saw different electrodes — silent corruption).

RUN-WHERE-CACHE-LIVES (DCC). Machine paths are argv, never baked in::

    python scripts/neuroprobe/verify_2band_cache_authenticity.py \\
        /work/.../v14_2band_v2_spec_lite --lite
    python scripts/neuroprobe/verify_2band_cache_authenticity.py \\
        /work/.../v14_2band_v2_spec_pretrain --expect-n-sessions 13
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from speech_decoding.studies.braintreebank.manifest import BT_LITE_SESSIONS

_LITE: set[tuple[int, int]] = set(BT_LITE_SESSIONS)

# Locked per-band freq-bin counts (the band geometry, view.STFT_2BAND_{LFS,HGA}):
# LFS 1024/hop512 over 2-56 Hz -> 28 bins; HGA 128/hop64 over 64-160 Hz -> 7 bins.
BAND_FREQ_BINS: dict[str, int] = {"lfs": 28, "hga": 7}

_VIEW_SUBDIR = "speech_decoding.extractors.view.MultiStftView._get_data,1"


def _band_dir(spec_cache_dir: str, band: str) -> str:
    base = os.path.join(spec_cache_dir, f"band_{band}", _VIEW_SUBDIR)
    uids = os.listdir(base)
    if len(uids) != 1:
        raise SystemExit(f"band_{band}: expected exactly one uid under {base}, got {uids}")
    return os.path.join(base, uids[0])


def _parse_json(json_path: str) -> tuple[tuple[int, int], int, tuple[str, ...]]:
    """(subject, trial), n_ch, extra_bad — from the exca cache JSON sidecar.

    The ``key`` field is ``{nested-json}_<start>_<end>`` (the suffix breaks a bare
    ``json.loads``), so the session id + STATIC drop are pulled by regex; n_ch is
    the length of the (clean-JSON) ``ch_names`` list."""
    d = json.load(open(json_path))
    key = d["key"]
    m_sid = re.search(r'"subject_id":\s*(\d+)', key)
    m_tid = re.search(r'"trial_id":\s*(\d+)', key)
    if m_sid is None or m_tid is None:
        raise SystemExit(f"{json_path}: key missing subject_id/trial_id: {key[:120]}")
    sid, tid = int(m_sid.group(1)), int(m_tid.group(1))
    m = re.search(r'"extra_bad":\s*\[([^\]]*)\]', key)
    bad = (
        tuple(sorted(x.strip().strip('"') for x in m.group(1).split(",") if x.strip()))
        if m
        else ()
    )
    return (sid, tid), len(d["ch_names"]), bad


def _scan_band(band_dir: str) -> dict[tuple[int, int], tuple[int, tuple[int, ...], tuple[str, ...]]]:
    rows: dict[tuple[int, int], tuple[int, tuple[int, ...], tuple[str, ...]]] = {}
    for jf in glob.glob(os.path.join(band_dir, "*.json")):
        sess, n_ch, bad = _parse_json(jf)
        shape = np.load(jf[:-5] + ".npy", mmap_mode="r").shape
        rows[sess] = (n_ch, tuple(shape), bad)
    return rows


def verify(spec_cache_dir: str, *, lite: bool, expect_n_sessions: int | None) -> None:
    per_band = {}
    for band, exp_bins in BAND_FREQ_BINS.items():
        rows = _scan_band(_band_dir(spec_cache_dir, band))
        per_band[band] = rows
        freq_bins = {r[1][1] for r in rows.values()}
        nch_consistent = all(r[0] == r[1][0] for r in rows.values())
        print(
            f"  band_{band}: {len(rows)} sessions | freq-bins={freq_bins} "
            f"(expect {exp_bins}) | shape[0]==n_ch: {nch_consistent}"
        )
        assert freq_bins == {exp_bins}, f"band_{band}: freq-bins {freq_bins} != {exp_bins}"
        assert nch_consistent, f"band_{band}: npy shape[0] != len(ch_names)"

    s_lfs, s_hga = set(per_band["lfs"]), set(per_band["hga"])
    nch_match = all(per_band["lfs"][s][0] == per_band["hga"][s][0] for s in s_lfs)
    bad_match = all(per_band["lfs"][s][2] == per_band["hga"][s][2] for s in s_lfs)
    print(f"  cross-band session-set lfs==hga: {s_lfs == s_hga}")
    print(f"  cross-band per-session n_ch lfs==hga: {nch_match}")
    print(f"  cross-band per-session extra_bad lfs==hga: {bad_match}")
    print(f"  n_ch by session: { {s: per_band['lfs'][s][0] for s in sorted(s_lfs)} }")
    drops = {s: per_band["lfs"][s][2] for s in sorted(s_lfs) if per_band["lfs"][s][2]}
    print(f"  STATIC extra_bad by session: {drops}")
    assert s_lfs == s_hga, "cross-band session-set mismatch"
    assert nch_match, "cross-band per-session n_ch mismatch (LFS/HGA desync)"
    assert bad_match, "cross-band per-session extra_bad mismatch (STATIC desync)"

    if lite:
        miss, extra = _LITE - s_lfs, s_lfs - _LITE
        print(f"  == BT_LITE_SESSIONS: {s_lfs == _LITE} (missing={miss} extra={extra})")
        assert s_lfs == _LITE, "session set != BT_LITE_SESSIONS"
    if expect_n_sessions is not None:
        assert len(s_lfs) == expect_n_sessions, f"expected {expect_n_sessions} sessions, got {len(s_lfs)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec_cache_dir", help="root holding band_lfs/ + band_hga/ (machine path)")
    ap.add_argument("--lite", action="store_true", help="assert session set == BT_LITE_SESSIONS")
    ap.add_argument("--expect-n-sessions", type=int, default=None)
    args = ap.parse_args()
    print(f"===== {args.spec_cache_dir} =====")
    verify(args.spec_cache_dir, lite=args.lite, expect_n_sessions=args.expect_n_sessions)
    print("\nALL AUTHENTICITY ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
