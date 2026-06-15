"""End-to-end integration test for the Layer-2 (CLIP) + Layer-3 (WINSOR) bad-electrode
defenses, run against the REAL rebuilt 2STFT cache + REAL bad-window sidecars on DCC
(#180). The unit tests (``test_bad_windows.py`` / ``test_v14_wiring.py``) prove the
mechanism on SYNTHETIC data; this proves it ``genuinely works for a real run`` (Ben,
2026-06-15) on actual BrainTreebank voltage.

It is NOT a reimplementation of the pipeline: it monkeypatch-captures the production
``Experiment`` that ``dispatch_v14.main()`` builds, so the ``Data`` under test carries
the EXACT 80-kwarg production config (front-end, atlas, segmenter, cache dirs). We then
drive the two defenses through the real ``Data.build()`` path and the real
``--session-z-winsor`` / ``--bad-window-dir`` CLI flags.

Checks (all on a single GLITCHY session, default (2,2) = 18 sidecar spans):

  CLIP-1  Data.build() with --bad-window-dir drops a NONZERO number of train clips vs
          without it (real overlapping clips removed).
  CLIP-2  EVERY surviving train clip's neural window misses every bad span (no overlap
          leaks through).
  CLIP-3  val/test clip counts are IDENTICAL with/without the filter is NOT asserted
          (SSL build splits all train); instead we assert the filter only ever REMOVES
          rows (filtered count <= unfiltered) for every split.
  CLIP-4  Phase-4 (eval) Data has bad_window_dir gated to None even when the flag is
          passed (parity-locked Neuroprobe clip sets untouched). Soft: SKIPs if the
          eval-mode session cache is absent.
  WINSOR-1 Without --session-z-winsor, the global max |z| over the session's glitch
          clips EXCEEDS the cap W (a real giant cell exists to clamp).
  WINSOR-2 With --session-z-winsor W, the global max |z| over the SAME clips is <= W
          (every cell the model ingests is bounded).
  WINSOR-3 For the single worst clip, the winsored front-end tensor equals
          clip(unwinsored, -W, +W) elementwise (WINSOR is exactly a symmetric clamp at
          W — a no-op wherever |z| <= W, nothing else perturbed).

RUN ON DCC (needs ROOT_DIR_BRAINTREEBANK + the rebuilt cache):

    EXCA_EXTRACTOR_CACHE_FOLDER=/work/ht203/cache/v14_extractors_static_fixed \\
    .venv/bin/python scripts/neuroprobe/integration_test_winsor_clip.py \\
        --spec-cache-dir /work/ht203/cache_neuroai/v14_2stft_spec_cache_static_fixed \\
        --bad-window-dir /work/ht203/v14_bad_windows \\
        --session-index 2 --winsor 2500

Exit code 0 iff every (non-skipped) check passes.
"""
from __future__ import annotations

import argparse
import sys
import typing as tp

import torch


# ----------------------------------------------------------------------------- capture
def _build_production_data(argv: list[str]):
    """Run ``dispatch_v14.main(argv)`` (with --cache-only so it returns before the
    trainer) and return the production ``Data`` it built — captured by wrapping
    ``build_v14_experiment`` so we inherit its exact kwarg mapping verbatim."""
    from speech_decoding.experiments import dispatch_v14

    captured: list[tp.Any] = []
    orig = dispatch_v14.build_v14_experiment

    def _wrap(*a, **k):
        xp = orig(*a, **k)
        captured.append(xp)
        return xp

    dispatch_v14.build_v14_experiment = _wrap
    try:
        rc = dispatch_v14.main(argv)
    finally:
        dispatch_v14.build_v14_experiment = orig
    if rc != 0:
        raise RuntimeError(f"dispatch_v14.main returned {rc} for argv={argv}")
    if not captured:
        raise RuntimeError("build_v14_experiment was never called — capture failed")
    return captured[-1].data


def _base_argv(args, *, phase: int) -> list[str]:
    """The production cache-only argv (mirrors submit_build_bt_2stft_cache.py) for one
    session. Extra flags (--bad-window-dir / --session-z-winsor) are appended by callers
    so each defense is driven through its real CLI front-door."""
    return [
        "--phase", str(phase), "--mode", "full",
        "--frontend", "2stft", "--pool", "mean", "--mean-pool-std",
        "--atlas", "dkt", "--exclude-single-electrode-parcels",
        "--clip-len", "5.0",
        "--spec-cache-dir", args.spec_cache_dir,
        "--num-workers", "0",
        "--cache-only", "--cache-session-index", str(args.session_index),
    ]


# ----------------------------------------------------------------------------- helpers
# Extractor-name substrings that are side-channels, NOT the front-end |STFT|-z the
# encoder ingests (support weights, valid mask, target label, reference index, parcel id).
_NON_FRONTEND = ("support", "valid", "mask", "target", "ref", "parcel", "id", "label")


def _batch_data(item: tp.Any) -> dict[str, torch.Tensor]:
    """The neuralset ``Batch.data`` dict (extractor-name -> batched tensor); a plain dict
    item is returned as-is."""
    data = getattr(item, "data", None)
    if data is None and isinstance(item, dict):
        data = item
    return data or {}


def _frontend_tensors(item: tp.Any) -> dict[str, torch.Tensor]:
    """The front-end |STFT|-z tensors out of one dataset item (the 2STFT path emits the
    low + high band under ``electrode_tokens*``). Side-channels (support / valid_mask /
    target / ref_idx / parcel-id) are excluded by name and by non-float dtype, so the
    max |.| we report is the per-cell robust-z the encoder actually sees."""
    out: dict[str, torch.Tensor] = {}
    for k, v in _batch_data(item).items():
        if not torch.is_tensor(v) or not v.is_floating_point() or v.ndim < 2:
            continue
        if any(tok in k.lower() for tok in _NON_FRONTEND):
            continue
        out[k] = v.detach().float()
    return out


def _global_max_abs(tensors: dict[str, torch.Tensor]) -> float:
    return max((float(v.abs().max()) for v in tensors.values()), default=0.0)


def _overlaps(lo: float, hi: float, spans: list[tuple[float, float]]) -> bool:
    return any(lo < b1 and b0 < hi for b0, b1 in spans)


# ------------------------------------------------------------------------------- checks
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []  # (id, status, detail)

    def add(self, cid: str, ok: bool | None, detail: str) -> None:
        status = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        self.rows.append((cid, status, detail))
        print(f"[{status}] {cid}: {detail}", flush=True)

    def failed(self) -> bool:
        return any(s == "FAIL" for _, s, _ in self.rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-cache-dir", required=True)
    ap.add_argument("--bad-window-dir", required=True)
    ap.add_argument("--session-index", type=int, default=2,
                    help="Index into V14_PRETRAIN_SESSIONS (default 2 = (2,2), 18 spans).")
    ap.add_argument("--winsor", type=float, default=2500.0)
    ap.add_argument("--diag", action="store_true",
                    help="Introspection only: build the Data, print dataset structure / "
                         "front-end tensor shapes (validates plumbing on any session, "
                         "incl. glitch-free ones). No assertions.")
    args = ap.parse_args()

    from speech_decoding.experiments.bad_windows import (
        bad_window_session_key,
        load_bad_windows,
    )
    from speech_decoding.studies.braintreebank.manifest import V14_PRETRAIN_SESSIONS

    sid, tid = V14_PRETRAIN_SESSIONS[args.session_index]
    key = bad_window_session_key(sid, tid)
    spans = load_bad_windows(args.bad_window_dir).get(key, [])
    print(f"=== session index {args.session_index} -> (subject {sid}, trial {tid}); "
          f"sidecar key {key}: {len(spans)} bad spans ===", flush=True)
    if not spans:
        print("WARNING: this session has NO bad spans — CLIP/WINSOR assertions cannot "
              "fire. Pick a glitchy --session-index (1,2,7,10).", flush=True)

    if args.diag:
        data = _build_production_data(_base_argv(args, phase=1))
        loaders = data.build()
        print(f"split sizes = {{s: len(loaders[s].dataset) for s}} -> "
              f"{ {s: len(loaders[s].dataset) for s in loaders} }", flush=True)
        sel = loaders["train"].dataset
        print(f"train dataset type = {type(sel).__module__}.{type(sel).__name__}", flush=True)
        trig = getattr(sel, "triggers", None)
        print(f"has .triggers = {trig is not None}; columns = "
              f"{list(getattr(trig, 'columns', []))[:25]}", flush=True)
        if trig is not None and "start" in getattr(trig, "columns", []):
            st = trig["start"].to_numpy()
            print(f"start[:5] = {st[:5]}  (n={len(st)})", flush=True)
        item = sel[0]
        data = _batch_data(item)
        print(f"item type = {type(item).__name__}; data keys = {list(data.keys())}", flush=True)
        for k, v in data.items():
            if torch.is_tensor(v):
                mx = float(v.abs().max()) if v.is_floating_point() else None
                print(f"  data[{k}] shape={tuple(v.shape)} dtype={v.dtype}"
                      + (f" max|.|={mx:.2f}" if mx is not None else ""), flush=True)
        ft = _frontend_tensors(item)
        for k, v in ft.items():
            print(f"  frontend[{k}] shape={tuple(v.shape)} max|.|={float(v.abs().max()):.2f} "
                  f"dtype={v.dtype}", flush=True)
        print(f"global max|z| on item[0] = {_global_max_abs(ft):.2f}", flush=True)
        return 0

    rep = Report()

    # ---- CLIP: real Data.build() with/without the sidecar dir --------------------
    data_unfiltered = _build_production_data(_base_argv(args, phase=1))
    data_filtered = _build_production_data(
        _base_argv(args, phase=1) + ["--bad-window-dir", args.bad_window_dir]
    )
    loaders_unf = data_unfiltered.build()
    loaders_fil = data_filtered.build()

    n_unf = {s: len(loaders_unf[s].dataset) for s in loaders_unf}
    n_fil = {s: len(loaders_fil[s].dataset) for s in loaders_fil}
    print(f"clip counts  unfiltered={n_unf}  filtered={n_fil}", flush=True)

    dropped = n_unf.get("train", 0) - n_fil.get("train", 0)
    rep.add("CLIP-1", dropped > 0,
            f"train clips dropped by filter = {dropped} "
            f"({n_unf.get('train')} -> {n_fil.get('train')})")

    rep.add("CLIP-3", all(n_fil[s] <= n_unf[s] for s in n_fil),
            f"filter only removes rows per split: {n_fil} <= {n_unf}")

    # CLIP-2: no surviving train clip overlaps any bad span.
    sel_fil = loaders_fil["train"].dataset
    trig = getattr(sel_fil, "triggers", None)
    if trig is not None and "start" in getattr(trig, "columns", []):
        seg_start = float(data_filtered.segmenter.start)
        seg_dur = float(data_filtered.segmenter.duration)
        leaks = 0
        for st in trig["start"].to_numpy():
            lo = float(st) + seg_start
            if _overlaps(lo, lo + seg_dur, spans):
                leaks += 1
        rep.add("CLIP-2", leaks == 0,
                f"surviving train clips overlapping a bad span = {leaks} "
                f"(of {len(trig)})")
    else:
        rep.add("CLIP-2", None, "dataset exposes no .triggers['start'] — cannot verify "
                                "per-clip overlap directly")

    # CLIP-4: Phase-4 gates bad_window_dir to None (soft — needs eval-mode cache).
    try:
        data_p4 = _build_production_data(
            ["--phase", "4", "--mode", "full",
             "--frontend", "2stft", "--pool", "mean", "--mean-pool-std",
             "--atlas", "dkt", "--exclude-single-electrode-parcels",
             "--clip-len", "5.0", "--spec-cache-dir", args.spec_cache_dir,
             "--num-workers", "0", "--cache-only", "--cache-session-index", "0",
             "--bad-window-dir", args.bad_window_dir]
        )
        rep.add("CLIP-4", data_p4.bad_window_dir is None,
                f"phase-4 Data.bad_window_dir = {data_p4.bad_window_dir!r} (must be None)")
    except Exception as exc:  # eval-mode session cache absent etc.
        rep.add("CLIP-4", None, f"phase-4 build unavailable here: {type(exc).__name__}: {exc}")

    # ---- WINSOR: real --session-z-winsor through the read path -------------------
    # Use the UNFILTERED phase-1 dataset (glitch clips present) twice: no-winsor and
    # winsor. Scan ONLY the clips overlapping a bad span (fast + targeted).
    W = args.winsor
    data_nowin = _build_production_data(_base_argv(args, phase=1))
    data_win = _build_production_data(
        _base_argv(args, phase=1) + ["--session-z-winsor", str(W)]
    )
    sel_nowin = data_nowin.build()["train"].dataset
    sel_win = data_win.build()["train"].dataset

    trig_nw = getattr(sel_nowin, "triggers", None)
    glitch_idx: list[int] = []
    if trig_nw is not None and "start" in getattr(trig_nw, "columns", []):
        seg_start = float(data_nowin.segmenter.start)
        seg_dur = float(data_nowin.segmenter.duration)
        starts = trig_nw["start"].to_numpy()
        for i, st in enumerate(starts):
            lo = float(st) + seg_start
            if _overlaps(lo, lo + seg_dur, spans):
                glitch_idx.append(i)

    if not glitch_idx:
        rep.add("WINSOR-1", None, "no train clip overlaps a bad span on this session — "
                                  "cannot locate a giant |z| clip")
        rep.add("WINSOR-2", None, "skipped (no glitch clip)")
        rep.add("WINSOR-3", None, "skipped (no glitch clip)")
    else:
        # Global max |z| over glitch clips, no-winsor vs winsor; and the single worst.
        worst_i, worst_max = glitch_idx[0], -1.0
        max_nowin = 0.0
        for i in glitch_idx:
            m = _global_max_abs(_frontend_tensors(sel_nowin[i]))
            max_nowin = max(max_nowin, m)
            if m > worst_max:
                worst_max, worst_i = m, i
        max_win = max(
            _global_max_abs(_frontend_tensors(sel_win[i])) for i in glitch_idx
        )
        print(f"glitch clips scanned={len(glitch_idx)}  "
              f"max|z| no-winsor={max_nowin:.1f}  winsor={max_win:.1f}  "
              f"worst clip idx={worst_i}", flush=True)

        rep.add("WINSOR-1", max_nowin > W,
                f"max |z| over glitch clips (no winsor) = {max_nowin:.1f} > W={W:.0f}")
        rep.add("WINSOR-2", max_win <= W + 1e-3,
                f"max |z| over glitch clips (winsor W={W:.0f}) = {max_win:.1f} <= W")

        # WINSOR-3: elementwise clamp identity on the worst clip.
        t_nowin = _frontend_tensors(sel_nowin[worst_i])
        t_win = _frontend_tensors(sel_win[worst_i])
        keys = sorted(set(t_nowin) & set(t_win))
        if not keys:
            rep.add("WINSOR-3", None, "no comparable front-end tensor keys")
        else:
            ok = True
            detail = []
            for k in keys:
                a = t_nowin[k]
                b = t_win[k]
                if a.shape != b.shape:
                    ok = False
                    detail.append(f"{k}: shape {tuple(a.shape)} != {tuple(b.shape)}")
                    continue
                clamped = a.clamp(-W, W)
                match = torch.allclose(b, clamped, atol=1e-3, rtol=0)
                n_diff = int((~torch.isclose(b, clamped, atol=1e-3, rtol=0)).sum())
                ok = ok and match
                detail.append(f"{k}: clamp-identity={'ok' if match else f'{n_diff} mismatch'}")
            rep.add("WINSOR-3", ok,
                    "winsored == clip(unwinsored, -W, +W) elementwise on worst clip; "
                    + "; ".join(detail))

    # ----------------------------------------------------------------- verdict
    print("\n=== SUMMARY ===", flush=True)
    for cid, status, detail in rep.rows:
        print(f"  {status:4s} {cid}: {detail}", flush=True)
    if rep.failed():
        print("\nRESULT: FAIL — a defense did not behave as specified.", flush=True)
        return 1
    print("\nRESULT: PASS (skips are environment-gated, not failures).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
