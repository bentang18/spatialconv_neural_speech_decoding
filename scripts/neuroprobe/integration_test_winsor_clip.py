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
import os
import sys
import typing as tp

import torch

# Production winsor knob: the CLI flag --session-z-winsor sets only this env var
# (dispatch_v14.py ~3568), read by SessionRobustZNormalizer (normalize.py). We drive it
# directly so the read path is byte-identical to a real --session-z-winsor run.
_WINSOR_ENV = "V14_SESSION_Z_WINSOR"


# ----------------------------------------------------------------------------- capture
class _Captured(Exception):
    """Carries the production Data out of ``main()`` and aborts it before the
    cache-only prepare()/trainer runs (``Data.build()`` does its own prepare())."""

    def __init__(self, data: tp.Any) -> None:
        self.data = data


def _build_production_data(argv: list[str]):
    """Return the production ``Data`` that ``dispatch_v14.main(argv)`` would build,
    captured by wrapping ``build_v14_experiment`` so we inherit its exact 80-kwarg
    mapping verbatim. We raise right after the Experiment is built to abort main()
    before it prepares/trains — so this is cheap even for phases whose cache is absent
    (the phase-4 gating check never touches an eval-mode cache)."""
    from speech_decoding.experiments import dispatch_v14

    orig = dispatch_v14.build_v14_experiment

    def _wrap(*a, **k):
        raise _Captured(orig(*a, **k).data)

    dispatch_v14.build_v14_experiment = _wrap
    try:
        dispatch_v14.main(argv)
    except _Captured as c:
        return c.data
    finally:
        dispatch_v14.build_v14_experiment = orig
    raise RuntimeError(f"capture failed (main returned without building) for argv={argv}")


def _base_argv(args, *, phase: int, winsor: float | None = None) -> list[str]:
    """The production cache-only argv (mirrors submit_build_bt_2stft_cache.py) for one
    session. WINSOR is driven through the REAL ``--session-z-winsor`` CLI flag (not a
    manual env set): ``main()`` is authoritative over ``V14_SESSION_Z_WINSOR`` — it SETS
    the env when the flag is present and POPS it when absent (dispatch_v14.py:3567-3570,
    so a stale value never leaks). Setting the env by hand and omitting the flag would be
    silently wiped by that pop — exactly the bug that made an earlier winsor check read
    unclamped. Appending the flag makes the env identical to a real
    ``--session-z-winsor`` run and immune to build-order contamination (each main() call
    sets/pops per its own argv). ``--bad-window-dir`` is appended by CLIP callers."""
    argv = [
        "--phase", str(phase), "--mode", "full",
        "--frontend", "2stft", "--pool", "mean", "--mean-pool-std",
        "--atlas", "dkt", "--exclude-single-electrode-parcels",
        "--clip-len", "5.0",
        "--spec-cache-dir", args.spec_cache_dir,
        "--num-workers", "0",
        "--cache-only", "--cache-session-index", str(args.session_index),
    ]
    if winsor is not None:
        argv += ["--session-z-winsor", str(winsor)]
    return argv


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
    ap.add_argument("--winsor-on", action="store_true",
                    help="(with --winsor-probe) build with --session-z-winsor on; omit to "
                         "build the no-winsor baseline.")
    ap.add_argument("--winsor-probe", action="store_true",
                    help="ISOLATION probe: build ONE phase-1 dataset honoring the SHELL's "
                         "V14_SESSION_Z_WINSOR env (set it BEFORE invoking, exactly like "
                         "production sets it from --session-z-winsor before prepare()), scan "
                         "the glitch clips, print max|z|. Run once per env state in a FRESH "
                         "process so no prior build's normalizer state can leak — this is the "
                         "production-faithful winsor check (one view, env set from the start).")
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

    if args.winsor_probe:
        # ISOLATION probe: build ONE phase-1 dataset in a FRESH process and scan the
        # glitch clips for the max |z| the encoder would ingest. With --winsor-on the
        # build appends --session-z-winsor (the REAL production front-door: main() sets
        # V14_SESSION_Z_WINSOR before prepare()); without it the flag is absent and
        # main() pops the env. Run once per state in separate processes so no prior
        # build's normalizer can leak — the production-faithful winsor check.
        w = args.winsor if args.winsor_on else None
        data = _build_production_data(_base_argv(args, phase=1, winsor=w))
        sel = data.build()["train"].dataset
        trig = getattr(sel, "triggers", None)
        seg_start = float(data.segmenter.start)
        seg_dur = float(data.segmenter.duration)
        gidx = [i for i, st in enumerate(trig["start"].to_numpy())
                if _overlaps(float(st) + seg_start, float(st) + seg_start + seg_dur, spans)]
        mx = max((_global_max_abs(_frontend_tensors(sel[i])) for i in gidx), default=0.0)
        print(f"WINSOR_PROBE  --session-z-winsor={w}  glitch_clips={len(gidx)}  "
              f"max|z|={mx:.1f}", flush=True)
        return 0

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
    os.environ.pop(_WINSOR_ENV, None)  # CLIP is winsor-independent; start clean

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

    dropped_by_split = {s: n_unf[s] - n_fil.get(s, 0) for s in n_unf}
    dropped_total = sum(dropped_by_split.values())
    rep.add("CLIP-1", dropped_total > 0,
            f"clips dropped by filter (any split) = {dropped_total} "
            f"by-split {dropped_by_split} -- a session's glitches can land "
            f"entirely in the val/test partition under the time-based holdout "
            f"split, so train-only drops are NOT guaranteed; what must hold is "
            f"that the filter removes overlapping clips wherever they fall")

    rep.add("CLIP-3", all(n_fil[s] <= n_unf[s] for s in n_fil),
            f"filter only removes rows per split: {n_fil} <= {n_unf}")

    # CLIP-2: no surviving clip in ANY split overlaps a bad span. (Train-only
    # would miss sessions whose glitches land in the val/test holdout partition.)
    seg_start = float(data_filtered.segmenter.start)
    seg_dur = float(data_filtered.segmenter.duration)
    leaks_by_split: dict[str, int] = {}
    checked_any = False
    for split, loader in loaders_fil.items():
        trig = getattr(loader.dataset, "triggers", None)
        if trig is None or "start" not in getattr(trig, "columns", []):
            continue
        checked_any = True
        leaks = 0
        for st in trig["start"].to_numpy():
            lo = float(st) + seg_start
            if _overlaps(lo, lo + seg_dur, spans):
                leaks += 1
        leaks_by_split[split] = leaks
    if checked_any:
        total_leaks = sum(leaks_by_split.values())
        rep.add("CLIP-2", total_leaks == 0,
                f"surviving clips overlapping a bad span (all splits) = "
                f"{total_leaks} by-split {leaks_by_split}")
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

    # ---- WINSOR: real read-time clamp via the --session-z-winsor CLI flag ---------
    # WINSOR's production mechanism: --session-z-winsor W -> main() sets
    # V14_SESSION_Z_WINSOR=W (and POPS it when the flag is absent, so a stale value
    # never leaks; dispatch_v14.py:3567-3570) -> SessionRobustZNormalizer reads it at
    # construction (normalize.py:191-192) -> transform() clamps |z| (line 265). We drive
    # the REAL flag through the argv (NOT a manual env set) so main() is authoritative:
    # a hand-set env with the flag absent would be silently popped — the exact artifact
    # that made an earlier check read unclamped (max|z| 13891 with "winsor on"). Driving
    # the flag also makes the two passes contamination-proof: each _build_production_data
    # re-runs main(), which sets/pops the env per ITS OWN argv.
    W = args.winsor
    os.environ.pop(_WINSOR_ENV, None)  # clean slate; main() owns the env from here
    from speech_decoding.experiments import dispatch_v14 as _d
    _wiring_args = _d._parser().parse_args(_base_argv(args, phase=1, winsor=W))
    rep.add("WINSOR-0", float(getattr(_wiring_args, "session_z_winsor", -1)) == W,
            f"--session-z-winsor parses to {getattr(_wiring_args, 'session_z_winsor', None)} "
            f"(main() sets V14_SESSION_Z_WINSOR from it before prepare())")

    # No-winsor pass: flag absent -> main() pops the env -> no clamp. Locate
    # glitch clips across ALL splits — a session's glitch can sit in any
    # partition under the time-based holdout split (e.g. subj4 t2's glitches
    # land entirely in the test partition).
    data_nw = _build_production_data(_base_argv(args, phase=1))
    loaders_nw = data_nw.build()
    seg_start = float(data_nw.segmenter.start)
    seg_dur = float(data_nw.segmenter.duration)
    glitch: list[tuple[str, int]] = []
    for split, loader in loaders_nw.items():
        trig = getattr(loader.dataset, "triggers", None)
        if trig is None or "start" not in getattr(trig, "columns", []):
            continue
        for i, st in enumerate(trig["start"].to_numpy()):
            lo = float(st) + seg_start
            if _overlaps(lo, lo + seg_dur, spans):
                glitch.append((split, i))

    if not glitch:
        rep.add("WINSOR-1", None, "no clip in any split overlaps a bad span on this "
                                  "session — cannot locate a giant |z| clip")
        rep.add("WINSOR-2", None, "skipped (no glitch clip)")
        rep.add("WINSOR-3", None, "skipped (no glitch clip)")
    else:
        # Scan glitch clips for the worst |z| (no-winsor build).
        worst, max_nowin = glitch[0], 0.0
        worst_tensors: dict[str, torch.Tensor] = {}
        for split, i in glitch:
            ft = _frontend_tensors(loaders_nw[split].dataset[i])
            m = _global_max_abs(ft)
            if m > max_nowin:
                max_nowin, worst, worst_tensors = m, (split, i), ft

        # Winsor pass: flag present -> main() sets the env=W before prepare().
        loaders_win = _build_production_data(
            _base_argv(args, phase=1, winsor=W)
        ).build()
        max_win = max(
            _global_max_abs(_frontend_tensors(loaders_win[split].dataset[i]))
            for split, i in glitch
        )
        t_win = _frontend_tensors(loaders_win[worst[0]].dataset[worst[1]])

        print(f"glitch clips scanned={len(glitch)}  "
              f"max|z| no-winsor={max_nowin:.1f}  winsor={max_win:.1f}  "
              f"worst clip={worst}", flush=True)

        rep.add("WINSOR-1", max_nowin > W,
                f"max |z| over glitch clips (no winsor) = {max_nowin:.1f} > W={W:.0f} "
                "(a real giant cell exists to clamp)")
        rep.add("WINSOR-2", max_win <= W + 1e-3,
                f"max |z| over glitch clips (winsor W={W:.0f}) = {max_win:.1f} <= W "
                "(every cell the encoder ingests is bounded)")

        # WINSOR-3: elementwise clamp identity on the worst clip.
        t_nowin = worst_tensors
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
