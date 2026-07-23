"""r4 (v14_converged_v3 Design B) depth-ladder probe — Stage 1 encode.

Adapts the r1 CS-erosion probe (``v3_probe_encode.py``) to the r4 tower. r4 is a DIFFERENT
architecture — ``PerBandStem`` decimates each band to its own token rate (HGA 32 / MID 16 /
SLOW 4 tokens per 1 s clip) and a flat-L1 encoder attends over the RAGGED per-(contact,band)
tokens packed varlen per shaft (``pack_r4.build_r4_grid``). There is no ``forward_padded``;
the teacher runs ``encoder.forward_flat`` over the packed grid.

Taps (Ben 2026-07-15): enc0, enc3, enc6, enc12. Perceiver taps (Ben 2026-07-16): dec, lat.
  - enc0 = the pre-projection DECIMATED raw band bins (``x[..., ::stride]`` per band, the exact
    tensor ``PerBandStem``'s per-band Linear consumes) — the M9 input-linear floor, decimated to
    what the model actually sees. No model, CPU-side (computed here so bands aren't re-read).
  - enc3/6/12 = raw block outputs of the EMA teacher (``_TargetTower``, the shipped
    representation), read in ONE forward via ``tap_blocks=(3,6,12)``.
  - dec/lat = the write-only secondary Perceiver, run on the SAME teacher ``z`` (the deep-sup
    1024-concat the tap forward already computes and the depth ladder discards).
      * ``dec`` (n,|P|,S·128) = the DECODE cross-attn per-(parcel,slot) query read, PRE-head.
        Already per-parcel — the model's own read, so NO electrode pooling is applied.
      * ``lat`` (n,1,S·M·128) = the processed latent bank. PARCEL-FREE by construction (the
        12 latents are shared learned params), hence natively subject-independent: CS needs
        NO parcel intersection. This is the bottleneck's information CEILING — ``dec`` is a
        deterministic function of ``(lat, parcel, slot)``, so no query set can exceed it.
        (That is also why a fixed-75-parcel query was dropped: provably redundant with lat.)
  Each Perceiver tap has a ``_rand`` twin from a FRESH-INIT Perceiver on the same teacher z —
  the control isolating what the Perceiver learned with the encoder held fixed by construction.
  Verified 2026-07-16: at the 10k ckpt every Perceiver tensor sits at std ~0.0033, a uniform
  decay from the 0.02 init ⇒ zero loss gradient ever (λ=0 until 10000), so "trained" means the
  10k→20k λ ramp alone.

Expect WEAK absolutes from dec/lat: the Perceiver's only loss was Gaussian NLL on a 6-D
band-power state target, so word-level information there is incidental, not optimized-for. The
informative contrasts are lat CS vs lat WS (does the code transfer?) and tap vs its _rand twin
(did the bottleneck learn to compress?) — both matched, hence immune to the teacher-vs-context
stream caveat below.

Stream caveat: the Perceiver trained reading the CONTEXT encoder over VISIBLE tokens; here it
reads the EMA TEACHER over the FULL grid, to stay on the same axis as enc0/3/6/12. τ=0.99925
makes the teacher a ~1333-step EMA (small); the full-vs-masked gap is the real one. Both sides
of every reported contrast carry it identically.

Feature = keep-time, NATIVE ragged (no hold-up: linear ridge, frame alignment irrelevant),
ELECTRODES POOLED TO PARCELS at encode (Ben 2026-07-15, OOM guard): per (band-slot, parcel)
MEAN over electrodes-in-parcel (Ben 2026-07-16: mean is the transferable parcel summary; std
conflates with per-subject electrode sampling, a CS nuisance). WS uses all present parcels; CS
intersects anchor/test parcels at readout. Pooling is per-(row, band-slot, parcel) hence
row-independent, so pre-pooling then row-subsetting is numerically identical to pool-at-readout
— but it shrinks the cache from ~14 GB/session (per-electrode keep-time) to well under 1 GB.

Runs from the LIVE r4 tree /projects/bhqk/htang13/speech (@ 2d3f52d = the ckpt's commit), so
the teacher state_dict matches by construction. GPU for enc3/6/12; enc0 + pooling are CPU.

Usage (one 1-GPU allocation):
  .venv/bin/python -m scripts.neuroprobe.v3_probe_encode_r4 \
      --ckpt /projects/bhqk/htang13/v3_ckpt_r4/ladder-step=10000.ckpt --tag r4_10k \
      --out-dir /projects/bhqk/htang13/v3_probe_cache_r4_10k \
      --band-cache-dir <slow> --band-cache-dir <mid> --band-cache-dir <hga> \
      --span-dir <spans> --bt-root <bt_root>

BOARD MODE (Ben 2026-07-16) — the Neuroprobe-Lite leaderboard-parity encode, consumed by
``v3_board_readout.py``. Three flags move the eval universe to the board's; the FEATURES and
the teacher forward are untouched, so the diagnostic and the board number are the same probe:
  --sessions board        the 12 Lite sessions (upstream NEUROPROBE_LITE_SUBJECT_TRIALS),
                          not the 7-session diagnostic cohort
  --tasks board15         the 15 leaderboard tasks — a RE-LABEL of the same task-agnostic
                          features (build_session_targets reads events["task"].unique())
  --electrode-set lite    the Lite montage, injected as load_v3_sessions(keep_labels_fn=...)
                          so keep_idx/parcel_id/sidecar/geom/band_stats are all BUILT on the
                          Lite axis — geom cannot be masked after the fact
  --no-perceiver          the board number is enc12; skip dec/lat

  .venv/bin/python -m scripts.neuroprobe.v3_probe_encode_r4 \
      --ckpt /projects/bhqk/htang13/v3_ckpt_r4/ladder-step=20000.ckpt --tag board_r4_20k \
      --out-dir /projects/bhqk/htang13/v3_board_cache \
      --band-cache-dir <slow_lite> --band-cache-dir <mid_lite> --band-cache-dir <hga_lite> \
      --span-dir <spans> --bt-root <bt_root> \
      --sessions board --tasks board15 --electrode-set lite --no-perceiver
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

PROBE_TASKS: tuple[str, ...] = ("onset", "delta_volume", "word_index", "gpt2_surprisal")
# The 15 Neuroprobe leaderboard tasks (upstream ``neuroprobe.config.NEUROPROBE_TASKS``).
# --tasks board re-labels the SAME task-agnostic features, so the only extra encode cost is
# materializing 15 label vectors instead of 4.
BOARD_TASKS: tuple[str, ...] = (
    "onset", "speech", "volume", "delta_volume", "pitch", "word_index",
    "word_gap", "gpt2_surprisal", "word_head_pos", "word_part_speech",
    "word_length", "global_flow", "local_flow", "frame_brightness", "face_num",
)
# The 12 Neuroprobe-Lite sessions (upstream ``NEUROPROBE_LITE_SUBJECT_TRIALS``): the board's
# CS anchor (2,4) + the 10 CS test cells + (2,0). --sessions board evaluates these.
BOARD_SESSIONS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2), (2, 0), (2, 4), (3, 0), (3, 1),
    (4, 0), (4, 1), (7, 0), (7, 1), (10, 0), (10, 1),
)
FPS = 32.0
CLIP_DUR_S = 1.0
N_PARCELS = 75
GPU_TAPS: tuple[int, ...] = (3, 6, 12)   # raw block outputs read in one teacher forward
PERC_RAND_SEED = 33                      # the launch seed — the fresh-init control's twin


def _load_ckpt(ckpt_path: str) -> dict:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return raw["state_dict"] if "state_dict" in raw else raw


def _subtree(sd: dict, pref: str) -> dict:
    """The ``pref``-rooted subtree of a LightningModule state_dict, de-compiled + de-``model.``d."""
    out = {}
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk.startswith("model."):
            kk = kk[len("model."):]
        if kk.startswith(pref):
            out[kk[len(pref):]] = v
    return out


def _freeze(m, *, device):
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _load_teacher(sd: dict, *, device: torch.device, pref: str = "objective.teacher.model.",
                  early_fusion: bool = False, no_fusion: bool = False, nf_decimate: int = 2):
    """Load ONLY the shipped encoder tower (`_TargetTower` = PerBandStem + encoder) from the ckpt.

    Filter the LightningModule state_dict to the ``pref``-rooted subtree and load it into a fresh
    ``_TargetTower`` (strict) — no need to build the full model / secondary head, so the load is
    independent of the objective's post-launch changes (#46 mean floor).

    ``pref`` selects which tower: the JEPA EMA teacher (default) or, for the MAE arm which has no
    teacher, ``objective.online.`` — the online encoder is the MAE arm's deployed representation.
    The online subtree is key- and shape-identical to the teacher subtree (verified on arm0, which
    carries both), so it loads into the same ``_TargetTower`` shell unchanged."""
    from speech_decoding.models.v14_converged_v3.objective import _TargetTower

    tsd = _subtree(sd, pref)
    if not tsd:
        raise RuntimeError(f"no '{pref}*' keys in ckpt; wrong ckpt layout")
    peek = [v.shape[0] for kk, v in tsd.items() if kk.endswith("parcel_embed.embed.weight")]
    if peek and int(peek[0]) != N_PARCELS:
        raise ValueError(f"ckpt parcel table {peek[0]} != expected {N_PARCELS}")
    tower = _TargetTower(n_parcels=N_PARCELS, deep_sup=True, early_fusion=early_fusion,
                         no_fusion=no_fusion, nf_decimate=nf_decimate)
    missing, unexpected = tower.load_state_dict(tsd, strict=False)
    bad = [m for m in missing if "num_batches_tracked" not in m]
    if bad or unexpected:
        raise RuntimeError(f"teacher state_dict mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
    return _freeze(tower, device=device)


def _load_perceivers(sd: dict, *, device: torch.device):
    """The ckpt's TRAINED Perceiver + a FRESH-INIT twin → ``{"": trained, "_rand": control}``.

    Both consume the SAME teacher z, so the pair holds the encoder fixed by construction and
    the only moving part is what the Perceiver learned. The twin re-inits at the launch seed,
    NOT the 10k ckpt's weights: 10k would confound Perceiver state with encoder state (10k vs
    20k encoder) and its 6× weight-decay shrink puts it off its own init distribution."""
    from speech_decoding.models.v14_converged_v3.perceiver import PerceiverHead

    pref = "objective.perceiver."
    psd = _subtree(sd, pref)
    if not psd:
        raise RuntimeError(f"no '{pref}*' keys in ckpt; a deep_sup=False run has no Perceiver")
    # A diag_nll / point-head run (r5 Arm 3, r5mod, ctx-loss) writes a mu-only head with NO
    # covariance params; a full-cov run writes head.chol_head.*. Match the head to the ckpt so
    # both load strict-clean. (The enc-only readout never reads the Perceiver taps, but the
    # encode loads it unconditionally, so a shape skew here is a hard crash.)
    point_only = not any("chol_head" in k for k in psd)
    trained = PerceiverHead(n_parcels=N_PARCELS, point_only=point_only)
    missing, unexpected = trained.load_state_dict(psd, strict=False)
    bad = [m for m in missing if "num_batches_tracked" not in m]
    if bad or unexpected:
        raise RuntimeError(f"perceiver state_dict mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
    torch.manual_seed(PERC_RAND_SEED)
    rand = PerceiverHead(n_parcels=N_PARCELS, point_only=point_only)
    return {"": _freeze(trained, device=device), "_rand": _freeze(rand, device=device)}


def _load_targets(session, bt_root, tasks=PROBE_TASKS):
    from scripts.neuroprobe.run_pretrain_probe_suite import _label_events
    from speech_decoding.experiments.pretrain_probe_labels import build_session_targets

    subject_id, trial_id = session
    events = _label_events(subject_id, trial_id, f"btbank{subject_id}_trial{trial_id}",
                           tasks, bt_root, lite_cap=True)
    # build_session_targets derives its task list from events["task"].unique(), so passing
    # `tasks` to _label_events is what widens 4 -> 15.
    return build_session_targets(events, subject_id=subject_id, trial_id=trial_id)


def _lite_keep_labels_fn(bt_root):
    """``keep_labels_fn`` restricting a session to its Neuroprobe-Lite montage.

    Injected into ``load_v3_sessions`` (NOT applied to the built spec) so keep_idx / parcel_id
    / sidecar / geom / band_stats are all constructed on the Lite axis by the normal code path
    — ``geom`` cannot be masked post-hoc, its gather_idx stores indices into the survivor axis.
    The realized montage is a SET intersection (``voltage_order ∩ lite_labels``), matching
    upstream ``datasets.py`` ``[full.index(e) for e in lite if e in full]``; we keep OUR voltage
    order, which is free for the encoder (the per-parcel pool is permutation-invariant within a
    parcel) and is what index-RoPE expects."""
    from speech_decoding.studies.braintreebank.anatomy import lite_electrode_set

    def fn(subject_id, trial_id, labels):
        return set(lite_electrode_set(subject_id))

    return fn


def _window_bands(spec, starts, clip_frames, rate_mult: int = 1):
    """Slice + robust-z every union window from the continuous spec caches. Returns one tensor
    per band, each (n_windows, N, F_band, T=clip_frames·rate_mult).

    ``rate_mult`` = cache frames per 32 Hz reference frame: 1 for the r4 uniform-32 Hz caches,
    2 for the r5 native-64 Hz caches (R5_BAND_RATES=((2,1),(2,1))) whose EarlyFusionStem
    conv-pools 64→32 Hz. Windows are sliced in the cache's own frame units (FPS·rate_mult), so
    the returned length is clip_frames·rate_mult = L = 2·T for r5.

    Bulk-loads each band's survivor rows into RAM ONCE (``mm[keep]`` → ~1 GB/band), then slices
    windows from RAM and robust-z's the whole (n,N,F,T) batch in one vectorized call. The naive
    per-window ``mm[keep, :, a:b]`` did ~n·N scattered Lustre reads (2.8M for one session) and
    dominated wall-clock; this is numerically identical (mm[keep] then slice == mm[keep,:,a:b];
    the normalizer is elementwise broadcast)."""
    keep = spec.keep_idx.numpy()
    # spec.n_frames is the 32 Hz reference count; the r5 caches carry rate_mult× that many
    # native frames. Bound the native-unit windows against the native length.
    n_frames_native = spec.n_frames * rate_mult
    t0 = np.rint(np.asarray(starts, dtype=float) * FPS * rate_mult).astype(np.int64)
    end = t0 + clip_frames * rate_mult
    oob = np.where((t0 < 0) | (end > n_frames_native))[0]
    if len(oob):
        raise RuntimeError(
            f"{spec.session_key}: {len(oob)} union windows out of cache bounds "
            f"(n_frames_native={n_frames_native}, first bad start={float(starts[oob[0]]):.4f}s)"
        )
    bands = []
    for path, norm in zip(spec.band_paths, spec.band_norms):
        mm = np.load(path, mmap_mode="r")
        full = np.asarray(mm[keep], dtype=np.float32)              # (N, F, T_total) bulk → RAM
        del mm
        clips = np.stack([full[:, :, a:b] for a, b in zip(t0.tolist(), end.tolist())], axis=0)
        bands.append(norm.transform(torch.from_numpy(clips)))     # (n, N, F_b, T32) vectorized
    return bands                                                  # 3 × (n, N, F_b, T32)


def _canon_parcels(grid, parcel_id):
    """Canonical (grid-order) contact indices + their parcel ids + present parcel atlas ids.

    build_r4_grid lays tokens contact-major (k_full block per contact); the first token of
    each block carries that contact's index (``grid.contact``), so reshaping to (n, k_full)
    and taking column 0 recovers the n canonical contacts and their parcels."""
    k = grid.k_full
    canon = grid.contact.reshape(-1, k)[:, 0].cpu().numpy()         # (n,) contact index into N
    parcel_canon = parcel_id.cpu().numpy()[canon]                   # (n,) DKT tag per canon contact
    present = np.unique(parcel_canon)                              # sorted present atlas ids
    return canon, parcel_canon, present


def _pool_parcels(x, parcel_canon, present):
    """Pool electrodes→parcels: x (B, n, *feat) → (B, |P|, prod(feat)) flattened last dim.

    Per present parcel, MEAN over its electrodes (Ben 2026-07-16: mean is the transferable
    parcel summary; std-over-electrodes conflates with per-subject electrode count/placement,
    a CS nuisance). Returns (B, |P|, F) fp16, parcel order == present."""
    B = x.shape[0]
    blocks = []
    for p in present:
        cols = np.where(parcel_canon == p)[0]
        sub = x[:, cols]                                          # (B, |cols|, *feat)
        blocks.append(sub.mean(1).reshape(B, -1))                 # (B, prod(feat))
    return torch.stack(blocks, dim=1).to(torch.float16)           # (B, |P|, F)


def _enc0_pooled(bands, canon, parcel_canon, present):
    """enc0 input floor: per band decimate ``x[..., ::stride]`` (the model's own input frames),
    reorder to canonical contacts, pool to parcels, concat bands → (n_win, |P|, F0)."""
    from speech_decoding.models.v14_converged_v3.pack_r4 import BAND_STRIDES

    per_band = []
    for x, st in zip(bands, BAND_STRIDES):                        # x (n, N, F_b, T32)
        xd = x[..., ::st]                                         # (n, N, F_b, T_b) decimated
        xd = xd.transpose(-1, -2).contiguous()                   # (n, N, T_b, F_b) time-major
        xd = xd[:, canon]                                        # (n, n_canon, T_b, F_b)
        per_band.append(_pool_parcels(xd, parcel_canon, present))  # (n, |P|, T_b·2F_b)
    return torch.cat(per_band, dim=-1)                            # (n, |P|, F0)


def _perc_queries(present, n_slots, *, device):
    """Parcel-major (q_parcel, q_slot), mirroring objective._secondary_nll's query build.

    Order [p0s0,p0s1,…,p1s0,…] so a (B, P·S, d) read reshapes to (B, P, S, d) — and the parcel
    axis then matches ``present``, i.e. the SAME order the enc taps are pooled into, so the
    readout's atlas-id intersection needs no special case for dec."""
    p = torch.as_tensor(present, dtype=torch.long, device=device)
    q_parcel = p.repeat_interleave(n_slots)                       # (Q,) atlas ids
    q_slot = torch.arange(n_slots, device=device).repeat(len(p))  # (Q,) 0..S-1
    return q_parcel, q_slot


def _perc_forward(perc, z, grid, q_parcel, q_slot, *, n_slots):
    """One Perceiver forward → (dec (B,Q,d_perc), lat (B,n_slots·M,d_perc)).

    BOTH taps come from forward HOOKS, and deliberately so — ``forward`` returns only the head's
    mu/cov, and hooks keep the training path untouched while r4b is queued.
      * ``dec`` = the DECODE cross-attn output: the per-(parcel,slot) query read, PRE-head.
      * ``lat`` = the output of the LAST ``process`` block, which is exactly the bank the
        forward's own ``return_latents`` hands back (it returns ``lat`` after ``for blk in
        self.process: lat = blk(...)``, pre-decode).
    We hook ``process[-1]`` rather than pass ``return_latents=True`` because that kwarg does NOT
    exist on the deployed r4 tree (2d3f52d) — only in a later local commit — and scp'ing src/ to
    add it would touch the queued r4b run for no gain. Hooks work on BOTH trees, and the weights
    are identical across them (the commit is a pure API addition, no structural change).

    ``noise=None`` is safe BY CONSTRUCTION, not by luck: the count-dependent floor enters only
    at ``head(dec, noise=...)``, strictly AFTER both taps, so it cannot reach dec or lat (and
    secondary_head.GaussianStateHead.forward falls back to its fixed buffer regardless).

    ``token_time`` on the full grid is ``grid.time_pos`` expanded — literally what the packed
    path gathers from (``pack.time_pos = grid.time_pos[idx]``, pack_r4.py:214), and what the
    teacher's own full-grid forward already builds (towers.py:243). No mask machinery needed;
    ``key_mask=None`` stays correct because every full-grid token is real."""
    B = z.shape[0]
    grab: dict = {}
    handles = [
        perc.decode.register_forward_hook(lambda _m, _i, out: grab.__setitem__("dec", out)),
        perc.process[-1].register_forward_hook(lambda _m, _i, out: grab.__setitem__("lat", out)),
    ]
    try:
        perc(
            z,
            grid.time_pos[None].expand(B, grid.total),
            None,
            q_parcel[None].expand(B, q_parcel.shape[0]),
            q_slot[None].expand(B, q_slot.shape[0]),
            n_slots=n_slots,
            noise=None,
        )
    finally:
        for h in handles:
            h.remove()
    return grab["dec"], grab["lat"]


def _perc_checks(percs, z, grid, q_parcel, q_slot, *, n_slots) -> bool:
    """[check] the Perceiver is a BOTTLENECK and its weights actually loaded.

    Rolling the parcel ids under a FIXED slot order must leave ``lat`` bit-identical — lat is
    computed before any query touches the model — while MOVING ``dec``, whose query is a parcel
    embedding. A lat that moves ⇒ we hooked the wrong module; a dec that doesn't ⇒ the parcel
    embedding is dead. Separately the trained tap must differ from its fresh-init twin, else the
    ckpt load silently no-op'd and every contrast below would be a random-vs-random null."""
    P = q_parcel.shape[0] // n_slots
    if P < 2:
        print("[check] perceiver bottleneck: SKIPPED (|P|<2, the parcel roll is a no-op)", flush=True)
        return True
    rolled = q_parcel.reshape(P, n_slots).roll(1, 0).reshape(-1)
    dec_a, lat_a = _perc_forward(percs[""], z, grid, q_parcel, q_slot, n_slots=n_slots)
    dec_b, lat_b = _perc_forward(percs[""], z, grid, rolled, q_slot, n_slots=n_slots)
    dec_r, _ = _perc_forward(percs["_rand"], z, grid, q_parcel, q_slot, n_slots=n_slots)
    lat_free = torch.equal(lat_a, lat_b)
    dec_sens = not torch.equal(dec_a, dec_b)
    loaded = not torch.equal(dec_a, dec_r)
    ok = lat_free and dec_sens and loaded
    print(f"[check] perceiver: lat parcel-free={lat_free} dec parcel-sensitive={dec_sens} "
          f"trained!=rand={loaded} -> {'OK' if ok else 'VIOLATED'}", flush=True)
    return ok


@torch.no_grad()
def _encode_taps(teacher, percs, bands, grid, parcel_packed, parcel_canon, present,
                 *, device, batch_size, n_slots, run_checks, elec_taps=()):
    """One forward of the teacher over all windows → per-tap parcel-pooled keep-time features.

    Cache stores the raw parcel-mean feature (n,|P|,k_full·d) — the most flexible storage: a
    readout can standardize columns on train stats (the FM linear-probe convention) or feed it
    raw (r1/M9-comparable), but neither is recoverable from a baked per-token LN. So we keep raw
    only. Returns {tap: {'raw': ...}} for the enc taps plus the Perceiver's dec/lat (+ _rand
    twins), which ride the SAME forward: ``z`` is the deep-sup 1024-concat the teacher already
    computes and the depth ladder throws away, so the control costs no extra job and no extra
    encoder pass. dec/lat are NOT electrode-pooled — dec is already per-parcel (that IS the
    model's read) and lat has no parcel axis at all, so it is stored under one pseudo-parcel."""
    n = bands[0].shape[0]
    k = grid.k_full
    acc: dict = {t: [] for t in GPU_TAPS}
    # Per-electrode keep-time (Ben 2026-07-16): WS keeps ALL electrodes; the parcel-mean is the
    # comparison. Stored UNPOOLED on the canonical-contact axis (same order as parcel_canon), so
    # it is the pooled tap's exact pre-mean input — the diff is the pooling and nothing else.
    for t in elec_taps:
        acc[f"elec{t}"] = []
    for name in percs:
        acc[f"dec{name}"], acc[f"lat{name}"] = [], []
    checked = not run_checks
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        bb = [b[s:e].to(device) for b in bands]
        Bb = e - s
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            z, taps = teacher.forward(bb, grid, parcel_packed, tap_blocks=GPU_TAPS)
            q_parcel, q_slot = _perc_queries(present, n_slots, device=device)
            if not checked and percs:
                if not _perc_checks(percs, z, grid, q_parcel, q_slot, n_slots=n_slots):
                    raise RuntimeError("perceiver invariant VIOLATED — refusing to write a cache")
                checked = True
            perc_out = {name: _perc_forward(p, z, grid, q_parcel, q_slot, n_slots=n_slots)
                        for name, p in percs.items()}
        for t in GPU_TAPS:
            enc = taps[t].float().reshape(Bb, -1, k, taps[t].shape[-1]).cpu()  # (Bb, n, k, d)
            acc[t].append(_pool_parcels(enc, parcel_canon, present))
            if t in elec_taps:
                # (Bb, n_contacts, k·d) fp16 — the SAME tensor, just unpooled.
                acc[f"elec{t}"].append(enc.reshape(Bb, enc.shape[1], -1).to(torch.float16))
        for name, (dec, lat) in perc_out.items():
            # dec (Bb, P·S, d) → (Bb, P, S·d): per-parcel already, no pooling.
            acc[f"dec{name}"].append(
                dec.float().reshape(Bb, len(present), -1).cpu().to(torch.float16))
            # lat (Bb, S·M, d) → (Bb, 1, S·M·d): parcel-FREE, one pseudo-parcel.
            acc[f"lat{name}"].append(lat.float().reshape(Bb, 1, -1).cpu().to(torch.float16))
    return {t: {"raw": torch.cat(v, 0)} for t, v in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True,
                   help="3× in v3 concat order: slow, mid, hga")
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sessions", choices=("cohort7", "board"), default="cohort7",
                   help="cohort7 = the 7-session diagnostic cohort; board = the 12 Neuroprobe-Lite sessions")
    p.add_argument("--tasks", choices=("probe4", "board15"), default="probe4",
                   help="board15 = the 15 leaderboard tasks (re-labels the same features)")
    p.add_argument("--electrode-set", choices=("all", "lite"), default="all",
                   help="lite = the Neuroprobe-Lite montage (leaderboard parity)")
    p.add_argument("--frontend", choices=("v3r4", "v3r5", "v3r5nf", "v3r5nffast"), default="v3r4",
                   help="v3r4 = 3-band PerBandStem @32Hz (r4/arm0, default). v3r5 = Chang "
                        "2-stream EarlyFusionStem: 2 native-64Hz caches (hga, lfs) → ONE 32Hz "
                        "token. v3r5nf = NoFusionStem: the SAME 2 caches, each stem-pooled into "
                        "its OWN 32Hz token stream (two-band grid, decimate 2). v3r5nffast = "
                        "v3r5nf with the first stem conv at stride 2 → 16Hz tokens (decimate 4). "
                        "All three 2-stream frontends imply --online (MAE has no teacher) + "
                        "--no-perceiver + no enc0 (band strides are r4-only).")
    p.add_argument("--no-perceiver", action="store_true",
                   help="skip the dec/lat taps — the board number is enc12, and the Perceiver "
                        "taps need the secondary head's ckpt subtree")
    p.add_argument("--online", action="store_true",
                   help="probe the ONLINE encoder (objective.online.*) instead of the EMA teacher. "
                        "Required for the MAE arm (no teacher); implies --no-perceiver (MAE has no "
                        "Perceiver). Use it on a JEPA ckpt too for an online-vs-online parity match.")
    p.add_argument("--elec-taps", default="",
                   help="comma-separated GPU taps to ALSO write per-electrode (unpooled), e.g. "
                        "'12' -> feats['enc12_elec']. WS keeps all electrodes by default (Ben "
                        "2026-07-16); each costs ~N/|P| (~5x) the pooled tap on disk.")
    args = p.parse_args()

    from speech_decoding.experiments.pretrain_probe_suite import PROBE_COHORT_7
    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.pack_r4 import (
        build_r4_grid, build_r5_grid, build_r5nf_grid,
    )
    from speech_decoding.models.v14_converged_v3.stem import nf_token_geometry
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    is_r5 = args.frontend == "v3r5"
    is_r5nf = args.frontend in ("v3r5nf", "v3r5nffast")
    is_two_stream = is_r5 or is_r5nf
    # NoFusionStem net decimation: 4 for v3r5nffast (16 Hz tokens), else 2 (v3r5nf, 32 Hz).
    nf_dec = 4 if args.frontend == "v3r5nffast" else 2
    if is_two_stream:
        # r5 (EarlyFusion) / nf (NoFusion): 2 native-64Hz caches, no teacher/perceiver/enc0.
        # Force the flags the frontend implies so a bare `--frontend` is sufficient.
        args.online = True
        args.no_perceiver = True
    n_cache = 2 if is_two_stream else 3
    if len(args.band_cache_dirs) != n_cache:
        want = "(hga, lfs)" if is_two_stream else "(slow, mid, hga)"
        raise SystemExit(
            f"need {n_cache} --band-cache-dir {want}, got {len(args.band_cache_dirs)}")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_frames = round(CLIP_DUR_S * FPS)
    # n_slots is the Perceiver slot count; the r5 tree has the perceiver path ripped out, so
    # import SLOT_STRIDE lazily only on the r4 perceiver path (n_slots is unused for r5).
    if is_two_stream:
        n_slots = 0
    else:
        from speech_decoding.models.v14_converged_v3.perceiver import SLOT_STRIDE
        # S = T // slot_stride, exactly as state_target.raw_state_vectors derives it (32//8 = 4).
        n_slots, rem = divmod(clip_frames, SLOT_STRIDE)
        if rem:
            raise SystemExit(f"clip_frames={clip_frames} not divisible by SLOT_STRIDE={SLOT_STRIDE}")
    parcel_fn = make_bt_parcel_fn(args.bt_root)
    cohort = BOARD_SESSIONS if args.sessions == "board" else tuple(PROBE_COHORT_7)
    tasks = BOARD_TASKS if args.tasks == "board15" else PROBE_TASKS
    keep_labels_fn = _lite_keep_labels_fn(args.bt_root) if args.electrode_set == "lite" else None
    elec_taps = tuple(int(t) for t in args.elec_taps.split(",") if t.strip())
    bad = [t for t in elec_taps if t not in GPU_TAPS]
    if bad:
        raise SystemExit(f"--elec-taps {bad} not in GPU_TAPS {GPU_TAPS}")
    sd = _load_ckpt(args.ckpt)
    tower_pref = "objective.online." if args.online else "objective.teacher.model."
    teacher = _load_teacher(sd, device=device, pref=tower_pref, early_fusion=is_r5,
                            no_fusion=is_r5nf, nf_decimate=nf_dec)
    # --online (MAE arm, or a JEPA online-vs-online match) has no Perceiver subtree to load.
    percs = {} if (args.no_perceiver or args.online) else _load_perceivers(sd, device=device)
    del sd
    perc_note = ("online encoder, no perceiver" if args.online
                 else "no perceiver" if args.no_perceiver
                 else "+ perceiver dec/lat (+_rand control)")
    print(f"[encode-r4] tag={args.tag} device={device} gpu_taps={GPU_TAPS} + enc0 "
          f"{perc_note}, n_slots={n_slots} sessions={args.sessions}({len(cohort)}) "
          f"tasks={args.tasks}({len(tasks)}) electrodes={args.electrode_set}", flush=True)

    first = True
    for session in cohort:
        subject_id, trial_id = session
        path = os.path.join(args.out_dir, f"enc_s{subject_id}_t{trial_id}_{args.tag}.pt")
        if os.path.exists(path):
            print(f"[encode-r4] {session}: exists, skip -> {path}", flush=True)
            continue
        load_kw = {}
        if is_two_stream:
            from speech_decoding.models.v14_converged_v3.dataset import R5_BAND_RATES
            load_kw["band_rates"] = R5_BAND_RATES
        spec = load_v3_sessions(
            sessions=[session], band_cache_dirs=args.band_cache_dirs, span_dir=args.span_dir,
            parcel_fn=parcel_fn, lof_report_path=None,
            winsor=(20.0, 15.0) if is_two_stream else (15.0, 15.0, 20.0),
            keep_labels_fn=keep_labels_fn, **load_kw,
        )[0]
        if keep_labels_fn is not None:
            # The montage is the WHOLE parity claim — print what was realized, per session.
            from speech_decoding.studies.braintreebank.anatomy import lite_electrode_set
            lite = lite_electrode_set(subject_id)
            kept = spec.setup.sidecar.labels
            ok = set(kept) <= set(lite)
            print(f"[check] lite montage s{subject_id}_t{trial_id}: kept={len(kept)} "
                  f"of lite-list {len(lite)} subset-of-lite={ok} -> {'OK' if ok else 'VIOLATED'}",
                  flush=True)
            if not ok:
                raise RuntimeError("lite montage kept a non-Lite electrode — refusing to write")
        targets = _load_targets(session, args.bt_root, tasks)
        # r5/nf caches are native-64Hz (R5_BAND_RATES 2×); read 2·clip_frames frames per window.
        bands = _window_bands(spec, targets.clip_starts, clip_frames,
                              rate_mult=2 if is_two_stream else 1)

        geom = spec.setup.geom.to(device)
        parcel_id = spec.setup.parcel_id.to(device)
        if is_r5:
            grid = build_r5_grid(geom, n_time=clip_frames)
        elif is_r5nf:
            n_tok, time_stride = nf_token_geometry(clip_frames, decimate=nf_dec)
            grid = build_r5nf_grid(geom, n_time=n_tok, time_stride=time_stride)
        else:
            grid = build_r4_grid(geom, n_time=clip_frames)
        parcel_packed = parcel_id[grid.contact]
        canon, parcel_canon, present = _canon_parcels(grid, parcel_id)

        # enc0 = the r4 per-band decimated input floor (BAND_STRIDES is r4-only); the r5/nf
        # 2-stream frontends have no analogous decimation, so skip it — enc3/6/12 are the arms.
        feats = ({} if is_two_stream
                 else {"enc0": {"raw": _enc0_pooled(bands, canon, parcel_canon, present)}})
        tap_pooled = _encode_taps(teacher, percs, bands, grid, parcel_packed, parcel_canon,
                                  present, device=device, batch_size=args.batch_size,
                                  n_slots=n_slots, run_checks=first, elec_taps=elec_taps)
        first = False
        for t in GPU_TAPS:
            feats[f"enc{t}"] = tap_pooled[t]
        for t in elec_taps:
            feats[f"enc{t}_elec"] = tap_pooled[f"elec{t}"]
        for name in percs:
            feats[f"dec{name}"] = tap_pooled[f"dec{name}"]
            feats[f"lat{name}"] = tap_pooled[f"lat{name}"]

        payload = {
            "subject_id": subject_id, "trial_id": trial_id, "ckpt_tag": args.tag,
            "present_parcels": np.asarray(present, dtype=np.int64),   # (|P|,) atlas ids, feature order
            "band_lengths": tuple(int(x) for x in grid.band_lengths),
            "feats": {k: {v: t for v, t in d.items()} for k, d in feats.items()},
            "clip_starts": np.asarray(targets.clip_starts),
            "labels": {lt: np.asarray(v) for lt, v in targets.labels.items()},
            "ws_split": targets.ws_split,
            "cs_split": targets.cs_split,
            "n_windows": int(bands[0].shape[0]),
        }
        torch.save(payload, path)
        shp = {k: tuple(next(iter(d.values())).shape) for k, d in feats.items()}
        print(f"[encode-r4] {session}: |P|={len(present)} n={payload['n_windows']} "
              f"shapes={shp} -> {path}", flush=True)
        del bands, feats, tap_pooled, payload


if __name__ == "__main__":
    main()
