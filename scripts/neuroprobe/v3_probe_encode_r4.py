"""r4 (v14_converged_v3 Design B) depth-ladder probe — Stage 1 encode.

Adapts the r1 CS-erosion probe (``v3_probe_encode.py``) to the r4 tower. r4 is a DIFFERENT
architecture — ``PerBandStem`` decimates each band to its own token rate (HGA 32 / MID 16 /
SLOW 4 tokens per 1 s clip) and a flat-L1 encoder attends over the RAGGED per-(contact,band)
tokens packed varlen per shaft (``pack_r4.build_r4_grid``). There is no ``forward_padded``;
the teacher runs ``encoder.forward_flat`` over the packed grid.

Taps (Ben 2026-07-15): enc0, enc3, enc6, enc12.
  - enc0 = the pre-projection DECIMATED raw band bins (``x[..., ::stride]`` per band, the exact
    tensor ``PerBandStem``'s per-band Linear consumes) — the M9 input-linear floor, decimated to
    what the model actually sees. No model, CPU-side (computed here so bands aren't re-read).
  - enc3/6/12 = raw block outputs of the EMA teacher (``_TargetTower``, the shipped
    representation), read in ONE forward via ``tap_blocks=(3,6,12)``.

The write-only secondary Perceiver (dec/lat taps, Ben 2026-07-16) is GONE: secondary = CUT, and
``v14_converged_v3.perceiver`` was deleted from the tree in ``acca0d4``. The encode kept importing
it on the r4 path, which is what broke the v3r4-vs-v3r6 enc0 parity job (2026-07-23).

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

  .venv/bin/python -m scripts.neuroprobe.v3_probe_encode_r4 \
      --ckpt /projects/bhqk/htang13/v3_ckpt_r4/ladder-step=20000.ckpt --tag board_r4_20k \
      --out-dir /projects/bhqk/htang13/v3_board_cache \
      --band-cache-dir <slow_lite> --band-cache-dir <mid_lite> --band-cache-dir <hga_lite> \
      --span-dir <spans> --bt-root <bt_root> \
      --sessions board --tasks board15 --electrode-set lite
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
GPU_TAPS: tuple[int, ...] = (3, 6, 9, 12)   # raw block outputs read in one teacher forward


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
                  early_fusion: bool = False, no_fusion: bool = False, nf_decimate: int = 2,
                  space_rope: bool = True):
    """Load ONLY the shipped encoder tower (`_TargetTower` = PerBandStem + encoder) from the ckpt.

    Filter the LightningModule state_dict to the ``pref``-rooted subtree and load it into a fresh
    ``_TargetTower`` (strict) — no need to build the full model / secondary head, so the load is
    independent of the objective's post-launch changes (#46 mean floor).

    ``pref`` selects which tower: the JEPA EMA teacher (default) or, for the MAE arm which has no
    teacher, ``objective.online.`` — the online encoder is the MAE arm's deployed representation.
    The online subtree is key- and shape-identical to the teacher subtree (verified on arm0, which
    carries both), so it loads into the same ``_TargetTower`` shell unchanged.

    ``deep_sup`` is READ OFF the ckpt, not passed in: deep-sup towers carry ``norms_block.*`` and
    no ``norm_out``, single-tap towers the reverse (towers.py:102/114), so the shell has to match
    or the strict-ish check below fires. The taps this script actually reads are captured BEFORE
    either norm (towers.py:171/311), so the flag changes which unused params exist, nothing else —
    deep-sup ckpts encode bit-identically to before."""
    from speech_decoding.models.v14_converged_v3.objective import _TargetTower

    tsd = _subtree(sd, pref)
    if not tsd:
        raise RuntimeError(f"no '{pref}*' keys in ckpt; wrong ckpt layout")
    peek = [v.shape[0] for kk, v in tsd.items() if kk.endswith("parcel_embed.embed.weight")]
    if peek and int(peek[0]) != N_PARCELS:
        raise ValueError(f"ckpt parcel table {peek[0]} != expected {N_PARCELS}")
    deep_sup = any(kk.startswith("encoder.norms_block.") for kk in tsd)
    # parcel_embed is READ OFF the ckpt for the same reason deep_sup is: the --no-parcel-embed
    # arm ships a tower with no parcel_embed submodule at all, so a shell built with one puts
    # parcel_embed.embed.weight in `missing` and the check below raises. Inferring it keeps that
    # check as the verifier — a wrong inference still fails loud, it is never silently absorbed.
    parcel_embed = bool(peek)
    # ⚠️ space_rope CANNOT be inferred and is NOT checked: L1RoPE zeroes idx_freq when
    # space=False and registers it persistent=False (pe.py), so a --no-space-rope ckpt is
    # key- AND value-identical to a space-rope-ON one. missing/unexpected both come back
    # empty and the check below PASSES while the tower rotates by contact index that the
    # trained model never saw. Nothing in the ckpt can catch it (no save_hyperparameters
    # anywhere), so the caller MUST pass it and the banner MUST record what was used.
    # d_model is READ OFF the ckpt like deep_sup/parcel_embed, and unlike space_rope it is
    # SELF-VERIFYING: every block's LayerNorm is d-wide, so a wrong width cannot load at all
    # (load_state_dict raises on a size mismatch whatever `strict` says). The width OFAT
    # therefore carries no silent-drift hazard and needs no CLI flag at encode.
    dkey = "encoder.blocks.0.norm1.weight"
    if dkey not in tsd:
        raise RuntimeError(f"ckpt has no '{dkey}'; cannot infer encoder width")
    d_model = int(tsd[dkey].shape[0])
    print(f"[encode] tower deep_sup={deep_sup} parcel_embed={parcel_embed} d_model={d_model} "
          f"(inferred from ckpt keys) space_rope={space_rope} (NOT inferable — from CLI)")
    tower = _TargetTower(n_parcels=N_PARCELS, deep_sup=deep_sup, parcel_embed=parcel_embed,
                         space_rope=space_rope, early_fusion=early_fusion,
                         no_fusion=no_fusion, nf_decimate=nf_decimate, d_model=d_model)
    missing, unexpected = tower.load_state_dict(tsd, strict=False)
    bad = [m for m in missing if "num_batches_tracked" not in m]
    if bad or unexpected:
        raise RuntimeError(f"teacher state_dict mismatch: missing={bad[:8]} unexpected={unexpected[:8]}")
    return _freeze(tower, device=device)


def _load_targets(session, bt_root, tasks=PROBE_TASKS):
    from scripts.neuroprobe.run_pretrain_probe_suite import _label_events
    from speech_decoding.experiments.pretrain_probe_labels import build_session_targets

    subject_id, trial_id = session
    events = _label_events(subject_id, trial_id, f"btbank{subject_id}_trial{trial_id}",
                           tasks, bt_root, lite_cap=True)
    # build_session_targets derives its task list from events["task"].unique(), so passing
    # `tasks` to _label_events is what widens 4 -> 15.
    return build_session_targets(events, subject_id=subject_id, trial_id=trial_id)


def _shift_and_trim(targets, *, offset_s: float, clip_frames: int, n_frames: int):
    """Shift every clip start by ``offset_s`` and drop windows that leave the session cache.

    Used by the visualization encode, which windows -0.5 -> +1.5 s around word onset instead
    of the leaderboard's [onset, onset+1 s]. A pre-onset offset pushes the earliest clips
    before frame 0 and a longer duration pushes the latest past the end; ``_window_bands``
    raises on both, so they must be dropped here.

    Dropping rows RENUMBERS the union axis, so labels and every ws/cs split index array are
    remapped together — a stale index would silently point at a different clip. Returns a new
    SessionTargets.
    """
    from speech_decoding.experiments.pretrain_probe_labels import SessionTargets

    starts = np.asarray(targets.clip_starts, dtype=float) + float(offset_s)
    t0 = np.rint(starts * FPS).astype(np.int64)
    valid = (t0 >= 0) & (t0 + clip_frames <= n_frames)
    remap = np.full(len(starts), -1, dtype=np.int64)
    remap[valid] = np.arange(int(valid.sum()), dtype=np.int64)

    def _idx(a):
        r = remap[np.asarray(a, dtype=np.int64)]
        return r[r >= 0]

    return SessionTargets(
        subject_id=targets.subject_id,
        trial_id=targets.trial_id,
        clip_starts=starts[valid],
        clip_durations=np.asarray(targets.clip_durations)[valid],
        clip_movie_onsets=np.asarray(targets.clip_movie_onsets)[valid],
        labels={k: np.asarray(v)[valid] for k, v in targets.labels.items()},
        ws_split={t: {f: {n: _idx(ix) for n, ix in sp.items()} for f, sp in folds.items()}
                  for t, folds in targets.ws_split.items()},
        cs_split={t: {n: _idx(ix) for n, ix in sp.items()}
                  for t, sp in targets.cs_split.items()},
    )


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


def _window_bands(spec, starts, clip_frames, rate_mult: int = 1, band_rates=None,
                  normalize: bool = True):
    """Slice + robust-z every union window from the continuous spec caches. Returns one tensor
    per band, each (n_windows, N, F_band, T_b).

    ``normalize=False`` returns the RAW (un-robust-z'd, un-winsored) |STFT| windows instead —
    the enc0 log-vs-abs gate needs the raw bins so it can apply ``log(x+eps)`` BEFORE a robust-z
    REFIT (frozen abs-stats are invalid for logged values: median commutes with log but MAD does
    not). Default True keeps the tap/production path byte-identical.

    ``rate_mult`` = cache frames per 32 Hz reference frame: 1 for the r4 uniform-32 Hz caches,
    2 for the r5 native-64 Hz caches (R5_BAND_RATES=((2,1),(2,1))) whose EarlyFusionStem
    conv-pools 64→32 Hz. Windows are sliced in the cache's own frame units (FPS·rate_mult), so
    the returned length is clip_frames·rate_mult = L = 2·T for r5.

    ``band_rates`` OVERRIDES the scalar ``rate_mult``: each band is sliced at its OWN cache rate,
    with the clip start snapped to the shared lattice (lcm of denominators, ``_start_align``) so
    all bands stay temporally aligned, exactly as the training dataset windows them
    (dataset.py:226, ``lo,hi = t0*num//den``). Per-band length clip_frames·num//den is integer by
    the dispatch invariant (clip_frames·num % den == 0). NO shipped frontend passes this today —
    every band cache is at the 32 Hz clip clock, so ``rate_mult`` covers r4/r6 (1) and r5 (2). It
    exists for a genuinely coarser-or-finer per-band bake, and is only correct when the declared
    rate matches the bake (``dataset.assert_band_rates_match_cache``); declaring a rate the cache
    does not have silently yields a compressed, time-shifted slice at the right SHAPE, which is
    what invalidated four r6 runs (memo project-r6-band-rates-cache-rate-bug-2026-07-23).

    Bulk-loads each band's survivor rows into RAM ONCE (``mm[keep]`` → ~1 GB/band), then slices
    windows from RAM and robust-z's the whole (n,N,F,T) batch in one vectorized call. The naive
    per-window ``mm[keep, :, a:b]`` did ~n·N scattered Lustre reads (2.8M for one session) and
    dominated wall-clock; this is numerically identical (mm[keep] then slice == mm[keep,:,a:b];
    the normalizer is elementwise broadcast)."""
    keep = spec.keep_idx.numpy()
    starts = np.asarray(starts, dtype=float)
    if band_rates is not None:
        from speech_decoding.models.v14_converged_v3.dataset import _start_align
        align = _start_align(band_rates)
        t0_ref = (np.rint(starts * FPS).astype(np.int64) // align) * align   # snap to lattice
        end_ref = t0_ref + clip_frames
        bands = []
        for (path, norm), (num, den) in zip(zip(spec.band_paths, spec.band_norms), band_rates):
            lo = t0_ref * num // den
            hi = end_ref * num // den
            mm = np.load(path, mmap_mode="r")
            full = np.asarray(mm[keep], dtype=np.float32)          # (N, F, T_native) bulk → RAM
            del mm
            n_native = full.shape[-1]
            oob = np.where((lo < 0) | (hi > n_native))[0]
            if len(oob):
                raise RuntimeError(
                    f"{spec.session_key}: band rate {num}/{den} {len(oob)} windows out of bounds "
                    f"(n_native={n_native}, first bad start={float(starts[oob[0]]):.4f}s)")
            clips = np.stack([full[:, :, a:b] for a, b in zip(lo.tolist(), hi.tolist())], axis=0)
            t = torch.from_numpy(clips)
            bands.append(norm.transform(t) if normalize else t)   # (n, N, F_b, T_b) vectorized
        return bands
    # uniform-rate path (r4 rate_mult=1, r5 rate_mult=2) — spec.n_frames is the 32 Hz reference
    # count; the r5 caches carry rate_mult× that many native frames.
    n_frames_native = spec.n_frames * rate_mult
    t0 = np.rint(starts * FPS * rate_mult).astype(np.int64)
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
        t = torch.from_numpy(clips)
        bands.append(norm.transform(t) if normalize else t)       # (n, N, F_b, T32) vectorized
    return bands                                                  # 3 × (n, N, F_b, T32)


def _robustz_refit(band, winsor, sigma_floor: float = 1e-6):
    """Per-(electrode, freq-bin) robust-z REFIT over a band's windows, then winsor clamp.

    band: (n_win, N, F, T). Stats (median, 1.4826*MAD) are computed per (N, F) over the flattened
    (n_win * T) frame axis — the windowed analogue of SessionRobustZNormalizer's per-session-time
    fit. Byte-mirrors ``extractors/normalize.robust_z`` (sigma-floor + constant-bin zeroing), then
    applies the read-time winsor clamp. Used ONLY by the enc0 log-vs-abs gate so both cells share
    identical robust-z machinery and differ solely by the pre-log; the abs cell is cross-checked
    against the frozen-stats enc0 at the CS level to confirm the window-support refit is sound."""
    n, N, F, T = band.shape
    flat = band.permute(1, 2, 0, 3).reshape(N, F, n * T)          # (N, F, n*T)
    median = flat.median(dim=-1, keepdim=True).values
    centered = flat - median
    mad = centered.abs().median(dim=-1, keepdim=True).values
    sigma = 1.4826 * mad
    z = centered / sigma.clamp(min=sigma_floor)
    z = torch.where(sigma >= sigma_floor, z, torch.zeros_like(z))  # constant bins -> 0
    z = z.reshape(N, F, n, T).permute(2, 0, 1, 3).contiguous()    # back to (n, N, F, T)
    if winsor is not None:
        z = z.clamp(-float(winsor), float(winsor))
    return z


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


def _parse_band_rates(spec: str, n_bands: int):
    """``'1/1,1/1,2/1'`` -> per-band ``(num, den)`` against the 32 Hz clip clock.

    This DECLARES each cache's own bake rate; it is not a resampling request. Every entry is
    checked against the cache's real ``band_hop`` by ``assert_band_rates_match_cache``, because a
    wrong declaration yields a compressed, time-shifted slice at the RIGHT shape rather than an
    error (the bug that invalidated four r6 runs, 2026-07-23)."""
    rates = []
    for part in spec.split(","):
        num, _, den = part.strip().partition("/")
        rates.append((int(num), int(den or 1)))
    if len(rates) != n_bands:
        raise SystemExit(
            f"--band-rates got {len(rates)} entries, need {n_bands} (one per --band-cache-dir)")
    return tuple(rates)


def _enc0_band_lengths(bands, strides):
    """enc0's per-band frame counts as the PAYLOAD actually carries them.

    ``grid.band_lengths`` is ``clip_frames // BAND_STRIDES`` on the 32 Hz clock, which is only
    the enc0 layout when every cache IS at 32 Hz. Under a mixed-rate bake (``--band-rates``) a
    64 Hz HGA carries 2x the frames, and a record claiming the grid's 32 would hand every
    downstream consumer a plausible, wrong slicing of enc0."""
    st = _enc0_strides(bands, strides)
    return tuple(int(-(-b.shape[-1] // s)) for b, s in zip(bands, st))  # len of x[..., ::s]


def _enc0_strides(bands, strides):
    """Resolve/validate the per-band decimation strides shared by enc0's pooled + elec paths."""
    from speech_decoding.models.v14_converged_v3.pack_r4 import BAND_STRIDES

    if strides is None:
        strides = BAND_STRIDES
    if len(strides) != len(bands):
        raise ValueError(f"enc0 got {len(bands)} bands but {len(strides)} strides")
    return strides


def _enc0_bands_canon(bands, canon, strides):
    """Per band: decimate ``x[..., ::stride]`` (the model's own input frames), reorder to
    canonical contacts, time-major. Yields (n_win, n_canon, T_b, F_b) tensors, one per band —
    the shared prefix of enc0's pooled and unpooled (elec) paths."""
    for x, st in zip(bands, strides):                             # x (n, N, F_b, T_clock)
        xd = x[..., ::st]                                         # (n, N, F_b, T_b) decimated
        xd = xd.transpose(-1, -2).contiguous()                   # (n, N, T_b, F_b) time-major
        yield xd[:, canon]                                       # (n, n_canon, T_b, F_b)


def _enc0_pooled(bands, canon, parcel_canon, present, strides=None):
    """enc0 input floor: per band decimate, reorder to canonical contacts, pool to parcels,
    concat bands → (n_win, |P|, F0).

    ``strides`` defaults to r4's ``BAND_STRIDES`` (8,2,1) on the 32 Hz clock. The 2-stream
    frontends pass their own: both r5/nf streams are cached at 64 Hz and their stem decimates
    64→32 by ``NOFUSION_DECIMATE`` (2; 4 for nffast), so ``(dec, dec)`` is the exact parity of
    r4's per-band strides — the raw bins each stem's first layer consumes."""
    strides = _enc0_strides(bands, strides)
    per_band = [_pool_parcels(xd, parcel_canon, present)
                for xd in _enc0_bands_canon(bands, canon, strides)]
    return torch.cat(per_band, dim=-1)                            # (n, |P|, F0)


def _enc0_elec(bands, canon, strides=None):
    """Unpooled enc0: same canonical-contact axis and band-concat order as ``_enc0_pooled``'s
    pre-pool input — the depth-0 sibling of ``_encode_taps``' ``enc{t}_elec`` (GPU taps
    3/6/12), which stores the equivalent unpooled tensor for those taps."""
    strides = _enc0_strides(bands, strides)
    per_band = []
    for xd in _enc0_bands_canon(bands, canon, strides):
        B = xd.shape[0]
        per_band.append(xd.reshape(B, xd.shape[1], -1).to(torch.float16))
    return torch.cat(per_band, dim=-1)                            # (n, n_canon, F0)


@torch.no_grad()
def _encode_taps(teacher, bands, grid, parcel_packed, parcel_canon, present,
                 *, device, batch_size, elec_taps=()):
    """One forward of the teacher over all windows → per-tap parcel-pooled keep-time features.

    Cache stores the raw parcel-mean feature (n,|P|,k_full·d) — the most flexible storage: a
    readout can standardize columns on train stats (the FM linear-probe convention) or feed it
    raw (r1/M9-comparable), but neither is recoverable from a baked per-token LN. So we keep raw
    only. Returns {tap: {'raw': ...}}."""
    n = bands[0].shape[0]
    k = grid.k_full
    # Per-electrode keep-time (Ben 2026-07-16): WS keeps ALL electrodes; the parcel-mean is the
    # comparison. Stored UNPOOLED on the canonical-contact axis (same order as parcel_canon), so
    # it is the pooled tap's exact pre-mean input — the diff is the pooling and nothing else.
    #
    # Each tap is preallocated at full length and written slice-by-slice. Accumulating a list of
    # per-batch tensors and torch.cat-ing at the end held the list AND its concatenation alive
    # simultaneously — a 2x peak on the ~40 GB enc12_elec tap, which is what forced --mem=300G
    # (measured 230-245 GiB on the 12-session board encode). Values, dtypes and row order are
    # unchanged; only the allocation pattern differs.
    acc: dict = {key: None for key in list(GPU_TAPS) + [f"elec{t}" for t in elec_taps]}

    def _write(key, lo, hi, x):
        if acc[key] is None:
            acc[key] = torch.empty((n, *x.shape[1:]), dtype=x.dtype)
        acc[key][lo:hi] = x

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        bb = [b[s:e].to(device) for b in bands]
        Bb = e - s
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            _z, taps = teacher.forward(bb, grid, parcel_packed, tap_blocks=GPU_TAPS)
        for t in GPU_TAPS:
            enc = taps[t].float().reshape(Bb, -1, k, taps[t].shape[-1]).cpu()  # (Bb, n, k, d)
            _write(t, s, e, _pool_parcels(enc, parcel_canon, present))
            if t in elec_taps:
                # (Bb, n_contacts, k·d) fp16 — the SAME tensor, just unpooled.
                _write(f"elec{t}", s, e, enc.reshape(Bb, enc.shape[1], -1).to(torch.float16))
    return {t: {"raw": v} for t, v in acc.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="required unless --enc0-only (enc0 never reads weights)")
    p.add_argument("--tag", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--band-cache-dir", dest="band_cache_dirs", action="append", required=True,
                   help="3× in v3 concat order: slow, mid, hga")
    p.add_argument("--span-dir", required=True)
    p.add_argument("--bt-root", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sessions", choices=("cohort7", "board"), default="cohort7",
                   help="cohort7 = the 7-session diagnostic cohort; board = the 12 Neuroprobe-Lite sessions")
    p.add_argument("--session-index", type=int, default=None,
                   help="encode ONLY cohort[i] — one Slurm array task per session "
                        "(feedback-always-parallel-shard-shorten-loop). Default = all.")
    p.add_argument("--tasks", choices=("probe4", "board15"), default="probe4",
                   help="board15 = the 15 leaderboard tasks (re-labels the same features)")
    p.add_argument("--electrode-set", choices=("all", "lite"), default="all",
                   help="lite = the Neuroprobe-Lite montage (leaderboard parity)")
    p.add_argument("--frontend", choices=("v3r4", "v3r5", "v3r5nf", "v3r5nffast", "v3r6"),
                   default="v3r4",
                   help="v3r4 = 3-band PerBandStem @32Hz (r4/arm0, default). v3r5 = Chang "
                        "2-stream EarlyFusionStem: 2 native-64Hz caches (hga, lfs) → ONE 32Hz "
                        "token. v3r5nf = NoFusionStem: the SAME 2 caches, each stem-pooled into "
                        "its OWN 32Hz token stream (two-band grid, decimate 2). v3r5nffast = "
                        "v3r5nf with the first stem conv at stride 2 → 16Hz tokens (decimate 4). "
                        "v3r6 = arm0's frontend VERBATIM (same 3 × 32 Hz caches, same "
                        "PerBandStem) — the encode path is byte-identical to v3r4; only --online "
                        "is implied (MAE has no EMA teacher), and enc0 is available exactly as on "
                        "v3r4. The two-stream frontends (v3r5/v3r5nf/v3r5nffast) imply --online + "
                        "no enc0 (band strides are r4-only).")
    p.add_argument("--parcel-perm-seed", dest="parcel_perm_seed", type=int, default=None,
                   help="the ckpt was trained with dispatch_v3 --parcel-perm-seed S (R35 "
                        "ablation). Like --no-space-rope this is NOT inferable from the ckpt "
                        "(no save_hyperparameters anywhere) and a wrong value silently feeds the "
                        "encoder a vocabulary it never learned, so it MUST be passed and the "
                        "[parcel-perm] fingerprint MUST match the training log. The readout's "
                        "parcel pooling keeps the TRUE atlas ids either way.")
    p.add_argument("--no-space-rope", dest="no_space_rope", action="store_true",
                   help="the ckpt was trained with dispatch_v3 --no-space-rope (ablation A2). "
                        "MUST be passed by hand: L1RoPE registers idx_freq persistent=False, so "
                        "a no-space-rope ckpt is key- and value-identical to a normal one and the "
                        "state_dict check CANNOT catch a mismatch — get this wrong and the encode "
                        "silently applies contact-index rotation the trained model never saw.")
    p.add_argument("--enc0-only", action="store_true",
                   help="compute ONLY the enc0 input floor (raw decimated band bins pooled to "
                        "parcels). No ckpt, no model, no GPU — enc0 never touched weights. Lets "
                        "the 2-stream frontends get an enc0 at parity with r4's (Ben 2026-07-23).")
    p.add_argument("--enc0-log", action="store_true",
                   help="enc0 log-vs-abs gate (implies --enc0-only). Writes TWO enc0 taps from the "
                        "same raw |STFT| windows with an identical robust-z REFIT: 'enc0' = "
                        "refit(abs), 'enc0_log' = refit(log(abs+1e-7)) — the ASR log-magnitude "
                        "preprocessing (view's own apply_log=True path, log_eps 1e-7). Both refit "
                        "over the clip windows so the ONLY difference is the pre-log; 'enc0' is "
                        "cross-checked at the CS level against the frozen-stats enc0 to confirm the "
                        "window-support refit is sound. mimics-asr scope-and-contract 2026-07-24.")
    p.add_argument("--enc0-log-eps", type=float, default=1e-7,
                   help="epsilon in log(|STFT|+eps) for --enc0-log; 1e-7 = the view's own log_eps "
                        "so this is byte-faithful to what apply_log=True would bake.")
    p.add_argument("--enc0-stride", type=int, default=None,
                   help="override the enc0 decimation applied to EVERY band (2-stream only). "
                        "Default = the frontend's own stem decimate (2 for v3r5/v3r5nf, 4 for "
                        "v3r5nffast) = stem parity @32Hz. Pass 1 for NO decimation = the full "
                        "native 64 Hz cache, which isolates what the stem's 2x downsample costs.")
    p.add_argument("--band-rates", default=None,
                   help="per-band NATIVE cache rate as num/den of the 32 Hz clip clock, aligned "
                        "with --band-cache-dir, e.g. '1/1,1/1,2/1' for slow/mid at 32 Hz beside an "
                        "HGA baked at 64 Hz. --enc0-only ONLY: no stem consumes mixed rates, so a "
                        "mixed-rate encoder tap would be meaningless. Each band keeps its own "
                        "BAND_STRIDES decimation, so 2/1 on HGA yields a 64 Hz enc0 band. The "
                        "declaration is validated against every cache's real band_hop.")
    p.add_argument("--online", action="store_true",
                   help="probe the ONLINE encoder (objective.online.*) instead of the EMA teacher. "
                        "Required for the MAE arm (no teacher). Use it on a JEPA ckpt too for an "
                        "online-vs-online parity match.")
    p.add_argument("--elec-taps", default="",
                   help="comma-separated taps to ALSO write per-electrode (unpooled), e.g. "
                        "'0,12' -> feats['enc0_elec'], feats['enc12_elec']. 0 = the |STFT| "
                        "frontend (enc0); 3/6/9/12 route through the teacher forward (GPU_TAPS). "
                        "WS keeps all electrodes by default (Ben 2026-07-16); each costs "
                        "~N/|P| (~5x) the pooled tap on disk.")
    p.add_argument("--clip-dur", type=float, default=CLIP_DUR_S,
                   help=f"clip seconds (default {CLIP_DUR_S} = Neuroprobe-Lite parity). The "
                        "visualization encode uses 2.0 — the SSL training clip length, so the "
                        "longer window is in-distribution rather than an extrapolation. Any "
                        "value other than the default BREAKS board parity: the resulting cache "
                        "is for figures, never for a leaderboard number.")
    p.add_argument("--clip-offset", type=float, default=0.0,
                   help="seconds added to every clip start; negative windows BEFORE word onset "
                        "(e.g. -0.5 with --clip-dur 2.0 gives -0.5 -> +1.5 s). Clips whose "
                        "shifted window leaves the session cache are dropped, with labels and "
                        "ws/cs split indices remapped to the surviving union axis.")
    args = p.parse_args()

    from speech_decoding.experiments.pretrain_probe_suite import PROBE_COHORT_7
    from speech_decoding.experiments.dispatch_v3 import make_bt_parcel_fn
    from speech_decoding.models.v14_converged_v3.pack_r4 import (
        build_r4_grid, build_r5_grid, build_r5nf_grid,
    )
    from speech_decoding.models.v14_converged_v3.stem import nf_token_geometry
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions
    from speech_decoding.models.v14_converged_v3.parcel_perm import apply_parcel_perm

    is_r5 = args.frontend == "v3r5"
    is_r5nf = args.frontend in ("v3r5nf", "v3r5nffast")
    is_r6 = args.frontend == "v3r6"
    is_two_stream = is_r5 or is_r5nf
    # NoFusionStem net decimation: 4 for v3r5nffast (16 Hz tokens), else 2 (v3r5nf, 32 Hz).
    nf_dec = 4 if args.frontend == "v3r5nffast" else 2
    if is_two_stream:
        # r5 (EarlyFusion) / nf (NoFusion): 2 native-64Hz caches, no EMA teacher, no enc0.
        # Force the flag the frontend implies so a bare `--frontend` is sufficient.
        args.online = True
    if is_r6:
        # r6 is MAE (--objective mae): no EMA teacher, so the deployed rep is the online encoder.
        # Its DATA path is r4's verbatim — same 3 × 32 Hz caches, same UNIFORM band_rates, same
        # PerBandStem decimate — so enc0 is bit-identical to v3r4 and nothing below needs an r6
        # branch (verified by the v3r4-vs-v3r6 enc0 parity job).
        args.online = True
    n_cache = 2 if is_two_stream else 3
    if len(args.band_cache_dirs) != n_cache:
        want = "(hga, lfs)" if is_two_stream else "(slow, mid, hga)"
        raise SystemExit(
            f"need {n_cache} --band-cache-dir {want}, got {len(args.band_cache_dirs)}")
    band_rates = None
    if args.band_rates:
        if not (args.enc0_only or args.enc0_log):  # --enc0-log implies --enc0-only, set below
            raise SystemExit("--band-rates is --enc0-only: no stem consumes bands at mixed rates")
        if is_two_stream:
            raise SystemExit("--band-rates conflicts with the 2-stream frontends (R5_BAND_RATES)")
        band_rates = _parse_band_rates(args.band_rates, n_cache)
        print(f"[check] band_rates={band_rates} (declared vs the 32 Hz clip clock; each cache's "
              f"real band_hop is asserted against this in load_v3_sessions)", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_frames = round(args.clip_dur * FPS)
    if args.clip_dur != CLIP_DUR_S or args.clip_offset != 0.0:
        print(f"[window] NON-PARITY window: dur={args.clip_dur}s offset={args.clip_offset}s "
              f"-> {clip_frames} frames @ {FPS} Hz. Figures only, not a board number.", flush=True)
    parcel_fn = make_bt_parcel_fn(args.bt_root)
    cohort = BOARD_SESSIONS if args.sessions == "board" else tuple(PROBE_COHORT_7)
    if args.session_index is not None:
        if not 0 <= args.session_index < len(cohort):
            raise SystemExit(f"--session-index {args.session_index} out of range [0,{len(cohort)})")
        cohort = (cohort[args.session_index],)
    tasks = BOARD_TASKS if args.tasks == "board15" else PROBE_TASKS
    keep_labels_fn = _lite_keep_labels_fn(args.bt_root) if args.electrode_set == "lite" else None
    elec_taps = tuple(int(t) for t in args.elec_taps.split(",") if t.strip())
    bad = [t for t in elec_taps if t not in (0,) + GPU_TAPS]
    if bad:
        raise SystemExit(f"--elec-taps {bad} not in {(0,) + GPU_TAPS} (0 = enc0_elec, off the "
                          f"|STFT| frontend directly; the rest route through GPU_TAPS {GPU_TAPS})")
    # tap 0 never goes through the teacher forward (_encode_taps/GPU_TAPS) — it's the frontend
    # input, handled by _enc0_elec below instead of _encode_taps' elec_taps loop.
    gpu_elec_taps = tuple(t for t in elec_taps if t != 0)
    want_enc0_elec = 0 in elec_taps
    if args.enc0_log:
        args.enc0_only = True  # the gate never touches weights: pure function of the cached bands
    # enc0 on a 2-stream frontend is opt-in: --enc0-only, or an explicit --enc0-stride.
    enc0_two_stream = args.enc0_only or args.enc0_stride is not None
    enc0_stride = args.enc0_stride
    if is_two_stream and enc0_stride is None:
        enc0_stride = nf_dec  # stem parity: the stem's own net decimate
    if args.enc0_only:
        # enc0 is a pure function of the CACHED BANDS (no weights ever touched), so there is
        # nothing to load and nothing to run on a GPU.
        teacher = None
        tower_note = "ENC0-ONLY (no ckpt, no model, CPU)"
    else:
        if not args.ckpt:
            raise SystemExit("--ckpt is required unless --enc0-only")
        sd = _load_ckpt(args.ckpt)
        tower_pref = "objective.online." if args.online else "objective.teacher.model."
        teacher = _load_teacher(sd, device=device, pref=tower_pref, early_fusion=is_r5,
                                no_fusion=is_r5nf, nf_decimate=nf_dec,
                                space_rope=not args.no_space_rope)
        del sd
        tower_note = "online encoder" if args.online else "EMA teacher"
    print(f"[encode-r4] tag={args.tag} device={device} gpu_taps={GPU_TAPS} + enc0 "
          f"{tower_note}, sessions={args.sessions}({len(cohort)}) "
          f"tasks={args.tasks}({len(tasks)}) electrodes={args.electrode_set}", flush=True)

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
        elif band_rates is not None:
            load_kw["band_rates"] = band_rates
        spec = load_v3_sessions(
            sessions=[session], band_cache_dirs=args.band_cache_dirs, span_dir=args.span_dir,
            parcel_fn=parcel_fn, lof_report_path=None,
            winsor=(20.0, 15.0) if is_two_stream else (15.0, 15.0, 20.0),
            keep_labels_fn=keep_labels_fn,
            parcel_perm_seed=args.parcel_perm_seed, **load_kw,
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
        if args.clip_dur != CLIP_DUR_S or args.clip_offset != 0.0:
            n_before = len(targets.clip_starts)
            targets = _shift_and_trim(targets, offset_s=args.clip_offset,
                                      clip_frames=clip_frames, n_frames=spec.n_frames)
            n_after = len(targets.clip_starts)
            print(f"[window] {session}: {n_before} -> {n_after} clips "
                  f"({n_before - n_after} dropped at session edges)", flush=True)
        # r5/nf caches are native-64Hz (R5_BAND_RATES 2×); read 2·clip_frames frames per window.
        # r4 AND r6 share the uniform 32 Hz caches ⇒ rate_mult 1, no per-band rates.
        bands = _window_bands(spec, targets.clip_starts, clip_frames,
                              rate_mult=2 if is_two_stream else 1, band_rates=band_rates)

        geom = spec.setup.geom.to(device)
        parcel_id = spec.setup.parcel_id.to(device)
        if is_r5:
            grid = build_r5_grid(geom, n_time=clip_frames)
        elif is_r5nf:
            n_tok, time_stride = nf_token_geometry(clip_frames, decimate=nf_dec)
            grid = build_r5nf_grid(geom, n_time=n_tok, time_stride=time_stride)
        else:
            grid = build_r4_grid(geom, n_time=clip_frames)
        # R35 SPLIT. ``spec.setup.parcel_id`` is the MODEL-SIDE tag: under --parcel-perm-seed
        # it is this subject's permuted vocabulary, which is what the parcel embed and the
        # predictor mask-query were trained on. The parcel POOLING below, and the CS
        # anchor/test parcel intersection it feeds, match parcels ACROSS subjects by atlas
        # id, so they MUST keep the TRUE tag -- permuting them would scramble the readout
        # instead of the encoder and the arm would measure the wrong thing.
        # ``parcel_fn`` maps by label, so passing the survivor labels returns survivor-aligned
        # true tags. The relation between the two is then ASSERTED on real data: with no seed
        # the recomputation must reproduce the spec exactly (which validates the recomputation
        # itself), and with a seed the spec must be exactly the permutation of it. This is the
        # guard the ckpt cannot provide.
        parcel_true = parcel_fn(subject_id, trial_id, list(spec.setup.sidecar.labels)).long()
        expect = (parcel_true if args.parcel_perm_seed is None else
                  apply_parcel_perm(parcel_true, subject_id, seed=args.parcel_perm_seed))
        if not torch.equal(expect, spec.setup.parcel_id.cpu()):
            raise RuntimeError(
                f"parcel tag parity FAILED for s{subject_id}_t{trial_id} with "
                f"--parcel-perm-seed {args.parcel_perm_seed}: the spec's model-side tag is not "
                f"the expected relabeling of the true atlas tag ({int((expect != spec.setup.parcel_id.cpu()).sum())} "
                f"of {expect.numel()} electrodes disagree). Refusing to encode."
            )
        print(f"[parcel-perm] s{subject_id}_t{trial_id} parity OK "
              f"(seed={args.parcel_perm_seed}, {int((parcel_true != spec.setup.parcel_id.cpu()).sum())} "
              f"of {parcel_true.numel()} electrodes relabeled; pooling uses TRUE atlas ids)",
              flush=True)
        parcel_packed = parcel_id[grid.contact]                        # MODEL-side tag
        canon, parcel_canon, present = _canon_parcels(grid, parcel_true)  # TRUE atlas ids

        # enc0 = the per-band decimated input floor. r4: BAND_STRIDES (8,2,1) on the 32 Hz clock.
        # 2-stream (Ben 2026-07-23): both caches are native 64 Hz and the stem decimates by
        # nf_dec, so (nf_dec,)*n_bands is the exact parity; --enc0-stride 1 keeps the full 64 Hz.
        # r6 takes NO branch here: it reads r4's uniform 32 Hz caches, so r4's default BAND_STRIDES
        # IS its enc0 floor. (The deleted r6 branch forced (1,1,1) on the never-baked native rates —
        # part of the R6_BAND_RATES bug, 2026-07-23.)
        if args.enc0_log:
            # enc0 log-vs-abs gate (CPU-only, r6 3-band). Reload the RAW |STFT| windows
            # (normalize=False) so the ONLY difference between the two taps is the pre-log:
            #   enc0     = robustz(|STFT|)                    (ASR: magnitude)
            #   enc0_log = robustz(log(|STFT| + eps))         (ASR: log-magnitude)
            # Both REFIT sigma over these same clip windows because frozen abs-stats are invalid
            # for log (median commutes with log, MAD does not), and both share the r6 read-time
            # winsor (15,15,20 for SLOW/MID/HGA) so winsorization is not a confound. The abs cell
            # is cross-checked against the frozen-stats enc0 below to prove the refit ≈ frozen path.
            raw = _window_bands(spec, targets.clip_starts, clip_frames,
                                rate_mult=2 if is_two_stream else 1, band_rates=band_rates,
                                normalize=False)
            st = (enc0_stride,) * len(raw) if enc0_stride is not None else None
            winsor_by_band = (15.0, 15.0, 20.0)
            abs_bands = [_robustz_refit(b, winsor_by_band[i]) for i, b in enumerate(raw)]
            log_bands = [_robustz_refit(torch.log(b + args.enc0_log_eps), winsor_by_band[i])
                         for i, b in enumerate(raw)]
            feats = {
                "enc0": {"raw": _enc0_pooled(abs_bands, canon, parcel_canon, present, st)},
                "enc0_log": {"raw": _enc0_pooled(log_bands, canon, parcel_canon, present, st)},
            }
            # Cross-check: frozen-stats enc0 (from the normalized `bands`) vs the refit-abs enc0.
            frozen0 = _enc0_pooled(bands, canon, parcel_canon, present, st)
            refit0 = feats["enc0"]["raw"]
            d = float((frozen0 - refit0).abs().max())
            rel = d / (float(frozen0.abs().max()) + 1e-9)
            print(f"[enc0-log check] {session}: frozen-abs vs refit-abs max|Δ|={d:.4g} "
                  f"rel={rel:.4g} -> {'OK' if rel < 5e-2 else 'DIVERGENT'}", flush=True)
        elif is_two_stream and not enc0_two_stream:
            feats = {}  # legacy: 2-stream enc0 off unless asked for
        else:
            st = (enc0_stride,) * len(bands) if enc0_stride is not None else None
            feats = {"enc0": {"raw": _enc0_pooled(bands, canon, parcel_canon, present, st)}}
            if want_enc0_elec:
                feats["enc0_elec"] = {"raw": _enc0_elec(bands, canon, st)}
        # band_lengths must describe the PAYLOAD. It is grid-derived (clip_frames // BAND_STRIDES
        # on the 32 Hz clock) everywhere else, which is only enc0's layout when every cache is at
        # 32 Hz. Recompute from the band tensors whenever enc0 is written, and refuse to write if
        # that disagrees with the grid on the DEFAULT path — that would mean the published enc0
        # layout moved under us.
        enc0_lengths = tuple(int(x) for x in grid.band_lengths)
        if feats:
            st_now = (enc0_stride,) * len(bands) if enc0_stride is not None else None
            enc0_lengths = _enc0_band_lengths(bands, st_now)
            fd = tuple(int(b.shape[-2]) for b in bands)
            default_path = band_rates is None and enc0_stride is None and not is_two_stream
            if default_path and enc0_lengths != tuple(int(x) for x in grid.band_lengths):
                raise RuntimeError(
                    f"enc0 band_lengths {enc0_lengths} != grid {tuple(grid.band_lengths)} on the "
                    f"DEFAULT path — the published enc0 layout changed; refusing to write")
            print(f"[check] enc0 band_lengths={enc0_lengths} band_fdims={fd} "
                  f"width={sum(t * f for t, f in zip(enc0_lengths, fd))} "
                  f"grid={tuple(int(x) for x in grid.band_lengths)}", flush=True)
        if args.enc0_only:
            tap_pooled = {}
        else:
            tap_pooled = _encode_taps(teacher, bands, grid, parcel_packed, parcel_canon,
                                      present, device=device, batch_size=args.batch_size,
                                      elec_taps=gpu_elec_taps)
            for t in GPU_TAPS:
                feats[f"enc{t}"] = tap_pooled[t]
            for t in gpu_elec_taps:
                feats[f"enc{t}_elec"] = tap_pooled[f"elec{t}"]

        payload = {
            "subject_id": subject_id, "trial_id": trial_id, "ckpt_tag": args.tag,
            "present_parcels": np.asarray(present, dtype=np.int64),   # (|P|,) atlas ids, feature order
            # (n_contacts,) atlas id per CANONICAL contact — the same axis the ``enc*_elec`` taps
            # are stored on, so a readout can re-pool electrodes→parcels itself instead of being
            # stuck with the mean baked in here. present_parcels alone is not enough: it gives the
            # parcel ORDER but not the membership.
            "parcel_canon": np.asarray(parcel_canon, dtype=np.int64),
            "band_lengths": enc0_lengths,
            # Frequency bins per band. band_lengths alone does NOT let a consumer slice enc0 by
            # band: the encoder taps are (k_full, d) so band_lengths is enough there, but enc0 is
            # the raw spectrogram at Σ_b F_b·T_b, and F_b is not recoverable from that total.
            "band_fdims": tuple(int(b.shape[-2]) for b in bands),
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
