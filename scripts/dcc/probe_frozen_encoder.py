#!/usr/bin/env python
"""Standalone linear-probe eval of a frozen v14 P1/P2 SSL encoder.

Loads a P1/P2 SSL checkpoint, reconstructs the frozen encoder, forwards
Neuroprobe clips, mean-pools the parcel latent (over valid parcels + time,
keeping freq + feature), then fits a deterministic L2 LogisticRegression per
task and reports downstream test AUROC.

------------------------------------------------------------------------------
WHAT THIS SCRIPT DOES (the Ben-confirmed probe spec)
------------------------------------------------------------------------------
1. Load a P1/P2 SSL checkpoint (either flavor — auto-detected, see below) and
   reconstruct a frozen ``V14ParcelPerceiverModel`` with the SSL-train config.
2. Build the Neuroprobe per-task labeled datasets (within-subject split) for
   ONE BrainTreeBank subject, tasks {speech, volume, onset} (all binary).
3. Forward clips -> encoder latent. Mean-pool over parcels(K, valid only) and
   time(T); KEEP freq(F_p) and feature(d) -> flatten to (B, F_p*d).
       - pool="mean"  (B37): latent is (B, K, F_p, T_p, d)  -> F_p survives natively.
       - pool="cross_attn" (B36, the DEFAULT SSL run): latent is (B, K*M, T_p, d)
         -- freq is ALREADY collapsed into d. There is NO F_p axis on this path,
         so the pooled feature is just (B, d). See the UNCERTAINTY note at the
         bottom of this file: confirm which pool your ckpt used.
4. Per task: fit sklearn L2 LogisticRegression on TRAIN features, report TEST
   AUROC (sklearn roc_auc_score). Deterministic (fixed seed, fixed split).
   Falls back to a torch logistic-regression solver if sklearn is unavailable.

------------------------------------------------------------------------------
KEY CODE-GROUNDING (file:line in the repo this mirrors)
------------------------------------------------------------------------------
ENCODER class + forward:
  - V14ParcelPerceiverModel.__init__:
        src/speech_decoding/models/v14_encoder.py:757   (and config build args
        enumerated at v14_encoder.py:3378-3406)
  - encoder.forward(electrode_tokens, support, valid_mask, *, return_taps=...):
        src/speech_decoding/models/v14_encoder.py:1585  (cross_attn path)
        src/speech_decoding/models/v14_encoder.py:1421  (_forward_meanpool, pool="mean")
  - cross_attn return: single tensor (B, K*M, T_p, d) when return_taps=False;
    {"M2","M4"} dict when return_taps=True (M4 = (B, K*M, T_p, d)).
  - mean-pool return: single tensor (B, K, F_p, T_p, d) when return_taps=False;
    {"M2","M4","latent_valid"} dict when return_taps=True
        (return statements: v14_encoder.py:1581-1583).
  - latent_valid (which parcels are valid; mean only over these):
        _compute_latent_valid(support, valid_mask, m_sub_slots) ->
        (B, K*M) bool   src/speech_decoding/models/v14_encoder.py:691
        (cross_attn path: we recompute it ourselves with this same logic).
        On the mean-pool path the encoder RETURNS latent_valid (B, K) directly.

CHECKPOINT structure (two flavors, both handled):
  (a) "transferable snapshot" (phase_N.ckpt, written by Experiment._snapshot):
        torch.save({"encoder": encoder.state_dict(), ...})
        src/speech_decoding/experiments/experiment.py:467-477
        -> encoder weights are ckpt["encoder"], BARE keys (no prefix).
        v14_joint_module.py:424  transferable_state() returns {"encoder": ...}.
  (b) Lightning ModelCheckpoint (best.ckpt / last.ckpt under checkpoints/):
        ckpt["state_dict"] with keys prefixed "student.encoder.*"
        (the SSL module stores the encoder at self.student.encoder:
         src/speech_decoding/experiments/v14_joint_module.py:151-153, 349).
  This script auto-detects and strips the right prefix.

NEUROPROBE data path:
  - build_v14_experiment(...): src/speech_decoding/experiments/dispatch_v14.py:?? (the
    builder; we call it with phase4_frozen_probe=True for the WithinSession data).
  - segmenter extractors dict (batch keys):
        dispatch_v14.py:1209-1215  ->  electrode_tokens / support / valid_mask /
        ref_idx / subject_subtype, plus "target" (EventField) at 1244-1249.
  - model is called with x_name=("electrode_tokens","support","valid_mask"):
        dispatch_v14.py:1601-1602   (so a batch carries those three tensors
        plus "target"). The P4 BrainModule.forward does
        self.model(**{n: batch.data[n] for n in x_name}):
        src/speech_decoding/experiments/v14_phase4.py:241-244.
  - within-subject split: eval_mode="WithinSession", KFold(n_splits=2,
    shuffle=False) per task, seed 42:
        src/speech_decoding/studies/braintreebank/word_events.py (_assign_within_session_split).
  - loaders = exp.data.build(worker_seed=exp.seed) ->
        {"train","val","test"} DataLoaders   src/speech_decoding/experiments/data.py:74-135.
  - valid BT subjects: V14_TRAIN_SUBJECT_IDS = {1,2,3,4,6,7,8,9,10}
        src/speech_decoding/studies/braintreebank/manifest.py.
    Neuroprobe-Lite (subject, trial) pairs include (2,4),(1,1),(3,0),(4,0),
    (7,0),(10,0): _neuroprobe_lite_tables.py.

ENV (from scripts/dcc/nano_worker_ddp_start.sh):
    ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank
    EXCA_CACHE_FOLDER=/hpc/group/coganlab/ht203/cache_neuroai
    EXCA_EXTRACTOR_CACHE_FOLDER=/work/ht203/cache/v14_extractors
  Run under /work/ht203/repo/speech/.venv (the repo is pip-installed, so
  `import speech_decoding` works once the venv is active).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Defaults (mirror dispatch_v14.py DEFAULT_* and the DCC env)
# ---------------------------------------------------------------------------

# dispatch_v14.py:68-86 production model defaults.
DEFAULT_D_MODEL = 256
DEFAULT_DEPTH = 6
DEFAULT_N_HEADS = 8
DEFAULT_M_SUB_SLOTS = 1
DEFAULT_K_PARCELS = 80          # len(V14_DK_PARCEL_LABELS)
DEFAULT_N_FREQ_BINS = 50        # FE-RAW-1 raw |STFT|
DEFAULT_SSL_CLIP_LEN = 5.0      # P1/P2 SSL clip length -> n_time_bins=81
DEFAULT_PROBE_CLIP_LEN = 1.0    # Neuroprobe-Lite P4 clip length -> T<=81 OK
DEFAULT_LATENT_DEPTH = 2        # B37 mean-pool parcel-SA depth (inert on cross_attn)

DEFAULT_TASKS = ("speech", "volume", "onset")
DEFAULT_SUBJECT_ID = 2
DEFAULT_TRIAL_ID = 4            # (2,4) is a valid Neuroprobe-Lite pair
DEFAULT_FOLD = 0

# DCC env (scripts/dcc/nano_worker_ddp_start.sh:29-31).
DCC_ENV_DEFAULTS = {
    "ROOT_DIR_BRAINTREEBANK": "/work/ht203/data/braintreebank",
    "EXCA_CACHE_FOLDER": "/hpc/group/coganlab/ht203/cache_neuroai",
    "EXCA_EXTRACTOR_CACHE_FOLDER": "/work/ht203/cache/v14_extractors",
}


# ---------------------------------------------------------------------------
# Checkpoint -> encoder state_dict
# ---------------------------------------------------------------------------

def load_encoder_state_dict(ckpt_path: str) -> Dict[str, Tensor]:
    """Return a BARE encoder state_dict (keys like 'patch_stem.conv.weight')
    from either checkpoint flavor.

    Flavor (a): transferable snapshot -> torch.save({"encoder": sd, ...})
                (experiment.py:467-477; v14_joint_module.py:424). The encoder
                sub-dict already has bare keys.
    Flavor (b): Lightning ckpt -> ckpt["state_dict"] with "student.encoder.*"
                prefixes (SSL module stores encoder at self.student.encoder,
                v14_joint_module.py:151-153,349). We strip "student.encoder.".
    """
    # weights_only=False so a raw Lightning ckpt (which pickles non-tensor
    # callback/optimizer state) still loads; we only read the tensor dicts.
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Flavor (a): transferable snapshot dict-of-dicts.
    if isinstance(ckpt, dict) and "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
        enc = ckpt["encoder"]
        # Heuristic: a transferable snapshot's "encoder" maps str->Tensor.
        if all(isinstance(v, torch.Tensor) for v in enc.values()):
            print(f"[ckpt] transferable-snapshot flavor: {len(enc)} encoder tensors")
            return enc

    # Flavor (b): Lightning ckpt with a flat state_dict.
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        raise ValueError(f"unrecognized checkpoint structure: {type(ckpt)}")

    # Find the encoder prefix. SSL module: self.student.encoder.* ; some
    # wrappers may store self.model.encoder.* or self.encoder.* . Pick the
    # prefix that captures the most tensor keys.
    candidate_prefixes = [
        "student.encoder.",
        "model.encoder.",
        "encoder.encoder.",  # P4 wrapper (V14ParcelPerceiverWithHead).encoder
        "encoder.",
    ]
    for prefix in candidate_prefixes:
        keys = [k for k in sd if k.startswith(prefix)]
        if keys:
            stripped = {k[len(prefix):]: sd[k] for k in keys}
            # Drop any nested teacher/predictor/readout sub-keys that may have
            # slipped under a short prefix (e.g. "encoder." also catching nothing
            # extra here, but be defensive).
            print(f"[ckpt] lightning flavor: prefix={prefix!r}, {len(stripped)} encoder tensors")
            return stripped

    raise KeyError(
        "could not locate encoder weights in checkpoint. Top-level keys: "
        f"{list(sd)[:8]}..."
    )


# ---------------------------------------------------------------------------
# Build the frozen encoder
# ---------------------------------------------------------------------------

def build_frozen_encoder(
    encoder_sd: Dict[str, Tensor],
    *,
    pool: str,
    n_freq_bins: int,
    ssl_clip_len: float,
    k_parcels: int,
    d_model: int,
    depth: int,
    n_heads: int,
    m_sub_slots: int,
    latent_depth: int,
    front_end_view,  # the MultiStftView instance, to size n_time_bins
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Construct a bare V14ParcelPerceiverModel with the SSL-train config and
    load the checkpoint weights (strict). Return (encoder, info).

    The encoder is built with the SSL-train n_time_bins (e.g. 5 s -> 81) so the
    RoPE table / weights match the checkpoint; it accepts shorter probe clips
    (T<=max_seq_len just works -- v14_encoder.py:1079-1080, 1465-1468).
    """
    from speech_decoding.models.v14_encoder import V14ParcelPerceiverModel

    # n_time_bins (RoPE ceiling) = SSL clip geometry, NOT the probe clip.
    # dispatch_v14.py:1098-1110 derives it from the front-end view; we reuse the
    # view when available, else the analytic torch.stft(center=True) frame count
    # 1 + floor(L/hop) with L = ssl_clip_len * 2048 Hz, hop=128 (view.py:6-7,59).
    if front_end_view is not None and hasattr(front_end_view, "n_time_bins_for_duration"):
        n_time_bins = front_end_view.n_time_bins_for_duration(ssl_clip_len)
    else:
        n_time_bins = 1 + int(ssl_clip_len * 2048) // 128   # 5 s -> 81
        print(f"[encoder] front-end view unavailable; analytic n_time_bins={n_time_bins}")

    # FE-RAW-1: kernel_freq=5 for F=50 (F=30 fbank sister uses 3). Match the
    # config the checkpoint was trained with. v14_encoder.py:3249-3250.
    patch_kernel_freq = 5 if n_freq_bins == 50 else 3

    encoder = V14ParcelPerceiverModel(
        n_freq_bins=n_freq_bins,
        n_time_bins=n_time_bins,
        k_parcels=k_parcels,
        d_model=d_model,
        n_heads=n_heads,
        depth_self_attn=depth,
        m_sub_slots=m_sub_slots,
        patch_kernel_freq=patch_kernel_freq,
        patch_kernel_time=2,
        pool=pool,
        latent_parcel_depth=latent_depth,
        # The front-end view emits electrode_tokens as (B, C, F, T) — freq then
        # time, time LAST (view.py: "(C,T) waveform -> (C, F, T_bin)"). The
        # dispatch builds the training encoder with time_last_input=True
        # (dispatch_v14.py:1538); the model DEFAULT is False, which would
        # transpose (B,C,F=50,T) -> (B,C,T,50) and read F=T (e.g. 17 for a 1 s
        # probe clip), tripping the "expected 50 freq bins, got 17" assertion.
        # Match training so F=n_freq_bins=50.
        time_last_input=True,
        # max_seq_len defaults to n_time_bins inside __init__ (v14_encoder.py:955).
    )

    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing or unexpected:
        # Strict-load is the contract (P4 uses strict=True), but we report the
        # diff loudly rather than silently mis-loading -- a non-empty diff means
        # the config (pool / d_model / depth / freq bins) does NOT match the
        # checkpoint. This is the #1 thing to verify on first run.
        print("[encoder] WARNING: state_dict mismatch (config likely differs from ckpt):")
        if missing:
            print(f"  missing ({len(missing)}): {missing[:6]}{' ...' if len(missing) > 6 else ''}")
        if unexpected:
            print(f"  unexpected ({len(unexpected)}): {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''}")
    else:
        print(f"[encoder] strict load OK ({len(encoder_sd)} tensors)")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    info = {
        "n_time_bins": n_time_bins,
        "n_freq_patches": int(encoder.n_freq_patches),
        "pool": pool,
        "k_parcels": k_parcels,
        "d_model": d_model,
        "m_sub_slots": m_sub_slots,
    }
    return encoder, info


# ---------------------------------------------------------------------------
# Pooling: encoder latent -> (B, F_p*d) feature vector
# ---------------------------------------------------------------------------

def _compute_latent_valid_cross_attn(
    support: Tensor, valid_mask: Optional[Tensor], m_sub_slots: int
) -> Tensor:
    """Mirror v14_encoder.py:691 _compute_latent_valid for the cross_attn path
    (which does NOT return latent_valid). Returns (B, K*M) bool: a parcel is
    valid iff it has >=1 covered & valid electrode."""
    if valid_mask is not None:
        eff = support * valid_mask.unsqueeze(-1).to(support.dtype)
    else:
        eff = support
    covered = eff.sum(dim=1) > 0          # (B, K)
    B, K = covered.shape
    return covered.unsqueeze(-1).expand(B, K, m_sub_slots).reshape(B, K * m_sub_slots)


@torch.no_grad()
def encode_and_pool(
    encoder: torch.nn.Module,
    batch_data: Dict[str, Tensor],
    *,
    pool: str,
    m_sub_slots: int,
    device: torch.device,
) -> Tensor:
    """Forward one batch through the frozen encoder, mean-pool over valid
    parcels(K) and time(T), keep freq(F_p) + feature(d), flatten -> (B, F_p*d).

    cross_attn path: latent (B, K*M, T_p, d), freq ALREADY collapsed -> pooled
        feature is (B, d) (no F_p axis exists on this path).
    mean path: latent (B, K, F_p, T_p, d), F_p preserved -> pooled (B, F_p*d).
    """
    electrode_tokens = batch_data["electrode_tokens"].to(device)
    support = batch_data["support"].to(device)
    valid_mask = batch_data.get("valid_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.to(device)

    if pool == "mean":
        # Encoder RETURNS latent_valid (B, K) directly under return_taps.
        out = encoder(
            electrode_tokens=electrode_tokens,
            support=support,
            valid_mask=valid_mask,
            return_taps=True,
        )
        m4 = out["M4"]                       # (B, K, F_p, T_p, d)
        latent_valid = out["latent_valid"]   # (B, K) bool
        B, K, F_p, T_p, d = m4.shape
        # Masked mean over valid parcels (K) and time (T_p); keep F_p + d.
        mask = latent_valid.to(m4.dtype).view(B, K, 1, 1, 1)   # (B,K,1,1,1)
        num = (m4 * mask).sum(dim=(1, 3))                       # sum over K,T_p -> (B,F_p,d)
        den = (latent_valid.to(m4.dtype).sum(dim=1) * T_p).clamp(min=1.0).view(B, 1, 1)
        pooled = num / den                                     # (B, F_p, d)
        feat = pooled.reshape(B, F_p * d)                      # (B, F_p*d)
        return feat

    # cross_attn (default): single tensor (B, K*M, T_p, d). Freq is collapsed.
    latent = encoder(
        electrode_tokens=electrode_tokens,
        support=support,
        valid_mask=valid_mask,
        return_taps=False,
    )
    if isinstance(latent, dict):           # defensive (shouldn't happen w/ return_taps=False)
        latent = latent["M4"]
    B, L, T_p, d = latent.shape
    latent_valid = _compute_latent_valid_cross_attn(support, valid_mask, m_sub_slots)  # (B, L)
    mask = latent_valid.to(latent.dtype).view(B, L, 1, 1)
    num = (latent * mask).sum(dim=(1, 2))                      # sum over L,T_p -> (B,d)
    den = (latent_valid.to(latent.dtype).sum(dim=1) * T_p).clamp(min=1.0).view(B, 1)
    feat = num / den                                           # (B, d)  -- NO F_p axis here
    return feat


@torch.no_grad()
def extract_features(
    encoder: torch.nn.Module,
    loader,
    *,
    pool: str,
    m_sub_slots: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run all batches in a loader -> (X (N, D), y (N,)) numpy arrays."""
    feats: List[Tensor] = []
    labels: List[Tensor] = []
    for batch in loader:
        data = batch.data
        x = encode_and_pool(
            encoder, data, pool=pool, m_sub_slots=m_sub_slots, device=device
        )
        y = data["target"]
        while y.ndim > 1 and y.shape[-1] == 1:   # collate may leave (B,1)
            y = y.squeeze(-1)
        feats.append(x.float().cpu())
        labels.append(y.long().cpu())
    X = torch.cat(feats).numpy()
    Y = torch.cat(labels).numpy()
    return X, Y


@torch.no_grad()
def encode_tokens(
    encoder: torch.nn.Module,
    batch_data: Dict[str, Tensor],
    *,
    pool: str,
    m_sub_slots: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """Forward one batch, return EVERY parcel x freq x time token (NO pooling)
    plus a per-token valid mask. Decouples feature construction from pooling so
    the flatten / attentive readouts can keep parcels + time (+ freq).

    mean path:  tokens (B, K*F_p*T_p, d), valid (B, K*F_p*T_p)
    cross_attn: tokens (B, L*T_p, d),     valid (B, L*T_p)   (freq pre-collapsed)
    """
    electrode_tokens = batch_data["electrode_tokens"].to(device)
    support = batch_data["support"].to(device)
    valid_mask = batch_data.get("valid_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.to(device)

    if pool == "mean":
        out = encoder(
            electrode_tokens=electrode_tokens,
            support=support,
            valid_mask=valid_mask,
            return_taps=True,
        )
        m4 = out["M4"]                       # (B, K, F_p, T_p, d)
        latent_valid = out["latent_valid"]   # (B, K) bool
        B, K, F_p, T_p, d = m4.shape
        tokens = m4.reshape(B, K * F_p * T_p, d)
        valid = latent_valid.view(B, K, 1, 1).expand(B, K, F_p, T_p).reshape(B, K * F_p * T_p)
        return tokens, valid.bool()

    latent = encoder(
        electrode_tokens=electrode_tokens,
        support=support,
        valid_mask=valid_mask,
        return_taps=False,
    )
    if isinstance(latent, dict):
        latent = latent["M4"]
    B, L, T_p, d = latent.shape
    latent_valid = _compute_latent_valid_cross_attn(support, valid_mask, m_sub_slots)  # (B, L)
    tokens = latent.reshape(B, L * T_p, d)
    valid = latent_valid.view(B, L, 1).expand(B, L, T_p).reshape(B, L * T_p)
    return tokens, valid.bool()


@torch.no_grad()
def extract_tokens(
    encoder: torch.nn.Module,
    loader,
    *,
    pool: str,
    m_sub_slots: int,
    device: torch.device,
    keep: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run all batches -> (T (N, N_keep, d) fp16 CPU, Y (N,) long, keep (N_tok,) bool).

    Drops tokens invalid for EVERY sample -- the padding parcels under the fixed
    within-(subject,trial) montage. Because the montage is constant across clips,
    a parcel is valid for all clips or none, so the surviving tokens need no
    per-sample mask downstream. `keep` (from the train split) is reused on test
    so the columns line up. fp16 keeps ~20x10x8x384 tokens x 3500 clips ~ 4 GB CPU.
    """
    toks: List[Tensor] = []
    vals: List[Tensor] = []
    labels: List[Tensor] = []
    for batch in loader:
        data = batch.data
        t, v = encode_tokens(
            encoder, data, pool=pool, m_sub_slots=m_sub_slots, device=device
        )
        y = data["target"]
        while y.ndim > 1 and y.shape[-1] == 1:
            y = y.squeeze(-1)
        toks.append(t.half().cpu())
        vals.append(v.cpu())
        labels.append(y.long().cpu())
    T = torch.cat(toks)        # (N, N_tok, d) fp16
    V = torch.cat(vals)        # (N, N_tok) bool
    Y = torch.cat(labels)
    if keep is None:
        keep = V.any(dim=0)    # valid for >=1 clip => valid for all (fixed montage)
    T = T[:, keep, :].contiguous()
    return T, Y, keep


# ---------------------------------------------------------------------------
# Linear probe (sklearn L2 LogReg; torch fallback)
# ---------------------------------------------------------------------------

def fit_probe_auroc(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, *, seed: int
) -> float:
    """Fit an L2 LogisticRegression on train, return test AUROC. Standardizes
    features first (the encoder latent is unscaled). Deterministic."""
    # Standardize using TRAIN statistics only.
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=seed
        )
        clf.fit(X_tr, y_tr)
        scores = clf.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, scores))
    except ImportError:
        print("[probe] sklearn unavailable -> torch L2 logistic-regression fallback")
        return _torch_logreg_auroc(X_tr, y_tr, X_te, y_te, seed=seed)


def _torch_logreg_auroc(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, *, seed: int
) -> float:
    """Deterministic full-batch L2 logistic regression in torch + rank-based AUROC."""
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr).float()
    Xe = torch.from_numpy(X_te).float()
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
    l2 = 1.0 / Xt.shape[0]  # ~ inverse-C weight decay, matched loosely to C=1

    def closure():
        opt.zero_grad()
        logits = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        scores = (Xe @ w + b).numpy()
    return _rank_auroc(y_te, scores)


def _rank_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney-U / rank-based AUROC (no sklearn)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# Token readouts: flatten-linear + lightweight->full attentive pooler
#
# The speech signal lives in a FEW parcels (selection beats averaging) and is
# structured in time (onset). Mean-over-parcels+time dilutes both. These heads
# keep that structure:
#   flatten : keep every valid parcel x freq x time token, one strong-L2 linear
#             (~N_keep*d weights -- ridge-like at p>>n, val early-stop regularizes).
#   attn    : a learned-query attentive pool. ONE module spanning a capacity axis:
#             - minimal "attention pool"  (queries=1, no qkv proj, no MLP,
#               pre_blocks=0): score_i = q.x_i/sqrt(d), softmax over tokens,
#               weighted sum -> LN -> linear.  ~ d + d*C params  (~1.2k).
#             - V-JEPA-2 attentive pooler (pre_blocks>0 self-attn refine tokens,
#               qkv_proj multi-head cross-attn, mlp_ratio>0 MLP).  ~1.5-2M params.
#             3500 clips tells us where on this axis the signal saturates.
# Param count is independent of N tokens for `attn` -- what overfits 3500 clips
# is PROBE PARAMS, not token count. That's why the attentive pool is the
# tractable way to "keep parcels and time" (flatten is ~N_keep*d-dim).
# ---------------------------------------------------------------------------

class FlattenLinear(torch.nn.Module):
    """Keep parcels+freq+time: flatten valid tokens -> a single strong-L2 linear.
    ~N_keep*d weights, trained with weight decay + val early-stop (ridge at p>>n)."""

    def __init__(self, n_tok: int, d: int, n_classes: int = 1):
        super().__init__()
        self.head = torch.nn.Linear(n_tok * d, n_classes)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:  # x (B,N,d)
        B = x.shape[0]
        return self.head(x.reshape(B, -1)).squeeze(-1)


class AttnProbe(torch.nn.Module):
    """Lightweight-to-full attentive pooler over (parcel x freq x time) tokens.

    Capacity knobs (minimal -> V-JEPA-2):
      n_queries  : learned query tokens that each pool the sequence (>1 => richer
                   readout, head sees n_queries*d).
      n_heads    : attention heads (qkv_proj path; also the pre-block heads).
      qkv_proj   : False => raw learned-query scoring (no k/v projections, ~d
                   params); True => projected multi-head cross-attention (~4d^2).
      mlp_ratio  : >0 adds a residual MLP (hidden = ratio*d) after pooling.
      pre_blocks : self-attention transformer blocks that refine the tokens
                   BEFORE pooling (the "two layers before cross-attn" in V-JEPA-2;
                   ~12*d^2 params each -- the heavy end, watch overfit on 3500).
    Tokens here are pre-filtered to valid parcels, so no padding mask is needed.
    """

    def __init__(
        self,
        d: int,
        n_classes: int = 1,
        *,
        n_queries: int = 1,
        n_heads: int = 1,
        qkv_proj: bool = False,
        mlp_ratio: float = 0.0,
        pre_blocks: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.n_queries = n_queries
        self.qkv_proj = qkv_proj
        self.drop = torch.nn.Dropout(dropout)
        self.pre = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=int((mlp_ratio or 2.0) * d),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(pre_blocks)
        ])
        self.query = torch.nn.Parameter(torch.randn(n_queries, d) * (d ** -0.5))
        if qkv_proj:
            self.attn = torch.nn.MultiheadAttention(
                embed_dim=d, num_heads=n_heads, dropout=dropout, batch_first=True
            )
        self.ln = torch.nn.LayerNorm(d)
        if mlp_ratio and mlp_ratio > 0:
            h = int(mlp_ratio * d)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(d, h), torch.nn.GELU(),
                torch.nn.Dropout(dropout), torch.nn.Linear(h, d),
            )
        else:
            self.mlp = None
        self.head = torch.nn.Linear(d * n_queries, n_classes)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x (B, N, d); mask (B, N) True=valid (None => all valid)
        pad = (~mask) if mask is not None else None
        for blk in self.pre:
            x = blk(x, src_key_padding_mask=pad)
        B = x.shape[0]
        q = self.query.unsqueeze(0).expand(B, -1, -1)            # (B, Q, d)
        if self.qkv_proj:
            pooled, _ = self.attn(q, x, x, key_padding_mask=pad, need_weights=False)
        else:
            scores = torch.einsum("bqd,bnd->bqn", q, x) * (self.d ** -0.5)  # (B,Q,N)
            if pad is not None:
                scores = scores.masked_fill(pad.unsqueeze(1), float("-inf"))
            attn = self.drop(torch.softmax(scores, dim=-1))
            pooled = torch.einsum("bqn,bnd->bqd", attn, x)        # (B, Q, d)
        pooled = self.ln(pooled)
        if self.mlp is not None:
            pooled = pooled + self.mlp(pooled)
        pooled = self.drop(pooled)
        return self.head(pooled.reshape(B, -1)).squeeze(-1)       # (B,) binary


def fit_torch_probe_auroc(
    model: torch.nn.Module,
    T_tr: Tensor, y_tr: Tensor,
    T_te: Tensor, y_te: Tensor,
    *,
    epochs: int, lr: float, wd: float, patience: int,
    val_frac: float, batch: int, seed: int, device: torch.device,
) -> Tuple[float, float]:
    """Train a token-readout head: per-d-channel standardize (TRAIN stats),
    AdamW + BCE, minibatch, early-stop on a held-out val AUROC, restore the
    best-val weights, return (test AUROC, best val AUROC). Deterministic."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    N, _, d = T_tr.shape
    flat = T_tr.reshape(-1, d).float()
    mu = flat.mean(0); sd = flat.std(0) + 1e-6

    def norm(T: Tensor) -> Tensor:
        return (T.float() - mu) / sd

    perm = torch.randperm(N, generator=g)
    n_val = max(1, int(round(val_frac * N)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss()
    yt = y_tr.float()

    best_auc, best_state, bad = -1.0, None, 0
    for _ep in range(epochs):
        model.train()
        order = tr_idx[torch.randperm(tr_idx.numel(), generator=g)]
        for i in range(0, order.numel(), batch):
            bi = order[i:i + batch]
            xb = norm(T_tr[bi]).to(device)
            yb = yt[bi].to(device)
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vs = [model(norm(T_tr[val_idx[i:i + batch]]).to(device)).float().cpu()
                  for i in range(0, val_idx.numel(), batch)]
        vauc = _rank_auroc(y_tr[val_idx].numpy(), torch.cat(vs).numpy())
        if vauc > best_auc + 1e-4:
            best_auc, bad = vauc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ts = [model(norm(T_te[i:i + batch]).to(device)).float().cpu()
              for i in range(0, T_te.shape[0], batch)]
    test_auc = _rank_auroc(y_te.numpy(), torch.cat(ts).numpy())
    return test_auc, best_auc


# ---------------------------------------------------------------------------
# Data loaders via build_v14_experiment
# ---------------------------------------------------------------------------

def build_task_loaders(
    *,
    task: str,
    subject_id: int,
    trial_id: int,
    fold: int,
    bt_root: str,
    clip_len: float,
    seed: int,
    batch_size: int,
    num_workers: int,
):
    """Return (loaders dict, front_end_view) for one Neuroprobe task, one
    subject, within-subject split. Reuses dispatch_v14.build_v14_experiment.

    NOTE: build_v14_experiment kwargs vary across repo revisions. We pass the
    minimal within-subject-probe set and rely on defaults for the rest. If a
    kwarg name has drifted, this is the FIRST place to fix on DCC (see the
    UNCERTAINTY notes at the bottom).
    """
    from speech_decoding.experiments.dispatch_v14 import build_v14_experiment

    # Default flags (joint_phase=False, p3_distill=False, cluster=None) build a
    # plain supervised Experiment with mode="lite" => the WithinSession eval
    # split on the 12 Neuroprobe-Lite sessions, built LOCALLY (no SLURM submit).
    # We only consume exp.data; the supervised head it builds is never used.
    exp = build_v14_experiment(
        bt_root=bt_root,
        mode="lite",                # Neuroprobe-Lite study universe
        task=task,
        eval_mode="WithinSession",
        test_subject_id=subject_id,
        test_trial_id=trial_id,
        fold_index=fold,
        clip_len=clip_len,
        neural_lag_s=0.0,
        electrode_set="lite",       # Neuroprobe-Lite electrode subset
        c_max=256,                  # BT-Lite max electrodes (dispatch default)
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    loaders = exp.data.build(worker_seed=exp.seed)
    # The front-end view (MultiStftView) sizes n_time_bins for the encoder.
    front_end_view = _extract_front_end_view(exp)
    return loaders, front_end_view


def _extract_front_end_view(exp) -> Any:
    """Pull the electrode-tokens extractor (MultiStftView) off the built Data
    object so we can size the encoder's n_time_bins via
    view.n_time_bins_for_duration(ssl_clip_len) (dispatch_v14.py:1098-1110).

    exp.data.segmenter is an ns.dataloader.Segmenter; its per-key extractors
    live on a .extractors mapping. We fall back to None (caller then uses the
    analytic n_time_bins formula) if the structure has drifted."""
    seg = getattr(exp.data, "segmenter", None)
    if seg is None:
        return None
    extractors = getattr(seg, "extractors", None)
    if extractors is None and isinstance(seg, dict):
        extractors = seg.get("extractors")
    if extractors is not None:
        try:
            return extractors["electrode_tokens"]
        except (KeyError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="P1/P2 SSL checkpoint (.ckpt)")
    ap.add_argument("--pool", choices=("cross_attn", "mean"), default="cross_attn",
                    help="Encoder pool the checkpoint was trained with. DEFAULT "
                         "SSL runs use cross_attn (freq collapsed); B37 joint-SSL "
                         "runs use mean (freq preserved -> real F_p*d feature).")
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    ap.add_argument("--subject-id", type=int, default=DEFAULT_SUBJECT_ID)
    ap.add_argument("--trial-id", type=int, default=DEFAULT_TRIAL_ID)
    ap.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    ap.add_argument("--bt-root", default=None,
                    help="BrainTreeBank root; default = $ROOT_DIR_BRAINTREEBANK")
    ap.add_argument("--ssl-clip-len", type=float, default=DEFAULT_SSL_CLIP_LEN,
                    help="Clip length the encoder was TRAINED at (sizes the RoPE "
                         "ceiling / n_time_bins so weights match). Default 5.0.")
    ap.add_argument("--probe-clip-len", type=float, default=DEFAULT_PROBE_CLIP_LEN,
                    help="Clip length for the Neuroprobe probe data. Default 1.0.")
    ap.add_argument("--n-freq-bins", type=int, default=DEFAULT_N_FREQ_BINS)
    ap.add_argument("--k-parcels", type=int, default=DEFAULT_K_PARCELS)
    ap.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    ap.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    ap.add_argument("--m-sub-slots", type=int, default=DEFAULT_M_SUB_SLOTS)
    ap.add_argument("--latent-depth", type=int, default=DEFAULT_LATENT_DEPTH)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ---- readout head ----
    ap.add_argument("--readout", choices=("mean", "flatten", "attn"), default="mean",
                    help="mean: pool parcels+time, keep freq -> sklearn L2 logistic "
                         "(baseline). flatten: keep every valid parcel x freq x time "
                         "token -> strong-L2 torch linear. attn: learned-query "
                         "attentive pool (capacity via --attn-* knobs).")
    # attentive-probe capacity knobs (minimal attention-pool -> V-JEPA-2 pooler)
    ap.add_argument("--attn-queries", type=int, default=1)
    ap.add_argument("--attn-heads", type=int, default=1)
    ap.add_argument("--attn-qkv-proj", action="store_true",
                    help="projected multi-head cross-attn (else raw learned-query scoring)")
    ap.add_argument("--attn-mlp-ratio", type=float, default=0.0)
    ap.add_argument("--attn-pre-blocks", type=int, default=0,
                    help="self-attn blocks refining tokens before pooling (V-JEPA-2 style)")
    ap.add_argument("--attn-dropout", type=float, default=0.1)
    # torch-probe training (shared by flatten + attn)
    ap.add_argument("--probe-epochs", type=int, default=200)
    ap.add_argument("--probe-lr", type=float, default=3e-3)
    ap.add_argument("--probe-wd", type=float, default=1e-2,
                    help="weight decay -- the main regularizer at p>>n on 3500 clips")
    ap.add_argument("--probe-patience", type=int, default=25)
    ap.add_argument("--probe-val-frac", type=float, default=0.2)
    ap.add_argument("--probe-batch", type=int, default=256)
    args = ap.parse_args()

    # ---- env (mirror nano_worker_ddp_start.sh) ----
    for k, v in DCC_ENV_DEFAULTS.items():
        os.environ.setdefault(k, v)
    bt_root = args.bt_root or os.environ.get("ROOT_DIR_BRAINTREEBANK", DCC_ENV_DEFAULTS["ROOT_DIR_BRAINTREEBANK"])

    # determinism
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)  # encoder has no nondeterministic ops we rely on; AUROC is rank-based

    device = torch.device(args.device)
    print(f"[env] bt_root={bt_root}  device={device}  pool={args.pool}")

    # ---- checkpoint -> encoder state_dict ----
    encoder_sd = load_encoder_state_dict(args.ckpt)

    results: Dict[str, float] = {}
    encoder = None  # built once on the first task's front-end view
    info = None

    for task in args.tasks:
        print(f"\n========== TASK: {task} ==========")
        loaders, front_end_view = build_task_loaders(
            task=task,
            subject_id=args.subject_id,
            trial_id=args.trial_id,
            fold=args.fold,
            bt_root=bt_root,
            clip_len=args.probe_clip_len,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        if encoder is None:
            encoder, info = build_frozen_encoder(
                encoder_sd,
                pool=args.pool,
                n_freq_bins=args.n_freq_bins,
                ssl_clip_len=args.ssl_clip_len,
                k_parcels=args.k_parcels,
                d_model=args.d_model,
                depth=args.depth,
                n_heads=args.n_heads,
                m_sub_slots=args.m_sub_slots,
                latent_depth=args.latent_depth,
                front_end_view=front_end_view,
            )
            encoder = encoder.to(device)
            print(f"[encoder] info={info}")

        if args.readout == "mean":
            X_tr, y_tr = extract_features(
                encoder, loaders["train"], pool=args.pool,
                m_sub_slots=args.m_sub_slots, device=device,
            )
            X_te, y_te = extract_features(
                encoder, loaders["test"], pool=args.pool,
                m_sub_slots=args.m_sub_slots, device=device,
            )
            print(f"[features] train X={X_tr.shape} y(+={int(y_tr.sum())}/{len(y_tr)}) "
                  f"| test X={X_te.shape} y(+={int(y_te.sum())}/{len(y_te)})")
            auroc = fit_probe_auroc(X_tr, y_tr, X_te, y_te, seed=args.seed)
            print(f"[result] {task}: AUROC = {auroc:.4f}")
        else:
            T_tr, yt_tr, keep = extract_tokens(
                encoder, loaders["train"], pool=args.pool,
                m_sub_slots=args.m_sub_slots, device=device,
            )
            T_te, yt_te, _ = extract_tokens(
                encoder, loaders["test"], pool=args.pool,
                m_sub_slots=args.m_sub_slots, device=device, keep=keep,
            )
            n_tok, d = T_tr.shape[1], T_tr.shape[2]
            print(f"[features] {args.readout} tokens kept={n_tok}/{keep.numel()} d={d} "
                  f"| train N={T_tr.shape[0]} (+={int(yt_tr.sum())}) "
                  f"test N={T_te.shape[0]} (+={int(yt_te.sum())})")
            torch.manual_seed(args.seed)  # reproducible head init, independent of task order
            if args.readout == "flatten":
                model = FlattenLinear(n_tok, d)
            else:
                model = AttnProbe(
                    d, n_queries=args.attn_queries, n_heads=args.attn_heads,
                    qkv_proj=args.attn_qkv_proj, mlp_ratio=args.attn_mlp_ratio,
                    pre_blocks=args.attn_pre_blocks, dropout=args.attn_dropout,
                )
            n_params = sum(p.numel() for p in model.parameters())
            auroc, vauc = fit_torch_probe_auroc(
                model, T_tr, yt_tr, T_te, yt_te,
                epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd,
                patience=args.probe_patience, val_frac=args.probe_val_frac,
                batch=args.probe_batch, seed=args.seed, device=device,
            )
            print(f"[result] {task}: AUROC = {auroc:.4f}  "
                  f"(val={vauc:.4f}, probe_params={n_params:,})")
        results[task] = auroc

    print("\n========== SUMMARY ==========")
    for task, auroc in results.items():
        print(f"  {task:>8}: {auroc:.4f}")
    valid = [v for v in results.values() if not np.isnan(v)]
    mean = float(np.mean(valid)) if valid else float("nan")
    print(f"  {'mean':>8}: {mean:.4f}")
    print({**{k: round(v, 4) for k, v in results.items()}, "mean": round(mean, 4)})


if __name__ == "__main__":
    sys.exit(main())
