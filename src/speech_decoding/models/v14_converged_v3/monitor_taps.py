"""Compact in-objective monitor reductions (#40 per-band JEPA).

The JEPA ``pred``/``tgt`` are ``(B, total, 1024)`` — GB-scale, too big to ship to the
Lightning callback as raw taps. This pure function reduces them to per-band SCALARS
INSIDE the objective's forward, on the monitor cadence only; the callback then just logs
the scalars.

Every split uses WEIGHTED reductions over the STATIC ``(B, total)`` grid (never a boolean
index), matching how the JEPA loss itself is computed — so the compiled forward's shapes
stay static and no data-dependent branch is introduced on the monitor path.
"""

from __future__ import annotations

from torch import Tensor

_EPS: float = 1e-8
BAND_NAMES: tuple[str, ...] = ("slow", "mid", "hga")


def per_band_jepa_stats(
    pred: Tensor, tgt: Tensor, w: Tensor, band_ids: Tensor
) -> dict[str, Tensor]:
    """Per-band explained-var / pred-target var-ratio / L1 over the margin-gated scored
    tokens — the r4 (per-band) replacement for the retired whole/intra tier split.

    ``pred``/``tgt`` (B, total, F); ``w`` (B, total) the ``in_loss`` weight (float, 1 at a
    scored token else 0); ``band_ids`` (total,) in {0,1,2}. Matches ``_tier_stats``
    semantics: variance is over the SCORED-TOKEN axis, per feature, averaged over features
    (``mean_f Var_tok``). Computed with weighted reductions (static shape), using

        Var = mean_f( E[x_f²] − E[x_f]² ),  E weighted over the band's scored tokens
            = E[‖x‖²]/F  −  ‖E[x]‖²/F

    so the per-feature mean vector is one matmul (no GB-scale elementwise temporary) and the
    ``E[‖x‖²]`` term reduces the feature axis first (a (B,total) intermediate)."""
    B, total, n_feat = pred.shape
    resid = tgt - pred
    # per-token squared norms — reduce the feature axis first (one GB transient each, freed).
    sqn_t = tgt.pow(2).sum(-1)  # (B, total)
    sqn_p = pred.pow(2).sum(-1)
    sqn_r = resid.pow(2).sum(-1)
    l1_tok = resid.abs().mean(-1)  # (B, total)
    pred_f = pred.reshape(B * total, n_feat)
    tgt_f = tgt.reshape(B * total, n_feat)
    resid_f = resid.reshape(B * total, n_feat)

    out: dict[str, Tensor] = {}
    for b, name in enumerate(BAND_NAMES):
        wb = w * (band_ids == b).to(w.dtype).unsqueeze(0)  # (B, total)
        denom = wb.sum().clamp_min(1.0)
        wf = wb.reshape(1, B * total)

        def _mean_f_var(sqn_tok: Tensor, x_f: Tensor, _wb=wb, _wf=wf, _denom=denom) -> Tensor:
            e_sq = (sqn_tok * _wb).sum() / _denom  # E[‖x‖²]
            mean_vec = (_wf @ x_f).squeeze(0) / _denom  # (F,) weighted per-feature mean
            return (e_sq - mean_vec.pow(2).sum()) / n_feat

        var_t = _mean_f_var(sqn_t, tgt_f)
        var_p = _mean_f_var(sqn_p, pred_f)
        var_r = _mean_f_var(sqn_r, resid_f)
        out[f"jepa_{name}_explained_var"] = 1.0 - var_r / (var_t + _EPS)
        out[f"jepa_{name}_pred_target_var_ratio"] = var_p / (var_t + _EPS)
        out[f"jepa_{name}_l1"] = (l1_tok * wb).sum() / denom
    return out


def visible_recon_gap_stats(
    pred: Tensor, tgt: Tensor, masked_w: Tensor, masked_stats: dict[str, Tensor], band_ids: Tensor
) -> dict[str, Tensor]:
    """Per-band VISIBLE-token reconstruction EV + the masked−visible GAP (MAE difficulty).

    Companion to :func:`per_band_jepa_stats`, which gives the MASKED (scored-token) EV. The
    gap ``visible_EV − masked_EV`` is the MAE-specific difficulty signal: → 0 means a masked
    token is as reconstructable as one the encoder saw ⇒ the model is interpolating from
    neighbours (task too easy, weak representation pressure); a large gap means masking poses
    a genuine prediction problem. It is the reconstruction analogue of the r5nf block-width
    tuning that watched masked EV directly.

    ``masked_w`` (B, total) float — the scored weight (r6: ``in_loss == masked``, no margin
    gate). VISIBLE = real tokens with ``masked_w == 0``. Shaft-batch tail-slack FILLER tokens
    carry all-zero target bins (``shaft_batch.py`` ``torch.zeros(pad_n, f, t)``) and are
    ALWAYS visible, so a naive ``1 − masked_w`` would fold their zero variance into the
    visible set and deflate its EV. They are removed by an EXACT target-energy gate
    (``‖tgt‖² > 0``); filler is constructed as exact zeros, so the gate is unambiguous.

    ``masked_stats`` is the already-computed :func:`per_band_jepa_stats` dict (reused for the
    masked EV term, so this adds ONE extra weighted reduction pass, not two). Emits, per band,
    ``jepa_{band}_visible_explained_var`` and ``jepa_{band}_recon_gap``.

    NOTE (r6 assumption): VISIBLE is defined as the complement of ``masked_w`` among real
    tokens, which is exact only when ``in_loss == masked`` (r6 drops the margin gate). Under a
    margin gate the gate-excluded masked tokens are neither scored nor truly visible; this fn
    is for the gate-free MAE arm and would mislabel them otherwise."""
    energy = tgt.pow(2).sum(-1)  # (B, total) — EXACTLY 0 at all-zero filler tokens
    real = (energy > 0).to(pred.dtype)
    vis_w = real * (1.0 - (masked_w > 0).to(pred.dtype))  # visible real tokens
    vis = per_band_jepa_stats(pred, tgt, vis_w, band_ids)
    out: dict[str, Tensor] = {}
    for name in BAND_NAMES:
        vev = vis[f"jepa_{name}_explained_var"]
        out[f"jepa_{name}_visible_explained_var"] = vev
        out[f"jepa_{name}_recon_gap"] = vev - masked_stats[f"jepa_{name}_explained_var"]
    return out


def early_fusion_recon_stats(
    pred: Tensor, tgt: Tensor, w: Tensor, *, n_hga: int, decimate: int
) -> dict[str, Tensor]:
    """r5 HGA-vs-LFS recon health: explained-var + pred-target var-ratio split by the early-fused
    CHANNEL groups (not token-band — r5 is a single band, the split lives in the reconstruction
    feature dim). ``pred``/``tgt`` (B, total, DEC·C) reshape to (B, total, DEC, C); channels
    ``[:n_hga]`` are the HGA |STFT| bins (combined), ``[n_hga:]`` the LFS raw. Same weighted
    per-feature-variance definition as :func:`per_band_jepa_stats` (variance over the scored-token
    axis, per feature, mean over features) so the {hga,lfs} curves land on the SAME wandb panel as
    r4's {slow,mid,hga}. ``w`` (B, total) = ``in_loss`` (accept-the-bleed ⇒ every masked token)."""
    B, total, dc = pred.shape
    c = dc // decimate
    pr = pred.reshape(B, total, decimate, c)
    tg = tgt.reshape(B, total, decimate, c)
    wf = w.reshape(1, B * total)
    denom = w.sum().clamp_min(1.0)
    out: dict[str, Tensor] = {}
    for name, lo, hi in (("hga", 0, n_hga), ("lfs", n_hga, c)):
        x_t = tg[..., lo:hi].reshape(B, total, -1)  # (B, total, g) group features (DEC·group_ch)
        x_p = pr[..., lo:hi].reshape(B, total, -1)
        n_feat = x_t.shape[-1]

        def _mean_f_var(x: Tensor, _nf=n_feat) -> Tensor:
            e_sq = (x.pow(2).sum(-1) * w).sum() / denom  # E[‖x‖²]
            mean_vec = (wf @ x.reshape(B * total, _nf)).squeeze(0) / denom  # (F,)
            return (e_sq - mean_vec.pow(2).sum()) / _nf

        var_t = _mean_f_var(x_t)
        var_p = _mean_f_var(x_p)
        var_r = _mean_f_var(x_t - x_p)
        out[f"jepa_{name}_explained_var"] = 1.0 - var_r / (var_t + _EPS)
        out[f"jepa_{name}_pred_target_var_ratio"] = var_p / (var_t + _EPS)
    return out


def nofusion_recon_stats(
    pred: Tensor,
    tgt: Tensor,
    w_hga: Tensor,
    w_lfs: Tensor,
    *,
    n_hga: int,
    n_lfs: int,
) -> dict[str, Tensor]:
    """v3r5nf per-stream recon health: explained-var + pred-target var-ratio for the HGA and LFS
    TOKEN streams (separate tokens, unlike r5's within-token channel split).

    ``pred``/``tgt`` (B, total, F_MAX) are the padded per-token targets; HGA tokens use all
    ``n_hga`` feats, LFS tokens only the first ``n_lfs`` (the rest pad-zero in BOTH pred and tgt).
    Each stream is sliced to its OWN valid feats (``[:n_b]``) so LFS's variance isn't deflated by
    the pad zeros, and weighted by that stream's masked weight (``w_hga``/``w_lfs``, already
    stream-disjoint) so only its own tokens contribute. Same weighted per-feature-variance as
    :func:`per_band_jepa_stats`, so {hga,lfs} land on the SAME wandb panel as r5-fused's split."""
    B, total, _ = pred.shape
    out: dict[str, Tensor] = {}
    for name, wsel, nf in (("hga", w_hga, n_hga), ("lfs", w_lfs, n_lfs)):
        x_t = tgt[..., :nf]  # (B, total, nf) this stream's valid feats
        x_p = pred[..., :nf]
        wf = wsel.reshape(1, B * total)
        denom = wsel.sum().clamp_min(1.0)

        def _mean_f_var(x: Tensor, _nf=nf, _wsel=wsel, _wf=wf, _denom=denom) -> Tensor:
            e_sq = (x.pow(2).sum(-1) * _wsel).sum() / _denom  # E[‖x‖²]
            mean_vec = (_wf @ x.reshape(B * total, _nf)).squeeze(0) / _denom  # (nf,)
            return (e_sq - mean_vec.pow(2).sum()) / _nf

        var_t = _mean_f_var(x_t)
        var_p = _mean_f_var(x_p)
        var_r = _mean_f_var(x_t - x_p)
        out[f"jepa_{name}_explained_var"] = 1.0 - var_r / (var_t + _EPS)
        out[f"jepa_{name}_pred_target_var_ratio"] = var_p / (var_t + _EPS)
    return out


__all__ = [
    "per_band_jepa_stats",
    "visible_recon_gap_stats",
    "early_fusion_recon_stats",
    "nofusion_recon_stats",
    "BAND_NAMES",
]
