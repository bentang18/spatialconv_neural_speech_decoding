"""Training + weight-averaging core for the attentive probe head.

Two pieces, both laptop-testable on synthetic token sets (the LOSO orchestration and
the WD/LS/dropout sweep are mechanical glue layered on top in the bench driver):

  - :func:`train_head` — train one :class:`AttentiveProbeHead` with BCE (+ optional
    binary label smoothing), nested-val EARLY STOP on val AUROC, and SWAD-style dense
    weight averaging (a running mean of post-warmup snapshots, the window bounded below
    by ``swad_warmup`` and above by early-stop — the dominant SWAD effect of averaging
    the dense tail BEFORE overfit, without the paper's finicky val-loss ts/te search).
    Returns both the best-val snapshot and the SWAD mean so the caller picks per val.
  - :func:`diwa_average` — DiWA: parameter-wise mean of several independently-trained
    head state-dicts (different seeds/HPs). Zero inference overhead, the top
    cross-subject extension (arXiv:2205.09739). Compose with per-run SWAD.

The val metric is AUROC (the eval target), NOT loss — early-stop on a held-out TRAIN
subject's AUROC is what selects for cross-subject transfer (the load-bearing reg)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from speech_decoding.experiments.v2_attentive_probe import AttentiveProbeHead

__all__ = [
    "HeadTrainConfig", "train_head", "diwa_average",
    "pad_concat", "loso_pooled_cs",
]


@dataclass
class HeadTrainConfig:
    d_model: int
    n_heads: int = 6
    n_queries: int = 1
    mlp_ratio: float = 4.0             # V-JEPA CrossAttentionBlock FFN ratio
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    residual_dropout: float = 0.1
    parcel_dropout: float = 0.0       # PRIMARY structured knob (drop whole parcels)
    token_dropout: float = 0.0        # optional i.i.d.-token ablation (off by default)
    tokens_per_parcel: int = 0        # k·S; set by the bench so parcel dropout knows blocks
    n_time_frames: int = 0            # S; >0 turns on the learnable time positional tag
    time_tag: bool = True             # readout ablation: False ⇒ _cfg_for leaves the tag off
    use_mlp: bool = True              # False ⇒ attn-pool→linear (matched linear baseline)
    lr: float = 1e-3
    weight_decay: float = 0.1
    label_smoothing: float = 0.0
    batch_size: int = 256
    max_steps: int = 2000
    eval_every: int = 50
    patience: int = 8           # eval windows w/o val-AUROC improvement → stop
    swad_warmup: int = 200      # no SWAD snapshot before this step
    seed: int = 0


def _build_head(cfg: HeadTrainConfig) -> AttentiveProbeHead:
    return AttentiveProbeHead(
        cfg.d_model, n_heads=cfg.n_heads, n_queries=cfg.n_queries,
        mlp_ratio=cfg.mlp_ratio, attn_dropout=cfg.attn_dropout,
        mlp_dropout=cfg.mlp_dropout, residual_dropout=cfg.residual_dropout,
        parcel_dropout=cfg.parcel_dropout, token_dropout=cfg.token_dropout,
        tokens_per_parcel=cfg.tokens_per_parcel, n_time_frames=cfg.n_time_frames,
        use_mlp=cfg.use_mlp, n_out=1,
    )


@torch.no_grad()
def _auroc(head: AttentiveProbeHead, x: Tensor, y: np.ndarray,
           mask: Tensor | None, device: torch.device) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    head.eval()
    s = head(x.to(device), None if mask is None else mask.to(device))
    return float(roc_auc_score(y, s.squeeze(-1).cpu().numpy()))


def _clone_state(head: AttentiveProbeHead) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in head.state_dict().items()}


def train_head(
    x_tr: Tensor, y_tr: Tensor,
    x_val: Tensor, y_val: Tensor,
    cfg: HeadTrainConfig,
    *,
    mask_tr: Tensor | None = None,
    mask_val: Tensor | None = None,
    token_parcel_ids_tr: Tensor | None = None,
    device: torch.device | None = None,
) -> dict:
    """Train one head; return ``{best_state, swad_state, best_val, swad_val, steps}``.

    ``x_*`` are ``(N, T, d)`` token sets, ``y_*`` ``(N,)`` in {0,1}, masks ``(N,T)``
    bool (True=valid) or None (constant T). ``token_parcel_ids_tr`` ``(T,)`` long drives
    irregular-group parcel dropout on the TRAIN tokens (M2 electrode tap); eval is
    dropout-free so val/test need none. Label smoothing is applied to the binary targets
    (``y(1-α)+α/2``). Early-stops on ``x_val`` AUROC; SWAD-averages post-warmup snapshots
    up to the stop."""
    device = device or torch.device("cpu")
    y_val_np = y_val.cpu().numpy().astype(float)
    head = _build_head(cfg).to(device)
    # torch.compile the TRAIN forward (GPU only). The single-query head is tiny, so the hot
    # loop is launch-bound — profiling one M2 cell showed ~143 kernel launches/step and 57%
    # of wall in cudaStreamSynchronize, with the GEMMs a small minority. Fusing the ~143
    # kernels into a handful ~halves the per-cell wall (31s→16s pdrop-off, 20s with the
    # parcel-dropout torch.unique graph break). Compiled artifacts cache by input shape, so
    # each distinct T recompiles once and is reused across that subject's 15 tasks. Eval,
    # SWAD state cloning, and the optimizer all keep the eager ``head`` (same params — the
    # compiled wrapper shares them), so state_dict keys stay clean and the numbers match.
    train_fwd = torch.compile(head) if device.type == "cuda" else head
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    n = x_tr.shape[0]
    # Move the whole cell's tokens to the GPU ONCE. The old per-step ``x_tr[idx].to(device)``
    # gathered + copied a minibatch host→device every step; for the M2 electrode tap that's
    # ~0.5 GB/step × up to 2000 steps × 105 cells, and it dominated wall (transfer-bound, not
    # compute-bound — the single-query head's GEMMs are ~1-2 min of real work). Resident on
    # device, minibatch indexing is a free on-GPU gather. The CPU-seeded generator still
    # draws ``idx`` (moved to device is a 2 KB copy), so the sampled rows are bit-identical
    # to before — same numbers, ~10-25× faster. One cell fits easily (≤~40 GB vs 97 GB).
    x_tr = x_tr.to(device)
    yb_tr = y_tr.float().view(n, 1).to(device)
    if mask_tr is not None:
        mask_tr = mask_tr.to(device)
    tpi = None if token_parcel_ids_tr is None else token_parcel_ids_tr.to(device)
    x_val = x_val.to(device)
    if mask_val is not None:
        mask_val = mask_val.to(device)
    swad_sum: dict[str, Tensor] | None = None
    swad_count = 0
    best_val = -float("inf")
    best_state = _clone_state(head)
    since_improve = 0
    step = 0
    while step < cfg.max_steps:
        head.train()
        idx = torch.randint(0, n, (min(cfg.batch_size, n),), generator=g).to(device)
        xb = x_tr[idx]
        ys = yb_tr[idx]
        ys = ys * (1.0 - cfg.label_smoothing) + 0.5 * cfg.label_smoothing
        mb = None if mask_tr is None else mask_tr[idx]
        loss = lossf(train_fwd(xb, mb, token_parcel_ids=tpi), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1

        if step % cfg.eval_every == 0:
            va = _auroc(head, x_val, y_val_np, mask_val, device)
            if step >= cfg.swad_warmup:
                state = _clone_state(head)
                if swad_sum is None:
                    swad_sum = {k: v.clone() for k, v in state.items()}
                else:
                    for k, v in state.items():
                        swad_sum[k] += v
                swad_count += 1
            if np.isnan(va):
                continue
            if va > best_val + 1e-4:
                best_val = va
                best_state = _clone_state(head)
                since_improve = 0
            else:
                since_improve += 1
                if since_improve >= cfg.patience:
                    break

    if swad_sum is None:                       # never reached warmup → fall back to best
        swad_state = best_state
    else:
        swad_state = {k: v / swad_count for k, v in swad_sum.items()}
    swad_head = _build_head(cfg).to(device)
    swad_head.load_state_dict(swad_state)
    swad_val = _auroc(swad_head, x_val, y_val_np, mask_val, device)
    return {
        "best_state": best_state, "swad_state": swad_state,
        "best_val": float(best_val), "swad_val": swad_val, "steps": step,
    }


def pad_concat(tokens_list: list[Tensor]) -> tuple[Tensor, Tensor]:
    """List of ``(N_s, T_s, d)`` → ``(ΣN, T_max, d)`` zero-padded + ``(ΣN, T_max)`` mask.

    Subjects have different token counts ``T_s = P_s·k·S`` (different parcel counts);
    the attentive head is set-valued so we pad to the batch max and mask the pad."""
    d = tokens_list[0].shape[-1]
    t_max = max(t.shape[1] for t in tokens_list)
    xs: list[Tensor] = []
    ms: list[Tensor] = []
    for t in tokens_list:
        n, ts, _ = t.shape
        x = t if ts == t_max else torch.cat([t, t.new_zeros(n, t_max - ts, d)], dim=1)
        m = torch.zeros(n, t_max, dtype=torch.bool)
        m[:, :ts] = True
        xs.append(x)
        ms.append(m)
    return torch.cat(xs, 0), torch.cat(ms, 0)


def _stack(
    tokens: dict[int, Tensor], labels: dict[int, np.ndarray], subs: list[int],
) -> tuple[Tensor, Tensor, Tensor]:
    """Concat finite-label clips across ``subs`` → padded ``(x, mask, y)``."""
    xs: list[Tensor] = []
    ys: list[np.ndarray] = []
    for s in subs:
        y = np.asarray(labels[s], dtype=float)
        fin = np.isfinite(y)
        xs.append(tokens[s][torch.from_numpy(fin)])
        ys.append(y[fin])
    x, m = pad_concat(xs)
    return x, m, torch.from_numpy(np.concatenate(ys)).float()


def loso_pooled_cs(
    tokens: dict[int, Tensor],
    labels: dict[int, np.ndarray],
    subjects: list[int],
    base_cfg: HeadTrainConfig,
    *,
    hp_grid: list[dict],
    n_diwa_seeds: int = 3,
    device: torch.device | None = None,
) -> dict:
    """Leave-one-SUBJECT-out cross-subject AUROC with NESTED-LOSO model selection.

    OUTER loop — each subject is the held-out TEST in turn (the reported zero-shot
    number). Within an outer fold ``train_s = cohort − test_s``:

    INNER-LOSO HP selection (the robustness fix). A single held-out val subject is a
    high-variance selection signal — inter-subject spread is large, so one quirky val
    subject can pick the wrong HP / stop time. Instead, for each HP we rotate the val
    subject across ALL of ``train_s``: train on ``train_s − val_s`` (pooled via
    :func:`pad_concat`), SWAD-early-stop on ``val_s``, and AVERAGE the SWAD val AUROC over
    every inner-val subject (1 seed — cheap; SWAD already denoises each run). The HP with
    the best MEAN inner-val AUROC wins, so selection reflects population-mean transfer,
    not one subject.

    FINAL refit at the chosen HP — a DiWA ensemble of ``n_diwa_seeds`` heads, each holding
    out a DIFFERENT rotating ``val_s`` for SWAD early-stop, parameter-averaged. Across the
    ensemble every train subject is covered and no single val subject drives the final
    model. The fused head is scored once on the untouched test subject.

    Firewall: ``test_s`` enters neither selection nor fit nor early-stop — only the final
    score. Returns ``{cs_mean, cs_std, folds}``; each fold logs
    ``test_s/fit_subjects/hp/sel_val/test``. ``tokens[s]`` ``(N_s,T_s,d)``, ``labels[s]``
    ``(N_s,)`` in {0,1}/NaN for ONE task. Cost ≈ ``|hp_grid|·|train_s| + n_diwa_seeds``
    head-trains per outer fold."""
    if not hp_grid:
        raise ValueError("hp_grid must be non-empty")
    device = device or torch.device("cpu")
    subs = list(subjects)
    folds: list[dict] = []
    for test_s in subs:
        train_s = [s for s in subs if s != test_s]
        x_te, m_te, y_te = _stack(tokens, labels, [test_s])
        y_te_np = y_te.numpy()

        def _fit_one(val_s: int, cfg: HeadTrainConfig) -> dict:
            fit_s = [s for s in train_s if s != val_s]
            x_tr, m_tr, y_tr = _stack(tokens, labels, fit_s)
            x_val, m_val, y_val = _stack(tokens, labels, [val_s])
            return train_head(
                x_tr, y_tr, x_val, y_val, cfg,
                mask_tr=m_tr, mask_val=m_val, device=device,
            )

        # --- inner-LOSO HP selection (1 seed, mean SWAD val AUROC over inner subjects) ---
        best: dict | None = None
        for hp in hp_grid:
            cfg = replace(base_cfg, **hp, seed=0)
            inner = [_fit_one(v, cfg)["swad_val"] for v in train_s]
            inner = [v for v in inner if np.isfinite(v)]
            mean_v = float(np.mean(inner)) if inner else float("nan")
            if best is None or (np.isfinite(mean_v) and mean_v > best["sel_val"]):
                best = {"sel_val": mean_v, "hp": hp}

        assert best is not None                            # hp_grid non-empty ⇒ set
        # --- final DiWA refit at the chosen HP (rotate val across seeds) ---
        cfg = replace(base_cfg, **best["hp"])
        states = [
            _fit_one(train_s[sd % len(train_s)], replace(cfg, seed=sd))["swad_state"]
            for sd in range(n_diwa_seeds)
        ]
        thead = _build_head(cfg).to(device)
        thead.load_state_dict(diwa_average(states))
        te = _auroc(thead, x_te, y_te_np, m_te, device)
        folds.append({
            "test_s": test_s, "fit_subjects": train_s,
            "hp": best["hp"], "sel_val": best["sel_val"], "test": te,
        })

    cs = [f["test"] for f in folds if np.isfinite(f["test"])]
    return {
        "cs_mean": float(np.mean(cs)) if cs else float("nan"),
        "cs_std": float(np.std(cs)) if cs else float("nan"),
        "folds": folds,
    }


def diwa_average(states: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """DiWA: parameter-wise mean of several head state-dicts (different seeds/HPs)."""
    if not states:
        raise ValueError("diwa_average needs at least one state dict")
    keys = states[0].keys()
    out: dict[str, Tensor] = {}
    for k in keys:
        acc = states[0][k].clone().float()
        for s in states[1:]:
            acc += s[k].float()
        out[k] = (acc / len(states)).to(states[0][k].dtype)
    return out
