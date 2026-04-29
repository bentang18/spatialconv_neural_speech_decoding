"""Training loop for v14-core (Task D3, First-Run Protocol).

Plain slot-wise cross-entropy on 3 phoneme slots (`#9` — no focal loss, no
label smoothing, no mixup, no augmentation). AdamW `lr=1e-3`, `wd=1e-4`,
cosine decay with 20-epoch linear warmup, grad clip `1.0`, mixed precision
on CUDA only. `trials_per_batch=8` with gradient accumulation to an effective
batch of 32. Val every 5 epochs; early-stop after 10 non-improving val checks.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from speech_decoding.v14.eval import slot_averaged_per


TRIALS_PER_BATCH = 8
EFFECTIVE_BATCH = 32
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH // TRIALS_PER_BATCH  # 4

LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
WARMUP_EPOCHS = 20
MAX_EPOCHS = 300
VAL_EVERY = 5
EARLY_STOP_PATIENCE = 10


def compute_slot_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over ``B · L`` slots. ``logits: (B, L, V)``, ``labels: (B, L)``."""

    B, L, V = logits.shape
    return F.cross_entropy(logits.reshape(B * L, V), labels.reshape(B * L))


def compute_flat_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-phoneme cross-entropy. ``logits: (B, V)``, ``labels: (B,)``."""

    return F.cross_entropy(logits, labels)


# --------------------------------------------------------------------------- #
# Default model call + loss wrappers.
# --------------------------------------------------------------------------- #


def _default_slot_ce_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    """Original B-1 slot-CE path — `model(batch, mode='train')` → (B, L, V)."""
    logits = model(batch, mode="train")
    return compute_slot_ce(logits, batch["labels"])


def per_phoneme_ce_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    """Per-phoneme path — `model(batch)` → (B, V), labels (B,)."""
    logits = model(batch)
    return compute_flat_ce(logits, batch["labels"])


def make_per_phoneme_ce_loss(
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    aug_cfg: "AugmentationConfig | None" = None,
) -> Callable[[nn.Module, dict], torch.Tensor]:
    """Build a per-phoneme CE loss with optional label smoothing + mixup + aug.

    Mixup interpolates the raw signal and the *loss* (not the labels, which
    are categorical). `prev_tokens` and all per-patient constants (layout,
    support, grid shape, active mask) are kept from the primary example —
    mixing is a one-sided signal perturbation, not a full context blend.

    `aug_cfg` (if given) applies signal-level augmentation BEFORE mixup, only
    in training mode. Augmentation may also update `electrode_active_mask`
    (channel dropout).
    """

    from speech_decoding.v14.augmentation import AugmentationConfig, augment_batch

    def _loss(model: nn.Module, batch: dict) -> torch.Tensor:
        labels = batch["labels"]
        if aug_cfg is not None and model.training:
            batch = augment_batch(batch, aug_cfg)
        if mixup_alpha > 0.0 and model.training:
            beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
            lam = float(beta.sample().item())
            B = labels.shape[0]
            perm = torch.randperm(B, device=labels.device)
            mixed_signal = lam * batch["signal"] + (1.0 - lam) * batch["signal"][perm]
            mixed_batch = dict(batch)
            mixed_batch["signal"] = mixed_signal
            logits = model(mixed_batch)
            loss_a = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            loss_b = F.cross_entropy(
                logits, labels[perm], label_smoothing=label_smoothing
            )
            return lam * loss_a + (1.0 - lam) * loss_b
        logits = model(batch)
        return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

    return _loss


def _move(batch: dict, device: torch.device) -> dict:
    moved: dict = {}
    for k, v in batch.items():
        moved[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return moved


def make_cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> float:
    """Single unaccumulated AdamW step. Returns the scalar loss."""

    model.train()
    device = next(model.parameters()).device
    batch = _move(batch, device)
    optimizer.zero_grad(set_to_none=True)
    if use_amp and scaler is not None:
        with torch.amp.autocast("cuda"):
            logits = model(batch, mode="train")
            loss = compute_slot_ce(logits, batch["labels"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(batch, mode="train")
        loss = compute_slot_ce(logits, batch["labels"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
    return float(loss.detach().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable[dict]) -> float:
    """Run exhaustive decode over ``loader`` and return slot-averaged PER."""

    model.eval()
    device = next(model.parameters()).device
    preds: list[torch.Tensor] = []
    trues: list[torch.Tensor] = []
    for batch in loader:
        batch = _move(batch, device)
        pred = model(batch, mode="eval")
        preds.append(pred.cpu())
        trues.append(batch["labels"].cpu())
    if not preds:
        raise ValueError("evaluate loader produced no batches")
    return slot_averaged_per(torch.cat(preds, dim=0), torch.cat(trues, dim=0))


@dataclass
class FoldResult:
    best_val_per: float
    final_epoch: int
    early_stopped: bool


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    max_epochs: int = MAX_EPOCHS,
    val_every: int = VAL_EVERY,
    patience: int = EARLY_STOP_PATIENCE,
    log_callback: Callable[[dict], None] | None = None,
    loss_fn: Callable[[nn.Module, dict], torch.Tensor] = _default_slot_ce_loss,
    evaluate_fn: Callable[[nn.Module, Iterable[dict]], float] | None = None,
    warmup_epochs: int = WARMUP_EPOCHS,
    grad_accum_steps: int | None = None,
) -> FoldResult:
    """Full First-Run Protocol training loop for one fold/seed.

    ``log_callback`` receives one dict per epoch with keys ``epoch``,
    ``train_loss``, ``lr``, and (on val epochs) ``val_per``. The CLI wires it
    to a JSONL writer; tests can capture into a list.

    ``loss_fn`` is the train-step loss wrapper. Default is the B-1 slot-CE
    path; pass ``per_phoneme_ce_loss`` (or any other callable) for the
    baseline-aligned per-phoneme path. Signature: ``(model, batch) -> loss``.

    ``evaluate_fn`` is the val-time PER reporter. Default is the slot-CE
    ``evaluate`` above; per-phoneme callers should pass their own.

    ``grad_accum_steps`` overrides the module default (4) — the pooled runner
    sets this to ``n_patients`` so each optimizer step sees one micro-batch
    from every patient.
    """

    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = make_cosine_warmup_schedule(optimizer, warmup_epochs, max_epochs)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    accum_steps = grad_accum_steps if grad_accum_steps is not None else GRAD_ACCUM_STEPS

    if evaluate_fn is None:
        evaluate_fn = evaluate

    best_val = math.inf
    best_state_dict: dict | None = None
    non_improving = 0
    final_epoch = 0
    early_stopped = False

    for epoch in range(max_epochs):
        final_epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum = 0.0
        epoch_micro = 0
        for i, batch in enumerate(train_loader):
            batch = _move(batch, device)
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    raw_loss = loss_fn(model, batch)
                    loss = raw_loss / accum_steps
                scaler.scale(loss).backward()
            else:
                raw_loss = loss_fn(model, batch)
                loss = raw_loss / accum_steps
                loss.backward()
            epoch_loss_sum += float(raw_loss.detach().item())
            epoch_micro += 1
            if (i + 1) % accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        # Flush any tail accumulation so small datasets (and partial-tail epochs
        # on any dataset) still get at least one optimizer step per epoch.
        if epoch_micro % accum_steps != 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        log: dict = {
            "epoch": epoch,
            "train_loss": epoch_loss_sum / max(1, epoch_micro),
            "lr": optimizer.param_groups[0]["lr"],
        }

        if (epoch + 1) % val_every == 0:
            val_per = evaluate_fn(model, val_loader)
            log["val_per"] = val_per
            if val_per < best_val:
                best_val = val_per
                best_state_dict = copy.deepcopy(model.state_dict())
                non_improving = 0
            else:
                non_improving += 1
            if non_improving >= patience:
                early_stopped = True

        if log_callback is not None:
            log_callback(log)

        if early_stopped:
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return FoldResult(
        best_val_per=best_val if math.isfinite(best_val) else 1.0,
        final_epoch=final_epoch,
        early_stopped=early_stopped,
    )
