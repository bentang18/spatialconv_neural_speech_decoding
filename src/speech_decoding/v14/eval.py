"""Slot-averaged phoneme error rate for v14-core (Task D2, `#33`).

PER = wrong-slot count / (L · B). Because every utterance has exactly `L = 3`
phonemes, slot-averaged PER reduces to a plain per-slot mismatch rate. Token-
level sequence alignment (edit distance) is unnecessary at this sequence
length and would confound phoneme-identity errors with segmentation errors
that cannot occur in this task.

The per-phoneme helpers below (plan P5) are the baseline-aligned path's eval
surface: `evaluate_per_phoneme` reports 1-slot PER via argmax for smoke tests;
`exhaustive_ar_per` enumerates 9³ hypotheses per trial and returns the
slot-averaged PER, matching the original `slot_averaged_per` metric.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def slot_averaged_per(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Wrong-slot count divided by `L · B`. Inputs must be `(B, L)` long."""

    if pred.shape != true.shape or pred.ndim != 2:
        raise ValueError(
            f"pred/true must match shape (B, L); got {tuple(pred.shape)} vs "
            f"{tuple(true.shape)}"
        )
    if pred.numel() == 0:
        raise ValueError("pred/true must be non-empty")
    return float((pred != true).float().mean().item())


# --------------------------------------------------------------------------- #
# Per-phoneme eval (plan P5 for the baseline-aligned per-phoneme stack).
# --------------------------------------------------------------------------- #


def _move_batch(batch: dict, device: torch.device) -> dict:
    moved: dict = {}
    for k, v in batch.items():
        moved[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return moved


@torch.no_grad()
def evaluate_per_phoneme(model: torch.nn.Module, loader: Iterable[dict]) -> float:
    """Per-phoneme 1-slot PER via argmax on logits. Smoke-test metric.

    ``model(batch)`` must return ``(B, V)`` logits.
    """

    model.eval()
    device = next(model.parameters()).device
    wrong = 0
    total = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        logits = model(batch)
        pred = logits.argmax(dim=-1).cpu()
        labels = batch["labels"].cpu()
        wrong += int((pred != labels).sum())
        total += int(labels.numel())
    if total == 0:
        raise ValueError("evaluate_per_phoneme loader produced no samples")
    return wrong / total


@torch.no_grad()
def exhaustive_ar_per(
    memories_per_pos: list[torch.Tensor],  # [(T, S, d), (T, S, d), (T, S, d)]
    labels: torch.Tensor,                   # (T, 3) long — ground-truth per slot
    decoder: torch.nn.Module,
    cell_queries_per_pos: list[torch.Tensor] | None = None,
) -> float:
    """Enumerate 9³ hypotheses per trial; return slot-averaged PER.

    Uses 1 + V + V decoder calls per pool of trials (V = 9 phonemes). The
    score for a hypothesis ``(p0, p1, p2)`` is the sum of per-slot log-probs
    under teacher-forced context (BOS, p0, p1).

    ``cell_queries_per_pos`` (optional) is a parallel list of `(T, n_cells, d)`
    tensors for the atlas-anchored hierarchical readout — one per position,
    passed through to every decoder call for that position.
    """

    if len(memories_per_pos) != 3:
        raise ValueError(
            f"expected memories for 3 positions, got {len(memories_per_pos)}"
        )
    T = memories_per_pos[0].shape[0]
    if labels.shape != (T, 3):
        raise ValueError(f"labels shape {tuple(labels.shape)} != ({T}, 3)")
    if cell_queries_per_pos is not None and len(cell_queries_per_pos) != 3:
        raise ValueError(
            f"cell_queries_per_pos must have 3 entries, got "
            f"{len(cell_queries_per_pos)}"
        )

    device = memories_per_pos[0].device
    V = 9
    BOS = -1

    mem0, mem1, mem2 = memories_per_pos
    if cell_queries_per_pos is None:
        cq0 = cq1 = cq2 = None
    else:
        cq0, cq1, cq2 = cell_queries_per_pos

    bos = torch.full((T,), BOS, dtype=torch.long, device=device)
    log_p0 = F.log_softmax(decoder(mem0, bos, cell_query=cq0), dim=-1)  # (T, V)

    # For each p0 ∈ 0..V-1, log P(p1 | mem1, prev=p0). Shape (V, T, V).
    log_p1 = torch.empty(V, T, V, device=device)
    for p in range(V):
        prev = torch.full((T,), p, dtype=torch.long, device=device)
        log_p1[p] = F.log_softmax(decoder(mem1, prev, cell_query=cq1), dim=-1)

    log_p2 = torch.empty(V, T, V, device=device)
    for p in range(V):
        prev = torch.full((T,), p, dtype=torch.long, device=device)
        log_p2[p] = F.log_softmax(decoder(mem2, prev, cell_query=cq2), dim=-1)

    # scores[t, p0, p1, p2] = log_p0[t, p0] + log_p1[p0, t, p1] + log_p2[p1, t, p2]
    scores = (
        log_p0[:, :, None, None]                     # (T, V, 1, 1)
        + log_p1.permute(1, 0, 2)[:, :, :, None]     # (T, V, V, 1) — indexed by [t, p0, p1]
        + log_p2.permute(1, 0, 2)[:, None, :, :]     # (T, 1, V, V) — indexed by [t, p1, p2]
    )                                                 # (T, V, V, V)

    flat = scores.view(T, V * V * V).argmax(dim=-1)   # (T,)
    p0_hat = (flat // (V * V)).cpu()
    p1_hat = ((flat // V) % V).cpu()
    p2_hat = (flat % V).cpu()
    pred = torch.stack([p0_hat, p1_hat, p2_hat], dim=1)  # (T, 3)

    return slot_averaged_per(pred, labels.cpu())


@torch.no_grad()
def exhaustive_ar_per_from_loader(
    model: torch.nn.Module,
    loader: Iterable[dict],
) -> float:
    """Convenience: gather memories from a per-phoneme loader, group by trial,
    run exhaustive AR decode. Expects ``model.encode_memory(batch)`` and
    ``model.decoder``; samples must carry ``trial_id`` and ``phoneme_pos``.
    """

    model.eval()
    device = next(model.parameters()).device

    mems_by_pos: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    labels_by_pos: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    trial_by_pos: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    cqs_by_pos: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    needs_cq = getattr(model, "compute_cell_query", None) is not None

    for batch in loader:
        batch = _move_batch(batch, device)
        memory = model.encode_memory(batch)  # (B, S, d)
        cq = model.compute_cell_query(batch) if needs_cq else None  # (B, C, d) | None
        for i in range(memory.shape[0]):
            pos = int(batch["phoneme_pos"][i])
            mems_by_pos[pos].append(memory[i : i + 1].cpu())
            labels_by_pos[pos].append(batch["labels"][i : i + 1].cpu())
            trial_by_pos[pos].append(batch["trial_id"][i : i + 1].cpu())
            if cq is not None:
                cqs_by_pos[pos].append(cq[i : i + 1].cpu())

    # Align by trial_id (sort) so positions 0/1/2 line up.
    def _concat_sorted(mems, labs, trials, cqs):
        mems_cat = torch.cat(mems, dim=0)
        labs_cat = torch.cat(labs, dim=0)
        trials_cat = torch.cat(trials, dim=0)
        order = torch.argsort(trials_cat)
        cq_cat = torch.cat(cqs, dim=0)[order] if cqs else None
        return mems_cat[order], labs_cat[order], trials_cat[order], cq_cat

    m0, l0, t0, c0 = _concat_sorted(
        mems_by_pos[0], labels_by_pos[0], trial_by_pos[0], cqs_by_pos[0]
    )
    m1, l1, t1, c1 = _concat_sorted(
        mems_by_pos[1], labels_by_pos[1], trial_by_pos[1], cqs_by_pos[1]
    )
    m2, l2, t2, c2 = _concat_sorted(
        mems_by_pos[2], labels_by_pos[2], trial_by_pos[2], cqs_by_pos[2]
    )
    if not (torch.equal(t0, t1) and torch.equal(t1, t2)):
        raise ValueError("trial_id mismatch across phoneme positions")

    labels = torch.stack([l0, l1, l2], dim=1)  # (T, 3)
    mems = [m0.to(device), m1.to(device), m2.to(device)]
    if c0 is None or c1 is None or c2 is None:
        cqs = None
    else:
        cqs = [c0.to(device), c1.to(device), c2.to(device)]
    return exhaustive_ar_per(mems, labels, model.decoder, cell_queries_per_pos=cqs)
