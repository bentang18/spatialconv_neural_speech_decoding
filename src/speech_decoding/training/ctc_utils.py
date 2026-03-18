"""CTC loss wrapper, greedy decoder, and training utilities.

Handles the PyTorch CTC quirk (T,B,C input order) and provides
phoneme error rate computation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def ctc_loss(
    log_probs: torch.Tensor,
    targets: list[list[int]],
) -> torch.Tensor:
    """Compute CTC loss with correct PyTorch input format.

    Args:
        log_probs: (B, T, C) log probabilities from model head.
        targets: List of integer label sequences (no blanks).

    Returns:
        Scalar loss (mean over batch).
    """
    B, T, C = log_probs.shape
    # PyTorch CTC expects (T, B, C)
    log_probs_t = log_probs.permute(1, 0, 2)  # (T, B, C)
    tgts, tgt_lens = encode_labels_for_ctc(targets)
    tgts = tgts.to(log_probs.device)
    tgt_lens = tgt_lens.to(log_probs.device)
    input_lengths = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

    return F.ctc_loss(
        log_probs_t, tgts, input_lengths, tgt_lens,
        blank=0, reduction="mean", zero_infinity=True,
    )


def encode_labels_for_ctc(
    labels: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten label sequences for PyTorch CTC.

    Returns:
        targets: 1D tensor of concatenated labels.
        target_lengths: 1D tensor of per-sequence lengths.
    """
    flat = []
    lengths = []
    for seq in labels:
        flat.extend(seq)
        lengths.append(len(seq))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def greedy_decode(log_probs: torch.Tensor) -> list[list[int]]:
    """Greedy CTC decoding: argmax → collapse repeats → remove blanks.

    Args:
        log_probs: (B, T, C) log probabilities.

    Returns:
        List of decoded integer sequences (one per batch item).
    """
    best = log_probs.argmax(dim=-1)  # (B, T)
    decoded = []
    for b in range(best.shape[0]):
        seq = best[b].tolist()
        # Collapse consecutive repeats
        collapsed = []
        prev = None
        for s in seq:
            if s != prev:
                collapsed.append(s)
            prev = s
        # Remove blanks (index 0)
        result = [s for s in collapsed if s != 0]
        decoded.append(result)
    return decoded


def compute_per(
    predictions: list[list[int]],
    targets: list[list[int]],
) -> float:
    """Compute Phoneme Error Rate (edit distance / target length).

    Args:
        predictions: Decoded sequences (from greedy_decode).
        targets: Ground truth label sequences.

    Returns:
        Mean PER across batch.
    """
    total_dist = 0
    total_len = 0
    for pred, tgt in zip(predictions, targets):
        total_dist += _edit_distance(pred, tgt)
        total_len += len(tgt)
    return total_dist / max(total_len, 1)


def _edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein distance between two integer sequences."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def blank_ratio(log_probs: torch.Tensor) -> float:
    """Fraction of time frames where blank (idx 0) is the argmax."""
    best = log_probs.argmax(dim=-1)  # (B, T)
    return (best == 0).float().mean().item()
