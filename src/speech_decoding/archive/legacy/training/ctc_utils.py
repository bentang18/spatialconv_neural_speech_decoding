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


def per_position_ce_loss(
    logits: torch.Tensor,
    targets: list[list[int]],
    n_positions: int = 3,
) -> torch.Tensor:
    """Cross-entropy loss with per-position classification heads.

    Mean-pools the ENTIRE temporal sequence, then applies CE for each
    phoneme position independently. This matches Spalding's per-position
    SVM approach — the model must encode all 3 phonemes into one shared
    representation vector.

    Args:
        logits: (B, T, n_positions * n_phonemes) raw logits from multi-head
                projection. First n_phonemes columns = position 1, etc.
        targets: List of integer label sequences, 1-indexed (e.g., [[1,3,5], ...]).
        n_positions: Number of phoneme positions (default 3 for CVC/VCV).

    Returns:
        Scalar loss (mean over batch and positions).
    """
    n_phonemes = logits.shape[2] // n_positions
    # Global temporal mean pool
    pooled = logits.mean(dim=1)  # (B, n_positions * n_phonemes)
    loss = torch.tensor(0.0, device=logits.device)

    for pos in range(n_positions):
        pos_logits = pooled[:, pos * n_phonemes : (pos + 1) * n_phonemes]  # (B, n_phonemes)
        tgt = torch.tensor(
            [t[pos] - 1 for t in targets], dtype=torch.long, device=logits.device
        )
        loss = loss + F.cross_entropy(pos_logits, tgt)

    return loss / n_positions


def per_position_ce_decode(
    logits: torch.Tensor,
    n_positions: int = 3,
) -> list[list[int]]:
    """Decode by argmax per position from globally pooled representation.

    Args:
        logits: (B, T, n_positions * n_phonemes) raw logits.
        n_positions: Number of phoneme positions.

    Returns:
        List of decoded sequences (1-indexed phoneme labels).
    """
    n_phonemes = logits.shape[2] // n_positions
    pooled = logits.mean(dim=1)  # (B, n_positions * n_phonemes)
    decoded = []
    for b in range(pooled.shape[0]):
        seq = []
        for pos in range(n_positions):
            pos_logits = pooled[b, pos * n_phonemes : (pos + 1) * n_phonemes]
            seq.append(pos_logits.argmax().item() + 1)  # 1-indexed
        decoded.append(seq)
    return decoded


def ce_pooled_loss(
    logits: torch.Tensor,
    targets: list[list[int]],
    n_positions: int = 3,
) -> torch.Tensor:
    """CE loss on already-pooled logits (no temporal dim).

    For use with heads that do their own temporal pooling (e.g.,
    CEPositionHead with attention).

    Args:
        logits: (B, n_positions * n_phonemes) — already pooled.
        targets: List of integer label sequences, 1-indexed.
        n_positions: Number of phoneme positions.
    """
    n_phonemes = logits.shape[1] // n_positions
    loss = torch.tensor(0.0, device=logits.device)
    for pos in range(n_positions):
        pos_logits = logits[:, pos * n_phonemes : (pos + 1) * n_phonemes]
        tgt = torch.tensor(
            [t[pos] - 1 for t in targets], dtype=torch.long, device=logits.device
        )
        loss = loss + F.cross_entropy(pos_logits, tgt)
    return loss / n_positions


def ce_pooled_decode(
    logits: torch.Tensor,
    n_positions: int = 3,
) -> list[list[int]]:
    """Decode from already-pooled logits (no temporal dim).

    Args:
        logits: (B, n_positions * n_phonemes) — already pooled.
        n_positions: Number of phoneme positions.
    """
    n_phonemes = logits.shape[1] // n_positions
    decoded = []
    for b in range(logits.shape[0]):
        seq = []
        for pos in range(n_positions):
            pos_logits = logits[b, pos * n_phonemes : (pos + 1) * n_phonemes]
            seq.append(pos_logits.argmax().item() + 1)
        decoded.append(seq)
    return decoded


def blank_ratio(log_probs: torch.Tensor) -> float:
    """Fraction of time frames where blank (idx 0) is the argmax."""
    best = log_probs.argmax(dim=-1)  # (B, T)
    return (best == 0).float().mean().item()


def mfa_guided_ce_loss(
    logits: torch.Tensor,
    targets: list[list[int]],
    segment_masks: torch.Tensor,
) -> torch.Tensor:
    """CE loss with MFA-derived per-position temporal pooling.

    Instead of mean-pooling over ALL frames (like per_position_ce_loss),
    each position classifier pools only over the frames corresponding to
    its phoneme, as defined by MFA segment boundaries.

    Args:
        logits: (B, T, n_phonemes) raw logits — one shared 9-way head.
        targets: List of integer label sequences, 1-indexed.
        segment_masks: (B, n_positions, T) per-trial, per-position frame
            weights. mask[b, p, t] = 1.0 if frame t belongs to phoneme p.

    Returns:
        Scalar loss (mean over batch and positions).
    """
    n_positions = segment_masks.shape[1]
    loss = torch.tensor(0.0, device=logits.device)
    n_active = 0

    for pos in range(n_positions):
        # (B, T) mask for this position
        mask = segment_masks[:, pos, :]  # (B, T)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        # Weighted mean pool: (B, T, C) * (B, T, 1) → sum → / denom → (B, C)
        pooled = (logits * mask.unsqueeze(-1)).sum(dim=1) / denom  # (B, C)

        tgt = torch.tensor(
            [t[pos] - 1 for t in targets], dtype=torch.long, device=logits.device
        )

        # Skip positions with empty masks for all trials
        if mask.sum() > 0:
            loss = loss + F.cross_entropy(pooled, tgt)
            n_active += 1

    return loss / max(n_active, 1)


def mfa_guided_ce_decode(
    logits: torch.Tensor,
    segment_masks: torch.Tensor,
) -> list[list[int]]:
    """Decode phonemes using MFA-guided per-position pooling.

    Args:
        logits: (B, T, n_phonemes) raw logits.
        segment_masks: (B, n_positions, T) per-position frame weights.

    Returns:
        List of decoded sequences (1-indexed phoneme labels).
    """
    n_positions = segment_masks.shape[1]
    decoded = []
    for b in range(logits.shape[0]):
        seq = []
        for pos in range(n_positions):
            mask = segment_masks[b, pos, :]  # (T,)
            denom = mask.sum().clamp_min(1.0)
            pooled = (logits[b] * mask.unsqueeze(-1)).sum(dim=0) / denom  # (C,)
            seq.append(pooled.argmax().item() + 1)  # 1-indexed
        decoded.append(seq)
    return decoded
