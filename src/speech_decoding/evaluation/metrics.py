"""Evaluation metrics for speech decoding.

Primary: PER (phoneme error rate).
Secondary: Balanced accuracy per position, CTC length accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from speech_decoding.data.phoneme_map import index_to_phoneme
from speech_decoding.training.ctc_utils import compute_per


def per_position_balanced_accuracy(
    predictions: list[list[int]],
    targets: list[list[int]],
    n_positions: int = 3,
) -> list[float]:
    """Balanced accuracy at each phoneme position.

    Only includes trials where prediction has exactly n_positions phonemes.
    """
    results = []
    for pos in range(n_positions):
        y_true = []
        y_pred = []
        for pred, tgt in zip(predictions, targets):
            if len(pred) >= pos + 1 and len(tgt) >= pos + 1:
                y_true.append(tgt[pos])
                y_pred.append(pred[pos])
        if len(y_true) > 0 and len(set(y_true)) > 1:
            results.append(balanced_accuracy_score(y_true, y_pred))
        else:
            results.append(0.0)
    return results


def ctc_length_accuracy(
    predictions: list[list[int]],
    target_length: int = 3,
) -> float:
    """Fraction of predictions with exactly target_length phonemes."""
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if len(p) == target_length)
    return correct / len(predictions)


def evaluate_predictions(
    predictions: list[list[int]],
    targets: list[list[int]],
    n_positions: int = 3,
) -> dict[str, float]:
    """Compute all evaluation metrics.

    Returns dict with: per, bal_acc_p1..p3, bal_acc_mean, length_accuracy.
    """
    per = compute_per(predictions, targets)
    pos_acc = per_position_balanced_accuracy(predictions, targets, n_positions)
    length_acc = ctc_length_accuracy(predictions, target_length=n_positions)

    result = {"per": per, "length_accuracy": length_acc}
    for i, acc in enumerate(pos_acc):
        result[f"bal_acc_p{i + 1}"] = acc
    result["bal_acc_mean"] = np.mean(pos_acc).item() if pos_acc else 0.0
    return result
