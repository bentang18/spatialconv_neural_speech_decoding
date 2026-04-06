"""Evaluation metrics for speech decoding.

Primary: PER (phoneme error rate).
Secondary: Balanced accuracy per position, CTC length accuracy.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, r2_score


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


def framewise_r2_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    top_k_dims: int = 5,
) -> dict[str, float]:
    """R^2 diagnostics for framewise regression targets.

    Reports variance-weighted R^2 on speech-only, silence-only, and all frames.
    Also reports per-dimension R^2 for the first top_k_dims dimensions using
    speech frames only.
    """
    result: dict[str, float] = {}
    selectors = {
        "speech": mask > 0.5,
        "silence": mask <= 0.5,
        "all": np.ones_like(mask, dtype=bool),
    }

    flat_pred = prediction.reshape(-1, prediction.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    flat_mask = mask.reshape(-1)

    for name, selector in selectors.items():
        sel = selector.reshape(-1) if selector.ndim > 1 else selector
        if sel.sum() == 0:
            result[f"r2_{name}"] = 0.0
            continue
        y_true = flat_target[sel]
        y_pred = flat_pred[sel]
        if np.allclose(np.var(y_true, axis=0), 0):
            result[f"r2_{name}"] = 0.0
            continue
        result[f"r2_{name}"] = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))

    speech_selector = flat_mask > 0.5
    if speech_selector.sum() > 0:
        y_true = flat_target[speech_selector]
        y_pred = flat_pred[speech_selector]
        n_dims = min(top_k_dims, y_true.shape[1])
        for idx in range(n_dims):
            if np.allclose(np.var(y_true[:, idx]), 0):
                result[f"r2_dim{idx + 1}"] = 0.0
            else:
                result[f"r2_dim{idx + 1}"] = float(r2_score(y_true[:, idx], y_pred[:, idx]))
    return result


def segment_r2_diagnostics(
    prediction: np.ndarray,
    target: np.ndarray,
    top_k_dims: int = 5,
) -> dict[str, float]:
    """R^2 diagnostics for segment-level regression targets."""
    result: dict[str, float] = {}
    flat_pred = prediction.reshape(-1, prediction.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    if np.allclose(np.var(flat_target, axis=0), 0):
        result["r2_segment"] = 0.0
    else:
        result["r2_segment"] = float(
            r2_score(flat_target, flat_pred, multioutput="variance_weighted")
        )
    n_dims = min(top_k_dims, flat_target.shape[1])
    for idx in range(n_dims):
        if np.allclose(np.var(flat_target[:, idx]), 0):
            result[f"r2_dim{idx + 1}"] = 0.0
        else:
            result[f"r2_dim{idx + 1}"] = float(
                r2_score(flat_target[:, idx], flat_pred[:, idx])
            )
    return result
