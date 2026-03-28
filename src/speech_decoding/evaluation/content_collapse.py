"""Content-collapse diagnostics for phoneme decoding.

Catches failure modes that loss curves miss: concentrated phoneme
distributions, stereotyped output sequences. Ref: RD-78.

Sequence diversity (unique / N_predictions) and possible-sequence coverage
(unique / 729) are kept separate — stereotypy_index measures the former,
possible_sequence_coverage measures the latter.
"""
from __future__ import annotations

import numpy as np


def output_entropy(preds: np.ndarray, n_classes: int = 9) -> float:
    """Shannon entropy (bits) of predicted phoneme distribution.

    Max = log2(n_classes) ≈ 3.17 for 9 phonemes. Alarm if < 1.5.
    """
    counts = np.bincount(preds, minlength=n_classes + 1)[1:n_classes + 1]
    probs = counts / counts.sum() if counts.sum() > 0 else np.ones(n_classes) / n_classes
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def unigram_kl(preds: np.ndarray, n_classes: int = 9) -> float:
    """KL divergence (nats) of predicted marginal from uniform.

    High KL = model favors certain phonemes regardless of input.
    """
    counts = np.bincount(preds, minlength=n_classes + 1)[1:n_classes + 1]
    p = counts / counts.sum() if counts.sum() > 0 else np.ones(n_classes) / n_classes
    q = np.ones(n_classes) / n_classes
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def stereotypy_index(sequences: list[list[int]]) -> float:
    """Fraction of unique predicted sequences out of produced sequences.

    stereotypy_index = unique / N_predictions.
    1.0 = all outputs distinct; 1/N = all outputs identical.
    """
    unique = set(tuple(s) for s in sequences)
    return len(unique) / len(sequences) if sequences else 0.0


def possible_sequence_coverage(
    sequences: list[list[int]],
    n_classes: int = 9,
    n_positions: int = 3,
) -> float:
    """Fraction of the full output space covered by unique predictions.

    possible_sequence_coverage = unique / n_classes**n_positions.
    Alarm threshold: < 0.10 (RD-78).
    """
    unique = set(tuple(s) for s in sequences)
    return len(unique) / float(n_classes ** n_positions)


def content_collapse_report(
    preds_per_position: list[np.ndarray],
    sequences: list[list[int]],
    n_classes: int = 9,
) -> dict:
    """Full content-collapse diagnostic report.

    Parameters
    ----------
    preds_per_position:
        List of length n_positions. Each element is a 1-D array of
        predicted phoneme labels (1-indexed) for that position across
        all trials.
    sequences:
        List of predicted 3-phoneme sequences, one per trial.
    n_classes:
        Number of phoneme classes (default 9 for PS task).

    Returns
    -------
    dict with keys:
        entropy                  – per-position Shannon entropy (bits)
        max_entropy              – log2(n_classes)
        unigram_kl               – per-position KL from uniform (nats)
        stereotypy_index         – unique / N_predictions
        possible_sequence_coverage – unique / n_classes**3
        collapsed                – bool alarm flag
    """
    entropies = [output_entropy(p, n_classes) for p in preds_per_position]
    kls = [unigram_kl(p, n_classes) for p in preds_per_position]
    stereo = stereotypy_index(sequences)
    coverage = possible_sequence_coverage(sequences, n_classes=n_classes)
    max_entropy = np.log2(n_classes)

    collapsed = (
        any(h < 1.5 for h in entropies)
        or coverage < 0.10
    )

    return {
        "entropy": entropies,
        "max_entropy": float(max_entropy),
        "unigram_kl": kls,
        "stereotypy_index": stereo,
        "possible_sequence_coverage": coverage,
        "collapsed": collapsed,
    }
