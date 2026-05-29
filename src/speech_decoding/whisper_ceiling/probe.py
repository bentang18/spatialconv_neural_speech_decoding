"""LogReg probe over Whisper-layer features (L.7.B0/C0/S-layer/S-combine).

Mirror of the Neuroprobe brain-side probe: for each task, binarize the
continuous label by top-quartile-vs-bottom-quartile (matching
`BrainTreebankSubjectTrialBenchmarkDataset` semantics), fit a balanced LogReg
on Whisper features, score AUROC on the held-out split.

Three split protocols:
    - within-trial:    random 80/20 inside one (subject, trial)
    - cross-session:   train on trial A, test on trial B of the same subject
                       (only defined for subjects with ≥2 trials)
    - cross-subject:   train on subjects {S \\ s}, test on subject s

Layer-merge sweep — five strategies:
    - per-layer L:     single layer's mean-pooled feature (W, d)
    - concat-all:      stack every hooked layer (W, L*d)
    - mean-all:        elementwise mean across layers (W, d) — tests redundancy
    - concat-top-k:    pick top-k layers by within-trial AUROC, concat (W, k*d)
    - late-fusion:     per-layer probes, ensemble by mean probability
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Continuous Neuroprobe Lite tasks (top-quartile vs bottom-quartile binary).
# Maps probe-task-name -> column name in words_df after labels merge.
DEFAULT_TASKS: dict[str, str] = {
    "volume": "rms",
    "pitch": "enhanced_pitch",
    "delta_volume": "delta_rms",
    "gpt2_surprisal": "gpt2_surprisal",
    "word_length": "word_length",
    "frame_brightness": "mean_pixel_brightness",
    "global_flow": "max_global_magnitude",
    "local_flow": "max_vector_magnitude",
    "face_num": "face_num",
}


@dataclass
class ProbeResult:
    task: str
    layer: int | str
    split: str
    auroc: float
    n_train: int
    n_test: int
    seed: int


def binary_labels_from_continuous(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce Neuroprobe's top-quartile-vs-bottom-quartile binarization.

    Returns (kept_indices, binary_labels). Bottom-25% -> 0, top-25% -> 1.
    Middle 50% dropped (matching Neuroprobe).
    """
    v = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(v)
    finite = v[finite_mask]
    if finite.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    percentiles = np.array([np.mean(finite < x) for x in v])
    pos = np.where(finite_mask & (percentiles > 0.75))[0]
    neg = np.where(finite_mask & (percentiles < 0.25))[0]
    kept = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)])
    order = np.argsort(kept)
    return kept[order], labels[order]


def binary_labels_face_num(face_nums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """face_num: >0 vs ==0 (Neuroprobe binary)."""
    v = np.asarray(face_nums)
    finite_mask = np.isfinite(v.astype(float))
    pos = np.where(finite_mask & (v > 0))[0]
    neg = np.where(finite_mask & (v == 0))[0]
    kept = np.concatenate([neg, pos])
    labels = np.concatenate([np.zeros(len(neg), dtype=int), np.ones(len(pos), dtype=int)])
    order = np.argsort(kept)
    return kept[order], labels[order]


def _fit_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> float:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return float("nan")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=seed, solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]
    return float(roc_auc_score(y_test, proba))


def _fit_predict_proba(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Same as _fit_score's pipeline but returns predicted probabilities for the
    positive class — used by late-fusion ensembles."""
    if len(np.unique(y_train)) < 2:
        return np.full(len(X_test), 0.5, dtype=float)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced",
        random_state=seed, solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)
    return clf.predict_proba(X_test_s)[:, 1]


def late_fusion_within(
    layer_features: dict,
    labels: np.ndarray,
    task: str,
    seed: int = 42,
    test_frac: float = 0.2,
) -> ProbeResult:
    """Per-layer LogReg → ensemble by mean of positive-class probability.

    layer_features: {layer_idx -> (W, d) array} for the SAME word set.
    """
    n = len(labels)
    if n < 20:
        return ProbeResult(task, "late_fusion", "within_trial", float("nan"), n, 0, seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return ProbeResult(task, "late_fusion", "within_trial", float("nan"),
                           len(y_train), len(y_test), seed)
    probas = []
    for L, feats in layer_features.items():
        proba = _fit_predict_proba(
            feats[train_idx].astype(np.float32), y_train,
            feats[test_idx].astype(np.float32), seed,
        )
        probas.append(proba)
    ensemble = np.mean(np.stack(probas, axis=0), axis=0)
    auc = float(roc_auc_score(y_test, ensemble))
    return ProbeResult(task, "late_fusion", "within_trial", auc,
                       len(y_train), len(y_test), seed)


def topk_layers_by_within_auroc(
    per_layer_results: list[ProbeResult],
    k: int,
) -> list[int]:
    """Pick top-k integer-layer keys by within-trial AUROC. Ties broken by lower
    layer index (favor earlier layers when tied)."""
    items = [
        (r.auroc, r.layer) for r in per_layer_results
        if r.split == "within_trial"
        and isinstance(r.layer, int)
        and np.isfinite(r.auroc)
    ]
    items.sort(key=lambda x: (-x[0], int(x[1])))
    return [int(layer) for _, layer in items[:k]]


def fit_probe_within_trial(
    features: np.ndarray,
    labels: np.ndarray,
    task: str,
    layer: int | str,
    seed: int = 42,
    test_frac: float = 0.2,
) -> ProbeResult:
    """80/20 random split within one trial."""
    if len(features) < 20:
        return ProbeResult(task, layer, "within_trial", float("nan"),
                           len(features), 0, seed)
    split = train_test_split(
        features.astype(np.float32), labels,
        test_size=test_frac, stratify=labels, random_state=seed,
    )
    X_train, X_test, y_train, y_test = (np.asarray(x) for x in split)
    auc = _fit_score(X_train, y_train, X_test, y_test, seed)
    return ProbeResult(task, layer, "within_trial", auc,
                       len(y_train), len(y_test), seed)


def fit_probe_cross_session(
    features_by_trial: dict[int, np.ndarray],
    labels_by_trial: dict[int, np.ndarray],
    train_trial: int,
    test_trial: int,
    task: str,
    layer: int | str,
    seed: int = 42,
) -> ProbeResult:
    """Train on trial A, test on trial B (same subject)."""
    X_tr, y_tr = features_by_trial[train_trial], labels_by_trial[train_trial]
    X_te, y_te = features_by_trial[test_trial], labels_by_trial[test_trial]
    if len(X_tr) < 20 or len(X_te) < 20:
        return ProbeResult(task, layer, "cross_session", float("nan"),
                           len(X_tr), len(X_te), seed)
    auc = _fit_score(X_tr.astype(np.float32), y_tr,
                     X_te.astype(np.float32), y_te, seed)
    return ProbeResult(task, layer, "cross_session", auc,
                       len(y_tr), len(y_te), seed)


def fit_probe_cross_subject(
    features_by_subject: dict[int, np.ndarray],
    labels_by_subject: dict[int, np.ndarray],
    held_out_subject: int,
    task: str,
    layer: int | str,
    seed: int = 42,
) -> ProbeResult:
    """Leave-one-subject-out."""
    train_subjects = [s for s in features_by_subject if s != held_out_subject]
    if not train_subjects:
        return ProbeResult(task, layer, "cross_subject", float("nan"), 0, 0, seed)
    X_tr = np.concatenate([features_by_subject[s] for s in train_subjects], axis=0)
    y_tr = np.concatenate([labels_by_subject[s] for s in train_subjects], axis=0)
    X_te = features_by_subject[held_out_subject]
    y_te = labels_by_subject[held_out_subject]
    if len(X_tr) < 20 or len(X_te) < 20:
        return ProbeResult(task, layer, "cross_subject", float("nan"),
                           len(X_tr), len(X_te), seed)
    auc = _fit_score(X_tr.astype(np.float32), y_tr,
                     X_te.astype(np.float32), y_te, seed)
    return ProbeResult(task, layer, "cross_subject", auc,
                       len(y_tr), len(y_te), seed)
