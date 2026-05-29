"""Tests for whisper_ceiling.probe."""
import numpy as np

from speech_decoding.whisper_ceiling.probe import (
    binary_labels_face_num,
    binary_labels_from_continuous,
    fit_probe_cross_session,
    fit_probe_cross_subject,
    fit_probe_within_trial,
    late_fusion_within,
    topk_layers_by_within_auroc,
)


def test_binary_labels_from_continuous_quartiles():
    """Bottom-25% -> 0, top-25% -> 1, middle 50% dropped."""
    values = np.arange(100, dtype=float)  # 0..99
    kept, labels = binary_labels_from_continuous(values)
    # n_pos ≈ 25, n_neg ≈ 25
    assert 22 <= (labels == 0).sum() <= 28
    assert 22 <= (labels == 1).sum() <= 28
    assert set(np.unique(labels)) == {0, 1}
    # All neg values < all pos values
    neg_vals = values[kept[labels == 0]]
    pos_vals = values[kept[labels == 1]]
    assert neg_vals.max() < pos_vals.min()


def test_binary_labels_from_continuous_handles_nans():
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0, 7.0, 9.0, 11.0, 13.0])
    kept, labels = binary_labels_from_continuous(values)
    # NaNs must be excluded from kept
    assert all(np.isfinite(values[i]) for i in kept)


def test_binary_labels_face_num():
    face = np.array([0, 1, 0, 2, 0, 3, 0])
    kept, labels = binary_labels_face_num(face)
    # Two classes: faces (1) and no-faces (0)
    assert set(labels.tolist()) == {0, 1}
    assert labels[face[kept] == 0].sum() == 0  # all face==0 are class 0
    assert labels[face[kept] > 0].all()        # all face>0 are class 1


def test_fit_probe_within_trial_perfect_separation_scores_high():
    rng = np.random.default_rng(0)
    n = 200
    # Class 0 mean -2, class 1 mean +2 on dim 0
    X0 = rng.normal(-2, 1, size=(n // 2, 8)).astype(np.float32)
    X1 = rng.normal(+2, 1, size=(n // 2, 8)).astype(np.float32)
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    result = fit_probe_within_trial(X, y, task="synthetic", layer=0, seed=42)
    assert result.split == "within_trial"
    assert result.task == "synthetic"
    assert result.layer == 0
    assert result.auroc > 0.95
    assert result.n_train + result.n_test == n


def test_fit_probe_within_trial_random_features_chance():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    result = fit_probe_within_trial(X, y, task="syn", layer=0, seed=42)
    assert 0.3 < result.auroc < 0.7  # near chance


def test_fit_probe_cross_session_uses_separate_trials():
    rng = np.random.default_rng(0)
    Xa = np.concatenate([
        rng.normal(-2, 1, size=(40, 4)), rng.normal(+2, 1, size=(40, 4))
    ]).astype(np.float32)
    ya = np.array([0] * 40 + [1] * 40)
    Xb = np.concatenate([
        rng.normal(-2, 1, size=(40, 4)), rng.normal(+2, 1, size=(40, 4))
    ]).astype(np.float32)
    yb = np.array([0] * 40 + [1] * 40)
    result = fit_probe_cross_session(
        features_by_trial={0: Xa, 1: Xb}, labels_by_trial={0: ya, 1: yb},
        train_trial=0, test_trial=1, task="syn", layer=0,
    )
    assert result.split == "cross_session"
    assert result.auroc > 0.85


def test_late_fusion_beats_uninformative_layers():
    """If one layer carries the signal and others are noise, ensemble should
    still pull AUROC above chance."""
    rng = np.random.default_rng(0)
    n = 200
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)]).astype(int)
    # Layer 0: informative (mean depends on class)
    X0 = np.concatenate([
        rng.normal(-1.5, 1, size=(n // 2, 8)), rng.normal(+1.5, 1, size=(n // 2, 8))
    ]).astype(np.float32)
    # Layers 1-3: pure noise
    Xn = rng.normal(0, 1, size=(n, 8)).astype(np.float32)
    layer_feats = {0: X0, 1: Xn, 2: Xn + rng.normal(0, 0.1, Xn.shape).astype(np.float32),
                   3: Xn + rng.normal(0, 0.1, Xn.shape).astype(np.float32)}
    result = late_fusion_within(layer_feats, y, task="syn", seed=42)
    assert result.split == "within_trial"
    assert result.layer == "late_fusion"
    assert result.auroc > 0.7  # informative layer still pulls signal up


def test_topk_layers_by_within_auroc_picks_best():
    from speech_decoding.whisper_ceiling.probe import ProbeResult as PR
    fake = [
        PR("v", 0, "within_trial", 0.60, 100, 25, 42),
        PR("v", 8, "within_trial", 0.85, 100, 25, 42),
        PR("v", 16, "within_trial", 0.72, 100, 25, 42),
        PR("v", 24, "within_trial", 0.80, 100, 25, 42),
        PR("v", 31, "within_trial", 0.50, 100, 25, 42),
        PR("v", 0, "cross_subject", 0.99, 100, 25, 42),  # wrong split — ignored
    ]
    top3 = topk_layers_by_within_auroc(fake, k=3)
    assert top3 == [8, 24, 16]


def test_fit_probe_cross_subject_loo():
    rng = np.random.default_rng(0)
    feats = {}
    labels = {}
    for sid in range(3):
        feats[sid] = np.concatenate([
            rng.normal(-2, 1, size=(40, 4)), rng.normal(+2, 1, size=(40, 4))
        ]).astype(np.float32)
        labels[sid] = np.array([0] * 40 + [1] * 40)
    result = fit_probe_cross_subject(
        feats, labels, held_out_subject=2, task="syn", layer=0,
    )
    assert result.split == "cross_subject"
    assert result.n_train == 160  # subjects 0 + 1
    assert result.n_test == 80    # subject 2
    assert result.auroc > 0.85
