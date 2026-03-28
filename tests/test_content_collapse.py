"""Tests for content-collapse diagnostic metrics."""
import pytest
import numpy as np

from speech_decoding.evaluation.content_collapse import (
    output_entropy,
    possible_sequence_coverage,
    unigram_kl,
    stereotypy_index,
    content_collapse_report,
)


class TestOutputEntropy:
    def test_uniform_distribution(self):
        """Uniform predictions → max entropy = log2(9) ≈ 3.17."""
        preds = np.array([list(range(1, 10))] * 10).flatten()
        h = output_entropy(preds, n_classes=9)
        assert h == pytest.approx(np.log2(9), abs=0.01)

    def test_collapsed_distribution(self):
        """All same prediction → entropy = 0."""
        preds = np.array([1] * 100)
        h = output_entropy(preds, n_classes=9)
        assert h == pytest.approx(0.0, abs=1e-6)

    def test_partial_collapse(self):
        """Concentrated on 2 phonemes → entropy ≈ 1.0."""
        preds = np.array([1] * 50 + [2] * 50)
        h = output_entropy(preds, n_classes=9)
        assert h == pytest.approx(1.0, abs=0.01)


class TestUnigramKL:
    def test_uniform_gives_zero(self):
        """Uniform predictions → KL from uniform = 0."""
        preds = np.array(list(range(1, 10)) * 10)
        kl = unigram_kl(preds, n_classes=9)
        assert kl == pytest.approx(0.0, abs=0.01)

    def test_concentrated_gives_high_kl(self):
        """All-same predictions → high KL."""
        preds = np.array([1] * 90)
        kl = unigram_kl(preds, n_classes=9)
        assert kl > 2.0  # KL(delta || uniform) = log(9) ≈ 2.2 nats


class TestStereotypyIndex:
    def test_all_unique_sequences(self):
        """All unique 3-phoneme sequences → index = 1.0."""
        sequences = [[i, j, k] for i in range(1, 4) for j in range(1, 4) for k in range(1, 4)]
        idx = stereotypy_index(sequences)
        assert idx == pytest.approx(1.0)

    def test_all_same_sequence(self):
        """All identical sequences → index = 1/N."""
        sequences = [[1, 2, 3]] * 100
        idx = stereotypy_index(sequences)
        assert idx == pytest.approx(0.01)

    def test_alarm_threshold(self):
        """< 10% of 729 possible = alarm (RD-78)."""
        sequences = [[i, 1, 1] for i in range(1, 6)] * 20
        coverage = possible_sequence_coverage(sequences, n_classes=9)
        assert coverage < 0.10


class TestContentCollapseReport:
    def test_report_structure(self):
        preds_per_position = [
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 3),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 3),
            np.array([1, 2, 3, 4, 5, 6, 7, 8, 9] * 3),
        ]
        sequences = [[1, 2, 3]] * 27
        report = content_collapse_report(preds_per_position, sequences, n_classes=9)
        assert "entropy" in report
        assert "unigram_kl" in report
        assert "stereotypy_index" in report
        assert "collapsed" in report
        assert len(report["entropy"]) == 3
