"""Tests for multi-patient collation."""
import numpy as np
import torch
import pytest

from speech_decoding.data.collate import collate_by_patient


def _make_dataset(patient_id, n_trials, grid_h, grid_w, T=300):
    """Create a list of (grid_data, ctc_label, patient_id) tuples."""
    samples = []
    for i in range(n_trials):
        x = np.random.randn(grid_h, grid_w, T).astype(np.float32)
        y = [1, 2, 3]
        samples.append((x, y, patient_id))
    return samples


class TestCollateByPatient:
    def test_single_patient(self):
        samples = _make_dataset("S14", 4, 8, 16)
        batches = collate_by_patient(samples)
        assert len(batches) == 1
        x, y, pid = batches["S14"]
        assert x.shape == (4, 8, 16, 300)
        assert isinstance(x, torch.Tensor)
        assert pid == "S14"

    def test_two_patients_different_grids(self):
        """Different grid sizes can't be stacked — collate groups by patient."""
        s1 = _make_dataset("S14", 3, 8, 16)
        s2 = _make_dataset("S33", 2, 12, 22)
        batches = collate_by_patient(s1 + s2)
        assert len(batches) == 2
        assert batches["S14"][0].shape == (3, 8, 16, 300)
        assert batches["S33"][0].shape == (2, 12, 22, 300)

    def test_labels_preserved(self):
        samples = _make_dataset("S14", 2, 8, 16)
        batches = collate_by_patient(samples)
        _, labels, _ = batches["S14"]
        assert len(labels) == 2
        assert labels[0] == [1, 2, 3]

    def test_dtype_is_float32(self):
        samples = _make_dataset("S14", 2, 8, 16)
        batches = collate_by_patient(samples)
        x, _, _ = batches["S14"]
        assert x.dtype == torch.float32
