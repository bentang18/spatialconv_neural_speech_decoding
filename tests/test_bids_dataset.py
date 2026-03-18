"""Tests for BIDS dataset loading."""
import numpy as np
import pytest
from pathlib import Path

from speech_decoding.data.bids_dataset import BIDSDataset, load_patient_data
from speech_decoding.data.phoneme_map import ARPA_PHONEMES


# ── Synthetic tests (no real data needed) ──────────────────────────

class TestBIDSDatasetSynthetic:
    """Test dataset interface with mock data."""

    def test_dataset_len(self):
        """Dataset length = number of trials."""
        n_trials = 10
        ds = BIDSDataset(
            grid_data=np.random.randn(n_trials, 8, 16, 300).astype(np.float32),
            ctc_labels=[[1, 2, 3]] * n_trials,
            patient_id="S14",
            grid_shape=(8, 16),
        )
        assert len(ds) == n_trials

    def test_dataset_getitem(self):
        """__getitem__ returns (grid_data, ctc_label, patient_id)."""
        ds = BIDSDataset(
            grid_data=np.random.randn(5, 8, 16, 300).astype(np.float32),
            ctc_labels=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5], [2, 4, 6]],
            patient_id="S14",
            grid_shape=(8, 16),
        )
        x, y, pid = ds[0]
        assert x.shape == (8, 16, 300)
        assert y == [1, 2, 3]
        assert pid == "S14"

    def test_dataset_grid_shape(self):
        ds = BIDSDataset(
            grid_data=np.random.randn(3, 12, 22, 300).astype(np.float32),
            ctc_labels=[[1, 2, 3]] * 3,
            patient_id="S33",
            grid_shape=(12, 22),
        )
        assert ds.grid_shape == (12, 22)


# ── Real data tests (slow, need BIDS data) ────────────────────────

PS_ROOT = Path(
    "BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
)
LEX_ROOT = Path(
    "BIDS_1.0_Lexical_µECoG/BIDS_1.0_Lexical_µECoG/BIDS"
)


@pytest.mark.slow
class TestLoadPatientDataPS:
    """Test loading real PS patient data."""

    def test_load_s14_basic(self):
        if not PS_ROOT.exists():
            pytest.skip("PS BIDS data not available")
        ds = load_patient_data("S14", PS_ROOT, task="PhonemeSequence", n_phons=3)
        assert len(ds) > 0
        x, y, pid = ds[0]
        assert pid == "S14"
        assert len(y) == 3  # 3-phoneme CTC label
        assert all(1 <= idx <= 9 for idx in y)  # valid phoneme indices

    def test_s14_grid_shape(self):
        if not PS_ROOT.exists():
            pytest.skip("PS BIDS data not available")
        ds = load_patient_data("S14", PS_ROOT, task="PhonemeSequence", n_phons=3)
        assert ds.grid_shape == (8, 16)
        x, _, _ = ds[0]
        assert x.shape[0] == 8
        assert x.shape[1] == 16

    def test_s14_time_crop(self):
        """Default crop [-0.5, 1.0] → T=300 at 200Hz."""
        if not PS_ROOT.exists():
            pytest.skip("PS BIDS data not available")
        ds = load_patient_data(
            "S14", PS_ROOT, task="PhonemeSequence", n_phons=3,
            tmin=-0.5, tmax=1.0,
        )
        x, _, _ = ds[0]
        # T should be 300 (1.5s at 200Hz) ± 1 for rounding
        assert 299 <= x.shape[2] <= 301

    def test_s14_labels_are_ps_phonemes(self):
        """All CTC labels should be indices of PS phonemes."""
        if not PS_ROOT.exists():
            pytest.skip("PS BIDS data not available")
        ds = load_patient_data("S14", PS_ROOT, task="PhonemeSequence", n_phons=3)
        for i in range(len(ds)):
            _, y, _ = ds[i]
            assert all(1 <= idx <= 9 for idx in y), f"Bad label at trial {i}: {y}"

    def test_s33_256ch_grid(self):
        """256ch patient should have 12x22 grid."""
        if not PS_ROOT.exists():
            pytest.skip("PS BIDS data not available")
        ds = load_patient_data("S33", PS_ROOT, task="PhonemeSequence", n_phons=3)
        assert ds.grid_shape == (12, 22)


@pytest.mark.slow
class TestLoadPatientDataLexical:
    """Test loading real Lexical patient data.

    NOTE: Lexical 5-phoneme words never have ALL phonemes from the PS set,
    so filter_ps_only=True yields 0 trials. Cross-task pooling is a future
    direction — for now we only verify the loading mechanics work.
    """

    def test_load_s41_ps_filter_yields_zero(self):
        """Lexical words don't contain only PS phonemes → 0 trials after filter."""
        if not LEX_ROOT.exists():
            pytest.skip("Lexical BIDS data not available")
        ds = load_patient_data(
            "S41", LEX_ROOT, task="lexical", n_phons=5,
            filter_ps_only=True,
        )
        assert len(ds) == 0  # expected: no all-PS trials exist
