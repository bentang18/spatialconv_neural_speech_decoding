"""E2E CPU integration smoke for the per-phoneme training loop (plan P5).

Trains the per-phoneme model on a tiny fake `.fif`, asserts val per-phoneme
PER drops below chance (8/9 ≈ 0.889) within a handful of epochs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from speech_decoding.v14.eval import evaluate_per_phoneme
from speech_decoding.v14.phoneme_dataset import (
    V14PhonemeDataset,
    collate_v14_phoneme_batch,
)
from speech_decoding.v14.phoneme_model import NeuralFieldPerceiverPerPhoneme
from speech_decoding.v14.train import per_phoneme_ce_loss, train_one_fold

from tests.v14.test_phoneme_dataset import _setup_patient_fixture


def test_phoneme_loop_drops_below_chance_on_tiny_fake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(0)
    fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
    ds = V14PhonemeDataset(
        "SXX", fif,
        support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
    )

    loader = DataLoader(
        ds, batch_size=6, shuffle=False, collate_fn=collate_v14_phoneme_batch,
    )
    # Val = train (overfit check).
    val_loader = DataLoader(
        ds, batch_size=6, shuffle=False, collate_fn=collate_v14_phoneme_batch,
    )

    model = NeuralFieldPerceiverPerPhoneme()

    result = train_one_fold(
        model,
        loader,
        val_loader,
        max_epochs=40,
        val_every=5,
        patience=100,
        loss_fn=per_phoneme_ce_loss,
        evaluate_fn=evaluate_per_phoneme,
        warmup_epochs=2,
    )

    # Chance for per-phoneme PER = 8/9 ≈ 0.889. Overfitting 6 samples must
    # comfortably beat that.
    assert result.best_val_per < 0.7, f"val_per={result.best_val_per}"
