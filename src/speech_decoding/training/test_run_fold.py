"""Runner test for per-phoneme fold expansion (plan P6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from speech_decoding.studies.cogan_ps.dataset import V14PhonemeDataset
from speech_decoding.training.run_fold import (
    _expand_trial_indices_to_phoneme_indices,
    run_one_fold,
)


class TestExpandTrialIndices:
    def test_expands_each_trial_to_three_phonemes(self) -> None:
        out = _expand_trial_indices_to_phoneme_indices([0, 2])
        assert out == [0, 1, 2, 6, 7, 8]

    def test_empty(self) -> None:
        assert _expand_trial_indices_to_phoneme_indices([]) == []


class TestRunOneFold:
    def test_produces_result_with_both_per_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Smoke: the runner writes a result JSON containing per-phoneme PER
        and exhaustive-AR slot-averaged PER. Requires ≥5 unique tokens for CV."""
        # 5 tokens → 5-fold CV is possible (each fold holds out one token).
        # We need the fake fixture to produce these tokens.
        from speech_decoding.studies.cogan_ps.fixtures.fake_phoneme_fif import build_fake_phoneme_epochs
        from speech_decoding.studies.cogan_ps.channels import GRID_SHAPES, PATIENT_KIND
        from speech_decoding.atlas.support import write_support_cache
        import scipy.io
        import numpy as np

        patient_id = "SXX"
        monkeypatch.setitem(PATIENT_KIND, patient_id, "map4_128")
        monkeypatch.setitem(GRID_SHAPES, patient_id, (8, 16))

        tokens = ("bak", "gub", "pik", "vap", "pav", "gap", "kab")
        epochs = build_fake_phoneme_epochs(n_electrodes=16, tokens=tokens)
        fif_path = tmp_path / "fake-phon-epo.fif"
        epochs.save(str(fif_path), overwrite=True, verbose="ERROR")

        channel_maps_dir = tmp_path / "cmaps"
        channel_maps_dir.mkdir()
        chanmap = np.arange(1, 129, dtype=np.uint8).reshape(8, 16)
        scipy.io.savemat(
            str(channel_maps_dir / f"{patient_id}_channelMap.mat"),
            {"chanMap": chanmap},
        )

        box_root = tmp_path / "box"
        elec_recon = box_root / "ECoG_Recon" / patient_id / "elec_recon"
        elec_recon.mkdir(parents=True)
        lines = [f"Subject {patient_id}", "# header"]
        for i in range(1, 129):
            lines.append(f"EL{i} G L")
        (elec_recon / f"{patient_id}.electrodeNames").write_text(
            "\n".join(lines) + "\n"
        )

        phys_names = tuple(f"EL{i}" for i in range(1, 17))
        support = np.zeros((16, 15), dtype=np.float32)
        support[0, 0] = 1.0
        support_cache_path = tmp_path / f"{patient_id}_support_tier1.csv"
        write_support_cache(support_cache_path, phys_names, support)

        ds = V14PhonemeDataset(
            patient_id, fif_path,
            support_cache_path=support_cache_path,
            channel_maps_dir=channel_maps_dir, box_root=box_root,
        )
        # 7 tokens → ensures 5 outer folds each have at least 1 test token.
        assert len(ds.trial_tokens()) == 7

        out_dir = tmp_path / "out"
        result = run_one_fold(
            ds,
            fold_idx=0,
            seed=0,
            backbone_depth=1,
            out_dir=out_dir,
            patient_id=patient_id,
            max_epochs=4,
            val_every=2,
            patience=100,
            warmup_epochs=1,
        )

        assert "test_per_phoneme" in result
        assert "test_slot_averaged_per" in result
        assert "best_val_per_phoneme" in result
        assert 0.0 <= result["test_per_phoneme"] <= 1.0
        assert 0.0 <= result["test_slot_averaged_per"] <= 1.0
