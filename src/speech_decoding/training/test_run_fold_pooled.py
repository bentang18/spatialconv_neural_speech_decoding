"""Smoke test for the pooled per-phoneme fold runner (Q1a).

Mirrors `test_phoneme_run_fold` but builds two small fake-patient datasets
and exercises the pooled runner end-to-end: pooled CV, round-robin train
loader, pooled val eval, per-patient test eval + population stats.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from speech_decoding.studies.cogan_ps.channels import GRID_SHAPES, PATIENT_KIND
from speech_decoding.studies.cogan_ps.dataset import V14PhonemeDataset
from speech_decoding.training.run_fold_pooled import (
    _RoundRobinInterleaveLoader,
    run_one_fold_pooled,
)
from speech_decoding.atlas.support import write_support_cache
from speech_decoding.studies.cogan_ps.fixtures.fake_phoneme_fif import build_fake_phoneme_epochs


def _register_fake_patient(monkeypatch: pytest.MonkeyPatch, patient_id: str) -> None:
    monkeypatch.setitem(PATIENT_KIND, patient_id, "map4_128")
    monkeypatch.setitem(GRID_SHAPES, patient_id, (8, 16))


def _build_fake_dataset(
    patient_id: str,
    root: Path,
    tokens: tuple[str, ...],
) -> V14PhonemeDataset:
    root.mkdir(parents=True, exist_ok=True)
    epochs = build_fake_phoneme_epochs(n_electrodes=16, tokens=tokens)
    fif_path = root / f"{patient_id}-phon-epo.fif"
    epochs.save(str(fif_path), overwrite=True, verbose="ERROR")

    channel_maps_dir = root / "cmaps"
    channel_maps_dir.mkdir(exist_ok=True)
    chanmap = np.arange(1, 129, dtype=np.uint8).reshape(8, 16)
    scipy.io.savemat(
        str(channel_maps_dir / f"{patient_id}_channelMap.mat"),
        {"chanMap": chanmap},
    )

    box_root = root / "box"
    elec_recon = box_root / "ECoG_Recon" / patient_id / "elec_recon"
    elec_recon.mkdir(parents=True, exist_ok=True)
    lines = [f"Subject {patient_id}", "# header"]
    for i in range(1, 129):
        lines.append(f"EL{i} G L")
    (elec_recon / f"{patient_id}.electrodeNames").write_text("\n".join(lines) + "\n")

    phys_names = tuple(f"EL{i}" for i in range(1, 17))
    support = np.zeros((16, 15), dtype=np.float32)
    support[0, 0] = 1.0
    support_cache_path = root / f"{patient_id}_support_tier1.csv"
    write_support_cache(support_cache_path, phys_names, support)

    return V14PhonemeDataset(
        patient_id,
        fif_path,
        support_cache_path=support_cache_path,
        channel_maps_dir=channel_maps_dir,
        box_root=box_root,
    )


class TestRoundRobinInterleaveLoader:
    def test_cycles_per_patient_per_step(self) -> None:
        batches_a = [{"patient_id": "A", "k": i} for i in range(3)]
        batches_b = [{"patient_id": "B", "k": i} for i in range(5)]

        class _FixedLoader:
            def __init__(self, batches: list[dict]) -> None:
                self._batches = batches

            def __iter__(self):
                return iter(self._batches)

            def __len__(self) -> int:
                return len(self._batches)

        loaders = {"A": _FixedLoader(batches_a), "B": _FixedLoader(batches_b)}
        rr = _RoundRobinInterleaveLoader(loaders)  # type: ignore[arg-type]
        assert len(rr) == 2 * 5  # n_patients * max_batches

        seen: list[tuple[str, int]] = []
        for b in rr:
            seen.append((b["patient_id"], b["k"]))
        patients = [p for p, _ in seen]
        # First two batches are A, B; then A, B etc. — perfect round-robin.
        for i in range(0, len(patients), 2):
            assert patients[i] == "A"
            assert patients[i + 1] == "B"


class TestRunOneFoldPooled:
    def test_pooled_smoke_writes_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register_fake_patient(monkeypatch, "SAA")
        _register_fake_patient(monkeypatch, "SBB")

        tokens = ("bak", "gub", "pik", "vap", "pav", "gap", "kab")
        ds_a = _build_fake_dataset("SAA", tmp_path / "a", tokens=tokens)
        # SBB has a different subset of tokens — still enough to CV-partition.
        ds_b = _build_fake_dataset("SBB", tmp_path / "b", tokens=tokens[:6])

        out_dir = tmp_path / "out"
        result = run_one_fold_pooled(
            {"SAA": ds_a, "SBB": ds_b},
            fold_idx=0,
            seed=0,
            backbone_depth=1,
            out_dir=out_dir,
            max_epochs=4,
            val_every=2,
            patience=100,
            warmup_epochs=1,
        )

        assert result["mode"] == "pooled"
        assert sorted(result["patients"]) == ["SAA", "SBB"]
        assert result["grad_accum_steps"] == 2
        assert set(result["per_patient_test"].keys()) == {"SAA", "SBB"}
        for pt_result in result["per_patient_test"].values():
            assert 0.0 <= pt_result["test_per_phoneme"] <= 1.0
            assert 0.0 <= pt_result["test_slot_averaged_per"] <= 1.0
        assert 0.0 <= result["pop_test_per_phoneme_mean"] <= 1.0
        assert result["pop_test_per_phoneme_std"] >= 0.0
