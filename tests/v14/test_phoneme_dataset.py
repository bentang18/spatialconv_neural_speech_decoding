"""Tests for V14PhonemeDataset + collator (plan P3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

from speech_decoding.v14.phoneme_dataset import (
    BOS_TOKEN,
    N_TIER1_PARCELS,
    T_RAW_SAMPLES,
    V14PhonemeDataset,
    V14PhonemeSample,
    collate_v14_phoneme_batch,
)

from tests.v14.fixtures.fake_phoneme_fif import build_fake_phoneme_epochs


def _setup_patient_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patient_id: str = "SXX"
) -> tuple[Path, Path, Path, Path]:
    """Wire up a fake 128-strip patient. Returns
    (fif_path, channel_maps_dir, box_root, support_cache_path)."""

    from speech_decoding.v14.channel_map import GRID_SHAPES, PATIENT_KIND
    from speech_decoding.v14.support_cache import write_support_cache

    monkeypatch.setitem(PATIENT_KIND, patient_id, "map4_128")
    monkeypatch.setitem(GRID_SHAPES, patient_id, (8, 16))

    epochs = build_fake_phoneme_epochs(n_electrodes=16)
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
    support = np.zeros((16, N_TIER1_PARCELS), dtype=np.float32)
    support[0, 0] = 0.77  # marker value
    support_cache_path = tmp_path / f"{patient_id}_support_tier1.csv"
    write_support_cache(support_cache_path, phys_names, support)

    return fif_path, channel_maps_dir, box_root, support_cache_path


class TestPhonemeDataset:
    def test_len_is_three_times_trials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
        )
        # 2 tokens -> 6 phoneme epochs, 2 trials
        assert len(ds) == 6
        assert ds.n_trials == 2

    def test_signal_shape_after_crop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
        )
        s = ds[0]
        # 16 electrodes (kept), cropped to 130 samples
        assert s.signal.shape == (16, T_RAW_SAMPLES)
        assert s.signal.dtype == torch.float32

    def test_labels_map_to_alphabetical_arpa_indices(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
        )
        # Tokens: bak, gub
        # ARPA indices: b=2, a=0, k=5; g=3, u=7, b=2
        expected = [2, 0, 5, 3, 7, 2]
        got = [int(ds[i].label) for i in range(6)]
        assert got == expected

    def test_trial_id_and_phoneme_pos_layout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
        )
        for i in range(6):
            s = ds[i]
            assert s.trial_id == i // 3
            assert s.phoneme_pos == i % 3

    def test_prev_phoneme_bos_at_pos_zero_else_previous_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
        )
        # bak → [2, 0, 5], gub → [3, 7, 2]
        # prev: BOS, 2, 0, BOS, 3, 7
        expected_prev = [BOS_TOKEN, 2, 0, BOS_TOKEN, 3, 7]
        got = [int(ds[i].prev_phoneme) for i in range(6)]
        assert got == expected_prev

    def test_support_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
        )
        s = ds[0]
        assert s.support.shape == (16, N_TIER1_PARCELS)
        assert float(s.support[0, 0]) == pytest.approx(0.77)

    def test_xyz_none_when_cache_dir_not_provided(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
        )
        assert ds[0].electrode_xyz is None

    def test_xyz_loaded_from_cache_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        # Write a minimal fsaverage coord cache for the 16 kept electrodes.
        coords_dir = tmp_path / "fsaverage_coords"
        coords_dir.mkdir()
        with (coords_dir / "SXX_fsaverage_pial.csv").open("w") as f:
            f.write("name,hemisphere,fsaverage_vertex,x,y,z\n")
            for i in range(1, 17):
                f.write(f"EL{i},L,{i},{i}.0,{i + 0.5},{i + 1.0}\n")
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
            fsaverage_coords_dir=coords_dir,
        )
        s = ds[0]
        assert s.electrode_xyz is not None
        assert s.electrode_xyz.shape == (16, 3)
        assert s.electrode_xyz.dtype == torch.float32
        # Row 0 is EL1 → (1.0, 1.5, 2.0)
        assert float(s.electrode_xyz[0, 0]) == pytest.approx(1.0)
        assert float(s.electrode_xyz[0, 1]) == pytest.approx(1.5)
        assert float(s.electrode_xyz[0, 2]) == pytest.approx(2.0)

    def test_xyz_collates_into_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        coords_dir = tmp_path / "fsaverage_coords"
        coords_dir.mkdir()
        with (coords_dir / "SXX_fsaverage_pial.csv").open("w") as f:
            f.write("name,hemisphere,fsaverage_vertex,x,y,z\n")
            for i in range(1, 17):
                f.write(f"EL{i},L,{i},{float(i)},{float(i)},{float(i)}\n")
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
            fsaverage_coords_dir=coords_dir,
        )
        samples = [ds[i] for i in range(3)]
        batch = collate_v14_phoneme_batch(samples)
        assert "electrode_xyz" in batch
        assert batch["electrode_xyz"].shape == (3, 16, 3)  # type: ignore[union-attr]


class TestCollator:
    def _make_samples(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, k: int = 4
    ) -> list[V14PhonemeSample]:
        fif, cmap, box, sup = _setup_patient_fixture(tmp_path, monkeypatch)
        ds = V14PhonemeDataset(
            "SXX", fif,
            support_cache_path=sup, channel_maps_dir=cmap, box_root=box,
        )
        return [ds[i] for i in range(k)]

    def test_batch_keys_and_shapes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = self._make_samples(tmp_path, monkeypatch, k=4)
        batch = collate_v14_phoneme_batch(samples)
        assert batch["signal"].shape == (4, 16, T_RAW_SAMPLES)
        assert batch["labels"].shape == (4,)
        assert batch["prev_tokens"].shape == (4,)
        assert batch["phoneme_pos"].shape == (4,)
        assert batch["trial_id"].shape == (4,)
        assert batch["electrode_grid_layout"].shape == (4, 16, 2)
        assert batch["electrode_active_mask"].shape == (4, 16)
        assert batch["support"].shape == (4, 16, N_TIER1_PARCELS)
        assert batch["electrode_grid_shape"] == (8, 16)
        assert batch["patient_id"] == "SXX"

    def test_rejects_mixed_patient(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = self._make_samples(tmp_path, monkeypatch, k=2)
        # Fabricate a second sample with a different patient id
        samples[1] = V14PhonemeSample(
            signal=samples[1].signal,
            patient_id="SYY",
            label=samples[1].label,
            prev_phoneme=samples[1].prev_phoneme,
            trial_id=samples[1].trial_id,
            phoneme_pos=samples[1].phoneme_pos,
            electrode_grid_layout=samples[1].electrode_grid_layout,
            electrode_grid_shape=samples[1].electrode_grid_shape,
            electrode_active_mask=samples[1].electrode_active_mask,
            support=samples[1].support,
            electrode_xyz=samples[1].electrode_xyz,
        )
        with pytest.raises(ValueError, match="patient_id"):
            collate_v14_phoneme_batch(samples)

    def test_empty_batch_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty batch"):
            collate_v14_phoneme_batch([])
