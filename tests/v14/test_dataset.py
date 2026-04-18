"""Unit tests for the v14-core dataset contract (plan Task B1).

Behavior-only. No Box mount, no real `.fif`, no support-cache materialization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from speech_decoding.v14.dataset import (
    N_PS_TOKENS,
    N_TIER1_PARCELS,
    T_SAMPLES,
    V14Sample,
    V14TrialDataset,
    build_sample_from_epoch,
    decompose_token_to_slot_indices,
)

from tests.v14.fixtures.fake_fif import build_fake_epochs


def _synthetic_metadata(n_e: int = 16) -> dict[str, object]:
    """Per-electrode metadata matching the fake-fif `N_e = 16` grid."""

    grid_shape = (8, 16)
    layout = np.array([(i // 4, (i % 4)) for i in range(n_e)], dtype=np.int64)
    active = np.ones(n_e, dtype=bool)
    support = np.zeros((n_e, N_TIER1_PARCELS), dtype=np.float32)
    support[0, 0] = 42.0
    return dict(
        electrode_grid_layout=layout,
        electrode_grid_shape=grid_shape,
        electrode_active_mask=active,
        support=support,
    )


# -----------------------------------------------------------------------------
# decompose_token_to_slot_indices


class TestDecompose:
    def test_bak_maps_to_2_0_5(self) -> None:
        assert decompose_token_to_slot_indices("bak") == [2, 0, 5]

    def test_gup_maps_to_3_7_6(self) -> None:
        assert decompose_token_to_slot_indices("gup") == [3, 7, 6]

    def test_ae_digraph_resolves_to_ae(self) -> None:
        # 'aeba' = ae(1) + b(2) + a(0)
        assert decompose_token_to_slot_indices("aeba") == [1, 2, 0]

    def test_rejects_4_phoneme_token(self) -> None:
        with pytest.raises(ValueError, match="expected 3"):
            decompose_token_to_slot_indices("baka")

    def test_rejects_unknown_ps_symbol(self) -> None:
        with pytest.raises(ValueError, match="unknown PS symbol"):
            decompose_token_to_slot_indices("bzk")


# -----------------------------------------------------------------------------
# build_sample_from_epoch / V14Sample contract


class TestBuildSample:
    def test_build_bak_and_gup_labels_and_dtypes(self) -> None:
        meta = _synthetic_metadata()
        sig = np.random.default_rng(0).standard_normal((16, T_SAMPLES)).astype(np.float32)
        s_bak = build_sample_from_epoch(
            signal=sig, token="bak", patient_id="SXX", **meta  # type: ignore[arg-type]
        )
        s_gup = build_sample_from_epoch(
            signal=sig, token="gup", patient_id="SXX", **meta  # type: ignore[arg-type]
        )
        assert isinstance(s_bak, V14Sample)
        assert torch.equal(s_bak.label, torch.tensor([2, 0, 5], dtype=torch.long))
        assert torch.equal(s_gup.label, torch.tensor([3, 7, 6], dtype=torch.long))
        assert s_bak.signal.dtype == torch.float32
        assert s_bak.signal.shape == (16, T_SAMPLES)
        assert s_bak.support.dtype == torch.float32
        assert s_bak.support.shape == (16, N_TIER1_PARCELS)
        assert s_bak.electrode_grid_layout.dtype == torch.long
        assert s_bak.electrode_active_mask.dtype == torch.bool
        assert s_bak.patient_id == "SXX"
        assert s_bak.electrode_grid_shape == (8, 16)

    def test_rejects_wrong_T(self) -> None:
        meta = _synthetic_metadata()
        sig = np.zeros((16, T_SAMPLES - 1), dtype=np.float32)
        with pytest.raises(ValueError, match="signal shape"):
            build_sample_from_epoch(
                signal=sig, token="bak", patient_id="SXX", **meta  # type: ignore[arg-type]
            )

    def test_rejects_layout_out_of_grid(self) -> None:
        meta = _synthetic_metadata()
        bad_layout = np.zeros((16, 2), dtype=np.int64)
        bad_layout[0, 0] = 99  # out of H_p=8
        meta["electrode_grid_layout"] = bad_layout
        sig = np.zeros((16, T_SAMPLES), dtype=np.float32)
        with pytest.raises(ValueError, match="outside grid"):
            build_sample_from_epoch(
                signal=sig, token="bak", patient_id="SXX", **meta  # type: ignore[arg-type]
            )

    def test_layout_rows_are_within_grid(self) -> None:
        meta = _synthetic_metadata()
        sig = np.zeros((16, T_SAMPLES), dtype=np.float32)
        s = build_sample_from_epoch(
            signal=sig, token="bak", patient_id="SXX", **meta  # type: ignore[arg-type]
        )
        H_p, W_p = s.electrode_grid_shape
        rows = s.electrode_grid_layout[:, 0]
        cols = s.electrode_grid_layout[:, 1]
        assert int(rows.min()) >= 0 and int(rows.max()) < H_p
        assert int(cols.min()) >= 0 and int(cols.max()) < W_p


# -----------------------------------------------------------------------------
# Fake-fif integration


class TestFakeFixture:
    def test_fake_epochs_have_52_event_ids_and_300_samples(self) -> None:
        epochs = build_fake_epochs()
        assert len(epochs.event_id) == N_PS_TOKENS
        # tmin=-0.5 inclusive, tmax=1.0 half-open, 300 samples, sfreq=200
        assert epochs.times.size == T_SAMPLES
        assert epochs.info["sfreq"] == 200.0
        # Two trials with tokens bak, gup
        id_to_tok = {c: t for t, c in epochs.event_id.items()}
        tokens = [id_to_tok[int(c)] for c in epochs.events[:, 2]]
        # "gup" in the plan doesn't exist in the real 52-token manifest;
        # use "gub" (G UW B -> [3, 7, 2]) instead for the round-trip test.
        assert tokens == ["bak", "gub"]

    def test_fake_fif_round_trip_through_trialdataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Write the fake epochs to a `.fif`, wire a synthetic support cache,
        and check `V14TrialDataset.__getitem__` returns the expected labels."""

        import scipy.io

        from speech_decoding.v14.channel_map import GRID_SHAPES, PATIENT_KIND
        from speech_decoding.v14.support_cache import write_support_cache

        patient_id = "SXX"
        monkeypatch.setitem(PATIENT_KIND, patient_id, "map4_128")
        monkeypatch.setitem(GRID_SHAPES, patient_id, (8, 16))

        # Build fake epochs and save to .fif
        epochs = build_fake_epochs(n_electrodes=16)
        fif_path = tmp_path / "fake-epo.fif"
        epochs.save(str(fif_path), overwrite=True, verbose="ERROR")

        # Fake 8x16 chanMap: identity permutation 1..128. Our 16 channels are "1".."16"
        channel_maps_dir = tmp_path / "cmaps"
        channel_maps_dir.mkdir()
        chanmap = np.arange(1, 129, dtype=np.uint8).reshape(8, 16)
        scipy.io.savemat(
            str(channel_maps_dir / f"{patient_id}_channelMap.mat"), {"chanMap": chanmap}
        )

        # Fake electrodeNames file with 128 entries "EL1".."EL128"
        box_root = tmp_path / "box"
        elec_recon = box_root / "ECoG_Recon" / patient_id / "elec_recon"
        elec_recon.mkdir(parents=True)
        lines = [f"Subject {patient_id}", "# header"]
        for i in range(1, 129):
            lines.append(f"EL{i} G L")
        (elec_recon / f"{patient_id}.electrodeNames").write_text("\n".join(lines) + "\n")

        # For channels "1".."16", phys_num = 1..16, so phys_name = "EL1".."EL16"
        phys_names = tuple(f"EL{i}" for i in range(1, 17))
        support = np.zeros((16, N_TIER1_PARCELS), dtype=np.float32)
        support[0, 0] = 42.0
        support_cache_path = tmp_path / f"{patient_id}_support_tier1.csv"
        write_support_cache(support_cache_path, phys_names, support)

        ds = V14TrialDataset(
            patient_id,
            fif_path,
            support_cache_path=support_cache_path,
            channel_maps_dir=channel_maps_dir,
            box_root=box_root,
        )
        assert len(ds) == 2
        s0 = ds[0]
        s1 = ds[1]
        assert torch.equal(s0.label, torch.tensor([2, 0, 5], dtype=torch.long))
        # "gub" = G UW B -> [3, 7, 2] (see fixture note about "gup" vs "gub")
        assert torch.equal(s1.label, torch.tensor([3, 7, 2], dtype=torch.long))
        assert s0.signal.shape == (16, T_SAMPLES)
        assert s0.support.shape == (16, N_TIER1_PARCELS)
        assert s0.electrode_grid_shape == (8, 16)
        # Row 0 of synthetic support survived the round-trip
        assert float(s0.support[0, 0]) == pytest.approx(42.0)
