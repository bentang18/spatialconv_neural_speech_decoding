"""Ground-truth phoneme fixture tests against `data/ps_tokens.csv` (#25).

The canonical 52-token PS manifest is the single source of truth for the
`event_id` assertion (`#18`), the per-trial 3-phoneme decomposition (`#19`,
`#29`), and the alphabetical ARPABET index (`#16`). Any drift between
`phoneme_map.py` and this manifest silently corrupts every downstream
label metric, so it is worth regressing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from speech_decoding.data.phoneme_map import (
    ARPA_PHONEMES,
    PS2ARPA,
    filter_to_ps_phonemes,
    normalize_label,
)


PS_TOKENS_CSV = Path(__file__).resolve().parents[2] / "data" / "ps_tokens.csv"


def _load_manifest() -> list[dict[str, str]]:
    with PS_TOKENS_CSV.open() as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def manifest() -> list[dict[str, str]]:
    if not PS_TOKENS_CSV.exists():
        pytest.skip(f"missing ps_tokens manifest: {PS_TOKENS_CSV}")
    rows = _load_manifest()
    assert len(rows) == 52, f"expected 52 PS tokens, found {len(rows)}"
    return rows


class TestManifestShape:
    def test_expected_columns_present(self, manifest: list[dict[str, str]]) -> None:
        required = {"token_id", "ps_notation", "ps_sequence", "arpabet_sequence", "structure"}
        assert required.issubset(set(manifest[0].keys()))

    def test_token_ids_are_0_to_51(self, manifest: list[dict[str, str]]) -> None:
        ids = sorted(int(row["token_id"]) for row in manifest)
        assert ids == list(range(52))

    def test_token_ps_notations_are_unique(self, manifest: list[dict[str, str]]) -> None:
        notations = [row["ps_notation"] for row in manifest]
        assert len(set(notations)) == 52

    def test_all_structures_are_cvc_or_vcv(self, manifest: list[dict[str, str]]) -> None:
        structures = {row["structure"] for row in manifest}
        assert structures.issubset({"CVC", "VCV"})


class TestPSSequenceDecomposition:
    def test_every_ps_sequence_has_three_tokens(self, manifest: list[dict[str, str]]) -> None:
        for row in manifest:
            parts = row["ps_sequence"].split()
            assert len(parts) == 3, f"{row['ps_notation']}: ps_sequence {row['ps_sequence']!r}"

    def test_every_ps_token_is_in_ps2arpa(self, manifest: list[dict[str, str]]) -> None:
        for row in manifest:
            for ps_tok in row["ps_sequence"].split():
                assert ps_tok in PS2ARPA, (
                    f"{row['ps_notation']}: unknown PS token {ps_tok!r}; "
                    f"PS2ARPA keys = {sorted(PS2ARPA)}"
                )

    def test_ps_to_arpa_matches_manifest_arpabet_sequence(
        self, manifest: list[dict[str, str]]
    ) -> None:
        """Decomposing ps_sequence via PS2ARPA must equal arpabet_sequence."""
        for row in manifest:
            ps_parts = row["ps_sequence"].split()
            arpa_parts = row["arpabet_sequence"].split()
            derived = [PS2ARPA[p] for p in ps_parts]
            assert derived == arpa_parts, (
                f"{row['ps_notation']}: PS2ARPA gave {derived}, "
                f"manifest says {arpa_parts}"
            )


class TestARPABETInventory:
    def test_every_arpabet_in_manifest_is_one_of_the_nine(
        self, manifest: list[dict[str, str]]
    ) -> None:
        seen: set[str] = set()
        for row in manifest:
            seen.update(row["arpabet_sequence"].split())
        # Every arpabet appearing in the manifest must be in ARPA_PHONEMES.
        assert seen.issubset(set(ARPA_PHONEMES)), (
            f"manifest has out-of-inventory phonemes: {sorted(seen - set(ARPA_PHONEMES))}"
        )

    def test_manifest_covers_all_nine_phonemes(
        self, manifest: list[dict[str, str]]
    ) -> None:
        """Every one of the 9 PS phonemes appears somewhere in the 52 tokens."""
        seen: set[str] = set()
        for row in manifest:
            seen.update(row["arpabet_sequence"].split())
        assert seen == set(ARPA_PHONEMES), (
            f"missing phonemes in manifest: {sorted(set(ARPA_PHONEMES) - seen)}"
        )


class TestNormalizeLabelRoundTrip:
    def test_every_arpabet_in_manifest_normalizes_to_itself(
        self, manifest: list[dict[str, str]]
    ) -> None:
        for row in manifest:
            for arpa_tok in row["arpabet_sequence"].split():
                assert normalize_label(arpa_tok) == arpa_tok, (
                    f"{row['ps_notation']}: {arpa_tok!r} did not round-trip"
                )

    def test_every_ps_token_normalizes_to_manifest_arpabet(
        self, manifest: list[dict[str, str]]
    ) -> None:
        for row in manifest:
            ps_parts = row["ps_sequence"].split()
            arpa_parts = row["arpabet_sequence"].split()
            for ps_tok, arpa_tok in zip(ps_parts, arpa_parts):
                assert normalize_label(ps_tok) == arpa_tok


class TestFilterToPSPhonemes:
    def test_every_manifest_arpabet_is_kept_by_filter(
        self, manifest: list[dict[str, str]]
    ) -> None:
        all_arpa = [tok for row in manifest for tok in row["arpabet_sequence"].split()]
        mask = filter_to_ps_phonemes(all_arpa)
        assert all(mask), "filter_to_ps_phonemes dropped a manifest phoneme"

    def test_every_manifest_ps_token_is_kept_by_filter(
        self, manifest: list[dict[str, str]]
    ) -> None:
        all_ps = [tok for row in manifest for tok in row["ps_sequence"].split()]
        mask = filter_to_ps_phonemes(all_ps)
        assert all(mask), "filter_to_ps_phonemes dropped a manifest PS token"
