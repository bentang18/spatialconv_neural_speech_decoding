from speech_decoding.v14.token_spec import (
    DEFAULT_BASE_PARCELS,
    DEFAULT_SPLIT_COUNTS,
    PROVISIONAL_TOKEN_SPEC,
    default_token_count,
)


def test_default_split_map_matches_base_parcels() -> None:
    assert set(DEFAULT_SPLIT_COUNTS) == set(DEFAULT_BASE_PARCELS)


def test_default_token_count_is_positive() -> None:
    # The exact count is pending `docs/implementation_tasks.md` #4 re-derivation.
    # Only the structural invariant (sum of split counts > 0) is asserted here.
    assert default_token_count() > 0


def test_token_spec_is_still_provisional() -> None:
    # #4 requires the provisional flag to be explicitly cleared by the
    # re-derivation before any v14 code path is allowed to consume these
    # constants. This test fails loudly the moment someone flips it without
    # updating the rest of the contract.
    assert PROVISIONAL_TOKEN_SPEC is True
