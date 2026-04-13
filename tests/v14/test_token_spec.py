from speech_decoding.v14.token_spec import DEFAULT_BASE_PARCELS, DEFAULT_SPLIT_COUNTS, default_token_count


def test_default_split_map_matches_base_parcels() -> None:
    assert set(DEFAULT_SPLIT_COUNTS) == set(DEFAULT_BASE_PARCELS)


def test_default_token_count_is_21() -> None:
    assert default_token_count() == 21
