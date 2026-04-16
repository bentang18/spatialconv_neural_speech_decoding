from speech_decoding.v14.token_spec import (
    DEFAULT_BASE_PARCELS,
    DEFAULT_SPLIT_COUNTS,
    PROVISIONAL_TOKEN_SPEC,
    assert_token_spec_frozen,
    default_token_count,
)


def test_default_base_parcels_frozen_fifteen() -> None:
    """The Phase-1 Tier-1 list is 15 LH parcels. See #4 in implementation_tasks.md.

    Selection rule: argmax_wins >= 10 on the frozen spatial pipeline
    (raw MNI + 8 mm PM dilation + σ=1.5 mm Gaussian). Primary auditory
    cortex TE1.0/1.2 (73) is the minimum-argmax entry at argmx=12.
    """
    assert len(DEFAULT_BASE_PARCELS) == 15
    assert all(isinstance(name, str) and isinstance(idx, int)
               for name, idx in DEFAULT_BASE_PARCELS)
    indices = [idx for _, idx in DEFAULT_BASE_PARCELS]
    assert all(1 <= idx <= 246 for idx in indices)
    assert len(set(indices)) == 15  # no duplicate BNA indices


def test_default_split_map_matches_base_parcels() -> None:
    names = {name for name, _ in DEFAULT_BASE_PARCELS}
    assert set(DEFAULT_SPLIT_COUNTS) == names


def test_default_split_map_is_uniform_k1_in_phase1() -> None:
    """Phase 1 uses k_parcel = 1 uniform. 2-token splits are a deferred ablation."""
    assert all(k == 1 for k in DEFAULT_SPLIT_COUNTS.values())


def test_default_token_count_equals_fifteen() -> None:
    assert default_token_count() == 15


def test_token_spec_is_frozen() -> None:
    """Frozen 2026-04-14 by #4. Flipping back to provisional must be a conscious act."""
    assert PROVISIONAL_TOKEN_SPEC is False
    assert_token_spec_frozen()  # no-op, raises if flipped back
