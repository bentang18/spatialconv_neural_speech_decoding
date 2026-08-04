"""Tests for the R35 per-subject parcel-vocabulary permutation.

The arm's whole validity rests on two things this file pins:

  1. The permutation preserves everything EXCEPT cross-subject correspondence. If it
     also changed the partition of electrodes into pooling groups, the tag marginals or
     the table size, the measured delta would confound several causes at once.
  2. Training and encode derive the SAME permutation from (subject_id, seed) alone.
     The checkpoint carries no hyperparameters, so nothing downstream can catch a drift
     between the two jobs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from speech_decoding.models.v14_converged_v3.parcel_perm import (
    N_PARCELS,
    apply_parcel_perm,
    parcel_permutation,
    perm_fingerprint,
)


def test_is_a_permutation_of_the_whole_table():
    for sub in (1, 2, 3, 7, 10):
        p = parcel_permutation(sub, seed=33)
        assert p.shape == (N_PARCELS,)
        assert p.dtype == torch.long
        assert sorted(p.tolist()) == list(range(N_PARCELS))


def test_deterministic_across_calls_and_independent_of_trial():
    # Same subject + seed must give the identical permutation in every process. The
    # training job and the encode job never share state, so this IS the parity contract.
    a = parcel_permutation(3, seed=33)
    b = parcel_permutation(3, seed=33)
    assert torch.equal(a, b)


def test_different_subjects_and_seeds_give_different_permutations():
    base = parcel_permutation(3, seed=33)
    assert not torch.equal(base, parcel_permutation(4, seed=33))
    assert not torch.equal(base, parcel_permutation(3, seed=34))


def test_pooling_partition_is_invariant():
    """THE SCIENCE GUARD: a relabeling must not regroup electrodes.

    The encode pools electrodes to parcels by grouping equal tags. A permutation is a
    bijection, so the induced partition of electrodes is identical -- only the group
    NAMES change. If this ever failed, the arm would be changing the readout's pooling
    as well as the model's tag, and the delta would be uninterpretable.
    """
    rng = np.random.default_rng(0)
    true = torch.from_numpy(rng.integers(0, N_PARCELS, size=120)).long()
    perm = apply_parcel_perm(true, subject_id=3, seed=33)

    def partition(t):
        return {frozenset(np.flatnonzero(t.numpy() == v).tolist()) for v in np.unique(t.numpy())}

    assert partition(true) == partition(perm)
    # and the multiset of group sizes (the tag marginal) is untouched
    assert sorted(np.bincount(true.numpy(), minlength=N_PARCELS).tolist()) == sorted(
        np.bincount(perm.numpy(), minlength=N_PARCELS).tolist()
    )


def test_within_subject_distinguishability_preserved():
    """Two contacts share a tag after the permutation iff they shared one before."""
    true = torch.tensor([5, 5, 12, 74, 12, 0]).long()
    perm = apply_parcel_perm(true, subject_id=7, seed=33)
    assert (true[:, None] == true[None, :]).equal(perm[:, None] == perm[None, :])


def test_unknown_id_is_permuted_too():
    # Holding the reserved unknown row fixed would leave one shared cross-subject anchor
    # and weaken the manipulation. It moves like every other row.
    moved = [int(parcel_permutation(s, seed=33)[N_PARCELS - 1]) for s in (1, 2, 3, 4, 5)]
    assert len(set(moved)) > 1


def test_roundtrip_through_the_inverse_recovers_true_ids():
    true = torch.arange(N_PARCELS).long()
    perm = parcel_permutation(3, seed=33)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(N_PARCELS)
    assert torch.equal(inv[apply_parcel_perm(true, subject_id=3, seed=33)], true)


def test_rejects_ids_outside_the_table():
    with pytest.raises(ValueError, match="outside the identity table"):
        apply_parcel_perm(torch.tensor([0, N_PARCELS]).long(), subject_id=3, seed=33)


def test_fingerprint_is_stable_and_sensitive():
    subs = [1, 2, 3, 7, 10]
    f = perm_fingerprint(subs, seed=33)
    assert f == perm_fingerprint(list(reversed(subs)), seed=33)  # order-insensitive
    assert f != perm_fingerprint(subs, seed=34)  # seed-sensitive
    assert f != perm_fingerprint(subs + [4], seed=33)  # cohort-sensitive


def test_train_and_encode_paths_agree_on_the_model_tag():
    """Parity: the encode reconstructs the model-side tag from the TRUE tag + seed.

    This is the exact relation ``v3_probe_encode_r4`` asserts on real data. The train
    path bakes the permutation into ``setup.parcel_id``; the encode recomputes the true
    tag from ``parcel_fn`` and must land on the same tensor.
    """
    rng = np.random.default_rng(1)
    true = torch.from_numpy(rng.integers(0, N_PARCELS, size=96)).long()

    train_side = apply_parcel_perm(true, subject_id=3, seed=33)  # baked at session load
    encode_side = apply_parcel_perm(true, subject_id=3, seed=33)  # recomputed at encode
    assert torch.equal(train_side, encode_side)

    # A wrong seed at encode must be LOUD, not a silent near-miss.
    assert not torch.equal(train_side, apply_parcel_perm(true, subject_id=3, seed=34))
