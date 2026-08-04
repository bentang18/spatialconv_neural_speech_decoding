"""v14_converged_v3 — per-subject parcel-vocabulary permutation (R35 ablation).

WHAT IT ISOLATES. The 2x2 geometry ablation showed the parcel embed is load-bearing
(noparcel: -.0205 / -.0129 / -.0178 ws / csession / cs) while the pretrain loss barely
moves (+.0037). Two accounts survive that:

  A. WITHIN-SUBJECT IDENTIFIABILITY. The tag lets one shared-weight encoder tell a
     contact's region apart from its neighbours inside the same brain. Any stable
     per-region code would do; the atlas is a convenience.
  B. CROSS-SUBJECT CORRESPONDENCE. Embedding row p accumulates gradient from every
     electrode tagged p in EVERY brain, so a true atlas makes the shared function
     conditional on a code that means the same thing across subjects. Pretraining is
     itself a cross-subject problem, and the tag is what makes it well posed.

A per-subject permutation of the parcel vocabulary separates them. It is a relabeling
INSIDE each brain, so A is preserved exactly; it destroys the correspondence BETWEEN
brains, so B is removed. Parameter count, table size, tag marginals and the partition
of electrodes into groups are all identical to baseline. One thing changes.

  perm ~ baseline  => A. The atlas is a convenience, not the mechanism.
  perm ~ noparcel  => B. Cross-subject alignment during pretraining is the mechanism.
  in between       => the -.0205/-.0129/-.0178 splits into an alignment term
                      (baseline - perm) and an identifiability term (perm - noparcel).

SCOPE — THIS IS A MODEL-SIDE TAG ONLY. The permuted id feeds the parcel embed and the
predictor mask-query seed. It must NEVER reach the readout's parcel pooling or the CS
anchor/test parcel intersection: those match parcels ACROSS subjects by atlas id, so
permuting them would scramble the readout instead of the encoder and the arm would
measure the wrong thing. ``v3_probe_encode_r4`` keeps the two apart explicitly and
asserts the relation between them on real data.

PER SUBJECT, NOT PER SESSION. Every trial of one subject shares a convention: the
scramble is between brains, not between recordings of one brain.

DETERMINISM. ``np.random.default_rng`` (PCG64) is stream-stable across platforms and
numpy versions by numpy's own compatibility policy, which ``torch.randperm`` does not
promise. The training run and the encode run happen in different jobs on different
nodes and MUST derive the identical permutation from (subject_id, seed) alone, so the
stability guarantee is the reason for numpy here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

# DKT vocab 74 (ids 0..73) + 1 reserved "unknown" row (id 74). Matches _n_parcels in
# dispatch_v3 and N_PARCELS in the encode/readout scripts.
N_PARCELS = 75


def parcel_permutation(subject_id: int, *, seed: int, n_parcels: int = N_PARCELS) -> Tensor:
    """Deterministic permutation of the parcel vocabulary for one subject.

    The whole table is permuted, INCLUDING the reserved unknown id. "Unknown" carries
    cross-subject meaning too (no atlas support, typically white matter), so holding it
    fixed would leave one shared anchor and weaken the manipulation.
    """
    rng = np.random.default_rng([int(seed), int(subject_id)])
    return torch.from_numpy(rng.permutation(int(n_parcels))).long()


def apply_parcel_perm(
    parcel_id: Tensor, subject_id: int, *, seed: int, n_parcels: int = N_PARCELS
) -> Tensor:
    """Relabel ``parcel_id`` (N,) long through this subject's permutation."""
    pid = parcel_id.long()
    hi = int(pid.max()) if pid.numel() else -1
    if hi >= n_parcels:
        raise ValueError(f"parcel id {hi} outside the identity table (n_parcels={n_parcels})")
    return parcel_permutation(subject_id, seed=seed, n_parcels=n_parcels).to(pid.device)[pid]


def perm_fingerprint(
    subject_ids: Sequence[int], *, seed: int, n_parcels: int = N_PARCELS
) -> str:
    """Short stable digest of the permutations for ``subject_ids``.

    Printed by BOTH the training banner and the encode banner. The checkpoint carries no
    hyperparameters (v3_probe_encode_r4:145), so nothing in the weights can catch an
    encode run that was handed the wrong --parcel-perm-seed. This digest makes that
    mismatch a one-line visual comparison instead of a silent wrong number.
    """
    h = hashlib.blake2s(digest_size=8)
    h.update(f"{int(seed)}:{int(n_parcels)}".encode())
    for s in sorted(int(x) for x in subject_ids):
        h.update(f"|{s}:".encode())
        h.update(parcel_permutation(s, seed=seed, n_parcels=n_parcels).numpy().astype("<i8").tobytes())
    return h.hexdigest()
