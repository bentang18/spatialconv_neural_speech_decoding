"""v14_converged_v3 r4 — Perceiver-IO secondary head invariants (B4).

The load-bearing properties, each asserted with a printed ``[check]`` (build-the-invariant
-into-the-probe): (1) shapes + the covariance is symmetric PD with the measured noise
floor as its ceiling on the min eigenvalue; (2) PADDED tokens are truly ignored —
perturbing them is bit-stable (else masked/absent contacts leak); (3) the encode stage is
permutation-invariant over the key tokens; (4) decode queries are INDEPENDENT (a Gaussian
per (parcel,slot) must not depend on the other queries in the batch row); (5) the module is
clip-length-agnostic (n_slots is a runtime arg; the parameters don't change).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_converged_v3.perceiver import PerceiverHead
from speech_decoding.models.v14_converged_v3.secondary_head import NOISE_VAR, STATE_DIM

N_PARCELS = 5
B, K, Q = 2, 6, 4
N_SLOTS = 12  # 3 s clip / 32 Hz / 8


def _head() -> PerceiverHead:
    torch.manual_seed(0)
    return PerceiverHead(n_parcels=N_PARCELS).eval()


def _inputs(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, K, 1024, generator=g)
    token_time = torch.randint(0, N_SLOTS * 8, (B, K), generator=g)
    token_mask = torch.ones(B, K, dtype=torch.bool)
    token_mask[0, 4:] = False  # row 0 has 2 padded tokens
    query_parcel = torch.randint(0, N_PARCELS, (B, Q), generator=g)
    query_slot = torch.randint(0, N_SLOTS, (B, Q), generator=g)
    return z, token_time, token_mask, query_parcel, query_slot


def test_shapes_and_covariance_is_pd_above_the_noise_floor() -> None:
    head = _head()
    z, tt, tm, qp, qs = _inputs()
    with torch.no_grad():
        mu, cov = head(z, tt, tm, qp, qs, n_slots=N_SLOTS)
    assert mu.shape == (B, Q, STATE_DIM)
    assert cov.shape == (B, Q, STATE_DIM, STATE_DIM)
    sym = torch.allclose(cov, cov.transpose(-1, -2), atol=1e-6)
    eig = torch.linalg.eigvalsh(cov)  # (B, Q, D) ascending
    floor = min(NOISE_VAR)
    min_eig = eig.min().item()
    ok = sym and min_eig >= floor - 1e-5
    print(f"[check] cov symmetric={sym}, min eig {min_eig:.4f} >= noise floor "
          f"{floor:.4f} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_padded_tokens_are_truly_ignored() -> None:
    # Perturbing the PADDED token features (and their time coords) must not move any
    # output — else absent/masked contacts leak into the global state.
    head = _head()
    z, tt, tm, qp, qs = _inputs()
    with torch.no_grad():
        mu0, cov0 = head(z, tt, tm, qp, qs, n_slots=N_SLOTS)
        z2 = z.clone()
        tt2 = tt.clone()
        pad = ~tm  # (B, K) padded positions
        z2[pad] += torch.randn_like(z2[pad]) * 10.0
        tt2[pad] = torch.randint(0, N_SLOTS * 8, (int(pad.sum()),))
        mu1, cov1 = head(z2, tt2, tm, qp, qs, n_slots=N_SLOTS)
    ok = torch.allclose(mu0, mu1, atol=1e-6) and torch.allclose(cov0, cov1, atol=1e-6)
    print(f"[check] padded tokens ignored (max Δmu "
          f"{(mu0 - mu1).abs().max():.2e}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_encode_is_permutation_invariant_over_tokens() -> None:
    # Row 1 is all-real (no pad); permuting its K tokens with their (time, mask) leaves
    # every output unchanged — attention is permutation-invariant over keys.
    head = _head()
    z, tt, tm, qp, qs = _inputs()
    tm = torch.ones(B, K, dtype=torch.bool)  # all real for a clean permutation
    perm = torch.randperm(K)
    with torch.no_grad():
        mu0, cov0 = head(z, tt, tm, qp, qs, n_slots=N_SLOTS)
        mu1, cov1 = head(z[:, perm], tt[:, perm], tm[:, perm], qp, qs, n_slots=N_SLOTS)
    ok = torch.allclose(mu0, mu1, atol=1e-5) and torch.allclose(cov0, cov1, atol=1e-5)
    print(f"[check] encode permutation-invariant over tokens (max Δmu "
          f"{(mu0 - mu1).abs().max():.2e}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_decode_queries_are_independent() -> None:
    # Changing query j's (parcel, slot) must not move query 0's Gaussian — decode
    # queries attend the latents only, never each other.
    head = _head()
    z, tt, tm, qp, qs = _inputs()
    with torch.no_grad():
        mu0, cov0 = head(z, tt, tm, qp, qs, n_slots=N_SLOTS)
        qp2, qs2 = qp.clone(), qs.clone()
        qp2[:, 1] = (qp[:, 1] + 1) % N_PARCELS  # perturb query 1 only
        qs2[:, 1] = (qs[:, 1] + 1) % N_SLOTS
        mu1, cov1 = head(z, tt, tm, qp2, qs2, n_slots=N_SLOTS)
    ok = (torch.allclose(mu0[:, 0], mu1[:, 0], atol=1e-6)
          and torch.allclose(cov0[:, 0], cov1[:, 0], atol=1e-6)
          and not torch.allclose(mu0[:, 1], mu1[:, 1], atol=1e-4))  # query 1 DID move
    print(f"[check] decode queries independent (query0 Δmu "
          f"{(mu0[:, 0] - mu1[:, 0]).abs().max():.2e}) {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_clip_length_agnostic() -> None:
    # The same parameters serve any n_slots (M mode vectors shared across slots).
    head = _head()
    z, tt, tm, qp, qs = _inputs()
    n_params = sum(p.numel() for p in head.parameters())
    with torch.no_grad():
        mu12, _ = head(z, tt, tm, qp, qs, n_slots=12)
        mu16, _ = head(z, tt, tm, qp, qs.clamp(max=15), n_slots=16)
    ok = mu12.shape == (B, Q, STATE_DIM) == mu16.shape and head.latents.shape == (12, 128)
    print(f"[check] clip-length agnostic: n_slots 12 & 16 both run, "
          f"latents {tuple(head.latents.shape)}, {n_params/1e6:.2f}M params "
          f"{'OK' if ok else 'VIOLATED'}")
    assert ok
