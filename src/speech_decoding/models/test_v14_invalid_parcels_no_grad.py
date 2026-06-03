"""Invariant: latents for invalid (no-coverage) parcels — and the inputs of
electrodes dropped via ``valid_mask`` — receive NO gradient from the
downstream readout (IE08).

**B36 (2026-06-01, hard block-diagonal pool)** tightens this from the
pre-B36 soft-leak tolerance to EXACT zero. Two structural facts make the
zero exact, not approximate:

1. The cross-attn pool is a hard one-hot DK assignment: an off-parcel /
   dropped electrode is masked to ``NEG_INF_MASK_VALUE`` pre-softmax AND
   zeroed post-softmax, so its pooling weight is **exactly 0.0** in every
   dtype. A no-coverage parcel's whole row is zero → its latent is left at
   its frozen init and is masked out of the latent-SA (B30 bidirectional)
   and the PMA readout (``latent_valid``).
2. The FE-04 front-end token blocks fold the electrode axis into the batch
   dim (``BC = B*C``, ``v14_encoder.py:1002``), so they are strictly
   per-electrode — a dropped electrode's tokens never bleed into any other
   electrode's tokens before the (masked) cross-attn.

Together: a dropped electrode has exactly one path to the readout (the
cross-attn pool) and that path is hard-zeroed, so its gradient is 0.0
exactly. This test pins the exact-zero invariant so any future change to
the masking stack (or a regression to a soft bias) gets caught.
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import V14ParcelPerceiver


def _make_tiny_model() -> tuple:
    """Returns a tiny built model + standard tensor shapes."""
    B, C, T, F = 1, 4, 6, 8
    K, M = 6, 2
    cfg = V14ParcelPerceiver(
        n_freq_bins=F,
        n_time_bins=T,
        k_parcels=K,
        d_model=32,
        n_heads=4,
        depth_self_attn=2,
        m_sub_slots=M,
        n_token_blocks=1,
    )
    model = cfg.build(n_classes=3)
    model.eval()  # deterministic; we still get grads via .backward()
    return model, B, C, T, F, K, M


def test_invalid_parcels_have_exactly_zero_grad_at_encoder_output() -> None:
    """One-hot DK assignment: electrode c → parcel c (c in 0..3); parcels
    4, 5 get NO electrode. After forward+backward, the encoder-output
    latents for the no-coverage parcel rows must have grad == 0.0 EXACTLY.
    Primary severance is the ``latent_valid`` mask (the no-coverage slots are
    dropped from both the latent-SA and the PMA readout — this alone would
    zero the grad even under the pre-B36 soft bias). The B36 hard pool is
    consistent with it (all-zero pooling row → frozen-init latent). The
    genuinely hard-pool-*specific* grad proof — where a soft bias would leak —
    is ``test_invalid_channel_input_has_exactly_zero_grad`` below.
    """
    torch.manual_seed(0)
    model, B, C, T, F, K, M = _make_tiny_model()

    # Clean one-hot assignment: electrode c assigned to parcel c. Parcels
    # 0..3 are covered (one electrode each); parcels 4, 5 are no-coverage.
    n_valid_parcels = 4
    assert C == n_valid_parcels
    support = torch.zeros(B, C, K)
    for c in range(C):
        support[:, c, c] = 1.0
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    electrode_tokens = torch.randn(B, C, T, F)

    captured: dict = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured["latents"] = output

    handle = model.encoder.register_forward_hook(hook)
    out = model(electrode_tokens, support, valid_mask)
    out.sum().backward()
    handle.remove()

    latents_grad = captured["latents"].grad  # (B, L=K*M, T, d)
    assert latents_grad is not None
    # Invalid parcels = indices [n_valid_parcels*M .. K*M)
    invalid_idx = list(range(n_valid_parcels * M, K * M))
    valid_idx = [i for i in range(K * M) if i not in invalid_idx]

    grad_invalid = latents_grad[:, invalid_idx].abs().max().item()
    grad_valid = latents_grad[:, valid_idx].abs().max().item()

    assert grad_invalid == 0.0, (
        f"no-coverage-parcel latents leaked gradient (B36 hard pool must give "
        f"them an exactly-zero pool row): max |grad| = {grad_invalid}"
    )
    assert grad_valid > 0.0, (
        f"sanity: covered-parcel latents must have nonzero grad, got {grad_valid}"
    )


def test_invalid_channel_input_has_exactly_zero_grad() -> None:
    """A channel marked invalid via ``valid_mask`` must receive grad == 0.0
    EXACTLY on its input. Under B36 the hard pool masks it to an exact-zero
    pooling weight for every parcel, and the per-electrode front-end
    (``BC = B*C``) never lets its tokens bleed into other electrodes — so it
    has no surviving path to the readout. (Pre-B36 this was only attenuated
    because the soft ``log(support+ε)`` bias still gave it a finite weight.)
    """
    torch.manual_seed(1)
    model, B, C, T, F, K, M = _make_tiny_model()

    # One-hot assignment so every parcel that is covered stays covered after
    # dropping the last channel (electrode c → parcel c).
    support = torch.zeros(B, C, K)
    for c in range(C):
        support[:, c, c] = 1.0
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    valid_mask[:, -1] = False  # last channel is invalid

    tokens_a = torch.randn(B, C, T, F, requires_grad=True)
    out_a = model(tokens_a, support, valid_mask)
    out_a.sum().backward()
    assert tokens_a.grad is not None
    grad_invalid_chan = tokens_a.grad[:, -1].abs().max().item()
    grad_valid_chan = tokens_a.grad[:, 0].abs().max().item()

    assert grad_invalid_chan == 0.0, (
        f"invalid-channel input leaked gradient (B36 hard pool + per-electrode "
        f"front-end must give it no path to the readout): "
        f"max |grad| = {grad_invalid_chan}"
    )
    assert grad_valid_chan > 0.0, (
        f"sanity: valid channel must carry gradient, got {grad_valid_chan}"
    )
