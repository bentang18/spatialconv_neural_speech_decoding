"""DETR-readout FFN tests.

Canonical DETR decoder layer (Carion 2020 §3.3): ``self-attn → cross-attn →
FFN``, pre-LN with residuals. v14 has N=1 task query so query self-attn is
moot, but the post-cross-attn FFN is standard and was missing. This commit
adds it so the readout is one full DETR decoder layer (minus the trivial
self-attn).

Three pins:
1. The head exposes ``ln_ffn`` and ``ffn`` modules (presence pin).
2. Perturbing the FFN parameters changes the head output (the FFN is on the
   forward path — not a vestigial module).
3. The residual connection from the task query is honored (perturbing the
   query embedding changes the output even when FFN is zeroed).
"""

from __future__ import annotations

import torch

from speech_decoding.models.v14_encoder import V14ClassifierHead


D_MODEL = 32
N_CLASSES = 3
N_HEADS = 4


def test_classifier_head_exposes_ffn_and_ffn_layernorm() -> None:
    """Presence pin: the canonical DETR decoder layer's FFN is now part of
    the head module."""
    head = V14ClassifierHead(d_model=D_MODEL, n_classes=N_CLASSES, n_heads=N_HEADS)
    assert hasattr(head, "ln_ffn"), "head must carry an FFN LayerNorm"
    assert hasattr(head, "ffn"), "head must carry an FFN block"


def test_classifier_head_output_depends_on_ffn_weights() -> None:
    """Perturbing the FFN parameters must change the output — confirms the
    FFN is on the forward path, not a dead branch."""
    torch.manual_seed(0)
    head = V14ClassifierHead(d_model=D_MODEL, n_classes=N_CLASSES, n_heads=N_HEADS)
    head.eval()

    B, L = 2, 6
    latents = torch.randn(B, L, D_MODEL)

    with torch.no_grad():
        out_ref = head(latents)
        # Perturb every FFN parameter.
        for p in head.ffn.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        out_perturbed = head(latents)

    assert not torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "FFN must be on the forward path — perturbing its weights must change output"
    )


def test_classifier_head_residual_keeps_query_information() -> None:
    """The residual connection from the task query to the post-cross-attn
    hidden must be active. With the FFN zeroed (so the FFN branch contributes
    nothing), changing the task query embedding must still propagate to the
    output via the residual."""
    torch.manual_seed(0)
    head = V14ClassifierHead(d_model=D_MODEL, n_classes=N_CLASSES, n_heads=N_HEADS)
    head.eval()

    # Zero out FFN so only the cross-attn + residual + classifier path matters.
    with torch.no_grad():
        for p in head.ffn.parameters():
            p.zero_()

    B, L = 1, 4
    latents = torch.randn(B, L, D_MODEL)

    with torch.no_grad():
        out_ref = head(latents)
        original_query = head.query.data.clone()
        head.query.data.add_(torch.randn_like(head.query.data) * 5.0)
        out_perturbed = head(latents)
        head.query.data.copy_(original_query)

    assert not torch.allclose(out_ref, out_perturbed, atol=1e-5), (
        "task-query residual must reach the classifier path even when FFN is zero"
    )
