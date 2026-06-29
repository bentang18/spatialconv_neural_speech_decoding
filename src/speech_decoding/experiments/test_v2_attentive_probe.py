"""Laptop TDD for the attentive probe head (:mod:`v2_attentive_probe`).

Pins the locked contract: folded single-query cross-attention (no W_q/W_k modules),
permutation-invariance over the token set, padding-mask correctness, token-dropout
safety (no NaN even at p=0.9), and a learnability smoke (separable set → AUROC≈1).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from speech_decoding.experiments.v2_attentive_probe import AttentiveProbeHead


def test_shapes_single_query():
    head = AttentiveProbeHead(16, n_heads=4, n_queries=1).eval()
    out = head(torch.randn(5, 7, 16))
    assert out.shape == (5, 1)


def test_shapes_multi_query_multi_head():
    head = AttentiveProbeHead(24, n_heads=6, n_queries=4, n_out=1).eval()
    out = head(torch.randn(3, 11, 24))
    assert out.shape == (3, 1)


def test_wq_wk_are_folded_not_modules():
    """The contract: W_q/W_k fold into `phi`; only W_v/W_o survive as Linears."""
    head = AttentiveProbeHead(16, n_heads=4)
    names = {n for n, _ in head.named_modules()}
    assert not any("W_q" in n or "to_q" in n or "W_k" in n or "to_k" in n for n in names)
    param_names = {n for n, _ in head.named_parameters()}
    assert "phi" in param_names and "query" in param_names
    assert any(n.startswith("W_v") for n in param_names)
    assert any(n.startswith("W_o") for n in param_names)


def test_permutation_invariant_over_tokens():
    head = AttentiveProbeHead(16, n_heads=4, attn_dropout=0.0).eval()
    x = torch.randn(4, 9, 16)
    perm = torch.randperm(9)
    a = head(x)
    b = head(x[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_padding_mask_ignores_padded_tokens():
    head = AttentiveProbeHead(16, n_heads=4).eval()
    x = torch.randn(3, 6, 16)
    base = head(x)
    pad = torch.randn(3, 4, 16) * 100.0                      # garbage padding
    xp = torch.cat([x, pad], dim=1)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[:, :6] = True
    masked = head(xp, key_padding_mask=mask)
    assert torch.allclose(base, masked, atol=1e-5)


def test_token_dropout_no_nan_and_train_eval_differ():
    head = AttentiveProbeHead(16, n_heads=4, token_dropout=0.9,
                              attn_dropout=0.2, mlp_dropout=0.2, residual_dropout=0.2)
    x = torch.randn(8, 12, 16)
    head.train()
    torch.manual_seed(0)
    o1 = head(x)
    torch.manual_seed(1)
    o2 = head(x)
    assert torch.isfinite(o1).all() and torch.isfinite(o2).all()
    assert not torch.allclose(o1, o2)                        # stochastic in train
    head.eval()
    assert torch.allclose(head(x), head(x))                  # deterministic in eval


def test_padding_mask_no_nan_when_dropout_would_empty_a_row():
    """token_dropout combined with a mask must never softmax an all -inf row."""
    head = AttentiveProbeHead(16, n_heads=4, token_dropout=0.95).train()
    x = torch.randn(6, 5, 16)
    mask = torch.zeros(6, 5, dtype=torch.bool)
    mask[:, 0] = True                                        # only 1 valid token/row
    torch.manual_seed(0)
    out = head(x, key_padding_mask=mask)
    assert torch.isfinite(out).all()


def test_parcel_dropout_drops_whole_parcels():
    """Parcel dropout keep/drop is decided per parcel and broadcast to all its tokens."""
    tpp = 4
    head = AttentiveProbeHead(16, n_heads=4, parcel_dropout=0.5,
                              tokens_per_parcel=tpp).train()
    x = torch.randn(8, 5 * tpp, 16)                          # 5 parcels × 4 tokens
    torch.manual_seed(0)
    mask = head._token_mask(x, None)
    assert mask is not None
    blocks = mask.view(mask.shape[0], -1, tpp)
    # every token in a parcel shares one decision (all kept or all dropped)
    assert (blocks.all(dim=-1) | (~blocks.any(dim=-1))).all()
    assert (mask.sum(dim=-1) > 0).all()                      # never-empty guard
    assert not mask.all()                                    # some parcel actually dropped


def test_parcel_dropout_noop_without_block_size():
    """Unknown block size (tokens_per_parcel=0) disables parcel dropout (no guess)."""
    head = AttentiveProbeHead(16, n_heads=4, parcel_dropout=0.9,
                              tokens_per_parcel=0).train()
    assert head._token_mask(torch.randn(4, 12, 16), None) is None


def test_parcel_dropout_forward_finite_and_train_eval_differ():
    head = AttentiveProbeHead(16, n_heads=4, parcel_dropout=0.5, tokens_per_parcel=3,
                              attn_dropout=0.0, mlp_dropout=0.0, residual_dropout=0.0)
    x = torch.randn(8, 12, 16)
    head.train()
    torch.manual_seed(0)
    o1 = head(x)
    torch.manual_seed(1)
    o2 = head(x)
    assert torch.isfinite(o1).all() and torch.isfinite(o2).all()
    assert not torch.allclose(o1, o2)
    head.eval()
    assert torch.allclose(head(x), head(x))


def test_learnability_separable_set():
    """A set whose token-mean encodes the label is learnable → train AUROC ≈ 1."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    n, t, d = 200, 8, 16
    y = rng.choice([0.0, 1.0], size=n).astype(np.float32)
    x = rng.standard_normal((n, t, d)).astype(np.float32)
    x += (2.0 * y[:, None, None] - 1.0) * 1.5               # shift the whole set by label
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)[:, None]
    head = AttentiveProbeHead(d, n_heads=4, attn_dropout=0.0, mlp_dropout=0.0,
                              residual_dropout=0.0)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss()
    head.train()
    for _ in range(300):
        opt.zero_grad()
        loss = lossf(head(xt), yt)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        scores = head(xt).squeeze(-1).numpy()
    assert roc_auc_score(y, scores) > 0.95
