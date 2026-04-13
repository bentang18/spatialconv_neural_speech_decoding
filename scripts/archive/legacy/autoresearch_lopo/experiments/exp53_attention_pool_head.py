#!/usr/bin/env python3
"""Exp 53: Attention pooling head instead of mean pooling.

Mean pooling treats all timesteps equally. Attention pooling learns which
timesteps are most informative for each phoneme position — should focus
on articulatory onset/offset events rather than steady-state.

Baseline: h.mean(dim=1) in ArticulatoryBottleneckHead, PER=0.762
Change: learned attention weights per position over time dimension
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from arch_ablation_base import run, SpatialReadIn, Backbone, ARTICULATORY_MATRIX


class AttentionPoolHead(nn.Module):
    """Articulatory bottleneck with learned attention pooling per position."""

    def __init__(self, d_in=64, n_positions=3, n_features=15, n_classes=9, dropout=0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        # Attention: one query per position
        self.attn_queries = nn.Parameter(torch.randn(n_positions, d_in) * 0.02)
        self.feature_heads = nn.ModuleList([
            nn.Linear(d_in, n_features) for _ in range(n_positions)
        ])
        self.register_buffer('A', torch.tensor(ARTICULATORY_MATRIX, dtype=torch.float32))
        self.temperature = nn.Parameter(torch.tensor(5.0))

    def forward(self, h):
        # h: (B, T, d_in)
        logits_list = []
        for pos_idx, head in enumerate(self.feature_heads):
            # Attention pooling for this position
            query = self.attn_queries[pos_idx]  # (d_in,)
            attn_scores = (h * query.unsqueeze(0).unsqueeze(0)).sum(dim=-1)  # (B, T)
            attn_weights = F.softmax(attn_scores, dim=1)  # (B, T)
            pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_in)
            pooled = self.drop(pooled)

            feat_pred = torch.sigmoid(head(pooled))
            feat_norm = F.normalize(feat_pred, dim=-1)
            A_norm = F.normalize(self.A.float(), dim=-1)
            sim = feat_norm @ A_norm.T * self.temperature
            logits_list.append(sim)
        return torch.stack(logits_list, dim=1)


if __name__ == "__main__":
    run(ReadInCls=SpatialReadIn, BackboneCls=Backbone,
        HeadCls=AttentionPoolHead,
        experiment_name="exp53_attention_pool_head")
