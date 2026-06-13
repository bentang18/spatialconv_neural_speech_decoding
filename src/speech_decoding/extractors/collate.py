"""SCAFFOLD-06 (DP01): variable-T (RoPE-compatible) + variable-C
(per-corpus) collate.

Phase-1 mixes 4 corpora — SWEC (anatomy-blind, k0–k21 valid bins),
AJILE12 (89.7% ECoG, k0–k20 valid bins), D-cohort (full 30-bin valid
range), BT (full 30-bin valid range). C_MAX and T_max differ per
corpus; this collate pads each batch to per-batch ``C_max`` /
``T_max`` and emits a ``valid_mask`` that the encoder consumes.

Frequency-bin count ``F`` and parcel count ``K`` are pre-padded to the
v14 cohort-wide constants (``F=30`` 30-bin log ⅓-octave filterbank,
``K=80`` DK parcels) by the extractor pipeline; mismatch is a config
error and is rejected explicitly.

The function is plain (not a class) so it can drop directly into
``torch.utils.data.DataLoader(collate_fn=v14_variable_tc_collate)``.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor


# ``electrode_tokens_high`` is the 2STFT (per_band_stem) HIGH band — present ONLY
# on the dual-band frontend. It shares the SAME electrodes (and therefore the same
# per-sample C and valid_mask/support) as ``electrode_tokens`` (the LOW band), so
# it is padded to the SAME ``C_max`` but keeps its OWN freq/time axes (F_high,
# T_high differ from the low band — different nperseg/hop). Absent on every
# non-2STFT path; handled separately below so it round-trips freq-major
# ``(C, F_high, T_high)`` untransposed (the encoder consumes it via
# ``_forward_dual_band`` with ``time_last_input``-style freq-major orientation).
_VARIABLE_FIELDS = ("electrode_tokens", "electrode_tokens_high", "support", "valid_mask")


def _stack_target(samples: Sequence[dict], key: str) -> Tensor:
    """Stack target / scalar fields. 0-d tensors stack to 1-d; 1-d
    tensors get a batch dim."""
    vals = [s[key] for s in samples]
    return torch.stack(vals, dim=0)


def v14_variable_tc_collate(samples: Sequence[dict[str, Any]]) -> dict[str, Tensor]:
    """Variable-T + variable-C collate.

    Inputs (per sample dict) ::

        electrode_tokens: (C_i, T_i, F)           float
        support:          (C_i, K)                float
        valid_mask:       (C_i,)                  bool   (True = real)
        target:           any fixed-shape tensor  (optional)
        ...               other fixed-shape fields round-trip via stack

    Output ::

        electrode_tokens: (B, C_max, T_max, F)    zero-padded
        support:          (B, C_max, K)           zero-padded
        valid_mask:       (B, C_max)              False at padded electrodes

        all other fields:  stacked along dim 0.

    Raises :class:`ValueError` if ``F`` or ``K`` differ across samples
    in a batch (per-corpus constants).
    """
    if len(samples) == 0:
        raise ValueError("v14_variable_tc_collate received an empty batch")

    # Validate per-corpus constants.
    F = samples[0]["electrode_tokens"].shape[-1]
    K = samples[0]["support"].shape[-1]
    for i, s in enumerate(samples):
        if s["electrode_tokens"].shape[-1] != F:
            raise ValueError(
                f"sample {i}: freq-bin count mismatch "
                f"{s['electrode_tokens'].shape[-1]} != batch-leader {F}"
            )
        if s["support"].shape[-1] != K:
            raise ValueError(
                f"sample {i}: parcel count mismatch "
                f"{s['support'].shape[-1]} != batch-leader {K}"
            )

    B = len(samples)
    C_max = max(s["electrode_tokens"].shape[0] for s in samples)
    T_max = max(s["electrode_tokens"].shape[1] for s in samples)

    e_dtype = samples[0]["electrode_tokens"].dtype
    s_dtype = samples[0]["support"].dtype

    electrode_tokens = torch.zeros(B, C_max, T_max, F, dtype=e_dtype)
    support = torch.zeros(B, C_max, K, dtype=s_dtype)
    valid_mask = torch.zeros(B, C_max, dtype=torch.bool)

    for b, sample in enumerate(samples):
        e = sample["electrode_tokens"]
        sp = sample["support"]
        vm = sample["valid_mask"]
        C_i, T_i = e.shape[0], e.shape[1]
        electrode_tokens[b, :C_i, :T_i, :] = e
        support[b, :C_i, :] = sp
        valid_mask[b, :C_i] = vm

    out: dict[str, Tensor] = {
        "electrode_tokens": electrode_tokens,
        "support": support,
        "valid_mask": valid_mask,
    }

    # 2STFT HIGH band (per_band_stem): same electrodes as the low band, so pad to
    # the SAME ``C_max`` (per-sample C_i must match the low band's), but keep its
    # own freq/time axes. Freq-major ``(C, F_high, T_high)`` round-trips untransposed.
    if "electrode_tokens_high" in samples[0]:
        eh0 = samples[0]["electrode_tokens_high"]
        F_high = eh0.shape[-1]
        A1_high = max(s["electrode_tokens_high"].shape[1] for s in samples)
        for i, s in enumerate(samples):
            if "electrode_tokens_high" not in s:
                raise ValueError(
                    f"sample {i}: electrode_tokens_high missing while batch-leader "
                    "carries it (all-or-none per batch)"
                )
            eh = s["electrode_tokens_high"]
            if eh.shape[-1] != F_high:
                raise ValueError(
                    f"sample {i}: high-band freq-bin count mismatch "
                    f"{eh.shape[-1]} != batch-leader {F_high}"
                )
            if eh.shape[0] != samples[i]["electrode_tokens"].shape[0]:
                raise ValueError(
                    f"sample {i}: high-band electrode count {eh.shape[0]} != "
                    f"low-band {samples[i]['electrode_tokens'].shape[0]} "
                    "(the two 2STFT bands must share electrodes)"
                )
        electrode_tokens_high = torch.zeros(
            B, C_max, A1_high, F_high, dtype=eh0.dtype
        )
        for b, sample in enumerate(samples):
            eh = sample["electrode_tokens_high"]
            C_i, A1_i = eh.shape[0], eh.shape[1]
            electrode_tokens_high[b, :C_i, :A1_i, :] = eh
        out["electrode_tokens_high"] = electrode_tokens_high

    # Any non-variable field round-trips via plain stack.
    other_keys = set(samples[0].keys()) - set(_VARIABLE_FIELDS)
    for key in other_keys:
        out[key] = _stack_target(samples, key)

    return out
