"""SCAFFOLD-06 (DP01): variable-T (RoPE-compatible) + variable-C
(per-corpus) collate.

Phase-1 mixes corpora with different ``C_MAX`` and ``T_max``; this collate
pads each batch to per-batch ``C_max`` / ``T_*_max`` and emits a
``valid_mask`` that the encoder consumes to ignore padded electrodes.

Two orientation modes, selected by whether the 2STFT high band is present:

* **Single-band** (default; ``front_end="raw"`` / ``"fbank"``) — the tokens
  are TIME-MAJOR ``electrode_tokens: (C, T, F)``. The FIXED axis is freq
  (last, a front-end constant); the VARIABLE axis is time (axis 1), padded to
  per-batch ``T_max``.
* **2STFT dual-band** (``electrode_tokens_high`` present) — BOTH bands are
  FREQ-MAJOR per the encoder contract (``_forward_dual_band`` reads
  ``(B, C, F, T)`` with time LAST): the low band rides ``electrode_tokens:
  (C, F_low, T_low)`` and the high band ``electrode_tokens_high: (C, F_high,
  T_high)``. The FIXED axis is freq (axis 1); the VARIABLE axis is TIME
  (last), padded to per-batch ``T_low_max`` / ``T_high_max``.

The per-corpus freq-bin count and parcel count ``K`` are fixed by the
extractor pipeline; a within-batch mismatch on the fixed axis is a config
error and is rejected explicitly. The variable (time) axis is padded, never
equality-checked, so a ragged-duration batch collates rather than crashing.

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
# T_high differ from the low band — different nperseg/hop). Its presence ALSO
# flips ``electrode_tokens`` into freq-major handling (the low band is freq-major
# in 2STFT mode); absent → the single-band time-major path. Both bands round-trip
# freq-major ``(C, F, T)`` untransposed for the encoder's ``_forward_dual_band``.
_VARIABLE_FIELDS = ("electrode_tokens", "electrode_tokens_high", "support", "valid_mask")


def _stack_target(samples: Sequence[dict], key: str) -> Tensor:
    """Stack target / scalar fields. 0-d tensors stack to 1-d; 1-d
    tensors get a batch dim."""
    vals = [s[key] for s in samples]
    return torch.stack(vals, dim=0)


def v14_variable_tc_collate(samples: Sequence[dict[str, Any]]) -> dict[str, Tensor]:
    """Variable-T + variable-C collate (orientation-mode-aware; see module docstring).

    Inputs (per sample dict) — single-band ::

        electrode_tokens: (C_i, T_i, F)           float  TIME-major
        support:          (C_i, K)                float
        valid_mask:       (C_i,)                  bool   (True = real)
        target:           any fixed-shape tensor  (optional)
        ...               other fixed-shape fields round-trip via stack

    Inputs — 2STFT dual-band (``electrode_tokens_high`` present) ::

        electrode_tokens:      (C_i, F_low, T_low_i)    float  FREQ-major (low band)
        electrode_tokens_high: (C_i, F_high, T_high_i)  float  FREQ-major (high band)
        support / valid_mask:  as above (shared across bands)

    Output ::

        single-band  electrode_tokens: (B, C_max, T_max, F)         zero-padded
        dual-band    electrode_tokens: (B, C_max, F_low, T_low_max)  zero-padded
                     electrode_tokens_high: (B, C_max, F_high, T_high_max)
        support:          (B, C_max, K)           zero-padded
        valid_mask:       (B, C_max)              False at padded electrodes
        all other fields:  stacked along dim 0.

    Raises :class:`ValueError` if the FIXED axis (freq, or ``K``) differs across
    samples in a batch. The variable (time) axis is padded, not rejected.
    """
    if len(samples) == 0:
        raise ValueError("v14_variable_tc_collate received an empty batch")

    B = len(samples)
    # The high band's presence selects the orientation mode for BOTH bands.
    is_dual_band = "electrode_tokens_high" in samples[0]

    # ``support`` / ``valid_mask`` are orientation-independent — keyed on the
    # electrode axis (axis 0), which is shared across the two 2STFT bands.
    K = samples[0]["support"].shape[-1]
    C_max = max(s["electrode_tokens"].shape[0] for s in samples)
    for i, s in enumerate(samples):
        if s["support"].shape[-1] != K:
            raise ValueError(
                f"sample {i}: parcel count mismatch "
                f"{s['support'].shape[-1]} != batch-leader {K}"
            )

    e0 = samples[0]["electrode_tokens"]
    e_dtype = e0.dtype
    if is_dual_band:
        # LOW band, FREQ-MAJOR (C, F_low, T_low): fixed axis 1 = freq (front-end
        # constant), variable last axis = time → pad time to per-batch max. (The
        # symmetric counterpart of the high-band branch below; pre-fix this band
        # ran the time-major path on a transposed tensor — a no-op at constant
        # clip_len but a false "freq mismatch" once T_low varied across a batch.)
        F_low = e0.shape[1]
        T_low_max = max(s["electrode_tokens"].shape[-1] for s in samples)
        for i, s in enumerate(samples):
            if s["electrode_tokens"].shape[1] != F_low:
                raise ValueError(
                    f"sample {i}: low-band freq-bin count mismatch "
                    f"{s['electrode_tokens'].shape[1]} != batch-leader {F_low}"
                )
        electrode_tokens = torch.zeros(B, C_max, F_low, T_low_max, dtype=e_dtype)
        for b, sample in enumerate(samples):
            e = sample["electrode_tokens"]
            C_i, T_i = e.shape[0], e.shape[-1]
            electrode_tokens[b, :C_i, :, :T_i] = e
    else:
        # Single-band TIME-MAJOR (C, T, F): fixed last axis = freq, variable axis
        # 1 = time → pad time to per-batch max (the original v14 behaviour).
        F = e0.shape[-1]
        T_max = max(s["electrode_tokens"].shape[1] for s in samples)
        for i, s in enumerate(samples):
            if s["electrode_tokens"].shape[-1] != F:
                raise ValueError(
                    f"sample {i}: freq-bin count mismatch "
                    f"{s['electrode_tokens'].shape[-1]} != batch-leader {F}"
                )
        electrode_tokens = torch.zeros(B, C_max, T_max, F, dtype=e_dtype)
        for b, sample in enumerate(samples):
            e = sample["electrode_tokens"]
            C_i, T_i = e.shape[0], e.shape[1]
            electrode_tokens[b, :C_i, :T_i, :] = e

    s_dtype = samples[0]["support"].dtype
    support = torch.zeros(B, C_max, K, dtype=s_dtype)
    valid_mask = torch.zeros(B, C_max, dtype=torch.bool)
    for b, sample in enumerate(samples):
        sp = sample["support"]
        vm = sample["valid_mask"]
        C_i = sp.shape[0]
        support[b, :C_i, :] = sp
        valid_mask[b, :C_i] = vm

    out: dict[str, Tensor] = {
        "electrode_tokens": electrode_tokens,
        "support": support,
        "valid_mask": valid_mask,
    }

    # 2STFT HIGH band (per_band_stem): FREQ-MAJOR (C, F_high, T_high). Same
    # electrodes as the low band, so pad to the SAME ``C_max`` (per-sample C_i must
    # match the low band's); fix freq (axis 1, a front-end constant) and pad TIME
    # (last) to per-batch ``T_high_max`` — honouring the encoder's freq-major
    # ``_forward_dual_band`` contract, symmetric with the low-band branch above.
    if is_dual_band:
        eh0 = samples[0]["electrode_tokens_high"]
        F_high = eh0.shape[1]
        T_high_max = max(s["electrode_tokens_high"].shape[-1] for s in samples)
        for i, s in enumerate(samples):
            if "electrode_tokens_high" not in s:
                raise ValueError(
                    f"sample {i}: electrode_tokens_high missing while batch-leader "
                    "carries it (all-or-none per batch)"
                )
            eh = s["electrode_tokens_high"]
            if eh.shape[1] != F_high:
                raise ValueError(
                    f"sample {i}: high-band freq-bin count mismatch "
                    f"{eh.shape[1]} != batch-leader {F_high}"
                )
            if eh.shape[0] != samples[i]["electrode_tokens"].shape[0]:
                raise ValueError(
                    f"sample {i}: high-band electrode count {eh.shape[0]} != "
                    f"low-band {samples[i]['electrode_tokens'].shape[0]} "
                    "(the two 2STFT bands must share electrodes)"
                )
        electrode_tokens_high = torch.zeros(
            B, C_max, F_high, T_high_max, dtype=eh0.dtype
        )
        for b, sample in enumerate(samples):
            eh = sample["electrode_tokens_high"]
            C_i, T_i = eh.shape[0], eh.shape[-1]
            electrode_tokens_high[b, :C_i, :, :T_i] = eh
        out["electrode_tokens_high"] = electrode_tokens_high

    # Any non-variable field round-trips via plain stack.
    other_keys = set(samples[0].keys()) - set(_VARIABLE_FIELDS)
    for key in other_keys:
        out[key] = _stack_target(samples, key)

    return out
