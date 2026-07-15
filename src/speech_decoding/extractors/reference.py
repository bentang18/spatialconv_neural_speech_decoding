"""CAR / shaft-CAR variants of NeuralSet's `IeegExtractor`.

NeuralSet's `IeegExtractor` exposes `bipolar_ref`, `notch_filter`, `filter`,
`apply_hilbert`, `scaler`, and `clamp` as constructor kwargs, which cover the
R0 raw / R3 bipolar / R4 shaft-Laplacian / I-family / F-family ablation cells
via config swaps. CAR has no first-class kwarg; this module fills the gap with
a thin subclass that subtracts the per-channel-mean post-load.

Two CAR modes:
  * ``car="global"`` — robust global CAR, subtracts the across-channel mean
    from every channel. Diagnostic for sparse asymmetric sEEG coverage.
  * ``car="shaft"`` — within-shaft CAR, subtracts the per-shaft mean. Physics-
    matched candidate for sEEG.

The shaft parser splits trailing integer suffix from stem (``"OFa12" -> ("OFa", 12)``).
The Stage-0 shaft/depth contract froze this parser on 2026-05-13; see
``reports/neuroprobe_stage0_shaft_depth_geometry_2026_05_13/README.md`` for the
admissibility verdict on signed-depth features, reference-scheme compatibility,
and the rejected encodings list. Audit: 1549 contacts across 9 subjects, 128
shafts, 0 parse anomalies, 0 cross-hemisphere shafts.
"""

from __future__ import annotations

import re
import typing as tp

import numpy as np
from neuralset.extractors.neuro import IeegExtractor, MneRaw

_SHAFT_RE = re.compile(r"^(.*?)(\d+)$")


def parse_shaft(channel_name: str) -> tuple[str, int | None]:
    """Split ``"OFa12"`` into ``("OFa", 12)``. Non-numeric suffix returns ``(name, None)``."""
    match = _SHAFT_RE.match(channel_name)
    if match is None:
        return channel_name, None
    return match.group(1), int(match.group(2))


class CARIeegExtractor(IeegExtractor):
    """`IeegExtractor` with a CAR (common-average reference) mode.

    Mutually exclusive with `reference="bipolar"` and explicit `bipolar_ref`.
    Order of operations: pick channels -> drop bads -> CAR -> (no bipolar)
    -> filter/notch/hilbert/scaler/clamp.
    """

    car: tp.Literal["global", "shaft", "group"] | None = None

    def model_post_init(self, log__):
        super().model_post_init(log__)
        if self.car is None:
            return
        if self.reference == "bipolar" or self.bipolar_ref is not None:
            raise ValueError(
                "CAR is mutually exclusive with bipolar reference. "
                "Set car=None or pick exactly one referencing scheme."
            )

    def _preprocess_raw(self, raw, event):
        raw = self._pick_channels(raw, self.picks)
        if self.drop_bads:
            raw.load_data()
            raw = raw.drop_channels(raw.info["bads"])
        if self.car is not None:
            raw = self._apply_car(raw)
        if self.reference == "bipolar":
            raw.load_data()
            raw = self._apply_bipolar_ref(raw)
        return MneRaw._preprocess_raw(self, raw, event)

    def _apply_car(self, raw):
        raw.load_data()
        data = raw._data
        if self.car == "global":
            data -= data.mean(axis=0, keepdims=True)
            return raw
        if self.car == "shaft":
            shafts = [parse_shaft(name)[0] for name in raw.ch_names]
            for shaft in set(shafts):
                idx = np.array([i for i, s in enumerate(shafts) if s == shaft])
                if idx.size == 0:
                    continue
                data[idx] -= data[idx].mean(axis=0, keepdims=True)
            return raw
        if self.car == "group":
            temp = raw.info.get("temp") or {}
            cmap = temp.get("car_groups")
            if cmap is None:
                raise ValueError(
                    "car='group' requires an authoritative per-run map at "
                    "raw.info['temp']['car_groups'] (set by the Study's _load_raw)."
                )
            groups = [cmap[name] for name in raw.ch_names]
            drop = []
            for g in set(groups):
                idx = np.array([i for i, s in enumerate(groups) if s == g])
                if idx.size <= 1:               # singleton: no valid CAR ref -> DROP
                    drop.extend(idx.tolist())   #   (orphaned tail / guard-1 remnant; dynamic)
                    continue
                data[idx] -= data[idx].mean(axis=0, keepdims=True)
            if drop:
                raw.drop_channels([raw.ch_names[i] for i in sorted(set(drop))])
            return raw
        raise ValueError(f"Unknown car mode: {self.car!r}")
