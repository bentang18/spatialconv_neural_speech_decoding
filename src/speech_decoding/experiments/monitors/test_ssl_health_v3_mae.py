"""SSLHealthMonitorV3 — the EMA-weight-gap read must tolerate the MAE arm (no teacher).

The MAE arm builds no EMA teacher (``objective.teacher is None``), so ``_ema_weight_gap``
must return ``None`` (metric simply not logged) instead of dereferencing ``None.model`` — the
crash the MAE GPU smoke caught. The JEPA path (teacher present) still returns a finite float.
"""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from speech_decoding.experiments.monitors.ssl_health_v3 import SSLHealthMonitorV3


def _pl(objective) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(objective=objective))


def test_ema_weight_gap_is_none_for_mae_no_teacher() -> None:
    mon = SSLHealthMonitorV3()
    obj = SimpleNamespace(online=nn.Linear(4, 4), teacher=None)  # MAE arm
    gap = mon._ema_weight_gap(_pl(obj))
    ok = gap is None
    print(f"[check] MAE (teacher=None): _ema_weight_gap -> {gap} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_ema_weight_gap_is_finite_float_for_jepa_teacher() -> None:
    mon = SSLHealthMonitorV3()
    obj = SimpleNamespace(online=nn.Linear(4, 4), teacher=SimpleNamespace(model=nn.Linear(4, 4)))
    gap = mon._ema_weight_gap(_pl(obj))
    ok = isinstance(gap, float) and gap == gap and gap >= 0.0  # finite, non-negative ratio
    print(f"[check] JEPA (teacher present): _ema_weight_gap -> {gap} {'OK' if ok else 'VIOLATED'}")
    assert ok


def test_ema_weight_gap_none_when_objective_absent() -> None:
    mon = SSLHealthMonitorV3()
    gap = mon._ema_weight_gap(SimpleNamespace(model=SimpleNamespace(objective=None)))
    print(f"[check] no objective: _ema_weight_gap -> {gap} {'OK' if gap is None else 'VIOLATED'}")
    assert gap is None
