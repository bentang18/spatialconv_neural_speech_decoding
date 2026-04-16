"""Deterministic verdict logic.

- any `must` check failed            → `fail`
- all `must` checks passed + no soft → `pass`
- all `must` checks passed + 1+ soft → `warn` (with that soft warning listed)
"""
from __future__ import annotations

from speech_decoding.v14.audit.schema import AuditResult, Check, Verdict


def compute_verdict(checks: list[Check]) -> Verdict:
    if any(c["level"] == "must" and not c["passed"] for c in checks):
        return "fail"
    if any(c["level"] == "soft" and not c["passed"] for c in checks):
        return "warn"
    return "pass"


def soft_warnings(checks: list[Check]) -> list[str]:
    return [c["name"] for c in checks if c["level"] == "soft" and not c["passed"]]


def assemble_result(
    patient: str, checks: list[Check], metadata: dict, workarounds: list[str]
) -> AuditResult:
    return {
        "patient": patient,
        "verdict": compute_verdict(checks),
        "checks": checks,
        "workarounds": workarounds,
        "metadata": metadata,
    }
