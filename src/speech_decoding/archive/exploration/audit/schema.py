"""Check / AuditResult schema.

Uses plain dicts for JSON-ready serialization. Each `Check` carries:
- `name`: short identifier
- `level`: `"must"` | `"soft"`  — only `must` checks gate the verdict
- `passed`: bool
- `detail`: human-readable string
"""
from __future__ import annotations

from typing import Literal, TypedDict

Level = Literal["must", "soft"]
Verdict = Literal["pass", "warn", "fail"]


class Check(TypedDict):
    name: str
    level: Level
    passed: bool
    detail: str


class AuditResult(TypedDict):
    patient: str
    verdict: Verdict
    checks: list[Check]
    workarounds: list[str]
    metadata: dict


def make_check(name: str, level: Level, passed: bool, detail: str) -> Check:
    return {"name": name, "level": level, "passed": passed, "detail": detail}
