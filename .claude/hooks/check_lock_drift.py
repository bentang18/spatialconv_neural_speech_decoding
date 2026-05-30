#!/usr/bin/env python3
"""PreToolUse hook: block Claude from (re)introducing superseded v14 lock
literals into the active source tree.

The failure mode this guards against: "the code is stale 5/22 scaffold, not
the current lock." Every rule below pairs a *superseded* literal with the
canonical memo that retired it. When a Claude Write/Edit/MultiEdit/NotebookEdit
to ``src/speech_decoding/**.py`` introduces one of those literals AND the
surrounding lines carry no historical / sister-cell marker, the hook exits 2
and tells Claude which lock the literal violates so it corrects rather than
ratifies the stale value.

Scope: ``src/speech_decoding/`` ``.py`` files only (source + colocated tests).
Docs, memos, and the blockers table legitimately enumerate superseded
literals, so they are out of scope.

Exemptions (a match is allowed when its ±2-line window contains any of these):
  * a sister-cell name (``R-<something>``) or the word ``sister``
  * historical framing — ``was``, ``were``, ``previous``, ``prior``,
    ``superseded``, ``retired``, ``deprecated``, ``legacy``, ``formerly``,
    ``no longer``, ``dropped``, ``pre-B``, ``predate``, ``merged``,
    ``collapsed``, ``restore``, ``lineage``, ``provenance``, ``historical``,
    ``not the default`` / ``NOT the default``, ``override``
  * an explicit ``# drift-ok`` annotation on the offending line

CLI (manual / CI full-tree audit, not used by the hook path):
  ``.claude/hooks/check_lock_drift.py --scan-tree``  → walk src/, exit 1 on
  any non-exempt violation. ``--scan <file>...`` → check specific files.

Sibling guard: ``.claude/hooks/paper_guard.py`` (paper-prose integrity).
"""
from __future__ import annotations

import json
import os
import re
import sys

SRC_SCOPE = "src/speech_decoding/"
SELF_NAME = "check_lock_drift.py"

# Each rule: (id, regex, canonical memo, must_also_contain, skip_if_line_contains).
# ``must_also_contain`` (optional, lowercased): the matched LINE must contain at
# least one of these for the rule to fire — stops the cross-attn ``[0, 3]`` rule
# from tripping on plain array indexing.
# ``skip_if_line_contains`` (optional, lowercased): if the matched LINE contains
# any of these, it is the legitimate *configurable knob* (not a stale default
# claim) — e.g. ``cross_attn_positions=[0, 3]`` is how the sister is exercised.
_RULES_RAW = [
    (
        "B29-m1-slots",
        r"\ball 320 slots?\b|\b320[- ]slots?\b",
        "project_v14_b29_joint_default_2026_05_27 (Item 13: M=1 → 80 slots)",
        None,
        None,
    ),
    (
        "B29-m1-default",
        r"\bM\s*=\s*4\b",
        "project_v14_b29_joint_default_2026_05_27 (Item 13: default M=1, not M=4)",
        None,
        None,
    ),
    (
        "B28-one-cross-attn",
        r"\{0,\s*3\}|\[0,\s*3\]",
        "project_v14_loss_design_amendment_b28_2026_05_27 (Item 2: 1 cross-attn @ {0})",
        ("cross", "attn", "perceiver"),
        ("cross_attn_positions",),
    ),
    (
        "B20-frame-rate",
        r"14\.7\s*hz",
        "project_v14_v4_invisible_frontend_lock_2026_05_24 (hop=256 → 8 Hz; '14.7 Hz' was a mislabel)",
        None,
        None,
    ),
    (
        "B30-latent-valid",
        r"parcels_supervised\[\s*subject\s*\]",
        "project_v14_anatomy_gated_symmetric_2026_05_28 (B30: latent_valid is the gate SSOT)",
        None,
        None,
    ),
    (
        "loss-composer-module",
        r"\bcompose\.py\b",
        "ssl/total_loss.py + ssl/aggregator.py (no 'compose.py' module exists)",
        None,
        None,
    ),
]

RULES = [
    (rid, re.compile(pat, re.IGNORECASE), memo, must, skip)
    for rid, pat, memo, must, skip in _RULES_RAW
]

EXEMPTION_MARKERS = (
    "sister",
    "was ",
    "were ",
    "previous",
    "prior",
    "superseded",
    "supersedes",
    "retired",
    "deprecated",
    "legacy",
    "formerly",
    "no longer",
    "dropped",
    "pre-b",
    "predate",
    "merged",
    "collapsed",
    "restore",
    "lineage",
    "provenance",
    "historical",
    "not the default",
    "override",
    "mislabel",
    "collapse",
    "m=1",
    "latent_valid",
    "drift-ok",
)

_SISTER_RE = re.compile(r"\bR-[a-z0-9]", re.IGNORECASE)


def _extract_added(tool: str, tool_input: dict) -> str:
    if tool == "Edit":
        return tool_input.get("new_string", "")
    if tool == "Write":
        return tool_input.get("content", "")
    if tool == "MultiEdit":
        edits = tool_input.get("edits") or []
        return "\n".join(e.get("new_string", "") for e in edits)
    if tool == "NotebookEdit":
        return tool_input.get("new_source", "")
    return ""


def _is_exempt(lines: list[str], idx: int) -> bool:
    lo = max(0, idx - 2)
    hi = min(len(lines), idx + 3)
    window = "\n".join(lines[lo:hi]).lower()
    if any(marker in window for marker in EXEMPTION_MARKERS):
        return True
    return bool(_SISTER_RE.search("\n".join(lines[lo:hi])))


def scan_text(text: str) -> list[tuple[str, str, str, str]]:
    """Return [(rule_id, literal, memo, line_text)] for non-exempt matches."""
    lines = text.splitlines()
    out: list[tuple[str, str, str, str]] = []
    for idx, line in enumerate(lines):
        low = line.lower()
        for rid, rx, memo, must, skip in RULES:
            if must and not any(tok in low for tok in must):
                continue
            if skip and any(tok in low for tok in skip):
                continue
            m = rx.search(line)
            if not m:
                continue
            if _is_exempt(lines, idx):
                continue
            out.append((rid, m.group(0), memo, line.strip()))
    return out


def _report(violations: list[tuple[str, str, str, str]], where: str) -> str:
    head = (
        f"[lock-drift guard] BLOCKED — {len(violations)} superseded v14 lock "
        f"literal(s) in {where}:\n"
    )
    body = []
    for rid, lit, memo, line in violations:
        body.append(
            f"  • [{rid}] '{lit}'  →  canonical lock: {memo}\n"
            f"      line: {line[:140]}"
        )
    tail = (
        "\nCorrect the literal to the current lock. If this is a deliberate "
        "historical / sister-cell reference, give it a context marker on a "
        "nearby line (e.g. 'sister', the R-<name>, 'was', 'superseded') or "
        "append '# drift-ok' to the line."
    )
    return head + "\n".join(body) + tail


def _scan_tree_mode(paths: list[str]) -> int:
    if not paths:
        root = os.path.join(SRC_SCOPE)
        paths = []
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if f.endswith(".py") and f != SELF_NAME:
                    paths.append(os.path.join(dirpath, f))
    total = 0
    for p in sorted(paths):
        try:
            with open(p, encoding="utf-8") as fh:
                text = fh.read()
        except OSError:
            continue
        v = scan_text(text)
        if v:
            total += len(v)
            print(_report(v, p), file=sys.stderr)
    if total:
        print(f"\n[lock-drift guard] {total} violation(s) across {len(paths)} files.", file=sys.stderr)
        return 1
    print(f"[lock-drift guard] clean — {len(paths)} files scanned, 0 violations.")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] in {"--scan-tree", "--scan"}:
        return _scan_tree_mode(argv[1:])

    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool = payload.get("tool_name", "")
    if tool not in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
        return 0

    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if SRC_SCOPE not in path or not path.endswith(".py") or path.endswith(SELF_NAME):
        return 0

    added = _extract_added(tool, tool_input)
    if not added.strip():
        return 0

    violations = scan_text(added)
    if violations:
        print(_report(violations, path), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
