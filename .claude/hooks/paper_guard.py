#!/usr/bin/env python3
"""PreToolUse hook for the v14 paper directory (paper/neuroprobe-hillclimb/).

THREE LAYERS — all IRONCLAD. No env-var bypass. If Ben needs a legitimate
large or unconventional edit, he writes it directly via his editor — the hook
only fires on Claude's Write/Edit/MultiEdit/NotebookEdit tool calls.

  L2a  Meta-LLM phrase block. Hard-rejects writes containing model-tells
       ("as an AI", "[INSERT CITATION]", "would you like me to",
       "illustrative numbers", etc.). Exit 2; reason surfaces to Claude.

  L2b  Bulk-drafting cap. Hard-rejects writes whose net additions exceed
       BULK_LIMIT lines. Ben drafts the bulk; Claude is surgical only.

  L5   Force-ask permission prompt. Writes that pass L2a + L2b emit
       hookSpecificOutput.permissionDecision = "ask" so the diff is shown
       in the permission dialog regardless of auto-accept mode. Ben MUST
       visually approve every diff. Cite keys + decimal numbers are listed
       in the dialog reason for fast scanning.

Full rule:  memory/feedback_arxiv_llm_content_responsibility_2026_05_16.md
Floor:      arXiv policy (Dietterich 2026-05-15) — un-checked LLM output →
            1-year ban + peer-review requirement for subsequent submissions.
Ceiling:    paper is Ben's — voice, framing, claim structure come from Ben.
"""
import json
import re
import sys


PAPER_DIR = "paper/neuroprobe-hillclimb/"
BULK_LIMIT = 25

META_LLM_PATTERNS = [
    r"\bas an AI(\s+(language\s+)?model)?\b",
    r"\bI('| a)m an AI\b",
    r"\bas a language model\b",
    r"\bwould you like me to\b",
    r"\bhere\s+is\s+(a|an|the)?\s*\d+[\s-]?word\s+summary\b",
    r"\[INSERT\b",
    r"\[CITATION\s+(HERE|TBD|NEEDED)\]?",
    r"\billustrative\s+(values?|numbers?|data|example)s?\b",
    r"\bplaceholder\s+(values?|numbers?|text)s?\b",
    r"\bfill\s+(in|with)\s+(the\s+)?real\b",
    r"\[TODO:?\s+(verify|cite|cit|check)",
]


def _extract_diff(tool: str, tool_input: dict) -> tuple[str, str]:
    if tool == "Edit":
        return tool_input.get("new_string", ""), tool_input.get("old_string", "")
    if tool == "Write":
        return tool_input.get("content", ""), ""
    if tool == "MultiEdit":
        edits = tool_input.get("edits") or []
        added = "\n".join(e.get("new_string", "") for e in edits)
        removed = "\n".join(e.get("old_string", "") for e in edits)
        return added, removed
    if tool == "NotebookEdit":
        return tool_input.get("new_source", ""), tool_input.get("old_source", "")
    return "", ""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    tool = payload.get("tool_name", "")
    if tool not in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
        return 0

    tool_input = payload.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if PAPER_DIR not in path:
        return 0

    added, removed = _extract_diff(tool, tool_input)
    if not added:
        return 0

    # L2a — meta-LLM phrase block
    for pat in META_LLM_PATTERNS:
        m = re.search(pat, added, re.IGNORECASE)
        if m:
            print(
                f"[paper guard] BLOCKED — meta-LLM phrase '{m.group(0)}' "
                f"(regex /{pat}/) in write to {path}.\n"
                "v14 paper directory rejects LLM-tells unconditionally. "
                "Remove the phrase and re-attempt.",
                file=sys.stderr,
            )
            return 2

    # L2b — bulk-drafting cap
    added_lines = added.count("\n") + (1 if added.strip() else 0)
    removed_lines = removed.count("\n") + (1 if removed.strip() else 0)
    net = added_lines - removed_lines
    if net > BULK_LIMIT:
        print(
            f"[paper guard] BLOCKED — bulk drafting: net +{net} lines "
            f"exceeds the {BULK_LIMIT}-line cap on writes to {path}.\n"
            "Ben drafts the bulk of the v14 paper; Claude is surgical only. "
            "For a legitimate large edit, Ben writes it directly.",
            file=sys.stderr,
        )
        return 2

    # L5 — force permission ask (MUST SHOW DIFF, even under auto-accept)
    cites = sorted(set(re.findall(r"\\cite[pt]?\*?\{([^}]+)\}", added)))
    nums = sorted(set(re.findall(r"(?<!\d)\d+\.\d{2,}(?!\d)", added)))
    reason_parts = ["v14 paper edit — verify every cite + number against source"]
    if cites:
        reason_parts.append(f"cites: {', '.join(cites)[:200]}")
    if nums:
        reason_parts.append(f"numbers: {', '.join(nums)[:200]}")

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": " | ".join(reason_parts),
        }
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
