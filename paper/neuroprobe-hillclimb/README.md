# v14 Paper — Neuroprobe Hillclimb

This directory holds the v14 preprint/paper draft.

## Authorship rule (IRONCLAD)

Ben drafts the bulk of the paper. Claude does NOT bulk-draft prose. We iterate
piece-by-piece, and every citation, number, and empirical claim Claude touches
is rigorously verified against a real source.

Mechanical guard: `.claude/hooks/paper_guard.py` fires on every Claude
Write/Edit/MultiEdit/NotebookEdit under this directory.

- Hard-block: meta-LLM phrases ("as an AI", "[INSERT CITATION]", etc.).
- Hard-block: net additions > 25 lines (bulk drafting cap).
- Force-ask: every clean edit triggers the permission prompt so Ben sees the
  diff regardless of auto-accept mode.

No env-var bypass. For a legitimate large edit, Ben writes it directly via his
editor — the hook only intercepts Claude's tool calls.

Full rule: `memory/feedback_arxiv_llm_content_responsibility_2026_05_16.md`.
CLAUDE.md section: "v14 Paper Authorship — IRONCLAD".
