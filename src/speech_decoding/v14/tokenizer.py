"""Shared temporal tokenizer for Neural Field Perceiver v14.

Implementation intentionally deferred. The exact temporal layer is still
an open blocker (`docs/implementation_tasks.md` #2 and #6) — the current
candidates are a simple Conv1d patch embed, a small dilated temporal CNN,
or another minimal shared encoder. No choice has been made. This file
will own whichever shared temporal front-end is agreed, along with the
explicit output contract (tensor shape, token rate, receptive field)
required by #6 before any downstream code consumes it.
"""

