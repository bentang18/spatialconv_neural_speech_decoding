"""Label derivation for BrainTreebank's 15-task Neuroprobe suite.

Populated by Stage-0 Block E3. The 15 tasks span sentence-onset / pitch /
volume / speaker / GPT-2 surprisal etc. For now the module is a stub so
`BraintreebankStudy._load_timeline_events()` can grow into it incrementally.
"""

from __future__ import annotations

NEUROPROBE_TASKS: tuple[str, ...] = ()
"""Names of the 15 Neuroprobe downstream tasks. Filled by Stage-0 E3."""
