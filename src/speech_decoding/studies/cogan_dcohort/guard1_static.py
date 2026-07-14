"""Guard-1 static-bad accessor for the Cogan D-cohort — the Cogan analogue of
``braintreebank.anatomy.extra_bad_electrodes``.

THE single source ``DCohortStudy.iter_timelines`` consults to fold each run's
guard-1 STATIC drop into its timeline (so the raw exca cache uid depends on the
drop, and ``_load_raw`` drops exactly those contacts pre-CAR). Mirrors BT's
accessor shape (a module function, per-run REPLACE semantics, empty fallback) but
reads a JSON MAP instead of a Python literal — a 938-entry (RAM: thousands)
literal does not scale, and the collector already emits this exact JSON.

Map schema = ``collect_guard1_static_cogan.py`` output (its ``per_run`` block):
    {"per_run": {"{global_subject_id}_{trial_id}": [label, ...], ...}, ...}
Labels are the loader's already-clean neural ``ch_names`` (the guard-1 scan tagged
them from ``cogan_load_raw``), so NO label cleaning is applied — verbatim match.

WHERE THE MAP LIVES (Ben-gated, governance): default is a package-relative
``guard1_static.json`` beside this module (git-tracked, BT-style provenance) — but
Cogan is IRB-restricted, so if the derived map may NOT live in the public repo,
set ``COGAN_GUARD1_STATIC`` to a ``/work`` path instead. Until the map is placed
(pre-array), the file is ABSENT → every lookup returns empty → NO static drop and
NO cache-key change (safe to land the wiring before the scan runs). A warning
fires ONCE so a cache-bake run without a committed map is never silent.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_VAR = "COGAN_GUARD1_STATIC"
_DEFAULT_PATH = Path(__file__).with_name("guard1_static.json")


def _map_path() -> Path:
    override = os.environ.get(_ENV_VAR)
    return Path(override) if override else _DEFAULT_PATH


@lru_cache(maxsize=1)
def _load_map(path_str: str) -> dict[str, list[str]]:
    """Load the ``per_run`` map from ``path_str``; absent file → empty (warn once)."""
    path = Path(path_str)
    if not path.exists():
        logger.warning(
            "guard-1 static map absent at %s (set %s or place the collector output); "
            "no static drops will be applied", path, _ENV_VAR,
        )
        return {}
    report = json.loads(path.read_text())
    per_run = report.get("per_run", report)  # accept the full report or a bare map
    return {str(k): list(v) for k, v in per_run.items()}


def cogan_extra_bad(subject_id: int, trial_id: int) -> frozenset[str]:
    """Guard-1 STATIC contact labels for one Cogan run (empty if none/unmapped)."""
    per_run = _load_map(str(_map_path()))
    return frozenset(per_run.get(f"{int(subject_id)}_{int(trial_id)}", ()))
