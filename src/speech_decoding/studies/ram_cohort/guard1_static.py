"""Guard-1 static-bad accessor for the RAM cohort — RAM analogue of
``cogan_dcohort.guard1_static``.

THE single source ``RamCohortStudy.iter_timelines`` consults to fold each run's
drop set into its timeline (so the raw exca cache uid depends on it, and
``_load_raw`` drops exactly those contacts pre-CAR). For RAM this set carries BOTH
guard-1 static-bad contacts AND the micro-electrode names (electrodes.tsv
``type == micro``, ECOG/SEEG-typed with a trailing index ⇒ not caught by the
neural whitelist) — both must leave before group-CAR.

Map schema (collector output, ``per_run`` block):
    {"per_run": {"{global_subject_id}_{trial_id}": [label, ...], ...}, ...}
Labels are the loader's already-clean neural ``ch_names`` — verbatim match, no
cleaning.

WHERE THE MAP LIVES: default is a package-relative ``guard1_static.json`` beside
this module; RAM is CC0 so the map MAY live in the repo, but set
``RAM_GUARD1_STATIC`` to a ``/work`` path to keep it out. Until the map is placed
the file is ABSENT → every lookup returns empty → NO drop and NO cache-key change
(safe to land the wiring before the scan runs). Warns once so a bake without a
map is never silent.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_VAR = "RAM_GUARD1_STATIC"
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
            "RAM guard-1 static map absent at %s (set %s or place the collector "
            "output); no static drops will be applied", path, _ENV_VAR,
        )
        return {}
    report = json.loads(path.read_text())
    per_run = report.get("per_run", report)  # accept the full report or a bare map
    return {str(k): list(v) for k, v in per_run.items()}


def ram_extra_bad(subject_id: int, trial_id: int) -> frozenset[str]:
    """Guard-1 static-bad + micro labels for one RAM run (empty if none/unmapped)."""
    per_run = _load_map(str(_map_path()))
    return frozenset(per_run.get(f"{int(subject_id)}_{int(trial_id)}", ()))
