"""Tripwire: no /work/ht203 write paths in active source.

`/work/ht203` is DCC's 75-day auto-purge scratch. Caches, checkpoints, and
experiment artifacts that need to survive a sweep must go through
`EXCA_CACHE_FOLDER` (configured to `/hpc/group/coganlab/ht203/cache_neuroai/`)
or another persistent path. Hard-coding `/work/ht203` in source silently sets
experiments on a clock — by the time the analysis runs, the data is gone.

This test scans the package for that string and fails if found. Reference
paths in docs/CLAUDE.md or in the `.cache/` clone are exempt — only source
under `src/speech_decoding/` is checked.
"""

from __future__ import annotations

from pathlib import Path

import speech_decoding

PURGE_PREFIX = "/work/ht203"


def test_no_purge_paths_in_src() -> None:
    pkg_root = Path(speech_decoding.__file__).resolve().parent
    offenders: list[tuple[Path, int, str]] = []
    for path in pkg_root.rglob("*.py"):
        # Skip this guard file itself.
        if path.name == "test_no_purge_paths.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if PURGE_PREFIX in line:
                offenders.append((path, lineno, line.strip()))
    assert not offenders, (
        f"{len(offenders)} source line(s) reference `{PURGE_PREFIX}`, which auto-purges "
        f"after 75 days. Route persistent cache through EXCA_CACHE_FOLDER instead.\n"
        + "\n".join(f"  {p}:{n}: {s}" for p, n, s in offenders)
    )
