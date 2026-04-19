"""CI guard: no module under `src/speech_decoding/v14/` may import legacy code.

The archived pre-v14 tree lives under `src/speech_decoding/archive/legacy/` and
adjacent stale paths. v14 must not silently reuse anything from there. Rewriting
fresh under `src/speech_decoding/v14/` is the only sanctioned path; if you think
a legacy helper is load-bearing, re-derive it from the Phase-1 contract instead
of importing the quarantine.

This test AST-parses every `.py` file under `src/speech_decoding/v14/` and
rejects any `import` or `from ... import` targeting a known-legacy prefix.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_V14_DIR = _REPO_ROOT / "src" / "speech_decoding" / "v14"

# Any module whose dotted path starts with one of these (exact match or with a
# trailing dot) is quarantined. Expand as new legacy paths appear.
_LEGACY_PREFIXES: tuple[str, ...] = (
    "speech_decoding.archive",
    "speech_decoding.models",
    "speech_decoding.training",
    "speech_decoding.pretraining",
    "speech_decoding.data.atlas",
    "speech_decoding.data.coordinates",
    "speech_decoding.data.bids_dataset",
    "speech_decoding.data.grid",
    "speech_decoding.data.augmentation",
    "speech_decoding.data.collate",
    "speech_decoding.data.sig_channels",
    "speech_decoding.data.audio_features",
    "speech_decoding.evaluation.metrics",
    "speech_decoding.evaluation.content_collapse",
)


def _is_legacy(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in _LEGACY_PREFIXES)


def _legacy_import_sites(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_legacy(alias.name):
                    offenders.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level != 0:
                continue  # relative imports stay inside v14 by construction
            module = node.module or ""
            if _is_legacy(module):
                offenders.append((node.lineno, module))
    return offenders


_V14_SOURCES = sorted(_V14_DIR.rglob("*.py"))


@pytest.mark.parametrize(
    "source_path",
    _V14_SOURCES,
    ids=[p.relative_to(_V14_DIR).as_posix() for p in _V14_SOURCES],
)
def test_v14_module_has_no_legacy_imports(source_path: Path) -> None:
    offenders = _legacy_import_sites(source_path)
    assert not offenders, (
        f"{source_path.relative_to(_REPO_ROOT)} imports legacy path(s) "
        f"{offenders}. v14 must not reuse pre-pivot code. Rewrite fresh "
        f"under src/speech_decoding/v14/ per docs/objectives.md."
    )


def test_v14_directory_is_non_empty() -> None:
    """Fail loudly if v14 gets deleted or emptied, so the guard above never
    passes vacuously on an empty rglob."""
    assert _V14_SOURCES, f"no python sources found under {_V14_DIR}"
