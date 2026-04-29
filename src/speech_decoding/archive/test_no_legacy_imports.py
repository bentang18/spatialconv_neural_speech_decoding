"""CI guard: no active module may import from `src/speech_decoding/archive/`.

All quarantined code (pre-v14 baselines, stage-2 stubs, parity oracles,
exploration prototypes) lives under `src/speech_decoding/archive/`. Active
code must not silently reuse anything from there. Rewrite fresh in the
active subpackages instead.

This test AST-parses every `.py` file under `src/speech_decoding/` (excluding
`archive/`) and rejects any `import` or `from ... import` targeting
`speech_decoding.archive.*`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
_ARCHIVE_DIR = _PKG_ROOT / "archive"
_REPO_ROOT = _PKG_ROOT.parent.parent

_LEGACY_PREFIX = "speech_decoding.archive"


def _is_legacy(module: str) -> bool:
    return module == _LEGACY_PREFIX or module.startswith(_LEGACY_PREFIX + ".")


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
                continue
            module = node.module or ""
            if _is_legacy(module):
                offenders.append((node.lineno, module))
    return offenders


_ACTIVE_SOURCES = sorted(
    p for p in _PKG_ROOT.rglob("*.py") if _ARCHIVE_DIR not in p.parents
)


@pytest.mark.parametrize(
    "source_path",
    _ACTIVE_SOURCES,
    ids=[p.relative_to(_PKG_ROOT).as_posix() for p in _ACTIVE_SOURCES],
)
def test_active_module_has_no_archive_imports(source_path: Path) -> None:
    offenders = _legacy_import_sites(source_path)
    assert not offenders, (
        f"{source_path.relative_to(_REPO_ROOT)} imports archived path(s) "
        f"{offenders}. Active code must not import from "
        f"`speech_decoding.archive.*`. Rewrite fresh in the active subpackages."
    )


def test_active_tree_is_non_empty() -> None:
    assert _ACTIVE_SOURCES, f"no active python sources found under {_PKG_ROOT}"
