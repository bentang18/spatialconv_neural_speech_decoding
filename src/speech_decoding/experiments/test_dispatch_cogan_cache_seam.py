"""Drift guard for the ``--study cogan`` spec-cache seam in dispatch_v14.

The Cogan D-cohort bake (``build_cogan_cache_experiment`` /
``_cogan_common_fe_kwargs`` / ``_COGAN_CACHE_BANDS``) is a HAND-COPIED mirror of the
front-end that ``build_v14_experiment`` constructs for a BT ``--cache-band`` build.
It is copied — not shared — so a Cogan run never traverses BT-specific pre-setup
(which reads bt_root anatomy). The copy MUST stay byte-identical to BT's, or the
Cogan cache would diverge from a BT v3 cache and the v3 consumer would mis-read it.

These tests pin the copy to the BT source STRUCTURALLY: they parse the literal
``common_fe_kwargs = dict(...)`` and the ``band_const = {...}[cache_band]`` map out of
``build_v14_experiment`` via ``ast`` and assert the Cogan mirror reproduces them. If
anyone edits BT's front-end (a filter cutoff, a band constant, a new kwarg), the
extracted AST values change and these tests fail — forcing the Cogan mirror to be
updated in lockstep. No neuralset/torch/study construction is needed (pure source +
the module's plain-dict band constants), so this runs anywhere.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from speech_decoding.experiments import dispatch_v14 as d


# ---- AST extraction of BT's front-end literals --------------------------------

_SRC = Path(inspect.getsourcefile(d)).read_text()
_TREE = ast.parse(_SRC)


def _func_node(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in dispatch_v14")


def _literal(node: ast.AST):
    """Reduce a constant / tuple-of-constants AST node to its Python value; return
    the sentinel ('name', id) for a bare Name reference (an arg-dependent kwarg)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_literal(e) for e in node.elts)
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand)
    raise AssertionError(f"unhandled AST literal node: {ast.dump(node)}")


def _extract_bt_common_fe_kwargs() -> dict:
    """The keyword args of ``common_fe_kwargs = dict(...)`` inside build_v14_experiment,
    as {name: literal-value-or-('name', ref)}."""
    fn = _func_node("build_v14_experiment")
    for node in ast.walk(fn):
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign)
            else getattr(node, "targets", [])
        )
        if any(isinstance(t, ast.Name) and t.id == "common_fe_kwargs" for t in targets):
            call = node.value
            assert isinstance(call, ast.Call) and getattr(call.func, "id", None) == "dict", (
                "common_fe_kwargs is expected to be a dict(...) call"
            )
            return {kw.arg: _literal(kw.value) for kw in call.keywords}
    raise AssertionError("common_fe_kwargs assignment not found in build_v14_experiment")


def _extract_bt_band_map() -> dict[str, str]:
    """The ``band_const = {<band>: <CONST>}[cache_band]`` map inside build_v14_experiment,
    as {band_name: constant_name}."""
    fn = _func_node("build_v14_experiment")
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "band_const" for t in node.targets)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Dict)
        ):
            dct = node.value.value
            out = {}
            for k, v in zip(dct.keys, dct.values):
                assert isinstance(k, ast.Constant) and isinstance(v, ast.Name)
                out[k.value] = v.id
            return out
    raise AssertionError("band_const map not found in build_v14_experiment")


_BT_FE = _extract_bt_common_fe_kwargs()
_BT_BANDS = _extract_bt_band_map()

# The v3 bands the Cogan bake supports (the BT map also has the older 3stft/2band
# names, which Cogan never bakes).
_V3_BANDS = ("v3slow", "v3mid", "hga")


# ---- 1. the front-end kwargs mirror -------------------------------------------

def test_cogan_common_fe_kwargs_matches_bt_literals():
    """Every LITERAL kwarg (corpus-agnostic front-end config) is identical between
    BT's common_fe_kwargs and the Cogan mirror."""
    cogan = d._cogan_common_fe_kwargs(
        notch_hz=60.0, c_max=128, session_robust_z=True, spec_only=True,
    )
    # keys must match exactly — no kwarg added/dropped on either side
    assert set(cogan) == set(_BT_FE), (
        f"kwarg key drift: cogan-only={set(cogan) - set(_BT_FE)}, "
        f"bt-only={set(_BT_FE) - set(cogan)}"
    )
    # literal (non-Name) BT kwargs must equal the Cogan value byte-for-byte
    for key, bt_val in _BT_FE.items():
        if isinstance(bt_val, tuple) and bt_val and bt_val[0] == "name":
            continue  # arg-dependent (notch/c_max/z/spec_only) — checked below
        assert cogan[key] == bt_val, f"{key}: cogan={cogan[key]!r} != bt-literal={bt_val!r}"


def test_cogan_arg_dependent_kwargs_are_the_expected_refs():
    """The arg-dependent BT kwargs are exactly notch_filter/c_max/session_robust_z/
    spec_only — and the Cogan mirror threads the SAME four through its signature."""
    bt_name_refs = {
        k: v[1] for k, v in _BT_FE.items()
        if isinstance(v, tuple) and v and v[0] == "name"
    }
    assert bt_name_refs == {
        "notch_filter": "effective_bt_notch_hz",
        "c_max": "c_max",
        "session_robust_z": "session_robust_z",
        "spec_only": "spec_only",
    }, f"BT arg-dependent kwargs changed: {bt_name_refs}"
    # Cogan passes notch through the notch_hz param; the value lands on notch_filter.
    cogan = d._cogan_common_fe_kwargs(
        notch_hz=50.0, c_max=99, session_robust_z=False, spec_only=False,
    )
    assert cogan["notch_filter"] == 50.0
    assert cogan["c_max"] == 99
    assert cogan["session_robust_z"] is False
    assert cogan["spec_only"] is False


# ---- 2. the band map mirror ----------------------------------------------------

def test_cogan_band_map_matches_bt_for_v3_bands():
    """For each v3 band, the Cogan _COGAN_CACHE_BANDS entry is the SAME STFT constant
    object BT's band_const map selects."""
    for band in _V3_BANDS:
        assert band in _BT_BANDS, f"BT band map lost {band!r}"
        bt_const_name = _BT_BANDS[band]
        bt_const = getattr(d, bt_const_name)
        assert d._COGAN_CACHE_BANDS[band] is bt_const, (
            f"{band}: cogan maps to a different object than BT's {bt_const_name}"
        )


def test_cogan_band_map_covers_exactly_the_v3_bands():
    assert set(d._COGAN_CACHE_BANDS) == set(_V3_BANDS)


# What the BAKE would write, per band: hop_length=int(band_const["band_hop"]).
# dbd95e9 (2026-07-21) moved SLOW 64→512 (4 Hz) and MID 64→128 (16 Hz) for the
# native-rate fine-HGA rebake. That rebake has NOT been run — every live cache is
# still hop=64 — so this map is the DECLARED bake, not the cache on disk. The two
# seams are pinned separately below.
_V3_DECLARED_HOP = {"v3slow": 512, "v3mid": 128, "hga": 64}

# What the v3 READ path accepts: cache_index.resolve_band_leaf is locked to this hop
# and fails loud on any other leaf, so this is the rate every trained v3 run has seen.
_V3_LIVE_CACHE_HOP = 64


def test_v3_band_constants_have_band_hop_matching_the_declared_bake():
    """The bake does hop_length=int(band_const['band_hop']); pin each v3 band's declared
    hop so a Cogan/BT drift or an unannounced re-hop can't land silently."""
    for band in _V3_BANDS:
        const = d._COGAN_CACHE_BANDS[band]
        assert "band_hop" in const, f"{band} missing band_hop"
        assert int(const["band_hop"]) == _V3_DECLARED_HOP[band], (
            f"{band} band_hop {const['band_hop']} != declared {_V3_DECLARED_HOP[band]}"
        )


def test_declared_bake_hop_divergence_from_the_live_read_lock_is_explicit():
    """SLOW/MID declare a native-rate bake the live cache does not have, and the reader is
    hard-locked to hop=64. Nothing downstream reconciles them: declaring a rate the cache
    lacks does NOT decimate — V3ClipDataset rescales the read INDEX, so an over-declared
    band returns a contiguous, TIME-SHIFTED slice at the right shape (the r6 bug, memo
    project-r6-band-rates-cache-rate-bug-2026-07-23; four runs trained on it).

    So this test states the seam rather than hiding it: any band whose declared bake hop
    differs from the read lock is a band that CANNOT be rebaked without updating
    ``resolve_band_leaf``'s ``band_hop`` default and the frontend's band_rates together.
    """
    from speech_decoding.models.v14_converged_v3.cache_index import resolve_band_leaf

    lock = inspect.signature(resolve_band_leaf).parameters["band_hop"].default
    assert lock == _V3_LIVE_CACHE_HOP, (
        f"read path now locks band_hop={lock}, not {_V3_LIVE_CACHE_HOP} — if the native-rate "
        "rebake landed, update _V3_LIVE_CACHE_HOP and every frontend's band_rates in lockstep"
    )
    pending = {b: h for b, h in _V3_DECLARED_HOP.items() if h != _V3_LIVE_CACHE_HOP}
    assert pending == {"v3slow": 512, "v3mid": 128}, (
        f"bake/read divergence changed: {pending}. This set must only ever shrink (by rebaking "
        "and re-locking) — a NEW entry means a band was re-hopped without a cache to match."
    )


# ---- 3. builder input-validation guards (fire before any study construction) ---

def test_builder_rejects_unknown_band():
    with pytest.raises(SystemExit, match="not a v3 band"):
        d.build_cogan_cache_experiment(
            cache_band="beta", cache_session_index=0, spec_cache_dir="/x",
            cogan_manifest="m.csv", notch_hz=60.0, c_max=128,
            session_robust_z=True, spec_only=True, clip_len=1.0,
        )


def test_builder_requires_manifest():
    with pytest.raises(SystemExit, match="requires --cogan-manifest"):
        d.build_cogan_cache_experiment(
            cache_band="v3slow", cache_session_index=0, spec_cache_dir="/x",
            cogan_manifest=None, notch_hz=60.0, c_max=128,
            session_robust_z=True, spec_only=True, clip_len=1.0,
        )


def test_builder_requires_spec_cache_dir():
    with pytest.raises(SystemExit, match="requires --spec-cache-dir"):
        d.build_cogan_cache_experiment(
            cache_band="v3slow", cache_session_index=0, spec_cache_dir=None,
            cogan_manifest="m.csv", notch_hz=60.0, c_max=128,
            session_robust_z=True, spec_only=True, clip_len=1.0,
        )


def test_builder_requires_session_index():
    with pytest.raises(SystemExit, match="requires --cache-session-index"):
        d.build_cogan_cache_experiment(
            cache_band="v3slow", cache_session_index=None, spec_cache_dir="/x",
            cogan_manifest="m.csv", notch_hz=60.0, c_max=128,
            session_robust_z=True, spec_only=True, clip_len=1.0,
        )


# ---- 4. argparse wiring --------------------------------------------------------

def test_parser_study_flag_defaults_bt_and_accepts_cogan():
    p = d._parser()
    base = ["--phase", "1", "--mode", "full"]
    assert p.parse_args(base).study == "bt"
    ns = p.parse_args(base + ["--study", "cogan"])
    assert ns.study == "cogan"


def test_parser_cogan_notch_default_is_duke_mains():
    """--cogan-notch-hz defaults to 60 Hz — Duke/US mains, same as BT's
    effective_bt_notch_hz, so a default Cogan bake notches identically to BT."""
    p = d._parser()
    ns = p.parse_args(["--phase", "1", "--mode", "full"])
    assert ns.cogan_notch_hz == 60.0
    assert ns.cogan_manifest is None


def test_parser_rejects_non_enum_study():
    p = d._parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--phase", "1", "--mode", "full", "--study", "bogus"])
