"""V14 Stage-1 architecture sweep — one Slurm array per sweep via job_array.

Star design over (eps, M, d_model, depth) with first-pass defaults from
``memory/project_v14_encoder_design_2026_05_13.md`` as the center. Each
non-combinatorial cell differs from the center by exactly one knob; identical
center configs collapse onto a single exca UID (13 nominal cells → 10 unique
runs).

Combinatorial mode is gated behind ``--combinatorial`` (4 × 3 × 2 × 4 = 96
cells) and is intended for the Stage-1 advance gate, not the first dispatch.

We use ``infra.job_array()`` directly rather than ``neuraltrain.utils.run_grid``
because ``run_grid`` routes through ``exca.ConfDict`` whose UID hasher rejects
the arbitrary-type ``ns.Step`` study inside ``Data``. ``infra.job_array`` only
needs each per-cell ``Experiment`` instance to have a stable ``infra.uid()``,
which exca's ``@infra.apply`` produces correctly for pydantic models with
``arbitrary_types_allowed``.

DCC invocation (via ``scripts/dcc/dispatch``):

    scripts/dcc/dispatch -m scripts.neuroprobe.submit_v14_stage1_grid \\
        --mode lite --task frame_perception --cluster slurm

Smoke-test (laptop, no BT data + no Slurm):

    ROOT_DIR_BRAINTREEBANK=/tmp/dummy EXCA_CACHE_FOLDER=/tmp/exca \\
    .venv/bin/python -m scripts.neuroprobe.submit_v14_stage1_grid --dry-run

The grid axes are LOCKED in this file (not configurable from the command
line) — they ARE the Stage-1 spec. Edit the constants below + bump
``GRID_VERSION`` if the axes change after Stage-1 closes.
"""

from __future__ import annotations

import argparse
import itertools
import os
import typing as tp
from pathlib import Path

import speech_decoding.models  # noqa: F401  # registers V14ParcelPerceiver
from speech_decoding.experiments.dispatch_v14 import (
    DEFAULT_D_MODEL,
    DEFAULT_DEPTH,
    DEFAULT_M_SUB_SLOTS,
    DEFAULT_N_EPOCHS,
    DEFAULT_TASK,
    build_v14_experiment,
)
from speech_decoding.studies.braintreebank.anatomy import DEFAULT_SUPPORT_BIAS_EPS


GRID_VERSION = "1"
EXP_NAME = f"v14_stage1_eps_M_d_depth_v{GRID_VERSION}"


# Star-design axes — first value of each list = center (matches encoder memo).
# Center configs across axes collapse to a single exca UID.
STAGE1_AXES: dict[str, list] = {
    "eps": [DEFAULT_SUPPORT_BIAS_EPS, 1e-4, 1e-3, 1e-1],
    "m_sub_slots": [DEFAULT_M_SUB_SLOTS, 2, 8],
    "d_model": [DEFAULT_D_MODEL, 64],
    "depth_self_attn": [DEFAULT_DEPTH, 4, 8, 12],
}


def _star_cells() -> list[dict[str, tp.Any]]:
    """Star design: each cell differs from the center by exactly one axis."""
    center = {axis: vs[0] for axis, vs in STAGE1_AXES.items()}
    cells = [dict(center)]
    seen: set[tuple] = {tuple(sorted(center.items()))}
    for axis, vs in STAGE1_AXES.items():
        for v in vs[1:]:
            cell = dict(center)
            cell[axis] = v
            key = tuple(sorted(cell.items()))
            if key not in seen:
                cells.append(cell)
                seen.add(key)
    return cells


def _combinatorial_cells() -> list[dict[str, tp.Any]]:
    """Full Cartesian product across the four axes."""
    axes = list(STAGE1_AXES.keys())
    cells: list[dict[str, tp.Any]] = []
    for combo in itertools.product(*STAGE1_AXES.values()):
        cells.append({axis: val for axis, val in zip(axes, combo)})
    return cells


def cell_uid(cell: dict[str, tp.Any]) -> str:
    """Stable folder-safe identifier for one cell."""
    return "_".join(
        f"{axis}{cell[axis]}".replace(".", "p")
        for axis in ("d_model", "depth_self_attn", "m_sub_slots", "eps")
    )


def build_cell_experiment(
    *,
    cell: dict[str, tp.Any],
    args: argparse.Namespace,
    base_folder: Path,
):
    """Build one ``Experiment`` for a single cell of the sweep.

    Each cell reuses ``build_v14_experiment`` and routes per-cell artifacts into
    ``{base_folder}/{EXP_NAME}/{cell_uid}/``.
    """
    exca_folder = str(base_folder / EXP_NAME / cell_uid(cell))
    return build_v14_experiment(
        bt_root=args.bt_root,
        mode=args.mode,
        task=args.task,
        eps=cell["eps"],
        m_sub_slots=cell["m_sub_slots"],
        d_model=cell["d_model"],
        depth=cell["depth_self_attn"],
        n_epochs=args.n_epochs,
        seed=args.seed,
        exca_folder=exca_folder,
        cluster=args.cluster,
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="V14 Stage-1 (eps × M × d_model × depth) sweep — one Slurm array.",
    )
    p.add_argument("--mode", choices=("nano", "lite", "full"), default="lite")
    p.add_argument(
        "--task", default=DEFAULT_TASK,
        help="Neuroprobe task (single task; cross-task lift evaluated at Stage-1 close).",
    )
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS)
    p.add_argument(
        "--seed", type=int, default=33,
        help="Single seed for first dispatch. 2-seed expand-on-decision is a separate sweep.",
    )
    p.add_argument(
        "--combinatorial", action="store_true",
        help="Full Cartesian sweep (96 cells) instead of star design (10 unique cells). "
             "Gated behind Stage-1 advance gate — not the first dispatch.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build all per-cell Experiment instances and print summary; do not submit.",
    )
    p.add_argument(
        "--max-workers", type=int, default=256,
        help="Slurm array parallelism cap (job_array max_workers).",
    )
    p.add_argument("--bt-root", default=None)
    p.add_argument("--exca-folder", default=None,
                   help="Base cache folder. Falls back to $EXCA_CACHE_FOLDER. "
                        "On DCC use /hpc/group/coganlab/ht203/cache_neuroai/ — "
                        "NOT /work/ht203/ (75-day purge).")
    p.add_argument("--cluster", default=None,
                   help="exca TaskInfra cluster ('slurm' on DCC, omit for local).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base_folder_str = args.exca_folder or os.environ.get("EXCA_CACHE_FOLDER")
    if base_folder_str is None:
        raise RuntimeError(
            "EXCA_CACHE_FOLDER must be set or --exca-folder passed explicitly."
        )
    base_folder = Path(base_folder_str)

    cells = _combinatorial_cells() if args.combinatorial else _star_cells()

    print(f"V14 Stage-1 sweep — {EXP_NAME}")
    print(f"  axes: {list(STAGE1_AXES.keys())}")
    print(f"  combinatorial={args.combinatorial}  → {len(cells)} unique cells")
    print(f"  mode={args.mode}  task={args.task}  n_epochs={args.n_epochs}  seed={args.seed}")
    print(f"  base_folder={base_folder}")

    experiments = [
        build_cell_experiment(cell=cell, args=args, base_folder=base_folder)
        for cell in cells
    ]

    if args.dry_run:
        for cell, xp in zip(cells, experiments):
            print(f"  [dry-run] {cell_uid(cell)}  eps={cell['eps']} "
                  f"M={cell['m_sub_slots']} d={cell['d_model']} "
                  f"depth={cell['depth_self_attn']}  uid={xp.infra.uid()}")
        return 0

    base_infra = experiments[0].infra
    with base_infra.job_array(max_workers=args.max_workers) as tasks:
        for xp in experiments:
            tasks.append(xp)
    print(f"V14 Stage-1 dispatch: {len(experiments)} cells submitted as one array.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
