#!/usr/bin/env python
"""Build ONE long-form CSV over every r6-era result artifact. Generated, never hand-edited.

WHY THIS EXISTS. Our numbers live scattered across board JSONs, per-cell shard dirs, LEACE arrays
and decoder-ablation dirs, and the merged JSONs are not trustworthy (the canonical 45k board carries
`csession_mean` EMPTY from a stale merge). Every provenance defect found in the 2026-07-29 audit was
one of four kinds:

    arm mix-up            -- an analysis run on r6_40k quoted as canonical cdlin_45k (5 instances)
    partial-cell macro    -- a 9- or 4-cell mean compared against a 10-cell mean
    definition swap       -- three different "event/level ratio"s used interchangeably
    window mix            -- a 1 s number quoted beside a 2 s one

A long-form row per (arm, regime, tap, norm, decoder, cell, task) makes all four visible in a
groupby instead of discoverable only by audit. `n_cells` is therefore something you COMPUTE from the
ledger, never something you trust from a memo.

THIS IS A READER. It writes exactly one file and never mutates a source artifact.

Usage:
  python scripts/neuroprobe/audit/build_results_ledger.py
  python scripts/neuroprobe/audit/build_results_ledger.py --out /tmp/ledger.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

R6 = Path("results/r6_era")

# Run IDs recovered by matching shard mtime to the sacct End time of the element that wrote it
# (these artifacts carry no provenance field). Exact to <=1s. See PROVENANCE.md.
RUN_IDS = {
    "leace": "20544396",
    "leace_enc3": "20565621",
    # pbs arms: sacct/JobName CANNOT tell them apart (one launcher, one JobName, space_frac by
    # env). These come from each job's OWN stdout header, /projects/bhqk/htang13/logs/
    # r6_pbspace_<jobid>.out, which prints run= and space_frac= verbatim.
    "pbs50_20k": "2764203",
    "pbs25_20k": "2764176",
    # pbs00_20k IS NOT FROM THE pbspace LAUNCHER. Its stdout header reads
    # "[r6 hard-mask OFAT] run=v3_r6_maskspace0_lr6e-3_40k ... --mask-space-frac 0.0", i.e. the
    # SHARED space knob with per_band_space=False, and cb=5500 accum=12 where pbs50/pbs25 run
    # cb=11000 accum=6. It is still the correct zero rung: masking.py:145 makes space_frac==0.0
    # the deliberate no-spatial-masking arm (Σd_s == 0 by construction), and per_band_space only
    # decides how a spatial draw is split across bands (:454), so it is VACUOUS when no contacts
    # are drawn. The cb/accum split is token-exact (66,000 contacts/step both) and accumulation
    # averages exactly, so what differs is the per-micro-batch mask REALIZATION -- a seed-level
    # difference, and there is 1 seed per arm. Quote the ladder with that caveat attached.
    "pbs00_20k": "2764373",
}

FIELDS = ["family", "artifact", "run_id", "arm_tag", "regime", "tap", "norm",
          "decoder", "cell", "task", "split", "value"]


# Readout columns that are a DECODER over one tap rather than a tap. The rules that pool or
# average within a single tap keep that tap; the ones whose input is a SET of layers report
# tap="multi", because inventing a tap named "ens:auto" would put a phantom layer in any
# groupby over depth.
_ONE_TAP_RULES = ("gpool", "bpool", "lamall", "lam3", "lamge")


def _split_col(tap: str) -> tuple[str, str]:
    """"lam3:enc12" -> ("enc12", "ridge_lam3"); "ens:auto" -> ("multi", "ridge_ens_auto")."""
    rule, sep, base = tap.partition(":")
    if not sep:
        return tap, "ridge"
    if rule in _ONE_TAP_RULES:
        return base, f"ridge_{rule}"
    return "multi", f"ridge_{rule}_{base}"


def _rows_board():
    """results/r6_era/board/shards_<tag>/<regime>_<cell>.json -- the trustworthy board source."""
    for sd in sorted(R6.glob("board/shards_*")):
        tag = sd.name.replace("shards_", "")
        for f in sorted(sd.glob("*.json")):
            d = json.load(open(f))
            regime, cell = d.get("kind", "?"), d.get("name", f.stem)
            for key, rec in d.get("cells", {}).items():
                arm_tag, task = (key.split("|") + [key])[:2] if "|" in key else (tag, key)
                for col, v in rec.get("cells", {}).items():
                    tap, _, norm = col.partition("|")
                    tap, decoder = _split_col(tap)
                    for split in ("val", "test"):
                        if split in v:
                            yield dict(family="board", artifact=str(sd), run_id="",
                                       arm_tag=arm_tag, regime=regime, tap=tap, norm=norm,
                                       decoder=decoder, cell=cell, task=task,
                                       split=split, value=v[split])


def _rows_pbs():
    """results/r6_era/pbs/results_v3_probe_<arm>.json -- the PRETRAIN probe, not the board.

    Kept as its own family because pooling probe numbers with board numbers is a recorded
    defect (probe CS ~.66-.69 vs board CS ~.60). ``ws_per_session`` is keyed by session,
    ``cs_per_test`` by held-out subject; both are held-out test values.
    """
    for f in sorted((R6 / "pbs").glob("results_v3_probe_*.json")):
        d = json.load(open(f))
        for key, rec in d.items():
            arm_tag, tap, norm, task = key.split("|")
            for regime, field in (("ws", "ws_per_session"), ("cs", "cs_per_test")):
                for cell, v in rec.get(field, {}).items():
                    yield dict(family="pbs_probe", artifact=str(f),
                               run_id=RUN_IDS.get(arm_tag, ""), arm_tag=arm_tag, regime=regime,
                               tap=tap, norm=norm, decoder="ridge", cell=cell, task=task,
                               split="test", value=v)


def _rows_leace():
    """LEACE erasure arms. `norm` carries the arm (std / leace / std_target)."""
    for sub in ("leace", "leace_enc3"):
        d0 = R6 / sub
        if not d0.is_dir():
            continue
        for f in sorted(d0.glob("leace_*.json")):
            cell = f.stem.split("_", 1)[1]
            for task, blob in json.load(open(f)).items():
                for col, v in blob.get("cells", {}).items():
                    tap, _, arm = col.partition("|")
                    if not isinstance(v, dict) or "test" not in v:
                        continue
                    for split in ("val", "test"):
                        if split in v:
                            yield dict(family="leace", artifact=str(d0),
                                       run_id=RUN_IDS.get(sub, ""),
                                       arm_tag="r6_40k", regime="cs", tap=tap, norm=arm,
                                       decoder="ridge", cell=cell, task=task,
                                       split=split, value=v[split])


def _rows_decoder():
    """decoder_ablation/<dir>/dec_<cell>.json -- keys are `<tap>|<variant>|<task>`.
    naive = HarnessCNN, perband = PerBandCNN, mlp = HarnessMLP (v3_board_decoder.py:273-280)."""
    for sd in sorted((R6 / "decoder_ablation").glob("*")):
        if not sd.is_dir():
            continue
        for f in sorted(sd.glob("dec_*.json")):
            d = json.load(open(f))
            for key, v in d.get("cells", {}).items():
                parts = key.split("|")
                if len(parts) != 3:
                    continue
                tap, variant, task = parts
                for split in ("val", "test"):
                    if split in v:
                        yield dict(family="decoder", artifact=str(sd), run_id="",
                                   arm_tag=d.get("tag", ""), regime="cs", tap=tap, norm="std",
                                   decoder=variant, cell=d.get("cell", f.stem),
                                   task=task, split=split, value=v[split])


def _rows_board_ft():
    """board_ft/<arm_tag>__k<K>/ft_<regime>_<cell>.json -- the partial fine-tune arms.

    Each file is a FLAT LIST of one record per (cell, task, FOLD); the board's macro unit is
    (session, task) with folds AVERAGED, so this reader averages them and emits one row per unit.
    Averaging here is what makes an FT row comparable to a `board` row: mixing a fold-level FT
    number with a fold-averaged board number is the same defect class this ledger exists to catch.

    Two decoders per unit, both fit on the SAME frozen ridge grid: `ridge` is A (frozen block12,
    val-selected lambda -- the published board entry, recomputed in-path at epoch 0) and
    `ridge_ft_k<K>` is C (block-12 MLP fine-tuned, then that same ridge). The headline is C - A,
    which you take as a groupby difference, never from the JSON's own `d` field -- `d` in the
    per-cell LOG is d(D-A), a different quantity. K is the DRIVER task count and lives only in the
    directory name because the records do not carry it (the run log prints `K_drv=`).
    """
    for sd in sorted((R6 / "board_ft").glob("*__k*")):
        if not sd.is_dir():
            continue
        arm_tag, _, k = sd.name.partition("__")
        for f in sorted(sd.glob("ft_*.json")):
            folds: dict[tuple, list] = {}
            for rec in json.load(open(f)):
                key = (rec["regime"], rec["cell"], rec["task"])
                folds.setdefault(key, []).append(rec)
            for (regime, cell, task), rs in sorted(folds.items()):
                decs = [("ridge", "test_frozen_vallam"), (f"ridge_ft_{k}", "test_c")]
                # EPOCH ENSEMBLES, when the arm ran with --dump-epoch-test. Each is the test AUROC
                # of a RANK-AVERAGED prediction over a val-only set of epochs, so it is a different
                # DECODER on the same (arm, regime, tap, cell, task) unit -- exactly what the
                # decoder column is for. `ens_top1` averages the single selected epoch and is
                # therefore a copy of test_c BY CONSTRUCTION; it is emitted anyway because a
                # groupby that shows it drifting from ridge_ft is the cheapest possible alarm that
                # an arm's curve and its selection came apart.
                for rule in ("ens_all", "ens_valge0", "ens_top3", "ens_top1"):
                    if all(rule in r and r[rule] == r[rule] for r in rs):
                        decs.append((f"ridge_ft_{k}_{rule}", rule))
                for dec, field in decs:
                    yield dict(family="board_ft", artifact=str(sd), run_id="", arm_tag=arm_tag,
                               regime=regime, tap="enc12", norm="std", decoder=dec, cell=cell,
                               task=task, split="test",
                               value=sum(r[field] for r in rs) / len(rs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R6 / "RESULTS_LEDGER.csv"))
    a = ap.parse_args()

    rows = (list(_rows_board()) + list(_rows_pbs()) + list(_rows_leace())
            + list(_rows_decoder()) + list(_rows_board_ft()))
    if not rows:
        raise SystemExit("no artifacts found -- run from the repo root")
    rows.sort(key=lambda r: tuple(str(r[k]) for k in FIELDS[:-1]))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {out}  {len(rows)} rows")

    # Coverage report. Printed, not assumed -- a family whose cell count is not what you expect is
    # exactly the bug this ledger exists to surface.
    print(f"\n{'family':9s} {'arm_tag':16s} {'regime':9s} {'decoder':9s} "
          f"{'taps':>4s} {'cells':>5s} {'tasks':>5s} {'rows':>6s}")
    seen = {}
    for r in rows:
        k = (r["family"], r["arm_tag"], r["regime"], r["decoder"])
        s = seen.setdefault(k, {"tap": set(), "cell": set(), "task": set(), "n": 0})
        s["tap"].add(r["tap"]); s["cell"].add(r["cell"]); s["task"].add(r["task"]); s["n"] += 1
    for k, s in sorted(seen.items()):
        flag = "" if len(s["task"]) == 15 else "  <-- NOT 15 TASKS"
        print(f"{k[0]:9s} {k[1]:16s} {k[2]:9s} {k[3]:9s} {len(s['tap']):4d} "
              f"{len(s['cell']):5d} {len(s['task']):5d} {s['n']:6d}{flag}")

    # Per-TAP coverage. The family-level table hides a stale slice: if one tap landed 10 complete
    # cells and another only 6 ragged ones, the union still reads "10 cells / 15 tasks". A local
    # copy going stale against Delta is a real, observed failure -- surface it per tap.
    print(f"\n{'family':9s} {'arm_tag':16s} {'regime':9s} {'tap':9s} {'norm/arm':22s} "
          f"{'cells':>5s} {'tasks':>5s}")
    pertap = {}
    for r in rows:
        label = r["decoder"] if r["family"] == "decoder" else r["norm"]
        k = (r["family"], r["arm_tag"], r["regime"], r["tap"], label)
        s = pertap.setdefault(k, {"cell": set(), "task": set()})
        s["cell"].add(r["cell"]); s["task"].add(r["task"])
    exp_cells = {"cs": 10, "ws": 12, "csession": 12}
    for k, s in sorted(pertap.items()):
        want = exp_cells.get(k[2])
        bad = (want and len(s["cell"]) != want) or len(s["task"]) != 15
        print(f"{k[0]:9s} {k[1]:16s} {k[2]:9s} {k[3]:9s} {k[4]:22s} "
              f"{len(s['cell']):5d} {len(s['task']):5d}" + ("  <-- INCOMPLETE" if bad else ""))

    arms = sorted({r["arm_tag"] for r in rows if r["arm_tag"]})
    print(f"\n[check] arm tags present: {arms}")
    print("[check] MIXING ARMS IS THE #1 RECORDED DEFECT -- always filter arm_tag before a macro.")


if __name__ == "__main__":
    main()
