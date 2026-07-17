"""Fork an NLL-arm v3 checkpoint into an r5 Arm 3 (point-head) checkpoint.

WHY THIS EXISTS. Arm 3 forks r4's step-10000 ckpt so it shares the byte-identical λ=0
prefix with Arms 0/2 (paired design). But that ckpt was written by a model that OWNS a
covariance head, and Arm 3's point head does not (``GaussianStateHead(point_only=True)``
does not construct ``chol_head`` — a point loss touches no covariance parameter, and
leaving the Linear in place would hand DDP a parameter with no gradient, the X1 failure
class that killed r3). Two things break, and BOTH fail loudly rather than silently:

  1. ``state_dict``: 2 unexpected ``chol_head.{weight,bias}`` keys → strict load rejects.
  2. ``optimizer_states``: AdamW state is restored BY POSITION within each param group, so
     2 fewer trainable params shift every later index →
     "loaded state dict contains a parameter group that doesn't match the size of
     optimizer's group".

(2) is the dangerous one and the reason this is a real utility and not a one-line filter:
dropping the keys alone leaves the optimizer indices stale, and index-shifted Adam state
would be silent corruption if PyTorch did not happen to size-check the group.

WHAT IT COSTS SCIENTIFICALLY: nothing. r4's λ ramp starts at step 10000, so the whole
0→10k prefix ran λ=0 and the perceiver took ZERO gradient — at the fork point its weights
are at init (modulo AdamW's decoupled decay) and its Adam moments are zero. So discarding
the covariance head's weights and moments discards zeros. Everything else — encoder,
predictor, stem, and all of THEIR optimizer state — is carried across EXACTLY, by name.

The remap is BY NAME, never by position: we build both modules, ask each one's real
``configure_optimizers`` for its parameter order, and move each surviving entry from its
old index to its new one. Position-based copying is precisely the bug being avoided.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any

import torch


def optimizer_param_names(module: Any) -> list[str]:
    """Global optimizer index → parameter name, in the exact order this module's OWN
    ``configure_optimizers`` lays them out (the no-WD split reorders params into two
    groups, so this must be read off the built optimizer, not ``named_parameters``)."""
    id2name = {id(p): n for n, p in module.named_parameters()}
    cfg = module.configure_optimizers()
    if isinstance(cfg, dict):
        opt = cfg["optimizer"]
    elif isinstance(cfg, (list, tuple)):
        first = cfg[0]
        opt = first["optimizer"] if isinstance(first, dict) else first
    else:
        opt = cfg
    names: list[str] = []
    for g in opt.param_groups:
        for p in g["params"]:
            names.append(id2name[id(p)])
    return names


def fork_to_point_ckpt(
    ckpt: dict, old_names: list[str], new_names: list[str]
) -> tuple[dict, dict]:
    """``ckpt`` (an NLL-arm Lightning checkpoint) → an Arm 3 checkpoint + a report.

    ``old_names``/``new_names``: optimizer index → param name for the NLL module and the
    POINT module respectively (:func:`optimizer_param_names`). Every name in ``new_names``
    must exist in ``old_names`` — the point model is a strict subset of the NLL model, so
    a missing name means the two were built from different configs and we refuse rather
    than guess."""
    out = copy.deepcopy(ckpt)
    dropped_names = [n for n in old_names if n not in set(new_names)]
    missing = [n for n in new_names if n not in set(old_names)]
    if missing:
        raise ValueError(
            f"point module has {len(missing)} params the NLL ckpt lacks (configs differ?): "
            f"{missing[:4]}"
        )

    # ---- 1. state_dict: drop the covariance head's tensors -------------------------
    # keyed by MODULE path (with Lightning's "model." / compile's "_orig_mod." prefixes),
    # which is a different namespace from the optimizer's param names — filter on the
    # module-path substring, and report the count so a silent no-op is visible.
    sd = out["state_dict"]
    sd_dropped = [k for k in sd if "chol_head" in k]
    out["state_dict"] = {k: v for k, v in sd.items() if "chol_head" not in k}

    # ---- 2. optimizer_states: remap surviving entries BY NAME -----------------------
    old_idx = {n: i for i, n in enumerate(old_names)}
    opt_reports = []
    for st in out.get("optimizer_states", []):
        old_state = st.get("state", {})
        new_state = {}
        for j, name in enumerate(new_names):
            i = old_idx[name]
            if i in old_state:
                new_state[j] = old_state[i]
            elif str(i) in old_state:  # some serializations stringify the keys
                new_state[j] = old_state[str(i)]
        # group hyper-params (lr, betas, weight_decay, ...) are unchanged; only the index
        # lists move. Rebuild each group's "params" from the NEW order, preserving which
        # group each surviving param belonged to.
        cursor = 0
        new_groups = []
        rebuilt_order: list[str] = []
        for g in st["param_groups"]:
            keep = [n for n in (old_names[i] for i in g["params"]) if n in set(new_names)]
            ng = {k: v for k, v in g.items() if k != "params"}
            ng["params"] = list(range(cursor, cursor + len(keep)))
            cursor += len(keep)
            rebuilt_order.extend(keep)
            new_groups.append(ng)
        # THE INVARIANT. ``new_state[j]`` is keyed by position in ``new_names``, while the
        # groups above are rebuilt by filtering the OLD group order. Those two orderings
        # must coincide or every Adam moment lands on the WRONG tensor — silently, since
        # the group SIZES would still match. Assert it rather than trust that dropping a
        # param preserves relative order.
        if rebuilt_order != new_names:
            first = next(
                (k for k, (x, y) in enumerate(zip(rebuilt_order, new_names)) if x != y),
                min(len(rebuilt_order), len(new_names)),
            )
            raise ValueError(
                "optimizer param ORDER diverges between the filtered NLL groups and the "
                f"point module's own optimizer at index {first}: "
                f"{rebuilt_order[first:first+3]} != {new_names[first:first+3]} — refusing "
                "to remap, Adam state would be misassigned"
            )
        opt_reports.append(
            {"n_state_in": len(old_state), "n_state_out": len(new_state),
             "group_sizes_in": [len(g["params"]) for g in st["param_groups"]],
             "group_sizes_out": [len(g["params"]) for g in new_groups]}
        )
        st["state"] = new_state
        st["param_groups"] = new_groups

    report = {
        "global_step": out.get("global_step"),
        "epoch": out.get("epoch"),
        "state_dict_keys_dropped": sd_dropped,
        "optimizer_params_dropped": dropped_names,
        "n_params_in": len(old_names),
        "n_params_out": len(new_names),
        "optimizers": opt_reports,
    }
    return out, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-ckpt", required=True, help="the NLL-arm fork point (r4@10k)")
    p.add_argument("--out-ckpt", required=True, help="Arm 3 fork ckpt to write")
    p.add_argument("--bt-root", required=True)
    p.add_argument("--band-cache-dir", action="append", required=True)
    p.add_argument("--span-dir", required=True)
    p.add_argument("--state-stats-dir", required=True)
    p.add_argument("--session", action="append", required=True, help="S:T")
    a = p.parse_args()

    from speech_decoding.experiments.dispatch_v3 import (
        _parse_sessions,
        build_arg_parser,
        build_v3_training,
        make_bt_parcel_fn,
    )
    from speech_decoding.models.v14_converged_v3.session_loader import load_v3_sessions

    sessions = _parse_sessions(a.session)
    specs = load_v3_sessions(
        sessions=sessions, band_cache_dirs=a.band_cache_dir, span_dir=a.span_dir,
        parcel_fn=make_bt_parcel_fn(a.bt_root), state_stats_dir=a.state_stats_dir,
    )

    def _mod(secondary_loss: str):
        base = ["--bt-root", a.bt_root, "--span-dir", a.span_dir,
                "--state-stats-dir", a.state_stats_dir, "--ssl-max-steps", "1",
                "--secondary-loss", secondary_loss, "--accelerator", "cpu", "--devices", "1"]
        for d in a.band_cache_dir:
            base += ["--band-cache-dir", d]
        for s in a.session:
            base += ["--session", s]
        args = build_arg_parser().parse_args(base)
        module, _, _ = build_v3_training(specs, args)
        return module

    old_names = optimizer_param_names(_mod("nll"))
    new_names = optimizer_param_names(_mod("l1"))
    ck = torch.load(a.in_ckpt, map_location="cpu", weights_only=False)
    out, rep = fork_to_point_ckpt(ck, old_names, new_names)
    torch.save(out, a.out_ckpt)

    ok = (
        rep["state_dict_keys_dropped"]
        and all("chol_head" in n for n in rep["optimizer_params_dropped"])
        and rep["n_params_in"] - rep["n_params_out"] == len(rep["optimizer_params_dropped"])
    )
    print(f"[check] fork {a.in_ckpt} -> {a.out_ckpt}")
    print(f"[check] global_step={rep['global_step']} epoch={rep['epoch']} (CARRIED — a "
          f"cold start would read 0)")
    print(f"[check] state_dict keys dropped: {rep['state_dict_keys_dropped']}")
    print(f"[check] optimizer params dropped: {rep['optimizer_params_dropped']} "
          f"({rep['n_params_in']} -> {rep['n_params_out']})")
    for i, o in enumerate(rep["optimizers"]):
        print(f"[check] opt[{i}] group sizes {o['group_sizes_in']} -> "
              f"{o['group_sizes_out']}; state entries {o['n_state_in']} -> "
              f"{o['n_state_out']}")
    print("[check] ONLY chol_head dropped, everything else remapped BY NAME "
          + ("OK" if ok else "VIOLATED"))


if __name__ == "__main__":
    main()
