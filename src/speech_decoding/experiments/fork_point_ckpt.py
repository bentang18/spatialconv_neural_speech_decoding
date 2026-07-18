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


def _optimizer_of(module: Any) -> Any:
    """The optimizer this module's OWN ``configure_optimizers`` builds (unwrapped from the
    dict / list forms Lightning accepts)."""
    cfg = module.configure_optimizers()
    if isinstance(cfg, dict):
        return cfg["optimizer"]
    if isinstance(cfg, (list, tuple)):
        first = cfg[0]
        return first["optimizer"] if isinstance(first, dict) else first
    return cfg


def optimizer_param_group_names(module: Any) -> list[list[str]]:
    """Per-group parameter names, in the exact order this module's OWN ``configure_optimizers``
    lays them out (list of groups, each a list of names). The no-WD split produces 2 groups
    (decayed weights, then ndim<=1 no-decay params); this is the authoritative NEW structure a
    fork rebuilds the optimizer from."""
    id2name = {id(p): n for n, p in module.named_parameters()}
    return [[id2name[id(p)] for p in g["params"]] for g in _optimizer_of(module).param_groups]


def optimizer_param_names(module: Any) -> list[str]:
    """Global optimizer index → parameter name, in the exact order this module's OWN
    ``configure_optimizers`` lays them out (the no-WD split reorders params into two
    groups, so this must be read off the built optimizer, not ``named_parameters``)."""
    return [n for group in optimizer_param_group_names(module) for n in group]


def _strip_prefixes(key: str) -> str:
    """Module-path key with Lightning's ``model.`` and compile's ``_orig_mod.`` prefixes
    removed, so a key from a COMPILED-run ckpt and one from a fresh (uncompiled) build of
    the same module compare equal."""
    return key.replace("_orig_mod.", "").replace("model.", "")


def fork_to_point_ckpt(
    ckpt: dict,
    old_names: list[str],
    new_names: list[str],
    *,
    seed_state_dict: dict | None = None,
    reinit_substrings: tuple[str, ...] = (),
) -> tuple[dict, dict]:
    """``ckpt`` (an NLL-arm Lightning checkpoint) → a point-head checkpoint + a report.

    ``old_names``/``new_names``: optimizer index → param name for the NLL module and the
    POINT module respectively (:func:`optimizer_param_names`). Every name in ``new_names``
    must exist in ``old_names`` — the point model is a strict subset of the NLL model, so
    a missing name means the two were built from different configs and we refuse rather
    than guess.

    ``reinit_substrings`` (r5-mod diag_nll): module-path substrings naming params that are
    KEPT in the optimizer's param groups but whose TENSORS are reseeded from
    ``seed_state_dict`` and whose OLD optimizer state is DROPPED (lazily re-init). This is
    for a head that changed SHAPE, not just presence: diag_nll's ``mu_head`` is (5, d) where
    the NLL ckpt's is (6, d), so carrying either its state_dict tensor (strict-load shape
    reject) or its Adam moments (optimizer shape reject) would fail. Scientifically free at
    the λ=0 fork point — the head took zero gradient over 0→10k, so its moments are zero and
    its weights are decayed init. ``chol_head`` is always fully DROPPED (as before); pass
    ``mu_head`` here when the head dim changed. Empty ⇒ the exact Arm-3 (l1) behavior."""
    out = copy.deepcopy(ckpt)
    dropped_names = [n for n in old_names if n not in set(new_names)]
    missing = [n for n in new_names if n not in set(old_names)]
    if missing:
        raise ValueError(
            f"point module has {len(missing)} params the NLL ckpt lacks (configs differ?): "
            f"{missing[:4]}"
        )
    if reinit_substrings and seed_state_dict is None:
        raise ValueError("reinit_substrings given but seed_state_dict is None")

    # ---- 1. state_dict: drop the covariance head; reseed any shape-changed head -----
    # keyed by MODULE path (with Lightning's "model." / compile's "_orig_mod." prefixes),
    # which is a different namespace from the optimizer's param names — filter on the
    # module-path substring, and report the count so a silent no-op is visible.
    sd = out["state_dict"]
    sd_dropped = [k for k in sd if "chol_head" in k]
    sd = {k: v for k, v in sd.items() if "chol_head" not in k}
    sd_reseeded: list[str] = []
    if reinit_substrings:
        assert seed_state_dict is not None  # guaranteed by the guard above
        seed_by_stripped = {_strip_prefixes(k): v for k, v in seed_state_dict.items()}
        for k in list(sd):
            if any(s in k for s in reinit_substrings):
                seed = seed_by_stripped.get(_strip_prefixes(k))
                if seed is None:
                    raise ValueError(f"no seed tensor for reinit key {k!r}")
                sd[k] = seed.clone()
                sd_reseeded.append(k)
    out["state_dict"] = sd

    # ---- 2. optimizer_states: remap surviving entries BY NAME -----------------------
    old_idx = {n: i for i, n in enumerate(old_names)}
    reinit = lambda name: any(s in name for s in reinit_substrings)  # noqa: E731
    opt_reports = []
    for st in out.get("optimizer_states", []):
        old_state = st.get("state", {})
        new_state = {}
        for j, name in enumerate(new_names):
            if reinit(name):
                continue  # shape-changed param: fresh, no carried Adam moment
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
        "state_dict_keys_reseeded": sd_reseeded,
        "optimizer_params_dropped": dropped_names,
        "n_params_in": len(old_names),
        "n_params_out": len(new_names),
        "optimizers": opt_reports,
    }
    return out, report


def _front_prefix(state_dict: dict) -> str:
    """The common leading prefix (``model.`` / ``model._orig_mod.``) the ckpt's state_dict keys
    carry, recovered from one key by removing its stripped tail. Used to key an INJECTED param
    under the same convention as the ckpt (so a compiled-run resume finds it)."""
    k0 = next(iter(state_dict))
    s0 = _strip_prefixes(k0)
    return k0[: len(k0) - len(s0)]


def fork_ckpt_general(
    ckpt: dict,
    *,
    old_names: list[str],
    new_group_names: list[list[str]],
    new_state_dict: dict,
) -> tuple[dict, dict]:
    """Reconcile a source ckpt to a NEW module that may DROP, RESHAPE, ADD, or carry params vs
    the module that wrote the ckpt — the fully general fork (the context-head arm both reshapes
    the diag head 6→5 AND injects a fresh ``pred_to_target_context`` the r4 ckpt never had).

    Each of the new module's params is classified against the ckpt state_dict (matched with
    Lightning ``model.`` / compile ``_orig_mod.`` prefixes stripped):
      * CARRY  — name in ckpt, SAME shape → keep the ckpt tensor + carry its Adam moment BY NAME.
      * RESHAPE — name in ckpt, DIFFERENT shape → reseed the tensor from ``new_state_dict`` + drop
                  the moment (AdamW lazily re-inits at the new shape).
      * ADD    — name only in the new module → inject the tensor from ``new_state_dict`` under the
                  ckpt's key convention + no moment (fresh).
    Any ckpt param the new module lacks is DROPPED (absent from both the out state_dict and the
    rebuilt optimizer). The optimizer param-groups are rebuilt from ``new_group_names`` (the
    authoritative new structure), taking each group's hyper-params from the SAME-index old group
    (the no-WD split gives both modules the same [decay, no-decay] group order). Reseeding/adding
    is scientifically free at the r4 λ-ramp fork point: the whole 0→10k prefix ran the reseeded/
    added heads at λ=0, so their moments are zero and their weights are decayed init.

    ``old_names``: optimizer index → name for the module that WROTE the ckpt (its integer state
    keys). ``new_group_names``: :func:`optimizer_param_group_names` of the fresh new module.
    ``new_state_dict``: the fresh new module's ``state_dict`` (bare keys)."""
    out = copy.deepcopy(ckpt)
    ck_sd = out["state_dict"]
    front = _front_prefix(ck_sd)
    ck_by_stripped = {_strip_prefixes(k): k for k in ck_sd}
    new_by_stripped = {_strip_prefixes(k): v for k, v in new_state_dict.items()}

    # ---- 1. state_dict: carry / reshape / add, keyed under the ckpt's prefix convention -------
    out_sd: dict = {}
    reshaped: set[str] = set()
    added: list[str] = []
    for sname, val in new_by_stripped.items():
        ck_key = ck_by_stripped.get(sname)
        if ck_key is not None:
            if tuple(ck_sd[ck_key].shape) == tuple(val.shape):
                out_sd[ck_key] = ck_sd[ck_key]  # CARRY (verbatim)
            else:
                out_sd[ck_key] = val.clone()  # RESHAPE (reseed from fresh)
                reshaped.add(sname)
        else:
            out_sd[front + sname] = val.clone()  # ADD (inject fresh)
            added.append(sname)
    dropped_sd = [k for k in ck_sd if _strip_prefixes(k) not in new_by_stripped]
    out["state_dict"] = out_sd

    # ---- 2. optimizer_states: rebuild groups from the new structure, moments carried BY NAME --
    new_names = [n for g in new_group_names for n in g]
    old_stripped_idx = {_strip_prefixes(n): i for i, n in enumerate(old_names)}
    opt_reports = []
    for st in out.get("optimizer_states", []):
        old_state = st.get("state", {})
        new_state: dict = {}
        for j, name in enumerate(new_names):
            sname = _strip_prefixes(name)
            if sname in reshaped or sname not in old_stripped_idx:
                continue  # reshaped (lazy re-init) or added (fresh) ⇒ no carried moment
            i = old_stripped_idx[sname]
            if i in old_state:
                new_state[j] = old_state[i]
            elif str(i) in old_state:
                new_state[j] = old_state[str(i)]
        if len(st["param_groups"]) != len(new_group_names):
            raise ValueError(
                f"old optimizer has {len(st['param_groups'])} param-groups but the new module "
                f"has {len(new_group_names)} — the no-WD split must produce the same group "
                "layout for a same-index hyper-param carry"
            )
        cursor = 0
        new_groups = []
        for old_g, gnames in zip(st["param_groups"], new_group_names):
            ng = {k: v for k, v in old_g.items() if k != "params"}
            ng["params"] = list(range(cursor, cursor + len(gnames)))
            cursor += len(gnames)
            new_groups.append(ng)
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
        "state_dict_keys_dropped": dropped_sd,
        "state_dict_keys_reshaped": sorted(reshaped),
        "state_dict_keys_added": sorted(added),
        "n_params_in": len(old_names),
        "n_params_out": len(new_names),
        "optimizers": opt_reports,
    }
    return out, report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-ckpt", required=True, help="the NLL-arm fork point (r4@10k)")
    p.add_argument("--out-ckpt", required=True, help="point-head fork ckpt to write")
    p.add_argument("--old-secondary-loss", default="nll",
                   help="secondary loss the IN ckpt was written with (owns chol_head). "
                        "arm1/r4 = 'nll' (full covariance).")
    p.add_argument("--new-secondary-loss", default="diag_nll",
                   help="secondary loss the FORKED module uses (point-only, no chol_head). "
                        "r5-mod = 'diag_nll'; r5 Arm 3 = 'l1'. Both drop the covariance head.")
    p.add_argument("--context-loss", action="store_true",
                   help="the FORKED module ALSO owns a V-JEPA 2.1 context head "
                        "(pred_to_target_context) the r4 ckpt never had. Routes through the "
                        "GENERAL fork (carry/reshape/drop/ADD) instead of the drop-only path, "
                        "since a param must be INJECTED, not just filtered.")
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

    def _mod(secondary_loss: str, context_loss: bool = False):
        base = ["--bt-root", a.bt_root, "--span-dir", a.span_dir,
                "--state-stats-dir", a.state_stats_dir, "--ssl-max-steps", "1",
                "--secondary-loss", secondary_loss, "--accelerator", "cpu", "--devices", "1"]
        if context_loss:
            base += ["--context-loss"]
        for d in a.band_cache_dir:
            base += ["--band-cache-dir", d]
        for s in a.session:
            base += ["--session", s]
        args = build_arg_parser().parse_args(base)
        module, _, _ = build_v3_training(specs, args)
        return module

    new_mod = _mod(a.new_secondary_loss, context_loss=a.context_loss)
    # old_names indexes the CKPT's optimizer state by position. Build it from a CURRENT-code
    # NLL module: its param COUNT/ORDER/NAMES match the module that wrote the ckpt (same arch),
    # so name-based remap is exact — only the head TENSOR SHAPES differ (5-dim code vs the
    # ckpt's 6-dim head), which the reinit path below handles.
    old_names = optimizer_param_names(_mod(a.old_secondary_loss))
    new_names = optimizer_param_names(new_mod)

    ck = torch.load(a.in_ckpt, map_location="cpu", weights_only=False)

    if a.context_loss:
        # The context arm INJECTS pred_to_target_context (absent from the r4 ckpt), so the
        # drop-only path can't do it — route through the general fork, which classifies every
        # new param carry/reshape/ADD/drop. It also handles the 6->5 diag reshape in the same
        # pass, so no separate reinit detection is needed here.
        out, rep = fork_ckpt_general(
            ck, old_names=old_names,
            new_group_names=optimizer_param_group_names(new_mod),
            new_state_dict=new_mod.state_dict(),
        )
        torch.save(out, a.out_ckpt)
        ctx_added = [k for k in rep["state_dict_keys_added"] if "pred_to_target_context" in k]
        ok = (
            rep["global_step"] == ck.get("global_step")
            and all("chol_head" in k for k in rep["state_dict_keys_dropped"])
            and len(ctx_added) == len(rep["state_dict_keys_added"])  # ONLY the context head added
            and rep["state_dict_keys_added"]                          # ...and it WAS added
        )
        print(f"[check] GENERAL fork {a.in_ckpt} -> {a.out_ckpt} "
              f"({a.old_secondary_loss} -> {a.new_secondary_loss} + context head)")
        print(f"[check] global_step={rep['global_step']} epoch={rep['epoch']} (CARRIED — a "
              f"cold start would read 0)")
        print(f"[check] dropped: {rep['state_dict_keys_dropped']}")
        print(f"[check] reshaped (fresh tensor, no carried moment): "
              f"{rep['state_dict_keys_reshaped']}")
        print(f"[check] ADDED (injected fresh, no moment): {rep['state_dict_keys_added']}")
        for i, o in enumerate(rep["optimizers"]):
            print(f"[check] opt[{i}] group sizes {o['group_sizes_in']} -> "
                  f"{o['group_sizes_out']}; state entries {o['n_state_in']} -> "
                  f"{o['n_state_out']}")
        print("[check] chol dropped + diag reshaped + context head injected, backbone remapped "
              "BY NAME " + ("OK" if ok else "VIOLATED"))
        return

    # Reseed EVERY persistent tensor whose shape changed between the CKPT and the new module,
    # not just mu_head. The 5-dim head resizes BOTH the mu_head Linear (5,d)<-(6,d) AND the
    # frozen ``noise_var`` BUFFER (5,)<-(6,) — carrying either at 6-dim fails the strict load.
    # Detect generically by comparing the ckpt's own state_dict against the fresh module's
    # (matched with prefixes stripped, so a compiled ckpt compares equal to the fresh build);
    # chol_head is excluded because it is DROPPED, not reseeded. When nothing changed shape
    # (l1 forking a same-dim ckpt) reinit is empty and Arm-3 behavior is unchanged.
    new_sd = new_mod.state_dict()
    new_by_stripped = {_strip_prefixes(k): v for k, v in new_sd.items()}
    reinit_keys = []
    for k, v in ck["state_dict"].items():
        sk = _strip_prefixes(k)
        if "chol_head" in sk:
            continue
        nv = new_by_stripped.get(sk)
        if nv is not None and tuple(nv.shape) != tuple(v.shape):
            reinit_keys.append(sk)
    reinit = tuple(sorted(set(reinit_keys)))
    seed = new_sd if reinit else None

    out, rep = fork_to_point_ckpt(
        ck, old_names, new_names, seed_state_dict=seed, reinit_substrings=reinit
    )
    torch.save(out, a.out_ckpt)

    ok = (
        rep["state_dict_keys_dropped"]
        and all("chol_head" in n for n in rep["optimizer_params_dropped"])
        and rep["n_params_in"] - rep["n_params_out"] == len(rep["optimizer_params_dropped"])
        and len(rep["state_dict_keys_reseeded"]) == len(reinit)  # one sd key per changed tensor
    )
    print(f"[check] fork {a.in_ckpt} -> {a.out_ckpt} "
          f"({a.old_secondary_loss} -> {a.new_secondary_loss}, reinit={reinit or 'none'})")
    print(f"[check] global_step={rep['global_step']} epoch={rep['epoch']} (CARRIED — a "
          f"cold start would read 0)")
    print(f"[check] state_dict keys dropped: {rep['state_dict_keys_dropped']}")
    print(f"[check] state_dict keys reseeded (fresh head, no carried moment): "
          f"{rep['state_dict_keys_reseeded']}")
    print(f"[check] optimizer params dropped: {rep['optimizer_params_dropped']} "
          f"({rep['n_params_in']} -> {rep['n_params_out']})")
    for i, o in enumerate(rep["optimizers"]):
        print(f"[check] opt[{i}] group sizes {o['group_sizes_in']} -> "
              f"{o['group_sizes_out']}; state entries {o['n_state_in']} -> "
              f"{o['n_state_out']}")
    print("[check] chol_head dropped + shape-changed head reseeded, backbone remapped BY "
          "NAME " + ("OK" if ok else "VIOLATED"))


if __name__ == "__main__":
    main()
