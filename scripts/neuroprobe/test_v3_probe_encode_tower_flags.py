"""_load_teacher must build a tower that MATCHES the ckpt's ablation flags.

Two failure modes, and they are not symmetric:

  parcel_embed  — LOUD. A --no-parcel-embed ckpt has no parcel_embed.embed.weight, so a shell
                  built with one puts that key in `missing` and _load_teacher raises. Before the
                  fix that raise was the ONLY outcome: the noparcel arm could not be probed at
                  all. Inferring the flag from the ckpt keys is what makes it loadable, and the
                  missing/unexpected check stays as the verifier.

  space_rope    — SILENT, and this is the dangerous one. L1RoPE zeroes idx_freq when space=False
                  and registers it persistent=False (pe.py), so a --no-space-rope ckpt is key-
                  AND value-identical to a normal one. No state_dict check can distinguish them.
                  If the flag is not passed the encode applies contact-index rotation the trained
                  model never saw, and nothing anywhere reports it. These tests pin that the flag
                  reaches the tower and that it genuinely changes the forward — a cosmetic flag
                  would be worse than none, since it would read as covered.
"""
from __future__ import annotations

import torch

from scripts.neuroprobe.v3_probe_encode_r4 import N_PARCELS
from speech_decoding.models.v14_converged_v3.objective import _TargetTower


def _sd(**kw) -> dict:
    """The `objective.online.`-rooted subtree a real ckpt would carry for these flags."""
    tower = _TargetTower(n_parcels=N_PARCELS, **kw)
    return {f"objective.online.{k}": v for k, v in tower.state_dict().items()}


def test_no_parcel_embed_ckpt_omits_the_embed_key() -> None:
    """The premise of the inference: the key is absent iff the arm ran --no-parcel-embed."""
    on = [k for k in _sd(parcel_embed=True) if k.endswith("parcel_embed.embed.weight")]
    off = [k for k in _sd(parcel_embed=False) if k.endswith("parcel_embed.embed.weight")]
    print(f"[check] parcel_embed=True -> {len(on)} embed key(s); False -> {len(off)} OK")
    assert len(on) == 1, on
    assert off == [], off


def test_load_teacher_infers_parcel_embed_and_loads_a_noparcel_ckpt() -> None:
    """Before the fix this raised RuntimeError('missing=parcel_embed.embed.weight')."""
    from scripts.neuroprobe.v3_probe_encode_r4 import _load_teacher

    for flag in (True, False):
        tower = _load_teacher(_sd(parcel_embed=flag), device=torch.device("cpu"),
                              pref="objective.online.")
        got = any("parcel_embed" in k for k in tower.state_dict())
        print(f"[check] ckpt parcel_embed={flag} -> tower has embed={got} OK")
        assert got is flag


def test_space_rope_ckpt_is_INDISTINGUISHABLE_so_the_flag_cannot_be_inferred() -> None:
    """Why --no-space-rope must be passed by hand: the state dicts are identical.

    This is a characterization test. If it ever FAILS, space RoPE has become visible in the
    ckpt and _load_teacher should infer it instead of trusting the CLI.
    """
    from scripts.neuroprobe.v3_probe_encode_r4 import _load_teacher

    trained_off = _sd(space_rope=False)          # what the A2 arm ships
    assert set(trained_off) == set(_sd(space_rope=True)), \
        "keys differ — space_rope is now inferable, go infer it"
    # The real danger: that ckpt loads into a space-rope-ON shell with NOTHING reported.
    # _load_teacher raises on any missing/unexpected key, so reaching this line at all is
    # the proof that the check is blind to this mismatch.
    tower = _load_teacher(trained_off, device=torch.device("cpu"), pref="objective.online.",
                          space_rope=True)       # deliberately WRONG for this ckpt
    assert tower is not None
    print("[check] a --no-space-rope ckpt loaded into a space_rope=True tower with NO error "
          "and NO missing/unexpected keys => only the CLI flag can prevent this OK")


def test_space_rope_flag_actually_changes_the_forward() -> None:
    """The flag is not cosmetic: with space=False the index axis takes the identity rotation."""
    from speech_decoding.models.v14_converged_v3.pe import L1RoPE

    idx = torch.arange(6).float().reshape(1, 6)
    t = torch.zeros(1, 6)
    cos_on, sin_on = L1RoPE(8, space=True).cos_sin(idx, t)
    cos_off, sin_off = L1RoPE(8, space=False).cos_sin(idx, t)
    differs = not torch.allclose(cos_on, cos_off) or not torch.allclose(sin_on, sin_off)
    print(f"[check] space=True vs False rotary tables differ={differs} "
          f"(sin_off max |.|={sin_off.abs().max():.3e}, want 0) OK")
    assert differs, "space_rope had no effect — the ablation would be a no-op"
    assert torch.allclose(sin_off, torch.zeros_like(sin_off)), "index axis should be identity"
