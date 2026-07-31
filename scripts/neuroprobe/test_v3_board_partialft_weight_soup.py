"""Averaging the WEIGHTS of the val-selected epochs, instead of their predictions.

WHY THIS EXISTS. Prediction-averaging over the top-3 val epochs put WS at .7014, but it ships an
ENSEMBLE: three forward passes, and a methods paragraph that has to justify them. Averaging the
weights instead ships ONE model -- the same procedure NMT has applied to its last-N checkpoints
since 2017 -- so if it recovers the gain it is strictly the cheaper claim to defend.

The two mechanisms are NOT the same thing and neither subsumes the other: prediction averaging
reduces the variance of the scores, weight averaging looks for a flatter point in the basin. That
is why this is a measurement and not a refactor.

WHY IT IS WELL-POSED HERE, WHICH IS NOT AUTOMATIC. Weight averaging across checkpoints is unsound
when the module carries BatchNorm running statistics -- the averaged buffers no longer describe the
averaged weights, which is why SWA re-estimates them in a separate pass. Our encoder block is
LayerNorm-only (attention.py:80-81,305-306) and its only buffers are persistent=False positional
tables, so there is nothing to re-estimate. If a BatchNorm ever enters block 12 this file's
assumption breaks silently, and `test_a_buffer_that_moves_between_epochs_is_still_averaged` is the
canary: it documents that we average whatever moved, which is correct ONLY while nothing
stateful moves.

THE LOAD-BEARING SELF-CHECK is `soup_top1`. Averaging one state is that state, so loading it must
reproduce `test_c` EXACTLY -- not to 1e-6 like the rank-tie case in prediction ensembling, but
bit-for-bit, because it is the same weights down the same ridge path. Anything else means the
saved per-epoch states are not the states the loop actually selected on.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, f"{name}.py"))
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


BFT = _mod("v3_board_partialft")


def _state(mlp_w, frozen_w=7.5):
    """One tail-block state: a tensor that moves (the tuned MLP) and one that never does."""
    return [{"mlp.fc1.weight": torch.tensor([[float(mlp_w)]]),
             "norm1.weight": torch.tensor([float(frozen_w)])}]


# ── averaging ───────────────────────────────────────────────────────────────────────────────

def test_average_of_a_single_state_is_that_state_bit_for_bit():
    """THE SELF-CHECK. soup_top1 averages one member, so it must be an identity -- otherwise the
    reported soup numbers are not comparable to test_c at full precision."""
    s = _state(0.1234567891234)
    out = BFT._average_states([s])
    assert torch.equal(out[0]["mlp.fc1.weight"], s[0]["mlp.fc1.weight"])
    assert torch.equal(out[0]["norm1.weight"], s[0]["norm1.weight"])


def test_averaging_identical_states_is_a_no_op():
    s = _state(0.3)
    out = BFT._average_states([s, _state(0.3), _state(0.3)])
    assert torch.equal(out[0]["mlp.fc1.weight"], s[0]["mlp.fc1.weight"])


def test_a_tensor_that_moved_is_averaged_elementwise():
    out = BFT._average_states([_state(0.0), _state(1.0), _state(2.0)])
    assert out[0]["mlp.fc1.weight"].item() == pytest.approx(1.0)


def test_a_tensor_frozen_across_epochs_comes_back_bit_identical():
    """Only the MLP is unfrozen, so every other tensor is the SAME object value at every epoch.
    Averaging three equal floats can perturb the last bit (x+x+x is not always 3x, and /3 need not
    invert it), which would make the 'frozen' half of the block drift for no reason. Entries that
    are identical across members must be passed through untouched, not recomputed."""
    odd = 0.1 + 0.2  # not representable; a naive mean of three copies can land a ulp away
    out = BFT._average_states([_state(1.0, odd), _state(2.0, odd), _state(3.0, odd)])
    assert torch.equal(out[0]["norm1.weight"], torch.tensor([odd])), "frozen tensor drifted"


def test_averaging_preserves_dtype_and_shape():
    a = [{"w": torch.zeros(2, 3, dtype=torch.float32)}]
    b = [{"w": torch.ones(2, 3, dtype=torch.float32)}]
    out = BFT._average_states([a, b])
    assert out[0]["w"].shape == (2, 3) and out[0]["w"].dtype == torch.float32
    assert out[0]["w"].flatten()[0].item() == pytest.approx(0.5)


def test_a_non_float_entry_that_disagrees_is_an_error_not_a_silent_mean():
    """An integer index buffer has no meaningful average. If one ever differs between epochs that
    is a bug upstream, and rounding a mean of indices would hide it."""
    a = [{"idx": torch.tensor([0, 1, 2])}]
    b = [{"idx": torch.tensor([0, 1, 3])}]
    with pytest.raises(Exception):
        BFT._average_states([a, b])


def test_a_non_float_entry_that_agrees_is_passed_through():
    a = [{"idx": torch.tensor([4, 5, 6])}]
    out = BFT._average_states([a, [{"idx": torch.tensor([4, 5, 6])}]])
    assert torch.equal(out[0]["idx"], torch.tensor([4, 5, 6]))


def test_every_block_in_the_tail_is_averaged_not_just_the_first():
    """--split-at 9 makes the tail three blocks. Averaging only blocks[0] would silently leave the
    rest at whatever epoch happened to be loaded last."""
    a = [{"w": torch.tensor([0.0])}, {"w": torch.tensor([0.0])}]
    b = [{"w": torch.tensor([2.0])}, {"w": torch.tensor([4.0])}]
    out = BFT._average_states([a, b])
    assert out[0]["w"].item() == pytest.approx(1.0)
    assert out[1]["w"].item() == pytest.approx(2.0)


def test_a_buffer_that_moves_between_epochs_is_still_averaged():
    """DOCUMENTS THE ASSUMPTION. We average whatever moved, with no BatchNorm re-estimation pass.
    That is sound only because block 12 is LayerNorm-only today. This test does not assert the
    behaviour is right for running stats -- it pins that they WOULD be averaged, so that if a
    stateful norm is ever added, the reviewer of that change sees the unhandled case here."""
    a = [{"running_mean": torch.tensor([0.0])}]
    b = [{"running_mean": torch.tensor([4.0])}]
    assert BFT._average_states([a, b])[0]["running_mean"].item() == pytest.approx(2.0)


# ── the soup rules are the SAME val-only sets as the prediction ensembles ────────────────────

def test_soups_reuse_the_prediction_ensembles_index_sets():
    """The comparison soup-vs-ensemble is only apples-to-apples if BOTH average the same epochs.
    Sharing `_ensemble_index_sets` is what makes that structural instead of a coincidence of two
    hand-written rule tables that can drift apart."""
    vals = [0.60, 0.71, 0.65, 0.70, 0.58]
    seen = {}

    def refit(avg):
        seen[avg[0]["tag"].item()] = True
        return 0.5

    states = [[{"tag": torch.tensor([float(i)])}] for i in range(len(vals))]
    out = BFT._weight_soups(vals, states, refit)
    assert set(out) == {"soup_all", "soup_valge0", "soup_top3", "soup_top1",
                        "soup_last3", "soup_last5", "soup_swa"}


def test_soup_top1_refits_exactly_the_epoch_the_loop_selected():
    """soup_top1 must hand refit() the UNAVERAGED state of the argmax epoch -- epoch 1 here, whose
    weight tensor is 1.0. This is what makes soup_top1 == test_c a real self-check rather than a
    number that merely lands nearby."""
    vals = [0.60, 0.71, 0.65]
    states = [[{"w": torch.tensor([float(i)])}] for i in range(3)]
    seen = {}

    def refit(avg):
        seen.setdefault("calls", []).append(avg[0]["w"].item())
        return 0.77

    out = BFT._weight_soups(vals, states, refit)
    assert BFT._ensemble_index_sets(vals)["ens_top1"] == [1]
    assert out["soup_top1"] == pytest.approx(0.77)
    assert pytest.approx(1.0) in seen["calls"], "top1 never refit the argmax epoch's own weights"


def test_soup_rules_are_a_function_of_val_alone():
    """Same submittability invariant as the prediction ensembles: the epoch SET may not depend on
    anything the test set produced. refit() returns wildly different numbers here and the requested
    index sets must be unchanged."""
    vals = [0.60, 0.71, 0.65, 0.70, 0.58]
    states = [[{"w": torch.tensor([float(i)])}] for i in range(len(vals))]
    seen_a, seen_b = [], []
    BFT._weight_soups(vals, states, lambda a: seen_a.append(a[0]["w"].item()) or 0.1)
    BFT._weight_soups(vals, states, lambda a: seen_b.append(a[0]["w"].item()) or 0.9)
    assert seen_a == seen_b


def test_nan_val_epochs_are_never_souped():
    """A failed fit's weights are not a checkpoint worth averaging in."""
    vals = [0.60, float("nan"), 0.72]
    states = [[{"w": torch.tensor([float(i)])}] for i in range(3)]
    seen = []
    BFT._weight_soups(vals, states, lambda a: seen.append(a[0]["w"].item()) or 0.5)
    # ens_all averages epochs 0 and 2 -> mean 1.0; epoch 1 (nan val) must not contribute
    assert pytest.approx(1.0) in seen


# ── greedy soup (Wortsman et al. 2022) — the rule with NO free parameter ─────────────────────

def test_greedy_soup_starts_from_the_best_val_epoch():
    """The published algorithm sorts candidates by val and seeds the soup with the best one. That
    seed is what makes greedy soup's floor the val-argmax model rather than an arbitrary point."""
    vals = [0.60, 0.71, 0.65]
    states = [[{"w": torch.tensor([float(i)])}] for i in range(3)]
    assert BFT._greedy_soup(vals, states, lambda avg: 0.0)[0] == 1


def test_greedy_soup_rejects_a_member_that_does_not_improve_the_soups_val():
    """THE DEFINING BEHAVIOUR, and what separates it from `valge0`/`last-N`: the criterion is the
    val of the RESULTING AVERAGE, not the val of the candidate on its own. A member with excellent
    solo val is still refused if souping it in makes the soup worse."""
    vals = [0.99, 0.98, 0.97]
    states = [[{"w": torch.tensor([float(i)])}] for i in range(3)]
    ing = BFT._greedy_soup(vals, states, lambda avg: 0.5 if avg[0]["w"].item() == 0.0 else 0.1)
    assert ing == [0], "a candidate that lowers the soup's val was souped in anyway"


def test_greedy_soup_accepts_a_member_that_improves_the_soups_val():
    vals = [0.99, 0.98]
    states = [[{"w": torch.tensor([0.0])}], [{"w": torch.tensor([2.0])}]]
    # solo best (w=0) scores 0.5; the average (w=1.0) scores better, so it must be kept
    ing = BFT._greedy_soup(vals, states, lambda avg: 0.9 if avg[0]["w"].item() == 1.0 else 0.5)
    assert ing == [0, 1]


def test_greedy_soup_never_sees_test():
    """Submittability. The rule is a function of the val callback alone -- there is no test
    argument it could read even by accident."""
    import inspect
    prm = list(inspect.signature(BFT._greedy_soup).parameters)
    assert not any("test" in p for p in prm), f"greedy soup takes a test-shaped argument: {prm}"


def test_greedy_soup_skips_failed_epochs():
    vals = [0.60, float("nan"), 0.72]
    states = [[{"w": torch.tensor([float(i)])}] for i in range(3)]
    seen = []

    def val_of(avg):
        seen.append(avg[0]["w"].item())
        return 0.5

    assert 1 not in BFT._greedy_soup(vals, states, val_of)


def test_greedy_soup_on_a_one_epoch_trace_is_that_epoch():
    assert BFT._greedy_soup([0.6], [[{"w": torch.tensor([1.0])}]], lambda avg: 0.5) == [0]


def test_greedy_soup_has_no_free_parameter():
    """WHY THIS RULE AND NOT last-N. `last3`/`last5` carry an N that we would have to justify --
    Vaswani used 5 for the base model and 20 for the big one, i.e. they tuned it. Greedy soup has
    nothing to tune: the val comparison decides the size of the soup. If a threshold or a count
    ever appears in this signature, that claim is no longer true."""
    import inspect
    prm = inspect.signature(BFT._greedy_soup).parameters
    extra = [n for n, p in prm.items() if p.default is not inspect.Parameter.empty]
    assert extra == [], f"greedy soup grew a tunable knob: {extra}"


def test_every_rule_returns_a_number_on_a_one_epoch_trace():
    """Degenerate cell (FT never ran): the reader must not see a ragged key set."""
    states = [[{"w": torch.tensor([1.0])}]]
    out = BFT._weight_soups([0.6], states, lambda a: 0.61)
    assert set(out) == {"soup_all", "soup_valge0", "soup_top3", "soup_top1",
                        "soup_last3", "soup_last5", "soup_swa"}
    assert all(v == pytest.approx(0.61) for v in out.values())
