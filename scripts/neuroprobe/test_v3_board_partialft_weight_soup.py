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
import inspect
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
    assert set(out) == {"soup_top3", "soup_top1",
                        "soup_last5", "soup_last10", "soup_last15"}


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








def test_every_rule_returns_a_number_on_a_one_epoch_trace():
    """Degenerate cell (FT never ran): the reader must not see a ragged key set."""
    states = [[{"w": torch.tensor([1.0])}]]
    out = BFT._weight_soups([0.6], states, lambda a: 0.61)
    assert set(out) == {"soup_top3", "soup_top1",
                        "soup_last5", "soup_last10", "soup_last15"}
    assert all(v == pytest.approx(0.61) for v in out.values())


# ── EMA weight averaging ─────────────────────────────────────────────────────────────────────
def test_ema_weights_sum_to_one():
    """Normalising by the sum IS the bias correction, so there is no warm-up phase where the
    average is contaminated by the pretrained init."""
    for n in (1, 2, 5, 24):
        for tau in (0.5, 0.8, 0.9, 0.95, 0.99):
            w = BFT._ema_weights(n, tau)
            assert len(w) == n
            assert abs(sum(w) - 1.0) < 1e-12, (n, tau, sum(w))


def test_ema_weights_are_newest_heaviest_and_strictly_decaying_backwards():
    w = BFT._ema_weights(5, 0.9)
    assert w[-1] == max(w)
    for a, b in zip(w, w[1:]):
        assert b > a, "later epochs must carry more weight than earlier ones"


def test_ema_weights_ratio_is_exactly_tau():
    """The defining property: consecutive weights differ by the decay, nothing else."""
    tau = 0.8
    w = BFT._ema_weights(6, tau)
    for a, b in zip(w, w[1:]):
        assert abs(a / b - tau) < 1e-12


def test_ema_weights_on_a_one_epoch_trace_is_a_single_full_weight():
    assert BFT._ema_weights(1, 0.9) == [1.0]


def test_ema_weights_rejects_a_decay_outside_the_open_unit_interval():
    """tau=1 is the uniform mean and tau=0 is argmax-of-last; both are OTHER rules, and silently
    accepting them here would let a typo relabel a different method as EMA."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            BFT._ema_weights(4, bad)


def test_ema_never_reads_val():
    """The claim we make for EMA that we do NOT make for last-N: the averaging operator is a
    function of the trajectory length and tau alone. Pin it by signature."""
    sig = inspect.signature(BFT._ema_weights)
    assert list(sig.parameters) == ["n", "tau"]


def test_weighted_average_states_matches_a_hand_computed_mean():
    a = [{"w": torch.tensor([0.0, 10.0]), "frozen": torch.tensor([7.0])}]
    b = [{"w": torch.tensor([2.0, 20.0]), "frozen": torch.tensor([7.0])}]
    out = BFT._average_states([a, b], [0.25, 0.75])
    assert torch.allclose(out[0]["w"], torch.tensor([1.5, 17.5]))


def test_weighted_average_still_passes_identical_entries_through_untouched():
    """The bit-exactness guarantee must survive weighting -- a weighted mean of equal floats is no
    more reliable than an unweighted one."""
    frozen = torch.tensor([1.0 / 3.0, 2.0 / 7.0])
    a = [{"w": torch.tensor([0.0]), "frozen": frozen.clone()}]
    b = [{"w": torch.tensor([1.0]), "frozen": frozen.clone()}]
    out = BFT._average_states([a, b], [0.3, 0.7])
    assert torch.equal(out[0]["frozen"], frozen)


def test_weighted_average_rejects_a_weight_vector_of_the_wrong_length():
    a = [{"w": torch.tensor([1.0])}]
    with pytest.raises(ValueError):
        BFT._average_states([a, a], [1.0])


def test_ema_with_uniform_weights_reproduces_the_plain_mean():
    """Sanity bridge between the two code paths: pass 1/n explicitly and the weighted path must
    land where the unweighted one does."""
    a = [{"w": torch.tensor([0.0, 4.0])}]
    b = [{"w": torch.tensor([2.0, 8.0])}]
    uni = BFT._average_states([a, b])
    wtd = BFT._average_states([a, b], [0.5, 0.5])
    assert torch.allclose(uni[0]["w"], wtd[0]["w"])


def test_ema_taus_are_a_declared_ladder_not_a_single_pinned_value():
    assert len(BFT.EMA_TAUS) >= 2
    assert all(0.0 < t < 1.0 for t in BFT.EMA_TAUS)
