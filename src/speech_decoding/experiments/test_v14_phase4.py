"""B36 WS-G tests for the Phase-4 frozen-encoder readout (B35).

G1 (freeze + load + optimizer scope):
* construction freezes BOTH the encoder AND the P3-PMA; only the readout Linear
  trains;
* the optimizer param set == the readout Linear only (not ``self.parameters()``);
* ``load_transferable_state`` loads the P3 encoder+PMA strict and DROPS the
  P3-only projector; a missing ``encoder`` key raises; the freeze is re-applied
  after load.

G2 (exact param-count pins, B35 pool-independent):
* frozen PMA == 263,424 (d=256, 8 heads);
* classifier == 514 (binary) / 2,570 (10-way);
* total < 30M and the freeze partition is ``frozen == encoder(measured) + PMA``
  with the per-task Linear the only trainable params. The exact total is
  measure-and-pinned from the assembled **B36 hard-pool** encoder built in-test
  (NOT the pre-B36 figure — see :data:`_EXPECTED_FULL_SCALE_TOTAL`).

G3 (CrossSubject split + BP20 + Neuroprobe metric + D8):
* ``leave_two_subjects_out`` partitions clips; ``assert_no_electrode_overlap``
  enforces BP20 (``train ∩ test electrodes == ∅``) and fires on a leak;
* the probe metric is a finite AUROC/accuracy scalar; backward updates ONLY the
  Linear;
* (D8) a head whose RoPE table is sized for T_p=20 reads a T_p=4 clip without a
  shape error (RoPE downward-generalization).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neuraltrain.optimizers import LightningOptimizer

from speech_decoding.experiments.v14_phase4 import (
    V14Phase4ReadoutExperiment,
    V14Phase4ReadoutModule,
    assert_no_electrode_overlap,
    leave_two_subjects_out,
)
from speech_decoding.models.v14_encoder import (
    V14ParcelCollapsePMA,
    V14ParcelPerceiver,
    V14ParcelPerceiverWithHead,
    V14PmaReadout,
)

_X_NAME = ("electrode_tokens", "support", "valid_mask")

# Measure-and-pin (G2): the exact param total of the **assembled B36 hard-pool**
# encoder + B35 readout, built by ``_make_full_scale_head`` below (d=256,
# depth=6, K=80, M=1, N=6, F=30, T=40, kernel 3/2 — the locked first-pass
# config). This is the freshly-measured post-WS-A figure: the test BUILDS the
# encoder and re-measures every run, so this constant is a regression tripwire,
# not a copied estimate. It happens to round to the pre-B36 "~12.12M" because
# the depth-6 d=256 inter-parcel transformer dominates the count and is
# unchanged — the hard block-diagonal pool replaced the soft-routing cross-attn
# at near-parity param count. If the encoder topology drifts, the structural
# asserts (frozen == encoder + PMA; total == frozen + trainable) still hold and
# this pin must be updated deliberately.
_EXPECTED_FULL_SCALE_TOTAL = 12_119_810


def _optim_config(lr: float = 1e-3) -> LightningOptimizer:
    return LightningOptimizer(optimizer={"name": "Adam", "lr": lr})


def _make_tiny_head(
    *,
    n_classes: int = 2,
    n_time_bins: int = 8,
    n_freq_bins: int = 4,
    k_parcels: int = 5,
    d_model: int = 16,
    n_heads: int = 4,
    readout: str = "pma_mean_linear",
) -> V14ParcelPerceiverWithHead:
    cfg = V14ParcelPerceiver(
        n_freq_bins=n_freq_bins,
        n_time_bins=n_time_bins,
        k_parcels=k_parcels,
        d_model=d_model,
        n_heads=n_heads,
        depth_self_attn=1,
        m_sub_slots=1,
        n_token_blocks=1,
        patch_kernel_freq=2,
        patch_kernel_time=2,
        readout=readout,  # type: ignore[arg-type]
    )
    head = cfg.build(n_outputs=n_classes)
    assert isinstance(head, V14ParcelPerceiverWithHead)
    return head


def _make_full_scale_head(n_classes: int = 2) -> V14ParcelPerceiverWithHead:
    """The locked B36 first-pass config (d=256, depth=6, K=80, M=1, N=6)."""
    cfg = V14ParcelPerceiver(
        n_freq_bins=30,
        n_time_bins=40,
        k_parcels=80,
        d_model=256,
        n_heads=8,
        depth_self_attn=6,
        m_sub_slots=1,
        n_token_blocks=6,
        patch_kernel_freq=3,
        patch_kernel_time=2,
        readout="pma_mean_linear",
    )
    head = cfg.build(n_outputs=n_classes)
    assert isinstance(head, V14ParcelPerceiverWithHead)
    return head


def _make_module(
    head: V14ParcelPerceiverWithHead | None = None,
    *,
    n_classes: int = 2,
    **head_kwargs,
) -> V14Phase4ReadoutModule:
    if head is None:
        head = _make_tiny_head(n_classes=n_classes, **head_kwargs)
    return V14Phase4ReadoutModule(
        model=head,
        loss=nn.CrossEntropyLoss(),
        optim_config=_optim_config(),
        x_name=_X_NAME,
        y_name="target",
    )


def _make_batch(
    *, B: int = 2, C: int = 5, K: int = 5, n_time_bins: int = 8,
    n_freq_bins: int = 4, n_classes: int = 2,
) -> SimpleNamespace:
    torch.manual_seed(0)
    electrode_tokens = torch.randn(B, C, n_time_bins, n_freq_bins)
    support = torch.zeros(B, C, K)
    for i in range(min(C, K)):
        support[:, i, i] = 1.0
    valid_mask = torch.ones(B, C, dtype=torch.bool)
    # Balanced labels so binary AUROC is well-defined (both classes present).
    target = torch.arange(B) % n_classes
    return SimpleNamespace(
        data={
            "electrode_tokens": electrode_tokens,
            "support": support,
            "valid_mask": valid_mask,
            "target": target,
        }
    )


def _opt(module: V14Phase4ReadoutModule) -> torch.optim.Optimizer:
    """Sibling-parity helper (see ``test_v14_phase_orchestration._opt``): with no
    trainer attached, ``configure_optimizers`` returns either a bare optimizer or
    a ``{"optimizer": ...}`` dict — the only two shapes ``LightningOptimizer.build``
    produces."""
    out = module.configure_optimizers()
    return out["optimizer"] if isinstance(out, dict) else out


# ---------------------------------------------------------------------------
# G1 — freeze + optimizer scope
# ---------------------------------------------------------------------------


def test_construct_freezes_encoder_and_pma() -> None:
    m = _make_module()
    assert all(not p.requires_grad for p in m.model.encoder.parameters())
    assert all(not p.requires_grad for p in m.model.readout.pma.parameters())
    # The Linear is the ONLY trainable thing.
    assert all(p.requires_grad for p in m.model.readout.classifier.parameters())


def test_trainable_set_is_the_classifier_only() -> None:
    m = _make_module()
    trainable = {id(p) for p in m._trainable_parameters()}
    classifier = {id(p) for p in m.model.readout.classifier.parameters()}
    assert trainable == classifier
    # Sanity: no encoder/PMA param leaked into the trainable set.
    frozen = {id(p) for p in m.model.encoder.parameters()} | {
        id(p) for p in m.model.readout.pma.parameters()
    }
    assert trainable.isdisjoint(frozen)


def test_optimizer_wraps_only_the_classifier() -> None:
    """configure_optimizers builds over ``_trainable_parameters`` — with the
    encoder + PMA frozen that is exactly the readout Linear (G1 contract)."""
    m = _make_module()
    optimizer = _opt(m)
    opt_ids = {id(p) for grp in optimizer.param_groups for p in grp["params"]}
    classifier = {id(p) for p in m.model.readout.classifier.parameters()}
    assert opt_ids == classifier


@pytest.mark.parametrize(
    "readout, extra_trainable",
    [
        ("pma_mean_linear", set()),
        ("pma_flatten_linear", set()),
        ("pma_timeattn_linear", {"time_score.weight", "time_score.bias"}),
    ],
)
def test_freeze_and_trainable_set_across_readout_variants(
    readout: str, extra_trainable: set[str],
) -> None:
    """G1 across all three B35 readouts (mean default + flatten + timeattn
    sisters): the encoder + PMA are frozen, and the trainable set is exactly the
    readout's classifier (plus the timeattn score vector for that sister) — the
    ``mean``-only G1 tests above never exercised the variant trainable sets."""
    m = _make_module(readout=readout)
    assert all(not p.requires_grad for p in m.model.encoder.parameters())
    assert all(not p.requires_grad for p in m.model.readout.pma.parameters())
    trainable = {id(p) for p in m._trainable_parameters()}
    expected_names = {"classifier.weight", "classifier.bias"} | extra_trainable
    expected = {
        id(p) for n, p in m.model.readout.named_parameters() if n in expected_names
    }
    assert trainable == expected


def test_optimizer_step_updates_only_the_linear() -> None:
    """End-to-end grad scope under a REAL optimizer step (the param-group test
    above only checks membership, never that a step leaves the frozen weights
    untouched): after one ``optimizer.step()`` every frozen encoder + PMA tensor
    is byte-identical and only the readout Linear has moved."""
    torch.manual_seed(0)
    m = _make_module(n_classes=2)
    m.train()
    opt = _opt(m)
    batch = _make_batch(n_classes=2)
    frozen_before = {
        n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad
    }
    # Pin the frozen set structurally so this test is non-vacuous in isolation:
    # if a regression unfroze the encoder/PMA, ``frozen_before`` would shrink and
    # the byte-identity loop below would silently check fewer params.
    expected_frozen = {
        n for n, _ in m.named_parameters()
        if n.startswith("model.encoder.") or n.startswith("model.readout.pma.")
    }
    assert set(frozen_before) == expected_frozen
    w_before = m.model.readout.classifier.weight.detach().clone()
    opt.zero_grad()
    loss = m.loss(m.forward(batch), m._target(batch))
    loss.backward()
    opt.step()
    for n, p in m.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p.detach(), frozen_before[n])
    assert not torch.equal(m.model.readout.classifier.weight.detach(), w_before)


def test_rejects_non_pma_readout() -> None:
    """The meanpool/attentive B34 sisters have no PMA to load — the P4 module
    must reject them loudly rather than silently skip the P3-PMA handoff."""
    head = _make_tiny_head(readout="meanpool")
    with pytest.raises(TypeError, match="V14PmaReadout"):
        V14Phase4ReadoutModule(
            model=head, loss=nn.CrossEntropyLoss(), optim_config=_optim_config(),
            x_name=_X_NAME, y_name="target",
        )


# ---------------------------------------------------------------------------
# G1 — cross-phase handoff (E4)
# ---------------------------------------------------------------------------


def test_transferable_state_is_encoder_pma_only() -> None:
    m = _make_module()
    state = m.transferable_state()
    # The per-task Linear is NOT a foundation weight — only encoder + PMA carry.
    assert set(state.keys()) == {"encoder", "pma"}


def test_load_from_p3_loads_encoder_pma_and_drops_projector() -> None:
    src = _make_module()
    with torch.no_grad():
        src.model.readout.pma.query.fill_(0.5)
        src.model.encoder.encoder_ln.weight.fill_(3.0)
    # P3 exports encoder + pma + projector; P4 must take encoder+pma, drop proj.
    p3_state = {
        "encoder": src.model.encoder.state_dict(),
        "pma": src.model.readout.pma.state_dict(),
        "projector": {"bogus": torch.zeros(1)},
    }
    dst = _make_module()
    dst.load_transferable_state(p3_state)
    torch.testing.assert_close(
        dst.model.readout.pma.query.detach(),
        torch.full_like(dst.model.readout.pma.query, 0.5),
    )
    torch.testing.assert_close(
        dst.model.encoder.encoder_ln.weight.detach(),
        torch.full_like(dst.model.encoder.encoder_ln.weight, 3.0),
    )
    # Freeze re-applied after load (encoder + PMA still frozen).
    assert all(not p.requires_grad for p in dst.model.encoder.parameters())
    assert all(not p.requires_grad for p in dst.model.readout.pma.parameters())


def test_load_missing_encoder_raises() -> None:
    m = _make_module()
    # Match the custom message, not the bare ``KeyError('encoder')`` that a
    # naive ``state["encoder"]`` access would also raise (which contains
    # "encoder" too) — pin that the explicit guard is what fires.
    with pytest.raises(KeyError, match="cannot warm-start"):
        m.load_transferable_state({"pma": m.model.readout.pma.state_dict()})


def test_load_refreezes_an_unfrozen_encoder_and_pma() -> None:
    """``load_transferable_state`` re-applies the freeze. A P3-3b checkpoint
    trains the encoder + PMA unfrozen; P4 must re-freeze them on load. Exercises
    the ``_apply_freeze`` call inside ``load_transferable_state`` doing real work
    (start from a deliberately unfrozen dst)."""
    src = _make_module()
    dst = _make_module()
    dst.model.encoder.requires_grad_(True)
    dst.model.readout.pma.requires_grad_(True)
    dst.load_transferable_state(src.transferable_state())
    assert all(not p.requires_grad for p in dst.model.encoder.parameters())
    assert all(not p.requires_grad for p in dst.model.readout.pma.parameters())


def test_transferable_state_roundtrips_through_a_p4_clone() -> None:
    """P4 is terminal but the WS-E driver snapshots it; save→load must restore
    the frozen foundation (encoder + PMA)."""
    src = _make_module()
    with torch.no_grad():
        src.model.readout.pma.query.fill_(-0.25)
    dst = _make_module()
    dst.load_transferable_state(src.transferable_state())
    torch.testing.assert_close(
        dst.model.readout.pma.query.detach(),
        torch.full_like(dst.model.readout.pma.query, -0.25),
    )


# ---------------------------------------------------------------------------
# G2 — exact param-count pins
# ---------------------------------------------------------------------------


def test_frozen_pma_param_count_is_263424() -> None:
    pma = V14ParcelCollapsePMA(d_model=256, n_heads=8)
    assert sum(p.numel() for p in pma.parameters()) == 263_424


def test_classifier_param_counts_binary_and_10way() -> None:
    r_bin = V14PmaReadout(
        pma=V14ParcelCollapsePMA(256, 8), temporal="mean", d_model=256,
        n_classes=2, n_time_patches=4,
    )
    assert sum(p.numel() for p in r_bin.classifier.parameters()) == 514
    r_10 = V14PmaReadout(
        pma=V14ParcelCollapsePMA(256, 8), temporal="mean", d_model=256,
        n_classes=10, n_time_patches=4,
    )
    assert sum(p.numel() for p in r_10.classifier.parameters()) == 2_570


def test_full_scale_total_under_30m_and_freeze_partition_binary() -> None:
    head = _make_full_scale_head(n_classes=2)
    m = V14Phase4ReadoutModule(
        model=head, loss=nn.CrossEntropyLoss(), optim_config=_optim_config(),
        x_name=_X_NAME, y_name="target",
    )
    total = sum(p.numel() for p in m.parameters())
    frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in m.model.encoder.parameters())

    # Freeze partition: the frozen set is exactly encoder ∪ PMA.
    assert frozen == encoder_params + 263_424
    # Only the binary classifier trains.
    assert trainable == 514
    assert total == frozen + trainable
    assert total < 30_000_000
    # Exact regression pin on the assembled B36 hard-pool total.
    assert total == _EXPECTED_FULL_SCALE_TOTAL


def test_full_scale_total_10way_classifier() -> None:
    head = _make_full_scale_head(n_classes=10)
    m = V14Phase4ReadoutModule(
        model=head, loss=nn.CrossEntropyLoss(), optim_config=_optim_config(),
        x_name=_X_NAME, y_name="target",
    )
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable == 2_570
    assert sum(p.numel() for p in m.parameters()) < 30_000_000


# ---------------------------------------------------------------------------
# G3 — CrossSubject leave-2-out split + BP20 electrode-overlap guard
# ---------------------------------------------------------------------------


def test_leave_two_subjects_out_partitions_and_no_electrode_overlap() -> None:
    subject_ids = [0, 0, 1, 2, 2, 3]
    electrode_ids = [[0, 1], [0, 1], [0, 1, 2], [0], [0], [5, 6]]
    train_idx, test_idx = leave_two_subjects_out(subject_ids, held_out=(1, 2))
    assert train_idx == [0, 1, 5]
    assert test_idx == [2, 3, 4]
    train_e, test_e = assert_no_electrode_overlap(
        subject_ids, electrode_ids, train_idx, test_idx,
    )
    # BP20: no electrode shared between the probe train and test pools.
    assert train_e & test_e == set()


def test_leave_two_subjects_out_rejects_nondistinct() -> None:
    with pytest.raises(ValueError, match="DISTINCT"):
        leave_two_subjects_out([0, 1, 2], held_out=(1, 1))


def test_leave_two_subjects_out_rejects_absent_subject() -> None:
    with pytest.raises(ValueError, match="absent"):
        leave_two_subjects_out([0, 1, 2], held_out=(1, 9))


def test_assert_no_electrode_overlap_fires_on_a_leak() -> None:
    # A (broken) split that puts the SAME (subject, electrode) in both pools.
    with pytest.raises(AssertionError, match="BP20"):
        assert_no_electrode_overlap([7, 7], [[0], [0]], train_idx=[0], test_idx=[1])


# ---------------------------------------------------------------------------
# G3 — Neuroprobe probe metric + grad scope
# ---------------------------------------------------------------------------


def test_probe_metric_finite_binary() -> None:
    m = _make_module(n_classes=2)
    batch = _make_batch(n_classes=2)
    logits = m.forward(batch)
    metrics = m._probe_metrics(logits, m._target(batch))
    assert set(metrics) == {"test_acc", "test_auroc"}
    assert torch.isfinite(metrics["test_acc"])
    assert torch.isfinite(metrics["test_auroc"])


def test_probe_metric_multiclass_accuracy_only() -> None:
    m = _make_module(n_classes=4)
    batch = _make_batch(B=4, n_classes=4)
    logits = m.forward(batch)
    metrics = m._probe_metrics(logits, m._target(batch))
    assert set(metrics) == {"test_acc"}
    assert torch.isfinite(metrics["test_acc"])


def test_probe_metric_binary_separable_is_perfect() -> None:
    """Pin the metric VALUES, not just finiteness: a confidently-correct binary
    batch scores acc==1.0 and AUROC==1.0. A constant-0.5 metric (the surviving
    mutation under the finiteness-only tests) fails here."""
    m = _make_module(n_classes=2)
    logits = torch.tensor([[3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [-3.0, 3.0]])
    target = torch.tensor([0, 1, 0, 1])
    metrics = m._probe_metrics(logits, target)
    assert metrics["test_acc"].item() == 1.0
    assert metrics["test_auroc"].item() == 1.0


def test_probe_metric_binary_inverted_auroc_is_zero() -> None:
    """A confidently-WRONG batch scores AUROC==0.0 — and would score 1.0 if the
    AUROC read class 0 instead of class 1. Pins the ``softmax(logits)[:, 1]``
    score wiring (the class-swap mutation flips this 0.0 → 1.0)."""
    m = _make_module(n_classes=2)
    logits = torch.tensor([[-3.0, 3.0], [3.0, -3.0], [-3.0, 3.0], [3.0, -3.0]])
    target = torch.tensor([0, 1, 0, 1])
    metrics = m._probe_metrics(logits, target)
    assert metrics["test_acc"].item() == 0.0
    assert metrics["test_auroc"].item() == 0.0


def test_probe_metric_accuracy_is_micro_not_macro() -> None:
    """A class-imbalanced batch where micro (0.75) ≠ macro (0.5): predict class 0
    for ``target=[0,0,0,1]``. Pins ``average="micro"`` — a micro→macro swap reads
    0.5 here and fails."""
    m = _make_module(n_classes=4)
    logits = torch.zeros(4, 4)
    logits[:, 0] = 5.0
    target = torch.tensor([0, 0, 0, 1])
    metrics = m._probe_metrics(logits, target)
    assert metrics["test_acc"].item() == pytest.approx(0.75)


def test_lifecycle_accumulates_metric_over_the_full_test_set(monkeypatch) -> None:
    """The G3 headline contract: the Neuroprobe metric is computed ONCE over the
    full probe test set, not averaged per batch. Drive the real test hooks
    (``on_test_epoch_start`` → ``test_step`` ×2 → ``on_test_epoch_end``) and
    assert the logged scalars equal the POOLED dataset metric. A first-batch-only
    accumulator, a no-op ``test_step``, or a per-batch average would log a
    different value (or none) — none of which the per-call ``_probe_metrics``
    tests above can catch."""
    torch.manual_seed(0)
    m = _make_module(n_classes=2)
    m.eval()  # deterministic forward (dropout off) so the reference matches
    b1 = _make_batch(B=4, n_classes=2)
    b2 = _make_batch(B=4, n_classes=2)
    b2.data["electrode_tokens"] = b2.data["electrode_tokens"] + 0.7  # differ from b1
    logged: dict = {}
    monkeypatch.setattr(m, "log", lambda name, value, **kw: logged.__setitem__(name, value))

    m.on_test_epoch_start()
    m.test_step(b1, 0)
    m.test_step(b2, 1)
    m.on_test_epoch_end()

    # Reference: pooled over the full 8-sample set, computed independently.
    l1, l2 = m.forward(b1).detach(), m.forward(b2).detach()
    t1, t2 = m._target(b1), m._target(b2)
    pooled = m._probe_metrics(torch.cat([l1, l2]), torch.cat([t1, t2]))
    assert "test_auroc" in logged and "test_acc" in logged
    torch.testing.assert_close(logged["test_auroc"], pooled["test_auroc"])
    torch.testing.assert_close(logged["test_acc"], pooled["test_acc"])
    # Discriminating: the pooled value differs from a first-batch-only metric, so
    # this would catch an ``on_test_epoch_end`` that read only batch 0.
    batch0 = m._probe_metrics(l1, t1)
    assert not torch.isclose(pooled["test_auroc"], batch0["test_auroc"])


def test_target_squeezes_trailing_singleton_and_casts_to_long() -> None:
    """``_target`` squeezes a ``(B, 1)`` target and casts to long (the CE loss
    needs int64 class indices). Exercises the squeeze/cast branch the 1-D-target
    tests never hit."""
    m = _make_module(n_classes=2)
    batch = _make_batch(n_classes=2)
    batch.data["target"] = torch.tensor([[0.0], [1.0]])  # (B, 1) float
    t = m._target(batch)
    assert t.shape == (2,)
    assert t.dtype == torch.long
    assert t.tolist() == [0, 1]


def test_backward_updates_only_the_linear() -> None:
    m = _make_module(n_classes=2)
    batch = _make_batch(n_classes=2)
    loss = m.loss(m.forward(batch), m._target(batch))
    loss.backward()
    # The Linear got a finite gradient.
    assert m.model.readout.classifier.weight.grad is not None
    assert torch.isfinite(m.model.readout.classifier.weight.grad).all()
    # The frozen encoder + PMA got NO gradient.
    assert all(p.grad is None for p in m.model.encoder.parameters())
    assert all(p.grad is None for p in m.model.readout.pma.parameters())


# ---------------------------------------------------------------------------
# G3 (D8) — RoPE downward-generalization: T_p=20 head reads T_p=4
# ---------------------------------------------------------------------------


def test_d8_rope_downward_generalization_t20_to_t4() -> None:
    # Build at the 5 s pretraining clip (n_time_bins=40 → T_p=20): the RoPE
    # table is sized for 20 patches. P4 then reads a 1 s clip (T=8 → T_p=4).
    head = _make_tiny_head(n_classes=2, n_time_bins=40)
    m = _make_module(head=head, n_classes=2)
    batch = _make_batch(n_time_bins=8, n_classes=2)  # T_p=4 < the 20-patch table
    logits = m.forward(batch)  # must NOT raise a shape error
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Experiment-layer phase guard (mirrors the V14Phase3Experiment guards)
# ---------------------------------------------------------------------------


def test_experiment_rejects_phase_not_4() -> None:
    for bad_phase in (1, 2, "3a", "3b"):
        with pytest.raises(ValueError, match="Phase-4"):
            V14Phase4ReadoutExperiment.model_construct(phase=bad_phase)


def test_experiment_accepts_phase_4() -> None:
    xp = V14Phase4ReadoutExperiment.model_construct(phase=4)
    xp.model_post_init(None)  # no raise
    assert xp.phase == 4
