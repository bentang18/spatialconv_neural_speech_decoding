"""B36 WS-I2 — the must-pass capstone: P1→P2→P3→P4 end-to-end on synthetic BT.

This is the definition of done for the whole B36 build (``docs/neuroprobe/
b36_implementation_plan.md`` §"Capstone test spec"). It runs all four staged
phases on a synthetic Nano-shaped fixture (no ``ROOT_DIR_BRAINTREEBANK``, no BT
data, no DCC), chaining one phase's snapshot into the next phase's warm-start
via the WS-E driver primitives (:meth:`Experiment._snapshot` /
:meth:`Experiment._load_pretrained`). Each phase is a real Lightning
``fast_dev_run`` fit, so the genuine train/val loop runs — EMA update (P1/P2),
the connector/encoder freeze regimes (P3), and the frozen-probe optimizer (P4)
all execute, not just ``_step``.

ONE encoder topology is shared across phases (the weights transfer); each phase
builds a fresh encoder of that topology and loads the prior phase's state. The
phases run the shared encoder at DIFFERENT clip lengths — P1/P2 at T=40
(``T_p``=20), **P3 at T=80** (``T_p``=40, to meet the Whisper teacher pool's
hard-pinned 40 frames = 8 Hz × 5 s), P4 at T=8 (``T_p``=4). The conv patch stem
is size-agnostic and RoPE is ``persistent=False`` (recomputed per forward, absent
from the state_dict), so the same parameters run at any ``T_p`` and the strict
cross-phase load stays clean (the P4 ``T_p``=20→4 RoPE-downward guard, D8).

The per-phase assertions below are keyed to the plan's capstone spec table.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from lightning import pytorch as pl
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neuraltrain.optimizers import LightningOptimizer
from neuraltrain.optimizers.base import AdamW

from speech_decoding.bt_alignment.teacher_cache import (
    TargetStandardizer,
    fit_channel_stats,
)
from speech_decoding.experiments.experiment import Experiment
from speech_decoding.experiments.v14_joint_module import V14JointBrainModule
from speech_decoding.experiments.v14_phase3 import V14Phase3DistillModule
from speech_decoding.experiments.v14_phase4 import V14Phase4ReadoutModule
from speech_decoding.models.v14_encoder import (
    V14ParcelPerceiver,
    V14ParcelPerceiverModel,
    V14ParcelPerceiverWithHead,
    V14PmaReadout,
)

# ── Shared fixture geometry (Nano-shaped; d=256 is REQUIRED for the B35 param
# pins — the PMA count + binary classifier are d-locked). depth/N small for
# speed; F=30/F_p=10 (kernel 3), n_time_bins=80 so the encoder default
# max_seq_len=80 admits the P3 T=80 clip without a rebuild. ──
_D_MODEL = 256
_N_HEADS = 8
_K_PARCELS = 80
_C = 8                       # electrodes; one DK parcel each → 8 covered slots
_B = 2                       # clips (matches the 2-clip Whisper cache)
_BASE_LR = 1e-3

_ENC_KW = dict(
    n_freq_bins=30,
    n_time_bins=80,
    k_parcels=_K_PARCELS,
    d_model=_D_MODEL,
    n_heads=_N_HEADS,
    depth_self_attn=2,
    m_sub_slots=1,
    n_token_blocks=2,
    patch_kernel_freq=3,      # F=30 → F_p=10
    patch_kernel_time=2,      # T → T_p = T/2
)

# B35 pool-independent param pins (d=256, 8 heads). Hold regardless of encoder
# depth, so they are asserted exactly here even at the capstone's depth-2 scale.
_PMA_PARAMS = 263_424
_BINARY_CLASSIFIER_PARAMS = 514   # Linear(256, 2) = 256·2 + 2


# ---------------------------------------------------------------------------
# Synthetic batch + loader helpers
# ---------------------------------------------------------------------------


def _support(B: int) -> torch.Tensor:
    """Diagonal one-hot DK support: electrode i → parcel i (``_C`` covered)."""
    s = torch.zeros(B, _C, _K_PARCELS)
    for i in range(_C):
        s[:, i, i] = 1.0
    return s


def _ssl_batch(*, T: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """P1/P2/P3 batch: electrode tokens + support + valid_mask (no labels)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "electrode_tokens": torch.randn(_B, _C, T, 30, generator=g),
        "support": _support(_B),
        "valid_mask": torch.ones(_B, _C, dtype=torch.bool),
    }


class _Batch(dict):
    """The collated batch.

    The modules read tensors via ``batch.data[name]``; here ``.data`` returns the
    dict itself. Crucially this is a ``Mapping`` of tensors, so Lightning's
    ``extract_batch_size`` can introspect it (a ``SimpleNamespace`` is opaque to
    that traversal → ``self.log(on_epoch=True)`` would raise
    ``could not infer the batch_size``).
    """

    @property
    def data(self) -> "_Batch":
        return self


class _DictDataset(Dataset):
    """Serve a pre-stacked batch dict one sample (leading axis) at a time."""

    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self._batch = batch

    def __len__(self) -> int:
        return _B

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {k: v[i] for k, v in self._batch.items()}


def _collate(samples: list[dict[str, torch.Tensor]]) -> _Batch:
    """Re-stack into the NeuralSet-style ``batch.data`` contract the modules read."""
    keys = samples[0]
    return _Batch({k: torch.stack([s[k] for s in samples]) for k in keys})


def _loader(batch: dict[str, torch.Tensor]) -> DataLoader:
    return DataLoader(_DictDataset(batch), batch_size=_B, collate_fn=_collate)


def _trainer() -> pl.Trainer:
    """One-batch CPU fast_dev_run: 1 train + 1 val step, no logger/ckpt."""
    return pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def _optim() -> LightningOptimizer:
    return LightningOptimizer(optimizer=AdamW(lr=_BASE_LR))


def _flat_optimizer(module):
    out = module.configure_optimizers()
    return out["optimizer"] if isinstance(out, dict) else out


def _encoder() -> V14ParcelPerceiverModel:
    return V14ParcelPerceiverModel(**_ENC_KW)


def _p4_head(n_classes: int) -> V14ParcelPerceiverWithHead:
    cfg = V14ParcelPerceiver(**_ENC_KW, readout="pma_mean_linear")
    head = cfg.build(n_outputs=n_classes)
    assert isinstance(head, V14ParcelPerceiverWithHead)
    assert isinstance(head.readout, V14PmaReadout)
    return head


def _snapshot(module, path) -> None:
    Experiment._snapshot(None, module, str(path))  # type: ignore[arg-type]


def _load(module, path) -> None:
    Experiment._load_pretrained(None, module, str(path))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The capstone
# ---------------------------------------------------------------------------


def test_p1_p2_p3_p4_end_to_end_on_bt(tmp_path) -> None:
    """Must-pass, no skip. P1→P2→P3→P4 on a synthetic BT fixture with ckpt
    handoff; every phase fast_dev_run-fits and the boundary loads are strict."""
    ckpt_p1 = tmp_path / "phase_p1.ckpt"
    ckpt_p2 = tmp_path / "phase_p2.ckpt"
    ckpt_p3a = tmp_path / "phase_p3a.ckpt"
    ckpt_p3b = tmp_path / "phase_p3b.ckpt"

    # ===================================================================
    # P1 — front-end masked JEPA (M2, BT-only, paradigm B)
    # ===================================================================
    m1 = V14JointBrainModule(encoder=_encoder(), optim_config=_optim(), phase="p1")

    # (a) finite loss + (c) M2 gradient scope. Done via an explicit backward
    # (robust — a trainer would zero the grads before we could inspect them):
    # P1 is paradigm B (exact parity with P2) — the visible-only front-end
    # produces the M2 context that feeds the P1 predictor, so the front-end AND
    # the predictor get gradient; the pool / inter-parcel encoder is downstream
    # of M2 (m2_only=True) and must get NO gradient.
    brk = m1._step(_ssl_batch(T=40).copy())
    assert torch.isfinite(brk.total)                                  # (a)
    brk.total.backward()
    frontend, parcel = m1.student.encoder.partition_parameters_for_staging()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in frontend)  # (c)
    assert all(p.grad is None for p in parcel)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m1.predictor.parameters()
    )
    m1.zero_grad(set_to_none=True)

    # (d) EMA teacher moves under the fixed-τ rule during the real fit.
    teacher_before = [p.clone() for p in m1.teacher.model.parameters()]
    _trainer().fit(m1, _loader(_ssl_batch(T=40)), _loader(_ssl_batch(T=40)))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(teacher_before, m1.teacher.model.parameters())
    )
    _snapshot(m1, ckpt_p1)
    p1_encoder_state = {k: v.clone() for k, v in m1.student.encoder.state_dict().items()}

    # ===================================================================
    # P2 — pool + encoder + M4 predictor (parcel×time JEPA, paradigm B)
    # ===================================================================
    m2 = V14JointBrainModule(encoder=_encoder(), optim_config=_optim(), phase="p2")
    _load(m2, ckpt_p1)
    # Boundary: the warm-started encoder == P1's post-fit encoder (front-end
    # included), with no missing/unexpected keys (strict load inside _load).
    for k, v in m2.student.encoder.state_dict().items():
        assert torch.equal(v, p1_encoder_state[k]), k

    # (c) discriminative LR: front-end @ base/10, pool/encoder/predictor @ base.
    opt2 = _flat_optimizer(m2)
    assert len(opt2.param_groups) == 2
    assert opt2.param_groups[0]["lr"] == pytest.approx(_BASE_LR * 0.1)
    assert opt2.param_groups[1]["lr"] == pytest.approx(_BASE_LR)

    # (a) finite + paradigm-B predictor gets gradient (P2 supervises M4).
    brk2 = m2._step(_ssl_batch(T=40, seed=1).copy())
    assert torch.isfinite(brk2.total)
    brk2.total.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in m2.predictor.parameters()
    )
    m2.zero_grad(set_to_none=True)

    # (e) RankMe logged from a TRAIN step (I1) on the post-encoder_ln M4 target.
    rank_logged: dict[str, float] = {}
    saved_log = m2.log
    m2.log = lambda k, v, **_kw: rank_logged.__setitem__(  # type: ignore[method-assign]
        k, float(v.detach()) if torch.is_tensor(v) else float(v)
    )
    m2._monitor_from_step(_ssl_batch(T=40, seed=1), step_name="train")
    m2.log = saved_log  # type: ignore[method-assign]
    assert "train_mon_rankme" in rank_logged

    _trainer().fit(m2, _loader(_ssl_batch(T=40)), _loader(_ssl_batch(T=40)))
    _snapshot(m2, ckpt_p2)

    # ===================================================================
    # P3 — Whisper distillation (project-up). 3a connector-only → 3b unfreeze.
    # ===================================================================
    # 2-clip synthetic Whisper cache (mean_all, 250 frames @ 50 Hz). fit_channel_stats
    # pools 250→40 and fits the per-channel z-score over the train clips.
    clips = torch.randn(_B, 250, 1280)
    cache_paths = []
    for i in range(_B):
        film_dir = tmp_path / "wcache" / "film0"
        film_dir.mkdir(parents=True, exist_ok=True)
        p = film_dir / f"clip{i}.pt"
        torch.save({"features": clips[i].to(torch.float16)}, p)
        cache_paths.append(p)
    stats = fit_channel_stats(cache_paths)
    standardizer = TargetStandardizer(stats["mean"], stats["inv_std"])

    def _p3_batch(seed: int) -> dict[str, torch.Tensor]:
        # P3 runs the encoder at T=80 → T_p=40 to meet the teacher's 40 frames.
        b = _ssl_batch(T=80, seed=seed)
        b["whisper_target"] = clips
        return b

    # --- 3a: encoder FROZEN, connector (PMA + projector) trains ---
    m3a = V14Phase3DistillModule(
        encoder=_encoder(), optim_config=_optim(), stage="3a",
        pma_n_heads=_N_HEADS, standardizer=standardizer,
    )
    _load(m3a, ckpt_p2)  # P2 → encoder only; connector stays at init for warmup
    assert all(not p.requires_grad for p in m3a.encoder.parameters())
    assert all(p.requires_grad for p in m3a.pma.parameters())
    assert all(p.requires_grad for p in m3a.projector.parameters())
    assert torch.isfinite(m3a._step(_p3_batch(2)))           # finite SmoothL1 in 1280-d
    _trainer().fit(m3a, _loader(_p3_batch(2)), _loader(_p3_batch(2)))
    _snapshot(m3a, ckpt_p3a)

    # --- 3b: encoder UNFREEZES with 3 discriminative-LR groups ---
    m3b = V14Phase3DistillModule(
        encoder=_encoder(), optim_config=_optim(), stage="3b",
        pma_n_heads=_N_HEADS, standardizer=standardizer,
    )
    _load(m3b, ckpt_p3a)  # carries encoder + warmed connector forward
    assert all(p.requires_grad for p in m3b.encoder.parameters())
    opt3b = _flat_optimizer(m3b)
    assert len(opt3b.param_groups) == 3
    assert opt3b.param_groups[0]["lr"] == pytest.approx(_BASE_LR * 0.1)   # front-end
    assert opt3b.param_groups[1]["lr"] == pytest.approx(_BASE_LR / 3.0)   # parcel
    assert opt3b.param_groups[2]["lr"] == pytest.approx(_BASE_LR)         # connector
    assert torch.isfinite(m3b._step(_p3_batch(3)))
    _trainer().fit(m3b, _loader(_p3_batch(3)), _loader(_p3_batch(3)))
    _snapshot(m3b, ckpt_p3b)

    # ===================================================================
    # P4 — frozen-encoder readout probe (B35: frozen PMA → mean → Linear)
    # ===================================================================
    m4 = V14Phase4ReadoutModule(
        model=_p4_head(n_classes=2),
        loss=nn.CrossEntropyLoss(),
        optim_config=_optim(),
        x_name=("electrode_tokens", "support", "valid_mask"),
        y_name="target",
    )
    _load(m4, ckpt_p3b)  # encoder + PMA strict; the P3-only projector is dropped

    # P4 invariant: encoder AND PMA frozen; the per-task Linear is the ONLY
    # trainable parameter.
    assert all(not p.requires_grad for p in m4.model.encoder.parameters())
    assert all(not p.requires_grad for p in m4.model.readout.pma.parameters())
    trainable = {n for n, p in m4.named_parameters() if p.requires_grad}
    assert trainable == {
        "model.readout.classifier.weight",
        "model.readout.classifier.bias",
    }

    # Exact param-count pins (B35 pool-independent) + structural total.
    pma_params = sum(p.numel() for p in m4.model.readout.pma.parameters())
    clf_params = sum(p.numel() for p in m4.model.readout.classifier.parameters())
    enc_params = sum(p.numel() for p in m4.model.encoder.parameters())
    total = sum(p.numel() for p in m4.parameters())
    assert pma_params == _PMA_PARAMS
    assert clf_params == _BINARY_CLASSIFIER_PARAMS
    assert total == enc_params + pma_params + clf_params  # nothing else in the head
    assert total < 30_000_000

    # RoPE downward-generalization (D8): the encoder loaded from P3 (run at
    # T_p=40) reads a T_p=4 probe clip without a shape error.
    probe = {
        "electrode_tokens": torch.randn(_B, _C, 8, 30),   # T=8 → T_p=4
        "support": _support(_B),
        "valid_mask": torch.ones(_B, _C, dtype=torch.bool),
        "target": torch.tensor([0, 1]),
    }
    logits = m4.forward(SimpleNamespace(data=probe))
    assert logits.shape == (_B, 2)
    assert torch.isfinite(logits).all()

    # (a) 1 train step → backward updates ONLY the Linear (encoder + PMA frozen).
    enc_before = [p.clone() for p in m4.model.encoder.parameters()]
    pma_before = [p.clone() for p in m4.model.readout.pma.parameters()]
    clf_before = [p.clone() for p in m4.model.readout.classifier.parameters()]
    _trainer().fit(m4, _loader(probe), _loader(probe))
    assert all(torch.equal(a, b) for a, b in zip(enc_before, m4.model.encoder.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(pma_before, m4.model.readout.pma.parameters()))
    assert any(
        not torch.equal(a, b)
        for a, b in zip(clf_before, m4.model.readout.classifier.parameters())
    )

    # (b) finite Neuroprobe probe metric over a full test pass (drive the test
    # lifecycle directly so the metric is pooled across the set, not per-batch).
    metric_logged: dict[str, float] = {}
    m4.log = lambda k, v, **_kw: metric_logged.__setitem__(  # type: ignore[method-assign]
        k, float(v.detach()) if torch.is_tensor(v) else float(v)
    )
    m4.eval()
    m4.on_test_epoch_start()
    m4.test_step(SimpleNamespace(data=probe), 0)
    m4.on_test_epoch_end()
    assert math.isfinite(metric_logged["test_acc"])
    assert "test_auroc" in metric_logged and math.isfinite(metric_logged["test_auroc"])
