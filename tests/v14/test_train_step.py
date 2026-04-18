"""Unit tests for the v14-core training loop (Task D3)."""

from __future__ import annotations

import pytest
import torch

from speech_decoding.v14.dataset import N_TIER1_PARCELS, T_SAMPLES
from speech_decoding.v14.model import NeuralFieldPerceiver
from speech_decoding.v14.train import (
    GRAD_ACCUM_STEPS,
    LR,
    WD,
    compute_slot_ce,
    make_cosine_warmup_schedule,
    train_one_fold,
    train_step,
)


def _fake_batch(B: int = 4, N_e: int = 16, H_p: int = 8, W_p: int = 16) -> dict:
    torch.manual_seed(0)
    signal = torch.randn(B, N_e, T_SAMPLES)
    layout = torch.tensor(
        [[[i // W_p, i % W_p] for i in range(N_e)] for _ in range(B)],
        dtype=torch.long,
    )
    active = torch.ones(B, N_e, dtype=torch.bool)
    support = torch.rand(B, N_e, N_TIER1_PARCELS)
    labels = torch.randint(0, 9, (B, 3), dtype=torch.long)
    prev_tokens = torch.full((B, 3), -1, dtype=torch.long)
    prev_tokens[:, 1:] = labels[:, :-1]
    return {
        "signal": signal,
        "electrode_grid_layout": layout,
        "electrode_grid_shape": (H_p, W_p),
        "electrode_active_mask": active,
        "support": support,
        "labels": labels,
        "prev_tokens": prev_tokens,
        "patient_id": "SXX",
    }


class TestComputeSlotCE:
    def test_random_logits_near_log_vocab(self) -> None:
        torch.manual_seed(0)
        B, L, V = 8, 3, 9
        logits = torch.randn(B, L, V) * 0.01
        labels = torch.randint(0, V, (B, L))
        ce = compute_slot_ce(logits, labels)
        # Near-uniform logits → CE ≈ log(V)
        assert abs(float(ce) - torch.log(torch.tensor(V).float()).item()) < 0.1


class TestCosineWarmup:
    def test_warmup_then_cosine(self) -> None:
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([p], lr=LR)
        sched = make_cosine_warmup_schedule(opt, warmup_epochs=5, max_epochs=20)
        lrs: list[float] = []
        for _ in range(20):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()
        assert lrs[0] == pytest.approx(LR / 5)
        assert lrs[4] == pytest.approx(LR)  # full LR at end of warmup
        assert lrs[-1] < 1e-4  # cosine decays to ~0


class TestTrainStep:
    def test_grad_accum_constant_is_four(self) -> None:
        assert GRAD_ACCUM_STEPS == 4

    def test_ten_steps_drop_loss_by_at_least_0_1(self) -> None:
        torch.manual_seed(0)
        model = NeuralFieldPerceiver()
        batch = _fake_batch()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        losses: list[float] = []
        for _ in range(10):
            losses.append(train_step(model, batch, opt))
        assert losses[0] - losses[-1] >= 0.1, f"losses={losses}"

    def test_train_step_returns_scalar_float(self) -> None:
        model = NeuralFieldPerceiver()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        loss = train_step(model, _fake_batch(B=2), opt)
        assert isinstance(loss, float)


class TestBestValRestoration:
    def test_restores_best_val_weights_after_overfit(self) -> None:
        """After training, model weights must equal the best-val checkpoint,
        not the final-epoch weights. Exposes the bug where test PER was
        evaluated on the overfit final model.
        """

        from torch.utils.data import DataLoader

        torch.manual_seed(0)
        model = NeuralFieldPerceiver()

        batch = _fake_batch(B=4)
        # Single-batch "dataset" repeated; train_loader and val_loader both wrap it.
        train_batches = [batch]
        val_batches = [batch]

        # Capture state after any val that is the best seen so far.
        capture: dict = {}

        def _captured_evaluate_factory():
            # Simulate: epoch 0 val=0.5 (best), epoch 1..N val=0.9 (all worse).
            # train_one_fold should snapshot model after epoch 0 and restore on exit.
            import speech_decoding.v14.train as T

            orig_eval = T.evaluate
            call_count = {"n": 0}

            def fake_eval(m, loader):
                call_count["n"] += 1
                return 0.5 if call_count["n"] == 1 else 0.9

            T.evaluate = fake_eval
            return orig_eval, T

        orig_eval, T = _captured_evaluate_factory()
        try:
            # Capture model.state_dict() after first val check.
            first_val_state: dict = {}

            def log(row: dict) -> None:
                # After the first val check fires (epoch 0 with val_every=1),
                # snapshot current model weights for comparison.
                if "val_per" in row and not first_val_state:
                    for k, v in model.state_dict().items():
                        first_val_state[k] = v.detach().clone()

            train_one_fold(
                model,
                DataLoader(train_batches, batch_size=None, collate_fn=lambda b: b),
                DataLoader(val_batches, batch_size=None, collate_fn=lambda b: b),
                max_epochs=6,
                val_every=1,
                patience=3,
                log_callback=log,
            )
        finally:
            T.evaluate = orig_eval

        # After training, model weights must match the best-val (epoch 0) snapshot,
        # NOT the further-trained epoch-N weights.
        final_state = model.state_dict()
        for k in first_val_state:
            assert torch.equal(final_state[k], first_val_state[k]), (
                f"param {k}: final weights diverged from best-val snapshot — "
                "best-val restoration is not happening"
            )
