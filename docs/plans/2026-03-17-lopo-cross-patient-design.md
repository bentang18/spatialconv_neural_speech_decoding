# Cross-Patient LOPO Training — Design

## Overview

Three-module decomposition for Leave-One-Patient-Out cross-validation:
1. **lopo_trainer.py** — Stage 1 multi-patient training
2. **adaptor.py** — Stage 2 target adaptation with source replay
3. **lopo.py** — LOPO orchestrator + Wilcoxon statistics

## Stage 1: Multi-Patient Training (`lopo_trainer.py`)

### Data Flow
```
Per step (1 of 2000):
  optimizer.zero_grad()
  for each source patient p:
    x, y = random_batch(train_data[p], B=16)
    x = augment(x, cfg=stage1.augmentation)  # on CPU
    x = x.to(device)
    shared = read_ins[p](x)       # per-patient Conv2d
    h = backbone(shared)          # shared
    log_probs = head(h)           # shared
    loss = ctc_loss(log_probs, y) / N_source
    loss.backward()               # accumulate grads
  clip_grad_norm_(all_params, 5.0)
  optimizer.step()
  scheduler.step()
```

### Key Details
- **Optimizer**: 9 param groups (7 source read-ins at 3× LR + backbone + head at 1× LR)
- **Scheduler**: warmup (20 epochs) + cosine decay (reused from per-patient trainer)
- **Validation**: every 100 steps, forward ALL held-out val data per patient → mean CTC loss
- **Val split**: 80/20 stratified per source patient (~840 train / ~210 val across 7 sources)
- **Early stopping**: patience 5 (= 500 steps without improvement)
- **No augmentation during validation**
- **Output**: checkpoint dict with backbone, head, and {pid: read_in} state dicts

## Stage 2: Target Adaptation (`adaptor.py`)

### Frozen/Unfrozen Sets
- **Frozen**: Conv1d + BiGRU (backbone temporal processing) + all source read-ins
- **Trainable**: target read-in (fresh init) + LayerNorm + articulatory head

### Data Flow Per Step
```
# Target trials (70%)
n_target = int(0.7 * B)  # ~11 of 16
x_tgt, y_tgt = random_batch(target_train, n_target)
x_tgt = augment(x_tgt, cfg=stage2.augmentation)  # lighter
shared_tgt = target_read_in(x_tgt.to(device))     # TRAINABLE
h_tgt = backbone(shared_tgt)                       # FROZEN
log_probs_tgt = head(h_tgt)                        # TRAINABLE
loss_tgt = ctc_loss(log_probs_tgt, y_tgt)

# Source replay (30%) — round-robin across source patients
n_source = B - n_target  # ~5
x_src, y_src = sample_from_random_source(source_data)  # round-robin
with torch.no_grad():
    shared_src = source_read_ins[pid](x_src.to(device))  # FROZEN
# backbone outside no_grad: LayerNorm gets grads, Conv1d+GRU don't
h_src = backbone(shared_src)                              # LayerNorm TRAINABLE
log_probs_src = head(h_src)                               # TRAINABLE
loss_src = ctc_loss(log_probs_src, y_src)

loss = 0.7 * loss_tgt + 0.3 * loss_src
loss.backward()
```

### Evaluation Structure
- **5-fold stratified CV** on target patient (stratify on first-phoneme label)
- **Within each fold**: 80/20 split of training portion for early stopping
  - ~150 trials → 120 train/fold → 96 train / 24 val, 30 test
- **Early stopping**: patience 10, eval every 1 step
- **Source replay sampling**: round-robin across source patients (uniform, not proportional to dataset size)

## LOPO Orchestrator (`lopo.py`)

```python
for target_pid in all_patients:
    sources = {pid: ds for pid, ds in all_datasets.items() if pid != target_pid}
    for seed in seeds:
        checkpoint = train_stage1(sources, config, seed, device)
        metrics = adapt_stage2(checkpoint, all_datasets[target_pid],
                               sources, config, seed, device)
        results[(target_pid, seed)] = metrics

# Per-patient mean across seeds → Wilcoxon signed-rank test
```

## Config Structure

LOPO uses `default.yaml` with augmentation nested inside each stage:
```yaml
training:
  stage1:
    steps: 2000           # 1 batch/patient/step; matches per-patient optimization budget
    lr: 1.0e-3
    warmup_epochs: 20
    readin_lr_mult: 3.0
    weight_decay: 1.0e-4
    batch_size: 16
    grad_clip: 5.0
    patience: 5           # × eval_every = 500 steps
    eval_every: 100
    augmentation:
      time_shift_frames: 20
      amp_scale_std: 0.15
      channel_dropout_max: 0.2
      noise_frac: 0.02
      feat_dropout_max: 0.3
      time_mask_min: 2
      time_mask_max: 4
      temporal_stretch: false

  stage2:
    steps: 100            # per-fold; eval_every=1 so patience=10 = 10 steps
    lr: 1.0e-3
    warmup_epochs: 0
    readin_lr_mult: 3.0
    weight_decay: 1.0e-3
    batch_size: 16
    grad_clip: 5.0
    patience: 10
    eval_every: 1
    cv_folds: 5
    val_fraction: 0.2
    source_replay_frac: 0.3
    augmentation:
      time_shift_frames: 10
      amp_scale_std: 0.1
      channel_dropout_max: 0.1
      noise_frac: 0.02
      feat_dropout_max: 0.2
      time_mask_min: 2
      time_mask_max: 4
      temporal_stretch: false
```

Per-patient config (`per_patient.yaml`) keeps its flat `training.augmentation` structure — no changes needed.

## Backward Compatibility

- Per-patient trainer reads `config["training"]["augmentation"]` — unchanged
- LOPO trainer reads `config["training"]["stage1"]["augmentation"]`
- Adaptor reads `config["training"]["stage2"]["augmentation"]`
- `assemble_model()` unchanged — builds components from `config["model"]`

## Per-Patient Baseline

The existing `trainer.py` serves as the per-patient baseline. Same architecture (assemble_model), same config. No separate baseline module needed.

## New Files

| File | Role |
|------|------|
| `src/speech_decoding/training/lopo_trainer.py` | Stage 1 multi-patient training |
| `src/speech_decoding/training/adaptor.py` | Stage 2 target adaptation |
| `src/speech_decoding/training/lopo.py` | LOPO orchestrator + statistics |
| `scripts/train_lopo.py` | CLI entry point |
| `tests/test_lopo_trainer.py` | Stage 1 tests (synthetic data) |
| `tests/test_adaptor.py` | Stage 2 tests (synthetic data) |
| `tests/test_lopo.py` | Orchestrator tests |

## Implementation Order

1. `lopo_trainer.py` (most complex, defines checkpoint format)
2. `adaptor.py` (depends on Stage 1 checkpoint format)
3. `lopo.py` (thin orchestrator over 1 + 2)
4. Tests for each (TDD — tests first)
5. `default.yaml` config updates
6. `train_lopo.py` CLI
