# Karpowicz et al. 2024 - FALCON: Few-shot Adaptive decoding for intracortical BCI recaLibratiON

## Citation
Karpowicz, B.M., Ali, Y.H., Wimalasena, L., Sedler, A.R., Keshtkaran, M.R., Bodkin, K., Ma, X., Miller, L.E., & Pandarinath, C. (2024). Stabilizing brain-computer interfaces through alignment of latent dynamics. *NeurIPS 2024 Datasets and Benchmarks Track*.

## Setup
- **Recording modalities**: Utah microelectrode arrays (96-192ch), intracortical spikes + spike-band power
- **Species**: Monkeys (M1, M2) and humans (H1, H2, B1 songbird)
- **Data scale**: 5 datasets. M1/M2: 2D cursor reaching, 20-109 sessions. H1: 2D cursor reaching, 9 sessions. H2: handwriting, 10 sessions. B1: zebra finch song, 10 days
- **Tasks**: Calibrated recalibration -- given 1-2 min of labeled data from a new session, adapt a pre-trained decoder
- **Compute**: Benchmark infrastructure, not training-heavy

## Architecture
The benchmark evaluates multiple architectures rather than proposing one. Key entrants:

- **NDT2 Multi**: Multi-session causal Transformer. Session-specific embeddings. Masked token prediction pretraining. Evaluated zero-shot and with few-shot recalibration via last-layer retraining
- **CORP (Karpowicz 2022)**: Fixed-point smoother with test-time adaptation. Online Procrustes alignment of neural manifold to reference session. Lightweight -- no deep network, just manifold rotation/scaling
- **NoMAD**: LFADS-based latent dynamics model with KL-based alignment network. See separate summary
- **H2 RNN baseline**: Per-session affine + shared 2-layer GRU(512) + 3-gram LM. The strongest communication decoder architecture
- **CycleGAN / ERASER / SABLE**: Unsupervised alignment methods. Map neural activity from new sessions to reference session distribution

## SSL/Pretraining
- **NDT2 Multi**: Masked token prediction on multi-session data (supervised sessions included)
- **CORP**: No pretraining -- purely online test-time adaptation
- **NoMAD**: LFADS autoencoder pretraining on reference session(s)
- **Benchmark**: All methods get access to the same held-in sessions for training/pretraining. Evaluated on held-out sessions with 1-2 min calibration budget

## Cross-Patient Handling
- **Within-subject only**: All datasets are multi-session from the SAME subject. Cross-subject is explicitly out of scope
- **Cross-session shift**: The core problem. Electrode impedance drift, micromotion, neural population turnover cause day-to-day distribution shifts
- **Calibration budget**: 1-2 minutes of labeled data from the new session -- must adapt decoder quickly
- **Per-session affine**: H2 baseline uses per-session affine layer (Linear + bias) as the adaptation mechanism. Same architecture as Nason/Willett day-specific layers

## I/O Features
- **Input**: Spike counts + threshold crossings at 20-50ms bins (M1/M2/H1), handwriting traces (H2), song spectrograms (B1)
- **Output**: 2D cursor velocity (M1/M2/H1), character probabilities (H2), song reconstruction (B1)
- **Calibration data**: 1-2 min paired neural + behavioral labels from the new session

## Key Results
| Dataset | Best Method | Metric | Score |
|---|---|---|---|
| M1 (reaching) | NDT2 Multi | R^2 (held-out) | 0.59 |
| M2 (reaching) | NDT2 Multi | R^2 (held-out) | 0.70 |
| H1 (reaching) | CORP | R^2 (held-out) | 0.60 |
| H2 (handwriting) | CORP | WER | 0.11 |
| B1 (song) | NDT2 Multi | R^2 | 0.66 |

Key findings:
- **Deep networks are unstable zero-shot**: NDT2 Multi degrades to R^2 = -0.60 without recalibration on new sessions. Deep models are MORE sensitive to distribution shift than simple linear decoders
- **CORP wins communication**: Test-time adaptation via online Procrustes alignment is the best strategy for H2 handwriting. Lightweight online methods beat heavyweight pretrained models for the most clinically relevant task
- **Unsupervised alignment is marginal**: CycleGAN, NoMAD (unsupervised mode), and ERASER provide only modest improvements. Behavioral labels during calibration are much more valuable than unsupervised manifold matching
- **1-2 min calibration is sufficient**: With the right method, a minute or two of calibration data recovers most decoder performance. This is clinically practical
- **No single winner**: NDT2 Multi dominates movement decoding, CORP dominates communication. The optimal recalibration strategy is task-dependent

## v12 Comparison

**Directly relevant for v12's future clinical deployment and recalibration strategy.** FALCON benchmarks the core problem of neural decoder stability across sessions -- the same problem v12 will face when deployed across patients.

**Key lessons for v12:**

1. **Deep networks need recalibration**: Zero-shot transfer of deep models fails catastrophically (R^2 = -0.60). This validates v12's per-patient layers as non-negotiable -- the model MUST adapt to each new patient/session. The 134-param diagonal normalization is v12's recalibration mechanism
2. **Per-session affine is the standard adaptation**: H2's per-session affine + shared GRU(512) + LM is architecturally identical to v12's design philosophy (per-patient diagonal + shared backbone + constrained decoder). This is the established pattern for clinical iBCI
3. **Online/lightweight adaptation beats heavyweight pretraining**: CORP's Procrustes alignment (test-time, no gradient) outperforms NDT2 Multi (pretrained Transformer) for handwriting. v12's per-patient layers (diagonal scale+bias, no retraining backbone) follow this lightweight adaptation principle
4. **1-2 min calibration budget**: v12 has ~1 min epoched data per patient -- exactly the FALCON calibration budget. This is a validated regime for few-shot adaptation

**What to import:**
- **Benchmark protocol**: FALCON's held-in/held-out session split with fixed calibration budget is a rigorous evaluation framework. v12 could adopt a similar protocol for evaluating cross-patient transfer (held-out patients, minimal per-patient calibration)
- **CORP's Procrustes alignment**: Online manifold alignment without gradient-based adaptation. Could serve as a fast initialization for v12's per-patient Fourier delta/omega before fine-tuning
- **Zero-shot as diagnostic**: Testing v12 zero-shot (without per-patient layers) is a critical diagnostic, following FALCON's finding that zero-shot deep models are catastrophically unstable

**What doesn't transfer:**
- **Within-subject only**: FALCON is entirely within-subject multi-session. v12's core challenge is cross-subject with different electrode placements, not same-subject temporal drift. The distribution shifts are qualitatively different (sensor permutation vs impedance drift)
- **Spike-based input**: All FALCON datasets use intracortical spikes from Utah arrays. Different biophysics from v12's surface HGA -- can't pool, can't directly compare numbers
- **Task simplicity**: 2D cursor reaching (2 DoF continuous) is far simpler than 9-class phoneme sequence decoding. Handwriting is closer but uses a constrained character vocabulary with language model support
