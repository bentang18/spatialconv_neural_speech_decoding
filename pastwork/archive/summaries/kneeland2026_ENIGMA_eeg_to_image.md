# Kneeland et al. 2026 - ENIGMA: EEG-to-Image Decoding

> NOTE: The file `kunz2025_inner_speech.pdf` at the expected path actually contains this paper, not a Kunz et al. inner speech paper.

## Citation
Kneeland R, Jiang W, Bruzadin Nunes U, Scotti PS, Delorme A, Xu J. "ENIGMA: EEG-to-Image in 15 Minutes Using Less Than 1% of the Parameters." arXiv:2602.10361v1 [q-bio.NC], February 2026.

## Setup
- **Task:** Reconstruct seen images from EEG recordings (EEG-to-Image)
- **Datasets:**
  - THINGS-EEG2: 64-channel research-grade EEG (ActiveTwo at 1000 Hz), 10 subjects, 16,740 unique images, ~820k trials
  - Alljoined-1.6M: 32-channel consumer-grade EEG (Emotiv Flex 2, ~$2,200, at 250 Hz), 20 subjects, ~1.6M trials
- **Paradigm:** Rapid serial visual presentation (RSVP); epochs -200 to 1000 ms relative to image onset
- **Architecture (ENIGMA):** 4 components:
  1. Shared spatio-temporal CNN backbone (temporal conv -> avg pool -> spatial conv -> flatten -> latent z_EEG in R^184)
  2. Subject-wise linear latent alignment layers (lightweight per-subject adaptation)
  3. MLP projector to CLIP ViT-H/14 embedding space (R^1024)
  4. SDXL Turbo image generation via IP-Adapter
- **Loss:** MSE + InfoNCE contrastive loss between EEG embedding and CLIP image embedding (lambda=0.5)
- **Baselines:** ATM-S (transformer-based, ~128-383M params), Perceptogram (linear map to CLIP)

## Key Findings
- **Parameter efficiency:** ENIGMA uses ~2.4M parameters total across all subjects vs. ATM-S's 383M (30 subjects) -- a ~165x reduction
- **SOTA performance:** Achieves best or tied-best scores on both THINGS-EEG2 and Alljoined-1.6M across low-level (PixCorr, SSIM), high-level (CLIP, Inception, AlexNet), and human rater metrics
- **Rapid fine-tuning:** With multi-subject pretraining, ENIGMA surpasses fully-trained ATM-S with only ~15 minutes of new subject data (~4,000 samples); without pretraining, 15 min is insufficient
- **Hardware robustness:** Unlike ATM-S (whose complex architecture breaks down on consumer-grade EEG), ENIGMA maintains performance across hardware quality levels
- **Ablations:** Latent alignment layers are critical for multi-subject generalization; spatio-temporal conv is essential in all configurations; transformer encoder and diffusion prior stages (from ATM-S) actually hurt performance in multi-subject and low-SNR settings
- **Scaling:** Performance scales log-linearly with training samples on both datasets; no saturation observed; research-grade hardware yields faster scaling
- **Channel count:** Performance degrades gracefully; meaningful decoding possible with as few as 24 channels

## Cross-Patient Relevance (brief)
This is a visual decoding / EEG paper -- **not directly relevant to uECOG overt speech decoding**. Key transferable ideas:

- **Subject-wise latent alignment layers** are a lightweight, effective mechanism for cross-subject generalization without full per-subject models -- analogous strategy could be applied to cross-patient speech decoding with uECOG
- **Multi-subject pretraining + rapid fine-tuning** framing (15 min of data to reach full performance) is a practical target worth aiming for in BCI speech work
- **Shared spatio-temporal backbone with per-subject heads** is a cleaner alternative to full per-patient retraining; the parameter count stays constant regardless of number of subjects
- **Contrastive (InfoNCE) + MSE loss** combination for aligning neural embeddings to a target space is applicable to speech decoding into phoneme/word embedding spaces
- The finding that architectural complexity (transformers, diffusion priors) hurts generalization in noisy/multi-subject settings is a useful cautionary note for speech decoder design
