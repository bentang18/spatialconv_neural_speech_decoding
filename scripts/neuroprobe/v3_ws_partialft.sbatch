#!/bin/bash
#SBATCH --job-name=ws_partialft
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=113560M
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/ws_partialft_%A_%a.out
#
# WS partial fine-tuning pilot — unfreeze the last encoder block, keep the ridge readout fixed,
# and ask whether the frozen-probe column we have always quoted is leaving a MAE-style gap.
#
# --mem=113560M is one GPU's DefMemPerGPU so ghx4 MAX_TRES billing stays at 1000 (same rule as
# v3_probe_encode_r6.sbatch). The x11 cache lives on the GPU (~25 GB of 96), not in host RAM.
#
# COST MODEL (why 2h is generous, not sloppy): the script caches tap-11 ONCE per session with a
# full 12-block forward (~40 s, matching the measured probe encode), then re-runs ONLY block 12
# per epoch — ~1.5 s train fwd+bwd + ~3.3 s eval fwd + ~0.7 s ridge/eigh = ~6 s/epoch. With
# early stopping that is 2-4 min/cell. Stage A shards 3 ways by lr (8 cells each, 20-35 min);
# stage B shards 7 ways by session (8 cells each, ~35 min). SLURM bills actual usage, so the
# walltime margin is free; it only affects backfill priority.
#
# STAGE A  (lr sweep on ONE session, val-selected):
#   sbatch --array=0-2 v3_ws_partialft.sbatch a
# STAGE A-wd (the wd leg; a + awd 0.5 + awd 3.0 tile a 3x3 lr x wd grid):
#   sbatch --array=0-2 v3_ws_partialft.sbatch awd 0.5
# STAGE B  (full 7x4x2 grid at the stage-A winner):
#   sbatch --array=0-6 v3_ws_partialft.sbatch b <LR>
# STAGE C  (the corrected head -- matched bn preprocessing + multitask, sharded by question):
#   sbatch --array=0-2 v3_ws_partialft.sbatch c
# STAGE D  (parameter-count ladder at the bn head -- norm/BitFit and mlp, never yet run):
#   sbatch --array=0-1 v3_ws_partialft.sbatch d
# STAGE DX (finish the mlp ladder -- stage D's mlp was still rising at its top lr):
#   sbatch --array=0 v3_ws_partialft.sbatch dx
# STAGE E  (the full 7-session grid at the corrected head -- the only stage that clears the null):
#   sbatch --array=0-6 v3_ws_partialft.sbatch e <LR>
# STAGE EM (the missing 2x2 corner: stage E with --arms mlp -- BCE + mlp over the same 28 cells):
#   sbatch --array=0-6%3 v3_ws_partialft.sbatch em <LR>      # run at BOTH 1e-2 and 1e-3
# STAGE F  (the block-DEPTH ladder: 2 and 3 trainable blocks vs the 1-block incumbent):
#   sbatch --array=0-1 v3_ws_partialft.sbatch f
# STAGE G  (two-sided: drive block 12 with the REPORTED ridge's leave-one-out risk, not BCE):
#   sbatch --array=0-1 v3_ws_partialft.sbatch g
# STAGE H  (stage E minus --multitask, same LR: E - H is the multitask term over 28 cells):
#   sbatch --array=0-6 v3_ws_partialft.sbatch h <LR>
# STAGE R  (stage G's driver over stage E's 28 cells; %3 -- 4 co-resident tasks OOM'd the loader):
#   sbatch --array=0-6%3 v3_ws_partialft.sbatch r <LR>
# STAGE RF (do depth and driver COMPOSE? 2 blocks + ridge driver, one session, precondition):
#   sbatch --array=0 v3_ws_partialft.sbatch rf
# STAGE RM (mlp sublayer vs whole block UNDER THE RIDGE DRIVER -- picks the CS arm):
#   sbatch --array=0 v3_ws_partialft.sbatch rm
# STAGE RMS (the candidate arm over 28 cells: mlp + ridge driver; R - RMS = the attention term):
#   sbatch --array=0-6%3 v3_ws_partialft.sbatch rms <LR>
# STAGE RMSB (is the winner an artifact of the MINIBATCH LOO? --train-batch ladder vs RMS's 128):
#   sbatch --array=0-6%3 v3_ws_partialft.sbatch rmsb 1e-2 256
#   sbatch --array=0-6%3 v3_ws_partialft.sbatch rmsb 1e-2 512
set -euo pipefail

STAGE="${1:?usage: sbatch v3_ws_partialft.sbatch <stage> [LR] [TRAIN_BATCH]}"
BASE=/projects/bhqk/htang13
CKPT=$BASE/v3_ckpt_v3_r6_pbspace111_sf50_lr6e-3_40k/ladder-step=20000.ckpt
TAG=pbs50_20k

# The arm is pbs50 (job 2764203) — the adoption candidate — so the FT delta lands on the number
# we are actually choosing, and pairs against the pbs50/pbs25 double dissociation read at the
# SAME step. Its frozen probe AND its encode cache already exist, which is what makes the
# feature-level parity gate (L0) possible at all.
BASELINE=$BASE/results_v3_probe_$TAG.json
PARITY_CACHE=$BASE/v3_probe_cache_$TAG

case "$STAGE" in
  a) LRS=(1e-4 3e-4 1e-3)
     # Both arms sweep here: for `head` the lr is the logistic head's own step size, so it needs
     # its own winner. Selecting one lr on block12 and imposing it on head would weaken cell B.
     SEL="--stage a --session-index 0 --lrs ${LRS[$SLURM_ARRAY_TASK_ID]} --arms block12"
     OUT=$BASE/ws_partialft_stageA_lr${LRS[$SLURM_ARRAY_TASK_ID]}.json ;;
  awd) WD="${2:?stage awd needs a weight decay}"
     # The WD leg of stage A (Ben 07-30: "add a weight decay arm as well"). Same one session and
     # same lr ladder as stage a, at a different wd, so a and awd together tile a 3x3 lr x wd grid
     # with no duplicated work. WHY IT IS A REAL KNOB HERE and not a copy of MAE's 0.05: the head
     # is ~213k params against ~3.6k training windows, ~60x overparameterized, where MAE's ViT-L
     # head faced 1.28M images. MAE itself brackets the range rather than settling it — 0.05 for
     # end-to-end FT, 0 for linear probing ("regularization is in general harmful for linear
     # probing", mae.tex:616), 0.3 for supervised-from-scratch ViT-L (mae.tex:701). No verified
     # source for wd 3.0 was found, so it is included as an ARM to be measured, not as a citation.
     LRS=(1e-4 3e-4 1e-3)
     SEL="--stage a --session-index 0 --lrs ${LRS[$SLURM_ARRAY_TASK_ID]} --wd $WD --arms block12"
     OUT=$BASE/ws_partialft_stageA_wd${WD}_lr${LRS[$SLURM_ARRAY_TASK_ID]}.json ;;
  c) # STAGE C -- THE CORRECTED-HEAD STAGE. Sharded by QUESTION, not by lr: stage A showed that
     # sweeping lr cannot rescue a head whose PREPROCESSING does not match the ridge's, so each
     # task here changes exactly ONE thing and carries its own lr ladder.
     #   0  head_bn      : encoder FROZEN, bn head. This is R0 measured in the PRIMAL -- the same
     #                     term v3_frozen_readout_sweep.py fits in the DUAL on the same features,
     #                     so the two experiments cross-check on real data. Free invariant: no
     #                     encoder param moves, so cell C MUST equal cell A; any C != A is a
     #                     feature-path bug, not a result. Its lr ladder is 10x higher because a
     #                     zero-init linear on standardized features wants a bigger step than a
     #                     pretrained block does.
     #   1  block12_bn   : the incumbent FT arm with MATCHED preprocessing. Against task 0 it
     #                     isolates "was d(D-A) = -.0060 a LayerNorm artifact?", because task 0
     #                     supplies the readout-cost offset that -.0060 confounds.
     #   2  block12_bn_mt: adds the multi-task target block -- 4 label columns on the SAME train
     #                     rows, which is VideoMAE v2's post-pre-training lever (arXiv 2303.16727)
     #                     in its leak-free form. Against task 1 it isolates "was the FT starved of
     #                     supervision?" -- our block 12 is ~790k params against ~1.3k rows.
     ARMS=(head block12 block12)
     CLRS=(1e-3,3e-3,1e-2 1e-4,3e-4,1e-3 1e-4,3e-4,1e-3)
     MTF=("" "" "--multitask")
     NAMES=(head_bn block12_bn block12_bn_mt)
     I=$SLURM_ARRAY_TASK_ID
     SEL="--stage a --session-index 0 --lrs ${CLRS[$I]} --arms ${ARMS[$I]} --head-norm bn ${MTF[$I]}"
     OUT=$BASE/ws_partialft_stageC_${NAMES[$I]}.json ;;
  d) # STAGE D -- THE PARAMETER-COUNT LADDER, at the corrected (bn) head. Stage A only ever ran
     # `block12`: ~790k trainable params against n_train ~1.3k rows. If the FT is failing because
     # it is overparameterized rather than because the features cannot improve, a SMALLER arm wins,
     # and the ladder is the instrument that separates those two explanations. Combined with stage
     # C task 0 (head, 0 encoder params) the full ladder is head < norm < mlp < block12.
     # DECISION: if norm or mlp beats block12 by more than noise, the adoption candidate CHANGES to
     # the smaller arm; if the ladder is flat, capacity is not the binding constraint and the
     # remaining suspect is supervision (stage C task 2).
     # Higher lr ladder than block12's: fewer, mostly 1-D params (norm is BitFit) tolerate and need
     # a larger step than a full pretrained block does.
     ARMS=(norm mlp)
     I=$SLURM_ARRAY_TASK_ID
     SEL="--stage a --session-index 0 --lrs 3e-4,1e-3,3e-3 --arms ${ARMS[$I]} --head-norm bn"
     OUT=$BASE/ws_partialft_stageD_${ARMS[$I]}_bn.json ;;
  dx) # STAGE DX -- FINISH THE mlp LADDER. Stage D ran mlp at 3e-4/1e-3/3e-3 and mean val_C rose
     # MONOTONICALLY across the whole range (.8191 -> .8213 -> .8243), so the ladder was truncated
     # at the wrong end: the arm is still improving where stage D stopped looking. norm is NOT
     # extended -- it moved .0006 total across the same decade and sits exactly at frozen on 5 of 8
     # cells, so its ladder is saturated and a higher lr cannot change the verdict.
     # DECISION: if mlp at a higher lr reaches block12's d(C-A), the CS build unfreezes the MLP
     # SUBLAYER ONLY -- half the parameters, no attention, and a cheaper arm to defend
     # cross-subject. If it turns over below block12, the attention sublayer is load-bearing and
     # block12 stays the candidate. Either way the ladder stops being open at its top end.
     SEL="--stage a --session-index 0 --lrs 1e-2,3e-2,1e-1 --arms mlp --head-norm bn"
     OUT=$BASE/ws_partialft_stageDX_mlp_bn_hi.json ;;
  rf) # STAGE RF -- DO THE TWO LEVERS COMPOSE? F varies DEPTH under the BCE driver; G varies the
     # DRIVER at one block. Both moved the number on their own, and nothing so far combines them,
     # so "use both" is currently an assumption. This is the smallest experiment that checks it:
     # ONE session, the deeper split F is testing (2 blocks, split-at 10 with F's train-batch 64 so
     # the memory profile is F's), driven by the ridge LOO risk at a ladder centred on G's winner.
     # It is a PRECONDITION for a 7-session combined run, not a result on its own.
     # DECISION: if RF beats BOTH single levers on the matched session, the 7-session combined arm
     # is the candidate for the morning and the CS build inherits it. If RF only matches the better
     # single lever, they are redundant -- take the CHEAPER one (one block, ridge driver) and do not
     # pay for depth. If RF is worse than both, they interfere and each stays standalone.
     SEL="--stage a --session-index 0 --lrs 1e-3,3e-3,1e-2 --arms blocks --split-at 10
          --head-norm bn --multitask --driver ridge --warmup-epochs 0 --train-batch 64"
     OUT=$BASE/ws_partialft_stageRF_split10_ridge.json ;;
  rm) # STAGE RM -- WHICH ARM GOES INTO THE CS BUILD: the MLP sublayer or the whole block?
     # Under the BCE driver the two TIED (mlp +.0059 vs block12 +.0065 on S1T0, a .0006 gap at
     # n=8), so the cheaper arm was already the better buy. But every ridge-driver number so far is
     # block12, and a tie under one driver does not transfer to another: the ridge LOO risk has a
     # different gradient scale and could easily need the attention sublayer that BCE did not.
     # WHY IT MATTERS BEYOND COST: attention mixes across the SPATIAL axis, and cross-subject is
     # exactly where electrode correspondence does not exist (v3_board_readout.py aligns csession by
     # electrode label; cs has no such alignment). A gain that lives in the per-token MLP has a
     # reason to survive the transfer that an attention-localized gain does not.
     # LR LADDER: mlp's BCE optimum sat 3-10x above block12's, so centre it 3-10x above the ridge
     # driver's 3e-3 winner rather than reusing block12's ladder.
     # DECISION: tie or better -> the CS build unfreezes blk.mlp only. Clearly worse -> attention is
     # load-bearing under this driver and block12 goes in despite the cost.
     SEL="--stage a --session-index 0 --lrs 3e-3,1e-2,3e-2 --arms mlp --head-norm bn
          --multitask --driver ridge --warmup-epochs 0"
     OUT=$BASE/ws_partialft_stageRM_mlp_ridge.json ;;
  rms) LR="${2:?stage rms needs the LR stage rm selected on val}"
     # STAGE RMS -- THE CANDIDATE ARM AT FULL SCALE. Stage RM overturned the BCE-era tie: under the
     # ridge driver the MLP SUBLAYER does not match the whole block, it BEATS it 2.1x on the matched
     # session (+.0202 vs +.0096, val_C .8464 vs .8376 -- val agrees, so not a test-side fluke).
     # THREE RUNS NOW AGREE THAT LESS CAPACITY IS BETTER UNDER THIS DRIVER: 2 blocks (stage RF) went
     # NEGATIVE, 1 block gives +.0096, mlp-only gives +.0202. The leave-one-out ridge risk is a
     # strong low-noise objective on ~1279 train rows and overfits itself when handed more
     # parameters, so dropping attention from the trainable set is the sweet spot.
     # Identical to stage R except --arms mlp, so R - RMS is the attention-sublayer term over the
     # full 28-cell null and the comparison stays a one-thing-changed comparison.
     SEL="--stage b --session-index $SLURM_ARRAY_TASK_ID --lrs $LR --arms mlp --head-norm bn
          --multitask --driver ridge --warmup-epochs 0"
     OUT=$BASE/ws_partialft_stageRMS_s$SLURM_ARRAY_TASK_ID.json ;;
  rmsb) LR="${2:?stage rmsb needs the LR -- 1e-2, the stage RMS winner}"
     TB="${3:?stage rmsb needs a --train-batch: 256 or 512 (128 IS stage RMS, already run)}"
     # STAGE RMSB -- IS THE WINNING RESULT AN ARTIFACT OF THE MINIBATCH LOO? (Ben 07-30)
     # The driver is the leave-one-out risk of the REPORTED estimator, but it is fit on a
     # MINIBATCH: --train-batch defaults to 128 against n_train=1279, and the reported ridge runs
     # at ridge_df = 540.6. A ridge on 128 rows has AT MOST 128 effective dof, so the objective we
     # optimize is a 4.2x-smaller-capacity estimator than the one we report. The first-principles
     # case for this driver is "match the reported estimator", and on sample size we do not.
     # 🚨 WHY IT IS NOT COSMETIC: the headline mechanism -- "the LOO risk overfits ITSELF when
     # handed more parameters" -- is far more true at n=128 than at n=1279. If the full-sample risk
     # is better conditioned it may tolerate the attention sublayer, which would RETRACT the
     # "attention hurts" capacity finding (RMS - R = +.0111, 21/28, p=.0125).
     # DESIGN: identical to stage RMS except --train-batch, so this is a one-thing-changed ladder
     # over the SAME 28 cells. LR is held at RMS's 1e-2 rather than re-swept, which makes the read
     # ASYMMETRIC and that is deliberate:
     #   FLAT or BETTER -> conclusive. A too-small lr for a bigger batch would show a DROP, never
     #                     flatness, so insensitivity at matched lr means the approximation is
     #                     benign. State it as a stated approximation and keep --train-batch 128.
     #   WORSE          -> AMBIGUOUS (n vs lr). Only then re-sweep lr at that batch before
     #                     concluding, and if it survives the sweep, build the exact two-pass
     #                     full-sample LOO (chunked no-grad forward -> Phi, closed-form dL/dPhi,
     #                     chunked backward) which costs ~1.5x a step and NOT more memory.
     # ⚠️ Batch size also changes UPDATES PER EPOCH (1279/tb: 10 at 128, 3 at 512). --epochs is the
     # same and best_epoch is selected on val, so each arm picks its own number of passes; that is
     # the fair comparison, but it is why a bare "worse" cannot be read as "n matters".
     SEL="--stage b --session-index $SLURM_ARRAY_TASK_ID --lrs $LR --arms mlp --head-norm bn
          --multitask --driver ridge --warmup-epochs 0 --train-batch $TB"
     OUT=$BASE/ws_partialft_stageRMSB_tb${TB}_s$SLURM_ARRAY_TASK_ID.json ;;
  b) LR="${2:?stage b needs the stage-A winning LR}"
     # `head` (encoder frozen, logistic head only) is the missing 2x2 cell: it separates readout
     # family from weights, so "FT beats the board" cannot be a logistic-vs-ridge artifact. It is
     # pointless in stage A (there is no encoder lr to sweep) and ~doubles stage B to ~6 GPU-h.
     SEL="--stage b --session-index $SLURM_ARRAY_TASK_ID --lrs $LR --arms block12"
     OUT=$BASE/ws_partialft_stageB_s$SLURM_ARRAY_TASK_ID.json ;;
  e) LR="${2:?stage e needs the stage-C winning LR}"
     # STAGE E -- THE FULL GRID AT THE CORRECTED HEAD. This is the only stage that can clear the
     # PRE-REGISTERED null (28 fold-meaned cells, paired two-sided sign test, 20/28 -> p=.036);
     # stage C runs ONE session, so its d(C-A) is a signal, not a verdict.
     # ⭐ THE HEADLINE IS d(C-A), NOT d(D-A): C and A are the SAME estimator (const-lambda dual
     # ridge) on different WEIGHTS, so their difference is attributable to the features. The frozen
     # sweep settled the other term -- R0[std] = -0.0005 over 56 cells (27/56, p=0.894), i.e. the
     # readout family is FREE at frozen features -- so a negative d(D-A) is now R1, a property of
     # the HEAD on fine-tuned features, and cannot be read as "fine-tuning did not help".
     # bn + multitask because both were established BEFORE this launch: bn beats per-row ln by
     # +.061 at p<1e-4 in the frozen sweep, and multitask is the arm whose C column came back
     # positive 7/8 in stage C.
     SEL="--stage b --session-index $SLURM_ARRAY_TASK_ID --lrs $LR --arms block12 --head-norm bn --multitask"
     OUT=$BASE/ws_partialft_stageE_s$SLURM_ARRAY_TASK_ID.json ;;
  em) LR="${2:?stage em needs an LR: 1e-2 for the mlp BCE optimum, 1e-3 to match stage E}"
     # STAGE EM -- THE MISSING CORNER OF THE 2x2 (Ben 07-30: "do bce+mlp now").
     # The attribution ladder has BCE/block12 (E), ridge/block12 (R) and ridge/mlp (RMS) at full
     # scale, but BCE/mlp ONLY ON ONE SESSION (stage DX: +.0073 at lr 1e-2, and mt=False). So the
     # claimed INTERACTION -- "restricting capacity pays only under the ridge driver" -- currently
     # rests on 4 fold-meaned cells against 28 for every other cell of the ladder.
     # This is stage E with --arms mlp and NOTHING else changed, so E - EM is the attention-sublayer
     # term under BCE over the SAME 28 cells on which R - RMS measures it under the ridge driver.
     # DECISION: if EM >= E, capacity restriction is driver-INDEPENDENT and the mechanism is simply
     # "fewer trainable params against ~1279 rows". If EM ~= E while RMS > R, the interaction is
     # real and the mechanism is specifically that the LOO risk overfits ITSELF when handed the
     # attention sublayer. Those are different claims about WHY, and the CS build inherits the
     # explanation, not just the arm.
     # 📐 PRECEDENT, verified (arXiv 2111.06377v3 Fig 9): MAE ran this exact cut and found the MLP
     # sub-block WORSE than a full block (79.1 vs 81.0) under its supervised/BCE-style objective.
     # Our ridge-driver result INVERTS that ordering (mlp +.0320 vs block12 +.0209). EM is the arm
     # that says whether the inversion is ours or the driver's.
     # LR: run BOTH 1e-2 (mlp's own BCE optimum -- stage DX is a PLATEAU .0073/.0069/.0068 across
     # 1e-2..1e-1, not a peak) and 1e-3 (stage E's lr, protocol-matched). The two arms have
     # different optima, so matching lr handicaps mlp while matching protocol is the other reading;
     # 7 extra tasks buys both and removes the ambiguity.
     SEL="--stage b --session-index $SLURM_ARRAY_TASK_ID --lrs $LR --arms mlp --head-norm bn --multitask"
     OUT=$BASE/ws_partialft_stageEM_lr${LR}_s$SLURM_ARRAY_TASK_ID.json ;;
  f) # STAGE F -- HOW MANY BLOCKS. Every FT number so far unfreezes exactly ONE block.
     # 🔴 CITATION CORRECTED 07-30 (verified against arXiv 2111.06377v3 Fig 9, ar5iv): an earlier
     # version of this comment read "one-block partial FT at 73.5 against 81.0 for the full
     # network", which MISMAPPED the figure. The real ladder is linear probing (0 blocks) 73.5,
     # MLP SUB-BLOCK ONLY 79.1, ONE block 81.0, full fine-tuning ~85. So one block recovers most
     # of the linear-probe->full-FT gap rather than leaving most of it on the table, and depth is
     # a smaller lever than the old misreading implied. The ladder is still worth running.
     # Depth is set by --split-at because the cached tap and the trainable set are the same cut, so
     # parity-L0 (recompute the tail, compare to the on-disk enc12) validates the deeper split
     # instead of taking it on trust. Task 0 = 2 blocks, task 1 = 3 blocks; with the existing
     # 1-block number that is a 3-point depth ladder on one session.
     # DECISION: if d(C-A) GROWS with depth, the CS build goes deep and stage E's 1-block number is
     # a floor, not the answer. If it is flat or falls, depth is not the binding constraint and
     # one block is the right arm -- which retires the biggest open alternative explanation.
     # --train-batch falls with depth to hold ACTIVATION memory roughly fixed: the 1-block arm
     # peaked at 70-72 GiB of 96 with batch 128, and backprop activations scale with the number of
     # trainable blocks while the ~25 GiB tap cache does not. Sized from that measurement, not by
     # halving; the run PRINTS peak_gib, so the next size comes from a number.
     SPLITS=(10 9)
     TB=(64 48)
     I=$SLURM_ARRAY_TASK_ID
     # Lower ladder than the 1-block arm's: 3x the trainable params off the SAME pretrained init,
     # and MAE's partial-FT recipe damps the lr further the deeper it goes (layer-wise decay 0.75).
     SEL="--stage a --session-index 0 --lrs 3e-5,1e-4,3e-4 --arms blocks --head-norm bn --multitask
          --split-at ${SPLITS[$I]} --train-batch ${TB[$I]}"
     OUT=$BASE/ws_partialft_stageF_split${SPLITS[$I]}.json ;;
  g) # STAGE G -- MAKE THE TEST TWO-SIDED. Every arm above drives block 12 with BCE on a trained
     # head, so the ridge is never in the autograd graph and d(C-A) cannot distinguish "the
     # features cannot improve" from "the head is too poor a teacher". Stage C MEASURED that the
     # teacher is poor: head_bn's own R0 came back -.003 to -.035 where the CONVERGED dual fit of
     # the same term is -0.0005, so most of the head's deficit is Adam not finishing.
     # --driver ridge replaces that loss with the LEAVE-ONE-OUT risk of the reported const-lambda
     # dual ridge -- closed form, train rows only, no head to underfit and no val split to
     # overfit. Same estimator in the loss as in the metric.
     # --warmup-epochs 0 because REVE's first step aligns a linear probe with the frozen features,
     # and there is no probe here: the ridge is refit exactly at every evaluation.
     # TWO TASKS ONLY TO BRACKET THE LR. The loss family is new, so its gradient scale is unknown
     # and a single ladder could sit entirely on the wrong side of the optimum; together the two
     # tasks sweep 3e-5 .. 1e-2 in half-decades. This is NOT two arms.
     # DECISION: if d(C-A) under the matched driver EXCEEDS the BCE arm's +.005, the BCE head was
     # the binding constraint and the CS build uses this driver. If it matches or falls, the
     # driver was not the constraint -- which makes the one-sided caveat moot and lets a null on
     # stage E be read as a real negative instead of an uninterpretable one.
     GLRS=(3e-5,1e-4,3e-4 1e-3,3e-3,1e-2)
     I=$SLURM_ARRAY_TASK_ID
     SEL="--stage a --session-index 0 --lrs ${GLRS[$I]} --arms block12 --head-norm bn --multitask
          --driver ridge --warmup-epochs 0"
     OUT=$BASE/ws_partialft_stageG_ridgedriver_$I.json ;;
  h) LR="${2:?stage h needs the same LR stage e used}"
     # STAGE H -- ATTRIBUTE THE +.010. Stage E moved THREE things off the incumbent at once (bn
     # preprocessing, multitask supervision, fine-tuning), so its d(C-A) cannot say which one paid.
     # This is stage E with --multitask REMOVED and nothing else changed: same lr, same 7 sessions,
     # same bn head, same folds, so E - H is the multitask term alone over the full 28 cells.
     # DECISION: if H reproduces E's +.010, multitask is not load-bearing and the CS build drops it
     # -- one fewer moving part in the arm that has to survive cross-subject. If H falls short, the
     # 4x supervision IS the mechanism and it must carry into CS, where n_train per anchor subject
     # is the binding constraint anyway.
     SEL="--stage b --session-index $SLURM_ARRAY_TASK_ID --lrs $LR --arms block12 --head-norm bn"
     OUT=$BASE/ws_partialft_stageH_s$SLURM_ARRAY_TASK_ID.json ;;
  r) LR="${2:?stage r needs the LR stage g bracketed}"
     # STAGE R -- THE RIDGE-DRIVER ARM AT FULL SCALE. Stage G is stage C's question on ONE session:
     # it brackets the lr for a loss family whose gradient scale we had never seen. This is that
     # winner over the SAME 7 sessions x 4 tasks x 2 folds as stage E, so R and E are the identical
     # design differing in ONE thing -- what moves block 12 -- and R - E is the driver term over the
     # full 28-cell null. Nothing else changes: same bn head, same multitask targets (under the
     # ridge driver the 4 label columns simply share one Gram), same folds, same const-lambda rule.
     # --warmup-epochs 0 for the same reason stage G has it: REVE's warmup aligns a linear probe
     # with frozen features and there is no probe here -- the ridge is refit at every evaluation.
     # DECISION: if R clears +.010 by more than E does, the CS build uses the ridge driver and the
     # BCE head is dropped from the pipeline entirely. If R ties E, the driver was not the binding
     # constraint, E's one-sided caveat is retired, and the CS build takes the simpler BCE path.
     SEL="--stage b --session-index $SLURM_ARRAY_TASK_ID --lrs $LR --arms block12 --head-norm bn
          --multitask --driver ridge --warmup-epochs 0"
     OUT=$BASE/ws_partialft_stageR_s$SLURM_ARRAY_TASK_ID.json ;;
  *) echo "FATAL: stage must be 'a', 'awd', 'b', 'c', 'd', 'e', 'em', 'f', 'g', 'h', 'r', 'dx', 'rf', 'rm', 'rms' or 'rmsb', got" \
          "'$STAGE'" >&2
     exit 1 ;;
esac

# pbs50 is an r6 arm => MAE => there is NO EMA teacher; the tower is objective.online.*
# (the r6 encode sbatch passes --online for the same reason). A wrong choice here fails the
# strict state_dict load rather than mis-loading silently, but the default belongs on the arm.
echo "[ws-partialft] stage=$STAGE task=${SLURM_ARRAY_TASK_ID:-0} ckpt=$CKPT"
echo "[ws-partialft] baseline=$BASELINE parity_cache=$PARITY_CACHE out=$OUT"

cd /projects/bhqk/htang13/speech
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

# NO `srun`. MEASURED, not stylistic: a bare srun step lost 4 array tasks to "Error configuring
# interconnect" in ~4 s each (2779639_0 gh078, 2779906_2 gh054, 2779981_0 gh054, 2779981_1 gh005).
# It is not a bad node -- gh054 and gh005 COMPLETED other tasks of the same arrays -- it tracks
# several array tasks landing on one ghx4 node and colliding on step network setup. With
# --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 the batch step already holds the allocation and
# the GPU binding, so srun adds a step that can fail and nothing else. Other launchers here keep
# srun because they submit ONE job; this is the only 7-way array.
#
# THE RISK THIS INTRODUCES, GUARDED: the pilot falls back to CPU when torch.cuda.is_available() is
# False, so a lost GPU binding would look like a very slow run rather than an error. Refuse first.
.venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "FATAL: no CUDA in the batch step -- refusing to run on CPU" >&2; exit 1; }

.venv/bin/python -u -m scripts.neuroprobe.v3_ws_partialft_pilot \
  --ckpt "$CKPT" \
  --baseline-json "$BASELINE" --baseline-prefix "$TAG" \
  --parity-cache-dir "$PARITY_CACHE" \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --tower online \
  --out "$OUT" \
  $SEL
