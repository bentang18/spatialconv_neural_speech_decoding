#!/bin/bash
#SBATCH --job-name=v3_r6_ms0_cb8250
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
# RACE maskspace0 at a DEMONSTRATED occupancy. This is v3_r6_maskspace0_lr6e-3_40k_cb5500.sbatch
# VERBATIM with exactly three deltas: contact-budget 5500 -> 8250, accum 12 -> 8, and a
# --resume-ckpt off the running arm's ladder. Everything else is byte-copied, deliberately.
#
# WHY. The original sized cb to 77% GPU occupancy for "keeper parity", but maskspace25 had
# ALREADY DEMONSTRATED 95% (90.3 GiB) runs fine. Under the fitted model GiB ~ 39.0 + 136.8*v
# at cb11000 (the 39.0 is cb-independent), the cb-dependent term scales with cb:
#   v=0.50, cb=5500 -> 39.0 + 136.8*0.50*(5500/11000)  = 73.2 GiB = 77%   <- what is running
#   v=0.50, cb=8250 -> 39.0 + 136.8*0.50*(8250/11000)  = 90.3 GiB = 95%   <- this file
# 90.3 GiB is not an extrapolation: maskspace25 at cb11000/v=0.375 is the SAME 90.3 GiB and it
# ran. Wall time is ~flat in visible fraction; the cliff is in cb -- 11000/6 = 3.35 s/step,
# 8250/8 = 3.53, 5500/12 = 4.806 (+41% over 8250 for nothing). Live micro-batch heartbeat on
# 2764373 is ~0.362 s x accum 12 = ~4.34 s/step, so the real gain is ~4.34 -> ~3.5 s/step.
#
# TOKEN-EXACT, so this is NOT a config change to the experiment: 8250 x 8 == 66000 ==
# 5500 x 12 contacts per OPTIMIZER step. Micro-batches are equal-sized within each config, so
# mean-of-means == global mean and the gradient is the same up to fp accumulation order. The
# asserted invariant below is the thing that makes the switch legitimate mid-run; if it ever
# fails, the arm is no longer comparable and the job MUST NOT start.
#
# RACE, DO NOT REPLACE. 2764373 keeps running. 95% occupancy is tight and the fallback for an
# OOM here is "nothing happened". Never scancel the incumbent to make room for this.
#
# SEPARATE CKPT_DIR IS LOAD-BEARING. Both jobs write ladder-step=<N>.ckpt every 1000 steps. If
# they shared a directory they would interleave ladders from two different throughput regimes
# and a later probe would silently read a MIXED ladder -- the cross-ckpt slip that already cost
# us one wrong csession number. So this writes its own dir and reads the incumbent's read-only.
#
# EVERY CONFIG FLAG IS RE-PASSED ON PURPOSE. Nothing about the masking config lives in the
# state_dict, and dispatch_v3.py:546 documents the precedent: L1RoPE.idx_freq is
# persistent=False, so a resume that drops a flag silently restores the DEFAULT. Drop
# --mask-space-frac 0.0 here and this arm silently becomes the 0.50 baseline and looks like a
# null result. --ssl-max-steps stays 40000 so the LR decay window is unchanged (no
# --stable-steps => LR decays over total_steps, lr_schedule.py); changing it would put this
# arm on a different LR at every shared step and break the 40k-wave comparison.
set -uo pipefail
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

CB=8250
ACCUM=8
TOKENS=$((CB * ACCUM))
if [ "$TOKENS" -ne 66000 ]; then
  echo "FATAL: cb($CB) x accum($ACCUM) = $TOKENS != 66000 contacts/step." >&2
  echo "  The whole basis for switching cb mid-run is that the optimizer step is TOKEN-EXACT." >&2
  echo "  Off-budget => this arm is not comparable to the 40k wave. Refusing to start." >&2
  exit 1
fi

RUN_NAME=v3_r6_maskspace0_lr6e-3_40k_cb8250
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
SRC_CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_v3_r6_maskspace0_lr6e-3_40k
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

if [ "$CKPT_DIR" = "$SRC_CKPT_DIR" ]; then
  echo "FATAL: race ckpt dir == incumbent ckpt dir; would interleave two ladders." >&2; exit 1
fi

# Resume from the LATEST ladder ckpt that has settled. Picked at job START, not at submit, so a
# long queue wait is not wasted -- the incumbent keeps laddering while this pends and we resume
# from wherever it got to. The -mmin +2 filter skips a ckpt the incumbent may still be writing;
# torch.save is not atomic and a torn ckpt would fail the load (loudly, but hours later).
RESUME=""
RESUME_STEP=""
for f in "$SRC_CKPT_DIR"/ladder-step=*.ckpt; do
  [ -f "$f" ] || continue
  [ -n "$(find "$f" -maxdepth 0 -mmin +2 2>/dev/null)" ] || continue
  s=${f##*step=}; s=${s%.ckpt}
  case "$s" in ''|*[!0-9]*) continue ;; esac
  if [ -z "$RESUME_STEP" ] || [ "$s" -gt "$RESUME_STEP" ]; then RESUME_STEP=$s; RESUME=$f; fi
done
if [ -z "$RESUME" ]; then
  echo "FATAL: no settled ladder-step=*.ckpt in $SRC_CKPT_DIR" >&2; exit 1
fi
if [ "$RESUME_STEP" -ge 40000 ]; then
  echo "FATAL: resume step $RESUME_STEP >= 40000; the arm is already done, nothing to race." >&2
  exit 1
fi

echo "[ms0 race] cb=$CB accum=$ACCUM tokens/step=$TOKENS (== incumbent 5500x12, TOKEN-EXACT)"
echo "[ms0 race] predicted mem = 39.0 + 136.8*0.50*($CB/11000) = 90.3 GiB = 95% (maskspace25 proved this footprint)"
echo "[ms0 race] resume <- $RESUME (step $RESUME_STEP), $((40000 - RESUME_STEP)) steps left to 40000"
echo "[ms0 race] ckpt-dir  -> $CKPT_DIR  (SEPARATE from incumbent, no ladder interleave)"
echo "[ms0 race] delta vs r6 baseline is STILL --mask-space-frac 0.0, re-passed below"

srun "$VENV" -u -m speech_decoding.experiments.dispatch_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --session 1:0 --session 2:1 --session 2:2 --session 2:3 --session 2:5 \
  --session 2:6 --session 3:2 --session 4:2 --session 6:0 --session 6:1 \
  --session 6:4 --session 8:0 --session 9:0 \
  --frontend v3r6 --objective mae --mask-space-frac 0.0 \
  --batch-unit shaft --contact-budget $CB --shaft-alpha 0.5 \
  --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACCUM \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 12 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --resume-ckpt "$RESUME" \
  --ssl-max-steps 40000
