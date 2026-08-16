#!/bin/bash
#SBATCH --job-name=r6_vits384_tube_2g_smoke
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=72
#SBATCH --mem=227120M
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   SMOKE TEST, RUN THIS BEFORE v3_r6_vits384_tube_2g_55k.sbatch.
#   sbatch scripts/neuroprobe/v3_r6_vits384_tube_2g_smoke.sbatch vits384
#
# ═══ WHAT THIS BUYS AND WHY IT IS NOT OPTIONAL ═══════════════════════════════════════════
# The 2-GPU single-job plan rests on ONE unmeasured number: DDP wall-clock scaling at this
# width. 55,000 x 4.005 s/step is 61.2 h on one GPU, which does not fit a 48 h partition wall;
# the plan only works if two ranks bring it under 47 h. 1.5x is enough (40.8 h) and 1.8x is
# comfortable (34.0 h), but NONE of that is profiled. 300 steps settles it.
# 🔴 --cpus-per-task=72 IS LOAD-BEARING, 36 DOES NOT LAUNCH. A ghx4 node is 288 CPUs / 4x GH200
# as gpu:...:4(S:0-3), ONE GPU PER SOCKET at 72 CPUs each. At 36 both ranks land on one socket
# and contend for its single cxi NIC, and srun dies in 3 s with "Error configuring interconnect"
# (measured, jobs 2950160-63). 🚫 --network=single_node_vni is NOT the fix, probe 2950167
# requested one and reproduced the failure exactly. Billing is unchanged: 1000 x max(cpus/72,
# mem_MB/113560, gpus) is pinned at 2 by the gpus term, so 144 CPUs and 227,120M are free.
# 🚨 THE REMAINING RISK IS MEMORY. Each rank now has its own 113,560M, but both still mmap the
# same band .npy files through 16 workers each. The failure modes are an OOM in the first
# seconds (that is always the LOADER, never the model) and the silent futex hang (GPU at 0%, no
# error, no exit). Both show up well inside 300 steps.
#
# ═══ WHAT TO READ OUT OF THE LOG ═════════════════════════════════════════════════════════
#   1. [mask] fingerprint  -- must say the shaft time-tube contract
#   2. the [hb] heartbeat lines from BOTH rank=0 and rank=1 -- if only rank 0 prints, that is
#      the hang, not slow startup
#   3. s/step after step ~100 (before that it is inductor warmup and cache writes)
#   4. peak host RSS. If this OOMs, drop NW to 12 and RECOMPUTE ipe -- do not just lower it.
# ⚠ Steps 0..5000 are the LR warmup, so the loss here says nothing. This measures throughput
# and survival ONLY. Do not read science off it.
#
# ═══ 🔴 IDENTICAL LOADER CONFIG TO THE REAL RUN ══════════════════════════════════════════
# CPS/NW/ACC/CB are copied from v3_r6_vits384_tube_2g_55k.sbatch verbatim and asserted below.
# A smoke test at a different worker count measures a different loader and proves nothing.
# It shares the real run's TORCHINDUCTOR_CACHE_DIR on purpose, so it warms the cache the
# 55k job then starts on (~20 min saved there).
# 🚫 Writes to a THROWAWAY ckpt dir. Nothing here is a rung and nothing here boards.
set -uo pipefail

ARM="${1:?usage: sbatch v3_r6_vits384_tube_2g_smoke.sbatch <vits384|vits384_noparcel|vits384_nosrope|vits384_noparcel_nosrope>}"
case "$ARM" in
  vits384)                  ABL="--enc-width 384" ;;
  vits384_noparcel)         ABL="--enc-width 384 --no-parcel-embed" ;;
  vits384_nosrope)          ABL="--enc-width 384 --no-space-rope" ;;
  vits384_noparcel_nosrope) ABL="--enc-width 384 --no-parcel-embed --no-space-rope" ;;
  *) echo "[FATAL] unknown arm '$ARM'"; exit 1 ;;
esac
case "$ARM"   in *nosrope*|*nospacerope*) NAME_SAYS_OFF=1 ;; *) NAME_SAYS_OFF=0 ;; esac
case " $ABL " in *" --no-space-rope "*)   FLAG_SAYS_OFF=1 ;; *) FLAG_SAYS_OFF=0 ;; esac
if [ "$NAME_SAYS_OFF" != "$FLAG_SAYS_OFF" ]; then
  echo "[FATAL] space_rope mismatch -- name says off=$NAME_SAYS_OFF, flags say off=$FLAG_SAYS_OFF"; exit 1
fi

CB=8250
ACC=4
WORLD=2
if [ "$((CB * ACC * WORLD))" -ne 66000 ]; then
  echo "[FATAL] cb=$CB x accum=$ACC x world=$WORLD = $((CB*ACC*WORLD)) != 66000 contacts/opt-step"; exit 1
fi
CPS=616
NSESS=13
NW=16
STREAMS=$(( WORLD * NW ))
IPE=$(( ((CPS * NSESS / STREAMS) * NW) / ACC ))
if [ "$IPE" -ne 1000 ]; then
  echo "[FATAL] ipe is $IPE, not 1000 -- MUST match the 55k job (cps=$CPS world=$WORLD workers=$NW accum=$ACC)"; exit 1
fi

module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_tube2g_${ARM}
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"

RUN_NAME=v3_r6_tube_pbspace111_sf50_${ARM}_2g_smoke
CKPT_DIR=/work/nvme/bhqk/htang13/v3_ckpt_SMOKE_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain

rm -rf "$CKPT_DIR"

echo "[SMOKE 2-GPU] arm=$ARM flags='$ABL' run=$RUN_NAME"
echo "[purpose]  measure s/step at world=$WORLD and prove the loader survives 2 ranks sharing 113560M. NOT science."
echo "[batch]    cb=$CB x accum=$ACC x world=$WORLD = $((CB*ACC*WORLD)) contacts/opt-step -- IDENTICAL to the 1-GPU contract"
echo "[epoch]    cps=$CPS workers=$NW/rank streams=$STREAMS -> ipe=$IPE, matches the 1-GPU pair exactly"
echo "[target]   the 55k job needs >=1.5x over 4.005 s/step (i.e. <=2.67 s/step) to fit a 47 h wall"
echo "[watch]    [hb] lines from rank=0 AND rank=1. Only rank 0 printing = the silent futex hang."
echo "[read]     🚫 throwaway ckpts under /work/nvme. Nothing here boards."

srun "$VENV" -u -m speech_decoding.experiments.dispatch_v3 \
  --bt-root /projects/bhqk/htang13/braintreebank \
  --band-cache-dir $ROOT/band_v3slow \
  --band-cache-dir $ROOT/band_v3mid \
  --band-cache-dir $ROOT/band_hga \
  --span-dir /work/nvme/bhqk/htang13/v14_bad_windows_v3 \
  --session 1:0 --session 2:1 --session 2:2 --session 2:3 --session 2:5 \
  --session 2:6 --session 3:2 --session 4:2 --session 6:0 --session 6:1 \
  --session 6:4 --session 8:0 --session 9:0 \
  --frontend v3r6 --objective mae \
  --batch-unit shaft --contact-budget $CB --shaft-alpha 0.0 \
  --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches $ACC \
  --accelerator gpu --devices $WORLD --precision bf16-mixed \
  --clips-per-session $CPS \
  --num-workers $NW --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac 0.50 --per-band-space --space-block-w-bands 1,1,1 \
  --band-time-unit shaft \
  $ABL \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 100000 \
  --ssl-max-steps 300
