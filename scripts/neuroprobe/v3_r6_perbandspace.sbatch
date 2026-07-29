#!/bin/bash
#SBATCH --job-name=r6_pbspace
#SBATCH --account=bhqk-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/projects/bhqk/htang13/logs/%x_%j.out
#
#   sbatch scripts/neuroprobe/v3_r6_perbandspace.sbatch <SLOW,MID,HGA> [space_frac]
#
# ABLATION R19+R20 (workshop paper) -- PER-BAND SPATIAL MASKING AT DERIVED WIDTHS, FROM SCRATCH.
# ONE conceptual delta off the 6e-3 keeper (2712191): the spatial mask stops being a TUBE.
#
# WHY (Ben 2026-07-29, "I like the symmetry for independent band mask for both space and time").
# r6 already draws the three TIME masks INDEPENDENTLY per band, each on its own 4/16/32 Hz grid --
# that is why a contact can be HGA-masked while SLOW/MID stay visible at the same instant, which is
# the only place the design forces cross-band mixing. SPACE was the lone shared axis: one (R,N)
# draw deleting every band at that contact. That asymmetry was never argued for. This arm makes
# BOTH axes follow one rule, so the contract states in a sentence: every mask is per band.
#
# MATCHED-VISIBLE BY CONSTRUCTION, which is what makes it readable. Exact-count snapping
# (d_s = round(space_frac*n_s) per shaft, masking.py) runs PER BAND, so each band masks the same
# number of contacts it did before. The masked TOKEN COUNT is bit-identical to the tube -- only
# WHICH contacts, and in which band, moves. Verified by test (test_masking_r6.py).
#
# THE SECOND KNOB, merged in at Ben's direction rather than run as its own arm: per-band block
# WIDTH. Unlike block_w_band=4 (TIME), which is derived from the STFT hop-overlap leak margin,
# block_w_space=4 was hard-coded and derived from NOTHING -- contacts share no samples, so there is
# no mechanical leak on the space axis. R20 measured along-shaft r(d) per band on our OWN data (NOT
# Duraivel 2023 Fig 2d: that curve is uECoG measured TANGENTIALLY across a surface grid and is
# HGA-only, while our shafts sample RADIALLY through gray->white->gray). RESULT, 13 sessions:
#
#   band | d=1  | d=2  | d=3  | d=4  | d=5  | d=6
#   SLOW | .557 | .364 | .331 | .343 | .351 | .349
#   MID  | .401 | .219 | .195 | .205 | .211 | .213
#   HGA  | .275 | .173 | .165 | .174 | .180 | .188
#
# r(d) does NOT decay -- it CLIFFS at d=1 then goes FLAT. So the expected SLOW-wide/HGA-narrow
# asymmetry DOES NOT EXIST (all three bands drop by the same proportion), and the widths are equal.
# The only thing a width can buy is moving masked contacts OFF the d=1 shortcut onto the flat floor.
#
# WIDTH + RATE, measured on the REAL sampler (realistic BT shaft mix, 64 rows). "HARD" = masked
# contacts whose nearest VISIBLE same-shaft contact is >=2 slots away, i.e. down on the floor:
#
#   space_frac 0.50: bw1 .224 | bw2 .342 | bw4 .517 | bw6 .547   <- knee at 4, 6 is diminishing
#   space_frac 0.25: bw1 .046 | bw2 .140 | bw4 .281 | bw6 .227   <- 4 is the MAXIMUM
#
# Ben 07-29 asked whether to mask at width 1 and let blocks emerge by chance. That works at 0.50
# (.224 hard) but COLLAPSES at 0.25 (.046 -- 95% of masked contacts keep a visible neighbour, and
# R20 says that neighbour carries r = .56/.40/.27). Rate and width trade off and nearly cancel:
# 0.50+bw1 (.224) ~= 0.25+bw4 (.281). You can have emergent blocks OR the lower rate, not both.
# This arm takes the lower rate and keeps width 4. Width changes arrangement only, never count.
#
# THE RATE (Ben 07-29, "we should probably spatial mask at a lower percent, 25 is better than 50").
# Correct, and it is what makes this arm READABLE: the closed visible-fraction sweep put space_frac
# 0.25 (37.5% visible, 0.625 total) at CS .6759 / WS .7621 vs the 0.50 keeper's .6724 / .7504 --
# CS +.0035 is a wash but WS +.0117 is real, and it is the best point measured on that axis. It
# also means this arm has a MATCHED CONTROL WE ALREADY OWN: maskspace25 (job 2742871, space 0.25,
# TUBE, width 4) at 30k. So per-band space stays a strictly ONE-KNOB delta -- against a control at
# the best rate rather than against the keeper at a weaker one. Ladders every 1000 and LR is
# constant (--min-lr-ratio default 1.0), so the 30k read is matched-step and legitimate.
#
# SCOPE, stated honestly: R18 put the whole spatial-arrangement lever at dr = +0.019 on
# reconstruction, and the mask-RATE sweep put its best step at CS +.0035 (a wash). So this is not
# expected to move the board much. It buys a DERIVED number where the paper currently has a
# hard-coded one, and a single-knob test of whether the cross-band bridge is load-bearing.
#
# FROM SCRATCH, not a branch: mask structure shapes what the model learns to predict from step 0.
# 40k @ ~3.467 s/step ~= 38.5 h; ladder every 1000 so a walltime cut costs at most the last 1000.
# NO cooldown and NO encode in this job (Ben 2026-07-28): LR is constant (--min-lr-ratio default
# 1.0) so EVERY ladder ckpt is a legitimate cooldown branch point.
# --- template provenance: 2756178 (v3_r6_randmask_lr6e-3_40k) stored batch script, verbatim ---
set -uo pipefail
SBW="${1:?usage: sbatch v3_r6_perbandspace.sbatch <SLOW,MID,HGA> [space_frac]  e.g. 4,4,4 0.25}"
SF="${2:-0.25}"
module load python/miniforge3_pytorch/2.10.0
export ROOT_DIR_BRAINTREEBANK=/projects/bhqk/htang13/braintreebank
export EXCA_CACHE_FOLDER=/work/nvme/bhqk/htang13/cache_neuroai
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/work/nvme/bhqk/htang13/torchinductor_cache_r6_pbspace
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export WANDB_API_KEY="$(cat /projects/bhqk/htang13/.wandb_key)"
TREE=/projects/bhqk/htang13/speech
VENV=$TREE/.venv/bin/python
export PYTHONPATH="$TREE/src:${PYTHONPATH:-}"
cd "$TREE"
RUN_NAME=v3_r6_pbspace${SBW//,/}_sf${SF#0.}_lr6e-3_40k
CKPT_DIR=/projects/bhqk/htang13/v3_ckpt_${RUN_NAME}
ROOT=/work/nvme/bhqk/htang13/cache_neuroai/v14_3band_v3_spec_pretrain
echo "[r6 R19+R20 ABLATION] run=$RUN_NAME PER-BAND SPACE widths=$SBW (SLOW,MID,HGA) space_frac=$SF (control: maskspace25 job 2742871 @30k) lr=6e-3 cb=11000 accum=6 tok/step=6864000 steps=40000 deep-sup ON [FILE-ONLY scp deploy]"
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
  --batch-unit shaft --contact-budget 11000 --shaft-alpha 0.5 \
  --lr 6e-3 --weight-decay 0.04 --grad-clip 3.0 --adam-beta2 0.95 --seed 33 \
  --clip-len 2.0 \
  --session-z-winsor-slow 15 --session-z-winsor-mid 15 --session-z-winsor-hga 20 \
  --batch-size 32 --accumulate-grad-batches 6 \
  --accelerator gpu --devices 1 --precision bf16-mixed \
  --num-workers 12 --monitor-every-n-steps 10 --log-every-n-steps 10 \
  --mask-space-frac "$SF" --per-band-space --space-block-w-bands "$SBW" \
  --compile \
  --wandb-project intuition --run-name $RUN_NAME \
  --ckpt-dir $CKPT_DIR --ckpt-ladder-every 1000 \
  --ssl-max-steps 40000
