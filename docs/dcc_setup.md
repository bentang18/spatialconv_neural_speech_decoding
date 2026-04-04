# Duke DCC Cluster Setup

Complete guide to running experiments on the Duke Compute Cluster. **Use DCC for all training runs.** Local MPS is for editing code only.

## Access

```bash
ssh ht203@dcc-login.oit.duke.edu
# No MFA required (as of 2026-04-04)
```

## Directory Layout

```
/work/ht203/
├── repo/speech/              # Git repo (branch: autoresearch/run1)
├── data/BIDS/                # Symlinked BIDS datasets
│   ├── BIDS_1.0_Phoneme_Sequence_uECoG/  # PS dataset
│   └── BIDS_1.0_Lexical_µECoG/           # Lexical dataset
├── miniconda3/envs/speech/   # Python 3.11 conda env
└── logs/                     # SLURM output logs

/hpc/group/coganlab/ht203/    # Permanent storage (NOT auto-purged)
```

**IMPORTANT**: `/work/ht203/` auto-purges after **75 days** of no access. Copy important results to `/hpc/group/coganlab/ht203/`.

## Python Environment

The base conda on DCC has a broken `frozendict` with Python 3.13. **Do NOT use `conda activate speech`**. Instead, use the python binary directly:

```bash
PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
```

### Environment Details
- Python 3.11
- PyTorch 2.5.1+cu121
- MNE, scikit-learn, speech_decoding (editable install)

### Known Issue: `packaging` Module
If you get `ModuleNotFoundError: No module named 'packaging'`, fix with:
```bash
cd /work/ht203/miniconda3/envs/speech/lib/python3.11/site-packages
rm -rf packaging-25.0.dist-info  # ghost dist-info with empty packaging/ dir
$PYTHON -m pip install packaging==24.2
```

## GPU Hardware

- **8× NVIDIA RTX 5000 Ada Generation** (32 GB VRAM each)
- CUDA driver 13.1
- Partition: `coganlab-gpu`

### Alternative Partitions

| Partition | Walltime | GPUs | Notes |
|-----------|----------|------|-------|
| `coganlab-gpu` | 90 days | RTX 5000 Ada | Dedicated, preferred |
| `gpu-common` | 2 days | Varies | Shared, no preemption |
| `scavenger-gpu` | 7 days | A6000/H200 | Preemptible, bigger GPUs |

## Submitting Jobs

### SBATCH Template

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ht203/logs/my_job_%j.out
#SBATCH --error=/work/ht203/logs/my_job_%j.err

set -e
cd /work/ht203/repo/speech

PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
export DEVICE=cuda

$PYTHON scripts/my_script.py --paths configs/paths.yaml --device cuda
```

### Submit and Monitor

```bash
sbatch scripts/my_job.sh          # Submit
squeue -u ht203                    # Check status
scancel <job_id>                   # Cancel
tail -f /work/ht203/logs/my_job_<id>.out  # Live output
```

## Data

### What's on DCC

**Pre-cached `.pt` tensors** (from autoresearch `prepare.py`):
- `/work/ht203/repo/speech/.cache/autoresearch_lopo/` — S14 target + 9 source patients
- These are grid-reshaped, tmin=0.0, tmax=1.0 tensors ready for training
- The autoresearch pipeline (`scripts/autoresearch_lopo/`) uses these caches and works on DCC

**Raw `.fif` files** (via symlink to coganlab):
- `/work/ht203/data/BIDS/` contains the BIDS directory structure
- Productionzscore HGA .fif files were transferred 2026-03-31
- Verify: `ls /work/ht203/data/BIDS/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS/derivatives/epoch\(phonemeLevel\)\(CAR\)/sub-S14/epoch\(band\)\(power\)/`

### What's NOT on DCC (known gaps)

- ~~Raw .fif files may be missing for some patients~~ — verify per above
- If `load_patient_data()` fails with `FileNotFoundError`, the .fif for that patient wasn't transferred
- Electrode TSV files (for grid inference) — check if present in BIDS sourcedata

### Transferring Data

From local Mac to DCC:
```bash
# Transfer a specific .fif file
scp /Users/bentang/Documents/Code/speech/BIDS_1.0_Phoneme_Sequence_uECoG/.../sub-S14_task-PhonemeSequence_desc-productionZscore_highgamma.fif \
  ht203@dcc-login.oit.duke.edu:/work/ht203/data/BIDS/BIDS_1.0_Phoneme_Sequence_uECoG/.../

# Transfer all .fif files (recursive)
rsync -avz --include='*/' --include='*highgamma.fif' --exclude='*' \
  /Users/bentang/Documents/Code/speech/BIDS_1.0_Phoneme_Sequence_uECoG/ \
  ht203@dcc-login.oit.duke.edu:/work/ht203/data/BIDS/BIDS_1.0_Phoneme_Sequence_uECoG/
```

### DCC `paths.yaml`

The repo's `configs/paths.yaml` is gitignored. On DCC, create/update it:
```yaml
bids_root: /work/ht203/data/BIDS
```

## Syncing Code

```bash
# On DCC — pull latest from GitHub
cd /work/ht203/repo/speech
git pull origin autoresearch/run1

# Or push from local, then pull on DCC
# Local:
git push origin autoresearch/run1
# DCC:
git pull
```

## Running the Sweep Script (Example)

```bash
#!/bin/bash
#SBATCH --job-name=sweep_tmin
#SBATCH --partition=coganlab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/ht203/logs/sweep_tmin_%j.out
#SBATCH --error=/work/ht203/logs/sweep_tmin_%j.err

set -e
cd /work/ht203/repo/speech

PYTHON=/work/ht203/miniconda3/envs/speech/bin/python
export DEVICE=cuda
export PYTHONUNBUFFERED=1

$PYTHON scripts/sweep_tmin_perpos.py \
    --paths configs/paths.yaml \
    --device cuda
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `frozendict` import error | Broken base conda Python 3.13 | Use `$PYTHON` binary directly, not `conda activate` |
| `ModuleNotFoundError: packaging` | Ghost dist-info directory | `rm -rf packaging-25.0.dist-info && pip install packaging==24.2` |
| `FileNotFoundError: HGA file not found` | .fif not transferred to DCC | Transfer via `rsync` (see above) |
| Python output not appearing in log | stdout buffering with `tee` | Add `export PYTHONUNBUFFERED=1` to SBATCH script |
| `CUDA out of memory` | Batch too large for 32GB | Reduce batch size or use gradient accumulation |
| Job stuck in `PD` state | GPU queue full | Try `scavenger-gpu` partition or wait |
