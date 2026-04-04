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
- PyTorch 2.10.0+cu126
- MNE, scikit-learn, speech_decoding (editable install)
- Note: `torch.cuda.is_available()` returns False on login node (no GPU). Works on compute nodes via SBATCH.

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
export PYTHONUNBUFFERED=1

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

**PS (Phoneme Sequence) dataset — all 11 patients transferred (2026-04-04)**:
- `.fif` files: `/work/ht203/data/BIDS/derivatives/epoch(phonemeLevel)(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`
- Electrode TSVs: `/work/ht203/data/BIDS/sub-{id}/ieeg/sub-{id}_acq-01_space-ACPC_electrodes.tsv`
- Patients: S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62
- `load_patient_data()` and `load_per_position_data()` work directly with `bids_root=/work/ht203/data/BIDS`

**Pre-cached `.pt` tensors** (from autoresearch `prepare.py`):
- `/work/ht203/repo/speech/.cache/autoresearch_lopo/` — S14 target + 9 source patients
- Grid-reshaped, tmin=0.0, tmax=1.0 tensors. Used by the autoresearch pipeline only.
- Note: this is how LOPO ran on DCC before the .fif files were transferred.

### What's NOT on DCC

- **Lexical dataset** (BIDS_1.0_Lexical_µECoG) — needed for cross-task pooling (future work)
- **S36** — excluded (duplicate of S32)
- **S18** — excluded (no preprocessed data)
- **Permanent storage** — `/hpc/group/coganlab/ht203/` not yet created (needed before /work purge)

### Transferring Data

**Important**: Do NOT transfer in parallel (multiple concurrent `scp` calls). SSH connection limits cause partial/corrupted transfers. Transfer sequentially:

```bash
LOCAL_BIDS="/Users/bentang/Documents/Code/speech/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS_1.0_Phoneme_Sequence_uECoG/BIDS"
REMOTE="ht203@dcc-login.oit.duke.edu:/work/ht203/data/BIDS"

for p in S14 S16 S22 S23 S26 S32 S33 S39 S57 S58 S62; do
  echo "Transferring $p..."
  # Create dirs
  ssh ht203@dcc-login.oit.duke.edu "mkdir -p /work/ht203/data/BIDS/derivatives/epoch\(phonemeLevel\)\(CAR\)/sub-$p/epoch\(band\)\(power\) && mkdir -p /work/ht203/data/BIDS/sub-$p/ieeg"
  # .fif file
  scp "${LOCAL_BIDS}/derivatives/epoch(phonemeLevel)(CAR)/sub-${p}/epoch(band)(power)/sub-${p}_task-PhonemeSequence_desc-productionZscore_highgamma.fif" \
    "${REMOTE}/derivatives/epoch(phonemeLevel)(CAR)/sub-${p}/epoch(band)(power)/"
  # Electrode TSV
  scp "${LOCAL_BIDS}/sub-${p}/ieeg/sub-${p}_acq-01_space-ACPC_electrodes.tsv" \
    "${REMOTE}/sub-${p}/ieeg/"
done
```

**Verify transfer integrity** (compare sizes):
```bash
# On DCC
for p in S14 S16 S22 S23 S26 S32 S33 S39 S57 S58 S62; do
  ls -lh /work/ht203/data/BIDS/derivatives/'epoch(phonemeLevel)(CAR)'/sub-$p/'epoch(band)(power)'/*.fif 2>/dev/null | awk '{print $NF, $5}' || echo "sub-$p: MISSING"
done
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
