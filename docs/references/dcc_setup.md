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
├── repo/speech/              # Git repo (branch: main)
│   └── .venv/                # Python 3.12 venv (uv-managed)
├── data/BIDS/                # Symlinked BIDS datasets
│   ├── BIDS_1.0_Phoneme_Sequence_uECoG/  # PS dataset
│   └── BIDS_1.0_Lexical_µECoG/           # Lexical dataset
├── miniconda3/envs/speech/   # Legacy Python 3.11 conda env (kept as fallback through 2026-05-05)
└── logs/                     # SLURM output logs

/hpc/group/coganlab/ht203/    # Permanent storage (NOT auto-purged)
```

**IMPORTANT**: `/work/ht203/` auto-purges after **75 days** of no access. Copy important results to `/hpc/group/coganlab/ht203/`.

## Python Environment

uv-managed venv inside the repo. Built 2026-04-28 to land Python 3.12 (required by `neuralset>=0.1.0`); the previous Python 3.11 conda env at `miniconda3/envs/speech/` stays untouched as a fallback through ~2026-05-05.

```bash
PYTHON=/work/ht203/repo/speech/.venv/bin/python
```

### Environment Details
- Python 3.12.13 (uv-managed via `python-build-standalone`)
- PyTorch 2.10.0+cu126 (bundled CUDA libs at `.venv/lib/python3.12/site-packages/torch/lib`)
- MNE, scikit-learn, speech_decoding (editable install via `pyproject.toml`)
- Note: `torch.cuda.is_available()` returns False on login node (no GPU). Works on compute nodes via SBATCH.

### Rebuild from scratch

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv not yet installed; lives at ~/.local/bin/uv
cd /work/ht203/repo/speech
~/.local/bin/uv venv .venv --python 3.12
~/.local/bin/uv pip install --python .venv/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
~/.local/bin/uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q tests/test_phoneme_map.py tests/test_grouped_cv.py tests/v14/test_no_legacy_imports.py
# expect: 65 passed
```

The 3-file canonical baseline avoids known pre-existing pyproject.toml gaps (nibabel/pandas not declared; tracked separately). Full `tests/` collection currently fails on those imports — fix lives in the Phase 3 NeuralSet-adoption PR.

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

PYTHON=/work/ht203/repo/speech/.venv/bin/python
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

### Ablation tooling — `scripts/ablation/` (preferred entry point)

For routine ablations (single mode, single patient or pooled set, fold × seed
cross-product, flag overrides) reach for `scripts/ablation/` instead of cloning
a hand-written sbatch. Seven small CLIs, all `--help`-documented:

```bash
# Pre-flight: verify local repo == DCC repo
scripts/ablation/dcc_sync_check.py

# Submit a new ablation cell
scripts/ablation/submit.py \
    --name q1e_d64 \
    --mode per-phoneme --patient S14 \
    --folds 0-4 --seeds 0-2 \
    --flag d_model=64 --flag backbone-depth=5 \
    --walltime 8:00:00 --mem 32G
# → job_id=12345  sbatch=scripts/v14_core/_gen/q1e_d64_dcc.sh  jobs=15  out=/work/ht203/results/v14_per_phoneme/_gen/q1e_d64

# Monitor (one line per job)
scripts/ablation/status.py 12345
scripts/ablation/status.py             # all known submissions

# Peek at logs without dragging full files into context
scripts/ablation/logs.py 12345                  # tail of task 0 .out + .err
scripts/ablation/logs.py 12345 --task 7
scripts/ablation/logs.py 12345 --grep ERROR
scripts/ablation/logs.py 12345 --failed
scripts/ablation/logs.py 12345 --stream         # tail -f via ssh

# Spot-check one result.json over ssh (no rsync)
scripts/ablation/peek.py 12345 --fold 2 --seed 1
scripts/ablation/peek.py 12345 --all

# Pull results back + run aggregator → docs/experiments/v14_ablation_log.csv
scripts/ablation/collect.py 12345

# Slice the ablation CSV (no need to read the whole file)
scripts/ablation/query.py --patient S14 --d 64
scripts/ablation/query.py --recent 10
```

Each submission is recorded to `.ablation_submissions.jsonl` (gitignored) so
`status.py`, `logs.py`, `peek.py`, `collect.py` can decode array task ids back
to `(fold, seed)` and find the remote `out_dir` without re-asking. Generated
sbatch wrappers land under `scripts/v14_core/_gen/` (gitignored). Shared
plumbing — DCC host, paths, `ssh()`, `rsync_repo()` — lives in
`scripts/ablation/_common.py`.

When **not** to use this tooling: non-standard array math (multi-cell
cross-products encoded into `task_id`, LOPO pretrain → finetune chains where
one task's flags depend on another). Hand-write those under
`scripts/v14_core/v14_*_dcc.sh` as before.

## Data

### What's on DCC

**PS (Phoneme Sequence) dataset — all 11 patients transferred (2026-04-04)**:
- `.fif` files (v14 Phase 1 trial-level input, per `#29`): `/work/ht203/data/BIDS/derivatives/epoch(CAR)/sub-{id}/epoch(band)(power)/sub-{id}_task-PhonemeSequence_desc-productionZscore_highgamma.fif`
- Electrode TSVs: `/work/ht203/data/BIDS/sub-{id}/ieeg/sub-{id}_acq-01_space-ACPC_electrodes.tsv`
- Patients: S14, S16, S22, S23, S26, S32, S33, S39, S57, S58, S62
- Pre-v14 phoneme-level files under `epoch(phonemeLevel)(CAR)/...` may still be on DCC from the earlier sync; they are audit-only per `#18` and not consumed by v14 training.

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
  ssh ht203@dcc-login.oit.duke.edu "mkdir -p /work/ht203/data/BIDS/derivatives/epoch\(CAR\)/sub-$p/epoch\(band\)\(power\) && mkdir -p /work/ht203/data/BIDS/sub-$p/ieeg"
  # .fif file (trial-level, v14 Phase 1 input per #29)
  scp "${LOCAL_BIDS}/derivatives/epoch(CAR)/sub-${p}/epoch(band)(power)/sub-${p}_task-PhonemeSequence_desc-productionZscore_highgamma.fif" \
    "${REMOTE}/derivatives/epoch(CAR)/sub-${p}/epoch(band)(power)/"
  # Electrode TSV
  scp "${LOCAL_BIDS}/sub-${p}/ieeg/sub-${p}_acq-01_space-ACPC_electrodes.tsv" \
    "${REMOTE}/sub-${p}/ieeg/"
done
```

**Verify transfer integrity** (compare sizes):
```bash
# On DCC
for p in S14 S16 S22 S23 S26 S32 S33 S39 S57 S58 S62; do
  ls -lh /work/ht203/data/BIDS/derivatives/'epoch(CAR)'/sub-$p/'epoch(band)(power)'/*.fif 2>/dev/null | awk '{print $NF, $5}' || echo "sub-$p: MISSING"
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

### When `git pull` is blocked by a dirty worktree

The DCC worktree drifts: files get edited in place, scripts get scp'd over before they're committed upstream, QC reports get written directly. A plain `git pull` then aborts with *"Your local changes to the following files would be overwritten by merge"* and *"The following untracked working tree files would be overwritten by merge"*. **Never blind-stash, blind-rm, or `reset --hard` to resolve it — some of those files may be mid-flight experiments or real local fixes.**

Recovery sequence that actually preserves work:

**1. Fetch origin (non-destructive) and see the full error.**
```bash
cd /work/ht203/repo/speech
git fetch origin
git pull --ff-only 2>&1   # will fail, but lists every blocker file
```

**2. Classify every blocker by hash-comparing disk ↔ `origin/main`.** This is the step that tells you what's safe to touch. For each blocker file:
```bash
on=$(md5sum "$f" | awk '{print $1}')
tg=$(git show origin/main:"$f" 2>/dev/null | md5sum | awk '{print $1}')
[ "$on" = "$tg" ] && echo MATCH || echo DIFFER
```
(Loop it over the full blocker list; a one-liner was run verbatim in the 2026-04-18 cleanup — see `git log` on that date for a reference invocation.)

Three classes emerge:
- **Untracked, MATCH origin/main** — DCC scp'd the file in before it was committed; `origin/main`'s tracked version is byte-identical. Safe to `rm`.
- **Modified-tracked, MATCH origin/main** — phantom modification; disk content already equals `origin/main`. `git checkout -- <file>` clears the "modified" flag without rewriting disk.
- **Modified-tracked, DIFFER** — real local divergence. Must preserve.

Do NOT skip the hash step. Two prior cleanups looked like the "safe" class but the blocker list included real local edits mixed in.

**3. Snapshot the DIFFER set before clobbering.** A named stash is the minimum; a backup branch is better because it's visible and recoverable via normal git commands:
```bash
git stash push -u -m "dcc-pre-sync-<ISO-date>-<task-name>"
```
Stash messages must be specific and dated. `git stash` with no message produces `WIP on main: <sha>` which is indistinguishable from any other idle stash months later.

**4. Pull.**
```bash
git pull --ff-only   # now succeeds
git status --short   # expect empty
```

**5. Check the in-flight Slurm jobs are unaffected.** Before pulling, verify which source files the running jobs import at runtime. If those files are in the MATCH class on disk, pull doesn't rewrite disk bytes and running tasks keep their cached modules. If any are DIFFER, either scancel first or wait until the job drains.

**6. Leave the stash/branch in place.** Never `git stash pop` an old DCC-sync stash blindly — they carry pre-push working-tree state that includes older versions of `CLAUDE.md`, `docs/objectives.md`, `docs/tactics.md`, `docs/strategy.md`, etc. Popping silently regresses doc progress. Drop with `git stash drop <ref>` only after `git stash show -p <ref>` confirms nothing worth keeping.

**7. Prefer deleting an identical untracked file over `git checkout --` on a modified file** when both would work — `rm` is a no-op on disk state (the tracked pull re-creates it byte-identical), whereas `checkout --` writes disk bytes and could race with a running task mid-import.

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

PYTHON=/work/ht203/repo/speech/.venv/bin/python
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
| `git pull` aborts: "local changes would be overwritten" | DCC worktree dirty (scp'd scripts, edited docs) | See "When `git pull` is blocked by a dirty worktree" above — hash-compare each blocker vs `origin/main`, stash DIFFERs before pulling |
