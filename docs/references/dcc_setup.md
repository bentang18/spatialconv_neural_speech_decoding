# Duke DCC Cluster Setup

Use DCC for all training runs. Local machine is for editing, tests, and small import checks only.

## Quick-start cheatsheet

This is the canonical reference. `CLAUDE.md §Compute` is intentionally a 4-line load-on-demand stub that says "Read this file when dispatching" — to avoid bloating every-session context with DCC details that don't apply when the session has nothing to do with the cluster. If you've been pointed here, you're probably about to dispatch, sync, or debug a DCC job.

### Identity + paths

| What | Where |
|---|---|
| SSH | `ssh ht203@dcc-login.oit.duke.edu` |
| Repo | `/work/ht203/repo/speech` |
| Python | `.venv/bin/python` (uv-managed Python 3.12) |
| BT raw data | `/work/ht203/data/braintreebank` |
| PS BIDS | `/work/ht203/data/BIDS` |
| Aux data | `/work/ht203/data/{mni_coords,channel_maps,transforms,atlas}/` |
| Slurm logs | `/work/ht203/logs/` (also per-report `slurm-*.out/.err`) |
| GPU | 8× RTX 5000 Ada (32 GB), partition `coganlab-gpu`, account `coganlab` |
| Upstream Neuroprobe clone | `/work/ht203/repo/neuroprobe_upstream/` (pinned `c7b955b0`) |
| Persistent cache (Exca, ckpts, results) | `/hpc/group/coganlab/ht203/cache_neuroai/` |

### Required env vars

```bash
export ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank
export EXCA_CACHE_FOLDER=/hpc/group/coganlab/ht203/cache_neuroai
```

### Sync + dispatch helpers (local-machine commands)

| Command | What it does |
|---|---|
| `scripts/dcc/sync` | `git push origin <branch>` then SSH DCC and `git fetch` + `git reset --hard origin/<branch>` + print last 3 commits. |
| `scripts/dcc/dispatch <submitter-relpath> [args...]` | Calls `sync` then runs `.venv/bin/python <submitter> [args...]` on DCC. Submitter path is repo-relative. |
| `scripts/dcc/status [report-glob]` | `squeue -u ht203` + `status_l_sweeps.py` per report dir. Default glob: `reports/neuroprobe_stage0_*/`. |
| `scripts/dcc/rerun-failed <report-dir> [--mem 64G] [--dry-run]` | Finds OOM / traceback / non-zero-exit jobs in a sweep dir, re-sbatches matching `slurm-<jobid>.sbatch` with optional bumped `--mem`. |
| `scripts/sync_dcc_repo.sh` | Heavyweight rsync; use only for non-git artifacts. |

Examples:

```bash
# Push + dispatch a Tier-C re-run with bumped memory:
scripts/dcc/dispatch scripts/neuroprobe/submit_tier_c_cross_subject.py \
    --l1-winner-norm train_set_fixed --l2-winner-ref bipolar \
    --l2-winner-view stft_abs --skip-baseline --mem 64G

# One-line status across all active Stage-0 sweeps:
scripts/dcc/status

# Re-run failed jobs in a specific sweep dir at 64G:
scripts/dcc/rerun-failed reports/neuroprobe_stage0_l2_neuralset_2026_05_07 --mem 64G

# Override host/repo if needed:
DCC_HOST=ht203@dcc-login.oit.duke.edu DCC_REPO=/work/ht203/repo/speech scripts/dcc/sync
```

### Standard sbatch header

All `submit_*` scripts emit headers in this shape:

```bash
#SBATCH -p coganlab-gpu          # or "common,scavenger,coganlab-gpu" to drain queue
#SBATCH --account=coganlab
#SBATCH --gres=gpu:1             # add for training; linear baselines are CPU-only
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G                # bump to 48G/64G/160G for OOM-prone cells
#SBATCH -t 04:00:00
```

### Critical rules

- `/work/ht203/` auto-purges every 75 days. Persistent caches/ckpts/results → `/hpc/group/coganlab/ht203/`.
- Never `conda activate`. Use `.venv/bin/python -m ...` directly.
- `EXCA_CACHE_FOLDER` MUST point at `/hpc/group/coganlab/ht203/cache_neuroai/`.

## Access

```bash
ssh ht203@dcc-login.oit.duke.edu
```

## Current Layout

```text
/work/ht203/
├── repo/speech/                  # repo checkout, uv-managed .venv
├── data/braintreebank/           # BrainTreebank Lite h5 files
├── data/BIDS/                    # PS data, paused
└── logs/                         # Slurm logs

/hpc/group/coganlab/ht203/        # persistent storage, not auto-purged
```

`/work/ht203/` auto-purges after 75 days. Long-lived caches, checkpoints, and result artifacts must go under `/hpc/group/coganlab/ht203/`.

## Python

```bash
cd /work/ht203/repo/speech
.venv/bin/python -m pytest -q
```

The DCC env is uv-managed Python 3.12. Do not use the old conda env for new work.

To rebuild:

```bash
cd /work/ht203/repo/speech
uv sync --extra dev
```

## Safe Repo Sync

Use the sync helper from the local repo root:

```bash
scripts/sync_dcc_repo.sh
```

It syncs to:

```text
ht203@dcc-login.oit.duke.edu:/work/ht203/repo/speech/
```

and excludes local data, BIDS folders, results, `pastwork/`, `.venv/`, `.git/`, caches, and notes. Do not run a broad unfiltered `rsync ./ ...:/work/ht203/repo/speech/`; this checkout may contain large local untracked data folders.

## BrainTreebank

Lite data root:

```bash
export ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank
```

Raw-voltage proof:

```bash
ROOT_DIR_BRAINTREEBANK=/work/ht203/data/braintreebank \
  .venv/bin/python scripts/neuroprobe/prove_wang2024treebank_raw_voltage.py \
  --subject-trial 1:1 --subject-trial 2:4 --subject-trial 10:1 \
  --out-dir reports/neuroai_raw_voltage_proof_YYYY_MM_DD
```

The 2026-04-29 proof passed and is stored locally at `reports/neuroai_raw_voltage_proof_2026_04_29/`.

## Slurm

Default GPU partition:

```bash
#SBATCH --partition=coganlab-gpu
#SBATCH --account=coganlab
#SBATCH --gres=gpu:1
```

Default CPU partition (probes, eval, postproc):

```bash
#SBATCH --partition=common         # NOT scavenger — preemptable + older nodes
#SBATCH --account=coganlab
```

Partition picker (5/28):

| Resource | Partition | Hardware | Why |
|---|---|---|---|
| GPU | `coganlab-gpu` (fallback `scavenger-gpu`) | RTX 5000 Ada × 8 @ 32 GB | Lab-owned, no preempt |
| CPU | `common` | 72-128 CPU, 385GB-1TB RAM, newer | NOT preemptable; lab-priority |
| Big-RAM CPU | `common` u-ab25-3-[1-8] | 128 CPU, 1 TB | Available without separate request |
| Short-burst CPU (<10 min) | `scavenger` | 48-96 CPU, 257GB+, mixed lab hw | OK if common queue jammed |

There is NO `coganlab` CPU partition. Only `coganlab-gpu`. Don't put `coganlab` in a CPU `-p` list — sbatch rejects.

`sinfo -p <part> -N -o '%N %c %m %f %G'` shows per-node CPU/RAM. `squeue -u ht203 -o "%i %P %j %R"` shows which node landed.

Monitor jobs directly when needed:

```bash
squeue -u ht203
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,ReqMem,ExitCode
tail -f /work/ht203/repo/speech/logs/<name>_<job_id>.out
```

## NeuroAI Execution

Use NeuralTrain experiments + Exca for all training and grid sweeps.

### Canonical cache folder

Every Exca-backed `infra` must point its `folder` at:

```text
/hpc/group/coganlab/ht203/cache_neuroai/<experiment_name>/
```

This is persistent. Never use `/work/ht203/` for caches or checkpoints — it auto-purges every 75 days, which silently invalidates every cached precompute on the cycle.

The local-side equivalent lives at `configs/paths.yaml:exca_cache_root` (gitignored, machine-specific). Experiment construction should read that key and set `infra.folder` from it; do not hardcode the path in scripts.

### Canonical sweep dispatch

`TaskInfra` decorates `Experiment.run`, then `neuraltrain.utils.run_grid` wraps the grid into one Slurm job array via `infra.job_array()`:

```python
from neuraltrain.utils import run_grid
run_grid(
    exp_cls=Experiment,
    exp_name="L1_normalization_sweep",
    base_config={"infra": {"folder": str(cache_root / "L1"), "cluster": "slurm"}, ...},
    grid={"data.normalization": ["window_local", "train_set", "session_fixed", "scale_only", "none"]},
)
```

`MapInfra` is for the *extractor* layer (per-subject precompute), not for hyperparameter grids. Don't conflate them.

### Old paths

Do not revive old `scripts/ablation/` or `scripts/v14_core/` workflows; they were backed up externally and removed from the active tree.
