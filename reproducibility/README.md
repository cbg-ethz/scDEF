# Reproducibility

This directory contains Snakemake workflows to generate all the benchmarking results in the paper and Jupyter notebooks to reproduce the figures from those results.

## Setup

Create the orchestration conda environment from the file in this directory:

```bash
conda env create -f scdef_benchmark.yml
conda activate scdef_benchmark
```

This environment contains Snakemake and the Slurm executor plugin. Method-specific environments are managed automatically via `--use-conda`.

## Configuration

The `config/methods.yaml` file contains hyperparameters for all methods. The default values are set for **local testing** (low iterations). For production runs, increase `n_epoch`, `lr`, `max_iter`, `max_epochs`, etc. to sensible values (see below).

## Running workflows

### Local mode (quick testing)

Use `--use-conda` with a small number of cores. The default `config/methods.yaml` has low iteration counts:

```bash
cd reproducibility

# Dry-run first to verify the DAG
snakemake --snakefile workflow/simulations/benchmark_singlebatch.smk \
    --use-conda -n

# Run locally with 4 cores
snakemake --snakefile workflow/simulations/benchmark_singlebatch.smk \
    --use-conda --conda-frontend mamba -j 4
```

### Local production mode (single GPU machine)

Use `--profile profile_local/` for a machine with multiple cores and 1 GPU. The profile uses 32 cores, 188 GB RAM, and serializes GPU jobs via `gpu=1`. CPU-only methods (PCA, NMF, scHPF, Harmony, Scanorama, nSBM) run in parallel while GPU methods (scDEF, scDEF\_un, scDEF\_corr, scDEF\_hclust, scVI, fscLVM) queue for the single GPU:

```bash
cd reproducibility

# Edit config/methods.yaml to set production hyperparameters first (see below)

snakemake --snakefile workflow/simulations/benchmark_singlebatch.smk \
    --use-conda --conda-frontend mamba --profile profile_local/
```

Edit `profile_local/config.yaml` to match your machine if different from 32 cores / 188 GB / 1 GPU.

### Cluster production mode (Slurm)

Use `--profile profile/` to submit jobs to Slurm. Each GPU job requests `--gpus=1` from Slurm independently, so if the cluster has multiple GPU nodes available, GPU methods for different reps/conditions will run on separate nodes in parallel (no global GPU cap):

```bash
cd reproducibility

# Edit config/methods.yaml to set production hyperparameters first (see below)

snakemake --snakefile workflow/simulations/benchmark_singlebatch.smk \
    --use-conda --conda-frontend mamba --profile profile/
```

Edit `profile/config.yaml` to change the Slurm account, default runtime, or memory per CPU.

### Production hyperparameters

Before any production run (local or cluster), edit `config/methods.yaml`:

```yaml
scDEF:       { n_epoch: [200], lr: [0.01] }
scDEF_un:    { n_epoch: [200], lr: [0.01] }
scDEF_hclust: { n_epoch: [200], lr: [0.01] }
NMF:         { max_iter: 1000 }
scHPF:       { min_iter: 30, max_iter: 500 }
scVI:        { max_epochs: 400 }
fscLVM:      { n_epochs: 1000 }
nSBM:        { n_init: 100 }
```

### Custom output directory

Override the output directory for any workflow:

```bash
snakemake --snakefile workflow/simulations/benchmark_multibatch.smk \
    --use-conda --conda-frontend mamba --profile profile_local/ \
    --config output_path=/data/benchmark_multibatch
```

## Resource allocation

All rules have `threads` and `resources` annotations in `workflow/run_methods.smk`:

| Method | Threads | RAM | GPU |
|---|---|---|---|
| scDEF, scDEF\_un, scDEF\_corr, scDEF\_hclust | 4 | 32 GB | Yes |
| scVI, fscLVM (MuVI) | 4 | 32 GB | Yes |
| nSBM | 2 | 16 GB | No |
| scHPF | 1 | 16 GB | No |
| PCA, NMF, Harmony, Scanorama | 1 | 8 GB | No |
| All evaluate\_\* rules | 1 | 8 GB | No |

On a 32-core / 1-GPU machine, Snakemake will run up to ~28 CPU-only jobs in parallel while GPU jobs queue one at a time.

## Simulation studies

### Single-batch benchmarking

```bash
snakemake --snakefile workflow/simulations/benchmark_singlebatch.smk \
    --use-conda --conda-frontend mamba --profile profile_local/
```

Methods: scDEF\_un, PCA, NMF, scHPF, nSBM, fscLVM. Results in `results/benchmark_singlebatch/`.

### Multi-batch benchmarking

```bash
snakemake --snakefile workflow/simulations/benchmark_multibatch.smk \
    --use-conda --conda-frontend mamba --profile profile_local/
```

Methods: all 12 (scDEF, scDEF\_un, scDEF\_corr, scDEF\_hclust, PCA, NMF, scHPF, scVI, Harmony, Scanorama, nSBM, fscLVM). Results in `results/benchmark_multibatch/`.

### Hyperparameter ablation

Sweeps `hierarchy_weight`, `brd_strength`, and `brd_mean`, each crossed with `de_prob` (DE density):

```bash
snakemake --snakefile workflow/simulations/hyperparams.smk \
    --use-conda --conda-frontend mamba --profile profile_local/
```

### Structure ablation

Sweeps `n_layers` and `n_factors`:

```bash
snakemake --snakefile workflow/simulations/structure.smk \
    --use-conda --conda-frontend mamba --profile profile_local/
```

### Wall-time scalability

Measures runtime across dataset sizes (100, 1k, 10k, 100k cells):

```bash
snakemake --snakefile workflow/simulations/walltime.smk \
    --use-conda --conda-frontend mamba --profile profile_local/
```

## Real data studies

Available datasets: `pbmcs3k`, `pbmcs2b`, `pbmcsifn`, `planaria`, `chemo`, `cnvs`, `visium`.

```bash
# Example: 3k PBMCs
snakemake --snakefile workflow/real_data/pbmcs3k.smk \
    --use-conda --conda-frontend mamba --profile profile_local/
```

Results will appear in `results/pbmcs3k/`. To generate the figures, run the corresponding notebook in `make_figures/`.

## Architecture

- **Shared method rules**: All `run_*` / `evaluate_*` rules live in `workflow/run_methods.smk`. Each parent workflow sets `RUN_SUFFIX`, `INPUT_ADATA`, and `METHOD_SEED` before including it. This avoids duplicating method rules across workflows.
- **Run/evaluate split**: Each method has separate `run_*` and `evaluate_*` rules. The `run` step fits the model and saves output (scDEF models as directories via `model.save()`, other methods as `.h5ad`). The `evaluate` step loads the output and computes metrics. This allows re-evaluation without re-running expensive model fits.
- **Conda environments**: Method-specific environments are in `workflow/envs/`. The `scdef_reproducibility.yml` environment (scdef==0.6.0) is used for all evaluation rules and scDEF model runs. Other methods each have their own isolated environment without scdef.
- **Shared scripts**: All Python runner and evaluation scripts in `workflow/scripts/` are shared between simulation and real-data workflows.
- **Resource management**: Rules are annotated with `threads`, `mem_mb`, and `gpu` resources. Use `--profile profile_local/` for single-machine runs or `--profile profile/` for Slurm clusters.

## Notes

- The workflows rely on `conda` environments to track the package versions.
- The workflows are designed for local testing, single-machine production, and HPC (Slurm) cluster use.
- GPU methods: scDEF, scDEF\_un, scDEF\_corr, scDEF\_hclust, scVI, fscLVM (MuVI).
- Output paths can be customized via config: `--config output_path=/path/to/results`.
