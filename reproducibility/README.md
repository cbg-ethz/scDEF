# Reproducibility
This directory contains Snakemake workflows to generate all the benchmarking results in the paper and Jupyter notebooks to reproduce the figures from those results. To run the workflows on a Slurm system, make sure you install `snakemake-executor-plugin-slurm` first.

To run the simulation studies, for example, the benchmarking, run
```snakemake --snakefile workflow/simulations/benchmark.smk --use-conda --conda-frontend mamba --profile profile/ --jobs 100```

The results will appear in `results/simulations/benchmark/`. To generate the figures for the paper, run the notebook `make_figures/simulations.ipynb`.

To run the real data studies, for example, the 3k PBMCs data, run 
```snakemake --snakefile workflow/real_data/pbmcs3k.smk --use-conda --conda-frontend mamba --profile profile/ --jobs 10```

The results will appear in `results/real_data/pbmcs3k/`. To generate the figures for the paper, run the notebook `make_figures/pbmcs3k.ipynb`.

Note:
* The workflows rely on `conda` environments to track the package versions.
* The workflows are designed to be used within a high-performance computing cluster.
* We run scDEF, scVI and MuVI on GPU nodes.
