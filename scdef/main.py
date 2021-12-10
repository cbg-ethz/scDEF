"""
Extract hierarchical signatures of cell state from single-cell data.
"""
from .scdef import scDEF

import numpy as np
import pandas as pd

import click
import os
from pathlib import Path

@click.command(help="Extract hierarchical signatures of cell state from single-cell data.")
@click.argument("--input-data",type=click.Path(exists=True, dir_okay=False))
@click.option("-l", "--level-sizes", default=[10, 3], help="List of sizes for different levels, starting from the lowest level.")
@click.option(
    "-e",
    "--epochs",
    default=1000,
    help="Number of epochs to run.",
)
@click.option(
    "-mb",
    "--minibatch-size",
    default=128,
    help="Minibatch size.",
)
@click.option(
    "-s",
    "--step-size",
    default=0.01,
    help="Step size.",
)
@click.option(
    "-mc",
    "--mc-samples",
    default=10,
    help="Number of Monte Carlo samples for gradient estimator.",
)
@click.option(
    "-g",
    "--max-genes",
    default=1000,
    help="Maximum number of genes to keep.",
)
@click.option(
    "-b",
    "--batch-key",
    default="batch",
    help="Batch indicator key in AnnData file.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Report inference diagnostics.",
)
@click.option(
    "-o",
    "--output-path",
    default='./',
    help="Output directory.",
)
def main(
    input_file, level_sizes, epochs, minibatch_size, step_size, mc_samples, max_genes, batch_key, verbose, output_path
):
    adata = sc.read_h5ad(input_file)

    # Do some basic pre-processing
    adata.var_names_make_unique() 
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.raw = adata
    raw_adata = adata.raw
    raw_adata = raw_adata.to_adata()
    raw_adata.X = raw_adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=max_genes)
    raw_adata = raw_adata[:, adata.var.highly_variable]

    # Run scDEF
    scdpf = scDPF(raw_adata, n_factors=level_sizes[0], n_hfactors=level_sizes[1], shape=1.)
    elbos = scdpf.optimize(n_epochs=epochs, batch_size=minibatch_size, step_size=step_size, num_samples=mc_samples)
   
    if verbose:
        # Check diagnostics
        scdpf.report_diagnostics()

    scdpf.adata.obsm['X_hfactors'] = scdpf.pmeans['hz']
    scdpf.adata.obsm['X_factors'] = scdpf.pmeans['z'] * scdpf.pmeans['cell_scale'].reshape(-1,1)
    scdpf.adata.obs['X_hfactor'] = np.argmax(scdpf.adata.obsm['X_hfactors'], axis=1).astype(str)
    scdpf.adata.obs['X_factor'] = np.argmax(scdpf.adata.obsm['X_factors'], axis=1).astype(str)

    for i in range(10):
        scdpf.adata.obs[f'{i}'] = scdpf.adata.obsm['X_factors'][:,i]
    
    for i in range(3):
        scdpf.adata.obs[f'h{i}'] = scdpf.adata.obsm['X_hfactors'][:,i]

    graph = scdpf.get_graph(enrichments=None, ard_filter=[.001, .001], top=10)

    # Save object, annotated AnnData and graph
    Path(output_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(scdpf, os.path.join(output_path, 'scdpf.pkl'))
    scdpf.adata.write(os.path.join(output_path), 'annotated_adata.h5')
    graph.render(directory=output_path)

    print(f"Saved results to {output_path}.")


if __name__ == "__main__":
    main()

