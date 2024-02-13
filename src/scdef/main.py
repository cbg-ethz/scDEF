"""
Extract hierarchical signatures of cell state from single-cell data.
"""
from .models import scDEF

import numpy as np
import pandas as pd

import logging
import click
import os
from pathlib import Path


@click.command(
    help="Extract hierarchical signatures of cell state from single-cell data."
)
@click.argument("--input-data", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-l",
    "--level-sizes",
    default=[10, 3],
    help="List of sizes for different levels, starting from the lowest level.",
)
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
    default="./",
    help="Output directory.",
)
def main(
    input_file,
    level_sizes,
    epochs,
    minibatch_size,
    step_size,
    mc_samples,
    max_genes,
    batch_key,
    verbose,
    output_path,
):
    logging.basicConfig(level=logging.INFO)

    adata = sc.read_h5ad(input_file)

    # Do some basic pre-processing
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var["mt"] = adata.var_names.str.startswith(
        "MT-"
    )  # annotate the group of mitochondrial genes as 'mt'
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    adata.raw = adata
    raw_adata = adata.raw
    raw_adata = raw_adata.to_adata()
    raw_adata.X = raw_adata.X.toarray()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=max_genes
    )
    raw_adata = raw_adata[:, adata.var.highly_variable]

    # Run scDEF
    scdef = scDEF(raw_adata, layer_sizes=level_sizes)
    elbos = scdef.optimize(
        n_epochs=epochs, batch_size=minibatch_size, lr=step_size, num_samples=mc_samples
    )

    summary = scdef.get_summary(top_genes=5)

    # Save object, annotated AnnData and graph
    Path(output_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(scdef, os.path.join(output_path, "scdef.pkl"))
    scdef.adata.write(os.path.join(output_path), "scdef_adata.h5ad")
    scd.graph.render(directory=output_path)

    logging.info(f"Saved results to {output_path}.")


if __name__ == "__main__":
    main()
