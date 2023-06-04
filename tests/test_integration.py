import pytest
from click.testing import CliRunner
import os

import scanpy as sc
import numpy as np

from scdef import main, scDEF

def test_scdef():
    n_epochs = 10

    # Download data
    adata = sc.datasets.pbmc3k()

    # Filter data
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[np.random.randint(adata.shape[0], size=200)]
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
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=100)
    raw_adata = raw_adata[:, adata.var.highly_variable]
    adata = adata[:, adata.var.highly_variable]

    scd = scDEF(raw_adata, layer_sizes= [60, 30, 15], layer_shapes=1., seed=1, batch_key='Experiment')
    assert hasattr(scd, 'adata')

    scd.learn(n_epoch=3)
    assert len(scd.elbos) == 1
    assert 'factor' in scd.adata.obs.columns
    assert 'hfactor' in scd.adata.obs.columns
    assert 'hhfactor' in scd.adata.obs.columns
#
#
# def test_cli():
#     runner = CliRunner()
#
#     # run program
#     with runner.isolated_filesystem():
#         result = runner.invoke(main, ["-o", "./outs"])
#
#         # test output
#         assert result.exit_code == 0
#         assert os.path.isfile("./outs/scdef.pkl")
#         assert os.path.isfile("./outs/scdef_adata.h5ad")
#         assert os.path.isfile("./outs/graph.pdf")
