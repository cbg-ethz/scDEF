import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

from scdef import scDPF

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

results_file = 'write/pbmc3k.h5ad'  # the file that will store the analysis results
adata = sc.read_10x_mtx(
    'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)
adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`
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
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=800)
raw_adata = raw_adata[:, adata.var.highly_variable]
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7'])

adata.shape
raw_adata.obsm['X_umap'] = adata.obsm['X_umap']
raw_adata.obs['leiden'] = adata.obs['leiden']
cluster_assignments = adata.obs['leiden']
unique_cluster_ids = np.unique(cluster_assignments.values)

scdpf = scDPF(raw_adata, n_factors=20, n_hfactors=10, shape=.1)
elbos = scdpf.optimize(n_epochs=3000, batch_size=2638, step_size=1e-1, num_samples=10)


plt.plot(elbos[:])
scdpf.plot_ard(layer_idx=0)
scdpf.plot_ard(layer_idx=1)

cell_scores = scdpf.pmeans['z'] / scdpf.pmeans['cell_scale'].reshape(-1,1)
cell_hscores = scdpf.pmeans['hz'] / scdpf.pmeans['cell_scale'].reshape(-1,1)
mean_cluster_scores = []
mean_cluster_hscores = []
for c in unique_cluster_ids:
    cell_idx = np.where(cluster_assignments == c)[0]
    mean_cluster_scores.append(np.mean(cell_scores[cell_idx], axis=0))
    mean_cluster_hscores.append(np.mean(cell_hscores[cell_idx], axis=0))

mean_cluster_scores = np.array(mean_cluster_scores)
mean_cluster_hscores = np.array(mean_cluster_hscores)

# scdpf.adata.obsm['X_hfactors'] = scdpf.pmeans['hz'] #/ scdpf.pmeans['cell_scale'].reshape(-1,1)
scdpf.adata.obs['X_hfactor'] = np.argmax(scdpf.adata.obsm['X_hfactors'], axis=1).astype(str)
scdpf.adata.obs['X_factor'] = np.argmax(scdpf.adata.obsm['X_factors'], axis=1).astype(str)


plt.scatter(np.log2(np.sum(scdpf.adata.X, axis=0)), np.log2(np.exp(scdpf.pmeans['gene_scale'])))

plt.scatter(np.log2(np.sum(scdpf.adata.X, axis=1)), np.log2(np.exp(scdpf.pmeans['cell_scale'])))


sc.pl.umap(scdpf.adata, color=['leiden', 'X_factor', 'X_hfactor'])

plt.pcolormesh(scdpf.pmeans['z']);plt.colorbar()

plt.pcolormesh(scdpf.pmeans['W']);plt.colorbar()
sns.clustermap(cell_hscores.dot(scdpf.pmeans['hW']));

lut = dict(zip(np.unique(adata.obs['leiden']), adata.uns['leiden_colors']))
row_colors = adata.obs['leiden'].map(lut)

sns.clustermap(scdpf.pmeans['W'] / scdpf.pmeans['gene_scale'], row_cluster=False)
plt.title('Topics')
plt.show()


sns.clustermap(scdpf.pmeans['z'] / scdpf.pmeans['cell_scale'].reshape(-1,1), col_cluster=False, row_colors=row_colors.values)
plt.title('Cell scores on topics')
plt.show()

sns.clustermap(scdpf.pmeans['hW'], row_cluster=False)
plt.title('Hierarchical topics')
plt.show()

sns.clustermap(scdpf.pmeans['hz'] / scdpf.pmeans['cell_scale'].reshape(-1,1), col_cluster=False, row_colors=row_colors.values)
plt.title('Cell scores on hierarchical topics')
plt.show()

fig = sns.clustermap(mean_cluster_scores, row_colors=list(lut.values()), col_cluster=False, row_cluster=False)
ax = fig.ax_heatmap
ax.set_xlabel("Factors")
ax.set_ylabel("Clusters")
plt.show()

fig = sns.clustermap(mean_cluster_hscores, row_colors=list(lut.values()), col_cluster=False, row_cluster=False)
ax = fig.ax_heatmap
ax.set_xlabel("H. Factors")
ax.set_ylabel("Clusters")
plt.show()
plt.pcolormesh(scdpf.pmeans['hW'])

ranks = scdpf.get_rankings()

scdpf.get_graph(enrichments=None, ard_filter=[.1, 0.1])
