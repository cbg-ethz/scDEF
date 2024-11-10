import scanpy as sc
import numpy as np
import pandas as pd

np.random.seed(snakemake.params.seed)

data_path = snakemake.params.data_path

adata = sc.read(data_path + '/dge.txt').T

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs['n_genes'] < 2500, :]
sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
filter_result = sc.pp.filter_genes_dispersion(
    adata.X, min_mean=0.01, max_mean=3, min_disp=0.4)
sc.pl.filter_genes_dispersion(filter_result)

adata = adata[:, filter_result.gene_subset]

adata.layers['counts'] = np.array(adata.X)

sc.pp.log1p(adata)
sc.pp.regress_out(adata, 'n_counts')
sc.pp.scale(adata, max_value=10)

sc.pp.pca(adata)
adata.obsm['X_pca'][:, 1] *= -1  # to match Seurat

X_pca_seurat = np.loadtxt(data_path + '/R_pca_seurat.txt').astype('float32')

adata.obsm['X_pca'] = X_pca_seurat

sc.pp.neighbors(adata, n_neighbors=30)

sc.tl.louvain(adata, resolution=1)

adata.obs['clusters'] = np.genfromtxt(data_path + '/R_annotation.txt', delimiter=',', dtype=str)
sc._utils.sanitize_anndata(adata)

adata.obs['clusters'].cat.reorder_categories([
    'early epidermal progenitors', 'activated early epidermal progenitors',
    'epidermal neoblasts', 'epidermis', 'epidermis DVb',
    'epidermis DVb neoblast', 'glia', 'phagocytes', 'goblet cells',
    'psd+ cells', 'gut progenitors', 'late epidermal progenitors 1',
    'late epidermal progenitors 2', 'muscle body', 'muscle pharynx',
    'muscle progenitors', 'neoblast 1', 'neoblast 2', 'neoblast 3',
    'neoblast 4', 'neoblast 5', 'neoblast 6', 'neoblast 7', 'neoblast 8',
    'neoblast 9', 'neoblast 10', 'neoblast 11', 'neoblast 12',
    'neoblast 13', 'ChAT neurons 1', 'ChAT neurons 2', 'GABA neurons',
    'otf+ cells 1', 'otf+ cells 2', 'spp-11+ neurons', 'npp-18+ neurons',
    'cav-1+ neurons', 'neural progenitors', 'pharynx cell type progenitors',
    'pgrn+ parenchymal cells', 'ldlrr-1+ parenchymal cells',
    'psap+ parenchymal cells', 'aqp+ parenchymal cells',
    'parenchymal progenitors', 'pharynx cell type', 'pigment',
    'protonephridia', 'secretory 1', 'secretory 2', 'secretory 3',
    'secretory 4'], inplace=True)

colors = pd.read_csv(data_path + '/colors_dataset.txt', header=None, sep='\t')
# transform to dict where keys are cluster names
colors = {k: c for k, c in colors.values}
adata.uns['clusters_colors'] = [colors[clus] for clus in adata.obs['clusters'].cat.categories]

sc.tl.tsne(adata)

sc.pl.tsne(adata, color='clusters', legend_loc='on data', legend_fontsize=5)

sc.pl.tsne(adata, color='louvain', legend_loc='on')

map_neoblasts = {c: c if not c.startswith('neoblast') else 'neoblast' for c in adata.obs['clusters'].cat.categories}

adata.obs['clusters_neoblasts'] = (
    adata.obs['clusters']
    .map(map_neoblasts)
    .astype('category')
)

cols = []
for clus in adata.obs['clusters_neoblasts'].cat.categories:
    if not clus == 'neoblast':
        cols.append(colors[clus])

    else:
        cols.append('grey80')

adata.uns['clusters_neoblasts_colors'] = cols

map_coarse = {}
colors_coarse = {}
for c in adata.obs['clusters'].cat.categories:
    if c.startswith('neoblast'):
        map_coarse[c] = 'stem cells'
        colors_coarse['stem cells'] = colors['neoblast 1']
    elif 'neur' in c:
        map_coarse[c] = 'neurons'
        colors_coarse['neurons'] = colors['neural progenitors']
    elif 'epider' in c:
        map_coarse[c] = 'epidermis'
        colors_coarse['epidermis'] = colors['early epidermal progenitors']
    elif 'muscle' in c:
        map_coarse[c] = 'muscle'
        colors_coarse['muscle'] = colors['muscle body']
    elif 'parenchy' in c:
        map_coarse[c] = 'parenchyma'
        colors_coarse['parenchyma'] = colors['parenchymal progenitors']
    elif 'secretory' in c:
        map_coarse[c] = 'secretory'
        colors_coarse['secretory'] = colors['secretory 4']
    elif 'pharynx' in c:
        map_coarse[c] = 'pharynx'
        colors_coarse['pharynx'] = colors['pharynx cell type progenitors']
    elif 'gut' in c or 'phagocyt' in c or 'goblet' in c:
        map_coarse[c] = 'gut'
        colors_coarse['gut'] = colors['gut progenitors']
    else:
        map_coarse[c] = c
        colors_coarse[c] = colors[c]

adata.obs['clusters_coarse'] = (
    adata.obs['clusters']
    .map(map_coarse)
    .astype('category')
)

adata.uns['clusters_coarse_colors'] = [colors_coarse[c] for c in adata.obs['clusters_coarse'].cat.categories]

sc.pl.tsne(adata, color=['clusters', 'clusters_neoblasts', 'clusters_coarse'],
           legend_loc='on data', legend_fontsize=5)

sc.pp.neighbors(adata, n_neighbors=30)

# Update annotations
adata.obs["Cell types"] = adata.obs["clusters_neoblasts"]
adata.obs["Coarse cell types"] = adata.obs["clusters_coarse"]

adata.uns["hierarchy_obs"] = ["Cell types", "Coarse cell types"]
for i, obs in enumerate(adata.uns["hierarchy_obs"]):
    adata.obs[obs] = "h" * i + adata.obs[obs].astype(str)

# Load marker genes
markers = pd.read_csv(snakemake.params.markers_fname, index_col=0)
markers_dict = dict()
for cell_type in adata.obs["Cell types"].unique():
    markers_dict[cell_type] = markers.loc[
        markers["clusters_neoblasts"] == cell_type
    ].index.tolist()
adata.uns["true_markers"] = markers_dict

# Load gene names
gene_names = pd.read_csv(snakemake.params.gene_names, index_col=0)
adata.var["gene_name"] = ""
for gene in gene_names.index:
    adata.var.loc[gene, "gene_name"] = gene.loc[gene, "name"]

# Keep only HVGs
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata, flavor="seurat_v3", layer="counts", n_top_genes=snakemake.params.n_top_genes
)  # Not required, but makes scDEF faster
adata = adata[:, adata.var.highly_variable]

# Process and visualize the data
sc.pp.regress_out(adata, ["total_counts", "pct_counts_mt"])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver="arpack")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(
    adata, color=adata.uns["hierarchy_obs"] + ["leiden"], show=False
)  # just to make the colors

# Write new h5ad file
adata.write(snakemake.output.fname)
