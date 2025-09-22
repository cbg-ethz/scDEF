"""
1. Generate data with different levels of DE genes and check if we still find the cell types: cell type ARI for different $\tau$ and different $\mu$, for the different scDEF layers
2. Generate data with different levels of coverage and check if we still find the cell types: hierarchy accuracy for different $\kappa$
"""

configfile: "config/realdata_config.yaml"

n_reps = 5
brds = [(0.1, 1.), (0.01, 1.), (0.01, 10.), (0.01, 100.)] # (mu,tau)
kappas = [1., 10., 100.]
layer_sizes_list = [[100], [100,10], [100,30,10], [100,60,30,10]]

rule all:
    input:
        'realdata_results/realdata_results.csv',

rule gather_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'realdata_results/brd_{brd_id}_kappa_{kappa}_layers_{n_layers}_rep_{rep_id}_scores.csv',
            brd_id=range(len(brds)),
            kappa=kappas,
            n_layers=[len(layer_sizes) for layer_sizes in layer_sizes_list],
            rep_id=[r for r in range(n_reps)],),
    output:
        fname = 'realdata_results/realdata_results.csv',
    run:
        import pandas as pd
        df = pd.read_csv(input['fname_list'][0], index_col=0)
        df = df.reset_index(names='score')
        df = df.rename(columns={"scDEF":"value"})
        for filename in input['fname_list'][1:]:
            print(filename)
            new_df = pd.read_csv(filename, index_col=0)
            new_df = new_df.reset_index(names='score')
            new_df = new_df.rename(columns={"scDEF":"value"})
            df = pd.concat([df, new_df])
        df = df.reset_index(drop=True)
        df.to_csv(output['fname'], index=False)   

rule preprocess_data:
    input:
        fname = config['input_data'],
    output:
        fname = 'realdata_results/adata.h5ad'
    params:
        n_top_genes = 2000,
    run:
        import scanpy as sc
        adata = sc.read_h5ad(input['fname'])
        
        adata = sc.pp.subsample(adata, n_obs=10000, copy=True)
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=10)

        # As in the paper
        adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        adata = adata[adata.obs.n_genes_by_counts < 5000, :]
        adata = adata[adata.obs.pct_counts_mt < 5, :]
        adata.raw = adata

        raw_adata = adata.raw
        raw_adata = raw_adata.to_adata()
        raw_adata.X = raw_adata.X.toarray()
        adata.layers['counts'] = raw_adata.X

        adata.layers['counts'] =  adata.X.toarray()

        # Keep only HVGs
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000, layer='counts',
                                batch_key='stim') # Not required, but makes scDEF faster

        adata = adata[:, adata.var.highly_variable]

        # Process and visualize the data
        sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)

        # Cell state
        adata.obs['cell_state'] = adata.obs['seurat_annotations']
        adata.obs['cell_state'] = adata.obs.apply(lambda r: f'{r["seurat_annotations"]}_{r["stim"]}' if "Mono" in r["seurat_annotations"] else r["seurat_annotations"], axis=1)

        sc.pl.umap(adata, color=['leiden', 'cell_state', 'stim'], frameon=False)

        adata.write_h5ad(output['fname'])

rule run_scdef:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        brds = brds,
        brd_id = "{brd_id}",
        kappa = "{kappa}",
        layer_sizes = layer_sizes_list,
        n_layers = "{n_layers}",
        seed = "{rep_id}",
        n_epoch = config["n_epoch"]
    input:
        fname = rules.preprocess_data.output.fname,
    output:
        scores_fname = 'realdata_results/brd_{brd_id}_kappa_{kappa}_layers_{n_layers}_rep_{rep_id}_scores.csv',
        graph_fname = 'realdata_results/brd_{brd_id}_kappa_{kappa}_layers_{n_layers}_rep_{rep_id}_graph.pdf',
    script:
        "scripts/run_scdef_realdata.py"