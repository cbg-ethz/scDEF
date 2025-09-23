output_path = "walltime_results"

configfile: "config/walltime_config.yaml"
configfile: "config/methods.yaml"

N_CELLS = config["n_cells"]
N_REPS = config["n_reps"]
METHODS = config["methods"]
METRICS = config["metrics"]

rule all:
    input:
        output_path + '/scores.csv'

rule gather_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            output_path + '/{method}/cellno_{cellno}/rep_{rep_id}_scores.csv',
            method=METHODS,
            cellno=N_CELLS,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/scores.csv'
    run:
        import pandas as pd

        rows = []
        for filename in snakemake.input['fname_list']:
            print(filename)

            # Parse filename
            l = filename.split("/")
            
            rep_id = l[-1].split("_")[1]
            method = l[-2] # method is the number of layers

            # Parse scores
            df = pd.read_csv(filename, index_col=0)
            print(df)

            for idx, score in enumerate(df.index): # must have the ARI per layer
                value = df.values[idx]
                value = float(value[0])
                rows.append(
                    [
                        method,
                        rep_id,
                        score,
                        value,
                    ]
                )

        columns = [
            "method",
            "rep_id",
            "score",
            "value",
        ]

        scores = pd.DataFrame.from_records(rows, columns=columns)
        print(scores)

        scores = pd.melt(
            scores,
            id_vars=[
                "method",
                "rep_id",
                "score",
            ],
            value_vars="value",
        )
        scores.to_csv(snakemake.output[0], index=False)    


rule generate_multibatch_data:
    resources:
        mem_per_cpu=10000,
        threads=10,
    params:
        de_fscale = config["de_fscale"],
        de_prob = config["de_prob"],
        batch_facscale = config["batch_facscale"],
        n_cells = "{n_cells}",
        n_batches = config["n_batches"],
        frac_shared = config["frac_shared"],
        seed = "{rep_id}",
        coverage = 0.,
    output:
        counts_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
        umap_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"


rule prepare_input:
    conda:
        "../envs/PCA.yml"
    params:
        seed = "{rep_id}",
    input:
        counts_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    script:
        '../scripts/prepare_input.py'

rule run_scdef:
    conda:
        "../envs/scdef.yml"
    params:
        n_factors = config['scDEF']['n_factors'],
        nmf_init = config['scDEF']['nmf_init'],
        pretrain = config['scDEF']['pretrain'],
        tau = config['scDEF']['tau'],
        mu = config['scDEF']['mu'],
        decay_factor = config['scDEF']['decay_factor'],
        kappa = config['scDEF']['kappa'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        metrics = METRICS,
        seed = "{rep_id}",
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/scDEF/cellno_{n_cells}/rep_{rep_id}.pkl',
        scores_fname = output_path + '/scDEF/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef.py'


rule run_scdef_un:
    conda:
        "../envs/scdef.yml"        
    params:
        n_factors = config['scDEF_un']['n_factors'],
        nmf_init = config['scDEF_un']['nmf_init'],
        pretrain = config['scDEF_un']['pretrain'],
        tau = config['scDEF_un']['tau'],
        mu = config['scDEF_un']['mu'],
        decay_factor = config['scDEF_un']['decay_factor'],
        kappa = config['scDEF_un']['kappa'],
        n_epoch = config['scDEF_un']['n_epoch'],
        lr = config['scDEF_un']['lr'],
        batch_size = config['scDEF_un']['batch_size'],
        num_samples = config['scDEF_un']['num_samples'],  
        metrics = METRICS,        
        seed = "{rep_id}",
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/scDEF_un/cellno_{n_cells}/rep_{rep_id}.pkl',
        scores_fname = output_path + '/scDEF_un/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef_un.py'

rule run_pca:
    conda:
        "../envs/PCA.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "PCA",
        n_top_genes = config["PCA"]['n_top_genes'],
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/PCA/cellno_{n_cells}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/PCA/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_unintegrated.py'

rule run_nmf:
    conda:
        "../envs/NMF.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "NMF",
        max_iter = config["NMF"]['max_iter'],
        n_top_genes = config["NMF"]['n_top_genes'],
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/NMF/cellno_{n_cells}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/NMF/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_nmf.py'

rule run_schpf:
    conda:
        "../envs/scHPF.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "scHPF",
        max_iter = config["scHPF"]['max_iter'],
        min_iter = config["scHPF"]['min_iter'],
        n_top_genes = config["scHPF"]['n_top_genes'],
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/scHPF/cellno_{n_cells}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/scHPF/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:    
        '../scripts/run_schpf.py'

rule run_scvi:
    conda:
        "../envs/scVI.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "scVI",
        n_top_genes = config["scVI"]['n_top_genes'],
        max_epochs = config["scVI"]['max_epochs'],
        batch_size = config["scVI"]['batch_size'],
        early_stopping = config["scVI"]['early_stopping'],
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/scVI/cellno_{n_cells}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/scVI/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scvi.py'

rule run_harmony:
    conda:
        "../envs/Harmony.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "Harmony",
        n_top_genes = config["Harmony"]['n_top_genes'],
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/Harmony/cellno_{n_cells}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/Harmony/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_harmony.py'

rule run_scanorama:
    conda:
        "../envs/Scanorama.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "Scanorama",
        n_top_genes = config["Scanorama"]['n_top_genes'],
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/Scanorama/cellno_{n_cells}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/Scanorama/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scanorama.py'

rule run_nsbm:
    conda:
        "../envs/nSBM.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "nSBM",
        n_top_genes = config["nSBM"]['n_top_genes'],
        n_init = config["nSBM"]['n_init'],
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/nSBM/cellno_{n_cells}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/nSBM/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_nsbm.py'

rule run_fsclvm:
    conda:
        "../envs/fscLVM.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "fscLVM",
        batch_size = config["fscLVM"]['batch_size'],
        n_epochs = config["fscLVM"]['n_epochs'],
        n_top_genes = config["fscLVM"]['n_top_genes'],
        store_full = True
    input:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    output:
        out_fname = output_path + '/fscLVM/cellno_{n_cells}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/fscLVM/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_muvi.py'