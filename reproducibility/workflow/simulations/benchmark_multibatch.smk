output_path = "results/multibatch"

localrules: gather_multibatch_scores

configfile: "config/benchmark_multibatch.yaml"
configfile: "config/methods.yaml"

METHODS = config["methods"]
METRICS = config["metrics"]

N_REPS = config["n_reps"]
SEPARABILITY = config["de_fscale"]
FRACS_SHARED = config["frac_shared"]

rule all:
    input:
        output_path + '/multibatch_scores.csv'

rule gather_multibatch_scores:
    conda:
        "../envs/PCA.yml"
    input:
        fname_list = expand(
            output_path + '/{method}/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
            method=METHODS,
            frac_shared=FRACS_SHARED, separability=SEPARABILITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        fname = output_path + '/multibatch_scores.csv'
    script:
        '../scripts/gather_multibatch_scores.py'

rule generate_multibatch_data:
    conda:
        "../envs/splatter.yml"
    params:
        de_fscale = "{separability}",
        de_prob = config["de_prob"],
        batch_facscale = config["batch_facscale"],
        n_cells = 1000,
        n_batches = config["n_batches"],
        frac_shared = "{frac_shared}",
        seed = "{rep_id}",
        coverage = 0.,
    output:
        counts_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
        umap_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_umap_nobatch.png',
    script:
        '../scripts/splatter_hierarchical.R'


rule prepare_input:
    conda:
        "../envs/PCA.yml"
    params:
        seed = "{rep_id}",
    input:
        counts_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad'
    script:
        '../scripts/prepare_input.py'

rule run_scdef:
    conda:
        "../envs/scdef.yml"
    params:
        n_layers = config['scDEF']['n_layers'],
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/scDEF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.pkl',
        scores_fname = output_path + '/scDEF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef.py'


rule run_scdef_un:
    conda:
        "../envs/scdef.yml"        
    params:
        n_factors = config['scDEF_un']['n_factors'],
        n_layers = config['scDEF_un']['n_layers'],
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/scDEF_un/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.pkl',
        scores_fname = output_path + '/scDEF_un/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/PCA/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/PCA/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/NMF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/NMF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/scHPF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/scHPF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/scVI/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/scVI/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/Harmony/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/Harmony/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/Scanorama/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/Scanorama/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/nSBM/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/nSBM/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
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
        adata = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/fscLVM/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
        scores_fname = output_path + '/fscLVM/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_muvi.py'