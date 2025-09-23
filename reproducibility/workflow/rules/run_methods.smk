envs_path = "../envs"
scripts_path = "../scripts"

rule run_scdef:
    conda:
        envs_path + "/scdef.yml"
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
        seed = SEED,
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/scDEF/scDEF.pkl',
        scores_fname = output_path + '/scDEF/scDEF.csv',
    script:
        scripts_path + "/run_scdef.py"


rule run_scdef_un:
    conda:
        envs_path + "/scdef.yml"        
    params:
        n_layers = config['scDEF_un']['n_layers'],
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
        seed = SEED,
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/scDEF_un/scDEF_un.pkl',
        scores_fname = output_path + '/scDEF_un/scDEF_un.csv',
    script:
        scripts_path + "/run_scdef_un.py"

rule run_pca:
    conda:
        envs_path + "/PCA.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "PCA",
        n_top_genes = config["PCA"]['n_top_genes'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/PCA/PCA.h5ad',
        scores_fname = output_path + '/PCA/PCA.csv',
    script:
        scripts_path + "/run_unintegrated.py"

rule run_nmf:
    conda:
        envs_path + "/NMF.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "NMF",
        max_iter = config["NMF"]['max_iter'],
        n_top_genes = config["NMF"]['n_top_genes'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/NMF/NMF.h5ad',
        scores_fname = output_path + '/NMF/NMF.csv',
    script:
        scripts_path + "/run_nmf.py"

rule run_schpf:
    conda:
        envs_path + "/scHPF.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "scHPF",
        max_iter = config["scHPF"]['max_iter'],
        min_iter = config["scHPF"]['min_iter'],
        n_top_genes = config["scHPF"]['n_top_genes'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/scHPF/scHPF.h5ad',
        scores_fname = output_path + '/scHPF/scHPF.csv',
    script:    
        scripts_path + "/run_schpf.py"

rule run_scvi:
    conda:
        envs_path + "/scVI.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "scVI",
        n_top_genes = config["scVI"]['n_top_genes'],
        max_epochs = config["scVI"]['max_epochs'],
        batch_size = config["scVI"]['batch_size'],
        early_stopping = config["scVI"]['early_stopping'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/scVI/scVI.h5ad',
        scores_fname = output_path + '/scVI/scVI.csv',
    script:
        scripts_path + "/run_scvi.py"

rule run_harmony:
    conda:
        envs_path + "/Harmony.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "Harmony",
        n_top_genes = config["Harmony"]['n_top_genes'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/Harmony/Harmony.h5ad',
        scores_fname = output_path + '/Harmony/Harmony.csv',
    script:
        scripts_path + "/run_harmony.py"

rule run_scanorama:
    conda:
        envs_path + "/Scanorama.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "Scanorama",
        n_top_genes = config["Scanorama"]['n_top_genes'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/Scanorama/Scanorama.h5ad',
        scores_fname = output_path + '/Scanorama/Scanorama.csv',
    script:
        scripts_path + "/run_scanorama.py"

rule run_nsbm:
    conda:
        envs_path + "/nSBM.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "nSBM",
        n_top_genes = config["nSBM"]['n_top_genes'],
        n_init = config["nSBM"]['n_init'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/nSBM/nSBM.h5ad',
        scores_fname = output_path + '/nSBM/nSBM.csv',
    script:
        scripts_path + "/run_nsbm.py"

rule run_fsclvm:
    conda:
        envs_path + "/fscLVM.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "fscLVM",
        batch_size = config["fscLVM"]['batch_size'],
        n_epochs = config["fscLVM"]['n_epochs'],
        n_top_genes = config["fscLVM"]['n_top_genes'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/fscLVM/fscLVM.h5ad',
        scores_fname = output_path + '/fscLVM/fscLVM.csv',
    script:
        scripts_path + "/run_muvi.py"