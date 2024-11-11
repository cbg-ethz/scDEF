scripts_path = "../../../../scripts"
envs_path = "../../../../envs"

rule run_scdef:
    conda:
        envs_path + "/scdef.yml"
    resources:
        partition = config['scDEF']['partition'],
    params:
        nmf_init = config['scDEF']['nmf_init'],
        tau = config['scDEF']['tau'],
        mu = config['scDEF']['mu'],
        layer_sizes = config['scDEF']['layer_sizes'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        metrics = METRICS,
        seed = SEED,
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/scDEF/scDEF.pkl',
        scores_fname = output_path + '/scDEF/scDEF.csv',
    script:
        scripts_path + "/run_scdef.py"


rule run_scdef_un:
    conda:
        envs_path + "/scdef.yml"        
    resources:
        partition = config['scDEF_un']['partition'],
        slurm_extra="--gres=gpu:1"
    params:
        nmf_init = config['scDEF_un']['nmf_init'],
        tau = config['scDEF_un']['tau'],
        mu = config['scDEF_un']['mu'],
        layer_sizes = config['scDEF_un']['layer_sizes'],
        n_epoch = config['scDEF_un']['n_epoch'],
        lr = config['scDEF_un']['lr'],
        batch_size = config['scDEF_un']['batch_size'],
        num_samples = config['scDEF_un']['num_samples'],  
        metrics = METRICS,        
        seed = SEED,
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
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
        settings = config["PCA"]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/PCA/PCA.h5ad',
        scores_fname = output_path + '/PCA/PCA.csv',
    script:
        scripts_path + "/run_method.py"

rule run_nmf:
    conda:
        envs_path + "/NMF.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "NMF",
        n_top_genes = config["NMF"]['n_top_genes'],
        settings = config["NMF"]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/NMF/NMF.h5ad',
        scores_fname = output_path + '/NMF/NMF.csv',
    script:
        scripts_path + "/run_method.py"

rule run_schpf:
    conda:
        envs_path + "/scHPF.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "scHPF",
        n_top_genes = config["scHPF"]['n_top_genes'],
        settings = config["scHPF"]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/scHPF/scHPF.h5ad',
        scores_fname = output_path + '/scHPF/scHPF.csv',
    script:
        scripts_path + "/run_method.py"

rule run_scvi:
    conda:
        envs_path + "/scVI.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "scVI",
        n_top_genes = config["scVI"]['n_top_genes'],
        settings = config["scVI"]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/scVI/scVI.h5ad',
        scores_fname = output_path + '/scVI/scVI.csv',
    script:
        scripts_path + "/run_method.py"

rule run_harmony:
    conda:
        envs_path + "/Harmony.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "Harmony",
        n_top_genes = config["Harmony"]['n_top_genes'],
        settings = config["Harmony"]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/Harmony/Harmony.h5ad',
        scores_fname = output_path + '/Harmony/Harmony.csv',
    script:
        scripts_path + "/run_method.py"

rule run_scanorama:
    conda:
        envs_path + "/Scanorama.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "Scanorama",
        n_top_genes = config["Scanorama"]['n_top_genes'],
        settings = config["Scanorama"]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/Scanorama/Scanorama.h5ad',
        scores_fname = output_path + '/Scanorama/Scanorama.csv',
    script:
        scripts_path + "/run_method.py"

rule run_nsbm:
    conda:
        envs_path + "/nSBM.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "nSBM",
        n_top_genes = config["nSBM"]['n_top_genes'],
        settings = config["nSBM"]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/nSBM/nSBM.h5ad',
        scores_fname = output_path + '/nSBM/nSBM.csv',
    script:
        scripts_path + "/run_method.py"

rule run_fsclvm:
    conda:
        envs_path + "/fscLVM.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "fscLVM",
        n_top_genes = config["fscLVM"]['n_top_genes'],
        settings = config["fscLVM"]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/fscLVM/fscLVM.h5ad',
        scores_fname = output_path + '/fscLVM/fscLVM.csv',
    script:
        scripts_path + "/run_method.py"