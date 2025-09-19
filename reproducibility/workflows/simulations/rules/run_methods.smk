scripts_path = "../../../../scripts"
envs_path = "../../../../envs"

rule run_scdef:
    conda:
        envs_path + "/scdef.yml"
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
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/scDEF_un/scDEF_un.pkl',
        scores_fname = output_path + '/scDEF_un/scDEF_un.csv',
    script:
        scripts_path + "/run_scdef_un.py"

rule run_method:
    conda:
        envs_path + "/{method}.yml"        
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "{method}",
        n_top_genes = lambda wildcards: config[wildcards.method]['n_top_genes'],
        settings = lambda wildcards: config[wildcards.method]['settings'],
        store_full = True
    input:
        adata = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/{method}/{method}.h5ad',
        scores_fname = output_path + '/{method}/{method}.csv',
    script:
        scripts_path + "/run_method.py"