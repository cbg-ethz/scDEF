scripts_path = "../../../../scripts"
envs_path = "../../../../envs"

rule run_scdef:
    conda:
        envs_path + "/scdef.yml"
    resources:
        partition = 'gpu'        
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
        partition = 'gpu'
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

rule run_method:
    conda:
        envs_path + "/{method}.yml"        
    resources:
        partition = lambda wildcards: config[wildcards.method]['partition'],
    params:
        metrics = METRICS,    
        seed = SEED,
        method = "{method}",
        n_top_genes = lambda wildcards: config[wildcards.method]['n_top_genes'],
        settings = lambda wildcards: config[wildcards.method]['settings'],
        store_full = True
    input:
        fname = output_path + '/prepared_input.h5ad',
    output:
        out_fname = output_path + '/{method}/{method}.h5ad',
        scores_fname = output_path + '/{method}/{method}.csv',
    script:
        scripts_path + "/run_method.py"

# rule run_unintegrated:
#     conda:
#         envs_path + "/scdef.yml"        
#     params:
#         metrics = METRICS,    
#         seed = SEED,
#     input:
#         adata = output_path + '/prepared_input.h5ad',
#     output:
#         out_fname = output_path + '/Unintegrated/Unintegrated.h5ad',
#         scores_fname = output_path + '/Unintegrated/Unintegrated.csv',
#     script:
#         scripts_path + "/run_unintegrated.py"

# rule run_nmf:
#     conda:
#         envs_path + "/scdef.yml"        
#     params:
#         metrics = METRICS,    
#         seed = SEED,
#         method = 'NMF',
#         settings = config['NMF'],
#     input:
#         adata = output_path + '/prepared_input.h5ad',
#     output:
#         out_fname = output_path + '/NMF/NMF.h5ad',
#         scores_fname = output_path + '/NMF/NMF.csv',
#     script:
#         scripts_path + "/run_method.py"

# rule run_schpf:
#     params:
#         metrics = METRICS,    
#         seed = SEED,
#         method = 'scHPF',
#         settings = config['scHPF'],
#     input:
#         adata = output_path + '/prepared_input.h5ad',
#     output:
#         out_fname = output_path + '/scHPF/scHPF.h5ad',
#         scores_fname = output_path + '/scHPF/scHPF.csv',
#     script:
#         scripts_path + "/run_method.py"

# rule run_scvi:
#     conda:
#         envs_path + "/scvi.yml"                
#     params:
#         metrics = METRICS,    
#         seed = SEED,
#         method = 'scVI',
#         settings = config['scVI']
#     input:
#         adata = output_path + '/prepared_input.h5ad',
#     output:
#         out_fname = output_path + '/scVI/scVI.h5ad',
#         scores_fname = output_path + '/scVI/scVI.csv',
#     script:
#         scripts_path + "/run_method.py"

# rule run_harmony:
#     conda:
#         envs_path + "/harmony.yml"
#     params:
#         metrics = METRICS,    
#         seed = SEED,
#         method = 'Harmony',
#     input:
#         adata = output_path + '/prepared_input.h5ad',
#     output:
#         out_fname = output_path + '/Harmony/Harmony.h5ad',
#         scores_fname = output_path + '/Harmony/Harmony.csv',
#     script:
#         scripts_path + "/run_method.py"

# rule run_scanorama:
#     conda:
#         envs_path + "/scanorama.yml"        
#     params:
#         metrics = METRICS,    
#         seed = SEED,
#         method = 'Scanorama',
#     input:
#         adata = output_path + '/prepared_input.h5ad',
#     output:
#         out_fname = output_path + '/Scanorama/Scanorama.h5ad',
#         scores_fname = output_path + '/Scanorama/Scanorama.csv',
#     script:
#         scripts_path + "/run_method.py"

# rule run_nsbm:
#     conda:
#         envs_path + "/nsbm.yml"                
#     params:
#         metrics = METRICS,    
#         seed = SEED,
#         method = 'nSBM',
#         settings = config['nSBM'],
#     input:
#         adata = output_path + '/prepared_input.h5ad',
#     output:
#         out_fname = output_path + '/nSBM/nSBM.h5ad',
#         scores_fname = output_path + '/nSBM/nSBM.csv',
#     script:
#         scripts_path + "/run_method.py"

# rule run_fsclvm:
#     conda:
#         envs_path + "/fsclvm.yml"                        
#     params:
#         metrics = METRICS,    
#         seed = SEED,
#         method = 'fscLVM',
#         batch_size = config['fscLVM']['batch_size'],   
#         n_epochs = config['fscLVM']['n_epochs'],   
#     input:
#         adata = output_path + '/prepared_input.h5ad',
#     output:
#         out_fname = output_path + '/fscLVM/fscLVM.h5ad',
#         scores_fname = output_path + '/fscLVM/fscLVM.csv',
#     script:
#         scripts_path + "/run_method.py"