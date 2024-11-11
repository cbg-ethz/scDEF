configfile: "config/benchmark_singlebatch.yaml"
configfile: "config/methods.yaml"

output_path = "results/singlebatch"
scripts_path = "scripts"
envs_path = "../../../envs"
methods_scripts_path = "../../../scripts"

METHODS = config["methods"]
METRICS = config["metrics"]

N_REPS = config["n_reps"]
SEPARABILITY = config["de_fscale"]

ruleorder: run_scdef_un > run_method

rule all:
    input:
        output_path + '/singlebatch_scores.csv'

rule gather_singlebatch_scores:
    conda:
        envs_path + "/scdef.yml"
    input:
        fname_list = expand(
            output_path + '/{method}/sep_{separability}/rep_{rep_id}_scores.csv',
            method=METHODS, separability=SEPARABILITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/singlebatch_scores.csv',
    script:
        scripts_path + '/gather_singlebatch_scores.py'

rule generate_singlebatch_data:
    conda:
        envs_path + "/splatter.yml"
    params:
        de_fscale = "{separability}",
        de_prob = config["de_prob"],
        batch_facscale = 0.,
        n_cells = config["n_cells"],
        n_batches = 1,
        frac_shared = 0.,
        seed = "{rep_id}",
        coverage = 0.,
    output:
        counts_fname = output_path + '/data/sep_{separability}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/sep_{separability}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/sep_{separability}/rep_{rep_id}_markers.csv',
        umap_fname = output_path + '/data/sep_{separability}/rep_{rep_id}_umap.png',
    script:
        scripts_path + '/splatter_hierarchical.R'

rule prepare_input:
    conda:
        envs_path + "/scdef.yml"
    params:
        seed = config['seed'],
    input:
        counts_fname = output_path + '/data/sep_{separability}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/sep_{separability}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/sep_{separability}/rep_{rep_id}_markers.csv',
    output:
        fname = output_path + '/data/sep_{separability}/rep_{rep_id}.h5ad'
    script:
        scripts_path + '/prepare_input.py'

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
        seed = "{rep_id}",
        store_full = False
    input:
        fname = output_path + '/data/sep_{separability}/rep_{rep_id}.h5ad'
    output:
        scores_fname = output_path + '/scDEF_un/sep_{separability}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_scdef_un.py"

rule run_method:
    conda:
        envs_path + "/{method}.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "{method}",
        n_top_genes = lambda wildcards: config[wildcards.method]['n_top_genes'],
        settings = lambda wildcards: config[wildcards.method]['settings'],
        store_full = False
    input:
        fname = output_path + '/data/sep_{separability}/rep_{rep_id}.h5ad'
    output:
        scores_fname = output_path + '/{method}/sep_{separability}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"
      
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
        fname = output_path + '/data/sep_{separability}/rep_{rep_id}.h5ad'
    output:
        scores_fname = output_path + '/PCA/sep_{separability}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

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
        fname = output_path + '/data/sep_{separability}/rep_{rep_id}.h5ad'
    output:
        scores_fname = output_path + '/NMF/sep_{separability}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

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
        fname = output_path + '/data/sep_{separability}/rep_{rep_id}.h5ad'
    output:
        scores_fname = output_path + '/scHPF/sep_{separability}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

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
        fname = output_path + '/data/sep_{separability}/rep_{rep_id}.h5ad'
    output:
        scores_fname = output_path + '/nSBM/sep_{separability}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

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
        fname = output_path + '/data/sep_{separability}/rep_{rep_id}.h5ad'
    output:
        scores_fname = output_path + '/fscLVM/sep_{separability}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"  