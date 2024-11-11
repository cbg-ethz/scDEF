configfile: "config/benchmark_multibatch.yaml"
configfile: "config/methods.yaml"

output_path = "results/multibatch"
scripts_path = "scripts"
envs_path = "../../../envs"
methods_scripts_path = "../../../scripts"

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
        envs_path + "/scdef.yml"
    input:
        fname_list = expand(
            output_path + '/{method}/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
            method=METHODS,
            frac_shared=FRACS_SHARED, separability=SEPARABILITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/multibatch_scores.csv'
    script:
        scripts_path + '/gather_multibatch_scores.py'

rule generate_multibatch_data:
    conda:
        envs_path + "/splatter.yml"
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
        scripts_path + "/splatter_hierarchical.R"


rule prepare_input:
    conda:
        envs_path + "/scdef.yml"
    params:
        seed = config['seed'],
    input:
        counts_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad'
    script:
        scripts_path + '/prepare_input.py'


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
        seed = "{rep_id}",
        store_full = False
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/scDEF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_scdef.py"


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
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/scDEF_un/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_scdef_un.py"      

rule run_pca:
    conda:
        envs_path + "/PCA.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "PCA",
        n_top_genes = config["PCA"]['n_top_genes'],
        settings = config["PCA"]['settings'],
        store_full = True
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/PCA/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

rule run_nmf:
    conda:
        envs_path + "/NMF.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "NMF",
        n_top_genes = config["NMF"]['n_top_genes'],
        settings = config["NMF"]['settings'],
        store_full = True
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/NMF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

rule run_schpf:
    conda:
        envs_path + "/scHPF.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "scHPF",
        n_top_genes = config["scHPF"]['n_top_genes'],
        settings = config["scHPF"]['settings'],
        store_full = True
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/scHPF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

rule run_scvi:
    conda:
        envs_path + "/scVI.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "scVI",
        n_top_genes = config["scVI"]['n_top_genes'],
        settings = config["scVI"]['settings'],
        store_full = True
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/scVI/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

rule run_harmony:
    conda:
        envs_path + "/Harmony.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "Harmony",
        n_top_genes = config["Harmony"]['n_top_genes'],
        settings = config["Harmony"]['settings'],
        store_full = True
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/Harmony/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

rule run_scanorama:
    conda:
        envs_path + "/Scanorama.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "Scanorama",
        n_top_genes = config["Scanorama"]['n_top_genes'],
        settings = config["Scanorama"]['settings'],
        store_full = True
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/Scanorama/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

rule run_nsbm:
    conda:
        envs_path + "/nSBM.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "nSBM",
        n_top_genes = config["nSBM"]['n_top_genes'],
        settings = config["nSBM"]['settings'],
        store_full = True
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/nSBM/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"

rule run_fsclvm:
    conda:
        envs_path + "/fscLVM.yml"        
    params:
        metrics = METRICS,    
        seed = "{rep_id}",
        method = "fscLVM",
        n_top_genes = config["fscLVM"]['n_top_genes'],
        settings = config["fscLVM"]['settings'],
        store_full = True
    input:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/fscLVM/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"     