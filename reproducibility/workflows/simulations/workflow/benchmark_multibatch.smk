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

ruleorder: run_scdef_un > run_scdef > run_method

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
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/{method}/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        methods_scripts_path + "/run_method.py"
      