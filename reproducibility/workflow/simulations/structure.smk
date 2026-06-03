"""
Structure ablation study.
A: Learn scDEF with varying n_layers and compute metrics per layer.
B: Learn scDEF with varying n_factors and compute all metrics.
"""

output_path = config.get("output_path", "structure_results")

configfile: "config/structure_config.yaml"
configfile: "config/methods.yaml"

N_REPS = config["n_reps"]
LAYERS = config["n_layers"]
FACTORS = config["n_factors"]
METRICS = config["metrics"]

envs_path = "../envs"
scripts_path = "../scripts"

wildcard_constraints:
    rep_id = r"\d+",
    n_layers = r"\d+",
    n_factors = r"\d+",

rule all:
    input:
        output_path + '/layers_scores.csv',
        output_path + '/factors_scores.csv'

rule gather_layers_scores:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    input:
        fname_list = expand(
            output_path + '/scDEF/layers_{n_layers}/rep_{rep_id}_scores.csv',
            n_layers=LAYERS,
            rep_id=[r for r in range(N_REPS)],)
    params:
        param_name = ["n_layers"],
        param_idx = [-2],
        method_idx = -3,
    output:
        output_path + '/layers_scores.csv'
    script:
        '../scripts/gather_scores.py'

rule gather_factors_scores:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    input:
        fname_list = expand(
            output_path + '/scDEF/factors_{n_factors}/rep_{rep_id}_scores.csv',
            n_factors=FACTORS,
            rep_id=[r for r in range(N_REPS)],)
    params:
        param_name = ["n_factors"],
        param_idx = [-2],
        method_idx = -3,
    output:
        output_path + '/factors_scores.csv'
    script:
        '../scripts/gather_scores.py'


rule generate_data:
    conda:
        envs_path + "/splatter.yml"
    params:
        de_fscale = config["de_fscale"],
        de_prob = config["de_prob"],
        batch_facscale = config["batch_facscale"],
        n_cells = config["n_cells"],
        n_batches = config["n_batches"],
        frac_shared = config["frac_shared"],
        coverage = 0.,
        seed = "{rep_id}",
    output:
        counts_fname = output_path + '/data/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/rep_{rep_id}_markers.csv',
        umap_fname = output_path + '/data/rep_{rep_id}_umap.png',
        umap_nobatch_fname = output_path + '/data/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"

rule prepare_input:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    params:
        seed = "{rep_id}",
    input:
        counts_fname = output_path + '/data/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/rep_{rep_id}_markers.csv',
    output:
        fname = output_path + '/data/rep_{rep_id}.h5ad'
    script:
        '../scripts/prepare_input.py'


# ---- n_layers sweep ----

rule run_scdef_layers:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        n_layers = "{n_layers}",
        n_factors = config['scDEF']['n_factors'],
        nmf_init = config['scDEF']['nmf_init'],
        pretraining = config['scDEF']['pretraining'],
        brd_strength = config['scDEF']['brd_strength'],
        brd_mean = config['scDEF']['brd_mean'],
        hierarchy_weight = config['scDEF']['hierarchy_weight'],
        top_factors = config['scDEF']['top_factors'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        seed = "{rep_id}",
    input:
        adata = output_path + '/data/rep_{rep_id}.h5ad',
    output:
        out_dir = directory(output_path + '/scDEF/layers_{n_layers}/rep_{rep_id}/'),
        duration_fname = output_path + '/scDEF/layers_{n_layers}/rep_{rep_id}_duration.txt',
    script:
        '../scripts/run_scdef.py'

rule evaluate_scdef_layers:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "scDEF",
        metrics = METRICS,
        is_scdef = True,
    input:
        adata = output_path + '/data/rep_{rep_id}.h5ad',
        model_state = output_path + '/scDEF/layers_{n_layers}/rep_{rep_id}/',
        duration_fname = output_path + '/scDEF/layers_{n_layers}/rep_{rep_id}_duration.txt',
    output:
        scores_fname = output_path + '/scDEF/layers_{n_layers}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/evaluate_results.py'


# ---- n_factors sweep ----

rule run_scdef_factors:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        n_layers = config['scDEF']['n_layers'],
        n_factors = "{n_factors}",
        nmf_init = config['scDEF']['nmf_init'],
        pretraining = config['scDEF']['pretraining'],
        brd_strength = config['scDEF']['brd_strength'],
        brd_mean = config['scDEF']['brd_mean'],
        hierarchy_weight = config['scDEF']['hierarchy_weight'],
        top_factors = config['scDEF']['top_factors'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        seed = "{rep_id}",
    input:
        adata = output_path + '/data/rep_{rep_id}.h5ad',
    output:
        out_dir = directory(output_path + '/scDEF/factors_{n_factors}/rep_{rep_id}/'),
        duration_fname = output_path + '/scDEF/factors_{n_factors}/rep_{rep_id}_duration.txt',
    script:
        '../scripts/run_scdef.py'

rule evaluate_scdef_factors:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "scDEF",
        metrics = METRICS,
        is_scdef = True,
    input:
        adata = output_path + '/data/rep_{rep_id}.h5ad',
        model_state = output_path + '/scDEF/factors_{n_factors}/rep_{rep_id}/',
        duration_fname = output_path + '/scDEF/factors_{n_factors}/rep_{rep_id}_duration.txt',
    output:
        scores_fname = output_path + '/scDEF/factors_{n_factors}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/evaluate_results.py'
