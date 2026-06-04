"""
Hyperparameter ablation study.
Sweep hierarchy_weight, brd_strength, and brd_mean, each crossed with
de_prob (DE density) to measure sensitivity to prior choices.
"""

output_path = config.get("output_path", "results/hyperparams")

configfile: "config/hyperparams_config.yaml"
configfile: "config/methods.yaml"

N_REPS = config["n_reps"]
DENSITY = config["de_prob"]
DENSITY_DEFAULT = config["de_prob_default"]
HIERARCHY_WEIGHT = config["hierarchy_weight"]
BRD_STRENGTH = config["brd_strength"]
BRD_MEAN = config["brd_mean"]
METRICS = config["metrics"]

envs_path = "../envs"
scripts_path = "../scripts"

wildcard_constraints:
    rep_id = r"\d+",
    density = r"[^/]+",
    hw = r"[^/]+",
    brd = r"[^/]+",
    brdm = r"[^/]+",

rule all:
    input:
        output_path + '/hierarchy_weight_scores.csv',
        output_path + '/brd_strength_scores.csv',
        output_path + '/brd_mean_scores.csv',


# ---- Gather rules ----

rule gather_hw_scores:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    input:
        fname_list = expand(
            output_path + '/hierarchy_weight/den_{density}/hw_{hw}/rep_{rep_id}_scores.csv',
            hw=HIERARCHY_WEIGHT,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],),
    output:
        output_path + '/hierarchy_weight_scores.csv',
    params:
        param_name = ["density", "hierarchy_weight"],
        param_idx = [-3, -2],
        method_idx = -4,
    script:
        '../scripts/gather_scores.py'

rule gather_brd_strength_scores:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    input:
        fname_list = expand(
            output_path + '/brd_strength/den_{density}/brd_{brd}/rep_{rep_id}_scores.csv',
            brd=BRD_STRENGTH,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/brd_strength_scores.csv',
    params:
        param_name = ["density", "brd_strength"],
        param_idx = [-3, -2],
        method_idx = -4,
    script:
        '../scripts/gather_scores.py'

rule gather_brd_mean_scores:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    input:
        fname_list = expand(
            output_path + '/brd_mean/den_{density}/brdm_{brdm}/rep_{rep_id}_scores.csv',
            brdm=BRD_MEAN,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/brd_mean_scores.csv',
    params:
        param_name = ["density", "brd_mean"],
        param_idx = [-3, -2],
        method_idx = -4,
    script:
        '../scripts/gather_scores.py'


# ---- Data generation ----

rule generate_density_data:
    conda:
        envs_path + "/splatter.yml"
    params:
        de_fscale = config["de_fscale"],
        de_prob = "{density}",
        batch_facscale = config["batch_facscale"],
        n_cells = config["n_cells"],
        n_batches = config["n_batches"],
        frac_shared = config["frac_shared"],
        coverage = 0.,
        seed = "{rep_id}",
    output:
        counts_fname = output_path + '/data/den_{density}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/den_{density}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/den_{density}/rep_{rep_id}_markers.csv',
        umap_fname = output_path + '/data/den_{density}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = output_path + '/data/den_{density}/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"

rule prepare_input:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    params:
        seed = "{rep_id}",
    input:
        counts_fname = output_path + '/data/den_{density}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/den_{density}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/den_{density}/rep_{rep_id}_markers.csv',
    output:
        fname = output_path + '/data/den_{density}/rep_{rep_id}.h5ad'
    script:
        '../scripts/prepare_input.py'


# ---- hierarchy_weight sweep ----

rule run_scdef_hw:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        nmf_init = config['scDEF']['nmf_init'],
        brd_strength = config['scDEF']['brd_strength'],
        brd_mean = config['scDEF']['brd_mean'],
        hierarchy_weight = "{hw}",
        n_layers = config['scDEF']['n_layers'],
        n_factors = config['scDEF']['n_factors'],
        top_factors = config['scDEF']['top_factors'],
        pretraining = config['scDEF']['pretraining'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        seed = "{rep_id}",
    input:
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
    output:
        out_dir = directory(output_path + '/hierarchy_weight/den_{density}/hw_{hw}/rep_{rep_id}/'),
        duration_fname = output_path + '/hierarchy_weight/den_{density}/hw_{hw}/rep_{rep_id}_duration.txt',
    script:
        '../scripts/run_scdef.py'

rule evaluate_scdef_hw:
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
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
        model_state = output_path + '/hierarchy_weight/den_{density}/hw_{hw}/rep_{rep_id}/',
        duration_fname = output_path + '/hierarchy_weight/den_{density}/hw_{hw}/rep_{rep_id}_duration.txt',
    output:
        scores_fname = output_path + '/hierarchy_weight/den_{density}/hw_{hw}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/evaluate_results.py'


# ---- brd_strength sweep ----

rule run_scdef_brd_strength:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        nmf_init = config['scDEF']['nmf_init'],
        brd_strength = "{brd}",
        brd_mean = config['scDEF']['brd_mean'],
        hierarchy_weight = config['scDEF']['hierarchy_weight'],
        n_layers = config['scDEF']['n_layers'],
        n_factors = config['scDEF']['n_factors'],
        top_factors = config['scDEF']['top_factors'],
        pretraining = config['scDEF']['pretraining'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        seed = "{rep_id}",
    input:
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
    output:
        out_dir = directory(output_path + '/brd_strength/den_{density}/brd_{brd}/rep_{rep_id}/'),
        duration_fname = output_path + '/brd_strength/den_{density}/brd_{brd}/rep_{rep_id}_duration.txt',
    script:
        '../scripts/run_scdef.py'

rule evaluate_scdef_brd_strength:
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
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
        model_state = output_path + '/brd_strength/den_{density}/brd_{brd}/rep_{rep_id}/',
        duration_fname = output_path + '/brd_strength/den_{density}/brd_{brd}/rep_{rep_id}_duration.txt',
    output:
        scores_fname = output_path + '/brd_strength/den_{density}/brd_{brd}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/evaluate_results.py'


# ---- brd_mean sweep ----

rule run_scdef_brd_mean:
    conda:
        envs_path + "/scdef_reproducibility.yml"
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        nmf_init = config['scDEF']['nmf_init'],
        brd_strength = config['scDEF']['brd_strength'],
        brd_mean = "{brdm}",
        hierarchy_weight = config['scDEF']['hierarchy_weight'],
        n_layers = config['scDEF']['n_layers'],
        n_factors = config['scDEF']['n_factors'],
        top_factors = config['scDEF']['top_factors'],
        pretraining = config['scDEF']['pretraining'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        seed = "{rep_id}",
    input:
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
    output:
        out_dir = directory(output_path + '/brd_mean/den_{density}/brdm_{brdm}/rep_{rep_id}/'),
        duration_fname = output_path + '/brd_mean/den_{density}/brdm_{brdm}/rep_{rep_id}_duration.txt',
    script:
        '../scripts/run_scdef.py'

rule evaluate_scdef_brd_mean:
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
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
        model_state = output_path + '/brd_mean/den_{density}/brdm_{brdm}/rep_{rep_id}/',
        duration_fname = output_path + '/brd_mean/den_{density}/brdm_{brdm}/rep_{rep_id}_duration.txt',
    output:
        scores_fname = output_path + '/brd_mean/den_{density}/brdm_{brdm}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/evaluate_results.py'
