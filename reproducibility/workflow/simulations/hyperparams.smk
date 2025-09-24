"""
1. Generate data with different levels of DE genes and check if we still find the cell types: cell type ARI for different $\tau$ and different $\mu$, for the different scDEF layers
2. Generate data with different levels of coverage and check if we still find the cell types: hierarchy accuracy for different $\kappa$
"""

output_path = "hyperparam_results"

configfile: "config/hyperparams_config.yaml"
configfile: "config/methods.yaml"

N_REPS = config["n_reps"]
DENSITY = config["de_prob"]
DENSITY_DEFAULT = config["de_prob_default"]
TAU = config["tau"]
MU = config["mu"]
KAPPA = config["kappa"]
METRICS = config["metrics"]

rule all:
    input:
        output_path + '/tau_scores.csv',
        output_path + '/mu_scores.csv',
        output_path + '/kappa_scores.csv',


rule gather_tau_scores:
    conda:
        "../envs/PCA.yml"
    input:
        fname_list = expand(
            output_path + '/tau/den_{density}/tau_{tau}/rep_{rep_id}_scores.csv',
            tau=TAU,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],),
    output:
        output_path + '/tau_scores.csv',
    params:
        param_name = ["density", "tau"],
        param_idx = [-3, -2],
        method_idx = -4,
    script:
        '../scripts/gather_scores.py'

rule gather_mu_scores:
    conda:
        "../envs/PCA.yml"
    input:
        fname_list = expand(
            output_path + '/mu/den_{density}/mu_{mu}/rep_{rep_id}_scores.csv',
            mu=MU,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/mu_scores.csv',
    params:
        param_name = ["density", "mu"],
        param_idx = [-3, -2],
        method_idx = -4,
    script:
        '../scripts/gather_scores.py'

rule gather_kappa_scores:
    conda:
        "../envs/PCA.yml"
    input:
        fname_list = expand(
            output_path + '/kappa/den_{density}/kappa_{kappa}/rep_{rep_id}_scores.csv',
            density=DENSITY,
            kappa=KAPPA,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/kappa_scores.csv',
    params:
        param_name = ["density", "kappa"],
        param_idx = [-3, -2],
        method_idx = -4,
    script:
        '../scripts/gather_scores.py'

rule generate_density_data:
    conda:
        "../envs/splatter.yml"
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
        "../envs/PCA.yml"
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

rule run_scdef_tau:
    conda:
        "../envs/scdef.yml"
    params:
        nmf_init = config['scDEF']['nmf_init'],
        tau = "{tau}",
        mu = config['scDEF']['mu'],
        kappa = config['scDEF']['kappa'],
        n_layers = config['scDEF']['n_layers'],
        n_factors = config['scDEF']['n_factors'],
        decay_factor = config['scDEF']['decay_factor'],
        pretrain = config['scDEF']['pretrain'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        metrics = METRICS,
        seed = "{rep_id}",
        store_full = False
    input:
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/tau/den_{density}/tau_{tau}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef.py'

rule run_scdef_mu:
    conda:
        "../envs/scdef.yml"
    params:
        nmf_init = config['scDEF']['nmf_init'],
        tau = config['scDEF']['tau'],
        mu = "{mu}",
        kappa = config['scDEF']['kappa'],
        n_layers = config['scDEF']['n_layers'],
        n_factors = config['scDEF']['n_factors'],
        decay_factor = config['scDEF']['decay_factor'],
        pretrain = config['scDEF']['pretrain'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        metrics = METRICS,
        seed = "{rep_id}",
        store_full = False
    input:
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/mu/den_{density}/mu_{mu}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef.py'

rule run_scdef_kappa:
    conda:
        "../envs/scdef.yml"
    params:
        nmf_init = config['scDEF']['nmf_init'],
        tau = config['scDEF']['tau'],
        mu = config['scDEF']['mu'],
        kappa = "{kappa}",
        n_layers = config['scDEF']['n_layers'],
        n_factors = config['scDEF']['n_factors'],
        decay_factor = config['scDEF']['decay_factor'],
        pretrain = config['scDEF']['pretrain'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        metrics = METRICS,
        seed = "{rep_id}",
        store_full = False
    input:
        adata = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/kappa/den_{density}/kappa_{kappa}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef.py'