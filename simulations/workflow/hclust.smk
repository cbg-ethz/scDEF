"""
Run hierarchical clustering on 1-layer scDEF, and on Leiden at a high resolution. Compare both the cell type ARI and the hierarchy accuracy to their other versions. 
"""

configfile: "config/config.yaml"

N_REPS = config["n_reps"]
SEPARABILITY = config["de_fscale"]
DENSITY = config["de_prob"]
COVERAGE = config["coverage"]
FRACS_SHARED = config["frac_shared"]
TAU = config["tau"]
MU = config["mu"]
KAPPA = config["kappa"]


rule all:
    input:
        'hyperparam_results/hyperparam_figure.pdf'

rule gather_tau_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'hyperparam_results/tau/den_{density}/tau_{tau}/rep_{rep_id}_scores.csv',
            tau=TAU,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'hyperparam_results/tau_scores.csv'
    script:
        '../scripts/gather_hyperparam_scores.py'


rule gather_mu_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'hyperparam_results/mu/den_{density}/mu_{mu}/rep_{rep_id}_scores.csv',
            mu=MU,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'hyperparam_results/mu_scores.csv'
    script:
        '../scripts/gather_hyperparam_scores.py'


rule gather_kappa_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'hyperparam_results/kappa/coverage_{coverage}/kappa_{mu}/rep_{rep_id}_scores.csv',
            kappa=KAPPA,
            coverage=COVERAGE,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'hyperparam_results/kappa_scores.csv'
    script:
        '../scripts/gather_hyperparam_scores.py'        

rule generate_density_data:
    resources:
        mem_per_cpu=10000,
        threads=10,
    params:
        de_fscale = "{separability}",
        de_prob = "{density}",
        batch_facscale = config["batch_facscale"],
        n_cells = 1000,
        n_batches = config["n_batches"],
        frac_shared = "{frac_shared}",
        seed = "{rep_id}",
    output:
        counts_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
        umap_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"

rule generate_coverage_data:
    resources:
        mem_per_cpu=10000,
        threads=10,
    params:
        de_fscale = "{separability}",
        de_prob = "{density}",
        batch_facscale = config["batch_facscale"],
        n_cells = 1000,
        n_batches = config["n_batches"],
        frac_shared = "{frac_shared}",
        seed = "{rep_id}",
    output:
        counts_fname = 'hyperparam_results/data/cov_{coverage}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/cov_{coverage}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/cov_{coverage}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
        umap_fname = 'hyperparam_results/data/cov_{coverage}/shared_{frac_shared}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = 'hyperparam_results/data/cov_{coverage}/shared_{frac_shared}/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"

rule run_scdef_tau:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        tau = "{tau}",
        mu = "{mu}",
        kappa = "{kappa}",
    input:
        counts_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'hyperparam_results/scDEF/den_{density}/shared_{frac_shared}/tau_{tau}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef.py"

rule run_scdef_mu:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        tau = "{tau}",
        mu = "{mu}",
        kappa = "{kappa}",
    input:
        counts_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/den_{density}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'hyperparam_results/scDEF/den_{density}/shared_{frac_shared}/mu_{mu}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef.py"

rule run_scdef_kappa:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        tau = "{tau}",
        mu = "{mu}",
        kappa = "{kappa}",
    input:
        counts_fname = 'hyperparam_results/data/cov_{coverage}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/cov_{coverage}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/cov_{coverage}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'hyperparam_results/scDEF/cov_{coverage}/shared_{frac_shared}/kappa_{kappa}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef.py"        