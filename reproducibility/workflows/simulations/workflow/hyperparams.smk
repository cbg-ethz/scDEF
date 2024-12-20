"""
1. Generate data with different levels of DE genes and check if we still find the cell types: cell type ARI for different $\tau$ and different $\mu$, for the different scDEF layers
2. Generate data with different levels of coverage and check if we still find the cell types: hierarchy accuracy for different $\kappa$
"""

configfile: "config/hyperparams_config.yaml"

N_REPS = config["n_reps"]
DENSITY = config["de_prob"]
COVERAGE = config["coverage"]
TAU = config["tau"]
MU = config["mu"]
KAPPA = config["kappa"]


rule all:
    input:
        'hyperparam_results/tau_scores.csv',
        'hyperparam_results/mu_scores.csv',
        'hyperparam_results/kappa_scores.csv',


rule gather_tau_mu_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        tau_fname_list = expand(
            'hyperparam_results/tau/den_{density}/tau_{tau}/rep_{rep_id}_scores.csv',
            tau=TAU,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],),
        mu_fname_list = expand(
            'hyperparam_results/mu/den_{density}/mu_{mu}/rep_{rep_id}_scores.csv',
            mu=MU,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        tau_output = 'hyperparam_results/tau_scores.csv',
        mu_output = 'hyperparam_results/mu_scores.csv',
    run:
        import pandas as pd
        import numpy as np
        methods_files = dict(zip(["tau", "mu"], [input['tau_fname_list'], input['mu_fname_list']]))
        for method in methods_files:
            rows = []
            for filename in methods_files[method]:
                print(filename)

                # Parse filename
                l = filename.split("/")
                
                rep_id = l[-1].split("_")[1]
                method_val = l[-2].split("_")[1]
                density = l[-3].split("_")[1]

                # Parse scores
                df = pd.read_csv(filename, index_col=0)
                print(df)

                for idx, score in enumerate(df.index): # must have the ARI per layer
                    value = df.values[idx]
                    value = value[0]
                    if isinstance(value, str):
                        value = np.mean(np.array(value.strip("][").split(", ")).astype(float))
                    value = float(value)
                    rows.append(
                        [
                            rep_id,
                            method_val,
                            density,
                            score,
                            value,
                        ]
                    )

            columns = [
                "rep_id",
                f"{method}",
                "density",
                "score",
                "value",
            ]

            scores = pd.DataFrame.from_records(rows, columns=columns)
            print(scores)

            if method == "tau":
                scores.to_csv(output['tau_output'], index=False)   
            else:
                scores.to_csv(output['mu_output'], index=False)   


rule gather_kappa_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'hyperparam_results/kappa/cov_{coverage}/kappa_{kappa}/rep_{rep_id}_scores.csv',
            kappa=KAPPA,
            coverage=COVERAGE,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'hyperparam_results/kappa_scores.csv'
    run:     
        import pandas as pd

        rows = []
        for filename in snakemake.input['fname_list']:
            print(filename)

            # Parse filename
            l = filename.split("/")
            
            rep_id = l[-1].split("_")[1]
            kappa = l[-2].split("_")[1]
            coverage = l[-3].split("_")[1]

            # Parse scores
            df = pd.read_csv(filename, index_col=0)
            print(df)

            for idx, score in enumerate(df.index): # must have the ARI per layer
                value = df.values[idx]
                value = float(value[0])
                rows.append(
                    [
                        rep_id,
                        kappa,
                        coverage,
                        score,
                        value,
                    ]
                )

        columns = [
            "rep_id",
            "kappa",
            "coverage",
            "score",
            "value",
        ]

        scores = pd.DataFrame.from_records(rows, columns=columns)
        print(scores)

        scores = pd.melt(
            scores,
            id_vars=[
                "kappa",
                "coverage",
                "rep_id",
                "score",
            ],
            value_vars="value",
        )
        scores.to_csv(snakemake.output, index=False)       

rule generate_density_data:
    resources:
        mem_per_cpu=10000,
        threads=10,
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
        counts_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_markers.csv',
        umap_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_umap_nobatch.png',
    script:
        "scripts/splatter_hierarchical.R"

rule generate_coverage_data:
    resources:
        mem_per_cpu=10000,
        threads=10,
    params:
        de_fscale = config["de_fscale"],
        de_prob = config["de_prob_default"],
        batch_facscale = config["batch_facscale"],
        n_cells = config["n_cells"],
        n_batches = config["n_batches"],
        frac_shared = config["frac_shared"],
        coverage = "{coverage}",
        seed = "{rep_id}",
    output:
        counts_fname = 'hyperparam_results/data/cov_{coverage}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/cov_{coverage}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/cov_{coverage}/rep_{rep_id}_markers.csv',
        umap_fname = 'hyperparam_results/data/cov_{coverage}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = 'hyperparam_results/data/cov_{coverage}/rep_{rep_id}_umap_nobatch.png',
    script:
        "scripts/splatter_hierarchical.R"

rule run_scdef_tau:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        tau = "{tau}",
        mu = config['default_mu'],
        kappa = config['default_kappa'],
        layer_sizes = config['layer_sizes'],
    input:
        counts_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'hyperparam_results/tau/den_{density}/tau_{tau}/rep_{rep_id}_scores.csv',
    script:
        "scripts/run_scdef.py"

rule run_scdef_mu:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        tau = config['default_tau'],
        mu = "{mu}",
        kappa = config['default_kappa'],
        layer_sizes = config['layer_sizes'],
    input:
        counts_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/den_{density}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'hyperparam_results/mu/den_{density}/mu_{mu}/rep_{rep_id}_scores.csv',
    script:
        "scripts/run_scdef.py"

rule run_scdef_kappa:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        tau = config['default_tau'],
        mu = config['default_mu'],
        kappa = "{kappa}",
        layer_sizes = config['layer_sizes'],
    input:
        counts_fname = 'hyperparam_results/data/cov_{coverage}/rep_{rep_id}_counts.csv',
        meta_fname = 'hyperparam_results/data/cov_{coverage}/rep_{rep_id}_meta.csv',
        markers_fname = 'hyperparam_results/data/cov_{coverage}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'hyperparam_results/kappa/cov_{coverage}/kappa_{kappa}/rep_{rep_id}_scores.csv',
    script:
        "scripts/run_scdef.py"        