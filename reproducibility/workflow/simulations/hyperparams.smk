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
        'hyperparam_results/tau_scores.csv',
        'hyperparam_results/mu_scores.csv',
        'hyperparam_results/kappa_scores.csv',


rule gather_tau_mu_scores:
    conda:
        "../envs/PCA.yml"
    input:
        tau_fname_list = expand(
            output_path + '/tau/den_{density}/tau_{tau}/rep_{rep_id}_scores.csv',
            tau=TAU,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],),
        mu_fname_list = expand(
            output_path + '/mu/den_{density}/mu_{mu}/rep_{rep_id}_scores.csv',
            mu=MU,
            density=DENSITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        tau_output = output_path + '/tau_scores.csv',
        mu_output = output_path + '/mu_scores.csv',
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
            f'hyperparam_results/kappa/den_{DENSITY_DEFAULT}/' + 'kappa_{kappa}/rep_{rep_id}_scores.csv',
            kappa=KAPPA,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'hyperparam_results/kappa_scores.csv'
    run:
        # Output:
        # data_rep_id model_rep_id model_rep_id
        # for each rep, run one model with random sizes and another with fixed sizes, both with random init
        import pandas as pd

        rows = []
        for filename in snakemake.input['fname_list']:
            print(filename)

            # Parse filename
            l = filename.split("/")
            
            rep_id = l[-1].split("_")[1]
            method = l[-2] # method is the kappa

            # Parse scores
            df = pd.read_csv(filename, index_col=0)
            print(df)

            for idx, score in enumerate(df.index): # must have the ARI per layer
                value = df.values[idx]
                value = float(value[0])
                rows.append(
                    [
                        method,
                        rep_id,
                        score,
                        value,
                    ]
                )

        columns = [
            "method",
            "rep_id",
            "score",
            "value",
        ]

        scores = pd.DataFrame.from_records(rows, columns=columns)
        print(scores)

        scores = pd.melt(
            scores,
            id_vars=[
                "method",
                "rep_id",
                "score",
            ],
            value_vars="value",
        )
        scores.to_csv(outFileName, index=False)   

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
        "scripts/splatter_hierarchical.R"

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
        fname = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
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
        fname = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
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
        fname = output_path + '/data/den_{density}/rep_{rep_id}.h5ad',
    output:
        scores_fname = output_path + '/kappa/den_{density}/kappa_{kappa}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef.py'