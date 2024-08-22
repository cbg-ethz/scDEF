"""
Generate hierarchical data from 4 batches and 3 layers.
A Learn scDEF with 1, 2, 3, and 4 layers and compute ARI wrt to each scDEF layer and each true layer
B Learn scDEF with 4 layers of sizes 100, 60, 30, 10, 1 and 100, 50, 30, 10, 1 and 100, 60, 20, 10, 1 and compute all metrics
"""

configfile: "config/structure_config.yaml"

N_REPS = config["n_reps"]
LAYERS = config["n_layers"]
layer_sizes = config["layer_sizes"]


rule all:
    input:
        'structure_results/layers_scores.csv',
        'structure_results/factors_scores.csv'

rule gather_layers_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'structure_results/{n_layers}_layers/rep_{rep_id}_scores.csv',
            n_layers=LAYERS,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'structure_results/layers_scores.csv'
    run:
        # Output:
        # data_rep_id, model1layer_layer1ari, model2layer_layer1ari, model3layer_layer1ari
        import pandas as pd

        rows = []
        for filename in snakemake.input['fname_list']:
            print(filename)

            # Parse filename
            l = filename.split("/")
            
            rep_id = l[-1].split("_")[1]
            method = l[-2] # method is the number of layers

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

rule gather_factors_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list_var = expand(
            'structure_results/factors/scDEF_var/rep_{rep_id}_scores.csv',
            rep_id=[r for r in range(N_REPS)],),
        fname_list_fixed = expand(
            'structure_results/factors/scDEF_fixed/rep_{rep_id}_scores.csv',
            rep_id=[r for r in range(N_REPS)],)
    output:
        'structure_results/factors_scores.csv'
    run:
        # Output:
        # data_rep_id model_rep_id model_rep_id
        # for each rep, run one model with random sizes and another with fixed sizes, both with random init
        import pandas as pd

        scdef_var_results = snakemake.input["fname_list_var"]
        scdef_fixed_results = snakemake.input["fname_list_fixed"]
        methods_files = dict(zip(['scDEF_var', 'scDEF_fixed'], [scdef_var_results, scdef_fixed_results]))

        rows = []
        for method in methods_files:
            for filename in methods_files[method]:
                print(filename)

                # Parse filename
                l = filename.split("/")
                
                rep_id = l[-1].split("_")[1]

                # Parse scores
                df = pd.read_csv(filename, index_col=0)
                print(df)

                for idx, score in enumerate(df.index):
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



rule generate_data:
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
        seed = "{rep_id}",
    output:
        counts_fname = 'structure_results/data/rep_{rep_id}_counts.csv',
        meta_fname = 'structure_results/data/rep_{rep_id}_meta.csv',
        markers_fname = 'structure_results/data/rep_{rep_id}_markers.csv',
        umap_fname = 'structure_results/data/rep_{rep_id}_umap.png',
        umap_nobatch_fname = 'structure_results/data/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"

rule run_scdef_layers:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        n_layers = "{n_layers}",
        seed = "{rep_id}",
        tau = config['default_tau'],
        mu = config['default_mu'],
        kappa = config['default_kappa'],
        layer_sizes = config['layer_sizes'],
        factors_var = 0,
    input:
        counts_fname = 'structure_results/data/rep_{rep_id}_counts.csv',
        meta_fname = 'structure_results/data/rep_{rep_id}_meta.csv',
        markers_fname = 'structure_results/data/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'structure_results/{n_layers}_layers/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef_factors_var.py"

rule run_scdef_factors_var:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        n_layers = len(config['layer_sizes']),
        seed = "{rep_id}",
        tau = config['default_tau'],
        mu = config['default_mu'],
        kappa = config['default_kappa'],
        layer_sizes = config['layer_sizes'],
        factors_var = config['factors_var'],
    input:
        counts_fname = 'structure_results/data/rep_{rep_id}_counts.csv',
        meta_fname = 'structure_results/data/rep_{rep_id}_meta.csv',
        markers_fname = 'structure_results/data/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'structure_results/factors/scDEF_var/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef_factors_var.py"

rule run_scdef_factors_fixed:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        tau = config['default_tau'],
        mu = config['default_mu'],
        kappa = config['default_kappa'],
        layer_sizes = config['layer_sizes'],
    input:
        counts_fname = 'structure_results/data/rep_{rep_id}_counts.csv',
        meta_fname = 'structure_results/data/rep_{rep_id}_meta.csv',
        markers_fname = 'structure_results/data/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'structure_results/factors/scDEF_fixed/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef.py"        