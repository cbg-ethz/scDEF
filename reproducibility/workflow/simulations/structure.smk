"""
Generate hierarchical data from 4 batches and 3 layers.
A Learn scDEF with 1, 2, 3, and 4 layers and compute ARI wrt to each scDEF layer and each true layer
B Learn scDEF with 4 layers of sizes 100, 60, 30, 10, 1 and 100, 50, 30, 10, 1 and 100, 60, 20, 10, 1 and compute all metrics
"""

output_path = "structure_results"

configfile: "config/structure_config.yaml"
configfile: "config/methods.yaml"

N_REPS = config["n_reps"]
LAYERS = config["n_layers"]
DECAY_FACTORS = config["decay_factors"]
METRICS = config["metrics"]


rule all:
    input:
        output_path + '/layers_scores.csv',
        output_path + '/factors_scores.csv'

rule gather_layers_scores:
    conda:
        "../envs/PCA.yml"
    input:
        fname_list = expand(
            output_path + '/{n_layers}_layers/rep_{rep_id}_scores.csv',
            n_layers=LAYERS,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/layers_scores.csv'
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
        scores.to_csv(snakemake.output[0], index=False)        

rule gather_factors_scores:
    conda:
        "../envs/PCA.yml"
    input:
        fname_list = expand(
            output_path + '/decay_{decay_factor}/rep_{rep_id}_scores.csv',
            decay_factor=DECAY_FACTORS,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/factors_scores.csv'
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
        scores.to_csv(snakemake.output[0], index=False)    



rule generate_data:
    conda:
        "../envs/splatter.yml"
    params:
        de_fscale = config["de_fscale"],
        de_prob = config["de_prob"],
        batch_facscale = config["batch_facscale"],
        n_cells = config["n_cells"],
        n_batches = config["n_batches"],
        frac_shared = config["frac_shared"],
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
        "../envs/PCA.yml"
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

rule run_scdef_layers:
    conda:
        "../envs/scdef.yml"
    params:
        n_layers = "{n_layers}",
        decay_factor = config['scDEF']['decay_factor'],
        kappa = config['scDEF']['kappa'],
        n_factors = config['scDEF']['n_factors'],
        nmf_init = config['scDEF']['nmf_init'],
        pretrain = config['scDEF']['pretrain'],
        tau = config['scDEF']['tau'],
        mu = config['scDEF']['mu'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        metrics = METRICS,
        seed = "{rep_id}",
        store_full = True
    input:
        adata = output_path + '/data/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/{n_layers}_layers/rep_{rep_id}.pkl',
        scores_fname = output_path + '/{n_layers}_layers/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef.py'


rule run_scdef_factors:
    conda:
        "../envs/scdef.yml"
    params:
        n_layers = config['scDEF']['n_layers'],
        decay_factor = "{decay_factor}",
        kappa = config['scDEF']['kappa'],        
        n_factors = config['scDEF']['n_factors'],
        nmf_init = config['scDEF']['nmf_init'],
        pretrain = config['scDEF']['pretrain'],
        tau = config['scDEF']['tau'],
        mu = config['scDEF']['mu'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        metrics = METRICS,
        seed = "{rep_id}",
        store_full = True
    input:
        adata = output_path + '/data/rep_{rep_id}.h5ad',
    output:
        out_fname = output_path + '/decay_{decay_factor}/rep_{rep_id}.pkl',
        scores_fname = output_path + '/decay_{decay_factor}/rep_{rep_id}_scores.csv',
    script:
        '../scripts/run_scdef.py'
