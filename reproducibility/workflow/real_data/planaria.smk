# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

output_path = "results/planaria"

localrules: gather_results

configfile: "config/planaria.yaml"
configfile: "config/methods.yaml"

METHODS = config["methods"]
METRICS = config["metrics"]
N_SEEDS = config.get("n_seeds", 1)
SEEDS = list(range(N_SEEDS))

wildcard_constraints:
    seed = r"\d+",

RUN_SUFFIX = "seed_{seed}"
INPUT_ADATA = output_path + '/prepared_input.h5ad'
METHOD_SEED = "{seed}"

include: "../run_methods.smk"

rule all:
    input:
        output_path + '/scores.csv'

rule gather_results:
    conda:
        "../envs/scdef_reproducibility.yml"
    input:
        fname_list = expand(
            output_path + '/{method}/seed_{seed}_scores.csv',
            method=METHODS, seed=SEEDS,)
    output:
        fname = output_path + '/scores.csv'
    script:
        '../scripts/gather_real_data_scores.py'

rule prepare_input:
    conda:
        "../envs/scdef_reproducibility.yml"
    params:
        data_path = config['data_path'],
        markers_fname = config['markers_path'],
        gene_names_fname = config['gene_names_path'],
        seed = config['seed'],
        genes_to_remove = config['genes_to_remove'],
        n_top_genes = config['n_top_genes'],
        min_genes = config['min_genes'],
        min_cells = config['min_cells'],
    output:
        fname = output_path + '/prepared_input.h5ad'
    script:
        '../scripts/prepare_planaria.py'
