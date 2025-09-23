# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

output_path = "results/pbmcsifn"

localrules: gather_results

configfile: "config/pbmcsifn.yaml"
configfile: "config/methods.yaml"

METHODS = config["methods"]
SEED = config["seed"]
METRICS = config["metrics"]

include: "run_methods.smk"

rule all:
    input:
        output_path + '/scores.csv'
rule gather_results:
    conda:
        "../envs/PCA.yml"
    input:
        fname_list = expand(
            output_path + '/{method}/{method}.csv',
            method=METHODS,)
    output:
        fname = output_path + '/scores.csv'
    script:
        'scripts/gather_scores.py'

rule prepare_input:
    conda:
        "../envs/PCA.yml"
    params:
        data_fname = config['data_path'],
        seed = config['seed'],
        genes_to_remove = config['genes_to_remove'],
        n_top_genes = config['n_top_genes'],
        min_genes = config['min_genes'],
        min_cells = config['min_cells'],
    output:
        fname = output_path + '/prepared_input.h5ad'
    script:
        '../scripts/prepare_pbmcsifn.py'