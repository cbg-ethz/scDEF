# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

localrules: gather_results

configfile: "config/visium.yaml"
configfile: "config/methods.yaml"

output_path = "results/visium"

SEED = config["seed"]
METRICS = config["metrics"]

include: "run_methods.smk"

rule all:
    input:
        output_path + '/scDEF/scDEF.pkl'

rule prepare_input:
    conda:
        "../../../envs/squidpy.yml"
    params:
        data_path = config['data_path'],
        data_fname = config['data_fname'],
        seed = config['seed'],
        genes_to_remove = config['genes_to_remove'],
        n_top_genes = config['n_top_genes'],
    output:
        fname = output_path + '/prepared_input.h5ad'
    script:
        'scripts/prepare_visium.py'
