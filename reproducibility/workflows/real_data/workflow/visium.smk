# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

configfile: "config/visium.yaml"
output_path = "results/visium"

SEED = config["seed"]

include: "rules/run_methods.smk"

rule all:
    input:
        output_path + '/scDEF.pkl'

rule prepare_input:
    params:
        data_fname = config['data_path'],
        annotations_fname = config['annotations_path'],
        seed = config['seed'],
        genes_to_remove = config['genes_to_remove'],
        n_top_genes = config['n_top_genes'],
    output:
        fname = output_path + '/prepared_input.h5ad'
    script:
        'scripts/prepare_visium.py'
