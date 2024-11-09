# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

configfile: "config/hgsoc_chemo.yaml"
configfile: "config/methods.yaml"

output_path = "results/hgsoc_chemo"

METHODS = config["methods"]
SEED = config["seed"]
METRICS = config["metrics"]

include: "rules/run_methods.smk"

rule all:
    input:
        output_path + '/scores.csv'

rule gather_results:
    conda:
        "../../../envs/scdef.yml"
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
        "../../../envs/scdef.yml"
    params:
        data_fname = config['data_path'],
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
        'scripts/prepare_hgsoc_chemo.py'
