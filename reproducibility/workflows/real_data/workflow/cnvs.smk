# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

localrules: gather_results

configfile: "config/cnvs.yaml"
configfile: "config/methods.yaml"

output_path = "results/cnvs"

METHODS = config["methods"]
SEED = config["seed"]
METRICS = config["metrics"]

include: "rules/run_methods.smk"

rule all:
    input:
        output_path + '/scores.csv'

rule gather_results:
    conda:
        "../../../envs/PCA.yml"
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
        "../../../envs/infercnv.yml"
    params:
        data_fname = config['data_path'],
        annotations_fname = config['annotations_path'],
        seed = config['seed'],
        genes_to_remove = config['genes_to_remove'],
        n_top_genes = config['n_top_genes'],
        min_genes = config['min_genes'],
        min_cells = config['min_cells'],
        donor_1 = config['donor_1'],
        donor_2 = config['donor_2'],
        celltype_1 = config['celltype_1'],
        celltype_2 = config['celltype_2'],
    output:
        # cnv_full_fname = output_path + '/cnv_full.png',
        cnv_subset_fname = output_path + '/cnv_subset.png',
        # adata_full_fname = output_path + '/adata_full.h5ad',
        adata_subset_fname = output_path + '/prepared_input.h5ad',
    script:
        'scripts/prepare_cnvs.py'
