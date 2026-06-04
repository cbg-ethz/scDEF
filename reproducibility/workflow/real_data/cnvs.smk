# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

localrules: gather_results

configfile: "config/cnvs.yaml"
configfile: "config/methods.yaml"

output_path = "results/cnvs"

METHODS = config["methods"]
SEED = config["seed"]
METRICS = config["metrics"]

RUN_SUFFIX = "run"
INPUT_ADATA = output_path + '/prepared_input.h5ad'
METHOD_SEED = SEED

_gpu_raw = config.get("gpu", True)
_gpu = _gpu_raw if isinstance(_gpu_raw, bool) else str(_gpu_raw).lower() not in ("false", "0", "no")
_scdef_env = "../envs/" + ("scdef_reproducibility.yml" if _gpu else "scdef_reproducibility_nogpu.yml")

include: "../run_methods.smk"

rule all:
    input:
        output_path + '/scores.csv'

rule gather_results:
    conda:
        _scdef_env
    input:
        fname_list = expand(
            output_path + '/{method}/run_scores.csv',
            method=METHODS,)
    output:
        fname = output_path + '/scores.csv'
    script:
        '../scripts/gather_real_data_scores.py'

rule download_data:
    conda:
        _scdef_env
    params:
        out_dir = output_path + '/raw_data',
    output:
        done = output_path + '/raw_data/.download_done',
    script:
        '../scripts/download_cnvs.py'

rule prepare_input:
    conda:
        "../envs/infercnv.yml"
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
    input:
        download_done = output_path + '/raw_data/.download_done',
    output:
        adata_full_fname = output_path + '/adata_full.h5ad',
        adata_subset_fname = output_path + '/prepared_input.h5ad',
    script:
        '../scripts/prepare_cnvs.py'
