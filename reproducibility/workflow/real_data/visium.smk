# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

localrules: gather_results

configfile: "config/visium.yaml"
configfile: "config/methods.yaml"

output_path = "results/visium"

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
        output_path + '/scDEF/run/'

rule download_data:
    conda:
        _scdef_env
    params:
        out_dir = output_path + '/raw_data',
    output:
        done = output_path + '/raw_data/.download_done',
    script:
        '../scripts/download_visium.py'

rule prepare_input:
    conda:
        "../envs/squidpy.yml"
    params:
        data_path = output_path + '/raw_data',
        data_fname = 'CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5',
        seed = config['seed'],
        genes_to_remove = config['genes_to_remove'],
        n_top_genes = config['n_top_genes'],
    input:
        download_done = output_path + '/raw_data/.download_done',
    output:
        fname = output_path + '/prepared_input.h5ad'
    script:
        '../scripts/prepare_visium.py'
