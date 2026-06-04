# This just generates the raw results from all the methods
# The figures are generated within a notebook that only requires the results

output_path = "results/pbmcsifn"

localrules: gather_results

configfile: "config/pbmcsifn.yaml"
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
            output_path + '/{method}/seed_{seed}_scores.csv',
            method=METHODS, seed=SEEDS,)
    output:
        fname = output_path + '/scores.csv'
    script:
        '../scripts/gather_real_data_scores.py'

rule download_data:
    conda:
        "../envs/seuratdata.yml"
    params:
        dataset = "ifnb",
    output:
        counts_fname = output_path + '/raw_data/counts.csv',
        meta_fname = output_path + '/raw_data/meta.csv',
    script:
        '../scripts/download_seuratdata.R'

rule convert_data:
    conda:
        _scdef_env
    input:
        counts_fname = output_path + '/raw_data/counts.csv',
        meta_fname = output_path + '/raw_data/meta.csv',
    output:
        fname = output_path + '/raw_data/raw.h5ad',
    script:
        '../scripts/convert_seuratdata.py'

rule prepare_input:
    conda:
        _scdef_env
    params:
        data_fname = output_path + '/raw_data/raw.h5ad',
        seed = config['seed'],
        genes_to_remove = config['genes_to_remove'],
        n_top_genes = config['n_top_genes'],
        min_genes = config['min_genes'],
        min_cells = config['min_cells'],
    input:
        raw_data = output_path + '/raw_data/raw.h5ad',
    output:
        fname = output_path + '/prepared_input.h5ad'
    script:
        '../scripts/prepare_pbmcsifn.py'
