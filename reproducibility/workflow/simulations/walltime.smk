"""
Wall-time scalability study.
Runs all methods on datasets of increasing cell counts and records runtime.
"""

output_path = config.get("output_path", "results/walltime")

localrules: gather_scores

configfile: "config/walltime_config.yaml"
configfile: "config/methods.yaml"

N_CELLS = config["n_cells"]
N_REPS = config["n_reps"]
METHODS = config["methods"]
METRICS = config["metrics"]

envs_path = "../envs"
scripts_path = "../scripts"

_gpu_raw = config.get("gpu", True)
_gpu = _gpu_raw if isinstance(_gpu_raw, bool) else str(_gpu_raw).lower() not in ("false", "0", "no")
_scdef_env = envs_path + ("/scdef_reproducibility.yml" if _gpu else "/scdef_reproducibility_nogpu.yml")

wildcard_constraints:
    rep_id = r"\d+",
    n_cells = r"\d+",

rule all:
    input:
        output_path + '/walltime_scores.csv'

rule gather_scores:
    conda:
        _scdef_env
    params:
        param_name = ["cellno"],
        param_idx = [-2],
        method_idx = -3,
    input:
        fname_list = expand(
            output_path + '/{method}/cellno_{cellno}/rep_{rep_id}_scores.csv',
            method=METHODS,
            cellno=N_CELLS,
            rep_id=[r for r in range(N_REPS)],)
    output:
        output_path + '/walltime_scores.csv'
    script:
        '../scripts/gather_scores.py'


rule generate_multibatch_data:
    conda:
        envs_path + "/splatter.yml"
    params:
        de_fscale = config["de_fscale"],
        de_prob = config["de_prob"],
        batch_facscale = config["batch_facscale"],
        n_cells = "{n_cells}",
        n_batches = config["n_batches"],
        frac_shared = config["frac_shared"],
        seed = "{rep_id}",
        coverage = 0.,
    output:
        counts_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
        umap_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"


rule prepare_input:
    conda:
        _scdef_env
    params:
        seed = "{rep_id}",
    input:
        counts_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        fname = output_path + '/data/cellno_{n_cells}/rep_{rep_id}.h5ad'
    script:
        '../scripts/prepare_input.py'

RUN_SUFFIX = "cellno_{n_cells}/rep_{rep_id}"
INPUT_ADATA = output_path + '/data/' + RUN_SUFFIX + '.h5ad'
METHOD_SEED = "{rep_id}"

include: "../run_methods.smk"
