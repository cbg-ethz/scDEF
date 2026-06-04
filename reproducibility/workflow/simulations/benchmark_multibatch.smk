output_path = config.get("output_path", "results/benchmark_multibatch")

localrules: gather_multibatch_scores

configfile: "config/benchmark_multibatch.yaml"
configfile: "config/methods.yaml"

METHODS = config["methods"]
METRICS = config["metrics"]

N_REPS = config["n_reps"]
SEPARABILITY = config["de_fscale"]
FRACS_SHARED = config["frac_shared"]

envs_path = "../envs"
scripts_path = "../scripts"

_gpu_raw = config.get("gpu", True)
_gpu = _gpu_raw if isinstance(_gpu_raw, bool) else str(_gpu_raw).lower() not in ("false", "0", "no")
_scdef_env = envs_path + ("/scdef_reproducibility.yml" if _gpu else "/scdef_reproducibility_nogpu.yml")

wildcard_constraints:
    rep_id = r"\d+",
    separability = r"[^/]+",
    frac_shared = r"[^/]+",

rule all:
    input:
        output_path + '/multibatch_scores.csv'

rule gather_multibatch_scores:
    conda:
        _scdef_env
    input:
        fname_list = expand(
            output_path + '/{method}/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
            method=METHODS,
            frac_shared=FRACS_SHARED, separability=SEPARABILITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        fname = output_path + '/multibatch_scores.csv'
    script:
        '../scripts/gather_multibatch_scores.py'

rule generate_multibatch_data:
    conda:
        envs_path + "/splatter.yml"
    params:
        de_fscale = "{separability}",
        de_prob = config["de_prob"],
        batch_facscale = config["batch_facscale"],
        n_cells = 1000,
        n_batches = config["n_batches"],
        frac_shared = "{frac_shared}",
        seed = "{rep_id}",
        coverage = 0.,
    output:
        counts_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
        umap_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_umap_nobatch.png',
    script:
        '../scripts/splatter_hierarchical.R'

rule prepare_input:
    conda:
        _scdef_env
    params:
        seed = "{rep_id}",
    input:
        counts_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        fname = output_path + '/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}.h5ad'
    script:
        '../scripts/prepare_input.py'

RUN_SUFFIX = "sep_{separability}/shared_{frac_shared}/rep_{rep_id}"
INPUT_ADATA = output_path + '/data/' + RUN_SUFFIX + '.h5ad'
METHOD_SEED = "{rep_id}"

include: "../run_methods.smk"
