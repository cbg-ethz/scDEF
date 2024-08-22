configfile: "config/benchmark_config.yaml"

N_REPS = config["n_reps"]
SEPARABILITY = config["de_fscale"]
FRACS_SHARED = config["frac_shared"]
METHODS = config["methods"]
SINGLE_METHODS = config["singlebatch_methods"]

include: "rules/multibatch.smk"
include: "rules/singlebatch.smk"

rule all:
    input:
        'results/benchmark_figure.pdf'

rule plot_benchmark:
    input:
        singlebatch_scores = 'results/singlebatch_scores.csv',
        multibatch_scores = 'results/multibatch_scores.csv'
    output:
        'results/benchmark_figure.pdf',
    script:
        "scripts/plot_benchmark.py"
