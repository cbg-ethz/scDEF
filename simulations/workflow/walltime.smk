configfile: "config/walltime_config.yaml"

N_CELLS = config["n_cells"]
N_REPS = config["n_reps"]
METHODS = config["methods"]


rule all:
    input:
        'walltime_results/benchmark_figure.pdf'

rule plot_benchmark:
    input:
        multibatch_scores = 'walltime_results/scores.csv'
    output:
        'walltime_results/benchmark_figure.pdf',
    run:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        df = pd.read_csv(snakemake.input[0])
        df = df.replace({"Unintegrated": "Leiden"})

        annots = {
            "scDEF": "True",
            "scDEF_un": "False",
            "NMF": "False",
            "scHPF": "False",
            "Unintegrated": "False",
            "scVI": "True",
            "Harmony": "True",
            "Scanorama": "True",
        }

        colordict = {
            "scDEF": "green",
            "scDEF (-)": "lightgreen",
            "NMF": "blue",
            "scHPF": "darkblue",
            "Leiden": "violet",
            "scVI": "orange",
            "Harmony": "red",
            "Scanorama": "darkred",
        }

        ax = sns.catplot(
            data=df,
            kind="box",
            x="Number of cells",
            y="Runtime",
            hue="Method",
            col="name",
            style="Annotations",
            palette=colordict,
            hue_order=[
                "scDEF",
                "scDEF_un",
                "scVI",
                "Harmony",
                "Scanorama",
                "Leiden",
                "scHPF",
                "NMF",
            ],
            aspect=0.5,
            dodge=True,
        )
        ax.fig.set_size_inches(12, 4)
        plt.savefig(snakemake.output)

rule gather_scores:
    resources:
        time = "03:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'walltime_results/{method}/cellno_{cellno}/rep_{rep_id}_scores.csv',
            method=METHODS,
            cellno=N_CELLS,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'walltime_results/scores.csv'
run:


rule generate_multibatch_data:
    resources:
        mem_per_cpu=10000,
        threads=10,
    params:
        de_fscale = config["de_fscale"]
        batch_facscale = config["batch_facscale"],
        n_cells = "{n_cells}",
        n_batches = config["n_batches"],
        frac_shared = config["frac_shared"],
        seed = "{rep_id}",
    output:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    script:
        "../scripts/splatter_hierarchical.R"

rule run_scdef:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/scDEF/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef.py"


rule run_scdef_un:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/scDEF_un/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef_un.py"

rule run_unintegrated:
    resources:
        time = "03:40:00",
        mem = 12000,
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/Unintegrated/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_unintegrated.py"

rule run_nmf:
    resources:
        time = "03:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/NMF/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_nmf.py"

rule run_schpf:
    resources:
        time = "03:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/scHPF/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_schpf.py"

rule run_scvi:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/scVI/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scvi.py"

rule run_harmony:
    resources:
        time = "03:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/Harmony/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_harmony.py"

rule run_scanorama:
    resources:
        time = "03:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/Scanorama/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scanorama.py"

rule run_nsbm:
    resources:
        time = "03:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/nSBM/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_nsbm.py"

rule run_muvi:
    resources:
        time = "03:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
    input:
        counts_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_counts.csv',
        meta_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_meta.csv',
        markers_fname = 'walltime_results/data/cellno_{n_cells}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'walltime_results/MuVI/cellno_{n_cells}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_muvi.py"