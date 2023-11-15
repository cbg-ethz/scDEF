rule generate_singlebatch_data:
    resources:
        mem_per_cpu=10000,
        threads=10,
    params:
        de_prob = "{separability}",
        batch_facscale = config["batch_facscale"],
        n_cells = config["n_cells"],
        n_batches = 1,
        seed = "{rep_id}",
    output:
        counts_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_markers.csv',
        umap_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_umap.png',
        umap_nobatch_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"

rule gather_singlebatch_scores:
    resources:
        time = "00:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'results/{method}/sep_{separability}/singlebatch/rep_{rep_id}_scores.csv',
            method=SINGLE_METHODS,separability=SEPARABILITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'results/singlebatch_scores.csv'
    script:
        'scripts/gather_singlebatch_scores.py'

rule run_scdef_singlebatch:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/scDEF/sep_{separability}/singlebatch/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef.py"

rule run_unintegrated_singlebatch:
    resources:
        time = "00:40:00",
        mem = 12000,
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/Unintegrated/sep_{separability}/singlebatch/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_unintegrated.py"

rule run_nmf_singlebatch:
    resources:
        time = "00:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/NMF/sep_{separability}/singlebatch/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_nmf.py"

rule run_schpf_singlebatch:
    resources:
        time = "00:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/singlebatch/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/scHPF/sep_{separability}/singlebatch/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_schpf.py"
