rule generate_multibatch_data:
    resources:
        mem_per_cpu=10000,
        threads=10,
    params:
        de_prob = "{separability}",
        batch_facscale = config["batch_facscale"],
        n_cells = config["n_cells"],
        n_batches = config["n_batches"],
        frac_shared = "{frac_shared}",
        seed = "{rep_id}",
    output:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
        umap_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_umap.png',
        umap_nobatch_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_umap_nobatch.png',
    script:
        "../scripts/splatter_hierarchical.R"

rule gather_multibatch_scores:
    resources:
        time = "00:40:00",
        mem_per_cpu = 12000,
    input:
        fname_list = expand(
            'results/{method}/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
            method=METHODS,
            frac_shared=FRACS_SHARED, separability=SEPARABILITY,
            rep_id=[r for r in range(N_REPS)],)
    output:
        'results/multibatch_scores.csv'
    script:
        'scripts/gather_multibatch_scores.py'

rule run_scdef:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/scDEF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef.py"


rule run_scdef_un:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/scDEF_un/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scdef_un.py"

rule run_unintegrated:
    resources:
        time = "00:40:00",
        mem = 12000,
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/Unintegrated/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_unintegrated.py"

rule run_nmf:
    resources:
        time = "00:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/NMF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_nmf.py"

rule run_schpf:
    resources:
        time = "00:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/scHPF/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_schpf.py"

rule run_scvi:
    resources:
        mem_per_cpu=10000,
        threads=10,
        slurm="gpus=1 ntasks-per-node=10",
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/scVI/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scvi.py"

rule run_harmony:
    resources:
        time = "00:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/Harmony/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_harmony.py"

rule run_scanorama:
    resources:
        time = "00:40:00",
        mem_per_cpu=10000,
        threads=10,
    params:
        seed = "{rep_id}",
        true_hrc = TRUE_HRC,
    input:
        counts_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_counts.csv',
        meta_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_meta.csv',
        markers_fname = 'results/data/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_markers.csv',
    output:
        scores_fname = 'results/Scanorama/sep_{separability}/shared_{frac_shared}/rep_{rep_id}_scores.csv',
    script:
        "../scripts/run_scanorama.py"
