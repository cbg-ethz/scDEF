"""
Shared run/evaluate rules for all benchmarking methods.

Including workflows must define before this include:
  RUN_SUFFIX   - path segment after the method name (e.g. "run" or "sep_{separability}/rep_{rep_id}")
  INPUT_ADATA  - full path to the prepared AnnData input
  METHOD_SEED  - seed passed to run scripts (int or wildcard string like "{rep_id}")
"""

# Paths relative to this file (workflow/run_methods.smk)
method_envs = "envs"
method_scripts = "scripts"

# GPU-aware environment selection: pass --config gpu=false for non-CUDA machines
_gpu_raw = config.get("gpu", True)
_gpu = _gpu_raw if isinstance(_gpu_raw, bool) else str(_gpu_raw).lower() not in ("false", "0", "no")
_run_scdef_env  = method_envs + ("/scdef_reproducibility.yml" if _gpu else "/scdef_reproducibility_nogpu.yml")
_run_scvi_env   = method_envs + ("/scVI.yml" if _gpu else "/scVI_nogpu.yml")
_run_fsclvm_env = method_envs + ("/fscLVM.yml" if _gpu else "/fscLVM_nogpu.yml")

# --- scDEF ---

rule run_scdef:
    conda:
        _run_scdef_env
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        n_layers = config['scDEF']['n_layers'],
        n_factors = config['scDEF']['n_factors'],
        nmf_init = config['scDEF']['nmf_init'],
        pretraining = config['scDEF']['pretraining'],
        brd_strength = config['scDEF']['brd_strength'],
        brd_mean = config['scDEF']['brd_mean'],
        hierarchy_weight = config['scDEF']['hierarchy_weight'],
        top_factors = config['scDEF']['top_factors'],
        n_epoch = config['scDEF']['n_epoch'],
        lr = config['scDEF']['lr'],
        batch_size = config['scDEF']['batch_size'],
        num_samples = config['scDEF']['num_samples'],
        seed = METHOD_SEED,
    input:
        adata = INPUT_ADATA,
    output:
        out_dir = directory(output_path + '/scDEF/' + RUN_SUFFIX + '/'),
        duration_fname = output_path + '/scDEF/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_scdef.py"

rule evaluate_scdef:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "scDEF",
        metrics = METRICS,
        is_scdef = True,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/scDEF/' + RUN_SUFFIX + '/',
        duration_fname = output_path + '/scDEF/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/scDEF/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- scDEF_un ---

rule run_scdef_un:
    conda:
        _run_scdef_env
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        n_layers = config['scDEF_un']['n_layers'],
        n_factors = config['scDEF_un']['n_factors'],
        nmf_init = config['scDEF_un']['nmf_init'],
        pretraining = config['scDEF_un']['pretraining'],
        brd_strength = config['scDEF_un']['brd_strength'],
        brd_mean = config['scDEF_un']['brd_mean'],
        hierarchy_weight = config['scDEF_un']['hierarchy_weight'],
        top_factors = config['scDEF_un']['top_factors'],
        n_epoch = config['scDEF_un']['n_epoch'],
        lr = config['scDEF_un']['lr'],
        batch_size = config['scDEF_un']['batch_size'],
        num_samples = config['scDEF_un']['num_samples'],
        seed = METHOD_SEED,
    input:
        adata = INPUT_ADATA,
    output:
        out_dir = directory(output_path + '/scDEF_un/' + RUN_SUFFIX + '/'),
        duration_fname = output_path + '/scDEF_un/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_scdef_un.py"

rule evaluate_scdef_un:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "scDEF_un",
        metrics = METRICS,
        is_scdef = True,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/scDEF_un/' + RUN_SUFFIX + '/',
        duration_fname = output_path + '/scDEF_un/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/scDEF_un/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- scDEF_corr ---

rule run_scdef_corr:
    conda:
        _run_scdef_env
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        n_epoch = config['scDEF_corr']['n_epoch'],
        lr = config['scDEF_corr']['lr'],
    input:
        scdef_un_dir = output_path + '/scDEF_un/' + RUN_SUFFIX + '/',
    output:
        out_dir = directory(output_path + '/scDEF_corr/' + RUN_SUFFIX + '/'),
        duration_fname = output_path + '/scDEF_corr/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_scdef_corr.py"

rule evaluate_scdef_corr:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "scDEF_corr",
        metrics = METRICS,
        is_scdef = True,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/scDEF_corr/' + RUN_SUFFIX + '/',
        duration_fname = output_path + '/scDEF_corr/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/scDEF_corr/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- scDEF_hclust ---

rule run_scdef_hclust:
    conda:
        _run_scdef_env
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        n_factors = config['scDEF_hclust']['n_factors'],
        n_layers = 1,
        nmf_init = config['scDEF_hclust']['nmf_init'],
        pretraining = config['scDEF_hclust']['pretraining'],
        brd_strength = config['scDEF_hclust']['brd_strength'],
        brd_mean = config['scDEF_hclust']['brd_mean'],
        hierarchy_weight = config['scDEF_hclust']['hierarchy_weight'],
        top_factors = config['scDEF_hclust']['top_factors'],
        n_epoch = config['scDEF_hclust']['n_epoch'],
        lr = config['scDEF_hclust']['lr'],
        batch_size = config['scDEF_hclust']['batch_size'],
        num_samples = config['scDEF_hclust']['num_samples'],
        resolutions = config['scDEF_hclust']['resolutions'],
        seed = METHOD_SEED,
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/scDEF_hclust/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/scDEF_hclust/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_scdef_hclust.py"

rule evaluate_scdef_hclust:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "scDEF_hclust",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/scDEF_hclust/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/scDEF_hclust/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/scDEF_hclust/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- PCA ---

rule run_pca:
    conda:
        method_envs + "/PCA.yml"
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        seed = METHOD_SEED,
        method = "PCA",
        n_top_genes = config["PCA"]['n_top_genes'],
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/PCA/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/PCA/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_unintegrated.py"

rule evaluate_pca:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "PCA",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/PCA/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/PCA/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/PCA/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- NMF ---

rule run_nmf:
    conda:
        method_envs + "/NMF.yml"
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        seed = METHOD_SEED,
        method = "NMF",
        max_iter = config["NMF"]['max_iter'],
        n_top_genes = config["NMF"]['n_top_genes'],
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/NMF/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/NMF/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_nmf.py"

rule evaluate_nmf:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "NMF",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/NMF/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/NMF/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/NMF/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- scHPF ---

rule run_schpf:
    conda:
        method_envs + "/scHPF.yml"
    threads: 1
    resources:
        mem_mb = 16000,
    params:
        seed = METHOD_SEED,
        method = "scHPF",
        max_iter = config["scHPF"]['max_iter'],
        min_iter = config["scHPF"]['min_iter'],
        n_top_genes = config["scHPF"]['n_top_genes'],
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/scHPF/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/scHPF/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_schpf.py"

rule evaluate_schpf:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "scHPF",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/scHPF/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/scHPF/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/scHPF/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- scVI ---

rule run_scvi:
    conda:
        _run_scvi_env
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        seed = METHOD_SEED,
        method = "scVI",
        n_top_genes = config["scVI"]['n_top_genes'],
        max_epochs = config["scVI"]['max_epochs'],
        batch_size = config["scVI"]['batch_size'],
        early_stopping = config["scVI"]['early_stopping'],
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/scVI/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/scVI/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_scvi.py"

rule evaluate_scvi:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "scVI",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/scVI/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/scVI/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/scVI/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- Harmony ---

rule run_harmony:
    conda:
        method_envs + "/Harmony.yml"
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        seed = METHOD_SEED,
        method = "Harmony",
        n_top_genes = config["Harmony"]['n_top_genes'],
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/Harmony/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/Harmony/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_harmony.py"

rule evaluate_harmony:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "Harmony",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/Harmony/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/Harmony/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/Harmony/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- Scanorama ---

rule run_scanorama:
    conda:
        method_envs + "/Scanorama.yml"
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        seed = METHOD_SEED,
        method = "Scanorama",
        n_top_genes = config["Scanorama"]['n_top_genes'],
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/Scanorama/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/Scanorama/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_scanorama.py"

rule evaluate_scanorama:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "Scanorama",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/Scanorama/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/Scanorama/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/Scanorama/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- nSBM ---

rule run_nsbm:
    conda:
        method_envs + "/nSBM.yml"
    threads: 2
    resources:
        mem_mb = 16000,
    params:
        seed = METHOD_SEED,
        method = "nSBM",
        n_top_genes = config["nSBM"]['n_top_genes'],
        n_init = config["nSBM"]['n_init'],
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/nSBM/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/nSBM/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_nsbm.py"

rule evaluate_nsbm:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "nSBM",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/nSBM/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/nSBM/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/nSBM/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"

# --- fscLVM ---

rule run_fsclvm:
    conda:
        _run_fsclvm_env
    threads: 4
    resources:
        mem_mb = 32000,
        gpu = 1,
    params:
        seed = METHOD_SEED,
        method = "fscLVM",
        batch_size = config["fscLVM"]['batch_size'],
        n_epochs = config["fscLVM"]['n_epochs'],
        n_top_genes = config["fscLVM"]['n_top_genes'],
    input:
        adata = INPUT_ADATA,
    output:
        out_fname = output_path + '/fscLVM/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/fscLVM/' + RUN_SUFFIX + '_duration.txt',
    script:
        method_scripts + "/run_muvi.py"

rule evaluate_fsclvm:
    conda:
        _run_scdef_env
    threads: 1
    resources:
        mem_mb = 8000,
    params:
        method = "fscLVM",
        metrics = METRICS,
        is_scdef = False,
    input:
        adata = INPUT_ADATA,
        model_state = output_path + '/fscLVM/' + RUN_SUFFIX + '.h5ad',
        duration_fname = output_path + '/fscLVM/' + RUN_SUFFIX + '_duration.txt',
    output:
        scores_fname = output_path + '/fscLVM/' + RUN_SUFFIX + '_scores.csv',
    script:
        method_scripts + "/evaluate_results.py"
