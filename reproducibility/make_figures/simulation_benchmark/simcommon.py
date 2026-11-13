"""Shared definitions for the simulation benchmark re-run (scDEF-markdown/experiments/exp-simulation-benchmark-rerun.md).

The design is fixed in that note (2026-11-05, before any simulation was run); this module only encodes it:

- generator: ``simulate.R`` (the 2024 ``workflow/scripts/splatter_hierarchical.R`` plus additions);
- grid (2024 ``config/benchmark_{single,multi}batch.yaml``): de_fscale 2.0 (low group similarity) and 0.5 (high);
  de_prob 0.2; one batch of 4,000 cells; four batches of 1,000 cells with batch_facscale 0.4 and frac_shared 1.0
  (similar batches) or 0.5 (distinct batches); replicates 0-4 -> 30 data sets;
- methods: single batch scDEF, PCA, NMF, scHPF, f-scLVM, nSBM; four batches scDEF with and without batch
  annotations, PCA, NMF, scHPF, f-scLVM, nSBM, scVI, Harmony, Scanorama;
- comparators through the real-data re-run's machinery (``../realdata_benchmark``), seed = replicate;
- scDEF: ``refit_best_of_seeds.fit_fixed`` from five seeds, lowest final loss.

The real-data modules are loaded under distinct names (``rd_common``, ``rd_run_method``, ``rd_score``) so that this
package's scripts can keep the same names; inside them ``import common`` still finds the real-data ``common.py``.
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAKE_FIGURES = os.path.dirname(HERE)
REPRO = os.path.dirname(MAKE_FIGURES)
RD_DIR = os.path.join(MAKE_FIGURES, "realdata_benchmark")
OUT_ROOT = os.environ.get("SIMBENCH_OUT_ROOT", os.path.join(MAKE_FIGURES, "out", "simulation_benchmark"))
REF_DIR = os.path.join(OUT_ROOT, "reference")
REF_MTX = os.path.join(REF_DIR, "pbmc3k_counts.mtx")
PARAMS_RDS = os.path.join(REF_DIR, "splat_params.rds")

if RD_DIR not in sys.path:
    sys.path.append(RD_DIR)          # after this directory: the real-data `common` has no namesake here


def _load(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(RD_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import common as rc  # noqa: E402  (realdata_benchmark/common.py)

assert os.path.dirname(os.path.abspath(rc.__file__)) == RD_DIR, rc.__file__


def rd_run_method():
    return _load("rd_run_method", "run_method.py")


def rd_score():
    return _load("rd_score", "score.py")


read_json, write_json = rc.read_json, rc.write_json

# --------------------------------------------------------------------------- grid (2024 configs)
DE_PROB = 0.2
DE_FSCALES = [2.0, 0.5]
SIMILARITY = {2.0: "Low", 0.5: "High"}           # group similarity, as the 2024 x axis
REPS = [0, 1, 2, 3, 4]
SCENARIOS = {
    "single": dict(n_batches=1, n_cells=4000, frac_shared=0.0, batch_facscale=0.0, label="Single batch"),
    "similar": dict(n_batches=4, n_cells=1000, frac_shared=1.0, batch_facscale=0.4, label="Four similar batches"),
    "distinct": dict(n_batches=4, n_cells=1000, frac_shared=0.5, batch_facscale=0.4, label="Four distinct batches"),
}
# The 2024 single-batch rule passed n_cells 4000 with one batch, the multi-batch rule 1000 per batch with four.

HIERARCHY_OBS = ["GroupC", "GroupB", "GroupA"]   # finest first, prefixed "", "h", "hh" (2024 prepare_input.py)
LEVEL_PREFIX = {"GroupC": "", "GroupB": "h", "GroupA": "hh"}
LEVEL_GROUPS = {"GroupC": 8, "GroupB": 4, "GroupA": 2}
LEVEL_BLOCK = {"GroupA": (1, 1000), "GroupB": (1001, 1500), "GroupC": (1501, 1750)}   # gene numbers
TOP_GENES = 20                                   # level signatures: top genes of a signature

SINGLE_METHODS = list(rc.SINGLE_METHODS)          # PCA, NMF, scHPF, fscLVM, nSBM
MULTI_METHODS = list(rc.MULTI_METHODS)            # + scVI, Harmony, Scanorama
METRICS_SINGLE = list(rc.METRICS_SINGLE)
METRICS_MULTI = list(rc.METRICS_MULTI)
SCDEF_SEEDS = [42, 1, 2, 3, 4]                   # refit_best_of_seeds.SEEDS


def dataset_id(scenario, de_fscale, rep):
    return f"{scenario}_fs{float(de_fscale)}_r{int(rep)}"


def parse_id(ds):
    """``<scenario>_fs<de_fscale>_r<rep>``, optionally ``_dp<de_prob>`` (the hyperparameter sweeps of
    hyper_grid.py, 2026-11-11; without it, de_prob is DE_PROB as in the benchmark)."""
    parts = ds.split("_")
    if len(parts) == 4 and parts[3].startswith("dp"):
        scenario, fs, rep, dp = parts[0], parts[1], parts[2], float(parts[3][2:])
    elif len(parts) == 3:
        (scenario, fs, rep), dp = parts, DE_PROB
    else:
        raise SystemExit(f"unknown data set {ds!r}")
    if scenario not in SCENARIOS:
        raise SystemExit(f"unknown data set {ds!r}")
    fs, rep = float(fs[2:]), int(rep[1:])
    sc = SCENARIOS[scenario]
    return dict(id=ds, scenario=scenario, de_fscale=fs, rep=rep, similarity=SIMILARITY[fs], de_prob=dp,
                multi=sc["n_batches"] > 1, **{k: v for k, v in sc.items() if k != "label"})


def all_ids():
    """Replicate-major, so that whole replicates finish early; the smoke data set comes first."""
    return [dataset_id(s, fs, r) for r in REPS for s in SCENARIOS for fs in DE_FSCALES]


def methods(ds):
    return MULTI_METHODS if parse_id(ds)["multi"] else SINGLE_METHODS


def scdef_variants(ds):
    """The scored multi-layer scDEF fits: label -> batch key given to scDEF."""
    return {"scDEF": "Batch", "scDEF_un": None} if parse_id(ds)["multi"] else {"scDEF": None}


# --------------------------------------------------------------------------- added 2026-11-05 (later the same day)
# Hierarchical-clustering baselines, the decomposition, and the shared/specific metric (design note, sections
# "Hierarchical-clustering baselines", "The decomposition is added", "Shared and batch-specific groups").
ONE_LAYER = "scDEF_1L"                 # one-layer scDEF (fit_scdef.py), used only through scDEF_1L_Ward
ONE_LAYER_SIZES = [100, 1]             # one layer of 100 factors under a single root, as the 2024 run_scdef_hclust
# With layer_sizes=[100] the fixed workflow fails (scd.tl.filter -> factor_diagnostics: "factor_obs is missing
# 'original_factor_idx'", 2026-11-05); [100, 1] passes fit, filter, refit and filter (checked on CPU the same day).
# SIMBENCH_ONE_LAYER=1 adds the one-layer fits and scDEF_1L_Ward; run_all.sh is resumable, so this adds only that work.
ONE_LAYER_ON = os.environ.get("SIMBENCH_ONE_LAYER", "0") == "1"
DECOMPOSED = "scDEF_decomposed"        # decompose.py, four batches only (top_layer=2, as the real-data workflow)
DECOMPOSED_T1 = "scDEF_decomposed_t1"  # decompose.py --top-layer 1 (package default; added 2026-11-10 after the nesting result)
REFERENCE_SCDEF = "scDEF"              # the kept model whose layer sizes cut every Ward tree: with batch
                                       # annotations on four batches (also for scDEF_un comparisons), plain on one
WARD_EMBEDDINGS = {"single": ["PCA", "NMF", "scHPF", "fscLVM"],
                   "multi": ["PCA", "NMF", "scHPF", "fscLVM", "scVI", "Harmony", "Scanorama"]}
SPECIFIC_MIN_FRACTION = 0.7            # a cluster is batch-specific if >= 70% of its cells are from one batch
SPECIFIC_MIN_CELLS = 20                # ... and it has >= 20 cells (Figure 5C, Table S1)
SPECIFIC_MIN_F1 = 0.5


def scdef_fits(ds):
    """Every scDEF fit of fit_scdef.py: label -> batch key (the one-layer model with batch annotations on four
    batches)."""
    v = dict(scdef_variants(ds))
    if ONE_LAYER_ON:
        v[ONE_LAYER] = "Batch" if parse_id(ds)["multi"] else None
    return v


def gpu_jobs(ds):
    """The GPU queue of a data set, in order: the multi-layer fits, the decomposition (needs scDEF), one layer."""
    jobs = list(scdef_variants(ds))
    if parse_id(ds)["multi"]:
        jobs.append(DECOMPOSED)
    return jobs + ([ONE_LAYER] if ONE_LAYER_ON else [])


def scdef_scored(ds):
    """scDEF models scored as scDEF objects."""
    return list(scdef_variants(ds)) + ([DECOMPOSED, DECOMPOSED_T1] if parse_id(ds)["multi"] else [])


def hc_variants(ds):
    """Hierarchical-clustering baselines (hc.py): variant -> dependencies (method dirs, scdef fit labels)."""
    emb = WARD_EMBEDDINGS["multi" if parse_id(ds)["multi"] else "single"]
    out = {f"{m}_Ward": dict(methods=[m], fits=[REFERENCE_SCDEF]) for m in emb}
    out["Leiden_Ward"] = dict(methods=["PCA"], fits=[REFERENCE_SCDEF])
    if ONE_LAYER_ON:
        out["scDEF_1L_Ward"] = dict(methods=[], fits=[ONE_LAYER, REFERENCE_SCDEF])
    return out


def scored_methods(ds):
    """Every method directory score.py reads (comparators, then the hierarchical-clustering baselines)."""
    return methods(ds) + list(hc_variants(ds))


def metrics(ds):
    return METRICS_MULTI if parse_id(ds)["multi"] else METRICS_SINGLE


def batch_key(ds):
    """Comparators get the batch key only with four batches. (The 2024 scripts passed "Batch" whenever the column
    existed, i.e. also for the one-batch data, where nSBM then ran Harmony on a single batch.)"""
    return "Batch" if parse_id(ds)["multi"] else None


def ds_dir(ds, *parts):
    d = os.path.join(OUT_ROOT, ds, *parts)
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- comparator settings
def settings(method, rep):
    s = dict(rc.SETTINGS[method])
    s["seed"] = rep
    s.pop("random_state", None)
    s.pop("numpy_seed", None)
    s.pop("random_seed", None)
    s["seed_note"] = "seed = replicate, as the 2024 simulation rules; Leiden keeps scanpy's random_state 0 (2024 too)"
    s["genes"] = "every gene of the simulated input (n_top_genes >= genes)"
    if method == "fscLVM":
        s["input"] = ("log-normalised counts (normalize_total 1e4, log1p), not regressed or scaled: the X of the "
                      "2024 simulation input (prepare_input.py)")
    if method == "nSBM":
        s["batch"] = "Harmony first with four batches; none with one batch"
    return s


# --------------------------------------------------------------------------- scDEF input
def scdef_adata(ds):
    """The AnnData every scDEF fit of a data set uses (and scoring reloads the saved model on): the shared input's
    cells and genes, raw counts in layers['counts'], X log-normalised, and the PCA hierarchical_init reads, made as
    in refit_best_of_seeds.tutorial_adata's multi-batch branch (normalize_total 1e4, log1p, scale max 10,
    PCA arpack) without regress_out (the simulated data have no mitochondrial genes)."""
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    inp = ad.read_h5ad(os.path.join(ds_dir(ds), "input.h5ad"))
    counts = inp.layers["counts"]
    counts = counts.toarray() if sp.issparse(counts) else np.asarray(counts)
    a = ad.AnnData(X=counts.astype(np.float32), obs=inp.obs.copy(), var=inp.var.copy())
    a.layers["counts"] = counts.astype(np.float32)
    a.obs["cell_type"] = a.obs["GroupC"].astype(str)          # refit_best_of_seeds.describe() reads it
    x = a.copy()
    sc.pp.normalize_total(x, target_sum=1e4)
    sc.pp.log1p(x)
    a.X = x.X.copy()
    sc.pp.scale(x, max_value=10)
    sc.tl.pca(x, svd_solver="arpack")
    a.obsm["X_pca"] = x.obsm["X_pca"]
    return a


def jsonable(o):
    return rc._jsonable(o)


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=jsonable)
