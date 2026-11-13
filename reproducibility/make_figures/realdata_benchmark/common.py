"""Shared definitions for the local re-run of the four standard real-data benchmarks.

Data sets, label columns and marker lists, method settings, output paths, and the import shims that
let the 2024 benchmark code (``reproducibility/workflow/scripts/_benchmark``) be reused unchanged.

Fairness rules: scDEF-markdown/decisions/decision-benchmark-no-manual-steps.md (every method gets the
same cells and genes, fixed package defaults, no manual steps; scDEF is not refitted, the saved
best-of-five models are scored).
"""
import importlib
import importlib.util
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
MAKE_FIGURES = os.path.dirname(HERE)
REPRO = os.path.dirname(MAKE_FIGURES)
SCDEF_SRC = os.path.join(os.path.dirname(REPRO), "src")
BENCH_PKG_DIR = os.path.join(REPRO, "workflow", "scripts", "_benchmark")
OUT_ROOT = os.environ.get("BENCH_OUT_ROOT", os.path.join(MAKE_FIGURES, "out", "realdata_benchmark"))
NB = os.path.expanduser("~/projects/scDEF/scdef_notebooks_revision")      # fitted scDEF models
REFIT_JSON = os.path.join(MAKE_FIGURES, "out", "refit_best_of_seeds_{}.json")
# The planaria marker lists of the 2024 benchmark exist only on the cluster
# (/cluster/work/bewi/members/pedrof/scdef_revisions/init_data/planaria/planaria_marker_genes.csv), and Table S3
# of Plass et al. 2018 cannot be downloaded automatically. derive_planaria_markers.py derives them from the
# annotation with the rule of the 2024 simulations (vault: exp-realdata-benchmark-local, "Planaria marker genes").
PLANARIA_MARKERS = os.environ.get(
    "PLANARIA_MARKERS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "realdata_benchmark",
                                     "planaria", "derived_markers.csv"))

SEED = 42
BUDGET_HOURS = float(os.environ.get("BENCH_BUDGET_HOURS", 4))   # the paper's fixed budget per method
METRICS_SINGLE = ["Cell Type ARI", "Cell Type ASW", "Hierarchy accuracy",
                  "Hierarchical signature consistency", "Signature sparsity", "Signature accuracy"]
METRICS_MULTI = ["Cell Type ARI", "Cell Type ASW", "Batch ARI", "Batch ASW", "Hierarchy accuracy",
                 "Hierarchical signature consistency", "Signature sparsity", "Signature accuracy"]

# The resolution sweeps of the 2024 code (_benchmark/constants.py: OTHERS_RES_SWEEPS). The Methods text
# lists four levels (1.0, 0.6, 0.3, 0.1 and 30, 20, 10, 5); the code that produced the figures used five.
LEIDEN_SWEEP = [1.0, 0.6, 0.3, 0.1, 0.01]
FACTOR_SWEEP = [30, 20, 10, 5, 1]

# run order: nSBM last, since it can take the whole budget
SINGLE_METHODS = ["PCA", "NMF", "scHPF", "fscLVM", "nSBM"]
MULTI_METHODS = ["PCA", "NMF", "scHPF", "fscLVM", "scVI", "Harmony", "Scanorama", "nSBM"]

# --------------------------------------------------------------------------- data sets
# refit: the name refit_best_of_seeds.tutorial_adata() and the saved models use; batch: the batch column
# of that AnnData (the 2024 configs call it "Batch"); scdef: label -> saved model.
DATASETS = {
    "pbmcs3k": dict(refit="pbmcs3k", batch=None, methods=SINGLE_METHODS, metrics=METRICS_SINGLE,
                    hierarchy_obs=["Cell types", "Coarse cell types"],
                    scdef={"scDEF": "scdef-pbmcs3k-best5"}),
    "planaria": dict(refit="planaria", batch=None, methods=SINGLE_METHODS, metrics=METRICS_SINGLE,
                     hierarchy_obs=["Cell types", "Coarse cell types"],
                     scdef={"scDEF": "scdef-planaria-best5"}),
    "pbmcs2b": dict(refit="pbmcs2b", batch="Experiment", methods=MULTI_METHODS, metrics=METRICS_MULTI,
                    hierarchy_obs=["Cell types", "Coarse cell types"],
                    scdef={"scDEF_un": "scdef-pbmcs2b-nobatch-best5", "scDEF": "scdef-pbmcs2b-ref-best5",
                           "scDEF_decomposed": "scdef-pbmcs2b-dec-top2-best5"}),
    "pbmcsifn": dict(refit="ifn", batch="stim", methods=MULTI_METHODS, metrics=METRICS_MULTI,
                     hierarchy_obs=["Cell states", "Coarse cell types"],
                     scdef={"scDEF_un": "scdef-ifn-nobatch-best5", "scDEF": "scdef-ifn-ref-best5",
                            "scDEF_decomposed": "scdef-ifn-dec-top2-best5"}),
}
ALIASES = {"ifn": "pbmcsifn", "3k": "pbmcs3k"}
# the refit JSON key for each scDEF label
SCDEF_KIND = {"scDEF_un": "nobatch", "scDEF": None, "scDEF_decomposed": "decomposed"}


def canonical(data):
    data = ALIASES.get(data, data)
    if data not in DATASETS:
        raise SystemExit(f"unknown data set {data!r}; one of {sorted(DATASETS)}")
    return data


def out_dir(data, *parts):
    d = os.path.join(OUT_ROOT, canonical(data), *parts)
    os.makedirs(d if not os.path.splitext(d)[1] else os.path.dirname(d), exist_ok=True)
    return d


# --------------------------------------------------------------------------- labels (2024 prepare_*.py)


def coarse_3k(c):                  # prepare_pbmcs3k.py
    return "T" if c.endswith(" T") else "Mono" if c.endswith("Mono") else c


def coarse_planaria(c):            # prepare_planaria.py
    if c.startswith("neoblast"):
        return "stem cells"
    for key, name in (("neur", "neurons"), ("epider", "epidermis"), ("muscle", "muscle"),
                      ("parenchy", "parenchyma"), ("secretory", "secretory"), ("pharynx", "pharynx")):
        if key in c:
            return name
    if "gut" in c or "phagocyt" in c or "goblet" in c:
        return "gut"
    return c


COARSE_2B = {"CD4+ T cell": "T cell", "Cytotoxic T cell": "T cell", "CD14+ monocyte": "Monocyte",
             "CD16+ monocyte": "Monocyte", "Dendritic cell": "Monocyte",
             "Plasmacytoid dendritic cell": "Monocyte"}                       # prepare_pbmcs2b.py
COARSE_IFN = {"B": "B", "B activated": "B", "CD4 Memory T": "T", "CD4 Naive T": "T", "CD8 T": "T",
              "CD14 Mono": "Monocyte", "CD16 Mono": "Monocyte", "DC": "Monocyte", "pDC": "Monocyte",
              "Eryth": "Eryth", "Mk": "Mk", "NK": "NK", "T activated": "T"}   # prepare_pbmcsifn.py

# Marker lists of the 2024 benchmark (prepare_pbmcs3k.py, prepare_pbmcs2b.py, prepare_pbmcsifn.py).
MARKERS = {
    "pbmcs3k": {
        "Memory CD4 T": ["IL7R", "CD3D", "CD3E", "IL32", "CD2"],
        "Naive CD4 T": ["IL7R", "CD3D", "CD3E", "TCF7", "CCR7", "CD2"],
        "CD8 T": ["CD8B", "CCL5", "CD2", "CD3D", "CD3E"],
        "NK": ["GNLY", "NKG7", "CD2"],
        "B": ["MS4A1", "CD79A", "CD79B", "CD74"],
        "CD14+ Mono": ["CD14", "LYZ"],
        "FCGR3A+ Mono": ["CD14", "FCGR3A", "MS4A7", "LYZ"],
        "DC": ["CD14", "FCER1A", "CST3", "CD74"],
        "Platelet": ["PPBP", "PF4"],
    },
    "pbmcs2b": {
        "CD4+ T cell": ["CD3D", "CD3E", "CD3G", "TRAC", "CD4", "TCF7", "CD27", "IL7R"],
        "Cytotoxic T cell": ["CD3D", "CD3E", "CD3G", "TRAC", "CD8A", "CD8B", "GZMK", "CCL5", "NKG7"],
        "CD14+ monocyte": ["VCAN", "FCN1", "S100A8", "S100A9", "CD14", "ITGAL", "ITGAM", "CSF3R",
                           "CSF1R", "CX3CR1", "TYROBP", "LYZ", "S100A12"],
        "CD16+ monocyte": ["FCN1", "FCGR3A", "FCGR3B", "ITGAL", "ITGAM", "CSF3R", "CSF1R", "CX3CR1",
                           "CDKN1C", "MS4A7"],
        "Dendritic cell": ["HLA-DPB1", "HLA-DPA1", "HLA-DQA1", "ITGAX", "CD1C", "CD1E", "FCER1A",
                           "CLEC10A", "FCGR2B", "MS4A1", "CD79A", "CD79B"],
        "Plasmacytoid dendritic cell": ["IL3RA", "GZMB", "JCHAIN", "IRF7", "TCF4", "LILRA4", "CLEC4C"],
        "B cell": ["CD19", "MS4A1", "CD79A", "CD79B", "MZB1", "IGHD", "IGHM"],
        "Natural killer cell": ["NCAM1", "NKG7", "KLRB1", "KLRD1", "KLRF1", "KLRC1", "KLRC2", "KLRC3",
                                "KLRC4", "FCGR3A", "FCGR3B", "ITGAL", "ITGAM", "FCER1G"],
        "Megakaryocyte": ["PF4", "PPBP", "GP5", "ITGA2B", "NRGN", "TUBB1", "SPARC", "RGS18", "MYL9",
                          "GNG11"],
    },
    "pbmcsifn": {
        "B": ["MS4A1", "CD79A"],
        "B activated": ["MS4A1", "CD79A", "MIR155HG", "NME1"],
        "CD4 Memory T": ["CD3D", "CREM"],
        "CD4 Naive T": ["CD3D", "GIMAP5", "CACYBP"],
        "CD8 T": ["CD3D", "CCL5", "CD8A"],
        "CD14 Mono_CTRL": ["FCGR3A", "S100A9"],
        "CD14 Mono_STIM": ["FCGR3A", "CCL2", "S100A9"],
        "CD16 Mono": ["FCGR3A", "VMO1"],
        "DC": ["HLA-DQA1", "GPR183"],
        "Eryth": ["HBA2", "HBB"],
        "Mk": ["PPBP", "GNG11"],
        "NK": ["GNLY", "NKG7", "CCL5"],
        "T activated": ["CD3D", "CREM", "HSPH1", "SELL", "CACYBP", "GPR183"],
        "pDC": ["TSPAN13", "IL3RA", "IGJ"],
    },
}

# --------------------------------------------------------------------------- method settings
# Package defaults throughout (decision-benchmark-no-manual-steps); config/methods.yaml held reduced smoke
# values (scDEF n_epoch 1, NMF max_iter 10, scHPF max_iter 20, scVI max_epochs 10, f-scLVM n_epochs 10)
# that are not used. n_top_genes: config/methods.yaml (the 2024 run_*.py scripts hard-coded 2000 for all).
SETTINGS = {
    "PCA": dict(implementation="_benchmark.other_methods.run_unintegrated via run_multiple_resolutions",
                n_top_genes=2000, hvg="seurat_v3 on counts (batch-aware with a batch key)",
                input="log-normalised counts (normalize_total 1e4, log1p), as run_unintegrated.py",
                pca="scanpy defaults (50 components)", neighbors="scanpy defaults (15 neighbours)",
                leiden="scanpy defaults (leidenalg, random_state 0)", resolutions=LEIDEN_SWEEP,
                signatures="Wilcoxon rank-sum (scanpy rank_genes_groups)"),
    "NMF": dict(implementation="_benchmark.other_methods.run_nmf via run_multiple_resolutions",
                n_top_genes=4000, input="counts, log(1e4 x / total + 1) inside run_nmf",
                resolutions=FACTOR_SWEEP, k_extra=2, k_choice="largest k whose factors all have >5 own top genes",
                max_iter="sklearn default (200)", init="sklearn default", random_state=SEED),
    "scHPF": dict(implementation="_benchmark.other_methods.run_schpf via run_multiple_resolutions",
                  n_top_genes=4000, input="raw counts", resolutions=FACTOR_SWEEP, k_extra=2,
                  k_choice="lowest final loss", schpf="scHPF class defaults (a=c=0.3, min_iter 30, "
                  "max_iter 1000, epsilon 0.001, better_than_n_ago 5); one trial per k, as 2024",
                  numpy_seed=SEED),
    "fscLVM": dict(implementation="_benchmark.other_methods.run_muvi via run_multiple_resolutions",
                   package="muvi 0.2.2", n_top_genes=4000,
                   input="log-normalised, regressed (library size, mitochondrial share), scaled "
                         "(max 10): the X of the 2024 prepared input",
                   resolutions=FACTOR_SWEEP, nmf=True, prior_masks=None,
                   fit="muvi defaults (full batch, 10000 epochs with early stopping, lr 0.005)",
                   factor_filter="factors explaining 95% of the variance", seed=SEED),
    "nSBM": dict(implementation="_benchmark.other_methods.run_nsbm", package="schist 0.10.0 (graph-tool 3.9)",
                 n_top_genes=2000, input="log-normalised counts; PCA (50), kNN graph (15); Harmony first "
                 "with a batch key", fit_model="schist defaults (nested, collect_marginals, n_samples 100, "
                 "n_iter 10, n_jobs -1)", random_seed=SEED,
                 note="config n_init 10 not passed: fit_model in schist 0.10 has no n_init"),
    "scVI": dict(implementation="local: train once, Leiden + Wilcoxon at each resolution as run_scvi",
                 n_top_genes=2000, input="raw counts", model="SCVI defaults (n_latent 10, zinb)",
                 train="scvi-tools defaults (max_epochs heuristic, batch_size 128, no early stopping)",
                 seed=SEED, resolutions=LEIDEN_SWEEP, leiden="scanpy defaults (random_state 0)"),
    "Harmony": dict(implementation="local: PCA + harmonypy once, Leiden + Wilcoxon at each resolution "
                    "as run_harmony", n_top_genes=2000, input="log-normalised counts; PCA (50)",
                    harmony="harmonypy defaults (theta 2)", random_state=SEED, resolutions=LEIDEN_SWEEP),
    "Scanorama": dict(implementation="local: PCA + scanorama once, Leiden + Wilcoxon at each resolution "
                      "as run_scanorama", n_top_genes=2000, input="log-normalised counts; PCA (50)",
                      scanorama="scanpy.external defaults", seed=SEED, resolutions=LEIDEN_SWEEP),
}
GPU_METHODS = {"scVI", "fscLVM"}


# --------------------------------------------------------------------------- import shims


def ensure_scdef_utils():
    """Make ``scdef.utils.hierarchy_utils`` importable.

    The comparators run in ``scdef-bench2``, which has no JAX stack. The 2024 code only needs scDEF's two
    pure-numpy utility modules there, so they are loaded from the source tree under a stub package; in the
    ``scdef`` environment the real package is used.
    """
    try:
        import scdef  # noqa: F401
        return "package"
    except Exception:
        pass
    for name in ("scdef", "scdef.utils", "scdef.models"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod

    class scDEF:  # placeholder: isinstance(x, scDEF) is False for every comparator output
        pass

    sys.modules["scdef.models"].scDEF = scDEF
    sys.modules["scdef"].scDEF = scDEF
    for sub in ("score_utils", "hierarchy_utils"):
        full = f"scdef.utils.{sub}"
        spec = importlib.util.spec_from_file_location(full, os.path.join(SCDEF_SRC, "scdef", "utils", f"{sub}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        setattr(sys.modules["scdef.utils"], sub, mod)
        setattr(sys.modules["scdef"], sub, mod)
    return "stub"


def bench_module(name):
    """Import ``_benchmark.<name>`` without running the package ``__init__`` (which imports every module)."""
    if "_benchmark" not in sys.modules:
        pkg = types.ModuleType("_benchmark")
        pkg.__path__ = [BENCH_PKG_DIR]
        sys.modules["_benchmark"] = pkg
    return importlib.import_module(f"_benchmark.{name}")


def refit_module():
    sys.path.insert(0, MAKE_FIGURES)
    import refit_best_of_seeds as rb
    return rb


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
