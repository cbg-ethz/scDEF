import scanpy as sc
import scdef

brd_id = int(snakemake.params["brd_id"])
brds = snakemake.params["brds"]
mu, tau = brds[brd_id]
kappa = float(snakemake.params["kappa"])
n_layers = int(snakemake.params["n_layers"])
layer_sizes = snakemake.params["layer_sizes"][n_layers - 1]
seed = int(snakemake.params["seed"])
n_epoch = int(snakemake.params["n_epoch"])

adata = sc.read_h5ad(snakemake.input["fname"])

# Run scDEF
scd = scdef.scDEF(
    adata,
    counts_layer="counts",
    layer_sizes=layer_sizes + [1],
    layer_shapes=[1.0] * n_layers + [1.0],
    layer_rates=[kappa] * n_layers + [1.0],
    brd_mean=mu,
    brd_strength=tau,
    seed=seed,
)
print(scd)
scd.learn(
    n_epoch=[n_epoch], patience=100, num_samples=10
)  # learn the hierarchical gene signatures
hierarchy_scdef = scd.get_hierarchy()
scd.make_graph(
    hierarchy=hierarchy_scdef,
    n_cells=True,
    wedged="cell_state",
    show_label=False,
)
scd.graph  # Graphviz object
scd.graph.render(snakemake.output["graph_fname"])

metrics_list = [
    "Cell Type ARI",
    "Cell Type ASW",
    "Batch ARI",
    "Batch ASW",
]

df = scdef.benchmark.evaluate_methods(
    adata,
    metrics_list,
    {"scDEF": scd},
    celltype_obs_key="cell_state",
    batch_obs_key="stim",
)
df["brd"] = brd_id
df["kappa"] = kappa
df["n_layers"] = len(layer_sizes)
df["rep"] = seed
df["elbo"] = scd.elbos[-1][0]
df.to_csv(snakemake.output["scores_fname"])
