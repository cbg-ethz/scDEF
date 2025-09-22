from _benchmark import evaluate_methods

import scanpy as sc
import scdef as scd

def main():
    brd_id = int(snakemake.params["brd_id"])
    brds = snakemake.params["brds"]
    mu, tau = brds[brd_id]
    kappa = float(snakemake.params["kappa"])
    n_factors = int(snakemake.params["n_factors"])
    decay_factor = float(snakemake.params["decay_factor"])
    seed = int(snakemake.params["seed"])
    n_epoch = int(snakemake.params["n_epoch"])

    adata = sc.read_h5ad(snakemake.input["fname"])

    # Run scDEF
    model = scd.scDEF(
        adata,
        counts_layer="counts",
        n_factors=n_factors,
        decay_factor=decay_factor,
        layer_concentration=kappa,
        brd_mean=mu,
        brd_strength=tau,
        seed=seed,
    )
    print(model)
    model.fit(
        pretrain=True,
        nmf_init=False,
        batch_size=256,
        n_epoch=[n_epoch],
        lr=[0.01],
        num_samples=100,
        patience=50,
    )  # learn the hierarchical gene signatures

    g = scd.pl.make_graph(
        model,
        n_cells=True,
        wedged="cell_state",
        show_label=False,
    )
    g.render(snakemake.output["graph_fname"])

    hierarchy_scdef = scd.get_hierarchy()
    g = scd.pl.make_graph(
        model,
        hierarchy=hierarchy_scdef,
        n_cells=True,
        wedged="cell_state",
        show_label=False,
    )
    g.render(snakemake.output["hierarchy_graph_fname"])

    metrics_list = [
        "Cell Type ARI",
        "Cell Type ASW",
        "Batch ARI",
        "Batch ASW",
    ]

    df = evaluate_methods(
        adata,
        metrics_list,
        {"scDEF": model},
        celltype_obs_key="cell_state",
        batch_obs_key="batch",
    )
    df["brd"] = brd_id
    df["kappa"] = kappa
    df["n_factors"] = n_factors
    df["decay_factor"] = decay_factor
    df["rep"] = seed
    df["elbo"] = model.elbos[-1][0]
    df.to_csv(snakemake.output["scores_fname"])

    import pickle

    with open(snakemake.output["out_fname"], "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    main()