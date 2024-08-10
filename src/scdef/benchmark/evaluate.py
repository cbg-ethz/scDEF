from scdef.utils import score_utils, hierarchy_utils
from scdef import scDEF

import pandas as pd
import numpy as np
import logging
from sklearn.metrics import adjusted_rand_score, silhouette_score


def evaluate_methods(
    adata,
    metrics_list,
    methods_results,
    true_hierarchy=None,
    hierarchy_obs_keys=None,
    markers=None,
    celltype_obs_key="celltype",
    batch_obs_key="batch",
    scdef_max_layers=5,
):
    methods_list = list(methods_results.keys())

    if not isinstance(celltype_obs_key, list):
        celltype_obs_key = [celltype_obs_key]

    df = pd.DataFrame(index=metrics_list, columns=methods_list)

    for method in methods_results:
        logging.info(f"Evaluating {method}...")
        method_outs = methods_results[method]

        # Hierarchy
        if "Hierarchy accuracy" in metrics_list:
            if isinstance(method_outs, scDEF):
                score = evaluate_scdef_hierarchy(
                    method_outs,
                    hierarchy_obs_keys,
                    true_hierarchy,
                )
            else:
                score = evaluate_hierarchy_from_cluster_levels(
                    adata,
                    hierarchy_obs_keys,
                    method_outs["assignments"],
                    true_hierarchy,
                )
            df.loc["Hierarchy accuracy", method] = score

        if "Hierarchical signature consistency" in metrics_list:
            if isinstance(method_outs, scDEF):
                method_simplified_hierarchy = method_outs.get_hierarchy(simplified=True)
                scdef_signatures, scdef_scores = method_outs.get_signatures_dict(
                    scores=True
                )
                scdef_sizes = method_outs.get_sizes_dict()
                score = evaluate_hierarchical_signatures_consistency(
                    adata.var_names,
                    method_simplified_hierarchy,
                    scdef_signatures,
                    scdef_scores,
                    scdef_sizes,
                    top_genes=20,
                )
            else:
                score = evaluate_hierarchical_signatures_consistency(
                    adata.var_names,
                    method_outs["simplified_hierarchy"],
                    method_outs["signatures"],
                    method_outs["scores"],
                    method_outs["sizes"],
                    top_genes=20,
                )
            df.loc["Hierarchical signature consistency", method] = score

        # Signatures
        if "Signature accuracy" in metrics_list:
            if isinstance(method_outs, scDEF):
                score = evaluate_scdef_signatures(
                    method_outs, hierarchy_obs_keys, markers
                )
            else:
                score = evaluate_cluster_signatures(
                    adata,
                    method_outs["assignments"],
                    method_outs["signatures"],
                    hierarchy_obs_keys,
                    markers,
                )
            df.loc["Signature accuracy", method] = score

        if "Signature sparsity" in metrics_list:
            ginis = []
            if isinstance(method_outs, scDEF):
                _, signature_scores = method_outs.get_signatures_dict(scores=True)
                for node in signature_scores:
                    ginis.append(score_utils.gini(signature_scores[node]))
            else:
                for node in method_outs["scores"]:
                    ginis.append(
                        score_utils.gini(
                            method_outs["scores"][node]
                            - np.min(method_outs["scores"][node])
                        )
                    )
            sparsity = np.mean(ginis)
            df.loc["Signature sparsity", method] = sparsity

        # Cell type clustering
        if "Cell Type ARI" in metrics_list:
            if isinstance(method_outs, scDEF):
                score = adjusted_rand_score(
                    method_outs.adata.obs[celltype_obs_key[0]],
                    method_outs.adata.obs["factor"],
                )
                # Do for all layers too
                for j, obs in enumerate(celltype_obs_key):
                    for i in range(scdef_max_layers):
                        lscore = np.nan
                        if method_outs.n_layers > i:
                            layer_name = f"{method_outs.layer_names[i]}factor"
                            lscore = adjusted_rand_score(
                                method_outs.adata.obs[obs],
                                method_outs.adata.obs[layer_name],
                            )
                        df.loc[f"Learned{i}vsTrue{j}", method] = lscore
            else:
                score = adjusted_rand_score(
                    adata.obs[celltype_obs_key[0]], method_outs["assignments"][0]
                )

            df.loc["Cell Type ARI", method] = score

        if "Cell Type ASW" in metrics_list:
            if isinstance(method_outs, scDEF):
                score = silhouette_score(
                    method_outs.adata.obsm["X_factors"],
                    method_outs.adata.obs[celltype_obs_key[0]],
                )
            else:
                score = silhouette_score(
                    method_outs["latents"][0],
                    adata.obs[celltype_obs_key[0]],
                )
            df.loc["Cell Type ASW", method] = score

        # Batch clustering
        if "Batch ARI" in metrics_list:
            if batch_obs_key in adata.obs.columns:
                if len(adata.obs[batch_obs_key].unique()) == 1:
                    score = 0.0
                else:
                    if isinstance(method_outs, scDEF):
                        score = adjusted_rand_score(
                            method_outs.adata.obs[batch_obs_key],
                            method_outs.adata.obs["factor"],
                        )

                        # Do for all layers too
                        for i in range(scdef_max_layers):
                            lscore = np.nan
                            if method_outs.n_layers > i:
                                layer_name = f"{method_outs.layer_names[i]}factor"
                                lscore = adjusted_rand_score(
                                    method_outs.adata.obs[batch_obs_key],
                                    method_outs.adata.obs[layer_name],
                                )
                                df.loc[f"Learned{i}vsBatch", method] = lscore
                    else:
                        score = adjusted_rand_score(
                            adata.obs[batch_obs_key], method_outs["assignments"][0]
                        )
            df.loc["Batch ARI", method] = score

        if "Batch ASW" in metrics_list:
            if batch_obs_key in adata.obs.columns:
                if len(adata.obs[batch_obs_key].unique()) == 1:
                    score = 0.0
                else:
                    if isinstance(method_outs, scDEF):
                        score = silhouette_score(
                            method_outs.adata.obsm["X_factors"],
                            method_outs.adata.obs[batch_obs_key],
                        )
                    else:
                        score = silhouette_score(
                            method_outs["latents"][0], adata.obs[batch_obs_key]
                        )
            df.loc["Batch ASW", method] = score

    return df


def evaluate_scdef_hierarchy(scd, obs_keys, true_hierarchy):
    simplified = scd.get_hierarchy(simplified=True)
    assignments, matches = scd.assign_obs_to_factors(
        obs_keys, hierarchy_utils.get_nodes_from_hierarchy(simplified)
    )
    annotated = hierarchy_utils.annotate_hierarchy(simplified, matches)

    obs_vals = [
        scd.adata.obs[obs_key].astype("category").cat.categories for obs_key in obs_keys
    ]
    obs_vals = list(set([item for sublist in obs_vals for item in sublist]))

    completed_annotated = hierarchy_utils.complete_hierarchy(annotated, obs_vals)
    completed_true_hierarchy = hierarchy_utils.complete_hierarchy(
        true_hierarchy, obs_vals
    )
    return hierarchy_utils.compare_hierarchies(
        completed_annotated, completed_true_hierarchy
    )


def evaluate_hierarchy_from_cluster_levels(
    adata, obs_keys, clusters_levels, true_hierarchy
):
    hierarchy = hierarchy_utils.get_hierarchy_from_clusters(clusters_levels)
    layer_names = ["h" * level for level in range(len(clusters_levels))]
    layer_sizes = [len(np.unique(cluster)) for cluster in clusters_levels]
    simplified = hierarchy_utils.simplify_hierarchy(hierarchy, layer_names, layer_sizes)
    assignments, matches = hierarchy_utils.assign_obs_to_clusters(
        adata,
        clusters_levels,
        layer_names,
        obs_keys,
        hierarchy_utils.get_nodes_from_hierarchy(simplified),
    )
    annotated = hierarchy_utils.annotate_hierarchy(simplified, matches)

    obs_vals = [
        adata.obs[obs_key].astype("category").cat.categories for obs_key in obs_keys
    ]
    obs_vals = list(set([item for sublist in obs_vals for item in sublist]))

    completed_annotated = hierarchy_utils.complete_hierarchy(annotated, obs_vals)
    completed_true_hierarchy = hierarchy_utils.complete_hierarchy(
        true_hierarchy, obs_vals
    )
    return hierarchy_utils.compare_hierarchies(
        completed_annotated, completed_true_hierarchy
    )


def evaluate_cluster_signatures(
    adata,
    clusters_levels,
    signatures,
    obs_keys,
    markers,
    cluster_names=[],
    top_genes=20,
):
    layer_names = ["h" * level for level in range(len(clusters_levels))]

    # Assign clusters to obs
    assignments, matches = hierarchy_utils.assign_obs_to_clusters(
        adata, clusters_levels, layer_names, obs_keys, cluster_names=cluster_names
    )

    # Compare each type's inferred signature with the true signature
    signature_scores = []
    for celltype in markers:
        cluster_name = assignments[celltype]
        signature = signatures[cluster_name][:top_genes]
        markers_type = markers[celltype]
        nonmarkers_type = [m for m in markers if m not in markers_type]
        signature_scores.append(
            score_utils.score_signature(signature, markers_type, nonmarkers_type)
        )
    return signature_scores


def evaluate_scdef_signatures(scd, obs_keys, markers, top_genes=20):
    # Assign factors to obs
    assignments, matches = scd.assign_obs_to_factors(obs_keys)
    signatures_dict = scd.get_signatures_dict()

    signature_scores = []
    for celltype in markers:
        factor_name = assignments[celltype]
        signature = signatures_dict[factor_name][:top_genes]
        markers_type = markers[celltype]
        nonmarkers_type = [m for m in markers if m not in markers_type]
        signature_scores.append(
            score_utils.score_signature(signature, markers_type, nonmarkers_type)
        )
    return signature_scores


def evaluate_hierarchical_signatures_consistency(
    var_names, hierarchy, signatures, scores, sizes, top_genes=20
):
    # sizes is a dict with the population sizes
    def get_consensus_signature(var_names, gene_scores_array, sizes_array):
        sizes_array = sizes_array / np.sum(sizes_array)
        avg_ranks = np.sum(sizes_array[:, None] * gene_scores_array, axis=0)
        consensus = var_names[np.argsort(avg_ranks)[::-1]].tolist()
        return consensus

    overlaps = []
    for parent in hierarchy:
        children = hierarchy[parent]
        if len(children) > 0:
            gene_scores = np.array(
                [scores[child] / np.max(scores[child]) for child in children]
            )  # ranking method-agnostic
            children_sizes = np.array([sizes[child] for child in children])
            consensus_signature = get_consensus_signature(
                var_names, gene_scores, children_sizes
            )
            overlap = score_utils.jaccard_similarity(
                [signatures[parent][:top_genes], consensus_signature[:top_genes]]
            )
            overlaps.append(overlap)
    return np.mean(overlaps)
