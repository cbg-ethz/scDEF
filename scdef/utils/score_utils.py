import numpy as np


def get_mean_cellscore_per_group(cell_scores, cell_groups):
    unique_cluster_ids = np.unique(cell_groups)
    mean_cluster_scores = []
    for c in unique_cluster_ids:
        cell_idx = np.where(cell_groups == c)[0]
        mean_cluster_scores.append(np.mean(cell_scores[cell_idx], axis=0))
    mean_cluster_scores = np.array(mean_cluster_scores)
    return mean_cluster_scores


def mod_score(factors_by_groups_matrix):
    total_factor_relative_weight = np.sum(factors_by_groups_matrix, axis=1)
    total_factor_relative_weight = total_factor_relative_weight / np.sum(
        total_factor_relative_weight
    )

    # Per factor
    n1 = (
        factors_by_groups_matrix
        / np.sum(factors_by_groups_matrix, axis=1)[:, np.newaxis]
    )
    n1 = np.sum(np.max(n1, axis=1) * total_factor_relative_weight)

    # Per group
    n2 = (
        factors_by_groups_matrix
        / np.sum(factors_by_groups_matrix, axis=0)[np.newaxis, :]
    )
    n2 = np.mean(np.max(n2, axis=0))

    return np.mean([n1, n2])


def entropy_score(factors_by_groups_matrix):
    total_factor_relative_weight = np.sum(factors_by_groups_matrix, axis=1)
    total_factor_relative_weight = total_factor_relative_weight / np.sum(
        total_factor_relative_weight
    )

    # Per factor
    n1 = (
        factors_by_groups_matrix
        / np.sum(factors_by_groups_matrix, axis=1)[:, np.newaxis]
    )
    n1 = np.sum(-np.sum(n1 * np.log(n1), axis=1) * total_factor_relative_weight)

    # Per group
    n2 = (
        factors_by_groups_matrix
        / np.sum(factors_by_groups_matrix, axis=0)[np.newaxis, :]
    )
    n2 = np.mean(-np.sum(n2 * np.log(n2), axis=0))

    return np.mean([n1, n2])


def compute_geneset_coherence(genes, counts_adata):
    # As in Spectra: https://github.com/dpeerlab/spectra/blob/ff0e5c456127a33938b1ea560432f228dc26a08b/spectra/initialization.py
    mat = np.array(counts_adata[:, genes].X)
    n_genes = len(genes)
    score = 0
    for i in range(1, n_genes):
        for j in range(i):
            dw1 = mat[:, i] > 0
            dw2 = mat[:, j] > 0
            dw1w2 = (dw1 & dw2).astype(float).sum()
            dw1 = dw1.astype(float).sum()
            dw2 = dw2.astype(float).sum()
            score += np.log((dw1w2 + 1) / (dw2))

    denom = n_genes * (n_genes - 1) / 2

    return score / denom


def coherence_score(marker_gene_sets, heldout_counts_adata):
    chs = []
    for marker_genes in marker_gene_sets:
        chs.append(compute_geneset_coherence(marker_genes, heldout_counts_adata))
    return np.mean(chs)


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / float(len(s1.union(s2))))


def overlap_index(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    size1 = len(s1)
    size2 = len(s2)
    return float(len(s1.intersection(s2)) / float(min(size1, size2)))


def compute_fscore(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn)


def score_signature(inf_signature, markers_list, nonmarkers_list):
    # true positive: marker for this type is in signature
    # true negative: marker for another type is not in signature
    tp = sum([1 for gene in inf_signature if gene in markers_list])
    fp = sum([1 for gene in inf_signature if gene in nonmarkers_list])
    fn = sum([1 for gene in markers_list if gene not in inf_signature])
    return compute_fscore(tp, fp, fn)


def compute_cluster_obs_association_score(
    adata, cluster_assignments, cluster_name, obs_key, obs_val
):
    # Cells attached to factor
    cells_in_factor = np.where(np.array(cluster_assignments) == cluster_name)[0]
    adata_cells_in_factor = adata[cells_in_factor]

    # Cells from obs_val
    adata_cells_from_obs = adata[np.where(adata.obs[obs_key] == obs_val)[0]]

    cells_from_obs = float(adata_cells_from_obs.shape[0])

    # Number of cells from obs_val that are not in factor
    cells_not_in_factor_from_obs = float(
        np.count_nonzero(
            list(
                set(adata_cells_from_obs.obs.index).difference(
                    set(adata_cells_in_factor.obs.index)
                )
            )
        )
    )

    # Number of cells in factor that are obs_val
    cells_in_factor_from_obs = float(
        np.count_nonzero(adata_cells_in_factor.obs[obs_key] == obs_val)
    )

    # Number of cells in factor that are not obs_val
    cells_in_factor_not_from_obs = float(
        np.count_nonzero(adata_cells_in_factor.obs[obs_key] != obs_val)
    )

    return compute_fscore(
        cells_in_factor_from_obs,
        cells_in_factor_not_from_obs,
        cells_not_in_factor_from_obs,
    )


def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad / np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g
