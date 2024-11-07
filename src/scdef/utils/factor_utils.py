import numpy as np
import matplotlib.pyplot as plt


def plot_cell_entropies(self, thres=0.9):
    entropies = []
    factor_ns = []
    for layer_idx in range(4):
        layer_name = self.layer_names[layer_idx]
        nf = len(self.factor_lists[layer_idx])
        a = (
            self.adata.obsm[f"X_{layer_name}factors"]
            / np.sum(self.adata.obsm[f"X_{layer_name}factors"], axis=1)[:, None]
        )
        a_sorted = np.vstack(
            [a[i, np.argsort(a, axis=1)[:, ::-1][i]] for i in range(a.shape[0])]
        )
        a_cumsums = np.cumsum(a_sorted, axis=1)
        n_factors = np.array(
            [(np.where(a_cumsums[i] > thres)[0][0] + 1) for i in range(a.shape[0])]
        )
        factor_ns.append(n_factors)
        entropy = np.array(
            [np.sum(-np.log(a[i]) * a[i]) / np.log(nf) for i in range(a.shape[0])]
        )
        entropies.append(entropy)
    fig, axes = plt.subplots(1, 2)
    plt.sca(axes[0])
    plt.boxplot(entropies)
    plt.sca(axes[1])
    plt.boxplot(factor_ns)
    plt.show()


def plot_factor_genes(self, thres=0.9):
    layer_kept_n_genes = []
    layer_removed_n_genes = []
    factor_n_kept_genes = dict()
    for layer_idx in range(len(self.layer_sizes)):
        layer_name = f"{self.layer_names[layer_idx]}"

        term_scores = self.pmeans[f"{self.layer_names[0]}W"]

        if layer_idx > 0:
            term_scores = self.pmeans[f"{self.layer_names[layer_idx]}W"]
            for layer in range(layer_idx - 1, 0, -1):
                lower_mat = self.pmeans[f"{self.layer_names[layer]}W"]
                term_scores = term_scores.dot(lower_mat)
            term_scores = term_scores.dot(self.pmeans[f"{self.layer_names[0]}W"])

        kept_factors_n_genes = []
        removed_factors_n_genes = []
        f_idx = 0
        for factor in range(self.layer_sizes[layer_idx]):
            vals = term_scores[factor] / np.sum(term_scores[factor])
            # sort
            vals_sorted = vals[np.argsort(vals)[::-1]]
            if factor in self.factor_lists[layer_idx]:
                kept_factors_n_genes.append(
                    np.where(np.cumsum(vals_sorted) > thres)[0][0]
                )
                factor_n_kept_genes[f"{layer_name}{f_idx}"] = kept_factors_n_genes[-1]
                f_idx += 1
            else:
                removed_factors_n_genes.append(
                    np.where(np.cumsum(vals_sorted) > thres)[0][0]
                )

        layer_kept_n_genes.append(kept_factors_n_genes)
        layer_removed_n_genes.append(removed_factors_n_genes)

    m = np.max(list(factor_n_kept_genes.values()))
    for f in factor_n_kept_genes:
        factor_n_kept_genes[f] = factor_n_kept_genes[f] / m
    factor_n_kept_genes

    fig, axes = plt.subplots(1, 4, figsize=(6, 3), sharey=True)
    for i in range(4):
        plt.sca(axes[i])
        plt.boxplot(
            [layer_kept_n_genes[i], layer_removed_n_genes[i]],
            tick_labels=["Kept", "Removed"],
        )
    plt.suptitle("Number of genes in factors")


def plot_factor_gini(self, idx, thres=0.9):
    a = self.pmeans[f"{self.layer_names[0]}W"][idx]
    # a -= np.mean(a)
    # a = np.maximum(a, 0)
    norm_a = a / np.sum(a)
    vals_sorted = norm_a[np.argsort(norm_a)[::-1]]
    plt.plot(np.cumsum(vals_sorted))
    plt.axvline(np.where(np.cumsum(vals_sorted) > thres)[0][0], color="gray")
    # plt.plot(a)
    plt.title(self.pmeans["brd"][idx])
    plt.show()
