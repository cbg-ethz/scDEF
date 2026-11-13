"""The spatial figure: the scDEF hierarchy of a Visium breast cancer section as a network of gene signatures, with
each factor's spatial autocorrelation and the neighbourhood enrichment between layer-0 factors, and spatial maps.

Data: 10x CytAssist FFPE human breast cancer (Janesick et al. 2023; ``refit_best_of_seeds.VISIUM``). The regions of
interest on the H&E are those of the original study, placed by the marker gene it mapped for each (``ROI_MARKERS``). Model:
``scdef-visium-best5`` (``refit_best_of_seeds.py visium``). The network follows scDEF-notebooks/scdef-spatial.ipynb
(cell 87); Moran's I and the neighbourhood enrichment are computed here as squidpy defines them (6 nearest spots,
row-normalised weights; z-scores against label permutations), since squidpy is not in the scdef environment.

    python reproducibility/make_figures/visium_figures.py [--model NAME] [--out DIR]
"""
import argparse
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4_batch as f4  # noqa: E402  (paths)
import refit_best_of_seeds as rb  # noqa: E402  (data)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scdef as scd
from scipy import sparse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
N_NEIGHBOURS = 6
N_PERMS = 1000


def load(name):
    a = rb.tutorial_adata("visium")
    m = scd.scDEF.load(os.path.join(f4.NB, name), adata=a.copy())
    m.annotate_adata()
    scd.tl.set_confident_signatures(m)
    return m


def spot_graph(xy, k=N_NEIGHBOURS):
    """Symmetric adjacency of each spot with its k nearest spots (the Visium hexagonal neighbours)."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(len(xy)), k)
    A = sparse.coo_matrix((np.ones(rows.size), (rows, idx.ravel())), shape=(len(xy), len(xy))).tocsr()
    return ((A + A.T) > 0).astype(float)


def morans_i(A, x):
    """Moran's I with row-normalised weights."""
    W = sparse.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    z = x - x.mean()
    return float(len(x) / W.sum() * (z @ (W @ z)) / (z @ z))


def factor_morans_i(m, A):
    out = {}
    for k, layer in enumerate(m.layer_names):
        P = np.asarray(m.adata.obsm[f"X_{layer}_probs"])
        for j, f in enumerate(m.factor_names[k]):
            if P[:, j].std() > 0:
                out[f] = morans_i(A, P[:, j])
    return out


def nhood_enrichment(A, labels, n_perms=N_PERMS, seed=0):
    """z-score of the number of neighbouring spot pairs between each two labels, against permuted labels."""
    cats = sorted(set(labels), key=lambda s: int(s.split("_")[-1]))
    code = pd.Categorical(labels, categories=cats).codes
    r, c = sparse.triu(A, k=1).nonzero()

    def counts(cd):
        M = np.zeros((len(cats), len(cats)))
        np.add.at(M, (cd[r], cd[c]), 1)
        return M + M.T - np.diag(np.diag(M))

    obs = counts(code)
    rng = np.random.default_rng(seed)
    perm = np.stack([counts(rng.permutation(code)) for _ in range(n_perms)])
    z = (obs - perm.mean(0)) / np.maximum(perm.std(0), 1e-12)
    return pd.DataFrame(z, index=cats, columns=cats)


NETWORK_TOP_LAYER = 3   # the network shows layers 0-3; layer 4 (two factors) only splits tumour from the rest.
                        # Tried with layer 4 on 2026-11-11 and dropped again as less readable (Pedro).
NETWORK_STRETCH = 1.15  # horizontal stretch of the shell layout, so that the network spans the page width at a legible
                        # size and the figure stays under a page (Pedro, 2026-11-11)


def network(m, moran, z, top_layer=NETWORK_TOP_LAYER):
    """The hierarchy up to ``top_layer``, every node filled by its Moran's I, and violet edges between layer-0 factors
    whose spots neighbour each other more than chance (as in the spatial notebook)."""
    keep = {f"L{k}" for k in range(top_layer + 1)}
    hier = {f: c for f, c in scd.tl.get_hierarchy(m, simplified=False).items() if f.split("_")[0] in keep}
    g = scd.pl.make_graph(m, hierarchy=hier, show_signatures=True,
                          top_genes=[4] * m.n_layers, show_label=True, color_edges=False, filled=moran, shell=True,
                          node_size_min=0.5, r=5, r_decay=0.55, gene_cmap="Purples_r")
    n = z.to_numpy() - z.to_numpy().min(axis=1, keepdims=True)
    n = n / n.sum(axis=1, keepdims=True)
    names = list(z.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            g.edge(names[i], names[j], penwidth=str((10 * n[i, j]) ** 4), color="violet")
    g.attr(outputorder="edgesfirst", splines="true")
    pin = re.compile(r'pos="?(-?[0-9.eE+-]+),(-?[0-9.eE+-]+)!"?')
    g.body = [pin.sub(lambda mm: f'pos="{float(mm.group(1)) * NETWORK_STRETCH},{mm.group(2)}!"', l) for l in g.body]
    return g


# Regions of interest: those of the original study (Janesick et al. 2023, Fig. 2c), placed on this section by the
# marker gene that study mapped for each on the same Visium data. They are independent of the scDEF factors.
ROI_MARKERS = {"DCIS #1": "SCGB2A2", "DCIS #2": "CPB1", "Invasive": "CDH2", "Adipose": "FABP4", "Stroma": "SFRP2"}
ROI_COLOURS = {"DCIS #1": "#1b7837", "DCIS #2": "#c51b7d", "Invasive": "#b2182b", "Adipose": "#e08214",
               "Stroma": "#2166ac"}
ROI_HOPS = 2          # each marker is averaged over the spot and its neighbours, this many times
ROI_MIN_Z = 1.0       # a spot belongs to the region whose smoothed marker z-score is largest, if at least this
ROI_MIN_SPOTS = 25    # connected pieces of a region smaller than this are not outlined


def roi_labels(m, A):
    """The region of each spot (empty when none), and the smoothed marker z-scores. Markers are read from the full
    count matrix, since not all of them are among the genes the model was fitted on."""
    import scanpy as sc
    raw = sc.read_visium(rb.VISIUM, count_file="CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5")
    raw.var_names_make_unique()
    raw = raw[m.adata.obs_names]
    lib = np.asarray(raw.X.sum(1)).ravel()
    X = raw[:, list(ROI_MARKERS.values())].X
    X = np.asarray(X.todense() if sparse.issparse(X) else X, dtype=float)
    E = np.log1p(X / lib[:, None] * 1e4)
    W = A + sparse.eye(A.shape[0])
    W = sparse.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
    for _ in range(ROI_HOPS):
        E = W @ E
    Z = (E - E.mean(0)) / E.std(0)
    names = np.array(list(ROI_MARKERS))
    lab = names[np.argmax(Z, 1)].astype(object)
    lab[Z.max(1) < ROI_MIN_Z] = ""
    return lab, Z


def roi_pieces(A, mask, min_spots=ROI_MIN_SPOTS):
    """Connected pieces (on the spot graph) of a region, largest first, keeping those of at least min_spots."""
    from scipy.sparse.csgraph import connected_components
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    n, comp = connected_components(A[idx][:, idx], directed=False)
    pieces = [idx[comp == c] for c in range(n)]
    return sorted([p for p in pieces if len(p) >= min_spots], key=len, reverse=True)


def roi_factors(m, lab):
    """For each region and layer, the factor with the largest mean normalised weight in its spots, and that share."""
    out = {}
    for r in ROI_MARKERS:
        sel = lab == r
        out[r] = dict(n_spots=int(sel.sum()))
        for k, layer in enumerate(m.layer_names[:-1]):
            P = np.asarray(m.adata.obsm[f"X_{layer}_probs"])
            mean = P[sel].mean(0) if sel.any() else np.zeros(P.shape[1])
            order = np.argsort(mean)[::-1][:3]
            out[r][layer] = [(m.factor_names[k][j], round(float(mean[j]), 3)) for j in order]
    return out


def he_image(m):
    lib = list(m.adata.uns["spatial"])[0]
    sp = m.adata.uns["spatial"][lib]
    return sp["images"]["hires"], sp["scalefactors"]["tissue_hires_scalef"]


def capture_box(m, pad=20):
    """The spots' extent in hires-image pixels (the capture area is a square inside a larger section)."""
    img, sf = he_image(m)
    xy = np.asarray(m.adata.obsm["spatial"]) * sf
    x0, y0 = np.maximum(np.floor(xy.min(0) - pad).astype(int), 0)
    x1, y1 = np.ceil(xy.max(0) + pad).astype(int)
    return x0, x1, y0, y1


def he_panel(ax, m, rois=None, A=None, fontsize=5.2):
    """The H&E of the capture area; with ``rois`` (labels from ``roi_labels``) and the spot graph, each region's
    outline (pieces of at least ROI_MIN_SPOTS spots) and its name at its largest piece."""
    img, sf = he_image(m)
    x0, x1, y0, y1 = capture_box(m)
    ax.imshow(img[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
    if rois is not None:
        xy = np.asarray(m.adata.obsm["spatial"]) * sf
        W = A + sparse.eye(A.shape[0])
        W = sparse.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
        for r, col in ROI_COLOURS.items():
            pieces = roi_pieces(A, rois == r)
            if not pieces:
                continue
            ind = np.zeros(len(xy)); ind[np.concatenate(pieces)] = 1.0
            ax.tricontour(xy[:, 0], xy[:, 1], W @ ind, levels=[0.5], colors=[col], linewidths=0.9)
            c = xy[pieces[0]].mean(0)
            j = pieces[0][np.argmin(((xy[pieces[0]] - c) ** 2).sum(1))]
            ax.text(xy[j, 0], xy[j, 1], r, fontsize=fontsize, color=col, ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.axis("off")


def spot_xy(m):
    """Spot centres in hires-image pixels (y grows downwards, as in the image)."""
    _, sf = he_image(m)
    return np.asarray(m.adata.obsm["spatial"]) * sf


TUMOUR_ROIS = ["DCIS #1", "DCIS #2", "Invasive"]   # outlined on the factor maps; the region map shows all five


def spot_size(ax, m, scale=1.15):
    """Marker area (points^2) at which the spots of ``m`` tile ``ax`` (call after the limits are set)."""
    xy = spot_xy(m)
    d = np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1])
    x0, x1 = ax.get_xlim()
    width_pt = ax.get_window_extent().width * 72 / ax.figure.dpi
    return float((scale * d * width_pt / abs(x1 - x0)) ** 2)


def roi_outlines(ax, m, rois, A, color="0.1", lw=0.45, labels=False, fontsize=5.2, regions=None):
    """Each region's outline (connected pieces of at least ROI_MIN_SPOTS spots), drawn on spot coordinates; with
    ``labels``, its name at its largest piece."""
    xy = spot_xy(m)
    W = A + sparse.eye(A.shape[0])
    W = sparse.diags(1.0 / np.asarray(W.sum(1)).ravel()) @ W
    for r in (regions or list(ROI_MARKERS)):
        pieces = roi_pieces(A, rois == r)
        if not pieces:
            continue
        ind = np.zeros(len(xy)); ind[np.concatenate(pieces)] = 1.0
        ax.tricontour(xy[:, 0], xy[:, 1], W @ ind, levels=[0.5], colors=[color], linewidths=lw)
        if labels:
            c = xy[pieces[0]].mean(0)
            j = pieces[0][np.argmin(((xy[pieces[0]] - c) ** 2).sum(1))]
            ax.text(xy[j, 0], xy[j, 1], r, fontsize=fontsize, color="0.1", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def frame(ax, m, pad=20):
    xy = spot_xy(m)
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].max() + pad, xy[:, 1].min() - pad)
    ax.set_aspect("equal")
    ax.axis("off")


def spatial_map(ax, m, values, title, note=None, cmap=scd.pl.FACTOR_CMAP, vmax=None, s=None, fontsize=6,
                rois=None, A=None):
    """Spots coloured by ``values`` (factor scores: the package's factor colormap), with the outlines of the tumour
    regions."""
    xy = spot_xy(m)
    frame(ax, m)
    ax.scatter(xy[:, 0], xy[:, 1], c=values, s=s or spot_size(ax, m), cmap=cmap, vmin=0,
               vmax=vmax or np.quantile(values, 0.99), lw=0, rasterized=True)
    if rois is not None:
        roi_outlines(ax, m, rois, A, color="0.05", lw=0.5, regions=TUMOUR_ROIS)
    frame(ax, m)
    ax.set_title(title, fontsize=fontsize, pad=1.5, linespacing=1.1)
    if note:
        ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=fontsize - 1,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def region_map(ax, m, rois, A, s=None, fontsize=6):
    """The regions of interest (the original study's, placed by its markers), named on the map; the tumour regions,
    outlined on the factor maps, in a darker outline."""
    xy = spot_xy(m)
    frame(ax, m)
    fill = {r: c for r, c in zip(ROI_MARKERS, ["#b9dcc2", "#efc0d9", "#f1aaa3", "#f8d9ad", "#bcd2ec"])}
    col = np.array([fill.get(r, "#efefef") for r in rois], dtype=object)
    ax.scatter(xy[:, 0], xy[:, 1], c=list(col), s=s or spot_size(ax, m), lw=0, rasterized=True)
    roi_outlines(ax, m, rois, A, color="0.55", lw=0.4, regions=[r for r in ROI_MARKERS if r not in TUMOUR_ROIS])
    roi_outlines(ax, m, rois, A, color="0.05", lw=0.5, regions=TUMOUR_ROIS)
    roi_outlines(ax, m, rois, A, color="none", lw=0, labels=True, fontsize=fontsize - 0.6)
    frame(ax, m)
    ax.set_title("Regions of interest", fontsize=fontsize, pad=1.5)


def layer_maps(m, k, moran, path, cols=4, rois=None, A=None):
    """Spatial maps of every factor of layer ``k`` (supplement)."""
    sig = m.get_signatures_dict(top_genes=3)
    names = list(m.factor_names[k])
    P = np.asarray(m.adata.obsm[f"X_{m.layer_names[k]}_probs"])
    rows = int(np.ceil(len(names) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(7.2, 1.95 * rows))
    axs = np.atleast_1d(axs).ravel()
    for j, f in enumerate(names):
        spatial_map(axs[j], m, P[:, j], f"{f}\n{' '.join(sig[f][:3])}", note=f"I = {moran.get(f, np.nan):.2f}",
                    rois=rois, A=A)
    for ax in axs[len(names):]:
        ax.axis("off")
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def fig7(m, moran, out, k_maps=3, rois=None, A=None):
    """Main-text figure: (A) the spatial maps of the layer-``k_maps`` factors on one colour scale, without
    annotations (Pedro, 2026-11-10: the factors annotate the regions; the text compares them with the original
    study's); (B) the network of layers 0-3 with Moran's I and neighbourhood enrichment. ``rois`` is unused here
    and kept for the numbers file."""
    plt.rcParams.update({"pdf.fonttype": 42})
    net = plt.imread(os.path.join(out, "visium_network.png"))
    names = list(m.factor_names[k_maps])
    P = np.asarray(m.adata.obsm[f"X_{m.layer_names[k_maps]}_probs"])
    vmax = float(np.quantile(P, 0.995))
    h_maps, w_net = 1.62, 7.2
    h_net = w_net * net.shape[0] / net.shape[1]
    fig = plt.figure(figsize=(7.2, h_maps + h_net + 0.25))
    G = GridSpec(2, 1, figure=fig, height_ratios=[h_maps, h_net], hspace=0.04, left=0.005, right=0.995,
                 top=1 - 0.2 / (h_maps + h_net + 0.25), bottom=0.005)
    gA = G[0].subgridspec(1, len(names) + 1, width_ratios=[1] * len(names) + [0.3], wspace=0.06)
    sig = m.get_signatures_dict(top_genes=2)
    axs = []
    for j, f in enumerate(names):
        ax = fig.add_subplot(gA[j]); axs.append(ax)
        spatial_map(ax, m, P[:, j], f"{f}  {' '.join(sig[f][:2])}", vmax=vmax, fontsize=6)
    axR = axs[0]
    cax = fig.add_subplot(gA[-1])
    pos = cax.get_position(); cax.set_position([pos.x0 + 0.05 * pos.width, pos.y0 + 0.12 * pos.height,
                                                pos.width * 0.22, 0.72 * pos.height])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=scd.pl.FACTOR_CMAP), cax=cax)
    cb.set_label(f"share of layer-{k_maps} weight", fontsize=5.2, labelpad=2); cb.ax.tick_params(labelsize=5, length=1.5)
    axB = fig.add_subplot(G[1]); axB.imshow(net, interpolation="lanczos"); axB.axis("off")
    # legends in the lower-right corner, which the shell layout leaves empty
    cax2 = axB.inset_axes([0.83, 0.115, 0.15, 0.016])
    cb2 = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="Purples_r"), cax=cax2,
                       orientation="horizontal")
    cb2.ax.xaxis.set_label_position("top")
    cb2.set_label("node colour: Moran's I of the factor", fontsize=5.4, labelpad=2)
    cb2.ax.tick_params(labelsize=5, length=1.5)
    axB.plot([0.83, 0.98], [0.035, 0.035], color="violet", lw=2, transform=axB.transAxes, clip_on=False)
    axB.text(0.905, 0.045, "edge: neighbourhood enrichment\nbetween layer-0 factors", ha="center", va="bottom",
             fontsize=5.4, transform=axB.transAxes)
    for ax, letter in ((axR, "A"), (axB, "B")):
        bb = ax.get_position()
        fig.text(0.006, bb.y1 + (0.004 if ax is axR else -0.01), letter, fontsize=10, fontweight="bold", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig7_visium.{ext}"), dpi=600 if ext == "pdf" else 300)
    plt.close(fig)


def main(name, out):
    os.makedirs(out, exist_ok=True)
    m = load(name)
    A = spot_graph(np.asarray(m.adata.obsm["spatial"]))
    moran = factor_morans_i(m, A)
    z = nhood_enrichment(A, m.adata.obs["L0"].astype(str).to_numpy())
    g = network(m, moran, z)
    g.format = "pdf"
    g.render(os.path.join(out, "visium_network"), cleanup=True)
    g.format = "png"
    g.graph_attr.update(dpi="200")
    g.render(os.path.join(out, "visium_network"), cleanup=True)
    rois, _ = roi_labels(m, A)
    for k in (2, 3):
        layer_maps(m, k, moran, os.path.join(out, f"visium_layer{k}_maps.png"))
    fig7(m, moran, out, rois=rois, A=A)
    numbers = dict(rois=dict(markers=ROI_MARKERS, hops=ROI_HOPS, min_z=ROI_MIN_Z, min_spots=ROI_MIN_SPOTS,
                             pieces={r: [len(p) for p in roi_pieces(A, rois == r)] for r in ROI_MARKERS},
                             factors=roi_factors(m, rois)),
                   layer_sizes=[len(f) for f in m.factor_lists], morans_i={k_: round(v, 3) for k_, v in moran.items()},
                   nhood_enrichment_L0=z.round(2).to_dict(), signatures={f: list(v) for f, v in m.get_signatures_dict(top_genes=8).items()})
    json.dump(numbers, open(os.path.join(out, "visium_numbers.json"), "w"), indent=1)
    print(json.dumps(dict(layer_sizes=numbers["layer_sizes"],
                          top_moran=sorted(moran.items(), key=lambda kv: -kv[1])[:8]), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="scdef-visium-best5")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "visium"))
    a_ = ap.parse_args()
    main(a_.model, a_.out)
