# Standard library imports
import logging
import time
import json
import pickle
from pathlib import Path
from typing import Optional, Union, Sequence, Mapping, Dict, List, Tuple, Any

# Third-party imports
import jax
import jax.numpy as jnp
from jax import jit, vmap, random, value_and_grad
from jax.scipy.stats import poisson
import optax
import tensorflow_probability.substrates.jax.distributions as tfd

import numpy as np
import pandas as pd
import scipy

import matplotlib
import seaborn as sns
from tqdm import tqdm

from anndata import AnnData
import anndata as ad

# Local imports
from scdef.utils import score_utils, color_utils
from scdef.utils.jax_utils import lognormal_sample, lognormal_entropy, gamma_logpdf

# Configure logging
logging.basicConfig()


class scDEF(object):
    """Single-cell Deep Exponential Families (scDEF) model.

    scDEF learns hierarchical, multi-level gene expression signatures from single-cell
    RNA-seq data provided in an AnnData object. This model can be used for a variety
    of analyses including dimensionality reduction, batch correction, clustering,
    and visualization of cell states and gene programs.

    The model fits multiple layers of latent factors ("gene signatures") to describe
    cellular heterogeneity at different resolutions. It supports batch correction,
    prior specification, and generation of corrected gene expression matrices.

    Model fitting, inference routines, and additional plotting utilities are
    implemented as methods of this class. The stored AnnData object is updated with
    model results during training.

    Args:
        adata: AnnData object containing the single-cell gene expression count matrix. Counts
            should be present in either `adata.X` or in the specified `adata.layers`.
        counts_layer: key for `adata.layers` specifying which layer to use as expression counts (if not `adata.X`).
        batch_key: key in `adata.obs` containing batch annotations; if provided, batch correction is performed.
            If None or not found, no batch correction is used.
        seed: random seed for model initialization and stochastic routines (uses JAX's pseudo-random number generator).
        n_factors: number of latent factors at the lowest layer (L0), denoted ``K_0`` in
            the geometric layer-size schedule when ``layer_sizes`` is None.
        n_layers: number of layers in the geometric schedule from L0 through the layer of
            width ``top_factors`` when ``layer_sizes`` is None. A size-``1`` root appended
            when ``top_factors > 1`` does not count toward ``n_layers``.
        top_factors: target width at the coarsest non-root layer (default ``1``), used with
            ``n_factors`` and ``n_layers`` in the geometric ladder
            ``K_l = K_0 * (K_top / K_0) ** (l / (n_layers - 1))`` for ``l = 0, ..., n_layers - 1``
            (last rung fixed at ``K_top``). When ``top_factors > 1``, a final root layer of
            width ``1`` is appended and is not included in ``n_layers``.
        layer_sizes: explicit list of the number of factors in each scDEF layer. If None, layer sizes are set automatically.
        layer_names: list of custom names for the layers. If None, layer names are enumerated as ["L0", "L1", ...].
        logginglevel: verbosity level for the logger.
        alpha: concentration parameter for the Gamma prior on z.
        kappa: rate scaling for the Gamma prior on z (non-top layers).
        shrinkage_shape: shape parameter for shrinkage prior controlling factor usage.
        shrinkage_rate: rate parameter for shrinkage prior controlling factor usage.
        shrinkage_mean: target prior mean for shrinkage/factor relevance.
        top_alpha: concentration parameter for the top layer Dirichlet prior over factor proportions.
        factor_shape: shape of the prior distribution for factor-gene loadings matrix W.
        brd_strength: BRD (Batch Relevance Determination) prior concentration parameter for factor relevance estimation.
        brd_mean: mean of the BRD prior for factor relevance estimation.
        use_brd: if True, use BRD prior for automatic selection of active factors.
        cell_scale_shape: precision/concentration parameter for cell-specific scaling priors.
        gene_scale_shape: precision/concentration parameter for gene-specific scaling priors.
        batch_cpal: default matplotlib color palette name used for batches.
        layer_cpal: matplotlib color palette for factors/colors at each scDEF layer.
        lightness_mult: lightness multiplier to define the base color for each new scDEF layer.
        set_alpha_from_cov: if True, set the alpha parameter from the data coverage.
        marginalize_alpha: if True, infer alpha from a variational posterior during training.
    """

    def make_corrected_data(self, layer_name: str = "scdef_corrected") -> None:
        """Compute and store the low-rank reconstruction of the UMI count matrix.

        The reconstructed matrix is saved to adata.layers[layer_name], providing a
        denoised, batch-corrected version of the expression data.

        Args:
            layer_name: name for the AnnData layer where the reconstructed matrix is stored
        """
        scdef_layer = self.layer_names[0]
        Z = self.pmeans[f"{scdef_layer}z"][:, self.factor_lists[0]]  # nxk
        W = self.pmeans[f"{scdef_layer}W"][self.factor_lists[0]]  # kxg
        self.adata.layers[layer_name] = np.array(Z.dot(W))

    def __init__(
        self,
        adata: AnnData,
        counts_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        seed: Optional[int] = 42,
        n_factors: Optional[int] = 100,
        top_factors: int = 1,
        n_layers: Optional[float] = 6,
        layer_sizes: Optional[list] = None,
        layer_names: Optional[list] = None,
        logginglevel: Optional[int] = logging.INFO,
        alpha: Optional[float] = 1.0,
        kappa: Optional[float] = 10.0,
        shrinkage_shape: Optional[float] = 1.0,
        shrinkage_rate: Optional[float] = 1.0,
        shrinkage_mean: Optional[float] = 1.0,
        top_alpha: Optional[float] = 1.0,
        factor_shape: Optional[float] = 0.1,
        brd_strength: Optional[float] = 1.0,
        brd_mean: Optional[float] = 1.0,
        use_brd: Optional[bool] = True,
        cell_scale_shape: Optional[float] = 1.0,
        gene_scale_shape: Optional[float] = 1.0,
        batch_cpal: Optional[str] = "Dark2",
        layer_cpal: Optional[str] = "tab10",
        lightness_mult: Optional[float] = 0.15,
        set_alpha_from_cov: Optional[bool] = True,
        marginalize_alpha: Optional[bool] = False,
    ):
        self.n_cells, self.n_genes = adata.shape

        self.n_batches = 1
        self.batches = [""]
        self.batch_key = batch_key

        self.seed = seed
        self.batch_cpal = batch_cpal
        self.layer_cpal = layer_cpal
        self.lightness_mult = lightness_mult

        self.logger = logging.getLogger("scDEF")
        self.logger.setLevel(logginglevel)

        self.load_adata(adata, layer=counts_layer, batch_key=batch_key)

        self.top_alpha = top_alpha
        self.alpha = alpha
        self.kappa = kappa
        self.factor_shape = factor_shape
        self.brd = brd_strength
        self.brd_mean = brd_mean
        self.shrinkage_shape = shrinkage_shape
        self.shrinkage_rate = shrinkage_rate
        self.shrinkage_mean = shrinkage_mean
        self.cell_scale_shape = cell_scale_shape
        self.gene_scale_shape = gene_scale_shape
        self.use_brd = use_brd
        self.set_alpha_from_cov = set_alpha_from_cov
        self.marginalize_alpha = bool(marginalize_alpha)
        self._marginalize_alpha_init = bool(marginalize_alpha)

        if n_layers is None:
            n_layers = 6.0
        self.n_layers_schedule = n_layers
        k0 = int(n_factors if n_factors is not None else 100)
        if k0 < 1:
            raise ValueError("n_factors must be >= 1.")
        self.top_factors = int(top_factors)
        if self.top_factors < 1:
            raise ValueError("top_factors must be >= 1.")
        if self.top_factors > k0:
            raise ValueError(
                "top_factors must be <= n_factors (layer sizes are non-increasing "
                "from L0 toward the top layer)."
            )
        self.n_factors = k0

        if layer_sizes is not None:
            self.layer_sizes = [int(x) for x in layer_sizes]
            self.n_factors = int(layer_sizes[0])
            self.n_layers = len(self.layer_sizes)
        else:
            self.update_model_size(
                n_layers=int(self.n_layers_schedule),
                use_decay_factor_schedule=False,
            )

        if layer_names is not None:
            self.layer_names = layer_names
        else:
            self.layer_names = [f"L{i}" for i in range(self.n_layers)]
        self.factor_lists = [np.arange(size) for size in self.layer_sizes]
        self.set_factor_names()

        self.update_model_priors()

        self.make_layercolors(layer_cpal=self.layer_cpal, lightness_mult=lightness_mult)

        self.init_var_params(nmf_init=False)  # just to get stub
        self.set_posterior_means()
        self.set_posterior_variances()
        self._has_fit = False
        self._fit_revision = 0
        self._learn_jit_cache = None

    def __repr__(self):
        out = f"scDEF object with {self.n_layers} layers"
        out += (
            "\n\t"
            + "Layer names: "
            + ", ".join([f"{name}" for name in self.layer_names])
        )
        out += (
            "\n\t"
            + "Layer sizes: "
            + ", ".join([str(len(factors)) for factors in self.factor_lists])
        )
        out += "\n\t" + "alpha parameter: " + str(self.alpha)
        if self.use_brd:
            out += "\n\t" + "Using BRD"
        out += "\n\t" + "Number of batches: " + str(self.n_batches)
        out += "\n" + "Contains " + self.adata.__str__()
        return out

    def save(
        self,
        dir_path: Union[str, Path],
        overwrite: bool = False,
        save_anndata: bool = False,
    ) -> None:
        """Save model state to disk, similarly to scvi-tools.

        This writes a model state pickle plus metadata to ``dir_path``. AnnData
        is saved separately as ``adata.h5ad`` only when ``save_anndata=True``.

        Args:
            dir_path: output directory path
            overwrite: whether to overwrite an existing non-empty directory
            save_anndata: whether to save ``model.adata`` as ``adata.h5ad``
        """
        out_dir = Path(dir_path).expanduser().resolve()
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Directory '{out_dir}' already exists and is not empty. "
                "Set overwrite=True to overwrite."
            )
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_anndata:
            self.adata.write(out_dir / "adata.h5ad")

        state = dict(self.__dict__)
        # Graphviz graph objects are environment-dependent and not required for
        # restoring model state.
        if "graph" in state:
            state["graph"] = None
        # JIT-compiled callables are runtime-only and not serializable.
        if "_learn_jit_cache" in state:
            state["_learn_jit_cache"] = None
        logger_level = (
            self.logger.level
            if hasattr(self, "logger") and self.logger is not None
            else None
        )
        state["logger"] = None

        payload = {
            "state": state,
            "class_name": self.__class__.__name__,
            "module": self.__class__.__module__,
            "save_anndata": bool(save_anndata),
            "logger_level": logger_level,
        }
        with open(out_dir / "model_state.pkl", "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        metadata = {
            "class_name": payload["class_name"],
            "module": payload["module"],
            "save_anndata": payload["save_anndata"],
            "files": ["model_state.pkl"] + (["adata.h5ad"] if save_anndata else []),
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        dir_path: Union[str, Path],
        adata: Optional[AnnData] = None,
    ) -> "scDEF":
        """Load model from disk.

        Args:
            dir_path: directory created by :meth:`save`
            adata: optional AnnData to attach when ``adata.h5ad`` was not saved.

        Returns:
            Loaded model instance.
        """
        in_dir = Path(dir_path).expanduser().resolve()
        state_path = in_dir / "model_state.pkl"
        if not state_path.exists():
            raise FileNotFoundError(f"Model state file not found: '{state_path}'.")

        with open(state_path, "rb") as f:
            try:
                payload = pickle.load(f)
            except ModuleNotFoundError as e:
                # Compatibility path for artifacts pickled under a different
                # NumPy internal module layout (e.g. numpy._core vs numpy.core).
                if "numpy._core" not in str(e):
                    raise
                f.seek(0)

                class _GraphPlaceholder:
                    """Fallback placeholder for optional Graphviz graph classes."""

                    def __init__(self, *args, **kwargs):
                        self.args = args
                        self.kwargs = kwargs

                class _CompatUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith("numpy._core"):
                            module = module.replace("numpy._core", "numpy.core", 1)
                        if module == "numpy.rec":
                            module = "numpy.core.records"
                        if module == "graphviz.graphs" and name in {"Graph", "Digraph"}:
                            return _GraphPlaceholder
                        return super().find_class(module, name)

                payload = _CompatUnpickler(f).load()

        obj = cls.__new__(cls)
        obj.__dict__.update(payload["state"])
        obj._learn_jit_cache = None
        if not hasattr(obj, "marginalize_alpha"):
            obj.marginalize_alpha = False
        if not hasattr(obj, "_marginalize_alpha_init"):
            obj._marginalize_alpha_init = bool(obj.marginalize_alpha)
        obj.logger = logging.getLogger(payload.get("class_name", cls.__name__))
        if payload.get("logger_level") is not None:
            obj.logger.setLevel(payload["logger_level"])

        adata_path = in_dir / "adata.h5ad"
        if adata is not None:
            obj.adata = adata
        elif getattr(obj, "adata", None) is not None:
            # Prefer embedded AnnData saved in the model state.
            pass
        elif payload.get("save_anndata", False) and adata_path.exists():
            obj.adata = ad.read_h5ad(adata_path)
        else:
            raise ValueError(
                "No AnnData available. Provide `adata` to load(...), or save with "
                "`save_anndata=True` so `adata.h5ad` is available."
            )

        obj.n_cells, obj.n_genes = obj.adata.shape
        return obj

    def make_layercolors(
        self,
        layer_cpal: Optional[str] = "tab10",
        lightness_mult: Optional[float] = 0.15,
    ):
        if isinstance(layer_cpal, str):
            layer_cpal = [layer_cpal] * self.n_layers
        elif isinstance(layer_cpal, list):
            if len(layer_cpal) != self.n_layers:
                raise ValueError("layer_cpal list must be of size scDEF.n_layers")
        else:
            raise TypeError("layer_cpal must be either a cmap str or a list of strs")

        layer_sizes = []
        for i in range(self.n_layers):
            layer_size = len(self.factor_lists[i])
            if layer_size > 10:
                layer_cpal[i] = "tab20"
            elif layer_size > 20:
                layer_cpal[i] = "hsl"
            layer_sizes.append(layer_size)

        self.layer_colorpalettes = [
            sns.color_palette(layer_cpal[idx], n_colors=size)
            for idx, size in enumerate(layer_sizes)
        ]

        if len(np.unique(layer_cpal)) == 1:
            # Make the layers have different lightness
            for layer_idx, size in enumerate(layer_sizes):
                for factor_idx in range(size):
                    col = self.layer_colorpalettes[layer_idx][factor_idx]
                    self.layer_colorpalettes[layer_idx][
                        factor_idx
                    ] = color_utils.adjust_lightness(
                        col, amount=1.0 + lightness_mult * layer_idx
                    )

    def load_adata(self, adata, layer=None, batch_key=None):
        if not isinstance(adata, AnnData):
            raise TypeError("adata must be an instance of AnnData.")
        self.adata = adata.copy()
        self.adata.raw = self.adata
        X = self.adata.raw.X
        if layer is not None:
            X = self.adata.layers[layer]

        if isinstance(X, scipy.sparse.csr_matrix):
            self.X = np.array(X.toarray())
        else:
            self.X = np.array(X)

        self.X = self.X.astype(float)

        self.batch_indices_onehot = np.ones((self.adata.shape[0], 1))
        self.batch_lib_sizes = np.sum(self.X, axis=1)
        self.batch_lib_ratio = (
            np.ones((self.X.shape[0], 1))
            * np.mean(self.batch_lib_sizes)
            / np.var(self.batch_lib_sizes)
        )
        self.exposure = jnp.array(self.batch_lib_sizes / np.mean(self.batch_lib_sizes))

        eps = 1e-12

        # ---- NEW: gene_ratio is per-gene and encodes baseline abundance ----
        # Choose a baseline definition. For raw UMI counts, using mean counts per cell is reasonable.
        # If you prefer library-normalized baseline, replace mu = X.mean(0) by mu = (X/lib).mean(0).
        mu = self.X.mean(axis=0) + eps
        mu_bar = float(mu.mean())
        # Prior mean of gene_scale_g will be ~ mu_g / mu_bar
        gene_ratio = mu_bar / mu  # vector length G
        gene_ratio = np.array(gene_ratio)[None, :]  # shape (1, G)
        self.gene_ratio_init = (
            gene_ratio  # will be overridden per batch if batch_key has >1 values
        )

        gene_size = np.sum(self.X, axis=0)
        self.gene_means = np.mean(gene_size)
        self.gene_vars = np.var(gene_size)
        self.gene_ratio = self.gene_means / self.gene_vars

        if batch_key is not None:
            if batch_key in self.adata.obs.columns:
                batches = np.unique(self.adata.obs[batch_key].values)
                self.batches = batches
                self.n_batches = len(batches)
                self.logger.info(
                    f"Found {self.n_batches} values for `{batch_key}` in data: {batches}"
                )
                if f"{batch_key}_colors" in self.adata.uns:
                    self.batch_colors = self.adata.uns[f"{batch_key}_colors"]
                else:
                    batch_colorpalette = sns.color_palette(
                        self.batch_cpal, n_colors=self.n_batches
                    )
                    self.batch_colors = [
                        matplotlib.colors.to_hex(batch_colorpalette[idx])
                        for idx in range(self.n_batches)
                    ]
                    self.adata.uns[f"{batch_key}_colors"] = self.batch_colors

                self.batch_indices_onehot = np.zeros(
                    (self.adata.shape[0], self.n_batches)
                )
                # If multiple batches: compute gene_ratio per batch (still per gene)
                if self.n_batches > 1:
                    self.gene_ratio = np.ones((self.n_batches, self.adata.shape[1]))
                    self.gene_ratio_init = np.ones(
                        (self.n_batches, self.adata.shape[1])
                    )
                    # self.gene_ratio_init = np.tile(
                    #     (mu_bar / mu)[None, :], (self.n_batches, 1)
                    # )
                    for i, b in enumerate(batches):
                        cells = np.where(self.adata.obs[batch_key] == b)[0]
                        self.batch_indices_onehot[cells, i] = 1
                        self.batch_lib_sizes[cells] = np.sum(self.X, axis=1)[cells]
                        self.batch_lib_ratio[cells] = np.mean(
                            self.batch_lib_sizes[cells]
                        ) / (np.var(self.batch_lib_sizes[cells]) + eps)
                        mu_b = self.X[cells].mean(axis=0) + eps
                        mu_b_bar = float(mu_b.mean())
                        self.gene_ratio_init[i] = mu_b_bar / mu_b
                        gene_size = np.sum(self.X[cells], axis=0)
                        gene_means = np.mean(gene_size)
                        gene_vars = np.var(gene_size)
                        self.gene_ratio[i] = (
                            self.gene_ratio[i] * 0 + gene_means / gene_vars
                        )

        self.batch_indices_onehot = jnp.array(self.batch_indices_onehot)
        self.batch_lib_sizes = jnp.array(self.batch_lib_sizes)
        self.batch_lib_ratio = jnp.array(self.batch_lib_ratio)
        self.gene_ratio = jnp.array(self.gene_ratio)

    def _geometric_layer_sizes(self, n_layers: int) -> List[int]:
        """Layer counts from ``n_factors`` (L0) through the ``top_factors`` layer, optional root.

        ``n_layers`` counts only layers from K0 up to and including the ``top_factors``
        width; a final root of width ``1`` (when ``top_factors > 1``) is appended and does
        not count toward ``n_layers``.

        For ``n_layers >= 3``, uses
        ``K_l = K_0 * (K_top / K_0) ** (l / (n_layers - 1))`` for ``l = 0, ..., n_layers - 2``,
        with the last geometric layer set to ``top_factors`` exactly, then appends ``1`` if
        ``top_factors > 1``.

        For ``n_layers == 2``: ``[n_factors, top_factors]``, then appends ``1`` if
        ``top_factors > 1``.
        """
        n_layers = max(2, int(n_layers))
        min_k = float(self.n_factors)
        top_k = float(self.top_factors)
        ratio = top_k / max(min_k, 1.0)
        add_root = int(self.top_factors) > 1
        if n_layers == 2:
            out = [max(1, int(round(min_k))), int(self.top_factors)]
            if out[0] < out[1]:
                out[0] = out[1]
            if add_root:
                out.append(1)
            return out

        denom = float(n_layers - 1)
        out: List[int] = []
        prev: Optional[int] = None
        for l in range(n_layers):
            r = l / denom
            if l == n_layers - 1:
                k_i = int(self.top_factors)
            else:
                k_f = min_k * (ratio**r)
                k_i = max(1, int(round(k_f)))
                if prev is not None and k_i >= prev:
                    k_i = max(1, prev - 1)
            out.append(k_i)
            prev = k_i
        out[-1] = int(self.top_factors)
        for i in range(len(out) - 2, -1, -1):
            if out[i] < out[i + 1]:
                out[i] = out[i + 1]
        if add_root:
            out.append(1)
        return out

    def update_model_size(
        self,
        max_n_factors=None,
        n_layers=None,
        layer_sizes=None,
        use_decay_factor_schedule: bool = False,
    ):
        """Update latent hierarchy dimensions.

        Args:
            max_n_factors: bottom-layer factor count when ``use_decay_factor_schedule``
                is True (``iscDEF`` marker layer 0 path).
            n_layers: target number of geometric layers from K0 through ``top_factors`` (a
                final root of size ``1`` when ``top_factors > 1`` is not counted), or maximum
                layers for the decay schedule.
            layer_sizes: explicit per-layer sizes. If provided, sizes are
                sanitized to be non-increasing and consecutive duplicates are
                collapsed.
            use_decay_factor_schedule: if True, use ``decay_factor``-based halving
                (``iscDEF`` only). If False, use ``n_factors``, ``top_factors``, and
                ``n_layers_schedule`` on ``self`` for a geometric ladder (``scDEF``).
        """
        if layer_sizes is not None:
            # If two consecutive layers have increasing sizes, clip
            for i in range(len(layer_sizes) - 1):
                if layer_sizes[i + 1] > layer_sizes[i]:
                    layer_sizes[i + 1] = layer_sizes[i]
            # If consecutive layers have the same size, keep only one.
            # Build a deduplicated list directly so chains like [...,1,1,1]
            # collapse to a single 1.
            dedup_layer_sizes = []
            for size in layer_sizes:
                if len(dedup_layer_sizes) == 0 or size != dedup_layer_sizes[-1]:
                    dedup_layer_sizes.append(size)
            layer_sizes = dedup_layer_sizes
            self.layer_sizes = layer_sizes
            self.n_layers = len(self.layer_sizes)
            self.layer_names = [f"L{i}" for i in range(self.n_layers)]
            self.factor_lists = [np.arange(size) for size in self.layer_sizes]
            self.set_factor_names()
            return

        if use_decay_factor_schedule:
            if n_layers is None:
                n_layers = self.n_layers_schedule
            if max_n_factors is None:
                raise ValueError(
                    "max_n_factors is required when use_decay_factor_schedule=True."
                )
            df = getattr(self, "decay_factor", None)
            if df is None or float(df) <= 0:
                raise ValueError(
                    "decay_factor must be set on the model when use_decay_factor_schedule=True."
                )
            n_factors = int(max_n_factors)
            self.layer_sizes = []
            self.layer_sizes.append(n_factors)
            if int(n_layers) > 1:
                while len(self.layer_sizes) < int(n_layers):
                    n_factors = int(np.floor(n_factors / float(df)))
                    self.layer_sizes.append(n_factors)
                    if n_factors == 1:
                        break
                if self.layer_sizes[-1] > 1:
                    self.layer_sizes[-1] = 1

            self.n_layers = len(self.layer_sizes)
            self.layer_names = [f"L{i}" for i in range(self.n_layers)]
            self.factor_lists = [np.arange(size) for size in self.layer_sizes]
            self.set_factor_names()
            return

        # scDEF geometric default
        if n_layers is None:
            n_layers = self.n_layers_schedule
        self.layer_sizes = self._geometric_layer_sizes(int(n_layers))
        self.n_layers = len(self.layer_sizes)
        self.n_factors = int(self.layer_sizes[0])
        self.layer_names = [f"L{i}" for i in range(self.n_layers)]
        self.factor_lists = [np.arange(size) for size in self.layer_sizes]
        self.set_factor_names()

    def update_model_priors(self):
        if self.use_brd:
            self.factor_shapes = [1.0] + [
                self.factor_shape  # * self.layer_sizes[layer_idx] / self.layer_sizes[0]
                for layer_idx in range(1, self.n_layers)
            ]
        else:
            self.factor_shapes = [
                self.factor_shape  # * self.layer_sizes[layer_idx] / self.layer_sizes[0]
                for layer_idx in range(self.n_layers)
            ]

        self.w_priors = []
        for idx in range(self.n_layers):
            prior_shapes = self.factor_shapes[idx]
            prior_rates = self.factor_shapes[idx]

            if idx > 0:
                prior_shapes = (
                    np.ones((self.layer_sizes[idx], self.layer_sizes[idx - 1]))
                    * self.factor_shapes[idx]
                    * 1.0
                )
                prior_rates = (
                    np.ones((self.layer_sizes[idx], self.layer_sizes[idx - 1]))
                    * self.factor_shapes[idx]
                    * 1.0
                    / self.kappa
                )
                prior_shapes = jnp.clip(jnp.array(prior_shapes), 1e-12, 1e12)
                prior_rates = jnp.clip(jnp.array(prior_rates), 1e-12, 1e12)

            self.w_priors.append([prior_shapes, prior_rates])

        self.cell_alpha_factor = self.batch_lib_sizes / float(
            np.median(self.batch_lib_sizes)
        )
        if self.set_alpha_from_cov:
            self.alpha = float(np.median(self.batch_lib_sizes)) / float(
                self.layer_sizes[0]
            )

    def get_effective_factors(
        self,
        brd_min: Optional[float] = 1.0,
        ard_min: Optional[float] = 0.001,
        clarity_min: Optional[float] = 0.5,
        n_eff_parents_max: float = 1.5,
        local_l0_scores: bool = False,
        min_cells: Optional[float] = 0.001,
    ):
        layer_name = self.layer_names[0]
        if min_cells != 0:
            min_cells = (
                np.maximum(10, min_cells * self.n_cells)
                if min_cells < 1.0
                else min_cells
            )

        normed = (
            self.pmeans[f"{layer_name}z"]
            / np.sum(self.pmeans[f"{layer_name}z"], axis=1)[:, None]
        )
        assignments = np.argmax(normed, axis=1)
        counts = np.array(
            [np.count_nonzero(assignments == a) for a in range(self.layer_sizes[0])]
        )

        keep = np.array(range(self.layer_sizes[0]))[np.where(counts >= min_cells)[0]]
        if self.use_brd:
            required_cols = {"BRD", "ARD", "original_factor_idx"}

            def _has_required(key):
                return key in self.adata.uns and required_cols.issubset(
                    set(self.adata.uns[key].columns)
                )

            if not _has_required("factor_obs_full") and not _has_required("factor_obs"):
                from scdef.tools.factor import factor_diagnostics

                factor_diagnostics(self)

            # Prefer the full (pre-filter) snapshot so re-filtering with
            # looser thresholds can reconsider previously dropped factors.
            source_key = (
                "factor_obs_full" if _has_required("factor_obs_full") else "factor_obs"
            )
            factor_obs = self.adata.uns[source_key]
            if "child_layer" in factor_obs.columns:
                factor_obs_l0 = factor_obs[
                    factor_obs["child_layer"] == self.layer_names[0]
                ]
            else:
                factor_obs_l0 = factor_obs

            original_idx = factor_obs_l0["original_factor_idx"].to_numpy(dtype=int)
            brd_vals = factor_obs_l0["BRD"].to_numpy(dtype=float)
            ard_vals = factor_obs_l0["ARD"].to_numpy(dtype=float)

            valid_idx = (original_idx >= 0) & (original_idx < self.layer_sizes[0])
            original_idx_f = original_idx[valid_idx]
            brd_vals = brd_vals[valid_idx]
            ard_vals = ard_vals[valid_idx]

            brd = np.full(self.layer_sizes[0], np.nan, dtype=float)
            ard = np.full(self.layer_sizes[0], np.nan, dtype=float)
            brd[original_idx_f] = brd_vals
            ard[original_idx_f] = ard_vals
            ard_sum = np.nansum(ard)

            if local_l0_scores:
                clarity_vals = factor_obs_l0["clarity_score_01"].to_numpy(dtype=float)[
                    valid_idx
                ]
                clarity = np.full(self.layer_sizes[0], np.nan, dtype=float)
                clarity[original_idx_f] = clarity_vals
                valid = np.isfinite(brd) & np.isfinite(ard) & np.isfinite(clarity)
                tree_ok = clarity >= float(clarity_min)
            elif "avg_n_eff_parents" in factor_obs_l0.columns:
                avg_vals = factor_obs_l0["avg_n_eff_parents"].to_numpy(dtype=float)[
                    valid_idx
                ]
                avg_neff = np.full(self.layer_sizes[0], np.nan, dtype=float)
                avg_neff[original_idx_f] = avg_vals
                valid = np.isfinite(brd) & np.isfinite(ard) & np.isfinite(avg_neff)
                tree_ok = avg_neff < float(n_eff_parents_max)
            else:
                clarity_vals = factor_obs_l0["clarity_score_01"].to_numpy(dtype=float)[
                    valid_idx
                ]
                clarity = np.full(self.layer_sizes[0], np.nan, dtype=float)
                clarity[original_idx_f] = clarity_vals
                valid = np.isfinite(brd) & np.isfinite(ard) & np.isfinite(clarity)
                tree_ok = clarity >= float(clarity_min)

            brd_keep = np.where(
                valid & (brd >= brd_min) & tree_ok & (ard >= ard_min * ard_sum)
            )[0]
            keep = np.unique(list(set(brd_keep).intersection(keep)))

        return keep

    def init_var_params(
        self,
        init_budgets=True,
        init_alpha=True,
        init_z=None,
        init_w=None,
        init_brd=None,
        init_ard=None,
        nmf_init=False,
        z_init_concentration=0.5,
        **kwargs,
    ):
        rngs = random.split(random.PRNGKey(self.seed), self.n_layers)
        init_z_provided = init_z is not None

        def _get_layer_init(init_value, layer_idx):
            if init_value is None:
                return None
            if isinstance(init_value, (list, tuple)):
                if layer_idx >= len(init_value) or init_value[layer_idx] is None:
                    return None
                return np.asarray(init_value[layer_idx], dtype=np.float32)
            if layer_idx == 0:
                return np.asarray(init_value, dtype=np.float32)
            return None

        if init_budgets:
            m = np.array(self.batch_lib_sizes / np.mean(self.batch_lib_sizes))[
                :, None
            ]  # (N,1)
            m = np.clip(m, 1e-3, 1e2)
            v = m / 10.0
            self.local_params = [
                jnp.array(
                    (
                        jnp.log(m**2 / jnp.sqrt(m**2 + v))
                        * jnp.ones((self.n_cells, 1)),  # cell_scales
                        jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2))))
                        * jnp.ones((self.n_cells, 1)),
                    )
                ),
            ]
            # initialize gene scales at the prior mean (per gene)
            # prior mean under Gamma(shape, rate=shape*gene_ratio) is 1/gene_ratio
            m = 1.0 / np.array(self.gene_ratio_init)  # shape: (G,) or (B,G)
            m = np.clip(m, 1e-6, 1e6)
            v = m / 10.0

            m = jnp.array(m, dtype=jnp.float32)
            v = jnp.array(v, dtype=jnp.float32)

            self.global_params = [
                jnp.array(
                    (
                        jnp.log(m**2 / jnp.sqrt(m**2 + v)),
                        jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))),
                    )
                ),
            ]
        else:
            self.local_params = [self.local_params[0]]
            self.global_params = [self.global_params[0]]
        # BRD
        m_brd = tfd.Gamma(100.0, 100.0 / (self.brd_mean * 0.1)).sample(
            seed=rngs[0],
            sample_shape=[self.layer_sizes[0], 1],
        )  # self.brd_mean
        v_brd = m_brd**2 / 100.0
        if init_brd is not None:
            m_brd = init_brd
            v_brd = m_brd**2 / 10.0
        self.global_params.append(
            jnp.array(
                (
                    jnp.log(m_brd**2 / jnp.sqrt(m_brd**2 + v_brd))
                    * jnp.ones((self.layer_sizes[0], 1)),
                    jnp.log(jnp.sqrt(jnp.log(1 + v_brd / (m_brd**2))))
                    * jnp.ones((self.layer_sizes[0], 1)),
                )
            ),
        )

        if nmf_init and init_w is None:
            self.logger.info("Initializing the factor weights with NMF.")
            init_z, init_w = self.get_nmf_init(**kwargs)

        z_shapes = []
        z_rates = []
        rng_cnt = 0

        for layer_idx in range(
            self.n_layers
        ):  # we go from layer 0 (bottom) to layer L (top)
            # Init z
            a = z_init_concentration
            clip = 1e-3  # alpha=1 and clip=1e-6 allows for many factors to be learned, but leads to BRD becoming kind of big

            if (
                layer_idx > 0
            ):  # If the top layer inits to very small numbers, the loss goes crazy when MC=10...
                clip = 1e-3
                a = 1.0

            if layer_idx == self.n_layers - 1:
                a = 100.0
                clip = 1e-3

            z_init_layer = _get_layer_init(init_z, layer_idx)
            if z_init_layer is not None and not nmf_init:
                m = z_init_layer.astype(jnp.float32)
            elif (
                nmf_init
                and (not init_z_provided)
                and layer_idx == 0
                and z_init_layer is not None
                and z_init_layer.shape[1] == self.layer_sizes[layer_idx]
                and z_init_layer.shape[0] == self.n_cells
            ):
                # Use NMF L0 z pattern, but keep balanced per-cell mass and prior scale.
                iz = z_init_layer.astype(np.float32)
                # Prevent single-factor domination by normalizing each cell across factors.
                iz = iz / np.maximum(np.max(iz, axis=1, keepdims=True), 1e-8)
                iz = iz + 1.0  # / self.layer_sizes[layer_idx]
                # iz = iz * float(self.layer_sizes[layer_idx])  # prior mean ~1 per factor
                m = jnp.asarray(np.clip(iz, max(1e-3, clip), 1e1), dtype=jnp.float32)
                m = jnp.clip(
                    tfd.Gamma(1.0, 1.0 / m).sample(
                        seed=rngs[rng_cnt],
                    ),
                    clip,
                    1e1,
                )
                rng_cnt += 1
            else:
                m = jnp.clip(
                    tfd.Gamma(a, a / 1.0).sample(
                        seed=rngs[rng_cnt],
                        sample_shape=[self.n_cells, self.layer_sizes[layer_idx]],
                    ),
                    clip,
                    4.0,
                )
                # sd = 2.0  # tunable
                # Lognormal with mean = 1 (corrected via -sd²/2 shift in log space)
                # m = tfd.LogNormal(loc=-sd**2 / 2, scale=sd).sample(
                #     seed=rngs[rng_cnt],
                #     sample_shape=[self.n_cells, self.layer_sizes[layer_idx]],
                # )
                # m = jnp.clip(m, clip, 1e1)
                # rng_cnt += 1

            v = m / 100.0

            if layer_idx > 0:
                v = m / 10.0
            z_shape = jnp.log(m**2 / jnp.sqrt(m**2 + v)) * jnp.ones(
                (self.n_cells, self.layer_sizes[layer_idx])
            )
            z_shapes.append(z_shape)
            z_rate = jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))) * jnp.ones(
                (self.n_cells, self.layer_sizes[layer_idx])
            )
            z_rates.append(z_rate)

        self.local_params.append(jnp.array((jnp.hstack(z_shapes), jnp.hstack(z_rates))))

        for layer_idx in range(
            self.n_layers
        ):  # the w don't have a shared axis across all, so we can't vectorize
            # Init w
            in_layer = self.layer_sizes[layer_idx]
            if layer_idx == 0:
                out_layer = self.n_genes
            else:
                out_layer = self.layer_sizes[layer_idx - 1]

            a = 10.0

            w_init_layer = _get_layer_init(init_w, layer_idx)
            if w_init_layer is not None and not nmf_init:
                m = w_init_layer.astype(jnp.float32)  # + 1e-6
                v = m  # / 100.0
            else:
                m = 1.0 / self.layer_sizes[layer_idx] * jnp.ones((in_layer, out_layer))
                m = m * self.w_priors[layer_idx][0] / self.w_priors[layer_idx][1]
                if layer_idx < self.n_layers - 1:
                    m = tfd.Gamma(a, a / (m)).sample(seed=rngs[rng_cnt])
                rng_cnt += 1
                v = m / 100.0
            # if nmf_init and layer_idx == 0:
            #     iw = init_w[layer_idx].astype(jnp.float32)
            #     # Rescale
            #     iw = 1.0 * iw + 1.0
            #     # iw = 10. * 100.**iw/100. + 1.
            #     iw = iw
            #     iw = iw  # * self.w_priors[layer_idx][0] / self.w_priors[layer_idx][1]
            #     iw = (
            #         iw / np.sum(iw, axis=1, keepdims=True)
            #         + 1.0 / self.layer_sizes[layer_idx]
            #     )
            #     iw = jnp.clip(iw, 1e-6, 1e2)  # need to set scales to avoid huge BRDs...
            #     m = tfd.Gamma(100.0, 100.0 / (iw)).sample(seed=rngs[rng_cnt])
            #     rng_cnt += 1
            #     v = m / 100.0
            v = m / 10.0
            if layer_idx > 0:
                v = m / 10.0
            w_shape = jnp.log(m**2 / jnp.sqrt(m**2 + v)) * jnp.ones(
                (in_layer, out_layer)
            )
            w_rate = jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))) * jnp.ones(
                (in_layer, out_layer)
            )

            self.global_params.append(jnp.array((w_shape, w_rate)))

        # Gamma(1e3, 1e3/m)
        m = (
            self.shrinkage_shape / self.shrinkage_rate
        )  # self.shrinkage_mean * self.shrinkage_shape  / self.shrinkage_rate
        v = m**2 / 10.0
        if init_ard is not None:
            m = init_ard
            v = m**2 / 10.0

        self.global_params.append(
            jnp.array(
                (
                    jnp.log(m**2 / jnp.sqrt(m**2 + v))
                    * jnp.ones((self.layer_sizes[0], 1)),
                    jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2))))
                    * jnp.ones((self.layer_sizes[0], 1)),
                )
            ),
        )

        if init_alpha:
            m = self.alpha  # + 0. * np.array(self.gene_ratio_init)
        else:
            m = self.pmeans["alpha"]  # + 0. * np.array(self.gene_ratio_init)
        v = m**2 / 10.0

        self.global_params.append(
            jnp.array(
                (
                    jnp.log(m**2 / jnp.sqrt(m**2 + v)) * jnp.ones((1, 1)),
                    jnp.log(jnp.sqrt(jnp.log(1 + v / (m**2)))) * jnp.ones((1, 1)),
                )
            ),
        )

    def get_nmf_init(self, max_cells=None):
        from sklearn.decomposition import NMF

        X_full = self.X
        X_norm = (
            X_full / np.maximum(np.sum(X_full, axis=1, keepdims=True), 1e-8) * 10_000
        )

        if max_cells is not None and max_cells < X_full.shape[0]:
            key = jax.random.PRNGKey(self.seed)
            idx = np.array(
                jax.random.choice(
                    key, X_full.shape[0], replace=False, shape=(max_cells,)
                )
            )
        else:
            idx = np.arange(X_full.shape[0])

        init_z = []
        init_W = []

        X_fit = X_norm[idx]  # subset for NMF
        X_proj = X_norm  # full data for projection

        for layer_idx in range(self.n_layers):
            model = NMF(
                n_components=self.layer_sizes[layer_idx],
                random_state=self.seed,
                beta_loss="kullback-leibler",
                solver="mu",
            )
            z_subset = model.fit_transform(X_fit)
            W = model.components_

            # Project full data through learned W
            WtW_inv = np.linalg.pinv(W @ W.T)
            z_full = np.clip(X_proj @ W.T @ WtW_inv, 1e-8, None)

            init_z.append(z_full)
            init_W.append(W)

            # For next layer: subset and full-data versions of z
            X_fit = z_subset
            X_proj = z_full

        return init_z, init_W

    def identify_mixture_factors(
        self, max_n_genes: int = 20, thres: float = 0.5
    ) -> np.ndarray:
        """Identify factors that might be better if broken apart.

        Args:
            max_n_genes: maximum number of genes per factor
            thres: threshold for identifying mixture factors

        Returns:
            array of factor indices that are mixture factors
        """
        # sparse_factors = np.where(self.pmeans['brd'].ravel() > 1.)[0]
        kept_factors = self.factor_lists[0]
        normed_factors = (
            self.pmeans["L0W"][kept_factors]
            / self.pmeans["L0W"][kept_factors].max(axis=1)[:, None]
        )
        n_genes_per_factor = np.sum(normed_factors > thres, axis=1)
        mixture_factors = kept_factors[np.where(n_genes_per_factor > max_n_genes)[0]]
        return mixture_factors

    def reinit_factors(
        self, mixture_factors=None, init_budgets=False, exponent=1.1, **kwargs
    ):
        kept_factors = self.factor_lists[0]
        if mixture_factors is None:
            mixture_factors = self.identify_mixture_factors(**kwargs)
        # Make W0 matrix with sparse factors, and with each mixture factor repeated twice
        W0 = np.ones((self.layer_sizes[0], self.n_genes)) * 1.0 / self.layer_sizes[0]
        for i, factor in enumerate(kept_factors):
            W0[i] = self.pmeans["L0W"][factor] ** exponent

        # If we can fit repetitions
        if len(mixture_factors) <= self.layer_sizes[0] - len(kept_factors):
            for factor in mixture_factors:
                # Add another copy
                W0[len(kept_factors) + i] = self.pmeans["L0W"][factor] ** exponent

        self.init_var_params(init_budgets=init_budgets, init_w=W0, nmf_init=False)
        return W0

    def elbo(
        self,
        rng,
        batch,
        indices,
        local_params,
        global_params,
        annealing_parameter,
        stop_gradients,
        stop_cell_budgets,
        stop_gene_budgets,
        alpha,
        min_shape=jnp.log(1e-10),
        max_shape=jnp.log(1e6),
        min_rate=jnp.log(1e-10),
        max_rate=jnp.log(1e10),
    ):
        # Only anneal the entropy of factor-related variables
        annealing_z = annealing_parameter  # for z
        annealing_w = annealing_parameter  # for W
        annealing_brd = 1.0  # fixed
        annealing_ard = 1.0  # fixed
        annealing_scales = 1.0  # fixed

        # Single-sample Monte Carlo estimate of the variational lower bound.
        batch_indices_onehot = self.batch_indices_onehot[indices]

        cell_budget_params = local_params[0]
        z_params = local_params[1]
        gene_budget_params = global_params[0]
        fscale_params = global_params[1]

        cell_budget_shape = jnp.clip(
            cell_budget_params[0][indices], min_shape, max_shape
        )
        cell_budget_rate = jnp.exp(
            jnp.clip(cell_budget_params[1][indices], min_rate, max_rate)
        )
        cell_budget_shape = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(cell_budget_shape),
            lambda: cell_budget_shape,
        )
        cell_budget_rate = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(cell_budget_rate),
            lambda: cell_budget_rate,
        )

        gene_budget_shape = jnp.maximum(gene_budget_params[0], min_shape)
        gene_budget_rate = jnp.exp(jnp.clip(gene_budget_params[1], min_rate, max_rate))
        gene_budget_shape = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(gene_budget_shape),
            lambda: gene_budget_shape,
        )
        gene_budget_rate = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(gene_budget_rate),
            lambda: gene_budget_rate,
        )

        fscale_shapes = jnp.clip(fscale_params[0], min_shape, max_shape)
        fscale_rates = jnp.exp(jnp.clip(fscale_params[1], min_rate, max_rate))
        fscale_shapes = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(fscale_shapes),
            lambda: fscale_shapes,
        )
        fscale_rates = jax.lax.cond(
            stop_gradients[0],
            lambda: jax.lax.stop_gradient(fscale_rates),
            lambda: fscale_rates,
        )

        wm_shape = jnp.clip(global_params[2 + self.n_layers][0], min_shape, max_shape)
        wm_rate = jnp.exp(
            jnp.clip(global_params[2 + self.n_layers][1], min_rate, max_rate)
        )

        s_shape = jnp.clip(
            global_params[2 + self.n_layers + 1][0], min_shape, max_shape
        )
        s_rate = jnp.exp(
            jnp.clip(global_params[2 + self.n_layers + 1][1], min_rate, max_rate)
        )

        z_shapes = jnp.clip(z_params[0][indices], min_shape, max_shape)
        z_rates = jnp.exp(jnp.clip(z_params[1][indices], min_rate, max_rate))

        # Sample from variational distribution
        cell_budget_shape = jax.lax.cond(
            stop_cell_budgets,
            lambda: jax.lax.stop_gradient(cell_budget_shape),
            lambda: cell_budget_shape,
        )
        cell_budget_rate = jax.lax.cond(
            stop_cell_budgets,
            lambda: jax.lax.stop_gradient(cell_budget_rate),
            lambda: cell_budget_rate,
        )
        cell_budget_sample = lognormal_sample(rng, cell_budget_shape, cell_budget_rate)
        gene_budget_shape = jax.lax.cond(
            stop_gene_budgets,
            lambda: jax.lax.stop_gradient(gene_budget_shape),
            lambda: gene_budget_shape,
        )
        gene_budget_rate = jax.lax.cond(
            stop_gene_budgets,
            lambda: jax.lax.stop_gradient(gene_budget_rate),
            lambda: gene_budget_rate,
        )
        gene_budget_sample = lognormal_sample(rng, gene_budget_shape, gene_budget_rate)
        if self.use_brd:
            fscale_samples = lognormal_sample(rng, fscale_shapes, fscale_rates)
            _wm_sample = lognormal_sample(rng, wm_shape, wm_rate)
            brd_samples = fscale_samples  # / jnp.maximum(_wm_sample, 1.0)
            # log_brd = jnp.log(fscale_samples + 1e-8)
            # log_brd = log_brd - jnp.mean(log_brd) + jnp.log(self.brd_mean)
            # brd_samples = jnp.exp(log_brd)
        # w will be sampled in a loop below because it cannot be vectorized

        # Compute ELBO
        global_pl = gamma_logpdf(
            gene_budget_sample,
            self.gene_scale_shape,
            self.gene_scale_shape * self.gene_ratio,
        )
        global_en = annealing_scales * lognormal_entropy(
            gene_budget_shape, gene_budget_rate
        )
        # exposure = (self.batch_lib_sizes)[:, None]
        # local_pl = 0.0
        # local_en = 0.0
        # local_pl = gamma_logpdf(cell_budget_sample,
        #                         self.cell_scale_shape,
        #                         self.cell_scale_shape / exposure[indices])
        local_pl = gamma_logpdf(
            cell_budget_sample,
            self.cell_scale_shape,
            self.cell_scale_shape * self.batch_lib_ratio[indices],
        )
        local_en = annealing_scales * lognormal_entropy(
            cell_budget_shape, cell_budget_rate
        )

        # Optional alpha marginalization via variational posterior q(alpha).
        alpha_value = alpha
        if self.marginalize_alpha:
            s_sample = lognormal_sample(rng, s_shape, s_rate)
            alpha_value = jnp.squeeze(s_sample)
            global_pl += gamma_logpdf(
                s_sample,
                2.0,
                2.0 / jnp.maximum(jnp.asarray(alpha), 1e-8),
            )
            global_en += lognormal_entropy(s_shape, s_rate)
        if self.use_brd:
            global_pl += gamma_logpdf(
                fscale_samples, self.brd, self.brd / self.brd_mean
            )
            global_en += annealing_brd * lognormal_entropy(fscale_shapes, fscale_rates)

            global_pl += gamma_logpdf(
                _wm_sample, self.shrinkage_shape, self.shrinkage_rate
            )

            global_en += annealing_ard * lognormal_entropy(wm_shape, wm_rate)

        z_mean = 1.0
        for idx in list(np.arange(0, self.n_layers)[::-1]):
            start = np.sum(self.layer_sizes[:idx]).astype(int)
            end = start + self.layer_sizes[idx]

            # w
            _w_shape = jnp.clip(global_params[2 + idx][0], min_shape, max_shape)
            _w_rate = jnp.exp(jnp.clip(global_params[2 + idx][1], min_rate, max_rate))

            _w_shape = jax.lax.cond(
                stop_gradients[idx],
                lambda: jax.lax.stop_gradient(_w_shape),
                lambda: _w_shape,
            )
            _w_rate = jax.lax.cond(
                stop_gradients[idx],
                lambda: jax.lax.stop_gradient(_w_rate),
                lambda: _w_rate,
            )

            _w_sample = lognormal_sample(rng, _w_shape, _w_rate)

            if idx == 0 and self.use_brd:
                global_pl += gamma_logpdf(
                    _w_sample,
                    self.w_priors[idx][0] * (1.0 / brd_samples),
                    self.w_priors[idx][1]
                    * (1.0 / brd_samples)
                    * (1.0 / _wm_sample)
                    * self.layer_sizes[idx],
                )
            elif idx == 0:
                global_pl += gamma_logpdf(
                    _w_sample,
                    self.w_priors[idx][0],
                    self.w_priors[idx][1] * self.layer_sizes[idx],
                )
            else:
                if idx == 1 and self.use_brd:
                    global_pl += gamma_logpdf(
                        _w_sample,
                        self.w_priors[idx][0],
                        self.w_priors[idx][1]
                        * 1.0
                        / brd_samples.T
                        * self.layer_sizes[idx],
                    )
                else:
                    global_pl += gamma_logpdf(
                        _w_sample,
                        self.w_priors[idx][0],
                        self.w_priors[idx][1] * self.layer_sizes[idx],
                    )
            global_en += annealing_w * lognormal_entropy(_w_shape, _w_rate)

            # z
            _z_shape = z_shapes[:, start:end]
            _z_rate = z_rates[:, start:end]

            _z_shape = jax.lax.cond(
                stop_gradients[idx],
                lambda: jax.lax.stop_gradient(_z_shape),
                lambda: _z_shape,
            )
            _z_rate = jax.lax.cond(
                stop_gradients[idx],
                lambda: jax.lax.stop_gradient(_z_rate),
                lambda: _z_rate,
            )
            _z_sample = lognormal_sample(rng, _z_shape, _z_rate)

            # if idx == 0 and self.use_brd:
            #     z_mean = z_mean * _wm_sample.T

            alpha_layer = alpha_value  # * (self.layer_sizes[0] / self.layer_sizes[idx])
            alpha_layer = jnp.maximum(
                alpha_layer * self.cell_alpha_factor[indices][:, None], 1.0
            )
            # Top-layer prior: ``top_alpha`` on the layer that is currently the "active top"
            # in the optimization schedule (via ``stop_gradients`` on the root).
            # - Root unfrozen: root ``z`` uses ``top_alpha``; all other layers ``alpha_layer``.
            # - Root frozen: penultimate ``z`` uses ``top_alpha``; root and other layers
            #   ``alpha_layer``.
            root_frozen = stop_gradients[self.n_layers - 1] > 0.5
            is_root = jnp.asarray(idx == self.n_layers - 1, dtype=jnp.bool_)
            is_penultimate = jnp.asarray(idx == self.n_layers - 2, dtype=jnp.bool_)
            use_top_prior = jnp.logical_or(
                jnp.logical_and(is_root, jnp.logical_not(root_frozen)),
                jnp.logical_and(is_penultimate, root_frozen),
            )
            local_pl += jax.lax.cond(
                use_top_prior,
                lambda: gamma_logpdf(_z_sample, self.top_alpha, self.top_alpha),
                lambda: gamma_logpdf(
                    _z_sample,
                    alpha_layer,
                    alpha_layer * self.kappa / (z_mean),
                ),
            )

            local_en += annealing_z * lognormal_entropy(_z_shape, _z_rate)

            z_mean = jnp.einsum("nk,kp->np", _z_sample, _w_sample)

        n_w_sample = _w_sample  # / jnp.sum(_w_sample, axis=1, keepdims=True)
        n_z_sample = _z_sample  # / jnp.sum(_z_sample, axis=1, keepdims=True)
        if self.use_brd:
            n_w_sample = n_w_sample  # * _wm_sample
        mean_bottom = (
            jnp.einsum("nk,kg->ng", n_z_sample, n_w_sample) * cell_budget_sample
        )  # self.exposures[indices]
        mean_bottom = mean_bottom * (batch_indices_onehot.dot(gene_budget_sample))

        # x = jnp.array(batch)
        # lam = mean_bottom  # same shape

        # mask = x > 0
        # x_nz = jnp.where(mask, x, 0.0)

        # # X log λ  -  log(X!)
        # ll_nz = jnp.sum(x_nz * jnp.log(lam + 1e-8))
        # ll_nz -= jnp.sum(jax.scipy.special.gammaln(x_nz + 1.0))

        # # -∑ λ over *all* entries
        # ll_zero = -jnp.sum(lam)

        # ll = ll_nz + ll_zero
        ll = jnp.sum(vmap(poisson.logpmf)(jnp.array(batch), mean_bottom))
        # x = jnp.asarray(batch)
        # x = jnp.maximum(x, 0.0)
        # x = jnp.rint(x).astype(jnp.int32)   # counts should be ints
        # mu = jnp.clip(mean_bottom, 1e-8, 1e8)

        # # phi_sample should be shape (1,G) or (B,G)->(N,G); make sure it broadcasts to mu
        # phi = jnp.clip(s_sample, 1e-3, 1e6)          # IMPORTANT: keep total_count away from 0

        # logits = jnp.log(mu + 1e-8) - jnp.log(phi + 1e-8)

        # nb = tfd.NegativeBinomial(total_count=phi, logits=logits)
        # ll = jnp.sum(nb.log_prob(x))
        ll = jax.lax.cond(
            stop_gradients[0],
            lambda: ll * 0.0,
            lambda: ll,
        )

        # Anneal the entropy
        # global_en *= annealing_parameter
        # local_en *= annealing_parameter

        return (
            (ll + local_pl + local_en) * (self.X.shape[0] / indices.shape[0])
            + global_pl
            + global_en
        )

    def batch_elbo(
        self,
        rng,
        X,
        indices,
        local_params,
        global_params,
        num_samples,
        annealing_parameter,
        stop_gradients,
        stop_cell_budgets,
        stop_gene_budgets,
        alpha,
    ):
        # Average over a batch of random samples.
        rngs = random.split(rng, num_samples)
        vectorized_elbo = vmap(
            self.elbo,
            in_axes=(0, None, None, None, None, None, None, None, None, None),
        )
        return jnp.mean(
            vectorized_elbo(
                rngs,
                X,
                indices,
                local_params,
                global_params,
                annealing_parameter,
                stop_gradients,
                stop_cell_budgets,
                stop_gene_budgets,
                alpha,
            )
        )

    def _optimize(
        self,
        local_update_func,
        global_update_func,
        local_params,
        global_params,
        local_opt_state,
        global_opt_state,
        n_epochs=500,
        batch_size=1,
        annealing_parameter=1.0,
        stop_gradients=None,
        stop_cell_budgets=None,
        stop_gene_budgets=None,
        seed=None,
        min_epochs=100,
        tolerance=1e-5,
        patience=5,
        update_locals=True,
        update_globals=True,
    ):
        if seed is None:
            seed = self.seed

        if stop_gradients is None:
            stop_gradients = np.zeros((self.n_layers))
        if stop_cell_budgets is None:
            stop_cell_budgets = 0.0
        if stop_gene_budgets is None:
            stop_gene_budgets = 0.0

        num_complete_batches, leftover = divmod(self.n_cells, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        self.logger.info(
            f"Each epoch contains {num_batches} batches of size {int(min(batch_size, self.n_cells))}"
        )

        def data_stream():
            rng = np.random.RandomState(0)
            while True:
                perm = rng.permutation(self.n_cells)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                    yield jnp.array(self.X[batch_idx]), jnp.array(batch_idx)

        batches = data_stream()

        losses = []
        global_grads = 0.0
        annealing_parameter = jnp.array(annealing_parameter)
        stop_gradients = jnp.array(stop_gradients)
        stop_cell_budgets = jnp.array(stop_cell_budgets)
        stop_gene_budgets = jnp.array(stop_gene_budgets)
        rng = random.PRNGKey(seed)
        t = 0
        min_loss = np.inf
        early_stop_counter = 0
        stop_early = False
        pbar = tqdm(range(n_epochs))
        try:
            for epoch in pbar:
                epoch_losses = []
                start_time = time.time()
                for it in range(num_batches):
                    rng, rng_input = random.split(rng)
                    X, indices = next(batches)
                    if update_locals:
                        loss, local_params, local_opt_state = local_update_func(
                            X,
                            indices,
                            t,
                            rng_input,
                            local_params,
                            global_params,
                            local_opt_state,
                            global_opt_state,
                            annealing_parameter,
                            stop_gradients,
                            stop_cell_budgets,
                            stop_gene_budgets,
                        )
                    if update_globals:
                        (
                            loss,
                            global_params,
                            global_opt_state,
                            global_grads,
                        ) = global_update_func(
                            X,
                            indices,
                            t,
                            rng_input,
                            local_params,
                            global_params,
                            local_opt_state,
                            global_opt_state,
                            annealing_parameter,
                            stop_gradients,
                            stop_cell_budgets,
                            stop_gene_budgets,
                        )
                    epoch_losses.append(loss)
                    t += 1
                current_loss = np.mean(epoch_losses)
                losses.append(current_loss)

                if epoch >= min_epochs:
                    if min_loss == np.inf:
                        min_loss = current_loss
                        stop_early = False

                    relative_improvement = (min_loss - current_loss) / np.abs(min_loss)
                    min_loss = min(min_loss, current_loss)

                    if relative_improvement < tolerance:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                    if early_stop_counter >= patience:
                        stop_early = True

                if stop_early:
                    break

                epoch_time = time.time() - start_time  # noqa: F841
                pbar.set_postfix({"Loss": losses[-1]})
                if epoch >= min_epochs:
                    pbar.set_postfix(
                        {"Loss": losses[-1], "Rel. improvement": relative_improvement}
                    )

            if stop_early:
                self.logger.info(
                    "Relative improvement of "
                    f"{relative_improvement:0.4g} < {tolerance:0.4g} "
                    f"for {patience} step(s) in a row, stopping early."
                )
                if (
                    relative_improvement < 0
                    and np.abs(relative_improvement) > 100 * tolerance
                ):
                    self.logger.info(
                        "It seems like the loss increased significantly. "
                        "Consider using a lower learning rate."
                    )
            else:
                self.logger.info(
                    "Learning stopped before convergence. "
                    "Consider increasing the number of iterations."
                )

        except KeyboardInterrupt:
            self.logger.info("Interrupted learning. Exiting safely...")

        return (
            losses,
            local_params,
            global_params,
            local_opt_state,
            global_opt_state,
            global_grads,
        )

    def fit(
        self,
        nmf_init: bool = False,
        max_cells_init: int = 5000,
        z_init_concentration: float = 0.5,
        n_rounds: int = 1,
        pretraining: bool = False,
        force_decay_factor: bool = False,
        root_epochs: int = 0,
        **kwargs: Any,
    ) -> None:
        """Fit scDEF, warm-starting from a previous fit when available.
        Args:
            nmf_init: whether to initialize the model with NMF.
            max_cells_init: maximum number of cells to use for initialization.
            z_init_concentration: concentration parameter of a Gamma distribution to sample the initial z values from. If high coverage, prefer higher values to avoid overfitting early.
            n_rounds: number of rounds to run the optimization.
            pretraining: whether to run ``pretrain`` before standard fit.
            force_decay_factor: on refit, whether to clip upper-layer sizes to the
                geometric template implied by ``n_factors``, ``top_factors``, and
                ``n_layers_schedule``. If False, preserves learned per-layer dimensions and
                initializes all layers from previous posterior means.
            root_epochs: if ``> 0`` (default ``0``), run a first phase with the root layer frozen, then
                a root-only refinement phase. In ``elbo``, ``top_alpha`` applies to the
                penultimate layer ``z`` while the root is frozen, and to the root ``z`` when
                the root is optimized; other layers use ``alpha_layer`` (see
                ``stop_gradients`` on the root). Factor ``filter`` and ``annotate`` (see
                ``_learn``) are skipped during the frozen-root phase and run only after the
                root step, using the ``filter`` / ``annotate`` values passed to ``fit``.
            **kwargs: additional keyword arguments.

            On the first call, parameters are initialized from priors (or NMF if enabled).
            On subsequent calls, the model is re-initialized from the current posterior
            quantities and the current `factor_lists`, enabling a fit -> filter -> fit
            workflow. During refit, upper-layer sizes are clipped to the geometric
            template when ``force_decay_factor`` is True before rebuilding the hierarchy.
        """
        marginalize_alpha_for_main_fit = bool(self.marginalize_alpha)
        self.root_epochs = int(root_epochs)
        if getattr(self, "_has_fit", False):
            # Keep original kept indices before resizing; update_model_size resets factor_lists.
            old_factor_lists = [np.array(f).copy() for f in self.factor_lists]
            layer_sizes = [len(self.factor_lists[i]) for i in range(self.n_layers)]
            if force_decay_factor:
                template = self._geometric_layer_sizes(int(self.n_layers_schedule))
                for i in range(min(len(layer_sizes), len(template))):
                    if layer_sizes[i] > template[i]:
                        layer_sizes[i] = template[i]
            self.update_model_size(
                max_n_factors=max(layer_sizes), layer_sizes=layer_sizes
            )
            self.update_model_priors()
            self.logger.info(
                f"Continuing scDEF from previous fit with layer sizes {self.layer_sizes}."
            )
            nmf_init = False
            init_budgets = False
            init_alpha = True
            if force_decay_factor:
                l0_keep = np.array(old_factor_lists[0], dtype=int)
                init_z = np.array(self.pmeans[f"{self.layer_names[0]}z"])[:, l0_keep]
                init_w = np.array(self.pmeans["L0W"])[l0_keep]
                init_brd = np.array(self.pmeans["brd"])[l0_keep]
                init_ard = np.array(self.pmeans["factor_means"])[l0_keep]
            else:
                init_z = [
                    np.array(self.pmeans[f"{self.layer_names[layer_idx]}z"])[
                        :, np.array(old_factor_lists[layer_idx], dtype=int)
                    ]
                    for layer_idx in range(
                        self.n_layers - 1
                    )  # ignore root and subroot z
                ]
                init_w = []
                for layer_idx in range(self.n_layers - 1):  # ignore root
                    w = np.array(self.pmeans[f"{self.layer_names[layer_idx]}W"])
                    row_keep = np.array(old_factor_lists[layer_idx], dtype=int)
                    if layer_idx == 0:
                        init_w.append(w[row_keep])
                    else:
                        col_keep = np.array(old_factor_lists[layer_idx - 1], dtype=int)
                        init_w.append(w[np.ix_(row_keep, col_keep)])
                init_brd = np.array(self.pmeans["brd"])[
                    np.array(old_factor_lists[0], dtype=int)
                ]
                init_ard = np.array(self.pmeans["factor_means"])[
                    np.array(old_factor_lists[0], dtype=int)
                ]
                # init_brd = None
                # init_ard = None
        else:
            init_budgets = True
            init_alpha = True
            init_z = None
            init_w = None
            init_brd = None
            init_ard = None
        self.init_var_params(
            init_budgets=init_budgets,
            init_alpha=init_alpha,
            init_z=init_z,
            init_w=init_w,
            init_brd=init_brd,
            init_ard=init_ard,
            nmf_init=nmf_init,
            max_cells=max_cells_init,
            z_init_concentration=z_init_concentration,
        )
        self._invalidate_cached_diagnostics()
        self.elbos = []
        self.step_sizes = []
        if pretraining:
            pretraining_n_epoch = int(
                kwargs.pop(
                    "pretraining_n_epoch",
                    max(10, int(kwargs.get("n_epoch", 1000)) // 5),
                )
            )
            pretraining_prune_alpha = kwargs.pop("pretraining_prune_alpha", None)
            if pretraining_prune_alpha is not None:
                pretraining_prune_alpha = float(pretraining_prune_alpha)
            pretrain_kwargs = dict(kwargs)
            pretrain_kwargs.pop("n_epoch", None)
            pretrain_kwargs.pop("n_epochs", None)
            self.pretrain(
                n_epoch=pretraining_n_epoch,
                prune_alpha=pretraining_prune_alpha,
                **pretrain_kwargs,
            )

        self.marginalize_alpha = marginalize_alpha_for_main_fit
        optimize_layers = list(range(self.n_layers))
        if root_epochs > 0:
            # Learn with frozen root
            optimize_layers = list(range(self.n_layers - 1))

        stop_gene_budgets = not init_budgets
        stop_cell_budgets = not init_budgets
        main_learn_kwargs = dict(kwargs)
        if int(root_epochs) > 0:
            # Defer factor filtering and adata annotation until after root refinement;
            # the frozen-root phase leaves root z unsettled for downstream summaries.
            main_learn_kwargs["filter"] = False
            main_learn_kwargs["annotate"] = False
        self._learn(
            n_rounds=n_rounds,
            optimize_layers=optimize_layers,
            stop_gene_budgets=stop_gene_budgets,
            stop_cell_budgets=stop_cell_budgets,
            **main_learn_kwargs,
        )
        self.qc_elbos = [np.asarray(x).copy() for x in self.elbos]
        preserved_traces = {
            "entropy_annealing_trace": np.asarray(
                getattr(self, "entropy_annealing_trace", np.array([]))
            ).copy(),
            "entropy_annealing_trace_epochs": np.asarray(
                getattr(self, "entropy_annealing_trace_epochs", np.array([]))
            ).copy(),
        }

        if root_epochs > 0:
            # Update with unfrozen root
            root_kwargs = dict(kwargs)
            root_kwargs["n_epoch"] = root_epochs
            # Root-only refinement should always run with baseline entropy
            # temperature, independent of the main fit entropy adaptation.
            root_kwargs["annealing"] = 1.0
            root_kwargs["entropy_anneal"] = False
            self._learn(
                n_rounds=1,
                optimize_layers=[self.n_layers - 1],
                **root_kwargs,
            )
            # Keep QC focused on the main optimization pass, not root-only updates.
            self.entropy_annealing_trace = preserved_traces[
                "entropy_annealing_trace"
            ].copy()
            self.entropy_annealing_trace_epochs = preserved_traces[
                "entropy_annealing_trace_epochs"
            ].copy()
            self.adata.uns.pop("alpha_trace", None)
            self.adata.uns.pop("alpha_trace_epochs", None)
            self.adata.uns.pop("n_eff_parents_trace", None)
            self.adata.uns.pop("n_eff_parents_trace_epochs", None)
            self.adata.uns.pop("active_l0_factor_counts_trace", None)
            self.adata.uns.pop("alpha_schedule_alphas", None)
            self.adata.uns.pop("alpha_schedule_losses", None)
            self.adata.uns.pop("alpha_schedule_epochs", None)
            if len(self.entropy_annealing_trace) > 0:
                self.adata.uns[
                    "entropy_annealing_trace"
                ] = self.entropy_annealing_trace.copy()
                self.adata.uns[
                    "entropy_annealing_trace_epochs"
                ] = self.entropy_annealing_trace_epochs.copy()
            else:
                self.adata.uns.pop("entropy_annealing_trace", None)
                self.adata.uns.pop("entropy_annealing_trace_epochs", None)
        self.clear_runtime_cache(clear_jax_cache=False)
        self._has_fit = True
        self._fit_revision += 1

    def pretrain(
        self,
        n_epoch: int = 200,
        prune_alpha: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Run two-pass alpha pretraining without the final full fit pass.

        Schedule:
        1) Full model pass with low alpha (=1)
        2) Full model pass with high alpha (pruning pressure)
        """
        if not hasattr(self, "elbos"):
            self.elbos = []
        if not hasattr(self, "step_sizes"):
            self.step_sizes = []

        base_alpha = float(self.alpha)
        if prune_alpha is None:
            prune_alpha = max(10.0, base_alpha * 5.0)

        pretrain_kwargs = dict(kwargs)
        pretrain_kwargs["n_rounds"] = 1
        pretrain_kwargs["n_epoch"] = int(n_epoch)
        pretrain_kwargs["filter"] = False
        pretrain_kwargs["annotate"] = False
        self.logger.info(
            "Starting two-pass pretraining (n_epoch=%s).",
            int(n_epoch),
        )
        prev_marginalize_alpha = bool(self.marginalize_alpha)
        self.marginalize_alpha = False
        try:
            self.alpha = 1.0
            self._learn(**pretrain_kwargs)

            self.alpha = float(prune_alpha)
            self._learn(**pretrain_kwargs)
        finally:
            self.alpha = base_alpha
            self.marginalize_alpha = prev_marginalize_alpha
            self.clear_runtime_cache(clear_jax_cache=False)

    def clear_runtime_cache(self, clear_jax_cache: bool = False) -> None:
        """Clear runtime-only compiled caches to reduce notebook memory pressure."""
        self._learn_jit_cache = None
        if clear_jax_cache:
            jax.clear_caches()

    def _invalidate_cached_diagnostics(self) -> None:
        """Drop cached diagnostics/signatures that become stale after refit."""
        if not hasattr(self, "adata") or self.adata is None:
            return
        if not hasattr(self.adata, "uns"):
            return
        for key in [
            "_factor_obs_upper_lists_fixed",
            "_factor_obs_fit_revision",
            "factor_obs",
            "factor_signatures",
            "confident_signatures",
            "obs_scores",
            "within_group_pairwise_dissimilarity",
            "technical_hierarchy",
        ]:
            self.adata.uns.pop(key, None)

    def _get_or_build_learn_grad_fns(self, num_samples: int):
        """Return cached JIT-compiled local/global gradient callables for _learn."""
        marginalize_alpha = bool(self.marginalize_alpha)
        cache = getattr(self, "_learn_jit_cache", None)
        cache_hit = (
            isinstance(cache, dict)
            and cache.get("num_samples", None) == int(num_samples)
            and cache.get("marginalize_alpha", None) == marginalize_alpha
            and "local_loss_grad" in cache
            and "global_loss_grad" in cache
        )
        if cache_hit:
            return cache["local_loss_grad"], cache["global_loss_grad"]

        def objective(
            X,
            indices,
            local_var_params,
            global_var_params,
            key,
            annealing_parameter,
            stop_gradients,
            stop_cell_budgets,
            stop_gene_budgets,
            alpha,
        ):
            return -self.batch_elbo(
                key,
                X,
                indices,
                local_var_params,
                global_var_params,
                num_samples,
                annealing_parameter,
                stop_gradients,
                stop_cell_budgets,
                stop_gene_budgets,
                alpha,
            )

        local_loss_grad = jit(value_and_grad(objective, argnums=2))
        global_loss_grad = jit(value_and_grad(objective, argnums=3))
        self._learn_jit_cache = {
            "num_samples": int(num_samples),
            "marginalize_alpha": marginalize_alpha,
            "local_loss_grad": local_loss_grad,
            "global_loss_grad": global_loss_grad,
        }
        return local_loss_grad, global_loss_grad

    def _compute_median_parents(
        self,
        local_params=None,
        global_params=None,
        return_active_l0_count: bool = False,
    ):
        """Estimate median effective parent count using variational means."""
        if self.n_layers < 2:
            if return_active_l0_count:
                return 1.0, 0
            return 1.0

        if local_params is None:
            local_params = self.local_params
        if global_params is None:
            global_params = self.global_params

        z_params = local_params[1]
        # Active L0 factors: at least 1 cell assigned.
        start0 = 0
        end0 = int(self.layer_sizes[0])
        z0 = np.array(
            jnp.exp(
                z_params[0][:, start0:end0]
                + 0.5 * jnp.exp(z_params[1][:, start0:end0]) ** 2
            )
        )
        assignments0 = np.argmax(z0, axis=1)
        counts0 = np.array(
            [np.sum(assignments0 == j) for j in range(self.layer_sizes[0])]
        )
        active_0 = np.where(counts0 >= 1)[0]

        start1 = int(np.sum(self.layer_sizes[:1]))
        end1 = start1 + int(self.layer_sizes[1])
        z1 = np.array(
            jnp.exp(
                z_params[0][:, start1:end1]
                + 0.5 * jnp.exp(z_params[1][:, start1:end1]) ** 2
            )
        )
        assignments = np.argmax(z1, axis=1)
        counts = np.array(
            [np.sum(assignments == j) for j in range(self.layer_sizes[1])]
        )
        active_1 = np.where(counts >= 1)[0]

        if len(active_0) == 0 or len(active_1) == 0:
            if return_active_l0_count:
                return 1.0, int(len(active_0))
            return 1.0

        w1_params = global_params[3]
        W1 = np.array(jnp.exp(w1_params[0] + 0.5 * jnp.exp(w1_params[1]) ** 2))[
            np.ix_(active_1, active_0)
        ]
        p = W1 / np.clip(W1.sum(axis=0, keepdims=True), 1e-12, None)
        H = -np.sum(p * np.log(p + 1e-12), axis=0)
        n_parents = np.exp(H)

        median_parents = np.median(n_parents)
        if return_active_l0_count:
            return median_parents, int(len(active_0))
        return median_parents

    def _learn(
        self,
        n_rounds=1,
        n_epoch=1000,
        lr=1e-1,
        local_lr=1e-2,
        annealing=1.0,
        num_samples=100,
        batch_size=256,
        layerwise=False,
        min_epochs=50,
        tolerance=1e-5,
        patience=50,
        update_locals=True,
        update_globals=True,
        stop_cell_budgets=0,
        stop_gene_budgets=0,
        stop_gene_budgets_after_burnin=False,
        opt_layer=None,
        optimize_layers=None,
        filter=True,
        annotate=True,
        entropy_anneal=False,
        entropy_window=50,
        entropy_check_every=10,
        entropy_rel_change_low=1e-3,
        entropy_rel_change_high=1e-2,
        entropy_increase_factor=1.2,
        entropy_decrease_factor=0.9,
        entropy_min_annealing=1.0,
        entropy_max_annealing=5.0,
        entropy_optimizer_reset_threshold=0.25,
        **kwargs,
    ):
        """Fit the model."""
        if "n_epochs" in kwargs:
            n_epoch = kwargs.pop("n_epochs")
        if len(kwargs) > 0:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments for _learn: {unknown}")

        if int(n_rounds) > 1:
            total_rounds = int(n_rounds)
            for r in range(total_rounds):
                round_lr = lr if layerwise else lr * (0.5**r)
                round_local_lr = local_lr if layerwise else local_lr * (0.5**r)
                is_last_round = r == (total_rounds - 1)
                round_filter = bool(filter) and is_last_round
                round_annotate = bool(annotate) and is_last_round
                self.logger.info(
                    "Starting learning round %s/%s (lr=%.5g, local_lr=%.5g, filter=%s, annotate=%s).",
                    r + 1,
                    total_rounds,
                    round_lr,
                    round_local_lr,
                    round_filter,
                    round_annotate,
                )
                self._learn(
                    n_rounds=1,
                    n_epoch=n_epoch,
                    lr=round_lr,
                    local_lr=round_local_lr,
                    annealing=annealing,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    layerwise=layerwise,
                    min_epochs=min_epochs,
                    tolerance=tolerance,
                    patience=patience,
                    update_locals=update_locals,
                    update_globals=update_globals,
                    stop_cell_budgets=stop_cell_budgets,
                    stop_gene_budgets=stop_gene_budgets,
                    stop_gene_budgets_after_burnin=stop_gene_budgets_after_burnin,
                    opt_layer=opt_layer,
                    optimize_layers=optimize_layers,
                    filter=round_filter,
                    annotate=round_annotate,
                    entropy_anneal=entropy_anneal,
                    entropy_window=entropy_window,
                    entropy_check_every=entropy_check_every,
                    entropy_rel_change_low=entropy_rel_change_low,
                    entropy_rel_change_high=entropy_rel_change_high,
                    entropy_increase_factor=entropy_increase_factor,
                    entropy_decrease_factor=entropy_decrease_factor,
                    entropy_min_annealing=entropy_min_annealing,
                    entropy_max_annealing=entropy_max_annealing,
                    entropy_optimizer_reset_threshold=entropy_optimizer_reset_threshold,
                    **kwargs,
                )
            return

        if batch_size is None:
            batch_size = self.n_cells

        num_complete_batches, leftover = divmod(self.n_cells, batch_size)
        num_batches = num_complete_batches + bool(leftover)
        self.logger.info(
            f"Each epoch contains {num_batches} batches of size "
            f"{int(min(batch_size, self.n_cells))}"
        )

        def clip_params(
            params, min_mu=-1e10, max_mu=1e4, min_logstd=-1e10, max_logstd=1e1
        ):
            for i in range(len(params))[:-2]:
                params[i] = params[i].at[0].set(jnp.clip(params[i][0], min_mu, max_mu))
                params[i] = (
                    params[i].at[1].set(jnp.clip(params[i][1], min_logstd, max_logstd))
                )
            return params

        local_loss_grad, global_loss_grad = self._get_or_build_learn_grad_fns(
            num_samples=num_samples
        )

        # --- Set up layer-wise stop gradients ---

        if optimize_layers is None:
            if opt_layer is not None:
                optimize_layers = [int(opt_layer)]
            else:
                optimize_layers = list(range(self.n_layers))

        layers_to_optimize = sorted({int(i) for i in optimize_layers})
        if len(layers_to_optimize) == 0:
            raise ValueError("optimize_layers must contain at least one layer index.")
        if min(layers_to_optimize) < 0 or max(layers_to_optimize) >= self.n_layers:
            raise ValueError(
                f"optimize_layers must be within [0, {self.n_layers - 1}]."
            )

        stop_gradients = jnp.ones((self.n_layers,))
        layers_to_optimize = jnp.array(layers_to_optimize, dtype=jnp.int32)
        stop_gradients = stop_gradients.at[layers_to_optimize].set(0.0)
        stop_cell_budgets = jnp.array(stop_cell_budgets)
        stop_gene_budgets = jnp.array(stop_gene_budgets)

        # --- Initialize optimizers ---

        local_optimizer = optax.adam(local_lr)
        global_optimizer = optax.adam(lr)
        local_opt_state = local_optimizer.init(self.local_params)
        global_opt_state = global_optimizer.init(self.global_params)

        local_params = self.local_params
        global_params = self.global_params

        def local_update(
            X,
            indices,
            key,
            local_params,
            global_params,
            local_opt_state,
            annealing_parameter,
            stop_gradients,
            stop_cell_budgets,
            stop_gene_budgets,
            alpha,
        ):
            value, gradient = local_loss_grad(
                X,
                indices,
                local_params,
                global_params,
                key,
                annealing_parameter,
                stop_gradients,
                stop_cell_budgets,
                stop_gene_budgets,
                alpha,
            )
            updates, local_opt_state_new = local_optimizer.update(
                gradient, local_opt_state, local_params
            )
            local_params_new = optax.apply_updates(local_params, updates)
            local_params_new = clip_params(local_params_new)
            return value, local_params_new, local_opt_state_new

        def global_update(
            X,
            indices,
            key,
            local_params,
            global_params,
            global_opt_state,
            annealing_parameter,
            stop_gradients,
            stop_cell_budgets,
            stop_gene_budgets,
            alpha,
        ):
            value, gradient = global_loss_grad(
                X,
                indices,
                local_params,
                global_params,
                key,
                annealing_parameter,
                stop_gradients,
                stop_cell_budgets,
                stop_gene_budgets,
                alpha,
            )
            updates, global_opt_state_new = global_optimizer.update(
                gradient, global_opt_state, global_params
            )
            global_params_new = optax.apply_updates(global_params, updates)
            global_params_new = clip_params(global_params_new)
            return value, global_params_new, global_opt_state_new

        # --- Data stream ---

        def data_stream():
            rng = np.random.RandomState(0)
            while True:
                perm = rng.permutation(self.n_cells)
                for i in range(num_batches):
                    batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                    yield jnp.array(self.X[batch_idx]), jnp.array(batch_idx)

        batches = data_stream()

        # --- Main training loop ---

        annealing_parameter = jnp.array(annealing)
        alpha_jnp = jnp.asarray(self.alpha, dtype=jnp.float32)
        rng = random.PRNGKey(self.seed)

        all_losses = []
        total_epochs = 0
        min_loss = np.inf
        early_stop_counter = 0
        stop_message = None
        interrupted = False
        if int(entropy_window) <= 0:
            raise ValueError("entropy_window must be > 0.")
        if int(entropy_check_every) <= 0:
            raise ValueError("entropy_check_every must be > 0.")
        if float(entropy_rel_change_low) < 0.0 or float(entropy_rel_change_high) < 0.0:
            raise ValueError("entropy relative-change thresholds must be >= 0.")
        if float(entropy_rel_change_low) > float(entropy_rel_change_high):
            raise ValueError(
                "entropy_rel_change_low must be <= entropy_rel_change_high."
            )
        if float(entropy_increase_factor) <= 1.0:
            raise ValueError("entropy_increase_factor must be > 1.0.")
        if not (0.0 < float(entropy_decrease_factor) < 1.0):
            raise ValueError("entropy_decrease_factor must be in (0, 1).")
        if float(entropy_min_annealing) <= 0.0:
            raise ValueError("entropy_min_annealing must be > 0.")
        if float(entropy_max_annealing) < float(entropy_min_annealing):
            raise ValueError("entropy_max_annealing must be >= entropy_min_annealing.")
        if float(entropy_optimizer_reset_threshold) < 0.0:
            raise ValueError("entropy_optimizer_reset_threshold must be >= 0.")
        entropy_annealing_trace: List[float] = []
        entropy_annealing_trace_epochs: List[int] = []
        gene_budgets_frozen = bool(float(stop_gene_budgets) >= 1.0)

        pbar = tqdm(range(n_epoch))
        try:
            for epoch in pbar:
                epoch_losses = []

                if (
                    stop_gene_budgets_after_burnin
                    and not gene_budgets_frozen
                    and total_epochs >= min_epochs
                ):
                    stop_gene_budgets = jnp.array(1.0)
                    gene_budgets_frozen = True
                    self.logger.info(
                        f"Freezing gene budgets at epoch {total_epochs} "
                        "(post burn-in)."
                    )

                for it in range(num_batches):
                    rng, rng_input = random.split(rng)
                    X, indices = next(batches)

                    if update_locals:
                        loss, local_params, local_opt_state = local_update(
                            X,
                            indices,
                            rng_input,
                            local_params,
                            global_params,
                            local_opt_state,
                            annealing_parameter,
                            stop_gradients,
                            stop_cell_budgets,
                            stop_gene_budgets,
                            alpha_jnp,
                        )
                    if update_globals:
                        loss, global_params, global_opt_state = global_update(
                            X,
                            indices,
                            rng_input,
                            local_params,
                            global_params,
                            global_opt_state,
                            annealing_parameter,
                            stop_gradients,
                            stop_cell_budgets,
                            stop_gene_budgets,
                            alpha_jnp,
                        )

                    epoch_losses.append(loss)

                current_loss = np.mean(epoch_losses)
                all_losses.append(current_loss)
                total_epochs += 1
                entropy_annealing_trace.append(float(annealing_parameter))
                entropy_annealing_trace_epochs.append(int(total_epochs))

                if bool(entropy_anneal) and (
                    int(total_epochs) % int(entropy_check_every) == 0
                ):
                    entropy_loss_buffer = all_losses
                    if len(entropy_loss_buffer) < int(entropy_window):
                        entropy_loss_buffer = []
                    if len(entropy_loss_buffer) == 0:
                        recent = None
                    else:
                        recent = np.asarray(
                            entropy_loss_buffer[-int(entropy_window) :], dtype=float
                        )
                else:
                    recent = None

                if recent is not None:
                    recent_mean = float(np.mean(recent))
                    denom = max(abs(recent_mean), 1e-12)
                    relative_change = float((np.max(recent) - np.min(recent)) / denom)
                    old_annealing = float(annealing_parameter)
                    new_annealing = old_annealing
                    if relative_change < float(entropy_rel_change_low):
                        new_annealing = min(
                            old_annealing * float(entropy_increase_factor),
                            float(entropy_max_annealing),
                        )
                    elif relative_change > float(entropy_rel_change_high):
                        new_annealing = max(
                            old_annealing * float(entropy_decrease_factor),
                            float(entropy_min_annealing),
                        )
                    if abs(new_annealing - old_annealing) > 0.0:
                        annealing_parameter = jnp.asarray(
                            new_annealing, dtype=annealing_parameter.dtype
                        )
                        rel_step = abs(new_annealing - old_annealing) / max(
                            abs(old_annealing), 1e-12
                        )
                        if rel_step >= float(entropy_optimizer_reset_threshold):
                            local_opt_state = local_optimizer.init(local_params)
                            global_opt_state = global_optimizer.init(global_params)
                        self.logger.info(
                            "Entropy annealing update at epoch %s: %.4f -> %.4f "
                            "(relative_change=%.6g).",
                            int(total_epochs),
                            float(old_annealing),
                            float(new_annealing),
                            float(relative_change),
                        )

                relative_improvement = np.nan
                if total_epochs >= min_epochs:
                    if min_loss == np.inf:
                        min_loss = current_loss
                    relative_improvement = (min_loss - current_loss) / np.abs(min_loss)
                    min_loss = min(min_loss, current_loss)
                    if relative_improvement < tolerance:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0

                postfix = {"Loss": current_loss}
                if bool(entropy_anneal):
                    postfix["anneal"] = float(annealing_parameter)
                if total_epochs >= min_epochs:
                    postfix["Rel. impr."] = relative_improvement
                pbar.set_postfix(postfix)
                if total_epochs >= min_epochs and early_stop_counter >= patience:
                    stop_message = f"Converged at epoch {total_epochs}."
                    break

        except KeyboardInterrupt:
            interrupted = True
            self.logger.info("Interrupted. Exiting safely...")
        finally:
            pbar.close()

        if stop_message is None and not interrupted:
            stop_message = (
                f"Stopping learning: reached max epochs (n_epoch={int(n_epoch)})."
            )
        if stop_message is not None:
            self.logger.info(stop_message)

        self.local_params = local_params
        self.global_params = global_params
        self.elbos.append(all_losses)
        self.step_sizes.append(lr)
        self.adata.uns.pop("alpha_trace", None)
        self.adata.uns.pop("alpha_trace_epochs", None)
        self.adata.uns.pop("n_eff_parents_trace", None)
        self.adata.uns.pop("n_eff_parents_trace_epochs", None)
        self.adata.uns.pop("active_l0_factor_counts_trace", None)
        self.adata.uns.pop("alpha_schedule_alphas", None)
        self.adata.uns.pop("alpha_schedule_losses", None)
        self.adata.uns.pop("alpha_schedule_epochs", None)
        self.entropy_annealing_trace = np.asarray(entropy_annealing_trace, dtype=float)
        self.entropy_annealing_trace_epochs = np.asarray(
            entropy_annealing_trace_epochs, dtype=int
        )
        self.adata.uns["entropy_annealing_trace"] = self.entropy_annealing_trace.copy()
        self.adata.uns[
            "entropy_annealing_trace_epochs"
        ] = self.entropy_annealing_trace_epochs.copy()

        self.set_posterior_means()
        self.set_posterior_variances()
        if self.use_brd and filter:
            self.filter_factors(upper_only=True)
        elif annotate:
            self.make_layercolors(
                layer_cpal=self.layer_cpal, lightness_mult=self.lightness_mult
            )
            self.annotate_adata()

    def set_posterior_means(self):
        cell_budget_params = self.local_params[0]
        gene_budget_params = self.global_params[0]
        fscale_params = self.global_params[1]
        wm_params = self.global_params[-2]
        z_params = self.local_params[1]

        self.pmeans = {
            "cell_scale": np.array(
                jnp.exp(
                    cell_budget_params[0] + 0.5 * jnp.exp(cell_budget_params[1]) ** 2
                )
            ),
            "gene_scale": np.array(
                jnp.exp(
                    gene_budget_params[0] + 0.5 * jnp.exp(gene_budget_params[1]) ** 2
                )
            ),
            "factor_concentrations": jnp.exp(
                fscale_params[0] + 0.5 * jnp.exp(fscale_params[1]) ** 2
            ),
            "factor_means": jnp.exp(wm_params[0] + 0.5 * jnp.exp(wm_params[1]) ** 2),
        }
        self.pmeans["brd"] = self.pmeans["factor_concentrations"]  # / jnp.maximum(
        #  self.pmeans["factor_means"], 1.0
        #  )
        self.pmeans["wm"] = self.pmeans["factor_means"]
        self.pmeans["ard"] = self.pmeans["wm"]
        for idx in range(self.n_layers):
            start = sum(self.layer_sizes[:idx])
            end = start + self.layer_sizes[idx]
            self.pmeans[f"{self.layer_names[idx]}z"] = jnp.exp(
                z_params[0][:, start:end]
                + 0.5 * jnp.exp(z_params[1][:, start:end]) ** 2
            )
            _w_shape = self.global_params[2 + idx][0]
            _w_rate = self.global_params[2 + idx][1]
            self.pmeans[f"{self.layer_names[idx]}W"] = jnp.exp(
                _w_shape + 0.5 * jnp.exp(_w_rate) ** 2
            )

        self.pmeans["alpha"] = np.exp(
            self.global_params[-1][0] + 0.5 * np.exp(self.global_params[-1][1]) ** 2
        )

    def set_posterior_variances(self):
        cell_budget_params = self.local_params[0]
        gene_budget_params = self.global_params[0]
        fscale_params = self.global_params[1]
        wm_params = self.global_params[-2]
        z_params = self.local_params[1]

        def _lognormal_var(mu, log_sigma):
            sigma2 = np.exp(log_sigma) ** 2
            return (np.exp(sigma2) - 1.0) * np.exp(2.0 * mu + sigma2)

        self.pvars = {
            "cell_scale": np.array(
                _lognormal_var(cell_budget_params[0], cell_budget_params[1])
            ),
            "gene_scale": np.array(
                _lognormal_var(gene_budget_params[0], gene_budget_params[1])
            ),
            "factor_concentrations": np.array(
                _lognormal_var(fscale_params[0], fscale_params[1])
            ),
            "factor_means": np.array(_lognormal_var(wm_params[0], wm_params[1])),
            "alpha": np.array(
                _lognormal_var(self.global_params[-1][0], self.global_params[-1][1])
            ),
        }

        for idx in range(self.n_layers):
            start = sum(self.layer_sizes[:idx])
            end = start + self.layer_sizes[idx]
            self.pvars[f"{self.layer_names[idx]}z"] = np.array(
                _lognormal_var(z_params[0][:, start:end], z_params[1][:, start:end])
            )
            _w_shape = self.global_params[2 + idx][0]
            _w_rate = self.global_params[2 + idx][1]
            self.pvars[f"{self.layer_names[idx]}W"] = np.array(
                _lognormal_var(_w_shape, _w_rate)
            )

    def filter_factors(
        self,
        brd_min: Optional[float] = 1.0,
        ard_min: Optional[float] = 0.001,
        clarity_min: Optional[float] = 0.5,
        n_eff_parents_max: float = 1.5,
        local_l0_scores: bool = False,
        min_cells_upper: Optional[float] = 0.001,
        min_cells_lower: Optional[float] = 0.0,
        filter_up: Optional[bool] = True,
        annotate: Optional[bool] = True,
        upper_only: Optional[bool] = False,
    ):
        """Filter irrelevant factors using BRD/ARD and hierarchy diagnostics.

        Args:
            brd_min: minimum factor BRD value for layer 0 when ``use_brd``.
            ard_min: minimum ARD fraction of total ARD for layer 0.
            clarity_min: minimum L0 ``clarity_score_01`` when not using lineage
                ``avg_n_eff_parents`` (``local_l0_scores`` or missing lineage columns).
            n_eff_parents_max: only when ``avg_n_eff_parents`` is present and
                ``local_l0_scores`` is False: cutoff ``avg_n_eff_parents < n_eff_parents_max``
                (default ``1.5``). Ignored for local-L0 / clarity-only filtering.
            local_l0_scores: if True, filter by ``clarity_score_01 >= clarity_min``; if
                False and ``avg_n_eff_parents`` exists, filter by ``n_eff_parents_max``.
            min_cells_upper: minimum cells attached to upper-layer factors (fraction if <1).
            min_cells_lower: minimum cells attached to layer-0 factors (fraction if <1).
            filter_up: whether to prune upper layers via inter-layer attachments.
            annotate: whether to run ``annotate_adata`` after filtering.
            upper_only: if True, only adjust upper layers (layer 0 unchanged).
        """
        if min_cells_upper != 0:
            if min_cells_upper < 1.0:
                min_cells_upper = max(min_cells_upper * self.adata.shape[0], 10)
        if min_cells_lower != 0:
            if min_cells_lower < 1.0:
                min_cells_lower = max(min_cells_lower * self.adata.shape[0], 10)

        new_factor_lists = []
        for i, layer_name in enumerate(self.layer_names):
            if i == 0:
                if upper_only:
                    keep = np.arange(self.layer_sizes[i])
                else:
                    keep = self.get_effective_factors(
                        brd_min=brd_min,
                        ard_min=ard_min,
                        clarity_min=clarity_min,
                        n_eff_parents_max=n_eff_parents_max,
                        local_l0_scores=local_l0_scores,
                        min_cells=min_cells_lower,
                    )
            else:
                assignments = np.argmax(self.pmeans[f"{layer_name}z"], axis=1)
                counts = np.array(
                    [
                        np.count_nonzero(assignments == a)
                        for a in range(self.layer_sizes[i])
                    ]
                )
                keep = np.array(range(self.layer_sizes[i]))[
                    np.where(counts >= min_cells_upper)[0]
                ]
                if filter_up and len(keep) > 0 and len(new_factor_lists[i - 1]) > 0:
                    mat = self.pmeans[f"{layer_name}W"][keep]
                    assignments = []
                    for factor in new_factor_lists[i - 1]:
                        assignments.append(keep[np.argmax(mat[:, factor])])

                    keep = np.unique(
                        list(set(np.unique(assignments)).intersection(keep))
                    )

            if len(keep) == 0:
                self.logger.info(
                    f"No factors in layer {i} satisfy the filtering criterion. Please adjust the filtering parameters."
                    f"Keeping all factors for layer {i} for now."
                )
                keep = np.arange(self.layer_sizes[i])
            new_factor_lists.append(keep)

        self.factor_lists = new_factor_lists
        self.set_factor_names()
        if "factor_obs" in self.adata.uns:
            self._sync_factor_obs_with_filter()

        self.make_layercolors(
            layer_cpal=self.layer_cpal, lightness_mult=self.lightness_mult
        )
        if annotate:
            self.annotate_adata()

    def _sync_factor_obs_with_filter(self):
        """Rebuild ``adata.uns['factor_obs']`` so it reflects the currently
        kept factors, with rows renamed to match ``self.factor_names``.

        The rebuild is sourced from ``adata.uns['factor_obs_full']`` when
        available (the complete snapshot written by
        ``scdef.tools.factor_diagnostics``), so that re-filtering with
        looser thresholds can restore rows for factors that had been
        previously dropped from the live view. Falls back to the current
        ``factor_obs`` if the full snapshot is not present.

        Rows for filtered-out factors are omitted, kept rows are renamed
        (index and ``child_factor`` column) to the current factor names, and
        ``technical`` is reset to False. Name-valued columns that reference
        other factors (e.g. ``best_parent``) retain their diagnostic-time
        values and may become stale if upper-layer factors were filtered;
        re-run ``scdef.tools.factor_diagnostics(model, recompute=True)`` to
        refresh those.
        """
        if "factor_obs_full" in self.adata.uns:
            source = self.adata.uns["factor_obs_full"]
        else:
            source = self.adata.uns["factor_obs"]

        if not (
            "child_layer" in source.columns and "original_factor_idx" in source.columns
        ):
            self.adata.uns["factor_obs"]["technical"] = False
            return

        layer_lookup = {}
        for layer_idx, layer_name in enumerate(self.layer_names):
            layer_lookup[layer_name] = {
                int(orig): (slot, self.factor_names[layer_idx][slot])
                for slot, orig in enumerate(self.factor_lists[layer_idx])
            }

        rows = []
        for pos in range(len(source)):
            row = source.iloc[pos]
            mapping = layer_lookup.get(row["child_layer"], {}).get(
                int(row["original_factor_idx"])
            )
            if mapping is None:
                continue
            slot, current_name = mapping
            rows.append((row["child_layer"], slot, pos, current_name))

        rows.sort(key=lambda t: (self.layer_names.index(t[0]), t[1]))
        positions = [t[2] for t in rows]
        new_index = [t[3] for t in rows]

        new_factor_obs = source.iloc[positions].copy()
        new_factor_obs.index = new_index
        if "child_factor" in new_factor_obs.columns:
            new_factor_obs["child_factor"] = new_index
        new_factor_obs["technical"] = False
        self.adata.uns["factor_obs"] = new_factor_obs

    def set_factor_names(self):
        self.factor_names = [
            [
                f"{self.layer_names[idx]}_{str(i)}"
                for i in range(len(self.factor_lists[idx]))
            ]
            for idx in range(self.n_layers)
        ]

    def annotate_adata(self):
        self.adata.obs["cell_scale"] = self.pmeans["cell_scale"]
        if self.n_batches == 1:
            self.adata.var["gene_scale"] = self.pmeans["gene_scale"][0]
            self.logger.info("Updated adata.var: `gene_scale`")
        else:
            for batch_idx in range(self.n_batches):
                name = f"gene_scale_{batch_idx}"
                self.adata.var[name] = self.pmeans["gene_scale"][batch_idx]
                self.logger.info(f"Updated adata.var: `{name}` for batch {batch_idx}.")

        if not getattr(self, "_preserve_factor_names_on_annotate", False):
            self.set_factor_names()
        ranked_genes, ranked_scores = self.get_signatures_dict(
            scores=True, sorted_scores=True
        )

        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            self.adata.obsm[f"X_{layer_name}"] = np.array(
                self.pmeans[f"{layer_name}z"][:, self.factor_lists[idx]]
            )
            technical_names = set()
            if (
                "factor_obs" in self.adata.uns
                and "technical" in self.adata.uns["factor_obs"].columns
            ):
                technical_names = set(
                    self.adata.uns["factor_obs"]
                    .index[self.adata.uns["factor_obs"]["technical"]]
                    .tolist()
                )
            biological_mask = np.array(
                [name not in technical_names for name in self.factor_names[idx]],
                dtype=bool,
            )
            scores_for_assignment = self.adata.obsm[f"X_{layer_name}"].copy()
            if np.any(biological_mask):
                scores_for_assignment[:, ~biological_mask] = -np.inf
            assignments = np.argmax(scores_for_assignment, axis=1)
            self.adata.obs[f"{layer_name}"] = [
                self.factor_names[idx][a] for a in assignments
            ]
            # Make sure factor colors in UMAP respect the palette
            factor_colors = [
                matplotlib.colors.to_hex(self.layer_colorpalettes[idx][i])
                for i in range(len(self.factor_lists[idx]))
            ]

            if layer_name == "marker":
                self.adata.obs[f"{layer_name}"] = pd.Categorical(
                    self.adata.obs[f"{layer_name}"]
                )
                sorted_factors = self.adata.obs_vector(f"{layer_name}")
                sorted_colors = []
                for fac in sorted_factors.categories:
                    pos = np.where(np.array(self.factor_names[idx]) == fac)[0][0]
                    col = factor_colors[pos]
                    sorted_colors.append(col)
                    self.adata.uns[f"{layer_name}_colors"] = sorted_colors
            else:
                self.adata.uns[f"{layer_name}_colors"] = factor_colors

            scores_names = [f + "_score" for f in self.factor_names[idx]]
            df = pd.DataFrame(
                self.adata.obsm[f"X_{layer_name}"],
                index=self.adata.obs.index,
                columns=scores_names,
            )
            if scores_names[0] not in self.adata.obs.columns:
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)
            else:
                self.adata.obs = self.adata.obs.drop(
                    columns=[col for col in self.adata.obs.columns if "score" in col]
                )
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)

            self.logger.info(
                f"Updated adata.obs with layer {idx}: `{layer_name}` and `{layer_name}_score` for all factors in layer {idx}"
            )
            self.logger.info(f"Updated adata.obsm with layer {idx}: `X_{layer_name}`")

            factor_names = self.factor_names[idx]
            names = np.array(
                [
                    tuple(
                        [ranked_genes[factor_name][i] for factor_name in factor_names]
                    )
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "O") for factor_name in factor_names],
            ).view(np.recarray)

            scores = np.array(
                [
                    tuple(
                        [ranked_scores[factor_name][i] for factor_name in factor_names]
                    )
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "<f4") for factor_name in factor_names],
            ).view(np.recarray)

            pvals = np.array(
                [
                    tuple([1.0 for factor_name in factor_names])
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "<f4") for factor_name in factor_names],
            ).view(np.recarray)

            pvals_adj = np.array(
                [
                    tuple([1.0 for factor_name in factor_names])
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "<f4") for factor_name in factor_names],
            ).view(np.recarray)

            logfoldchanges = np.array(
                [
                    tuple([0.0 for factor_name in factor_names])
                    for i in range(self.n_genes)
                ],
                dtype=[(factor_name, "<f4") for factor_name in factor_names],
            ).view(np.recarray)

            self.adata.uns[f"{layer_name}_signatures"] = {
                "params": {
                    "reference": "rest",
                    "method": "scDEF",
                    "groupby": f"{layer_name}",
                },
                "names": names,
                "scores": scores,
                "pvals": pvals,
                "pvals_adj": pvals_adj,
                "logfoldchanges": logfoldchanges,
            }

            self.logger.info(
                f"Updated adata.uns with layer {idx} signatures: `{layer_name}_signatures`."
            )

        self.normalize_cellscores()

    def normalize_cellscores(self):
        for idx in range(self.n_layers):
            layer_name = self.layer_names[idx]
            technical_names = set()
            if (
                "factor_obs" in self.adata.uns
                and "technical" in self.adata.uns["factor_obs"].columns
            ):
                technical_names = set(
                    self.adata.uns["factor_obs"]
                    .index[self.adata.uns["factor_obs"]["technical"]]
                    .tolist()
                )
            biological_mask = np.array(
                [name not in technical_names for name in self.factor_names[idx]],
                dtype=bool,
            )
            probs = np.zeros_like(self.adata.obsm[f"X_{layer_name}"], dtype=float)
            if np.any(biological_mask):
                bio_scores = self.adata.obsm[f"X_{layer_name}"][:, biological_mask]
                bio_den = np.sum(bio_scores, axis=1, keepdims=True)
                bio_den = np.clip(bio_den, 1e-12, None)
                probs[:, biological_mask] = bio_scores / bio_den
            self.adata.obsm[f"X_{layer_name}_probs"] = probs
            scores_names = [f + "_prob" for f in self.factor_names[idx]]
            df = pd.DataFrame(
                self.adata.obsm[f"X_{layer_name}_probs"],
                index=self.adata.obs.index,
                columns=scores_names,
            )
            if scores_names[0] not in self.adata.obs.columns:
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)
            else:
                self.adata.obs = self.adata.obs.drop(
                    columns=[col for col in self.adata.obs.columns if "prob" in col]
                )
                self.adata.obs = pd.concat([self.adata.obs, df], axis=1)

    def get_annotations(
        self,
        marker_reference: Mapping[str, Sequence[str]],
        gene_rankings: Optional[List[List[str]]] = None,
    ) -> List[List[str]]:
        """Get annotations for factors based on marker gene reference.

        Args:
            marker_reference: dictionary mapping annotation names to gene lists
            gene_rankings: gene rankings for each factor, if None will be computed

        Returns:
            list of annotation lists, one per factor
        """
        if gene_rankings is None:
            gene_rankings = self.get_rankings(layer_idx=0)

        annotations = []
        keys = list(marker_reference.keys())
        for rank in gene_rankings:
            # Get annotation in marker_reference that contains the highest score
            # score is length of intersection over length of union
            scores = np.array(
                [
                    len(set(rank).intersection(set(marker_reference[a])))
                    / len(set(rank).union(set(marker_reference[a])))
                    for a in keys
                ]
            )
            sorted_ann_idx = np.argsort(scores)[::-1]
            sorted_scores = scores[sorted_ann_idx]

            # Get only annotations for which score is not zero
            sorted_ann_idx = sorted_ann_idx[np.where(sorted_scores > 0)[0]]

            ann = np.array(keys)[sorted_ann_idx]
            ann = np.array(
                [f"{a} ({sorted_scores[i]:.4f})" for i, a in enumerate(ann)]
            ).tolist()
            annotations.append(ann)
        return annotations

    def get_rankings(
        self,
        layer_idx: int = 0,
        top_genes: Optional[int] = None,
        genes: bool = True,
        return_scores: bool = False,
        sorted_scores: bool = True,
        drop_factors: Optional[List[str]] = None,
    ) -> Union[List[List[str]], Tuple[List[List[str]], List[List[float]]]]:
        """Get gene or factor rankings for each factor in a layer.

        Args:
            layer_idx: layer index to get rankings for
            top_genes: number of top genes/factors to return
            genes: whether to return gene rankings (True) or factor rankings (False).
                Gene rankings use cached confidence+mean combined scores when
                ``sorted_scores=True`` and ``drop_factors`` is not provided.
            return_scores: whether to return scores along with rankings
            sorted_scores: whether to return scores sorted by ranking
            drop_factors: list of factors to drop from rankings

        Returns:
            list of rankings per factor, or tuple of (rankings, scores) if return_scores is True
        """
        top_genes_provided = top_genes is not None
        if top_genes is None:
            top_genes = len(self.adata.var_names)

        if (
            genes
            and sorted_scores
            and top_genes_provided
            and (drop_factors is None or len(drop_factors) == 0)
        ):
            from scdef.tools.factor import (
                get_stored_confident_signatures,
                set_confident_signatures,
            )

            try:
                top_terms_dict, top_scores_dict = get_stored_confident_signatures(
                    self,
                    layer_idx=layer_idx,
                    max_genes=int(top_genes),
                    return_combined_scores=True,
                )
            except KeyError:
                set_confident_signatures(self)
                top_terms_dict, top_scores_dict = get_stored_confident_signatures(
                    self,
                    layer_idx=layer_idx,
                    max_genes=int(top_genes),
                    return_combined_scores=True,
                )

            top_terms = []
            top_scores = []
            for factor_name in self.factor_names[layer_idx]:
                genes_k = list(top_terms_dict.get(factor_name, []))
                scores_k = np.asarray(top_scores_dict.get(factor_name, []), dtype=float)
                top_terms.append(genes_k)
                top_scores.append(scores_k.tolist())

            if return_scores:
                return top_terms, top_scores
            return top_terms

        term_names = np.array(self.adata.var_names)
        factor_names_0 = np.array(self.factor_names[0])
        indices_0 = np.arange(len(factor_names_0))

        # Prepare mask for drop_factors to select factors in dot products for upper layers
        if drop_factors is not None and len(drop_factors) > 0:
            drop_set = set(drop_factors)
            keep_indices_0 = [
                i for i, name in enumerate(factor_names_0) if name not in drop_set
            ]
        else:
            keep_indices_0 = indices_0

        term_scores = self.pmeans[f"{self.layer_names[0]}W"][self.factor_lists[0]]
        n_factors = len(self.factor_lists[layer_idx])

        if layer_idx > 0:
            if genes:
                # Compute factor-to-gene mapping for this upper layer
                term_scores = self.pmeans[f"{self.layer_names[layer_idx]}W"][
                    self.factor_lists[layer_idx]
                ][:, self.factor_lists[layer_idx - 1]]
                for layer in range(layer_idx - 1, 0, -1):
                    lower_mat = self.pmeans[f"{self.layer_names[layer]}W"][
                        self.factor_lists[layer]
                    ][:, self.factor_lists[layer - 1]]
                    term_scores = term_scores.dot(lower_mat)
                # For mapping to genes, drop columns of w0 corresponding to factors in drop_factors
                w0 = self.pmeans[f"{self.layer_names[0]}W"][self.factor_lists[0]]
                if len(keep_indices_0) != len(factor_names_0):
                    w0 = w0[keep_indices_0, :]
                term_scores = (
                    term_scores[:, keep_indices_0]
                    if term_scores.shape[1] == len(factor_names_0)
                    else term_scores
                )
                term_scores = term_scores.dot(w0)
            else:
                n_factors_below = len(self.factor_lists[layer_idx - 1])
                term_names = np.arange(n_factors_below).astype(str)
                term_scores = self.pmeans[f"{self.layer_names[layer_idx]}W"][
                    self.factor_lists[layer_idx]
                ][:, self.factor_lists[layer_idx - 1]]

        top_terms = []
        top_scores = []
        for k in range(n_factors):
            top_terms_idx = (term_scores[k, :]).argsort()[::-1]
            top_terms_idx = top_terms_idx[:top_genes]
            top_terms_list = term_names[top_terms_idx].tolist()
            top_terms.append(top_terms_list)
            if sorted_scores:
                sorted_term_scores_k = term_scores[k, :][top_terms_idx]
                top_scores_list = sorted_term_scores_k.tolist()
            else:
                top_scores_list = term_scores[k, :].tolist()
            top_scores.append(top_scores_list)

        if return_scores:
            return top_terms, top_scores
        return top_terms

    def get_signature_sample(
        self,
        rng: Any,
        factor_idx: int,
        layer_idx: int,
        top_genes: int = 10,
        return_scores: bool = False,
    ) -> Union[List[str], Tuple[List[str], np.ndarray]]:
        """Get a single signature sample from the posterior for a factor.

        Args:
            rng: JAX random number generator key
            factor_idx: index of the factor
            layer_idx: layer index of the factor
            top_genes: number of top genes to return
            return_scores: whether to return scores along with gene names

        Returns:
            list of gene names, or tuple of (gene_names, scores) if return_scores is True
        """
        term_names = np.array(self.adata.var_names)

        term_scores_shape = self.global_params[2 + 0][0][self.factor_lists[0]]
        term_scores_rate = np.exp(self.global_params[2 + 0][1][self.factor_lists[0]])
        rng, sample_rng = random.split(rng)
        term_scores_sample = lognormal_sample(
            sample_rng, term_scores_shape, term_scores_rate
        )

        if layer_idx > 0:
            term_scores_shape = self.global_params[2 + layer_idx][0][
                self.factor_lists[layer_idx]
            ][:, self.factor_lists[layer_idx - 1]]

            term_scores_rate = np.exp(
                self.global_params[2 + layer_idx][1][self.factor_lists[layer_idx]][
                    :, self.factor_lists[layer_idx - 1]
                ]
            )
            rng, sample_rng = random.split(rng)
            term_scores_sample = lognormal_sample(
                sample_rng, term_scores_shape, term_scores_rate
            )

            for layer in range(layer_idx - 1, 0, -1):
                lower_mat_shape = self.global_params[2 + layer][0][
                    self.factor_lists[layer]
                ][:, self.factor_lists[layer - 1]]

                lower_mat_rate = np.exp(
                    self.global_params[2 + layer][1][self.factor_lists[layer]][
                        :, self.factor_lists[layer - 1]
                    ]
                )
                rng, sample_rng = random.split(rng)
                lower_mat_sample = lognormal_sample(
                    sample_rng, lower_mat_shape, lower_mat_rate
                )
                term_scores_sample = term_scores_sample.dot(lower_mat_sample)

            lower_term_scores_shape = self.global_params[2 + 0][0][self.factor_lists[0]]

            lower_term_scores_rate = np.exp(
                self.global_params[2 + 0][1][self.factor_lists[0]]
            )
            rng, sample_rng = random.split(rng)
            lower_term_scores_sample = lognormal_sample(
                sample_rng, lower_term_scores_shape, lower_term_scores_rate
            )

            term_scores_sample = term_scores_sample.dot(lower_term_scores_sample)

        top_terms_idx = (term_scores_sample[factor_idx, :]).argsort()[::-1][:top_genes]
        top_terms = term_names[top_terms_idx].tolist()
        top_scores = term_scores_sample[factor_idx, :].tolist()

        if return_scores:
            return top_terms, top_scores
        return top_terms

    def get_signature_confidence(
        self,
        factor_idx: int,
        layer_idx: int,
        mc_samples: int = 100,
        top_genes: int = 10,
        pairwise: bool = False,
    ) -> float:
        """Get confidence score for a factor signature using Monte Carlo sampling.

        Args:
            factor_idx: index of the factor
            layer_idx: layer index of the factor
            mc_samples: number of Monte Carlo samples to take
            top_genes: number of top genes to consider in each sample
            pairwise: whether to compute pairwise Jaccard similarities

        Returns:
            confidence score as Jaccard similarity
        """
        signatures = []
        base_rng = random.PRNGKey(0)
        for i in range(mc_samples):
            rng = random.fold_in(base_rng, i)
            signature_sample = self.get_signature_sample(
                rng,
                factor_idx=factor_idx,
                layer_idx=layer_idx,
                top_genes=top_genes,
            )
            signatures.append(signature_sample)

        if pairwise:
            jaccs = np.zeros((mc_samples, mc_samples))
            for i in range(mc_samples):
                for j in range(mc_samples):
                    jaccs[i, j] = score_utils.jaccard_similarity(
                        [signatures[i], signatures[j]]
                    )
            return np.mean(jaccs)
        else:
            return score_utils.jaccard_similarity(signatures)

    def get_relevances_dict(self) -> Dict[str, float]:
        """Get dictionary of factor relevance scores.

        Returns:
            dictionary mapping factor names to relevance scores
        """
        relevance_dict = {}
        for layer_idx in range(self.n_layers):
            for factor_idx, factor_name in enumerate(self.factor_names[layer_idx]):
                relevance_dict[factor_name] = self.pmeans["factor_means"][
                    self.factor_lists[layer_idx][factor_idx]
                ]
        return relevance_dict

    def get_sizes_dict(self) -> Dict[str, float]:
        """Get dictionary of factor sizes (number of cells per factor).

        Returns:
            dictionary mapping factor names to cell counts
        """
        sizes_dict = {}
        for layer_idx in range(self.n_layers):
            layer_sizes = self.adata.obs[
                f"{self.layer_names[layer_idx]}"
            ].value_counts()
            for factor_idx, factor_name in enumerate(self.factor_names[layer_idx]):
                if factor_name in layer_sizes.keys():
                    sizes_dict[factor_name] = layer_sizes[factor_name]
                else:
                    sizes_dict[factor_name] = 0
        return sizes_dict

    def get_signatures_dict(
        self,
        top_genes: Optional[int] = None,
        scores: bool = False,
        sorted_scores: bool = False,
        layer_normalize: bool = False,
        drop_factors: Optional[List[str]] = None,
    ) -> Union[
        Dict[str, List[str]], Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]
    ]:
        """Get dictionary of gene signatures for all factors across all layers.

        Args:
            top_genes: number of top genes per signature
            scores: whether to return scores along with signatures
            sorted_scores: whether to return scores sorted by ranking
            layer_normalize: whether to normalize scores within each layer
            drop_factors: list of factors to exclude

        Returns:
            dictionary mapping factor names to gene lists, or tuple of (signatures, scores) if scores is True
        """
        signatures_dict = {}
        scores_dict = {}
        for layer_idx in range(self.n_layers):
            layer_signatures, layer_scores = self.get_rankings(
                layer_idx=layer_idx,
                top_genes=top_genes,
                return_scores=True,
                sorted_scores=sorted_scores,
                drop_factors=drop_factors,
            )
            for factor_idx, factor_name in enumerate(self.factor_names[layer_idx]):
                val = np.array(layer_scores[factor_idx])
                if layer_normalize:
                    val = val - np.min(val)
                    val = val / np.max(val)
                signatures_dict[factor_name] = layer_signatures[factor_idx]
                scores_dict[factor_name] = val

        if scores:
            return signatures_dict, scores_dict
        return signatures_dict

    def get_summary(self, top_genes: int = 10, reindex: bool = True) -> str:
        """Get a text summary of the model factors and their top genes.

        Args:
            top_genes: number of top genes to show per factor
            reindex: whether to reindex factors

        Returns:
            string summary of the model
        """
        tokeep = self.factor_lists[0]
        n_factors_eff = len(tokeep)
        genes, scores = self.get_rankings(return_scores=True)

        summary = f"Found {n_factors_eff} factors grouped in the following way:\n"

        # Group the factors
        assignments = []
        for i, factor in enumerate(tokeep):
            assignments.append(
                np.argmax(self.pmeans[f"{self.layer_names[1]}W"][:, factor])
            )
        assignments = np.array(assignments)
        factor_order = np.argsort(np.array(assignments))  # noqa: F841

        for group in np.unique(assignments):
            factors = tokeep[np.where(assignments == group)[0]]
            if len(factors) > 0:
                summary += f"Group {group}:\n"
            for i, factor in enumerate(factors):
                summary += f"    Factor {i}: "
                summary += ", ".join(
                    [
                        f"{genes[factor][j]} ({scores[factor][j]:.3f})"
                        for j in range(top_genes)
                    ]
                )
                summary += "\n"
            summary += "\n"

        self.logger.info(summary)

        return summary

    def get_layer_factor_orders(self) -> List[np.ndarray]:
        """Get the ordering of factors in each layer for plotting.

        Returns:
            list of arrays, one per layer, containing factor indices in plotting order
        """
        layer_factor_orders = []
        for layer_idx in np.arange(0, self.n_layers)[::-1]:  # Go top down
            factors = self.factor_lists[layer_idx]
            n_factors = len(factors)
            if layer_idx < self.n_layers - 1:
                # Assign factors to upper factors to set the plotting order
                mat = self.pmeans[f"{self.layer_names[layer_idx+1]}W"][
                    self.factor_lists[layer_idx + 1]
                ][:, self.factor_lists[layer_idx]]
                normalized_factor_weights = mat / np.sum(mat, axis=1).reshape(-1, 1)
                assignments = []
                for factor_idx in range(n_factors):
                    assignments.append(
                        np.argmax(normalized_factor_weights[:, factor_idx])
                    )
                assignments = np.array(assignments)

                factor_order = []
                for upper_factor_idx in layer_factor_orders[-1]:
                    factor_order.append(np.where(assignments == upper_factor_idx)[0])
                factor_order = np.concatenate(factor_order).astype(int)
                layer_factor_orders.append(factor_order)
            else:
                layer_factor_orders.append(np.arange(n_factors))
        layer_factor_orders = layer_factor_orders[::-1]
        return layer_factor_orders

    def attach_factors_to_obs(self, obs_key: str) -> List[List[str]]:
        """Attach factors to observation categories.

        Args:
            obs_key: key in model.adata.obs to use for attachment

        Returns:
            list of attachment lists, one per layer
        """
        attachments = []
        for layer, layer_name in enumerate(self.layer_names):
            layer_attachments = []
            for factor_idx in range(len(self.factor_lists[layer])):
                factor_name = f"{self.factor_names[layer][int(factor_idx)]}"
                # cells attached to this factor
                cells = np.where(self.adata.obs[f"{layer_name}"] == factor_name)[0]
                if len(cells) > 0:
                    # cells in this factor that belong to each obs
                    prevs = [
                        np.count_nonzero(self.adata.obs[obs_key].iloc[cells] == b)
                        / len(np.where(self.adata.obs[obs_key] == b)[0])
                        for b in self.adata.obs[obs_key].cat.categories
                    ]
                    obs_idx = np.argmax(prevs)  # obs attachment
                    layer_attachments.append(
                        self.adata.obs[obs_key].cat.categories[obs_idx]
                    )
            attachments.append(layer_attachments)
        return attachments

    def compute_weight(self, upper_factor_name: str, lower_factor_name: str) -> float:
        """Compute the weight between two factors across any number of layers.

        Args:
            upper_factor_name: name of the upper factor
            lower_factor_name: name of the lower factor

        Returns:
            weight value between the two factors
        """
        upper_factor_idx = -1
        upper_factor_layer_idx = -1
        for layer_idx in range(self.n_layers):
            layer_factor_names = np.array(self.factor_names[layer_idx])
            if upper_factor_name in layer_factor_names:
                upper_factor_idx = np.where(upper_factor_name == layer_factor_names)[0][
                    0
                ]
                upper_factor_layer_idx = layer_idx
                break

        assert upper_factor_idx != -1

        lower_factor_idx = -1
        lower_factor_layer_idx = -1
        for layer_idx in range(self.n_layers):
            layer_factor_names = np.array(self.factor_names[layer_idx])
            if lower_factor_name in layer_factor_names:
                lower_factor_idx = np.where(lower_factor_name == layer_factor_names)[0][
                    0
                ]
                lower_factor_layer_idx = layer_idx
                break

        assert lower_factor_idx != -1

        upper_layer_name = self.layer_names[upper_factor_layer_idx]
        mat = self.pmeans[f"{upper_layer_name}W"][
            self.factor_lists[upper_factor_layer_idx]
        ][:, self.factor_lists[upper_factor_layer_idx - 1]]
        for layer_idx in range(upper_factor_layer_idx - 1, lower_factor_layer_idx, -1):
            layer_name = self.layer_names[layer_idx]
            lower_mat = self.pmeans[f"{layer_name}W"][self.factor_lists[layer_idx]][
                :, self.factor_lists[layer_idx - 1]
            ]
            mat = mat.dot(lower_mat)

        return mat[upper_factor_idx][lower_factor_idx]

    def compute_factor_obs_association_score(
        self, layer_idx: int, factor_name: str, obs_key: str, obs_val: str
    ) -> float:
        """Compute association score between a factor and an observation category.

        Args:
            layer_idx: layer index of the factor
            factor_name: name of the factor
            obs_key: key in model.adata.obs
            obs_val: value in obs_key to compute association with

        Returns:
            association score value
        """
        layer_name = self.layer_names[layer_idx]

        # Cells attached to factor
        adata_cells_in_factor = self.adata[
            np.where(self.adata.obs[f"{layer_name}"] == factor_name)[0]
        ]

        # Cells from obs_val
        adata_cells_from_obs = self.adata[
            np.where(self.adata.obs[obs_key] == obs_val)[0]
        ]

        cells_from_obs = float(adata_cells_from_obs.shape[0])  # noqa: F841

        # Number of cells from obs_val that are not in factor
        cells_not_in_factor_from_obs = float(
            np.count_nonzero(adata_cells_from_obs.obs[f"{layer_name}"] != factor_name)
        )

        # Number of cells in factor that are obs_val
        cells_in_factor_from_obs = float(
            np.count_nonzero(adata_cells_in_factor.obs[obs_key] == obs_val)
        )

        # Number of cells in factor that are not obs_val
        cells_in_factor_not_from_obs = float(
            np.count_nonzero(adata_cells_in_factor.obs[obs_key] != obs_val)
        )

        return score_utils.compute_fscore(
            cells_in_factor_from_obs,
            cells_in_factor_not_from_obs,
            cells_not_in_factor_from_obs,
        )

    def compute_factor_obs_assignment_fracs(
        self, layer_idx: int, factor_name: str, obs_key: str, obs_val: str
    ) -> float:
        """Compute assignment fraction between a factor and an observation category.

        Args:
            layer_idx: layer index of the factor
            factor_name: name of the factor
            obs_key: key in model.adata.obs
            obs_val: value in obs_key to compute fraction with

        Returns:
            assignment fraction value
        """
        layer_name = self.layer_names[layer_idx]

        # Cells attached to factor
        adata_cells_in_factor = self.adata[
            np.where(self.adata.obs[f"{layer_name}"] == factor_name)[0]
        ]

        # Cells in factor
        cells_in_factor = float(adata_cells_in_factor.shape[0])

        # Cells from factor in obs
        cells_in_factor_from_obs = float(
            np.count_nonzero(adata_cells_in_factor.obs[obs_key] == obs_val)
        )

        score = 0.0
        if cells_in_factor != 0:
            score = cells_in_factor_from_obs / cells_in_factor

        return score

    def compute_factor_obs_weight_score(
        self, layer_idx: int, factor_name: str, obs_key: str, obs_val: str
    ) -> float:
        """Compute weight score between a factor and an observation category.

        Args:
            layer_idx: layer index of the factor
            factor_name: name of the factor
            obs_key: key in model.adata.obs
            obs_val: value in obs_key to compute weight with

        Returns:
            weight score value
        """
        layer_name = self.layer_names[layer_idx]  # noqa: F841

        # Cells from obs_val
        adata_cells_from_obs = self.adata[
            np.where(self.adata.obs[obs_key] == obs_val)[0]
        ]
        adata_cells_not_from_obs = self.adata[
            np.where(self.adata.obs[obs_key] != obs_val)[0]
        ]

        # Weight of cells from obs in factor
        avg_in = np.mean(adata_cells_from_obs.obs[f"{factor_name}_score"])

        # Weight of cells not from obs in factor
        avg_out = np.mean(adata_cells_not_from_obs.obs[f"{factor_name}_score"])

        score = avg_in / np.sum(avg_in + avg_out)

        return score
