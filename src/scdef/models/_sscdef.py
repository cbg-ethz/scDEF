from scdef.models._scdef import scDEF
from anndata import AnnData
import jax
import jax.numpy as jnp
from jax import vmap, random
from jax.scipy.stats import poisson
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import optax

from scdef.utils.jax_utils import lognormal_sample, lognormal_entropy, gamma_logpdf

from typing import Any, Dict, List, Optional, Sequence


class sscDEF(scDEF):
    """Supervised scDEF (sscDEF): cell-type labels fix the top hierarchy.

    Layer sizes and :math:`W` priors match :class:`scDEF` (geometric schedule from
    ``n_factors`` / ``n_layers``, or explicit ``layer_sizes``). Unlike default scDEF
    with multiple top factors, no width-1 root is appended: the coarsest layer is the
    supervised one with one factor per category in ``adata.obs[top_key]``.

    Top-layer cell-factor usage ``z`` is fixed to binary assignments (1 on the annotated
    population, near-zero elsewhere). During fitting the top ``z`` is not sampled from
    the variational posterior and its parameters are not optimized; all other layers
    are learned as in scDEF.

    Args:
        adata: AnnData with counts in ``adata.X`` or ``counts_layer``.
        top_key: ``adata.obs`` column with cell population / cell-type labels.
        n_factors, n_layers, layer_sizes, layer_names: same meaning as :class:`scDEF`
            (``n_layers`` counts layers from L0 through the supervised top; no extra root).
        **kwargs: passed to :class:`scDEF`.
    """

    _Z_OFF_TYPE = 1e-3
    _Z_ON = 1.0

    @staticmethod
    def _geometric_layer_sizes_no_root(
        n_factors: int, top_factors: int, n_layers: int
    ) -> List[int]:
        """Same geometric ladder as :meth:`scDEF._geometric_layer_sizes` without a root."""
        n_layers = max(2, int(n_layers))
        min_k = float(n_factors)
        top_k = float(top_factors)
        ratio = top_k / max(min_k, 1.0)
        if n_layers == 2:
            out = [max(1, int(round(min_k))), int(top_factors)]
            if out[0] < out[1]:
                out[0] = out[1]
            return out

        denom = float(n_layers - 1)
        out: List[int] = []
        prev: Optional[int] = None
        for l in range(n_layers):
            r = l / denom
            if l == n_layers - 1:
                k_i = int(top_factors)
            else:
                k_f = min_k * (ratio**r)
                k_i = max(1, int(round(k_f)))
                if prev is not None and k_i >= prev:
                    k_i = max(1, prev - 1)
            out.append(k_i)
            prev = k_i
        out[-1] = int(top_factors)
        for i in range(len(out) - 2, -1, -1):
            if out[i] < out[i + 1]:
                out[i] = out[i + 1]
        return out

    def __init__(
        self,
        adata: AnnData,
        top_key: str,
        **kwargs: Any,
    ):
        if top_key not in adata.obs.columns:
            raise KeyError(f"top_key `{top_key}` not found in adata.obs.")

        self.top_key = str(top_key)

        labels = adata.obs[self.top_key].astype(str)
        if labels.isna().any():
            raise ValueError(
                f"obs[`{self.top_key}`] contains missing values; fill or drop them first."
            )
        self.population_names: List[str] = sorted(labels.unique())
        self.n_populations = len(self.population_names)
        if self.n_populations < 1:
            raise ValueError(f"obs[`{self.top_key}`] must have at least one category.")
        self.population_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(self.population_names)
        }

        n_factors = int(kwargs.get("n_factors", 100))
        n_layers_schedule = int(kwargs.get("n_layers", 6))
        if n_layers_schedule < 2:
            raise ValueError(
                "sscDEF requires n_layers >= 2 (one supervised top layer plus "
                "at least one learned layer below)."
            )

        layer_sizes = kwargs.get("layer_sizes")
        if layer_sizes is not None:
            layer_sizes = [int(x) for x in layer_sizes]
            if int(layer_sizes[-1]) != self.n_populations:
                raise ValueError(
                    "The last entry of layer_sizes must equal the number of "
                    f"categories in obs[`{self.top_key}`] ({self.n_populations})."
                )
        else:
            layer_sizes = self._geometric_layer_sizes_no_root(
                n_factors, self.n_populations, n_layers_schedule
            )

        layer_names = kwargs.get("layer_names")
        if layer_names is None:
            layer_names = [f"L{i}" for i in range(len(layer_sizes) - 1)] + [
                self.top_key
            ]
        elif len(layer_names) != len(layer_sizes):
            raise ValueError("layer_names must have the same length as layer_sizes.")

        kwargs = dict(kwargs)
        kwargs["layer_sizes"] = layer_sizes
        kwargs["layer_names"] = layer_names
        kwargs["n_factors"] = int(layer_sizes[0])
        kwargs["top_factors"] = self.n_populations
        kwargs["n_layers"] = len(layer_sizes)

        self.supervised_top_layer_idx = len(layer_sizes) - 1

        super(sscDEF, self).__init__(adata, **kwargs)

        logginglevel = self.logger.level
        self.logger = logging.getLogger("sscDEF")
        self.logger.setLevel(logginglevel)
        self._preserve_factor_names_on_annotate = True
        if int(self.layer_sizes[self.supervised_top_layer_idx]) != self.n_populations:
            raise ValueError(
                "Top layer size must match the number of supervised populations."
            )

        self._supervised_top_z = self._build_supervised_top_z()
        self._sscdef_fixed_top_layer_idx = self.supervised_top_layer_idx
        self._sscdef_fixed_top_z = jnp.array(self._supervised_top_z, dtype=jnp.float32)
        self.set_factor_names()
        self.init_var_params(
            init_budgets=True,
            init_alpha=True,
            init_z=self._build_init_z_list(),
            nmf_init=False,
        )
        self.set_posterior_means()
        self.set_posterior_variances()
        self._pin_supervised_top_z_local_params()

    def set_factor_names(self) -> None:
        super(sscDEF, self).set_factor_names()
        top_idx = int(self.n_layers - 1)
        self.factor_names[top_idx] = [
            self.population_names[int(i)] for i in self.factor_lists[top_idx]
        ]

    def _build_supervised_top_z(self) -> np.ndarray:
        labels = self.adata.obs[self.top_key].astype(str).to_numpy()
        k_top = int(self.layer_sizes[self.supervised_top_layer_idx])
        z = np.full((self.n_cells, k_top), self._Z_OFF_TYPE, dtype=np.float32)
        for cell_idx, label in enumerate(labels):
            if label not in self.population_to_idx:
                raise ValueError(
                    f"Cell has population `{label}` not in the training set of "
                    f"`{self.top_key}` categories."
                )
            z[cell_idx, self.population_to_idx[label]] = self._Z_ON
        return z

    def _gamma_params_from_mean(
        self, mean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=np.float64)
        mean = np.clip(mean, 1e-3, 10.0)
        v = np.maximum(mean * 1e-3, 1e-12)
        z_shape = np.log(mean**2 / np.sqrt(mean**2 + v))
        z_rate = np.log(np.sqrt(np.log(1.0 + v / (mean**2))))
        return z_shape.astype(np.float32), z_rate.astype(np.float32)

    def _layer_column_range(self, layer_idx: int) -> tuple[int, int]:
        start = int(sum(self.layer_sizes[:layer_idx]))
        end = start + int(self.layer_sizes[layer_idx])
        return start, end

    def _pin_supervised_top_z_local_params(self) -> None:
        """Reset top-layer variational z parameters to the supervised binary assignment."""
        if len(self.local_params) < 2:
            return
        z_shapes, z_rates = self.local_params[1]
        start, end = self._layer_column_range(self.supervised_top_layer_idx)
        m = self._supervised_top_z
        shape_np, rate_np = self._gamma_params_from_mean(m)
        z_shapes = np.array(z_shapes)
        z_rates = np.array(z_rates)
        z_shapes[:, start:end] = shape_np
        z_rates[:, start:end] = rate_np
        self.local_params[1] = jnp.array((jnp.array(z_shapes), jnp.array(z_rates)))

    def _build_init_z_list(self) -> List[Optional[np.ndarray]]:
        init_z: List[Optional[np.ndarray]] = [None] * self.n_layers
        init_z[self.supervised_top_layer_idx] = self._supervised_top_z
        return init_z

    def _supervised_optimize_layers(
        self, optimize_layers: Optional[Sequence[int]]
    ) -> List[int]:
        if optimize_layers is None:
            return list(range(self.n_layers - 1))
        layers = sorted({int(i) for i in optimize_layers})
        top = self.supervised_top_layer_idx
        if top in layers:
            self.logger.warning(
                "Top supervised layer %s is fixed; excluding it from optimize_layers.",
                top,
            )
            layers = [i for i in layers if i != top]
        return layers

    # Copied from :meth:`scDEF.elbo` with a fixed (non-sampled) supervised top ``z``.
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

            if idx == self.supervised_top_layer_idx:
                _z_shape = jax.lax.stop_gradient(_z_shape)
                _z_rate = jax.lax.stop_gradient(_z_rate)
                _z_sample = self._sscdef_fixed_top_z[indices]
            else:
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
                    alpha_layer / (z_mean),
                ),
            )

            if idx != self.supervised_top_layer_idx:
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

    # Copied from :meth:`scDEF._learn` with supervised ``optimize_layers`` and top-``z`` pinning.
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
        optimize_layers = self._supervised_optimize_layers(optimize_layers)

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
                        self._pin_supervised_top_z_local_params()
                        local_params = self.local_params
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

    def fit(self, **kwargs: Any) -> None:
        """Fit sscDEF; the supervised top layer ``z`` is held fixed."""
        kwargs = dict(kwargs)
        kwargs.pop("optimize_layers", None)
        self._learn_jit_cache = None
        self._pin_supervised_top_z_local_params()
        super(sscDEF, self).fit(**kwargs)
        self._pin_supervised_top_z_local_params()
        self.set_posterior_means()
        self.set_posterior_variances()

    def set_posterior_means(self) -> None:
        super(sscDEF, self).set_posterior_means()
        if not hasattr(self, "_supervised_top_z"):
            return
        layer_name = self.layer_names[self.supervised_top_layer_idx]
        self.pmeans[f"{layer_name}z"] = np.asarray(self._supervised_top_z, dtype=float)

    def set_posterior_variances(self) -> None:
        super(sscDEF, self).set_posterior_variances()
        if not hasattr(self, "_supervised_top_z"):
            return
        layer_name = self.layer_names[self.supervised_top_layer_idx]
        self.pvars[f"{layer_name}z"] = np.zeros_like(
            self._supervised_top_z, dtype=float
        )

    def annotate_adata(self) -> None:
        super(sscDEF, self).annotate_adata()
        layer_name = self.layer_names[self.supervised_top_layer_idx]
        labels = self.adata.obs[self.top_key].astype(str)
        self.adata.obs[layer_name] = labels.values
        self.adata.obs[layer_name] = pd.Categorical(self.adata.obs[layer_name])
        sorted_factors = self.adata.obs_vector(layer_name)
        sorted_colors = []
        for fac in sorted_factors.categories:
            pos = self.population_names.index(str(fac))
            sorted_colors.append(
                self.layer_colorpalettes[self.supervised_top_layer_idx][pos]
            )
        self.adata.uns[f"{layer_name}_colors"] = sorted_colors
        self.adata.obsm[f"X_{layer_name}"] = np.asarray(
            self._supervised_top_z, dtype=float
        )
