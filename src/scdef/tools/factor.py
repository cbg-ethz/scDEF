import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from scipy.stats import norm
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist
from sklearn.preprocessing import minmax_scale
import pandas as pd
from .hierarchy import get_hierarchy, compute_hierarchy_scores
from typing import Optional, Sequence, Dict, List, Tuple, Union, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def _confidence_mean_score(
    confidences: np.ndarray,
    means: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Combine confidence and mean loading into a DE-style ranking score.

    Interprets confidence as ``1 - pvalue`` and mean loading as an effect-size
    proxy. The final score is:

    ``score = mean * -log10(1 - confidence)``.
    """
    confidences = np.asarray(confidences, dtype=float)
    means = np.asarray(means, dtype=float)
    significance = -np.log10(np.clip(1.0 - confidences, eps, 1.0))
    return means * significance


def _get_layer_term_means(model: "scDEF", layer_idx: int) -> np.ndarray:
    """Return per-factor mean loadings aligned with ``adata.var_names``."""
    layer_name = model.layer_names[layer_idx]
    if layer_idx == 0:
        kept = np.asarray(model.factor_lists[layer_idx], dtype=int)
        return np.asarray(model.pmeans[f"{layer_name}W"], dtype=float)[kept]
    term_scores = np.asarray(
        model.pmeans[f"{model.layer_names[layer_idx]}W"], dtype=float
    )[np.asarray(model.factor_lists[layer_idx], dtype=int)][
        :, np.asarray(model.factor_lists[layer_idx - 1], dtype=int)
    ]
    for layer in range(layer_idx - 1, 0, -1):
        lower_mat = np.asarray(
            model.pmeans[f"{model.layer_names[layer]}W"], dtype=float
        )[np.asarray(model.factor_lists[layer], dtype=int)][
            :, np.asarray(model.factor_lists[layer - 1], dtype=int)
        ]
        term_scores = term_scores.dot(lower_mat)
    w0 = np.asarray(model.pmeans[f"{model.layer_names[0]}W"], dtype=float)[
        np.asarray(model.factor_lists[0], dtype=int),
        :,
    ]
    return term_scores.dot(w0)


def _get_confident_signatures_cache(model: "scDEF") -> Dict[str, object]:
    cache = model.adata.uns.get("confident_signatures", None)
    if cache is None:
        raise KeyError(
            "Confident signatures were not precomputed. "
            "Run `scd.tl.set_confident_signatures(model)` first."
        )
    cache_fit_rev = int(cache.get("fit_revision", -1))
    current_fit_rev = int(getattr(model, "_fit_revision", 0))
    if cache_fit_rev != current_fit_rev:
        raise KeyError(
            "Stored confident signatures are stale for this fitted model. "
            "Run `scd.tl.set_confident_signatures(model)` again."
        )
    return cache


def get_stored_confident_signatures(
    model: "scDEF",
    layer_idx: int = 0,
    max_genes: Optional[int] = None,
    return_confidences: bool = False,
    return_combined_scores: bool = False,
) -> Union[
    Dict[str, List[str]],
    Tuple[Dict[str, List[str]], Dict[str, np.ndarray]],
    Tuple[Dict[str, List[str]], Dict[str, np.ndarray], Dict[str, np.ndarray]],
]:
    """Load precomputed confident signatures (and optional scores) from cache."""
    if layer_idx < 0 or layer_idx >= model.n_layers:
        raise ValueError(f"layer_idx must be in [0, {model.n_layers - 1}].")
    cache = _get_confident_signatures_cache(model)
    layer_data = cache["by_layer"][str(int(layer_idx))]
    signatures: Dict[str, List[str]] = {
        k: list(v) for k, v in layer_data["signatures"].items()
    }
    confidences: Dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=float) for k, v in layer_data["confidences"].items()
    }
    combined_scores: Dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=float) for k, v in layer_data["combined_scores"].items()
    }
    if max_genes is not None:
        kmax = int(max_genes)
        signatures = {k: v[:kmax] for k, v in signatures.items()}
        confidences = {k: v[:kmax] for k, v in confidences.items()}
        combined_scores = {k: v[:kmax] for k, v in combined_scores.items()}

    if return_confidences and return_combined_scores:
        return signatures, confidences, combined_scores
    if return_confidences:
        return signatures, confidences
    if return_combined_scores:
        return signatures, combined_scores
    return signatures


def set_confident_signatures(
    model: "scDEF",
    confidence_threshold: float = 0.9,
    tau_quantile: float = 0.99,
    min_effect: Optional[float] = None,
    mc_samples_upper: int = 100,
    random_seed: int = 0,
) -> Dict[str, List[str]]:
    """Precompute and cache confident signatures/scores for all layers.

    Stores signatures, confidence values and combined scores in
    ``model.adata.uns['confident_signatures']`` for reuse by plotting/utilities.
    """
    cache: Dict[str, object] = {
        "fit_revision": int(getattr(model, "_fit_revision", 0)),
        "params": {
            "confidence_threshold": float(confidence_threshold),
            "tau_quantile": float(tau_quantile),
            "min_effect": None if min_effect is None else float(min_effect),
            "mc_samples_upper": int(mc_samples_upper),
            "random_seed": int(random_seed),
        },
        "by_layer": {},
    }
    term_names = np.asarray(model.adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(term_names)}
    signatures_flat: Dict[str, List[str]] = {}

    for layer_idx in range(model.n_layers):
        sigs, confs = get_confident_signatures(
            model,
            layer_idx=layer_idx,
            confidence_threshold=confidence_threshold,
            tau_quantile=tau_quantile,
            min_effect=min_effect,
            max_genes=None,
            mc_samples_upper=mc_samples_upper,
            random_seed=random_seed,
            return_confidences=True,
        )
        term_means = _get_layer_term_means(model, layer_idx)
        layer_combined_scores: Dict[str, List[float]] = {}
        for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
            genes = list(sigs.get(factor_name, []))
            conf_arr = np.asarray(confs.get(factor_name, np.array([])), dtype=float)
            if len(genes) > 0:
                gene_idx = np.asarray([gene_to_idx[g] for g in genes], dtype=int)
                mean_arr = np.asarray(term_means[factor_idx, gene_idx], dtype=float)
            else:
                mean_arr = np.array([], dtype=float)
            n = min(len(genes), len(conf_arr), len(mean_arr))
            genes = genes[:n]
            conf_arr = conf_arr[:n]
            mean_arr = mean_arr[:n]
            combined_arr = _confidence_mean_score(conf_arr, mean_arr)

            sigs[factor_name] = genes
            confs[factor_name] = conf_arr
            layer_combined_scores[factor_name] = combined_arr.tolist()
            signatures_flat[factor_name] = genes

        cache["by_layer"][str(int(layer_idx))] = {
            "layer_name": model.layer_names[layer_idx],
            "signatures": {k: list(v) for k, v in sigs.items()},
            "confidences": {
                k: np.asarray(v, dtype=float).tolist() for k, v in confs.items()
            },
            "combined_scores": layer_combined_scores,
        }

    model.adata.uns["confident_signatures"] = cache
    model.adata.uns["factor_signatures"] = signatures_flat


def factor_diagnostics(
    model: "scDEF",
    recompute: bool = False,
    batch_key: Optional[str] = None,
) -> None:
    """Compute/store factor diagnostics in ``model.adata.uns['factor_obs']``.

    Args:
        model: scDEF model instance
        recompute: if True, force recomputation of the cached fixed upper-layer
            factor subset used for clarity scores, even if the fit revision
            did not change.
        batch_key: optional key in ``model.adata.obs`` used to compute
            per-factor batch entropy and batch purity on top-loaded cells.
            For each layer, cells are assigned to the factor with maximal
            variational ``z`` loading, and purity is defined as
            ``1 - batch_entropy / log(n_batches)``.
    """
    # Keep layer 0 unfiltered, but use a fixed filtered subset on upper layers.
    # Cache and reuse upper-layer factor lists so diagnostics remain stable across
    # later calls to filter/annotate routines.
    cache_key = "_factor_obs_upper_lists_fixed"
    cache_rev_key = "_factor_obs_fit_revision"
    current_fit_rev = int(getattr(model, "_fit_revision", 0))
    reset_reasons: List[str] = []
    if recompute:
        reset_reasons.append("explicit recompute=True")
    if cache_key not in model.adata.uns:
        reset_reasons.append("missing cached upper-layer factor lists")
    elif len(model.adata.uns[cache_key]) != max(model.n_layers - 1, 0):
        reset_reasons.append("cached upper-layer list length mismatch")
    if int(model.adata.uns.get(cache_rev_key, -1)) != current_fit_rev:
        reset_reasons.append(
            f"fit revision changed ({model.adata.uns.get(cache_rev_key, -1)} -> {current_fit_rev})"
        )
    reset_cache = len(reset_reasons) > 0
    if not reset_cache:
        # Validate cached indices against current layer sizes.
        for i, idxs in enumerate(model.adata.uns[cache_key], start=1):
            arr = np.asarray(idxs, dtype=int)
            if np.any(arr < 0) or np.any(arr >= model.layer_sizes[i]):
                reset_cache = True
                reset_reasons.append(
                    f"invalid cached indices for layer {i} (out of bounds)"
                )
                break
    if hasattr(model, "logger"):
        if reset_cache:
            model.logger.info(
                "factor_diagnostics: recomputing cached diagnostics (%s).",
                "; ".join(reset_reasons),
            )
        else:
            model.logger.info(
                "factor_diagnostics: using cached diagnostics (fit revision %s).",
                current_fit_rev,
            )
    if reset_cache:
        model.adata.uns[cache_key] = [
            np.asarray(model.factor_lists[i], dtype=int).tolist()
            for i in range(1, model.n_layers)
        ]
        model.adata.uns[cache_rev_key] = current_fit_rev

    fixed_upper_lists = [
        np.asarray(idxs, dtype=int) for idxs in model.adata.uns[cache_key]
    ]
    old_factor_lists = [np.asarray(f, dtype=int).copy() for f in model.factor_lists]
    old_factor_names = (
        [list(names) for names in model.factor_names]
        if hasattr(model, "factor_names")
        else None
    )
    try:
        model.factor_lists = [
            np.arange(model.layer_sizes[0], dtype=int)
        ] + fixed_upper_lists
        if hasattr(model, "set_factor_names"):
            model.set_factor_names()
        res = compute_hierarchy_scores(
            model,
            use_filtered=True,
            filter_upper_layers=True,
        )
    finally:
        model.factor_lists = old_factor_lists
        if old_factor_names is not None:
            model.factor_names = old_factor_names
    model.adata.uns["factor_obs"] = res["per_factor"]
    model.adata.uns["factor_obs"]["ARD"] = np.array(
        [np.nan] * len(model.adata.uns["factor_obs"])
    )
    model.adata.uns["factor_obs"]["BRD"] = np.array(
        [np.nan] * len(model.adata.uns["factor_obs"])
    )
    factor_obs = model.adata.uns["factor_obs"]
    if "original_factor_idx" not in factor_obs.columns:
        raise KeyError(
            "factor_obs is missing 'original_factor_idx'. Recompute diagnostics with updated compute_hierarchy_scores."
        )

    if "child_layer" in factor_obs.columns:
        l0_rows = factor_obs.index[factor_obs["child_layer"] == model.layer_names[0]]
    else:
        l0_rows = factor_obs.index

    original_idx = factor_obs.loc[l0_rows, "original_factor_idx"].to_numpy(dtype=int)
    valid = (original_idx >= 0) & (original_idx < int(model.layer_sizes[0]))
    l0_rows = np.asarray(l0_rows)[valid]
    original_idx = original_idx[valid]

    ard_all = np.asarray(model.pmeans["factor_means"]).ravel()
    brd_all = np.asarray(model.pmeans["factor_concentrations"]).ravel()
    factor_obs.loc[l0_rows, "ARD"] = ard_all[original_idx]
    factor_obs.loc[l0_rows, "BRD"] = brd_all[original_idx]
    factor_obs["technical"] = False
    factor_obs["batch_entropy"] = np.nan
    factor_obs["batch_purity"] = np.nan

    if batch_key is not None:
        if batch_key not in model.adata.obs.columns:
            raise KeyError(
                f"batch_key '{batch_key}' not found in model.adata.obs. "
                f"Available keys: {list(model.adata.obs.columns)}"
            )

        batch_values = model.adata.obs[batch_key].to_numpy()
        valid_mask = ~pd.isna(batch_values)
        unique_batches = np.unique(batch_values[valid_mask])
        n_batches = int(len(unique_batches))

        if n_batches >= 2:
            batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
            batch_idx = np.full(model.n_cells, -1, dtype=int)
            for i, b in enumerate(batch_values):
                if not pd.isna(b):
                    batch_idx[i] = batch_to_idx[b]

            z_means_full = np.asarray(model.local_params[1][0], dtype=float)
            offsets = np.cumsum([0] + [int(s) for s in model.layer_sizes]).astype(int)

            for factor_name, row in factor_obs.iterrows():
                if "child_layer" in factor_obs.columns:
                    layer_name = str(row["child_layer"])
                    if layer_name not in model.layer_names:
                        continue
                    layer_idx = model.layer_names.index(layer_name)
                else:
                    if not isinstance(factor_name, str) or "_" not in factor_name:
                        continue
                    layer_name = factor_name.rsplit("_", 1)[0]
                    if layer_name not in model.layer_names:
                        continue
                    layer_idx = model.layer_names.index(layer_name)

                if "original_factor_idx" in factor_obs.columns:
                    original_factor_idx = int(row["original_factor_idx"])
                else:
                    try:
                        original_factor_idx = int(str(factor_name).rsplit("_", 1)[1])
                    except (ValueError, IndexError):
                        continue
                if original_factor_idx < 0 or original_factor_idx >= int(
                    model.layer_sizes[layer_idx]
                ):
                    continue

                start = int(offsets[layer_idx])
                end = int(offsets[layer_idx + 1])
                layer_scores = z_means_full[:, start:end]
                winner = np.argmax(layer_scores, axis=1)
                selected_cells = np.where(winner == original_factor_idx)[0]
                if selected_cells.size == 0:
                    continue

                selected_batches = batch_idx[selected_cells]
                selected_batches = selected_batches[selected_batches >= 0]
                if selected_batches.size == 0:
                    continue

                counts = np.bincount(selected_batches, minlength=n_batches).astype(
                    float
                )
                probs = counts / counts.sum()
                probs = probs[probs > 0.0]
                entropy = float(-np.sum(probs * np.log(probs)))
                max_entropy = float(np.log(n_batches))
                purity = 1.0 - entropy / max_entropy if max_entropy > 0 else np.nan

                factor_obs.at[row.name, "batch_entropy"] = entropy
                factor_obs.at[row.name, "batch_purity"] = float(
                    np.clip(purity, 0.0, 1.0)
                )

        if "factor_diagnostics" not in model.adata.uns:
            model.adata.uns["factor_diagnostics"] = {}
        model.adata.uns["factor_diagnostics"]["batch_key"] = str(batch_key)
        model.adata.uns["factor_diagnostics"]["n_batches"] = int(n_batches)
        model.adata.uns["factor_diagnostics"]["batch_values"] = [
            str(b) for b in unique_batches
        ]

    # Cache a complete snapshot of factor_obs keyed by (child_layer,
    # original_factor_idx). This is the source of truth for filtering
    # decisions (e.g. get_effective_factors / filter_factors), so that
    # re-filtering with looser thresholds still sees diagnostics for
    # factors previously dropped from the live factor_obs view.
    model.adata.uns["factor_obs_full"] = factor_obs.copy()


def set_factor_signatures(
    model: "scDEF",
    signatures: Optional[Dict[str, List[str]]] = None,
    top_genes: int = 10,
) -> Dict[str, List[str]]:
    if signatures is None:
        signatures = {}
        for layer_idx in range(model.n_layers):
            layer_sigs = get_stored_confident_signatures(
                model, layer_idx=layer_idx, max_genes=top_genes
            )
            signatures.update(layer_sigs)
    model.adata.uns["factor_signatures"] = signatures
    return signatures


def get_obs_score_rankings(
    model: "scDEF",
    layer: Union[int, str],
    obs_key: str,
    obs_values: Union[str, Sequence[str]],
    score_model: Literal["f1", "fracs", "weights", "prob"] = "fracs",
    ascending: bool = False,
    recompute: bool = False,
) -> pd.DataFrame:
    """Return per-obs-value factor rankings by observation association score.

    This reads cached matrices from ``model.adata.uns['obs_scores']`` (written by
    ``scd.pl.obs_scores``). If cache is missing/stale for the requested key/model,
    it is recomputed on demand for the requested ``obs_key`` and ``score_model``.
    """
    if isinstance(layer, str):
        if layer not in model.layer_names:
            raise ValueError(f"Unknown layer '{layer}'. Valid: {model.layer_names}.")
        layer_idx = model.layer_names.index(layer)
    else:
        layer_idx = int(layer)
    if layer_idx < 0 or layer_idx >= model.n_layers:
        raise ValueError(f"layer must be in [0, {model.n_layers - 1}].")

    if isinstance(obs_values, str):
        obs_values = [obs_values]
    obs_values = list(obs_values)
    if len(obs_values) == 0:
        raise ValueError("obs_values must contain at least one value.")

    from ..utils import data_utils

    fit_rev = int(getattr(model, "_fit_revision", 0))
    cache_root = model.adata.uns.get("obs_scores", {})
    mode_cache = cache_root.get(score_model, {})
    need_recompute = recompute
    if int(mode_cache.get("fit_revision", -1)) != fit_rev:
        need_recompute = True
    if "obs_keys" not in mode_cache or obs_key not in mode_cache["obs_keys"]:
        need_recompute = True

    if need_recompute:
        if score_model == "f1":
            score_func = data_utils.get_assignment_scores
        elif score_model == "fracs":
            score_func = data_utils.get_assignment_fracs
        elif score_model == "weights":
            score_func = data_utils.get_weight_scores
        elif score_model == "prob":
            score_func = data_utils.get_prob_scores
        else:
            raise ValueError(
                "score_model must be one of ['f1', 'fracs', 'weights', 'prob']."
            )

        obs_mats, obs_clusters, obs_vals_dict = data_utils.prepare_obs_factor_scores(
            model,
            [obs_key],
            score_func,
        )
        data_utils.cache_obs_factor_scores(
            model=model,
            obs_keys=[obs_key],
            mode=score_model,
            obs_mats=obs_mats,
            obs_clusters=obs_clusters,
            obs_vals_dict=obs_vals_dict,
        )
        cache_root = model.adata.uns.get("obs_scores", {})
        mode_cache = cache_root.get(score_model, {})

    obs_entry = mode_cache["obs_keys"][obs_key]
    available_obs_values = list(obs_entry["obs_values"])
    missing = [v for v in obs_values if v not in available_obs_values]
    if len(missing) > 0:
        raise ValueError(
            f"obs_values {missing} not found for obs_key '{obs_key}'. "
            f"Available values: {available_obs_values}."
        )

    layer_entry = obs_entry["layers"][str(int(layer_idx))]
    factor_names = list(layer_entry["factor_names"])
    score_mat = np.asarray(layer_entry["scores"], dtype=float)
    row_idx = [available_obs_values.index(v) for v in obs_values]
    selected = score_mat[row_idx, :]
    if selected.ndim == 1:
        selected = selected[None, :]

    per_obs_frames = []
    for i, obs_value in enumerate(obs_values):
        per_obs_frames.append(
            pd.DataFrame(
                {
                    "factor": factor_names,
                    "layer": model.layer_names[layer_idx],
                    "layer_idx": int(layer_idx),
                    "obs_key": obs_key,
                    "obs_value": obs_value,
                    "score_model": score_model,
                    "score": selected[i, :],
                }
            )
        )
    df = pd.concat(per_obs_frames, axis=0, ignore_index=True)
    obs_order = {v: i for i, v in enumerate(obs_values)}
    df["_obs_value_order"] = df["obs_value"].map(obs_order)
    df = df.sort_values(
        by=["_obs_value_order", "score"],
        ascending=[True, ascending],
    ).reset_index(drop=True)
    return df.drop(columns=["_obs_value_order"])


def get_obs_value_specific_factors(
    model: "scDEF",
    layer: Union[int, str],
    obs_key: str,
    obs_values: Union[str, Sequence[str]],
    score_model: Literal["f1", "fracs", "weights", "prob"] = "fracs",
    min_specificity: float = 0.0,
    top_n: Optional[int] = None,
    recompute: bool = False,
    return_scores: bool = False,
) -> Union[Dict[str, List[str]], pd.DataFrame]:
    """Get factors specific to each obs value in a layer.

    Specificity is defined within the provided ``obs_values`` as:
    ``specificity = score(obs_value) - max(score(other_obs_values))``.
    Higher values indicate stronger specificity for that obs category.
    """
    if isinstance(obs_values, str):
        obs_values = [obs_values]
    obs_values = list(obs_values)
    if len(obs_values) == 0:
        raise ValueError("obs_values must contain at least one value.")

    ranked = get_obs_score_rankings(
        model=model,
        layer=layer,
        obs_key=obs_key,
        obs_values=obs_values,
        score_model=score_model,
        recompute=recompute,
        ascending=False,
    )

    score_table = ranked.pivot(index="factor", columns="obs_value", values="score")
    rows: List[Dict[str, object]] = []
    for obs_value in obs_values:
        others = [v for v in obs_values if v != obs_value]
        score_v = score_table[obs_value].to_numpy(dtype=float)
        if len(others) == 0:
            best_other = np.zeros_like(score_v)
            best_other_name = np.array([""] * len(score_v), dtype=object)
        else:
            other_mat = score_table[others].to_numpy(dtype=float)
            best_other_idx = np.argmax(other_mat, axis=1)
            best_other = other_mat[np.arange(other_mat.shape[0]), best_other_idx]
            best_other_name = np.asarray(others, dtype=object)[best_other_idx]

        specificity = score_v - best_other
        for i, factor_name in enumerate(score_table.index.tolist()):
            rows.append(
                {
                    "factor": factor_name,
                    "obs_value": obs_value,
                    "score": float(score_v[i]),
                    "best_other_obs_value": best_other_name[i],
                    "best_other_score": float(best_other[i]),
                    "specificity": float(specificity[i]),
                }
            )

    spec_df = pd.DataFrame(rows)
    spec_df = spec_df[spec_df["specificity"] >= float(min_specificity)].copy()
    spec_df["layer"] = ranked["layer"].iloc[0]
    spec_df["layer_idx"] = int(ranked["layer_idx"].iloc[0])
    spec_df["obs_key"] = obs_key
    spec_df["score_model"] = score_model
    spec_df = spec_df.sort_values(
        ["obs_value", "specificity", "score"],
        ascending=[True, False, False],
    )

    if top_n is not None:
        top_n = int(top_n)
        spec_df = spec_df.groupby("obs_value", as_index=False, group_keys=False).head(
            top_n
        )

    spec_df = spec_df.reset_index(drop=True)
    if return_scores:
        return spec_df
    return {
        obs_value: spec_df.loc[spec_df["obs_value"] == obs_value, "factor"].tolist()
        for obs_value in obs_values
    }


def assign_confident(
    model: "scDEF",
    n_samples: int = 500,
    tau: float = 0.3,
    credible_level: float = 0.9,
    key_added: str = "confident",
    rng_key=None,
) -> None:
    """Pick the finest scDEF layer at which each cell is confidently assigned.

    For each cell ``c`` and layer ``k`` (restricted to filtered factors
    ``model.factor_lists[k]``), draws ``n_samples`` reparameterized samples
    ``z^(s) ~ q(z_{c,k})`` from the log-normal variational posterior and
    normalizes each sample: ``ẑ^(s) = z^(s) / sum(z^(s))``.

    The confidence score is defined on the **gap** between the
    cell-level winner and its nearest competitor, so the score is
    invariant to layer size ``K_k`` (only the top two factors enter,
    dormant factors don't inflate it):

    - **Cell-level winner** ``f* = argmax_f E_s[ẑ_f]``.
    - **Per-sample gap** ``gap^(s) = ẑ^(s)_{f*} - max_{g != f*} ẑ^(s)_g``.
      Equals 0 when the winner and runner-up are tied, ``1`` when the
      winner holds all the mass, and goes negative in samples where
      some other factor out-competes ``f*`` (so identity flipping
      directly penalizes the score).
    - **Effect size** ``= E_s[gap^(s)]`` — posterior-mean margin by
      which the winner beats its nearest competitor.
    - **Posterior SD** ``= SD_s[gap^(s)]`` — uncertainty of that
      margin (diagnostic).
    - **Confidence** ``= quantile_{1 - credible_level}({gap^(s)})`` —
      the empirical ``(1 - credible_level)``-quantile of the gap across
      posterior samples. Reads as:

        "In at least ``credible_level`` of posterior samples, the
         winning factor ``f*`` had at least ``confidence`` more
         normalized mass than any competing factor."

      Layer-size invariant: a value of e.g. ``0.3`` means "winner leads
      runner-up by 30 percentage points of normalized mass" regardless
      of the number of factors at the layer.

    Two auxiliary diagnostic scores are also computed:

    - ``winner_probability[c, k] = max_f P_s[argmax = f]`` — how often
      the winner is the argmax across samples. Identity-stability only;
      ignores magnitude (a tight ``0.51/0.49`` scores ``1.0`` here).
    - ``entropy_confidence[c, k] = 1 - H(p) / log(K_k)`` — normalized
      entropy of the argmax distribution.

    **Selection rule (finest-that-clears)**. For each cell, the "best"
    layer is the finest (lowest index) multi-factor layer
    (``K_k >= 2``) whose ``confidence`` clears ``tau``. If no
    multi-factor layer clears ``tau``, the cell is assigned to the
    top-most single-factor layer (typically the root) as a
    "stem-cell-like" catch-all. This encodes the biological intuition:

    - Terminally differentiated cells have a clear dominant factor at
      L0 with tight posterior → assigned at L0.
    - Partially differentiated cells are ambiguous between siblings at
      L0 but clear at their parent layer L1 → assigned at L1.
    - Stem-like cells are ambiguous everywhere → assigned at the root.

    Writes:

    - ``adata.obsm[f"{key_added}_effect_size"]`` — ``(n_cells, n_layers)``
      float. Posterior-mean **gap** between winner and nearest competitor.
    - ``adata.obsm[f"{key_added}_posterior_sd"]`` — ``(n_cells, n_layers)``
      float. Posterior SD of the gap.
    - ``adata.obsm[f"{key_added}_confidence"]`` — ``(n_cells, n_layers)``
      float. Lower empirical quantile of the gap — the layer-size-invariant
      score that ``tau`` gates on.
    - ``adata.obsm[f"{key_added}_winner_mass"]`` — ``(n_cells, n_layers)``
      float. Diagnostic: posterior mean of the winner's normalized mass
      (``E[ẑ_{f*}]``). Not K-invariant.
    - ``adata.obsm[f"{key_added}_winner_probability"]`` — ``(n_cells, n_layers)``
      float. Diagnostic: posterior argmax-identity probability.
    - ``adata.obsm[f"{key_added}_entropy_confidence"]`` — ``(n_cells, n_layers)``
      float. Diagnostic: normalized-entropy score.
    - ``adata.obsm[f"{key_added}_argmax_factor"]`` — ``(n_cells, n_layers)`` int,
      slot indices into the filtered factor list of each layer (``-1`` if the
      layer has no kept factors).
    - ``adata.obs[f"{key_added}_confidence_{layer_name}"]`` per layer.
    - ``adata.obs[f"{key_added}_argmax_{layer_name}"]`` per layer (factor name).
    - ``adata.obs[f"{key_added}_best_layer"]`` — layer name of the chosen layer.
    - ``adata.obs[f"{key_added}_best_layer_idx"]`` — integer index of that layer
      in ``model.layer_names`` (``0`` = finest). ``-1`` if no layer was chosen.
    - ``adata.obs[f"{key_added}_best_factor_idx"]`` — slot within the best
      layer's filtered factor list.
    - ``adata.obs[f"{key_added}_factor"]`` — factor name at the best layer.
    - ``adata.obs[f"{key_added}_best_effect_size"]`` — effect size at the
      best layer.
    - ``adata.obs[f"{key_added}_best_posterior_sd"]`` — posterior SD at the
      best layer.
    - ``adata.obs[f"{key_added}_best_confidence"]`` — combined confidence
      at the best layer.
    - ``adata.obs[f"{key_added}_depth_score"]`` — assignment-centric depth
      in ``[0, 1]``: ``best_layer_index / (n_layers - 1)``, so ``0`` at the
      finest layer (L0) and ``1`` at the coarsest layer (root). If
      ``n_layers == 1``, the score is ``0`` for assigned cells. Cells with
      no valid layer (``best_layer_index < 0``) get ``NaN``.
    - ``adata.uns[key_added]`` — metadata (layer names, sizes, ``tau``,
      ``n_samples``, ``credible_level``, metric name, fit revision).

    Args:
        model: scDEF model instance (must already be fitted).
        n_samples: number of Monte Carlo samples ``S`` drawn per cell/layer.
        tau: minimum confidence (lower quantile of the winner-runner-up
            gap) for a multi-factor layer to be eligible as the cell's
            best layer (in ``[0, 1]``). Reads as "in at least
            ``credible_level`` of posterior samples, the winner must lead
            the runner-up by at least ``tau`` of the normalized mass".
        credible_level: posterior credibility of the lower bound, in
            ``(0, 1)``. The confidence is the empirical
            ``(1 - credible_level)``-quantile of the per-sample gap.
            Default ``0.9`` → "in 90% of posterior samples, the winner
            led by at least ``confidence``". Set higher (e.g. ``0.95``)
            for a stricter score.
        key_added: prefix used for all written keys in ``adata``.
        rng_key: optional ``jax.random`` key; if ``None``, derived from
            ``model.seed``.
    """
    import jax
    import jax.numpy as jnp
    from jax import random, lax

    if int(n_samples) <= 0:
        raise ValueError("n_samples must be > 0.")
    if not (0.0 <= float(tau) <= 1.0):
        raise ValueError("tau must be in [0, 1].")
    if not (0.0 < float(credible_level) < 1.0):
        raise ValueError("credible_level must be in (0, 1).")

    if rng_key is None:
        rng_key = random.PRNGKey(int(getattr(model, "seed", 0)))

    z_params = model.local_params[1]
    mu_full = jnp.asarray(z_params[0])
    log_std_full = jnp.asarray(z_params[1])

    n_cells = int(mu_full.shape[0])
    n_layers = int(model.n_layers)
    layer_sizes = [int(s) for s in model.layer_sizes]
    layer_names = [str(model.layer_names[i]) for i in range(n_layers)]
    n_samples = int(n_samples)

    effect_size_mat = np.zeros((n_cells, n_layers), dtype=float)
    posterior_sd_mat = np.zeros((n_cells, n_layers), dtype=float)
    confidence_mat = np.zeros((n_cells, n_layers), dtype=float)
    winner_mass_mat = np.zeros((n_cells, n_layers), dtype=float)
    winner_prob_mat = np.zeros((n_cells, n_layers), dtype=float)
    entropy_conf_mat = np.zeros((n_cells, n_layers), dtype=float)
    argmax_mat = np.full((n_cells, n_layers), -1, dtype=int)
    label_mat = np.empty((n_cells, n_layers), dtype=object)
    label_mat[:] = ""

    layer_rng_keys = random.split(rng_key, max(n_layers, 1))

    for layer_idx in range(n_layers):
        start = int(sum(layer_sizes[:layer_idx]))
        kept = np.asarray(model.factor_lists[layer_idx], dtype=int)
        K_k = int(len(kept))
        if K_k == 0:
            continue

        kept_cols = jnp.asarray(start + kept, dtype=jnp.int32)
        mu_k = mu_full[:, kept_cols]
        sigma_k = jnp.exp(log_std_full[:, kept_cols])
        sample_keys = random.split(layer_rng_keys[layer_idx], n_samples)

        # Single-factor layer: the only factor has all the normalized
        # mass (ẑ ≡ 1). No runner-up exists; define confidence = 1.
        if K_k == 1:
            argmax_slot = np.zeros(n_cells, dtype=int)
            effect_size_mat[:, layer_idx] = 1.0
            posterior_sd_mat[:, layer_idx] = 0.0
            confidence_mat[:, layer_idx] = 1.0
            winner_mass_mat[:, layer_idx] = 1.0
            winner_prob_mat[:, layer_idx] = 1.0
            entropy_conf_mat[:, layer_idx] = 1.0
            argmax_mat[:, layer_idx] = argmax_slot
            names_arr = np.asarray(model.factor_names[layer_idx], dtype=object)
            label_mat[:, layer_idx] = names_arr[argmax_slot]
            continue

        # Pass 1: accumulate summary statistics over posterior samples.
        # Memory O(n_cells * K_k) for the state.
        #   - argmax one-hot counts → posterior P(argmax = f)
        #   - normalized mass       → posterior E[ẑ_f] (used to pick f*)
        def _accumulate_step(state, key):
            counts, sum_zhat = state
            eps = random.normal(key, shape=mu_k.shape)
            z = jnp.exp(mu_k + sigma_k * eps)
            zhat = z / jnp.clip(jnp.sum(z, axis=-1, keepdims=True), 1e-30, None)
            a = jnp.argmax(zhat, axis=-1)
            one_hot = jax.nn.one_hot(a, num_classes=K_k, dtype=jnp.float32)
            return (counts + one_hot, sum_zhat + zhat), None

        init_state = (
            jnp.zeros((n_cells, K_k), dtype=jnp.float32),
            jnp.zeros((n_cells, K_k), dtype=jnp.float32),
        )
        (counts_final, sum_zhat_final), _ = lax.scan(
            _accumulate_step, init_state, sample_keys
        )
        p = counts_final / float(n_samples)
        mean_zhat = sum_zhat_final / float(n_samples)
        argmax_slot_jax = jnp.argmax(mean_zhat, axis=-1)  # (n_cells,)

        # Pass 2: re-draw samples with the same keys and, for each sample,
        # compute gap^(s) = ẑ^(s)[f*] - max_{g != f*} ẑ^(s)[g] at the
        # cell-level winner slot f*. Collect into a (S, n_cells) matrix
        # from which we take the exact empirical quantile, mean, and SD.
        # Memory O(S * n_cells) — no (S, n_cells, K_k) tensor.
        rows_jax = jnp.arange(n_cells)
        winner_onehot = jax.nn.one_hot(argmax_slot_jax, K_k, dtype=bool)

        def _gather_step(_, key):
            eps = random.normal(key, shape=mu_k.shape)
            z = jnp.exp(mu_k + sigma_k * eps)
            zhat = z / jnp.clip(jnp.sum(z, axis=-1, keepdims=True), 1e-30, None)
            winner_mass = zhat[rows_jax, argmax_slot_jax]
            zhat_others = jnp.where(winner_onehot, -jnp.inf, zhat)
            runner_up_mass = jnp.max(zhat_others, axis=-1)
            gap = winner_mass - runner_up_mass
            return None, (gap, winner_mass)

        _, (gap_samples, winner_mass_samples) = lax.scan(
            _gather_step, None, sample_keys
        )
        # gap_samples, winner_mass_samples: (S, n_cells)

        q_level = 1.0 - float(credible_level)
        conf_jax = jnp.quantile(gap_samples, q=q_level, axis=0)
        mean_gap_jax = jnp.mean(gap_samples, axis=0)
        sd_gap_jax = jnp.std(gap_samples, axis=0)
        winner_mass_mean_jax = jnp.mean(winner_mass_samples, axis=0)

        p_np = np.asarray(p, dtype=float)
        argmax_slot = np.asarray(argmax_slot_jax, dtype=int)

        # Effect size: posterior-mean margin of winner over runner-up.
        effect_size = np.clip(np.asarray(mean_gap_jax, dtype=float), -1.0, 1.0)

        # Posterior SD of the gap (uncertainty of the margin).
        posterior_sd = np.clip(np.asarray(sd_gap_jax, dtype=float), 0.0, 1.0)

        # Primary confidence: exact empirical lower quantile of the gap.
        conf = np.clip(np.asarray(conf_jax, dtype=float), 0.0, 1.0)

        # Diagnostic: posterior mean of the winner's normalized mass
        # (not K-invariant; included as a direct interpretability aid).
        winner_mass = np.clip(np.asarray(winner_mass_mean_jax, dtype=float), 0.0, 1.0)

        # Diagnostic: argmax-identity probability.
        winner_prob = np.clip(np.max(p_np, axis=-1), 0.0, 1.0)

        # Diagnostic: normalized entropy of argmax distribution.
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_p = np.clip(p_np, 1e-30, 1.0)
            H = -np.sum(p_np * np.log(safe_p), axis=-1)
        entropy_conf = np.clip(1.0 - H / float(np.log(K_k)), 0.0, 1.0)

        effect_size_mat[:, layer_idx] = effect_size
        posterior_sd_mat[:, layer_idx] = posterior_sd
        confidence_mat[:, layer_idx] = conf
        winner_mass_mat[:, layer_idx] = winner_mass
        winner_prob_mat[:, layer_idx] = winner_prob
        entropy_conf_mat[:, layer_idx] = entropy_conf
        argmax_mat[:, layer_idx] = argmax_slot
        names_arr = np.asarray(model.factor_names[layer_idx], dtype=object)
        label_mat[:, layer_idx] = names_arr[argmax_slot]

    # Per-cell aggregation across layers.
    #
    # Selection rule: "finest layer that clears tau".
    #   1) Scan multi-factor layers (K_k >= 2) from finest (lowest idx)
    #      to coarsest and take the first one whose confidence clears
    #      `tau`.
    #   2) Otherwise fall back to the top-most single-factor layer
    #      (typically the root), which acts as a "stem-cell-like"
    #      catch-all for cells not confidently assigned at any finer
    #      multi-factor layer.
    is_primary_layer = np.asarray(
        [len(model.factor_lists[i]) >= 2 for i in range(n_layers)],
        dtype=bool,
    )
    has_factors = np.asarray(
        [len(model.factor_lists[i]) > 0 for i in range(n_layers)],
        dtype=bool,
    )

    primary_mask = (confidence_mat >= float(tau)) & is_primary_layer[None, :]
    has_primary = np.any(primary_mask, axis=1)

    # Finest-first: argmax over a mask is the first True index, which is
    # the finest (lowest-index) multi-factor layer clearing tau.
    finest_primary = np.argmax(primary_mask, axis=1)

    fallback_layer = -1
    for i in range(n_layers - 1, -1, -1):
        if has_factors[i] and not is_primary_layer[i]:
            fallback_layer = i
            break

    best_layer_idx_raw = np.where(
        has_primary,
        finest_primary,
        fallback_layer,
    )
    any_cand = best_layer_idx_raw >= 0

    best_layer_idx = np.full(n_cells, -1, dtype=int)
    best_confidence = np.full(n_cells, np.nan, dtype=float)
    best_effect_size = np.full(n_cells, np.nan, dtype=float)
    best_posterior_sd = np.full(n_cells, np.nan, dtype=float)
    best_factor_slot = np.full(n_cells, -1, dtype=int)
    best_layer_name = np.empty(n_cells, dtype=object)
    best_layer_name[:] = ""
    best_label = np.empty(n_cells, dtype=object)
    best_label[:] = ""

    if np.any(any_cand):
        idxs = np.where(any_cand)[0]
        sel = best_layer_idx_raw[any_cand]
        best_layer_idx[idxs] = sel
        best_confidence[idxs] = confidence_mat[idxs, sel]
        best_effect_size[idxs] = effect_size_mat[idxs, sel]
        best_posterior_sd[idxs] = posterior_sd_mat[idxs, sel]
        best_factor_slot[idxs] = argmax_mat[idxs, sel]
        best_layer_name[idxs] = np.asarray(
            [layer_names[int(k)] for k in sel], dtype=object
        )
        best_label[idxs] = label_mat[idxs, sel]

    depth_score = np.full(n_cells, np.nan, dtype=float)
    if n_layers <= 1:
        depth_score[best_layer_idx >= 0] = 0.0
    else:
        valid_depth = best_layer_idx >= 0
        depth_score[valid_depth] = best_layer_idx[valid_depth].astype(float) / float(
            n_layers - 1
        )

    adata = model.adata
    adata.obsm[f"{key_added}_effect_size"] = effect_size_mat
    adata.obsm[f"{key_added}_posterior_sd"] = posterior_sd_mat
    adata.obsm[f"{key_added}_confidence"] = confidence_mat
    adata.obsm[f"{key_added}_winner_mass"] = winner_mass_mat
    adata.obsm[f"{key_added}_winner_probability"] = winner_prob_mat
    adata.obsm[f"{key_added}_entropy_confidence"] = entropy_conf_mat
    adata.obsm[f"{key_added}_argmax_factor"] = argmax_mat

    for layer_idx in range(n_layers):
        layer_name = layer_names[layer_idx]
        adata.obs[f"{key_added}_confidence_{layer_name}"] = confidence_mat[:, layer_idx]
        adata.obs[f"{key_added}_argmax_{layer_name}"] = pd.Categorical(
            label_mat[:, layer_idx].astype(str)
        )

    adata.obs[f"{key_added}_best_layer"] = pd.Categorical(best_layer_name.astype(str))
    adata.obs[f"{key_added}_best_layer_idx"] = best_layer_idx
    adata.obs[f"{key_added}_best_factor_idx"] = best_factor_slot
    adata.obs[f"{key_added}_factor"] = pd.Categorical(best_label.astype(str))
    adata.obs[f"{key_added}_best_effect_size"] = best_effect_size
    adata.obs[f"{key_added}_best_posterior_sd"] = best_posterior_sd
    adata.obs[f"{key_added}_best_confidence"] = best_confidence
    adata.obs[f"{key_added}_depth_score"] = depth_score

    adata.uns[key_added] = {
        "layer_names": list(layer_names),
        "layer_sizes_filtered": [
            int(len(model.factor_lists[i])) for i in range(n_layers)
        ],
        "tau": float(tau),
        "n_samples": int(n_samples),
        "credible_level": float(credible_level),
        "quantile_level": float(1.0 - float(credible_level)),
        "metric": "empirical_lower_quantile_winner_runner_up_gap",
        "selection_rule": "finest_layer_clearing_tau",
        "depth_score": "best_layer_index / max(n_layers - 1, 1); single-layer models use 0",
        "fit_revision": int(getattr(model, "_fit_revision", 0)),
    }


def set_cell_entropies(
    model: "scDEF",
    layers: Optional[Sequence[Union[int, str]]] = None,
    key_suffix: str = "entropy",
    effective_suffix: str = "effective_n_factors",
    normalize: bool = True,
    eps: float = 1e-12,
) -> List[str]:
    """Compute per-cell assignment entropy and store one column per layer.

    For each selected layer, uses ``model.adata.obsm[f"X_{layer_name}"]`` to
    build per-cell membership probabilities and computes Shannon entropy.

    If ``normalize=True``, entropy is divided by ``log(n_factors_layer)`` so
    values are approximately in ``[0, 1]`` (for layers with >1 factors).

    Also stores an effective number of factors per cell, defined as
    ``exp(H)`` where ``H`` is the non-normalized Shannon entropy.

    Returns:
        List of created/updated entropy column names.
    """
    if layers is None:
        layer_indices = list(range(model.n_layers))
    else:
        layer_indices = []
        for layer in layers:
            if isinstance(layer, str):
                if layer not in model.layer_names:
                    raise ValueError(
                        f"Unknown layer '{layer}'. Valid: {model.layer_names}."
                    )
                layer_indices.append(model.layer_names.index(layer))
            else:
                layer_idx = int(layer)
                if layer_idx < 0 or layer_idx >= model.n_layers:
                    raise ValueError(f"layer must be in [0, {model.n_layers - 1}].")
                layer_indices.append(layer_idx)

    created_cols: List[str] = []
    for layer_idx in layer_indices:
        layer_name = model.layer_names[layer_idx]
        obsm_key = f"X_{layer_name}"
        if obsm_key not in model.adata.obsm:
            raise KeyError(
                f"Missing '{obsm_key}' in model.adata.obsm. "
                "Run `model.annotate_adata()` (or `model.fit(...)`) first."
            )

        x = np.asarray(model.adata.obsm[obsm_key], dtype=float)
        if x.ndim != 2:
            raise ValueError(f"{obsm_key} must be a 2D array.")
        probs = x / np.clip(x.sum(axis=1, keepdims=True), eps, None)
        ent_raw = -np.sum(probs * np.log(np.clip(probs, eps, None)), axis=1)
        ent = ent_raw.copy()
        if normalize:
            n_factors = x.shape[1]
            if n_factors > 1:
                ent = ent / np.log(float(n_factors))
            else:
                ent = np.zeros_like(ent)

        col = f"{layer_name}_{key_suffix}"
        model.adata.obs[col] = ent
        eff_col = f"{layer_name}_{effective_suffix}"
        model.adata.obs[eff_col] = np.exp(ent_raw)
        created_cols.append(col)

    return created_cols


def compute_within_group_pairwise_dissimilarity(
    model: "scDEF",
    layer: Union[int, str],
    obs_key: str,
    metric: Literal["jsd", "euclidean", "cosine"] = "jsd",
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute within-group pairwise cell dissimilarity for one layer.

    Cells are represented by normalized factor memberships from
    ``model.adata.obsm[f"X_{layer_name}"]``. Pairwise distances are computed
    within each category of ``obs_key`` and summarized per group.

    Results are cached in ``model.adata.uns['within_group_pairwise_dissimilarity']``.
    """
    if obs_key not in model.adata.obs.columns:
        raise KeyError(f"obs_key '{obs_key}' not found in model.adata.obs.")

    if isinstance(layer, str):
        if layer not in model.layer_names:
            raise ValueError(f"Unknown layer '{layer}'. Valid: {model.layer_names}.")
        layer_idx = model.layer_names.index(layer)
    else:
        layer_idx = int(layer)
    if layer_idx < 0 or layer_idx >= model.n_layers:
        raise ValueError(f"layer must be in [0, {model.n_layers - 1}].")

    layer_name = model.layer_names[layer_idx]
    x_key = f"X_{layer_name}"
    if x_key not in model.adata.obsm:
        raise KeyError(
            f"Missing '{x_key}' in model.adata.obsm. "
            "Run `model.annotate_adata()` (or `model.fit(...)`) first."
        )

    x = np.asarray(model.adata.obsm[x_key], dtype=float)
    x = x / np.clip(x.sum(axis=1, keepdims=True), eps, None)
    groups = model.adata.obs[obs_key]
    group_values = list(pd.unique(groups))

    metric_name = "jensenshannon" if metric == "jsd" else metric
    summary_rows = []
    distributions: Dict[str, List[float]] = {}

    for group_value in group_values:
        mask = np.asarray(groups == group_value)
        group_x = x[mask]
        n_cells = int(group_x.shape[0])
        if n_cells < 2:
            dists = np.array([], dtype=float)
        else:
            dists = pdist(group_x, metric=metric_name).astype(float)
        distributions[str(group_value)] = dists.tolist()

        summary_rows.append(
            {
                "layer_idx": int(layer_idx),
                "layer": layer_name,
                "obs_key": obs_key,
                "obs_value": group_value,
                "metric": metric,
                "n_cells": n_cells,
                "n_pairs": int(dists.size),
                "mean_distance": float(np.mean(dists)) if dists.size > 0 else np.nan,
                "median_distance": float(np.median(dists))
                if dists.size > 0
                else np.nan,
                "std_distance": float(np.std(dists)) if dists.size > 0 else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "mean_distance", ascending=False, na_position="last"
    )
    cache = model.adata.uns.get("within_group_pairwise_dissimilarity", {})
    cache_key = f"{layer_name}::{obs_key}::{metric}"
    cache[cache_key] = {
        "fit_revision": int(getattr(model, "_fit_revision", 0)),
        "layer_idx": int(layer_idx),
        "layer": layer_name,
        "obs_key": obs_key,
        "metric": metric,
        "summary": summary_df.to_dict(orient="records"),
        "distributions": distributions,
    }
    model.adata.uns["within_group_pairwise_dissimilarity"] = cache
    return summary_df.reset_index(drop=True)


def get_confident_signatures(
    model: "scDEF",
    layer_idx: int = 0,
    confidence_threshold: float = 0.9,
    tau_quantile: float = 0.99,
    min_effect: Optional[float] = None,
    max_genes: Optional[int] = None,
    mc_samples_upper: int = 100,
    random_seed: int = 0,
    return_confidences: bool = False,
) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]]:
    """Get confidence-based signatures per factor using posterior mean/variance.

    For each factor independently, this computes a per-factor threshold
    ``tau = quantile(E[W_k,:], tau_quantile)`` and keeps genes that satisfy
    ``P(W_k,g > tau) >= confidence_threshold`` under a normal approximation
    using the posterior mean and variance of ``W``.

    For ``layer_idx > 0``, confidences are estimated with Monte Carlo sampling
    from the variational posterior via ``model.get_signature_sample``.

    Args:
        model: scDEF model instance
        layer_idx: layer index to use
        confidence_threshold: minimum posterior confidence to keep a gene
        tau_quantile: quantile of factor mean loadings used as threshold tau
        min_effect: optional minimum posterior mean loading ``E[W_k,g]``
        max_genes: optional maximum number of genes to keep per factor
        mc_samples_upper: number of Monte Carlo samples used for
            ``layer_idx > 0`` confidence estimation
        random_seed: random seed for Monte Carlo sampling in upper layers
        return_confidences: whether to also return per-gene confidence arrays

    Genes are ranked by a combined DE-style score that uses both confidence and
    posterior mean loading:
    ``score = E[W_k,g] * -log10(1 - confidence_k,g)``.

    Returns:
        Dictionary mapping factor names to confident gene lists. If
        ``return_confidences`` is True, also returns a dictionary mapping
        factor names to confidence arrays aligned with each gene list.
    """
    if layer_idx < 0 or layer_idx >= model.n_layers:
        raise ValueError(f"layer_idx must be in [0, {model.n_layers - 1}].")
    if not (0.0 < confidence_threshold < 1.0):
        raise ValueError("confidence_threshold must be in (0, 1).")
    if not (0.0 < tau_quantile < 1.0):
        raise ValueError("tau_quantile must be in (0, 1).")
    if mc_samples_upper <= 0:
        raise ValueError("mc_samples_upper must be > 0.")

    layer_name = model.layer_names[layer_idx]
    term_names = np.asarray(model.adata.var_names)
    signatures: Dict[str, List[str]] = {}
    signature_confidences: Dict[str, np.ndarray] = {}

    if layer_idx == 0:
        kept = np.asarray(model.factor_lists[layer_idx], dtype=int)
        term_means = np.asarray(model.pmeans[f"{layer_name}W"], dtype=float)[kept]
        term_vars = np.asarray(model.pvars[f"{layer_name}W"], dtype=float)[kept]
        term_vars = np.maximum(term_vars, 0.0)
        term_stds = np.sqrt(term_vars + 1e-12)

        for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
            mu = term_means[factor_idx]
            sigma = term_stds[factor_idx]
            tau = float(np.quantile(mu, tau_quantile))
            z = (mu - tau) / sigma
            confidences = norm.cdf(z)

            keep_mask = confidences >= confidence_threshold
            if min_effect is not None:
                keep_mask = keep_mask & (mu >= min_effect)
            keep_idx = np.where(keep_mask)[0]

            # Rank by a DE-style combined score of confidence and mean loading.
            if len(keep_idx) > 0:
                combined_scores = _confidence_mean_score(
                    confidences[keep_idx], mu[keep_idx]
                )
                order = np.argsort(combined_scores)[::-1]
                keep_idx = keep_idx[order]
            if max_genes is not None:
                keep_idx = keep_idx[: int(max_genes)]

            signatures[factor_name] = term_names[keep_idx].tolist()
            signature_confidences[factor_name] = confidences[keep_idx]
    else:
        from jax import random

        if hasattr(model, "logger"):
            model.logger.info(
                "Estimating confident signatures for layer %s with Monte Carlo "
                "(mc_samples_upper=%s). This may be slower than layer 0.",
                layer_idx,
                mc_samples_upper,
            )
        term_means = _get_layer_term_means(model, layer_idx)
        base_rng = random.PRNGKey(int(random_seed))
        n_genes = len(term_names)

        for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
            mu = term_means[factor_idx]
            tau = float(np.quantile(mu, tau_quantile))

            samples = []
            for s_idx in range(int(mc_samples_upper)):
                rng = random.fold_in(
                    base_rng, factor_idx * int(mc_samples_upper) + s_idx
                )
                _, sample_scores = model.get_signature_sample(
                    rng,
                    factor_idx=factor_idx,
                    layer_idx=layer_idx,
                    top_genes=n_genes,
                    return_scores=True,
                )
                samples.append(np.asarray(sample_scores, dtype=float))
            sample_arr = np.vstack(samples)
            confidences = np.mean(sample_arr > tau, axis=0)

            keep_mask = confidences >= confidence_threshold
            if min_effect is not None:
                keep_mask = keep_mask & (mu >= min_effect)
            keep_idx = np.where(keep_mask)[0]

            # Rank by a DE-style combined score of confidence and mean loading.
            if len(keep_idx) > 0:
                combined_scores = _confidence_mean_score(
                    confidences[keep_idx], mu[keep_idx]
                )
                order = np.argsort(combined_scores)[::-1]
                keep_idx = keep_idx[order]
            if max_genes is not None:
                keep_idx = keep_idx[: int(max_genes)]

            signatures[factor_name] = term_names[keep_idx].tolist()
            signature_confidences[factor_name] = confidences[keep_idx]

    if return_confidences:
        return signatures, signature_confidences
    return signatures


def _resolve_factor_obs_names(
    model: "scDEF", names: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Map user-supplied factor names to ``factor_obs`` index entries.

    User names are first matched directly against ``factor_obs.index``. If a
    name is not present, it is interpreted as an entry of the current
    ``model.factor_names[layer]`` and translated to the corresponding
    ``factor_obs`` row via ``original_factor_idx``. This makes it safe to pass
    names taken from the *current* (possibly filtered) model, even when
    ``factor_obs`` was populated before filtering.

    Returns:
        (resolved_names, unknown_names)
    """
    factor_obs = model.adata.uns["factor_obs"]
    has_meta = (
        "child_layer" in factor_obs.columns
        and "original_factor_idx" in factor_obs.columns
    )

    current_to_orig: dict = {}
    for layer_idx, layer_names in enumerate(model.factor_names):
        for slot, name in enumerate(layer_names):
            current_to_orig[name] = (
                layer_idx,
                int(model.factor_lists[layer_idx][slot]),
            )

    resolved: List[str] = []
    unknown: List[str] = []
    for name in names:
        if has_meta and name in current_to_orig:
            layer_idx, orig = current_to_orig[name]
            layer_name = model.layer_names[layer_idx]
            mask = (factor_obs["child_layer"] == layer_name) & (
                factor_obs["original_factor_idx"].astype(int) == orig
            )
            matches = factor_obs.index[mask].tolist()
            if matches:
                resolved.append(matches[0])
                continue
        if name in factor_obs.index:
            resolved.append(name)
        else:
            unknown.append(name)
    return resolved, unknown


def set_technical_factors(
    model: "scDEF",
    factors: Optional[Sequence[str]] = None,
    brd_min: Optional[float] = 1.0,
    ard_min: Optional[float] = 0.001,
    clarity_min: Optional[float] = 0.5,
    n_eff_parents_max: float = 1.5,
    local_l0_scores: bool = False,
    min_cells_lower: Optional[float] = 0.0,
) -> None:
    """Set the technical factors of the model.

    Technical factors must be layer 0 factors.

    Args:
        model: scDEF model instance
        factors: list of factor names to mark as technical. Names are resolved
            against the current ``model.factor_names`` (and translated to the
            corresponding ``factor_obs`` rows via ``original_factor_idx``), so
            it is safe to pass names from the model after ``filter_factors()``.
            When provided, criteria-based selection is skipped.
        brd_min: minimum BRD threshold for keeping biological layer-0 factors
            when ``factors`` is None.
        ard_min: minimum ARD fraction threshold for keeping biological layer-0
            factors when ``factors`` is None.
        clarity_min: minimum L0 clarity when not using lineage ``avg_n_eff_parents``.
        n_eff_parents_max: only for lineage diagnostics: ceiling on ``avg_n_eff_parents``
            (default ``1.5``; matches ``scd.pl.factor_diagnostics``).
        local_l0_scores: if True, biological factors are chosen using ``n_eff_parents``
            and ``n_eff_parents_max`` instead of lineage averages / ``clarity_min``.
        min_cells_lower: minimum cell-count criterion for keeping biological
            layer-0 factors when ``factors`` is None. Same semantics as
            ``scDEF.filter_factors(..., min_cells_lower=...)``.

    Notes:
        When ``factors`` is None, the candidate pool is restricted to the
        layer-0 factors currently kept in ``model.factor_lists[0]``. Already
        filtered-out factors are never re-introduced as technical.
    """
    if "factor_obs" not in model.adata.uns:
        factor_diagnostics(model)
    if "technical" not in model.adata.uns["factor_obs"].columns:
        model.adata.uns["factor_obs"]["technical"] = False
    model.adata.uns["factor_obs"]["technical"] = False

    factor_obs = model.adata.uns["factor_obs"]
    has_meta = (
        "child_layer" in factor_obs.columns
        and "original_factor_idx" in factor_obs.columns
    )

    technical_factors: List[str] = []
    if factors is not None:
        resolved, unknown = _resolve_factor_obs_names(model, factors)
        if len(unknown) > 0:
            raise ValueError(
                "Unknown factor name(s) in `factors`: " + ", ".join(map(str, unknown))
            )
        technical_factors = resolved
    else:
        bio_orig = set(
            int(i)
            for i in model.get_effective_factors(
                brd_min=brd_min,
                ard_min=ard_min,
                clarity_min=clarity_min,
                n_eff_parents_max=n_eff_parents_max,
                local_l0_scores=local_l0_scores,
                min_cells=min_cells_lower,
            )
        )
        kept_orig = set(int(o) for o in model.factor_lists[0])

        if has_meta:
            l0_mask = factor_obs["child_layer"] == model.layer_names[0]
            l0_rows = factor_obs.index[l0_mask].tolist()
            orig_arr = factor_obs.loc[l0_rows, "original_factor_idx"].astype(int)
            technical_factors = [
                name
                for name, o in zip(l0_rows, orig_arr)
                if int(o) in kept_orig and int(o) not in bio_orig
            ]
        else:
            kept_slots_bio = [
                slot
                for slot, orig in enumerate(model.factor_lists[0])
                if int(orig) in bio_orig
            ]
            keep_names = set(model.factor_names[0][slot] for slot in kept_slots_bio)
            l0_prefix = f"{model.layer_names[0]}_"
            l0_names = [
                name for name in factor_obs.index if str(name).startswith(l0_prefix)
            ]
            technical_factors = [name for name in l0_names if name not in keep_names]

    if len(technical_factors) > 0:
        model.adata.uns["factor_obs"].loc[technical_factors, "technical"] = True

    # Get complete hierarchy
    complete_hierarchy = get_hierarchy(model, simplified=False)
    # Traverse hierarchy. If all the children of a factor are technical, set the factor as technical.
    for factor, children in complete_hierarchy.items():
        if all(
            [
                model.adata.uns["factor_obs"].loc[child, "technical"]
                for child in children
            ]
        ):
            model.adata.uns["factor_obs"].loc[factor, "technical"] = True

    # Refresh annotations/probabilities so cell assignments and factor probabilities
    # are computed using biological factors only.
    model.annotate_adata()


def __build_consensus_signature(var_names, gene_scores_array, sizes_array):
    sizes_array = sizes_array / np.sum(sizes_array)
    avg_ranks = np.sum(sizes_array[:, None] * gene_scores_array, axis=0)
    idx_sorted = np.argsort(avg_ranks)[::-1]
    consensus = var_names[idx_sorted].tolist()
    consensus_scores = avg_ranks[idx_sorted]
    return consensus, consensus_scores


def get_technical_signature(
    model: "scDEF", top_genes: int = 10, return_scores: bool = False
) -> Union[List[str], Tuple[List[str], np.ndarray]]:
    hierarchy = model.adata.uns["technical_hierarchy"]
    gene_rankings, gene_scores = model.get_rankings(
        layer_idx=0,
        genes=True,
        return_scores=True,
    )

    # Reorder each gene_rankings and gene_scores by model.adata.var_names
    var_names = np.array(model.adata.var_names)
    n_factors = len(gene_scores)
    gene_scores_ordered = []
    for i in range(n_factors):
        ranking = np.array(gene_rankings[i])
        scores = np.array(gene_scores[i])
        # Map gene ranking to index in model.adata.var_names
        gene_order = np.argsort(
            [np.where(var_names == gene)[0][0] for gene in ranking]
        )  # noqa: F841
        reordered_idx = np.argsort(
            [np.where(ranking == g)[0][0] for g in var_names]
        )  # noqa: F841
        # Pad/truncate scores to fit var_names if necessary
        scores_full = np.full(len(var_names), np.nan)
        mask = np.in1d(var_names, ranking)
        scores_full[mask] = scores[
            [np.where(ranking == g)[0][0] for g in var_names[mask]]
        ]
        # Replace nans with 0 if needed, or keep as nan
        scores_full = np.nan_to_num(scores_full, nan=0)
        gene_scores_ordered.append(scores_full)
    gene_scores = np.array(
        [s / np.max(s) if np.max(s) > 0 else s for s in gene_scores_ordered]
    )

    relevances = model.get_relevances_dict()
    children = hierarchy["tech_top"]
    factors = [
        factor
        for i, factor in enumerate(range(len(gene_scores)))
        if model.factor_names[0][i] in children
    ]
    gene_scores = np.array([gene_scores[f] / np.max(gene_scores[f]) for f in factors])
    children_sizes = np.array([relevances[child] for child in children]).ravel()

    consensus_signature, consensus_scores = __build_consensus_signature(
        model.adata.var_names, gene_scores, children_sizes
    )
    if return_scores:
        return consensus_signature[:top_genes], consensus_scores[:top_genes]
    return consensus_signature[:top_genes]


def get_biological_signature(model: "scDEF", top_genes: int = 10) -> List[str]:
    # Get the top signature
    technical_factors = model.adata.uns["factor_obs"][
        model.adata.uns["factor_obs"]["technical"]
    ].index.tolist()
    top_layer_idx = model.n_layers - 1
    signatures_dict = get_stored_confident_signatures(
        model, layer_idx=top_layer_idx, max_genes=top_genes
    )
    for tf in technical_factors:
        signatures_dict.pop(tf, None)
    top_factor = f"{model.layer_names[top_layer_idx]}_0"
    signature = signatures_dict.get(top_factor, [])
    return signature


def gsea(
    model: "scDEF",
    libs: Sequence[str] = ("KEGG_2019_Human",),
    custom_gene_sets: Optional[Dict[str, Sequence[str]]] = None,
    organism: str = "Human",
    background_genes: Optional[Sequence[str]] = None,
    layers: Optional[Sequence[int]] = None,
    top_genes: Optional[int] = None,
    cutoff: float = 0.05,
    outdir: Optional[str] = None,
) -> pd.DataFrame:
    """Run Enrichr pathway enrichment for cached signatures across layers.

    This utility uses signatures from ``scd.tl.get_stored_confident_signatures``
    and does not rely on model-level ranking by raw ``W``.
    Online libraries in ``libs`` are fetched to local dicts and merged with
    ``custom_gene_sets`` so each factor is tested against one combined
    gene-set universe using a single ``gp.enrich`` call. By default, runs
    for all layers and stores per-layer results in ``adata.uns['factor_enrichments']``.
    """
    import gseapy as gp

    if layers is None:
        layers = list(range(model.n_layers))
    else:
        layers = [int(i) for i in layers]
    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= model.n_layers:
            raise ValueError(f"layer index {layer_idx} out of bounds.")

    use_online_libs = libs is not None and len(list(libs)) > 0
    use_custom_sets = custom_gene_sets is not None and len(custom_gene_sets) > 0
    if not use_online_libs and not use_custom_sets:
        raise ValueError(
            "Provide at least one online library in `libs` and/or `custom_gene_sets`."
        )

    # Build one merged gene-set dictionary (shared universe for all factors).
    all_sets: Dict[str, List[str]] = {}
    term_sources: Dict[str, List[str]] = {}
    if use_online_libs:
        for lib_name in libs:
            lib_sets = gp.get_library(name=lib_name, organism=organism)
            for term, genes in lib_sets.items():
                all_sets[term] = list(genes)
                term_sources.setdefault(term, [])
                if lib_name not in term_sources[term]:
                    term_sources[term].append(lib_name)
    if use_custom_sets:
        for term, genes in custom_gene_sets.items():
            all_sets[term] = list(genes)
            term_sources.setdefault(term, [])
            if "custom" not in term_sources[term]:
                term_sources[term].append("custom")

    if len(all_sets) == 0:
        raise ValueError("Combined gene-set dictionary is empty.")

    bg = list(model.adata.var_names) if background_genes is None else background_genes

    cache = {}
    all_results: List[pd.DataFrame] = []
    for layer_idx in layers:
        signatures = get_stored_confident_signatures(
            model,
            layer_idx=layer_idx,
            max_genes=top_genes,
        )
        layer_frames: List[pd.DataFrame] = []
        for factor_name, genes in signatures.items():
            if len(genes) == 0:
                continue
            enr = gp.enrich(
                gene_list=genes,
                gene_sets=all_sets,
                background=bg,
                outdir=outdir,
            )
            df = enr.results.copy()
            if len(df) == 0:
                continue

            cols_ci = {c.lower(): c for c in df.columns}
            term_col = cols_ci.get("term", None)
            if term_col is not None:
                source_labels = df[term_col].map(
                    lambda t: "|".join(term_sources.get(str(t), ["merged"]))
                )
                df["Gene_set"] = source_labels.values
                df["gene_set_source"] = source_labels.values

            padj_col = cols_ci.get("adjusted p-value", None)
            if padj_col is None:
                raise KeyError(
                    "Enrichr results missing 'Adjusted P-value' column; cannot filter significance."
                )
            df = df[df[padj_col] <= float(cutoff)].copy()
            if len(df) == 0:
                continue

            combined_col = None
            for candidate in ["combined score", "combined_score", "combinedscore"]:
                if candidate in cols_ci:
                    combined_col = cols_ci[candidate]
                    break
            if combined_col is None:
                raise KeyError(
                    "Enrichment results missing 'Combined Score' column; cannot sort by combined score."
                )
            df = df.sort_values(combined_col, ascending=False)
            df["factor"] = factor_name
            df["layer_idx"] = int(layer_idx)
            df["layer"] = model.layer_names[layer_idx]
            layer_frames.append(df)

        if len(layer_frames) > 0:
            layer_df = pd.concat(layer_frames, axis=0, ignore_index=True)
            all_results.append(layer_df)
            cache[str(int(layer_idx))] = {
                "fit_revision": int(getattr(model, "_fit_revision", 0)),
                "results": layer_df.to_dict(orient="records"),
            }

    model.adata.uns["factor_enrichments"] = cache
    if len(all_results) == 0:
        return pd.DataFrame()
    return pd.concat(all_results, axis=0, ignore_index=True)


def umap(
    model: "scDEF",
    layers: Optional[List[int]] = None,
    use_log: bool = False,
    metric: str = "euclidean",
) -> None:
    """Compute UMAP embeddings for each scDEF layer.

    The resulting embeddings are stored in
    ``model.adata.obsm[f"X_umap_{layer_name}"]`` for each layer.

    Args:
        model: scDEF model instance
        layers: which layers to compute UMAPs for. If None, all layers
            with more than one factor are used (in descending order).
        use_log: whether to use log-transformed cell-factor weights for
            the neighbor graph computation.
        metric: distance metric for neighbors computation.
    """
    if layers is None:
        layers = [
            i
            for i in range(model.n_layers - 1, -1, -1)
            if len(model.factor_lists[i]) > 1
        ]

    for layer in layers:
        layer_name = model.layer_names[layer]
        # Compute log representation
        model.adata.obsm[f"X_{layer_name}_log"] = np.log(
            model.adata.obsm[f"X_{layer_name}"]
        )
        if use_log:
            sc.pp.neighbors(model.adata, use_rep=f"X_{layer_name}_log")
        else:
            sc.pp.neighbors(
                model.adata,
                use_rep=f"X_{layer_name}",
                metric=metric,
            )
        sc.tl.umap(model.adata)
        # Store under a layer-specific key
        model.adata.obsm[f"X_umap_{layer_name}"] = model.adata.obsm["X_umap"].copy()


def multilayer_umap(
    model: "scDEF",
    layers: Optional[Sequence[int]] = None,
    weights: Optional[Sequence[float]] = None,
    normalize_per_layer: bool = False,
    use_log: bool = False,
    metric: str = "euclidean",
    key_added: str = "multilayer",
    eps: float = 1e-8,
    neighbors_kwargs: Optional[Dict[str, object]] = None,
    umap_kwargs: Optional[Dict[str, object]] = None,
) -> np.ndarray:
    """Compute a single UMAP from the concatenation of all scDEF layers.

    Each cell's representation is the concatenation of its soft factor
    assignments ``X_{layer_name}`` across the selected layers, producing
    a multi-resolution signature that encodes identity at every level
    of the hierarchy simultaneously.

    This tends to produce a **lineage-aware** embedding that smooths
    through differentiation trajectories: terminally differentiated
    cells form tight clusters by shared fine-layer factor, progenitor
    cells bridge siblings through their shared parent-layer mass, and
    stem-like cells with diffuse loadings sit in their own regime.
    Contrast with ``umap`` (per-layer), which only captures a single
    scale.

    By default it uses raw per-layer scores as stored in ``X_{layer}``.
    Set ``normalize_per_layer=True`` to convert each layer's row to
    proportions before optional log/weighting; this makes geometry
    reflect relative composition at each layer rather than absolute
    loading scale.

    Use ``weights`` to bias the embedding toward fine or coarse
    resolution.

    Single-factor layers (``K_k == 1``, typically the root) are skipped
    automatically — every cell has mass 1 there, so they add no
    discriminative signal.

    Args:
        model: fitted scDEF model.
        layers: layer indices to include. If None, uses all layers with
            ``K_k >= 2``, in ascending order.
        weights: optional per-layer multiplicative weights applied to
            each layer's sub-vector before concatenation. Must have the
            same length as ``layers``. Larger weight → that layer has
            more influence on the embedding geometry. Default: uniform.
        normalize_per_layer: if True, row-normalize each selected layer
            block so rows sum to 1 before optional ``log`` and
            weighting. Useful when you want distances to depend on
            relative factor composition rather than total per-layer
            loading magnitude.
        use_log: if True, replace each sub-vector by ``log(X + eps)``
            before applying weights/concatenation. Helpful when fine
            resolution is dominated by a single factor and small
            proportions get crushed by Euclidean distance.
        metric: distance metric for ``sc.pp.neighbors``.
        key_added: suffix used to store results. Writes
            ``adata.obsm[f"X_{key_added}"]`` (the concatenated
            representation) and ``adata.obsm[f"X_umap_{key_added}"]``
            (the UMAP embedding).
        eps: floor for ``log`` when ``use_log=True``.
        neighbors_kwargs: extra kwargs forwarded to ``sc.pp.neighbors``.
        umap_kwargs: extra kwargs forwarded to ``sc.tl.umap``.

    Returns:
        The concatenated representation of shape ``(n_cells, sum K_k)``.
    """
    if layers is None:
        layers = [i for i in range(model.n_layers) if len(model.factor_lists[i]) >= 2]
    else:
        layers = [int(i) for i in layers]
        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= model.n_layers:
                raise ValueError(f"layer index {layer_idx} out of bounds.")
        layers = [i for i in layers if len(model.factor_lists[i]) >= 2]

    if len(layers) == 0:
        raise ValueError(
            "No layers with K >= 2 to concatenate. "
            "Run `model.annotate_adata()` and check `factor_lists`."
        )

    if weights is None:
        weights_arr = np.ones(len(layers), dtype=float)
    else:
        weights_arr = np.asarray(list(weights), dtype=float)
        if weights_arr.shape != (len(layers),):
            raise ValueError(
                f"weights must have length {len(layers)} (one per selected layer)."
            )
        if np.any(weights_arr < 0):
            raise ValueError("weights must be non-negative.")

    parts: List[np.ndarray] = []
    for w, layer_idx in zip(weights_arr, layers):
        layer_name = model.layer_names[layer_idx]
        obsm_key = f"X_{layer_name}"
        if obsm_key not in model.adata.obsm:
            raise KeyError(
                f"Missing '{obsm_key}' in model.adata.obsm. "
                "Run `model.annotate_adata()` (or `model.fit(...)`) first."
            )
        x = np.asarray(model.adata.obsm[obsm_key], dtype=float)
        if normalize_per_layer:
            x = x / np.clip(np.sum(x, axis=1, keepdims=True), float(eps), None)
        if use_log:
            x = np.log(np.clip(x, float(eps), None))
        if w != 1.0:
            x = x * float(w)
        parts.append(x)

    concat = np.concatenate(parts, axis=1)
    rep_key = f"X_{key_added}"
    model.adata.obsm[rep_key] = concat

    nb_kwargs = dict(neighbors_kwargs or {})
    nb_kwargs.setdefault("use_rep", rep_key)
    nb_kwargs.setdefault("metric", metric)
    sc.pp.neighbors(model.adata, **nb_kwargs)
    sc.tl.umap(model.adata, **(umap_kwargs or {}))
    model.adata.obsm[f"X_umap_{key_added}"] = model.adata.obsm["X_umap"].copy()

    model.adata.uns[f"{key_added}_umap_config"] = {
        "layers": [int(i) for i in layers],
        "layer_names": [model.layer_names[i] for i in layers],
        "layer_sizes_filtered": [int(len(model.factor_lists[i])) for i in layers],
        "weights": weights_arr.tolist(),
        "normalize_per_layer": bool(normalize_per_layer),
        "use_log": bool(use_log),
        "metric": str(metric),
        "fit_revision": int(getattr(model, "_fit_revision", 0)),
    }

    return concat


from .trajectory import multilevel_paga  # noqa: E402,F401
