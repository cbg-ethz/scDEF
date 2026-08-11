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
from .batch import (
    get_factor_batch_gene_scale_affinity,
    TOP_BATCH_COL,
    TOP_SCORE_COL,
)
from typing import (
    Optional,
    Sequence,
    Dict,
    List,
    Tuple,
    Union,
    TYPE_CHECKING,
    Literal,
    Mapping,
    Any,
)

if TYPE_CHECKING:
    from scdef.models._scdef import scDEF


def _jaccard_gene_lists(reference: Sequence[str], sample: Sequence[str]) -> float:
    """Unweighted Jaccard similarity between two gene lists (as sets)."""
    ref = set(reference)
    samp = set(sample)
    if len(ref) == 0 and len(samp) == 0:
        return 1.0
    union = ref | samp
    if len(union) == 0:
        return 1.0
    return float(len(ref & samp) / len(union))


def _weighted_signature_jaccard(
    reference: Sequence[str],
    weights: Sequence[float],
    sample: Sequence[str],
) -> float:
    """Weighted Jaccard of a posterior draw top-k list vs the confident signature.

    Genes in the confident signature are weighted by normalized ``combined_scores``
    (higher-ranked signature genes count more). Genes in the draw but not in the
    reference add ``1 / k`` each to the union denominator (``k = |reference|``).
    """
    ref_genes = list(reference)
    k = len(ref_genes)
    if k == 0:
        return 1.0
    w = np.asarray(weights[:k], dtype=float)
    if w.size < k:
        w = np.pad(w, (0, k - w.size), constant_values=0.0)
    if np.all(w <= 0):
        w = np.ones(k, dtype=float)
    w = w / np.sum(w)

    draw = set(sample)
    ref_set = set(ref_genes)
    intersection_weight = float(
        np.sum([w[i] for i, gene in enumerate(ref_genes) if gene in draw])
    )
    n_draw_only = len(draw - ref_set)
    union_weight = 1.0 + float(n_draw_only) / float(k)
    return intersection_weight / union_weight


def _compute_signature_jaccard_confidences(
    model: "scDEF",
    layer_idx: int,
    signatures: Dict[str, List[str]],
    combined_scores: Dict[str, Sequence[float]],
    mc_samples: int,
    random_seed: int,
) -> Dict[str, float]:
    """Average weighted Jaccard of posterior top-k draws vs confident signatures.

    For each factor, ``k`` is the number of genes in that factor's confident
    signature. Each Monte Carlo draw yields a top-``k`` gene list from
    ``get_signature_sample``. Weighted Jaccard uses ``combined_scores`` so genes
    ranked higher in the confident signature contribute more to the score.
    """
    from jax import random

    base_rng = random.PRNGKey(int(random_seed))
    layer_offset = int(layer_idx) * 1_000_003
    out: Dict[str, float] = {}
    for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
        reference = list(signatures.get(factor_name, []))
        k = len(reference)
        if k == 0:
            out[factor_name] = float("nan")
            continue
        weights = list(combined_scores.get(factor_name, []))[:k]
        jaccs: List[float] = []
        for s_idx in range(int(mc_samples)):
            rng = random.fold_in(
                base_rng, layer_offset + factor_idx * int(mc_samples) + s_idx
            )
            draw_genes = model.get_signature_sample(
                rng,
                factor_idx=factor_idx,
                layer_idx=layer_idx,
                top_genes=k,
            )
            jaccs.append(_weighted_signature_jaccard(reference, weights, draw_genes))
        out[factor_name] = float(np.mean(jaccs))
    return out


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


def factor_obs_row_label(layer_name: str, original_factor_idx: int) -> str:
    """Stable ``factor_obs`` row label for a factor.

    Keyed by the *original* factor index, which never changes, rather than by
    the contiguous ``model.factor_names`` entry, which ``filter_factors``
    rewrites. This is the canonical key for everything stored per factor.
    """
    return f"{layer_name}_{int(original_factor_idx)}"


def original_key_of_factor(
    model: "scDEF", factor_name: str
) -> Optional[Tuple[int, int]]:
    """``current name -> (layer_idx, original_factor_idx)``, or None if unknown."""
    for layer_idx, names in enumerate(model.factor_names):
        for slot, name in enumerate(names):
            if str(name) == str(factor_name):
                return layer_idx, int(model.factor_lists[layer_idx][slot])
    return None


def current_name_of_original(
    model: "scDEF", layer_idx: int, original_factor_idx: int
) -> Optional[str]:
    """``(layer_idx, original_factor_idx) -> current name``, or None if dropped."""
    if layer_idx < 0 or layer_idx >= len(model.factor_names):
        return None
    factor_list = np.asarray(model.factor_lists[layer_idx], dtype=int)
    matches = np.where(factor_list == int(original_factor_idx))[0]
    if matches.size == 0:
        return None
    return str(model.factor_names[layer_idx][int(matches[0])])


def factor_obs_row_for_name(model: "scDEF", factor_name: str) -> Optional[str]:
    """Stable ``factor_obs`` row label for a *current* model factor name."""
    key = original_key_of_factor(model, factor_name)
    if key is None:
        return None
    layer_idx, orig = key
    return factor_obs_row_label(str(model.layer_names[layer_idx]), orig)


def _current_name_by_factor_obs_key(model: "scDEF") -> Dict[Tuple[str, int], str]:
    """Map ``(layer_name, original_factor_idx)`` to the current model factor name.

    ``factor_obs`` is keyed by the factor names in place when diagnostics ran,
    while ``model.factor_names`` is renamed contiguously by ``filter_factors``.
    This map is the bridge between the two.
    """
    out: Dict[Tuple[str, int], str] = {}
    factor_names = getattr(model, "factor_names", None)
    if factor_names is None:
        return out
    for layer_idx, names in enumerate(factor_names):
        layer_name = str(model.layer_names[layer_idx])
        factor_list = np.asarray(model.factor_lists[layer_idx], dtype=int)
        for slot, name in enumerate(names):
            if slot >= factor_list.size:
                break
            out[(layer_name, int(factor_list[slot]))] = str(name)
    return out


def _factor_obs_rows_to_current_names(model: "scDEF", rows: Sequence[str]) -> List[str]:
    """Translate ``factor_obs`` row names to current model factor names.

    Rows whose factor is no longer kept by the model (no current name) are
    dropped.
    """
    factor_obs = model.adata.uns.get("factor_obs")
    if factor_obs is None or len(rows) == 0:
        return []
    has_meta = (
        "child_layer" in factor_obs.columns
        and "original_factor_idx" in factor_obs.columns
    )
    if not has_meta:
        return [str(row) for row in rows]
    key_to_current = _current_name_by_factor_obs_key(model)
    out: List[str] = []
    for row in rows:
        if row not in factor_obs.index:
            continue
        key = (
            str(factor_obs.at[row, "child_layer"]),
            int(factor_obs.at[row, "original_factor_idx"]),
        )
        current = key_to_current.get(key)
        if current is not None:
            out.append(current)
    return out


def get_technical_factors(model: "scDEF") -> List[str]:
    """Current model names of the factors marked technical in ``factor_obs``.

    ``factor_obs`` rows are keyed by the factor names in place when diagnostics
    ran; after ``filter_factors`` the model renames factors contiguously. The
    returned names are always the *current* ``model.factor_names`` entries, so
    they can be compared directly against the live model. Technical factors that
    are no longer kept by the model are omitted.
    """
    factor_obs = model.adata.uns.get("factor_obs")
    if factor_obs is None or "technical" not in factor_obs.columns:
        return []
    rows = factor_obs.index[factor_obs["technical"].astype(bool)].tolist()
    return _factor_obs_rows_to_current_names(model, rows)


def get_batch_technical_factors(model: "scDEF") -> List[str]:
    """Current model names of the factors marked ``batch_technical``.

    The counterpart of :func:`set_batch_technical_factors`, and the batch-side
    analogue of :func:`get_technical_factors`. Names are translated to the
    *current* ``model.factor_names`` entries the same way, so they can be
    compared directly against the live model, and flagged factors the model no
    longer keeps are omitted.

    The two flags mean different things and are not interchangeable. A
    ``technical`` factor is a candidate for deletion by :func:`drop_technical`,
    and the flag propagates up the tree. A ``batch_technical`` factor is a
    layer-0 per-batch view of a program the corrected parent layer already
    represents: nothing is deleted and nothing propagates —
    :func:`factor_batch_correction` merges or drops it in the corrected
    representation and leaves the model itself untouched.

    Returns:
        Current layer-0 factor names flagged batch-technical, or ``[]`` if none
        are flagged or diagnostics have not been computed.
    """
    factor_obs = model.adata.uns.get("factor_obs")
    if factor_obs is None or "batch_technical" not in factor_obs.columns:
        return []
    rows = factor_obs.index[factor_obs["batch_technical"].astype(bool)].tolist()
    return _factor_obs_rows_to_current_names(model, rows)


def _resolve_signature_drop_factors(
    model: "scDEF", drop_factors: Optional[Sequence[str]]
) -> List[str]:
    """Drop list for hierarchical gene signatures (defaults to technical factors)."""
    if drop_factors is not None:
        return list(drop_factors)
    return get_technical_factors(model)


def _l0_keep_indices(model: "scDEF", drop_factors: Sequence[str]) -> np.ndarray:
    """Indices into ``model.factor_names[0]`` to keep when building L0→gene maps."""
    if not drop_factors:
        return np.arange(len(model.factor_names[0]), dtype=int)
    drop_set = set(drop_factors)
    return np.array(
        [i for i, name in enumerate(model.factor_names[0]) if name not in drop_set],
        dtype=int,
    )


def _filter_l0_factor_columns(
    model: "scDEF", matrix: np.ndarray, drop_factors: Sequence[str]
) -> np.ndarray:
    """Drop L0 factor columns used when mapping upper-layer loadings to genes."""
    keep = _l0_keep_indices(model, drop_factors)
    if keep.size == len(model.factor_names[0]):
        return matrix
    return np.asarray(matrix, dtype=float)[:, keep]


def _filter_l0_factor_rows(
    model: "scDEF", matrix: np.ndarray, drop_factors: Sequence[str]
) -> np.ndarray:
    """Drop L0 factor rows (``L0W``) when mapping upper-layer loadings to genes."""
    keep = _l0_keep_indices(model, drop_factors)
    if keep.size == len(model.factor_names[0]):
        return matrix
    return np.asarray(matrix, dtype=float)[keep, :]


def _get_layer_term_means(
    model: "scDEF",
    layer_idx: int,
    drop_factors: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return per-factor mean loadings aligned with ``adata.var_names``.

    For ``layer_idx > 0``, loadings are propagated through ``W`` down to L0 and then
    to genes. When ``drop_factors`` is omitted, factors marked ``technical`` in
    ``factor_obs`` are excluded from the L0→gene map (biological signatures only).
    """
    drop_factors = _resolve_signature_drop_factors(model, drop_factors)
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
    w0 = _filter_l0_factor_rows(model, w0, drop_factors)
    if term_scores.shape[1] == len(model.factor_names[0]):
        term_scores = _filter_l0_factor_columns(model, term_scores, drop_factors)
    return term_scores.dot(w0)


def _hierarchy_gene_scores_draw(
    model: "scDEF",
    rng: Any,
    max_layer_idx: Optional[int] = None,
    drop_factors: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, np.ndarray], Any]:
    """Draw one hierarchical posterior sample of gene scores for upper layers.

    Samples each ``W`` matrix once and propagates down to genes. Returns a
    mapping ``layer_idx -> (n_factors, n_genes)`` for layers ``1..max_layer_idx``.
    """
    from jax import random

    from scdef.utils.jax_utils import lognormal_sample

    drop_factors = _resolve_signature_drop_factors(model, drop_factors)
    max_layer = model.n_layers - 1 if max_layer_idx is None else int(max_layer_idx)
    if max_layer <= 0:
        return {}, rng

    l0_rows = np.asarray(model.factor_lists[0], dtype=int)
    w0_shape = model.global_params[2 + 0][0][l0_rows]
    w0_rate = np.exp(model.global_params[2 + 0][1][l0_rows])
    rng, sample_rng = random.split(rng)
    w0 = lognormal_sample(sample_rng, w0_shape, w0_rate)
    w0 = _filter_l0_factor_rows(model, w0, drop_factors)

    scores: Dict[int, np.ndarray] = {}
    path_to_l0: Optional[np.ndarray] = None

    for layer_idx in range(1, max_layer + 1):
        kept = np.asarray(model.factor_lists[layer_idx], dtype=int)
        kept_prev = np.asarray(model.factor_lists[layer_idx - 1], dtype=int)
        w_shape = model.global_params[2 + layer_idx][0][kept][:, kept_prev]
        w_rate = np.exp(model.global_params[2 + layer_idx][1][kept][:, kept_prev])
        rng, sample_rng = random.split(rng)
        w_layer = lognormal_sample(sample_rng, w_shape, w_rate)

        if path_to_l0 is None:
            path_to_l0 = w_layer
        else:
            path_to_l0 = w_layer.dot(path_to_l0)

        gene_scores = path_to_l0
        if gene_scores.shape[1] == len(model.factor_names[0]):
            gene_scores = _filter_l0_factor_columns(model, gene_scores, drop_factors)
        scores[layer_idx] = np.asarray(gene_scores.dot(w0), dtype=float)

    return scores, rng


def _collect_hierarchy_mc_scores(
    model: "scDEF",
    mc_samples: int,
    random_seed: int,
    max_layer_idx: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """Monte Carlo gene-score tensors shared across factors and upper layers.

    Returns ``layer_idx -> (mc_samples, n_factors, n_genes)`` for each upper
    layer included in the draw.
    """
    from jax import random

    base_rng = random.PRNGKey(int(random_seed))
    by_layer: Dict[int, List[np.ndarray]] = {}
    for s_idx in range(int(mc_samples)):
        rng = random.fold_in(base_rng, s_idx)
        scores, _ = _hierarchy_gene_scores_draw(model, rng, max_layer_idx=max_layer_idx)
        for layer_idx, mat in scores.items():
            by_layer.setdefault(layer_idx, []).append(mat)
    return {layer_idx: np.stack(mats, axis=0) for layer_idx, mats in by_layer.items()}


def _confident_signatures_from_mc_scores(
    model: "scDEF",
    layer_idx: int,
    mc_scores: np.ndarray,
    confidence_threshold: float,
    tau_quantile: float,
    min_effect: Optional[float],
    max_genes: Optional[int],
    return_confidences: bool,
) -> Union[Dict[str, List[str]], Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]]:
    """Build confident signatures from precomputed Monte Carlo gene scores."""
    term_names = np.asarray(model.adata.var_names)
    term_means = _get_layer_term_means(model, layer_idx)
    signatures: Dict[str, List[str]] = {}
    signature_confidences: Dict[str, np.ndarray] = {}

    for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
        mu = term_means[factor_idx]
        tau = float(np.quantile(mu, tau_quantile))
        sample_arr = np.asarray(mc_scores[:, factor_idx, :], dtype=float)
        confidences = np.mean(sample_arr > tau, axis=0)

        keep_mask = confidences >= confidence_threshold
        if min_effect is not None:
            keep_mask = keep_mask & (mu >= min_effect)
        keep_idx = np.where(keep_mask)[0]

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


def _signature_jaccard_from_mc_scores(
    model: "scDEF",
    layer_idx: int,
    signatures: Dict[str, List[str]],
    combined_scores: Dict[str, Sequence[float]],
    mc_scores: np.ndarray,
) -> Dict[str, float]:
    """Weighted signature Jaccard from the same Monte Carlo draws as confidences."""
    term_names = np.asarray(model.adata.var_names)
    out: Dict[str, float] = {}
    for factor_idx, factor_name in enumerate(model.factor_names[layer_idx]):
        reference = list(signatures.get(factor_name, []))
        k = len(reference)
        if k == 0:
            out[factor_name] = float("nan")
            continue
        weights = list(combined_scores.get(factor_name, []))[:k]
        jaccs: List[float] = []
        for s_idx in range(mc_scores.shape[0]):
            scores = np.asarray(mc_scores[s_idx, factor_idx, :], dtype=float)
            top_idx = np.argsort(scores)[::-1][:k]
            draw_genes = term_names[top_idx].tolist()
            jaccs.append(_weighted_signature_jaccard(reference, weights, draw_genes))
        out[factor_name] = float(np.mean(jaccs))
    return out


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
    return_signature_confidences: bool = False,
) -> Union[
    Dict[str, List[str]],
    Tuple[Dict[str, List[str]], Dict[str, np.ndarray]],
    Tuple[Dict[str, List[str]], Dict[str, np.ndarray], Dict[str, np.ndarray]],
    Tuple[Dict[str, List[str]], Dict[str, float]],
    Tuple[
        Dict[str, List[str]],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, float],
    ],
]:
    """Load precomputed confident signatures (and optional scores) from cache."""
    if layer_idx < 0 or layer_idx >= model.n_layers:
        raise ValueError(f"layer_idx must be in [0, {model.n_layers - 1}].")
    cache = _get_confident_signatures_cache(model)
    layer_data = cache["by_layer"][str(int(layer_idx))]

    # Stored keys are the stable original-index labels; translate them to the
    # current contiguous names so callers keep working with model.factor_names.
    # Entries whose factor is no longer kept are dropped rather than mislabelled.
    layer_name = str(model.layer_names[layer_idx])
    stable_to_current: Dict[str, str] = {}
    for slot, name in enumerate(model.factor_names[layer_idx]):
        orig = int(model.factor_lists[layer_idx][slot])
        stable_to_current[factor_obs_row_label(layer_name, orig)] = str(name)

    def _remap(d: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in d.items():
            current = stable_to_current.get(str(key))
            if current is None:
                # Either an already-current key (legacy cache) or a factor that
                # is no longer kept. Pass through only if it names a live factor.
                if str(key) in stable_to_current.values():
                    current = str(key)
                else:
                    continue
            out[current] = value
        return out

    signatures: Dict[str, List[str]] = {
        k: list(v) for k, v in _remap(layer_data["signatures"]).items()
    }
    confidences: Dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=float)
        for k, v in _remap(layer_data["confidences"]).items()
    }
    combined_scores: Dict[str, np.ndarray] = {
        k: np.asarray(v, dtype=float)
        for k, v in _remap(layer_data["combined_scores"]).items()
    }
    signature_confidences: Dict[str, float] = {
        k: float(v)
        for k, v in _remap(layer_data.get("signature_confidences", {})).items()
    }
    if max_genes is not None:
        kmax = int(max_genes)
        signatures = {k: v[:kmax] for k, v in signatures.items()}
        confidences = {k: v[:kmax] for k, v in confidences.items()}
        combined_scores = {k: v[:kmax] for k, v in combined_scores.items()}

    if return_signature_confidences and return_confidences and return_combined_scores:
        return signatures, confidences, combined_scores, signature_confidences
    if return_signature_confidences and return_confidences:
        return signatures, confidences, signature_confidences
    if return_signature_confidences and return_combined_scores:
        return signatures, combined_scores, signature_confidences
    if return_signature_confidences:
        return signatures, signature_confidences
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
    mc_samples: int = 100,
    random_seed: int = 0,
) -> Dict[str, List[str]]:
    """Precompute and cache confident signatures/scores for all layers.

    Stores signatures, per-gene confidences, combined scores, and per-factor
    weighted signature Jaccard confidences (posterior stability of each
    confident gene list, weighted by ``combined_scores``) in
    ``model.adata.uns['confident_signatures']`` for reuse by plotting.
    """
    cache: Dict[str, object] = {
        "fit_revision": int(getattr(model, "_fit_revision", 0)),
        "params": {
            "confidence_threshold": float(confidence_threshold),
            "tau_quantile": float(tau_quantile),
            "min_effect": None if min_effect is None else float(min_effect),
            "mc_samples": int(mc_samples),
            "random_seed": int(random_seed),
        },
        "by_layer": {},
    }
    term_names = np.asarray(model.adata.var_names)
    gene_to_idx = {g: i for i, g in enumerate(term_names)}
    signatures_flat: Dict[str, List[str]] = {}

    upper_mc_scores: Optional[Dict[int, np.ndarray]] = None
    if model.n_layers > 1:
        upper_mc_scores = _collect_hierarchy_mc_scores(
            model, mc_samples=mc_samples, random_seed=random_seed
        )

    for layer_idx in range(model.n_layers):
        if layer_idx == 0:
            sigs, confs = get_confident_signatures(
                model,
                layer_idx=layer_idx,
                confidence_threshold=confidence_threshold,
                tau_quantile=tau_quantile,
                min_effect=min_effect,
                max_genes=None,
                mc_samples=mc_samples,
                random_seed=random_seed,
                return_confidences=True,
            )
        else:
            layer_mc = upper_mc_scores[layer_idx]
            sigs, confs = _confident_signatures_from_mc_scores(
                model,
                layer_idx=layer_idx,
                mc_scores=layer_mc,
                confidence_threshold=confidence_threshold,
                tau_quantile=tau_quantile,
                min_effect=min_effect,
                max_genes=None,
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

        if layer_idx == 0:
            signature_jaccard = _compute_signature_jaccard_confidences(
                model,
                layer_idx=layer_idx,
                signatures=sigs,
                combined_scores=layer_combined_scores,
                mc_samples=mc_samples,
                random_seed=random_seed,
            )
        else:
            signature_jaccard = _signature_jaccard_from_mc_scores(
                model,
                layer_idx=layer_idx,
                signatures=sigs,
                combined_scores=layer_combined_scores,
                mc_scores=upper_mc_scores[layer_idx],
            )

        # Store under the STABLE original-index label, not the contiguous name,
        # so a later filter_factors rename cannot re-attach these genes to a
        # different factor. `get_stored_confident_signatures` maps back to the
        # current names when reading.
        layer_name = str(model.layer_names[layer_idx])

        def _stable(key: str) -> str:
            orig = original_key_of_factor(model, str(key))
            if orig is None:
                return str(key)
            return factor_obs_row_label(layer_name, orig[1])

        cache["by_layer"][str(int(layer_idx))] = {
            "layer_name": model.layer_names[layer_idx],
            "key_scheme": "original_factor_idx",
            "signatures": {_stable(k): list(v) for k, v in sigs.items()},
            "confidences": {
                _stable(k): np.asarray(v, dtype=float).tolist()
                for k, v in confs.items()
            },
            "combined_scores": {
                _stable(k): v for k, v in layer_combined_scores.items()
            },
            "signature_confidences": {
                _stable(k): v for k, v in signature_jaccard.items()
            },
        }

    model.adata.uns["confident_signatures"] = cache
    model.adata.uns["factor_signatures"] = signatures_flat


def _entropy_purity_from_batch_masses(
    masses: np.ndarray, n_batches: int
) -> Tuple[float, float]:
    """Shannon entropy and normalized purity from non-negative batch masses."""
    total = float(np.sum(masses))
    if total <= 0.0 or n_batches < 2:
        return np.nan, np.nan
    probs = masses[masses > 0.0] / total
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(n_batches))
    purity = 1.0 - entropy / max_entropy if max_entropy > 0 else np.nan
    return entropy, float(np.clip(purity, 0.0, 1.0))


def _get_layer_cell_probs(model: "scDEF", layer_idx: int) -> np.ndarray:
    """Per-cell factor probabilities for one layer (rows sum to 1)."""
    layer_name = model.layer_names[layer_idx]
    probs_key = f"X_{layer_name}_probs"
    if probs_key in model.adata.obsm:
        return np.asarray(model.adata.obsm[probs_key], dtype=float)
    scores_key = f"X_{layer_name}"
    if scores_key in model.adata.obsm:
        scores = np.asarray(model.adata.obsm[scores_key], dtype=float)
    else:
        z_key = f"{layer_name}z"
        if z_key not in model.pmeans:
            raise KeyError(
                f"Missing cell scores for layer {layer_name!r}. Run "
                "model.annotate_adata() or model.fit(annotate=True) first."
            )
        scores = np.asarray(model.pmeans[z_key], dtype=float)
        kept = np.asarray(model.factor_lists[layer_idx], dtype=int)
        if scores.shape[1] != len(kept):
            scores = scores[:, kept]
    den = np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)
    return scores / den


def _soft_factor_membership(
    model: "scDEF",
    layer_idx: int,
    original_factor_idx: int,
    layer_probs: np.ndarray,
) -> Optional[np.ndarray]:
    """Per-cell soft membership for one factor (length n_cells)."""
    layer_name = model.layer_names[layer_idx]
    factor_list = np.asarray(model.factor_lists[layer_idx], dtype=int)
    slot_matches = np.where(factor_list == original_factor_idx)[0]
    if slot_matches.size > 0:
        return layer_probs[:, int(slot_matches[0])]
    z_key = f"{layer_name}z"
    if z_key not in model.pmeans:
        return None
    z = np.asarray(model.pmeans[z_key], dtype=float)
    if original_factor_idx < 0 or original_factor_idx >= z.shape[1]:
        return None
    z_norm = z / (np.clip(z.sum(axis=1, keepdims=True), 1e-12, None))
    return z_norm[:, original_factor_idx]


def hard_assignment_name_indices(model: "scDEF", layer_idx: int) -> np.ndarray:
    """Per-cell winner index into ``factor_names[layer_idx]`` (kept factors only).

    Uses the same posterior-mean ``z`` argmax as :meth:`scDEF.annotate_adata` and
    ``make_graph(..., assignments=True)``.
    """
    layer_name = model.layer_names[layer_idx]
    keep = np.asarray(model.factor_lists[layer_idx], dtype=int)
    z = np.asarray(
        model.pmeans[f"{layer_name}z"][:, keep],
        dtype=float,
    )
    return np.argmax(z, axis=1)


def hard_assignment_factor_slots(model: "scDEF", layer_idx: int) -> np.ndarray:
    """Per-cell winning original factor slot at ``layer_idx`` (kept factors only)."""
    keep = np.asarray(model.factor_lists[layer_idx], dtype=int)
    return keep[hard_assignment_name_indices(model, layer_idx)]


def count_hard_assigned_cells(
    model: "scDEF", layer_idx: int, original_factor_idx: int
) -> int:
    """Number of cells whose hard assignment at ``layer_idx`` is ``original_factor_idx``."""
    winners = hard_assignment_factor_slots(model, layer_idx)
    return int(np.sum(winners == int(original_factor_idx)))


def lookup_factor_obs_n_cells(model: "scDEF", factor_name: str) -> Optional[int]:
    """Return ``factor_obs['n_cells']`` for ``factor_name`` when diagnostics exist."""
    factor_obs = model.adata.uns.get("factor_obs")
    if factor_obs is None or "n_cells" not in factor_obs.columns:
        return None
    # factor_obs is keyed by the stable original-index label, so a current
    # (contiguous) name must be resolved before lookup.
    row = factor_obs_row_for_name(model, factor_name)
    if row is None or row not in factor_obs.index:
        if factor_name in factor_obs.index:
            row = factor_name
        else:
            return None
    value = factor_obs.at[row, "n_cells"]
    if pd.isna(value):
        return None
    return int(value)


def _compute_l0_batch_split(
    model: "scDEF",
    factor_obs: pd.DataFrame,
    z_means_full: np.ndarray,
    offsets: np.ndarray,
    batch_idx: np.ndarray,
    n_batches: int,
    unique_batches: np.ndarray,
    min_batch_frac: float = 0.7,
) -> None:
    """Score how much each L0 factor looks like one half of a per-batch split.

    ``decompose_batch_effects`` re-learns L0 without a ``batch_key``, so a purely
    technical per-batch duplication shows up as two L0 factors that (a) sit under
    the same L1 parent, (b) are each dominated by a *different* batch, and (c)
    have nearly identical cell-score fingerprints over the *other* L0 factors —
    i.e. the same cells, split by batch rather than by biology.

    Everything here works on the per-cell factor scores ``z`` (``pmeans['L0z']``,
    the cell side of the factorization), *not* on the factor-gene loadings ``W``.
    A factor's "cell-score profile" is the mean ``log1p`` score its own cells
    place on every L0 factor.

    The score is a **weighted average over every opposite-batch candidate**, not
    a single best partner, so it is dense: every kept L0 factor gets a value and
    ``NaN`` never appears. For factor ``i`` and candidate ``j``:

    - ``corr_ij``: Pearson correlation of the two cell-score profiles over all
      kept L0 columns *except* ``{i, j}``, so the trivially anti-correlated
      self-scores cannot drive it.
    - ``s_ij = sum_k a_i[k] * a_j[k]`` where ``a_i`` is column ``i`` of ``L1W``
      normalized to a distribution — a **soft shared-parent** weight that keeps
      the hierarchy in play without a hard ``argmax`` parent assignment.
    - ``skew_j``: a ramp on the candidate's dominant-batch fraction, centered on
      ``min_batch_frac`` (``0.6 -> 0``, ``0.8 -> 1`` at the default ``0.7``).

    ``batch_split_corr[i] = sum_j w_ij * corr_ij / sum_j w_ij`` with
    ``w_ij = s_ij * skew_j``, over ``j != i`` whose dominant batch differs from
    ``i``'s. A degenerate case (single batch, or no opposite-batch contributor)
    scores ``0.0`` rather than ``NaN``.

    ``skew_j`` is what spares biology. A real per-batch split needs a batch-*pure*
    partner; two genuine subtypes that merely lean opposite ways by batch (CD4
    Memory ~52% control vs CD4 Naive ~52% stimulated) have batch-*balanced*
    partners, which the ramp weights to ~0. A balanced factor is the
    batch-corrected cell-type factor — the roll-up target, not a split half.

    Note this metric deliberately does **not** gate on factor ``i``'s own batch
    skew; that is a separate, plottable condition. Keeping it out here is what
    makes the column dense and lets a balanced factor's high correlation stay
    *visible*.

    .. warning::
        Read this column as a diagnostic, never as a ranking of "how technical"
        a factor is. On the paired pbmcs2b data it saturates in the 0.79-0.99
        band across many factor pairs, including pairs that share no genes, and
        it ranks the genuine Cytotoxic-T per-batch split *below* unrelated
        pairs. The verdict layer that used to be built on it has been removed
        for exactly this reason; use :func:`batch_structure_report`, which
        describes the same geometry without a verdict.

    Writes three layer-0 columns into ``factor_obs``: ``batch_split_corr``,
    plus the informational ``batch_split_partner`` (the single
    ``argmax_j w_ij * corr_ij`` contributor) and ``batch_split_batch`` (that
    partner's dominant batch).

    Args:
        model: scDEF model instance (needs ``pmeans`` from a fit).
        factor_obs: diagnostics frame to fill in place.
        z_means_full: per-cell variational ``z`` means for all layers, accepted
            for API symmetry with the surrounding batch diagnostics (the
            cell-score profiles use the posterior means in ``pmeans`` instead).
        offsets: per-layer column offsets into ``z_means_full``, likewise.
        batch_idx: per-cell batch index (``-1`` for missing).
        n_batches: number of distinct batches.
        unique_batches: batch values, ordered as in ``batch_idx``.
        min_batch_frac: center of the candidate batch-skew ramp.
    """
    for column, default in (
        ("batch_split_corr", np.nan),
        ("batch_split_partner", ""),
        ("batch_split_batch", ""),
    ):
        if column not in factor_obs.columns:
            factor_obs[column] = default

    if int(model.n_layers) < 2:
        return
    pmeans = getattr(model, "pmeans", None)
    if not isinstance(pmeans, dict):
        return
    l0_name = model.layer_names[0]
    w1_key = f"{model.layer_names[1]}W"
    z0_key = f"{l0_name}z"
    if w1_key not in pmeans or z0_key not in pmeans:
        return

    # W at layer 1 is (n_L1, n_L0); column i is factor i's parent affinity.
    w1 = np.asarray(pmeans[w1_key], dtype=float)
    # Posterior mean per-cell factor scores (non-negative). Note: `local_params`
    # holds the unconstrained log-space parameters, for which log1p(clip(., 0))
    # would zero out everything -- the posterior means are what we need here.
    z0 = np.asarray(pmeans[z0_key], dtype=float)
    if w1.ndim != 2 or z0.ndim != 2 or w1.shape[1] != z0.shape[1]:
        return

    # Same cell partition as n_cells / the batch metrics: argmax over the kept
    # factors only, so a dropped factor never claims cells.
    kept_l0 = np.asarray(model.factor_lists[0], dtype=int)
    if kept_l0.size == 0:
        return
    winner = kept_l0[np.argmax(z0[:, kept_l0], axis=1)]
    logz = np.log1p(np.clip(z0, 0.0, None))

    if "child_layer" in factor_obs.columns:
        l0_rows = factor_obs.index[factor_obs["child_layer"] == l0_name]
    else:
        l0_rows = factor_obs.index
    if "original_factor_idx" not in factor_obs.columns:
        return

    kept_orig = set(int(o) for o in model.factor_lists[0])
    orig_to_row: Dict[int, str] = {}
    for row_name in l0_rows:
        orig = int(factor_obs.at[row_name, "original_factor_idx"])
        if orig in kept_orig and 0 <= orig < z0.shape[1]:
            orig_to_row[orig] = row_name
    if len(orig_to_row) < 2:
        return

    kept_cols = np.asarray(sorted(orig_to_row.keys()), dtype=int)

    # Soft parent membership: normalize each L0 column of L1W to a distribution.
    parent_affinity = np.clip(w1[:, kept_cols], 0.0, None)
    den = np.clip(parent_affinity.sum(axis=0, keepdims=True), 1e-12, None)
    parent_affinity = parent_affinity / den

    dom_batch: Dict[int, int] = {}
    frac_dom: Dict[int, float] = {}
    profile: Dict[int, np.ndarray] = {}
    for orig in kept_cols:
        cells = np.where(winner == int(orig))[0]
        if cells.size == 0:
            continue
        cell_batches = batch_idx[cells]
        cell_batches = cell_batches[cell_batches >= 0]
        if cell_batches.size == 0:
            continue
        counts = np.bincount(cell_batches, minlength=n_batches).astype(float)
        total = float(np.sum(counts))
        if total <= 0.0:
            continue
        dom_batch[int(orig)] = int(np.argmax(counts))
        frac_dom[int(orig)] = float(np.max(counts) / total)
        profile[int(orig)] = logz[cells].mean(axis=0)

    col_pos = {int(c): pos for pos, c in enumerate(kept_cols)}
    ramp_lo = float(min_batch_frac) - 0.1
    skew = {
        j: float(np.clip((frac_dom[j] - ramp_lo) / 0.2, 0.0, 1.0)) for j in frac_dom
    }

    for i in kept_cols:
        i = int(i)
        row_name = orig_to_row[i]
        if i not in profile:
            # No cells of its own: nothing to compare, but keep the column dense.
            factor_obs.at[row_name, "batch_split_corr"] = 0.0
            continue

        a_i = parent_affinity[:, col_pos[i]]
        num = 0.0
        den_w = 0.0
        best_contrib = -np.inf
        best_partner: Optional[int] = None
        for j in profile:
            if j == i or dom_batch[j] == dom_batch[i]:
                continue
            w = float(np.dot(a_i, parent_affinity[:, col_pos[j]])) * skew[j]
            if w <= 0.0:
                continue
            mask = (kept_cols != i) & (kept_cols != j)
            other = kept_cols[mask]
            if other.size < 2:
                continue
            u = profile[i][other]
            v = profile[j][other]
            if not (np.std(u) > 0 and np.std(v) > 0):
                continue
            corr = float(np.corrcoef(u, v)[0, 1])
            if not np.isfinite(corr):
                continue
            num += w * corr
            den_w += w
            if w * corr > best_contrib:
                best_contrib = w * corr
                best_partner = j

        score = 0.0 if den_w <= 0.0 else num / den_w
        factor_obs.at[row_name, "batch_split_corr"] = round(float(score), 4)
        if best_partner is not None:
            factor_obs.at[row_name, "batch_split_partner"] = str(
                orig_to_row[best_partner]
            )
            factor_obs.at[row_name, "batch_split_batch"] = str(
                unique_batches[dom_batch[best_partner]]
            )


def factor_diagnostics(
    model: "scDEF",
    recompute: bool = False,
    batch_key: Optional[str] = None,
    sensible_top_n_eff_parents_max: float = 1.5,
    sensible_top_min_best_parent_prob: Optional[float] = None,
    sensible_top_min_clear_children: int = 2,
    sensible_top_ignore_root: bool = True,
    sensible_top_use_filtered: bool = True,
    confidence_threshold: float = 0.9,
    tau_quantile: float = 0.99,
    min_effect: Optional[float] = None,
    mc_samples: int = 100,
    random_seed: int = 0,
    batch_split_min_batch_frac: float = 0.7,
    gene_scale_reference: Optional[Any] = None,
) -> None:
    """Compute/store factor diagnostics in ``model.adata.uns['factor_obs']``.

    Populates per-factor hierarchy scores plus ``ARD``, ``BRD``, ``n_cells``
    (hard argmax posterior ``z`` assignments among kept factors — the same rule
    as ``annotate_adata`` / ``make_graph(..., assignments=True)``), and optional
    batch metrics when ``batch_key`` is set.

    Also runs :func:`set_confident_signatures` so plotting helpers
    (``make_graph``, ``pl.factor_diagnostics(color='signature_confidence')``,
    etc.) can use cached signatures without a separate call.

    Args:
        model: scDEF model instance
        recompute: if True, force recomputation of the cached fixed upper-layer
            factor subset used for clarity scores, even if the fit revision
            did not change.
        batch_key: optional key in ``model.adata.obs`` used to compute
            per-factor batch metrics. ``batch_purity`` uses hard winner cells
            (argmax variational ``z``). ``batch_purity_soft`` uses the batch
            distribution of per-cell memberships from ``X_<layer>_probs``
            (or row-normalized ``X_<layer>`` / posterior ``z`` if probs are
            missing). Both are ``1 - entropy / log(n_batches)``. Also writes
            ``dom_batch`` / ``frac_dom_batch`` (the factor's dominant batch and
            its share of the factor's cells) and, for layer 0, the
            ``batch_split_*`` columns from :func:`_compute_l0_batch_split`.
            All are plottable via ``scdef.pl.factor_diagnostics``.
        sensible_top_n_eff_parents_max: threshold used to classify
            sensible-top factors on the hierarchy walk.
        sensible_top_min_best_parent_prob: optional best-parent probability
            threshold used alongside ``n_eff_parents_max`` when deciding
            whether a child merge is clear.
        sensible_top_min_clear_children: parent escape-hatch count; a parent
            is accepted if it owns at least this many clear best-parent
            children, even when weighted ambiguity is high.
        sensible_top_ignore_root: whether to ignore a width-1 final root layer
            when classifying sensible-top factors.
        sensible_top_use_filtered: whether to use ``model.factor_lists`` /
            ``model.factor_names`` for hierarchy transitions.
        confidence_threshold: passed to :func:`set_confident_signatures`.
        tau_quantile: passed to :func:`set_confident_signatures`.
        min_effect: passed to :func:`set_confident_signatures`.
        mc_samples: passed to :func:`set_confident_signatures`.
        random_seed: passed to :func:`set_confident_signatures`.
        batch_split_min_batch_frac: center of the batch-skew ramp that weights
            candidate partners in :func:`_compute_l0_batch_split` (default
            ``0.7``: a candidate at ``0.6`` contributes nothing, one at ``0.8``
            contributes fully). Only used when ``batch_key`` is set.
        gene_scale_reference: optional gene-side batch diagnostic, off by
            default. Pass the REFERENCE fit (the model made with ``batch_key``
            and per-batch ``gene_scale``) — or anything else
            :func:`~scdef.tools.batch.get_factor_batch_gene_scale_affinity`
            accepts — to add two layer-0 columns: ``gene_scale_affinity``, the
            Spearman correlation between the factor's gene loadings and the
            per-batch ``gene_scale`` log-ratio of the batch it matches best,
            and ``gene_scale_affinity_batch``, that batch. High values say the
            factor is built from the genes the reference fit's per-batch term
            absorbed; they do **not** say the factor is technical (on a
            condition-like batch key that programme is the biology of the
            experiment). See that function's docstring.
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
    # Re-key by the stable (child_layer, original_factor_idx) label so the frame
    # survives the renaming that filter_factors performs.
    per_factor = res["per_factor"]
    if (
        "child_layer" in per_factor.columns
        and "original_factor_idx" in per_factor.columns
    ):
        per_factor = per_factor.copy()
        stable_index = [
            factor_obs_row_label(str(layer), int(orig))
            for layer, orig in zip(
                per_factor["child_layer"], per_factor["original_factor_idx"]
            )
        ]
        per_factor.index = pd.Index(stable_index, name=per_factor.index.name)
        if "child_factor" in per_factor.columns:
            per_factor["child_factor"] = stable_index
    model.adata.uns["factor_obs"] = per_factor
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
    factor_obs["global"] = False
    factor_obs["batch_entropy"] = np.nan
    factor_obs["batch_purity"] = np.nan
    factor_obs["batch_purity_soft"] = np.nan
    factor_obs["dom_batch"] = ""
    factor_obs["frac_dom_batch"] = np.nan
    factor_obs["batch_split_corr"] = np.nan
    factor_obs["batch_split_partner"] = ""
    factor_obs["batch_split_batch"] = ""
    factor_obs["n_cells"] = 0

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

        factor_obs.at[row.name, "n_cells"] = count_hard_assigned_cells(
            model, layer_idx, original_factor_idx
        )

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
            layer_probs_cache: Dict[int, np.ndarray] = {}

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

                # Use the SAME cell partition as `n_cells`: argmax over the
                # factors the model currently keeps. Scoring against the full
                # layer would attribute cells to dropped factors and would give
                # kept factors a different denominator than their own n_cells.
                kept_layer = np.asarray(model.factor_lists[layer_idx], dtype=int)
                if original_factor_idx not in kept_layer:
                    continue  # dropped factor: leave its batch metrics as NaN
                start = int(offsets[layer_idx])
                end = int(offsets[layer_idx + 1])
                layer_scores = z_means_full[:, start:end][:, kept_layer]
                winner = kept_layer[np.argmax(layer_scores, axis=1)]
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
                entropy, purity = _entropy_purity_from_batch_masses(counts, n_batches)
                factor_obs.at[row.name, "batch_entropy"] = entropy
                factor_obs.at[row.name, "batch_purity"] = purity

                total = float(np.sum(counts))
                if total > 0.0:
                    dominant = int(np.argmax(counts))
                    factor_obs.at[row.name, "dom_batch"] = str(unique_batches[dominant])
                    factor_obs.at[row.name, "frac_dom_batch"] = float(
                        counts[dominant] / total
                    )

                if layer_idx not in layer_probs_cache:
                    try:
                        layer_probs_cache[layer_idx] = _get_layer_cell_probs(
                            model, layer_idx
                        )
                    except KeyError:
                        layer_probs_cache[layer_idx] = None
                layer_probs = layer_probs_cache[layer_idx]
                if layer_probs is not None:
                    membership = _soft_factor_membership(
                        model,
                        layer_idx,
                        original_factor_idx,
                        layer_probs,
                    )
                    if membership is not None:
                        valid = batch_idx >= 0
                        soft_masses = np.bincount(
                            batch_idx[valid],
                            weights=membership[valid],
                            minlength=n_batches,
                        ).astype(float)
                        _, purity_soft = _entropy_purity_from_batch_masses(
                            soft_masses, n_batches
                        )
                        factor_obs.at[row.name, "batch_purity_soft"] = purity_soft

            _compute_l0_batch_split(
                model,
                factor_obs,
                z_means_full,
                offsets,
                batch_idx,
                n_batches,
                unique_batches,
                min_batch_frac=batch_split_min_batch_frac,
            )

        if "factor_diagnostics" not in model.adata.uns:
            model.adata.uns["factor_diagnostics"] = {}
        model.adata.uns["factor_diagnostics"]["batch_key"] = str(batch_key)
        model.adata.uns["factor_diagnostics"]["n_batches"] = int(n_batches)
        model.adata.uns["factor_diagnostics"]["batch_values"] = [
            str(b) for b in unique_batches
        ]

    from scdef.tools.hierarchy import _annotate_sensible_top_factors

    factor_obs = _annotate_sensible_top_factors(
        model,
        factor_obs,
        n_eff_parents_max=sensible_top_n_eff_parents_max,
        min_best_parent_prob=sensible_top_min_best_parent_prob,
        min_clear_children=sensible_top_min_clear_children,
        ignore_root=sensible_top_ignore_root,
        use_filtered=sensible_top_use_filtered,
    )
    model.adata.uns["factor_obs"] = factor_obs
    if "factor_diagnostics" not in model.adata.uns:
        model.adata.uns["factor_diagnostics"] = {}
    model.adata.uns["factor_diagnostics"]["sensible_top"] = {
        "n_eff_parents_max": float(sensible_top_n_eff_parents_max),
        "min_best_parent_prob": sensible_top_min_best_parent_prob,
        "min_clear_children": int(sensible_top_min_clear_children),
        "ignore_root": bool(sensible_top_ignore_root),
        "use_filtered": bool(sensible_top_use_filtered),
    }

    if gene_scale_reference is not None:
        from scdef.tools.batch import get_factor_batch_gene_scale_affinity

        affinity = get_factor_batch_gene_scale_affinity(model, gene_scale_reference)
        factor_obs["gene_scale_affinity"] = np.nan
        factor_obs["gene_scale_affinity_batch"] = ""
        l0_layer_name = model.layer_names[0]
        for original_factor_idx, (_, aff_row) in zip(
            np.asarray(model.factor_lists[0], dtype=int), affinity.iterrows()
        ):
            label = factor_obs_row_label(l0_layer_name, int(original_factor_idx))
            if label in factor_obs.index:
                factor_obs.at[label, "gene_scale_affinity"] = float(
                    aff_row["top_score"]
                )
                factor_obs.at[label, "gene_scale_affinity_batch"] = str(
                    aff_row["top_batch"]
                )
        model.adata.uns["factor_obs"] = factor_obs

    # Cache a complete snapshot of factor_obs keyed by (child_layer,
    # original_factor_idx). This is the source of truth for filtering
    # decisions (e.g. get_effective_factors / filter_factors), so that
    # re-filtering with looser thresholds still sees diagnostics for
    # factors previously dropped from the live factor_obs view.
    model.adata.uns["factor_obs_full"] = factor_obs.copy()

    set_confident_signatures(
        model,
        confidence_threshold=confidence_threshold,
        tau_quantile=tau_quantile,
        min_effect=min_effect,
        mc_samples=mc_samples,
        random_seed=random_seed,
    )


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
    mode: Literal["f1", "fracs", "weights", "prob", "soft_prec", "score"] = "fracs",
    ascending: bool = False,
    recompute: bool = False,
) -> pd.DataFrame:
    """Return per-obs-value factor rankings by observation association score.

    This reads cached matrices from ``model.adata.uns['obs_scores']`` (written by
    ``scd.pl.obs_scores``). If cache is missing/stale for the requested key/model,
    it is recomputed on demand for the requested ``obs_key`` and ``mode``.
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
    mode_cache = cache_root.get(mode, {})
    need_recompute = recompute
    if int(mode_cache.get("fit_revision", -1)) != fit_rev:
        need_recompute = True
    if "obs_keys" not in mode_cache or obs_key not in mode_cache["obs_keys"]:
        need_recompute = True

    if need_recompute:
        if mode == "f1":
            score_func = data_utils.get_assignment_scores
        elif mode == "fracs":
            score_func = data_utils.get_assignment_fracs
        elif mode == "weights":
            score_func = data_utils.get_weight_scores
        elif mode == "prob":
            score_func = data_utils.get_prob_scores
        elif mode == "soft_prec":
            score_func = data_utils.get_soft_prec_scores
        elif mode == "score":
            score_func = data_utils.get_score_means
        else:
            raise ValueError(
                "mode must be one of ['f1', 'fracs', 'weights', 'prob', 'soft_prec', 'score']."
            )

        obs_mats, obs_clusters, obs_vals_dict = data_utils.prepare_obs_factor_scores(
            model,
            [obs_key],
            score_func,
        )
        data_utils.cache_obs_factor_scores(
            model=model,
            obs_keys=[obs_key],
            mode=mode,
            obs_mats=obs_mats,
            obs_clusters=obs_clusters,
            obs_vals_dict=obs_vals_dict,
        )
        cache_root = model.adata.uns.get("obs_scores", {})
        mode_cache = cache_root.get(mode, {})

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
                    "mode": mode,
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
    mode: Literal["f1", "fracs", "weights", "prob", "soft_prec", "score"] = "fracs",
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
        mode=mode,
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
    spec_df["mode"] = mode
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
    exclude_technical: bool = False,
    exclude_batch_technical: bool = False,
    batch_technical_top_layer: Optional[int] = None,
) -> None:
    """Pick the finest scDEF layer at which each cell is confidently assigned.

    For each cell ``c`` and layer ``k`` (restricted to filtered factors
    ``model.factor_lists[k]``), draws ``n_samples`` reparameterized samples
    ``z^(s) ~ q(z_{c,k})`` from the log-normal variational posterior and
    normalizes each sample: ``ẑ^(s) = z^(s) / sum(z^(s))``.
    ``exclude_technical`` / ``exclude_batch_technical`` narrow that per-layer
    candidate set further, and all scores below are then computed among the
    remaining factors only.

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
      slot indices into the *effective* candidate list of each layer — the
      filtered factor list minus any factors excluded by
      ``exclude_technical`` / ``exclude_batch_technical`` (``-1`` if the layer
      has no candidates left).
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
        exclude_technical: if True, factors marked ``technical`` in
            ``factor_obs`` (see :func:`set_technical_factors`) are removed from
            the candidate set at **every** layer, so no cell is ever assigned to
            an ambient/stress program. Confidence is then computed among the
            remaining factors only.
        exclude_batch_technical: if True, factors marked ``batch_technical``
            (see :func:`set_batch_technical_factors`) are removed from the
            candidate set at every layer **below** ``batch_technical_top_layer``,
            and any cell whose layer-0 winner is one of them is rolled up to
            that layer. Per-batch splits only exist below the roll-up layer, so
            the flag is deliberately not applied at or above it.
        batch_technical_top_layer: layer index the batch-technical roll-up
            targets. Defaults to ``adata.uns['batch_technical_top_layer']``,
            recorded by :meth:`scDEF.decompose_batch_effects`, else ``1``.

    Example:
        >>> # After flagging per-batch splits, their cells attach to the
        >>> # batch-corrected L1 parent instead of overshooting to the root.
        >>> scdef.tl.set_batch_technical_factors(model, splits)
        >>> scdef.tl.assign_confident(model, exclude_batch_technical=True)
        >>> model.adata.obs["confident_best_layer"].value_counts()
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

    # Effective per-layer candidate sets. Technical factors are dropped
    # everywhere; batch-technical factors only below the roll-up layer, which
    # still carries the batch-corrected signal those cells belong to.
    technical_names: set = set()
    if exclude_technical:
        technical_names = set(get_technical_factors(model))

    bt_names: set = set()
    if exclude_batch_technical:
        bt_names = set(get_batch_technical_factors(model))

    if batch_technical_top_layer is None:
        bt_top_layer = int(model.adata.uns.get("batch_technical_top_layer", 1))
    else:
        bt_top_layer = int(batch_technical_top_layer)
    bt_top_layer = int(np.clip(bt_top_layer, 0, max(n_layers - 1, 0)))

    eff_lists: List[np.ndarray] = []
    eff_names: List[List[str]] = []
    for i in range(n_layers):
        layer_factor_list = np.asarray(model.factor_lists[i], dtype=int)
        keep_slots = [
            slot
            for slot, name in enumerate(model.factor_names[i])
            if name not in technical_names
            and not (name in bt_names and i < bt_top_layer)
        ]
        eff_lists.append(layer_factor_list[np.asarray(keep_slots, dtype=int)])
        eff_names.append([str(model.factor_names[i][slot]) for slot in keep_slots])

    for layer_idx in range(n_layers):
        start = int(sum(layer_sizes[:layer_idx]))
        kept = np.asarray(eff_lists[layer_idx], dtype=int)
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
            names_arr = np.asarray(eff_names[layer_idx], dtype=object)
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
        names_arr = np.asarray(eff_names[layer_idx], dtype=object)
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
        [len(eff_lists[i]) >= 2 for i in range(n_layers)],
        dtype=bool,
    )
    has_factors = np.asarray(
        [len(eff_lists[i]) > 0 for i in range(n_layers)],
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

    # Roll-up cap for batch-technical splits. A cell owned by a per-batch split
    # at L0 belongs to that split's batch-corrected parent, so pin it there
    # deterministically. Without this, dropping the split halves from the
    # candidate set leaves those cells ambiguous among the remaining L0 factors
    # and the finest-clears-tau rule overshoots all the way to the root.
    if len(bt_names) > 0 and len(eff_lists[bt_top_layer]) > 0:
        l0_name = layer_names[0]
        l0_scores_key = f"X_{l0_name}"
        if l0_scores_key in model.adata.obsm:
            l0_scores = np.asarray(model.adata.obsm[l0_scores_key], dtype=float)
        else:
            l0_scores = np.asarray(model.pmeans[f"{l0_name}z"], dtype=float)
            l0_kept = np.asarray(model.factor_lists[0], dtype=int)
            if l0_scores.shape[1] != l0_kept.size:
                l0_scores = l0_scores[:, l0_kept]
        # argmax over ALL kept L0 factors, including the flagged splits.
        l0_all_names = np.asarray(model.factor_names[0], dtype=object)
        if l0_scores.shape[1] == l0_all_names.size:
            l0_winner_name = l0_all_names[np.argmax(l0_scores, axis=1)]
            rolled = np.isin(l0_winner_name, np.asarray(sorted(bt_names), dtype=object))
            if np.any(rolled):
                idxs = np.where(rolled)[0]
                best_layer_idx[idxs] = bt_top_layer
                best_confidence[idxs] = confidence_mat[idxs, bt_top_layer]
                best_effect_size[idxs] = effect_size_mat[idxs, bt_top_layer]
                best_posterior_sd[idxs] = posterior_sd_mat[idxs, bt_top_layer]
                best_factor_slot[idxs] = argmax_mat[idxs, bt_top_layer]
                best_layer_name[idxs] = layer_names[bt_top_layer]
                best_label[idxs] = label_mat[idxs, bt_top_layer]

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
        "layer_sizes_effective": [int(len(eff_lists[i])) for i in range(n_layers)],
        "exclude_technical": bool(exclude_technical),
        "exclude_batch_technical": bool(exclude_batch_technical),
        "batch_technical_top_layer": int(bt_top_layer),
        "n_batch_technical_factors": int(len(bt_names)),
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
    mc_samples: int = 100,
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
        mc_samples: number of Monte Carlo samples used for
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
    if mc_samples <= 0:
        raise ValueError("mc_samples must be > 0.")

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
        if hasattr(model, "logger"):
            model.logger.info(
                "Estimating confident signatures for layer %s with Monte Carlo "
                "(mc_samples=%s). This may be slower than layer 0.",
                layer_idx,
                mc_samples,
            )
        mc_by_layer = _collect_hierarchy_mc_scores(
            model,
            mc_samples=mc_samples,
            random_seed=random_seed,
            max_layer_idx=layer_idx,
        )
        return _confident_signatures_from_mc_scores(
            model,
            layer_idx=layer_idx,
            mc_scores=mc_by_layer[layer_idx],
            confidence_threshold=confidence_threshold,
            tau_quantile=tau_quantile,
            min_effect=min_effect,
            max_genes=max_genes,
            return_confidences=return_confidences,
        )

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


def annotate_factors(
    model: "scDEF",
    annotations: Mapping[str, str],
) -> pd.DataFrame:
    """Attach descriptive annotations to factors in ``adata.uns['factor_obs']``.

    Annotations are stored in the ``annotation`` column of ``factor_obs``, keyed
    by the resolved factor rows (see :func:`_resolve_factor_obs_names`). Factor
    names may be current model names (e.g. ``L0_4``) even after filtering.

    Args:
        model: scDEF model instance.
        annotations: mapping ``{factor_name: description}``.

    Returns:
        Updated ``factor_obs`` dataframe.
    """
    if len(annotations) == 0:
        raise ValueError("annotations must contain at least one factor.")
    if "factor_obs" not in model.adata.uns:
        factor_diagnostics(model)
    factor_obs = model.adata.uns["factor_obs"]
    if "annotation" not in factor_obs.columns:
        factor_obs["annotation"] = pd.Series(
            pd.NA, index=factor_obs.index, dtype="object"
        )
    names = [str(k) for k in annotations.keys()]
    resolved, unknown = _resolve_factor_obs_names(model, names)
    if len(unknown) > 0:
        raise ValueError(
            "Unknown factor name(s) in `annotations`: " + ", ".join(map(str, unknown))
        )
    for user_name, obs_name in zip(names, resolved):
        factor_obs.loc[obs_name, "annotation"] = str(annotations[user_name])
    model.adata.uns["factor_obs"] = factor_obs
    return factor_obs


def get_factor_annotations(
    model: "scDEF",
    factor_names: Sequence[str],
) -> List[Optional[str]]:
    """Look up ``factor_obs['annotation']`` values for factor names.

    Args:
        model: scDEF model instance.
        factor_names: factor names as in ``model.factor_names[layer]``.

    Returns:
        List parallel to ``factor_names`` with annotation strings or ``None``.
    """
    if "factor_obs" not in model.adata.uns:
        return [None for _ in factor_names]
    factor_obs = model.adata.uns["factor_obs"]
    if "annotation" not in factor_obs.columns:
        return [None for _ in factor_names]
    resolved, unknown = _resolve_factor_obs_names(model, [str(n) for n in factor_names])
    out: List[Optional[str]] = []
    for i, name in enumerate(factor_names):
        if str(name) in unknown:
            out.append(None)
            continue
        val = factor_obs.loc[resolved[i], "annotation"]
        if pd.isna(val) or str(val).strip() == "":
            out.append(None)
        else:
            out.append(str(val))
    return out


def set_technical_factors(
    model: "scDEF",
    factors: Optional[Sequence[str]] = None,
    brd_min: Optional[float] = 1.0,
    ard_min: Optional[float] = 0.001,
    clarity_min: Optional[float] = 0.5,
    n_eff_parents_max: float = 1.5,
    brd_exceptional: Optional[float] = None,
    local_l0_scores: bool = False,
    batch_purity_max: Optional[float] = None,
    batch_purity_soft_max: Optional[float] = None,
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
            (default ``1.5``; matches ``scd.pl.factor_diagnostics``). When
            ``brd_exceptional`` is set, factors with ``BRD >= brd_exceptional`` are
            kept regardless.
        brd_exceptional: if set, high-BRD escape hatch when lineage effective parents
            exceed ``n_eff_parents_max``. Default ``None`` (disabled).
        local_l0_scores: if True, biological factors are chosen using ``n_eff_parents``
            and ``n_eff_parents_max`` instead of lineage averages / ``clarity_min``.
        batch_purity_max: if set, layer-0 factors with hard ``batch_purity`` above
            this value are not biological (requires
            ``factor_diagnostics(..., batch_key=...)``).
        batch_purity_soft_max: if set, same for soft ``batch_purity_soft``.
            Same semantics as ``filter_factors`` / ``factor_diagnostics`` plot.
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
                brd_exceptional=brd_exceptional,
                local_l0_scores=local_l0_scores,
                batch_purity_max=batch_purity_max,
                batch_purity_soft_max=batch_purity_soft_max,
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

    model.adata.uns.pop("confident_signatures", None)

    # Propagate upwards: a parent whose children are all technical is technical.
    #
    # `get_hierarchy` speaks *current* model names while `factor_obs` is keyed by
    # the names in place when diagnostics ran, so every name is resolved before
    # use. Parents are visited bottom-up (by layer) so that a newly technical
    # parent propagates to its own parent in the same pass.
    complete_hierarchy = get_hierarchy(model, simplified=False)
    layer_of_name: Dict[str, int] = {}
    for layer_idx, names in enumerate(model.factor_names):
        for name in names:
            layer_of_name[str(name)] = layer_idx

    factor_obs = model.adata.uns["factor_obs"]
    for parent in sorted(
        complete_hierarchy, key=lambda name: layer_of_name.get(str(name), 0)
    ):
        children = list(complete_hierarchy[parent])
        # A parent with no children carries no evidence of being technical.
        if len(children) == 0:
            continue
        parent_rows, parent_unknown = _resolve_factor_obs_names(model, [parent])
        if parent_unknown or not parent_rows:
            continue
        parent_row = parent_rows[0]
        if parent_row not in factor_obs.index:
            continue
        child_rows, child_unknown = _resolve_factor_obs_names(model, children)
        # Conservative: an unknown-status child must never sweep its parent into
        # the technical set.
        if child_unknown or len(child_rows) != len(children):
            continue
        if all(bool(factor_obs.loc[row, "technical"]) for row in child_rows):
            factor_obs.loc[parent_row, "technical"] = True

    model.annotate_adata()


def set_batch_technical_factors(
    model: "scDEF",
    factors: Sequence[str],
) -> None:
    """Mark layer-0 factors as *batch*-technical (per-batch splits of one type).

    Unlike :func:`set_technical_factors`, this flag does **not** propagate up the
    tree and no factor is deleted. Batch-technical factors are layer-0 per-batch
    views of a program that the parent layer already represents once, so that
    parent is the roll-up target and propagating the flag into it would throw
    away the very signal the roll-up depends on.

    This works on any fitted model with a batch column in ``adata.obs``, not only
    on one from :meth:`scDEF.decompose_batch_effects` — a plain fit made without
    a ``batch_key`` leaves batch structure in the factors directly, and can be
    flagged and corrected the same way. See :func:`batch_structure_report` for
    how the reading differs between the two.

    This records *which* factors are batch-technical and nothing else. The
    roll-up target comes from ``adata.uns['batch_technical_top_layer']``, written
    by :meth:`scDEF.decompose_batch_effects` as the layer whose ``z`` it froze —
    the only place that knows it. Consumers
    (:func:`factor_batch_correction`, :func:`assign_confident`) read it by
    default and take an override argument, so there is no second copy here to
    fall out of step with the decomposition.

    Sets ``factor_obs['batch_technical']``.

    Args:
        model: scDEF model instance.
        factors: factor names to flag. Names are resolved against the current
            ``model.factor_names`` (see :func:`_resolve_factor_obs_names`), so
            names taken from a filtered model are safe.

    Raises:
        ValueError: if any name in ``factors`` cannot be resolved.

    Example:
        >>> import scdef
        >>> # Stage 1: batch-corrected reference fit.
        >>> ref = scdef.scDEF(adata, counts_layer="counts", batch_key="Experiment")
        >>> ref.fit()
        >>> # Stage 2: re-learn L0/L1 without the batch key to expose batch programs.
        >>> model = scdef.scDEF.decompose_batch_effects(ref, top_layer=1)
        >>> # Describe the batch geometry, then decide from the design which
        >>> # branch splits are technical (here both batches are one donor).
        >>> rep = scdef.tl.batch_structure_report(model, batch_key="Experiment")
        >>> splits = rep.index[rep["shape"] == "branch_split"]
        >>> scdef.tl.set_batch_technical_factors(model, splits)
        >>> scdef.tl.factor_batch_correction(model, top_layer=1)
        >>> # Cells owned by a split are assigned at their L1 parent instead.
        >>> scdef.tl.assign_confident(model, exclude_batch_technical=True)
    """
    if "factor_obs" not in model.adata.uns:
        factor_diagnostics(model)
    factor_obs = model.adata.uns["factor_obs"]
    factor_obs["batch_technical"] = False

    names = [str(name) for name in factors]
    if len(names) > 0:
        resolved, unknown = _resolve_factor_obs_names(model, names)
        if len(unknown) > 0:
            raise ValueError(
                "Unknown factor name(s) in `factors`: " + ", ".join(map(str, unknown))
            )
        factor_obs.loc[resolved, "batch_technical"] = True

    model.adata.uns["factor_obs"] = factor_obs


def _batch_technical_l0_slots(model: "scDEF") -> List[int]:
    """Slots into ``model.factor_names[0]`` of the batch-technical L0 factors."""
    flagged = set(get_batch_technical_factors(model))
    if len(flagged) == 0:
        return []
    return [
        slot for slot, name in enumerate(model.factor_names[0]) if str(name) in flagged
    ]


def _resolve_batch_technical_top_layer(
    model: "scDEF", top_layer: Optional[int]
) -> int:
    """The batch-corrected parent layer flagged L0 factors group and roll up to.

    ``None`` reads ``adata.uns['batch_technical_top_layer']``, which
    :meth:`scDEF.decompose_batch_effects` records as the layer whose ``z`` it
    froze, falling back to ``1``. That record is the single source of truth: no
    other function writes it, so the roll-up target cannot drift from what the
    decomposition actually froze.
    """
    if int(model.n_layers) < 2:
        raise KeyError(
            "Batch-technical factors are flagged but the model has only one "
            "layer, so there is no parent layer to group them by."
        )
    if top_layer is None:
        top_layer = int(model.adata.uns.get("batch_technical_top_layer", 1))
    top_layer = int(top_layer)
    if top_layer < 1 or top_layer >= int(model.n_layers):
        raise ValueError(
            f"top_layer must be in [1, {int(model.n_layers) - 1}]; got {top_layer}."
        )
    return top_layer


def _l0_parent_slots(model: "scDEF", top_layer: int) -> np.ndarray:
    """Parent slot at ``top_layer`` for every kept layer-0 factor.

    Indexes ``model.factor_lists[top_layer]`` / ``factor_names[top_layer]``, and
    the ``argmax`` runs over the **kept** parents only, so a filtered-out factor
    can never win. Shared by :func:`factor_batch_correction` and
    :func:`_rollup_batch_factors` so the two cannot disagree about which flagged
    factors are siblings.
    """
    weights = _l0_to_top_layer_weights(model, top_layer)
    kept0 = np.asarray(model.factor_lists[0], dtype=int)
    if kept0.size == 0:
        return np.empty(0, dtype=int)
    if np.any(kept0 >= weights.shape[1]):
        raise KeyError(
            "model.factor_lists[0] indexes outside the layer-1 connection "
            "weights; the model's factor lists and posterior means are out of sync."
        )
    return np.argmax(weights[:, kept0], axis=0).astype(int)


def factor_batch_correction(
    model: "scDEF",
    reduce: Literal["sum", "max"] = "sum",
    key_added: str = "X_L0_batch_corrected",
    labels_key_added: str = "batch_corrected",
    top_layer: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Apply the batch-technical correction to the scores and to the labels.

    Factors flagged by :func:`set_batch_technical_factors` are per-batch views of
    a program that the batch-corrected ``top_layer`` already represents once.
    This removes them from the layer-0 representation and writes the same
    correction as cell-level labels, so an embedding, a heatmap and a UMAP
    colouring all describe one corrected view.

    Flagged factors are grouped by their ``top_layer`` parent and handled by
    group size:

    - **Two or more flagged siblings** under one parent -- the per-batch halves
      of a single program -- collapse into one merged column, labelled by
      joining the members with ``+`` in layer order (e.g. ``"L0_7+L0_14"``).
    - **A lone flagged factor** under a parent, with no flagged counterpart --
      a batch-skewed program with no opposite-batch half -- has nothing to merge
      with, so its column is **dropped**. The cells it claimed are then described
      by the factors they still score on, and their labels roll up to the parent.

    Anything not flagged is untouched and keeps its own column, including a
    batch-restricted factor that is genuine biology (an ISG program, say) and any
    non-flagged sibling of a flagged one. An embedding built on the result
    therefore mixes batches where the split was judged technical and keeps them
    apart where it was not.

    Two ``adata.obs`` columns are always written, from the same grouping:
    ``f"L0_{labels_key_added}"`` labels merged siblings by their joined name and
    a lone flagged factor by its parent, while ``labels_key_added`` labels every
    flagged cell by its parent. All other cells keep their layer-0 label in both.

    This addresses one shape of batch structure only: the one that appears as an
    *extra column per batch*. It does nothing about a factor that both batches
    use and merely score differently in magnitude -- on pbmcs2b, correcting the
    ``branch_split`` columns moves the median within-branch batch AUC only
    0.970 -> 0.930, because what is left is not confined to a clean pair of
    sibling columns.

    Args:
        model: scDEF model instance with cell scores annotated (``X_L0`` in
            ``adata.obsm``) and factors flagged by
            :func:`set_batch_technical_factors`.
        reduce: how to merge a group of two or more flagged siblings. ``"sum"``
            (default) adds them, which is exact when the halves are disjoint --
            the usual case, since each cell is dominated by the half from its own
            batch. ``"max"`` takes the per-cell peak instead. The two differ only
            for cells carrying real mass on *both* halves, where ``"sum"``
            reports the combined program and ``"max"`` the stronger half alone.
            Irrelevant to a lone flagged factor, which is dropped either way.
        key_added: ``adata.obsm`` key for the corrected matrix. The column labels
            go to ``adata.uns[key_added + '_factors']``, the per-column member
            names to ``..._members``, the flagged factors to
            ``..._batch_technical`` and the dropped ones to ``..._dropped``, so a
            cached matrix can be checked against the current flags.
        labels_key_added: base name for the two ``adata.obs`` columns.
        top_layer: the batch-corrected parent layer flagged factors group and
            roll up to. Defaults to ``adata.uns['batch_technical_top_layer']``,
            recorded by :meth:`scDEF.decompose_batch_effects` as the layer whose
            ``z`` it froze, else ``1``. Override only to inspect a different
            grouping.

    Returns:
        ``(matrix, labels)`` -- the corrected ``(n_cells, n_corrected_factors)``
        matrix and its column labels. The matrix has **fewer columns** than
        ``X_L0`` whenever anything was flagged.

    Raises:
        KeyError: ``X_L0`` is missing, the model has no parent layer to group by,
            or the connection weights are unavailable.
        ValueError: ``reduce`` is not ``"sum"`` or ``"max"``, ``top_layer`` is out
            of range, the stored scores are stale, or every column was dropped.

    Example:
        >>> model = scdef.scDEF.decompose_batch_effects(ref, top_layer=1)
        >>> rep = scdef.tl.batch_structure_report(model, batch_key="Experiment")
        >>> flagged = rep.index[rep["shape"].isin(["branch_split", "branch_skewed"])]
        >>> scdef.tl.set_batch_technical_factors(model, flagged)
        >>> matrix, labels = scdef.tl.factor_batch_correction(model)
        >>> scdef.pl.umap(model, color=["L0_batch_corrected", "batch_corrected"])
    """
    if reduce not in ("sum", "max"):
        raise ValueError(f"reduce must be 'sum' or 'max'; got {reduce!r}.")

    l0_name = model.layer_names[0]
    scores_key = f"X_{l0_name}"
    if scores_key not in model.adata.obsm:
        raise KeyError(
            f"{scores_key} not found in adata.obsm. Run model.annotate_adata() "
            "(or model.fit(annotate=True)) first."
        )
    scores = np.asarray(model.adata.obsm[scores_key], dtype=float)
    current_names = [str(name) for name in model.factor_names[0]]
    if scores.shape[1] != len(current_names):
        # Stale scores (e.g. the model was re-filtered without re-annotating)
        # would silently misalign every column, so fail loudly instead.
        raise ValueError(
            f"adata.obsm['{scores_key}'] has {scores.shape[1]} columns but the "
            f"model has {len(current_names)} layer-0 factors. Re-run "
            "model.annotate_adata() after filtering."
        )

    flagged_slots = _batch_technical_l0_slots(model)
    flagged_on = sorted(current_names[slot] for slot in flagged_slots)
    dropped: List[str] = []

    def _store(
        matrix: np.ndarray,
        labels: List[str],
        members: List[List[str]],
    ) -> Tuple[np.ndarray, List[str]]:
        model.adata.obsm[key_added] = matrix
        model.adata.uns[f"{key_added}_factors"] = list(labels)
        # Per-column member names. Consumers that need to map a column back to
        # its factors (e.g. hierarchy ordering) must use this rather than
        # splitting the label on '+', since factor names may themselves contain
        # '+' (iscDEF names factors after marker sets, e.g. 'CD14+ monocyte').
        model.adata.uns[f"{key_added}_members"] = [list(m) for m in members]
        # Provenance: which factors were flagged and which lost their column, so
        # a cached matrix can be checked against the current flags.
        model.adata.uns[f"{key_added}_batch_technical"] = list(flagged_on)
        model.adata.uns[f"{key_added}_dropped"] = list(dropped)
        return matrix, list(labels)

    if len(flagged_slots) == 0:
        # Nothing flagged: the corrected representation *is* the original one,
        # and there are no labels to write.
        return _store(scores.copy(), current_names, [[n] for n in current_names])

    top_layer = _resolve_batch_technical_top_layer(model, top_layer)
    parent_slot_of = _l0_parent_slots(model, top_layer)
    flagged_set = set(flagged_slots)

    # Group flagged factors by parent, preserving layer order within a group.
    groups: Dict[int, List[int]] = {}
    for slot in flagged_slots:
        groups.setdefault(int(parent_slot_of[slot]), []).append(slot)

    columns: List[np.ndarray] = []
    labels: List[str] = []
    column_members: List[List[str]] = []
    emitted_parents: set = set()
    for slot, name in enumerate(current_names):
        if slot not in flagged_set:
            columns.append(scores[:, slot])
            labels.append(name)
            column_members.append([name])
            continue
        parent = int(parent_slot_of[slot])
        members = groups[parent]
        if len(members) < 2:
            # No opposite-batch half to fold into, so the batch-specific program
            # leaves the representation; the cells roll up to the parent in obs.
            dropped.append(name)
            continue
        if parent in emitted_parents:
            continue  # already merged at the position of the group's first member
        emitted_parents.add(parent)
        block = scores[:, members]
        merged = block.sum(axis=1) if reduce == "sum" else block.max(axis=1)
        columns.append(merged)
        member_names = [current_names[s] for s in members]
        labels.append("+".join(member_names))
        column_members.append(member_names)

    if len(columns) == 0:
        raise ValueError(
            "Every layer-0 factor was flagged batch-technical, so the corrected "
            "representation would have no columns left. Flag fewer factors -- "
            "check the `shape` column of batch_structure_report."
        )

    result = _store(np.column_stack(columns).astype(float), labels, column_members)
    # The labels are part of the correction, not an optional extra: they express
    # the same grouping over cells that the columns express over factors.
    _rollup_batch_factors(model, key_added=labels_key_added, top_layer=top_layer)
    return result


def _factor_name_sort_key(name: str) -> Tuple[int, int, str]:
    """Sort factor names by their numeric suffix (``L0_7`` before ``L0_14``)."""
    tail = str(name).rsplit("_", 1)[-1]
    if tail.isdigit():
        return (0, int(tail), str(name))
    return (1, 0, str(name))


def _l0_to_top_layer_weights(model: "scDEF", top_layer: int) -> np.ndarray:
    """Connection weights from every L0 factor up to the kept ``top_layer`` factors.

    Returns an ``(n_kept_top, n_L0)`` matrix. For ``top_layer == 1`` this is just
    ``pmeans['L1W']`` restricted to the kept L1 rows; above that the per-layer
    ``W`` slices are chained the same way :func:`_get_layer_term_means`
    propagates loadings, always restricted to the currently kept factors.
    """
    weights = np.asarray(
        model.pmeans[f"{model.layer_names[1]}W"], dtype=float
    )  # (n_L1, n_L0)
    weights = weights[np.asarray(model.factor_lists[1], dtype=int), :]
    for layer in range(2, int(top_layer) + 1):
        upper = np.asarray(model.pmeans[f"{model.layer_names[layer]}W"], dtype=float)[
            np.asarray(model.factor_lists[layer], dtype=int)
        ][:, np.asarray(model.factor_lists[layer - 1], dtype=int)]
        weights = upper.dot(weights)
    return weights


def _rollup_batch_factors(
    model: "scDEF",
    flagged: Optional[Sequence[str]] = None,
    key_added: str = "batch_corrected",
    base_obs: str = "L0",
    top_layer: Optional[int] = None,
) -> Dict[str, str]:
    """Write batch-technical roll-up assignments to ``adata.obs``.

    Internal: :func:`factor_batch_correction` calls this so the labels always
    express the same grouping as the corrected columns. It is not part of the
    public API, because writing the labels without the matching score correction
    produces two views of the data that disagree.

    Two columns are written. They are the **same partition** of cells, labelled
    two ways:

    - ``f"{base_obs}_{key_added}"`` (default ``"L0_batch_corrected"``): flagged
      L0 siblings sharing a parent are merged into one ``"L0_a+L0_b"`` label. A
      *lone* flagged factor has no sibling to merge with, so it takes its
      parent's name -- matching :func:`factor_batch_correction`, which drops that
      factor's column outright. Every other cell keeps its ``base_obs`` label.
    - ``key_added`` (default ``"batch_corrected"``): every flagged cell is
      labelled by its parent factor at ``top_layer`` instead; every other cell
      keeps its ``base_obs`` label.

    Args:
        model: scDEF model instance.
        flagged: current L0 factor names to roll up. Defaults to the factors
            marked ``batch_technical`` in ``factor_obs``.
        key_added: name of the parent-labelled column; the sibling-labelled
            column is ``f"{base_obs}_{key_added}"``.
        base_obs: source hard-assignment column in ``adata.obs`` (default
            ``"L0"``, written by ``annotate_adata``).
        top_layer: parent layer the split cells roll up to. Defaults to
            ``adata.uns['batch_technical_top_layer']``, else ``1``.

    Returns:
        ``{merged_label: parent_name}`` — the correspondence between the two
        columns for groups of two or more siblings. Lone flagged factors are
        omitted: both columns already show the parent for those.

    Raises:
        ValueError: if no factors are flagged and ``flagged`` is None, or if
            ``base_obs`` is missing from ``adata.obs``.

    Example:
        >>> scdef.tl.set_batch_technical_factors(model, splits)
        >>> scdef.tl.assign_confident(model, exclude_batch_technical=True)
        >>> mapping = _rollup_batch_factors(model)
        >>> scdef.pl.umap(model, color=["L0_batch_corrected", "batch_corrected"])
    """
    if base_obs not in model.adata.obs.columns:
        raise ValueError(
            f"base_obs '{base_obs}' not found in adata.obs. Run "
            "model.annotate_adata() first, or pass the column holding the "
            "layer-0 hard assignments."
        )

    if flagged is None:
        flagged = get_batch_technical_factors(model)
        if len(flagged) == 0:
            raise ValueError(
                "No batch-technical factors are set. Run "
                "scdef.tools.set_batch_technical_factors(model, splits) first, "
                "or pass `flagged` explicitly."
            )
    flagged = [str(name) for name in flagged]

    top_layer = _resolve_batch_technical_top_layer(model, top_layer)
    # Same parent resolution as `factor_batch_correction`, so the labels and the
    # corrected columns cannot disagree about which factors are siblings.
    parent_slot_of = _l0_parent_slots(model, top_layer)
    kept_top = np.asarray(model.factor_lists[top_layer], dtype=int)
    slot_of_orig = {
        int(orig): slot
        for slot, orig in enumerate(np.asarray(model.factor_lists[0], dtype=int))
    }

    # Resolve each flagged L0 factor to its parent at `top_layer`.
    parent_of: Dict[str, str] = {}
    for name in flagged:
        key = original_key_of_factor(model, name)
        if key is None or key[0] != 0:
            raise ValueError(
                f"{name!r} is not a current layer-0 factor of this model. "
                "Roll-up applies to L0 splits."
            )
        slot = slot_of_orig.get(int(key[1]))
        if slot is None:
            raise ValueError(
                f"{name!r} maps to layer-0 index {key[1]}, which is not among the "
                "factors currently kept."
            )
        parent_orig = int(kept_top[int(parent_slot_of[slot])])
        parent_name = current_name_of_original(model, top_layer, parent_orig)
        if parent_name is None:
            raise ValueError(
                f"Could not resolve a kept parent at layer {top_layer} for {name!r}."
            )
        parent_of[name] = str(parent_name)

    # Group flagged factors by parent: only same-parent siblings merge.
    groups: Dict[str, List[str]] = {}
    for name, parent in parent_of.items():
        groups.setdefault(parent, []).append(name)

    sibling_label_of: Dict[str, str] = {}
    mapping: Dict[str, str] = {}
    for parent, members in groups.items():
        members = sorted(members, key=_factor_name_sort_key)
        if len(members) < 2:
            # Nothing to merge with: a lone flagged factor rolls up to its parent
            # in both columns, matching `factor_batch_correction`, which drops
            # that factor's column outright.
            sibling_label_of[members[0]] = parent
            continue
        label = "+".join(members)
        for member in members:
            sibling_label_of[member] = label
        mapping[label] = parent

    base = model.adata.obs[base_obs].astype(str)
    sibling_values = base.map(lambda v: sibling_label_of.get(v, v))
    parent_values = base.map(lambda v: parent_of.get(v, v))

    def _ordered(source: pd.Series) -> pd.Categorical:
        # Preserve the base column's category order, with each merged label
        # sitting where its first member was.
        if isinstance(model.adata.obs[base_obs].dtype, pd.CategoricalDtype):
            base_order = [str(c) for c in model.adata.obs[base_obs].cat.categories]
        else:
            base_order = sorted(base.unique(), key=_factor_name_sort_key)
        categories: List[str] = []
        for value in base_order:
            mapped = sibling_label_of.get(value) if source is sibling_values else None
            if source is parent_values:
                mapped = parent_of.get(value)
            mapped = mapped if mapped is not None else value
            if str(mapped) not in categories:
                categories.append(str(mapped))
        for value in source.unique():
            if str(value) not in categories:
                categories.append(str(value))
        return pd.Categorical(source.astype(str), categories=categories, ordered=True)

    model.adata.obs[f"{base_obs}_{key_added}"] = _ordered(sibling_values)
    model.adata.obs[key_added] = _ordered(parent_values)
    return mapping


# `shape` cut points. Deliberately module constants rather than parameters:
# `shape` is a coarse descriptive bucket, and letting callers tune the cut
# points would invite reading it as a tuned classifier, which it is not.
_SHAPE_FRAC_DOM_MIN = 0.7
_SHAPE_EFF_PARENTS_MAX = 1.5

# Cross-validation settings for the per-branch batch AUC.
_BRANCH_AUC_N_SPLITS = 3
_BRANCH_AUC_MIN_CLASS = 15


def _batch_codes_for_report(
    model: "scDEF", batch_key: str, caller: str = "batch_structure_report"
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell integer batch codes (``-1`` for missing) and the batch labels.

    Raises with an actionable message when ``batch_key`` is absent or carries
    fewer than two observed values, since every quantity built from these codes
    is a contrast between batches. ``caller`` names the public function in the
    error text, so the message points at what the user actually called.
    """
    obs = model.adata.obs
    if batch_key not in obs.columns:
        raise KeyError(
            f"batch_key '{batch_key}' not found in model.adata.obs. "
            f"Available keys: {list(obs.columns)}"
        )
    values = pd.Series(np.asarray(obs[batch_key].to_numpy(), dtype=object))
    codes, uniques = pd.factorize(values, sort=True)
    codes = np.asarray(codes, dtype=int)
    labels = np.asarray([str(u) for u in uniques], dtype=object)
    if labels.size < 2:
        observed = list(labels)
        raise ValueError(
            f"batch_key '{batch_key}' has {labels.size} distinct non-missing "
            f"value(s) ({observed}); {caller} contrasts batches "
            "and needs at least 2."
        )
    return codes, labels


def _branch_batch_auc(
    features: np.ndarray,
    batch_codes: np.ndarray,
    cells: np.ndarray,
    min_cells: int,
    random_seed: int,
) -> float:
    """Cross-validated AUC for predicting batch from L0 scores inside one branch.

    ``features`` is ``log1p`` of the kept-L0 cell scores for *all* cells;
    ``cells`` selects the branch. Features are standardized inside each fold.
    Returns ``NaN`` when the branch is too small or too imbalanced to score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    cells = np.asarray(cells, dtype=int)
    cells = cells[batch_codes[cells] >= 0]
    if cells.size < int(min_cells):
        return np.nan
    y = batch_codes[cells]
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2 or int(counts.min()) < _BRANCH_AUC_MIN_CLASS:
        return np.nan

    X = features[cells]
    splitter = StratifiedKFold(
        n_splits=_BRANCH_AUC_N_SPLITS, shuffle=True, random_state=int(random_seed)
    )
    scores: List[float] = []
    for train_idx, test_idx in splitter.split(X, y):
        y_train, y_test = y[train_idx], y[test_idx]
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(X[train_idx]), y_train)
        proba = clf.predict_proba(scaler.transform(X[test_idx]))
        try:
            if len(clf.classes_) == 2:
                score = roc_auc_score(y_test, proba[:, 1])
            else:
                score = roc_auc_score(
                    y_test,
                    proba,
                    multi_class="ovr",
                    average="macro",
                    labels=clf.classes_,
                )
        except ValueError:
            # A class missing from this test fold: skip rather than crash.
            continue
        scores.append(float(score))
    if len(scores) == 0:
        return np.nan
    return float(np.mean(scores))


def batch_structure_report(
    model: "scDEF",
    batch_key: str,
    group_layer: int = 1,
    min_group_cells: int = 80,
    random_seed: int = 0,
    reference: Union["scDEF", pd.DataFrame, np.ndarray, None] = None,
) -> pd.DataFrame:
    """Describe *how* batch structure appears among the layer-0 factors.

    Summarises the **shape** of that structure — how batch-skewed each L0 factor
    is, whether it has an opposite-batch sibling under the same parent, whether
    it is confined to one branch or overlaid across several, and how separable
    the batches are inside each branch — so that the analyst can decide what to
    filter, correct (:func:`factor_batch_correction`) or keep.

    **Any fitted model with two layers works**, provided ``batch_key`` is a
    column of ``adata.obs``. The model need not have been fitted with that key,
    and need not have come from :meth:`scDEF.decompose_batch_effects`. What the
    report *means* does depend on which it is:

    - a **plain fit with no** ``batch_key`` — the batch was never corrected, so
      this is the raw batch structure as the factorization happened to capture
      it. The natural first look, before deciding whether to use a batch key at
      all.
    - a **decomposed model** — the upper layers are batch-corrected and frozen
      while L0 was re-learned without per-batch gene scales, so batch structure
      is pushed into L0 and read against a corrected hierarchy. This is the
      configuration the ``shape`` buckets were designed around.
    - a **fit made with** ``batch_key`` — what is left is the *residual*
      structure the per-batch ``gene_scale`` did not absorb, which is a smaller
      and different quantity. Useful for auditing a correction, but do not read
      its magnitudes as the batch effect in the data.

    It deliberately emits **no verdict**. Nothing here distinguishes a technical
    per-batch duplication of one cell type from a genuine condition-specific
    biological program: in paired reference data the two are indistinguishable on
    every column below (same ``eff_parents`` ~1, same ``opp_batch_sibling``, same
    near-ceiling ``branch_auc``). Only the experimental design can settle that,
    which is why ``shape`` is a geometric label and never a cause.

    This function supersedes the removed verdict-style "which factors look
    technical" suggester, whose flag columns did not hold up: its split rule
    fired on the IFN data's *biological* monocyte factors, and the
    ``batch_split_corr`` it ranked on saturates at 0.79-0.99 across factor pairs
    sharing no genes, placing the genuine pbmcs2b Cytotoxic-T split below
    unrelated pairs. The same geometry is reported here, descriptively.

    No cell-type or other biological annotation is read. The inputs are
    ``model.pmeans``, ``model.factor_lists`` / ``factor_names``,
    ``adata.obs[batch_key]``, and — for the two optional gene-side columns — the
    reference fit's per-batch ``gene_scale``.

    **Cell side and gene side.** Every column below except the last two is
    *cell*-side: it asks which cells score on a factor and how they are spread
    over batches. ``gene_scale_affinity`` is *gene*-side: it asks whether the
    factor's gene programme is the one the reference fit's per-batch
    ``gene_scale`` absorbed, and never looks at a cell. A factor can be perfectly
    mixed across batches and still be built from exactly those genes, or the
    reverse, so the two are worth reading together.

    .. warning::

       ``gene_scale_affinity`` is **not** a technical-vs-biological score and
       must not be sorted on as if it were. The per-batch ``gene_scale`` absorbs
       whatever differs between batches at the gene level, and what that is
       depends entirely on what the batch key encodes. On Kang CTRL/STIM the
       top-scoring factor is the interferon-response factor — the biology the
       experiment is about, which must be kept. On pbmcs2b (two runs of one
       donor) the top-scoring factor is a stress/ambient programme — an artefact
       to remove. Identical statistic, opposite verdicts; only the experimental
       design settles it. This is the same trap that the removed
       "which factors look technical" suggester fell into.

    Notation: ``kept0 = model.factor_lists[0]``, ``Z = pmeans['L0z'][:, kept0]``,
    ``W`` the connection weights from L0 up to the kept ``group_layer`` factors,
    ``a0(c) = argmax_k Z[c, k]`` the hard L0 assignment, ``S_k`` the cells
    assigned to factor ``k``, and ``Znorm`` the row-normalized ``Z`` (each cell's
    loading distributed over the kept L0 factors).

    Columns, one row per kept L0 factor:

    - ``n_cells``: ``|S_k|``. Sums to ``n_obs`` over the frame. This is the only
      column that counts cells with a missing ``batch_key`` value; every
      batch-derived column below is computed on the labelled cells alone.
    - ``dom_batch``: modal batch of ``S_k`` (empty string if the factor has no
      cells with a batch label).
    - ``frac_dom_batch``: fraction of the *batch-labelled* cells of ``S_k`` that
      lie in ``dom_batch``. Hard purity, with a floor at the largest batch prior
      and a ceiling of 1.
    - ``batch_purity_soft``: ``max_b mass_b / sum_b mass_b`` where
      ``mass_b = sum_{c in batch b} Znorm[c, k]`` over **all** cells. Floored near
      ``1/n_batches`` and strongly compressed relative to ``frac_dom_batch``,
      because normalized loading has a wide background across cells that the
      factor does not explain.
    - ``loading_ratio``: median ``Znorm[:, k]`` in ``dom_batch`` over the median
      in the other batches, taken over **all** cells. See the note on estimator
      sensitivity below — read it as a rough direction check, not as a ranking.
    - ``eff_parents``: ``exp(H(p))`` for ``p`` the factor's ``W`` column
      normalized to a distribution, i.e. the effective number of ``group_layer``
      parents it loads on. Range ``[1, n_kept_group]``. Values below ~1.1 are
      indistinguishable from the numerical noise in tiny ``W`` entries; read the
      column as a coarse "one parent" vs "several parents" flag.
    - ``parent``: current name of the ``argmax`` ``group_layer`` parent.
    - ``opp_batch_sibling``: ``True`` when another kept L0 factor shares this
      ``parent`` and has a *different* ``dom_batch`` — the geometry of a
      per-batch split of one branch, whatever its cause.
    - ``branch_auc``: cross-validated AUC (3-fold stratified logistic regression
      on ``log1p(Z)``, standardized in-fold) for predicting batch from the L0
      scores of the cells in ``parent``'s branch. A **branch-level** quantity: it
      is identical for all children of a parent and cannot separate siblings.
      ``NaN`` when the branch has fewer than ``min_group_cells`` labelled cells,
      fewer than two batches, or a batch with fewer than 15 cells.
    - ``shape``: descriptive bucket, **geometry only, never cause**:

      ==================  ==========================================================
      ``branch_split``    ``frac_dom_batch >= 0.7``, ``eff_parents < 1.5``, has an
                          opposite-batch sibling. One branch appearing as two
                          batch-skewed halves.
      ``branch_skewed``   same but with no opposite-batch sibling. A batch-skewed
                          factor whose branch has no counterpart half.
      ``overlaid``        ``frac_dom_batch >= 0.7`` and ``eff_parents >= 1.5``. A
                          batch-skewed program spread over several branches rather
                          than confined to one.
      ``balanced``        everything else, i.e. ``frac_dom_batch < 0.7``.
      ==================  ==========================================================

    - ``gene_scale_affinity_<batch>``, one per batch, plus
      ``gene_scale_affinity_max`` and ``gene_scale_affinity_batch``: **present
      only when the reference fit's per-batch gene_scale is available** (see
      ``reference``). Each per-batch column is the Spearman correlation, over all
      genes, between the factor's gene loadings and the per-gene log-ratio of
      that batch's ``gene_scale`` against the other batches. ``_max`` is the
      largest of them and ``_batch`` names which batch attains it — the
      ``top_score`` and ``top_batch`` of
      :func:`get_factor_batch_gene_scale_affinity`. With exactly two batches the
      two contrasts are mirror images, so the per-batch columns are exact
      negatives of each other, ``_max`` is non-negative and ``_batch`` only says
      which side; the magnitude is the information. With many batches the frame
      gets correspondingly wide — ``attrs['gene_scale_affinity']`` holds the same
      numbers. Read the warning above before using any of them.

    Rows are sorted by ``frac_dom_batch`` descending.

    ``result.attrs`` carries ``batch_key``, ``group_layer``, ``min_group_cells``,
    ``random_seed``, the ``shape`` cut points (``frac_dom_min``,
    ``eff_parents_max``) and ``branch_summary``: a DataFrame indexed by kept
    ``group_layer`` factor with ``n_cells`` (cells whose hard assignment at
    ``group_layer`` is this factor), ``n_l0_children`` (kept L0 factors whose
    ``argmax`` parent is this factor), ``batch_auc`` and ``max_child_frac_dom``.
    When the gene-side columns are present it also carries
    ``gene_scale_affinity``: the full factors-by-batches frame, with one column
    per batch rather than only the best one.

    Args:
        model: any fitted scDEF model with at least two layers; see above for
            how the reading changes with how it was fitted.
        batch_key: key in ``adata.obs`` holding the batch labels. Needs at least
            two observed values; the model itself need not have been fitted with
            this key.
        group_layer: layer whose factors define the branches (default ``1``,
            the layer usually frozen by the decomposition). Above ``1`` the
            per-layer ``W`` slices are chained, as in
            :func:`factor_batch_correction`.
        min_group_cells: minimum labelled cells in a branch before its
            ``batch_auc`` is estimated; smaller branches report ``NaN``.
        random_seed: seed for the cross-validation splits.
        reference: where the per-batch ``gene_scale`` contrast for the two
            gene-side columns comes from — a fitted reference scDEF model, a
            genes-by-batches DataFrame of log-ratios, or a
            ``(n_batches, n_genes)`` array, as accepted by
            :func:`get_factor_batch_gene_scale_affinity`. ``None`` (default) uses
            the profile :meth:`scDEF.decompose_batch_effects` stored on this
            model. If neither is available the two columns are simply **omitted**
            and the rest of the report is unaffected — a model decomposed before
            that record existed needs the reference passed explicitly.

    Returns:
        DataFrame indexed by current L0 factor name with columns ``n_cells``,
        ``dom_batch``, ``frac_dom_batch``, ``batch_purity_soft``,
        ``loading_ratio``, ``eff_parents``, ``parent``, ``opp_batch_sibling``,
        ``branch_auc`` and ``shape``, sorted by ``frac_dom_batch`` descending.

    Raises:
        KeyError: ``batch_key`` is not in ``adata.obs``.
        ValueError: ``batch_key`` has fewer than two observed values, or
            ``group_layer`` is out of range.

    Note:
        Three caveats worth carrying into any reading of the frame.

        ``loading_ratio`` is estimator-sensitive. Taking the median over all
        cells measures the factor's *background* loading level rather than its
        loading where it is active, and it can point at the opposite batch from
        ``frac_dom_batch``. Restricting the median to ``S_k`` compresses every
        factor into a narrow band; using means tracks ``frac_dom_batch`` much
        more closely. The all-cell median is reported for continuity, but the
        column should not be used to rank factors.

        ``parent`` and ``branch_auc`` use *different* definitions of the
        hierarchy: ``parent`` is the ``argmax`` over the ``W`` column, while the
        branch is defined by the ``argmax`` over the ``group_layer`` cell scores.
        These disagree for some factors, and where they do, ``branch_auc`` is
        measured on a cell population that largely excludes the factor's own
        cells.

        Small ``n_cells`` rows (a few tens of cells) have a binomial standard
        error on ``frac_dom_batch`` of 0.05-0.10 and should not be compared with
        rows of several hundred cells.

    Example:
        >>> ref = scdef.scDEF(adata, counts_layer="counts", batch_key="stim")
        >>> ref.fit()
        >>> model = scdef.scDEF.decompose_batch_effects(ref, top_layer=1)
        >>> report = scdef.tl.batch_structure_report(model, batch_key="stim")
        >>> report.loc[report["shape"] == "branch_split", ["parent", "dom_batch"]]
        >>> report.attrs["branch_summary"].head()
    """
    batch_codes, batch_labels = _batch_codes_for_report(model, batch_key)

    group_layer = int(group_layer)
    if group_layer < 1 or group_layer >= int(model.n_layers):
        raise ValueError(
            f"group_layer must be in [1, {int(model.n_layers) - 1}]; got {group_layer}."
        )

    l0_name = str(model.layer_names[0])
    kept0 = np.asarray(model.factor_lists[0], dtype=int)
    kept_group = np.asarray(model.factor_lists[group_layer], dtype=int)
    if kept0.size == 0 or kept_group.size == 0:
        raise ValueError(
            "The model has no kept factors at layer 0 or at "
            f"layer {group_layer}; nothing to report."
        )

    # Read the cell scores from `pmeans` rather than `adata.obsm`, so the
    # partition here is by construction the same one `hard_assignment_*` and
    # `factor_diagnostics` use even if `annotate_adata` has not been re-run.
    scores = np.asarray(model.pmeans[f"{l0_name}z"], dtype=float)[:, kept0]
    n_cells_total = scores.shape[0]
    if batch_codes.size != n_cells_total:
        raise ValueError(
            f"adata.obs['{batch_key}'] has {batch_codes.size} entries but the "
            f"model's layer-0 scores have {n_cells_total} cells."
        )
    assignment = np.argmax(scores, axis=1)
    normalized = scores / np.clip(scores.sum(axis=1, keepdims=True), 1e-9, None)
    features = np.log1p(np.clip(scores, 0.0, None))

    # (n_kept_group, n_L0_full) -> restrict columns to the kept L0 factors.
    weights = _l0_to_top_layer_weights(model, group_layer)[:, kept0]

    # Branch = hard assignment at `group_layer`; slot indexes factor_names.
    group_assignment = hard_assignment_name_indices(model, group_layer)
    group_names = [str(n) for n in model.factor_names[group_layer]]
    branch_auc_by_slot: Dict[int, float] = {}
    branch_cells_by_slot: Dict[int, int] = {}
    for slot in range(kept_group.size):
        cells = np.where(group_assignment == slot)[0]
        branch_cells_by_slot[slot] = int(cells.size)
        branch_auc_by_slot[slot] = _branch_batch_auc(
            features, batch_codes, cells, min_group_cells, random_seed
        )

    n_batches = int(batch_labels.size)
    labelled = batch_codes >= 0

    names: List[str] = []
    records: List[Dict[str, Any]] = []
    parent_slots: List[int] = []
    for slot, orig in enumerate(kept0):
        name = str(model.factor_names[0][slot])
        cells = np.where(assignment == slot)[0]

        # --- batch skew of the cells this factor claims (hard) ---
        dom_batch = ""
        frac_dom = np.nan
        dom_code = -1
        cell_codes = batch_codes[cells]
        cell_codes = cell_codes[cell_codes >= 0]
        if cell_codes.size > 0:
            counts = np.bincount(cell_codes, minlength=n_batches).astype(float)
            dom_code = int(np.argmax(counts))
            dom_batch = str(batch_labels[dom_code])
            frac_dom = float(counts[dom_code] / counts.sum())

        # --- batch skew of the loading mass, over every cell (soft) ---
        column = normalized[:, slot]
        purity_soft = np.nan
        loading_ratio = np.nan
        if n_batches >= 2:
            mass = np.array(
                [
                    float(column[labelled & (batch_codes == b)].sum())
                    for b in range(n_batches)
                ]
            )
            total_mass = float(mass.sum())
            if total_mass > 0.0:
                purity_soft = float(mass.max() / total_mass)
        if dom_code >= 0:
            in_dom = labelled & (batch_codes == dom_code)
            out_dom = labelled & (batch_codes != dom_code)
            if in_dom.any() and out_dom.any():
                loading_ratio = float(
                    np.median(column[in_dom]) / (np.median(column[out_dom]) + 1e-9)
                )

        # --- position in the hierarchy ---
        parent_column = np.clip(weights[:, slot], 0.0, None)
        parent_total = float(parent_column.sum())
        eff_parents = np.nan
        if parent_total > 0.0:
            probs = parent_column / parent_total
            probs = probs[probs > 0.0]
            eff_parents = float(np.exp(-np.sum(probs * np.log(probs))))
        parent_slot = int(np.argmax(parent_column))
        parent_slots.append(parent_slot)
        parent_name = current_name_of_original(
            model, group_layer, int(kept_group[parent_slot])
        )

        names.append(name)
        records.append(
            {
                "n_cells": int(cells.size),
                "dom_batch": dom_batch,
                "frac_dom_batch": frac_dom,
                "batch_purity_soft": purity_soft,
                "loading_ratio": loading_ratio,
                "eff_parents": eff_parents,
                "parent": "" if parent_name is None else str(parent_name),
                "branch_auc": branch_auc_by_slot.get(parent_slot, np.nan),
            }
        )

    # --- opposite-batch sibling: same argmax parent, different dominant batch ---
    for i, record in enumerate(records):
        opp = False
        if record["n_cells"] > 0 and record["dom_batch"]:
            for j, other in enumerate(records):
                if j == i or other["n_cells"] <= 0 or not other["dom_batch"]:
                    continue
                if (
                    parent_slots[j] == parent_slots[i]
                    and other["dom_batch"] != record["dom_batch"]
                ):
                    opp = True
                    break
        record["opp_batch_sibling"] = opp

        # `shape` is a description of geometry, not a claim about cause.
        frac_dom = record["frac_dom_batch"]
        eff_parents = record["eff_parents"]
        skewed = bool(np.isfinite(frac_dom) and frac_dom >= _SHAPE_FRAC_DOM_MIN)
        multi_parent = bool(
            np.isfinite(eff_parents) and eff_parents >= _SHAPE_EFF_PARENTS_MAX
        )
        if not skewed:
            shape = "balanced"
        elif multi_parent:
            shape = "overlaid"
        elif opp:
            shape = "branch_split"
        else:
            shape = "branch_skewed"
        record["shape"] = shape

    columns = [
        "n_cells",
        "dom_batch",
        "frac_dom_batch",
        "batch_purity_soft",
        "loading_ratio",
        "eff_parents",
        "parent",
        "opp_batch_sibling",
        "branch_auc",
        "shape",
    ]
    result = pd.DataFrame(records, index=pd.Index(names, name=l0_name))[columns]

    # --- gene-side evidence, when the reference profile is available ---
    # Everything above is cell-side. This asks whether a factor's gene programme
    # is the one the reference fit's per-batch `gene_scale` absorbed. It is
    # optional: a model decomposed before that profile was recorded, or a model
    # that never had a per-batch gene side, simply gets no such columns.
    affinity: Optional[pd.DataFrame] = None
    try:
        affinity = get_factor_batch_gene_scale_affinity(model, reference)
    except (ValueError, TypeError, KeyError) as exc:
        if reference is not None:
            # The caller explicitly asked for it, so do not silently drop it.
            raise
        if hasattr(model, "logger"):
            model.logger.info(
                "batch_structure_report: no gene_scale affinity columns (%s)", exc
            )
    if affinity is not None:
        aff = affinity.reindex(result.index)
        summary_names = ("gene_scale_affinity_max", "gene_scale_affinity_batch")
        for label in affinity.attrs.get("batch_columns", []):
            name = f"gene_scale_affinity_{label}"
            # A batch literally named "max" or "batch" would shadow a summary
            # column; suffix it rather than silently overwrite.
            if name in summary_names or name in result.columns:
                name = f"{name}_"
            result[name] = aff[label]
        result["gene_scale_affinity_max"] = aff[TOP_SCORE_COL]
        result["gene_scale_affinity_batch"] = aff[TOP_BATCH_COL]

    if len(result) > 0:
        result = result.sort_values("frac_dom_batch", ascending=False, kind="stable")

    # --- per-branch summary ---
    summary_records: List[Dict[str, Any]] = []
    for slot in range(kept_group.size):
        children = [i for i, p in enumerate(parent_slots) if p == slot]
        child_fracs = [
            records[i]["frac_dom_batch"]
            for i in children
            if np.isfinite(records[i]["frac_dom_batch"])
        ]
        summary_records.append(
            {
                "n_cells": branch_cells_by_slot.get(slot, 0),
                "n_l0_children": len(children),
                "batch_auc": branch_auc_by_slot.get(slot, np.nan),
                "max_child_frac_dom": (
                    float(np.max(child_fracs)) if len(child_fracs) > 0 else np.nan
                ),
            }
        )
    branch_summary = pd.DataFrame(
        summary_records,
        index=pd.Index(group_names, name=str(model.layer_names[group_layer])),
        columns=["n_cells", "n_l0_children", "batch_auc", "max_child_frac_dom"],
    )

    result.attrs["batch_key"] = str(batch_key)
    result.attrs["batches"] = [str(b) for b in batch_labels]
    result.attrs["group_layer"] = group_layer
    result.attrs["min_group_cells"] = int(min_group_cells)
    result.attrs["random_seed"] = int(random_seed)
    result.attrs["frac_dom_min"] = float(_SHAPE_FRAC_DOM_MIN)
    result.attrs["eff_parents_max"] = float(_SHAPE_EFF_PARENTS_MAX)
    result.attrs["branch_auc_n_splits"] = int(_BRANCH_AUC_N_SPLITS)
    result.attrs["branch_auc_min_class"] = int(_BRANCH_AUC_MIN_CLASS)
    result.attrs["branch_summary"] = branch_summary
    if affinity is not None:
        # The full factors-by-batches frame, not just the best-matching batch.
        result.attrs["gene_scale_affinity"] = affinity
    return result


def filter_factors(
    model: "scDEF",
    batch_key: Optional[str] = None,
    diagnostics_kwargs: Optional[Mapping[str, Any]] = None,
    **filter_kwargs: Any,
) -> None:
    """Filter factors and refresh the diagnostics and signatures in one step.

    ``model.filter_factors`` renames factors, which invalidates everything keyed
    by those names — the stored signatures, the hierarchies, and the frozen
    upper-layer subset used by :func:`factor_diagnostics`. Calling the two
    separately leaves the model in that in-between state, where
    ``scd.pl.make_graph(show_signatures=True)`` raises until diagnostics are
    re-run. This wrapper does both, so the model is immediately usable::

        scd.tl.filter(model, batch_key="Experiment", brd_min=1.0)

    Equivalent to::

        model.filter_factors(brd_min=1.0)
        scd.tl.factor_diagnostics(model, batch_key="Experiment")

    Args:
        model: scDEF model instance.
        batch_key: passed to :func:`factor_diagnostics`; needed for the batch
            metrics (``batch_purity``, ``frac_dom_batch``, ``batch_split_corr``).
        diagnostics_kwargs: optional extra keyword arguments for
            :func:`factor_diagnostics` (e.g. ``{"mc_samples": 200}``).
        **filter_kwargs: forwarded to :meth:`scDEF.filter_factors`
            (``brd_min``, ``ard_min``, ``n_eff_parents_max``, ``keep``, ...).

    Example:
        >>> scdef.tl.filter(model, batch_key="Experiment", brd_min=1.0)
        >>> scdef.pl.make_graph(model, show_signatures=True)   # works right away
    """
    model.filter_factors(**filter_kwargs)
    factor_diagnostics(model, batch_key=batch_key, **(diagnostics_kwargs or {}))


def drop_technical(model: "scDEF") -> None:
    """Remove factors marked ``technical`` from ``factor_lists`` and re-annotate.

    Technical flags in ``factor_obs`` are cleared for the remaining factors.

    Args:
        model: fitted scDEF model instance
    """
    if "factor_obs" not in model.adata.uns:
        factor_diagnostics(model)

    if "technical" in model.adata.uns["factor_obs"].columns:
        technical_names = set(get_technical_factors(model))
        if len(technical_names) > 0:
            new_factor_lists = []
            for layer_idx in range(model.n_layers):
                keep = [
                    int(model.factor_lists[layer_idx][slot])
                    for slot, name in enumerate(model.factor_names[layer_idx])
                    if name not in technical_names
                ]
                if len(keep) == 0:
                    raise ValueError(
                        f"Cannot drop technical factors: layer "
                        f"{model.layer_names[layer_idx]} would have no factors left."
                    )
                new_factor_lists.append(np.array(keep, dtype=int))

            model.factor_lists = new_factor_lists
            model.set_factor_names()
            model._sync_factor_obs_with_filter()
            model.make_layercolors(
                layer_cpal=model.layer_cpal, lightness_mult=model.lightness_mult
            )
            model.adata.uns.pop("confident_signatures", None)
            model.adata.uns.pop("biological_hierarchy", None)
            model.adata.uns.pop("technical_hierarchy", None)

    model.annotate_adata()


def set_global_factors(
    model: "scDEF",
    factors: Optional[Sequence[str]] = None,
    layer_idx: int = 0,
    n_eff_parents_min: float = 1.5,
    exclude_technical: bool = True,
) -> None:
    """Mark global (shared-across-lineages) factors in ``factor_obs``.

    Global factors are identified from hierarchy diagnostics (high effective
    parents). They are excluded from :func:`make_biological_hierarchy`. Use
    :func:`drop_technical` to remove technical factors from the active model.

    Args:
        model: scDEF model instance
        factors: explicit factor names to mark as global (resolved like
            :func:`set_technical_factors`). When ``None``, uses
            :func:`scdef.tools.lineage.get_global_factors`.
        layer_idx: child layer for automatic selection (default 0).
        n_eff_parents_min: minimum effective-parent score when ``factors`` is None.
        exclude_technical: do not mark technical factors as global.
    """
    from scdef.tools.lineage import get_global_factors

    if "factor_obs" not in model.adata.uns:
        factor_diagnostics(model)
    if "global" not in model.adata.uns["factor_obs"].columns:
        model.adata.uns["factor_obs"]["global"] = False
    model.adata.uns["factor_obs"]["global"] = False

    factor_obs = model.adata.uns["factor_obs"]
    global_factor_rows: List[str] = []
    if factors is not None:
        resolved, unknown = _resolve_factor_obs_names(model, factors)
        if len(unknown) > 0:
            raise ValueError(
                "Unknown factor name(s) in `factors`: " + ", ".join(map(str, unknown))
            )
        global_factor_rows = resolved
    else:
        for name in get_global_factors(
            model,
            layer_idx=layer_idx,
            n_eff_parents_min=n_eff_parents_min,
            exclude_technical=exclude_technical,
        ):
            resolved, unknown = _resolve_factor_obs_names(model, [name])
            if unknown:
                continue
            global_factor_rows.extend(resolved)

    if len(global_factor_rows) > 0:
        model.adata.uns["factor_obs"].loc[global_factor_rows, "global"] = True

    complete_hierarchy: Dict[str, Sequence[str]] = {}
    pmeans = getattr(model, "pmeans", None)
    if isinstance(pmeans, dict):
        can_build = True
        for layer_idx in range(model.n_layers - 1):
            key = f"{model.layer_names[layer_idx + 1]}W"
            if key not in pmeans:
                can_build = False
                break
        if can_build:
            complete_hierarchy = get_hierarchy(model, simplified=False)

    for factor, children in complete_hierarchy.items():
        if len(children) == 0:
            continue
        if all(
            bool(model.adata.uns["factor_obs"].loc[child, "global"])
            for child in children
            if child in model.adata.uns["factor_obs"].index
        ):
            if factor in model.adata.uns["factor_obs"].index:
                model.adata.uns["factor_obs"].loc[factor, "global"] = True


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


def get_global_signature(
    model: "scDEF", top_genes: int = 10, return_scores: bool = False
) -> Union[List[str], Tuple[List[str], np.ndarray]]:
    """Consensus gene signature over global layer-0 factors.

    Requires :func:`make_global_hierarchy` (or :func:`make_hierarchies`) to have
    been run so ``model.adata.uns['global_hierarchy']`` exists.

    Args:
        model: scDEF model instance
        top_genes: number of top genes to return
        return_scores: if True, also return consensus scores

    Returns:
        Gene list, or ``(genes, scores)`` when ``return_scores=True``.
    """
    hierarchy = model.adata.uns["global_hierarchy"]
    gene_rankings, gene_scores = model.get_rankings(
        layer_idx=0,
        genes=True,
        return_scores=True,
    )

    var_names = np.array(model.adata.var_names)
    n_factors = len(gene_scores)
    gene_scores_ordered = []
    for i in range(n_factors):
        ranking = np.array(gene_rankings[i])
        scores = np.array(gene_scores[i])
        scores_full = np.full(len(var_names), np.nan)
        mask = np.in1d(var_names, ranking)
        scores_full[mask] = scores[
            [np.where(ranking == g)[0][0] for g in var_names[mask]]
        ]
        scores_full = np.nan_to_num(scores_full, nan=0)
        gene_scores_ordered.append(scores_full)
    gene_scores = np.array(
        [s / np.max(s) if np.max(s) > 0 else s for s in gene_scores_ordered]
    )

    relevances = model.get_relevances_dict()
    children = hierarchy["global_top"]
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


_SCANPY_GRAPH_UNS_KEYS = ("neighbors", "umap")


def _snapshot_scanpy_umap_obsm(adata) -> Optional[np.ndarray]:
    """Copy ``adata.obsm['X_umap']`` if present, else ``None``."""
    if "X_umap" not in adata.obsm:
        return None
    return np.asarray(adata.obsm["X_umap"], dtype=float).copy()


def _restore_scanpy_umap_obsm(adata, snapshot: Optional[np.ndarray]) -> None:
    """Restore or remove scanpy's shared ``X_umap`` slot."""
    if snapshot is None:
        if "X_umap" in adata.obsm:
            del adata.obsm["X_umap"]
    else:
        adata.obsm["X_umap"] = snapshot


def _snapshot_scanpy_graph_state(adata) -> Dict[str, Any]:
    """Snapshot scanpy slots touched by ``pp.neighbors`` / ``tl.umap``."""
    return {
        "x_umap": _snapshot_scanpy_umap_obsm(adata),
        "uns": {
            key: copy.deepcopy(adata.uns[key])
            for key in _SCANPY_GRAPH_UNS_KEYS
            if key in adata.uns
        },
        "obsp_keys": list(adata.obsp.keys()),
        "obsp": {key: adata.obsp[key].copy() for key in adata.obsp.keys()},
    }


def _restore_scanpy_graph_state(adata, snapshot: Dict[str, Any]) -> None:
    """Restore ``X_umap``, ``uns['neighbors']`` / ``uns['umap']``, and ``obsp``."""
    _restore_scanpy_umap_obsm(adata, snapshot["x_umap"])

    snapshot_obsp_keys = set(snapshot["obsp_keys"])
    for key in list(adata.obsp.keys()):
        if key not in snapshot_obsp_keys:
            del adata.obsp[key]
    for key in snapshot_obsp_keys:
        adata.obsp[key] = snapshot["obsp"][key]

    for key in _SCANPY_GRAPH_UNS_KEYS:
        if key in snapshot["uns"]:
            adata.uns[key] = copy.deepcopy(snapshot["uns"][key])
        elif key in adata.uns:
            del adata.uns[key]


def umap(
    model: "scDEF",
    layers: Optional[List[int]] = None,
    use_log: bool = False,
    metric: str = "euclidean",
) -> None:
    """Compute UMAP embeddings for each scDEF layer.

    The resulting embeddings are stored in
    ``model.adata.obsm[f"X_umap_{layer_name}"]`` for each layer. Any pre-existing
    ``adata.obsm['X_umap']``, ``adata.uns['neighbors']``, ``adata.uns['umap']``,
    and ``adata.obsp`` neighbor graphs are restored afterward (removed if absent),
    so generic scanpy calls keep using the original embedding and graph.

    Args:
        model: scDEF model instance
        layers: which layers to compute UMAPs for, in processing order. If
            None, all layers with more than one factor are used, coarse-to-fine
            (descending layer index).
        use_log: whether to use log-transformed cell-factor weights for
            the neighbor graph computation.
        metric: distance metric for neighbors computation.

    When a corrected layer-0 representation is already present in
    ``adata.obsm['X_L0_batch_corrected']`` (written by
    :func:`factor_batch_correction`), it is embedded as well and stored
    as ``adata.obsm['X_umap_L0_corrected']``, in addition to the per-layer
    embeddings. This function never *builds* that representation.
    """
    if layers is None:
        layers = [
            i
            for i in range(model.n_layers - 1, -1, -1)
            if len(model.factor_lists[i]) > 1
        ]
    else:
        layers = list(layers)

    graph_snapshot = _snapshot_scanpy_graph_state(model.adata)

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

    corrected_key = "X_L0_batch_corrected"
    if corrected_key in model.adata.obsm:
        # Embed the stored corrected representation; never build it here.
        # Same recipe as the per-layer loop above, on the merged matrix.
        model.adata.obsm[f"{corrected_key}_log"] = np.log(
            model.adata.obsm[corrected_key]
        )
        if use_log:
            sc.pp.neighbors(model.adata, use_rep=f"{corrected_key}_log")
        else:
            sc.pp.neighbors(
                model.adata,
                use_rep=corrected_key,
                metric=metric,
            )
        sc.tl.umap(model.adata)
        model.adata.obsm["X_umap_L0_corrected"] = model.adata.obsm["X_umap"].copy()

    _restore_scanpy_graph_state(model.adata, graph_snapshot)


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
    stem-like cells with diffuse scores sit in their own regime.
    Contrast with ``umap`` (per-layer), which only captures a single
    scale.

    By default it uses raw per-layer scores as stored in ``X_{layer}``.
    Set ``normalize_per_layer=True`` to convert each layer's row to
    proportions before optional log/weighting; this makes geometry
    reflect relative composition at each layer rather than absolute
    score scale.

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
            score magnitude.
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
