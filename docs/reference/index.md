# API Reference

scDEF's Python API has three parts: the **model classes** you fit, the **tools**
(`scdef.tl`) that compute and store derived quantities on the fitted model, and the
**plotting** functions (`scdef.pl`) that draw them. Import the package as
`import scdef as scd`.

Most tools follow a `set_*` / `get_*` convention: the `set_*` function computes
something and stores it on `model.adata`, and the matching `get_*` reads it back.
Plotting functions generally expect the corresponding tool to have run first.

For how these fit together end to end, see the [Analysis workflow](../workflow.md);
for a short introduction, see [Basic usage](../examples/basicusage.md).

## Models

The model classes. `scDEF` is the unsupervised model; the other two add prior
information about cell identity.

| | |
|---|---|
| [`scdef.scDEF`](scDEF.md) | Single-cell Deep Exponential Families (scDEF) model. |
| [`scdef.iscDEF`](iscDEF.md) | Informed Single-cell Deep Exponential Families (iscDEF) model. |
| [`scdef.sscDEF`](sscDEF.md) | **Experimental.** Supervised scDEF: cell-type labels fix the top hierarchy. Not validated on real data. |

## Model construction

These build a new `scDEF` from an existing one — or, for `from_hierarchy`, from a
hierarchy that need not come from a fitted model. Each is also available as a
classmethod on `scDEF` (for example `scdef.scDEF.decompose_batch_effects`), which
forwards to the function documented here.

| | |
|---|---|
| [`scdef.from_reference`](from_reference.md) | Create a new model initialized from a fitted reference hierarchy. |
| [`scdef.add_batch_correction`](add_batch_correction.md) | Warm-start a batch-corrected model from a fitted hierarchy. |
| [`scdef.decompose_batch_effects`](decompose_batch_effects.md) | Re-learn lower layers under a frozen upper hierarchy to discover batch programs. |
| [`scdef.from_hierarchy`](from_hierarchy.md) | Create a model for new data initialized from a learned hierarchy. |

## Tools

### Factors and filtering

Inspect the fitted factors, and prune the ones that are not worth keeping.
[`scdef.tl.filter`](tl.filter.md) is the entry point: it drops factors *and* refreshes
the diagnostics and signatures that depend on them, which is why it should be
preferred over calling the model's own `filter_factors` directly.

| | |
|---|---|
| [`scdef.tl.filter`](tl.filter.md) | Filter factors and refresh the diagnostics and signatures in one step. |
| [`scdef.tl.factor_diagnostics`](tl.factor_diagnostics.md) | Compute/store factor diagnostics in `model.adata.uns['factor_obs']`. |
| [`scdef.tl.annotate_factors`](tl.annotate_factors.md) | Attach descriptive annotations to factors in `adata.uns['factor_obs']`. |
| [`scdef.tl.get_factor_annotations`](tl.get_factor_annotations.md) | Look up `factor_obs['annotation']` values for factor names. |
| [`scdef.tl.drop_technical`](tl.drop_technical.md) | Remove factors marked `technical` from `factor_lists` and re-annotate. |
| [`scdef.tl.assign_confident`](tl.assign_confident.md) | Pick the finest scDEF layer at which each cell is confidently assigned. |
| [`scdef.tl.set_cell_entropies`](tl.set_cell_entropies.md) | Compute per-cell assignment entropy and store one column per layer. |
| [`scdef.tl.compute_within_group_pairwise_dissimilarity`](tl.compute_within_group_pairwise_dissimilarity.md) | Compute within-group pairwise cell dissimilarity for one layer. |
| [`scdef.tl.get_obs_score_rankings`](tl.get_obs_score_rankings.md) | Return per-obs-value factor rankings by observation association score. |
| [`scdef.tl.get_obs_value_specific_factors`](tl.get_obs_value_specific_factors.md) | Get factors specific to each obs value in a layer. |

### Gene signatures

Gene signatures per factor. scDEF can call a signature statistically — keeping the
genes whose posterior loading exceeds a threshold with a given confidence — rather
than simply taking the top N.

| | |
|---|---|
| [`scdef.tl.set_factor_signatures`](tl.set_factor_signatures.md) | Store a signature per factor in `adata.uns['factor_signatures']`. |
| [`scdef.tl.set_confident_signatures`](tl.set_confident_signatures.md) | Precompute and cache confident signatures/scores for all layers. |
| [`scdef.tl.get_confident_signatures`](tl.get_confident_signatures.md) | Get confidence-based signatures per factor using posterior mean/variance. |
| [`scdef.tl.get_stored_confident_signatures`](tl.get_stored_confident_signatures.md) | Load precomputed confident signatures (and optional scores) from cache. |
| [`scdef.tl.get_biological_signature`](tl.get_biological_signature.md) | Gene signature of the top-layer factor — the programme every cell shares. |
| [`scdef.tl.get_technical_signature`](tl.get_technical_signature.md) | Consensus gene signature over the factors flagged as technical. |
| [`scdef.tl.get_global_signature`](tl.get_global_signature.md) | Consensus gene signature over global layer-0 factors. |
| [`scdef.tl.gsea`](tl.gsea.md) | Run Enrichr pathway enrichment for cached signatures across layers. |

### Batch and technical structure

Identify factors that track batch or other technical variation, and correct for them.
The batch workflow is a two-step process described in the
[Analysis workflow](../workflow.md).

| | |
|---|---|
| [`scdef.tl.batch_structure_report`](tl.batch_structure_report.md) | Describe how batch structure appears among the layer-0 factors. |
| [`scdef.tl.set_technical_factors`](tl.set_technical_factors.md) | Set the technical factors of the model. |
| [`scdef.tl.get_technical_factors`](tl.get_technical_factors.md) | Current model names of the factors marked technical in `factor_obs`. |
| [`scdef.tl.set_batch_technical_factors`](tl.set_batch_technical_factors.md) | Mark layer-0 factors as batch-technical (per-batch splits of one type). |
| [`scdef.tl.get_batch_technical_factors`](tl.get_batch_technical_factors.md) | Current model names of the factors marked `batch_technical`. |
| [`scdef.tl.factor_batch_correction`](tl.factor_batch_correction.md) | Apply the batch-technical correction to the scores and to the labels. |
| [`scdef.tl.set_global_factors`](tl.set_global_factors.md) | Mark global (shared-across-lineages) factors in `factor_obs`. |
| [`scdef.tl.get_batch_specific_genes_from_gene_scale`](tl.get_batch_specific_genes_from_gene_scale.md) | Per-gene log-ratios of batch-specific `gene_scale` vs a reference profile. |
| [`scdef.tl.get_factor_batch_gene_scale_affinity`](tl.get_factor_batch_gene_scale_affinity.md) | Score each layer-0 factor by its affinity for the reference fit's per-batch `gene_scale` contrast. |

### Hierarchy

Extract and score the hierarchy that relates factors across layers.

| | |
|---|---|
| [`scdef.tl.get_hierarchy`](tl.get_hierarchy.md) | Get a dictionary containing the polytree contained in the scDEF graph. |
| [`scdef.tl.make_hierarchies`](tl.make_hierarchies.md) | Store the biological, technical, and global hierarchies of the model. |
| [`scdef.tl.make_biological_hierarchy`](tl.make_biological_hierarchy.md) | Make the biological hierarchy of the model. |
| [`scdef.tl.make_technical_hierarchy`](tl.make_technical_hierarchy.md) | Make the technical hierarchy of the model. |
| [`scdef.tl.make_global_hierarchy`](tl.make_global_hierarchy.md) | Make the global (shared-across-lineages) hierarchy of the model. |
| [`scdef.tl.compute_hierarchy_scores`](tl.compute_hierarchy_scores.md) | Compute per-factor and global hierarchy scores from learned W matrices. |
| [`scdef.tl.add_l0_lineage_aggregate_scores`](tl.add_l0_lineage_aggregate_scores.md) | Add lineage-averaged clarity and effective parents for layer-0 factors only. |
| [`scdef.tl.find_sensible_top_layer`](tl.find_sensible_top_layer.md) | Find the coarsest hierarchy layer supported by confident merges. |
| [`scdef.tl.find_sensible_top_factors`](tl.find_sensible_top_factors.md) | Return factors marked as sensible top factors in `factor_obs`. |
| [`scdef.tl.get_lineage_factors`](tl.get_lineage_factors.md) | Return factors at `layer_idx` that are clear descendants of `top_factor_label`. |
| [`scdef.tl.get_global_factors`](tl.get_global_factors.md) | Return the factors at `layer_idx` shared across lineages (high effective parents). |

### Trajectories

Build and score paths through the hierarchy.

| | |
|---|---|
| [`scdef.tl.multilevel_paga`](tl.multilevel_paga.md) | Compute and cache multilevel PAGA results for plotting. |
| [`scdef.tl.build_differentiation_paths`](tl.build_differentiation_paths.md) | Build hierarchy-consistent differentiation paths (top->leaf). |
| [`scdef.tl.build_transition_paths`](tl.build_transition_paths.md) | Build transition paths on a soft inter-layer factor graph. |
| [`scdef.tl.score_paths`](tl.score_paths.md) | Score per-cell position and affinity on previously built paths. |

### Embeddings

| | |
|---|---|
| [`scdef.tl.umap`](tl.umap.md) | Compute UMAP embeddings for each scDEF layer. |
| [`scdef.tl.multilayer_umap`](tl.multilayer_umap.md) | Compute a single UMAP from the concatenation of all scDEF layers. |

## Plotting

### Hierarchy graphs

Graphviz renderings of the factor hierarchy.

| | |
|---|---|
| [`scdef.pl.make_graph`](pl.make_graph.md) | Make Graphviz-formatted scDEF graph. |
| [`scdef.pl.biological_hierarchy`](pl.biological_hierarchy.md) | Plot the biological hierarchy of the model. |
| [`scdef.pl.technical_hierarchy`](pl.technical_hierarchy.md) | Plot the technical hierarchy of the model. |
| [`scdef.pl.global_hierarchy`](pl.global_hierarchy.md) | Plot the global (shared-across-lineages) factor hierarchy. |

### Factors and signatures

| | |
|---|---|
| [`scdef.pl.obs_cell_factor_heatmap`](pl.obs_cell_factor_heatmap.md) | Heatmap of per-cell factor scores or probabilities for one or more obs subsets. |
| [`scdef.pl.obs_factor_dotplot`](pl.obs_factor_dotplot.md) | Plot dotplot showing factor assignments for observations. |
| [`scdef.pl.factor_genes`](pl.factor_genes.md) | Plot number of genes in factors across layers. |
| [`scdef.pl.factor_gene_uncertainty_boxplot`](pl.factor_gene_uncertainty_boxplot.md) | Plot per-gene uncertainty boxes for a factor using posterior mean/variance. |
| [`scdef.pl.factors_bars`](pl.factors_bars.md) | Plot factor scores as bar charts. |
| [`scdef.pl.signatures_scores`](pl.signatures_scores.md) | Plot the association between a set of cell annotations and a set of gene signatures. |
| [`scdef.pl.pathway_scores`](pl.pathway_scores.md) | Plot the association between the factors and a set of pathways. |
| [`scdef.pl.obs_scores`](pl.obs_scores.md) | Plot the association between a set of cell annotations and factors. |
| [`scdef.pl.continuous_obs_scores`](pl.continuous_obs_scores.md) | Plot the correlations between a set of cell annotations and factors. |
| [`scdef.pl.layers_obs`](pl.layers_obs.md) | Plot observation matrices across layers. |
| [`scdef.pl.umap`](pl.umap.md) | Plot pre-computed UMAPs for different layers. |
| [`scdef.pl.cell_entropies`](pl.cell_entropies.md) | Plot cell entropies and factor numbers across layers. |
| [`scdef.pl.within_group_pairwise_dissimilarity`](pl.within_group_pairwise_dissimilarity.md) | Plot within-group pairwise dissimilarity distributions. |
| [`scdef.pl.factor_gini`](pl.factor_gini.md) | Plot Gini coefficient for a specific factor. |

### Trajectories

| | |
|---|---|
| [`scdef.pl.multilevel_paga`](pl.multilevel_paga.md) | Plot cached multilevel PAGA graphs across scDEF layers. |
| [`scdef.pl.trajectory_heatmap`](pl.trajectory_heatmap.md) | Plot stacked trajectory heatmap from precomputed confident signatures. |
| [`scdef.pl.path_trajectory_heatmap`](pl.path_trajectory_heatmap.md) | Trajectory heatmap along a stored multi-layer path (differentiation or transition). |
| [`scdef.pl.path_embedding`](pl.path_embedding.md) | Plot cells on an embedding colored by position along one path. |

### Model diagnostics

Check whether a fit is healthy before interpreting it.

| | |
|---|---|
| [`scdef.pl.qc`](pl.qc.md) | Plot QC metrics for scDEF run. |
| [`scdef.pl.loss`](pl.loss.md) | Plot training loss over epochs. |
| [`scdef.pl.scales`](pl.scales.md) | Plot both cell and gene scales. |
| [`scdef.pl.scale`](pl.scale.md) | Plot learned scale factors vs observed scales. |
| [`scdef.pl.relevance`](pl.relevance.md) | Plot relevance determination scores. |
| [`scdef.pl.gini_brd`](pl.gini_brd.md) | Plot Gini coefficient vs BRD scores. |
| [`scdef.pl.factor_diagnostics`](pl.factor_diagnostics.md) | Diagnostic scatter plot of layer-0 factors with flexible axis/color/size mapping. |
