# Analysis workflow

The reference workflow for applying scDEF to a data set. Every step states what
it needs and what it writes, so the steps can be reordered or skipped as long as
the prerequisites hold. For a short introduction see
[Basic usage](examples/basicusage.md); for the objects themselves see the
[API reference](reference/index.md).

If you would rather start from working code, the
[reference workflow](#reference-workflow) at the end of this page is an
end-to-end skeleton you can copy and adapt.

## Input requirements

scDEF models raw UMI counts directly, so the input must be counts and **must not
be normalized or log-transformed**.

- Put the raw counts in a layer and name it: `scd.scDEF(adata, counts_layer='counts')`.
  Without `counts_layer`, `adata.X` is used and must itself hold counts.
- Filtering to highly variable genes is strongly recommended. Between 2000 and
  4000 genes is typical, selected on the counts:

```python
adata.layers['counts'] = adata.X.copy()
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', layer='counts', n_top_genes=4000)
adata = adata[:, adata.var.highly_variable].copy()
```

- Any marker genes you intend to check should be added back after HVG selection,
  since selection may drop them.
- Cell and gene QC filtering (minimum counts, mitochondrial fraction) is the
  user's responsibility and should happen before fitting.

## Fitting

```python
model = scd.scDEF(adata, counts_layer='counts')
model.fit()
```

Construction defaults: `n_factors=100` at layer 0, `n_layers=6`, `seed=42`.
`layer_sizes` overrides the geometric schedule with explicit per-layer sizes.
The number of *used* factors is not `n_factors` — the relevance priors switch
most of them off, and the explicit filtering step below removes them.

Training defaults, all passed to `fit()`: `n_epoch=1000`, `lr=0.1`,
`batch_size=256`, `num_samples=100`. `n_epoch` is a **maximum per round**:
training stops early once the *relative* improvement over the best loss so far
stays below `tolerance=1e-5` for `patience=50` consecutive epochs, and never
before `min_epochs=50`.

`fit()` annotates the model when it finishes, so the cell scores are in
`model.adata.obsm['X_L0']`, `['X_L1']`, … with no separate annotate step.

It also filters, but **only the upper layers**: `filter=True` calls
`filter_factors(upper_only=True)`, which keeps every layer-0 factor. Pruning
layer 0 — the layer whose factors you actually interpret — is a separate,
deliberate step, so that the thresholds can be chosen after looking at the
diagnostics. That is what the loop below is for.

### Fit, filter, re-fit

One fit is rarely the end. The usual loop is to fit, look at the diagnostics,
filter the factors that do not hold up, and **fit again** — the model
re-initializes from the current posterior and the surviving `factor_lists`, so
the second fit refines a smaller, cleaner hierarchy rather than starting over:

```python
model.fit()
scd.pl.qc(model)

scd.tl.factor_diagnostics(model)
scd.pl.factor_diagnostics(model, brd_min=1.0, ard_min=1e-3, clarity_min=0.5,
                          annotate_factors=True)     # inspect before cutting
scd.tl.filter(model, brd_min=1.0, ard_min=1e-3, clarity_min=0.5)

model.fit()                                          # re-fit the reduced hierarchy
scd.pl.qc(model)
```

Use [`scd.tl.filter`][scdef.tl.filter] rather than
`model.filter_factors` directly. Filtering renames factors, which invalidates
everything keyed by those names — stored signatures, the hierarchies, and the
frozen upper-layer subset the diagnostics use. `scd.tl.filter` filters *and*
recomputes the diagnostics, so the model is immediately usable; calling
`model.filter_factors` on its own leaves it in a state where, for example,
`scd.pl.make_graph(model, show_signatures=True)` raises.

Those three numbers are the defaults, so the first pass can simply call
`scd.tl.filter(model)`. Pass explicit values on a later pass when the
diagnostics plot shows factors you want to cut: `clarity_min` is usually the one
to raise, since factors with many effective parents are those that fail to sit
cleanly in the hierarchy. Repeat until the hierarchy stops changing.

### What a re-filter invalidates

Filtering or re-fitting renames factors, so anything keyed by factor name goes
stale. After either, re-run in this order:

```python
scd.tl.factor_diagnostics(model)      # done for you by scd.tl.filter
scd.tl.set_technical_factors(model, ...)
scd.tl.make_hierarchies(model)
```

**The refresh invariant:** run
[`scd.tl.factor_diagnostics`][scdef.tl.factor_diagnostics] after any fit, re-fit
or [`decompose_batch_effects`][scdef.decompose_batch_effects] — it recomputes the
diagnostics *and* the confident gene signatures, so there is no separate
[`set_confident_signatures`][scdef.tl.set_confident_signatures] step.
[`scd.tl.filter`][scdef.tl.filter] already bundles it.

The one case that needs a rebuild is
[`drop_technical`][scdef.tl.drop_technical], which actually removes factors and so
changes the hierarchy the upper-layer signatures were computed from. Merely
*flagging* factors with
[`set_technical_factors`][scdef.tl.set_technical_factors] does not: the cache is
per factor and the flags are applied when a consensus signature is read.

Fitting is the expensive step. Save the result rather than refitting:

```python
model.save('my_model')
model = scd.scDEF.load('my_model')   # do not pass adata; the model carries its own
```

## Checking the fit

Do this before interpreting anything.

```python
scd.pl.qc(model)
```

[`qc`][scdef.pl.qc] is the composite panel and is normally all you need: it draws
the loss curve, BRD against the Gini coefficient, learned versus observed cell
and gene scales, and the relevance distributions. [`loss`][scdef.pl.loss],
[`gini_brd`][scdef.pl.gini_brd] and [`scale`][scdef.pl.scale] draw those
individual pieces at full size.

What to look for, and what to assert on when running unattended:

- **Convergence.** If the loss is still descending at the end, the fit hit the
  `n_epoch` ceiling rather than the early-stopping rule; raise `n_epoch`.
- **Factor count.** `[len(fl) for fl in model.factor_lists]` gives the surviving
  factors per layer. Layer 0 collapsing to a handful, or keeping nearly all 100,
  both mean the relevance priors are mis-scaled for this data: raise
  `brd_strength` to prune harder, lower it to keep more.
- **Scales.** The learned cell and gene scales should track the observed library
  sizes and gene totals; a cloud with no relationship means the fit did not take.

If you are generating a notebook unattended and cannot inspect the figures, use
the defaults throughout and do not tune — they are sensible for typical data.

## What the fit writes to `model.adata`

**`scDEF` copies the AnnData at construction.** Your original object is never
touched — every result lives on `model.adata`, and any `obs` annotation you want
to use downstream (`wedged=`, `obs_keys=`, `subset_obs_key=`) must be present
*before* you construct the model.

- `model.adata.obs['L0']`, `['L1']`, … — the hard factor assignment per layer.
- `model.adata.obs['<factor>_prob']` — each cell's weight on one factor, e.g.
  `[f'{f}_prob' for f in model.factor_names[0]]`.
- `model.adata.obsm['X_L0']`, `['X_L1']`, … — the per-layer representations, and
  `X_<layer>_probs` for the normalized memberships.
- `model.adata.uns['L0_signatures']` — signatures in scanpy's
  differential-expression format, so
  `sc.pl.rank_genes_groups(model.adata, key='L0_signatures', groups=['L0_0'])`
  renders them. `model.get_signatures_dict()` returns the same as a dict.
- `model.adata.uns['factor_obs']` — the per-factor diagnostics table, and where
  the `technical` / `batch_technical` / `annotation` flags live.

These are plottable with plain `scanpy` — `sc.pl.umap(model.adata, color=['L0'])`
— **but only if you computed a scanpy UMAP yourself before fitting**.
[`scd.tl.umap`][scdef.tl.umap] will not do it for you: it writes per-layer
`X_umap_L0`, `X_umap_L1`, … and deliberately restores any pre-existing `X_umap`
so generic scanpy calls keep using your own embedding.
[`multilayer_umap`][scdef.tl.multilayer_umap] is the exception — it runs
`sc.pp.neighbors` and `sc.tl.umap` on `model.adata` directly, so it **overwrites**
`X_umap` and the neighbour graph as a side effect.

## Step prerequisites

**Hard** — these raise if the prerequisite has not run:

| Step | Requires |
| --- | --- |
| any plotting of cell scores | `fit()` (or `model.annotate_adata()`) so `X_<layer>` is in `obsm` |
| `scd.pl.factor_diagnostics` | `scd.tl.factor_diagnostics` |
| `scd.pl.make_graph(show_signatures=True)`, `scd.pl.biological_hierarchy(show_signatures=True)` | `scd.tl.factor_diagnostics` (it caches the signatures) |
| `scd.pl.biological_hierarchy`, `technical_hierarchy`, `global_hierarchy` | `scd.tl.make_hierarchies` |
| any `wedged=<key>` | `adata.uns['<key>_colors']`, written by a scanpy plotting call **before** the model is constructed |
| `scd.tl.drop_technical` | `set_technical_factors` |
| `scd.tl.get_technical_signature` | `make_technical_hierarchy` or `make_hierarchies` |
| `scd.tl.get_global_signature` | `make_global_hierarchy` or `make_hierarchies` |
| `scd.tl.get_biological_signature` | `factor_diagnostics`, plus `set_technical_factors` for the flags it applies |
| `scd.pl.trajectory_heatmap` | `scd.tl.factor_diagnostics` (it caches the signatures) |
| `scd.tl.gsea` | `scd.tl.factor_diagnostics` (it caches the signatures) |
| `scd.tl.score_paths` | `build_differentiation_paths` or `build_transition_paths` |
| `scd.pl.path_trajectory_heatmap` | `score_paths` |
| `scd.pl.path_embedding` | `score_paths`, plus `scd.tl.multilayer_umap` for the default `basis='umap_multilayer'` |
| `scd.pl.umap` | `scd.tl.umap` |
| `scd.tl.factor_batch_correction` | `set_batch_technical_factors` |
| `scd.tl.batch_structure_report` | a `batch_key` column in `adata.obs` |

**Soft** — computed for you if missing, so calling them first is optional:

| Step | Computes if absent |
| --- | --- |
| `scd.tl.set_technical_factors` | factor diagnostics |
| `scd.pl.cell_entropies` | `set_cell_entropies` |
| `scd.tl.get_obs_score_rankings` | its `uns['obs_scores']` cache |
| `scd.pl.within_group_pairwise_dissimilarity` | `compute_within_group_pairwise_dissimilarity` |

Note that `scd.pl.pathway_scores` needs **neither** — it reads the per-layer
`uns['<layer>_signatures']` that `fit()` writes.

## Gene signatures

Signature membership is a per-gene call, not a top-*N* cut:
[`get_confident_signatures`][scdef.tl.get_confident_signatures] thresholds each
factor at a high quantile of its own mean loadings and keeps genes whose
posterior probability of exceeding it clears `confidence_threshold`.
[`set_confident_signatures`][scdef.tl.set_confident_signatures] precomputes this
for every layer and caches per-gene confidences, combined scores and a per-factor
posterior stability (Jaccard across samples) that flags unstable signatures.
[`factor_gene_uncertainty_boxplot`][scdef.pl.factor_gene_uncertainty_boxplot]
shows the posterior of each gene's loading in one factor as a box — quartiles,
median and 5th/95th whiskers — so the genes that genuinely define a program
separate visually from those with a high point estimate and a wide posterior.
With `color_by_confidence=True` each box is coloured by its own
`P(W > tau)`, which puts the per-gene call and the uncertainty behind it in one
picture:

```python
scd.pl.factor_gene_uncertainty_boxplot(
    model, factor='L0_3', layer_idx=0,
    color_by_confidence=True, confidence_tau_quantile=0.99,
)
```

## Pathway enrichment

Once signatures are cached, [`gsea`][scdef.tl.gsea] runs Enrichr enrichment on
them for every layer at once and stores the result in
`adata.uns['factor_enrichments']`:

```python
scd.tl.factor_diagnostics(model)                # caches the signatures gsea tests
enrichments = scd.tl.gsea(model, libs=['KEGG_2019_Human'])
```

It tests the *confident* signatures rather than a top-*N* ranking of `W`, so a
factor with a short, well-supported signature is tested on exactly those genes.
`custom_gene_sets` merges your own sets into the same universe as the online
libraries, `background_genes` sets the background (use the HVG list, not the
whole genome, if the input was filtered), and `layers` restricts which layers
are tested. The returned frame is in Enrichr's format — one row per
factor-by-term hit, with `Term`, `Adjusted P-value`, `Odds Ratio` and the
overlapping `Genes`.

`gsea` calls Enrichr over the network, so it needs internet access and is the
slowest step here. Cache `adata.uns['factor_enrichments']` rather than re-running.

[`pathway_scores`][scdef.pl.pathway_scores] is a **separate** route and does not
consume `gsea`'s output. It expects a decoupler/PROGENy network — a long-format
frame with `source` and `target` columns — and scores it against the per-layer
signatures with over-representation analysis:

```python
import decoupler
progeny = decoupler.get_progeny(organism='human')
scd.pl.pathway_scores(model, pathways=progeny)
```

## Naming factors

Factors are `L0_0`, `L0_1`, … by default. Once you know what they are,
[`annotate_factors`][scdef.tl.annotate_factors] attaches labels that persist in
`factor_obs` and are reused by the plots:

```python
scd.tl.annotate_factors(model, {'L0_3': 'CD14 monocyte', 'L0_7': 'interferon response'})
scd.tl.get_factor_annotations(model, ['L0_3'])
```

[`get_obs_score_rankings`][scdef.tl.get_obs_score_rankings] and
[`get_obs_value_specific_factors`][scdef.tl.get_obs_value_specific_factors] help
decide those labels: the first ranks factors by association with each value of an
annotation, the second returns the factors *specific* to a value, scoring
`score(value) - max(score(others))`.

## Splitting the hierarchy

Beyond technical factors, [`set_global_factors`][scdef.tl.set_global_factors]
marks programs shared across lineages rather than confined to one — a cell-cycle
or stress program typically. With both flags set,
[`make_hierarchies`][scdef.tl.make_hierarchies] stores three views of the tree,
each with its own plot:

```python
scd.tl.set_technical_factors(model)
scd.tl.set_global_factors(model)
scd.tl.make_hierarchies(model)

scd.pl.biological_hierarchy(model)   # lineage tree, technical and global removed
scd.pl.technical_hierarchy(model)
scd.pl.global_hierarchy(model)
```

[`make_graph`][scdef.pl.make_graph] draws the full graph and takes the same
options as the hierarchy plots: `show_signatures=True` prints the top genes in
each node, `wedged='cell_type'` colours nodes by the composition of an
annotation, `n_cells_label=True` shows how many cells attach, and
`show_confidences=True` adds signature confidence. Passing
`hierarchy=model.adata.uns['biological_hierarchy']` with `top_factor=` restricts
the drawing to one lineage:

```python
scd.pl.make_graph(model, show_signatures=True, wedged='cell_type',
                  color_edges=True, n_cells_label=True)
```

To label the nodes automatically,
`scd.utils.factor_utils.assign_obs_to_factors` matches factors to the values of
an annotation and returns assignments usable as `top_factor` and
`factor_annotations`.

The corresponding consensus signatures are
[`get_biological_signature`][scdef.tl.get_biological_signature],
[`get_technical_signature`][scdef.tl.get_technical_signature] and
[`get_global_signature`][scdef.tl.get_global_signature]. The last needs
`make_global_hierarchy` (or `make_hierarchies`) to have run.

## Cells and annotations

[`obs_scores`][scdef.pl.obs_scores] associates annotations with factors;
[`signatures_scores`][scdef.pl.signatures_scores] associates marker sets with the
learned signatures. [`obs_cell_factor_heatmap`][scdef.pl.obs_cell_factor_heatmap]
draws one row per cell rather than a group mean, stacking a panel per value of
`subset_obs` over shared factor columns, which is how to see whether a population
uses one program cleanly or mixes several. Use `values='prob'` for normalized
memberships. For continuous annotations use
[`continuous_obs_scores`][scdef.pl.continuous_obs_scores], which correlates them
against the factors instead of scoring overlap, and
[`factors_bars`][scdef.pl.factors_bars] for factor scores as bar charts per
annotation.

## Embeddings

`fit()` writes one representation per layer, `X_L0`, `X_L1`, …
[`umap`][scdef.tl.umap] embeds every layer that has more than one factor,
writing `X_umap_<layer>` per layer, and [`scdef.pl.umap`][scdef.pl.umap] plots
them; pass `layers=` to restrict which. [`multilayer_umap`][scdef.tl.multilayer_umap] instead concatenates the
layers into a single representation, so one embedding reflects every resolution
at once rather than one layer at a time.

## How mixed are the cells

[`set_cell_entropies`][scdef.tl.set_cell_entropies] computes, per cell and per
layer, the Shannon entropy of its factor memberships and the implied effective
number of factors, writing two `obs` columns per layer
(`<layer>_entropy` and `<layer>_effective_n_factors`);
[`cell_entropies`][scdef.pl.cell_entropies] plots them across layers. High
entropy at every layer is the same signal that sends a cell to the root in the
confident assignment below.

To ask whether a labelled group is internally homogeneous,
[`compute_within_group_pairwise_dissimilarity`][scdef.tl.compute_within_group_pairwise_dissimilarity]
computes pairwise distances between cells of the same group in factor space
(Jensen-Shannon by default), and
[`within_group_pairwise_dissimilarity`][scdef.pl.within_group_pairwise_dissimilarity]
plots the distributions — a group that is really two states shows up as a wide or
bimodal distribution.

For the factors themselves, [`factor_genes`][scdef.pl.factor_genes] shows how many
genes each factor uses across layers and [`factor_gini`][scdef.pl.factor_gini] the
sparsity of one factor.

## Resolution and undifferentiated cells

[`assign_confident`][scdef.tl.assign_confident] picks, per cell, the finest layer
whose winning factor beats its runner-up by a margin that survives the posterior.
Cells ambiguous at every layer fall back to the root, so `confident_depth_score`
(`0` at layer 0, `1` at the root) ranks cells by how undifferentiated they are
without any labelling. Pass `exclude_technical=True` or
`exclude_batch_technical=True` to respect the flags set earlier.

## Multiple batches

Two steps. Fit with `batch_key`, then re-learn layer 0 without the per-gene batch
term to expose what that term absorbed:

```python
ref = scd.scDEF(adata, counts_layer='counts', batch_key='Experiment')
ref.fit()
dec = scd.scDEF.decompose_batch_effects(ref, top_layer=1)

rep = scd.tl.batch_structure_report(dec)
flagged = rep.index[rep['shape'].isin(['branch_split', 'branch_skewed'])]
scd.tl.set_batch_technical_factors(dec, flagged)
scd.tl.factor_batch_correction(dec)
```

[`get_batch_technical_factors`][scdef.tl.get_batch_technical_factors] reads back
what was flagged. Two gene-side views complement the report:
[`get_batch_specific_genes_from_gene_scale`][scdef.tl.get_batch_specific_genes_from_gene_scale]
gives the per-gene contrast the batch term absorbed, and
[`get_factor_batch_gene_scale_affinity`][scdef.tl.get_factor_batch_gene_scale_affinity]
scores which decomposed factors look like it.

`ref` is the representation to use downstream; `dec` is the diagnostic. The
report describes geometry, not cause — the same columns rank a stress program
highest on two runs of one donor and the interferon response highest on
stimulated versus control cells, so which factors to flag is a decision from the
experimental design, not a threshold. See
[`batch_structure_report`][scdef.tl.batch_structure_report] for the columns and
[`factor_batch_correction`][scdef.tl.factor_batch_correction] for what the
correction does to the scores and labels.

### Interpreting layer 0 of the decomposed model

Decomposing and then reading layer 0 is the **standard step for any multi-batch
data set**, not a special-case diagnostic. The decomposed L0 is where the batch
term's contents become visible as ordinary factors, so each one can be judged on
what it is rather than removed wholesale.

Every layer-0 factor of `dec` falls into one of three cases:

| What you see | What it is | What to do |
| --- | --- | --- |
| One factor per batch splitting a single population that the corrected parent already represents | A per-batch split — technical | [`set_batch_technical_factors`][scdef.tl.set_batch_technical_factors], then [`factor_batch_correction`][scdef.tl.factor_batch_correction] |
| A factor loading across all cell types in one batch — ambient RNA, dissociation stress, a depth artefact | Global or technical | [`set_technical_factors`][scdef.tl.set_technical_factors] or [`set_global_factors`][scdef.tl.set_global_factors] |
| A factor confined to particular cell types and coherent with a treatment or condition | Biology that happens to align with a batch | **Keep it** — this is the finding, not an artefact |

The third row is why this cannot be automated. The same procedure gives opposite
answers on the two tutorial data sets: in
[Integrating two batches of PBMCs](examples/scdef-pbmcs-2batches.ipynb) the
batch-aligned factors are technical and get flagged, while in
[Identifying signatures of interferon-response in PBMCs](examples/scdef-ifn.ipynb)
the batch-aligned factor is the interferon response — the result the experiment
was run to find — and flagging it would delete the answer.

The geometry alone does not separate the two. Use the experimental design:
a per-batch split of one population is technical; a coherent, cell-type-restricted
programme that tracks a *condition* is biological. `eff_parents` from the
diagnostics helps here — a programme shared across lineages reads differently
from one confined to a branch.

## Trajectories

```python
scd.tl.multilevel_paga(model, neighbors_rep='X_L0')
scd.pl.multilevel_paga(model)

scd.tl.build_differentiation_paths(model)          # top-to-leaf, uns['differentiation_paths']
scd.tl.score_paths(model, paths_key='differentiation_paths')
scd.pl.path_embedding(model, paths_key='differentiation_paths')
```

[`build_transition_paths`][scdef.tl.build_transition_paths] is the alternative
that also allows moves between branches, in `uns['transition_paths']`. Path ids
are indices into the stored list, so check
`model.adata.uns['differentiation_paths']` before selecting one.

## Known markers

[`iscDEF`][scdef.iscDEF] shapes the layer-0 loading prior with marker gene sets,
supplied as `{name: [genes]}`. `markers_layer=0` enforces one factor per set at
the finest layer and learns the structure above; `markers_layer > 0` puts the
sets at a coarse layer and resolves finer states within each. Raise
`gs_big_scale` and `marker_strength` to stay close to the lists, lower them to
let the data add genes, and set `add_other` when the sets are not expected to
cover every cell.

## Reference workflow

An end-to-end skeleton. Later blocks assume the earlier ones have run.

`CELLTYPE_KEY` stands for your own annotation column; drop those arguments
entirely if the data is unannotated.

```python
import scanpy as sc
import scdef as scd

CELLTYPE_KEY = 'celltypes'      # your annotation column, or None

# 1. input: QC'd, raw counts kept in a layer, HVG-filtered
adata.layers['counts'] = adata.X.copy()
sc.pp.highly_variable_genes(adata, flavor='seurat_v3', layer='counts', n_top_genes=4000)
adata = adata[:, adata.var.highly_variable].copy()

# a scanpy embedding, if you want to plot scDEF's obs columns with sc.pl.umap
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=CELLTYPE_KEY)   # also writes uns[f'{CELLTYPE_KEY}_colors'],
                                        # which `wedged=` below requires

# 2. fit. Annotations and their colors must already be on `adata`: the model copies it.
#    See "Multiple batches" for batch_key and "Known markers" for iscDEF.
model = scd.scDEF(adata, counts_layer='counts')
print(model)                                   # layer sizes sanity check
model.fit()
scd.pl.qc(model)

# 3. diagnose, filter, re-fit
scd.tl.factor_diagnostics(model)               # must precede the diagnostics plot
scd.pl.factor_diagnostics(model, annotate_factors=True)
scd.tl.filter(model)                           # filters AND refreshes diagnostics
model.fit()
scd.pl.qc(model)
print([len(fl) for fl in model.factor_lists])  # factors surviving per layer

# 4. diagnostics + statistically called gene signatures, cached for all layers
scd.tl.factor_diagnostics(model)               # also refreshes the signatures

# 5. flag technical factors: names read off the diagnostics plot, or by criteria
scd.tl.set_technical_factors(model)

# 6. split the hierarchy into its biological / technical / global views
scd.tl.make_hierarchies(model)
scd.tl.get_technical_signature(model, top_genes=10)   # needs the technical hierarchy
scd.pl.biological_hierarchy(model, show_signatures=True, wedged=CELLTYPE_KEY)

# 7. assign each cell at the finest layer it is confident at
scd.tl.assign_confident(model, exclude_technical=True)

# 8. inspect
scd.pl.make_graph(model, show_signatures=True, wedged=CELLTYPE_KEY)
scd.pl.obs_scores(model, obs_keys=[CELLTYPE_KEY], mode='weights')
scd.tl.umap(model)
scd.pl.umap(model, color=CELLTYPE_KEY)

# 9. if marker sets are known, check the signatures recovered them
scd.pl.signatures_scores(model, CELLTYPE_KEY, markers, top_genes=20)
```

The ordering above is load-bearing: the diagnostics plot and
`show_signatures=True` both need `factor_diagnostics` to have run, since that is
what caches the confident signatures, and `get_technical_signature` needs
`make_hierarchies`. Flagging factors with `set_technical_factors` does *not*
invalidate the signatures — only `drop_technical`, which removes factors
outright, requires `factor_diagnostics` to be run again.
