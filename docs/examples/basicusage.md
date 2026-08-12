# Basic usage

scDEF learns a multi-layer matrix factorization of single-cell counts, where each
layer describes the same cells at a coarser resolution: gene programs at layer 0,
progressively broader ones above. From one fit you get a hierarchy of cell states,
a gene signature for each factor, and a low-dimensional representation per layer.

The input is an [AnnData](https://anndata.readthedocs.io/en/latest/) object with
**raw UMI counts** — not normalized or log-transformed — and, ideally, filtered to
highly variable genes. With scDEF [installed](../installation.md):

```python
import scdef as scd

model = scd.scDEF(your_anndata_object, counts_layer='counts')
model.fit()
```

That is the whole fit. The number of factors is chosen automatically: layer 0
starts from `n_factors=100`, and the Automatic and Biological Relevance
Determination priors switch off the ones that are unused or not sparse enough to
be a plausible gene program. When `fit()` returns, the results are written onto
`model.adata` — the model copies the AnnData you pass it, so your own object is
left untouched. See the workflow page for the full list of keys.

Fitting prunes the upper layers only; trimming layer 0 is a deliberate step you
take after looking at the diagnostics, which is what `scd.tl.filter` below does.

From there, the common things to do are:

```python
scd.pl.qc(model)                          # check the fit converged
scd.tl.filter(model)                      # drop factors that fail the diagnostics
scd.pl.make_graph(model, show_signatures=True)   # the hierarchy, with its genes
scd.tl.assign_confident(model)            # assign cells at the resolution they support
scd.tl.umap(model)                        # embed each layer
scd.pl.umap(model)
```

scDEF also handles several things that usually need separate tools: flagging
technical factors, integrating batches while keeping batch-specific biology,
trajectories across the hierarchy, and guiding the factorization with known
marker genes.

## Where to go next

The [Analysis workflow](../workflow.md) is the full reference: input requirements,
training parameters, how to check a fit, what each step needs, and the complete
end-to-end skeleton. Start there when applying scDEF to your own data.

The example notebooks work it through on real data:
[Getting started with 3k PBMCs](scdef-pbmcs3k.ipynb) for hierarchical signatures
and clustering in a single batch,
[Cell type hierarchies in a whole adult animal](scdef-planaria.ipynb) for a deep
hierarchy, [Integrating two batches of PBMCs](scdef-pbmcs-2batches.ipynb) where
the batch difference is mostly technical, and
[Identifying interferon-response between two batches of PBMCs](scdef-ifn.ipynb)
where it is mostly biological.
