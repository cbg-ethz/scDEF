# scDEF
Extract hierarchical signatures of cell state from single-cell data using a deep exponential family model and variational inference.

# Installation
```
$ pip install scdef
```

# Usage
## Command line
`scdef` can be used from the command line:
```
$ scdef data.h5
```
This will produce the following files:
* `scdef.pkl`: file containing the `scDEF` object;
* `scdef_data.h5`: file containing the `AnnData` object with annotations obtained by `scDEF`;
* `graph.pdf`: the `scDEF` graph containing the learned factors and their hierarchical groupings.

Full overview:
```
$ scdef --help
```

## Python API
Alternatively, it is possible to use it in code:
```python
>>> from scdef import scDEF
>>> scdpf = scDPF(raw_adata, n_factors=10, n_hfactors=3, shape=1.)
>>> elbos = scdpf.optimize(n_epochs=2000, batch_size=100, step_size=.01, num_samples=100)
>>> scdpf.get_graph(enrichments=None, ard_filter=[.001, .001], top=10)
```
