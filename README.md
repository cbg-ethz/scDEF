# scDEF

[![pypi](https://img.shields.io/pypi/v/scdef.svg?style=flat)](https://pypi.python.org/pypi/scdef)
[![build](https://github.com/pedrofale/scdef/actions/workflows/main.yaml/badge.svg)](https://github.com/pedrofale/scdef/actions/workflows/main.yaml)

Extract hierarchical signatures of cell state from single-cell data using a deep exponential family model.

## Installation
```
$ pip install scdef
```

## Usage
### Command line
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

### Python API
Alternatively, it is possible to use it in code:
```python
>>> from scdef import scDEF
>>> model = scDEF(raw_adata)
>>> elbos = model.optimize()
>>> model.get_summary()
Found 10 factors grouped in the following way:
Group 0:
    Factor 0: FTH1 (1.065), RPS27A (1.035), LYZ (0.864), S100A9 (0.833), CD74 (0.752)
    Factor 1: RPS27A (1.105), LYZ (1.067), FTH1 (1.000), RPS3A (0.848), CD74 (0.717)
    Factor 2: RPS27A (1.150), FTH1 (1.083), LYZ (1.052), RPS3A (0.766), S100A9 (0.731)
    Factor 3: FTH1 (1.006), RPS27A (0.964), LYZ (0.945), RPS3A (0.895), S100A9 (0.749)
    Factor 4: RPS27A (1.049), LYZ (1.023), RPS3A (0.936), FTH1 (0.880), S100A9 (0.830)

Group 1:
    Factor 0: FTH1 (1.128), RPS27A (1.090), RPS3A (0.902), LYZ (0.882), S100A9 (0.775)
    Factor 1: RPS27A (1.192), LYZ (1.053), FTH1 (0.989), RPS3A (0.921), HLA-DRA (0.837)

Group 2:
    Factor 0: RPS27A (1.175), LYZ (1.173), FTH1 (0.938), RPS3A (0.874), S100A9 (0.750)
    Factor 1: RPS27A (1.068), LYZ (1.028), RPS3A (0.937), FTH1 (0.829), S100A9 (0.813)
    Factor 2: FTH1 (1.042), RPS27A (1.023), LYZ (0.982), RPS3A (0.929), CD74 (0.774)

>>> model.get_graph()
```
