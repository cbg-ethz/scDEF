<div align="left">
  <img src="https://github.com/cbg-ethz/scDEF/raw/main/figures/scdef.png", width="300px">
</div>
<p></p>

[![pypi](https://img.shields.io/pypi/v/scdef.svg?style=flat)](https://pypi.python.org/pypi/scdef)
[![build](https://github.com/cbg-ethz/scDEF/actions/workflows/main.yaml/badge.svg)](https://github.com/cbg-ethz/scDEF/actions/workflows/main.yaml)

Deep exponential families for single-cell data. scDEF learns hierarchies of cell states and their gene signatures from scRNA-seq data. The method can be used for dimensionality reduction, visualization, gene signature identification, clustering at multiple levels of resolution, and batch integration. The informed version (iscDEF) can additionally take known gene lists to jointly assign cells to types and find clusters within each type.

## Installation
```
pip install scdef
```

Please be sure to install a version of [JAX](https://jax.readthedocs.io/) that is compatible with your GPU (if applicable).

## Getting started
scDEF takes as input an [AnnData](https://anndata.readthedocs.io/) object containing UMI counts. The [notebooks](https://github.com/cbg-ethz/scDEF/tree/main/notebooks) directory contains IPython notebooks with examples showcasing the analyses enabled by scDEF.
