<div align="left">
  <img src="https://github.com/cbg-ethz/scDEF/raw/main/docs/assets/images/scdef.png", width="300px">
</div>
<p></p>

[![pypi](https://img.shields.io/pypi/v/scdef.svg?style=flat)](https://pypi.python.org/pypi/scdef)
[![CI](https://github.com/cbg-ethz/scDEF/actions/workflows/ci.yml/badge.svg)](https://github.com/cbg-ethz/scDEF/actions/workflows/ci.yml)
[![docs](https://github.com/cbg-ethz/scDEF/actions/workflows/docs.yml/badge.svg)](https://cbg-ethz.github.io/scDEF/)

Deep exponential families for single-cell data. scDEF learns hierarchies of cell states and their gene signatures from scRNA-seq data. The method enables model-based exploration of biological and technical effects in the data and can be used for dimensionality reduction, visualization, gene signature identification, clustering at multiple levels of resolution, and batch integration. The informed version (iscDEF) can additionally take known gene lists to jointly assign cells to types and find clusters within each type.

## Getting started

[Install scDEF](installation.md), then:

- **[Basic usage](examples/basicusage.md)** — the shortest path from counts to a
  fitted hierarchy and its signatures.
- **[Analysis workflow](workflow.md)** — the full reference workflow, stating what
  each step needs and what it writes.
- **[API reference](reference/index.md)** — every public object, grouped by task.

## Tutorials

- [Introduction to scDEF on 3k PBMCs](examples/scdef-pbmcs3k.ipynb)
- [Identifying cell type hierarchies in a whole adult animal](examples/scdef-planaria.ipynb)
- [Integration of two batches of PBMCs](examples/scdef-pbmcs-2batches.ipynb)
- [Identifying signatures of interferon-response in PBMCs](examples/scdef-ifn.ipynb)

## Contributors

Pedro Falé Ferreira [@pedrofale](https://github.com/pedrofale)
