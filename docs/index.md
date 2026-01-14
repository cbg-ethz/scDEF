<div align="left">
  <img src="https://github.com/cbg-ethz/scDEF/raw/main/docs/assets/images/scdef.png", width="300px">
</div>
<p></p>

[![pypi](https://img.shields.io/pypi/v/scdef.svg?style=flat)](https://pypi.python.org/pypi/scdef)
[![build](https://github.com/cbg-ethz/scDEF/actions/workflows/test.yaml/badge.svg)](https://github.com/cbg-ethz/scDEF/actions/workflows/test.yaml)
[![docs](https://github.com/cbg-ethz/scDEF/actions/workflows/docs.yaml/badge.svg)](https://cbg-ethz.github.io/scDEF/)

Deep exponential families for single-cell data. scDEF learns hierarchies of cell states and their gene signatures from scRNA-seq data. The method enables model-based exploration of biological and technical effects in the data and can be used for dimensionality reduction, visualization, gene signature identification, clustering at multiple levels of resolution, and batch integration. The informed version (iscDEF) can additionally take known gene lists to jointly assign cells to types and find clusters within each type.

## Installation
scDEF is available through [PyPI](https://pypi.org/project/scdef):

```
pip install scdef
```

Please be sure to install a version of [JAX](https://jax.readthedocs.io/) that is compatible with your GPU (if applicable). scDEF is much faster on the GPU than on the CPU.

## Example notebooks
To get started with scDEF, please see the example notebooks:

- [Introduction to scDEF on 3k PBMCs](http://cbg-ethz.github.io/scDEF/examples/scdef-pbmcs3k/)

- [Identifying cell type hierarchies in a whole adult animal](http://cbg-ethz.github.io/scDEF/examples/scdef-planaria/)

- [Integration of two batches of PBMCs](http://cbg-ethz.github.io/scDEF/examples/scdef-pbmcs-2batches/)

- [Identifying signatures of interferon-response in PBMCs](http://cbg-ethz.github.io/scDEF/examples/scdef-ifn/)


## Contributors
Pedro Falé Ferreira [@pedrofale](https://github.com/pedrofale)
