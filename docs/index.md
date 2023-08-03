<div align="left">
  <img src="https://github.com/cbg-ethz/scDEF/raw/main/figures/scdef.png", width="300px">
</div>
<p></p>

[![pypi](https://img.shields.io/pypi/v/scdef.svg?style=flat)](https://pypi.python.org/pypi/scdef)
[![build](https://github.com/cbg-ethz/scDEF/actions/workflows/test.yaml/badge.svg)](https://github.com/cbg-ethz/scDEF/actions/workflows/test.yaml)
[![docs](https://github.com/cbg-ethz/scDEF/actions/workflows/docs.yaml/badge.svg)](https://cbg-ethz.github.io/scDEF/)

Deep exponential families for single-cell data. scDEF learns hierarchies of cell states and their gene signatures from scRNA-seq data. The method can be used for dimensionality reduction, visualization, gene signature identification, clustering at multiple levels of resolution, and batch integration. The informed version (iscDEF) can additionally take known gene lists to jointly assign cells to types and find clusters within each type.

## Installation
scDEF is available through [PyPI](https://pypi.org/project/scdef):

```
pip install scdef
```

Please be sure to install a version of [JAX](https://jax.readthedocs.io/) that is compatible with your GPU (if applicable). scDEF is much faster on the GPU than on the CPU.

### Optional: using the `scdef.benchmark` module
The `scdef.benchmark` module includes wrapper functions to other methods. If you wish to use it, please install the extras:

```
pip install scdef[extras]
```

The `scdef.benchmark` also contains a wrapper function to `scVI` from [scvi-tools](https://scvi-tools.org/), but `scvi-tools` is not included in the extras and it must be installed separately if you wish to use it. The same applies to [scHPF](https://github.com/simslab/scHPF).

## Example notebooks
To get started with scDEF, please see the example notebooks:

- [Introduction to scDEF on 3k PBMCs](http://cbg-ethz.github.io/scDEF/examples/scdef-pbmcs3k/)


## Contributors
Pedro Falé Ferreira [@pedrofale](https://github.com/pedrofale)
