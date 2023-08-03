scDEF is available through [PyPI](https://pypi.org/project/scdef):

```
pip install scdef
```

Please be sure to install a version of [JAX](https://jax.readthedocs.io/) that is compatible with your GPU (if applicable). scDEF is much faster on the GPU than on the CPU.


The `scdef.benchmark` module includes wrapper functions to other methods. If you wish to use it, please install the extras:

```
pip install scdef[extras]
```

The `scdef.benchmark` also contains a wrapper function to `scVI` from [scvi-tools](https://scvi-tools.org/), but `scvi-tools` is not included in the extras and it must be installed separately if you wish to use it. The same applies to [scHPF](https://github.com/simslab/scHPF).
