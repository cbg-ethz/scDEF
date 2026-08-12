# Installation

scDEF is available through [PyPI](https://pypi.org/project/scdef) and requires
Python 3.10 or 3.11:

```bash
pip install scdef
```

## GPU support

scDEF runs on [JAX](https://jax.readthedocs.io/), and is much faster on a GPU
than on a CPU. Installing `scdef` pulls in a CPU build of JAX, which is enough to
get started but will be the bottleneck on anything beyond a small data set.

To use a GPU, install the JAX build matching your CUDA version *after* installing
scDEF, so it replaces the CPU build:

```bash
pip install scdef
pip install --upgrade "jax[cuda12]"
```

Check which backend you ended up with:

```python
import jax
jax.devices()      # [CudaDevice(id=0)] on GPU, [CpuDevice(id=0)] on CPU
```

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
for other accelerators and CUDA versions. scDEF pins `jax>=0.4.31,<0.7`, so pick a
build in that range.

## Graph rendering

Plotting the factor hierarchy with [`scdef.pl.make_graph`](reference/pl.make_graph.md)
needs the Graphviz system library, which is not a Python package and so is not
installed by pip:

```bash
conda install -c conda-forge graphviz   # or: brew install graphviz
```

Everything else in scDEF works without it.

## Development install

To work on scDEF itself, clone the repository and install with
[Poetry](https://python-poetry.org/), including the development and optional
dependency groups:

```bash
git clone https://github.com/cbg-ethz/scDEF.git
cd scDEF
poetry install --with dev,extras
poetry run pytest tests/test_integration.py
```
