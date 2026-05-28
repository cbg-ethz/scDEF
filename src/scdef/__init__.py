import os as _os

# Silence noisy XLA/TF C++ logs (e.g. GPU autotuner "All configs were filtered out…"
# and "Compiling N configs for ..." warnings) that appear during scDEF.fit(). Must be
# set before jax/XLA is imported. Users can override by exporting TF_CPP_MIN_LOG_LEVEL
# themselves before importing scdef.
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from .models import scDEF as scDEF, iscDEF as iscDEF, sscDEF as sscDEF
from .utils import (
    hierarchy_utils as hierarchy_utils,
    jax_utils as jax_utils,
    score_utils as score_utils,
    color_utils as color_utils,
)
from . import plotting as pl  # noqa: F401
from . import tools as tl  # noqa: F401

from importlib import metadata

__version__ = metadata.version("scdef")
