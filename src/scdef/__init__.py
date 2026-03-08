from .models import scDEF as scDEF, iscDEF as iscDEF
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
