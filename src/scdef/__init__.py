from .models import scDEF, iscDEF
from .utils import hierarchy_utils, jax_utils, score_utils, color_utils
from .benchmark import evaluate, other_methods

from importlib import metadata

__version__ = metadata.version("scdef")
