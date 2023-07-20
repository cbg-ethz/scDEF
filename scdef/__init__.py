from importlib import metadata

from .main import main
from .models import scDEF, iscDEF
from .utils import eval_utils, hierarchy_utils, jax_utils, score_utils

__version__ = metadata.version("scdef")
