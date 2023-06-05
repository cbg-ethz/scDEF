from importlib import metadata

from .main import main
from .scdef import scDEF
from .util import *

__version__ = metadata.version("scdef")
