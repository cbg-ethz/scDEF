from importlib import metadata

from .main import main
from .scdef import scDEF
from .iscdef import iscDEF
from .util import *

__version__ = metadata.version("scdef")
