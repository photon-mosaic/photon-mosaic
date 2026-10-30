from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("photon-mosaic")
except PackageNotFoundError:
    # package is not installed; spikeinterface parses this when loading a saved object, so it must stay a valid version
    __version__ = "0+unknown"

from .core import *
