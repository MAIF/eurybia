"""Top-level package."""

from importlib.metadata import PackageNotFoundError, version

from eurybia.core.smartdrift import SmartDrift

__author__ = """Thomas Bouche, Clément Guérin, Johann Martin, Nicolas Roux, Guillaume Vignal"""
__email__ = "thomas.bouche@maif.fr"


try:
    __version__ = version("eurybia")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [__version__, "SmartDrift"]  # noqa: PLE0604
