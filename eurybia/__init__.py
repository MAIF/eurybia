"""Top-level package."""

import importlib.metadata

from eurybia.core.smartdrift import SmartDrift

__author__ = """Thomas Bouche, Johann Martin, Nicolas Roux"""
__email__ = "thomas.bouche@maif.fr"


__version__ = importlib.metadata.metadata("eurybia")["Version"]

__all__ = [__version__, "SmartDrift"]  # noqa: PLE0604
