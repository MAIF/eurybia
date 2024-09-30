"""Top-level package."""

__author__ = """Thomas Bouche, Johann Martin, Nicolas Roux"""
__email__ = "thomas.bouche@maif.fr"

from eurybia.core.smartdrift import SmartDrift

VERSION = (1, 3, 1)

__version__ = ".".join(map(str, VERSION))

__all__ = ["SmartDrift"]
