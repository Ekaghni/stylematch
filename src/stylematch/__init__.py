"""stylematch: compare the writing style of two texts."""

from .core import Comparison, compare, interpret, similarity

__version__ = "0.1.1"
__all__ = ["compare", "similarity", "interpret", "Comparison", "__version__"]
