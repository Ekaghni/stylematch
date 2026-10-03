"""stylematch: detect AI-written text and compare writing styles."""

from .core import Comparison, compare, interpret, similarity

__version__ = "0.1.2"
__all__ = ["compare", "similarity", "interpret", "Comparison", "__version__"]
