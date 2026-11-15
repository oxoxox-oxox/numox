"""numox: High-performance float64 matrix C-extension for Python."""

from ._core import Matrix, to_list

__version__ = "1.0.5"
__all__ = ["Matrix", "to_list", "__version__"]
