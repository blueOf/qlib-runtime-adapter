"""Built-in model adapters registered by :mod:`quant_project.modeling`."""

from .composite import CompositeGraphAdapter
from .deterministic import DeterministicAdapter
from .identity import IdentityAdapter
from .sklearn_base import LogisticAdapter, RidgeAdapter
from .sklearn_tree import ExtraTreesAdapter, HistGradientBoostingAdapter

__all__ = [
    "CompositeGraphAdapter",
    "DeterministicAdapter",
    "IdentityAdapter",
    "LogisticAdapter",
    "RidgeAdapter",
    "HistGradientBoostingAdapter",
    "ExtraTreesAdapter",
]
