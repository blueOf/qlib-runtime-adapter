"""Optional tree-based adapters."""
from __future__ import annotations

from .sklearn_base import SklearnAdapter


class HistGradientBoostingAdapter(SklearnAdapter):
    def __init__(self):
        super().__init__("sklearn-hist-gradient-boosting.v1")
        self.estimator_kind = "hist_gradient_boosting"


class ExtraTreesAdapter(SklearnAdapter):
    def __init__(self):
        super().__init__("sklearn-extra-trees.v1")
        self.estimator_kind = "extra_trees"
