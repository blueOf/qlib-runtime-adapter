"""Explicit model-adapter registry.

Configuration contains registry keys only.  It never supplies Python import
paths, so an adapter can execute only after it is registered by repository
code and its factory has passed the controlled dependency boundary.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping

from .contracts import AdapterUnknownError, DependencyUnavailableError, ModelAdapter


Factory = Callable[[], ModelAdapter]


class AdapterRegistry:
    def __init__(self):
        self._factories: dict[str, Factory] = {}

    def register(self, adapter_id: str, factory: Factory, *, replace: bool = False) -> None:
        key = str(adapter_id)
        if not key or any(char.isspace() for char in key):
            raise ValueError("adapter registry key must be a non-empty token")
        if key in self._factories and not replace:
            raise ValueError(f"duplicate adapter registry key: {key}")
        self._factories[key] = factory

    def has(self, adapter_id: str) -> bool:
        return str(adapter_id) in self._factories

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def resolve(self, adapter_id: str) -> ModelAdapter:
        key = str(adapter_id)
        factory = self._factories.get(key)
        if factory is None:
            raise AdapterUnknownError(f"unknown model adapter: {key}")
        try:
            adapter = factory()
        except DependencyUnavailableError:
            raise
        except ImportError as error:
            raise DependencyUnavailableError(f"adapter dependency unavailable: {key}") from error
        if getattr(adapter, "adapter_id", None) != key:
            raise ValueError(f"factory returned adapter with mismatched id: {key}")
        return adapter

    def capabilities(self) -> dict[str, Mapping[str, Any]]:
        result = {}
        for key in self.keys():
            try:
                adapter = self.resolve(key)
                capabilities = dict(adapter.capabilities())
                availability = getattr(adapter, "availability", None)
                if callable(availability):
                    available, reason = availability()
                    capabilities["available"] = bool(available)
                    if reason:
                        capabilities["availability_error"] = reason
                else:
                    capabilities.setdefault("available", True)
                result[key] = capabilities
            except DependencyUnavailableError as error:
                result[key] = {"adapter_id": key, "available": False, "error_code": error.code,
                               "error": str(error)}
        return result


def build_default_registry() -> AdapterRegistry:
    from .adapters.composite import CompositeGraphAdapter
    from .adapters.deterministic import DeterministicAdapter
    from .adapters.identity import IdentityAdapter
    from .adapters.sklearn_base import LogisticAdapter, RidgeAdapter
    from .adapters.sklearn_tree import ExtraTreesAdapter, HistGradientBoostingAdapter

    registry = AdapterRegistry()
    registry.register("identity.v1", IdentityAdapter)
    registry.register("fixed-rank-blend.v1", lambda: DeterministicAdapter("fixed-rank-blend.v1", transform="rank"))
    registry.register("fixed-linear.v1", lambda: DeterministicAdapter("fixed-linear.v1", transform="linear"))
    registry.register("sklearn-logistic.v1", LogisticAdapter)
    registry.register("sklearn-ridge.v1", RidgeAdapter)
    registry.register("sklearn-hist-gradient-boosting.v1", HistGradientBoostingAdapter)
    registry.register("sklearn-extra-trees.v1", ExtraTreesAdapter)
    registry.register("composite-graph.v1", lambda: CompositeGraphAdapter(registry))
    return registry
