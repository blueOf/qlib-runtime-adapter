"""Load component YAML and the central JSON experiment catalog."""
from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML

from .models import ConfigError


def load_yaml(path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = YAML(typ="safe", pure=True).load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return value


def load_json(path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"JSON root must be an object: {path}")
    return value


def resolve_research_component(value, loader, method):
    """Resolve/schema-check a model research component without importing services."""
    from ..modeling.contracts import ConfigError as ModelConfigError
    from jsonschema import Draft202012Validator
    if isinstance(value, str):
        try:
            value = getattr(loader, method)(value)[1]
        except (ValueError, FileNotFoundError) as error:
            raise ModelConfigError(f"cannot resolve {method} config: {value}") from error
    if not isinstance(value, dict):
        raise ModelConfigError(f"{method} must be a resolved object or a config ID")
    schema_name = {"factor_set": "factor-set", "model": "model-spec", "label": "label-spec",
                   "split_policy": "split-policy", "evaluation_policy": "evaluation-policy"}.get(method)
    if schema_name:
        schema = load_json(Path(__file__).resolve().parents[3] / f"contracts/{schema_name}.schema.json")
        issue = next(Draft202012Validator(schema).iter_errors(value), None)
        if issue:
            raise ModelConfigError(f"resolved {method} violates its schema: {issue.message}")
    return value


class ConfigLoader:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def factor_registry(self):
        path = self.root / "factors/registry.yaml"
        data = load_yaml(path)
        factors = data.get("factors")
        if not isinstance(factors, dict):
            raise ConfigError("factor registry must contain a factors mapping")
        return path, factors

    def _by_id(self, folder: str, identifier: str, *, reject_duplicates=False):
        directory = self.root / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        matches = []
        for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
            data = load_yaml(path)
            if data.get("id") == identifier:
                if not reject_duplicates:
                    return path, data
                matches.append((path, data))
        if len(matches) > 1:
            raise ConfigError(f"duplicate {folder} ID: {identifier}")
        if matches:
            return matches[0]
        raise ConfigError(f"{folder} ID not found: {identifier}")

    def signal(self, identifier, *, reject_duplicates=False):
        return self._by_id("signals", identifier, reject_duplicates=reject_duplicates)

    def strategy(self, identifier, *, reject_duplicates=False):
        return self._by_id("strategies", identifier, reject_duplicates=reject_duplicates)

    def factor_set(self, identifier):
        return self._by_id("factor_sets", identifier, reject_duplicates=True)

    def model(self, identifier):
        return self._by_id("models", identifier, reject_duplicates=True)

    def label(self, identifier):
        return self._by_id("labels", identifier, reject_duplicates=True)

    def split_policy(self, identifier):
        return self._by_id("split_policies", identifier, reject_duplicates=True)

    def evaluation_policy(self, identifier):
        return self._by_id("evaluation_policies", identifier, reject_duplicates=True)

    def experiment(self, identifier):
        # A separate catalog preserves V1 history. IDs are unique across both;
        # a V2 request must never silently fall through to a legacy composition.
        catalogs = [self.root / "experiments/combination_catalog.json",
                    self.root / "experiments/combination_catalog_v2.json"]
        seen, found = set(), None
        for path in catalogs:
            if not path.is_file():
                continue
            catalog = load_json(path)
            version = "v2" if path.name == "combination_catalog_v2.json" else "v1"
            if catalog.get("schema_version") != f"quant-project-combination-catalog-{version}":
                raise ConfigError(f"unsupported combination catalog schema: {path}")
            if set(catalog) != {"schema_version", "combinations"} or not isinstance(catalog["combinations"], list):
                raise ConfigError("combination catalog requires only schema_version and combinations array")
            for item in catalog["combinations"]:
                if not isinstance(item, dict) or set(item) != {"id", "name", "sources"}:
                    raise ConfigError("each combination entry must contain only id, name, and sources")
                item_id = item["id"]
                if not isinstance(item_id, str) or not item_id:
                    raise ConfigError("combination requires a non-empty id")
                if item_id in seen:
                    raise ConfigError(f"duplicate combination ID: {item_id}")
                seen.add(item_id)
                if item_id != identifier:
                    continue
                if not isinstance(item["name"], str) or not item["name"].strip():
                    raise ConfigError(f"combination {identifier} requires a non-empty name")
                sources = item["sources"]
                required = {"factor_set", "model", "signal", "strategy"} if version == "v2" else {"factor", "signal", "strategy"}
                if not isinstance(sources, dict) or set(sources) != required or any(
                        not isinstance(sources[key], str) or not sources[key] for key in required):
                    raise ConfigError(f"combination {identifier} sources must name {', '.join(sorted(required))} IDs")
                found = path, {"schema_version": f"stock-experiment-{version}", "id": identifier,
                               **sources, "metadata": {"name": item["name"].strip()}}
        if found is None:
            raise ConfigError(f"experiment ID not found in combination catalog: {identifier}")
        return found
