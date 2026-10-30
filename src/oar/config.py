"""Strict exploratory configuration; later owners extend this schema explicitly."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class SeedStreams:
    construction: int
    audit: int
    data: int
    model: int


@dataclass(frozen=True)
class RunPaths:
    data: Path
    runs: Path
    decisions: Path


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    phase: str
    method: str
    device: str
    seeds: SeedStreams
    paths: RunPaths


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("configuration keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration field: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _fields(value: object, expected: set[str], where: str) -> Mapping:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    unknown = value.keys() - expected
    missing = expected - value.keys()
    if unknown:
        raise ValueError(f"unknown {where} fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing {where} fields: {sorted(missing)}")
    return value


def _parse(payload: object) -> RunConfig:
    raw = _fields(payload, {"schema_version", "phase", "method", "device", "seeds", "paths"}, "config")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported config schema_version")
    if raw["phase"] not in ("exploratory", "confirmatory"):
        raise ValueError("phase must be exploratory or confirmatory")
    if raw["phase"] == "confirmatory":
        raise ValueError("unresolved confirmatory protocol: schema not implemented yet")
    if raw["method"] != "dense_adamw":
        raise ValueError("unsupported method; only dense_adamw is currently configured")
    if raw["device"] not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
    seeds = _fields(raw["seeds"], {"construction", "audit", "data", "model"}, "seeds")
    for name, value in seeds.items():
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError(f"seed {name} must be an integer in [0, 2**63)")
    paths = _fields(raw["paths"], {"data", "runs", "decisions"}, "paths")
    for name, value in paths.items():
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError(f"path {name} must be a nonempty path string")
    return RunConfig(1, raw["phase"], raw["method"], raw["device"],
                     SeedStreams(**seeds), RunPaths(**{key: Path(value) for key, value in paths.items()}))


def load_config(path: Path) -> RunConfig:
    """Read YAML without silently discarding duplicate or unknown fields."""
    try:
        payload = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError("invalid YAML configuration") from exc
    return _parse(payload)


def resolve_config(config: RunConfig) -> dict[str, object]:
    """Return independent JSON metadata; relative paths stay relative to run cwd."""
    payload = asdict(config)
    payload["paths"] = {name: str(value) for name, value in payload["paths"].items()}
    _parse(payload)  # Also validate instances constructed directly by callers.
    payload["scope"] = "correctness_structural" if config.device == "cpu" else "exploratory"
    return payload


def config_sha256(config: RunConfig) -> str:
    canonical = json.dumps(resolve_config(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DiscoveryConfig:
    schema_version: int = 1
    device: str = "auto"
    allow_mps_fallback: bool = False


def load_discovery_config(path: Path) -> DiscoveryConfig:
    """Discovery is separate from training; automatic fallback is never permitted."""
    try:
        payload = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError("invalid YAML configuration") from exc
    raw = _fields(payload, {"schema_version", "device", "allow_mps_fallback"}, "discovery")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported discovery schema_version")
    if raw["device"] not in ("auto", "cpu", "mps"):
        raise ValueError("discovery device must be auto, cpu or mps")
    if raw["allow_mps_fallback"] is not False:
        raise ValueError("MPS fallback must be false")
    return DiscoveryConfig(**raw)
