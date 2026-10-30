"""Versioned transport records. Tensor payloads belong in checkpoints, not JSON.

Frozen metadata protects field assignment, not the storage underlying a tensor.
Numerical and transaction owners enforce tensor lifetime/version invariants.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
import json
from types import MappingProxyType, UnionType
from typing import TypeAlias, Union, get_args, get_origin, get_type_hints

from torch import Tensor


ParameterId: TypeAlias = str
LayerId: TypeAlias = ParameterId


def _freeze_descriptor(value):
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("descriptor keys must be strings")
        return MappingProxyType({key: _freeze_descriptor(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_descriptor(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise TypeError("replay descriptors must contain immutable scalar metadata")


class DecisionKind(str, Enum):
    ACCEPT_APPROX = "accept_approx"
    SCHEDULED_DENSE = "scheduled_dense"
    REJECT_REPLAY = "reject_replay"
    NUMERICAL_REJECT = "numerical_reject"
    STOP_INVALID = "stop_invalid"


@dataclass(frozen=True)
class ReplayableBatch:
    source_ids: tuple[str, ...]
    microbatches: tuple[Mapping[str, object], ...]
    target_masks: tuple[tuple[bool, ...], ...]
    valid_counts: tuple[int, ...]
    pre_cursor: Mapping[str, object]
    post_cursor: Mapping[str, object]
    schema_version: int = field(default=1, init=False)

    def __post_init__(self):
        for name in ("source_ids", "microbatches", "target_masks", "valid_counts", "pre_cursor", "post_cursor"):
            object.__setattr__(self, name, _freeze_descriptor(getattr(self, name)))


@dataclass(frozen=True)
class StepContext:
    logical_step: int
    parameter_version: int
    state_version: int
    valid_tokens: int
    rng_snapshots: dict[str, Tensor]
    schedule_position: int
    batch: ReplayableBatch
    timing_boundary_s: float
    schema_version: int = field(default=1, init=False)


@dataclass(frozen=True)
class LayerAudit:
    layer_id: LayerId
    shape: tuple[int, int]
    rank: int
    basis_version: int
    probe_id: str
    probe_count: int
    allocation: float
    residual_sketch: Tensor | None = None
    statistic: float | None = None
    bound: float | None = None
    orthogonality: float | None = None
    guardband: float | None = None
    profiling: dict[str, float] | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self):
        if self.residual_sketch is not None and self.residual_sketch.requires_grad:
            raise ValueError("audit sketches must be detached")


@dataclass(frozen=True)
class AdamWParamState:
    step: int
    exp_avg: Tensor
    exp_avg_sq: Tensor
    schema_version: int = field(default=1, init=False)


@dataclass(frozen=True)
class AdamWGroup:
    parameter_ids: tuple[ParameterId, ...]
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    schema_version: int = field(default=1, init=False)


@dataclass
class PendingUpdate:
    parameter_version: int
    state_version: int
    candidate_parameters: dict[ParameterId, Tensor]
    decrements: dict[ParameterId, Tensor]
    optimizer_state: dict[ParameterId, AdamWParamState]
    consumed: bool = False
    schema_version: int = field(default=1, init=False)


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reasons: tuple[str, ...]
    a: float | None = None
    B: float | None = None
    allowance: float | None = None
    next_step_changes: dict[LayerId, int | str] = field(default_factory=dict)
    schema_version: int = field(default=1, init=False)


@dataclass(frozen=True)
class StepRecord:
    logical_step: int
    parameter_version: int
    state_version: int
    decision: Decision
    valid_tokens: int
    processed_tokens: int
    active_wall_s: float | None = None
    checkpoint: str | None = None
    schema_version: int = field(default=1, init=False)


@dataclass(frozen=True)
class GateRecord:
    package: str
    status: str
    reason: str
    next_action: str
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    measured_resources: dict[str, float | int] | None = None
    forecast_resources: dict[str, float | int] | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self):
        if self.status not in ("pass", "stop", "inconclusive", "blocked"):
            raise ValueError("invalid gate status")
        if self.package not in {f"WP{index:02d}" for index in range(10)}:
            raise ValueError("unknown gate package")
        if not self.reason.strip() or not self.next_action.strip():
            raise ValueError("gate reason and next_action are required")


_RECORDS = {cls.__name__: cls for cls in (
    ReplayableBatch, StepContext, LayerAudit, AdamWParamState, AdamWGroup,
    PendingUpdate, Decision, StepRecord, GateRecord,
)}


def _encode(value):
    if is_dataclass(value) and type(value).__name__ in _RECORDS:
        return {"record_type": type(value).__name__, **{
            item.name: _encode(getattr(value, item.name)) for item in fields(value)
        }}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Tensor) or value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return {key: _encode(item) for key, item in value.items()}
    raise TypeError(f"unsupported payload type: {type(value).__name__}")


def to_payload(record) -> dict[str, object]:
    """Encode a registered record without copying/stringifying tensor storage."""
    if type(record).__name__ not in _RECORDS:
        raise TypeError("expected a registered OAR record")
    payload = _encode(record)
    from_payload(payload)  # Validate direct constructors at the serialization boundary.
    return payload


def _decode(value, expected):
    if expected is object:
        return value
    origin, args = get_origin(expected), get_args(expected)
    if origin in (UnionType, Union):
        # Preserve integer counters before considering permitted float coercion.
        for choice in sorted(args, key=lambda hint: hint is not type(value)):
            try:
                return _decode(value, choice)
            except (TypeError, ValueError):
                pass
        raise ValueError(f"invalid payload value for {expected}")
    if origin in (dict, Mapping):
        if not isinstance(value, dict) or not all(type(key) is str for key in value):
            raise ValueError("expected string-keyed payload mapping")
        return {key: _decode(item, args[1]) for key, item in value.items()}
    if origin is tuple:
        if not isinstance(value, (tuple, list)):
            raise ValueError("expected payload sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(item, args[0]) for item in value)
        if len(value) != len(args):
            raise ValueError("wrong payload sequence length")
        return tuple(_decode(item, hint) for item, hint in zip(value, args))
    if isinstance(expected, type) and issubclass(expected, Enum):
        return expected(value)
    if isinstance(expected, type) and is_dataclass(expected):
        record = from_payload(value)
        if type(record) is not expected:
            raise ValueError("wrong nested record type")
        return record
    if expected is Tensor:
        if not isinstance(value, Tensor):
            raise ValueError("expected Tensor checkpoint payload")
        return value
    if expected is float and type(value) in (int, float):
        return float(value)
    if type(value) is not expected:
        raise ValueError(f"expected {expected}, got {type(value).__name__}")
    return value


def from_payload(payload: dict[str, object]):
    """Strict schema-v1 reader; reject unknown record types, fields and versions."""
    if not isinstance(payload, dict):
        raise ValueError("expected record payload mapping")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported record schema_version")
    cls = _RECORDS.get(payload.get("record_type"))
    if cls is None:
        raise ValueError("unknown record_type")
    expected = {item.name for item in fields(cls)} | {"record_type"}
    if unknown := payload.keys() - expected:
        raise ValueError(f"unknown payload fields: {sorted(unknown)}")
    if missing := expected - payload.keys():
        raise ValueError(f"missing payload fields: {sorted(missing)}")
    hints = get_type_hints(cls)
    return cls(**{item.name: _decode(payload[item.name], hints[item.name])
                  for item in fields(cls) if item.init})


def to_json(record) -> str:
    """Encode metadata only, retaining explicit nulls and refusing nonfinite data."""
    def unsupported(value):
        raise TypeError(f"{type(value).__name__} requires a checkpoint payload, not JSON")

    return json.dumps(to_payload(record), sort_keys=True, allow_nan=False, default=unsupported)
