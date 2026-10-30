"""Gaussian real-arithmetic confidence model and empirically qualified margins.

The scalar root and sketch reduction run explicitly on CPU FP64. Transferring
an MPS sketch here synchronizes and must be charged by the enclosing active
clock. These routines do not certify FP32 projection/reduction/cancellation;
that uncertainty belongs in a shape-specific measured ``NumericalPolicy``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math

import torch
from torch import Tensor

from .records import AuditCandidate, AuditReservation, Decision, DecisionKind


def lower_fraction(q: int, delta: float) -> float:
    """Conservative FP64 root of c - 1 - log(c) = -2 log(delta) / q.

    Bisect t=-log(c), evaluating expm1(-t)+t to avoid cancellation near
    c=1. Return a slightly smaller representable c; fail closed when the
    positive root cannot be represented. This scalar rounding precaution is
    not a machine-arithmetic proof for the full audit.
    """
    if type(q) is not int or q <= 0 or type(delta) not in (int, float) or not 0 < delta < 1:
        raise ValueError('require integer q > 0 and finite 0 < delta < 1')
    try:
        target = (-2 * math.log(delta)) / q
    except OverflowError as exc:
        raise ValueError('confidence scale is not representable') from exc
    if not math.isfinite(target) or target <= 0:
        raise ValueError('confidence scale underflow or not representable')
    lo, hi = 0., target + 1.
    for _ in range(1100):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if math.expm1(-mid) + mid < target:
            lo = mid
        else:
            hi = mid
    c = math.nextafter(math.exp(-math.nextafter(hi, math.inf)), 0.)
    if not 0 < c < 1:
        raise ValueError('positive confidence root underflow or not representable')
    return c


def residual_bound(sketch: Tensor, q: int, delta: float) -> float:
    """Return ||sketch||_F / sqrt(q*c), conditional on independent Gaussian Z.

    ``sketch`` is the detached, aggregated E Z, with exactly q columns. A
    scaled reduction avoids squaring overflow/underflow; an unrepresentable
    final bound is infinity, which the margin decision rejects. Invalid
    evidence/root raises ValueError and must cause numerical rejection.
    """
    c = lower_fraction(q, delta)
    if (not isinstance(sketch, Tensor) or sketch.ndim != 2 or sketch.shape[1] != q
            or sketch.shape[0] == 0 or sketch.dtype not in (torch.float32, torch.float64)
            or sketch.requires_grad):
        raise ValueError('require nonempty detached FP32/FP64 residual matrix with q columns')
    r = sketch.detach().to('cpu').to(torch.float64)
    if not torch.isfinite(r).all():
        raise ValueError('residual sketch must be finite')
    scale = r.abs().max().item()
    if scale == 0:
        return 0.
    sum_scaled = ((r / scale) ** 2).sum().item()
    log_bound = math.log(scale) + .5 * (math.log(sum_scaled) - math.log(q) - math.log(c))
    try:
        bound = math.exp(log_bound)
    except OverflowError:
        return math.inf
    # Positive evidence never turns into an accepting zero on underflow.
    return math.nextafter(bound, math.inf)


def _finite_number(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value)


def decide_margin(a: float, contributions: dict[str, float], tau: float,
                  u_margin: float | None) -> Decision:
    """Accept only positive alignment with B + u_margin <= tau*a.

    ``Decision.allowance`` retains raw u_margin, never normalized slack.
    Finite metadata is retained for rejected decisions; nonfinite metadata is
    null so that failure records remain serializable. No future rank changes
    are proposed here; the online controller owns those.
    """
    safe_a = float(a) if _finite_number(a) else None
    safe_u = float(u_margin) if _finite_number(u_margin) else None
    if (safe_a is None or not _finite_number(tau) or not 0 < tau < 1
            or safe_u is None or safe_u < 0
            or not isinstance(contributions, dict)
            or any(type(k) is not str or not k or not _finite_number(v) or v < 0
                   for k, v in contributions.items())):
        return Decision(DecisionKind.NUMERICAL_REJECT, ('invalid_or_unresolved_numerics',),
                        a=safe_a, allowance=safe_u)
    try:
        total = math.fsum(contributions.values())
    except OverflowError:
        total = math.inf
    if not math.isfinite(total) or not math.isfinite(total + safe_u):
        return Decision(DecisionKind.NUMERICAL_REJECT, ('nonfinite_total_margin',),
                        a=safe_a, allowance=safe_u)
    if a <= 0:
        kind, reason = DecisionKind.REJECT_REPLAY, 'nonpositive_alignment'
    elif total > tau * a:
        kind, reason = DecisionKind.REJECT_REPLAY, 'residual_margin_exceeded'
    elif total + safe_u > tau * a:
        kind, reason = DecisionKind.NUMERICAL_REJECT, 'inside_numerical_uncertainty'
    else:
        kind, reason = DecisionKind.ACCEPT_APPROX, 'qualified_first_order_margin'
    return Decision(kind, (reason,), a=safe_a, B=total, allowance=safe_u)


def _config_bytes(configuration: Mapping) -> bytes:
    if not isinstance(configuration, Mapping) or not configuration:
        raise ValueError('calibration configuration must be a nonempty mapping')
    # A JSON round trip makes a defensive snapshot; NaNs cannot name a policy.
    try:
        return json.dumps(dict(configuration), sort_keys=True, separators=(',', ':'),
                          allow_nan=False).encode('utf-8')
    except (ValueError, TypeError) as exc:
        raise ValueError('calibration configuration must be finite JSON metadata') from exc


@dataclass(frozen=True, init=False)
class NumericalPolicy:
    """A measured total allowance for one exact hashed calibration context.

    Unresolved by default. No coefficient is inferred from tiny operator
    tests. A resolved caller supplies the measured nonnegative inner-product
    allowance and SHA-256 of its empirical evidence. The context must include
    actual shapes/backend/dtype/scales and all uncertainty sources; validating
    that experiment and freezing its policy remain later operator/protocol
    work. This record itself cannot establish evidence quality.
    """

    config_sha256: str
    measured_u_margin: float | None
    evidence_sha256: str | None
    _configuration: bytes = field(repr=False)
    schema_version: int = field(default=1, init=False)

    def __init__(self, configuration: Mapping, measured_u_margin: float | None = None,
                 evidence_sha256: str | None = None):
        config = _config_bytes(configuration)
        if measured_u_margin is not None:
            if not _finite_number(measured_u_margin) or measured_u_margin < 0:
                raise ValueError('measured u_margin must be finite and nonnegative')
            if (type(evidence_sha256) is not str or len(evidence_sha256) != 64
                    or any(c not in '0123456789abcdef' for c in evidence_sha256)):
                raise ValueError('resolved calibration requires an empirical evidence SHA-256')
        object.__setattr__(self, '_configuration', config)
        object.__setattr__(self, 'config_sha256', hashlib.sha256(config).hexdigest())
        object.__setattr__(self, 'measured_u_margin', measured_u_margin)
        object.__setattr__(self, 'evidence_sha256', evidence_sha256)

    def allowance_for(self, configuration: Mapping) -> float | None:
        """An unmatched or unresolved calibration cannot authorize acceptance."""
        try:
            context = _config_bytes(configuration)
        except ValueError:
            return None
        return self.measured_u_margin if context == self._configuration else None

    def decide(self, a: float, contributions: dict[str, float], tau: float, *,
               configuration: Mapping) -> Decision:
        return decide_margin(a, contributions, tau, self.allowance_for(configuration))


def validate_candidate(candidate: AuditCandidate, reservation: AuditReservation, *,
                       current_rank: int, current_basis_version: int,
                       changes_effective_step: int | None = None,
                       current_probe_ids: tuple[str, ...] = ()) -> None:
    """Check pre-probe declarations and post-probe frozen rank/version.

    Call before probe generation and again with actual current capture
    rank/version before using the result. Supply the effective batch's full
    ``current_probe_ids`` when validating outside the journal. Future controller
    changes must name a strictly later logical step. Checks cannot prove arbitrary
    candidate construction code truthfully declared its data dependencies.
    """
    if (candidate.layer_id != reservation.layer_id or candidate.logical_step != reservation.logical_step
            or type(candidate.rank) is not int or candidate.rank < 0
            or type(candidate.basis_version) is not int or candidate.basis_version < 0
            or type(candidate.logical_step) is not int or candidate.logical_step < 0
            or type(candidate.basis_source_step) is not int
            or not 0 <= candidate.basis_source_step < candidate.logical_step
            or candidate.construction_stream != 'construction'
            or set(candidate.probe_ids_used).intersection((*current_probe_ids, reservation.probe_id))
            or type(current_rank) is not int or current_rank != candidate.rank
            or type(current_basis_version) is not int or current_basis_version != candidate.basis_version):
        raise ValueError('candidate must be independent, past-only and frozen at its reserved rank/version')
    if changes_effective_step is not None and (type(changes_effective_step) is not int
                                              or changes_effective_step <= candidate.logical_step):
        raise ValueError('audit-informed changes are next-step only')
