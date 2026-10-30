"""Margin uncertainty stays in raw inner-product units and blocks acceptance."""

from dataclasses import replace
import math

import pytest

from oar.audit import NumericalPolicy, decide_margin, validate_candidate
from oar.audit_journal import AuditJournal
from oar.records import AuditCandidate, DecisionKind, from_payload, to_payload


def test_uncertainty_blocks_boundary_acceptance():
    decision = decide_margin(1., {'w': .49}, .5, .02)
    assert decision.kind == DecisionKind.NUMERICAL_REJECT
    assert decision.allowance == .02
    assert decide_margin(1., {'w': .49}, .5, .005).kind == DecisionKind.ACCEPT_APPROX
    assert decide_margin(1., {'w': .51}, .5, .02).kind == DecisionKind.REJECT_REPLAY


def test_margin_uses_all_contributions_and_raw_allowance():
    decision = decide_margin(4., {'w1': .4, 'w2': .5}, .25, .1)
    assert decision.kind == DecisionKind.ACCEPT_APPROX
    assert (decision.a, decision.B, decision.allowance) == (4., .9, .1)
    assert from_payload(to_payload(decision)) == decision
    assert decide_margin(0., {}, .5, 0.).kind == DecisionKind.REJECT_REPLAY
    assert decide_margin(-1., {}, .5, 0.).kind == DecisionKind.REJECT_REPLAY


@pytest.mark.parametrize('a,contributions,tau,u', [
    (math.nan, {}, .5, 0.), (math.inf, {}, .5, 0.), (1., {'w': math.nan}, .5, 0.),
    (1., {'w': -1.}, .5, 0.), (1., {}, 0., 0.), (1., {}, 1., 0.),
    (1., {}, .5, None), (1., {}, .5, math.inf), (1., {}, .5, -.1),
    (1., {'w1': 1e308, 'w2': 1e308}, .5, 0.),
])
def test_nonfinite_or_unresolved_values_reject_and_remain_serializable(a, contributions, tau, u):
    decision = decide_margin(a, contributions, tau, u)
    assert decision.kind == DecisionKind.NUMERICAL_REJECT
    assert from_payload(to_payload(decision)) == decision


def test_policy_requires_bound_configuration_and_measured_evidence():
    config = {'shape': [5, 7], 'dtype': 'float32', 'backend': 'mps',
              'uncertainty_sources': ['residual cancellation', 'orthogonality', 'reduction']}
    unresolved = NumericalPolicy(config)
    assert unresolved.allowance_for(config) is None
    assert unresolved.decide(1., {'w': 0.}, .5, configuration=config).kind == DecisionKind.NUMERICAL_REJECT
    measured = NumericalPolicy(config, measured_u_margin=.03, evidence_sha256='b' * 64)
    assert measured.allowance_for(dict(reversed(list(config.items())))) == .03
    config['shape'][0] = 900
    assert measured.allowance_for(config) is None
    assert measured.config_sha256 != NumericalPolicy(config).config_sha256
    with pytest.raises(ValueError):
        NumericalPolicy(config, measured_u_margin=.03)
    with pytest.raises(ValueError):
        NumericalPolicy(config, measured_u_margin=-1., evidence_sha256='b' * 64)


def test_candidate_metadata_binds_current_rank_and_requires_next_step_changes(tmp_path):
    reservation = AuditJournal(tmp_path / 'audit.jsonl', .01, 0).reserve(('w',), 3)[0]
    candidate = AuditCandidate('w', 3, 2, 7, 1)
    assert from_payload(to_payload(candidate)) == candidate
    validate_candidate(candidate, reservation, current_rank=2, current_basis_version=7, changes_effective_step=4)
    for changed in [replace(candidate, layer_id='other'), replace(candidate, logical_step=4),
                    replace(candidate, basis_source_step=3), replace(candidate, construction_stream='audit'),
                    replace(candidate, probe_ids_used=(reservation.probe_id,))]:
        with pytest.raises(ValueError):
            validate_candidate(changed, reservation, current_rank=2, current_basis_version=7)
    for kwargs in [dict(current_rank=3, current_basis_version=7),
                   dict(current_rank=2, current_basis_version=8),
                   dict(current_rank=2, current_basis_version=7, changes_effective_step=3)]:
        with pytest.raises(ValueError):
            validate_candidate(candidate, reservation, **kwargs)


def test_policy_mismatched_or_nonfinite_runtime_context_fails_closed():
    config = {'shape': [2, 2], 'dtype': 'float32'}
    policy = NumericalPolicy(config, measured_u_margin=.01, evidence_sha256='c' * 64)
    for context in ({'shape': [2, 3], 'dtype': 'float32'}, {'scale': math.nan}, {}):
        assert policy.decide(1., {}, .5, configuration=context).kind == DecisionKind.NUMERICAL_REJECT


def test_measured_orthogonality_defect_requires_allowance_even_with_exact_residual_norm():
    import torch
    # Stored FP32 projector is almost orthogonal; exact arithmetic on its
    # represented values exposes the error in assuming U.T E == 0.
    u = torch.tensor([[1. + 2 ** -22], [0.]], dtype=torch.float32).double()
    gradient = torch.tensor([[1.], [0.]], dtype=torch.float64)
    projected = u @ (u.T @ gradient)
    displacement = projected
    error = gradient - projected
    complement = displacement - u @ (u.T @ displacement)
    a = (projected * displacement).sum().item()
    b = torch.linalg.vector_norm(error).item() * torch.linalg.vector_norm(complement).item()
    actual = (gradient * displacement).sum().item()
    # B alone overstates the alignment lower bound because of U's defect.
    uncertainty = (a - b) - actual
    assert uncertainty > 1e-7
    assert decide_margin(a, {'w': b}, 1e-8, 0.).kind == DecisionKind.ACCEPT_APPROX
    assert decide_margin(a, {'w': b}, 1e-8, uncertainty).kind == DecisionKind.NUMERICAL_REJECT


def test_cancellation_zero_sketch_is_not_machine_arithmetic_certification():
    import torch
    from oar.audit import residual_bound
    from oar.linear import LinearCapture, projected_linear
    x = torch.tensor([[1e8], [1.], [-1e8]], dtype=torch.float32)
    weight = torch.ones(1, 1, requires_grad=True)
    capture = LinearCapture(torch.empty(1, 0), torch.ones(1, 1), basis_version=0)
    projected_linear(x, weight, None, capture).sum().backward()
    exact = torch.ones(1, 3, dtype=torch.float64) @ x.double()
    assert exact.item() == 1.
    assert capture.r_sum.item() == 0.
    assert residual_bound(capture.r_sum, 1, .1) == 0.
    assert NumericalPolicy({'shape': [1, 1], 'scale': 1e8}).decide(
        1., {'w': 0.}, .5, configuration={'shape': [1, 1], 'scale': 1e8}
    ).kind == DecisionKind.NUMERICAL_REJECT
    assert decide_margin(1., {'w': 0.}, .5, exact.item()).kind == DecisionKind.NUMERICAL_REJECT
