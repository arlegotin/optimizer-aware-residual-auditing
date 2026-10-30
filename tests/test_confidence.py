"""Independent scalar references and resolvable Gaussian confidence checks."""

import math

import numpy as np
import pytest
import torch

from oar.audit import lower_fraction, residual_bound


@pytest.mark.parametrize('q,delta,multiplier', [
    (8, .01, 2.7434), (16, .01, 1.9196), (32, .01, 1.5435),
    (8, 1e-6, 9.2170), (16, 1e-6, 3.7749), (32, 1e-6, 2.3122),
])
def test_scalar_root_matches_hand_tabulated_norm_multipliers(q, delta, multiplier):
    c = lower_fraction(q, delta)
    assert 0 < c < 1
    assert 1 / math.sqrt(c) == pytest.approx(multiplier, abs=5e-5)
    assert c - 1 - math.log(c) == pytest.approx(-2 * math.log(delta) / q, rel=1e-13)


@pytest.mark.parametrize('q,delta', [(0, .1), (-1, .1), (True, .1), (1.5, .1),
                                      (8, 0), (8, 1), (8, -1), (8, math.nan), (8, math.inf)])
def test_invalid_confidence_inputs_cannot_form_bound(q, delta):
    with pytest.raises(ValueError):
        lower_fraction(q, delta)


def test_log_domain_handles_small_delta_and_rejects_unrepresentable_root():
    c = lower_fraction(8, 1e-300)
    assert c > 0
    assert c - 1 - math.log(c) == pytest.approx(-2 * math.log(1e-300) / 8, rel=1e-13)
    with pytest.raises(ValueError, match='represent|underflow'):
        lower_fraction(1, math.ulp(0.0))
    with pytest.raises(ValueError):
        residual_bound(torch.zeros(2, 1), 1, math.ulp(0.0))


def test_delta_close_to_one_is_not_rounded_to_zero_risk():
    c = lower_fraction(32, math.nextafter(1., 0.))
    assert 0 < c < 1
    t = -math.log(c)
    assert math.expm1(-t) + t == pytest.approx(-math.log(math.nextafter(1., 0.)) / 16, rel=2e-6)


@pytest.mark.parametrize('scale', [1e-300, 1., 1e300])
def test_bound_uses_scaled_norm_without_squaring_underflow_or_overflow(scale):
    sketch = torch.tensor([[3., 0.], [0., 4.]], dtype=torch.float64) * scale
    # q=2 and c=.25: delta = exp(-q/2 * (.25 - 1 - log(.25))).
    delta = .25 * math.exp(.75)
    actual = residual_bound(sketch, 2, delta)
    assert math.isfinite(actual) and actual > 0
    assert actual / scale == pytest.approx(5 * math.sqrt(2), rel=2e-13)
    assert residual_bound(torch.zeros_like(sketch), 2, delta) == 0


@pytest.mark.parametrize('sketch,q', [(torch.ones(3), 3), (torch.ones(2, 2), 3),
    (torch.ones(0, 2), 2), (torch.ones(2, 2, dtype=torch.int64), 2),
    (torch.full((2, 2), math.nan), 2), (torch.full((2, 2), math.inf), 2),
    (torch.ones(2, 2, requires_grad=True), 2)])
def test_malformed_sketch_is_not_an_accepting_zero(sketch, q):
    with pytest.raises(ValueError):
        residual_bound(sketch, q, .05)


@pytest.mark.parametrize('spectrum', [[1., 0., 0., 0.], [1., 1., 1., 1.], [1., .5, .25, .125]])
def test_gaussian_failure_frequency_is_conservative_at_resolvable_probability(spectrum):
    # 20,000 independent NumPy draws; q=8, delta=.1. A binomial test at this
    # scale can detect gross undercoverage, not validate tiny run-wide risks.
    z = np.random.default_rng(8401).normal(size=(20_000, 4, 8))
    r = z * np.asarray(spectrum)[None, :, None]
    c = lower_fraction(8, .1)
    squared_bounds = (r * r).sum(axis=(1, 2)) / (8 * c)
    failures = np.count_nonzero(squared_bounds < sum(s * s for s in spectrum))
    assert failures / 20_000 < .1 + 5 * math.sqrt(.1 * .9 / 20_000)
    # Also verify the public reduction on independently drawn sketches.
    for sample in r[:3]:
        expected = math.sqrt(float(np.square(sample).sum()) / (8 * c))
        assert residual_bound(torch.from_numpy(sample), 8, .1) == pytest.approx(expected, rel=2e-13)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires actual visible MPS')
def test_mps_sketch_transfer_then_cpu_fp64_reduction():
    sketch = torch.tensor([[3., 0.], [0., 4.]], device='mps')
    before = torch.mps.get_rng_state().clone()
    actual = residual_bound(sketch, 2, .25 * math.exp(.75))
    assert actual == pytest.approx(5 * math.sqrt(2), rel=2e-13)
    assert torch.equal(before, torch.mps.get_rng_state())
