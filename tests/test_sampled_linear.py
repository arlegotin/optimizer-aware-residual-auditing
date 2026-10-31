# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic CPU correctness for fixed-budget row-sampled dW."""
import hashlib
import importlib
import itertools
import json
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F


@pytest.fixture
def op():
    # Missing implementation is a behavioral RED failure, not collection failure.
    assert importlib.util.find_spec('oar.sampled_linear') is not None, 'standalone sampler missing'
    return importlib.import_module('oar.sampled_linear')


def close(a, b):
    torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)


def fixture(n=5, dtype=torch.float64):
    g = torch.Generator().manual_seed(91)
    return (torch.randn(n, 4, generator=g, dtype=dtype),
            torch.randn(3, 4, generator=g, dtype=dtype),
            torch.randn(3, generator=g, dtype=dtype),
            torch.randn(n, 3, generator=g, dtype=dtype))


@pytest.mark.parametrize('n,k', [(2, 1), (3, 1), (3, 2)])
@pytest.mark.parametrize('sampler', ['importance', 'uniform'])
def test_enumeration_unbiased_variance_duplicates(op, n, k, sampler):
    # Omitting duplicates, 1/(kp), or dividing by n breaks expectation/variance.
    x = torch.tensor([[1., 2.], [-3., .5], [2., -1.]], dtype=torch.float64)[:n]
    d = torch.tensor([[.5, -2.], [4., 1.], [-1., 3.]], dtype=torch.float64)[:n]
    # Enumeration probabilities are a separate FP64 reference, independent of
    # production FP32 score rounding. Only gather/weight/GEMM uses runtime code.
    norms = torch.linalg.vector_norm(x, dim=1)*torch.linalg.vector_norm(d, dim=1)
    p = .9*norms/norms.sum()+.1/n if sampler == 'importance' else torch.full((n,), 1/n, dtype=torch.float64)
    exact = d.T @ x
    expected = torch.zeros_like(exact)
    variance = torch.zeros_like(exact)
    duplicate_seen = False
    for rows in itertools.product(range(n), repeat=k):
        indices = torch.tensor(rows)
        estimate = op.gather_weight_gradient(x, d, indices, p, k)
        probability = math.prod(p[i].item() for i in rows)
        expected += probability * estimate
        variance += probability * (estimate - exact).square()
        duplicate_seen |= len(set(rows)) < k
    single_second = sum(torch.outer(d[i], x[i]).square() / p[i] for i in range(n))
    close(expected, exact)
    close(variance, (single_second - exact.square()) / k)
    if k > 1:
        assert duplicate_seen


@pytest.mark.parametrize('sampler', ['importance', 'uniform'])
@pytest.mark.parametrize('k', [1, 4, 5, 8])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_linear_forward_exact_adjoints_and_native_dispatch(op, k, dtype, sampler):
    x, w, b, d = fixture(dtype=dtype)
    x, w, b = [t.requires_grad_() for t in (x, w, b)]
    state = op.SamplingState(2, 'block0.ffn_in', sampler=sampler)
    saved = state.generator.get_state().clone()
    y = op.sampled_linear(x, w, b, state, k=k)
    native = F.linear(x, w, b)
    torch.testing.assert_close(y, native, rtol=0, atol=0)
    actual = torch.autograd.grad(y, (x, w, b), d)
    expected = torch.autograd.grad(native, (x, w, b), d)
    for i in (0, 2):
        torch.testing.assert_close(actual[i], expected[i], rtol=1e-10 if dtype == torch.float64 else 1e-4,
                                   atol=1e-12 if dtype == torch.float64 else 1e-6)
    if k >= len(x):
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        assert torch.equal(saved, state.generator.get_state())
        assert type(y.grad_fn) is type(native.grad_fn)
    else:
        assert not torch.equal(saved, state.generator.get_state())


def test_deep_nonlinear_residual_keeps_root_bias_and_unsampled_grads_exact(op):
    def run(sample):
        g = torch.Generator().manual_seed(6)
        x = torch.randn(2, 3, 4, generator=g, dtype=torch.float64).requires_grad_()
        h, params, exact_indices = x, [], []
        for layer in range(3):
            w1 = (torch.randn(7, 4, generator=g, dtype=x.dtype)/4).requires_grad_()
            w2 = (torch.randn(4, 7, generator=g, dtype=x.dtype)/4).requires_grad_()
            b1 = torch.randn(7, generator=g, dtype=x.dtype).requires_grad_()
            b2 = torch.randn(4, generator=g, dtype=x.dtype).requires_grad_()
            hidden = op.sampled_linear(h, w1, b1, op.SamplingState(3, str(layer)), k=2) if sample else F.linear(h, w1, b1)
            h = h + F.linear(F.gelu(hidden), w2, b2)
            exact_indices.extend([1+len(params)+1, 1+len(params)+2, 1+len(params)+3])
            params.extend((w1, w2, b1, b2))
        return h, torch.autograd.grad((h.sin()+h.square()/3).mean(), (x, *params)), exact_indices
    a, ag, indices = run(True)
    b, bg, _ = run(False)
    close(a, b)
    for i in [0, *indices]:
        close(ag[i], bg[i])


@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('freeze', ['x', 'weight', 'neither'])
def test_layout_leading_axes_optional_bias_and_needs_input_grad(op, bias, freeze):
    x = torch.arange(48., dtype=torch.float64).reshape(2, 4, 6).transpose(0, 1)[..., ::2]
    assert not x.is_contiguous()
    x.requires_grad_(freeze != 'x')
    w = torch.ones(2, 3, dtype=x.dtype, requires_grad=freeze != 'weight')
    b = torch.zeros(2, dtype=x.dtype, requires_grad=True) if bias else None
    state = op.SamplingState(9, 'layout')
    saved = state.generator.get_state().clone()
    y = op.sampled_linear(x, w, b, state, k=1)
    leaves = [v for v in (x, w, b) if v is not None and v.requires_grad]
    grads = torch.autograd.grad(y.sum(), leaves)
    assert all(torch.isfinite(v).all() for v in grads)
    if freeze == 'weight':
        assert torch.equal(saved, state.generator.get_state())
    if x.requires_grad:
        close(grads[0], torch.full_like(x, 2))


@pytest.mark.parametrize('which', ['x', 'd'])
@pytest.mark.parametrize('sampler', ['importance', 'uniform'])
def test_proved_zero_dense_gradient_without_draws(op, which, sampler):
    x, w, b, d = fixture()
    if which == 'x': x.zero_()
    else: d.zero_()
    x, w, b = [v.requires_grad_() for v in (x, w, b)]
    state = op.SamplingState(4, 'zero', sampler=sampler)
    saved = state.generator.get_state().clone()
    gx, gw, gb = torch.autograd.grad(op.sampled_linear(x, w, b, state, k=1), (x, w, b), d)
    assert gw is not None
    close(gw, torch.zeros_like(w))
    close(gx, d @ w)
    close(gb, d.sum(0))
    assert torch.equal(saved, state.generator.get_state())


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_reciprocal_scale_score_underflow_aborts_before_draw(op, dtype):
    x = torch.tensor([[1e30], [1e-30]], dtype=dtype, requires_grad=True)
    d = torch.tensor([[1e-30], [1e30]], dtype=dtype)
    w = torch.ones(1, 1, dtype=dtype, requires_grad=True)
    state = op.SamplingState(1, 'underflow')
    saved = state.generator.get_state().clone()
    with pytest.raises(RuntimeError, match='score'):
        op.sampled_linear(x, w, None, state, k=1).backward(d)
    assert torch.equal(saved, state.generator.get_state())


@pytest.mark.parametrize('n', [1, 3, 8192])
def test_probability_bounds_concentration_and_cast_inverse_weights(op, n):
    x = torch.ones(n, 2)
    d = torch.ones(n, 2)
    if n > 1:
        x[0] = 1e30
        x[-1] = 1e-30
    p = op.importance_probabilities(x, d)
    assert p.dtype == torch.float64 and p.device.type == 'cpu'
    assert abs(p.sum().item()-1) <= 5e-15
    assert p.min() >= .1/n*(1-1e-12)
    k = max(1, n//4)
    weights = (k*p).rsqrt().float().double()
    assert ((k*p*weights.square()-1).abs() <= 8*torch.finfo(torch.float32).eps).all()


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
@pytest.mark.parametrize('operand', ['x', 'd'])
def test_nonfinite_operands_abort(op, value, operand):
    x, w, _, d = fixture(dtype=torch.float32)
    (x if operand == 'x' else d)[0, 0] = value
    w.requires_grad_()
    state = op.SamplingState(3, 'bad')
    with pytest.raises(RuntimeError, match='finite'):
        op.sampled_linear(x, w, None, state, k=1).backward(d)


def test_finite_operands_with_gradient_overflow_abort(op):
    x = torch.full((3, 1), 1e30)
    w = torch.ones(1, 1, requires_grad=True)
    d = torch.full((3, 1), 1e30)
    with pytest.raises(RuntimeError, match='finite'):
        op.sampled_linear(x, w, None, op.SamplingState(1, 'overflow'), k=1).backward(d)


@pytest.mark.parametrize('scale', [1e-150, 1e150])
def test_fp64_reference_scale_is_preserved_before_score_cast(op, scale):
    x = torch.tensor([[1., 2.], [2., -1.], [-1., 3.]], dtype=torch.float64)*scale
    d = torch.tensor([[2.], [1.], [3.]], dtype=torch.float64)/scale
    p = op.importance_probabilities(x, d)
    expected = torch.tensor([math.sqrt(5)*2, math.sqrt(5), math.sqrt(10)*3], dtype=torch.float64)
    # Score FP32 rounding, independent of reference operand scale, is allowed.
    torch.testing.assert_close(p, .9*expected/expected.sum()+.1/3, rtol=1e-4, atol=1e-6)
    got = op.gather_weight_gradient(x, d, torch.tensor([1]), p, 1)
    close(got, torch.outer(d[1], x[1])/p[1])


def test_rank_deficient_cancellation_wide_and_tiny(op):
    x = torch.tensor([[1., -1.], [1., -1.], [1e-20, -1e-20]], dtype=torch.float64).repeat(1, 32)
    d = torch.tensor([[1.], [-1.], [1e-20]], dtype=torch.float64)
    p = op.importance_probabilities(x, d)
    expected = d.T @ x
    mean = sum(p[i]*op.gather_weight_gradient(x, d, torch.tensor([i]), p, 1) for i in range(3))
    # Absolute roundoff <= 8 eps * sum absolute contributions; fixed before assert.
    bound = 8*torch.finfo(torch.float64).eps*(d.abs().T@x.abs()).max().item()
    assert (mean-expected).abs().max() <= bound


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_reject_half(op, dtype):
    with pytest.raises(ValueError, match='dtype'):
        op.sampled_linear(torch.ones(3, 2, dtype=dtype), torch.ones(1, 2, dtype=dtype), None, op.SamplingState(1, 'dtype'))


def test_reject_amp_higher_order_and_empty(op):
    x, w, b, _ = fixture()
    x, w = x.requires_grad_(), w.requires_grad_()
    with torch.autocast('cpu'):
        with pytest.raises(RuntimeError, match='autocast|AMP'):
            op.sampled_linear(x, w, b, op.SamplingState(1, 'amp'))
    with pytest.raises(RuntimeError, match='higher'):
        torch.autograd.grad(op.sampled_linear(x, w, b, op.SamplingState(1, 'higher'), k=1).sum(), w, create_graph=True)
    with pytest.raises(ValueError, match='empty|rows'):
        op.sampled_linear(x[:0], w, b, op.SamplingState(1, 'empty'))


def test_named_streams_isolation_consumption_and_restore(op):
    expected = int.from_bytes(hashlib.sha256(json.dumps([19, 'sampling', 'L', 'importance'], separators=(',', ':'), ensure_ascii=True).encode('ascii')).digest()[:8], 'big') % (2**63-1)
    state = op.SamplingState(19, 'L')
    assert state.seed == expected == op.named_seed(19, 'sampling', 'L', 'importance')
    assert len({op.SamplingState(19, name, sampler=s).seed for name in ['L', 'M'] for s in ['importance', 'uniform']}) == 4
    global_before = torch.random.get_rng_state().clone()
    x, w, _, d = fixture()
    w.requires_grad_()
    snapshot = state.state_dict()
    first = torch.autograd.grad(op.sampled_linear(x, w, None, state, k=2), w, d)[0]
    after = state.generator.get_state().clone()
    p = op.importance_probabilities(x, d)
    reference_rng = torch.Generator().manual_seed(expected)
    torch.multinomial(p, 2, replacement=True, generator=reference_rng)
    assert torch.equal(after, reference_rng.get_state())
    state.load_state_dict(snapshot)
    again = torch.autograd.grad(op.sampled_linear(x, w, None, state, k=2), w, d)[0]
    close(first, again)
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert torch.equal(after, state.generator.get_state())


def test_uniform_direct_randint_and_no_score_dependency(op, monkeypatch):
    x, w, _, d = fixture()
    state = op.SamplingState(20, 'uniform', sampler='uniform')
    reference_rng = torch.Generator().manual_seed(state.seed)
    rows = torch.randint(5, (2,), generator=reference_rng)
    expected = sum(torch.outer(d[i], x[i]) for i in rows)*5/2
    def forbidden(*args): raise AssertionError('uniform evaluated scores')
    monkeypatch.setattr(op, 'importance_probabilities', forbidden)
    w.requires_grad_()
    actual = torch.autograd.grad(op.sampled_linear(x, w, None, state, k=2), w, d)[0]
    close(actual, expected)
    assert torch.equal(state.generator.get_state(), reference_rng.get_state())


def test_microbatch_valid_token_expectation_and_independent_samples(op):
    x = torch.tensor([[1.], [3.], [2.], [-1.], [5.]], dtype=torch.float64)
    d = torch.tensor([[2.], [-1.], [3.], [4.], [-2.]], dtype=torch.float64)
    # Each local mean is weighted by valid/total: D_local / local_n * local_n/5.
    full = d/5
    weighted_expectation = torch.zeros(1, 1, dtype=x.dtype)
    for start, stop in [(0, 2), (2, 5)]:
        xx, dd = x[start:stop], d[start:stop]/(stop-start)*(stop-start)/5
        p = op.importance_probabilities(xx, dd)
        weighted_expectation += sum(p[i]*op.gather_weight_gradient(xx, dd, torch.tensor([i]), p, 1) for i in range(stop-start))
    close(weighted_expectation, full.T@x)
    # Independent local draws have support differing from one concatenated draw.
    local_values = {float(d[i]*x[i]*2/5+d[j]*x[j]*3/5) for i in range(2) for j in range(2, 5)}
    concat_values = {float(d[i]*x[i]) for i in range(5)}
    assert local_values != concat_values
    w = torch.ones(1, 1, dtype=x.dtype, requires_grad=True)
    state = op.SamplingState(5, 'accum', sampler='uniform')
    for start, stop in [(0, 2), (2, 5)]:
        op.sampled_linear(x[start:stop], w, None, state, k=1).backward(full[start:stop])
    assert w.grad is not None


def test_shared_module_alias_storage_and_optimizer_exclusion(op):
    model = nn.ModuleDict({'a': nn.Linear(3, 2), 'b': nn.Linear(3, 2)})
    op.validate_sampled_parameters(model, ['a', 'b'], list(model.parameters()))
    with pytest.raises(ValueError, match='optimizer'):
        op.validate_sampled_parameters(model, ['a'], [*model.parameters(), model['a'].weight])
    with pytest.raises(ValueError, match='optimizer'):
        op.validate_sampled_parameters(model, ['a'], list(model.parameters())[:-1])
    model['b'].weight = model['a'].weight
    with pytest.raises(ValueError, match='shared|owner|alias'):
        op.validate_sampled_parameters(model, ['a'], list(model.parameters()))
    model['b'] = model['a']
    with pytest.raises(ValueError, match='shared|owner|alias|reused'):
        op.validate_sampled_parameters(model, ['a'], list(model.parameters()))
    storage = nn.ModuleDict({'a': nn.Linear(3, 2), 'b': nn.Linear(3, 2)})
    storage['b'].weight = nn.Parameter(storage['a'].weight.view_as(storage['a'].weight))
    with pytest.raises(ValueError, match='storage|shared|alias'):
        op.validate_sampled_parameters(storage, ['a'], list(storage.parameters()))
    # Exact embedding/output tie remains legal when neither owner is sampled.
    tied = nn.ModuleDict({'embed': nn.Embedding(3, 2), 'head': nn.Linear(2, 3, bias=False), 'a': nn.Linear(2, 2)})
    tied['head'].weight = tied['embed'].weight
    op.validate_sampled_parameters(tied, ['a'], list(tied.parameters()))


def test_dynamic_same_weight_in_one_backward_rejected(op):
    x, w, _, _ = fixture()
    w.requires_grad_()
    state = op.SamplingState(3, 'reused')
    a = op.sampled_linear(x, w, None, state, k=1)
    b = op.sampled_linear(x+1, w, None, state, k=1)
    with pytest.raises(RuntimeError, match='shared|reused|same weight'):
        (a+b).sum().backward()


def test_adamw_multistep_identical_injected_sample_reference(op):
    x, weight, bias, d = fixture()
    pairs = [[weight.clone().requires_grad_(), bias.clone().requires_grad_()] for _ in range(2)]
    opts = [torch.optim.AdamW([{'params': [p[0]], 'weight_decay': .07}, {'params': [p[1]], 'weight_decay': 0}], lr=.03, betas=(.8, .93), eps=.002, foreach=False) for p in pairs]
    for opt, params in zip(opts, pairs):
        for param in params:
            opt.state[param] = {'step': torch.tensor(3.), 'exp_avg': torch.full_like(param, .2), 'exp_avg_sq': torch.full_like(param, .4)}
    state = op.SamplingState(33, 'adam')
    reference_rng = torch.Generator().manual_seed(state.seed)
    for step in range(4):
        for opt in opts: opt.zero_grad(set_to_none=True)
        dd = d*(step+1)/4 if step != 2 else torch.zeros_like(d)
        op.sampled_linear(x, *pairs[0], state, k=2).backward(dd)
        if step == 2:
            pairs[1][0].grad = torch.zeros_like(weight)
        else:
            p = op.importance_probabilities(x, dd)
            rows = torch.multinomial(p, 2, replacement=True, generator=reference_rng)
            pairs[1][0].grad = sum(torch.outer(dd[i], x[i])/(2*p[i]) for i in rows)
        pairs[1][1].grad = dd.sum(0)
        for opt, params in zip(opts, pairs):
            torch.nn.utils.clip_grad_norm_(params, .7, error_if_nonfinite=True)
            opt.step()
        for a, b in zip(*pairs): close(a, b)
        for pa, pb in zip(*pairs):
            for key in ['step', 'exp_avg', 'exp_avg_sq']:
                close(opts[0].state[pa][key], opts[1].state[pb][key])
    assert torch.equal(state.generator.get_state(), reference_rng.get_state())


def test_unbiased_gradient_does_not_imply_unbiased_adam_displacement():
    # Equiprobable -1,3 has mean1. beta=0 makes first Adam displacement sign(g).
    displacements = []
    for grad in [-1., 3., 1.]:
        p = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        opt = torch.optim.AdamW([p], lr=.1, betas=(0., 0.), eps=.01, weight_decay=0)
        p.grad = torch.tensor(grad, dtype=p.dtype)
        opt.step()
        displacements.append(p.item())
    assert (-1+3)/2 == 1
    assert abs(sum(displacements[:2])/2-displacements[2]) > .09


@pytest.mark.parametrize('rho', [.125, .25, .5, 1.])
@pytest.mark.parametrize('n', [1, 2, 9])
def test_production_budget_rule_and_single_row_native(op, rho, n):
    x = torch.arange(1., n+1, dtype=torch.float64)[:, None]
    w = torch.ones(1, 1, dtype=x.dtype, requires_grad=True)
    d = torch.ones_like(x)
    state = op.SamplingState(17, 'budget', rho=rho)
    reference = torch.Generator().manual_seed(state.seed)
    k = max(1, math.ceil(rho*n))
    actual = torch.autograd.grad(op.sampled_linear(x, w, None, state), w, d)[0]
    if k >= n:
        close(actual, d.T@x)
    else:
        p = op.importance_probabilities(x, d)
        rows = torch.multinomial(p, k, replacement=True, generator=reference)
        expected = sum(torch.outer(d[i], x[i])/(k*p[i]) for i in rows)
        close(actual, expected)
    assert torch.equal(state.generator.get_state(), reference.get_state())


def test_probability_mixture_from_hand_computed_scores(op):
    # Exact normalized scores .5,1,0 yield independent probabilities 1/3,19/30,1/30.
    x = torch.tensor([[1.], [2.], [0.]], dtype=torch.float64)
    d = torch.tensor([[2.], [2.], [1.]], dtype=torch.float64)
    close(op.importance_probabilities(x, d), torch.tensor([1/3, 19/30, 1/30], dtype=torch.float64))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('sampler', ['importance', 'uniform'])
@pytest.mark.parametrize('scale', [1e-10, 1., 1e10])
def test_sampled_dw_matches_independent_outer_product_reference(op, dtype, sampler, scale):
    x, w, _, d = fixture(dtype=dtype)
    x = x*scale
    d = d/scale
    w.requires_grad_()
    state = op.SamplingState(71, 'outer', sampler=sampler)
    gen = torch.Generator().manual_seed(state.seed)
    p = op.importance_probabilities(x, d) if sampler == 'importance' else torch.full((5,), .2, dtype=torch.float64)
    rows = torch.multinomial(p, 4, replacement=True, generator=gen) if sampler == 'importance' else torch.randint(5, (4,), generator=gen)
    expected = sum(torch.outer(d[i].double(), x[i].double())/(4*p[i]) for i in rows)
    actual = torch.autograd.grad(op.sampled_linear(x, w, None, state, k=4), w, d)[0]
    torch.testing.assert_close(actual.double(), expected, rtol=1e-4 if dtype == torch.float32 else 1e-10, atol=1e-6 if dtype == torch.float32 else 1e-12)


def test_independent_microbatch_draws_match_precomputed_accumulation(op):
    x, w, _, d = fixture()
    w.requires_grad_()
    state = op.SamplingState(82, 'local', sampler='uniform')
    gen = torch.Generator().manual_seed(state.seed)
    expected = torch.zeros_like(w)
    for lo, hi in [(0, 2), (2, 5)]:
        rows = torch.randint(hi-lo, (1,), generator=gen)+lo
        # d/5 is valid-token weighted local mean in each unequal partition.
        expected += torch.outer(d[rows[0]]/5, x[rows[0]])*(hi-lo)
        op.sampled_linear(x[lo:hi], w, None, state, k=1).backward(d[lo:hi]/5)
    close(w.grad, expected)
    assert torch.equal(state.generator.get_state(), gen.get_state())


def test_overflow_draw_is_consumed_once_without_retry(op):
    x = torch.full((3, 1), 1e30)
    d = torch.full_like(x, 1e30)
    w = torch.ones(1, 1, requires_grad=True)
    state = op.SamplingState(1, 'no-retry')
    gen = torch.Generator().manual_seed(state.seed)
    torch.multinomial(torch.full((3,), 1/3, dtype=torch.float64), 1, replacement=True, generator=gen)
    with pytest.raises(RuntimeError, match='finite'):
        op.sampled_linear(x, w, None, state, k=1).backward(d)
    assert torch.equal(state.generator.get_state(), gen.get_state())


@pytest.mark.parametrize('derivative', ['x', 'bias'])
def test_zero_dw_still_checks_exact_derivative_overflow(op, derivative):
    x = torch.zeros(5, 1, requires_grad=True)
    w = torch.tensor([[1e30 if derivative == 'x' else 0.]], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    d = torch.full((5, 1), 1e30 if derivative == 'x' else 1e38)
    state = op.SamplingState(4, 'exact-overflow')
    before = state.generator.get_state()
    with pytest.raises(RuntimeError, match='finite'):
        op.sampled_linear(x, w, b, state, k=1).backward(d)
    assert torch.equal(before, state.generator.get_state())


def test_state_restore_rejects_different_named_stream_and_does_not_alias(op):
    a = op.SamplingState(5, 'a')
    b = op.SamplingState(5, 'b')
    with pytest.raises(ValueError, match='metadata'):
        b.load_state_dict(a.state_dict())
    snapshot = a.state_dict()
    initial = a.generator.get_state().clone()
    snapshot['generator_state'].zero_()
    assert torch.equal(a.generator.get_state(), initial)


@pytest.mark.parametrize('k', [0, -1, 1.5, True])
def test_reject_invalid_explicit_budget(op, k):
    x, w, b, _ = fixture()
    with pytest.raises(ValueError, match='k'):
        op.sampled_linear(x, w, b, op.SamplingState(1, 'invalid'), k=k)


@pytest.mark.parametrize('kwargs', [{'sampler': 'other'}, {'rho': .3}])
def test_reject_unsupported_sampler_or_ratio(op, kwargs):
    with pytest.raises(ValueError):
        op.SamplingState(1, 'invalid', **kwargs)


def test_distinct_state_objects_cannot_evade_shared_weight_detection(op):
    x, w, _, _ = fixture()
    w.requires_grad_()
    a = op.sampled_linear(x, w, None, op.SamplingState(1, 'a'), k=1)
    b = op.sampled_linear(x, w, None, op.SamplingState(1, 'b'), k=1)
    with pytest.raises(RuntimeError, match='shared|reused|same weight'):
        (a+b).sum().backward()


def test_missing_backward_task_primitive_fails_visibly(op, monkeypatch):
    x, w, _, _ = fixture()
    w.requires_grad_()
    monkeypatch.delattr(torch._C, '_current_graph_task_id')
    with pytest.raises(RuntimeError, match='primitive unavailable'):
        op.sampled_linear(x, w, None, op.SamplingState(1, 'missing'), k=1).sum().backward()


def test_zero_tensor_gradient_and_none_have_different_adamw_effects():
    zero, absent = [torch.tensor(1., requires_grad=True) for _ in range(2)]
    opts = [torch.optim.AdamW([p], lr=.1, betas=(.8, .9), eps=.01, weight_decay=.1) for p in (zero, absent)]
    for opt, p in zip(opts, (zero, absent)):
        opt.state[p] = {'step': torch.tensor(2.), 'exp_avg': torch.tensor(.3), 'exp_avg_sq': torch.tensor(.4)}
    zero.grad = torch.zeros_like(zero)
    for opt in opts: opt.step()
    assert zero.item() < 1 and absent.item() == 1
    assert opts[0].state[zero]['step'] == 3
    assert opts[1].state[absent]['step'] == 2


@pytest.mark.parametrize('n', [3, 8192])
def test_actual_fp32_gather_inverse_weight_rounding_bound(op, n):
    scoring_x = torch.ones(n, 1)
    scoring_x[0] = 1e30
    scoring_d = torch.ones_like(scoring_x)
    p = op.importance_probabilities(scoring_x, scoring_d)
    # Unit operands isolate the two cast inverse-square-root scales in actual GEMM.
    unit = torch.ones(n, 1)
    for row in [0, n-1]:
        got = op.gather_weight_gradient(unit, unit, torch.tensor([row]), p, 1).double().item()
        assert abs(p[row].item()*got-1) <= 8*torch.finfo(torch.float32).eps


@pytest.mark.parametrize('sampler', ['importance', 'uniform'])
def test_zero_operand_does_not_hide_nonfinite_other_operand(op, sampler):
    x = torch.zeros(3, 1)
    d = torch.tensor([[float('nan')], [1.], [1.]])
    w = torch.ones(1, 1, requires_grad=True)
    state = op.SamplingState(9, 'zero-with-nan', sampler=sampler)
    before = state.generator.get_state()
    with pytest.raises(RuntimeError, match='finite'):
        op.sampled_linear(x, w, None, state, k=1).backward(d)
    assert torch.equal(before, state.generator.get_state())


def test_registry_releases_dead_weights_and_keeps_only_last_task(op):
    import gc
    import weakref
    x, w, _, _ = fixture()
    w.requires_grad_()
    state = op.SamplingState(1, 'lifecycle')
    key, ref = id(w), weakref.ref(w)
    for _ in range(3):
        op.sampled_linear(x, w, None, state, k=1).sum().backward()
        assert key in op._last_backward
    del w
    gc.collect()
    assert ref() is None
    assert key not in op._last_backward
