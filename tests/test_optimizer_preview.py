"""Independent native AdamW oracles catch optimizer and transaction defects."""

import random

import numpy as np
import pytest
import torch

from oar.optimizer_preview import commit_pending, preview_adamw
from oar.records import AdamWGroup, AdamWParamState, from_payload, to_payload


def fixture(device='cpu', dtype=torch.float64, eps=1e-8, decay=.03):
    # Strided matrix, zero gradient, skipped initialized/uninitialized states.
    p = {'w': torch.tensor([[1., -2.], [3., 4.]], dtype=dtype, device=device).T,
         'b': torch.tensor([.5, -.3], dtype=dtype, device=device),
         'skip': torch.ones(2, dtype=dtype, device=device),
         'new': torch.ones(2, dtype=dtype, device=device)}
    p = {k: v.requires_grad_() for k, v in p.items()}
    grads = {'w': torch.tensor([[.1, -.8], [3., .02]], dtype=dtype, device=device).T,
             'b': torch.zeros_like(p['b']), 'skip': None, 'new': None}
    state = {k: AdamWParamState(step, torch.full_like(p[k], .4),
                                torch.tensor([1e-8, 1e4], dtype=dtype, device=device).expand_as(p[k]).clone())
             for k, step in [('w', 100), ('b', 3), ('skip', 8)]}
    groups = (AdamWGroup(('w', 'skip'), .03, (.9, .999), eps, decay),
              AdamWGroup(('b', 'new'), .07, (.5, .98), eps, 0.))
    return p, grads, state, groups


def native_copy(p, state, groups):
    params = {k: v.detach().clone().requires_grad_() for k, v in p.items()}
    opt = torch.optim.AdamW([dict(params=[params[k] for k in g.parameter_ids], lr=g.lr,
                                  betas=g.betas, eps=g.eps, weight_decay=g.weight_decay)
                             for g in groups], foreach=False, fused=False)
    for k, s in state.items():
        opt.state[params[k]] = dict(step=torch.tensor(float(s.step)),
                                   exp_avg=s.exp_avg.clone(), exp_avg_sq=s.exp_avg_sq.clone())
    return params, opt


def native_step(params, opt, grads, groups, clip):
    for k, p in params.items():
        p.grad = None if grads[k] is None else grads[k].detach().clone()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_([params[k] for g in groups for k in g.parameter_ids],
                                      clip, foreach=False, error_if_nonfinite=True)
    opt.step()


def snapshot(p, grads, state):
    return ({k: (v.clone(), v._version) for k, v in p.items()},
            {k: None if v is None else (v.clone(), v._version) for k, v in grads.items()},
            {k: (s.step, s.exp_avg.clone(), s.exp_avg_sq.clone(), s.exp_avg._version,
                 s.exp_avg_sq._version) for k, s in state.items()})


def assert_snapshot(saved, p, grads, state):
    after = snapshot(p, grads, state)
    for first, second in zip(saved, after):
        assert first.keys() == second.keys()
        for k in first:
            if first[k] is None:
                assert second[k] is None
            else:
                for a, b in zip(first[k], second[k]):
                    if isinstance(a, torch.Tensor):
                        assert torch.equal(a, b)
                    else:
                        assert a == b


@pytest.mark.parametrize('clip', [None, 0., 1e-7, .5, 1e9])
@pytest.mark.parametrize('eps,decay', [(1e-8, .03), (.2, 0.)])
def test_multistep_native_equivalence_and_no_preview_mutation(clip, eps, decay):
    p, grads, state, groups = fixture(eps=eps, decay=decay)
    refs, opt = native_copy(p, state, groups)
    pv, sv = 4, 9
    for iteration in range(4):
        if iteration == 1:
            grads['new'] = torch.tensor([.7, -.9], dtype=torch.float64)
            grads['w'] = None
        if iteration == 2:
            grads['w'] = torch.zeros_like(p['w'])
            grads['b'] = None
        saved = snapshot(p, grads, state)
        rng = torch.get_rng_state().clone()
        nrng, prng = np.random.get_state(), random.getstate()
        pending = preview_adamw(p, grads, state, groups, pv, sv, clip)
        assert_snapshot(saved, p, grads, state)
        assert torch.equal(rng, torch.get_rng_state())
        assert np.array_equal(nrng[1], np.random.get_state()[1])
        assert nrng[2:] == np.random.get_state()[2:] and prng == random.getstate()
        active = {k for k, v in grads.items() if v is not None}
        assert pending.displacements.keys() == pending.optimizer_state.keys() == active
        assert pending.candidate_parameters.keys() == active
        alignment = sum((grads[k] * (p[k] - pending.candidate_parameters[k])).sum().item() for k in active)
        assert pending.raw_alignment == pytest.approx(alignment, rel=1e-12, abs=1e-14)
        for k in active:
            assert torch.equal(pending.displacements[k], p[k] - pending.candidate_parameters[k])
            assert not pending.candidate_parameters[k].requires_grad
        pv, sv = commit_pending(pending, p, state, pv, sv)
        assert (pv, sv) == (5 + iteration, 10 + iteration)
        native_step(refs, opt, grads, groups, clip)
        for k in p:
            torch.testing.assert_close(p[k], refs[k], rtol=1e-10, atol=1e-12)
            rs = opt.state.get(refs[k])
            if rs:
                assert state[k].step == int(rs['step'])
                torch.testing.assert_close(state[k].exp_avg, rs['exp_avg'], rtol=1e-10, atol=1e-12)
                torch.testing.assert_close(state[k].exp_avg_sq, rs['exp_avg_sq'], rtol=1e-10, atol=1e-12)
            else:
                assert k not in state
        with pytest.raises(RuntimeError, match='consumed'):
            commit_pending(pending, p, state, pv, sv)


@pytest.mark.parametrize('target', ['parameter_version', 'state_version', 'other_parameters',
                                   'other_state', 'parameter_mutation', 'moment_mutation',
                                   'candidate_mutation', 'restored_pending'])
def test_invalid_commit_leaves_all_live_values_untouched(target):
    p, grads, state, groups = fixture()
    pending = preview_adamw(p, grads, state, groups, 4, 9, None)
    pv, sv = 4, 9
    if target == 'parameter_version':
        pv += 1
    elif target == 'state_version':
        sv += 1
    elif target == 'other_parameters':
        p = {k: v.detach().clone() for k, v in p.items()}
    elif target == 'other_state':
        state = dict(state)
    elif target == 'parameter_mutation':
        with torch.no_grad():
            p['w'].add_(1)
    elif target == 'moment_mutation':
        state['w'].exp_avg.add_(1)
    elif target == 'candidate_mutation':
        pending.candidate_parameters['b'].add_(1)
    elif target == 'restored_pending':
        pending = from_payload(to_payload(pending))
    saved = snapshot(p, grads, state)
    with pytest.raises(RuntimeError, match='version|owner|modified'):
        commit_pending(pending, p, state, pv, sv)
    assert_snapshot(saved, p, grads, state)
    assert not pending.consumed


def test_rejected_preview_can_be_discarded_without_advancing_state():
    p, grads, state, groups = fixture()
    first = preview_adamw(p, grads, state, groups, 0, 0, .5)
    second = preview_adamw(p, grads, state, groups, 0, 0, .5)
    for k in first.displacements:
        assert torch.equal(first.candidate_parameters[k], second.candidate_parameters[k])
    commit_pending(second, p, state, 0, 0)
    with pytest.raises(RuntimeError, match='modified|version'):
        commit_pending(first, p, state, 0, 0)


@pytest.mark.parametrize('project', [False, True])
def test_high_cosine_adam_counterexample_including_orthogonal_projection(project):
    g = torch.tensor([1., .02], dtype=torch.float64)
    direction = torch.tensor([1., -.02], dtype=torch.float64)
    if project:
        u = direction / direction.norm()
        approximation = u * (u @ g)
        torch.testing.assert_close(u @ (g - approximation), torch.tensor(0., dtype=g.dtype), atol=1e-15, rtol=0)
    else:
        approximation = direction
    p = {'w': torch.zeros(2, dtype=torch.float64)}
    # Appendix A uses the updated counter t=100: native pre-step state is 99.
    state = {'w': AdamWParamState(99, torch.zeros(2, dtype=torch.float64),
                                 torch.tensor([1e4, 1e-8], dtype=torch.float64))}
    groups = (AdamWGroup(('w',), 1., (.9, .999), 1e-8, 0.),)
    pending = preview_adamw(p, {'w': approximation}, state, groups, 0, 0, None)
    refs, opt = native_copy(p, state, groups)
    native_step(refs, opt, {'w': approximation}, groups, None)
    torch.testing.assert_close(pending.displacements['w'], -refs['w'], rtol=1e-12, atol=1e-14)
    assert (g @ approximation / (g.norm() * approximation.norm())).item() > .999
    assert (g @ pending.displacements['w']).item() < 0 < pending.raw_alignment
    if not project:
        torch.testing.assert_close(pending.displacements['w'], torch.tensor([.0003087203, -.9638050377], dtype=g.dtype), rtol=1e-7, atol=1e-10)


def test_fp32_displacement_uses_stored_candidate_rounding():
    p = {'w': torch.tensor([1e8, 1.], requires_grad=True)}
    grads = {'w': torch.tensor([1., 1.])}
    groups = (AdamWGroup(('w',), 1e-5, (.9, .999), 1e-8, .1),)
    pending = preview_adamw(p, grads, {}, groups, 0, 0, None)
    refs, opt = native_copy(p, {}, groups)
    native_step(refs, opt, grads, groups, None)
    assert torch.equal(pending.candidate_parameters['w'], refs['w'])
    assert torch.equal(pending.displacements['w'], p['w'] - refs['w'])
    # Decay at 1e8 rounds to a multiple of the FP32 spacing (8), not nominal 100.
    assert pending.displacements['w'][0].item() == 104.


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires actual visible MPS')
@pytest.mark.parametrize('clip', [None, .5, 0.])
def test_mps_fp32_native_equivalence_and_rng_preservation(clip):
    p, grads, state, groups = fixture(device='mps', dtype=torch.float32)
    refs, opt = native_copy(p, state, groups)
    for i in range(3):
        if i == 1:
            grads['new'] = torch.tensor([.7, -.9], device='mps')
            grads['w'] = None
        if i == 2:
            grads['w'] = torch.zeros_like(p['w'])
            grads['b'] = None
        saved = snapshot(p, grads, state)
        rng = torch.mps.get_rng_state().clone()
        pending = preview_adamw(p, grads, state, groups, i, i, clip)
        assert torch.equal(rng, torch.mps.get_rng_state())
        assert_snapshot(saved, p, grads, state)
        commit_pending(pending, p, state, i, i)
        native_step(refs, opt, grads, groups, clip)
        for k in p:
            torch.testing.assert_close(p[k], refs[k], rtol=2e-6, atol=2e-7)
            if k in state:
                rs = opt.state[refs[k]]
                assert state[k].step == int(rs['step'])
                torch.testing.assert_close(state[k].exp_avg, rs['exp_avg'], rtol=2e-6, atol=2e-7)
                torch.testing.assert_close(state[k].exp_avg_sq, rs['exp_avg_sq'], rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('bad', ['duplicate', 'unassigned', 'unknown_gradient', 'shape',
                                 'dtype', 'negative_variance', 'bad_step', 'bad_lr',
                                 'bad_beta', 'bad_clip', 'nonfinite_gradient', 'half'])
def test_invalid_preview_fails_without_mutating_live_state(bad):
    p, grads, state, groups = fixture()
    if bad == 'duplicate':
        groups += (groups[0],)
    elif bad == 'unassigned':
        groups = groups[:1]
    elif bad == 'unknown_gradient':
        grads['typo'] = torch.ones(2, dtype=torch.float64)
    elif bad == 'shape':
        grads['w'] = torch.ones(2, dtype=torch.float64)
    elif bad == 'dtype':
        grads['w'] = grads['w'].float()
    elif bad == 'negative_variance':
        state['w'].exp_avg_sq.fill_(-1)
    elif bad == 'bad_step':
        state['w'] = AdamWParamState(-1, state['w'].exp_avg, state['w'].exp_avg_sq)
    elif bad == 'bad_lr':
        groups = (AdamWGroup(groups[0].parameter_ids, -1., (.9, .999), 1e-8, 0.), groups[1])
    elif bad == 'bad_beta':
        groups = (AdamWGroup(groups[0].parameter_ids, .1, (1., .999), 1e-8, 0.), groups[1])
    elif bad == 'nonfinite_gradient':
        grads['w'].fill_(float('inf'))
    elif bad == 'half':
        p['new'] = p['new'].half()
    saved = snapshot(p, grads, state)
    with pytest.raises(ValueError):
        preview_adamw(p, grads, state, groups, 0, 0, float('nan') if bad == 'bad_clip' else None)
    assert_snapshot(saved, p, grads, state)


def test_none_gradient_preserves_state_while_zero_applies_decay_and_momentum():
    p = {'none': torch.tensor([2.], dtype=torch.float64), 'zero': torch.tensor([2.], dtype=torch.float64)}
    state = {k: AdamWParamState(10, torch.tensor([1.], dtype=torch.float64),
                                torch.tensor([1.], dtype=torch.float64)) for k in p}
    old_none = state['none']
    groups = (AdamWGroup(tuple(p), .1, (.9, .999), 1e-8, .1),)
    pending = preview_adamw(p, {'none': None, 'zero': torch.zeros_like(p['zero'])}, state, groups, 0, 0, None)
    commit_pending(pending, p, state, 0, 0)
    assert p['none'].item() == 2 and state['none'] is old_none
    assert p['zero'].item() < 1.98 and state['zero'].step == 11
    assert pending.raw_alignment == 0


def test_clipping_stabilizer_changes_submicro_gradient_with_large_epsilon():
    p = {'w': torch.zeros(1, dtype=torch.float64)}
    grads = {'w': torch.tensor([1e-7], dtype=torch.float64)}
    group = (AdamWGroup(('w',), 1., (0., 0.), 1., 0.),)
    pending = preview_adamw(p, grads, {}, group, 0, 0, 1e-7)
    # Native factor is 1e-7 / (1e-7 + 1e-6) = 1/11, not 1.
    assert pending.optimizer_state['w'].exp_avg.item() == pytest.approx(1e-7 / 11, rel=1e-14)
    assert pending.raw_alignment == pytest.approx(1e-7 * (1e-7 / 11) / (1 + 1e-7 / 11), rel=1e-14)


def test_native_float32_counter_rounding_at_precision_boundary():
    p = {'w': torch.zeros(1, dtype=torch.float64)}
    grads = {'w': torch.ones(1, dtype=torch.float64)}
    state = {'w': AdamWParamState(2**24, torch.zeros_like(p['w']), torch.ones_like(p['w']))}
    groups = (AdamWGroup(('w',), 1., (.9, .99999999), 1e-8, 0.),)
    refs, opt = native_copy(p, state, groups)
    pending = preview_adamw(p, grads, state, groups, 0, 0, None)
    native_step(refs, opt, grads, groups, None)
    # Noncapturable native AdamW's CPU FP32 scalar cannot represent 2**24 + 1.
    assert pending.optimizer_state['w'].step == int(opt.state[refs['w']]['step'])
    torch.testing.assert_close(pending.candidate_parameters['w'], refs['w'], rtol=1e-12, atol=1e-14)


def test_shared_parameter_storage_is_rejected_before_ambiguous_preview():
    storage = torch.ones(4, dtype=torch.float64)
    p = {'a': storage[:2], 'b': storage[1:3]}
    groups = (AdamWGroup(tuple(p), .1, (.9, .999), 1e-8, 0.),)
    with pytest.raises(ValueError, match='storage|alias'):
        preview_adamw(p, {k: torch.ones_like(v) for k, v in p.items()}, {}, groups, 0, 0, None)
    assert torch.equal(storage, torch.ones_like(storage))


@pytest.mark.parametrize('omit', [False, True])
def test_all_missing_gradients_preserve_live_gradients_parameters_and_state(omit):
    p, grads, state, groups = fixture()
    for v in p.values():
        v.grad = torch.full_like(v, 42.)
    grads = {} if omit else {k: None for k in p}
    saved = snapshot(p, grads, state)
    pending = preview_adamw(p, grads, state, groups, 0, 0, .5)
    assert not pending.candidate_parameters and not pending.optimizer_state and not pending.displacements
    assert pending.raw_alignment == 0
    commit_pending(pending, p, state, 0, 0)
    assert_snapshot(saved, p, grads, state)
    assert all(torch.equal(v.grad, torch.full_like(v, 42.)) for v in p.values())
