"""Reproduce tiny AdamW correctness evidence; no training or timing claims.

Run from the repository root in a fresh process on the actual machine:
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python analysis/collect_optimizer_correctness.py
--device cpu can produce a CPU-only artifact, which cannot satisfy the MPS gate.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from oar.optimizer_preview import commit_pending, preview_adamw
from oar.records import AdamWGroup, AdamWParamState


def case(device, dtype, clip, eps):
    p = {'matrix': torch.tensor([[1., -2.], [3., 4.]], device=device, dtype=dtype).T.requires_grad_(),
         'bias': torch.tensor([.5, -.3], device=device, dtype=dtype).requires_grad_(),
         'late': torch.ones(2, device=device, dtype=dtype).requires_grad_()}
    raw = {'matrix': torch.tensor([[.1, -.8], [3., .02]], device=device, dtype=dtype).T,
           'bias': torch.zeros_like(p['bias']), 'late': None}
    state = {'matrix': AdamWParamState(100, torch.full_like(p['matrix'], .4),
                                      torch.tensor([1e-8, 1e4], device=device, dtype=dtype).expand_as(p['matrix']).clone())}
    groups = (AdamWGroup(('matrix',), .03, (.9, .999), eps, .03),
              AdamWGroup(('bias', 'late'), .07, (.5, .98), eps, 0.))
    ref = {k: v.detach().clone().requires_grad_() for k, v in p.items()}
    native = torch.optim.AdamW([dict(params=[ref[k] for k in g.parameter_ids], lr=g.lr,
                                     betas=g.betas, eps=g.eps, weight_decay=g.weight_decay)
                                for g in groups], foreach=False, fused=False)
    for k, s in state.items():
        native.state[ref[k]] = dict(step=torch.tensor(float(s.step)), exp_avg=s.exp_avg.clone(),
                                    exp_avg_sq=s.exp_avg_sq.clone())
    errors = dict(parameters=0., first_moments=0., second_moments=0., displacements=0., raw_alignment=0.)
    normalized = dict(errors)
    rng_cpu = torch.get_rng_state().clone()
    rng_mps = torch.mps.get_rng_state().clone() if device == 'mps' else None
    for step in range(4):
        if step == 1:
            raw['late'] = torch.tensor([.7, -.9], device=device, dtype=dtype)
            raw['matrix'] = None
        if step == 2:
            raw['matrix'] = torch.zeros_like(p['matrix'])
            raw['bias'] = None
        before = {k: v.detach().clone() for k, v in p.items()}
        moments = {k: (v.exp_avg.clone(), v.exp_avg_sq.clone()) for k, v in state.items()}
        raw_before = {k: None if v is None else v.clone() for k, v in raw.items()}
        pending = preview_adamw(p, raw, state, groups, step, step, clip)
        assert all(torch.equal(p[k], v) for k, v in before.items())
        assert all(torch.equal(state[k].exp_avg, m) and torch.equal(state[k].exp_avg_sq, v)
                   for k, (m, v) in moments.items())
        assert all(v is None or torch.equal(raw[k], v) for k, v in raw_before.items())
        for k, v in raw.items():
            ref[k].grad = None if v is None else v.clone()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(list(ref.values()), clip, foreach=False, error_if_nonfinite=True)
        native.step()
        expected_alignment = 0.
        for k, v in pending.displacements.items():
            expected_d = before[k] - ref[k].detach()
            error = (v - expected_d).abs().max().item()
            errors['displacements'] = max(errors['displacements'], error)
            normalized['displacements'] = max(normalized['displacements'], error / max(expected_d.abs().max().item(), 1e-30))
            torch.testing.assert_close(v, expected_d, rtol=1e-10 if dtype == torch.float64 else 2e-6,
                                       atol=1e-12 if dtype == torch.float64 else 2e-7)
            expected_alignment += (raw[k].to('cpu').double() * expected_d.to('cpu').double()).sum().item()
        errors['raw_alignment'] = max(errors['raw_alignment'], abs(pending.raw_alignment - expected_alignment))
        normalized['raw_alignment'] = max(normalized['raw_alignment'], abs(pending.raw_alignment - expected_alignment) / max(abs(expected_alignment), 1e-30))
        commit_pending(pending, p, state, step, step)
        for k in p:
            pairs = [('parameters', p[k], ref[k])]
            if k in state:
                ns = native.state[ref[k]]
                assert state[k].step == int(ns['step'])
                pairs += [('first_moments', state[k].exp_avg, ns['exp_avg']),
                          ('second_moments', state[k].exp_avg_sq, ns['exp_avg_sq'])]
            for name, actual, expected in pairs:
                error = (actual - expected).abs().max().item()
                errors[name] = max(errors[name], error)
                normalized[name] = max(normalized[name], error / max(expected.abs().max().item(), 1e-30))
                torch.testing.assert_close(actual, expected, rtol=1e-10 if dtype == torch.float64 else 2e-6,
                                           atol=1e-12 if dtype == torch.float64 else 2e-7)
    assert torch.equal(rng_cpu, torch.get_rng_state())
    if rng_mps is not None:
        assert torch.equal(rng_mps, torch.mps.get_rng_state())
    return dict(device=device, dtype=str(dtype), steps=4, clip=clip, eps=eps,
                max_absolute_errors=errors, max_error_over_reference_maxnorm=normalized,
                per_parameter_counts_match=True, previews_preserve_sources_and_rng=True,
                rtol=1e-10 if dtype == torch.float64 else 2e-6,
                atol=1e-12 if dtype == torch.float64 else 2e-7)


def counterexample(project):
    g = torch.tensor([1., .02], dtype=torch.float64)
    gh = torch.tensor([1., -.02], dtype=torch.float64)
    if project:
        u = gh / gh.norm()
        gh = u * (u @ g)
    p = {'w': torch.zeros_like(g)}
    state = {'w': AdamWParamState(99, torch.zeros_like(g), torch.tensor([1e4, 1e-8], dtype=g.dtype))}
    groups = (AdamWGroup(('w',), 1., (.9, .999), 1e-8, 0.),)
    pending = preview_adamw(p, {'w': gh}, state, groups, 0, 0, None)
    return dict(orthogonal_projection=project, committed_counter=99, updated_counter=100,
                cosine=(g @ gh / (g.norm() * gh.norm())).item(),
                true_alignment=(g @ pending.displacements['w']).item(),
                approximate_alignment=pending.raw_alignment,
                displacement=pending.displacements['w'].tolist(),
                projection_residual_dot=None if not project else (u @ (g-gh)).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['all', 'cpu'], default='all')
    parser.add_argument('--output', type=Path, default=Path('analysis/optimizer_correctness.json'))
    args = parser.parse_args()
    devices = [('cpu', torch.float64), ('cpu', torch.float32)]
    if args.device == 'all':
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or not torch.backends.mps.is_available():
            raise RuntimeError('all-device evidence requires actual MPS and fallback=0 before interpreter startup')
        devices.append(('mps', torch.float32))
    rows = [case(device, dtype, clip, eps) for device, dtype in devices
            for clip in (None, .5, 0.) for eps in (1e-8, .2)]
    # Full participation and initialized state: count unique tensor payload bytes,
    # not RSS or a peak-memory benchmark. Live parameters/gradients/moments=16P;
    # owned pending candidates/displacements/moments=16P, hence retained=32P.
    p = {'w': torch.ones(16)}
    raw = {'w': torch.ones(16)}
    state = {'w': AdamWParamState(2, torch.ones(16), torch.ones(16))}
    pending = preview_adamw(p, raw, state, (AdamWGroup(('w',), .1, (.9,.999), 1e-8, .01),), 0, 0, None)
    payloads = [*pending.candidate_parameters.values(), *pending.displacements.values(),
                *(s.exp_avg for s in pending.optimizer_state.values()),
                *(s.exp_avg_sq for s in pending.optimizer_state.values())]
    pending_bytes = sum(t.numel() * t.element_size() for t in payloads)
    sources = ['src/oar/optimizer_preview.py', 'src/oar/records.py', 'tests/test_optimizer_preview.py',
               'analysis/collect_optimizer_correctness.py']
    report = dict(schema_version=1, torch_version=torch.__version__,
                  scope='tiny native single-tensor AdamW correctness; no trainer/replay/performance qualification',
                  command='PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python analysis/collect_optimizer_correctness.py',
                  caller_fallback_environment=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
                  cases=rows, counterexamples=[counterexample(False), counterexample(True)],
                  retained_fp32_storage=dict(parameters=16, pending_payload_bytes=pending_bytes,
                                             pending_bytes_per_parameter=pending_bytes / 16,
                                             live_plus_pending_bytes_per_parameter=16 + pending_bytes / 16,
                                             excludes='activations, workspaces, clipping copies, denominators, CPU FP64 reduction temporaries; not peak memory'),
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources})
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'{len(rows)} native reference cases; {args.output}')


if __name__ == '__main__':
    main()
