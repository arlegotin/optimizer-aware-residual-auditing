"""Collect tiny numerical evidence, never performance or gate qualification."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess

import torch
from torch.nn import functional as F

from oar.basis import build_basis
from oar.linear import LinearCapture, projected_linear


def cpu64(t):
    return t.detach().cpu().double()


def error(actual, expected):
    difference = cpu64(actual) - cpu64(expected)
    return {'max_abs': difference.abs().max().item() if difference.numel() else 0.0,
            'frobenius': torch.linalg.vector_norm(difference).item()}


assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
assert torch.backends.mps.is_available()
out = {
    'schema_version': 1,
    'scope': 'tiny operator correctness only; no performance, training, guardband or complete WP01 claim',
    'captured_at_utc': datetime.now(timezone.utc).isoformat(),
    'provenance': {'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   'python': platform.python_version(), 'machine': platform.machine(),
                   'torch': torch.__version__, 'torch_revision': torch.version.git_version,
                   'mps_available': True, 'fallback_environment_before_import': '0'},
    'operator_cases': [], 'basis_cases': [],
}
for device, dtype in [('cpu', torch.float64), ('mps', torch.float32)]:
    for rank in (0, 2, 5):
        for bias in (False, True):
            g = torch.Generator().manual_seed(409)
            xr = torch.randn(4, 3, 7, generator=g, dtype=torch.float64).transpose(0, 1).requires_grad_()
            wr = torch.randn(5, 7, generator=g, dtype=torch.float64).requires_grad_()
            br = torch.randn(5, generator=g, dtype=torch.float64).requires_grad_() if bias else None
            u = torch.linalg.qr(torch.randn(5, 5, generator=g, dtype=torch.float64)).Q[:, :rank]
            z = torch.randn(7, 6, generator=g, dtype=torch.float64)
            d = torch.randn(3, 4, 5, generator=g, dtype=torch.float64)
            exact = torch.autograd.grad(F.linear(xr, wr, br), (xr, wr, br) if bias else (xr, wr), d)
            x = xr.detach().to(dtype=dtype, device=device).requires_grad_()
            w = wr.detach().to(dtype=dtype, device=device).requires_grad_()
            b = br.detach().to(dtype=dtype, device=device).requires_grad_() if bias else None
            capture = LinearCapture(u.to(dtype=dtype, device=device), z.to(dtype=dtype, device=device), 3)
            y = projected_linear(x, w, b, capture)
            actual = torch.autograd.grad(y, (x, w, b) if bias else (x, w), d.to(dtype=dtype, device=device))
            projected = u @ (u.T @ exact[1])
            refs = {'forward': F.linear(xr, wr, br), 'dx': exact[0], 'dw': projected,
                    'h': u.T @ exact[1], 'residual': (exact[1] - projected) @ z}
            values = {'forward': y, 'dx': actual[0], 'dw': actual[1], 'h': capture.h_sum, 'residual': capture.r_sum}
            if bias:
                refs['db'], values['db'] = exact[2], actual[2]
            rtol, atol = (1e-10, 1e-12) if device == 'cpu' else (2e-5, 1e-5)
            for key in values:
                torch.testing.assert_close(cpu64(values[key]), cpu64(refs[key]), rtol=rtol, atol=atol)
            out['operator_cases'].append({'device': device, 'dtype': str(dtype), 'seed': 409,
                'x_shape': list(x.shape), 'weight_shape': [5, 7], 'rank': rank, 'probe_count': 6,
                'bias': bias, 'x_noncontiguous': not x.is_contiguous(), 'rtol': rtol, 'atol': atol,
                'reference': 'native dense CPU FP64 autograd; independently materialized projector/residual',
                'errors': {key: error(values[key], refs[key]) for key in values}})
    for nonzero_rank in (0, 1, 2):
        g = torch.Generator().manual_seed(705)
        a = torch.randn(8, nonzero_rank, generator=g, dtype=dtype)
        b = torch.randn(nonzero_rank, 6, generator=g, dtype=dtype)
        gradient = (a @ b).to(device)
        state = build_basis(gradient, 5, 8, g, 0)
        assert state.numerical_rank == nonzero_rank
        u, ref = cpu64(state.vectors), cpu64(gradient)
        out['basis_cases'].append({'device': device, 'dtype': str(dtype), 'seed': 705,
            'gradient_shape': [8, 6], 'fixture_rank': nonzero_rank, 'max_rank': 5, 'oversampling': 8,
            'numerical_rank': state.numerical_rank, 'stored_rank': state.vectors.shape[1],
            'orthogonality': state.orthogonality, 'reconstruction_error': error(u @ (u.T @ ref), ref),
            'construction_width': state.construction_width, 'construction_wall_s': state.construction_wall_s,
            'transfer_bytes': state.transfer_bytes, 'decomposition_device': state.decomposition_device,
            'decomposition_dtype': state.decomposition_dtype,
            'timing_interpretation': 'single inclusive diagnostic, not a benchmark'})

accum = runpy.run_path('tests/test_accumulation.py')['accumulated_case'].__wrapped__()
out['accumulation'] = {'seed': 502, 'device': 'cpu', 'dtype': 'float64',
    'microbatch_sizes': [2, 5, 3, 8, 6, 9, 7], 'valid_counts': [1, 3, 2, 6, 4, 7, 5],
    'rtol': 1e-10, 'atol': 1e-12, 'backward_contributions': accum.capture.backward_contributions,
    'residual_error': error(accum.capture.r_sum, accum.exact_residual @ accum.z),
    'input_gradient_error': error(accum.input_grad, accum.dense_input_grad),
    'weight_gradient_error': error(accum.w.grad, accum.u @ (accum.u.T @ accum.exact[-2])),
    'bias_gradient_error': error(accum.b.grad, accum.exact[-1])}
deep = runpy.run_path('tests/test_exact_adjoints.py')
actual, _ = deep['residual_gradients'](projected_linear)
dense, _ = deep['residual_gradients'](lambda x, w, b, capture: F.linear(x, w, b))
out['deep_residual'] = {'seed': 605, 'device': 'cpu', 'dtype': 'float64', 'blocks': 4,
    'x_shape': [2, 3, 5], 'hidden_width': 7, 'ranks': [2, 1], 'rtol': 1e-10, 'atol': 1e-12,
    'input_gradient_error': error(actual[0], dense[0]),
    'bias_gradient_errors': [error(actual[4 * i + offset], dense[4 * i + offset]) for i in range(4) for offset in (3, 4)]}
sources = ['src/oar/linear.py', 'src/oar/basis.py', 'tests/test_algebra.py', 'tests/test_basis.py',
           'tests/test_accumulation.py', 'tests/test_exact_adjoints.py', 'pyproject.toml']
out['provenance']['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources}
Path('analysis').mkdir(exist_ok=True)
Path('analysis/operator_correctness.json').write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps({'operator_cases': len(out['operator_cases']), 'basis_cases': len(out['basis_cases']),
                  'artifact': 'analysis/operator_correctness.json',
                  'max_cpu_abs': max(v['max_abs'] for c in out['operator_cases'] if c['device'] == 'cpu' for v in c['errors'].values()),
                  'max_mps_abs': max(v['max_abs'] for c in out['operator_cases'] if c['device'] == 'mps' for v in c['errors'].values())}))
