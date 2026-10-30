"""Bounded shape/scale stress diagnostics; leaves online NumericalPolicy unresolved.

Run in a fresh interpreter with PYTORCH_ENABLE_MPS_FALLBACK=0. This empirical
distribution is deliberately not converted into a universal margin constant.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

assert 'torch' not in sys.modules
assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
import numpy as np
import torch

from oar.audit import NumericalPolicy, residual_bound
from oar.bench import load_benchmark_config
from oar.linear import LinearCapture, projected_linear
from oar.optimizer_preview import preview_adamw
from oar.records import AdamWGroup, AdamWParamState


def fp64(x):
    return x.detach().cpu().double()


def norm(x):
    return torch.linalg.vector_norm(x).item()


config = load_benchmark_config(Path('configs/operator.yaml'))
assert torch.backends.mps.is_available()
rows = []
started = __import__('time').perf_counter()
for shape_index, shape in enumerate(config.benchmark.shapes):
    # Both expansion and contraction matrices of the actual residual MLP.
    for direction, di, do in [('expand', shape.di, shape.do), ('contract', shape.do, shape.di)]:
        for seed, split in [(62101, 'development'), (62102, 'held_out_stress')]:
            generator = torch.Generator().manual_seed(seed)
            u64 = torch.linalg.qr(torch.randn(do, shape.rank, dtype=torch.float64, generator=generator)).Q
            u = u64.float().to('mps')
            for scale in (1e-4, 1., 1e4):
                for spectrum in ('generic', 'near_subspace'):
                    x = (torch.randn(shape.n, di, generator=generator)*scale).to('mps').requires_grad_()
                    if spectrum == 'generic':
                        d = (torch.randn(shape.n, do, generator=generator)*scale/(shape.n**.5)).to('mps')
                    else:
                        low = torch.randn(shape.n, shape.rank, generator=generator).double() @ u64.T
                        d = ((low + 1e-6*torch.randn(shape.n, do, generator=generator).double())*scale/(shape.n**.5)).float().to('mps')
                    p = torch.nn.Parameter(torch.randn(do, di, generator=generator).to('mps')/(di**.5))
                    probes = np.random.Generator(np.random.PCG64(seed+shape_index+777)).standard_normal((di, shape.q), dtype=np.float32)
                    z = torch.from_numpy(probes).to('mps')
                    capture = LinearCapture(u, z, 0)
                    for xm, dm in zip(x.chunk(shape.accumulation), d.chunk(shape.accumulation)):
                        projected_linear(xm, p, None, capture).backward(dm)
                    # References use the actual stored rounded FP32 values and
                    # the same reduction/objective weighting, calculated in FP64.
                    xc, dc, uc, zc = map(fp64, (x, d, u, z))
                    exact = dc.T @ xc
                    ideal = uc @ (uc.T @ exact)
                    residual = exact-ideal
                    sketch_reference = residual @ zc
                    actual_sketch, gradient = fp64(capture.r_sum), fp64(p.grad)
                    old = {'w': AdamWParamState(7, torch.full_like(p, .002), torch.full_like(p, .003))}
                    pending = preview_adamw({'w': p}, {'w': p.grad}, old,
                                            (AdamWGroup(('w',), 1e-4, (.9, .999), 1e-8, .01),), 0, 0, 1.)
                    displacement = pending.displacements['w']
                    comp = displacement-u @ (u.T @ displacement)
                    disp64, comp64 = fp64(displacement), fp64(comp)
                    ideal_comp = disp64-uc @ (uc.T @ disp64)
                    actual_bound = residual_bound(capture.r_sum.detach(), shape.q, .01)
                    reference_bound = residual_bound(sketch_reference.detach(), shape.q, .01)
                    b_actual = actual_bound*norm(comp64)
                    b_reference = reference_bound*norm(ideal_comp)
                    true_alignment = (exact*disp64).sum().item()
                    lower = pending.raw_alignment-b_actual
                    # This deterministic numerical discrepancy holds the same
                    # probes/displacement fixed. It does not remove Gaussian
                    # model risk or calibrate unobserved optimization histories.
                    a_reference = (ideal*disp64).sum().item()
                    numerical_margin_discrepancy = abs((pending.raw_alignment-b_actual)-(a_reference-b_reference))
                    rows.append(dict(shape=asdict(shape), direction=direction, weight_shape=[do, di], seed=seed,
                        split=split, scale=scale, spectrum=spectrum, dtype='float32', backend='mps',
                        orthogonality_defect=norm(uc.T@uc-torch.eye(shape.rank, dtype=torch.float64)),
                        gradient_error_frobenius=norm(gradient-ideal), residual_sketch_error_frobenius=norm(actual_sketch-sketch_reference),
                        explicit_complement_error_frobenius=norm(comp64-ideal_comp),
                        raw_alignment=pending.raw_alignment, reference_projected_alignment=a_reference,
                        true_dense_alignment=true_alignment, computed_lower_margin=lower,
                        b_computed=b_actual, b_reference_same_probe= b_reference,
                        orthogonality_identity_defect=abs((residual*disp64).sum().item()-(residual*ideal_comp).sum().item()),
                        numerical_margin_discrepancy=numerical_margin_discrepancy))
                    del x, d, p, pending, old, capture, xc, dc, exact, ideal, residual, sketch_reference, gradient

sources = ['analysis/collect_numerical_calibration.py', 'configs/operator.yaml', 'src/oar/linear.py',
           'src/oar/audit.py', 'src/oar/optimizer_preview.py', 'src/oar/basis.py']
policy_context = dict(backend='mps', dtype='float32', shapes=[asdict(s) for s in config.benchmark.shapes],
                      scales=[1e-4, 1., 1e4], spectra=['generic', 'near_subspace'])
policy = NumericalPolicy(policy_context)
output = dict(schema_version=1, captured_at_utc=datetime.now(timezone.utc).isoformat(),
    fallback_environment_before_torch_import='0', torch_preimported=False, torch=torch.__version__,
    rows=rows, source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
    active_wall_s=__import__('time').perf_counter()-started,
    numerical_policy=dict(status='unresolved', measured_u_margin=None, config_sha256=policy.config_sha256,
                          reason='Observed inner-product errors are shape/scale/seed diagnostics. No conservative formula validated across optimizer histories, near-threshold margins, adversarial cancellation, and production distributions; no production guardband fitted.'),
    development_max_numerical_margin_discrepancy=max(r['numerical_margin_discrepancy'] for r in rows if r['split']=='development'),
    held_out_stress_max_numerical_margin_discrepancy=max(r['numerical_margin_discrepancy'] for r in rows if r['split']=='held_out_stress'))
Path('analysis/guardband_calibration.json').write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+'\n')
print(json.dumps(dict(cases=len(rows), policy=output['numerical_policy'],
                      development_max=output['development_max_numerical_margin_discrepancy'],
                      held_out_max=output['held_out_stress_max_numerical_margin_discrepancy'])))
