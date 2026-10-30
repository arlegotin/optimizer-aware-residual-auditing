"""Small deterministic Task 4 evidence; no fitted guardband or training claim."""
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from oar.audit import lower_fraction
from oar.linear import LinearCapture, projected_linear


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [max(0., center-half), min(1., center+half)]


roots = []
for q, delta in [(q, d) for q in (8, 16, 32) for d in (.01, 1e-6)] + [(8, 1e-300), (32, math.nextafter(1., 0.))]:
    c = lower_fraction(q, delta)
    with localcontext() as context:
        context.prec = 80
        dc, dd = Decimal.from_float(c), Decimal.from_float(delta)
        residual = dc - 1 - dc.ln() + 2 * dd.ln() / Decimal(q)
    roots.append(dict(q=q, delta=delta, lower_fraction=c, norm_multiplier=1/math.sqrt(c),
                      equation_residual_decimal80=float(residual)))

confidence = []
z = np.random.default_rng(8401).normal(size=(20_000, 4, 8))
for name, spectrum in [('rank_one', [1., 0., 0., 0.]), ('flat', [1., 1., 1., 1.]), ('decaying', [1., .5, .25, .125])]:
    r = z * np.asarray(spectrum)[None, :, None]
    squared_norm = sum(x*x for x in spectrum)
    s = (r*r).sum(axis=(1, 2)) / 8
    k = int(np.count_nonzero(s / lower_fraction(8, .1) < squared_norm))
    confidence.append(dict(spectrum_name=name, singular_values=spectrum, seed=8401,
        generator='numpy.default_rng/PCG64', replicates=20_000, q=8, delta=.1,
        failures=k, fraction=k/20_000, wilson_95=wilson(k, 20_000),
        mean_squared_estimate=float(s.mean()), true_squared_norm=squared_norm))

numerics = []
for device in ('cpu', 'mps'):
    if device == 'mps' and not torch.backends.mps.is_available():
        continue
    for scale in (1e-8, 1., 1e8):
        x = (torch.tensor([[1e8], [1.], [-1e8]], dtype=torch.float32) * scale).to(device)
        weight = torch.ones(1, 1, dtype=torch.float32, device=device, requires_grad=True)
        capture = LinearCapture(torch.empty(1, 0, device=device), torch.ones(1, 1, device=device), 0)
        projected_linear(x, weight, None, capture).sum().backward()
        actual = capture.r_sum.cpu().double().item()
        reference = (torch.ones(1, 3, dtype=torch.float64) @ x.cpu().double()).item()
        numerics.append(dict(device=device, dtype='float32', x_shape=[3, 1], rank=0,
            probe_count=1, scale=scale, residual_computed=actual,
            residual_cpu_fp64_of_stored_values=reference, absolute_error=abs(actual-reference)))

prior = json.loads(Path('analysis/operator_correctness.json').read_text())
prior_envelopes = []
for dtype in ('torch.float64', 'torch.float32'):
    rows = [x for x in prior['operator_cases'] if x['dtype'] == dtype]
    prior_envelopes.append(dict(dtype=dtype, device=rows[0]['device'], cases=len(rows),
        max_abs_error=max(e['max_abs'] for row in rows for e in row['errors'].values()),
        residual_max_abs_error=max(row['errors']['residual']['max_abs'] for row in rows)))

torch_a = torch.randn(16, generator=torch.Generator().manual_seed(123), dtype=torch.float64)
torch_b = torch.randn(16, generator=torch.Generator().manual_seed(123 + 2**32), dtype=torch.float64)
np_a = np.random.Generator(np.random.PCG64(123)).standard_normal(16)
np_b = np.random.Generator(np.random.PCG64(123 + 2**32)).standard_normal(16)
files = ['src/oar/audit.py', 'src/oar/audit_journal.py', 'src/oar/records.py',
         'tests/test_confidence.py', 'tests/test_audit_decision.py', 'tests/test_audit_journal.py']
evidence = dict(schema_version=1, captured_at_utc=datetime.now(timezone.utc).isoformat(),
    scope='Tiny confidence/numerical diagnostics, not shape-specific empirical calibration, timing, training or formal machine certification',
    provenance=dict(base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), collector_sha256=sha(__file__), python=platform.python_version(),
        machine=platform.machine(), torch=torch.__version__, numpy=np.__version__,
        fallback_environment_before_import=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
        mps_available=torch.backends.mps.is_available(), source_sha256={p: sha(p) for p in files}),
    scalar_roots=roots, gaussian_confidence=confidence,
    confidence_interpretation='Conditional real-arithmetic Gaussian bound; 20,000 replicates at delta=.1 do not establish tiny run-wide risk. Wilson intervals describe Monte Carlo sampling uncertainty, not theorem validity.',
    scalar_underflow=dict(q=1, delta=math.ulp(0.), expected='ValueError: unrepresentable positive root'),
    cancellation_stress=numerics,
    task3_operator_envelopes=dict(source='analysis/operator_correctness.json', sha256=sha('analysis/operator_correctness.json'), cases=prior_envelopes,
        interpretation='Tiny Task3 fixtures only. Raw residual/gradient errors cannot be copied as an inner-product margin allowance.'),
    seed_high_bits=dict(seeds=[123, 123+2**32], vector_length=16,
        cpu_torch_normal_streams_equal=bool(torch.equal(torch_a, torch_b)),
        numpy_pcg64_normal_streams_equal=bool(np.array_equal(np_a, np_b)),
        chosen_audit_stream='oar.audit.pcg64.gaussian.v1',
        interpretation='Actual Torch CPU high-bit seed collision motivates PCG64; stored NumPy version is checked on journal resume.'),
    numerical_policy=dict(status='unresolved', measured_u_margin=None,
        reason='Actual shape/backend/dtype/scale-specific residual, orthogonality and reduction calibration remains Tasks6/14. No coefficient inferred here.'))
Path('analysis/audit_confidence.json').write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + '\n')
print(json.dumps(dict(root_max_abs_residual=max(abs(x['equation_residual_decimal80']) for x in roots),
    failures={x['spectrum_name']: x['failures'] for x in confidence},
    cancellation_stress=numerics, seed_high_bits=evidence['seed_high_bits']), indent=2))
