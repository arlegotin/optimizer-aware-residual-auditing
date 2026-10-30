"""Bounded operator gate, with a fresh fallback-disabled worker per whole study.

Headline blocks execute complete residual MLP updates. Profile blocks are
separate diagnostics and must never be summed into a claimed full-step time.
Conditional commit timing bypasses an unresolved policy solely to price p=0;
it is explicitly ineligible for online acceptance or learning claims.
"""

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

from .config import RunConfig, RunPaths, SeedStreams, _fields, _UniqueLoader
from .timing import ActiveClock, measure_blocks, paired_blocks, warmup


@dataclass(frozen=True)
class Shape:
    n: int
    di: int
    do: int
    rank: int
    q: int
    accumulation: int
    noncontiguous: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int
    blocks: int
    max_blocks: int
    bootstrap_replicates: int
    noise_relative_width: float
    warmup: dict
    shapes: tuple[Shape, ...]
    output: str


@dataclass
class CostTable:
    rows: dict[tuple, dict]
    metadata: dict


def load_benchmark_config(path: Path) -> RunConfig:
    try:
        payload = yaml.load(Path(path).read_text(), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError('invalid YAML configuration') from exc
    return _parse_benchmark(payload)


def _parse_benchmark(payload):
    raw = _fields(payload,
                  {'schema_version', 'device', 'seed', 'blocks', 'max_blocks',
                   'bootstrap_replicates', 'noise_relative_width', 'warmup', 'shapes', 'output'}, 'benchmark')
    if type(raw['schema_version']) is not int or raw['schema_version'] != 1 or raw['device'] not in ('cpu', 'mps'):
        raise ValueError('require schema 1 and cpu/mps')
    for name in ('seed', 'blocks', 'max_blocks', 'bootstrap_replicates'):
        if type(raw[name]) is not int or not 0 < raw[name] < 2**63-3:
            raise ValueError(f'{name} must be a positive integer')
    if raw['blocks'] < 3 or raw['max_blocks'] < raw['blocks']:
        raise ValueError('require at least three blocks and max_blocks >= blocks')
    w = _fields(raw['warmup'], {'minimum', 'maximum', 'window', 'relative_range'}, 'warmup')
    if any(type(w[k]) is not int or w[k] < 2 for k in ('minimum', 'maximum', 'window')) or w['maximum'] < max(w['minimum'], w['window']):
        raise ValueError('invalid warmup bounds')
    for value in (w['relative_range'], raw['noise_relative_width']):
        if type(value) not in (int, float) or not 0 < value < 1:
            raise ValueError('relative noise settings must be in (0,1)')
    shapes = []
    if not isinstance(raw['shapes'], (list, tuple)) or not raw['shapes']:
        raise ValueError('require nonempty shapes')
    for s in raw['shapes']:
        _fields(s, set(Shape.__dataclass_fields__), 'shape')
        if any(type(s[k]) is not int or s[k] <= 0 for k in ('n', 'di', 'do', 'rank', 'q', 'accumulation')):
            raise ValueError('shape sizes must be positive integers')
        if type(s['noncontiguous']) is not bool or s['rank'] >= min(s['di'], s['do']) or s['n'] % s['accumulation']:
            raise ValueError('invalid rank, layout or accumulation')
        shapes.append(Shape(**s))
    if not isinstance(raw['output'], str) or not raw['output'].strip():
        raise ValueError('require output path')
    b = BenchmarkConfig(**{k: raw[k] for k in BenchmarkConfig.__dataclass_fields__ if k != 'shapes'}, shapes=tuple(shapes))
    return RunConfig(1, 'exploratory', 'dense_adamw', raw['device'],
                     SeedStreams(b.seed, b.seed + 1, b.seed + 2, b.seed + 3),
                     RunPaths(Path('data'), Path('runs'), Path('decisions')), b)


def account_attempt(components):
    if not components or any(not math.isfinite(v) or v < 0 for v in components.values()):
        raise ValueError('costs must be finite and nonnegative')
    candidate = components['forward_backward_reconstruction']
    total = math.fsum(components.values())
    return dict(active_wall_s=total, candidate_only_s=candidate, audit_and_preview_s=total-candidate)


def cycle_forecast(dense, attempt, replay, refresh_extra, interval, rejection_probability):
    if (any(not math.isfinite(v) or v < 0 for v in (dense, attempt, replay, refresh_extra))
            or replay <= 0 or type(interval) is not int or interval < 1 or not 0 <= rejection_probability <= 1):
        raise ValueError('invalid cycle costs')
    return dict(cycle_mean_s=(dense + refresh_extra + (interval-1)*(attempt+rejection_probability*replay))/interval,
                local_break_even_p=(dense-attempt-refresh_extra/interval)/replay)


def candidate_linear(x, weight, bias, basis):
    """Cheapest faithful candidate: dense reconstruction and exact dX/db, no audit."""
    import torch
    from torch.nn import functional as F

    class Candidate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, basis):
            ctx.save_for_backward(x, weight, basis)
            return F.linear(x, weight, bias)

        @staticmethod
        def backward(ctx, d):
            x, w, u = ctx.saved_tensors
            return d @ w, u @ ((d @ u).T @ x), d.sum(0), None

    return Candidate.apply(x, weight, bias, basis)


def native_state(parameters, optimizer):
    """Borrow current native moments after exact replay, without copying tensors."""
    from .records import AdamWParamState
    return {name: AdamWParamState(int(optimizer.state[p]['step'].item()),
                                  optimizer.state[p]['exp_avg'], optimizer.state[p]['exp_avg_sq'])
            for name, p in parameters.items() if optimizer.state.get(p)}


def _summary(raw, seed, replicates):
    values = np.asarray(raw, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (replicates, len(values)))].mean(1)
    return dict(mean_s=float(values.mean()), median_s=float(np.median(values)),
                min_s=float(values.min()), max_s=float(values.max()),
                mean_bootstrap_95_s=np.quantile(means, [.025, .975]).tolist(), raw_s=list(raw))


def _paired_compare(a, b, synchronize, config, context=None):
    initial_context = context() if context else None
    warm = {name: warmup(op, synchronize, **config.warmup) for name, op in [('dense', a), ('other', b)]}
    rows = []
    # Add blocks only for unresolved/noisy cost differences, never for a p-value.
    while len(rows) < config.max_blocks:
        count = config.blocks if not rows else min(4, config.max_blocks-len(rows))
        batch = paired_blocks({'dense': a, 'other': b}, synchronize, count, config.seed+len(rows), context=context)
        for r in batch:
            r['pair'] += len(rows)
        rows.extend(batch)
        differences = [r['seconds']['dense']-r['seconds']['other'] for r in rows]
        diff = _summary(differences, config.seed, config.bootstrap_replicates)
        width = diff['mean_bootstrap_95_s'][1]-diff['mean_bootstrap_95_s'][0]
        scale = np.mean([r['seconds']['dense'] for r in rows])
        if width <= config.noise_relative_width*scale:
            break
    return dict(context_before_warmup=initial_context, warmup=warm, pairs=rows, dense=_summary([r['seconds']['dense'] for r in rows], config.seed, config.bootstrap_replicates),
                other=_summary([r['seconds']['other'] for r in rows], config.seed, config.bootstrap_replicates),
                dense_minus_other=diff, noise_target_met=bool(width <= config.noise_relative_width*scale))


def _compare_many(dense, others, synchronize, config, *, context=None, clock=time.perf_counter):
    """Shared randomized rounds prevent a method owning an earlier cost regime."""
    rng = random.Random(config.seed)
    operations = {'dense': dense, **others}
    warm_order = list(operations)
    rng.shuffle(warm_order)
    initial_context = context() if context else None
    warm = {name:warmup(operations[name], synchronize, **config.warmup, clock=clock) for name in warm_order}
    results = {name:dict(pairs=[], warmup={'dense':warm['dense'], 'other':warm[name]},
                              context_before_warmup=initial_context,
                              timing_schedule='randomized shared rounds; paired AB/BA within each round') for name in others}
    for index in range(config.max_blocks):
        order = list(others)
        rng.shuffle(order)
        for name in order:
            pair = paired_blocks({'dense':dense, 'other':others[name]}, synchronize, 1,
                                 rng.randrange(2**63), context=context, clock=clock)[0]
            pair.update(pair=index, comparison_order=order)
            results[name]['pairs'].append(pair)
        if index+1 >= config.blocks:
            for result in results.values():
                pairs = result['pairs']
                result['dense'] = _summary([p['seconds']['dense'] for p in pairs], config.seed, config.bootstrap_replicates)
                result['other'] = _summary([p['seconds']['other'] for p in pairs], config.seed, config.bootstrap_replicates)
                result['dense_minus_other'] = _summary([p['seconds']['dense']-p['seconds']['other'] for p in pairs], config.seed, config.bootstrap_replicates)
                ci = result['dense_minus_other']['mean_bootstrap_95_s']
                result['noise_target_met'] = bool(ci[1]-ci[0] <= config.noise_relative_width*result['dense']['mean_s'])
            if all(r['noise_target_met'] for r in results.values()):
                break
    return results


def benchmark(config: RunConfig) -> CostTable:
    """One fresh worker for the complete study; fallback=0 precedes Torch import."""
    if not isinstance(config.benchmark, BenchmarkConfig):
        raise ValueError('benchmark configuration required')
    payload = dict(device=config.device, benchmark=asdict(config.benchmark))
    with tempfile.TemporaryDirectory(prefix='oar-bench-') as directory:
        request, response = Path(directory)/'request.json', Path(directory)/'response.json'
        request.write_text(json.dumps(payload))
        worker = subprocess.run([sys.executable, '-m', 'oar.bench', '--worker', str(request), str(response)],
                                env={**os.environ, 'PYTORCH_ENABLE_MPS_FALLBACK': '0'}, check=False)
        if worker.returncode:
            raise RuntimeError(f'benchmark worker failed: {worker.returncode}')
        result = json.loads(response.read_text())
    return CostTable({tuple(r['key']): r for r in result['rows']}, result['metadata'])


def write_cost_table(table: CostTable, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(schema_version=1, metadata=table.metadata, rows=list(table.rows.values())), indent=2, sort_keys=True, allow_nan=False)+'\n')
    with output.with_suffix('.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['backend', 'dtype', 'n', 'di', 'do', 'rank', 'q', 'accumulation', 'comparison', 'dense_s', 'other_s', 'saving_low_s', 'saving_high_s'])
        for r in table.rows.values():
            for name, c in r['headline'].items():
                writer.writerow([*r['key'], r['shape']['accumulation'], name, c['dense']['mean_s'], c['other']['mean_s'], *c['dense_minus_other']['mean_bootstrap_95_s']])


def _run_worker(request):
    preimported = 'torch' in sys.modules
    fallback = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')
    if preimported or fallback != '0':
        raise RuntimeError('fallback policy must precede Torch import')
    import torch
    from torch.nn import functional as F
    import psutil
    from .audit import NumericalPolicy, residual_bound
    from .audit_journal import AuditJournal
    from .basis import build_basis
    from .hardware import resource_snapshot, profile_resources, redact_hardware, _collect_hardware, _command
    from .linear import LinearCapture, projected_linear
    from .optimizer_preview import preview_adamw, commit_pending, _validate
    from .records import AdamWGroup, AuditCandidate

    active = ActiveClock()
    device = request['device']
    b = request['benchmark']
    b['shapes'] = tuple(Shape(**s) for s in b['shapes'])
    config = BenchmarkConfig(**b)
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('actual MPS unavailable; sandbox absence is not a hardware finding')
    synchronize = torch.mps.synchronize if device == 'mps' else lambda: None
    before = resource_snapshot(device)
    physical = psutil.virtual_memory().total
    recommended = torch.mps.recommended_max_memory() if device == 'mps' else physical
    ceiling = min(.7*physical, .8*recommended)
    hardware = redact_hardware(_collect_hardware())
    def thermal_context():
        # This command reports policy/throttling limits when exposed, not chip
        # temperature. Keep only named numeric fields; no arbitrary shell text.
        import re
        raw = _command(['pmset', '-g', 'therm']) or ''
        return {name: int(m.group(1)) if (m := re.search(name+r'\s*=\s*(\d+)', raw)) else None
                for name in ('CPU_Speed_Limit', 'CPU_Scheduler_Limit', 'CPU_Available_CPUs')}
    thermal_before = thermal_context()
    rows = []

    class Fixture:
        def __init__(self, shape, directory, variant='single'):
            self.s = shape
            self.step = 0
            self.version = (0, 0)
            self.state = {}
            self.group = (AdamWGroup(('w1', 'b1', 'w2', 'b2'), 1e-4, (.9, .999), 1e-8, .01),)
            g = torch.Generator().manual_seed(config.seed)
            # A view with stride 2 exercises the noncontiguous input path.
            self.x = torch.randn(shape.n, shape.di*(2 if shape.noncontiguous else 1), generator=g).to(device)
            if shape.noncontiguous:
                self.x = self.x[:, ::2]
            self.x.requires_grad_()
            self.parameters = {name: torch.nn.Parameter(value.to(device)) for name, value in {
                'w1': torch.randn(shape.do, shape.di, generator=g)/math.sqrt(shape.di),
                'b1': torch.zeros(shape.do),
                'w2': torch.randn(shape.di, shape.do, generator=g)/math.sqrt(shape.do),
                'b2': torch.zeros(shape.di)}.items()}
            kwargs = dict(foreach=variant == 'foreach', fused=variant == 'fused')
            if variant == 'default':
                kwargs = {}
            self.optimizer = torch.optim.AdamW(list(self.parameters.values()), lr=1e-4, weight_decay=.01, **kwargs)
            self.backward('dense')
            self.bases = {name: build_basis(self.parameters[name].grad, shape.rank, 8,
                           torch.Generator().manual_seed(config.seed+100+i), 0)
                          for i, name in enumerate(('w1', 'w2'))}
            torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1., foreach=False, error_if_nonfinite=True)
            self.optimizer.step()  # initial native dense update, basis source 0
            self.state = native_state(self.parameters, self.optimizer)
            self.initialization_s = active.elapsed_s()
            self.journal = AuditJournal(Path(directory)/'audit.jsonl', .01, 0, audit_seed=config.seed+1)
            self.policy = NumericalPolicy(dict(device=device, shape=asdict(shape), dtype='float32'))
            self.decisions = {}

        def backward(self, mode, captures=None):
            for p in self.parameters.values():
                p.grad = None
            self.x.grad = None
            p = self.parameters
            def linear(x, name, bias):
                if mode == 'dense':
                    return F.linear(x, p[name], p[bias])
                if mode == 'candidate':
                    return candidate_linear(x, p[name], p[bias], self.bases[name].vectors)
                return projected_linear(x, p[name], p[bias], captures[name])
            for x in self.x.chunk(self.s.accumulation):
                y = x + linear(F.gelu(linear(x, 'w1', 'b1')), 'w2', 'b2')
                (y.square().mean()/self.s.accumulation).backward()

        def native(self, mode='dense'):
            self.backward(mode)
            torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1., foreach=False, error_if_nonfinite=True)
            self.optimizer.step()

        def captures(self):
            self.step += 1
            reservations = self.journal.reserve(('w1', 'w2'), self.step)
            result = {}
            for name, r in zip(('w1', 'w2'), reservations):
                candidate = AuditCandidate(name, self.step, self.s.rank, 0, 0)
                z = self.journal.draw_probes(r, candidate, input_width=self.parameters[name].shape[1],
                                             q=self.s.q, dtype=torch.float32, device=device)
                result[name] = LinearCapture.from_basis(self.bases[name], current_step=self.step,
                                                        probes=z, basis_version=0)
            return result, reservations

        def audit(self):
            captures, reservations = self.captures()
            self.backward('full', captures)
            raw = {name: p.grad for name, p in self.parameters.items()}
            pending = preview_adamw(self.parameters, raw, self.state, self.group, *self.version, 1.)
            contributions = {}
            for (name, c), reservation in zip(captures.items(), reservations):
                d, u = pending.displacements[name], c.basis
                complement = d-u @ (u.T @ d)
                norm = torch.linalg.vector_norm(complement).item()
                contributions[name] = residual_bound(c.r_sum, self.s.q, reservation.delta)*norm
            decision = self.policy.decide(pending.raw_alignment, contributions, .5,
                                          configuration=dict(device=device, shape=asdict(self.s), dtype='float32'))
            self.decisions[decision.kind.value] = self.decisions.get(decision.kind.value, 0)+1
            return pending

        def conditional(self):
            pending = self.audit()
            # Deliberate counterfactual: price commit at p=0, never authorize it.
            self.version = commit_pending(pending, self.parameters, self.state, *self.version)

        def reject(self):
            pending = self.audit()
            del pending
            # No stochastic model operators: the exact same tensor batch is replayed.
            self.replay()

        def replay(self):
            self.native()
            self.state = native_state(self.parameters, self.optimizer)

        def refresh(self):
            self.backward('dense')
            for i, name in enumerate(('w1', 'w2')):
                self.bases[name] = build_basis(self.parameters[name].grad, self.s.rank, 8,
                                              torch.Generator().manual_seed(config.seed+900+i), self.step)
            torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1., foreach=False, error_if_nonfinite=True)
            self.optimizer.step()
            self.state = native_state(self.parameters, self.optimizer)

    with tempfile.TemporaryDirectory(prefix='oar-operator-') as directory:
        for index, shape in enumerate(config.shapes):
            shape_start = active.elapsed_s()
            # Dense+candidate+full fixtures and FP64 refresh temporaries: conservative
            # allocation screening, not a claim about observed peak memory.
            predicted = 3*64*(2*shape.di*shape.do+shape.di+shape.do)+32*shape.n*shape.do
            if predicted > ceiling or predicted > psutil.virtual_memory().available:
                raise RuntimeError('bounded shape exceeds current memory screening ceiling')
            root = Path(directory)/str(index)
            root.mkdir()
            variants, unsupported, native_fixtures = {}, {}, {}
            for variant in ('single', 'foreach', 'fused', 'default'):
                sub = root/variant
                sub.mkdir()
                try:
                    f = Fixture(shape, sub, variant)
                    # Native executable variants must match the qualified native
                    # single-tensor update before they may compete on throughput.
                    params = [p.detach().clone().requires_grad_() for p in f.parameters.values()]
                    reference = [p.detach().clone().requires_grad_() for p in params]
                    for a, r in zip(params, reference):
                        a.grad = torch.ones_like(a)*.001
                        r.grad = a.grad.clone()
                    kw = {} if variant == 'default' else dict(foreach=variant=='foreach', fused=variant=='fused')
                    test_opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01, **kw)
                    ref_opt = torch.optim.AdamW(reference, lr=1e-4, weight_decay=.01, foreach=False, fused=False)
                    for _ in range(3):
                        test_opt.step()
                        ref_opt.step()
                    for a, r in zip(params, reference):
                        torch.testing.assert_close(a, r, rtol=2e-6, atol=2e-7)
                        for moment in ('exp_avg', 'exp_avg_sq'):
                            torch.testing.assert_close(test_opt.state[a][moment], ref_opt.state[r][moment], rtol=2e-6, atol=2e-7)
                    native_fixtures[variant] = f
                    del f, params, reference, test_opt, ref_opt
                except (RuntimeError, AssertionError, NotImplementedError) as exc:
                    unsupported[variant] = dict(error_type=type(exc).__name__, message=str(exc)[:200])
            if not native_fixtures:
                raise RuntimeError('no validated native AdamW baseline')
            native_warm_order = list(native_fixtures)
            random.Random(config.seed).shuffle(native_warm_order)
            native_warm = {v:warmup(native_fixtures[v].native, synchronize, **config.warmup) for v in native_warm_order}
            native_blocks = paired_blocks({v:f.native for v,f in native_fixtures.items()}, synchronize,
                                           config.blocks, config.seed)
            for variant in native_fixtures:
                variants[variant] = dict(validation='three native updates vs single, FP32 rtol=2e-6 atol=2e-7',
                                         warmup=native_warm[variant],
                                         timing=_summary([p['seconds'][variant] for p in native_blocks], config.seed, config.bootstrap_replicates))
            fastest = min(variants, key=lambda name: variants[name]['timing']['mean_s'])
            del native_fixtures
            fixtures = {}
            for name in ('dense', 'candidate', 'conditional', 'reject', 'profile'):
                sub = root/name
                sub.mkdir()
                fixtures[name] = Fixture(shape, sub, fastest)
                if name == 'conditional':
                    # The transactional branch owns its moments after preview;
                    # do not retain orphan native moments after the first commit.
                    fixtures[name].optimizer.state.clear()
            headline = _compare_many(fixtures['dense'].native, {
                'candidate':lambda:fixtures['candidate'].native('candidate'),
                'conditional':fixtures['conditional'].conditional,
                'reject':fixtures['reject'].reject,
                'replay_only':fixtures['reject'].replay}, synchronize, config,
                context=lambda:dict(reservations={n:2*f.step for n,f in fixtures.items()},
                                    native_variant=fastest))
            # Separate profiling path: explicit stage timings/snapshots are never
            # substituted for the above minimally instrumented whole operations.
            f = fixtures['profile']
            p = f.parameters
            f.backward('dense')
            raw = {n: v.grad for n, v in p.items()}
            pending = preview_adamw(p, raw, f.state, f.group, *f.version, 1.)
            memory_after_preview = resource_snapshot(device)
            u, d = f.bases['w1'].vectors, pending.displacements['w1']
            x = f.x.detach()
            derivative = torch.ones(shape.n, shape.do, device=device)/shape.n
            z = torch.ones(shape.di, shape.q, device=device)
            profile_ops = {
                'forward': lambda: F.linear(x, p['w1'], p['b1']),
                'exact_dx': lambda: derivative @ p['w1'],
                'exact_dw': lambda: derivative.T @ x,
                'project_reconstruct': lambda: u @ ((derivative @ u).T @ x),
                'probes': lambda: derivative.T @ (x @ z)-u @ (((derivative @ u).T @ x) @ z),
                'explicit_complement': lambda: torch.linalg.vector_norm(d-u @ (u.T @ d)).item(),
                'preview_including_raw_alignment': lambda: preview_adamw(p, raw, f.state, f.group, *f.version, 1.),
                'preview_input_validation': lambda: _validate(p, raw, f.state, f.group, *f.version, 1.),
                'raw_alignment_cpu_fp64_transfer': lambda: (raw['w1'].detach().cpu().double()*d.detach().cpu().double()).sum().item(),
                'refresh_cpu_qr_svd_transfers': lambda: build_basis(raw['w1'], shape.rank, 8, torch.Generator().manual_seed(config.seed), 0),
                'sampling_gather_scatter': lambda: torch.zeros_like(x).index_add_(0, torch.arange(0, shape.n, 2, device=device), x.index_select(0, torch.arange(0, shape.n, 2, device=device))),
                'generation_journal_transfer_capture': f.captures,
                'native_optimizer_only': f.optimizer.step,
            }
            profiles = {name: measure_blocks(op, synchronize, 3) for name, op in profile_ops.items()}
            journal_age = []
            for age in (0, 64, 256):
                sub = root/f'aged-{age}'
                sub.mkdir()
                aged = Fixture(shape, sub, fastest)
                if age:
                    for j in range(age):
                        aged.journal.reserve(('w1',), j+1)
                journal_age.append(dict(prior_reservations=age, raw_s=measure_blocks(aged.captures, synchronize, 3),
                                        bytes_after=aged.journal.path.stat().st_size))
                del aged
            refresh = _paired_compare(fixtures['dense'].native, f.refresh, synchronize, config)
            memory_profiles = {name: profile_resources(op, device) for name, op in {
                'dense_full_step': fixtures['dense'].native, 'conditional_full_attempt': fixtures['conditional'].conditional,
                'forced_reject_replay': fixtures['reject'].reject, 'dense_refresh_transfers': f.refresh}.items()}
            # Refresh comparison uses the entire dense+construction block. The
            # difference is extra refresh work; scheduled dense work is counted once.
            row = dict(key=[device, 'float32', shape.n, shape.di, shape.do, shape.rank, shape.q],
                       shape=asdict(shape), fastest_native=fastest, native_variants=variants,
                       native_variant_randomized_blocks=native_blocks,
                       unsupported_native_variants=unsupported, headline=headline,
                       refresh=refresh, profile_raw_s=profiles, journal_age=journal_age,
                       resource_profiles=memory_profiles,
                       memory_after_preview=memory_after_preview,
                       retained_parameter_optimizer_preview_bytes=32*(2*shape.di*shape.do+shape.di+shape.do),
                       resource_after=resource_snapshot(device), allocation_screen_bytes=predicted,
                       actual_policy_decisions={name: f.decisions for name, f in fixtures.items()},
                       shape_active_wall_s=active.elapsed_s()-shape_start)
            rows.append(row)
            print(json.dumps(dict(shape=asdict(shape), fastest_native=fastest,
                                  means={k: [v['dense']['mean_s'], v['other']['mean_s']] for k,v in headline.items()},
                                  savings95={k:v['dense_minus_other']['mean_bootstrap_95_s'] for k,v in headline.items()})), flush=True)
            del fixtures, f, p, raw, pending, u, d, x, derivative, z, profile_ops
    sources = ['src/oar/bench.py', 'src/oar/timing.py', 'src/oar/hardware.py', 'src/oar/config.py', 'src/oar/cli.py', 'src/oar/linear.py', 'src/oar/basis.py',
               'src/oar/optimizer_preview.py', 'src/oar/audit.py', 'src/oar/audit_journal.py', 'configs/operator.yaml']
    return dict(schema_version=1, rows=rows, metadata=dict(
        torch_preimported=preimported, fallback_environment_before_torch_import=fallback,
        backend=device, python=platform.python_version(), torch=torch.__version__, torch_revision=torch.version.git_version,
        hardware=hardware, thermal_limits_before=thermal_before, thermal_limits_after=thermal_context(),
        resources_before=before, resources_after=resource_snapshot(device),
        engineering_screen_ceiling_bytes=int(ceiling), recommended_metal_working_set_bytes=recommended,
        config=asdict(config), source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        active_wall_s=active.elapsed_s(), null_block_raw_s=measure_blocks(lambda: None, synchronize, config.blocks),
        interpretation='Conditional p=0 timing bypasses unresolved NumericalPolicy solely to price complete commit. Actual policy always rejects. No learning value or deployable acceptance evidence. Paired bootstrap intervals are block variability, not independent-run inferential intervals. Profile sums are not headline time. Separate 5ms resource profiles give sampled lower bounds on peaks; short peaks may be missed. Five comparison fixtures reside in this worker; process resource peaks are study working set, not a single model training footprint. RSS and Metal overlap and are never added. No checkpoint/trainer exists; checkpoint memory and full-cycle training throughput remain unmeasured.'))


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[1] != '--worker':
        raise SystemExit(2)
    result = _run_worker(json.loads(Path(sys.argv[2]).read_text()))
    Path(sys.argv[3]).write_text(json.dumps(result, allow_nan=False))
