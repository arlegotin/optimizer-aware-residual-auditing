# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded, append-only dW sampling experiment. Importing never starts work.

Only A--D are implemented before their gates. Corpus/trainer work stays absent
while the recorded asset restriction is unresolved. No old oracle is invoked.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REVIEWED = '9d8ad069c51b4d0e6c13e9c6e465579e9b44a339'
PROTOCOL = '8d0ce1920bdc06d0c02f94a0ca14a7a7e9e976060225820270bf8c1affbeca67'
STAGES = ('prepare', 'feasibility', 'correctness', 'cost', 'pilot', 'confirm', 'summarize')
CORE = ('analysis/run_dw_sampling.py', 'configs/dw_sampling.yaml', 'decisions/dw_sampling_scope.md',
        'decisions/dw_sampling_reopening.md',
        'src/oar/timing.py', 'src/oar/hardware.py')
SAMPLED = ('src/oar/sampled_linear.py', 'tests/test_sampled_linear.py', 'tests/test_dw_sampling_protocol.py')
HISTORICAL = ('decisions/gates.jsonl', 'decisions/amendments.jsonl', 'manifests/assets.json',
              'analysis/evidence_index.json', 'analysis/report.md', 'requirements.lock.txt', 'manifests/wheels.sha256')


class MeasurementInvalid(RuntimeError):
    def __init__(self, condition, reason):
        super().__init__(reason)
        self.condition = condition


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_config(path):
    import yaml
    from oar.config import _UniqueLoader
    try:
        value = yaml.load(Path(path).read_text(), Loader=_UniqueLoader)
        valid = hashlib.sha256(canonical(value)).hexdigest() == PROTOCOL
    except (yaml.YAMLError, TypeError, ValueError) as exc:
        raise ValueError('invalid strict sampling configuration') from exc
    if not valid:
        raise ValueError('unknown or changed protected protocol; freeze a separately reviewed source/config version')
    return value


def allocate_attempt(out, stage):
    if stage not in STAGES:
        raise ValueError('unknown stage')
    directory = Path(out)/stage
    directory.mkdir(parents=True, exist_ok=True)
    for number in range(1, 100000):
        attempt = directory/f'attempt-{number:04d}'
        try:
            attempt.mkdir()
            return attempt
        except FileExistsError:
            continue
    raise RuntimeError('attempt limit reached')


def seal_record(attempt, payload):
    attempt = Path(attempt)
    if (attempt/'record.json').exists():
        raise FileExistsError('stage records cannot be overwritten')
    artifacts = {str(p.relative_to(attempt)): digest(p) for p in sorted(attempt.rglob('*'))
                 if p.is_file() and p.name not in ('record.json', 'record.sha256')}
    record = {**payload, 'schema_version': 1, 'recorded_utc': utc(), 'artifacts': artifacts}
    write_json(attempt/'record.json', record)
    with (attempt/'record.sha256').open('x') as stream:
        stream.write(digest(attempt/'record.json')+'\n')
    return record


def read_record(path):
    path = Path(path)
    if digest(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError(f'record hash mismatch: {path}')
    record = json.loads(path.read_text())
    for name, expected in record['artifacts'].items():
        item = (path.parent/name).resolve()
        if not item.is_relative_to(path.parent.resolve()) or not item.is_file() or digest(item) != expected:
            raise ValueError(f'artifact hash mismatch: {name}')
    return record


def require_stage(out, stage, measurement_hashes):
    paths = sorted((Path(out)/stage).glob('attempt-*/record.json'))
    if not paths:
        raise ValueError(f'missing prerequisite: {stage}')
    # Latest attempt is authoritative; an older pass cannot hide a later stop.
    record = read_record(paths[-1])
    if record.get('stage') != stage or record['status'] != 'pass':
        raise ValueError(f'failed prerequisite {stage}: {record["status"]}')
    previous = record['measurement_hashes']
    for name, expected in previous.items():
        # A native-only consumer does not depend on the sampled source/tests.
        # Every consumer passes its complete, explicit dependency map.
        if name in measurement_hashes and measurement_hashes[name] != expected:
            return prerequisite_carryover(paths[-1], record, stage, measurement_hashes)
    return record


def git_blob(head, name):
    """Read the exact original committed bytes; never use a working-tree copy."""
    if not re.fullmatch('[0-9a-f]{40}', head) or name not in CORE+SAMPLED:
        raise ValueError('invalid carryover Git source identity')
    try:
        return subprocess.check_output(['git', 'show', f'{head}:{name}'], cwd=ROOT, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        raise ValueError('unavailable carryover Git source identity') from exc


def prerequisite_carryover(path, record, stage, measurement_hashes):
    """Explicit A/B continuity only; preserve the prior sealed attribution."""
    import yaml
    from oar.config import _UniqueLoader
    amendment = json.loads((ROOT/'decisions/dw_sampling_amendment.json').read_text())
    entries = amendment.get('prerequisite_carryover', [])
    entry = next((item for item in entries if item.get('stage') == stage
                  and item.get('record_sha256') == digest(path)
                  and item.get('previous_measurement_hashes') == record['measurement_hashes']
                  and item.get('current_measurement_hashes') == measurement_hashes), None)
    if stage not in ('prepare', 'feasibility') or record['status'] != 'pass' or entry is None:
        raise ValueError('prerequisite source/config mismatch: no exact approved carryover')
    expected_previous = CORE if stage == 'feasibility' else CORE+SAMPLED
    if set(record['measurement_hashes']) != set(expected_previous):
        raise ValueError('carryover incomplete original measurement map')
    current = source_hashes('cost')
    native = {name: current[name] for name in CORE}
    if measurement_hashes not in (native, current) or not set(record['measurement_hashes']).issubset(current):
        raise ValueError('carryover current source/config map differs from actual consumer sources')
    manifest_path = Path(path).parent/'source_manifest.json'
    if record.get('source_manifest_sha256') != digest(manifest_path) or 'source_manifest.json' not in record['artifacts']:
        raise ValueError('carryover original source manifest is not sealed')
    manifest = json.loads(manifest_path.read_text())
    if manifest['measurement_hashes'] != record['measurement_hashes']:
        raise ValueError('carryover original source identity mismatch')
    head = manifest['actual_head']
    if not re.fullmatch('[0-9a-f]{40}', head):
        raise ValueError('invalid carryover original Git head')
    original = {name: git_blob(head, name) for name in current}
    old_hashes = {name: hashlib.sha256(blob).hexdigest() for name, blob in original.items()}
    if any(old_hashes[name] != expected for name, expected in record['measurement_hashes'].items()):
        raise ValueError('carryover original Git blobs differ from sealed measurement hashes')
    allowed = {'analysis/run_dw_sampling.py', 'configs/dw_sampling.yaml',
               'decisions/dw_sampling_reopening.md', 'tests/test_dw_sampling_protocol.py'}
    changed = {name for name in current if old_hashes[name] != current[name]}
    if changed-allowed:
        raise ValueError('carryover changed protected measured source: '+', '.join(sorted(changed-allowed)))
    # A native-only consumer omits protocol tests from its map. Their changed
    # hash must still be explicitly pinned by the current amendment's freeze.
    for name in changed-set(measurement_hashes):
        if amendment.get('bound_files', {}).get(name) != current[name]:
            raise ValueError('carryover omitted changed source is not explicitly pinned')
    measured = ('configure_torch', 'Fixture', 'zero_dw_linear', 'native_variant_qualification',
                'timed_pairs', 'state_streams', 'cost_session', 'select_native_variant',
                'assert_native_equivalent', 'choose_block_length', 'primitive_qualification',
                'environment', 'dependency_inventory', 'prepare')
    def measured_asts(source):
        nodes = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in measured}
        if set(nodes) != set(measured):
            raise ValueError('carryover is missing a measured function or class')
        return nodes
    if measured_asts(original['analysis/run_dw_sampling.py']) != measured_asts((ROOT/'analysis/run_dw_sampling.py').read_bytes()):
        raise ValueError('carryover measured function/class AST changed')
    def native_config(source):
        value = yaml.load(source, Loader=_UniqueLoader)
        value['timing'].pop('cost_gate_upper_ratio', None)
        # Native A/B do not consume model/data settings. This exclusion does
        # not establish any learning/asset permission or model qualification.
        value.pop('learning', None)
        value.pop('asset', None)
        return canonical(value)
    if native_config(original['configs/dw_sampling.yaml']) != native_config((ROOT/'configs/dw_sampling.yaml').read_bytes()):
        raise ValueError('carryover native configuration changed')
    return {**record, 'carryover': {'stage': stage, 'record_path': str(path), 'record_sha256': digest(path),
            'previous_actual_head': head, 'previous_measurement_hashes': record['measurement_hashes'],
            'current_measurement_hashes': measurement_hashes, 'measured_functions_unchanged': list(measured),
            'native_config_exclusions': ['timing.cost_gate_upper_ratio', 'learning', 'asset']}}


def source_hashes(stage):
    files = CORE + (() if stage == 'feasibility' else SAMPLED)
    return {name: digest(ROOT/name) for name in files}


def validate_amendment(config):
    amendment = json.loads((ROOT/'decisions/dw_sampling_amendment.json').read_text())
    if amendment['reviewed_head'] != REVIEWED or amendment['device'] != 'mps' or amendment['dtype'] != 'float32':
        raise ValueError('invalid prospective amendment')
    subprocess.run(['git', 'cat-file', '-e', REVIEWED+'^{commit}'], cwd=ROOT, check=True, capture_output=True)
    for name, expected in amendment['bound_files'].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f'prospective source/history hash mismatch: {name}')
    if digest(config) != amendment['bound_files']['configs/dw_sampling.yaml']:
        raise ValueError('configuration bytes differ from prospective binding')
    raw = (ROOT/'decisions/gates.jsonl').read_bytes().splitlines()
    wp02 = [line for line in raw if json.loads(line)['package'] == 'WP02'][-1]
    if hashlib.sha256(wp02).hexdigest() != amendment['wp02_record_sha256']:
        raise ValueError('WP02 stop record mismatch')
    if json.loads(wp02)['status'] != 'stop':
        raise ValueError('historical WP02 must stay stopped')
    return amendment


def paired_ratio(dense, candidate):
    import numpy as np
    a, b = np.asarray(dense, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all() or (a <= 0).any() or (b <= 0).any():
        raise ValueError('paired times must be finite, positive and equally sized')
    z = np.log(b/a)
    rng = np.random.default_rng(62001)
    draws = z[rng.integers(0, len(z), (10000, len(z)))].mean(1)
    return {'ratio': float(np.exp(z.mean())), 'ci95': np.exp(np.quantile(draws, [.025, .975], method='linear')).tolist(),
            'pairs': len(z), 'analysis_seed': 62001, 'bootstrap_replicates': 10000, 'unit': 'timing block, not training seed'}


def freeze_delta(absolute_aa_errors, *, floor=0.0):
    import numpy as np
    values = np.asarray(absolute_aa_errors, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('invalid A/A observations')
    if not math.isfinite(floor) or floor < 0:
        raise ValueError('invalid noise threshold floor')
    return max(floor, 2*float(np.quantile(values, .95, method='linear')))


_zero_type = None


def zero_dw_linear(x, weight, bias):
    """Nonlearning optimistic diagnostic: dense zero dW, exact dX/db."""
    import torch
    from torch.nn import functional as F
    global _zero_type
    if _zero_type is None:
        class ZeroDW(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, w, b):
                ctx.save_for_backward(x, w)
                ctx.has_bias = b is not None
                return F.linear(x, w, b)

            @staticmethod
            def backward(ctx, d):
                x, w = ctx.saved_tensors
                d2 = d.reshape(-1, w.shape[0])
                return ((d2@w).reshape(x.shape) if ctx.needs_input_grad[0] else None,
                        torch.zeros_like(w) if ctx.needs_input_grad[1] else None,
                        d2.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None)
        _zero_type = ZeroDW
    return _zero_type.apply(x, weight, bias)


def command(attempt, args, *, timeout=240, env=None):
    """Serial subprocess with retained stdout/stderr, invocation and exit code."""
    started = time.perf_counter()
    logs = list(Path(attempt).glob('command-*.json'))
    stem = Path(attempt)/f'command-{len(logs)+1:03d}'
    try:
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout = exc.stdout or b''
        stderr = exc.stderr or b''
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors='replace')
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors='replace')
    stem.with_suffix('.stdout.txt').write_text(stdout)
    stem.with_suffix('.stderr.txt').write_text(stderr)
    write_json(stem.with_suffix('.json'), {'argv': args, 'exit_code': code, 'wall_s': time.perf_counter()-started,
                                          'utc': utc(), 'environment_overrides': {k: (env or {}).get(k) for k in
                                          ('PYTHONPATH', 'PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH', 'PYTORCH_MPS_PREFER_METAL')}})
    with (Path(attempt)/'commands.log').open('a') as stream:
        stream.write(json.dumps({'argv': args, 'exit_code': code})+'\n')
    return code, stdout, stderr


def environment(cfg):
    return {**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'PYTORCH_ENABLE_MPS_FALLBACK': cfg['fallback'],
            'PYTORCH_MPS_FAST_MATH': cfg['fast_math'], 'PYTORCH_MPS_PREFER_METAL': cfg['prefer_metal']}


def pin_mismatches(lock_text, installed):
    installed = {re.sub(r'[-_.]+', '-', k).lower(): v for k, v in installed.items()}
    differences = []
    for name, expected in re.findall(r'^([A-Za-z0-9_.-]+)==([^\s]+)', lock_text, re.MULTILINE):
        observed = installed.get(re.sub(r'[-_.]+', '-', name).lower(), 'unavailable')
        if observed != expected:
            differences.append(f'{name}: expected {expected}, observed {observed}')
    return differences


def remaining_timeout(deadline, *, clock=time.perf_counter):
    if deadline is None:
        return None
    remaining = deadline-clock()
    if remaining <= 0:
        raise TimeoutError('shared prospective stage deadline reached')
    return remaining


def check_command_deadline(exit_code, deadline, *, clock=time.perf_counter):
    if exit_code == 124:
        raise TimeoutError('subprocess exceeded remaining shared stage deadline')
    remaining_timeout(deadline, clock=clock)


@contextmanager
def exclusive_session(path):
    """Advisory host-wide ownership for this repository's experiment driver.

    External applications are not controlled. Competing repository test/bench
    processes are screened separately; unknown external GPU work is a retained
    limitation of laptop measurements, never silently killed.
    """
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('exclusive measurement session already owned') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def is_competing_work(argv, cwd):
    text = ' '.join(argv)
    in_repository = str(ROOT) in text or (cwd is not None and Path(cwd).resolve().is_relative_to(ROOT))
    return in_repository and any(k in text for k in ('pytest', 'collect_', 'run_dw_sampling.py', 'oar.cli bench'))


def competing_processes():
    import psutil
    own, ancestors = psutil.Process(), {os.getpid()}
    ancestors.update(p.pid for p in own.parents())
    blockers, unavailable = [], 0
    for process in psutil.process_iter(['pid', 'cmdline', 'cwd']):
        try:
            argv = process.info['cmdline'] or []
            if process.pid not in ancestors and is_competing_work(argv, process.info['cwd']):
                blockers.append({'pid': process.pid, 'kind': 'competing repository test/collector/benchmark'})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            unavailable += 1
    return {'blockers': blockers, 'unavailable_processes': unavailable,
            'external_policy': 'No other agent tests/collectors/benchmarks scheduled; advisory lock excludes other driver sessions. External applications remain running, are not controlled or killed.'}


def storage_guard(cfg, out):
    free = shutil.disk_usage(out).free
    used = sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file())
    maximum = cfg['resources']['maximum_output_bytes']
    if free < cfg['resources']['minimum_free_disk_bytes'] or (maximum is not None and used > maximum):
        raise MemoryError('prospective disk reserve/output allowance exceeded')
    return {'free_disk_bytes': free, 'run_output_bytes': used}


def choose_block_length(measure_for_count, initial, *, minimum_s=.5):
    if type(initial) is not int or initial <= 0:
        raise ValueError('initial dense block length must be a positive integer')
    if not math.isfinite(minimum_s) or minimum_s <= 0:
        raise ValueError('minimum dense block duration must be finite and positive')
    count, observations = initial, []
    while True:
        seconds = measure_for_count(count)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('invalid dense block duration')
        observations.append({'updates': count, 'seconds': seconds})
        if seconds >= minimum_s:
            return count, observations
        count = max(count+1, math.ceil(count*minimum_s/seconds))


def dependency_inventory():
    packages = []
    for distribution in metadata.distributions():
        verified, missing, mismatches = 0, [], []
        record = distribution.read_text('RECORD')
        for item in distribution.files or []:
            if item.hash is None or item.hash.mode != 'sha256':
                continue
            path = Path(distribution.locate_file(item))
            if not path.is_file():
                missing.append(str(item))
                continue
            observed = base64.urlsafe_b64encode(bytes.fromhex(digest(path))).decode().rstrip('=')
            if observed != item.hash.value:
                mismatches.append(str(item))
            verified += 1
        packages.append({'name': distribution.metadata['Name'], 'version': distribution.version,
                         'record_sha256': hashlib.sha256(record.encode()).hexdigest() if record else None,
                         'verified_sha256_files': verified, 'missing': missing, 'mismatches': mismatches})
    return {'packages': packages, 'lock_sha256': digest(ROOT/'requirements.lock.txt'),
            'wheel_inventory_sha256': digest(ROOT/'manifests/wheels.sha256'),
            'wheel_hashes': (ROOT/'manifests/wheels.sha256').read_text(),
            'note': 'Installed files checked against wheel RECORD SHA256 entries; wheel archive identities retained separately.'}


def configure_torch():
    if 'torch' in sys.modules:
        raise RuntimeError('fresh hardware worker must configure policy before Torch import')
    for key in ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH', 'PYTORCH_MPS_PREFER_METAL'):
        if os.environ.get(key) != '0':
            raise RuntimeError(f'{key}=0 required before Torch import')
    import torch
    torch.set_float32_matmul_precision('highest')
    if not torch.backends.mps.is_available():
        raise RuntimeError('actual MPS unavailable')
    return torch


class Fixture:
    """One residual block; baseline has no sampling state or hooks."""
    def __init__(self, cfg, shape, variant='single', mode='dense', rho=.25, session=0):
        import torch
        from oar.sampled_linear import SamplingState, named_seed
        self.cfg, self.s, self.mode = cfg, shape, mode
        self.torch, self.states = torch, {}
        self.seed = cfg['seed']+session
        init = torch.Generator().manual_seed(named_seed(self.seed, 'init', shape['id']))
        data = torch.Generator().manual_seed(named_seed(self.seed, 'data', shape['id']))
        self.init_seed, self.data_seed = init.initial_seed(), data.initial_seed()
        self.init_generator, self.data_generator = init, data
        n, di, do, stride = (shape[k] for k in ('n', 'di', 'do', 'stride'))
        storage = torch.randn(n, di*stride, generator=data).to('mps')
        self.x = storage[:, ::stride].detach().requires_grad_()
        self.p = {k: torch.nn.Parameter(v.to('mps')) for k, v in {
            'w1': torch.randn(do, di, generator=init)/math.sqrt(di), 'b1': torch.zeros(do),
            'w2': torch.randn(di, do, generator=init)/math.sqrt(do), 'b2': torch.zeros(di)}.items()}
        options = {} if variant == 'default' else {'foreach': variant == 'foreach', 'fused': variant == 'fused'}
        o = cfg['optimizer']
        self.opt = torch.optim.AdamW([{'params': [self.p['w1'], self.p['w2']], 'weight_decay': o['weight_decay']},
                                     {'params': [self.p['b1'], self.p['b2']], 'weight_decay': 0}],
                                    lr=o['lr'], betas=tuple(o['betas']), eps=o['eps'], **options)
        self.sampling_setup_s = 0.
        if mode in ('importance', 'uniform'):
            setup_start = time.perf_counter()
            self.states = {k: SamplingState(self.seed, layer, mode, rho) for k, layer in
                           (('w1', 'block0.ffn_in'), ('w2', 'block0.ffn_out'))}
            self.sampling_setup_s = time.perf_counter()-setup_start
        self.updates = 0

    def loss(self, *, diagnostic=False):
        from torch.nn import functional as F
        captured = {}
        losses = []
        for j, x in enumerate(self.x.chunk(self.s['microbatches'])):
            def linear(value, w, b):
                if self.mode == 'dense':
                    y = F.linear(value, self.p[w], self.p[b])
                elif self.mode == 'zero':
                    y = zero_dw_linear(value, self.p[w], self.p[b])
                else:
                    from oar.sampled_linear import sampled_linear
                    y = sampled_linear(value, self.p[w], self.p[b], self.states[w])
                if diagnostic:
                    y.retain_grad()
                    captured[(j, w)] = (value.detach(), y)
                return y
            y = x+linear(F.gelu(linear(x, 'w1', 'b1'), approximate='none'), 'w2', 'b2')
            losses.append(y.square().mean()/self.s['microbatches'])
        return losses, captured

    def backward(self, diagnostic=False):
        self.opt.zero_grad(set_to_none=True)
        self.x.grad = None
        losses, captured = self.loss(diagnostic=diagnostic)
        for loss in losses:
            loss.backward()
        return captured

    def step(self):
        torch = self.torch
        self.backward()
        # Native clipping checks the complete total norm before native step;
        # candidate-local probability/gradient checks are inside its backward.
        torch.nn.utils.clip_grad_norm_(list(self.p.values()), 1., error_if_nonfinite=True)
        self.opt.step()
        self.updates += 1

    def snapshot(self):
        return {'parameters': {k: v.detach().clone() for k, v in self.p.items()},
                'optimizer': copy.deepcopy(self.opt.state_dict()), 'updates': self.updates,
                'sampling': {k: v.state_dict() for k, v in self.states.items()},
                'init_state': self.init_generator.get_state().clone(), 'data_state': self.data_generator.get_state().clone()}

    def restore(self, snapshot):
        with self.torch.no_grad():
            for k, p in self.p.items():
                p.copy_(snapshot['parameters'][k])
        self.opt.load_state_dict(copy.deepcopy(snapshot['optimizer']))
        self.updates = snapshot['updates']
        for k, state in self.states.items():
            if k in snapshot['sampling']:
                state.load_state_dict(snapshot['sampling'][k])
        self.init_generator.set_state(snapshot['init_state'])
        self.data_generator.set_state(snapshot['data_state'])
        self.opt.zero_grad(set_to_none=True)
        self.x.grad = None


def primitive_qualification(cfg):
    import torch
    from oar.sampled_linear import SamplingState, sampled_linear, importance_probabilities
    rows, calibrations = [], []
    from oar.timing import measure_blocks
    for s in cfg['shapes']:
        f = Fixture(cfg, s)
        p = f.p
        # Actual shape FP32 device reduction, score-only CPU transfer,
        # FP64 CPU categorical, integer/weight transfers, gather and adjoints.
        x = f.x
        for sampler in cfg['samplers']:
            state = SamplingState(cfg['seed'], 'block0.ffn_in', sampler)
            f.opt.zero_grad(set_to_none=True)
            x.grad = None
            y = sampled_linear(x, p['w1'], p['b1'], state)
            d = torch.ones_like(y)/y.numel()
            y.backward(d)
            torch.mps.synchronize()
            torch.testing.assert_close(x.grad, d@p['w1'], rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(p['b1'].grad, d.sum(0), rtol=1e-4, atol=1e-6)
            assert p['w1'].grad is not None and bool(torch.isfinite(p['w1'].grad).all())
            probabilities = importance_probabilities(x, d)
            rows.append({'shape': s, 'sampler': sampler, 'device': str(y.device), 'status': 'pass',
                         'score_count_transferred': s['n'], 'p_sum': probabilities.sum().item(),
                         'p_min': probabilities.min().item(), 'input_stride': list(x.stride())})
        f.step()  # native shader warmup only, not a performance observation
        raw = measure_blocks(f.step, torch.mps.synchronize, 3)
        calibrations.append({'shape': s, 'native_complete_step_s': raw,
                             'boundary': 'resource forecast calibration only; not comparative performance'})
        del f
    return {'operations': rows, 'native_resource_calibration': calibrations,
            'matmul_precision': torch.get_float32_matmul_precision(), 'fallback_before_import': '0'}


def prepare(cfg, config, out, attempt):
    import platform
    from oar.hardware import resource_snapshot
    started = time.perf_counter()
    forecast = cfg['resources']['qualification_forecast_s']
    deadline = started+2*forecast
    write_json(attempt/'resource_forecast.json', {'forecast_s': forecast, 'deadline_s': 2*forecast,
               'maximum_output_bytes': cfg['resources']['maximum_output_bytes']})
    env = environment(cfg)
    write_json(attempt/'dependencies.json', dependency_inventory())
    remaining_timeout(deadline)
    dependencies = json.loads((attempt/'dependencies.json').read_text())
    if any(p['missing'] or p['mismatches'] for p in dependencies['packages']):
        return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'installed dependency RECORD hashes differ; preserve environment, do not change pins'}
    code, _, _ = command(attempt, [sys.executable, '-m', 'oar.cli', 'doctor', '--config', 'configs/discovery.yaml', '--out', str(attempt/'doctor.json')], env=env, timeout=remaining_timeout(deadline))
    check_command_deadline(code, deadline)
    if code:
        raise RuntimeError('doctor command failed')
    doctor = json.loads((attempt/'doctor.json').read_text())
    differences = pin_mismatches((ROOT/'requirements.lock.txt').read_text(), {p['name']: p['version'] for p in dependencies['packages']})
    if sys.version_info[:2] != (3, 13) or platform.machine() != 'arm64':
        differences.append('target runtime requires native arm64 CPython 3.13')
    write_json(attempt/'environment.json', {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
               'doctor': doctor, 'resource_snapshot_cpu': resource_snapshot('cpu'), 'flags': {k: env[k] for k in env if k.startswith('PYTORCH_')},
               'thermal': 'unavailable unless exposed by host; no fabricated temperature', 'start_utc': utc(),
               'pin_differences': differences, 'clock': vars(time.get_clock_info('perf_counter'))})
    for args in ([sys.executable, '-m', 'pytest', '-q', '-m', 'not mps'],
                 [sys.executable, '-m', 'oar.cli', 'report', '--gates', 'decisions/gates.jsonl', '--run-index', 'analysis/evidence_index.json', '--out', str(attempt/'historical-report-check.md')]):
        code, _, _ = command(attempt, args, env=env, timeout=remaining_timeout(deadline))
        check_command_deadline(code, deadline)
        if code:
            return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'existing CPU tests or historical report failed; inspect retained command logs'}
    if not doctor.get('gpu_claim_eligible'):
        return {'status': 'BLOCKED_HARDWARE', 'reason': 'actual qualified MPS unavailable in this process; CPU reference work only'}
    if differences:
        return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'target runtime differs from frozen native pins; CPU checks retained, no performance qualification', 'pin_differences': differences}
    for args in ([sys.executable, '-m', 'pytest', '-q', '-m', 'mps'], [sys.executable, '-m', 'pytest', '-q']):
        code, _, _ = command(attempt, args, env=env, timeout=remaining_timeout(deadline))
        check_command_deadline(code, deadline)
        if code:
            return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'actual-device existing tests failed; inspect retained logs'}
    request = {'kind': 'qualification', 'config': str(config), 'out': str(out), 'attempt': str(attempt)}
    write_json(attempt/'qualification-request.json', request)
    code, _, _ = command(attempt, worker_args('prepare', config, out, attempt/'qualification-request.json'), env=env, timeout=remaining_timeout(deadline))
    check_command_deadline(code, deadline)
    if code:
        return {'status': 'BLOCKED_HARDWARE', 'reason': 'essential actual-shape primitive failed; retained worker failure, no cost qualification'}
    if time.perf_counter()-started > 2*forecast:
        return {'status': 'INCOMPLETE_RESOURCE', 'reason': 'qualification exceeded prospective deadline'}
    return {'status': 'pass', 'reason': 'existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified',
            'asset_status': 'BLOCKED_ASSET', 'qualification': str(attempt/'qualification.json')}


def worker_args(stage, config, out, request):
    return [sys.executable, str(ROOT/'analysis/run_dw_sampling.py'), '--stage', stage, '--config', str(config), '--out', str(out), '--session-worker', str(request)]


def validate_worker_request(request, stage, config, out, request_path):
    out, config, request_path = Path(out).resolve(), Path(config).resolve(), Path(request_path).resolve()
    attempt = Path(request['attempt']).resolve()
    permitted = {'prepare': {'qualification'}, 'feasibility': {'feasibility'}, 'cost': {'cost', 'diagnostic'}}
    if (request['kind'] not in permitted.get(stage, set()) or Path(request['out']).resolve() != out
            or Path(request['config']).resolve() != config or request_path.parent != attempt
            or attempt.parent != out/stage or not (attempt/'source_manifest.json').is_file()):
        raise ValueError('invalid frozen worker request or stage boundary')
    identity = json.loads((attempt/'source_manifest.json').read_text())
    if identity['measurement_hashes'] != source_hashes(stage):
        raise ValueError('worker source hashes changed')
    if stage in ('feasibility', 'cost'):
        require_stage(out, 'prepare', source_hashes(stage))
    if stage == 'cost':
        require_stage(out, 'feasibility', source_hashes(stage))
        require_stage(out, 'correctness', source_hashes(stage))


def assert_native_equivalent(parameters, optimizer, reference_parameters, reference_optimizer):
    import torch
    for key, parameter in parameters.items():
        reference = reference_parameters[key]
        torch.testing.assert_close(parameter, reference, rtol=1e-4, atol=1e-6)
        for field in ('exp_avg', 'exp_avg_sq', 'step'):
            actual_state, reference_state = optimizer.state[parameter][field], reference_optimizer.state[reference][field]
            if field == 'step':
                # Native fused AdamW keeps its scalar counter on the device;
                # scalar native keeps it on CPU. Compare exact counter values.
                torch.testing.assert_close(actual_state.detach().cpu(), reference_state.detach().cpu(), rtol=0, atol=0)
            else:
                torch.testing.assert_close(actual_state, reference_state, rtol=1e-4, atol=1e-6)


def select_native_variant(results):
    """Choose baseline-only median speed; warmup stability is diagnostic."""
    from statistics import median
    supported = {variant: row for variant, row in results.items() if row['status'] == 'pass'}
    if not supported:
        raise RuntimeError('no qualified native AdamW variant')
    for row in supported.values():
        raw = row['pilot_s']
        if not raw or any(not math.isfinite(seconds) or seconds <= 0 for seconds in raw):
            raise ValueError('invalid native pilot duration')
    return min(supported, key=lambda variant: median(supported[variant]['pilot_s']))


def native_variant_qualification(cfg, shape, session, updates):
    import torch
    from statistics import median
    from oar.sampled_linear import named_seed
    from oar.timing import warmup, measure_blocks
    results = {}
    order_seed = named_seed(cfg['seed'], 'order', 'feasibility', str(session), shape['id'], 'native_variants')
    order = list(cfg['timing']['native_variants'])
    random.Random(order_seed).shuffle(order)
    if session % 2:
        order.reverse()
    for variant in order:
        try:
            actual, reference = Fixture(cfg, shape, variant, session=session), Fixture(cfg, shape, 'single', session=session)
            for _ in range(5):
                actual.step()
                reference.step()
                assert_native_equivalent(actual.p, actual.opt, reference.p, reference.opt)
            def block():
                for _ in range(updates):
                    actual.step()
            w = warmup(block, torch.mps.synchronize, **cfg['timing']['warmup'])
            raw = measure_blocks(block, torch.mps.synchronize, 3)
            results[variant] = {'status': 'pass', 'warmup': w, 'pilot_s': raw, 'mean_s': sum(raw)/len(raw),
                                'median_s': median(raw), 'pilot_order': order, 'order_seed': order_seed}
        except (RuntimeError, NotImplementedError, AssertionError) as exc:
            # Unsupported kernels/options or numerical disagreement disqualify
            # a native implementation; actual execution/device failures do not
            # become a reason to silently choose a different implementation.
            message = str(exc).lower()
            fatal = any(word in message for word in ('non-finite', 'nonfinite', 'invalid gradient', 'probabil',
                        'device error', 'command buffer', 'out of memory', 'allocation'))
            unsupported = not fatal and (isinstance(exc, (NotImplementedError, AssertionError)) or any(
                word in message for word in ('not supported', 'unsupported', 'not implemented', 'fused=true requires')))
            if not unsupported:
                raise
            results[variant] = {'status': 'unsupported', 'reason': str(exc), 'exception': type(exc).__name__,
                                'pilot_order': order, 'order_seed': order_seed}
    return select_native_variant(results), results


def timed_pairs(cfg, dense, candidate, common, updates, seed, raw_path=None):
    import torch
    from oar.timing import warmup, measure_blocks
    fixtures = {'dense': dense, 'candidate': candidate}
    sync, resets = torch.mps.synchronize, []
    def restore(f):
        start = time.perf_counter()
        f.restore(common)
        sync()
        resets.append(time.perf_counter()-start)
    def block(f):
        for _ in range(updates):
            f.step()
    warming = {}
    # Common initialization/moment history; no arm inherits another warmup.
    for name, f in fixtures.items():
        restore(f)
        warming[name] = warmup(lambda: block(f), sync, **cfg['timing']['warmup'])
    if raw_path:
        write_json(Path(raw_path).with_suffix('.warmup.json'), warming)
        Path(raw_path).open('x').close()
    rows, rng = [], random.Random(seed)
    for pair in range(cfg['timing']['pairs']):
        order = list(fixtures)
        rng.shuffle(order)
        seconds = {}
        for name in order:
            f = fixtures[name]
            restore(f)
            seconds[name] = measure_blocks(lambda: block(f), sync, 1)[0]
        row = {'pair': pair, 'order': order, 'seconds': seconds}
        rows.append(row)
        if raw_path:
            with Path(raw_path).open('a') as stream:
                stream.write(json.dumps(row)+'\n')
                stream.flush()
    # One-time layer-generator setup is amortized over the declared 30 timed
    # blocks. Paired reset/cloning remains separately recorded experimental
    # work, as prescribed, and is not pretended to be runtime training work.
    setup_per_block = candidate.sampling_setup_s/len(rows)
    return {'warmup': warming, 'pairs': rows, 'reset_s': resets, 'updates_per_block': updates,
            'order_seed': seed, 'sampling_setup_s': candidate.sampling_setup_s,
            'setup_amortization_updates': updates*len(rows),
            'statistics': paired_ratio([r['seconds']['dense'] for r in rows], [r['seconds']['candidate']+setup_per_block for r in rows]),
            'steady_state_statistics': paired_ratio([r['seconds']['dense'] for r in rows], [r['seconds']['candidate'] for r in rows])}


def state_streams(f):
    return {'init_seed': f.init_seed, 'data_seed': f.data_seed,
            'init_state': base64.b64encode(bytes(f.init_generator.get_state().tolist())).decode(),
            'data_state': base64.b64encode(bytes(f.data_generator.get_state().tolist())).decode(),
            'sampling': {k: {'seed': s.seed, 'layer_id': s.layer_id, 'sampler': s.sampler,
                            'state': base64.b64encode(bytes(s.generator.get_state().tolist())).decode()} for k, s in f.states.items()}}


def cost_session(cfg, request):
    import torch
    from oar.hardware import resource_snapshot
    from oar.sampled_linear import named_seed
    started = time.perf_counter()
    session, kind = request['session'], request['kind']
    deadline = None if request['deadline_s'] is None else started+request['deadline_s']
    results = {'session': session, 'kind': kind, 'start_utc': utc(), 'shapes': [], 'resources_before': resource_snapshot('mps')}
    result_path = Path(request['result'])
    def check_deadline():
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError('prospectively bound complete session deadline exceeded')
    try:
        shapes = list(cfg['shapes'])
        for shape in shapes:
            check_deadline()
            # Available memory screening precedes shape allocation; never disable
            # the native allocator's safeguards. Observed RSS/Metal overlap.
            resource = resource_snapshot('mps')
            n, di, do = (shape[k] for k in ('n', 'di', 'do'))
            predicted = 3*64*(2*di*do+di+do)+64*n*do
            ceiling = min(.7*resource['physical_memory_bytes'], .8*torch.mps.recommended_max_memory(), resource['available_memory_bytes'])
            if predicted > ceiling:
                raise MemoryError('fixed shape does not fit conservative current memory screening')
            calibration = next(v for v in request['calibration'] if v['shape']['id'] == shape['id'])
            step_s = min(calibration['native_complete_step_s'])
            minimum_s = cfg['timing']['minimum_block_s']
            updates = max(1, math.ceil(minimum_s/step_s))
            from oar.timing import measure_blocks
            pilot_observations = []
            if kind == 'feasibility':
                pilot = Fixture(cfg, shape, session=session)
                initial_raw = []
                def measure_initial(count):
                    def block():
                        for _ in range(count):
                            pilot.step()
                    raw = measure_blocks(block, torch.mps.synchronize, 3)
                    initial_raw.append(raw)
                    return min(raw)
                updates, pilot_observations = choose_block_length(measure_initial, updates, minimum_s=minimum_s)
                for observation, raw in zip(pilot_observations, initial_raw):
                    observation['raw_s'] = raw
                del pilot
                variant, variants = native_variant_qualification(cfg, shape, session, updates)
            else:
                previous = next(v for v in request['baseline']['shapes'] if v['shape']['id'] == shape['id'])
                variant, variants = previous['selected_variant'], previous['variants']
                updates = previous['updates_per_block']
            write_json(result_path.parent/(result_path.stem+'-'+shape['id']+'-native-variants.json'), variants)
            dense = Fixture(cfg, shape, variant, session=session)
            selected_raw = []
            def measure_selected(count):
                def block():
                    for _ in range(count):
                        dense.step()
                raw = measure_blocks(block, torch.mps.synchronize, 3)
                selected_raw.append(raw)
                return min(raw)
            # Baseline-only calibration starts at B's count in D, increasing
            # as needed before any candidate is measured. All matched arms
            # then use this count, including the complete native optimizer.
            initial_updates = updates
            updates, selected_pilot = choose_block_length(measure_selected, updates, minimum_s=minimum_s)
            for observation, raw in zip(selected_pilot, selected_raw):
                observation['raw_s'] = raw
            # Calibration history is discarded; declared common history is 8.
            del dense
            dense = Fixture(cfg, shape, variant, session=session)
            for _ in range(cfg['timing']['history_updates']):
                dense.step()
            common = dense.snapshot()
            row = {'shape': shape, 'selected_variant': variant, 'variants': variants, 'updates_per_block': updates,
                   'dense_pilot_s': [p['seconds'] for p in pilot_observations],
                   'dense_pilot': pilot_observations, 'comparisons': {}, 'resources_before': resource,
                   'selected_dense_pilot': selected_pilot, 'initial_updates_per_block': initial_updates,
                   'minimum_block_s': minimum_s, 'baseline_only_recalibrated': updates != initial_updates,
                   'predicted_working_set_bytes': predicted, 'streams': state_streams(dense)}
            if kind == 'feasibility':
                arms = [('AA', 'dense', 1.), ('zero', 'zero', 1.)]
            else:
                arms = [(f'{sampler}-{rho}', sampler, rho) for sampler in cfg['samplers'] for rho in cfg['rhos']]
                random.Random(named_seed(cfg['seed'], 'order', kind, str(session), shape['id'], 'arms')).shuffle(arms)
            for name, mode, rho in arms:
                check_deadline()
                candidate = Fixture(cfg, shape, variant, mode, rho, session)
                comparison_common = {**common, 'sampling': {k: v.state_dict() for k, v in candidate.states.items()}}
                row['comparisons'][name] = timed_pairs(cfg, dense, candidate, comparison_common, updates,
                    named_seed(cfg['seed'], 'order', kind, str(session), shape['id'], name),
                    result_path.parent/(result_path.stem+'-'+shape['id']+'-'+name+'-pairs.jsonl'))
                row['comparisons'][name]['sampling_streams_final'] = state_streams(candidate)
                torch.mps.synchronize()
                row['comparisons'][name]['resources_candidate_live'] = resource_snapshot('mps')
                row['comparisons'][name]['memory_context'] = 'endpoint snapshot with dense/common and candidate live, plus allocator caches; overlapping fields are not added and are not exact peaks'
                del candidate
            if kind == 'feasibility':
                dense.restore(common)
                operands = list(dense.backward(diagnostic=True).values())
                def dw_products():
                    for x, y in operands:
                        _gradient = y.grad.T@x
                row['dense_dw_group_profile_s'] = measure_blocks(dw_products, torch.mps.synchronize, 3)
                row['profile_note'] = 'separate dense-dW GEMM group diagnostic; synchronization/graph boundaries differ, never add component profiles into an observed full-step time'
            row['resources_after'] = resource_snapshot('mps')
            results['shapes'].append(row)
            # Persist each completed shape in its own immutable file.
            write_json(result_path.parent/(result_path.stem+'-'+shape['id']+'.json'), row)
            del dense, common
            check_deadline()
        results['status'] = 'pass'
    except (RuntimeError, TimeoutError, MemoryError) as exc:
        results.update(status='INCOMPLETE_RESOURCE' if isinstance(exc, (TimeoutError, MemoryError)) else 'INCONCLUSIVE_MEASUREMENT',
                       failure=str(exc), invalid_condition=getattr(exc, 'condition', None), traceback=traceback.format_exc())
    results.update(wall_s=time.perf_counter()-started, resources_after=resource_snapshot('mps'))
    write_json(result_path, results)
    return results


def forecast_stage(cfg, calibration, kind):
    # Conservative measured native maximum; candidate components use an explicit
    # 3x allowance, with all warmup, pairs, variants, cloning and startup included.
    timing = cfg['timing']
    minimum_s = timing['minimum_block_s']
    arms = 2 if kind == 'feasibility' else len(cfg['samplers'])*len(cfg['rhos'])
    per_shape = []
    for row in calibration:
        maximum = max(row['native_complete_step_s'])
        updates = max(1, math.ceil(minimum_s/min(row['native_complete_step_s'])))
        block = max(minimum_s, updates*maximum)
        blocks = arms*(2*timing['pairs']+2*timing['warmup']['maximum'])
        blocks += len(timing['native_variants'])*(timing['warmup']['maximum']+3) if kind == 'feasibility' else 0
        per_shape.append({'shape': row['shape']['id'], 'updates_per_block': updates,
                          'minimum_block_s': minimum_s, 'bound_block_s': block, 'bound_blocks': blocks,
                          'forecast_s': 3*block*blocks+128*maximum+15})
    forecast = timing['sessions']*sum(r['forecast_s'] for r in per_shape)+60
    deadline_s = None if cfg['resources']['maximum_stage_s'] is None else cfg['resources']['deadline_multiplier']*forecast
    return {'kind': kind, 'sessions': timing['sessions'], 'automatic_replacement_sessions': 0,
            'per_shape': per_shape, 'forecast_s': forecast,
            'deadline_s': deadline_s, 'includes': 'two sessions, 3x component allowance, histories, warmup, reset, variants, startup and serialization; advisory without a configured deadline',
            'diagnostic_allowance_s': 0 if kind == 'feasibility' else 300}


def offline_diagnostics(cfg, request):
    """Paid native-state information study, never consumed by timed training."""
    import torch
    from oar.sampled_linear import SamplingState, importance_probabilities, gather_weight_gradient
    from oar.timing import measure_blocks
    start = time.perf_counter()
    deadline = None if request['deadline_s'] is None else start+request['deadline_s']
    shape = next(s for s in cfg['shapes'] if s['id'] == 'middle4096')
    f = Fixture(cfg, shape, request['variant'])
    rows, profiles = [], []
    for target_update in cfg['diagnostic']['native_states']:
        while f.updates < target_update:
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError('offline diagnostic deadline')
            f.step()
        common = f.snapshot()
        captures = f.backward(diagnostic=True)
        dense_grads = {k: p.grad.detach().clone() for k, p in f.p.items()}
        operands = {w: (x, y.grad.detach()) for (_micro, w), (x, y) in captures.items()}
        layer_info = {}
        for w, (x, d) in operands.items():
            # FP64 CPU full operands are allowed ONLY in this charged offline
            # diagnostic, never in the runtime sampler or headline timed step.
            xd, dd = x.detach().cpu().double(), d.detach().cpu().double()
            a = xd.norm(dim=1)*dd.norm(dim=1)
            exact = dd.T@xd
            norm2 = exact.square().sum().item()
            p = importance_probabilities(x, d)
            info = {'n': len(x), 'exact_norm2': norm2,
                    'heterogeneity': (len(a)*a.square().sum()/a.sum().square()).item(),
                    'score_min': a.min().item(), 'score_max': a.max().item(), 'variance': {}}
            for sampler in cfg['samplers']:
                probabilities = p if sampler == 'importance' else torch.full((len(x),), 1/len(x), dtype=torch.float64)
                for rho in cfg['rhos']:
                    k = math.ceil(rho*len(x))
                    variance = ((a.square()/probabilities).sum().item()-norm2)/k
                    info['variance'][f'{sampler}-{rho}'] = {'conditional_variance': variance,
                        'normalized_variance': variance/norm2 if norm2 else None}
            layer_info[w] = info
            profiles.append({'native_update': target_update, 'layer': w,
                'dense_dw_s': measure_blocks(lambda: d.T@x, torch.mps.synchronize, 3),
                'scores_cpu_probabilities_checks_s': measure_blocks(lambda: importance_probabilities(x, d), torch.mps.synchronize, 3),
                'note': 'separate synchronized profiles, not additive full-step observations'})
        # Native displacement/reference loss from the same committed history.
        def apply_grads(grads):
            f.restore(common)
            for key, parameter in f.p.items():
                parameter.grad = grads[key].detach().clone()
            torch.nn.utils.clip_grad_norm_(list(f.p.values()), 1., error_if_nonfinite=True)
            f.opt.step()
            return {key: parameter.detach().clone() for key, parameter in f.p.items()}
        native_parameters = apply_grads(dense_grads)
        with torch.no_grad():
            native_loss = sum(f.loss()[0]).item()
        for sampler in cfg['samplers']:
            for rho in cfg['rhos']:
                states = {w: SamplingState(cfg['seed'], 'diagnostic.'+w+'.'+str(target_update), sampler, rho) for w in operands}
                for draw in range(cfg['diagnostic']['draws']):
                    if deadline is not None and time.perf_counter() > deadline:
                        raise TimeoutError('offline diagnostic deadline')
                    grads = dict(dense_grads)
                    errors = {}
                    for w, (x, d) in operands.items():
                        k, generator = math.ceil(rho*len(x)), states[w].generator
                        if sampler == 'importance':
                            p = importance_probabilities(x, d)
                            indices = torch.multinomial(p, k, replacement=True, generator=generator)
                            grads[w] = gather_weight_gradient(x, d, indices, p, k)
                        else:
                            indices = torch.randint(len(x), (k,), generator=generator).to(x.device)
                            grads[w] = d.index_select(0, indices).T@(x.index_select(0, indices)*(len(x)/k))
                        errors[w] = (grads[w]-dense_grads[w]).square().sum().item()/layer_info[w]['exact_norm2']
                    sampled_parameters = apply_grads(grads)
                    with torch.no_grad():
                        loss = sum(f.loss()[0]).item()
                    displacement_difference = math.sqrt(sum((sampled_parameters[w]-native_parameters[w]).square().sum().item() for w in f.p))
                    rows.append({'native_update': target_update, 'sampler': sampler, 'rho': rho, 'draw': draw,
                                 'normalized_gradient_squared_errors': errors, 'displacement_difference_l2': displacement_difference,
                                 'native_one_step_loss': native_loss, 'sampled_one_step_loss': loss, 'one_step_loss_difference': loss-native_loss})
        write_json(Path(request['attempt'])/f'diagnostic-state-{target_update}.json', {'layers': layer_info, 'rows': [r for r in rows if r['native_update'] == target_update]})
        f.restore(common)
    torch.mps.synchronize()
    return {'shape': shape, 'draws': cfg['diagnostic']['draws'], 'states': cfg['diagnostic']['native_states'], 'rows': rows,
            'profiles': profiles, 'wall_s': time.perf_counter()-start,
            'interpretation': 'offline conditional variance and copied native-AdamW one-step diagnostics; not learning success or a runtime acceptance rule'}


def hardware_stage(cfg, config, out, attempt, kind):
    bindings = source_hashes(kind)
    prepared = require_stage(out, 'prepare', bindings)
    qualification = json.loads(Path(prepared['qualification']).read_text())
    calibration = qualification['native_resource_calibration']
    baseline = None
    if kind == 'cost':
        feasibility = require_stage(out, 'feasibility', bindings)
        require_stage(out, 'correctness', bindings)
        baseline = feasibility
    forecast = forecast_stage(cfg, calibration, kind)
    if kind == 'cost':
        forecast['forecast_s'] += 300
        if forecast['deadline_s'] is not None:
            forecast['deadline_s'] = cfg['resources']['deadline_multiplier']*forecast['forecast_s']
    write_json(attempt/'resource_forecast.json', forecast)
    maximum_stage_s = cfg['resources']['maximum_stage_s']
    if maximum_stage_s is not None and forecast['deadline_s'] > maximum_stage_s:
        return {'status': 'INCOMPLETE_RESOURCE', 'reason': 'inclusive measured forecast exceeds configured stage capacity; no expensive session launched'}
    deadline = None if forecast['deadline_s'] is None else time.perf_counter()+forecast['deadline_s']
    noise_shapes = cfg['timing']['noise_calibration_shapes']
    frozen = {'source': 'native-only pooled two-session A/A; frozen before sampler timing',
              'noise_calibration_shapes': noise_shapes, 'floor': cfg['timing']['delta_floor']}
    if baseline:
        delta = baseline['delta']
        write_json(attempt/'frozen_delta.json', {'delta': delta, **frozen})
    sessions, invocation = [], 0
    def run_session(session, slot):
        nonlocal invocation
        invocation += 1
        request = {'kind': kind, 'session': session, 'config': str(config), 'out': str(out), 'attempt': str(attempt),
                   'result': str(attempt/f'session-invocation-{invocation}.json'), 'calibration': calibration,
                   'deadline_s': remaining_timeout(deadline)}
        if baseline:
            request['baseline'] = baseline['sessions'][slot]
        request_path = attempt/f'session-invocation-{invocation}-request.json'
        write_json(request_path, request)
        code, _, _ = command(attempt, worker_args(kind, config, out, request_path), env=environment(cfg),
                             timeout=remaining_timeout(deadline))
        if code or not Path(request['result']).is_file():
            return {'status': 'INCOMPLETE_RESOURCE', 'failure': 'session interrupted or deadline exceeded; raw prefix retained'}
        return json.loads(Path(request['result']).read_text())
    for slot in range(cfg['timing']['sessions']):
        result = run_session(slot, slot)
        if result['status'] != 'pass':
            return {'status': result['status'], 'reason': result.get('failure'), 'sessions': sessions,
                    'failed_session': result}
        sessions.append(result)
    if not baseline:
        delta = freeze_delta([abs(p['seconds']['dense']/p['seconds']['candidate']-1)
            for s in sessions for row in s['shapes'] if row['shape']['id'] in noise_shapes
            for p in row['comparisons']['AA']['pairs']], floor=cfg['timing']['delta_floor'])
        write_json(attempt/'frozen_delta.json', {'delta': delta, **frozen})
    unstable = [(s['session'], row['shape']['id'], name) for s in sessions for row in s['shapes']
                for name, comparison in row['comparisons'].items() if not all(v['stable'] for v in comparison['warmup'].values())]
    diagnostics = None
    if kind == 'cost':
        request = {'kind': 'diagnostic', 'config': str(config), 'out': str(out), 'attempt': str(attempt),
                   'variant': next(row for row in baseline['sessions'][0]['shapes'] if row['shape']['id'] == 'middle4096')['selected_variant'],
                   'deadline_s': remaining_timeout(deadline)}
        request_path = attempt/'diagnostic-request.json'
        write_json(request_path, request)
        code, _, _ = command(attempt, worker_args(kind, config, out, request_path), env=environment(cfg), timeout=remaining_timeout(deadline))
        if code:
            return {'status': 'INCOMPLETE_RESOURCE', 'reason': 'paid offline diagnostic incomplete; raw session timing retained', 'sessions': sessions, 'delta': delta}
        diagnostics = str(attempt/'diagnostic.json')
    eligible, noise_guard_eligible = [], []
    gate_upper_ratio = 1-delta if kind == 'feasibility' else cfg['timing']['cost_gate_upper_ratio']
    names = ['zero'] if kind == 'feasibility' else [f'{s}-{r}' for s in cfg['samplers'] for r in cfg['rhos']]
    for name in names:
        for shape_id in ('middle2048', 'middle4096'):
            comparisons = [next(row for row in s['shapes'] if row['shape']['id'] == shape_id)['comparisons'][name] for s in sessions]
            if all(c['statistics']['ci95'][1] < 1-delta for c in comparisons):
                noise_guard_eligible.append({'arm': name, 'shape': shape_id})
            if all(c['statistics']['ci95'][1] < gate_upper_ratio for c in comparisons):
                eligible.append({'arm': name, 'shape': shape_id})
    return {'status': 'pass' if eligible else 'STOP_COST', 'delta': delta, 'eligible': eligible, 'sessions': sessions,
            'gate_upper_ratio': gate_upper_ratio, 'noise_guard_upper_ratio': 1-delta, 'noise_guard_eligible': noise_guard_eligible,
            'diagnostics': diagnostics, 'unstable_warmup_diagnostics': unstable,
            'automatic_replacement_sessions': 0,
            'reason': f'upper95% paired ratio must be below {gate_upper_ratio} in both sessions at a planned-model middle shape',
            'zero_semantics': 'nonlearning dense zero dW with exact dX/db and native optimizer moments/decay; optimistic diagnostic, not universal lower bound'}


def correctness(cfg, config, out, attempt):
    paths = sorted((out/'prepare').glob('attempt-*/record.json'))
    prepared = read_record(paths[-1]) if paths else None
    if prepared and prepared['status'] not in ('pass', 'BLOCKED_HARDWARE'):
        raise ValueError('CPU correctness permits only qualified or hardware-blocked preparation')
    env = environment(cfg)
    code, stdout, _ = command(attempt, [sys.executable, '-m', 'pytest', '-q', 'tests/test_sampled_linear.py', 'tests/test_dw_sampling_protocol.py'], env=env)
    return {'status': 'pass' if not code else 'INCONCLUSIVE_MEASUREMENT', 'reason': 'CPU operator/protocol reference tests; actual-device qualification separately in prepare',
            'cpu_correctness_only': prepared is None or prepared['status'] != 'pass', 'test_output': stdout.strip()}


def summarize_run(out):
    records = []
    for stage in STAGES[:-1]:
        for path in sorted((Path(out)/stage).glob('attempt-*/record.json')):
            records.append({'path': str(path), 'record_sha256': digest(path), **read_record(path)})
    latest = {r['stage']: r for r in records}
    terminal = next((latest[stage]['status'] for stage in reversed(STAGES[:-1])
                     if stage in latest and latest[stage]['status'] not in ('pass', 'not_run')), None)
    if terminal is None:
        terminal = 'INCOMPLETE_RESOURCE'
    return {'schema_version': 1, 'terminal_state': terminal, 'records': records, 'primary_time_to_target': None,
            'new_performance_claim': False, 'historical_evidence': 'inherited, unchanged, separately source-attributed',
            'recommendation': 'Stop at the recorded boundary; reopening requires the unmet prerequisite and a new hash-bound prospective version.'}


def render_summary(summary):
    lines = ['<!-- SPDX-License-Identifier: CC-BY-4.0 -->', '# Bounded dW sampling report', '',
             f'Terminal state: **{summary["terminal_state"]}**.', '',
             'Fixed-budget norm-importance dW sampling and the direct-randint uniform control retain native forward, exact input/bias adjoints and native AdamW. Historical OAR results are unchanged.', '',
             'No time-to-held-out-target or endpoint language quality result exists. No corpus bodies were ingested; no trainer was added before its prerequisites.', '',
             '| Stage / attempt | State | Reason |', '| --- | --- | --- |']
    for r in summary['records']:
        lines.append(f'| {r["stage"]} / {Path(r["path"]).parent.name} | {r["status"]} | {r.get("reason", "")} |')
    lines += ['', 'Raw command exits/stdout/stderr, resource forecasts, source identities, stage records and their hashes accompany this report. Timing block intervals describe blocks, never independent training seeds.', '', summary['recommendation'], '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=STAGES)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true', help='prepare inventory only; no tests, training, bodies or device operations')
    parser.add_argument('--session-worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    args.config, args.out = args.config.resolve(), args.out.resolve()
    amendment = validate_amendment(args.config)
    if args.session_worker:
        request = json.loads(args.session_worker.read_text())
        validate_worker_request(request, args.stage, args.config, args.out, args.session_worker)
        configure_torch()
        if request['kind'] == 'qualification':
            write_json(Path(request['attempt'])/'qualification.json', primitive_qualification(cfg))
        elif request['kind'] == 'diagnostic':
            write_json(Path(request['attempt'])/'diagnostic.json', offline_diagnostics(cfg, request))
        else:
            cost_session(cfg, request)
        return 0
    attempt = allocate_attempt(args.out, args.stage)
    start = time.perf_counter()
    identity = {'reviewed_head': REVIEWED, 'actual_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                'amendment_sha256': digest(ROOT/'decisions/dw_sampling_amendment.json'), 'config_path': str(args.config),
                'measurement_hashes': source_hashes(args.stage), 'bound_files': amendment['bound_files'], 'utc': utc()}
    write_json(attempt/'source_manifest.json', identity)
    result = {}
    ownership = ExitStack()
    try:
        if args.stage in ('prepare', 'feasibility', 'correctness', 'cost') and not args.dry_run:
            ownership.enter_context(exclusive_session(ROOT/'.superpowers/dw_sampling/measurement.lock'))
            competing = competing_processes()
            write_json(attempt/'exclusive_session.json', {'pid': os.getpid(), 'start_utc': utc(), **competing,
                       'storage': storage_guard(cfg, args.out)})
            if competing['blockers']:
                raise MeasurementInvalid('overlapping_process', 'competing repository test/collector/benchmark process detected')
        if args.dry_run:
            if args.stage != 'prepare':
                raise ValueError('--dry-run is prepare-only')
            result = {'status': 'not_run', 'reason': 'prepare-only dry run: validated immutable config/source/history; no device work/tests/training/downloads'}
        elif args.stage == 'prepare':
            result = prepare(cfg, args.config, args.out, attempt)
        elif args.stage in ('feasibility', 'cost'):
            result = hardware_stage(cfg, args.config, args.out, attempt, args.stage)
        elif args.stage == 'correctness':
            result = correctness(cfg, args.config, args.out, attempt)
        elif args.stage in ('pilot', 'confirm'):
            prerequisite = 'cost' if args.stage == 'pilot' else 'pilot'
            require_stage(args.out, prerequisite, source_hashes(args.stage))
            result = {'status': 'BLOCKED_ASSET', 'reason': 'WikiText license/use restriction unresolved; language infrastructure deliberately not implemented or launched'}
        else:
            summary = summarize_run(args.out)
            write_json(attempt/'summary.json', summary)
            (attempt/'report.md').write_text(render_summary(summary))
            result = {'status': 'pass', 'reason': 'incomplete/blocked summary rendered without upgrading scientific evidence', 'terminal_state': summary['terminal_state']}
        storage_guard(cfg, args.out)
    except Exception as exc:
        (attempt/'failure.txt').write_text(traceback.format_exc())
        result = {'status': 'INCONCLUSIVE_MEASUREMENT' if isinstance(exc, MeasurementInvalid) else 'INCOMPLETE_RESOURCE', 'reason': str(exc), 'exception': type(exc).__name__}
    finally:
        ownership.close()
    record = seal_record(attempt, {'stage': args.stage, **result, 'measurement_hashes': source_hashes(args.stage),
                                  'invocation_wall_s': time.perf_counter()-start, 'source_manifest_sha256': digest(attempt/'source_manifest.json')})
    print(json.dumps({'stage': args.stage, 'status': record['status'], 'attempt': str(attempt), 'reason': record['reason']}))
    return 0 if record['status'] in ('pass', 'not_run', 'BLOCKED_ASSET', 'BLOCKED_HARDWARE', 'STOP_COST', 'INCONCLUSIVE_MEASUREMENT', 'INCOMPLETE_RESOURCE') else 1


if __name__ == '__main__':
    raise SystemExit(main())
