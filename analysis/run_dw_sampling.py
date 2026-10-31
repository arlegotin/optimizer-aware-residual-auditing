# SPDX-License-Identifier: AGPL-3.0-only
"""Append-only dW sampling and gated full-model qualification.

Language qualification requires the frozen native cost and asset prerequisites.
Tuning/confirmation remain staged behind a future dispatcher. Importing starts
no work, and no historical oracle is invoked.
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
PROTOCOL = '916e919075f7ded0b253e9fe7c6441439a345c765decc62cbaa8d167e5e2d399'
STAGES = ('prepare', 'feasibility', 'correctness', 'cost', 'pilot', 'confirm', 'summarize')
CORE = ('analysis/run_dw_sampling.py', 'configs/dw_sampling.yaml', 'decisions/dw_sampling_scope.md',
        'decisions/dw_sampling_reopening.md',
        'src/oar/timing.py', 'src/oar/hardware.py')
LEARNING = ('src/oar/sampling_lm.py', 'src/oar/sampling_data.py')
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
    attempts = sorted(path for path in (Path(out)/stage).glob('attempt-*') if path.is_dir())
    if not attempts:
        raise ValueError(f'missing prerequisite: {stage}')
    paths = [attempts[-1]/'record.json']
    if not paths[-1].is_file() or not paths[-1].with_suffix('.sha256').is_file():
        raise ValueError(f'incomplete prerequisite: {stage}')
    # Latest attempt is authoritative; an older pass cannot hide a later stop.
    record = read_record(paths[-1])
    if record.get('stage') != stage or record['status'] != 'pass':
        raise ValueError(f'failed prerequisite {stage}: {record["status"]}')
    if measurement_hashes not in (source_hashes('feasibility'), source_hashes('cost'), source_hashes('pilot')):
        raise ValueError('prerequisite source/config map differs from actual consumer sources')
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
    """Exact reviewed prerequisite continuity; preserve prior sealed attribution.

    Selected AST equality is a consistency check, never a transitive proof.
    Each approved new source map requires renewed review of the complete diff.
    """
    amendment = json.loads((ROOT/'decisions/dw_sampling_amendment.json').read_text())
    entries = amendment.get('prerequisite_carryover', [])
    entry = next((item for item in entries if item.get('stage') == stage
                  and item.get('record_sha256') == digest(path)
                  and item.get('previous_measurement_hashes') == record['measurement_hashes']
                  and item.get('current_measurement_hashes') == measurement_hashes), None)
    if stage not in ('prepare', 'feasibility', 'cost') or record['status'] != 'pass' or entry is None:
        raise ValueError('prerequisite source/config mismatch: no exact approved carryover')
    expected_previous = CORE if stage == 'feasibility' else CORE+SAMPLED
    if set(record['measurement_hashes']) != set(expected_previous):
        raise ValueError('carryover incomplete original measurement map')
    current = source_hashes('cost')
    native = {name: current[name] for name in CORE}
    learning = source_hashes('pilot')
    if measurement_hashes not in (native, current, learning) or not set(record['measurement_hashes']).issubset(current):
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
                'environment', 'dependency_inventory', 'paired_ratio', 'freeze_delta')
    if stage == 'prepare':
        measured += ('prepare',)
    if stage == 'cost':
        measured += ('offline_diagnostics', 'hardware_stage', 'forecast_stage')
    def measured_asts(source):
        nodes = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in measured}
        if set(nodes) != set(measured):
            raise ValueError('carryover is missing a measured function or class')
        return nodes
    if measured_asts(original['analysis/run_dw_sampling.py']) != measured_asts((ROOT/'analysis/run_dw_sampling.py').read_bytes()):
        raise ValueError('carryover measured function/class AST changed')
    if native_carryover_config(original['configs/dw_sampling.yaml'], stage) != native_carryover_config((ROOT/'configs/dw_sampling.yaml').read_bytes(), stage):
        raise ValueError('carryover native configuration changed')
    for name, expected in measurement_hashes.items():
        if amendment.get('bound_files', {}).get(name) != expected:
            raise ValueError('carryover consumer source is not explicitly pinned by current amendment')
    return {**record, 'carryover': {'stage': stage, 'record_path': str(path), 'record_sha256': digest(path),
            'previous_actual_head': head, 'previous_measurement_hashes': record['measurement_hashes'],
            'current_measurement_hashes': measurement_hashes, 'measured_functions_unchanged': list(measured),
            'native_config_exclusions': ([] if stage == 'cost' else ['timing.cost_gate_upper_ratio'])+['learning', 'asset', 'schema_version'],
            'supported_config_schemas': [2, 3]}}


def native_carryover_config(source, stage):
    """Native-only dependency comparison; explicitly supported schema metadata."""
    import yaml
    from oar.config import _UniqueLoader
    value = yaml.load(source, Loader=_UniqueLoader)
    if type(value.get('schema_version')) is not int or value['schema_version'] not in (2, 3):
        raise ValueError('unsupported native carryover configuration schema')
    value.pop('schema_version')
    if stage != 'cost':
        value['timing'].pop('cost_gate_upper_ratio', None)
    value.pop('learning', None)
    value.pop('asset', None)
    return canonical(value)


def source_hashes(stage):
    files = CORE + (() if stage == 'feasibility' else SAMPLED)
    if stage in ('prepare', 'correctness', 'pilot', 'confirm'):
        files += LEARNING
    return {name: digest(ROOT/name) for name in files}


def validate_source_stability(identity, stage, config):
    """Reject source/config/amendment drift before attributing a stage result."""
    try:
        stable = (source_hashes(stage) == identity['measurement_hashes']
                  and digest(config) == identity['config_sha256']
                  and digest(ROOT/'decisions/dw_sampling_amendment.json') == identity['amendment_sha256']
                  and all(digest(ROOT/name) == expected for name, expected in identity['bound_files'].items()))
    except (OSError, KeyError) as exc:
        raise MeasurementInvalid('source_changed', 'source/config/amendment changed during stage attempt') from exc
    if not stable:
        raise MeasurementInvalid('source_changed', 'source/config/amendment changed during stage attempt')


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


def command(attempt, args, *, timeout=None, env=None):
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
    maximum_stage_s = cfg['resources']['maximum_stage_s']
    deadline = None if maximum_stage_s is None else started+maximum_stage_s
    write_json(attempt/'resource_forecast.json', {'forecast_s': forecast, 'deadline_s': maximum_stage_s,
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
    return {'status': 'pass', 'reason': 'existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified',
            'asset_status': 'BLOCKED_ASSET', 'qualification': str(attempt/'qualification.json')}


def worker_args(stage, config, out, request):
    return [sys.executable, str(ROOT/'analysis/run_dw_sampling.py'), '--stage', stage, '--config', str(config), '--out', str(out), '--session-worker', str(request)]


def validate_worker_request(request, stage, config, out, request_path):
    out, config, request_path = Path(out).resolve(), Path(config).resolve(), Path(request_path).resolve()
    attempt = Path(request['attempt']).resolve()
    permitted = {'prepare': {'qualification'}, 'feasibility': {'feasibility'}, 'cost': {'cost', 'diagnostic'},
                 'pilot': {'model_native', 'model_cost', 'model_calibration'}}
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
    if stage == 'pilot':
        prerequisites = {name: require_stage(out, name, source_hashes(stage))
                         for name in ('prepare', 'feasibility', 'correctness', 'cost')}
        validate_model_worker_request(request, load_config(config), attempt, identity, config, prerequisites['cost'])


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
    unfinished = terminal is None
    if terminal is None:
        terminal = 'INCOMPLETE_RESOURCE'
    return {'schema_version': 1, 'terminal_state': terminal, 'records': records, 'primary_time_to_target': None,
            'new_performance_claim': False, 'historical_evidence': 'inherited, unchanged, separately source-attributed',
            'study_unfinished': unfinished,
            'recommendation': ('Continue the authorized unfinished workflow after its frozen prerequisites; qualification alone does not establish learning benefit.'
                               if unfinished else 'Retain the recorded boundary and evidence; any repair must freeze its changed sources before repeating affected work.')}


def render_summary(summary):
    lines = ['<!-- SPDX-License-Identifier: CC-BY-4.0 -->', '# Bounded dW sampling report', '',
             f'Terminal state: **{summary["terminal_state"]}**.', '',
             'Fixed-budget norm-importance dW sampling and the direct-randint uniform control retain native forward, exact input/bias adjoints and native AdamW. Historical OAR results are unchanged.', '',
             'Qualification and component costs do not establish time to a held-out language target. Asset acquisition, parser manifests and any reached model qualification are recorded in their pilot attempt; the official test remains protected until its final reporting gate.', '',
             '| Stage / attempt | State | Reason |', '| --- | --- | --- |']
    for r in summary['records']:
        lines.append(f'| {r["stage"]} / {Path(r["path"]).parent.name} | {r["status"]} | {r.get("reason", "")} |')
    lines += ['', 'Raw command exits/stdout/stderr, resource forecasts, source identities, stage records and their hashes accompany this report. Timing block intervals describe blocks, never independent training seeds.', '', summary['recommendation'], '']
    return '\n'.join(lines)


def assert_language_asset_permission(cfg):
    """Honor only the frozen, narrow documentary decision; never old metadata."""
    asset = cfg['asset']
    if asset.get('body_downloads') is not True:
        raise ValueError('corpus body acquisition is disabled by the prospective configuration')
    path = ROOT/asset['decision_path']
    if digest(path) != asset['decision_sha256']:
        raise ValueError('asset decision hash mismatch')
    decision = json.loads(path.read_text())
    if (decision.get('dataset') != asset['name'] or decision.get('revision') != asset['revision']
            or decision.get('decision') != 'permission_evidence_reviewed_for_conditional_local_only_research'):
        raise ValueError('asset decision does not permit this exact local research use')
    return decision


def fetch_pinned_file(url, destination, expected_size, expected_sha256):
    """One verified HTTPS fetch; existing evidence and failed prefixes stay intact."""
    import urllib.request
    import uuid
    destination = Path(destination)
    if (not url.startswith('https://') or type(expected_size) is not int or expected_size <= 0
            or not re.fullmatch('[0-9a-f]{64}', expected_sha256)):
        raise ValueError('invalid pinned HTTPS file identity')
    started = time.perf_counter()
    if destination.exists():
        if destination.stat().st_size != expected_size or digest(destination) != expected_sha256:
            raise ValueError('existing cached file differs from the pinned bytes; preserved without overwrite')
        return {'path': str(destination), 'sha256': expected_sha256, 'size_bytes': expected_size,
                'reused': True, 'wall_s': time.perf_counter()-started}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    with urllib.request.urlopen(url) as response, partial.open('xb') as output:
        shutil.copyfileobj(response, output, length=1024*1024)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != expected_size or digest(partial) != expected_sha256:
        raise ValueError('downloaded bytes differ from the pinned identity; unparsed partial retained')
    if destination.exists():
        raise FileExistsError('cached destination appeared during fetch; no replacement')
    partial.rename(destination)
    return {'path': str(destination), 'sha256': expected_sha256, 'size_bytes': expected_size,
            'reused': False, 'wall_s': time.perf_counter()-started}


def parse_language_parquet(path, split):
    """Use the supported reader, and check lossless normalization before training."""
    import pyarrow
    import pyarrow.parquet as parquet
    from oar.sampling_data import normalize_articles, article_manifest
    table = parquet.read_table(path, columns=['text'], use_threads=False)
    rows = table.column('text').to_pylist()
    articles = normalize_articles(rows, split)
    normalized = hashlib.sha256()
    for row in rows:
        normalized.update(row.replace('\r\n', '\n').replace('\r', '\n').encode('utf8'))
    reconstructed = hashlib.sha256()
    for article in articles:
        reconstructed.update(article.text.encode('utf8'))
    if normalized.digest() != reconstructed.digest() or not articles:
        raise ValueError('article parser did not preserve normalized source bytes')
    return articles, {'source_split': split, 'source_sha256': digest(path), 'source_rows': len(rows),
                      'article_count': len(articles), 'normalized_utf8_sha256': normalized.hexdigest(),
                      'reader': 'pyarrow', 'reader_version': pyarrow.__version__,
                      'articles': article_manifest(articles), 'text_or_tokens_distributed': False}


def prepare_language_data(cfg, out, attempt):
    """Shared paid preprocessing, strictly after the cost and local-use gates."""
    from dataclasses import asdict
    import torch
    import pyarrow
    from oar.sampling_data import (encode_articles, split_training_articles, exclude_training_duplicates,
                                  training_chunks, evaluation_windows, choose_monitor, article_manifest)
    require_stage(out, 'cost', source_hashes('pilot'))
    assert_language_asset_permission(cfg)
    if pyarrow.__version__ != cfg['asset']['parquet_reader_version']:
        raise ValueError('Parquet reader differs from the frozen version')
    started = time.perf_counter()
    asset = next(a for a in json.loads((ROOT/'manifests/assets.json').read_text())['datasets'] if a['name'] == cfg['asset']['name'])
    if asset['revision'] != cfg['asset']['revision']:
        raise ValueError('pinned dataset revision changed')
    cache = ROOT/'.cache/dw_sampling/wikitext'/asset['revision']
    receipts, parsed, source_manifests = {}, {}, {}
    # Official test is deliberately absent from this acquisition loop.
    for split in ('train', 'validation'):
        item = next(v for v in asset['files'] if Path(v['path']).name == split+'-00000-of-00001.parquet')
        receipt = fetch_pinned_file(item['source'], cache/Path(item['path']).name,
                                   item['size_bytes'], item['upstream_lfs_sha256'])
        write_json(attempt/f'asset-fetch-{split}.json', receipt)
        receipts[split] = receipt
        parsed[split], source_manifests[split] = parse_language_parquet(receipt['path'], split)
    fitting, development = split_training_articles(parsed['train'])
    validation, exclusions = exclude_training_duplicates(parsed['validation'], parsed['train'])
    if not fitting or not development or not validation:
        raise ValueError('fitting/development/validation article split is empty')
    train_blocks = training_chunks(fitting)
    dev_full, val_full = evaluation_windows(development), evaluation_windows(validation)
    dev_monitor, val_monitor = choose_monitor(development, dev_full), choose_monitor(validation, val_full)
    manifest = {'schema_version': 1, 'dataset_revision': asset['revision'],
                'asset_decision_sha256': cfg['asset']['decision_sha256'], 'raw_files': receipts,
                'data_source_sha256': digest(ROOT/'src/oar/sampling_data.py'),
                'preparation_driver_sha256': digest(ROOT/'analysis/run_dw_sampling.py'),
                'source_manifests': source_manifests, 'fitting_articles': article_manifest(fitting),
                'development_articles': article_manifest(development), 'validation_articles': article_manifest(validation),
                'validation_exclusions': exclusions,
                'training_chunks': [asdict(b) for b in train_blocks],
                'development_full': [asdict(b) for b in dev_full], 'development_monitor': [asdict(b) for b in dev_monitor],
                'validation_full': [asdict(b) for b in val_full], 'validation_monitor': [asdict(b) for b in val_monitor],
                'official_validation_loss_observed': False, 'official_test_body_opened': False,
                'byte_metrics_include_EOS': True, 'copyrighted_text_or_tokens_in_public_manifest': False}
    key = hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != 'raw_files'})).hexdigest()
    private = cache/'processed'/key
    private.mkdir(parents=True, exist_ok=True)
    train_data = {'schema_version': 1, 'fitting_tokens': encode_articles(fitting),
                  'training_blocks': manifest['training_chunks'], 'development_tokens': encode_articles(development),
                  'development_full': manifest['development_full'], 'development_monitor': manifest['development_monitor']}
    val_data = {'schema_version': 1, 'validation_tokens': encode_articles(validation),
                'validation_full': manifest['validation_full'], 'validation_monitor': manifest['validation_monitor']}
    files = {}
    for name, value in (('fitting-development', train_data), ('validation-sealed-loss', val_data)):
        path = private/(name+'.pt')
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            with temporary.open('xb') as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.rename(path)
        else:
            existing = torch.load(path, weights_only=True, map_location='cpu')
            if existing != value:
                raise ValueError('processed cache differs from exact parser output; preserved')
        files[name] = {'path': str(path), 'sha256': digest(path), 'size_bytes': path.stat().st_size}
    manifest['processed_files'] = files
    manifest['shared_preprocessing_wall_s'] = time.perf_counter()-started
    write_json(attempt/'data_manifest.json', manifest)
    notice = private/'NOTICE.txt'
    if not notice.exists():
        with notice.open('x') as stream:
            stream.write('WikiText creators: Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher; underlying Wikipedia contributors.\n'
                         'Pinned Salesforce/wikitext revision '+asset['revision']+'. Original supplied notices remain in third_party/discovery/wikitext/README.md.\n'
                         'Conflicting CC-BY-SA-3.0/GFDL metadata and CC-BY-SA-4.0 prose retained; no authoritative version selected.\n'
                         'Local research only. Changes: line-ending normalization, article parsing, UTF8 byte encoding and target windows. No corpus/checkpoint redistribution.\n')
    return {'manifest_path': str(attempt/'data_manifest.json'), 'manifest_sha256': digest(attempt/'data_manifest.json'),
            'files': files, 'shared_preprocessing_wall_s': manifest['shared_preprocessing_wall_s']}


def learning_lr(elapsed, *, t0, cutoff, peak):
    """One time-budget schedule; restart preserves the original first-block t0."""
    from oar.sampling_lm import time_budget_lr
    return time_budget_lr(elapsed, t0=t0, cutoff=cutoff, peak=peak)


def calibrate_horizon(*, startup_s, monitor_s, checkpoint_s, block_s, update_s,
                      endpoint_s, endpoint_checkpoint_s, aa_jitter, block_updates):
    """Baseline-only 4096-update-equivalent budget; A/A noise is diagnostic."""
    values = (startup_s, monitor_s, checkpoint_s, block_s, update_s, endpoint_s, endpoint_checkpoint_s, aa_jitter)
    if any(not math.isfinite(v) or v < 0 for v in values) or block_s <= 0 or update_s <= 0 or block_updates not in (1, 8):
        raise ValueError('invalid baseline-only horizon calibration')
    observations = []
    for count in (100, 50, 25):
        horizon = startup_s+4096*update_s+(count-1)*monitor_s+((count-1)//5)*checkpoint_s
        interval = horizon/count
        overhead = monitor_s/interval+checkpoint_s/(5*interval)
        observations.append({'monitor_count': count, 'H': horizon, 'Delta': interval, 'overhead_fraction': overhead})
        if overhead <= .02 and block_s <= interval/10:
            if block_s > .01*horizon:
                return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'common block-size1 fallback required before learning',
                        'fallback_required': block_updates == 8, 'observations': observations}
            reserve = 2*(endpoint_s+endpoint_checkpoint_s)
            return {'status': 'pass', 'H': horizon, 'Delta': interval, 'R': reserve, 'T_budget': horizon+reserve,
                    'r_time': interval+block_s, 'B_max': block_s, 'monitor_count': count,
                    'block_updates': block_updates, 'overshoot_limit_s': .01*horizon,
                    'diagnostic_delta': max(aa_jitter, 2*(interval+block_s)/horizon),
                    'required_time_ratio': 1., 'observations': observations}
    return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'baseline cadence overhead or block resolution unsuitable',
            'fallback_required': block_updates == 8, 'observations': observations}


def confirmed_target_time(monitors, q_star, *, cutoff):
    """Second completed consecutive crossing; missing/censored times stay None."""
    if not math.isfinite(q_star) or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('invalid frozen monitor target')
    previous, consecutive = -1., 0
    for row in monitors:
        elapsed, nll = row['completed_s'], row['nll']
        if not math.isfinite(elapsed) or not math.isfinite(nll) or elapsed < previous or elapsed < 0:
            raise ValueError('invalid or unordered monitor observation')
        previous = elapsed
        if elapsed >= cutoff:
            break
        consecutive = consecutive+1 if nll <= q_star else 0
        if consecutive == 2:
            return elapsed
    return None


def select_learning_arms(records, *, cutoff, epsilon_eval, r_time):
    """Freeze the target from R0, then choose fastest quality-eligible arms."""
    if not all(math.isfinite(v) and v > 0 for v in (cutoff, epsilon_eval, r_time)):
        raise ValueError('invalid selection thresholds')
    complete = [r for r in records if r['status'] == 'pass' and math.isfinite(r['endpoint_full']['nll'])]
    dense = [r for r in complete if r['arm']['sampler'] is None]
    if not dense:
        return {'status': 'STOP_QUALITY', 'reason': 'no complete finite native grid arm'}
    minimum = min(r['endpoint_full']['nll'] for r in dense)
    reference = min((r for r in dense if r['endpoint_full']['nll'] <= minimum+epsilon_eval),
                    key=lambda r: (r['arm']['batch'], r['arm']['lr']))
    gain = reference['initial_full']['nll']-reference['endpoint_full']['nll']
    if not math.isfinite(gain) or gain <= 10*epsilon_eval:
        return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'reference full-development improvement unresolved'}
    epsilon_q = min(math.log(1.01), .1*gain)
    calibration = next(((i, m) for i, m in enumerate(reference['monitors']) if m['completed_s'] >= .6*cutoff), None)
    if calibration is None:
        return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'reference has no pre-endpoint calibration monitor'}
    index, observation = calibration
    q_star = observation['nll']
    if (not math.isfinite(q_star) or reference['initial_monitor']['nll']-q_star < 10*epsilon_eval
            or confirmed_target_time(reference['monitors'][index+1:], q_star, cutoff=cutoff) is None):
        return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'target lacks two later pre-cutoff confirmations'}
    eligible = []
    for record in complete:
        if record['endpoint_full']['nll'] > reference['endpoint_full']['nll']+epsilon_q:
            continue
        crossing = confirmed_target_time([record['initial_monitor'], *record['monitors']], q_star, cutoff=cutoff)
        if crossing is not None:
            eligible.append({**record, 'target_time_s': crossing})
    dense_eligible = [r for r in eligible if r['arm']['sampler'] is None]
    candidates = [r for r in eligible if r['arm']['sampler'] in ('importance', 'uniform')]
    def fastest_ties(rows, key):
        if not rows:
            return None
        first = min(r['target_time_s'] for r in rows)
        return min((r for r in rows if r['target_time_s'] <= first+r_time), key=key)
    selected_dense = fastest_ties(dense_eligible, lambda r: (r['arm']['batch'], r['arm']['lr']))
    selected_candidate = fastest_ties(candidates, lambda r: (r['arm']['sampler'] != 'uniform', -r['arm']['rho'], r['arm']['lr']))
    result = {'R0': reference, 'q_star': q_star, 'epsilon_q': epsilon_q,
              'dense': selected_dense, 'candidate': selected_candidate,
              'eligible_dense': dense_eligible, 'eligible_candidates': candidates}
    if selected_dense is None or selected_candidate is None:
        return {'status': 'STOP_QUALITY', 'reason': 'no sampler and native pair meets target plus endpoint guard', **result}
    ratio = selected_candidate['target_time_s']/selected_dense['target_time_s']
    return {'status': 'pass' if ratio < 1 else 'STOP_QUALITY', 'grid_target_time_ratio': ratio, **result}


def plan_confirmation(time_ratios, quality_differences, *, epsilon_q):
    from statistics import mean, stdev
    if (len(time_ratios) != 3 or len(quality_differences) != 3
            or any(not math.isfinite(v) or v <= 0 for v in time_ratios)
            or any(not math.isfinite(v) for v in quality_differences)
            or not math.isfinite(epsilon_q) or epsilon_q <= 0):
        raise ValueError('confirmation planning requires three finite independent pilot pairs')
    logs = [math.log(v) for v in time_ratios]
    m_t, s_t, m_q, s_q = mean(logs), stdev(logs), mean(quality_differences), stdev(quality_differences)
    if m_t >= 0 or epsilon_q-max(0., m_q) <= 0:
        raise ValueError('nonpositive time or quality planning margin')
    n_time = math.ceil(((1.96+.84)*s_t/(-m_t))**2)
    n_quality = math.ceil(((1.96+.84)*s_q/(epsilon_q-max(0., m_q)))**2)
    return {'N': max(8, n_time, n_quality), 'N_time': n_time, 'N_quality': n_quality,
            'mean_log_time_ratio': m_t, 'sd_log_time_ratio': s_t, 'mean_quality_difference': m_q,
            'sd_quality_difference': s_q, 'unit': 'independent training-seed pair',
            'caution': 'three pilot seeds estimate variability; observed pilot effect is an uncertain planning assumption'}


def analyze_confirmation(candidate_times, dense_times, candidate_nll, dense_nll, *, epsilon_q, critical_value=None):
    import numpy as np
    n = len(candidate_times)
    if n < 8 or any(len(v) != n for v in (dense_times, candidate_nll, dense_nll)):
        raise ValueError('confirmation requires at least eight complete paired seeds')
    vectors = [np.asarray(v, dtype=np.float64) for v in (candidate_times, dense_times, candidate_nll, dense_nll)]
    if any(v.shape != (n,) or not np.isfinite(v).all() for v in vectors) or any((v <= 0).any() for v in vectors[:2]):
        raise ValueError('nonfinite or censored confirmation observation; no substitution is allowed')
    if not math.isfinite(epsilon_q) or epsilon_q <= 0:
        raise ValueError('invalid frozen endpoint guard')
    provided = (2.364624, 2.306004, 2.262157, 2.228139, 2.200985, 2.178813, 2.160369, 2.144787, 2.131450)
    if critical_value is None:
        if n > 16:
            raise ValueError('freeze a vetted t critical value for the planned N before protected outcomes')
        critical_value = provided[n-8]
    if not math.isfinite(critical_value) or critical_value <= 0:
        raise ValueError('invalid frozen t critical value')
    z, q = np.log(vectors[0]/vectors[1]), vectors[2]-vectors[3]
    indices = np.random.default_rng(62002).integers(0, n, (10000, n))
    def intervals(values):
        center, error = float(values.mean()), critical_value*float(values.std(ddof=1))/math.sqrt(n)
        bootstrap = np.quantile(values[indices].mean(axis=1), [.025, .975], method='linear').tolist()
        return center, [center-error, center+error], bootstrap
    mean_z, t_z, b_z = intervals(z)
    mean_q, t_q, b_q = intervals(q)
    time_t, time_b = t_z[1] < 0, b_z[1] < 0
    quality_t, quality_b = t_q[1] <= epsilon_q, b_q[1] <= epsilon_q
    status = ('INCONCLUSIVE_PRECISION' if time_t != time_b or quality_t != quality_b else
              'CONFIRMED_SCOPED_BENEFIT' if time_t and quality_t else 'STOP_QUALITY')
    return {'status': status, 'N': n, 'df': n-1, 't_critical_975': critical_value,
            'time': {'ratio': math.exp(mean_z), 't_ci95': [math.exp(v) for v in t_z],
                     'bootstrap_ci95': [math.exp(v) for v in b_z], 'paired_log_ratios': z.tolist()},
            'quality': {'mean_difference': mean_q, 't_ci95': t_q, 'bootstrap_ci95': b_q,
                        'paired_differences': q.tolist(), 'epsilon_q': epsilon_q},
            'analysis_seed': 62002, 'bootstrap_replicates': 10000, 'unit': 'independent training-seed pair'}


def simpler_dominates(simple_times, candidate_times, simple_nll, candidate_nll, *, r_time, epsilon_eval):
    from statistics import mean
    n = len(simple_times)
    if n != 3 or any(len(v) != n for v in (candidate_times, simple_nll, candidate_nll)):
        raise ValueError('simpler control requires the three matched independent pilot seeds')
    if any(not math.isfinite(v) or v <= 0 for v in [*simple_times, *candidate_times]):
        raise ValueError('simpler-control target times must be reached and finite')
    if (any(not math.isfinite(v) for v in [*simple_nll, *candidate_nll, r_time, epsilon_eval])
            or r_time < 0 or epsilon_eval < 0):
        raise ValueError('invalid simpler-control quality or resolution')
    candidate_geomean = math.exp(mean(math.log(v) for v in candidate_times))
    ratio = math.exp(mean(math.log(a/b) for a, b in zip(simple_times, candidate_times)))
    difference = mean(a-b for a, b in zip(simple_nll, candidate_nll))
    return {'dominates': ratio <= 1+r_time/candidate_geomean and difference <= epsilon_eval,
            'R_simple': ratio, 'time_limit': 1+r_time/candidate_geomean, 'mean_nll_difference': difference}


def model_fixture(cfg, data, request, *, variant, sampler=None, rho=.25):
    """Production qualification always uses the actual MPS model and fixed B16."""
    from oar.sampling_lm import ModelConfig, ModelFixture
    learning = cfg['learning']
    if learning['candidate_batch'] != 16:
        raise ValueError('full-model qualification requires the declared batch16')
    architecture = ModelConfig(**{name: learning[name] for name in ('blocks', 'width', 'heads', 'ffn_width', 'context')})
    return ModelFixture(data, base_seed=request['base_seed'], variant=variant, sampler=sampler, rho=rho,
                        batch_size=16, device='mps', model_config=architecture,
                        synchronize_each_update=request['synchronize_each_update'])


def model_data(request):
    from oar.sampling_lm import load_worker_data
    return load_worker_data(request['data_path'], request['data_sha256'], phase='development')


def model_resource_guard(cfg, out):
    import torch
    from oar.hardware import resource_snapshot
    resource = resource_snapshot('mps')
    l = cfg['learning']
    w, f, layers, context = (l[k] for k in ('width', 'ffn_width', 'blocks', 'context'))
    parameters = (259+context)*w+layers*(4*w*w+2*w*f+9*w+f)+2*w
    # Two full native parameter/moment sets, CPU common state, gradients,
    # activation/attention buffers and temporary native optimizer work.
    predicted = parameters*4*24+64*16*context*(w+f)*layers
    ceiling = min(cfg['resources']['memory_fraction']*resource['physical_memory_bytes'],
                  .8*torch.mps.recommended_max_memory(), resource['available_memory_bytes'])
    if predicted > ceiling:
        raise MemoryError('full-model fixtures exceed current conservative available memory')
    return {'snapshot': resource, 'predicted_working_set_bytes': predicted,
            'estimated_checkpoint_bytes': parameters*12+1048576, 'storage': storage_guard(cfg, out)}


def model_dense_calibration(cfg, fixture, initial=1):
    import torch
    from oar.timing import measure_blocks
    raw = []
    def measure(count):
        def block():
            for _ in range(count):
                fixture.step()
        values = measure_blocks(block, torch.mps.synchronize, 3)
        raw.append(values)
        return min(values)
    updates, observations = choose_block_length(measure, initial, minimum_s=cfg['timing']['minimum_block_s'])
    return updates, [{**row, 'raw_s': values} for row, values in zip(observations, raw)]


def model_unsupported_native(exc):
    """Capability failures only; numerical/device failures must stop the run."""
    message = str(exc).lower()
    fatal = any(word in message for word in ('nonfinite', 'non-finite', 'invalid gradient', 'probabil', 'device error',
                                            'command buffer', 'out of memory', 'allocation'))
    return not fatal and (isinstance(exc, NotImplementedError) or
        (isinstance(exc, RuntimeError) and any(word in message for word in
        ('not implemented for', 'not supported on', 'does not support', 'fused=true requires', 'does not have an mps implementation'))))


def qualify_model_native_gradients(actual, reference):
    """Isolate native AdamW arithmetic from independent-backward variation.

    Actual training supplies real current fitting gradients and native clipping.
    Both original optimizers consume identical detached copies for five checks;
    the reference model never executes a separate forward/backward trajectory.
    The temporary entry is removed before any warmup or timing operation.
    """
    absent = object()
    previous_step = actual.opt.__dict__.get('step', absent)
    native_actual_step, native_reference_step = actual.opt.step, reference.opt.step
    def shared_gradient_step(*args, **kwargs):
        reference.opt.zero_grad(set_to_none=True)
        for name, parameter in actual.p.items():
            if parameter.grad is None:
                raise RuntimeError(f'invalid gradient: native model parameter {name} has no gradient')
            reference.p[name].grad = parameter.grad.detach().clone()
        result = native_actual_step(*args, **kwargs)
        native_reference_step()
        return result
    try:
        actual.opt.step = shared_gradient_step
        for _ in range(5):
            actual.step()
            assert_native_equivalent(actual.p, actual.opt, reference.p, reference.opt)
    finally:
        if previous_step is absent:
            del actual.opt.step
        else:
            actual.opt.step = previous_step


def model_native_variants(cfg, data, request):
    import torch
    from oar.sampled_linear import named_seed
    from oar.timing import warmup, measure_blocks
    from statistics import median
    pilot = model_fixture(cfg, data, request, variant='single')
    updates, initial_pilot = model_dense_calibration(cfg, pilot)
    write_json(Path(request['result']).with_suffix('.initial-dense-pilot.json'), initial_pilot)
    del pilot
    order_seed = named_seed(cfg['seed'], 'order', 'full_model', str(request['session']), 'native_variants')
    order = list(cfg['timing']['native_variants'])
    random.Random(order_seed).shuffle(order)
    if request['session'] % 2:
        order.reverse()
    results = {}
    for variant in order:
        try:
            actual = model_fixture(cfg, data, request, variant=variant)
            reference = model_fixture(cfg, data, request, variant='single')
            if actual.model.initial_parameters_sha256 != reference.model.initial_parameters_sha256:
                raise ValueError('native model references do not share exact fresh initialization')
            # Fresh identical init/data streams; isolate optimizer arithmetic
            # with the actual model's current, natively clipped gradients.
            qualify_model_native_gradients(actual, reference)
            del reference
            def block():
                for _ in range(updates):
                    actual.step()
            warming = warmup(block, torch.mps.synchronize, **cfg['timing']['warmup'])
            raw = measure_blocks(block, torch.mps.synchronize, 3)
            results[variant] = {'status': 'pass', 'warmup': warming, 'pilot_s': raw,
                                'median_s': median(raw), 'updates_per_block': updates, 'qualification_updates': 5}
            del actual
        except (RuntimeError, NotImplementedError) as exc:
            if not model_unsupported_native(exc):
                raise
            results[variant] = {'status': 'unsupported', 'reason': str(exc), 'exception': type(exc).__name__}
        results[variant]['qualification_mode'] = 'identical current full-model gradients after native clipping'
        write_json(Path(request['result']).with_suffix('.native-'+variant+'.json'), results[variant])
    selected = select_native_variant(results)
    return {'status': 'pass', 'selected_variant': selected, 'variants': results, 'initial_dense_pilot': initial_pilot,
            'order_seed': order_seed, 'pilot_order': order}


def pooled_model_native_variant(sessions):
    if len(sessions) != 2:
        raise ValueError('full-model native selection requires two independent sessions')
    common = set.intersection(*(set(s['variants']) for s in sessions))
    pooled = {}
    for name in sessions[0]['variants']:
        if name not in common or not all(s['variants'][name]['status'] == 'pass' for s in sessions):
            continue
        raw_sessions, per_update = [], []
        for index, session in enumerate(sessions):
            row = session['variants'][name]
            count = row['updates_per_block']
            if type(count) is not int or count <= 0:
                raise ValueError('native pooling requires each session own positive update count')
            raw = row['pilot_s']
            if not raw or any(not math.isfinite(value) or value <= 0 for value in raw):
                raise ValueError('native pooling requires finite positive pilot observations')
            per_update.extend(value/count for value in raw)
            raw_sessions.append({'session': session.get('session', index), 'updates_per_block': count, 'pilot_s': raw})
        pooled[name] = {'status': 'pass', 'pilot_s': per_update,
                        'unit': 'seconds per native update', 'raw_sessions': raw_sessions}
    return select_native_variant(pooled), pooled


def model_common_history(cfg, fixture):
    import torch
    for _ in range(cfg['timing']['history_updates']):
        fixture.step()
    torch.mps.synchronize()
    common = fixture.snapshot()
    steps = [state['step'].detach().cpu().item() for state in fixture.opt.state.values()]
    nonzero = any(bool(state['exp_avg'].detach().cpu().ne(0).any()) for state in fixture.opt.state.values())
    if not steps or not all(value == cfg['timing']['history_updates'] for value in steps) or not nonzero:
        raise RuntimeError('model common history requires native updates and nonzero optimizer moments')
    metadata = {'history_updates': fixture.updates, 'native_optimizer_steps': steps,
                'nonzero_native_moments': nonzero, 'streams': state_streams(fixture),
                'initial_parameters_sha256': fixture.model.initial_parameters_sha256,
                'data_cursor': common['stream']['cursor'], 'data_epoch': common['stream']['epoch']}
    return common, metadata


def model_native_session(cfg, request):
    from oar.hardware import resource_snapshot
    data = model_data(request)
    resource = model_resource_guard(cfg, request['out'])
    if request['phase'] == 'variants':
        result = model_native_variants(cfg, data, request)
    else:
        variant = request['fixed_variant']
        dense = model_fixture(cfg, data, request, variant=variant)
        updates, calibration = model_dense_calibration(cfg, dense)
        write_json(Path(request['result']).with_suffix('.selected-dense-pilot.json'), calibration)
        del dense
        dense = model_fixture(cfg, data, request, variant=variant)
        common, history = model_common_history(cfg, dense)
        candidate = model_fixture(cfg, data, request, variant=variant)
        result = {'status': 'pass', 'selected_variant': variant, 'updates_per_block': updates,
                  'selected_dense_pilot': calibration, 'history': history,
                  'AA': timed_pairs(cfg, dense, candidate, common, updates, request['order_seed'],
                                    Path(request['result']).with_suffix('.AA.jsonl'))}
    result.update(kind='model_native', phase=request['phase'], session=request['session'],
                  base_seed=request['base_seed'], synchronize_each_update=request['synchronize_each_update'],
                  resources=resource, resources_after=resource_snapshot('mps'))
    write_json(request['result'], result)
    return result


def model_cost_session(cfg, request):
    from oar.sampled_linear import named_seed
    from oar.hardware import resource_snapshot
    data = model_data(request)
    resource = model_resource_guard(cfg, request['out'])
    frozen = json.loads(Path(request['frozen_delta_path']).read_text())
    if digest(request['frozen_delta_path']) != request['frozen_delta_sha256'] or frozen['fixed_variant'] != request['fixed_variant']:
        raise ValueError('model frozen native/noise identity mismatch before candidate timing')
    variant = request['fixed_variant']
    dense = model_fixture(cfg, data, request, variant=variant)
    updates, calibration = model_dense_calibration(cfg, dense, request['updates_per_block'])
    write_json(Path(request['result']).with_suffix('.selected-dense-pilot.json'), calibration)
    del dense
    dense = model_fixture(cfg, data, request, variant=variant)
    common, history = model_common_history(cfg, dense)
    arms = list(request['arms'])
    random.Random(request['order_seed']).shuffle(arms)
    comparisons = {}
    for arm in arms:
        candidate = model_fixture(cfg, data, request, variant=variant, sampler=arm['sampler'], rho=arm['rho'])
        name = arm['arm']
        comparison = timed_pairs(cfg, dense, candidate, common, updates,
            named_seed(cfg['seed'], 'order', 'full_model', 'cost', str(request['session']), name,
                       str(request['synchronize_each_update'])), Path(request['result']).with_suffix('.'+name+'.jsonl'))
        comparison['sampling_streams_final'] = state_streams(candidate)
        comparison['resources_candidate_live'] = resource_snapshot('mps')
        comparison['memory_context'] = 'dense/common/candidate live with allocator caches; overlapping fields are not summed or claimed as exact peaks'
        comparisons[name] = comparison
        # Persist each completed arm even if another arm later fails.
        write_json(Path(request['result']).with_suffix('.'+name+'.json'), comparison)
        del candidate
    result = {'status': 'pass', 'kind': 'model_cost', 'session': request['session'], 'selected_variant': variant,
              'updates_per_block': updates, 'initial_updates_per_block': request['updates_per_block'],
              'selected_dense_pilot': calibration, 'history': history, 'comparisons': comparisons,
              'arm_order': [a['arm'] for a in arms], 'order_seed': request['order_seed'], 'resources': resource,
              'synchronize_each_update': request['synchronize_each_update'], 'delta': frozen['delta'],
              'resources_after': resource_snapshot('mps')}
    write_json(request['result'], result)
    return result


def model_cost_eligible(sessions, arms):
    if len(sessions) != 2:
        raise ValueError('full-model cost requires two independent sessions')
    return [arm for arm in arms if all(s['comparisons'][arm['arm']]['statistics']['ci95'][1] < 1. for s in sessions)]


def model_calibration_session(cfg, request):
    """Cold baseline startup, exact initial evaluations and paid trained costs."""
    import torch
    from oar.hardware import resource_snapshot
    from oar.sampling_lm import ChargedClock, evaluate, save_checkpoint
    if request['clock_implementation'] != time.get_clock_info('perf_counter').implementation:
        raise ValueError('parent and worker monotonic clock implementations differ')
    clock = ChargedClock(request['launch_perf_s'])
    data = model_data(request)
    resource = model_resource_guard(cfg, request['out'])
    fixture = model_fixture(cfg, data, request, variant=request['fixed_variant'])
    sync = torch.mps.synchronize
    events_path = Path(request['result']).with_suffix('.events.jsonl')
    with events_path.open('x'):
        pass
    def event(kind, **row):
        with events_path.open('a') as stream:
            stream.write(json.dumps({'kind': kind, 'elapsed_s': clock.elapsed(), **row}, allow_nan=False)+'\n')
            stream.flush()
    def evaluate_split(windows, kind):
        sync()
        started = time.perf_counter()
        value = evaluate(fixture.model, data['evaluation_tokens'], windows, batch_size=16)
        sync()
        seconds = time.perf_counter()-started
        event(kind, cost_s=seconds, nll=value['nll'] if kind.startswith('initial') else None,
              valid_targets=value['valid_targets'])
        return value, seconds
    initial_full, initial_full_s = evaluate_split(data['evaluation_full'], 'initial_full')
    initial_monitor, initial_monitor_s = evaluate_split(data['evaluation_monitor'], 'initial_monitor')
    t0 = clock.elapsed()
    receipts = []
    def checkpoint(kind):
        sync()
        started = time.perf_counter()
        path = Path(request['result']).with_suffix('.'+kind+'.pt')
        disk = storage_guard(cfg, request['out'])
        if resource.get('estimated_checkpoint_bytes', 0) > disk['free_disk_bytes']-cfg['resources']['minimum_free_disk_bytes']:
            raise MemoryError('trained checkpoint would consume the actual free-disk reserve')
        payload = {'schema_version': 1, 'bindings': {'data_sha256': request['data_sha256'],
                    'base_seed': request['base_seed'], 'native_variant': request['fixed_variant']},
                   'training_state': fixture.snapshot(), 'clock': clock.state_dict(), 't0_s': t0,
                   'scheduler_rule': {'qualification_only': True, 'block_updates': request['block_updates']}}
        receipt = {'path': str(path), **save_checkpoint(path, payload), 'kind': kind}
        storage_guard(cfg, request['out'])
        sync()
        seconds = time.perf_counter()-started
        receipts.append(receipt)
        event('checkpoint', receipt=receipt, inclusive_cost_s=seconds)
        return seconds
    initial_checkpoint_s = checkpoint('initial')
    startup = clock.elapsed()
    _, history = model_common_history(cfg, fixture)
    _, monitor_s = evaluate_split(data['evaluation_monitor'], 'trained_monitor_cost')
    checkpoint_s = checkpoint('trained')
    _, endpoint_s = evaluate_split(data['evaluation_full'], 'trained_endpoint_cost')
    # End-to-end controller block includes one fence, scalar transfer and JSON
    # logging. The optional fallback fixture fences after every native update.
    sync()
    started = time.perf_counter()
    event('update_block_start', updates=request['block_updates'], lr=fixture.lr)
    raw = [fixture.step() for _ in range(request['block_updates'])]
    sync()
    scalars = torch.stack([torch.stack((r['loss'], r['norm'])) for r in raw]).detach().cpu().tolist()
    event('update_block', updates=request['block_updates'], stats=scalars,
          valid_targets=sum(r['valid_targets'] for r in raw), physical_rows=sum(r['physical_rows'] for r in raw))
    block_s = time.perf_counter()-started
    if not math.isfinite(block_s) or block_s <= 0:
        raise ValueError('invalid complete model controller block duration')
    result = {'status': 'pass', 'kind': 'model_calibration', 'session': request['session'],
              'initial_parameters_sha256': fixture.model.initial_parameters_sha256,
              'initial_full': initial_full, 'initial_monitor': initial_monitor,
              'initial_full_s': initial_full_s, 'initial_monitor_s': initial_monitor_s,
              'initial_checkpoint_s': initial_checkpoint_s, 'startup_s': startup, 't0_s': t0,
              'monitor_s': monitor_s, 'checkpoint_s': checkpoint_s, 'endpoint_s': endpoint_s,
              'endpoint_checkpoint_s': checkpoint_s, 'update_block_s': block_s,
              'block_updates': request['block_updates'], 'base_seed': request['base_seed'],
              'selected_variant': request['fixed_variant'], 'history': history, 'resources': resource,
              'checkpoint_receipts': receipts, 'clock': clock.state_dict(),
              'synchronize_each_update': request['synchronize_each_update'], 'resources_after': resource_snapshot('mps')}
    write_json(request['result'], result)
    return result


def aggregate_model_calibration(cfg, rows, aa_jitter, *, baseline_blocks=()):
    if len(rows) != 3 or len({r['initial_parameters_sha256'] for r in rows}) != 1:
        raise ValueError('cold calibration requires three identical initial parameter identities')
    if len({r['block_updates'] for r in rows}) != 1:
        raise ValueError('cold calibration block boundaries differ')
    values = [[r[k]['nll'] for r in rows] for k in ('initial_full', 'initial_monitor')]
    if any(not math.isfinite(v) for group in values for v in group):
        raise ValueError('nonfinite initial evaluation calibration')
    epsilon = max(cfg['learning']['epsilon_eval_floor'], 2*max(max(group)-min(group) for group in values))
    if epsilon > cfg['learning']['epsilon_eval_ceiling']:
        return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': 'initial evaluation repeatability exceeds declared ceiling',
                'epsilon_eval': epsilon, 'fallback_required': False, 'cold_workers': rows}
    block_updates = rows[0]['block_updates']
    costs = {name: max(r[name] for r in rows) for name in
             ('startup_s', 'monitor_s', 'checkpoint_s', 'endpoint_s', 'endpoint_checkpoint_s')}
    blocks = [*baseline_blocks, *({'source': 'cold controller block', 'seconds': row['update_block_s'],
                                  'updates': block_updates} for row in rows)]
    for block in blocks:
        if (type(block['updates']) is not int or block['updates'] <= 0
                or not math.isfinite(block['seconds']) or block['seconds'] <= 0):
            raise ValueError('invalid qualified native block for horizon calibration')
    block_s = max(block['seconds'] for block in blocks)
    update_s = max(block['seconds']/block['updates'] for block in blocks)
    horizon = calibrate_horizon(**costs, block_s=block_s, update_s=update_s,
                                aa_jitter=aa_jitter, block_updates=block_updates)
    return {'status': horizon['status'], 'epsilon_eval': epsilon, 'horizon': horizon,
            'fallback_required': horizon.get('fallback_required', False), 'cold_workers': rows,
            'baseline_update_s_bound': update_s, 'baseline_blocks': blocks,
            'reason': horizon.get('reason', 'baseline-only full-model calibration complete')}


def model_baseline_blocks(variants, native, costs, fixed_variant):
    """Every qualified fixed-native measurement, without deleting slow blocks."""
    blocks = []
    for index, session in enumerate(variants):
        row = session['variants'][fixed_variant]
        blocks.extend({'source': f'variant pilot session{index}', 'seconds': seconds,
                       'updates': row['updates_per_block']} for seconds in row['pilot_s'])
    for index, session in enumerate(native):
        for pair in session['AA']['pairs']:
            blocks.extend({'source': f'AA session{index} {arm}', 'seconds': seconds,
                           'updates': session['updates_per_block']} for arm, seconds in pair['seconds'].items())
    for index, session in enumerate(costs):
        for arm, comparison in session['comparisons'].items():
            blocks.extend({'source': f'candidate-comparison native session{index} {arm}',
                           'seconds': pair['seconds']['dense'], 'updates': comparison['updates_per_block']}
                          for pair in comparison['pairs'])
    return blocks


def model_qualification_forecast(cfg, baseline, arms, manifest):
    """Advisory inclusive forecast; no prospective runtime or update ceiling."""
    timing, learning = cfg['timing'], cfg['learning']
    relevant = [row for session in baseline['sessions'] for row in session['shapes'] if row['shape']['id'] == 'middle4096']
    native_step = max(p['seconds']/p['updates'] for row in relevant for p in row['selected_dense_pilot'])
    attention_allowance = 2.
    model_step = native_step*learning['blocks']*attention_allowance
    updates = max(1, math.ceil(timing['minimum_block_s']/model_step))
    block = max(timing['minimum_block_s'], updates*model_step)
    warm, pairs = timing['warmup']['maximum'], timing['pairs']
    variants = len(timing['native_variants'])
    variant_blocks = 2*(3+variants*(warm+3)+3)
    aa_blocks = 2*(3+2*warm+2*pairs)
    candidate_blocks = 2*(3+len(arms)*(2*warm+2*pairs))
    resets = max((r for row in relevant for r in row['comparisons']['AA']['reset_s']), default=0.)
    reset_allowance = 2*(1+len(arms))*(2+2*pairs)*resets*learning['blocks']*attention_allowance*3
    full_batches = math.ceil(len(manifest['development_full'])/16)
    monitor_batches = math.ceil(len(manifest['development_monitor'])/16)
    startup = 9*cfg['resources']['qualification_forecast_s']
    cold = 3*((2*full_batches+2*monitor_batches+16)*model_step+2*30.)
    # All three initial/trained full native checkpoints remain private; reserve
    # space for both initially planned round and the sole possible fallback.
    w, f, layers, context = (learning[k] for k in ('width', 'ffn_width', 'blocks', 'context'))
    parameter_bytes = 4*((259+context)*w+layers*(4*w*w+2*w*f+9*w+f)+2*w)
    checkpoints = 12*(3*parameter_bytes+1048576)
    equivalence_updates = 2*variants*10
    paired_history_updates = 4*timing['history_updates']
    per_round = 3*block*(variant_blocks+aa_blocks+candidate_blocks)+reset_allowance+startup+cold
    per_round += (equivalence_updates+paired_history_updates)*model_step
    return {'forecast_s': 2*per_round, 'deadline_s': None, 'rounds_including_single_possible_fallback': 2,
            'native_variant_workers_per_round': 2, 'native_AA_workers_per_round': 2, 'candidate_workers_per_round': 2,
            'cold_workers_per_round': 3, 'native_operator_step_s': native_step,
            'model_step_s_allowance': model_step, 'attention_allowance': attention_allowance,
            'updates_per_block_estimate': updates, 'native_variant_blocks': variant_blocks,
            'AA_blocks': aa_blocks, 'candidate_blocks': candidate_blocks, 'reset_allowance_s': reset_allowance,
            'native_equivalence_updates': equivalence_updates, 'paired_history_updates': paired_history_updates,
            'cold_full_batches': full_batches, 'cold_monitor_batches': monitor_batches,
            'expected_private_checkpoint_bytes': checkpoints, 'startup_allowance_s': startup,
            'policy': 'advisory only; all declared work and the one common-block fallback charged'}


def language_budget_forecast(cfg, calibration, arms):
    learning = cfg['learning']
    grid = len(learning['dense_batches'])*len(learning['peak_lrs'])+len(arms)*len(learning['peak_lrs'])
    planning_n = learning['confirmation_minimum']
    stages = {'eligible_grid': grid, 'independent_development_maximum': learning['independent_development_maximum'],
              'precision_screens_maximum': learning['precision_screens_maximum'],
              'confirmation_runs_at_planning_N': 2*planning_n,
              'secondary_batch16_maximum': learning['secondary_batch16_maximum'],
              'independent_repeat_runs': 2*learning['independent_repeat_pairs']}
    total = sum(stages.values())
    horizon = calibration['horizon']
    return {'stages': stages, 'full_budget_equivalents': total, 'H': horizon['H'], 'T_budget': horizon['T_budget'],
            'forecast_s': total*horizon['T_budget'], 'deadline_s': None,
            'confirmation_planning_N': planning_n, 'confirmation_N_selected': False,
            'confirmation_configured_maximum': learning['confirmation_maximum'],
            'budget_equivalents_formula': f'{total-2*planning_n}+2*N',
            'forecast_s_without_confirmation': (total-2*planning_n)*horizon['T_budget'],
            'additional_forecast_s_per_confirmation_seed': 2*horizon['T_budget'],
            'note': 'complete work formula; planning N is the declared minimum, not a cap; regenerate with precision-selected N before confirmation; duplicate development seeds/runs deduplicate'}


def validate_model_worker_request(request, cfg, attempt, identity, config, direct):
    """Pilot-only exact prepared development data; no generic training worker."""
    if identity.get('config_sha256') != digest(config) or cfg['learning'].get('action') != 'qualification':
        raise ValueError('model worker configuration differs from the frozen qualification action')
    assert_language_asset_permission(cfg)
    if 'device' in request or Path(request['result']).resolve().parent != attempt or Path(request['result']).exists():
        raise ValueError('model worker result path/device is outside its stage boundary')
    flag = request.get('synchronize_each_update')
    if type(flag) is not bool or request.get('block_updates') != (1 if flag else 8):
        raise ValueError('model worker common synchronization boundary mismatch')
    if (type(request.get('session')) is not int or type(request.get('base_seed')) is not int
            or type(request.get('order_seed')) is not int or not 0 <= request['session'] < (3 if request['kind'] == 'model_calibration' else 2)):
        raise ValueError('invalid model worker declared session/order stream')
    seed = cfg['learning']['tuning_seed'] if request['kind'] == 'model_calibration' else cfg['seed']+request['session']
    if request['base_seed'] != seed or not math.isfinite(request.get('launch_perf_s', -1)) or request.get('launch_perf_s', -1) < 0:
        raise ValueError('model worker seed/cold launch timestamp mismatch')
    manifest = Path(request['data_manifest_path']).resolve()
    if manifest.parent != attempt or digest(manifest) != request['data_manifest_sha256']:
        raise ValueError('model worker prepared data manifest differs')
    data = json.loads(manifest.read_text())['processed_files']['fitting-development']
    if (Path(request['data_path']).resolve() != Path(data['path']).resolve()
            or request['data_sha256'] != data['sha256'] or digest(request['data_path']) != data['sha256']):
        raise ValueError('model worker development data differs from exact prepared bytes')
    if request['kind'] == 'model_native' and request.get('phase') not in ('variants', 'AA'):
        raise ValueError('unknown native model qualification phase')
    if request['kind'] != 'model_native' or request['phase'] == 'AA':
        if request.get('fixed_variant') not in cfg['timing']['native_variants']:
            raise ValueError('unknown fixed full-model native variant')
        pool = Path(request['native_pool_path']).resolve()
        if (pool.parent != attempt or digest(pool) != request['native_pool_sha256']
                or json.loads(pool.read_text())['fixed_variant'] != request['fixed_variant']):
            raise ValueError('fixed native model variant differs from pooled qualification')
    if request['kind'] == 'model_cost':
        expected = [{'arm': f'{sampler}-{rho}', 'sampler': sampler, 'rho': rho}
                    for sampler in cfg['samplers'] for rho in cfg['rhos']
                    if {'arm': f'{sampler}-{rho}', 'shape': 'middle4096'} in direct['eligible']]
        frozen = Path(request['frozen_delta_path']).resolve()
        if (request['arms'] != expected or not expected or type(request['updates_per_block']) is not int
                or request['updates_per_block'] <= 0 or frozen.parent != attempt
                or digest(frozen) != request['frozen_delta_sha256']):
            raise ValueError('model candidate arms/frozen noise differ from prospectively qualified work')


def launch_model_worker(cfg, config, out, attempt, kind, tag, payload):
    from oar.sampled_linear import named_seed
    attempt = Path(attempt)
    request = {'kind': kind, 'config': str(config), 'out': str(out), 'attempt': str(attempt),
               'result': str(attempt/(tag+'.json')), 'order_seed': named_seed(cfg['seed'], 'order', tag), **payload,
               'clock_implementation': time.get_clock_info('perf_counter').implementation,
               'launch_perf_s': time.perf_counter()}
    path = attempt/(tag+'-request.json')
    write_json(path, request)
    code, _, _ = command(attempt, worker_args('pilot', config, out, path), env=environment(cfg), timeout=None)
    if code or not Path(request['result']).is_file():
        return {'status': 'INCOMPLETE_RESOURCE' if code == 124 else 'INCONCLUSIVE_MEASUREMENT',
                'reason': 'full-model worker did not complete; raw prefix and command failure retained',
                'request_path': str(path), 'exit_code': code}
    return json.loads(Path(request['result']).read_text())


def pilot_qualification(cfg, config, out, attempt):
    """Qualification only: never promote Stage E before tuning exists."""
    if cfg['learning'].get('action') != 'qualification':
        return {'status': 'not_run', 'reason': 'full learning dispatcher requires a future prospective freeze'}
    bindings = source_hashes('pilot')
    prerequisites = {stage: require_stage(out, stage, bindings) for stage in ('prepare', 'feasibility', 'correctness', 'cost')}
    baseline = prerequisites['feasibility']
    direct = prerequisites['cost']
    arms = []
    for sampler in cfg['samplers']:
        for rho in cfg['rhos']:
            name = f'{sampler}-{rho}'
            if {'arm': name, 'shape': 'middle4096'} in direct['eligible']:
                arms.append({'arm': name, 'sampler': sampler, 'rho': rho})
    if not arms:
        return {'status': 'STOP_COST', 'reason': 'no sampled arm survived direct D cost at middle4096'}
    assert_language_asset_permission(cfg)
    prepared = prepare_language_data(cfg, out, attempt)
    manifest = json.loads(Path(prepared['manifest_path']).read_text())
    forecast = model_qualification_forecast(cfg, baseline, arms, manifest)
    write_json(Path(attempt)/'model_qualification_forecast.json', forecast)
    storage = storage_guard(cfg, out)
    if forecast['expected_private_checkpoint_bytes'] > storage['free_disk_bytes']-cfg['resources']['minimum_free_disk_bytes']:
        raise MemoryError('full-model checkpoint forecast would consume the actual free-disk reserve')
    data = prepared['files']['fitting-development']
    rounds = []
    for fallback in (False, True):
        label = 'model-block1' if fallback else 'model-block8'
        common = {'data_path': data['path'], 'data_sha256': data['sha256'],
                  'data_manifest_path': prepared['manifest_path'], 'data_manifest_sha256': prepared['manifest_sha256'],
                  'synchronize_each_update': fallback, 'block_updates': 1 if fallback else 8}
        def run(kind, suffix, session, **extra):
            value = launch_model_worker(cfg, config, out, attempt, kind, label+'-'+suffix+'-'+str(session),
                {**common, 'session': session, 'base_seed': cfg['seed']+session, **extra})
            if value['status'] != 'pass':
                raise MeasurementInvalid('model_worker_failed', json.dumps(value))
            return value
        variants = [run('model_native', 'variants', session, phase='variants') for session in range(2)]
        variant, pooled = pooled_model_native_variant(variants)
        pool_path = Path(attempt)/(label+'-pooled-native.json')
        write_json(pool_path, {'fixed_variant': variant, 'pooled': pooled, 'sessions': variants})
        common.update(native_pool_path=str(pool_path), native_pool_sha256=digest(pool_path))
        native = [run('model_native', 'AA', session, phase='AA', fixed_variant=variant) for session in range(2)]
        aa_delta = freeze_delta([abs(pair['seconds']['dense']/pair['seconds']['candidate']-1)
                                for row in native for pair in row['AA']['pairs']], floor=0.)
        frozen_path = Path(attempt)/(label+'-frozen_delta.json')
        frozen = {'delta': max(direct['delta'], aa_delta), 'previous_D_delta': direct['delta'],
                  'model_AA_delta': aa_delta, 'fixed_variant': variant,
                  'source': 'pooled full-model native A/A; frozen before all model candidate timings',
                  'synchronize_each_update': fallback}
        write_json(frozen_path, frozen)
        costs = [run('model_cost', 'cost', session, fixed_variant=variant, arms=arms,
                     updates_per_block=native[session]['updates_per_block'],
                     frozen_delta_path=str(frozen_path), frozen_delta_sha256=digest(frozen_path)) for session in range(2)]
        eligible = model_cost_eligible(costs, arms)
        round_result = {'synchronize_each_update': fallback, 'variant_sessions': variants, 'native_sessions': native,
                        'fixed_variant': variant, 'frozen_delta': frozen, 'cost_sessions': costs, 'eligible': eligible}
        rounds.append(round_result)
        if not eligible:
            write_json(Path(attempt)/(label+'-qualification.json'), {**round_result, 'status': 'STOP_COST'})
            return {'status': 'STOP_COST', 'reason': 'no D-surviving arm beat full native model upper95%<1 in both sessions',
                    'qualification': {'status': 'STOP_COST', 'rounds': rounds}, 'data': prepared}
        cold = [run('model_calibration', 'cold', session, fixed_variant=variant,
                    base_seed=cfg['learning']['tuning_seed']) for session in range(3)]
        calibrated = aggregate_model_calibration(cfg, cold, frozen['delta'],
            baseline_blocks=model_baseline_blocks(variants, native, costs, variant))
        round_result['calibration'] = calibrated
        write_json(Path(attempt)/(label+'-qualification.json'), {**round_result, 'status': calibrated['status']})
        if calibrated['status'] != 'pass':
            if calibrated.get('fallback_required') and not fallback:
                continue
            return {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': calibrated['reason'],
                    'qualification': {'status': calibrated['status'], 'rounds': rounds}, 'data': prepared}
        whole = language_budget_forecast(cfg, calibrated, eligible)
        write_json(Path(attempt)/'language_budget_forecast.json', whole)
        qualification = {'status': 'pass', 'native_variant': variant, 'eligible': eligible, 'calibration': calibrated,
                         'forecast': whole, 'qualification_forecast': forecast, 'rounds': rounds}
        write_json(Path(attempt)/'model_qualification.json', qualification)
        return {'status': 'not_run', 'reason': 'full-model qualification passed; Stage E tuning/selection awaits full learning dispatcher and future freeze',
                'qualification': qualification, 'data': prepared, 'learning_success': False}
    raise RuntimeError('unreachable model fallback state')


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
        elif request['kind'] == 'model_native':
            model_native_session(cfg, request)
        elif request['kind'] == 'model_cost':
            model_cost_session(cfg, request)
        elif request['kind'] == 'model_calibration':
            model_calibration_session(cfg, request)
        else:
            cost_session(cfg, request)
        return 0
    attempt = allocate_attempt(args.out, args.stage)
    start = time.perf_counter()
    identity = {'reviewed_head': REVIEWED, 'actual_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                'amendment_sha256': digest(ROOT/'decisions/dw_sampling_amendment.json'), 'config_path': str(args.config),
                'config_sha256': digest(args.config),
                'measurement_hashes': source_hashes(args.stage), 'bound_files': amendment['bound_files'], 'utc': utc()}
    write_json(attempt/'source_manifest.json', identity)
    result = {}
    ownership = ExitStack()
    try:
        if args.stage in ('prepare', 'feasibility', 'correctness', 'cost', 'pilot', 'confirm') and not args.dry_run:
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
        elif args.stage == 'pilot':
            result = pilot_qualification(cfg, args.config, args.out, attempt)
        elif args.stage == 'confirm':
            require_stage(args.out, 'pilot', source_hashes('confirm'))
            result = {'status': 'not_run', 'reason': 'confirmation dispatcher awaits qualified development/selection and a future prospective freeze'}
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
    try:
        validate_source_stability(identity, args.stage, args.config)
    except MeasurementInvalid as exc:
        (attempt/'source_stability_failure.txt').write_text(traceback.format_exc())
        result = {'status': 'INCONCLUSIVE_MEASUREMENT', 'reason': str(exc),
                  'exception': type(exc).__name__, 'invalidated_result': result}
    record = seal_record(attempt, {'stage': args.stage, **result, 'measurement_hashes': identity['measurement_hashes'],
                                  'invocation_wall_s': time.perf_counter()-start, 'source_manifest_sha256': digest(attempt/'source_manifest.json')})
    print(json.dumps({'stage': args.stage, 'status': record['status'], 'attempt': str(attempt), 'reason': record['reason']}))
    return 0 if record['status'] in ('pass', 'not_run', 'BLOCKED_ASSET', 'BLOCKED_HARDWARE', 'STOP_COST', 'INCONCLUSIVE_MEASUREMENT', 'INCOMPLETE_RESOURCE') else 1


if __name__ == '__main__':
    raise SystemExit(main())
