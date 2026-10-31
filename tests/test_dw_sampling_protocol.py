# SPDX-License-Identifier: AGPL-3.0-only
"""Protocol tests exercise validation and persisted evidence, not source text."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def driver():
    spec = importlib.util.spec_from_file_location('dw_sampling_driver', ROOT / 'analysis/run_dw_sampling.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configuration_rejects_changed_protected_protocol_and_unknown_fields(tmp_path):
    d = driver()
    original = ROOT / 'configs/dw_sampling.yaml'
    assert d.load_config(original)['device'] == 'mps'
    for mutation in ('extra', 'rho', 'shape', 'timing', 'asset', 'schema'):
        cfg = yaml.safe_load(original.read_text())
        if mutation == 'extra':
            cfg['unapproved_sampler'] = True
        elif mutation == 'rho':
            cfg['rhos'] = [0.01]
        elif mutation == 'shape':
            cfg['shapes'][1]['di'] = 1024
        elif mutation == 'timing':
            cfg['timing']['pairs'] = 3
        elif mutation == 'asset':
            cfg['asset']['body_downloads'] = True
        else:
            cfg['schema_version'] = 1
        path = tmp_path / (mutation + '.yaml')
        path.write_text(yaml.safe_dump(cfg))
        with pytest.raises(ValueError):
            d.load_config(path)
    duplicate = tmp_path / 'duplicate.yaml'
    duplicate.write_text(original.read_text() + '\ndevice: cpu\n')
    with pytest.raises(ValueError):
        d.load_config(duplicate)


def test_attempts_append_and_sealed_records_detect_tampering(tmp_path):
    d = driver()
    one = d.allocate_attempt(tmp_path, 'prepare')
    two = d.allocate_attempt(tmp_path, 'prepare')
    assert one != two and one.is_dir() and two.is_dir()
    artifact = one / 'raw.json'
    artifact.write_text('{"time":1}\n')
    d.seal_record(one, {'stage': 'prepare', 'status': 'pass', 'measurement_hashes': {'x': 'a'}})
    assert d.read_record(one / 'record.json')['status'] == 'pass'
    with pytest.raises(FileExistsError):
        d.seal_record(one, {'status': 'pass'})
    artifact.write_text('{"time":0.1}\n')
    with pytest.raises(ValueError, match='hash'):
        d.read_record(one / 'record.json')


def test_missing_failed_and_source_changed_prerequisites_cannot_launch(tmp_path):
    d = driver()
    with pytest.raises(ValueError, match='prerequisite'):
        d.require_stage(tmp_path, 'feasibility', {'source': 'v1'})
    p = d.allocate_attempt(tmp_path, 'feasibility')
    d.seal_record(p, {'stage': 'feasibility', 'status': 'STOP_COST', 'measurement_hashes': {'source': 'v1'}})
    with pytest.raises(ValueError, match='prerequisite'):
        d.require_stage(tmp_path, 'feasibility', {'source': 'v1'})
    p2 = d.allocate_attempt(tmp_path, 'feasibility')
    d.seal_record(p2, {'stage': 'feasibility', 'status': 'pass', 'measurement_hashes': {'source': 'v1'}})
    with pytest.raises(ValueError, match='source'):
        d.require_stage(tmp_path, 'feasibility', {'source': 'v2'})
    assert d.require_stage(tmp_path, 'feasibility', {'source': 'v1'})['status'] == 'pass'


def test_geometric_paired_estimator_and_noise_threshold():
    d = driver()
    # Ratios .5,2 have geometric mean1, arithmetic mean1.25.
    result = d.paired_ratio([1., 1.], [.5, 2.])
    assert result['ratio'] == pytest.approx(1.)
    assert result['ci95'] == pytest.approx([.5, 2.])
    assert d.paired_ratio([1., 2., 4.], [.8, 1.6, 3.2])['ci95'] == pytest.approx([.8, .8])
    aa = [0., .01, .02, .03]
    assert d.freeze_delta(aa) == pytest.approx(.057)
    assert d.freeze_delta([.001, .002]) == pytest.approx(.0039)
    assert d.freeze_delta([.1, .2]) == pytest.approx(.39)
    assert d.freeze_delta([.001], floor=.01) == .01
    for observations, floor in (([], 0.), ([float('nan')], 0.), ([-.1], 0.), ([.1], -1.), ([.1], float('inf'))):
        with pytest.raises(ValueError):
            d.freeze_delta(observations, floor=floor)
    for x, y in (([], []), ([0.], [1.]), ([1.], [float('nan')])):
        with pytest.raises(ValueError):
            d.paired_ratio(x, y)


def test_zero_dw_control_keeps_adjoint_bias_and_native_moment_decay():
    import torch
    d = driver()
    x = torch.tensor([[1., 2.]], requires_grad=True)
    w = torch.nn.Parameter(torch.tensor([[3., 4.]]))
    b = torch.nn.Parameter(torch.tensor([5.]))
    opt = torch.optim.AdamW([w, b], lr=.01, foreach=False)
    # First give the weight nonzero native history.
    w.grad = torch.ones_like(w)
    b.grad = torch.ones_like(b)
    opt.step()
    previous = opt.state[w]['exp_avg'].clone()
    opt.zero_grad(set_to_none=True)
    d.zero_dw_linear(x, w, b).sum().backward()
    torch.testing.assert_close(x.grad, w.detach())
    torch.testing.assert_close(b.grad, torch.ones_like(b))
    assert w.grad is not None
    torch.testing.assert_close(w.grad, torch.zeros_like(w))
    opt.step()
    torch.testing.assert_close(opt.state[w]['exp_avg'], .9 * previous)
    assert opt.state[w]['step'].item() == 2


def test_summary_handles_incomplete_run_without_promoting_it(tmp_path):
    d = driver()
    path = d.allocate_attempt(tmp_path, 'prepare')
    d.seal_record(path, {'stage': 'prepare', 'status': 'BLOCKED_HARDWARE', 'reason': 'actual MPS unavailable', 'measurement_hashes': {}})
    report = d.summarize_run(tmp_path)
    assert report['terminal_state'] == 'BLOCKED_HARDWARE'
    assert report['primary_time_to_target'] is None
    assert report['new_performance_claim'] is False


def test_help_has_no_output_directory_or_training_side_effect(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'analysis/run_dw_sampling.py'), '--help'], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0
    assert 'feasibility' in result.stdout and 'confirm' in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_native_environment_checks_actual_versions_against_pins():
    d = driver()
    lock = 'torch==2.15.0 --hash=sha256:abc\nPyYAML==6.0.3 --hash=sha256:def\n'
    assert d.pin_mismatches(lock, {'torch': '2.15.0', 'pyyaml': '6.0.3'}) == []
    assert d.pin_mismatches(lock, {'torch': '2.14.0', 'PyYAML': '6.0.3'}) == ['torch: expected 2.15.0, observed 2.14.0']
    assert d.pin_mismatches(lock, {'torch': '2.15.0'}) == ['PyYAML: expected 6.0.3, observed unavailable']


def test_shared_deadline_never_restarts_for_later_commands():
    d = driver()
    assert d.remaining_timeout(100., clock=lambda: 75.) == 25.
    with pytest.raises(TimeoutError):
        d.remaining_timeout(100., clock=lambda: 101.)
    assert d.remaining_timeout(None, clock=lambda: 101.) is None
    d.check_command_deadline(0, None)


def test_worker_stage_boundary_rejects_injected_cost_request(tmp_path):
    d = driver()
    attempt = d.allocate_attempt(tmp_path, 'prepare')
    (attempt/'source_manifest.json').write_text('{}')
    request = {'kind': 'cost', 'attempt': str(attempt), 'out': str(tmp_path), 'config': str(ROOT/'configs/dw_sampling.yaml')}
    with pytest.raises(ValueError, match='stage boundary'):
        d.validate_worker_request(request, 'prepare', ROOT/'configs/dw_sampling.yaml', tmp_path, attempt/'request.json')


def test_exclusive_session_rejects_second_owner(tmp_path):
    d = driver()
    lock = tmp_path/'measurement.lock'
    with d.exclusive_session(lock):
        with pytest.raises(RuntimeError, match='exclusive'):
            with d.exclusive_session(lock):
                pytest.fail('a second session acquired the measurement device')
    with d.exclusive_session(lock):
        pass


def test_selected_dense_block_exceeds_100_updates_and_reaches_measured_minimum():
    d = driver()
    count, observations = d.choose_block_length(lambda n: .001*n, 3)
    assert count >= 500
    assert observations[0] == {'updates': 3, 'seconds': .003}
    assert observations[-1]['updates'] == count
    assert observations[-1]['seconds'] >= .5


def test_dense_calibration_has_no_fixed_iteration_limit_and_honors_target():
    d = driver()
    durations = iter([.499]*120 + [.75])
    count, observations = d.choose_block_length(lambda n: next(durations), 1)
    assert len(observations) == 121
    assert count > 100
    assert observations[-1]['seconds'] == .75
    count, observations = d.choose_block_length(lambda n: .01*n, 1, minimum_s=.8)
    assert count >= 80 and observations[-1]['seconds'] >= .8


@pytest.mark.parametrize('initial', [0, -1, 1.2, True, float('inf')])
def test_dense_calibration_rejects_nonpositive_or_noninteger_initial(initial):
    with pytest.raises(ValueError):
        driver().choose_block_length(lambda n: .5, initial)


@pytest.mark.parametrize('seconds', [0., -.1, float('nan'), float('inf')])
def test_dense_calibration_rejects_invalid_observations(seconds):
    with pytest.raises(ValueError):
        driver().choose_block_length(lambda n: seconds, 1)


@pytest.mark.parametrize('minimum', [0., -.1, float('nan'), float('inf')])
def test_dense_calibration_rejects_invalid_duration_target(minimum):
    with pytest.raises(ValueError):
        driver().choose_block_length(lambda n: .5, 1, minimum_s=minimum)


def test_fastest_supported_native_uses_median_despite_unstable_warmup():
    d = driver()
    variants = {
        'single': {'status': 'pass', 'warmup': {'stable': True}, 'pilot_s': [.3, .3, .3]},
        'fused': {'status': 'pass', 'warmup': {'stable': False}, 'pilot_s': [.1, .1, 10.]},
        'foreach': {'status': 'unsupported', 'reason': 'unsupported native implementation'},
    }
    assert d.select_native_variant(variants) == 'fused'
    with pytest.raises(RuntimeError, match='qualified'):
        d.select_native_variant({'foreach': variants['foreach']})
    with pytest.raises(ValueError):
        d.select_native_variant({'single': {'status': 'pass', 'pilot_s': [float('nan')]}})


def test_native_qualification_retains_seeded_pilot_order_and_raw_diagnostics(monkeypatch):
    import torch
    import oar.timing
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    class CPUFixture:
        last_variant = None
        def __init__(self, cfg, shape, variant, session):
            self.variant = variant
            self.p = {'p': torch.nn.Parameter(torch.tensor([1., -2.]))}
            options = {} if variant == 'default' else {'foreach': variant == 'foreach', 'fused': variant == 'fused'}
            self.opt = torch.optim.AdamW(list(self.p.values()), lr=.001, **options)
        def step(self):
            CPUFixture.last_variant = self.variant
            self.p['p'].grad = torch.tensor([.1, -.2])
            self.opt.step()
    def warmup(operation, sync, **kwargs):
        operation()
        return {'stable': False, 'raw_s': [.1, .4]}
    def measure(operation, sync, blocks):
        operation()
        return [.1, .1, 10.] if CPUFixture.last_variant == 'fused' else [.3, .3, .3]
    monkeypatch.setattr(d, 'Fixture', CPUFixture)
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: None)
    monkeypatch.setattr(oar.timing, 'warmup', warmup)
    monkeypatch.setattr(oar.timing, 'measure_blocks', measure)
    orders, seeds = [], []
    for session in (0, 1):
        selected, results = d.native_variant_qualification(cfg, cfg['shapes'][0], session, 1)
        assert selected == 'fused'
        assert all(v['status'] == 'pass' and not v['warmup']['stable'] for v in results.values())
        assert results['fused']['pilot_s'] == [.1, .1, 10.]
        assert all(v['pilot_order'] == list(results) for v in results.values())
        orders.append(list(results))
        seeds.append(results['fused']['order_seed'])
    assert seeds[0] != seeds[1]
    assert orders[0] != orders[1]


@pytest.mark.parametrize('failure', ['non-finite gradient requires repair', 'MPS device error: unsupported command buffer'])
def test_native_qualification_does_not_hide_invalid_gradient_or_device_failure(monkeypatch, failure):
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    class BrokenFixture:
        def __init__(self, *args, **kwargs):
            pass
        def step(self):
            raise RuntimeError(failure)
    monkeypatch.setattr(d, 'Fixture', BrokenFixture)
    with pytest.raises(RuntimeError, match=failure):
        d.native_variant_qualification(cfg, cfg['shapes'][0], 0, 1)


def test_native_qualification_excludes_numerically_unsupported_variant(monkeypatch):
    import torch
    import oar.timing
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    class CPUFixture:
        def __init__(self, cfg, shape, variant, session):
            self.variant = variant
            self.p = {'p': torch.nn.Parameter(torch.tensor([1.]))}
            self.opt = torch.optim.AdamW(list(self.p.values()), lr=.001, foreach=False)
        def step(self):
            self.p['p'].grad = torch.tensor([100. if self.variant == 'fused' else .1])
            self.opt.step()
    monkeypatch.setattr(d, 'Fixture', CPUFixture)
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: None)
    monkeypatch.setattr(oar.timing, 'warmup', lambda *args, **kwargs: {'stable': False, 'raw_s': [.1, .4]})
    monkeypatch.setattr(oar.timing, 'measure_blocks', lambda *args: [.3, .3, .3])
    selected, results = d.native_variant_qualification(cfg, cfg['shapes'][0], 0, 1)
    assert results['fused']['status'] == 'unsupported'
    assert selected != 'fused'


def test_storage_guard_allows_nullable_output_ceiling_but_keeps_disk_reserve(tmp_path):
    d = driver()
    (tmp_path/'raw').write_bytes(b'abc')
    cfg = {'resources': {'minimum_free_disk_bytes': 0, 'maximum_output_bytes': None}}
    assert d.storage_guard(cfg, tmp_path)['run_output_bytes'] == 3
    cfg['resources']['maximum_output_bytes'] = 2
    with pytest.raises(MemoryError):
        d.storage_guard(cfg, tmp_path)
    cfg['resources'].update(maximum_output_bytes=None, minimum_free_disk_bytes=10**30)
    with pytest.raises(MemoryError):
        d.storage_guard(cfg, tmp_path)


def test_forecast_uses_uncapped_count_current_target_and_two_sessions():
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    calibration = [{'shape': cfg['shapes'][0], 'native_complete_step_s': [.001, .002, .001]}]
    result = d.forecast_stage(cfg, calibration, 'feasibility')
    assert result['sessions'] == 2
    assert result['deadline_s'] is None
    assert result['per_shape'][0]['updates_per_block'] == 500
    assert result['per_shape'][0]['bound_block_s'] == pytest.approx(1.)
    json.dumps(result, allow_nan=False)
    cfg['timing']['minimum_block_s'] = 1.
    longer = d.forecast_stage(cfg, calibration, 'feasibility')
    assert longer['per_shape'][0]['updates_per_block'] == 1000
    assert longer['forecast_s'] > result['forecast_s']


def test_unstable_warmup_retains_pairs_and_charged_setup(tmp_path, monkeypatch):
    import torch
    import oar.timing
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: None)
    monkeypatch.setattr(oar.timing, 'warmup', lambda operation, sync, **kwargs: {'stable': False, 'raw_s': [.1, .4]})
    def measure(operation, sync, blocks):
        operation()
        return [1.]*blocks
    monkeypatch.setattr(oar.timing, 'measure_blocks', measure)
    class CPUFixture:
        sampling_setup_s = 3.
        def restore(self, common):
            self.updates = common['updates']
        def step(self):
            self.updates += 1
    path = tmp_path/'pairs.jsonl'
    result = d.timed_pairs(cfg, CPUFixture(), CPUFixture(), {'updates': 8}, 2, 42, path)
    assert len(result['pairs']) == 30
    assert all(not w['stable'] for w in result['warmup'].values())
    assert len(path.read_text().splitlines()) == 30
    assert json.loads(path.with_suffix('.warmup.json').read_text()) == result['warmup']
    assert result['statistics']['ratio'] == pytest.approx(1.1)
    assert result['steady_state_statistics']['ratio'] == pytest.approx(1.)


@pytest.mark.parametrize('noise', [.01, .12])
@pytest.mark.parametrize('kind', ['feasibility', 'cost'])
def test_stage_uses_only_declared_noise_shapes_without_warmup_veto_or_replacement(tmp_path, monkeypatch, noise, kind):
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    qualification = tmp_path/'qualification.json'
    qualification.write_text(json.dumps({'native_resource_calibration': [
        {'shape': s, 'native_complete_step_s': [.01, .01, .01]} for s in cfg['shapes']]}))
    monkeypatch.setattr(d, 'source_hashes', lambda stage: {})
    def prerequisite(out, stage, bindings):
        if stage == 'feasibility':
            return {'delta': 2*noise, 'sessions': [{'shapes': [
                {'shape': s, 'selected_variant': 'single'} for s in cfg['shapes']]}]*2}
        return {'qualification': str(qualification)}
    monkeypatch.setattr(d, 'require_stage', prerequisite)
    def cpu_worker(attempt, args, *, timeout, env):
        request = json.loads(Path(args[-1]).read_text())
        assert timeout is None and request['deadline_s'] is None
        if kind == 'cost':
            assert (attempt/'frozen_delta.json').is_file(), 'delta must be frozen before candidate workers start'
            frozen = json.loads((attempt/'frozen_delta.json').read_text())
            assert frozen['delta'] == pytest.approx(2*noise)
        if request['kind'] == 'diagnostic':
            d.write_json(attempt/'diagnostic.json', {'cpu_boundary_fixture': True})
            return 0, '', ''
        shapes = []
        for shape in cfg['shapes']:
            error = noise if shape['id'] in ('middle2048', 'middle4096') else .9
            comparison = {'warmup': {'dense': {'stable': False, 'raw_s': [.1, .3]},
                                     'candidate': {'stable': False, 'raw_s': [.1, .3]}},
                          'pairs': [{'seconds': {'dense': 1+error, 'candidate': 1.}}]*30,
                          'statistics': {'ci95': [.4, .5]}}
            names = ['AA', 'zero'] if kind == 'feasibility' else [f'{s}-{r}' for s in cfg['samplers'] for r in cfg['rhos']]
            shapes.append({'shape': shape, 'comparisons': {name: comparison for name in names}})
        d.write_json(request['result'], {'status': 'pass', 'session': request['session'], 'shapes': shapes})
        return 0, '', ''
    monkeypatch.setattr(d, 'command', cpu_worker)
    attempt = d.allocate_attempt(tmp_path, kind)
    result = d.hardware_stage(cfg, ROOT/'configs/dw_sampling.yaml', tmp_path, attempt, kind)
    assert result['status'] == 'pass'
    assert result['delta'] == pytest.approx(2*noise)
    assert len(result['sessions']) == 2
    assert all(len(s['shapes']) == 5 for s in result['sessions'])
    assert len(list(attempt.glob('session-invocation-*-request.json'))) == 2
    assert len(result['eligible']) == (2 if kind == 'feasibility' else 12)
    frozen = json.loads((attempt/'frozen_delta.json').read_text())
    assert frozen['noise_calibration_shapes'] == ['middle2048', 'middle4096']


@pytest.mark.parametrize('session', [0, 1])
@pytest.mark.parametrize('kind', ['feasibility', 'cost'])
def test_cost_session_recalibrates_baseline_only_and_keeps_shape_order(tmp_path, monkeypatch, session, kind):
    import torch
    import oar.hardware
    import oar.timing
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.mps, 'recommended_max_memory', lambda: 10**14)
    monkeypatch.setattr(oar.hardware, 'resource_snapshot', lambda device: {
        'physical_memory_bytes': 10**14, 'available_memory_bytes': 10**14})
    # CPU fixture performs real reset/update counting; only device allocation
    # and elapsed-time measurement are replaced at the hardware boundary.
    class CPUFixture:
        elapsed = 0.
        def __init__(self, cfg, shape, variant='single', mode='dense', rho=.25, session=0):
            self.mode, self.updates, self.states = mode, 0, {}
            self.sampling_setup_s = 0.
        def step(self):
            self.updates += 1
            CPUFixture.elapsed += .001 if self.mode == 'dense' else .0005
        def snapshot(self):
            return {'updates': self.updates, 'sampling': {}}
        def restore(self, common):
            self.updates = common['updates']
        def backward(self, diagnostic=False):
            return {}
    def measure(operation, sync, blocks):
        raw = []
        for _ in range(blocks):
            start = CPUFixture.elapsed
            operation()
            raw.append(CPUFixture.elapsed-start)
        return raw
    monkeypatch.setattr(d, 'Fixture', CPUFixture)
    monkeypatch.setattr(d, 'state_streams', lambda f: {'test_cpu_updates': f.updates})
    monkeypatch.setattr(d, 'native_variant_qualification', lambda *args: ('single', {
        'single': {'status': 'pass', 'warmup': {'stable': False, 'raw_s': [.1, .4]}, 'pilot_s': [.5]*3}}))
    monkeypatch.setattr(oar.timing, 'measure_blocks', measure)
    monkeypatch.setattr(oar.timing, 'warmup', lambda *args, **kwargs: {'stable': False, 'raw_s': [.1, .4]})
    request = {'session': session, 'kind': kind, 'deadline_s': None, 'result': str(tmp_path/'session.json'),
               'calibration': [{'shape': s, 'native_complete_step_s': [.001]*3} for s in cfg['shapes']],
               'baseline': {'shapes': [{'shape': s, 'selected_variant': 'single', 'variants': {},
                                        'updates_per_block': 3} for s in cfg['shapes']]}}
    result = d.cost_session(cfg, request)
    assert result['status'] == 'pass'
    assert [r['shape']['id'] for r in result['shapes']] == [s['id'] for s in cfg['shapes']]
    for row in result['shapes']:
        assert row['updates_per_block'] >= 500
        if kind == 'cost':
            assert row['selected_dense_pilot'][0]['updates'] == 3
        else:
            assert row['dense_pilot'][-1]['seconds'] >= .5
        assert row['selected_dense_pilot'][-1]['seconds'] >= .5
        assert all(len(p.get('raw_s', [])) == 3 and p['seconds'] == min(p['raw_s']) for p in row['selected_dense_pilot'])
        assert len(row['comparisons']) == (6 if kind == 'cost' else 2)
        assert all(c['updates_per_block'] == row['updates_per_block'] and len(c['pairs']) == 30
                   for c in row['comparisons'].values())
    assert len(list(tmp_path.glob('*-pairs.jsonl'))) == (30 if kind == 'cost' else 10)


def test_offline_diagnostics_accepts_no_deadline_and_keeps_native_state_rows(tmp_path, monkeypatch):
    import copy
    import torch
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    cfg['diagnostic'] = {'native_states': [0, 1], 'draws': 1}
    cfg['rhos'] = [.5]
    class CPUFixture:
        def __init__(self, *args):
            self.p = {'w1': torch.nn.Parameter(torch.tensor([[.2, .3]]))}
            self.x = torch.tensor([[1., 2.], [3., 4.]])
            self.opt = torch.optim.AdamW(list(self.p.values()), lr=.001, foreach=False)
            self.updates = 0
        def loss(self):
            y = self.x@self.p['w1'].T
            if y.requires_grad:
                y.retain_grad()
            return [y.square().mean()], {(0, 'w1'): (self.x, y)}
        def backward(self, diagnostic=False):
            self.opt.zero_grad(set_to_none=True)
            losses, captured = self.loss()
            losses[0].backward()
            return captured
        def step(self):
            self.backward()
            self.opt.step()
            self.updates += 1
        def snapshot(self):
            return {'p': self.p['w1'].detach().clone(), 'opt': copy.deepcopy(self.opt.state_dict()), 'updates': self.updates}
        def restore(self, state):
            with torch.no_grad():
                self.p['w1'].copy_(state['p'])
            self.opt.load_state_dict(copy.deepcopy(state['opt']))
            self.updates = state['updates']
    monkeypatch.setattr(d, 'Fixture', CPUFixture)
    monkeypatch.setattr(torch.mps, 'synchronize', lambda: None)
    result = d.offline_diagnostics(cfg, {'deadline_s': None, 'variant': 'single', 'attempt': str(tmp_path)})
    assert result['states'] == [0, 1]
    assert len(result['rows']) == 4
    assert {r['sampler'] for r in result['rows']} == {'importance', 'uniform'}
    assert len(list(tmp_path.glob('diagnostic-state-*.json'))) == 2
    assert all(r['normalized_gradient_squared_errors']['w1'] >= 0 for r in result['rows'])


def test_latest_passing_attempt_supersedes_historical_hardware_failure(tmp_path):
    d = driver()
    for status in ('BLOCKED_HARDWARE', 'pass'):
        p = d.allocate_attempt(tmp_path, 'prepare')
        d.seal_record(p, {'stage': 'prepare', 'status': status, 'measurement_hashes': {}})
    result = d.summarize_run(tmp_path)
    assert result['terminal_state'] == 'INCOMPLETE_RESOURCE'
    assert len(result['records']) == 2


def test_relative_argv_competing_work_is_detected_by_working_directory():
    d = driver()
    assert d.is_competing_work(['python', '-m', 'pytest', '-q'], str(ROOT))
    assert d.is_competing_work(['python', 'analysis/collect_operator_correctness.py'], str(ROOT))
    assert not d.is_competing_work(['python', 'ordinary_script.py'], '/tmp')


def test_timeout_exit_is_a_resource_stop():
    d = driver()
    with pytest.raises(TimeoutError):
        d.check_command_deadline(124, 100., clock=lambda: 99.)
    with pytest.raises(TimeoutError):
        d.check_command_deadline(0, 100., clock=lambda: 101.)
    d.check_command_deadline(0, 100., clock=lambda: 99.)


@pytest.mark.mps
def test_native_optimizer_step_counter_device_does_not_disqualify_fused():
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip('actual MPS unavailable; not qualification evidence')
    d = driver()
    p = torch.nn.Parameter(torch.tensor([1., -2.], device='mps'))
    q = torch.nn.Parameter(p.detach().clone())
    actual = torch.optim.AdamW([p], foreach=False, fused=True)
    reference = torch.optim.AdamW([q], foreach=False, fused=False)
    for step in range(5):
        gradient = torch.tensor([.2+step/10, -.3], device='mps')
        p.grad, q.grad = gradient.clone(), gradient.clone()
        actual.step()
        reference.step()
        d.assert_native_equivalent({'p': p}, actual, {'p': q}, reference)
    assert actual.state[p]['step'].device.type == 'mps'
    assert reference.state[q]['step'].device.type == 'cpu'
