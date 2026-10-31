# SPDX-License-Identifier: AGPL-3.0-only
"""Protocol tests exercise validation and persisted evidence, not source text."""
import importlib.util
import copy
import hashlib
import json
import math
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
            cfg['asset']['body_downloads'] = not cfg['asset']['body_downloads']
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


def test_missing_failed_and_source_changed_prerequisites_cannot_launch(tmp_path, monkeypatch):
    d = driver()
    current = {'source': 'v1'}
    monkeypatch.setattr(d, 'source_hashes', lambda stage: dict(current))
    with pytest.raises(ValueError, match='prerequisite'):
        d.require_stage(tmp_path, 'feasibility', {'source': 'v1'})
    p = d.allocate_attempt(tmp_path, 'feasibility')
    d.seal_record(p, {'stage': 'feasibility', 'status': 'STOP_COST', 'measurement_hashes': {'source': 'v1'}})
    with pytest.raises(ValueError, match='prerequisite'):
        d.require_stage(tmp_path, 'feasibility', {'source': 'v1'})
    p2 = d.allocate_attempt(tmp_path, 'feasibility')
    d.seal_record(p2, {'stage': 'feasibility', 'status': 'pass', 'measurement_hashes': {'source': 'v1'}})
    current['source'] = 'v2'
    with pytest.raises(ValueError, match='source'):
        d.require_stage(tmp_path, 'feasibility', {'source': 'v2'})
    current['source'] = 'v1'
    assert d.require_stage(tmp_path, 'feasibility', {'source': 'v1'})['status'] == 'pass'


@pytest.fixture
def approved_carryover(tmp_path, monkeypatch):
    """Real sealed filesystem evidence with Git blobs isolated at its boundary."""
    d = driver()
    repository = tmp_path/'repository'
    old = {name: (ROOT/name).read_bytes() for name in d.CORE+d.SAMPLED}
    old_driver = old['analysis/run_dw_sampling.py'].decode().replace("REVIEWED = '", "OLD_METADATA = True\nREVIEWED = '", 1)
    old['analysis/run_dw_sampling.py'] = old_driver.encode()
    old_cfg = yaml.safe_load(old['configs/dw_sampling.yaml'])
    old_cfg['timing'].pop('cost_gate_upper_ratio')
    old['configs/dw_sampling.yaml'] = yaml.safe_dump(old_cfg).encode()
    for name in d.CORE+d.SAMPLED+('src/oar/sampling_lm.py', 'src/oar/sampling_data.py'):
        path = repository/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    (repository/'decisions/dw_sampling_reopening.md').write_text('explicit prospective carryover scope\n')
    (repository/'tests/test_dw_sampling_protocol.py').write_text('# current, explicitly pinned protocol tests\n')
    monkeypatch.setattr(d, 'ROOT', repository)
    monkeypatch.setattr(d, 'git_blob', lambda head, name: old[name], raising=False)
    out = tmp_path/'run'
    def make(stage='feasibility', status='pass', missing=None):
        if stage == 'cost':
            old['configs/dw_sampling.yaml'] = (repository/'configs/dw_sampling.yaml').read_bytes()
        names = d.CORE if stage == 'feasibility' else d.CORE+d.SAMPLED
        previous = {name: hashlib.sha256(old[name]).hexdigest() for name in names if name != missing}
        attempt = d.allocate_attempt(out, stage)
        d.write_json(attempt/'source_manifest.json', {'actual_head': 'a'*40, 'measurement_hashes': previous,
                                                    'original_identity': 'retain this source attribution'})
        d.seal_record(attempt, {'stage': stage, 'status': status, 'measurement_hashes': previous,
                               'source_manifest_sha256': d.digest(attempt/'source_manifest.json'), 'delta': .2085})
        entry = {'stage': stage, 'record_sha256': d.digest(attempt/'record.json'),
                 'previous_measurement_hashes': previous, 'current_measurement_hashes': d.source_hashes('cost')}
        amendment = {'prerequisite_carryover': [entry], 'bound_files': d.source_hashes('cost')}
        (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
        return attempt, entry, amendment
    return d, repository, out, old, make


@pytest.mark.parametrize('stage', ['prepare', 'feasibility', 'cost'])
def test_exact_approved_carryover_preserves_original_record_and_identity(approved_carryover, stage):
    d, repository, out, old, make = approved_carryover
    attempt, entry, amendment = make(stage)
    original = (attempt/'record.json').read_bytes()
    result = d.require_stage(out, stage, d.source_hashes('cost'))
    assert result['status'] == 'pass' and result['delta'] == .2085
    assert result['measurement_hashes'] == entry['previous_measurement_hashes']
    assert result['carryover']['record_sha256'] == entry['record_sha256']
    assert result['carryover']['previous_actual_head'] == 'a'*40
    assert result['carryover']['current_measurement_hashes'] == entry['current_measurement_hashes']
    assert (attempt/'record.json').read_bytes() == original


def test_native_carryover_excludes_only_unconsumed_model_and_asset_dictionaries(approved_carryover):
    d, repository, out, old, make = approved_carryover
    attempt, entry, amendment = make()
    path = repository/'configs/dw_sampling.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['learning'] = {'separately_frozen_model': 42}
    cfg['asset'] = {'separately_gated_asset': 'not authorized by native carryover'}
    path.write_text(yaml.safe_dump(cfg))
    entry['current_measurement_hashes'] = d.source_hashes('cost')
    amendment['bound_files'] = dict(entry['current_measurement_hashes'])
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    result = d.require_stage(out, 'feasibility', d.source_hashes('cost'))
    assert result['carryover']['native_config_exclusions'] == ['timing.cost_gate_upper_ratio', 'learning', 'asset', 'schema_version']
    assert result['measurement_hashes'] == entry['previous_measurement_hashes']


@pytest.mark.parametrize('setting', ['shape', 'rho', 'optimizer', 'flags', 'timing'])
def test_carryover_rejects_every_changed_native_setting_even_when_new_hash_is_approved(approved_carryover, setting):
    d, repository, out, old, make = approved_carryover
    attempt, entry, amendment = make()
    path = repository/'configs/dw_sampling.yaml'
    cfg = yaml.safe_load(path.read_text())
    if setting == 'shape':
        cfg['shapes'][0]['n'] = 4097
    elif setting == 'rho':
        cfg['rhos'] = [.1]
    elif setting == 'optimizer':
        cfg['optimizer']['lr'] = .002
    elif setting == 'flags':
        cfg['fallback'] = '1'
    else:
        cfg['timing']['pairs'] = 31
    path.write_text(yaml.safe_dump(cfg))
    entry['current_measurement_hashes'] = d.source_hashes('cost')
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match='native configuration'):
        d.require_stage(out, 'feasibility', d.source_hashes('cost'))


def test_current_carryover_map_must_match_actual_files_even_if_caller_and_entry_agree(approved_carryover):
    d, repository, out, old, make = approved_carryover
    attempt, entry, amendment = make()
    entry['current_measurement_hashes']['analysis/run_dw_sampling.py'] = '0'*64
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match='actual consumer'):
        d.require_stage(out, 'feasibility', entry['current_measurement_hashes'])


@pytest.mark.parametrize('missing', ['analysis/run_dw_sampling.py', 'configs/dw_sampling.yaml'])
def test_carryover_requires_complete_original_measurement_map(approved_carryover, missing):
    d, repository, out, old, make = approved_carryover
    make(missing=missing)
    with pytest.raises(ValueError, match='original measurement map'):
        d.require_stage(out, 'feasibility', d.source_hashes('cost'))


@pytest.mark.parametrize('pin_protocol_test', [True, False])
def test_native_consumer_carryover_still_requires_explicit_current_protocol_test_pin(approved_carryover, pin_protocol_test):
    d, repository, out, old, make = approved_carryover
    attempt, entry, amendment = make(stage='prepare')
    entry['current_measurement_hashes'] = d.source_hashes('feasibility')
    if not pin_protocol_test:
        amendment['bound_files'].pop('tests/test_dw_sampling_protocol.py')
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    if pin_protocol_test:
        result = d.require_stage(out, 'prepare', d.source_hashes('feasibility'))
        assert result['carryover']['current_measurement_hashes'] == d.source_hashes('feasibility')
    else:
        with pytest.raises(ValueError, match='explicitly pinned'):
            d.require_stage(out, 'prepare', d.source_hashes('feasibility'))


@pytest.mark.parametrize('stage', ['feasibility', 'cost'])
@pytest.mark.parametrize('mutation', ['missing_entry', 'record', 'stage', 'previous_hash', 'current_hash',
                                    'sampler', 'hardware', 'timing', 'original_scope', 'function',
                                    'config', 'git_blob', 'failed', 'stale', 'unsealed_manifest'])
def test_carryover_rejects_wrong_identity_or_changed_measurement(approved_carryover, mutation, stage):
    d, repository, out, old, make = approved_carryover
    attempt, entry, amendment = make(stage=stage, status='STOP_COST' if mutation == 'failed' else 'pass')
    if mutation == 'missing_entry':
        amendment.pop('prerequisite_carryover')
    elif mutation == 'record':
        entry['record_sha256'] = '0'*64
    elif mutation == 'stage':
        entry['stage'] = 'prepare'
    elif mutation == 'previous_hash':
        entry['previous_measurement_hashes'] = {**entry['previous_measurement_hashes'], 'analysis/run_dw_sampling.py': '0'*64}
    elif mutation == 'current_hash':
        entry['current_measurement_hashes'] = {**entry['current_measurement_hashes'], 'analysis/run_dw_sampling.py': '0'*64}
    elif mutation in ('sampler', 'hardware', 'timing', 'original_scope'):
        name = {'sampler': 'src/oar/sampled_linear.py', 'hardware': 'src/oar/hardware.py',
                'timing': 'src/oar/timing.py', 'original_scope': 'decisions/dw_sampling_scope.md'}[mutation]
        with (repository/name).open('a') as stream:
            stream.write('\n# unapproved change, even if the entry pins its new hash\n')
        entry['current_measurement_hashes'] = d.source_hashes('cost')
    elif mutation == 'function':
        path = repository/'analysis/run_dw_sampling.py'
        path.write_text(path.read_text().replace("self.seed = cfg['seed']+session", "self.seed = cfg['seed']+session+1", 1))
        entry['current_measurement_hashes'] = d.source_hashes('cost')
    elif mutation == 'config':
        path = repository/'configs/dw_sampling.yaml'
        cfg = yaml.safe_load(path.read_text())
        cfg['timing']['pairs'] = 31
        path.write_text(yaml.safe_dump(cfg))
        entry['current_measurement_hashes'] = d.source_hashes('cost')
    elif mutation == 'git_blob':
        old['analysis/run_dw_sampling.py'] += b'\n# differs from the sealed source hash\n'
    elif mutation == 'stale':
        later = d.allocate_attempt(out, stage)
        d.seal_record(later, {'stage': stage, 'status': 'STOP_COST', 'measurement_hashes': {}})
    elif mutation == 'unsealed_manifest':
        path = attempt/'source_manifest.json'
        identity = json.loads(path.read_text())
        identity['actual_head'] = 'b'*40
        path.write_text(json.dumps(identity))
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    with pytest.raises(ValueError):
        d.require_stage(out, stage, d.source_hashes('cost'))


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
@pytest.mark.parametrize('uppers', [(.5, .5), (.9, .9), (.9, 1.01), (.99, 1.)])
def test_stage_uses_only_declared_noise_shapes_without_warmup_veto_or_replacement(tmp_path, monkeypatch, noise, kind, uppers):
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
                          'statistics': {'ci95': [.4, uppers[request['session']]]}}
            names = ['AA', 'zero'] if kind == 'feasibility' else [f'{s}-{r}' for s in cfg['samplers'] for r in cfg['rhos']]
            shapes.append({'shape': shape, 'comparisons': {name: comparison for name in names}})
        d.write_json(request['result'], {'status': 'pass', 'session': request['session'], 'shapes': shapes})
        return 0, '', ''
    monkeypatch.setattr(d, 'command', cpu_worker)
    attempt = d.allocate_attempt(tmp_path, kind)
    result = d.hardware_stage(cfg, ROOT/'configs/dw_sampling.yaml', tmp_path, attempt, kind)
    cutoff = 1-2*noise if kind == 'feasibility' else 1.
    passes = all(upper < cutoff for upper in uppers)
    assert result['status'] == ('pass' if passes else 'STOP_COST')
    assert result['delta'] == pytest.approx(2*noise)
    assert len(result['sessions']) == 2
    assert all(len(s['shapes']) == 5 for s in result['sessions'])
    assert len(list(attempt.glob('session-invocation-*-request.json'))) == 2
    expected_arms = 2 if kind == 'feasibility' else 12
    assert len(result['eligible']) == (expected_arms if passes else 0)
    guarded = all(upper < 1-2*noise for upper in uppers)
    assert len(result['noise_guard_eligible']) == (expected_arms if guarded else 0)
    assert result['gate_upper_ratio'] == pytest.approx(cutoff)
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


# Invented byte fixtures for the post-cost-gate data helper.
def data_module():
    assert (ROOT/'src/oar/sampling_data.py').is_file(), 'byte data helper is not implemented'
    from oar import sampling_data
    return sampling_data


@pytest.mark.parametrize(('rows', 'expected'), [
    ([' pre\r\n', ' = Title = \r\n', 'x\r', '= = Sub = =\n', '= Next =\n'],
     [' pre\n', ' = Title = \nx\n= = Sub = =\n', '= Next =\n']),
    (['pre', '= Title =\nbody'], ['pre', '= Title =\nbody']),
    (['= Ti', 'tle =\n'], ['= Title =\n']),
    (['x\r', '\ny'], ['x\n\ny']),
    (['', '\r\n', 'a\rb'], ['\na\nb']),
    (['', ''], []),
    ([' \t'], [' \t']),
    (['= A =\n= Punctuation! =\n= Alpha = beta =\n'],
     ['= A =\n', '= Punctuation! =\n', '= Alpha = beta =\n']),
    (['= = Subsection = =\n= =\n=  =\n== Deep ==\n= =bad ='],
     ['= = Subsection = =\n= =\n=  =\n== Deep ==\n= =bad =']),
])
def test_data_parser_preserves_normalized_rows_and_heading_boundaries(rows, expected):
    d = data_module()
    articles = d.normalize_articles(iter(rows), 'train')
    assert [a.text for a in articles] == expected
    assert ''.join(a.text for a in articles) == ''.join(row.replace('\r\n', '\n').replace('\r', '\n') for row in rows)
    assert [a.source_article_index for a in articles] == list(range(len(expected)))
    assert [a.source_split for a in articles] == ['train']*len(expected)
    assert [a.sha256 for a in articles] == [hashlib.sha256(s.encode('utf8')).hexdigest() for s in expected]


@pytest.mark.parametrize('bad', [None, 5, b'bytes', {'text': 'invented'}])
def test_data_parser_rejects_nonstring_rows(bad):
    d = data_module()
    with pytest.raises(TypeError, match='row'):
        d.normalize_articles(['valid', bad], 'train')


def test_data_records_are_frozen_and_utf8_encoding_includes_each_boundary_once():
    from dataclasses import FrozenInstanceError
    d = data_module()
    article = d.normalize_articles(['é'], 'train')[0]
    empty = d.Article('train', 9, hashlib.sha256(b'').hexdigest(), '')
    assert d.encode_articles([article, empty]) == [(256, 195, 169, 257), (256, 257)]
    with pytest.raises(FrozenInstanceError):
        article.text = 'changed'
    block = d.TargetBlock(0, 1, 3, 0)
    with pytest.raises(FrozenInstanceError):
        block.target_start = 2


def test_data_hash_split_preserves_source_order_and_duplicate_multiplicity():
    d = data_module()
    articles = [d.Article('train', i, f'{h:064x}', str(i)) for i, h in enumerate([10, 11, 10, 21, 30])]
    fitting, development = d.split_training_articles(articles)
    assert [a.source_article_index for a in fitting] == [1, 3]
    assert [a.source_article_index for a in development] == [0, 2, 4]
    assert fitting[0] is articles[1] and development[1] is articles[2]


def test_data_exact_training_exclusions_keep_evaluation_duplicates_and_source_indices():
    d = data_module()
    fitting = d.normalize_articles(['fitting'], 'train')
    development = d.normalize_articles(['development'], 'train')
    texts = ['new', 'fitting', 'new', 'development']
    evaluation = [d.Article('validation', i, hashlib.sha256(s.encode()).hexdigest(), s) for i, s in enumerate(texts)]
    kept, excluded = d.exclude_training_duplicates(evaluation, fitting+development)
    assert [a.source_article_index for a in kept] == [0, 2]
    assert excluded == [
        {'split': 'validation', 'source_article_index': 1, 'sha256': fitting[0].sha256},
        {'split': 'validation', 'source_article_index': 3, 'sha256': development[0].sha256},
    ]


def test_data_training_chunks_reset_at_256_and_keep_utf8_eos():
    d = data_module()
    articles = [d.normalize_articles([s], 'train')[0] for s in ['x'*256, 'é']]
    # Explicit empty articles remain valid fixtures even though the parser emits none.
    articles.append(d.Article('train', 2, hashlib.sha256(b'').hexdigest(), ''))
    blocks = d.training_chunks(articles)
    assert blocks == [d.TargetBlock(0, 1, 256, 0), d.TargetBlock(0, 257, 257, 256),
                      d.TargetBlock(1, 1, 3, 0), d.TargetBlock(2, 1, 1, 0)]


def test_data_evaluation_windows_have_exact_context_and_cover_every_target_once():
    d = data_module()
    articles = [d.normalize_articles([s], 'validation')[0] for s in ['x'*299, 'é']]
    windows = d.evaluation_windows(articles)
    assert windows == [d.TargetBlock(0, 1, 128, 0), d.TargetBlock(0, 129, 256, 0),
                       d.TargetBlock(0, 257, 300, 44), d.TargetBlock(1, 1, 3, 0)]
    assert [(w.target_end-w.input_start, w.target_start-w.input_start-1,
             w.target_end-w.input_start-1) for w in windows] == [(128, 0, 127), (256, 128, 255), (256, 212, 255), (3, 0, 2)]
    for i, end in [(0, 300), (1, 3)]:
        assert [t for w in windows if w.article_index == i for t in range(w.target_start, w.target_end+1)] == list(range(1, end+1))


def test_data_monitor_hash_ranking_ties_multiplicity_and_final_source_order():
    d = data_module()
    # The digest is deliberately identical, making source index the last tie break.
    articles = [d.Article('validation', i, 'a'*64, 'z'*299) for i in [8, 2, 5]]
    windows = d.evaluation_windows(articles)
    # SHA ranks target starts 257, 1, 129. The fourth slot uses source index 2.
    expected = [d.TargetBlock(0, 257, 300, 44), d.TargetBlock(1, 1, 128, 0),
                d.TargetBlock(1, 257, 300, 44), d.TargetBlock(2, 257, 300, 44)]
    chosen = d.choose_monitor(articles, list(reversed(windows)), maximum=4)
    assert chosen == expected
    assert len(chosen) == 4 and len(set(chosen)) == 4
    assert d.choose_monitor(articles, windows, maximum=64) == windows


@pytest.mark.parametrize('maximum', [0, -1, 65, 1.5, True])
def test_data_monitor_rejects_invalid_maximum(maximum):
    d = data_module()
    with pytest.raises((ValueError, TypeError)):
        d.choose_monitor([], [], maximum=maximum)


def test_data_collation_has_shifted_targets_actual_partial_rows_and_right_padding():
    import torch
    d = data_module()
    articles = [d.normalize_articles([s], 'train')[0] for s in ['x'*256, 'é']]
    tokens = d.encode_articles(articles)
    blocks = d.training_chunks(articles)
    batch = d.collate_blocks(tokens, blocks[1:])
    assert batch['input_ids'].shape == batch['target_ids'].shape == batch['target_mask_cpu'].shape == (2, 256)
    assert batch['input_ids'].dtype == batch['target_ids'].dtype == torch.int64
    assert batch['input_ids'].device.type == batch['target_ids'].device.type == 'cpu'
    assert batch['target_mask_cpu'].dtype == torch.bool and batch['target_mask_cpu'].device.type == 'cpu'
    assert batch['input_ids'][0, :4].tolist() == [120, 258, 258, 258]
    assert batch['input_ids'][1, :4].tolist() == [256, 195, 169, 258]
    assert batch['target_ids'][0, :4].tolist() == [257, -100, -100, -100]
    assert batch['target_ids'][1, :4].tolist() == [195, 169, 257, -100]
    assert batch['target_mask_cpu'].sum(1).tolist() == [1, 3]
    assert torch.equal(batch['target_mask_cpu'], batch['target_ids'].ne(-100))
    assert batch['physical_rows'] == 512 and batch['valid_targets'] == 4
    assert batch['target_ids'][:, 3:].eq(-100).all()
    last = d.collate_blocks(tokens, blocks[1:2])
    assert last['input_ids'].shape == (1, 256)
    assert last['physical_rows'] == 256 and last['valid_targets'] == 1


def test_data_collation_scores_only_window_targets_with_exact_local_positions():
    d = data_module()
    tokens = [(256, *([120]*299), 257)]
    windows = [d.TargetBlock(0, 1, 128, 0), d.TargetBlock(0, 129, 256, 0), d.TargetBlock(0, 257, 300, 44)]
    batch = d.collate_blocks(tokens, windows)
    assert batch['target_mask_cpu'].sum(1).tolist() == [128, 128, 44]
    assert batch['target_ids'][1, :128].tolist() == [-100]*128
    assert batch['target_ids'][2, :212].tolist() == [-100]*212
    assert batch['target_ids'][2, 212:].tolist() == [120]*43+[257]
    assert batch['input_ids'][0, 128:].tolist() == [258]*128
    assert batch['valid_targets'] == 300 and batch['physical_rows'] == 768


def test_data_masks_support_target_weighting_instead_of_mean_of_batch_means():
    import torch
    d = data_module()
    tokens = [(256, 257), (256, 195, 169, 257)]
    batch = d.collate_blocks(tokens, [d.TargetBlock(0, 1, 1, 0), d.TargetBlock(1, 1, 3, 0)])
    nll = torch.zeros((2, 256), dtype=torch.float32)
    nll[0, 0] = 1.
    nll[1, :3] = torch.tensor([3., 3., 3.])
    mean = nll[batch['target_mask_cpu']].to(torch.float64).sum().item()/batch['valid_targets']
    assert mean == 2.5 and mean != (1.+3.)/2.
    assert batch['valid_targets'] == 4


@pytest.mark.parametrize('fields', [(-1, 1, 1, 0), (1, 1, 1, 0), (0, 0, 1, 0),
    (0, 2, 1, 0), (0, 1, 4, 0), (0, 1, 1, -1), (0, 1, 1, 1),
    (0, 1, 1, .5), (0, True, 1, 0)])
def test_data_collation_rejects_malformed_block_bounds(fields):
    d = data_module()
    with pytest.raises((ValueError, TypeError)):
        d.collate_blocks([(256, 195, 169, 257)], [d.TargetBlock(*fields)])


def test_data_collation_rejects_empty_overlong_and_malformed_token_batches():
    d = data_module()
    with pytest.raises(ValueError, match='empty'):
        d.collate_blocks([], [])
    with pytest.raises(ValueError):
        d.collate_blocks([(256, *([120]*257), 257)], [d.TargetBlock(0, 1, 257, 0)])
    for tokens in [(256, 999, 257), (256, 258, 257), (1, 2, 257), (256, 2, 3), (256, 1.5, 257)]:
        with pytest.raises((ValueError, TypeError)):
            d.collate_blocks([tokens], [d.TargetBlock(0, 1, 2, 0)])


def test_data_manifests_retain_provenance_counts_without_text_or_tokens():
    d = data_module()
    article = d.Article('validation', 17, hashlib.sha256('é'.encode()).hexdigest(), 'é')
    assert d.article_manifest([article]) == [{'split': 'validation', 'source_article_index': 17,
        'sha256': article.sha256, 'utf8_bytes': 2, 'targets': 3}]
    assert d.block_manifest([article], [d.TargetBlock(0, 1, 3, 0)]) == [{
        'split': 'validation', 'source_article_index': 17, 'sha256': article.sha256,
        'target_start': 1, 'target_end': 3, 'input_start': 0, 'input_length': 3, 'scored_targets': 3}]
    with pytest.raises(ValueError):
        d.block_manifest([article], [d.TargetBlock(0, 1, 4, 0)])


def test_data_module_import_does_not_load_torch():
    data_module()
    result = subprocess.run([sys.executable, '-c',
        "import sys; import oar.sampling_data; assert 'torch' not in sys.modules"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_data_monitor_ranks_article_hash_and_restores_local_evaluation_order():
    d = data_module()
    articles = [d.Article('validation', i, h*64, 'z'*129) for i, h in enumerate(['1', 'f', '0'])]
    windows = d.evaluation_windows(articles)
    # Smallest digest keys are (hash0,1), (hashf,1), (hash1,129).
    assert d.choose_monitor(articles, windows, maximum=3) == [
        d.TargetBlock(0, 129, 130, 0), d.TargetBlock(1, 1, 128, 0), d.TargetBlock(2, 1, 128, 0)]


def test_data_training_collation_covers_all_bytes_and_eos_without_bos_targets():
    d = data_module()
    article = d.normalize_articles(['x'*256], 'train')
    batch = d.collate_blocks(d.encode_articles(article), d.training_chunks(article))
    assert batch['target_ids'][batch['target_mask_cpu']].tolist() == [120]*256+[257]
    assert batch['input_ids'][0].tolist() == [256]+[120]*255
    assert batch['input_ids'][1, 0].item() == 120
    assert batch['valid_targets'] == 257


@pytest.mark.parametrize('tokens', [(256., 120, 120, 257), (256, 120, 120, 257.)])
def test_data_collation_rejects_noninteger_boundaries_outside_selected_slice(tokens):
    d = data_module()
    with pytest.raises((TypeError, ValueError)):
        d.collate_blocks([tokens], [d.TargetBlock(0, 2, 2, 1)])


@pytest.mark.parametrize('limit,elapsed,passes', [(None, 1000., True), (60., 20., True), (60., 1000., False)])
def test_prepare_qualification_obeys_only_explicit_stage_limit(tmp_path, monkeypatch, limit, elapsed, passes):
    import platform
    import oar.hardware
    d = driver()
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    cfg['resources']['maximum_stage_s'] = limit
    clock = [0.]
    timeouts = []
    monkeypatch.setattr(d.time, 'perf_counter', lambda: clock[0])
    original_remaining = d.remaining_timeout
    monkeypatch.setattr(d, 'remaining_timeout', lambda deadline, **kwargs: original_remaining(deadline, clock=lambda: clock[0]))
    original_check = d.check_command_deadline
    monkeypatch.setattr(d, 'check_command_deadline', lambda code, deadline: original_check(code, deadline, clock=lambda: clock[0]))
    monkeypatch.setattr(d, 'dependency_inventory', lambda: {'packages': []})
    monkeypatch.setattr(d, 'pin_mismatches', lambda *args: [])
    monkeypatch.setattr(platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(d.sys, 'version_info', (3, 13))
    monkeypatch.setattr(oar.hardware, 'resource_snapshot', lambda *args: {})
    attempt = d.allocate_attempt(tmp_path, 'prepare')
    def qualify(attempt, args, *, env, timeout):
        timeouts.append(timeout)
        if 'doctor' in args:
            d.write_json(attempt/'doctor.json', {'gpu_claim_eligible': True})
        if '--session-worker' in args:
            clock[0] = elapsed
            d.write_json(attempt/'qualification.json', {'actual_shapes_qualified': True})
        return 0, '', ''
    monkeypatch.setattr(d, 'command', qualify)
    if passes:
        assert d.prepare(cfg, ROOT/'configs/dw_sampling.yaml', tmp_path, attempt)['status'] == 'pass'
        assert json.loads((attempt/'qualification.json').read_text())['actual_shapes_qualified']
    else:
        with pytest.raises(TimeoutError):
            d.prepare(cfg, ROOT/'configs/dw_sampling.yaml', tmp_path, attempt)
    assert timeouts and all(t is None if limit is None else t == 60. for t in timeouts)
    forecast = json.loads((attempt/'resource_forecast.json').read_text())
    assert forecast['forecast_s'] == 120 and forecast['deadline_s'] == limit


@pytest.mark.parametrize('mutation', ['source', 'config', 'amendment', 'bound_file', 'unrelated', 'none'])
def test_end_source_stability_checks_exact_started_dependencies(tmp_path, monkeypatch, mutation):
    d = driver()
    paths = {'source.py': 'measured', 'config.yaml': 'config', 'decisions/dw_sampling_amendment.json': '{}', 'pinned.txt': 'pin'}
    for name, content in paths.items():
        p = tmp_path/name
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_text(content)
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    monkeypatch.setattr(d, 'source_hashes', lambda stage: {'source.py': d.digest(tmp_path/'source.py')})
    identity = {'measurement_hashes': {'source.py': d.digest(tmp_path/'source.py')},
                'config_sha256': d.digest(tmp_path/'config.yaml'),
                'amendment_sha256': d.digest(tmp_path/'decisions/dw_sampling_amendment.json'),
                'bound_files': {'pinned.txt': d.digest(tmp_path/'pinned.txt')}}
    changed = {'source': 'source.py', 'config': 'config.yaml', 'amendment': 'decisions/dw_sampling_amendment.json',
               'bound_file': 'pinned.txt', 'unrelated': 'unrelated.py'}
    if mutation != 'none':
        (tmp_path/changed[mutation]).write_text('changed')
    if mutation in ('unrelated', 'none'):
        d.validate_source_stability(identity, 'pilot', tmp_path/'config.yaml')
    else:
        with pytest.raises(d.MeasurementInvalid, match='changed during'):
            d.validate_source_stability(identity, 'pilot', tmp_path/'config.yaml')


@pytest.mark.parametrize('stage', ['prepare', 'correctness', 'pilot', 'confirm', 'feasibility', 'cost'])
def test_stage_source_maps_bind_only_declared_learning_dependencies(stage):
    d = driver()
    names = set(d.source_hashes(stage))
    expected = set(d.CORE)
    if stage != 'feasibility':
        expected.update(d.SAMPLED)
    if stage in ('prepare', 'correctness', 'pilot', 'confirm'):
        expected.update(('src/oar/sampling_lm.py', 'src/oar/sampling_data.py'))
    assert names == expected


@pytest.mark.parametrize('stage', ['feasibility', 'cost'])
def test_native_pass_continuity_to_exact_full_learning_map_without_old_learning_git_blobs(approved_carryover, stage):
    d, repository, out, old, make = approved_carryover
    # D was frozen with the current cost gate; B predates that policy field.
    if stage == 'cost':
        old['configs/dw_sampling.yaml'] = (repository/'configs/dw_sampling.yaml').read_bytes()
    attempt, entry, amendment = make(stage)
    entry['current_measurement_hashes'] = {**d.source_hashes('pilot'), **{name: d.digest(repository/name) for name in ('src/oar/sampling_lm.py', 'src/oar/sampling_data.py')}}
    amendment['bound_files'] = dict(entry['current_measurement_hashes'])
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    result = d.require_stage(out, stage, entry['current_measurement_hashes'])
    assert result['carryover']['current_measurement_hashes'] == entry['current_measurement_hashes']
    assert 'prepare' not in result['carryover']['measured_functions_unchanged']
    assert {'paired_ratio', 'freeze_delta'} <= set(result['carryover']['measured_functions_unchanged'])
    if stage == 'cost':
        assert {'offline_diagnostics', 'hardware_stage', 'forecast_stage'} <= set(result['carryover']['measured_functions_unchanged'])
        assert result['carryover']['native_config_exclusions'] == ['learning', 'asset', 'schema_version']


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'unbound_learning', 'stats', 'gate', 'diagnostic', 'forecast'])
def test_cost_continuity_cannot_approve_changed_statistics_gates_or_inexact_consumers(approved_carryover, mutation):
    d, repository, out, old, make = approved_carryover
    old['configs/dw_sampling.yaml'] = (repository/'configs/dw_sampling.yaml').read_bytes()
    attempt, entry, amendment = make('cost')
    if mutation == 'gate':
        path = repository/'configs/dw_sampling.yaml'
        cfg = yaml.safe_load(path.read_text())
        cfg['timing']['cost_gate_upper_ratio'] = .99
        path.write_text(yaml.safe_dump(cfg))
    if mutation in ('stats', 'diagnostic', 'forecast'):
        path = repository/'analysis/run_dw_sampling.py'
        name = {'stats': 'paired_ratio', 'diagnostic': 'offline_diagnostics', 'forecast': 'forecast_stage'}[mutation]
        source = path.read_text()
        start = source.index(f'def {name}(')
        end = source.index('\n', start)
        path.write_text(source[:end+1]+"    raise RuntimeError('changed protected body')\n"+source[end+1:])
    current = {**d.source_hashes('pilot'), **{name: d.digest(repository/name) for name in ('src/oar/sampling_lm.py', 'src/oar/sampling_data.py')}}
    amendment['bound_files'] = dict(current)
    if mutation == 'extra':
        current['arbitrary.py'] = 'f'*64
    elif mutation == 'missing':
        current.pop('src/oar/sampling_data.py')
    elif mutation == 'unbound_learning':
        amendment['bound_files'].pop('src/oar/sampling_lm.py')
    entry['current_measurement_hashes'] = current
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    with pytest.raises(ValueError):
        d.require_stage(out, 'cost', current)


@pytest.mark.parametrize('stage', ['feasibility', 'cost'])
def test_later_incomplete_attempt_supersedes_prior_approved_native_pass(approved_carryover, stage):
    d, repository, out, old, make = approved_carryover
    if stage == 'cost':
        old['configs/dw_sampling.yaml'] = (repository/'configs/dw_sampling.yaml').read_bytes()
    make(stage)
    latest = d.allocate_attempt(out, stage)
    (latest/'partial-worker.json').write_text('{"partial":true}')
    with pytest.raises(ValueError, match='incomplete prerequisite'):
        d.require_stage(out, stage, d.source_hashes('cost'))


@pytest.mark.parametrize('stage,function', [('feasibility', 'freeze_delta'), ('cost', 'freeze_delta'),
    ('cost', 'hardware_stage')])
def test_native_continuity_rejects_critical_statistic_or_gate_ast_changes(approved_carryover, stage, function):
    d, repository, out, old, make = approved_carryover
    if stage == 'cost':
        old['configs/dw_sampling.yaml'] = (repository/'configs/dw_sampling.yaml').read_bytes()
    attempt, entry, amendment = make(stage)
    path = repository/'analysis/run_dw_sampling.py'
    source = path.read_text()
    start = source.index('def '+function+'(')
    end = source.index('\n', start)
    path.write_text(source[:end+1]+"    raise RuntimeError('changed protected statistic/gate')\n"+source[end+1:])
    entry['current_measurement_hashes'] = d.source_hashes('pilot')
    amendment['bound_files'] = dict(entry['current_measurement_hashes'])
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    with pytest.raises(ValueError, match='AST changed'):
        d.require_stage(out, stage, entry['current_measurement_hashes'])


@pytest.mark.parametrize('mutation', ['source', 'amendment', 'unrelated'])
def test_main_seals_failure_with_started_identity_when_stage_sources_drift(tmp_path, monkeypatch, mutation):
    d = driver()
    repository = tmp_path/'repository'
    (repository/'decisions').mkdir(parents=True)
    (repository/'measured.py').write_text('start')
    config = repository/'config.yaml'
    config.write_text('frozen config')
    amendment_path = repository/'decisions/dw_sampling_amendment.json'
    amendment_path.write_text('{}')
    monkeypatch.setattr(d, 'ROOT', repository)
    monkeypatch.setattr(d, 'load_config', lambda path: {})
    monkeypatch.setattr(d, 'validate_amendment', lambda path: {'bound_files': {}})
    monkeypatch.setattr(d, 'storage_guard', lambda *args: {})
    monkeypatch.setattr(d, 'competing_processes', lambda: {'blockers': []})
    monkeypatch.setattr(d.subprocess, 'check_output', lambda args, **kwargs: 'a'*40 if args[1] == 'rev-parse' else '')
    monkeypatch.setattr(d, 'source_hashes', lambda stage: {'measured.py': d.digest(repository/'measured.py')})
    started = d.digest(repository/'measured.py')
    def qualifying(*args):
        changed = {'source': repository/'measured.py', 'amendment': amendment_path,
                   'unrelated': repository/'new-unrelated.py'}[mutation]
        changed.write_text('changed during attempt')
        return {'status': 'pass', 'reason': 'qualification succeeded'}
    monkeypatch.setattr(d, 'prepare', qualifying)
    out = tmp_path/'out'
    assert d.main(['--stage', 'prepare', '--config', str(config), '--out', str(out)]) == 0
    record = d.read_record(out/'prepare/attempt-0001/record.json')
    assert record['measurement_hashes'] == {'measured.py': started}
    assert record['status'] == ('pass' if mutation == 'unrelated' else 'INCONCLUSIVE_MEASUREMENT')
    if mutation != 'unrelated':
        assert record['invalidated_result'] == {'status': 'pass', 'reason': 'qualification succeeded'}
        assert 'source_stability_failure.txt' in record['artifacts']


@pytest.mark.parametrize('stage', ['prepare', 'feasibility', 'cost'])
def test_prepare_only_change_preserves_native_measurements_but_requires_prepare_rerun(approved_carryover, stage):
    d, repository, out, old, make = approved_carryover
    attempt, entry, amendment = make(stage)
    path = repository/'analysis/run_dw_sampling.py'
    source = path.read_text()
    start = source.index('def prepare(')
    end = source.index('\n', start)
    path.write_text(source[:end+1]+"    qualification_metadata = 'prospective deadline change'\n"+source[end+1:])
    entry['current_measurement_hashes'] = d.source_hashes('pilot')
    amendment['bound_files'] = dict(entry['current_measurement_hashes'])
    (repository/'decisions/dw_sampling_amendment.json').write_text(json.dumps(amendment))
    if stage == 'prepare':
        with pytest.raises(ValueError, match='AST changed'):
            d.require_stage(out, stage, entry['current_measurement_hashes'])
    else:
        result = d.require_stage(out, stage, entry['current_measurement_hashes'])
        assert result['status'] == 'pass' and result['measurement_hashes'] == entry['previous_measurement_hashes']


@pytest.mark.parametrize('mutation', ['extra', 'missing', 'wrong_hash'])
def test_unchanged_prerequisite_still_requires_complete_actual_consumer_map(approved_carryover, mutation):
    d, repository, out, old, make = approved_carryover
    current = d.source_hashes('cost')
    attempt = d.allocate_attempt(out, 'cost')
    d.seal_record(attempt, {'stage': 'cost', 'status': 'pass', 'measurement_hashes': current})
    consumer = dict(current)
    if mutation == 'extra':
        consumer['arbitrary.py'] = 'f'*64
    elif mutation == 'missing':
        consumer.pop('src/oar/sampled_linear.py')
    else:
        # A changed current value must be rejected even without a shared-key mismatch.
        consumer['src/oar/sampling_lm.py'] = 'f'*64
        consumer['src/oar/sampling_data.py'] = 'e'*64
    with pytest.raises(ValueError, match='actual consumer'):
        d.require_stage(out, 'cost', consumer)


# SPDX-License-Identifier: AGPL-3.0-only
"""Tiny CPU fixtures for the fixed byte model; contains no corpus text."""


def test_byte_model_isolated_declared_init_draw_order_and_exact_tie():
    import torch
    from oar.sampled_linear import named_seed
    from oar.sampling_lm import ByteLM, ModelConfig
    config = ModelConfig(blocks=2, width=8, heads=2, ffn_width=16)
    before = torch.random.get_rng_state().clone()
    model = ByteLM(config, base_seed=71)
    assert torch.equal(before, torch.random.get_rng_state())
    assert model.init_seed == named_seed(71, 'init')
    generator = torch.Generator(device='cpu').manual_seed(model.init_seed)
    ordered = [model.token_embedding.weight, model.position_embedding.weight]
    for block in model.blocks:
        ordered.extend([block.qkv.weight, block.attention_out.weight, block.ffn_in.weight, block.ffn_out.weight])
    for weight in ordered:
        expected = torch.empty(weight.shape).normal_(0, .02, generator=generator)
        assert torch.equal(weight, expected)
    assert torch.equal(generator.get_state(), model.init_generator.get_state())
    assert model.head.weight is model.token_embedding.weight
    for name, value in model.named_parameters():
        if name.endswith('bias'):
            assert torch.equal(value, torch.zeros_like(value))
        elif value.ndim == 1:
            assert torch.equal(value, torch.ones_like(value))
    model.to(dtype=torch.float64)
    assert model.head.weight is model.token_embedding.weight
    state = model.state_dict()
    model.load_state_dict(state, assign=True)
    assert model.head.weight is model.token_embedding.weight


def test_byte_model_defaults_and_invalid_config():
    import dataclasses
    import pytest
    from oar.sampling_lm import ModelConfig, ByteLM
    config = ModelConfig()
    assert dataclasses.asdict(config) == dict(blocks=4, width=512, heads=8, ffn_width=2048, context=256, vocab_size=259)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.width = 1
    for overrides in ({'width': 7, 'heads': 2}, {'context': 257}, {'blocks': 0}, {'vocab_size': 258}):
        with pytest.raises(ValueError):
            ModelConfig(**overrides)
    with pytest.raises(ValueError):
        ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=1, sampler='invalid')


def test_byte_model_dense_candidate_forward_and_unsampled_gradients_match():
    import torch
    from oar.sampling_lm import ByteLM, ModelConfig
    config = ModelConfig(blocks=2, width=8, heads=2, ffn_width=16)
    dense = ByteLM(config, base_seed=3)
    sampled = ByteLM(config, base_seed=3, sampler='importance')
    assert dense.state_dict().keys() == sampled.state_dict().keys()
    assert dense.sampling_states == {}
    for key, value in dense.state_dict().items():
        assert torch.equal(value, sampled.state_dict()[key])
    ids = torch.tensor([[256, 65, 66, 258], [256, 67, 258, 258]])
    a, b = dense(ids), sampled(ids)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    a.square().sum().backward()
    b.square().sum().backward()
    candidate_parameters = dict(sampled.named_parameters())
    for name, parameter in dense.named_parameters():
        if name.endswith(('ffn_in.weight', 'ffn_out.weight')):
            continue
        torch.testing.assert_close(parameter.grad, candidate_parameters[name].grad, rtol=1e-4, atol=1e-6)
    assert dense.blocks[0].ffn_in.__class__.forward is torch.nn.Linear.forward
    assert not hasattr(dense.blocks[0].ffn_in, 'sampling_state')
    assert not dense.blocks[0].ffn_in._forward_hooks


def test_byte_model_causal_padding_and_single_target_are_finite():
    import torch
    from oar.sampling_lm import ByteLM, ModelConfig
    model = ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=9)
    original = torch.tensor([[256, 11, 12, 13]])
    changed = torch.tensor([[256, 11, 88, 89]])
    torch.testing.assert_close(model(original)[:, :2], model(changed)[:, :2], rtol=1e-4, atol=1e-6)
    padded = torch.full((2, 256), 258, dtype=torch.long)
    padded[0, :4] = original[0]
    padded[1, 0] = 256
    logits = model(padded)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[0, :4], model(original)[0], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(logits[1, :1], model(torch.tensor([[256]]))[0], rtol=1e-4, atol=1e-6)


def test_byte_model_sampling_physical_rows_eval_bypass_and_state_restore(monkeypatch):
    import torch
    import oar.sampling_lm as lm
    model = lm.ByteLM(lm.ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=8, sampler='uniform')
    assert set(model.sampling_states) == {'block0.ffn_in', 'block0.ffn_out'}
    assert set(model.resolved_seeds) == {'init', 'sampling'}
    assert set(model.resolved_seeds['sampling']) == set(model.sampling_states)
    before = model.sampling_state_dict()
    observed = []
    real = lm.sampled_linear
    def record(x, weight, bias, state):
        observed.append((state.layer_id, x.numel() // weight.shape[1]))
        return real(x, weight, bias, state)
    monkeypatch.setattr(lm, 'sampled_linear', record)
    ids = torch.full((2, 256), 258, dtype=torch.long)
    ids[:, 0] = 256
    model(ids).square().sum().backward()
    assert observed == [('block0.ffn_in', 512), ('block0.ffn_out', 512)]
    after = model.sampling_state_dict()
    assert any(not torch.equal(before[k]['generator_state'], after[k]['generator_state']) for k in before)
    model.load_sampling_state_dict(before)
    for key in before:
        assert torch.equal(before[key]['generator_state'], model.sampling_state_dict()[key]['generator_state'])
    observed.clear()
    with torch.no_grad():
        model(ids)
    assert observed == []
    for block in model.blocks:
        block.ffn_in.weight.requires_grad_(False)
        block.ffn_out.weight.requires_grad_(False)
    model(ids)
    assert observed == []


def test_byte_model_native_optimizer_unique_groups_and_static_owner_checks():
    import torch
    import pytest
    from oar.sampling_lm import ByteLM, ModelConfig, optimizer_groups, make_optimizer
    model = ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=2, sampler='uniform')
    groups = optimizer_groups(model)
    values = [p for g in groups for p in g['params']]
    assert len(values) == len({id(p) for p in values}) == len(list(model.parameters()))
    assert {id(p) for p in values} == {id(p) for p in model.parameters()}
    for group in groups:
        assert group['weight_decay'] == (.01 if group['params'][0].ndim >= 2 else 0.)
        assert all((p.ndim >= 2) == (group['weight_decay'] == .01) for p in group['params'])
    for variant in ('default', 'single', 'foreach', 'fused'):
        optimizer = make_optimizer(model, variant=variant, lr=.003)
        assert type(optimizer) is torch.optim.AdamW
        assert optimizer.defaults['lr'] == .003
        assert optimizer.defaults['betas'] == (.9, .999)
        assert optimizer.defaults['eps'] == 1e-8
        assert optimizer.defaults['foreach'] == {'default': None, 'single': False, 'foreach': True, 'fused': False}[variant]
        assert optimizer.defaults['fused'] == {'default': None, 'single': False, 'foreach': False, 'fused': True}[variant]
    model.shared_alias = model.blocks[0].ffn_in.weight
    with pytest.raises(ValueError, match='owners|aliases'):
        make_optimizer(model)


def test_byte_model_evaluation_global_token_mean_order_rng_and_mode():
    import math
    import torch
    import torch.nn.functional as F
    from oar.sampling_data import normalize_articles, encode_articles, evaluation_windows, collate_blocks
    from oar.sampling_lm import ByteLM, ModelConfig, evaluate
    articles = normalize_articles(['= A =\nx\n= B =\n' + 'y' * 150 + '\n= C =\nz'], 'fixture')
    tokens = encode_articles(articles)
    windows = evaluation_windows(articles)
    model = ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=5, sampler='uniform')
    prior_rng = model.sampling_state_dict()
    model.train()
    expected_sum, expected_count = 0., 0
    expected_seen = []
    with torch.no_grad():
        for start in range(0, len(windows), 2):
            batch = collate_blocks(tokens, windows[start:start+2])
            expected_seen.append(batch['input_ids'].clone())
            logits = model(batch['input_ids'])
            losses = F.cross_entropy(logits.reshape(-1, 259), batch['target_ids'].reshape(-1), reduction='none', ignore_index=-100).reshape(batch['target_ids'].shape)
            expected_sum += losses.cpu().double()[batch['target_mask_cpu']].sum().item()
            expected_count += batch['valid_targets']
    seen = []
    def observe(module, args):
        assert not module.training and not torch.is_grad_enabled()
        seen.append(args[0].clone())
    handle = model.register_forward_pre_hook(observe)
    result = evaluate(model, tokens, windows, batch_size=2)
    handle.remove()
    assert model.training
    assert len(seen) == len(expected_seen)
    for a, b in zip(seen, expected_seen):
        assert torch.equal(a, b)
    assert seen[-1].shape[0] == len(windows) % 2 or len(windows) % 2 == 0
    assert result['valid_targets'] == expected_count
    assert result['window_count'] == len(windows)
    assert result['nll'] == expected_sum / expected_count
    assert result['perplexity'] == math.exp(result['nll'])
    for key, value in prior_rng.items():
        assert torch.equal(value['generator_state'], model.sampling_state_dict()[key]['generator_state'])
    model.eval()
    evaluate(model, tokens, windows, batch_size=2)
    assert not model.training


def test_byte_model_initialization_uses_cpu_fp32_under_nondefault_dtype():
    import torch
    from oar.sampling_lm import ByteLM, ModelConfig
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=3)
        assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in model.parameters())
    finally:
        torch.set_default_dtype(previous)


def test_byte_model_evaluation_disables_autocast_and_restores_mode_after_failure():
    import pytest
    import torch
    from oar.sampling_data import normalize_articles, encode_articles, evaluation_windows
    from oar.sampling_lm import ByteLM, ModelConfig, evaluate
    articles = normalize_articles(['= A =\nx'], 'fixture')
    tokens = encode_articles(articles)
    windows = evaluation_windows(articles)
    model = ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=8, sampler='importance')
    reference = evaluate(model, tokens, windows)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        result = evaluate(model, tokens, windows)
    assert result['nll'] == reference['nll']
    def fail(module, args):
        raise RuntimeError('fixture failure')
    handle = model.register_forward_pre_hook(fail)
    model.train()
    with pytest.raises(RuntimeError, match='fixture failure'):
        evaluate(model, tokens, windows)
    handle.remove()
    assert model.training


def test_byte_model_four_blocks_have_eight_independent_stable_sampler_owners():
    import torch
    from oar.sampled_linear import named_seed
    from oar.sampling_lm import ByteLM, ModelConfig, optimizer_groups
    prior = torch.random.get_rng_state().clone()
    model = ByteLM(ModelConfig(blocks=4, width=8, heads=2, ffn_width=16), base_seed=12, sampler='importance')
    assert torch.equal(prior, torch.random.get_rng_state())
    expected = [f'block{i}.{suffix}' for i in range(4) for suffix in ('ffn_in', 'ffn_out')]
    assert list(model.sampling_states) == expected
    modules = dict(model.named_modules())
    generators = []
    for name in expected:
        index, suffix = name.split('.')
        module = modules[f'blocks.{index[5:]}.{suffix}']
        state = model.sampling_states[name]
        assert module.sampling_state is state
        assert state.seed == named_seed(12, 'sampling', name, 'importance')
        generators.append(id(state.generator))
    assert len(set(generators)) == 8
    optimizer_groups(model)


def test_byte_model_evaluation_rejects_nonfinite_scored_losses_and_restores_mode():
    import pytest
    import torch
    from oar.sampling_data import normalize_articles, encode_articles, evaluation_windows
    from oar.sampling_lm import ByteLM, ModelConfig, evaluate
    articles = normalize_articles(['= A =\nx'], 'fixture')
    model = ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=15, sampler='importance')
    before = model.sampling_state_dict()
    with torch.no_grad():
        model.final_ln.weight.fill_(float('nan'))
    model.train()
    with pytest.raises(RuntimeError, match='nonfinite.*evaluation|evaluation.*nonfinite'):
        evaluate(model, encode_articles(articles), evaluation_windows(articles))
    assert model.training
    for key, state in before.items():
        assert torch.equal(state['generator_state'], model.sampling_state_dict()[key]['generator_state'])


def test_byte_model_evaluation_rejects_overflowing_perplexity_with_clear_failure():
    import pytest
    import torch
    from oar.sampling_data import normalize_articles, encode_articles, evaluation_windows
    from oar.sampling_lm import ByteLM, ModelConfig, evaluate
    articles = normalize_articles(['x'], 'fixture')
    model = ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=15)
    def large_finite_logits(module, args, output):
        result = torch.zeros_like(output)
        result[..., 0] = 1000.
        return result
    handle = model.register_forward_hook(large_finite_logits)
    with pytest.raises(RuntimeError, match='evaluation.*finite|perplexity'):
        evaluate(model, encode_articles(articles), evaluation_windows(articles))
    handle.remove()
    assert model.training


# SPDX-License-Identifier: AGPL-3.0-only
"""Tiny CPU training and restart fixtures; no external data or measurements."""


def _byte_recovery_fixture(sampler='uniform', seed=29, batch_size=2, variant='single'):
    from oar.sampling_data import normalize_articles, encode_articles, training_chunks
    from oar.sampling_lm import ByteLM, ModelConfig, BatchStream, TrainingState, make_optimizer
    articles = normalize_articles(['= A =\nx\n= B =\n' + 'y' * 40 + '\n= C =\nz\n= D =\nw\n= E =\nv'], 'fixture')
    tokens = encode_articles(articles)
    blocks = training_chunks(articles)
    model = ByteLM(ModelConfig(blocks=1, width=8, heads=2, ffn_width=16), base_seed=seed, sampler=sampler)
    stream = BatchStream(tokens, blocks, batch_size=batch_size, base_seed=seed)
    return TrainingState(model, make_optimizer(model, variant=variant), stream)


def _assert_byte_recovery_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_byte_recovery_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_byte_recovery_equal(x, y)
    else:
        assert a == b


def _byte_recovery_payload(state):
    return {'schema_version': 1, 'bindings': {'source': 'a' * 64, 'asset': 'b' * 64},
            'training_state': state.snapshot(), 'clock': {'launch_perf_s': 1., 'prior_elapsed_s': 0., 'elapsed_s': 2.},
            't0_s': 1.25, 'scheduler_rule': {'kind': 'frozen_fixture', 'lr': .001}}


def test_byte_recovery_stream_randperm_final_actual_batch_and_lazy_next_epoch():
    import torch
    from oar.sampled_linear import named_seed
    from oar.sampling_data import collate_blocks
    state = _byte_recovery_fixture()
    stream = state.stream
    expected_generator = torch.Generator(device='cpu').manual_seed(named_seed(29, 'data'))
    assert torch.equal(stream.state_dict()['generator_state'], expected_generator.get_state())
    expected_order = torch.randperm(len(stream.blocks), generator=expected_generator)
    for start in range(0, len(stream.blocks), 2):
        selected = [stream.blocks[i] for i in expected_order[start:start+2].tolist()]
        expected = collate_blocks(stream.tokenized_articles, selected)
        actual = stream.next_batch()
        _assert_byte_recovery_equal(actual, expected)
    saved = stream.state_dict()
    assert saved['cursor'] == len(stream.blocks) and saved['epoch'] == 1
    assert torch.equal(saved['generator_state'], expected_generator.get_state())
    next_order = torch.randperm(len(stream.blocks), generator=expected_generator)
    stream.next_batch()
    assert torch.equal(stream.state_dict()['order'], next_order)
    assert stream.state_dict()['epoch'] == 2


def test_byte_recovery_stream_metadata_and_cursor_fail_before_mutation():
    import copy
    import pytest
    from oar.sampling_data import TargetBlock
    from oar.sampling_lm import BatchStream
    state = _byte_recovery_fixture()
    state.stream.next_batch()
    saved = state.stream.state_dict()
    other = BatchStream(state.stream.tokenized_articles, state.stream.blocks, batch_size=1, base_seed=29)
    with pytest.raises(ValueError, match='metadata|binding|batch'):
        other.load_state_dict(saved)
    mutated_tokens = list(state.stream.tokenized_articles)
    mutated_tokens[0] = (256, 99, 257)
    other = BatchStream(mutated_tokens, [TargetBlock(0, 1, 2, 0)], batch_size=2, base_seed=29)
    with pytest.raises(ValueError, match='metadata|binding|manifest'):
        other.load_state_dict(saved)
    before = state.stream.state_dict()
    broken = copy.deepcopy(saved)
    broken['cursor'] = len(state.stream.blocks) + 1
    with pytest.raises(ValueError, match='cursor'):
        state.stream.load_state_dict(broken)
    _assert_byte_recovery_equal(before, state.stream.state_dict())


def test_byte_recovery_charged_clock_accounts_restart_setup_and_excludes_gap():
    import math
    import pytest
    import time
    from oar.sampling_lm import ChargedClock
    now = [12.]
    first = ChargedClock(10., clock=lambda: now[0])
    assert first.elapsed() == 2.
    now[0] = 13.5
    first_state = first.state_dict()
    assert first_state == {'launch_perf_s': 10., 'prior_elapsed_s': 0., 'elapsed_s': 3.5}
    now[0] = 101.
    resumed = ChargedClock(100., prior_elapsed_s=first_state['elapsed_s'], clock=lambda: now[0])
    assert resumed.elapsed() == 4.5
    now[0] = 102.5
    assert resumed.elapsed() == 6.
    for launch, prior in ((math.nan, 0), (1, math.inf), (2, -1), (103, 0)):
        with pytest.raises(ValueError, match='clock|finite|nonnegative|domain'):
            ChargedClock(launch, prior_elapsed_s=prior, clock=lambda: now[0])
    now[0] = 101.
    with pytest.raises(ValueError, match='monotonic|domain|clock'):
        resumed.elapsed()
    real_launch = time.perf_counter()
    real_first = ChargedClock(real_launch)
    sum(range(1000))
    prior = real_first.elapsed()
    restart_launch = time.perf_counter()
    sum(range(1000))
    real_resumed = ChargedClock(restart_launch, prior_elapsed_s=prior)
    charged = real_resumed.elapsed()
    assert prior <= charged <= prior + time.perf_counter() - restart_launch


def test_byte_recovery_step_matches_global_valid_token_objective_and_native_update():
    import torch
    import torch.nn.functional as F
    from oar.sampling_lm import ByteLM, make_optimizer
    state = _byte_recovery_fixture(sampler=None)
    reference = ByteLM(state.model.config, base_seed=29)
    optimizer = make_optimizer(reference, lr=.002)
    stream_before = state.stream.state_dict()
    batch = state.stream.next_batch()
    state.stream.load_state_dict(stream_before)
    assert batch['valid_targets'] != batch['physical_rows']
    optimizer.zero_grad(set_to_none=True)
    logits = reference(batch['input_ids'])
    loss = F.cross_entropy(logits.reshape(-1, 259), batch['target_ids'].reshape(-1), reduction='sum', ignore_index=-100) / batch['valid_targets']
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    actual = state.step(.002)
    torch.testing.assert_close(actual['loss'], loss.detach(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(actual['norm'], norm.detach(), rtol=1e-4, atol=1e-6)
    assert {key: actual[key] for key in ('valid_targets', 'physical_rows', 'logical_update')} == {
        'valid_targets': batch['valid_targets'], 'physical_rows': batch['physical_rows'], 'logical_update': 1}
    _assert_byte_recovery_equal(state.model.state_dict(), reference.state_dict())
    _assert_byte_recovery_equal(state.optimizer.state_dict(), optimizer.state_dict())
    assert (state.logical_updates, state.valid_tokens, state.physical_tokens) == (1, batch['valid_targets'], batch['physical_rows'])


def test_byte_recovery_interrupted_resume_replays_data_draws_model_and_adam():
    for sampler in (None, 'uniform', 'importance'):
        state = _byte_recovery_fixture(sampler=sampler)
        lrs = [.001, .0008, .0007, .0006, .0004, .0002]
        for lr in lrs[:2]:
            state.step(lr)
        saved = state.snapshot()
        expected_records = [state.step(lr) for lr in lrs[2:]]
        expected = state.snapshot()
        resumed = _byte_recovery_fixture(sampler=sampler)
        resumed.step(.005)
        resumed.restore(saved)
        actual_records = [resumed.step(lr) for lr in lrs[2:]]
        assert actual_records == expected_records
        _assert_byte_recovery_equal(resumed.snapshot(), expected)
        assert resumed.model.head.weight is resumed.model.token_embedding.weight
        assert resumed.logical_updates == 6


def test_byte_recovery_snapshot_is_unaliased_and_metadata_checked_first(monkeypatch):
    import copy
    import pytest
    import torch
    state = _byte_recovery_fixture()
    monkeypatch.setattr(torch.mps, 'get_rng_state', lambda: (_ for _ in ()).throw(AssertionError('CPU must not access MPS RNG')))
    state.step(.001)
    saved = state.snapshot()
    pristine = copy.deepcopy(saved)
    with torch.no_grad():
        state.model.token_embedding.weight.add_(1.)
        next(iter(state.optimizer.state.values()))['exp_avg'].add_(1.)
    state.stream.next_batch()
    _assert_byte_recovery_equal(saved, pristine)
    state.restore(saved)
    _assert_byte_recovery_equal(state.snapshot(), pristine)
    different = _byte_recovery_fixture(seed=30)
    before = different.snapshot()
    with pytest.raises(ValueError, match='metadata|binding'):
        different.restore(saved)
    _assert_byte_recovery_equal(different.snapshot(), before)
    invalid = copy.deepcopy(saved)
    invalid['stream']['cursor'] = 999
    before = state.snapshot()
    with pytest.raises(ValueError, match='cursor'):
        state.restore(invalid)
    _assert_byte_recovery_equal(state.snapshot(), before)


def test_byte_recovery_failed_loss_gradient_and_native_step_do_not_commit(monkeypatch):
    import pytest
    import torch
    for cause in ('loss', 'gradient', 'optimizer'):
        state = _byte_recovery_fixture(sampler=None)
        before = state.snapshot()
        if cause == 'loss':
            with torch.no_grad():
                state.model.final_ln.weight.fill_(float('nan'))
        elif cause == 'gradient':
            state.model.final_ln.weight.register_hook(lambda grad: grad * float('nan'))
        else:
            def broken_step(*args, **kwargs):
                with torch.no_grad():
                    state.model.token_embedding.weight.add_(1.)
                raise RuntimeError('native fixture failure')
            monkeypatch.setattr(state.optimizer, 'step', broken_step)
        with pytest.raises(RuntimeError, match='finite|native fixture'):
            state.step(.001)
        assert (state.logical_updates, state.valid_tokens, state.physical_tokens) == (0, 0, 0)
        with pytest.raises(RuntimeError, match='failed|uncommitted'):
            state.snapshot()
        state.restore(before)
        assert state.logical_updates == 0
        _assert_byte_recovery_equal(state.snapshot(), before)


def test_byte_recovery_zero_gradient_step_succeeds_but_zero_targets_fails(monkeypatch):
    import pytest
    state = _byte_recovery_fixture(sampler=None)
    handle = state.model.register_forward_hook(lambda module, args, output: output * 0.)
    result = state.step(.001)
    handle.remove()
    assert result['norm'] == 0. and state.logical_updates == 1
    good = state.snapshot()
    actual_next = state.stream.next_batch
    def empty_targets(*args, **kwargs):
        batch = actual_next(*args, **kwargs)
        batch['valid_targets'] = 0
        return batch
    monkeypatch.setattr(state.stream, 'next_batch', empty_targets)
    with pytest.raises(ValueError, match='valid target'):
        state.step(.001)
    assert state.logical_updates == 1
    state.restore(good)


def test_byte_recovery_atomic_checkpoint_load_checks_hash_schema_bindings_and_t0(tmp_path, monkeypatch):
    import hashlib
    import pytest
    import torch
    import oar.sampling_lm as lm
    state = _byte_recovery_fixture()
    state.step(.001)
    payload = _byte_recovery_payload(state)
    path = tmp_path / 'committed-0001.pt'
    info = lm.save_checkpoint(path, payload)
    assert info['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info['size_bytes'] == path.stat().st_size and info['wall_s'] >= 0
    real_load = torch.load
    calls = []
    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', checked_load)
    loaded = lm.load_checkpoint(path, info['sha256'], expected_bindings=payload['bindings'])
    assert calls == [{'weights_only': True, 'map_location': 'cpu'}]
    _assert_byte_recovery_equal(loaded, payload)
    assert loaded['t0_s'] == 1.25 and loaded['scheduler_rule'] == payload['scheduler_rule']
    with pytest.raises(ValueError, match='binding'):
        lm.load_checkpoint(path, info['sha256'], expected_bindings={'source': 'wrong'})
    with pytest.raises(FileExistsError):
        lm.save_checkpoint(path, payload)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
    wrong_schema = dict(payload, schema_version=999)
    invalid = tmp_path / 'schema.pt'
    torch.save(wrong_schema, invalid)
    with pytest.raises(ValueError, match='schema'):
        lm.load_checkpoint(invalid, hashlib.sha256(invalid.read_bytes()).hexdigest(), expected_bindings=payload['bindings'])
    path.write_bytes(path.read_bytes() + b'tampered')
    call_count = len(calls)
    with pytest.raises(ValueError, match='SHA|hash|integrity'):
        lm.load_checkpoint(path, info['sha256'], expected_bindings=payload['bindings'])
    assert len(calls) == call_count


def test_byte_recovery_failed_write_or_rename_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import hashlib
    import pytest
    import oar.sampling_lm as lm
    state = _byte_recovery_fixture()
    payload = _byte_recovery_payload(state)
    prior = tmp_path / 'committed-0000.pt'
    info = lm.save_checkpoint(prior, payload)
    for cause in ('write', 'rename'):
        with monkeypatch.context() as patch:
            if cause == 'write':
                def broken_save(obj, file, *args, **kwargs):
                    file.write(b'incomplete fixture')
                    raise OSError('fixture write failure')
                patch.setattr(lm.torch, 'save', broken_save)
            else:
                def broken_rename(*args, **kwargs):
                    raise OSError('fixture rename failure')
                patch.setattr(lm.os, 'rename', broken_rename)
            with pytest.raises(OSError, match='fixture'):
                lm.save_checkpoint(tmp_path / f'failed-{cause}.pt', payload)
        assert sorted(p.name for p in tmp_path.iterdir()) == ['committed-0000.pt']
        assert hashlib.sha256(prior.read_bytes()).hexdigest() == info['sha256']


def test_byte_recovery_step_rejects_autocast_before_drawing_data():
    import pytest
    import torch
    state = _byte_recovery_fixture(sampler=None)
    before = state.snapshot()
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match='autocast|AMP'):
            state.step(.001)
    _assert_byte_recovery_equal(state.snapshot(), before)


def test_byte_recovery_restore_binds_optimizer_parameter_order_and_static_options():
    import copy
    import pytest
    from oar.sampling_lm import TrainingState
    state = _byte_recovery_fixture(sampler=None)
    state.step(.001)
    saved = state.snapshot()
    other = _byte_recovery_fixture(sampler=None)
    other.optimizer.param_groups[0]['params'].reverse()
    reordered = TrainingState(other.model, other.optimizer, other.stream)
    with pytest.raises(ValueError, match='binding|metadata'):
        reordered.restore(saved)
    altered = copy.deepcopy(saved)
    altered['optimizer']['param_groups'][0]['weight_decay'] = .123
    before = state.snapshot()
    with pytest.raises(ValueError, match='optimizer.*metadata|optimizer.*static'):
        state.restore(altered)
    _assert_byte_recovery_equal(state.snapshot(), before)


def test_byte_recovery_clock_adds_delta_before_prior_to_avoid_spurious_overflow():
    from oar.sampling_lm import ChargedClock
    clock = ChargedClock(1e308, prior_elapsed_s=1e308, clock=lambda: 1e308)
    assert clock.elapsed() == 1e308


def test_byte_recovery_checkpoint_rejects_nonprimitive_payload_and_respects_existing_lock(tmp_path):
    import pytest
    from oar.sampling_lm import save_checkpoint
    state = _byte_recovery_fixture()
    payload = _byte_recovery_payload(state)
    path = tmp_path / 'new.pt'
    invalid = dict(payload, forbidden_object=object())
    with pytest.raises(ValueError, match='unsupported checkpoint'):
        save_checkpoint(path, invalid)
    assert list(tmp_path.iterdir()) == []
    lock = tmp_path / '.new.pt.lock'
    lock.write_bytes(b'other fixture writer')
    with pytest.raises(FileExistsError):
        save_checkpoint(path, payload)
    assert lock.read_bytes() == b'other fixture writer'
    assert not path.exists()


def test_byte_recovery_restore_rejects_changed_native_optimizer_parameter_ids():
    import copy
    import pytest
    state = _byte_recovery_fixture(sampler=None)
    state.step(.001)
    saved = state.snapshot()
    altered = copy.deepcopy(saved)
    altered['optimizer']['param_groups'][0]['params'].reverse()
    before = state.snapshot()
    with pytest.raises(ValueError, match='optimizer.*parameter'):
        state.restore(altered)
    _assert_byte_recovery_equal(state.snapshot(), before)


def test_byte_recovery_checkpoint_fsync_failure_preserves_prior_file(tmp_path, monkeypatch):
    import hashlib
    import pytest
    import oar.sampling_lm as lm
    payload = _byte_recovery_payload(_byte_recovery_fixture())
    prior = tmp_path / 'prior.pt'
    info = lm.save_checkpoint(prior, payload)
    def fail_sync(*args, **kwargs):
        raise OSError('fixture fsync failure')
    monkeypatch.setattr(lm.os, 'fsync', fail_sync)
    with pytest.raises(OSError, match='fixture fsync'):
        lm.save_checkpoint(tmp_path / 'later.pt', payload)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['prior.pt']
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == info['sha256']


def test_byte_recovery_step_defers_scalar_reads_until_controller_block_boundary(monkeypatch):
    import torch
    state = _byte_recovery_fixture(sampler=None)
    native_step = state.optimizer.step
    real_item = torch.Tensor.item
    after_optimizer = [False]
    def observed_optimizer_step(*args, **kwargs):
        result = native_step(*args, **kwargs)
        after_optimizer[0] = True
        return result
    def guarded_item(tensor, *args, **kwargs):
        assert not after_optimizer[0], 'post-optimizer scalar extraction must wait for block boundary'
        return real_item(tensor, *args, **kwargs)
    monkeypatch.setattr(state.optimizer, 'step', observed_optimizer_step)
    monkeypatch.setattr(torch.Tensor, 'item', guarded_item)
    result = state.step(.001)
    assert after_optimizer[0]
    for key in ('loss', 'norm'):
        assert isinstance(result[key], torch.Tensor)
        assert result[key].ndim == 0
        assert not result[key].requires_grad and result[key].grad_fn is None
    assert type(result['logical_update']) is int
    assert type(result['valid_targets']) is int
    assert type(result['physical_rows']) is int
    # The real controller synchronizes its device once per completed block here.
    # CPU needs no synchronization, but scalar logging starts at the same boundary.
    after_optimizer[0] = False
    logged = {key: result[key].item() for key in ('loss', 'norm')}
    assert logged['loss'] > 0 and logged['norm'] >= 0


def test_byte_recovery_restore_rejects_bad_moment_semantics_before_live_mutation():
    import copy
    import pytest
    import torch
    source = _byte_recovery_fixture(sampler=None)
    source.step(.001)
    pristine = source.snapshot()
    damages = ('negative_second', 'nan_second', 'inf_second', 'negative_step', 'nan_step',
               'inf_step', 'fractional_step', 'step_vector', 'integer_step_dtype',
               'wrong_moment_dtype', 'extra_moment_field', 'missing_second', 'extra_parameter_state')
    for damage in damages:
        saved = copy.deepcopy(pristine)
        pid = next(iter(saved['optimizer']['state']))
        moment = saved['optimizer']['state'][pid]
        if damage == 'negative_second':
            moment['exp_avg_sq'].fill_(-1.)
        elif damage in ('nan_second', 'inf_second'):
            moment['exp_avg_sq'].fill_(float('nan') if damage == 'nan_second' else float('inf'))
        elif damage in ('negative_step', 'nan_step', 'inf_step', 'fractional_step'):
            moment['step'].fill_({'negative_step': -1., 'nan_step': float('nan'), 'inf_step': float('inf'), 'fractional_step': 1.5}[damage])
        elif damage == 'step_vector':
            moment['step'] = moment['step'].reshape(1)
        elif damage == 'integer_step_dtype':
            moment['step'] = moment['step'].long()
        elif damage == 'wrong_moment_dtype':
            moment['exp_avg'] = moment['exp_avg'].double()
        elif damage == 'extra_moment_field':
            moment['unknown_state'] = torch.tensor(1.)
        elif damage == 'missing_second':
            del moment['exp_avg_sq']
        elif damage == 'extra_parameter_state':
            saved['optimizer']['state'][9999] = copy.deepcopy(moment)
        target = _byte_recovery_fixture(sampler=None)
        before = target.snapshot()
        with pytest.raises(ValueError, match='optimizer|moment|step'):
            target.restore(saved)
        _assert_byte_recovery_equal(target.snapshot(), before)


def test_byte_recovery_restore_rejects_nonfinite_model_and_nonempty_zero_update_adam():
    import copy
    import pytest
    source = _byte_recovery_fixture(sampler=None)
    source.step(.001)
    pristine = source.snapshot()
    for damage in ('nan_model', 'inf_model', 'moments_at_zero'):
        saved = copy.deepcopy(pristine)
        if damage in ('nan_model', 'inf_model'):
            saved['model']['final_ln.weight'].fill_(float('nan') if damage == 'nan_model' else float('inf'))
        else:
            saved['logical_updates'] = saved['valid_tokens'] = saved['physical_tokens'] = 0
        target = _byte_recovery_fixture(sampler=None)
        before = target.snapshot()
        with pytest.raises(ValueError, match='finite|optimizer|moment|step'):
            target.restore(saved)
        _assert_byte_recovery_equal(target.snapshot(), before)


def test_byte_recovery_engine_rejects_frozen_model_parameters():
    import pytest
    from oar.sampling_lm import TrainingState
    state = _byte_recovery_fixture(sampler=None)
    state.model.blocks[0].ffn_in.weight.requires_grad_(False)
    with pytest.raises(ValueError, match='trainable|frozen'):
        TrainingState(state.model, state.optimizer, state.stream)
    with pytest.raises(ValueError, match='trainable|frozen'):
        state.step(.001)


def test_byte_recovery_native_adam_variants_keep_step_policy_on_cpu_resume():
    import torch
    for variant in ('default', 'single', 'foreach', 'fused'):
        state = _byte_recovery_fixture(sampler='uniform', variant=variant)
        state.step(.001)
        saved = state.snapshot()
        if variant == 'fused':
            assert all(value['step'].dtype == torch.float32 for value in saved['optimizer']['state'].values())
        resumed = _byte_recovery_fixture(sampler='uniform', variant=variant)
        resumed.restore(saved)
        expected, actual = state.step(.0007), resumed.step(.0007)
        _assert_byte_recovery_equal(expected, actual)
        _assert_byte_recovery_equal(state.snapshot(), resumed.snapshot())


def test_byte_recovery_nonfused_native_float64_cpu_step_counter_remains_valid():
    import torch
    state = _byte_recovery_fixture(sampler=None)
    prior_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        state.step(.001)
    finally:
        torch.set_default_dtype(prior_dtype)
    saved = state.snapshot()
    assert all(value['step'].dtype == torch.float64 for value in saved['optimizer']['state'].values())
    resumed = _byte_recovery_fixture(sampler=None)
    resumed.restore(saved)
    _assert_byte_recovery_equal(state.step(.0005), resumed.step(.0005))
    _assert_byte_recovery_equal(state.snapshot(), resumed.snapshot())


def _learning_driver():
    return driver()


def test_learning_time_schedule_preserves_t0_and_zero_first_lr():
    d = _learning_driver()
    assert d.learning_lr(20., t0=20., cutoff=120., peak=.001) == 0.
    assert d.learning_lr(25., t0=20., cutoff=120., peak=.001) == pytest.approx(.001)
    assert d.learning_lr(120., t0=20., cutoff=120., peak=.001) == pytest.approx(.0001)
    assert d.learning_lr(200., t0=20., cutoff=120., peak=.001) == pytest.approx(.0001)
    with pytest.raises(ValueError): d.learning_lr(10., t0=20., cutoff=20., peak=.001)


def test_learning_horizon_uses_baseline_cost_not_inherited_noise_cap():
    d = _learning_driver()
    r = d.calibrate_horizon(startup_s=10.,monitor_s=.2,checkpoint_s=.5,block_s=2.,update_s=.25,
                            endpoint_s=12.,endpoint_checkpoint_s=.6,aa_jitter=.3,block_updates=8)
    assert r['status'] == 'pass'
    assert r['monitor_count'] == 50
    assert r['H'] == pytest.approx(10+4096*.25+49*.2+9*.5)
    assert r['R'] == pytest.approx(25.2)
    assert r['diagnostic_delta'] >= .3
    assert r['overshoot_limit_s'] == pytest.approx(.01*r['H'])
    assert r['T_budget'] == pytest.approx(r['H']+r['R'])


def test_learning_target_is_second_observed_completion_not_interpolation():
    d = _learning_driver()
    rows=[{'nll':2.,'completed_s':10.},{'nll':1.8,'completed_s':20.},
          {'nll':2.1,'completed_s':30.},{'nll':1.7,'completed_s':40.},
          {'nll':1.6,'completed_s':52.}]
    assert d.confirmed_target_time(rows,1.8,cutoff=100.)==52.
    assert d.confirmed_target_time(rows,1.8,cutoff=50.) is None
    assert d.confirmed_target_time(rows,1.5,cutoff=100.) is None


def _learning_grid_record(name,batch,lr,end,monitor,*,sampler=None,rho=None):
    return {'id':name,'status':'pass','arm':{'batch':batch,'lr':lr,'sampler':sampler,'rho':rho},
            'initial_full':{'nll':4.},'initial_monitor':{'nll':4.,'completed_s':5.},
            'endpoint_full':{'nll':end},
            'monitors':[{'nll':q,'completed_s':t} for t,q in monitor]}


def test_learning_grid_uses_quality_guard_and_simpler_time_ties():
    d = _learning_driver()
    dense=_learning_grid_record('reference',16,.001,1.5,[(40,2.5),(60,2.),(70,1.9),(80,1.8)])
    smaller=_learning_grid_record('small',4,.0003,1.505,[(30,1.9),(40,1.8),(60,1.7),(70,1.6)])
    imp=_learning_grid_record('imp',16,.001,1.507,[(20,1.9),(30,1.8),(60,1.7),(70,1.6)],sampler='importance',rho=.125)
    uniform=_learning_grid_record('uniform',16,.0003,1.507,[(21,1.9),(31,1.8),(60,1.7),(70,1.6)],sampler='uniform',rho=.5)
    bad=_learning_grid_record('badquality',16,.001,2.,[(10,1.9),(20,1.8),(60,1.7),(70,1.6)],sampler='uniform',rho=.125)
    r=d.select_learning_arms([dense,smaller,imp,uniform,bad],cutoff=100.,epsilon_eval=1e-6,r_time=2.)
    assert r['status']=='pass' and r['R0']['id']=='reference'
    assert r['q_star']==2. and r['epsilon_q']==pytest.approx(math.log(1.01))
    assert r['dense']['id']=='small' and r['candidate']['id']=='uniform'
    assert 'badquality' not in [v['id'] for v in r['eligible_candidates']]


def test_learning_target_calibration_needs_two_later_crossings():
    d = _learning_driver()
    ref=_learning_grid_record('reference',16,.001,1.5,[(60,2.),(70,1.9),(80,2.1)])
    r=d.select_learning_arms([ref],cutoff=100.,epsilon_eval=1e-6,r_time=2.)
    assert r['status']=='INCONCLUSIVE_MEASUREMENT'


def test_learning_confirmation_count_uses_pilot_effect_and_no_arbitrary16_cap():
    d = _learning_driver()
    q=d.plan_confirmation([.99,.90,1.01],[0.,0.,0.],epsilon_q=.01)
    assert q['N']>16 and q['N']>=q['N_time']
    assert q['unit']=='independent training-seed pair'
    with pytest.raises(ValueError): d.plan_confirmation([1.,1.,1.],[0.,0.,0.],epsilon_q=.01)
    with pytest.raises(ValueError): d.plan_confirmation([.8,.8,.8],[.01,.01,.01],epsilon_q=.01)


def test_learning_confirmation_intervals_and_censoring_fail_closed():
    d = _learning_driver()
    q=d.analyze_confirmation([80.]*8,[100.]*8,[1.]*8,[1.]*8,epsilon_q=.01)
    assert q['status']=='CONFIRMED_SCOPED_BENEFIT'
    assert q['time']['ratio']==pytest.approx(.8)
    assert q['time']['t_ci95']==pytest.approx([.8,.8])
    assert q['quality']['t_ci95']==pytest.approx([0.,0.])
    with pytest.raises(ValueError):d.analyze_confirmation([None]*8,[100.]*8,[1.]*8,[1.]*8,epsilon_q=.01)
    q=d.analyze_confirmation([110.]*8,[100.]*8,[1.]*8,[1.]*8,epsilon_q=.01)
    assert q['status']=='STOP_QUALITY'


def test_learning_simpler_control_dominance_uses_frozen_resolution():
    d = _learning_driver()
    q=d.simpler_dominates([101.]*3,[100.]*3,[1.]*3,[1.]*3,r_time=2.,epsilon_eval=1e-6)
    assert q['dominates']
    q=d.simpler_dominates([110.]*3,[100.]*3,[1.]*3,[1.]*3,r_time=2.,epsilon_eval=1e-6)
    assert not q['dominates']


def test_learning_confirmation_interval_disagreement_is_inconclusive():
    d = _learning_driver()
    # One asymmetric positive log-ratio observation makes the t upper bound
    # positive while this frozen percentile-bootstrap upper bound stays below0.
    candidate=[100*math.exp(-.1)]*7+[100*math.exp(.15)]
    q=d.analyze_confirmation(candidate,[100.]*8,[1.]*8,[1.]*8,epsilon_q=.01)
    assert q['time']['t_ci95'][1]>1
    assert q['time']['bootstrap_ci95'][1]<1
    assert q['status']=='INCONCLUSIVE_PRECISION'


@pytest.mark.parametrize('key', ['r_time','epsilon_eval'])
def test_learning_dominance_rejects_nonfinite_frozen_threshold(key):
    d = _learning_driver()
    values={'r_time':2.,'epsilon_eval':1e-6};values[key]=float('nan')
    with pytest.raises(ValueError):
        d.simpler_dominates([101.]*3,[100.]*3,[1.]*3,[1.]*3,**values)


def test_byte_recovery_drawn_epoch_rejects_zero_cursor_before_mutation():
    import copy
    from oar.sampling_lm import BatchStream
    state = _byte_recovery_fixture()
    state.stream.next_batch()
    saved = state.stream.state_dict()
    changed = copy.deepcopy(saved)
    changed['cursor'] = 0
    with pytest.raises(ValueError, match='cursor'):
        state.stream.load_state_dict(changed)
    _assert_byte_recovery_equal(state.stream.state_dict(), saved)


@pytest.mark.parametrize('damage', [
    'missing_state', 'wrong_moment_shape', 'nan_moment', 'wrong_step', 'missing_step',
])
def test_review_rejects_malformed_adam_before_mutating_live_state(damage):
    import torch
    source = _byte_recovery_fixture(sampler=None)
    source.step(.001)
    saved = source.snapshot()
    pid = next(iter(saved['optimizer']['state']))
    native = saved['optimizer']['state'][pid]
    if damage == 'missing_state':
        del saved['optimizer']['state'][pid]
    elif damage == 'wrong_moment_shape':
        native['exp_avg'] = torch.zeros(1)
    elif damage == 'nan_moment':
        native['exp_avg'].fill_(float('nan'))
    elif damage == 'wrong_step':
        native['step'].fill_(99.)
    elif damage == 'missing_step':
        del native['step']

    target = _byte_recovery_fixture(sampler=None)
    before = target.snapshot()
    with pytest.raises(ValueError):
        target.restore(saved)
    _assert_byte_recovery_equal(target.snapshot(), before)


def test_review_rejects_cursor_that_is_not_a_possible_batch_boundary():
    import copy
    stream = _byte_recovery_fixture(batch_size=2).stream
    stream.next_batch()
    before = stream.state_dict()
    invalid = copy.deepcopy(before)
    assert len(stream.blocks) == 5 and before['cursor'] == 2
    invalid['cursor'] = 1
    with pytest.raises(ValueError, match='cursor|boundary'):
        stream.load_state_dict(invalid)
    _assert_byte_recovery_equal(stream.state_dict(), before)


def _asset_driver():
    return driver()


def test_language_asset_guard_requires_body_permission_and_exact_document(tmp_path):
    d=_asset_driver()
    doc={'dataset':'wikitext-2-raw-v1','revision':'b08601e04326c79dfdd32d625aee71d232d685c3',
         'decision':'permission_evidence_reviewed_for_conditional_local_only_research'}
    p=tmp_path/'decision.json';p.write_text(json.dumps(doc));h=hashlib.sha256(p.read_bytes()).hexdigest()
    cfg={'asset':{'body_downloads':False,'name':doc['dataset'],'revision':doc['revision'],
                  'decision_path':str(p),'decision_sha256':h}}
    with pytest.raises(ValueError):d.assert_language_asset_permission(cfg)
    cfg['asset']['body_downloads']=True
    assert d.assert_language_asset_permission(cfg)==doc
    p.write_text(json.dumps({**doc,'revision':'wrong'}))
    with pytest.raises(ValueError):d.assert_language_asset_permission(cfg)


def test_language_pinned_fetch_verifies_before_use_and_never_overwrites(tmp_path,monkeypatch):
    d=_asset_driver();body=b'fictional parquet bytes';h=hashlib.sha256(body).hexdigest()
    import urllib.request
    import io
    calls=[]
    def fetch(url):calls.append(url);return io.BytesIO(body)
    monkeypatch.setattr(urllib.request,'urlopen',fetch)
    p=tmp_path/'data.parquet'
    assert not d.fetch_pinned_file('https://example.invalid/fixture',p,len(body),h)['reused']
    assert d.fetch_pinned_file('https://example.invalid/fixture',p,len(body),h)['reused']
    assert len(calls)==1
    with pytest.raises(ValueError):d.fetch_pinned_file('https://example.invalid/fixture',p,len(body),'0'*64)
    assert p.read_bytes()==body


def test_language_parquet_parser_lossless_real_library_no_text_in_manifest(tmp_path):
    d=_asset_driver()
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows=['\r\n preamble\r','= A =\r\né\n','= = Sub = =\n','= B =\nx']
    path=tmp_path/'fixture.parquet';pq.write_table(pa.table({'text':rows}),path)
    articles,manifest=d.parse_language_parquet(path,'fixture')
    assert ''.join(a.text for a in articles)==''.join(r.replace('\r\n','\n').replace('\r','\n') for r in rows)
    assert len(articles)==3 and manifest['article_count']==3
    assert manifest['normalized_utf8_sha256']==hashlib.sha256(''.join(a.text for a in articles).encode()).hexdigest()
    assert 'preamble' not in json.dumps(manifest) and 'Sub' not in json.dumps(manifest)


def test_byte_recovery_native_forward_backward_has_no_extra_loss_barrier(monkeypatch):
    import torch
    state = _byte_recovery_fixture(sampler=None)
    awaiting_backward = [False]
    original_forward = state.model.forward
    def forward(inputs):
        value = original_forward(inputs)
        awaiting_backward[0] = True
        return value
    state.model.forward = forward
    def observed_backward(gradient):
        awaiting_backward[0] = False
        return gradient
    handle = state.model.final_ln.weight.register_hook(observed_backward)
    original_bool = torch.Tensor.__bool__
    def checked_bool(value):
        assert not awaiting_backward[0], 'extra host scalar barrier between native forward and backward'
        return original_bool(value)
    monkeypatch.setattr(torch.Tensor, '__bool__', checked_bool)
    try:
        state.step(.001)
    finally:
        handle.remove()
    assert state.logical_updates == 1


@pytest.mark.mps
@pytest.mark.parametrize('sampler', [None, 'uniform', 'importance'])
def test_byte_model_mps_mask_exact_raw_adjoints_and_checkpoint_replay(sampler):
    import os
    import torch
    import torch.nn.functional as F
    from oar.sampling_data import normalize_articles, encode_articles, training_chunks, collate_blocks
    from oar.sampling_lm import ByteLM, ModelConfig, BatchStream, TrainingState, make_optimizer
    if not torch.backends.mps.is_available():
        pytest.skip('actual MPS unavailable; CPU tests are not qualification evidence')
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    articles = normalize_articles(['= Invented one =\nabc\n= Invented two =\n'+'xyz '*20], 'fixture')
    tokens, blocks = encode_articles(articles), training_chunks(articles)
    batch = collate_blocks(tokens, blocks, device='mps')
    config = ModelConfig(blocks=2, width=64, heads=1, ffn_width=128)
    dense = ByteLM(config, base_seed=89).to('mps')
    sampled = ByteLM(config, base_seed=89, sampler=sampler).to('mps')
    assert dense.initial_parameters_sha256 == sampled.initial_parameters_sha256
    outputs = [model(batch['input_ids']) for model in (dense, sampled)]
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-6)
    for output in outputs:
        (F.cross_entropy(output.reshape(-1,259), batch['target_ids'].reshape(-1),
                         reduction='sum', ignore_index=-100)/batch['valid_targets']).backward()
    reference = dict(dense.named_parameters())
    for name, parameter in sampled.named_parameters():
        if sampler is not None and ('ffn_in.weight' in name or 'ffn_out.weight' in name):
            continue
        assert parameter.grad.device.type == 'mps'
        torch.testing.assert_close(parameter.grad.cpu(), reference[name].grad.cpu(), rtol=1e-4, atol=1e-6)
    state = TrainingState(sampled, make_optimizer(sampled), BatchStream(tokens,blocks,batch_size=2,base_seed=89))
    state.step(.001)
    torch.mps.synchronize()
    saved = state.snapshot()
    resumed_model = ByteLM(config,base_seed=89,sampler=sampler).to('mps')
    resumed = TrainingState(resumed_model,make_optimizer(resumed_model),BatchStream(tokens,blocks,batch_size=2,base_seed=89))
    resumed.restore(saved)
    state.step(.0003)
    resumed.step(.0003)
    torch.mps.synchronize()
    for name, parameter in state.model.named_parameters():
        assert parameter.device.type == 'mps'
        torch.testing.assert_close(parameter.cpu(),dict(resumed.model.named_parameters())[name].cpu(),rtol=1e-4,atol=1e-6)
    assert state.logical_updates == resumed.logical_updates == 2
    for name in state.model.sampling_states:
        assert torch.equal(state.model.sampling_states[name].generator.get_state(),resumed.model.sampling_states[name].generator.get_state())


def _budget_case(tmp_path,monkeypatch,*,horizon=80.,interval=5.,initial_cost=1.):
    import torch
    import oar.sampling_lm as lm
    from oar.sampling_data import normalize_articles,encode_articles,training_chunks,evaluation_windows,choose_monitor
    articles=normalize_articles(['= Fiction =\n'+'abcd '*30],'fixture');tokens=encode_articles(articles)
    stream=lm.BatchStream(tokens,training_chunks(articles),batch_size=1,base_seed=8)
    model=lm.ByteLM(lm.ModelConfig(blocks=1,width=8,heads=2,ffn_width=16),base_seed=8,sampler='uniform')
    state=lm.TrainingState(model,lm.make_optimizer(model),stream)
    class Clock:
        now=0.
        def __call__(self):return self.now
    source=Clock();clock=lm.ChargedClock(0.,clock=source)
    real_step=state.step
    def step(lr):
        value=real_step(lr);source.now+=1.;return value
    state.step=step
    calls=[]
    def evaluation(model,tokens,windows,*,batch_size=16):
        calls.append(len(windows));source.now+=initial_cost
        return {'nll':4.-min(1.,source.now/horizon),'perplexity':math.e**3,'valid_targets':100,'window_count':len(windows),'wall_s':initial_cost}
    monkeypatch.setattr(lm,'evaluate',evaluation)
    windows=evaluation_windows(articles)
    data={'evaluation_tokens':tokens,'evaluation_full':windows,'evaluation_monitor':choose_monitor(articles,windows)}
    request={'id':'fixture','H':horizon,'Delta':interval,'R':20.,'T_budget':horizon+20.,
             'block_updates':1,'peak_lr':.001,'overshoot_limit_s':.01*horizon,
             'bindings':{'fixture':'frozen'},'run_dir':str(tmp_path/'events'),
             'checkpoint_dir':str(tmp_path/'private-checkpoints'),'phase':'development'}
    return lm,state,data,request,clock,source,calls


def test_budget_worker_charges_initial_evals_fixed_blocks_and_endpoint(tmp_path,monkeypatch):
    lm,state,data,request,clock,source,calls=_budget_case(tmp_path,monkeypatch,horizon=100.)
    result=lm.train_budget(state,data,request,clock)
    assert result['status']=='pass'
    assert result['initial_full']['completed_s']==1.
    assert result['initial_monitor']['completed_s']==2.
    assert result['t0_s']==2.
    assert result['steps'][0]['lr']==0.
    assert len(result['monitors'])>=8
    assert all(r['completed_s']>=r['deadline_s'] for r in result['monitors'])
    assert result['cutoff_completion_s']>=100.
    assert result['completion_s']<=120.
    assert result['endpoint_full']['completed_s']>=result['cutoff_completion_s']
    assert result['final_checkpoint']['sha256']
    assert result['steps'][-1]['logical_update']==state.logical_updates
    assert all(type(r['loss']) is float and math.isfinite(r['norm']) for r in result['steps'])


def test_budget_worker_missed_initial_deadlines_are_not_caught_up(tmp_path,monkeypatch):
    lm,state,data,request,clock,source,calls=_budget_case(tmp_path,monkeypatch,horizon=100.,interval=5.,initial_cost=6.)
    result=lm.train_budget(state,data,request,clock)
    assert [r['deadline_s'] for r in result['missed_deadlines'][:2]]==[5.,10.]
    assert all(r['deadline_s']>12. for r in result['monitors'])
    assert result['initial_monitor']['completed_s']==12.


def test_budget_worker_resume_preserves_schedule_and_original_t0(tmp_path,monkeypatch):
    import json
    lm,state,data,request,clock,source,calls=_budget_case(tmp_path/'first',monkeypatch,horizon=100.)
    real_step=state.step
    def interrupted(lr):
        if state.logical_updates==25:raise KeyboardInterrupt('injected external interruption')
        return real_step(lr)
    state.step=interrupted
    with pytest.raises(KeyboardInterrupt):lm.train_budget(state,data,request,clock)
    receipts=[json.loads(line) for line in (Path(request['run_dir'])/'checkpoint_receipts.jsonl').read_text().splitlines()]
    checkpoint=receipts[-1]
    prior=clock.elapsed()
    lm,resumed,data2,request2,clock2,source2,calls2=_budget_case(tmp_path/'second',monkeypatch,horizon=100.)
    clock2=lm.ChargedClock(0.,prior_elapsed_s=prior,clock=source2)
    request2['resume']={'path':checkpoint['path'],'sha256':checkpoint['sha256']}
    result=lm.train_budget(resumed,data2,request2,clock2)
    assert result['t0_s']==2.
    assert result['initial_full']['completed_s']==1.
    assert result['steps'][0]['lr']>0.
    assert result['restart']['prior_charged_s']==prior
    assert result['completion_s']>=100.


def test_model_cost_fixture_restores_native_history_with_isolated_candidate_streams():
    import torch
    from oar.sampling_data import normalize_articles,encode_articles,training_chunks
    from oar.sampling_lm import ModelFixture,ModelConfig
    articles=normalize_articles(['= Made up =\n'+'word '*60],'fixture')
    data={'training_tokens':encode_articles(articles),'training_blocks':training_chunks(articles)}
    tiny=ModelConfig(blocks=1,width=8,heads=2,ffn_width=16)
    dense=ModelFixture(data,base_seed=31,device='cpu',model_config=tiny,batch_size=2)
    dense.step();dense.step();common=dense.snapshot()
    candidate=ModelFixture(data,base_seed=31,sampler='uniform',device='cpu',model_config=tiny,batch_size=2)
    initial={k:v.generator.get_state().clone() for k,v in candidate.states.items()}
    candidate.restore(common)
    assert candidate.updates==2
    for name,p in dense.p.items():assert torch.equal(p,candidate.p[name])
    for k,v in candidate.states.items():assert torch.equal(v.generator.get_state(),initial[k])
    candidate.step();candidate.restore(common)
    for k,v in candidate.states.items():assert torch.equal(v.generator.get_state(),initial[k])


@pytest.mark.parametrize('fallback', [False, True, 'unresolved'])
@pytest.mark.parametrize('cost_failure', [False, True])
def test_controller_freezes_pooled_native_aa_before_all_candidates_and_never_passes_stage_e(tmp_path, monkeypatch, fallback, cost_failure):
    d, cfg = driver(), yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    arms = [{'arm': name, 'shape': 'middle4096'} for name in
            ('importance-0.125', 'importance-0.25', 'uniform-0.125', 'uniform-0.25', 'uniform-0.5')]
    baseline = {'status': 'pass', 'delta': .2085, 'eligible': arms,
                'sessions': [{'shapes': [{'shape': {'id': 'middle4096'}, 'updates_per_block': 10,
                              'selected_dense_pilot': [{'seconds': .5, 'updates': 10}],
                              'comparisons': {'AA': {'reset_s': [.01]}}}]}]*2}
    monkeypatch.setattr(d, 'source_hashes', lambda stage: {})
    monkeypatch.setattr(d, 'require_stage', lambda out, stage, hashes: baseline)
    monkeypatch.setattr(d, 'storage_guard', lambda *args: {'free_disk_bytes': 10**15, 'run_output_bytes': 0})
    monkeypatch.setattr(d, 'assert_language_asset_permission', lambda cfg: {})
    manifest = tmp_path/'data_manifest.json'
    manifest.write_text(json.dumps({'development_full': [{}]*32, 'development_monitor': [{}]*16}))
    monkeypatch.setattr(d, 'prepare_language_data', lambda *args: {
        'manifest_path': str(manifest), 'manifest_sha256': d.digest(manifest),
        'files': {'fitting-development': {'path': 'private.pt', 'sha256': 'a'*64}}, 'shared_preprocessing_wall_s': .2})
    events = []
    def command(attempt, args, *, env, timeout):
        request = json.loads(Path(args[-1]).read_text())
        assert timeout is None and request['launch_perf_s'] > 0
        assert request['data_sha256'] == 'a'*64
        kind, phase = request['kind'], request.get('phase')
        events.append((kind, phase, request['synchronize_each_update']))
        result = {'status': 'pass', 'session': request['session']}
        if kind == 'model_native' and phase == 'variants':
            result.update(variants={
                'single': {'status': 'pass', 'updates_per_block': 1, 'pilot_s': [.2]*3},
                'foreach': {'status': 'pass', 'updates_per_block': 1, 'pilot_s': [.1]*3},
                'fused': {'status': 'unsupported', 'reason': 'unsupported kernel'}})
        elif kind == 'model_native':
            assert request['fixed_variant'] == 'foreach'
            result.update(selected_variant='foreach', updates_per_block=2,
                          AA={'pairs': [{'seconds': {'dense': 1., 'candidate': 1.}}]*30})
        elif kind == 'model_cost':
            assert request['fixed_variant'] == 'foreach'
            assert len(request['arms']) == 5
            frozen = json.loads(Path(request['frozen_delta_path']).read_text())
            assert frozen['delta'] == .2085
            assert frozen['fixed_variant'] == 'foreach'
            result['comparisons'] = {arm['arm']: {'statistics': {'ci95': [.8, 1. if cost_failure and request['session'] == 1 else .9]}, 'updates_per_block': 2, 'pairs': [{'seconds': {'dense': 1., 'candidate': .8}}]*30} for arm in arms}
        else:
            assert kind == 'model_calibration' and request['fixed_variant'] == 'foreach'
            assert request['base_seed'] == 3101
            count = 1 if request['synchronize_each_update'] else 8
            result.update(initial_full={'nll': 5.}, initial_monitor={'nll': 4.},
                          initial_parameters_sha256='identical-initial-weights', startup_s=1., t0_s=.9,
                          monitor_s=10. if fallback and (count == 8 or fallback == 'unresolved') else .001,
                          checkpoint_s=.001, endpoint_s=.1, endpoint_checkpoint_s=.001,
                          update_block_s=.01*count, block_updates=count)
        d.write_json(request['result'], result)
        return 0, '', ''
    monkeypatch.setattr(d, 'command', command)
    result = d.pilot_qualification(cfg, ROOT/'configs/dw_sampling.yaml', tmp_path, tmp_path)
    if cost_failure:
        assert result['status'] == 'STOP_COST'
        assert len(events) == 6 and not any(kind == 'model_calibration' for kind, _, _ in events)
        return
    if fallback == 'unresolved':
        assert result['status'] == 'INCONCLUSIVE_MEASUREMENT'
        assert len(events) == 18 and len(result['qualification']['rounds']) == 2
        assert not (tmp_path/'model_qualification.json').exists()
        return
    assert result['status'] == 'not_run'
    qualified = result['qualification']
    assert qualified['status'] == 'pass' and qualified['native_variant'] == 'foreach'
    assert len(qualified['eligible']) == 5
    assert qualified['calibration']['epsilon_eval'] == 1e-6
    assert qualified['calibration']['horizon']['block_updates'] == (1 if fallback else 8)
    expected = [('model_native', 'variants'), ('model_native', 'variants'),
                ('model_native', 'AA'), ('model_native', 'AA'), ('model_cost', None), ('model_cost', None),
                ('model_calibration', None), ('model_calibration', None), ('model_calibration', None)]
    assert [(kind, phase) for kind, phase, flag in events] == expected*(2 if fallback else 1)
    assert [flag for _, _, flag in events] == [False]*9+([True]*9 if fallback else [])
    assert qualified['forecast']['full_budget_equivalents'] >= 12+15+1+16+3+6
    assert qualified['forecast']['deadline_s'] is None
    assert len(list(tmp_path.glob('*-request.json'))) == len(events)


@pytest.mark.parametrize('upper_second', [1., 1.01])
def test_model_cost_failure_stops_before_cold_calibration(tmp_path, monkeypatch, upper_second):
    d = driver()
    sessions = [{'comparisons': {'importance-0.125': {'statistics': {'ci95': [.8, upper]}}}} for upper in (.9, upper_second)]
    assert d.model_cost_eligible(sessions, [{'arm': 'importance-0.125', 'sampler': 'importance', 'rho': .125}]) == []


def test_cold_aggregation_uses_within_full_and_monitor_triples_without_crossing_scales():
    d = driver()
    rows = [{'initial_parameters_sha256': 'same', 'initial_full': {'nll': f}, 'initial_monitor': {'nll': m},
             'startup_s': 2+i, 't0_s': 1., 'monitor_s': .001, 'checkpoint_s': .002,
             'endpoint_s': .1, 'endpoint_checkpoint_s': .003, 'update_block_s': .08+i*.001,
             'block_updates': 8} for i, (f, m) in enumerate(zip((5., 5.00001, 5.00002), (3., 3.00004, 3.00002)))]
    result = d.aggregate_model_calibration(yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text()), rows, .3)
    assert result['status'] == 'pass'
    assert result['epsilon_eval'] == pytest.approx(.00008)
    assert result['horizon']['diagnostic_delta'] >= .3
    assert result['horizon']['B_max'] == .082
    rows[2]['initial_monitor']['nll'] += .01
    assert d.aggregate_model_calibration(yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text()), rows, .3)['status'] == 'INCONCLUSIVE_MEASUREMENT'
    rows[2]['initial_parameters_sha256'] = 'different'
    with pytest.raises(ValueError, match='initial'):
        d.aggregate_model_calibration(yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text()), rows, .3)


def test_variant_pool_requires_supported_intersection_and_baseline_medians():
    d = driver()
    rows = [{'variants': {'single': {'status': 'pass', 'updates_per_block': 1, 'pilot_s': [.3]*3},
                         'foreach': {'status': 'pass', 'updates_per_block': 1, 'pilot_s': [.1, .1, 10.]},
                         'fused': {'status': 'pass', 'updates_per_block': 1, 'pilot_s': [.001]*3}}},
            {'variants': {'single': {'status': 'pass', 'updates_per_block': 1, 'pilot_s': [.3]*3},
                         'foreach': {'status': 'pass', 'updates_per_block': 1, 'pilot_s': [.1]*3},
                         'fused': {'status': 'unsupported'}}}]
    selected, pooled = d.pooled_model_native_variant(rows)
    assert selected == 'foreach' and pooled['foreach']['pilot_s'] == [.1, .1, 10., .1, .1, .1]
    assert 'fused' not in pooled


def test_model_worker_requests_cannot_escape_pilot_or_result_directory(tmp_path, monkeypatch):
    d, cfg = driver(), yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    attempt = d.allocate_attempt(tmp_path, 'pilot')
    d.write_json(attempt/'source_manifest.json', {'measurement_hashes': {}, 'config_sha256': d.digest(ROOT/'configs/dw_sampling.yaml')})
    monkeypatch.setattr(d, 'source_hashes', lambda stage: {})
    monkeypatch.setattr(d, 'require_stage', lambda *args: {'status': 'pass'})
    request = {'kind': 'model_native', 'phase': 'variants', 'attempt': str(attempt), 'out': str(tmp_path),
               'config': str(ROOT/'configs/dw_sampling.yaml'), 'result': str(tmp_path/'escaped.json')}
    with pytest.raises(ValueError):
        d.validate_worker_request(request, 'pilot', ROOT/'configs/dw_sampling.yaml', tmp_path, attempt/'request.json')
    with pytest.raises(ValueError, match='boundary'):
        d.validate_worker_request(request, 'cost', ROOT/'configs/dw_sampling.yaml', tmp_path, attempt/'request.json')


def test_new_model_worker_accepts_only_bound_development_data_and_native_phase(tmp_path, monkeypatch):
    d = driver()
    attempt = d.allocate_attempt(tmp_path, 'pilot')
    cfg_path = ROOT/'configs/dw_sampling.yaml'
    d.write_json(attempt/'source_manifest.json', {'measurement_hashes': {}, 'config_sha256': d.digest(cfg_path)})
    data = tmp_path/'private.pt'
    data.write_bytes(b'exact CPU fixture data')
    manifest = attempt/'data_manifest.json'
    d.write_json(manifest, {'processed_files': {'fitting-development': {'path': str(data), 'sha256': d.digest(data)}}})
    monkeypatch.setattr(d, 'source_hashes', lambda stage: {})
    monkeypatch.setattr(d, 'require_stage', lambda *args: {'status': 'pass'})
    monkeypatch.setattr(d, 'assert_language_asset_permission', lambda cfg: {})
    cfg = yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    request = {'kind': 'model_native', 'phase': 'variants', 'session': 0, 'attempt': str(attempt),
               'out': str(tmp_path), 'config': str(cfg_path), 'result': str(attempt/'native.json'),
               'base_seed': cfg['seed'], 'synchronize_each_update': False, 'block_updates': 8,
               'launch_perf_s': 1., 'clock_implementation': __import__('time').get_clock_info('perf_counter').implementation, 'order_seed': 42, 'data_path': str(data), 'data_sha256': d.digest(data),
               'data_manifest_path': str(manifest), 'data_manifest_sha256': d.digest(manifest)}
    d.validate_worker_request(request, 'pilot', cfg_path, tmp_path, attempt/'request.json')
    for change in ({'phase': 'online_oracle'}, {'data_sha256': '0'*64}, {'base_seed': 0}, {'block_updates': 1},
                   {'data_path': str(tmp_path/'other.pt')}, {'device': 'cpu'}):
        with pytest.raises(ValueError):
            d.validate_worker_request({**request, **change}, 'pilot', cfg_path, tmp_path, attempt/'request.json')


@pytest.mark.parametrize('failure', [AssertionError('native numerical disagreement'),
    RuntimeError('invalid gradient does not support repair-by-fallback'), RuntimeError('MPS device error not supported on device')])
def test_model_native_numerical_and_device_errors_are_fatal(failure):
    d = driver()
    assert d.model_unsupported_native(failure) is False
    assert d.model_unsupported_native(NotImplementedError('kernel not implemented for MPS')) is True


@pytest.fixture
def actual_cpu_model(tmp_path, monkeypatch):
    import torch
    import oar.sampling_lm as lm
    import oar.timing
    import oar.hardware
    from oar.sampling_data import TargetBlock
    d, cfg = driver(), yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    cfg['timing']['minimum_block_s'] = .001
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    data = {'training_tokens': [[256, 97, 98, 257]],
            'training_blocks': [TargetBlock(0, 1, 3, 0)]*32,
            'evaluation_tokens': [[256, 97, 98, 257]],
            'evaluation_full': [TargetBlock(0, 1, 3, 0)]*2,
            'evaluation_monitor': [TargetBlock(0, 1, 3, 0)]}
    counter = {'work': 0., 'restores': 0, 'sync': 0}
    def fixture(cfg, data, request, *, variant, sampler=None, rho=.25):
        actual = lm.ModelFixture(data, base_seed=request['base_seed'], variant=variant, sampler=sampler, rho=rho,
                                 batch_size=16, device='cpu', model_config=lm.ModelConfig(blocks=1, width=8, heads=2, ffn_width=16),
                                 synchronize_each_update=request['synchronize_each_update'])
        original_step, original_restore = actual.step, actual.restore
        def step():
            result = original_step()
            counter['work'] += 1. if sampler is None else .5
            return result
        def restore(common):
            counter['restores'] += 1
            original_restore(common)
        actual.step, actual.restore = step, restore
        return actual
    def measure(operation, sync, blocks):
        rows = []
        for _ in range(blocks):
            before = counter['work']
            operation()
            rows.append(.01*(counter['work']-before))
        return rows
    def warmup(operation, sync, **kwargs):
        operation()
        return {'stable': False, 'raw_s': [.01, .04]}
    def synchronize():
        counter['sync'] += 1
    monkeypatch.setattr(d, 'model_fixture', fixture)
    monkeypatch.setattr(d, 'model_data', lambda request: data)
    monkeypatch.setattr(d, 'model_resource_guard', lambda *args: {'cpu_fixture': True})
    monkeypatch.setattr(torch.mps, 'synchronize', synchronize)
    monkeypatch.setattr(oar.timing, 'measure_blocks', measure)
    monkeypatch.setattr(oar.timing, 'warmup', warmup)
    monkeypatch.setattr(oar.hardware, 'resource_snapshot', lambda *args: {'cpu_fixture': True})
    request = {'kind': 'model_native', 'phase': 'variants', 'session': 0, 'base_seed': cfg['seed'], 'out': str(tmp_path),
               'result': str(tmp_path/'variants.json'), 'synchronize_each_update': False, 'order_seed': 42,
               'block_updates': 8, 'data_sha256': 'a'*64, 'clock_implementation': __import__('time').get_clock_info('perf_counter').implementation}
    yield d, cfg, data, request, counter
    torch.set_num_threads(previous_threads)


def test_actual_model_native_variants_use_fresh_scalar_reference_without_snapshot_restore(actual_cpu_model):
    d, cfg, data, request, counter = actual_cpu_model
    result = d.model_native_session(cfg, request)
    assert all(row['status'] == 'pass' for row in result['variants'].values())
    assert counter['restores'] == 0
    assert all(not row['warmup']['stable'] for row in result['variants'].values())
    assert len(list(Path(request['out']).glob('*.native-*.json'))) == 4


def test_model_native_qualification_shares_identical_clipped_gradients_and_leaves_pilots_native(actual_cpu_model, monkeypatch):
    import torch
    import oar.timing
    d, cfg, data, request, counter = actual_cpu_model
    original_factory = d.model_fixture
    original_assert = d.assert_native_equivalent
    original_warmup, original_measure = oar.timing.warmup, oar.timing.measure_blocks
    fixtures, checks, norms = [], [], []
    absent = object()
    def fixture(*args, **kwargs):
        value = original_factory(*args, **kwargs)
        index = len(fixtures)
        entry = {'fixture': value, 'step_attribute': value.opt.__dict__.get('step', absent),
                 'native_step_function': value.opt.step.__func__}
        fixtures.append(entry)
        if index > 0 and index % 2 == 0:
            def no_reference_model(*args, **kwargs):
                pytest.fail('scalar reference must consume cloned gradients without forward/backward')
            value.model.forward = no_reference_model
            value.step = no_reference_model
        else:
            step = value.step
            def actual_step():
                result = step()
                norms.append(float(result['norm']))
                return result
            value.step = actual_step
        return value
    def assert_equivalent(parameters, optimizer, reference_parameters, reference_optimizer):
        original_assert(parameters, optimizer, reference_parameters, reference_optimizer)
        actual = next(entry['fixture'] for entry in fixtures if entry['fixture'].opt is optimizer)
        reference = next(entry['fixture'] for entry in fixtures if entry['fixture'].opt is reference_optimizer)
        squared_norm = 0.
        for name, parameter in parameters.items():
            gradient, copied = parameter.grad, reference_parameters[name].grad
            assert gradient is not None and copied is not None
            torch.testing.assert_close(gradient, copied, rtol=0, atol=0)
            assert gradient.data_ptr() != copied.data_ptr() and not copied.requires_grad
            squared_norm += gradient.double().square().sum().item()
            assert optimizer.state[parameter]['step'].item() == actual.updates
        assert 0 < squared_norm <= 1.00001
        assert any(bool(state['exp_avg'].ne(0).any()) for state in reference_optimizer.state.values())
        assert all(group['eps'] == 1e-8 and group['betas'] == (.9, .999)
                   for opt in (optimizer, reference_optimizer) for group in opt.param_groups)
        assert reference.updates == 0
        checks.append(actual.updates)
    def check_native_pilot():
        for entry in fixtures:
            optimizer = entry['fixture'].opt
            assert optimizer.__dict__.get('step', absent) is entry['step_attribute']
            assert optimizer.step.__func__ is entry['native_step_function']
    def warmup(*args, **kwargs):
        check_native_pilot()
        return original_warmup(*args, **kwargs)
    def measure(*args, **kwargs):
        check_native_pilot()
        return original_measure(*args, **kwargs)
    monkeypatch.setattr(d, 'model_fixture', fixture)
    monkeypatch.setattr(d, 'assert_native_equivalent', assert_equivalent)
    monkeypatch.setattr(oar.timing, 'warmup', warmup)
    monkeypatch.setattr(oar.timing, 'measure_blocks', measure)
    result = d.model_native_session(cfg, request)
    assert checks == [1, 2, 3, 4, 5]*4
    assert any(norm > 1. for norm in norms), 'fixture must exercise native clipping'
    assert counter['restores'] == 0
    for variant, row in result['variants'].items():
        assert row['status'] == 'pass'
        assert row['qualification_mode'] == 'identical current full-model gradients after native clipping'
        assert row['qualification_updates'] == 5
        persisted = json.loads(Path(request['result']).with_suffix('.native-'+variant+'.json').read_text())
        assert persisted['qualification_mode'] == row['qualification_mode']


@pytest.mark.parametrize('preexisting_step', [False, True])
@pytest.mark.parametrize('fail', [False, True])
def test_model_shared_gradient_qualification_restores_exact_optimizer_step_on_success_and_failure(actual_cpu_model, monkeypatch, preexisting_step, fail):
    d, cfg, data, request, counter = actual_cpu_model
    actual = d.model_fixture(cfg, data, request, variant='foreach')
    reference = d.model_fixture(cfg, data, request, variant='single')
    native_step = actual.opt.step
    calls = []
    if preexisting_step:
        def previous_step(*args, **kwargs):
            calls.append('original native entry')
            return native_step(*args, **kwargs)
        actual.opt.step = previous_step
    else:
        assert 'step' not in actual.opt.__dict__
    if fail:
        def numerical_failure(*args):
            raise AssertionError('fixed native equivalence failure')
        monkeypatch.setattr(d, 'assert_native_equivalent', numerical_failure)
        with pytest.raises(AssertionError, match='fixed native equivalence'):
            d.qualify_model_native_gradients(actual, reference)
    else:
        d.qualify_model_native_gradients(actual, reference)
    expected = 1 if fail else 5
    assert actual.updates == expected and reference.updates == 0
    assert all(state['step'].item() == expected for state in reference.opt.state.values())
    if preexisting_step:
        assert actual.opt.__dict__['step'] is previous_step
        assert len(calls) == expected
    else:
        assert 'step' not in actual.opt.__dict__
        assert actual.opt.step.__func__ is native_step.__func__
    actual.step()
    assert actual.updates == expected+1
    assert all(state['step'].item() == expected for state in reference.opt.state.values())


def test_actual_model_aa_candidates_keep_native_history_streams_pairs_and_setup(actual_cpu_model):
    d, cfg, data, request, counter = actual_cpu_model
    request.update(phase='AA', fixed_variant='single', result=str(Path(request['out'])/'native-AA.json'))
    native = d.model_native_session(cfg, request)
    assert len(native['AA']['pairs']) == 30
    assert native['history']['history_updates'] == 8 and native['history']['nonzero_native_moments']
    frozen = Path(request['out'])/'frozen.json'
    d.write_json(frozen, {'delta': .3, 'fixed_variant': 'single'})
    request.update(kind='model_cost', result=str(Path(request['out'])/'cost.json'),
        updates_per_block=native['updates_per_block'], frozen_delta_path=str(frozen), frozen_delta_sha256=d.digest(frozen),
        arms=[{'arm': 'importance-0.25', 'sampler': 'importance', 'rho': .25},
              {'arm': 'uniform-0.5', 'sampler': 'uniform', 'rho': .5}])
    result = d.model_cost_session(cfg, request)
    assert len(result['comparisons']) == 2
    from oar.sampled_linear import named_seed
    for arm_name, row in result['comparisons'].items():
        assert len(row['pairs']) == 30 and len(row['reset_s']) == 62
        assert row['statistics']['ratio'] >= row['steady_state_statistics']['ratio']
        assert row['updates_per_block'] == result['updates_per_block']
        assert row['sampling_streams_final']['sampling']
        assert row['order_seed'] == named_seed(cfg['seed'], 'order', 'full_model', 'cost',
                                               str(request['session']), arm_name, str(request['synchronize_each_update']))
    assert len(list(Path(request['out']).glob('*.jsonl'))) == 3


def test_actual_cold_worker_charges_initial_checkpoint_and_logs_trained_controller_block(actual_cpu_model):
    import time
    d, cfg, data, request, counter = actual_cpu_model
    request.update(kind='model_calibration', base_seed=3101, fixed_variant='single',
                   launch_perf_s=time.perf_counter()-.1, result=str(Path(request['out'])/'cold.json'))
    result = d.model_calibration_session(cfg, request)
    assert result['startup_s'] > .1 and result['startup_s'] >= result['t0_s']
    assert result['history']['history_updates'] == 8
    assert len(result['checkpoint_receipts']) == 2 and all(Path(r['path']).is_file() for r in result['checkpoint_receipts'])
    assert result['block_updates'] == 8 and result['update_block_s'] > 0
    events = [json.loads(line) for line in Path(request['result']).with_suffix('.events.jsonl').read_text().splitlines()]
    assert [row['kind'] for row in events] == ['initial_full', 'initial_monitor', 'checkpoint', 'trained_monitor_cost',
                                              'checkpoint', 'trained_endpoint_cost', 'update_block_start', 'update_block']
    assert len(events[-1]['stats']) == 8


def test_native_continuity_allows_only_supported_schema_metadata_changes():
    d, cfg = driver(), yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text())
    old = copy.deepcopy(cfg)
    old['schema_version'] = 2
    old['learning'] = {'old_unconsumed_settings': True}
    old['asset'] = {'old_unconsumed_permission': False}
    assert d.native_carryover_config(yaml.safe_dump(old), 'cost') == d.native_carryover_config(yaml.safe_dump(cfg), 'cost')
    changed = copy.deepcopy(cfg)
    changed['timing']['cost_gate_upper_ratio'] = .95
    assert d.native_carryover_config(yaml.safe_dump(old), 'cost') != d.native_carryover_config(yaml.safe_dump(changed), 'cost')
    assert d.native_carryover_config(yaml.safe_dump(old), 'feasibility') == d.native_carryover_config(yaml.safe_dump(changed), 'feasibility')
    cfg['schema_version'] = 4
    with pytest.raises(ValueError, match='schema'):
        d.native_carryover_config(yaml.safe_dump(cfg), 'cost')


def test_actual_native_numerical_failure_stops_without_trying_weaker_model(actual_cpu_model, monkeypatch):
    d, cfg, data, request, counter = actual_cpu_model
    def disagree(*args):
        raise AssertionError('native model parameters/moments disagree')
    monkeypatch.setattr(d, 'assert_native_equivalent', disagree)
    with pytest.raises(AssertionError, match='native model'):
        d.model_native_session(cfg, request)
    assert len(list(Path(request['out']).glob('*.native-*.json'))) == 0
    assert counter['restores'] == 0


def test_model_variant_pool_compares_equal_work_when_session_counts_differ():
    d = driver()
    sessions = [{'variants': {
        'single': {'status':'pass','updates_per_block':1,'pilot_s':[.6]*3},
        'fused': {'status':'pass','updates_per_block':1,'pilot_s':[.5]*3}}},
        {'variants': {
        'single': {'status':'pass','updates_per_block':3,'pilot_s':[1.2]*3},
        'fused': {'status':'pass','updates_per_block':3,'pilot_s':[1.32]*3}}}]
    variant, pooled = d.pooled_model_native_variant(sessions)
    assert variant == 'fused'
    assert pooled['fused']['pilot_s'] == pytest.approx([.5]*3+[.44]*3)
    assert pooled['fused']['unit'] == 'seconds per native update'
    assert pooled['fused']['raw_sessions'][1]['updates_per_block'] == 3


def test_model_horizon_uses_slowest_qualified_native_block_not_only_cold_blocks():
    d = driver()
    rows = [{'initial_parameters_sha256':'same','initial_full':{'nll':5.},'initial_monitor':{'nll':4.},
             'startup_s':2.,'monitor_s':.001,'checkpoint_s':.002,'endpoint_s':.1,'endpoint_checkpoint_s':.002,
             'update_block_s':.8,'block_updates':8} for _ in range(3)]
    q = d.aggregate_model_calibration(yaml.safe_load((ROOT/'configs/dw_sampling.yaml').read_text()),rows,.2,
          baseline_blocks=[{'source':'valid native AA spike','seconds':1.2,'updates':4}])
    assert q['horizon']['H'] >= 4096*.3
    assert q['horizon']['B_max'] == 1.2
    assert q['baseline_update_s_bound'] == pytest.approx(.3)


def test_budget_worker_final_receipt_retires_own_ephemeral_resume_files(tmp_path, monkeypatch):
    lm,state,data,request,clock,source,calls = _budget_case(tmp_path,monkeypatch,horizon=100.)
    result = lm.train_budget(state,data,request,clock)
    retained = list(Path(request['checkpoint_dir']).glob('*.pt'))
    assert retained == [Path(result['final_checkpoint']['path'])]
    assert all(row['pruned'] for row in result['checkpoint_manifest'] if row['kind'] != 'final')


def test_budget_worker_interrupted_partial_block_records_prefix_without_checkpoint(tmp_path, monkeypatch):
    lm,state,data,request,clock,source,calls = _budget_case(tmp_path,monkeypatch,horizon=100.)
    request['block_updates'] = 8
    original_step = state.step
    def interrupted(lr):
        if state.logical_updates == 3:
            raise KeyboardInterrupt('injected partial-block interruption')
        return original_step(lr)
    state.step = interrupted
    with pytest.raises(KeyboardInterrupt):
        lm.train_budget(state,data,request,clock)
    failure = json.loads((Path(request['run_dir'])/'failure.json').read_text())
    assert failure['successful_updates_in_block'] == 3
    assert failure['logical_updates'] == 3
    assert failure['last_committed_checkpoint']['logical_updates'] == 0
    assert failure['lr'] == 0.
    assert len(list(Path(request['checkpoint_dir']).glob('*.pt'))) == 1
    with pytest.raises(RuntimeError, match='failed|uncommitted'):
        state.snapshot()


def test_actual_cold_worker_rejects_different_parent_monotonic_clock(actual_cpu_model):
    d,cfg,data,request,counter = actual_cpu_model
    import time
    request.update(kind='model_calibration', base_seed=3101, fixed_variant='single',
                   launch_perf_s=time.perf_counter()-.1, result=str(Path(request['out'])/'wrong-clock.json'))
    request['clock_implementation'] = 'different host clock'
    with pytest.raises(ValueError,match='clock'):
        d.model_calibration_session(cfg,request)
