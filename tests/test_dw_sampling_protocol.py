# SPDX-License-Identifier: AGPL-3.0-only
"""Protocol tests exercise validation and persisted evidence, not source text."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
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
    for mutation in ('extra', 'rho', 'shape', 'timing', 'asset'):
        cfg = yaml.safe_load(original.read_text())
        if mutation == 'extra':
            cfg['unapproved_sampler'] = True
        elif mutation == 'rho':
            cfg['rhos'] = [0.01]
        elif mutation == 'shape':
            cfg['shapes'][1]['di'] = 1024
        elif mutation == 'timing':
            cfg['timing']['pairs'] = 3
        else:
            cfg['asset']['body_downloads'] = True
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
    assert d.freeze_delta(aa) == pytest.approx(max(.05, 2 * np.quantile(aa, .95, method='linear')))
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


def test_selected_dense_block_is_remeasured_until_duration_or_cap():
    d = driver()
    count, observations = d.choose_block_length(lambda n: .001*n, 3)
    assert count == 100 and observations[-1]['seconds'] == pytest.approx(.1)
    durations = iter([.05, .15, .21])
    count, observations = d.choose_block_length(lambda n: next(durations), 5)
    assert count == 27
    assert observations[-1]['seconds'] >= .2


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
