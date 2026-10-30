from pathlib import Path
import json
from dataclasses import replace

import pytest
import torch

from oar.bench import account_attempt, cycle_forecast, candidate_linear, load_benchmark_config, _paired_compare, _compare_many, BenchmarkConfig, native_state
from oar.cli import main


def test_complete_attempt_cannot_hide_preview():
    result = account_attempt({'forward_backward_reconstruction': 3., 'preview': 2.,
                              'probes_generation_journal': .5, 'complement': .5,
                              'decision_commit': 1.})
    assert result['active_wall_s'] == 7.
    assert result['candidate_only_s'] == 3.
    assert result['audit_and_preview_s'] == 4.


def test_refresh_transfers_and_replay_are_additional_costs():
    result = cycle_forecast(dense=10., attempt=6., replay=11., refresh_extra=8.,
                            interval=4, rejection_probability=.25)
    assert result['cycle_mean_s'] == 11.0625  # (10+8+3*(6+.25*11))/4
    assert result['local_break_even_p'] == pytest.approx(4/33)
    at_threshold = cycle_forecast(10., 6., 11., 8., 4, 4/33)
    assert at_threshold['cycle_mean_s'] == pytest.approx(10.)
    assert cycle_forecast(10., 6., 11., 8., 4, 0.)['cycle_mean_s'] < 10.
    assert cycle_forecast(10., 6., 11., 8., 4, .25)['cycle_mean_s'] > 10.
    assert cycle_forecast(10., 12., 11., 0., 4, 0.)['local_break_even_p'] < 0


def test_all_dense_cycle_has_no_approximate_segment_or_rejection_threshold():
    result = cycle_forecast(10., 6., 11., 8., 1, .25)
    assert result['cycle_mean_s'] == 18.
    assert result['has_approximate_segment'] is False
    assert result['local_break_even_p'] is None


def test_worker_request_preserves_resolved_seed_streams(monkeypatch):
    from types import SimpleNamespace
    from oar.bench import benchmark
    from oar.config import SeedStreams
    config = replace(load_benchmark_config(Path('configs/operator.yaml')),
                     seeds=SeedStreams(11, 22, 33, 44))
    def worker(command, **kwargs):
        request = json.loads(Path(command[-2]).read_text())
        assert request['seeds'] == {'construction':11,'audit':22,'data':33,'model':44}
        Path(command[-1]).write_text(json.dumps({'rows':[], 'metadata':{'seeds':request['seeds']}}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr('oar.bench.subprocess.run', worker)
    assert benchmark(config).metadata['seeds']['model'] == 44


def test_data_shape_layout_and_seed_do_not_shift_model_initialization():
    from oar.bench import _initialize_inputs_and_parameters
    from oar.config import SeedStreams
    shape = load_benchmark_config(Path('configs/operator.yaml')).benchmark.shapes[0]
    seeds = SeedStreams(11, 22, 33, 44)
    x, parameters = _initialize_inputs_and_parameters(shape, seeds, 'cpu')
    variants = [(replace(shape,n=256),seeds), (replace(shape,noncontiguous=True),seeds),
                (shape,replace(seeds,data=34)), (shape,replace(seeds,construction=12,audit=23))]
    for variant, streams in variants:
        _, actual = _initialize_inputs_and_parameters(variant, streams, 'cpu')
        for name in parameters:
            torch.testing.assert_close(actual[name],parameters[name],rtol=0,atol=0)
    same_x, changed_model = _initialize_inputs_and_parameters(shape,replace(seeds,model=45),'cpu')
    torch.testing.assert_close(x,same_x,rtol=0,atol=0)
    assert not torch.equal(parameters['w1'],changed_model['w1'])


def test_summary_bootstrap_uses_the_same_scheduled_mixture_threshold(tmp_path,monkeypatch):
    import runpy
    repo = Path(__file__).resolve().parents[1]
    table = json.loads((repo/'analysis/timing_calibration.json').read_text())
    row = table['rows'][0]
    row['headline']['conditional']['dense']['mean_s'] = 10.
    row['headline']['conditional']['dense']['raw_s'] = [10.]*3
    row['headline']['conditional']['other']['mean_s'] = 6.
    row['headline']['conditional']['other']['raw_s'] = [6.]*3
    row['headline']['replay_only']['other']['mean_s'] = 11.
    row['headline']['replay_only']['other']['raw_s'] = [11.]*3
    row['refresh']['dense']['mean_s'] = 10.
    row['refresh']['other']['mean_s'] = 18.
    row['refresh']['pairs'] = [{'seconds':{'dense':10.,'other':18.}}]*3
    table['rows'] = [row]
    (tmp_path/'analysis').mkdir()
    for name in ('timing_calibration.json','timing_calibration_wide.json'):
        (tmp_path/'analysis'/name).write_text(json.dumps(table))
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(repo/'analysis/summarize_headroom.py'))
    result = json.loads((tmp_path/'analysis/headroom_decision.json').read_text())['rows'][0]
    assert result['local_break_even_p_K16_bootstrap_95'] == pytest.approx([52/165,52/165])
    at_four = next(f for f in result['forecasts'] if f['interval']==4)
    assert at_four['local_break_even_p'] == pytest.approx(4/33)


@pytest.mark.parametrize('kwargs', [{'preview': -1.}, {'preview': float('nan')}])
def test_accounting_rejects_invalid_costs(kwargs):
    with pytest.raises(ValueError):
        account_attempt({'forward_backward_reconstruction': 3., **kwargs})


def test_candidate_only_derivative_keeps_exact_dx_and_reconstructs_dw():
    x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    w = torch.tensor([[2., 1.], [4., 3.]], requires_grad=True)
    b = torch.zeros(2, requires_grad=True)
    u = torch.tensor([[1.], [0.]])
    d = torch.tensor([[1., 3.], [2., 4.]])
    y = candidate_linear(x, w, b, u)
    y.backward(d)
    torch.testing.assert_close(x.grad, torch.tensor([[14., 10.], [20., 14.]]))
    torch.testing.assert_close(w.grad, torch.tensor([[7., 10.], [0., 0.]]))
    torch.testing.assert_close(b.grad, torch.tensor([3., 7.]))


def test_strict_benchmark_config_and_cli_errors(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('schema_version: 1\nunknown: true\n')
    with pytest.raises(ValueError):
        load_benchmark_config(path)
    assert main(['bench', '--config', str(path)]) == 2


def test_repository_operator_config_resolves():
    config = load_benchmark_config(Path('configs/operator.yaml'))
    assert config.device == 'mps'
    assert all(s.di < s.do and s.rank < min(s.di, s.do) for s in config.benchmark.shapes)


def test_measured_comparisons_serialize_as_strict_json():
    config = BenchmarkConfig(1, 3, 3, 10, .1,
                             dict(minimum=2, maximum=2, window=2, relative_range=.1), (), 'unused')
    result = _paired_compare(lambda: None, lambda: None, lambda: None, config)
    assert json.loads(json.dumps(result, allow_nan=False))['noise_target_met'] in (True, False)


def test_native_state_transport_uses_current_post_replay_moments():
    from oar.optimizer_preview import preview_adamw
    from oar.records import AdamWGroup
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    opt = torch.optim.AdamW([p], lr=.01, foreach=False, fused=False)
    for g in ([1., -1.], [2., 3.]):
        p.grad = torch.tensor(g)
        opt.step()
    state = native_state({'p': p}, opt)
    assert state['p'].step == 2
    pending = preview_adamw({'p': p}, {'p': p.grad}, state,
                            (AdamWGroup(('p',), .01, (.9, .999), 1e-8, .01),), 0, 0, None)
    opt.step()
    torch.testing.assert_close(pending.candidate_parameters['p'], p)


def test_benchmark_config_hash_includes_operator_shape_and_noise_policy():
    from oar.config import config_sha256, resolve_config
    config = load_benchmark_config(Path('configs/operator.yaml'))
    altered = replace(config, benchmark=replace(config.benchmark, blocks=7))
    assert config_sha256(config) != config_sha256(altered)
    assert resolve_config(config)['benchmark']['shapes'][0]['n'] == 128


def test_invalid_benchmark_yaml_returns_config_error(tmp_path):
    path = tmp_path / 'invalid.yaml'
    path.write_text('schema_version: [unterminated')
    assert main(['bench', '--config', str(path)]) == 2


def test_unavailable_swap_counters_are_unknown_not_zero(monkeypatch):
    from oar.hardware import resource_snapshot
    def denied():
        raise OSError('sandbox')
    monkeypatch.setattr('oar.hardware.psutil.swap_memory', denied)
    facts = resource_snapshot('cpu')
    assert facts['swap_used_bytes'] is None
    assert facts['swap_in_bytes'] is None
    assert facts['process_rss_bytes'] > 0


def test_headline_comparisons_interleave_in_a_shared_window():
    now = [0.]
    counts = {name: 0 for name in ('candidate', 'conditional', 'reject')}
    def dense():
        now[0] += 4
    def op(name, cost):
        def run():
            counts[name] += 1
            now[0] += cost
        return run
    config = BenchmarkConfig(1, 3, 3, 20, .1,
                             dict(minimum=2, maximum=2, window=2, relative_range=.1), (), 'unused')
    results = _compare_many(dense, {n:op(n,c) for n,c in [('candidate',2),('conditional',6),('reject',8)]},
                            lambda: None, config, context=lambda:dict(counts), clock=lambda:now[0])
    assert [results[n]['other']['mean_s'] for n in counts] == [2.,6.,8.]
    for result in results.values():
        assert len(result['pairs']) == 3
        for pair in result['pairs']:
            values = pair['context_after'].values()
            assert max(values)-min(values) <= 1


def test_duplicate_cost_table_keys_cannot_silently_discard_accumulation_results(tmp_path):
    import yaml
    payload = yaml.safe_load(Path('configs/operator.yaml').read_text())
    payload['shapes'].append({**payload['shapes'][0], 'accumulation': 2})
    path = tmp_path/'duplicate.yaml'
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match='duplicate cost-table key'):
        load_benchmark_config(path)


def test_direct_benchmark_validates_keys_before_launching_hardware(monkeypatch):
    from oar.bench import benchmark
    config = load_benchmark_config(Path('configs/operator.yaml'))
    config = replace(config, benchmark=replace(config.benchmark, shapes=(config.benchmark.shapes[0],)*2))
    def forbidden_worker(*args, **kwargs):
        raise RuntimeError('invalid input reached hardware worker')
    monkeypatch.setattr('oar.bench.subprocess.run', forbidden_worker)
    with pytest.raises(ValueError, match='duplicate cost-table key'):
        benchmark(config)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.mps)])
def test_benchmark_input_layout_is_created_on_target_device(device):
    from oar.bench import _make_input
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('actual MPS unavailable')
    shape = load_benchmark_config(Path('configs/operator.yaml')).benchmark.shapes[1]
    x = _make_input(shape, torch.Generator().manual_seed(61001), device)
    assert x.device.type == device
    assert x.shape == (512, 256)
    assert x.stride() == (512, 2)
    assert not x.is_contiguous() and x.requires_grad and x.is_leaf
    source = torch.randn(512, 512, generator=torch.Generator().manual_seed(61001))[:, ::2]
    torch.testing.assert_close(x.detach().cpu(), source, rtol=0, atol=0)
