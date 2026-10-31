"""Bounded same-seed checks cannot silently drop rows or accept changed science."""
import copy
import json
from pathlib import Path

import pytest


def api():
    from oar import reproduction
    return reproduction


def row():
    r = json.loads(Path('analysis/structural_oracle_audit.jsonl').read_text().splitlines()[0])
    r['split'] = 'evaluation'
    return r


def test_comparison_keeps_geometry_labels_and_acceptance_not_wall_times():
    o = api()
    original = row()
    repeat = dict(original, oracle_cost_s=999., checkpoint_id='different/local/path', basis_sha256='regenerated')
    result = o.compare_rows([original], [repeat])
    assert result['status'] == 'reproduced' and result['rows_compared'] == 1
    repeat['development_delta'] += .02
    assert o.compare_rows([original], [repeat])['status'] == 'failed'


def test_categorical_signs_are_exact_even_inside_scalar_tolerance():
    o = api()
    original = dict(row(), development_delta=1e-9)
    repeat = dict(original, development_delta=-1e-9)
    result = o.compare_rows([original], [repeat])
    assert result['status'] == 'failed'
    assert any('sign' in failure for failure in result['failures'])


@pytest.mark.parametrize('alter', ['missing', 'duplicate', 'nonfinite', 'scope', 'acceptance'])
def test_bad_reproduction_is_retained_as_failure(alter):
    o = api()
    a = row(); b = dict(a)
    original, repeat = [a], [b]
    if alter == 'missing': repeat = []
    if alter == 'duplicate': repeat.append(b)
    if alter == 'nonfinite': b['energy'] = float('nan')
    if alter == 'scope': b['count_unit'] = 'tokens'
    if alter == 'acceptance':
        a['proposal_raw_alignment'] = b['proposal_raw_alignment'] = 1.
        a['proposal_complement_bound'] = .5-1e-9
        b['proposal_complement_bound'] = .5+1e-9
    result = o.compare_rows(original, repeat)
    assert result['status'] == 'failed' and result['failures']


def test_path_clone_preserves_every_scientific_setting():
    o = api()
    import yaml
    original = yaml.safe_load(Path('configs/structural_oracle.yaml').read_text())
    clone = dict(original, output='analysis/reproduction/test', checkpoints_dir='checkpoints/reproduction/test')
    o.validate_clone(original, clone)
    for key, value in [('seeds', [1101]), ('max_wall_s', 700), ('shape', {'n': 12, 'di': 5, 'do': 9})]:
        with pytest.raises(ValueError, match='scientific'):
            o.validate_clone(original, dict(clone, **{key: value}))


def test_comparison_policy_predeclares_exact_field_coverage():
    o = api(); r = row()
    assert set(o.NUMERIC_FIELDS) <= r.keys()
    assert set(o.CATEGORICAL_FIELDS) <= r.keys()
    assert {'proposal_true_alignment', 'proposal_complement_bound', 'dense_development_delta', 'lost_progress'} <= set(o.NUMERIC_FIELDS)
    assert o.POLICY['atol'] == 2e-6 and o.POLICY['rtol'] == 2e-4
    assert o.POLICY['taus'] == [.25, .5, .75]


def test_curve_comparison_uses_actual_committed_update_field():
    o = api()
    reference = json.loads(Path('analysis/structural_oracle_summary.json').read_text())['curves']
    repeated = copy.deepcopy(reference)
    result = o.compare_curves(reference, repeated)
    assert result['curves_compared'] == 30 and result['curve_max_absolute_difference'] == 0
    assert result['failures'] == []
    repeated[0]['development_mse'] += .01
    assert o.compare_curves(reference, repeated)['failures']
    assert o.compare_curves(reference, reference[:-1])['failures']
    assert o.compare_curves(reference, reference+[reference[0]])['failures']


@pytest.mark.parametrize('mode', ['normal', 'recovery'])
@pytest.mark.parametrize('alter', ['duplicate', 'missing', 'hash', 'valid'])
def test_reproduction_helper_validates_checkpoint_identity_count_and_bytes(tmp_path, monkeypatch, mode, alter):
    """The external worker is replaced; real helper validation reads real fixture bytes."""
    import runpy
    import shutil
    from types import SimpleNamespace
    helpers = runpy.run_path('analysis/reproduce_structural_oracle.py')
    original = json.loads(Path('analysis/structural_oracle_summary.json').read_text())
    reference_row = row()
    for name in ('configs/structural_oracle.yaml', 'src/oar/oracle.py', 'src/oar/reproduction.py',
                 'analysis/reproduce_structural_oracle.py'):
        target = tmp_path/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(name, target)
    monkeypatch.chdir(tmp_path)
    Path('decisions').mkdir()
    prior = '{"fixture_permission": true}\n'
    Path('decisions/amendments.jsonl').write_text(prior)
    entries = []
    for step in (2, 4):
        checkpoint = tmp_path/f'{step}.pt'
        checkpoint.write_bytes(f'fixture checkpoint {step}'.encode())
        entries.append(dict(condition='full_rank', seed=17, step=step, path=str(checkpoint),
                            sha256=helpers['sha'](checkpoint)))
    original.update(artifact_index=entries, row_count=1, curves=original['curves'][:1])
    Path('analysis/structural_oracle_summary.json').write_text(json.dumps(original))
    Path('analysis/structural_oracle_audit.jsonl').write_text(json.dumps(reference_row)+'\n')
    out = Path('analysis/reproduction/fixture')
    g = helpers['run'].__globals__
    monkeypatch.setitem(g, 'require_structural_oracle', lambda *a, **k: {})

    def fixture_worker(command, **kwargs):
        import yaml
        summary = copy.deepcopy(original)
        summary['config'] = yaml.safe_load((out/'config.yaml').read_text())
        if alter == 'duplicate': summary['artifact_index'][1] = copy.deepcopy(entries[0])
        if alter == 'missing': summary['artifact_index'].pop()
        if alter == 'hash': summary['artifact_index'][1]['sha256'] = '0'*64
        (out/'structural_oracle_summary.json').write_text(json.dumps(summary))
        (out/'structural_oracle_audit.jsonl').write_text(json.dumps(reference_row)+'\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setitem(g, 'subprocess', SimpleNamespace(
        check_output=lambda *a, **k: '0'*40, run=fixture_worker, STDOUT=-2))
    helpers['prepare'](out)
    status = helpers['run'](out/'plan.json')
    if mode == 'recovery':
        # Recreate only the retained schema-failure condition accepted by recovery.
        failure_path = out/'comparison.json'
        failed = json.loads(failure_path.read_text())
        failed.update(status='failed', failures=["KeyError: 'step'"])
        failure_path.write_text(json.dumps(failed))
        status = helpers['recover_comparison'](out/'plan.json')
        result = json.loads((out/'comparison_recovered.json').read_text())
    else:
        result = json.loads((out/'comparison.json').read_text())
    assert status == (0 if alter == 'valid' else 1)
    assert result['status'] == ('reproduced' if alter == 'valid' else 'failed')
    if alter == 'duplicate':
        assert any('identity' in f for f in result['failures'])
    if alter == 'valid':
        assert result['checkpoint_hashes_validated'] == 2
        assert result['failures'] == []
    assert json.loads(Path('decisions/amendments.jsonl').read_text().splitlines()[-1]) == json.loads(prior)
