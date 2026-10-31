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
