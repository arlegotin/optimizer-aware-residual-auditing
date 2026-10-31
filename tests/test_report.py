"""Truthfulness and evidence integrity, independent of cosmetic Markdown layout."""
import hashlib
import json
from pathlib import Path

import pytest

from oar.records import GateRecord, to_json


def build(*args):
    from oar.report import build_report
    return build_report(*args)


def stopped(tmp_path, reason='CPU-only stop: actual MPS unavailable'):
    gates = tmp_path/'gates.jsonl'
    gates.write_text(to_json(GateRecord('WP00', 'stop', reason, 'Synthesize only', {}, {}, {}))+'\n')
    return gates


def evidence(tmp_path, gates, **changes):
    source = tmp_path/'raw.json'
    source.write_text('{"observed": true}\n')
    payload = dict(schema_version=1, root='.', gate_snapshot=dict(lines=1, sha256=hashlib.sha256(gates.read_bytes()).hexdigest()),
                   artifacts=[dict(path='raw.json', sha256=hashlib.sha256(source.read_bytes()).hexdigest())],
                   evidence=[], audits=[])
    payload.update(changes)
    index = tmp_path/'index.json'
    index.write_text(json.dumps(payload))
    return index


def item(kind='backend timing', **changes):
    result = dict(evidence_class=kind, title='Complete operator', artifact='raw.json', status='measured',
                  scope='isolated residual block; actual MPS', summary='Correct but slower OAR.',
                  uncertainty='Within-study interval; independent runs unavailable.',
                  metrics={'attempt': {'value': 54.4, 'unit': 'ms'}, 'native': {'value': 36.7, 'unit': 'ms'}})
    result.update(changes)
    return result


def test_early_stop_is_reportable_without_speedup(tmp_path):
    reason = 'CPU-only stop: actual MPS unavailable'
    gates = stopped(tmp_path, reason)
    text = build(gates, None, tmp_path/'report.md').read_text()
    assert 'No training speedup was established' in text and reason in text
    assert 'No run index supplied' in text and 'No training measurements supplied' in text
    assert 'confirmed learning: 0' in text


@pytest.mark.parametrize(('entry', 'expected'), [
    (item(), 'Correct but slower OAR'),
    (item('exploratory learning', summary='Unhelpful oracle: no accepted past proposal at the declared grid.'), 'Unhelpful oracle'),
    (item('unresolved', status='blocked', summary='INSTANT unavailable: faithful MPS port not executed.'), 'faithful MPS port not executed'),
    (item(metrics={'difference': {'value': 0.1, 'unit': 'ms', 'interval': [-2., 3.]}},
          uncertainty='Wide intervals include meaningful benefit and harm; inconclusive.'), 'Wide intervals'),
    (item('exploratory learning', summary='Rare regression observed; population risk unresolved.'), 'Rare regression'),
    (item('unresolved', status='failed', summary='Reproduction failed: numerical comparison differed.'), 'Reproduction failed'),
])
def test_reached_outcomes_keep_exact_class_scope_units_and_limits(tmp_path, entry, expected):
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates, evidence=[entry])
    text = build(gates, index, tmp_path/'report.md').read_text()
    assert expected in text and entry['scope'] in text and entry['uncertainty'] in text
    assert f"{entry['evidence_class']}: 1" in text
    assert 'No training speedup was established' in text and 'confirmed learning: 0' in text
    assert 'ms' in text and 'raw.json' in text


def test_confirmation_assertion_cannot_promote_an_exploratory_result(tmp_path):
    gates = stopped(tmp_path)
    claim = item('confirmed learning', title='SUCCESS BREAKTHROUGH', summary='Training speedup proven',
                 metrics={'speedup': {'value': 9., 'unit': 'x'}})
    text = build(gates, evidence(tmp_path, gates, evidence=[claim]), tmp_path/'report.md').read_text()
    assert 'Unsupported confirmation claim' in text and 'unresolved: 1' in text
    assert 'confirmed learning: 0' in text
    assert 'SUCCESS BREAKTHROUGH' not in text and 'Training speedup proven' not in text and '9.0' not in text


@pytest.mark.parametrize('change', [
    {'evidence_class': 'unknown'}, {'metrics': {'time': {'value': float('nan'), 'unit': 's'}}},
    {'metrics': {'time': {'value': True, 'unit': 's'}}}, {'scope': ''},
    {'metrics': {'time': {'value': 1., 'unit': ''}}},
    {'metrics': {'time': {'value': 1., 'unit': 's', 'interval': [2., -1.]}}},
    {'artifact': 'missing.json'}, {'status': 'confirmed'},
])
def test_invalid_evidence_fails_closed_without_report(tmp_path, change):
    gates = stopped(tmp_path)
    out = tmp_path/'report.md'
    with pytest.raises(ValueError):
        build(gates, evidence(tmp_path, gates, evidence=[item(**change)]), out)
    assert not out.exists()


def test_missing_values_stay_unavailable(tmp_path):
    gates = stopped(tmp_path)
    text = build(gates, evidence(tmp_path, gates, evidence=[item(metrics={'quality': {'value': None, 'unit': 'MSE/output'}})]), tmp_path/'report.md').read_text()
    assert 'quality: unavailable (MSE/output)' in text
    assert 'quality: 0' not in text


def test_hash_tampering_and_absent_artifacts_rejected(tmp_path):
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates)
    (tmp_path/'raw.json').write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        build(gates, index, tmp_path/'report.md')
    (tmp_path/'raw.json').unlink()
    with pytest.raises(ValueError, match='artifact'):
        build(gates, index, tmp_path/'report.md')


def test_prefix_binding_ignores_only_appended_synthesis_gate(tmp_path):
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates)
    first = build(gates, index, tmp_path/'report.md').read_bytes()
    with gates.open('a') as stream:
        stream.write(to_json(GateRecord('WP09', 'pass', 'Synthesis only', 'Stop', {}, {}, {}))+'\n')
    assert build(gates, index, tmp_path/'report.md').read_bytes() == first
    gates.write_text(gates.read_text().replace('CPU-only', 'Changed!'))
    with pytest.raises(ValueError, match='snapshot'):
        build(gates, index, tmp_path/'report.md')


def test_snapshot_cannot_hide_later_scientific_gate(tmp_path):
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates)
    with gates.open('a') as stream:
        stream.write(to_json(GateRecord('WP02', 'stop', 'New evidence', 'Stop', {}, {}, {}))+'\n')
    with pytest.raises(ValueError, match='scientific'):
        build(gates, index, tmp_path/'report.md')


def test_cli_report_and_invalid_index_exit(tmp_path, capsys):
    from oar.cli import main
    gates = stopped(tmp_path)
    out = tmp_path/'report.md'
    assert main(['report', '--gates', str(gates), '--out', str(out)]) == 0
    assert out.exists()
    index = evidence(tmp_path, gates, schema_version=99)
    assert main(['report', '--gates', str(gates), '--run-index', str(index), '--out', str(out)]) == 2
    assert 'invalid evidence' in capsys.readouterr().err


@pytest.mark.parametrize('target', ['gates.jsonl', 'index.json', 'raw.json'])
def test_output_cannot_overwrite_input_evidence(tmp_path, target):
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates)
    out = tmp_path/target
    before = out.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        build(gates, index, out)
    assert out.read_bytes() == before


def test_historical_revision_is_verified_without_relabeling_current_bytes(tmp_path):
    import subprocess
    root = Path('.').resolve()
    revision = subprocess.check_output(['git', 'rev-parse', '6f04c6e'], text=True).strip()
    data = subprocess.check_output(['git', 'show', f'{revision}:src/oar/cli.py'])
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates)
    payload = json.loads(index.read_text())
    payload['root'] = str(root)
    payload['artifacts'] = [dict(path='src/oar/cli.py', sha256=hashlib.sha256(data).hexdigest(), revision=revision)]
    index.write_text(json.dumps(payload))
    text = build(gates, index, tmp_path/'report.md').read_text()
    assert f'verified historical revision {revision}' in text
    assert hashlib.sha256(data).hexdigest() in text
    payload['artifacts'][0]['sha256'] = '0'*64
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='hash'):
        build(gates, index, tmp_path/'report.md')


def test_unknown_schema_duplicate_keys_and_unbound_gate_reference_rejected(tmp_path):
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates)
    payload = json.loads(index.read_text())
    payload['artifacts'][0]['path'] = '../raw.json'
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        build(gates, index, tmp_path/'report.md')
    index.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ValueError, match='duplicate'):
        build(gates, index, tmp_path/'report.md')
    record = GateRecord('WP00', 'stop', 'Missing evidence', 'Stop', {'absent': '0'*64}, {}, {})
    gates.write_text(to_json(record)+'\n')
    index = evidence(tmp_path, gates)
    with pytest.raises(ValueError, match='reference'):
        build(gates, index, tmp_path/'report.md')


def test_archived_bytes_keep_original_path_attribution(tmp_path):
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates)
    payload = json.loads(index.read_text())
    raw = payload['artifacts'][0]
    raw.update(path='original.json', archived_path='raw.json')
    index.write_text(json.dumps(payload))
    text = build(gates, index, tmp_path/'report.md').read_text()
    assert 'original.json' in text and 'verified archive raw.json' in text
    payload['evidence'] = [item(artifact='original.json')]
    index.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='current artifact'):
        build(gates, index, tmp_path/'report.md')


def test_output_cannot_overwrite_archived_input_bytes(tmp_path):
    gates = stopped(tmp_path)
    index = evidence(tmp_path, gates)
    payload = json.loads(index.read_text())
    payload['artifacts'][0].update(path='historical.json', archived_path='raw.json')
    index.write_text(json.dumps(payload))
    before = (tmp_path/'raw.json').read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        build(gates, index, tmp_path/'raw.json')
    assert (tmp_path/'raw.json').read_bytes() == before
