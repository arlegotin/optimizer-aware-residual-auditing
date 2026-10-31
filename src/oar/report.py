"""Deterministic stopped-study synthesis, with explicitly bounded evidence support.

Schema 1 is an annotated artifact index, not a confirmatory analysis engine.
It verifies bytes and attribution; scientific interpretation remains authored.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

from .records import GateRecord, from_payload

CLASSES = ('algebra/reference', 'backend timing', 'exploratory learning', 'confirmed learning', 'unresolved')


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys()-set(required)-set(optional):
        raise ValueError(f'invalid evidence fields: expected {sorted(required)}')


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('evidence text/scope/unit must be nonempty')
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('evidence values must be finite numbers')


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate evidence key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _local(root, name):
    _text(name)
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not (root/path).resolve().is_relative_to(root):
        raise ValueError('artifact path must stay within evidence root')
    return root/path


def _load_index(path):
    index = _json(path)
    _fields(index, ('schema_version', 'root', 'gate_snapshot', 'artifacts', 'evidence', 'audits'))
    if type(index['schema_version']) is not int or index['schema_version'] != 1:
        raise ValueError('unsupported evidence schema')
    root = (path.parent/_text(index['root'])).resolve()
    if not all(isinstance(index[k], list) for k in ('artifacts', 'evidence', 'audits')):
        raise ValueError('index collections must be lists')
    current = {}
    identities = set()
    for artifact in index['artifacts']:
        _fields(artifact, ('path', 'sha256'), ('revision', 'archived_path', 'unavailable_reason'))
        local = _local(root, artifact['path'])
        if not isinstance(artifact['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', artifact['sha256']):
            raise ValueError('invalid artifact hash')
        identity = artifact['path'], artifact['sha256']
        if identity in identities:
            raise ValueError('duplicate artifact identity')
        identities.add(identity)
        if sum(k in artifact for k in ('revision', 'archived_path', 'unavailable_reason')) > 1:
            raise ValueError('artifact has ambiguous historical attribution')
        if 'unavailable_reason' in artifact:
            _text(artifact['unavailable_reason'])
            if 'revision' in artifact:
                raise ValueError('unavailable artifact cannot assert revision verification')
            continue
        if 'revision' in artifact:
            if not re.fullmatch('[a-f0-9]{40}', artifact['revision']):
                raise ValueError('historical artifact requires full source revision')
            try:
                data = subprocess.check_output(['git', '-C', str(root), 'show', f"{artifact['revision']}:{artifact['path']}"], stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:
                raise ValueError(f'unavailable historical artifact: {artifact["path"]}') from exc
        elif 'archived_path' in artifact:
            try:
                data = _local(root, artifact['archived_path']).read_bytes()
            except OSError as exc:
                raise ValueError('unavailable archived artifact') from exc
        else:
            try:
                data = local.read_bytes()
            except OSError as exc:
                raise ValueError(f'unavailable artifact: {artifact["path"]}') from exc
            if artifact['path'] in current:
                raise ValueError('ambiguous current artifact')
            current[artifact['path']] = local
        if _digest(data) != artifact['sha256']:
            raise ValueError(f'artifact hash mismatch: {artifact["path"]}')
    for entry in index['evidence']:
        _fields(entry, ('evidence_class', 'title', 'artifact', 'status', 'scope', 'summary', 'uncertainty', 'metrics'))
        if entry['evidence_class'] not in CLASSES or entry['status'] not in ('measured', 'blocked', 'failed', 'unresolved'):
            raise ValueError('unsupported evidence class/status')
        if entry['artifact'] not in current:
            raise ValueError('evidence requires a verified current artifact reference')
        for key in ('title', 'scope', 'summary', 'uncertainty'):
            _text(entry[key])
        if not isinstance(entry['metrics'], dict):
            raise ValueError('metrics must be an object')
        for name, metric in entry['metrics'].items():
            _text(name)
            _fields(metric, ('value', 'unit'), ('interval',))
            _text(metric['unit'])
            if metric['value'] is not None:
                _number(metric['value'])
            if 'interval' in metric:
                interval = metric['interval']
                if not isinstance(interval, list) or len(interval) != 2 or metric['value'] is None:
                    raise ValueError('invalid evidence interval')
                for value in interval:
                    _number(value)
                if interval[0] > interval[1]:
                    raise ValueError('reversed evidence interval')
    for audit in index['audits']:
        _fields(audit, ('kind', 'summary', 'artifacts'))
        if audit['kind'] not in ('scientific', 'experimental', 'implementation', 'decision', 'reproduction'):
            raise ValueError('unsupported challenge audit')
        _text(audit['summary'])
        if not isinstance(audit['artifacts'], list) or not audit['artifacts'] or any(p not in current for p in audit['artifacts']):
            raise ValueError('audit requires verified artifact references')
    return index, current


def build_report(gates: Path, run_index: Path | None, out: Path) -> Path:
    """Render honest reached-stage evidence; never mutate gates or infer a pass.

    A bound prefix may omit only subsequent WP09 synthesis records. Schema 1
    cannot validate confirmation protocols and reclassifies those claims unresolved.
    Invalid inputs fail before writing output, including nonfinite metrics.
    """
    gates, out = Path(gates), Path(out)
    lines = gates.read_bytes().splitlines(keepends=True)
    records = [from_payload(json.loads(line)) for line in lines]
    if not records or not all(isinstance(r, GateRecord) for r in records):
        raise ValueError('expected nonempty gate ledger')
    index, current = _load_index(Path(run_index)) if run_index is not None else (None, {})
    snapshot = None
    if index:
        snapshot = index['gate_snapshot']
        _fields(snapshot, ('lines', 'sha256'))
        n = snapshot['lines']
        if type(n) is not int or not 0 < n <= len(lines) or _digest(b''.join(lines[:n])) != snapshot['sha256']:
            raise ValueError('gate snapshot hash/length mismatch')
        if any(r.package != 'WP09' for r in records[n:]):
            raise ValueError('later scientific gate requires refreshed snapshot')
        records = records[:n]
        identities = {(a['path'], a['sha256']) for a in index['artifacts']}
        for record in records:
            if any((p, h) not in identities for p, h in record.artifact_hashes.items()):
                raise ValueError('gate artifact reference absent from index')
    protected = {gates.resolve(), *(p.resolve() for p in current.values())}
    if run_index is not None:
        protected.add(Path(run_index).resolve())
        root = (Path(run_index).parent/index['root']).resolve()
        for artifact in index['artifacts']:
            protected.add((root/artifact['path']).resolve())
            if 'archived_path' in artifact:
                protected.add((root/artifact['archived_path']).resolve())
    if out.resolve() in protected:
        raise ValueError('report output would overwrite input evidence')
    entries = index['evidence'] if index else []
    classified = {kind: [] for kind in CLASSES}
    for entry in entries:
        kind = entry['evidence_class']
        if kind == 'confirmed learning' or entry['status'] != 'measured':
            kind = 'unresolved'
        classified[kind].append(entry)
    def link(path):
        return f'[{path}]({Path(os.path.relpath(current[path], out.parent.resolve())).as_posix()})'
    text = ['# OAR reached-stage evidence and execution outcome', '',
            '**No training speedup was established.** All supported measurements are exploratory or component/reference evidence. '
            'Training quality, time-to-quality, complete training memory and independent confirmation are unavailable unless explicitly measured; '
            'this schema supplies no confirmation validator.', '']
    if not index:
        text += ['No run index supplied. No training measurements supplied. Gate reasons below are records, not newly verified measurements.', '']
    else:
        text += [f"Evidence binds the first {snapshot['lines']} gate lines, SHA256 `{snapshot['sha256']}`. "
                 'Later WP09 records do not enter this scientific snapshot. Artifact annotations are authored interpretations; hashes establish attribution, not scientific validation.', '']
        # Lead with the actual trade-off before long ledger/history tables.
        for kind in ('backend timing', 'exploratory learning'):
            for entry in classified[kind]:
                text += [f"{entry['summary']} Scope: {entry['scope']} Uncertainty: {entry['uncertainty']} Source: {link(entry['artifact'])}.", '']
    text += ['## Evidence classes', '']
    for kind in CLASSES:
        text += [f'{kind}: {len(classified[kind])}', '']
        for entry in classified[kind]:
            if entry['evidence_class'] == 'confirmed learning':
                text += [f"Unsupported confirmation claim; classified unresolved. Schema 1 cannot establish protected evaluation, independent replication or a valid protocol. Source: {link(entry['artifact'])}.", '']
                continue
            text += [f"### {entry['title']}", '', f"Status: {entry['status']}. {entry['summary']}", '',
                     f"Scope: {entry['scope']} Uncertainty: {entry['uncertainty']}", '']
            for name, metric in entry['metrics'].items():
                value = 'unavailable' if metric['value'] is None else str(metric['value'])
                interval = f"; interval {metric['interval']}" if 'interval' in metric else ''
                text += [f"- {name}: {value} ({metric['unit']}){interval}"]
            text += ['', f"Source: {link(entry['artifact'])}.", '']
    latest = {r.package: i for i, r in enumerate(records, 1)}
    text += ['## All gate outcomes, including superseded stops', '']
    for i, record in enumerate(records, 1):
        state = 'latest in snapshot' if latest[record.package] == i else 'superseded record retained'
        text += [f'### Gate {i}: {record.package} — {record.status} ({state})', '', record.reason, '', f'Next action: {record.next_action}', '']
    absent = [f'WP{i:02d}' for i in range(9) if f'WP{i:02d}' not in latest]
    text += [f"No recorded gate outcome: {', '.join(absent) or 'none'}. Absence never implies completion.", '',
             '## Outcome decision and challenge audits', '',
             'Blueprint §14: stop the current online branch where a prerequisite is stopped or inconclusive; narrow conclusions to reached measurements. '
             'A component benefit alone does not establish learning efficiency. Wide intervals are inconclusive, not equivalence. '
             'Continue/revise/pivot requires the applicable evidence and a new specification; synthesis grants no experimental permission.', '']
    if index:
        for audit in index['audits']:
            text += [f"### {audit['kind'].capitalize()} audit", '', audit['summary'], '',
                     'Sources: '+', '.join(link(p) for p in audit['artifacts'])+'.', '']
        text += ['## Raw artifact and historical hash index', '',
                 'Current artifacts are checked from disk; historical revisions are checked using `git show REV:PATH`. '
                 'Explicitly unavailable historical artifacts stay unavailable. No index/report/WP09 hash cycle is used.', '']
        for a in index['artifacts']:
            attribution = ('unavailable: '+a['unavailable_reason'] if 'unavailable_reason' in a else
                           'verified historical revision '+a['revision'] if 'revision' in a else
                           'verified archive '+a['archived_path'] if 'archived_path' in a else 'verified current bytes')
            label = link(a['path']) if a['path'] in current and 'revision' not in a and 'archived_path' not in a and 'unavailable_reason' not in a else f"`{a['path']}`"
            text += [f"- {label}: `{a['sha256']}` ({attribution})"]
        text += ['']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(text)+'\n', encoding='utf-8')
    return out
