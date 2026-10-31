"""Rebuild the small WP09 annotated hash index, without running experiments.

The explicit scientific prefix is fixed separately from the append-only WP09
ledger. Authored annotations are not an automatic inferential analysis.
"""
import hashlib
import json
from pathlib import Path
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def resolve_artifact(root, name, digest):
    item = dict(path=name, sha256=digest)
    if name.startswith('.superpowers/'):
        return dict(item, unavailable_reason='Historical disposable workflow report; excluded from current scientific support. Durable Task11 report is decisions/reports/structural_oracle.md.')
    local = root/name
    if local.is_file() and sha(local.read_bytes()) == digest:
        return item
    for archived in sorted((root/'analysis/history').glob('*')):
        if archived.is_file() and sha(archived.read_bytes()) == digest:
            return dict(item, archived_path=str(archived.relative_to(root)))
    commits = subprocess.check_output(['git','-C',str(root),'log','--all','--format=%H','--',name],text=True).splitlines()
    for commit in commits:
        process = subprocess.run(['git','-C',str(root),'show',f'{commit}:{name}'],capture_output=True)
        if process.returncode == 0 and sha(process.stdout) == digest:
            return dict(item, revision=commit)
    raise ValueError(f'unresolved scientific artifact {name}: {digest}')


def collect_index(root=Path('.')):
    root = root.resolve()
    snapshot = json.loads((root/'analysis/scientific_gate_snapshot.json').read_text())
    lines = (root/'decisions/gates.jsonl').read_bytes().splitlines(keepends=True)
    if sha(b''.join(lines[:snapshot['lines']])) != snapshot['sha256']:
        raise ValueError('scientific prefix changed; update snapshot deliberately')
    gates = [json.loads(line) for line in lines[:snapshot['lines']]]
    if any(json.loads(line)['package'] != 'WP09' for line in lines[snapshot['lines']:]):
        raise ValueError('new scientific evidence requires a new snapshot')
    annotations = json.loads((root/'decisions/reports/synthesis_annotations.json').read_text())
    bindings = {}
    def add(name, digest):
        if (name,digest) not in bindings:
            bindings[name,digest] = resolve_artifact(root,name,digest)
    for gate in gates:
        for name, digest in gate['artifact_hashes'].items():
            add(name,digest)
    # Preserve actual measured-source attribution, including changed CLI/doc bytes.
    for name in ('timing_calibration.json','timing_calibration_wide.json','structural_oracle_summary.json'):
        raw = json.loads((root/'analysis'/name).read_text())
        sources = raw.get('provenance',raw.get('metadata',{}).get('source_sha256',{}))
        for path,digest in sources.items():
            add(path,digest)
    for name in annotations['artifacts']:
        add(name,sha((root/name).read_bytes()))
    for entry in annotations['evidence']:
        add(entry['artifact'],sha((root/entry['artifact']).read_bytes()))
    for audit in annotations['audits']:
        for name in audit['artifacts']:
            add(name,sha((root/name).read_bytes()))
    return dict(schema_version=1,root='..',gate_snapshot=snapshot,
                artifacts=[bindings[k] for k in sorted(bindings)],
                evidence=annotations['evidence'],audits=annotations['audits'])


if __name__ == '__main__':
    result = collect_index()
    Path('analysis/evidence_index.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(f"{len(result['artifacts'])} artifact/hash bindings; {len(result['evidence'])} evidence annotations; {result['gate_snapshot']['lines']} scientific gate lines")
