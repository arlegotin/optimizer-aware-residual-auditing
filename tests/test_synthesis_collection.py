"""Collector must distinguish current bytes from historical attribution."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess


def collector():
    path = Path('analysis/collect_synthesis.py')
    spec = importlib.util.spec_from_file_location('collect_synthesis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_evidence_finds_exact_history_without_relabeling():
    module = collector()
    root = Path('.').resolve()
    revision = subprocess.check_output(['git', 'rev-parse', '6f04c6e'], text=True).strip()
    data = subprocess.check_output(['git', 'show', f'{revision}:src/oar/cli.py'])
    digest = hashlib.sha256(data).hexdigest()
    resolved = module.resolve_artifact(root, 'src/oar/cli.py', digest)
    assert resolved['sha256'] == digest and 'revision' in resolved
    assert resolved['path'] == 'src/oar/cli.py'
    assert 'unavailable_reason' not in resolved


def test_disposable_historical_workflow_is_explicitly_unavailable():
    resolved = collector().resolve_artifact(Path('.').resolve(), '.superpowers/removed-report.md', '0'*64)
    assert resolved['sha256'] == '0'*64 and 'unavailable_reason' in resolved
    assert 'revision' not in resolved


def test_uncommitted_historical_bytes_resolve_to_hash_identical_archive():
    resolved = collector().resolve_artifact(Path('.').resolve(), 'analysis/evidence_history.json',
        '53c7cfcab0a5edfa66f841c2073e7d2d5b9b93593603d4147008e75ea5e0a860')
    assert resolved['archived_path'] == 'analysis/history/evidence-index-before-layout-53c7cfcab0a5.json'
    assert 'revision' not in resolved
