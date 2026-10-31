# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path
import hashlib, importlib.util, json, platform, subprocess, sys
root = Path.cwd()
run = root/'analysis/dw_sampling/20261031T105213Z'
def sha(data): return hashlib.sha256(data).hexdigest()
def digest(path): return sha(path.read_bytes())
spec = importlib.util.spec_from_file_location('reviewed_driver',root/'analysis/run_dw_sampling.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
assert driver.ROOT == root.resolve()
cfg = driver.load_config(root/'configs/dw_sampling.yaml')
amendment = driver.validate_amendment(root/'configs/dw_sampling.yaml')
initial = json.loads((run/'initial_inspection.json').read_text())
historical = initial['historical_tracked_sha256']
reviewed = '9d8ad069c51b4d0e6c13e9c6e465579e9b44a339'
tracked_baseline = subprocess.check_output(['git','ls-tree','-r','--name-only',reviewed],text=True).splitlines()
assert set(tracked_baseline) == set(historical)
assert len(historical) == 184
for name, expected in historical.items():
    assert digest(root/name) == expected, name
    assert sha(subprocess.check_output(['git','show',reviewed+':'+name])) == expected, name
records = []
for path in sorted(run.glob('*/attempt-*/record.json')):
    r = driver.read_record(path)
    sm = json.loads((path.parent/'source_manifest.json').read_text())
    assert r['source_manifest_sha256'] == digest(path.parent/'source_manifest.json')
    assert r['measurement_hashes'] == sm['measurement_hashes']
    assert all(sm['bound_files'][n] == h for n,h in r['measurement_hashes'].items())
    # Recover the amendment by its content hash, including retained source revisions.
    amendments = [root/'decisions/dw_sampling_amendment.json',*sorted((run/'frozen_versions').glob('*.json'))]
    matched = [p for p in amendments if digest(p) == sm['amendment_sha256']]
    assert len(matched)==1, str(path)
    assert json.loads(matched[0].read_text())['bound_files'] == sm['bound_files']
    git_source = {}
    for name, expected in sm['bound_files'].items():
        observed = sha(subprocess.check_output(['git','show',sm['actual_head']+':'+name]))
        assert observed == expected, (str(path),name,expected,observed)
        git_source[name] = observed
    artifacts_present = {str(p.relative_to(path.parent)) for p in path.parent.rglob('*') if p.is_file() and p.name not in ('record.json','record.sha256')}
    assert artifacts_present == set(r['artifacts']), str(path)
    records.append(dict(path=str(path.relative_to(root)),status=r['status'],record_sha256=digest(path),artifact_count=len(r['artifacts']),source_head=sm['actual_head'],source_git_files_verified=len(git_source),amendment_sha256=sm['amendment_sha256']))
summary = json.loads((run/'summarize/attempt-0002/summary.json').read_text())
assert summary['terminal_state'] == 'INCONCLUSIVE_MEASUREMENT'
assert summary['new_performance_claim'] is False
assert summary['primary_time_to_target'] is None
assert len(summary['records']) == 5
for name in ('cost','pilot','confirm'):
    assert not (run/name).exists(), name
# These checks consume prerequisites without executing a stage or allocating artifacts.
blocked = {}
for stage in ('prepare','feasibility'):
    try:
        driver.require_stage(run,stage,driver.source_hashes('cost'))
    except ValueError as exc:
        blocked[stage]=str(exc)
    else:
        raise AssertionError('current branch unexpectedly accepts cost prerequisite '+stage)
assert 'source/config mismatch' in blocked['prepare']
assert 'INCONCLUSIVE_MEASUREMENT' in blocked['feasibility']
# No tracked checkout changes from tests/help/render/summarize.
assert subprocess.check_output(['git','diff','HEAD','--name-only'],text=True) == ''
report_path = root.parent/'historical-rendered.md'
print(json.dumps(dict(checkout=str(root),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),historical_files_verified=len(historical),amendment_bound_files_verified=len(amendment['bound_files']),current_source_hashes=driver.source_hashes('summarize'),amendment_sha256=digest(root/'decisions/dw_sampling_amendment.json'),historical_inventory_sha256=digest(run/'initial_inspection.json'),sealed_records_verified=len(records),sealed_artifacts_verified=sum(r['artifact_count'] for r in records),records=records,summary_terminal=summary['terminal_state'],summary_records=len(summary['records']),blocked_prerequisite_checks=blocked,historical_render_sha256=digest(report_path),historical_render_matches_archived_report=digest(report_path)==digest(root/'analysis/report.md'),tracked_diff='',hardware_identification=dict(platform=platform.platform(),machine=platform.machine()),limitations='CPU-only same-machine/day/operator review; shared pinned environment; no independent performance or learning replication'),indent=2))
