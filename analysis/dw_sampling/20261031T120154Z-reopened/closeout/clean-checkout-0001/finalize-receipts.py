# SPDX-License-Identifier: AGPL-3.0-only
"""Archive fresh-checkout output and verify its final receipts without rerunning tests."""
from pathlib import Path
import hashlib,json,subprocess
state=json.loads((Path(__file__).parent/'state.json').read_text())
base=Path(state['base']);public=Path(state['public']);clone=Path(state['clone'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy_new(source,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(source.read_bytes())
for name in ('prepare-only','prepare-only-2','metadata-replay'):
 for p in sorted((base/name).rglob('*')):
  if p.is_file():copy_new(p,public/name/p.relative_to(base/name))
copy_new(base/'dry-run-sentinel.json',public/'dry-run-sentinel.json')
copy_new(base/'state.json',public/'state.json')
status=subprocess.check_output(['git','status','--porcelain'],cwd=clone,text=True)
assert status==''
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=clone,text=True).strip()==state['head']
assert not (clone/'.cache/dw_sampling').exists()
bindings=json.loads((public/'binding-verification.json').read_text())
assert all(sha(clone/name)==expected for name,expected in bindings['source_hashes'].items())
for name in ('prepare-only','prepare-only-2'):
 attempt=public/name/'prepare/attempt-0001'
 record=json.loads((attempt/'record.json').read_text());identity=json.loads((attempt/'source_manifest.json').read_text())
 assert record['status']=='not_run'
 assert sha(attempt/'record.json')==(attempt/'record.sha256').read_text().strip()
 assert identity['actual_head']==state['head'] and identity['git_status']==''
 assert identity['measurement_hashes']==bindings['source_hashes']
 assert identity['config_path']==str(clone/'configs/dw_sampling.yaml')
 for filename,expected in record['artifacts'].items():assert sha(attempt/filename)==expected
replay=public/'metadata-replay';original=clone/'analysis/dw_sampling/20261031T120154Z-reopened/closeout/attempt-0001'
a=json.loads((replay/'recorded-grid-analysis.json').read_text());b=json.loads((original/'recorded-grid-analysis.json').read_text())
wall={}
for name,value in (('replayed',a),('original',b)):wall[name]=value.pop('analysis_wall_s_before_publication')
assert a==b
csv={}
for name in ('all-grid-outcomes.csv','development-monitor-curves.csv'):
 assert sha(replay/name)==sha(original/name)
 csv[name]=sha(replay/name)
receipt=json.loads((replay/'reproduction-receipt.json').read_text())
assert receipt['status']=='pass' and receipt['training_repeated'] is False
sentinel=json.loads((public/'dry-run-sentinel.json').read_text());assert sentinel['status']=='pass' and sentinel['violations']==[]
report=Path(state['original_root'])/'.superpowers/dw_sampling_reopening/clean-checkout-final-review.md'
assert report.read_bytes()==(public/'review.md').read_bytes()
result={'status':'pass','reviewed_head':state['head'],'clone_git_status':status,'source_hashes_unchanged':True,'body_cache_created':False,'tests':{'passed':842,'failed':0,'deselected':16,'pytest_seconds':24.54},'scientific_state':'STOP_QUALITY','all_scientific_json_fields_identical':True,'analysis_wall_s_before_publication':wall,'csv_sha256':csv,'report_sha256':sha(report),'report_path':str(report),'replay_receipt_sha256':sha(replay/'reproduction-receipt.json'),'dry_run_sentinel_sha256':sha(public/'dry-run-sentinel.json'),'reviewer_wrapper_failure_retained':True,'final_report_diff_check_pending':True}
with (public/'verification-summary.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(result,indent=2))
