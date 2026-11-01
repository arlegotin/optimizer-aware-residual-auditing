# SPDX-License-Identifier: AGPL-3.0-only
"""Independent immutable Git/file/environment verification; reads public bytes only."""
from pathlib import Path
import hashlib, importlib.util, importlib.metadata, json, platform, subprocess, sys
state=json.loads((Path(__file__).parent/'state.json').read_text())
root=Path.cwd()
assert str(root)==state['clone']
def git(*args):return subprocess.check_output(['git',*args],cwd=root)
def sha(data):return hashlib.sha256(data).hexdigest()
head=git('rev-parse','HEAD').decode().strip()
assert head==state['head']
assert git('rev-parse','--is-shallow-repository').strip()==b'false'
assert not (root/'.git/objects/info/alternates').exists()
assert all(p.stat().st_nlink==1 for p in (root/'.git/objects').rglob('*') if p.is_file())
assert git('status','--porcelain')==b''
run=root/'analysis/dw_sampling/20261031T120154Z-reopened'
freeze_path=run/'prospective/version10-source-freeze.json'
freeze=json.loads(freeze_path.read_text())
reviewed=freeze['reviewed_head'];source=freeze['source_head']
assert reviewed=='9d8ad069c51b4d0e6c13e9c6e465579e9b44a339'
assert source=='5361bb59aedccf12af66314f15391c5af60ad595'
git('cat-file','-e',reviewed+'^{commit}')
names=[n.decode() for n in git('ls-tree','-rz','--name-only',reviewed).split(b'\0') if n]
assert len(names)==184
assert set(names)==set(freeze['historical_tracked_sha256'])
historical={}
for name in names:
 old=git('show',reviewed+':'+name);current=(root/name).read_bytes()
 assert old==current,name
 assert sha(current)==freeze['historical_tracked_sha256'][name],name
 historical[name]=sha(current)
assert len(freeze['source_hashes'])==11
for name,expected in freeze['source_hashes'].items():
 current=(root/name).read_bytes()
 assert sha(current)==expected,name
 assert current==git('show',source+':'+name)==git('show',head+':'+name),name
amendment=json.loads((root/'decisions/dw_sampling_amendment.json').read_text())
for name,expected in amendment['bound_files'].items():assert sha((root/name).read_bytes())==expected,name
assert amendment['closeout']['source_freeze_sha256']==sha(freeze_path.read_bytes())
assert amendment['learning_scope']['new_scientific_runs_authorized'] is False
record_path=run/'pilot/attempt-0004/record.json';record=json.loads(record_path.read_text())
assert sha(record_path.read_bytes())==freeze['sealed_learning_record_sha256']==record_path.with_suffix('.sha256').read_text().strip()
for name,expected in record['artifacts'].items():
 p=(record_path.parent/name).resolve();assert p.is_relative_to(record_path.parent)
 assert sha(p.read_bytes())==expected,name
spec=importlib.util.find_spec('oar');assert Path(spec.origin).resolve().is_relative_to(root/'src')
assert not (root/'.cache/dw_sampling').exists()
report_names=['scientific_report-draft-v1.md','prospective/version10-recovery-review.md','prospective/version10-math-sealed-outcome-review.md','closeout/attempt-0001/reproduce-recorded-grid.py']
result={'status':'pass','head':head,'source_head':source,'reviewed_head':reviewed,'full_history':True,'no_hardlinks':True,'git_status':'','historical_files_count':len(names),'historical_tracked_sha256':historical,'source_hashes':freeze['source_hashes'],'bound_files_count':len(amendment['bound_files']),'bound_files':amendment['bound_files'],'sealed_public_artifact_count':len(record['artifacts']),'sealed_record_sha256':sha(record_path.read_bytes()),'report_sha256':{n:sha((run/n).read_bytes()) for n in report_names},'shared_environment':{'python_executable':sys.executable,'python_version':platform.python_version(),'dependency_versions':{n:importlib.metadata.version(n) for n in ['torch','numpy','PyYAML','psutil','pytest','pyarrow']},'package_origin':spec.origin},'clone_has_private_data_cache':False,'training_repeated':False}
with (Path(state['public'])/'binding-verification.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['historical_tracked_sha256','source_hashes','bound_files']},indent=2))
