# SPDX-License-Identifier: AGPL-3.0-only
"""One-time prospective freeze for code-only negative-result closeout."""
import argparse,datetime,hashlib,importlib.util,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--review-sha',required=True);args=p.parse_args()
root=pathlib.Path.cwd();run=root/'analysis/dw_sampling/20261031T120154Z-reopened';pros=run/'prospective'
def h(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def put(path,data):
    if path.exists():assert path.read_bytes()==data,'immutable artifact differs: '+str(path)
    else:
        with path.open('xb') as f:f.write(data)
def save(path,value):put(path,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert head=='5361bb59aedccf12af66314f15391c5af60ad595'
review_source=root/'.superpowers/dw_sampling_reopening/version10-recovery-review.md'
assert h(review_source)==args.review_sha
review=pros/'version10-recovery-review.md';put(review,review_source.read_bytes())
package=pros/'version10-recovery-review-package.txt';put(package,(root/'.superpowers/dw_sampling_reopening/version10-recovery-review-package.txt').read_bytes())
for name in ('math-sealed-outcome-review.md','math-sealed-outcome-check.json'):
    put(pros/('version10-'+name),(root/'.superpowers/dw_sampling_reopening'/name).read_bytes())
base=json.loads((pros/'frozen-amendment-v9.json').read_text());assert base['version']==9
assert (root/'decisions/dw_sampling_amendment.json').read_bytes()==(pros/'frozen-amendment-v9.json').read_bytes()
spec=importlib.util.spec_from_file_location('dw_closeout_freeze',root/'analysis/run_dw_sampling.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
config=root/'configs/dw_sampling.yaml';cfg=d.load_config(config);current=d.source_hashes('pilot')
for name,want in current.items():assert hashlib.sha256(subprocess.check_output(['git','show',head+':'+name])).hexdigest()==want,name
for name,want in base['bound_files'].items():
    if name not in current:assert h(root/name)==want,name
oldfiles=subprocess.check_output(['git','ls-tree','-r','--name-only',base['reviewed_head']],text=True).splitlines();assert len(oldfiles)==184
historical={}
for name in oldfiles:
    original=subprocess.check_output(['git','show',base['reviewed_head']+':'+name]);assert (root/name).read_bytes()==original,name
    historical[name]=hashlib.sha256(original).hexdigest()
e4=run/'pilot/attempt-0004/record.json';record=d.read_record(e4)
assert h(e4)=='3e15fe5e4526ae001b5112a896b69d43669a2044aa60f4febc7c5ee05125f7bd'
assert record['status']=='STOP_QUALITY' and record['candidate'] is None and not record['eligible_candidates']
qpath=root/cfg['learning']['qualification_record_path'];q=d.read_record(qpath);qsm=qpath.parent/'source_manifest.json';qid=json.loads(qsm.read_text())
freeze={'schema_version':1,'version':10,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reviewed_head':base['reviewed_head'],'source_head':head,'source_hashes':current,'config_sha256':h(config),
        'previous_authorization_sha256':h(pros/'frozen-amendment-v9.json'),'backend_review_sha256':h(review),
        'historical_tracked_sha256':historical,'sealed_learning_record_sha256':h(e4),
        'authority':'code-only STOP_QUALITY closeout; no new training or protected losses',
        'resource_forecast_s':{'prepare':120,'correctness':60},'deadline_multiplier':2,
        'forecast_basis':'prior complete preparation max62.058s/correctness20.092s plus22 new recovery cases; startup/serialization included; no expensive learning',
        'measurement_scope':'current correctness/backend checks only; old uninterrupted v9 learning and B/D/model timing retain original measured identities'}
freeze_path=pros/'version10-source-freeze.json';save(freeze_path,freeze)
a=base.copy();a.update(version=10,frozen_utc=freeze['utc'],actual_head_at_freeze=head,
    previous_authorization_sha256=h(pros/'frozen-amendment-v9.json'),
    authorization='Owner-authorized engineering completion after sealed STOP_QUALITY. Repair admitted environmental-recovery preparation accounting, retain failures/unknown intervals; verify current CPU/MPS correctness, independently review and summarize. No new grid, protected confirmation/test or algorithm expansion; no retuning of frozen v9 science.',
    config_sha256=h(config),reopening_scope_sha256=h(root/'decisions/dw_sampling_reopening.md'),carryover_review_sha256=h(review),
    carryover_scope='Exact unchanged B/D and v7 model/native qualification paths; old observations retain original source identities. Current freeze authorizes code-only checks, never upgrades failed E or grants changed-source learning.')
a['bound_files']={name:(current[name] if name in current else want) for name,want in base['bound_files'].items()}
extras=[e4,e4.parent/'source_manifest.json',e4.parent/'grid-results.json',review,package,freeze_path,
        pros/'version10-math-sealed-outcome-review.md',pros/'version10-math-sealed-outcome-check.json']
extras.extend(x for x in (pros/'version10-recovery-accounting').rglob('*') if x.is_file())
extras.extend(x for x in (run/'closeout/attempt-0001').glob('*') if x.is_file())
extras.extend(x for x in (run/'closeout/metadata-replay-0001').glob('*') if x.is_file())
for path in extras:a['bound_files'][str(path.relative_to(root))]=h(path)
for entry in a['prerequisite_carryover']:entry['current_measurement_hashes']=current
changed=sorted(name for name in q['measurement_hashes'] if q['measurement_hashes'][name]!=current[name])
a['model_qualification_continuity']={'status':'approved','record_path':str(qpath.relative_to(root)),
    'record_sha256':h(qpath),'source_manifest_sha256':h(qsm),'measured_head':qid['actual_head'],
    'previous_measurement_hashes':q['measurement_hashes'],'current_measurement_hashes':current,'changed_files':changed,
    'executed_paths_unchanged':True,'initializers_and_transitive_callers_reviewed':True,
    'reviews':[{'path':str(review.relative_to(root)),'sha256':h(review)}]}
for stage in ('prepare','correctness'):a['prerequisites'][stage]='Current code-only freeze and actual checks; no body acquisition or real learning; retain all failures.'
a['prerequisites']['pilot']='STOP_QUALITY at immutable v9 E4. Archived v9 manifest is not changed-source training authority; no new learning in this closeout.'
a['prerequisites']['confirm']='Not reached: actual E failed, no selected sampler/fixedN/confirmation freeze; protected validation/test stay sealed.'
a['allowed_commands']=[f'PYTHONPATH=src .venv/bin/python analysis/run_dw_sampling.py --stage {stage} --config configs/dw_sampling.yaml --out analysis/dw_sampling/20261031T120154Z-reopened' for stage in ('prepare','correctness','summarize')]
a['allowed_commands']+=['PYTHONPATH=src .venv/bin/python analysis/run_dw_sampling.py --help','PYTHONPATH=src .venv/bin/python analysis/run_dw_sampling.py --stage prepare --config configs/dw_sampling.yaml --out <new-validation-root> --dry-run',"PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'",'PYTHONPATH=src .venv/bin/python analysis/dw_sampling/20261031T120154Z-reopened/closeout/attempt-0001/reproduce-recorded-grid.py --out <fresh-output-directory>']
a['learning_scope']={**base['learning_scope'],'status':'STOP_QUALITY','authority':'archived v9 execution only; no current changed-source learning admission','sealed_record_path':str(e4.relative_to(root)),'sealed_record_sha256':h(e4),'new_scientific_runs_authorized':False}
a['closeout']={'source_freeze_path':str(freeze_path.relative_to(root)),'source_freeze_sha256':h(freeze_path),'forecast_s':freeze['resource_forecast_s'],'deadline_multiplier':2,'pre_admission_boundary':'unchanged shared static metadata/ownership; in whole-stage cost, outside worker trajectory; admitted recovery paid before allocation/authentication','training_repetition':False,'public_metadata_replay':True,'future_private_hydration_required_for_relocated_training':True}
a['source_commit_mapping']='Exact5361bb59 source/scope/test bytes precede this metadata-only freeze commit. New A/C retain actual execution HEAD and direct hashes; sealed v9 data remains measured89c5bd8/launch43c2cec. No new-source performance claim.'
save(pros/'frozen-amendment-v10.json',a);(root/'decisions/dw_sampling_amendment.json').write_bytes((pros/'frozen-amendment-v10.json').read_bytes())
d.validate_amendment(config)
for stage in ('feasibility','cost'):d.require_stage(run,stage,current)
d.require_model_qualification(run,qpath,h(qpath),current)
try:d.load_learning_freeze(cfg,config,run)
except ValueError:pass
else:raise AssertionError('archived v9 manifest must not authorize changed-source learning')
print(json.dumps({'version':10,'source_head':head,'historical_files_unchanged':len(historical),'bound_files':len(a['bound_files']),'amendment_sha256':h(root/'decisions/dw_sampling_amendment.json'),'source_hashes':current,'new_learning_authorized':False},indent=2))
