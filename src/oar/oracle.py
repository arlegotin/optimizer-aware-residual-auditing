"""Bounded synthetic research scaffolding; no online or general trainer authority.

All loss/finite deltas are MSE per output element and counts are observations.
Exact gradients, CPU FP64 decompositions and copied-state work are paid offline.
"""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.nn import functional as F
import yaml

from .audit import residual_bound
from .basis import BasisState, build_basis
from .bench import CostTable, native_state
from .config import _fields, _UniqueLoader
from .gates import GateBlocked
from .linear import LinearCapture, projected_linear
from .optimizer_preview import preview_adamw
from .records import AdamWGroup, AdamWParamState, ReplayableBatch, GateRecord, from_payload


class StudyIncomplete(RuntimeError):
    """Bounded resource stop; valid incomplete evidence, never a successful prefix."""


@dataclass(frozen=True)
class OracleConfig:
    schema_version: int
    phase: str
    synthetic_only: bool
    loss_unit: str
    device: str
    stage: str
    shape: dict
    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    checkpoints: tuple[int, ...]
    ages: tuple[int, ...]
    ranks: tuple[int, ...]
    teacher_rank: int
    shift_step: int
    pilot_steps: int
    learning_rates: tuple[float, ...]
    selected_learning_rate: float | None
    taus: tuple[float, ...]
    relative_thresholds: tuple[float, ...]
    queries_per_split: int
    max_wall_s: float
    output: str
    checkpoints_dir: str


def parse_oracle_config(payload) -> OracleConfig:
    r = dict(_fields(payload, {f.name for f in fields(OracleConfig)}, 'structural oracle'))
    if (type(r['schema_version']) is not int or r['schema_version'] != 1 or
            r['phase'] != 'exploratory' or r['synthetic_only'] is not True or
            r['loss_unit'] != 'mse_per_output_element' or r['device'] not in ('cpu', 'mps') or
            r['stage'] not in ('pilot', 'study')):
        raise ValueError('require schema1 exploratory synthetic MSE oracle on cpu/mps')
    _fields(r['shape'], {'n', 'di', 'do'}, 'shape')
    def integer(v): return type(v) is int and 0 < v < 2**31
    def positive(v): return type(v) in (int, float) and math.isfinite(v) and v > 0
    if not all(integer(v) for v in r['shape'].values()): raise ValueError('invalid shape')
    for k in ('seeds', 'checkpoints', 'ages', 'ranks'):
        if (not isinstance(r[k], (list, tuple)) or not r[k] or
                not all(integer(v) for v in r[k]) or len(set(r[k])) != len(r[k])):
            raise ValueError(f'invalid {k}')
        r[k] = tuple(r[k])
    if list(r['checkpoints']) != sorted(r['checkpoints']): raise ValueError('checkpoints must increase')
    if max(r['ranks']) > min(r['shape']['di'], r['shape']['do']): raise ValueError('rank exceeds dimension')
    for k in ('teacher_rank', 'shift_step', 'pilot_steps', 'queries_per_split'):
        if not integer(r[k]): raise ValueError(f'invalid {k}')
    if r['teacher_rank'] > min(r['shape']['di'], r['shape']['do']): raise ValueError('invalid teacher rank')
    if (not isinstance(r['conditions'], (list, tuple)) or not r['conditions'] or
            len(set(r['conditions'])) != len(r['conditions']) or
            not set(r['conditions']) <= {'low_rank', 'full_rank', 'direction_shift'}):
        raise ValueError('invalid conditions')
    r['conditions'] = tuple(r['conditions'])
    for k in ('learning_rates', 'taus', 'relative_thresholds'):
        if not isinstance(r[k], (list, tuple)) or not r[k] or not all(positive(v) for v in r[k]):
            raise ValueError(f'invalid {k}')
        r[k] = tuple(r[k])
    if any(v >= 1 for v in r['taus']): raise ValueError('tau must be below1')
    lr = r['selected_learning_rate']
    if lr is not None and (not positive(lr) or lr not in r['learning_rates']): raise ValueError('invalid selected learning rate')
    if not positive(r['max_wall_s']): raise ValueError('invalid time cap')
    for k in ('output', 'checkpoints_dir'):
        if not isinstance(r[k], str) or not r[k].strip() or '\x00' in r[k]: raise ValueError('invalid output path')
    return OracleConfig(**r)


def load_oracle_config(path: Path) -> OracleConfig:
    try:
        payload = yaml.load(Path(path).read_text(), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError('invalid YAML configuration') from exc
    return parse_oracle_config(payload)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_sha(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require_structural_oracle(config, config_path, *, root=Path('.'), command='audit-checkpoints'):
    """Narrow command permission; never changes require_gate or normal prerequisites."""
    if command != 'audit-checkpoints': raise GateBlocked('exception belongs only to oracle command')
    parse_oracle_config(asdict(config))
    try:
        if config != load_oracle_config(config_path): raise ValueError('config object does not match hashed file')
        ledger = [json.loads(s) for s in (root/'decisions/gates.jsonl').read_text().splitlines()]
        if not all(isinstance(from_payload(r), GateRecord) for r in ledger): raise ValueError('invalid gate ledger')
        latest = {r['package']: r for r in ledger}
        amendment = json.loads((root/'decisions/amendments.jsonl').read_text().splitlines()[-1])
        _fields(amendment, {'schema_version', 'kind', 'command', 'scope_sha256', 'config_sha256',
                           'wp02_record_sha256', 'evidence_hashes'}, 'oracle amendment')
        if (amendment['schema_version'] != 1 or amendment['kind'] != 'synthetic_structural_oracle' or
                amendment['command'] != command or amendment['config_sha256'] != sha(config_path) or
                amendment['scope_sha256'] != sha(root/'decisions/structural_oracle_scope.md') or
                amendment['wp02_record_sha256'] != canonical_sha(latest['WP02'])):
            raise ValueError('amendment hash or command mismatch')
        if latest['WP00']['status'] != 'pass' or latest['WP01']['status'] != 'pass' or latest['WP02']['status'] != 'stop':
            raise ValueError('entry correctness/stop evidence changed')
        required = {'analysis/headroom_decision.json', 'analysis/timing_calibration.json',
                    'analysis/operator_correctness.json', 'analysis/audit_confidence.json',
                    'analysis/optimizer_correctness.json'}
        if not required <= amendment['evidence_hashes'].keys(): raise ValueError('missing evidence bindings')
        for name, digest in amendment['evidence_hashes'].items():
            if name not in latest['WP01']['artifact_hashes'] and name not in latest['WP02']['artifact_hashes']:
                raise ValueError('evidence not bound by genuine entry gate')
            expected = latest['WP02']['artifact_hashes'].get(name, latest['WP01']['artifact_hashes'].get(name))
            if digest != expected or sha(root/name) != digest: raise ValueError('evidence hash mismatch')
        decision = json.loads((root/'analysis/headroom_decision.json').read_text())
        matched = [r for r in decision['rows'] if r['shape']['n'] == config.shape['n'] and
                   r['shape']['di'] == config.shape['di'] and r['shape']['do'] == config.shape['do'] and
                   r['shape']['rank'] == 16 and r['shape']['accumulation'] == 1]
        if (decision['decision'] != 'audit_cost_failure_candidate_potential' or not matched or
                matched[0]['candidate_headroom'] != 'observed_positive' or
                not matched[0]['candidate_measurement_qualified'] or
                matched[0]['conditional_full_headroom'] != 'observed_negative'):
            raise ValueError('no matching candidate-potential/audit-failure evidence')
        if config.max_wall_s>600 or len(config.seeds)>2 or config.queries_per_split>1 or max(config.checkpoints)>128 or max(config.ranks)>64:
            raise ValueError('bounded scope coverage/resource ceiling exceeded')
        if config.stage=='study' and config.selected_learning_rate is None:raise ValueError('study requires locked pilot learning rate')
        if config.device != 'mps' or config.shape != {'n': 2048, 'di': 512, 'do': 2048}:
            raise ValueError('amended research shape/backend changed')
        return amendment
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise GateBlocked(f'structural oracle permission denied: {exc}') from exc


_DOMAINS = ('student', 'data', 'teacher', 'construction', 'tuning', 'evaluation', 'development', 'probe')

def domain_seed(seed, domain, *ids):
    if domain not in _DOMAINS: raise ValueError('unknown RNG domain')
    return int.from_bytes(hashlib.sha256(json.dumps([seed, domain, *ids]).encode()).digest()[:8], 'big') % (2**63-1)


def generator(seed, domain, *ids):
    return torch.Generator(device='cpu').manual_seed(domain_seed(seed, domain, *ids))


def initialize(config, seed, domain):
    di, do = config.shape['di'], config.shape['do']
    g = generator(seed, domain)
    p = {'w1': torch.randn(do, di, generator=g)/math.sqrt(di), 'b1': torch.zeros(do),
         'w2': torch.randn(di, do, generator=g)/math.sqrt(do), 'b2': torch.zeros(di)}
    return {k: v.to(config.device).requires_grad_(domain == 'student') for k, v in p.items()}


def forward(x, p, captures=None):
    def linear(x, w, b):
        return F.linear(x, p[w], p[b]) if captures is None else projected_linear(x, p[w], p[b], captures[w])
    return x + linear(F.gelu(linear(x, 'w1', 'b1')), 'w2', 'b2')


def synthetic_batch(config, seed, condition, step, domain, index):
    if condition not in config.conditions or domain not in ('data', 'tuning', 'evaluation', 'development'):
        raise ValueError('invalid synthetic descriptor domain/condition')
    d = dict(generator='gaussian_teacher_residual_v1', seed=seed, condition=condition,
             step=step, domain=domain, index=index, n=config.shape['n'])
    identity = canonical_sha(d)
    return ReplayableBatch((identity,), (d,), (tuple(True for _ in range(config.shape['n'])),),
                           (config.shape['n'],), {'domain': domain, 'step': step}, {'domain': domain, 'step': step+1})


def replay_synthetic(batch, config, seed, condition):
    if len(batch.microbatches) != 1: raise ValueError('require single synthetic descriptor')
    d = dict(batch.microbatches[0])
    expected = synthetic_batch(config, seed, condition, d['step'], d['domain'], d['index'])
    if d != dict(expected.microbatches[0]) or batch.source_ids != expected.source_ids or batch.valid_counts != expected.valid_counts:
        raise ValueError('invalid synthetic descriptor identity/counts')
    x = torch.randn(config.shape['n'], config.shape['di'], generator=generator(seed, d['domain'], d['step'], d['index'])).to(config.device)
    teacher = initialize(config, seed, 'teacher')
    if condition == 'low_rank':
        g = generator(seed, 'teacher', 'low_rank_output')
        a = torch.randn(config.shape['di'], config.teacher_rank, generator=g)/math.sqrt(config.teacher_rank)
        b = torch.randn(config.teacher_rank, config.shape['do'], generator=g)/math.sqrt(config.shape['do'])
        teacher['w2'] = (a@b).to(config.device)
    if condition == 'direction_shift' and d['step'] >= config.shift_step:
        teacher['w2'] = teacher['w2'].roll(config.shape['di']//2, dims=0)
    with torch.no_grad(): y = forward(x, teacher)
    return x.requires_grad_(), y


def synchronize(device):
    if device == 'mps': torch.mps.synchronize()


def loss_value(x, y, p):
    with torch.no_grad(): return float(F.mse_loss(forward(x, p), y).item())


def bootstrap_trajectory(config, seed, condition, lr, *, retain=True, steps=None, progress=None):
    """Native scaffolding with incremental evidence owned by the registered attempt.

    Last committed update counts and saved checkpoints are distinct: a failure
    can follow committed updates newer than the latest selected checkpoint.
    """
    started = time.perf_counter()
    steps = max(config.checkpoints) if steps is None else steps
    paths, curves = [], []
    progress = {} if progress is None else progress
    progress.update(condition=condition, seed=seed, lr=lr, status='running',
                    curves=curves, checkpoint_paths=[], artifact_index=[],
                    last_committed_update=0, last_checkpoint=None)
    try:
        p = initialize(config, seed, 'student')
        optimizer = torch.optim.AdamW(list(p.values()), lr=lr, betas=(.9,.999), eps=1e-8, weight_decay=.01)
        source_steps = {q-a for q in config.checkpoints for a in config.ages if q-a >= 0} if retain else set()
        sources = {}
        directory = Path(config.checkpoints_dir)/f'{condition}-{seed}'
        if retain:
            directory.mkdir(parents=True, exist_ok=True)

        def save_checkpoint(step):
            selected = {q: b for q, b in sources.items() if q in {step-a for a in config.ages}}
            encoded = {s: {name: dict(vectors=b.vectors.cpu(), source_step=b.source_step,
                numerical_rank=b.numerical_rank, orthogonality=b.orthogonality,
                construction_width=b.construction_width, construction_wall_s=b.construction_wall_s,
                transfer_bytes=b.transfer_bytes) for name, b in bases.items()} for s, bases in selected.items()}
            payload = dict(schema_version=1, config=asdict(config), seed=seed, condition=condition,
                step=step, parameters={k: v.detach().cpu() for k, v in p.items()},
                state={k: dict(step=v.step, exp_avg=v.exp_avg.cpu(), exp_avg_sq=v.exp_avg_sq.cpu())
                       for k, v in native_state(p, optimizer).items()},
                group=asdict(AdamWGroup(tuple(p), lr, (.9,.999), 1e-8, .01)), sources=encoded,
                native_variant='default', loss_unit=config.loss_unit)
            path = directory/f'{step}.pt'
            torch.save(payload, path)
            paths.append(path)
            entry = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, step=step,
                condition=condition, seed=seed, sources=sorted(selected),
                provenance='native_committed_state_and_actual_pre_update_bases')
            progress['checkpoint_paths'].append(str(path))
            progress['artifact_index'].append(entry)
            progress['last_checkpoint'] = entry

        def measure(step):
            # Fixed independent inputs; the teacher changes only at the declared shift.
            d = synthetic_batch(config, seed, condition, 0, 'development', 0)
            x, y = replay_synthetic(d, config, seed, condition)
            if condition == 'direction_shift' and step >= config.shift_step:
                _, y = replay_synthetic(synthetic_batch(config, seed, condition, config.shift_step, 'development', 0), config, seed, condition)
                t = initialize(config, seed, 'teacher')
                t['w2'] = t['w2'].roll(config.shape['di']//2, dims=0)
                with torch.no_grad():
                    y = forward(x, t)
            value = loss_value(x, y, p)
            if not math.isfinite(value):
                raise FloatingPointError(f'nonfinite development MSE at committed update {step}')
            curves.append(dict(condition=condition, seed=seed, lr=lr, committed_updates=step,
                target_phase='post_shift' if condition == 'direction_shift' and step >= config.shift_step else 'initial',
                development_mse=value, elapsed_s=time.perf_counter()-started))

        measure(0)
        for s in range(steps):
            if time.perf_counter()-started > config.max_wall_s:
                raise StudyIncomplete('trajectory engineering ceiling reached')
            if retain and s in config.checkpoints:
                save_checkpoint(s)
            optimizer.zero_grad(set_to_none=True)
            batch = synthetic_batch(config, seed, condition, s, 'data', 0)
            x, y = replay_synthetic(batch, config, seed, condition)
            F.mse_loss(forward(x, p), y).backward()
            if s in source_steps:
                sources[s] = {name: build_basis(p[name].grad, max(config.ranks), 8,
                    generator(seed, 'construction', s, name), s) for name in ('w1', 'w2')}
            torch.nn.utils.clip_grad_norm_(list(p.values()), 1., foreach=False, error_if_nonfinite=True)
            optimizer.step()
            progress['last_committed_update'] = s+1
            if s+1 in set(config.checkpoints) | {config.shift_step, steps}:
                measure(s+1)
        if retain and steps in config.checkpoints:
            save_checkpoint(steps)
        progress['status'] = 'completed'
        return paths, curves
    except Exception as exc:
        progress['status'] = 'incomplete' if isinstance(exc, StudyIncomplete) else 'failed'
        progress['failure'] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        synchronize(config.device)
        progress['native_wall_s'] = time.perf_counter()-started
        progress['last_committed_evidence'] = dict(
            committed_updates=progress['last_committed_update'],
            next_training_source_step=progress['last_committed_update'],
            last_saved_checkpoint=progress['last_checkpoint'],
            limitation='Only the last selected checkpoint persists tensors; later committed-update counts are execution evidence, not a resumable snapshot.')


def fp64(x): return x.detach().to('cpu').to(torch.float64)

def layer_geometry(g, gh, u, d):
    g,gh,u,d = map(fp64,(g,gh,u,d))
    e = g-gh; complement = d-u@(u.T@d)
    norm = lambda t: float(torch.linalg.vector_norm(t).item())
    ng,nh,ne=norm(g),norm(gh),norm(e)
    return dict(gradient_norm=ng, projected_norm=nh,residual_norm=ne,
                energy=nh**2/ng**2 if ng else None, cosine=float((g*gh).sum())/(ng*nh) if ng*nh else None,
                relative_residual=ne/nh if nh else None, raw_alignment=float((gh*d).sum()),
                true_alignment=float((g*d).sum()),ordinary_bound=ne*norm(d),complement_bound=ne*norm(complement),
                displacement_norm=norm(d),complement_norm=norm(complement),
                orthogonality=norm(u.T@u-torch.eye(u.shape[1],dtype=u.dtype)),
                projection_defect=norm(u.T@e), missing_reason=None if ng*nh else 'zero_gradient_norm')


def audit_checkpoint(checkpoint: Path, past_batches: list[ReplayableBatch], query_batches: list[ReplayableBatch],
                     ranks: tuple[int,...], costs: CostTable) -> list[dict[str,object]]:
    started=time.perf_counter(); snap=torch.load(checkpoint,weights_only=False)
    if snap['schema_version'] != 1: raise ValueError('unsupported oracle checkpoint')
    c=parse_oracle_config(snap['config']); seed=snap['seed']; condition=snap['condition']; step=snap['step']
    if not set(ranks)<=set(c.ranks): raise ValueError('rank outside checkpoint coverage')
    expected={synthetic_batch(c,seed,condition,s,'data',0).source_ids[0]:s for s in snap['sources']}
    actual={}
    for b in past_batches:
        ident=b.source_ids[0]
        if ident not in expected or b.microbatches[0]['step'] >= step: raise ValueError('past basis source mismatch')
        replay_synthetic(b,c,seed,condition);actual[expected[ident]]=ident
    if set(actual)!=set(snap['sources']): raise ValueError('past descriptors omit basis sources')
    p={k:v.to(c.device).requires_grad_() for k,v in snap['parameters'].items()}
    state={k:AdamWParamState(v['step'],v['exp_avg'].to(c.device),v['exp_avg_sq'].to(c.device)) for k,v in snap['state'].items()}
    gr=dict(snap['group']);gr.pop('schema_version');gr['parameter_ids']=tuple(gr['parameter_ids']);gr['betas']=tuple(gr['betas']);groups=(AdamWGroup(**gr),)
    rows=[]
    for query in query_batches:
        qstart=time.perf_counter();d=dict(query.microbatches[0])
        if d['step'] != step or d['domain'] not in ('tuning','evaluation'): raise ValueError('query step/domain mismatch')
        x,y=replay_synthetic(query,c,seed,condition)
        dev=synthetic_batch(c,seed,condition,step,'development',1+d['index']+(1000 if d['domain']=='evaluation' else 0))
        dx,dy=replay_synthetic(dev,c,seed,condition)
        for v in p.values(): v.grad=None
        base=F.mse_loss(forward(x,p),y);base.backward()
        exact={k:v.grad.detach().clone() for k,v in p.items()}
        dense=preview_adamw(p,exact,state,groups,step,step,1.)
        baseline=float(base.item());devbase=loss_value(dx,dy,p)
        dense_same=loss_value(x,y,dense.candidate_parameters)-baseline
        dense_dev=loss_value(dx,dy,dense.candidate_parameters)-devbase
        svdstart=time.perf_counter()
        current={name:torch.linalg.svd(fp64(exact[name]),full_matrices=False).U[:,:max(ranks)].to(dtype=p[name].dtype,device=c.device) for name in ('w1','w2')}
        synchronize(c.device);svdwall=time.perf_counter()-svdstart
        options=[('past',s,{n:BasisState(**dict(v, vectors=v['vectors'].to(c.device))) for n,v in b.items()}) for s,b in snap['sources'].items()]+[('current_svd_oracle',step,None)]
        commonwall=time.perf_counter()-qstart
        for kind,source,bases in options:
            for rank in ranks:
                t=time.perf_counter()
                us={name:(bases[name].for_step(step, min(rank,bases[name].vectors.shape[1])) if bases is not None else current[name][:,:rank]) for name in ('w1','w2')}
                captures={name:LinearCapture(u,torch.randn(p[name].shape[1],8,generator=generator(seed,'probe',step,d['domain'],d['index'],name)).to(c.device),source) for name,u in us.items()}
                for v in p.values():v.grad=None
                F.mse_loss(forward(x,p,captures),y).backward()
                gh={k:v.grad.detach().clone() for k,v in p.items()}
                pending=preview_adamw(p,gh,state,groups,step,step,1.)
                same=loss_value(x,y,pending.candidate_parameters)-baseline
                development=loss_value(dx,dy,pending.candidate_parameters)-devbase
                geometry={name:layer_geometry(exact[name],gh[name],us[name],pending.displacements[name]) for name in ('w1','w2')}
                a=pending.raw_alignment;true=sum(float((fp64(exact[k])*fp64(pending.displacements[k])).sum()) for k in p)
                ordinary=sum(r['ordinary_bound'] for r in geometry.values());comp=sum(r['complement_bound'] for r in geometry.values())
                reln=math.sqrt(sum(r['residual_norm']**2 for r in geometry.values()));reld=math.sqrt(sum(r['projected_norm']**2 for r in geometry.values()))
                ident=f'{condition}/{seed}/{step}/{d["domain"]}/{d["index"]}/{kind}/{source}/{rank}'
                matching=costs.rows.get((c.device,'float32',c.shape['n'],c.shape['di'],c.shape['do'],rank,8))
                probe_bounds={name:residual_bound(captures[name].r_sum,8,.01) for name in geometry}
                probe_complement=sum(probe_bounds[name]*r['complement_norm'] for name,r in geometry.items())
                synchronize(c.device);elapsed=time.perf_counter()-t
                for name,r in geometry.items():
                    bound=probe_bounds[name]
                    row=dict(schema_version=1,proposal_id=ident,checkpoint_id=str(checkpoint),query_id=query.source_ids[0],
                        development_id=dev.source_ids[0],condition=condition,seed=seed,query_step=step,split=d['domain'],
                        layer=name,rank=rank,actual_rank=us[name].shape[1],basis_kind=kind,basis_source_step=source,
                        basis_age=step-source,basis_source_id=actual.get(source),basis_sha256=hashlib.sha256(fp64(us[name]).numpy().tobytes()).hexdigest(),
                        source_provenance='actual_pre_update_before_clipping' if kind=='past' else 'paid_exact_current_gradient',
                        loss_unit=c.loss_unit,count_unit='synthetic_observations',observations=c.shape['n'],
                        same_batch_delta=same,development_delta=development,dense_same_batch_delta=dense_same,
                        dense_development_delta=dense_dev,lost_progress=development-dense_dev,
                        proposal_raw_alignment=a,proposal_true_alignment=true,proposal_ordinary_bound=ordinary,
                        proposal_complement_bound=comp,proposal_probe_complement_bound=probe_complement,proposal_relative_residual=reln/reld if reld else None,
                        probe_residual_bound=bound,probe_complement_bound=bound*r['complement_norm'],
                        dense_displacement_norm=float(torch.linalg.vector_norm(fp64(dense.displacements[name]))),
                        oracle_cost_s=elapsed,query_shared_cost_s=commonwall,current_svd_cost_s=svdwall,
                        cost_accounting='proposal wall repeated per layer; shared query overhead recorded once per query; use total study wall',
                        runtime_evidence_reason='matched_rank16_cost_only' if matching else 'no_matching_measured_cost',
                        measured_candidate_s=matching['headline']['candidate']['other']['mean_s'] if matching else None,
                        measured_complete_attempt_s=matching['headline']['conditional']['other']['mean_s'] if matching else None,
                        online_eligible=False,numerical_allowance=None,numerical_reason='NumericalPolicy_unresolved',**r)
                    rows.append(row)
    synchronize(c.device)
    # File inputs and all caller runtime state are read-only; no learner is passed here.
    return rows


def _auc(scores,labels):
    yes=[s for s,y in zip(scores,labels) if y];no=[s for s,y in zip(scores,labels) if not y]
    if not yes or not no:return None
    return sum((a>b)+.5*(a==b) for a in yes for b in no)/(len(yes)*len(no))


def summarize_policies(rows,taus,relative_thresholds):
    # Collapse layer rows to the shared simultaneous two-weight proposal.
    unique={r['proposal_id']:r for r in rows if r['basis_kind']=='past'}
    items=list(unique.values());tune=[r for r in items if r['split']=='tuning'];evaluation=[r for r in items if r['split']=='evaluation']
    def get(r,name):return r.get('proposal_'+name,r.get(name))
    def accept(r,kind,threshold):
        if kind in ('constant','always_project'):return threshold
        if kind=='fixed':return (r['rank'],r['basis_age'])==threshold
        if kind=='relative':return get(r,'relative_residual') is not None and get(r,'relative_residual')<=threshold
        a=get(r,'raw_alignment');b=get(r,kind+'_bound')
        return a>0 and b<=threshold*a
    # Minimize dev MSE change against exact fallback on tuning, no evaluation labels.
    fitted={};performance={}
    choices={'constant':(False,True),'always_project':(True,), 'fixed':sorted({(r['rank'],r['basis_age']) for r in tune}), 'relative':relative_thresholds,'ordinary':taus,'complement':taus}
    for kind,grid in choices.items():
        if not tune or not grid:continue
        objective=lambda threshold:sum(r['development_delta'] if accept(r,kind,threshold) else r.get('dense_development_delta',0.) for r in tune)/len(tune)
        threshold=min(grid,key=objective)
        fitted[kind]=dict(threshold=threshold,tuning_objective=objective(threshold),tuning_query_ids=sorted({r.get('query_id',r['proposal_id']) for r in tune}))
        accepted=[r for r in evaluation if accept(r,kind,threshold)]
        performance[kind]=dict(evaluation_queries=sorted({r.get('query_id',r['proposal_id']) for r in evaluation}),
            proposals=len(evaluation),accepted=len(accepted),accepted_development_harm=sum(r['development_delta']>0 for r in accepted),
            mean_development_delta_with_exact_fallback=sum(r['development_delta'] if accept(r,kind,threshold) else r.get('dense_development_delta',0.) for r in evaluation)/len(evaluation) if evaluation else None)
    for kind,perf in performance.items():
        threshold=fitted[kind]['threshold']
        perf['labels']={}
        for label,predicate in {'first_order_harm':lambda r:get(r,'true_alignment')<0, 'same_batch_harm':lambda r:r['same_batch_delta']>0, 'development_harm':lambda r:r['development_delta']>0, 'lost_progress':lambda r:r['lost_progress']>0}.items():
            harmful=[r for r in evaluation if predicate(r)];safe=[r for r in evaluation if not predicate(r)]
            caught=sum(not accept(r,kind,threshold) for r in harmful);false=sum(not accept(r,kind,threshold) for r in safe)
            perf['labels'][label]=dict(events=len(harmful),rejected_events=caught,safe=len(safe),false_rejections=false,
                sensitivity=caught/len(harmful) if len(harmful)>=5 else None,
                sensitivity_reason='descriptive_correlated_queries' if len(harmful)>=5 else 'insufficient_harm_events',
                false_rejection_fraction=false/len(safe) if safe else None)
    labels={'first_order_harm':lambda r:get(r,'true_alignment')<0,'same_batch_harm':lambda r:r['same_batch_delta']>0,
            'development_harm':lambda r:r['development_delta']>0,'lost_progress':lambda r:r['lost_progress']>0}
    discrimination={}
    for label,fn in labels.items():
        ys=[fn(r) for r in evaluation]
        for score in ('relative','ordinary','complement'):
            ss=[get(r,'relative_residual') if score=='relative' else (get(r,score+'_bound')/get(r,'raw_alignment') if get(r,'raw_alignment')>0 else math.inf) for r in evaluation]
            valid=[(s,y) for s,y in zip(ss,ys) if s is not None]
            auc=_auc([s for s,y in valid],[y for s,y in valid]);n=sum(y for s,y in valid)
            reason='only_one_label_class' if auc is None else ('insufficient_label_support' if min(n,len(valid)-n)<5 else 'descriptive_correlated_queries')
            if reason!='descriptive_correlated_queries':auc=None
            discrimination[label+'/'+score]=dict(events=n,total=len(valid),auc=auc,reason=reason)
    return dict(fitted_policies=fitted,evaluation=performance,discrimination=discrimination,
                limitation='Proposals share checkpoints; query counts are not independent runs. No online or confirmatory inference.')


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def run_oracle_worker(config, *, pilot_only):
    """Internal bounded scaffolding; public entry validates permission in fresh process."""
    import psutil
    start=time.perf_counter();out=Path(config.output);out.mkdir(parents=True,exist_ok=True)
    summary=dict(schema_version=1,scope='synthetic_structural_exploratory',online_eligible=False,
                 config=asdict(config),loss_unit=config.loss_unit,count_unit='synthetic_observations',
                 coverage_complete=False,curves=[],artifact_index=[],attempts=[],rng_domains={str(seed):{d:domain_seed(seed,d) for d in _DOMAINS} for seed in config.seeds},
                 code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 fallback_environment=os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))
    def run_bootstrap(seed, condition, lr, *, phase, **kwargs):
        attempt = dict(condition=condition, seed=seed, lr=lr, phase=phase, status='registered',
            curves=[], checkpoint_paths=[], artifact_index=[], last_committed_update=0,
            last_checkpoint=None, native_wall_s=0.)
        summary['attempts'].append(attempt)
        if phase == 'pilot':
            summary['pilot'].append(attempt)
        try:
            if time.perf_counter()-start > config.max_wall_s:
                raise StudyIncomplete('study engineering ceiling reached')
            paths, curves = bootstrap_trajectory(config, seed, condition, lr, progress=attempt, **kwargs)
            return paths, curves, attempt
        except Exception as exc:
            attempt['status'] = 'incomplete' if isinstance(exc, StudyIncomplete) else 'failed'
            attempt.setdefault('failure', dict(type=type(exc).__name__, message=str(exc)))
            raise
        finally:
            summary['curves'].extend(attempt['curves'])
            summary['artifact_index'].extend(attempt['artifact_index'])
            attempt['wall_s'] = attempt['native_wall_s']

    try:
        if config.selected_learning_rate is None:
            pilots=[];summary['pilot']=pilots
            for lr in config.learning_rates:
                _,curves,attempt=run_bootstrap(config.seeds[0],'full_rank',lr,phase='pilot',retain=False,steps=config.pilot_steps)
                print(f'pilot lr={lr} MSE {curves[0]["development_mse"]:.6g} -> {curves[-1]["development_mse"]:.6g}',flush=True)
            selected=min(pilots,key=lambda r:r['curves'][-1]['development_mse'])['lr']
            summary['pilot']=pilots;summary['selected_learning_rate']=selected
        else:selected=config.selected_learning_rate
        if not pilot_only:
            raw_cost=json.loads(Path('analysis/timing_calibration.json').read_text())
            costs=CostTable({tuple(r['key']):r for r in raw_cost['rows']},raw_cost['metadata'])
            allrows=[]
            auditpath=out/'structural_oracle_audit.jsonl'
            with auditpath.open('w') as stream:
                for condition in config.conditions:
                    for seed in config.seeds:
                        paths,curves,attempt=run_bootstrap(seed,condition,selected,phase='study')
                        phases={r['target_phase'] for r in curves}
                        learnable=all(next(r for r in curves if r['target_phase']==phase)['development_mse'] >
                            [r for r in curves if r['target_phase']==phase][-1]['development_mse']+1e-7 for phase in phases)
                        attempt['learnable']=learnable
                        for path in paths:
                            if time.perf_counter()-start>config.max_wall_s:raise StudyIncomplete('study engineering ceiling reached')
                            snap=torch.load(path,weights_only=False)
                            if not learnable:continue
                            past=[synthetic_batch(config,seed,condition,s,'data',0) for s in snap['sources']]
                            queries=[synthetic_batch(config,seed,condition,snap['step'],domain,i) for domain in ('tuning','evaluation') for i in range(config.queries_per_split)]
                            at=time.perf_counter()
                            rows=audit_checkpoint(path,past,queries,config.ranks,costs)
                            auditwall=time.perf_counter()-at
                            for r in rows:stream.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
                            stream.flush();allrows.extend(rows)
                            if 'resource_forecast' not in summary:
                                planned=len(config.conditions)*len(config.seeds)*sum((1+sum(q-a>=0 for a in config.ages))*len(config.ranks)*2*2*config.queries_per_split for q in config.checkpoints)
                                forecast=auditwall*planned/len(rows)+summary['attempts'][-1]['native_wall_s']*len(config.conditions)*len(config.seeds)
                                summary['resource_forecast']=dict(estimated_study_wall_s=forecast,engineering_ceiling_s=config.max_wall_s,
                                    measured_first_checkpoint_audit_s=auditwall,measured_first_native_history_s=summary['attempts'][-1]['native_wall_s'],
                                    planned_layer_rows=planned,method='first native history times histories plus first checkpoint cost per layer-row times planned rows; conservative shared-overhead scaling; not a confidence bound')
                                print(f'forecast study wall={forecast:.2f}s ceiling={config.max_wall_s}s',flush=True)
                                if forecast>config.max_wall_s:raise StudyIncomplete('first-query forecast exceeds conservative engineering ceiling; partial coverage retained')
                            print(f'audit {condition} seed={seed} step={snap["step"]} rows={len(rows)} elapsed={time.perf_counter()-start:.2f}s',flush=True)
            summary['policies']=summarize_policies(allrows,config.taus,config.relative_thresholds)
            summary['row_count']=len(allrows)
            summary['strata']=[]
            for condition in config.conditions:
                for seed in config.seeds:
                    for step in config.checkpoints:
                        subset=[r for r in allrows if r['condition']==condition and r['seed']==seed and r['query_step']==step]
                        summary['strata'].append(dict(condition=condition,seed=seed,step=step,policies=summarize_policies(subset,config.taus,config.relative_thresholds)))
            write_json(out/'structural_oracle_schema.json',dict(schema_version=1,fields=sorted(allrows[0]) if allrows else [],
                units={'loss':'MSE per output element','counts':'synthetic observations','alignment':'raw gradient dot actual displacement','cost':'seconds'},
                timing='oracle_cost_s repeats on two layer rows; query_shared_cost_s repeats across proposals; total_wall_s is authoritative',
                oracle='current_svd_oracle uses paid exact current gradient, never a past/deployable proposal'))
        summary['resources_note']='RSS and Metal are overlapping endpoint snapshots, not measured peaks; CPU SVD and transfers are charged, no online speedup.'
        summary['provenance']={name:sha(name) for name in ('src/oar/oracle.py','src/oar/linear.py','src/oar/basis.py','src/oar/optimizer_preview.py','configs/structural_oracle.yaml','decisions/structural_oracle_scope.md','analysis/timing_calibration.json','analysis/headroom_decision.json')}
        summary['coverage_complete']=True
        summary['outcome']='pilot_complete' if pilot_only else 'synthetic_only_online_still_blocked'
    except StudyIncomplete as exc:
        summary['outcome']='incomplete_inconclusive';summary['coverage_complete']=False
        summary['incomplete_reason']=str(exc)
        summary['row_count']=len(allrows) if 'allrows' in locals() else 0
    except Exception as exc:
        summary['outcome']='failed';summary['coverage_complete']=False
        summary['row_count']=len(allrows) if 'allrows' in locals() else 0
        summary['failure']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        synchronize(config.device);summary['total_wall_s']=time.perf_counter()-start
        summary['rss_at_finish_bytes']=psutil.Process().memory_info().rss
        if config.device=='mps':summary['mps_driver_at_finish_bytes']=torch.mps.driver_allocated_memory()
        path=out/('structural_oracle_pilot.json' if pilot_only else 'structural_oracle_summary.json');write_json(path,summary)
    return path


def run_structural_oracle(config_path: Path) -> Path:
    config=load_oracle_config(config_path);require_structural_oracle(config,config_path)
    env=os.environ.copy();env['PYTORCH_ENABLE_MPS_FALLBACK']='0'
    try:
        subprocess.run([sys.executable,'-m','oar.oracle',str(config_path)],env=env,check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('synthetic oracle worker failed; retained failure summary and partial rows') from exc
    return Path(config.output)/('structural_oracle_pilot.json' if config.stage=='pilot' else 'structural_oracle_summary.json')


if __name__=='__main__':
    config_path=Path(sys.argv[1]);config=load_oracle_config(config_path)
    require_structural_oracle(config,config_path)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('fresh fallback0 worker required')
    if config.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('actual MPS required')
    run_oracle_worker(config,pilot_only=config.stage=='pilot')
