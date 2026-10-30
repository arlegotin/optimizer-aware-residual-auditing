"""Independent references for the bounded synthetic oracle, never population claims."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from oar.bench import CostTable


def api():
    from oar import oracle
    return oracle


def tiny_config(tmp_path):
    raw = yaml.safe_load(Path('configs/structural_oracle.yaml').read_text())
    raw.update(device='cpu', shape={'n': 12, 'di': 5, 'do': 9}, seeds=[17],
               checkpoints=[2, 4], ages=[1, 2], ranks=[1, 3], pilot_steps=4,
               learning_rates=[.001, .003], shift_step=2, teacher_rank=2,
               output=str(tmp_path/'analysis'), checkpoints_dir=str(tmp_path/'checkpoints'))
    return api().parse_oracle_config(raw)


def test_oracle_module_and_strict_synthetic_config(tmp_path):
    o = api()
    c = tiny_config(tmp_path)
    assert c.phase == 'exploratory' and c.loss_unit == 'mse_per_output_element'
    raw = yaml.safe_load(Path('configs/structural_oracle.yaml').read_text())
    for change in ({'phase': 'confirmatory'}, {'synthetic_only': False}, {'corpus': 'text'},
                   {'seeds': [True]}, {'ranks': [0]}, {'learning_rates': [float('nan')]}):
        with pytest.raises(ValueError):
            o.parse_oracle_config(dict(raw, **change))


def test_geometry_matches_independent_numpy_and_adam_counterexample():
    o = api()
    g = torch.tensor([[1.], [.02]], dtype=torch.float64)
    u = torch.tensor([[1.], [-.02]], dtype=torch.float64)
    u /= torch.linalg.vector_norm(u)
    gh = u @ (u.T @ g)
    from oar.optimizer_preview import preview_adamw
    from oar.records import AdamWGroup, AdamWParamState
    p = {'w': torch.zeros_like(g)}
    pending = preview_adamw(p, {'w': gh}, {'w': AdamWParamState(99, torch.zeros_like(g),
                              torch.tensor([[1e4], [1e-8]], dtype=torch.float64))},
                            (AdamWGroup(('w',), 1., (.9, .999), 1e-8, 0.),), 0, 0, None)
    d = pending.displacements['w']
    r = o.layer_geometry(g, gh, u, d)
    gn, un, dn = g.numpy(), u.numpy(), d.numpy()
    e = gn - un@(un.T@gn)
    assert r['residual_norm'] == pytest.approx(np.linalg.norm(e))
    assert r['complement_bound'] == pytest.approx(np.linalg.norm(e)*np.linalg.norm(dn-un@(un.T@dn)))
    assert r['true_alignment'] == pytest.approx(float((gn*dn).sum()))
    assert r['true_alignment'] < 0 < r['raw_alignment']
    assert r['cosine'] > .999


@pytest.mark.parametrize('kind', ['flat', 'rotating', 'rare'])
def test_constructed_projected_sgd_descent_and_residual_geometry(kind):
    o = api()
    # Fixed past mode e1; query rotation changes its left singular direction.
    u = torch.tensor([[1.], [0.], [0.]], dtype=torch.float64)
    if kind == 'flat':
        g = torch.eye(3, dtype=torch.float64)
    elif kind == 'rotating':
        direction = torch.tensor([.5, 3**.5/2, 0.], dtype=torch.float64)
        g = torch.outer(direction, torch.tensor([1., 0., 0.], dtype=torch.float64))
    else:
        g = torch.diag(torch.tensor([1., 1., 1e-5], dtype=torch.float64))
    gh = u@(u.T@g)
    r = o.layer_geometry(g, gh, u, .1*gh)
    assert r['true_alignment'] == pytest.approx(.1*float(gh.square().sum()))
    assert r['complement_bound'] < 1e-15
    if kind == 'flat':
        assert r['energy'] == pytest.approx(1/3)
        assert r['residual_norm'] == pytest.approx(2**.5)
    if kind == 'rotating':
        assert r['energy'] == pytest.approx(.25)
        assert r['residual_norm'] == pytest.approx(3**.5/2)
    if kind == 'rare':
        assert float(g[2, 2]) != 0 and float(gh[2, 2]) == 0


def test_descriptor_replay_and_rng_domains(tmp_path):
    o = api(); c = tiny_config(tmp_path)
    global_before = torch.get_rng_state().clone()
    a = o.synthetic_batch(c, 17, 'full_rank', 3, 'evaluation', 0)
    x, y = o.replay_synthetic(a, c, 17, 'full_rank')
    x2, y2 = o.replay_synthetic(a, c, 17, 'full_rank')
    assert torch.equal(x, x2) and torch.equal(y, y2)
    assert a.valid_counts == (12,) and a.pre_cursor['domain'] == 'evaluation'
    train = o.synthetic_batch(c, 17, 'full_rank', 3, 'data', 0)
    assert not torch.equal(x, o.replay_synthetic(train, c, 17, 'full_rank')[0])
    assert torch.equal(global_before, torch.get_rng_state())


def test_temporal_leakage_and_copied_state_isolation(tmp_path):
    o = api(); c = tiny_config(tmp_path)
    paths, curves = o.bootstrap_trajectory(c, 17, 'full_rank', .003)
    checkpoint = paths[-1]
    before = checkpoint.read_bytes(); rng = torch.get_rng_state().clone()
    snap = torch.load(checkpoint, weights_only=False)
    past = [o.synthetic_batch(c, 17, 'full_rank', s, 'data', 0) for s in snap['sources']]
    queries = [o.synthetic_batch(c, 17, 'full_rank', 4, domain, 0) for domain in ('tuning', 'evaluation')]
    rows = o.audit_checkpoint(checkpoint, past, queries, c.ranks, CostTable({}, {}))
    assert rows and checkpoint.read_bytes() == before and torch.equal(rng, torch.get_rng_state())
    assert all(r['basis_source_step'] < r['query_step'] for r in rows if r['basis_kind'] == 'past')
    assert all(r['loss_unit'] == 'mse_per_output_element' and r['oracle_cost_s'] >= 0 for r in rows)
    assert all(r['runtime_evidence_reason'] == 'no_matching_measured_cost' for r in rows)
    assert {r['basis_kind'] for r in rows} == {'past', 'current_svd_oracle'}
    leaked = [o.synthetic_batch(c, 17, 'full_rank', 4, 'data', 0)]
    with pytest.raises(ValueError, match='past|source'):
        o.audit_checkpoint(checkpoint, leaked, queries, c.ranks, CostTable({}, {}))
    other = [o.synthetic_batch(c, 17, 'full_rank', 3, 'evaluation', 0)]
    with pytest.raises(ValueError, match='query'):
        o.audit_checkpoint(checkpoint, past, other, c.ranks, CostTable({}, {}))


def test_policies_fit_only_tuning_and_absent_harm_is_unresolved():
    o = api()
    rows = []
    for split in ('tuning', 'evaluation'):
        for i in range(4):
            rows.append(dict(proposal_id=f'{split}{i}', split=split, basis_kind='past',
                condition='full_rank', seed=1, query_step=8, rank=4, basis_age=1,
                relative_residual=.1*i, raw_alignment=1., ordinary_bound=.2*i,
                complement_bound=.1*i, development_delta=-.1, same_batch_delta=-.1,
                true_alignment=.1, lost_progress=.01))
    a = o.summarize_policies(rows, (.25,.5,.75), (.1,.5,1.))
    changed = copy.deepcopy(rows)
    for r in changed:
        if r['split'] == 'evaluation': r['development_delta'] = 100
    b = o.summarize_policies(changed, (.25,.5,.75), (.1,.5,1.))
    assert a['fitted_policies'] == b['fitted_policies']
    assert all(v['auc'] is None and v['reason'] == 'only_one_label_class' for v in a['discrimination'].values())


def test_permission_guard_rejects_missing_amendment_and_non_oracle_command(tmp_path):
    o = api()
    c = tiny_config(tmp_path)
    with pytest.raises(o.GateBlocked):
        o.require_structural_oracle(c, Path('configs/structural_oracle.yaml'), root=tmp_path)
    with pytest.raises(o.GateBlocked, match='command'):
        o.require_structural_oracle(c, Path('configs/structural_oracle.yaml'), command='train')


def test_cpu_end_to_end_worker_outputs_units_and_costs(tmp_path):
    o = api(); c = tiny_config(tmp_path)
    result = o.run_oracle_worker(c, pilot_only=False)
    assert result.exists()
    summary = json.loads(result.read_text())
    assert summary['online_eligible'] is False
    assert summary['total_wall_s'] > 0
    assert summary['curves'] and summary['artifact_index']
    assert summary['resource_forecast']['estimated_study_wall_s'] > 0
    assert summary['scope'] == 'synthetic_structural_exploratory'


def test_saved_bases_are_original_preupdate_gradients_and_dense_counterfactual_is_native(tmp_path):
    """Recomputing a source batch at query weights or using postclip history fails."""
    o = api(); c = tiny_config(tmp_path)
    paths, _ = o.bootstrap_trajectory(c, 17, 'full_rank', .003)
    snap = torch.load(paths[-1], weights_only=False)
    p = o.initialize(c, 17, 'student')
    opt = torch.optim.AdamW(list(p.values()), lr=.003, weight_decay=.01)
    for step in range(4):
        opt.zero_grad(set_to_none=True)
        x, y = o.replay_synthetic(o.synthetic_batch(c, 17, 'full_rank', step, 'data', 0),c,17,'full_rank')
        torch.nn.functional.mse_loss(o.forward(x,p),y).backward()
        if step in snap['sources']:
            for name in ('w1', 'w2'):
                g = p[name].grad.double()
                u = torch.linalg.svd(g,full_matrices=False).U[:,:3]
                saved = snap['sources'][step][name]['vectors'].double()
                torch.testing.assert_close(saved@saved.T,u@u.T,atol=2e-5,rtol=2e-5)
        torch.nn.utils.clip_grad_norm_(list(p.values()),1.,foreach=False,error_if_nonfinite=True)
        opt.step()
    for name in p:
        assert torch.equal(p[name],snap['parameters'][name])
        assert torch.equal(opt.state[p[name]]['exp_avg'],snap['state'][name]['exp_avg'])
        assert torch.equal(opt.state[p[name]]['exp_avg_sq'],snap['state'][name]['exp_avg_sq'])
    past=[o.synthetic_batch(c,17,'full_rank',s,'data',0) for s in snap['sources']]
    query=o.synthetic_batch(c,17,'full_rank',4,'evaluation',0)
    rows=o.audit_checkpoint(paths[-1],past,[query],c.ranks,CostTable({},{}))
    opt.zero_grad(set_to_none=True)
    x,y=o.replay_synthetic(query,c,17,'full_rank')
    before=torch.nn.functional.mse_loss(o.forward(x,p),y)
    before.backward()
    torch.nn.utils.clip_grad_norm_(list(p.values()),1.,foreach=False,error_if_nonfinite=True)
    opt.step()
    delta=o.loss_value(x,y,p)-before.item()
    assert all(r['dense_same_batch_delta']==pytest.approx(delta,abs=2e-7) for r in rows)


def test_guard_positive_and_tamper_binding(tmp_path):
    import shutil
    o=api()
    (tmp_path/'decisions').mkdir()
    ledger=[json.loads(s) for s in Path('decisions/gates.jsonl').read_text().splitlines()]
    latest={r['package']:r for r in ledger}
    shutil.copy('decisions/gates.jsonl',tmp_path/'decisions/gates.jsonl')
    shutil.copy('decisions/structural_oracle_scope.md',tmp_path/'decisions/structural_oracle_scope.md')
    config_path=tmp_path/'config.yaml';shutil.copy('configs/structural_oracle.yaml',config_path)
    evidence={}
    for name in ('analysis/headroom_decision.json','analysis/timing_calibration.json',
                 'analysis/operator_correctness.json','analysis/audit_confidence.json','analysis/optimizer_correctness.json'):
        target=tmp_path/name;target.parent.mkdir(exist_ok=True)
        shutil.copy(name,target);evidence[name]=o.sha(target)
    amendment=dict(schema_version=1,kind='synthetic_structural_oracle',command='audit-checkpoints',
        config_sha256=o.sha(config_path),scope_sha256=o.sha(tmp_path/'decisions/structural_oracle_scope.md'),
        wp02_record_sha256=o.canonical_sha(latest['WP02']),evidence_hashes=evidence)
    (tmp_path/'decisions/amendments.jsonl').write_text(json.dumps(amendment)+'\n')
    c=o.load_oracle_config(config_path)
    assert o.require_structural_oracle(c,config_path,root=tmp_path)==amendment
    (tmp_path/'analysis/headroom_decision.json').write_text('{}')
    with pytest.raises(o.GateBlocked,match='hash'):
        o.require_structural_oracle(c,config_path,root=tmp_path)
    shutil.copy('analysis/headroom_decision.json',tmp_path/'analysis/headroom_decision.json')
    config_path.write_text(config_path.read_text()+'\n')
    with pytest.raises(o.GateBlocked,match='hash'):
        o.require_structural_oracle(c,config_path,root=tmp_path)


def test_cli_owns_oracle_command_and_reports_blocked(tmp_path,capsys):
    from oar.cli import main
    path=tmp_path/'config.yaml';path.write_text('schema_version: 99')
    assert main(['audit-checkpoints','--config',str(path)])==2


def test_rare_harm_events_do_not_produce_claimed_discrimination():
    o=api()
    rows=[]
    for i in range(8):
        rows.append(dict(proposal_id=str(i),split='evaluation',basis_kind='past',condition='full_rank',
            seed=1,query_step=8,rank=4,basis_age=1,relative_residual=i,raw_alignment=1.,
            ordinary_bound=i,complement_bound=i,development_delta=1. if i==7 else -1.,
            same_batch_delta=-1.,true_alignment=1.,lost_progress=0.))
    result=o.summarize_policies(rows,(.25,.5,.75),(.1,.5,1.))
    item=result['discrimination']['development_harm/relative']
    assert item['events']==1 and item['auc'] is None
    assert item['reason']=='insufficient_label_support'


def test_synthetic_native_boundary_preserves_root_input_gradients(tmp_path):
    o=api();c=tiny_config(tmp_path)
    x,y=o.replay_synthetic(o.synthetic_batch(c,17,'full_rank',0,'data',0),c,17,'full_rank')
    assert x.requires_grad
    p=o.initialize(c,17,'student')
    torch.nn.functional.mse_loss(o.forward(x,p),y).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_engineering_ceiling_preserves_incomplete_result(tmp_path):
    from dataclasses import replace
    o=api();c=replace(tiny_config(tmp_path),selected_learning_rate=.003,max_wall_s=1e-9)
    result=o.run_oracle_worker(c,pilot_only=False)
    summary=json.loads(result.read_text())
    assert summary['outcome']=='incomplete_inconclusive'
    assert summary['coverage_complete'] is False
    assert 'ceiling' in summary['incomplete_reason']


def test_constant_and_unconditional_projection_controls_share_proposals():
    o=api()
    rows=[dict(proposal_id=split,split=split,basis_kind='past',condition='full_rank',seed=1,
        query_step=8,rank=4,basis_age=1,relative_residual=.5,raw_alignment=1.,ordinary_bound=.5,
        complement_bound=.25,development_delta=-.1,dense_development_delta=-.2,
        same_batch_delta=-.1,true_alignment=.1,lost_progress=.1) for split in ('tuning','evaluation')]
    summary=o.summarize_policies(rows,(.25,.5,.75),(.1,.5,1.))
    assert summary['evaluation']['always_project']['accepted']==1
    assert summary['evaluation']['constant']['accepted']==0


def test_descriptive_postprocessor_deduplicates_layer_rows_and_keeps_dense_harm():
    import runpy
    helpers=runpy.run_path('analysis/summarize_structural_oracle.py')
    base=dict(schema_version=1,proposal_id='p',split='evaluation',condition='full_rank',seed=1,
        query_step=8,basis_kind='past',rank=4,basis_age=1,query_id='q',development_id='d',
        proposal_true_alignment=-1.,same_batch_delta=.2,development_delta=.3,
        dense_same_batch_delta=.1,dense_development_delta=.1,lost_progress=.2,
        energy=.5,proposal_raw_alignment=1.,proposal_ordinary_bound=.5,
        proposal_complement_bound=.2,proposal_probe_complement_bound=.8,
        oracle_cost_s=2.,query_shared_cost_s=3.,current_svd_cost_s=1.)
    rows=[dict(base,layer=layer) for layer in ('w1','w2')]
    result=helpers['aggregate_rows'](rows,(.25,.5,.75))
    assert result['evaluation_proposals']==1 and result['evaluation_layer_rows']==2
    assert result['bound_distributions']['past']['complement_over_a']['median']==pytest.approx(.2)
    assert result['acceptance_envelope']['past']['complement']['0.25']['accepted']==1
    assert result['acceptance_envelope']['past']['ordinary']['0.25']['accepted']==0
    item=result['strata'][0]
    assert item['first_order_harm']==1 and item['dense_development_harm']==1
    assert item['ordinary_margin_ranges']['0.25']==[-.25,-.25]
    assert item['complement_margin_ranges']['0.25']==pytest.approx([.05,.05])


def test_cli_rejects_syntactically_invalid_oracle_yaml(tmp_path):
    from oar.cli import main
    path=tmp_path/'bad.yaml';path.write_text('shape: [\n')
    assert main(['audit-checkpoints','--config',str(path)])==2
