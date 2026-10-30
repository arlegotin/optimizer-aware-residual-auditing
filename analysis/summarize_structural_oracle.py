"""Descriptive postprocessing only: no threshold fitting or experiment selection.

Run `.venv/bin/python analysis/summarize_structural_oracle.py` after the oracle.
Paired layer rows collapse to a single simultaneous two-weight proposal.
"""
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean


def aggregate_rows(rows, taus, relative_thresholds=(.1,.25,.5,1.,2.)):
    evaluation = [r for r in rows if r['split'] == 'evaluation']
    unique = {r['proposal_id']: r for r in evaluation}
    groups = {}
    for r in unique.values():
        key = tuple(r[k] for k in ('condition', 'seed', 'query_step', 'basis_kind', 'rank', 'basis_age'))
        groups.setdefault(key, []).append(r)
    strata = []
    for key, group in sorted(groups.items()):
        item = dict(zip(('condition', 'seed', 'query_step', 'basis_kind', 'rank', 'basis_age'), key))
        item.update(proposals=len(group), query_ids=sorted({r['query_id'] for r in group}),
            first_order_harm=sum(r['proposal_true_alignment'] < 0 for r in group),
            same_batch_harm=sum(r['same_batch_delta'] > 0 for r in group),
            development_harm=sum(r['development_delta'] > 0 for r in group),
            dense_same_batch_harm=sum(r['dense_same_batch_delta'] > 0 for r in group),
            dense_development_harm=sum(r['dense_development_delta'] > 0 for r in group),
            lost_progress=sum(r['lost_progress'] > 0 for r in group),
            mean_same_batch_delta=mean(r['same_batch_delta'] for r in group),
            mean_development_delta=mean(r['development_delta'] for r in group),
            mean_dense_development_delta=mean(r['dense_development_delta'] for r in group),
            mean_lost_progress=mean(r['lost_progress'] for r in group))
        for name, column in (('ordinary', 'proposal_ordinary_bound'), ('complement', 'proposal_complement_bound'),
                             ('probe', 'proposal_probe_complement_bound')):
            margins = {str(tau): [tau*r['proposal_raw_alignment']-r[column] for r in group] for tau in taus}
            item[name+'_margin_ranges'] = {tau: [min(v), max(v)] for tau, v in margins.items()}
        strata.append(item)
    envelope={}
    for kind in sorted({r['basis_kind'] for r in unique.values()}):
        candidates=[r for r in unique.values() if r['basis_kind']==kind]
        envelope[kind]={}
        for score in ('ordinary','complement','probe_complement','relative'):
            grid=relative_thresholds if score=='relative' else taus
            result={}
            for threshold in grid:
                if score=='relative':
                    accepted=[r for r in candidates if r.get('proposal_relative_residual') is not None and r['proposal_relative_residual']<=threshold]
                else:
                    accepted=[r for r in candidates if r['proposal_raw_alignment']>0 and r['proposal_'+score+'_bound']<=threshold*r['proposal_raw_alignment']]
                result[str(threshold)]=dict(proposals=len(candidates),accepted=len(accepted),
                    first_order_harm=sum(r['proposal_true_alignment']<0 for r in accepted),
                    same_batch_harm=sum(r['same_batch_delta']>0 for r in accepted),
                    development_harm=sum(r['development_delta']>0 for r in accepted),
                    lost_progress=sum(r['lost_progress']>0 for r in accepted),
                    accepted_proposal_ids=[r['proposal_id'] for r in accepted])
            envelope[kind][score]=result
    distributions={}
    def distribution(values):
        values=sorted(values)
        def quantile(p):
            position=(len(values)-1)*p;index=int(position)
            return values[index]+(values[min(index+1,len(values)-1)]-values[index])*(position-index)
        return dict(n=len(values),**{name:quantile(p) if values else None for name,p in (('minimum',0),('q25',.25),('median',.5),('q75',.75),('maximum',1))})
    for kind in sorted(envelope):
        candidates=[r for r in unique.values() if r['basis_kind']==kind and r['proposal_raw_alignment']>0]
        distributions[kind]=dict(positive_alignment_proposals=len(candidates),
            complement_over_ordinary=distribution([r['proposal_complement_bound']/r['proposal_ordinary_bound'] for r in candidates if r['proposal_ordinary_bound']>0]),
            complement_over_a=distribution([r['proposal_complement_bound']/r['proposal_raw_alignment'] for r in candidates]))
    return dict(schema_version=1, bound_distributions=distributions, acceptance_envelope=envelope, scope='descriptive_synthetic_only', evaluation_proposals=len(unique),
                evaluation_layer_rows=len(evaluation), evaluation_query_ids=sorted({r['query_id'] for r in unique.values()}),
                strata=strata, loss_unit='mse_per_output_element',
                limitation='Shared checkpoints, paired seeds and duplicated early full/shift stages; these are not independent training-run estimates. Margins are algebraic without a validated NumericalPolicy.')


def main():
    out = Path('analysis')
    raw = out/'structural_oracle_audit.jsonl'
    summary_path = out/'structural_oracle_summary.json'
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    result = aggregate_rows(rows, summary['config']['taus'], summary['config']['relative_thresholds'])
    result['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (raw, summary_path)}
    (out/'structural_oracle_aggregate.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    curves = summary['curves']
    with (out/'structural_oracle_learning.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader(); writer.writerows(curves)
    (out/'structural_oracle_checkpoint_index.json').write_text(json.dumps(dict(schema_version=1,
        checkpoints=summary['artifact_index'], regeneration='PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli audit-checkpoints --config configs/structural_oracle.yaml',
        raw_tensors_ignored=True), indent=2, sort_keys=True)+'\n')
    print(f'{len(rows)} layer rows; {result["evaluation_proposals"]} evaluation proposals; {len(curves)} curve observations')


if __name__ == '__main__':
    main()
