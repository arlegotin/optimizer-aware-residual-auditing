# SPDX-License-Identifier: AGPL-3.0-only
"""Post-hoc, stdlib-only analysis of sealed metadata; never loads model/data."""
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]
ROOT = RUN.parents[2]
PILOT = RUN / 'pilot/attempt-0004'
EXPECTED_RECORD = '3e15fe5e4526ae001b5112a896b69d43669a2044aa60f4febc7c5ee05125f7bd'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def target_time(observations, target, cutoff):
    previous = False
    for row in observations:
        if row['completed_s'] >= cutoff:
            break
        below = row['nll'] <= target
        if previous and below:
            return row['completed_s']
        previous = below
    return None


def norms(steps):
    values = [step['norm'] for step in steps]
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    return {'updates': len(values), 'fraction_norm_gt_1': sum(v > 1 for v in values)/len(values),
            'median_preclip_norm': statistics.median(values),
            'median_nominal_clip_factor': statistics.median(1/max(1, v) for v in values),
            'minimum_preclip_norm': min(values), 'maximum_preclip_norm': max(values)}


def main():
    started = time.perf_counter()
    record_path = PILOT / 'record.json'
    assert digest(record_path) == EXPECTED_RECORD
    record = json.loads(record_path.read_text())
    assert record['status'] == 'STOP_QUALITY' and not record['learning_success']
    assert record['candidate'] is None and not record['eligible_candidates']
    for relative, expected in record['artifacts'].items():
        path = (PILOT / relative).resolve()
        assert path.is_relative_to(PILOT) and digest(path) == expected, relative
    identity = json.loads((PILOT / 'source_manifest.json').read_text())
    assert digest(PILOT/'source_manifest.json') == record['source_manifest_sha256']
    frozen_path = RUN / 'prospective/learning-v9.json'
    frozen = json.loads(frozen_path.read_text())
    horizon = frozen['calibration']['horizon']
    target = record['q_star']
    reference_nll = record['R0']['endpoint_full']['nll']
    quality_limit = reference_nll + record['epsilon_q']
    aggregate = json.loads((PILOT / 'grid-results.json').read_text())
    assert len(aggregate) == len(frozen['grid']) == 27
    by_id = {row['id']: row for row in aggregate}
    assert len(by_id) == 27 and set(by_id) == {row['id'] for row in frozen['grid']}
    rows, curves, hashes = [], [], {}
    for trial in frozen['grid']:
        path = PILOT / (trial['id'] + '-row.json')
        raw = json.loads(path.read_text())
        assert raw == by_id[trial['id']] and raw['arm'] == trial['arm']
        assert raw['seed'] == 3101 and raw['status'] == 'pass' and raw['restart'] is None
        assert raw['canonical_completion_s'] <= horizon['T_budget']
        assert raw['cutoff_overshoot_s'] <= horizon['overshoot_limit_s']
        steps = raw['steps']
        assert len(steps) == raw['logical_updates']
        assert [s['logical_update'] for s in steps] == list(range(1, len(steps)+1))
        assert sum(s['valid_targets'] for s in steps) == raw['valid_tokens']
        assert sum(s['physical_rows'] for s in steps) == raw['physical_tokens']
        observations = [raw['initial_monitor'], *raw['monitors']]
        reached = target_time(observations, target, horizon['H'])
        endpoint = raw['endpoint_full']['nll']
        family = raw['arm']['sampler'] or 'dense'
        row = {'id': raw['id'], 'family': family, **raw['arm'], 'seed': raw['seed'],
               'full_development_nll': endpoint, 'nll_difference_from_R0': endpoint-reference_nll,
               'confirmed_target_time_s': reached, 'target_censored': reached is None,
               'endpoint_guard_pass': endpoint <= quality_limit,
               'joint_eligible': reached is not None and endpoint <= quality_limit,
               'logical_updates': raw['logical_updates'], 'valid_targets': raw['valid_tokens'],
               'physical_targets': raw['physical_tokens'], 'canonical_completion_s': raw['canonical_completion_s'],
               'cutoff_overshoot_s': raw['cutoff_overshoot_s'], 'unused_reserve_s': raw['unused_reserve_s'],
               'periodic_monitors': len(raw['monitors']), 'all_updates': norms(steps),
               'last_quarter_blocks': norms([s for s in steps if s['block_end_s'] >= .75*horizon['H']])}
        rows.append(row)
        for i, observation in enumerate(observations):
            curves.append({'id': raw['id'], 'family': family, 'kind': 'initial' if i == 0 else 'periodic',
                           'completed_s': observation['completed_s'], 'nll': observation['nll'],
                           'valid_targets': observation['valid_targets']})
        hashes[str(path.relative_to(ROOT))] = digest(path)
    assert len({r['initial_parameters_sha256'] for r in aggregate}) == 1
    assert not any(row['joint_eligible'] for row in rows if row['family'] != 'dense')
    families = {}
    for family in ('dense', 'importance', 'uniform'):
        group = [row for row in rows if row['family'] == family]
        best = min(group, key=lambda row: row['full_development_nll'])
        families[family] = {'trials': len(group), 'best_endpoint': best,
                            'summed_canonical_s': sum(row['canonical_completion_s'] for row in group),
                            'censored_target_trials': sum(row['target_censored'] for row in group),
                            'endpoint_guard_trials': sum(row['endpoint_guard_pass'] for row in group)}
    result = {'schema_version': 1, 'status': 'STOP_QUALITY', 'analysis_kind': 'post_hoc_descriptive_metadata_only',
              'training_seed_count': 1, 'analysis_code_sha256': digest(Path(__file__)),
              'record_sha256': EXPECTED_RECORD, 'source_manifest_sha256': digest(PILOT/'source_manifest.json'),
              'learning_manifest_sha256': digest(frozen_path), 'input_row_hashes': hashes,
              'measured_launch_head': identity['actual_head'], 'measured_source_hashes': identity['measurement_hashes'],
              'horizon': horizon, 'q_star': target, 'epsilon_q': record['epsilon_q'],
              'R0_nll': reference_nll, 'endpoint_quality_limit': quality_limit,
              'selected_dense_id': record['dense']['id'], 'selected_dense_target_s': record['dense']['target_time_s'],
              'pilot_invocation_wall_s': record['invocation_wall_s'],
              'summed_canonical_s': sum(row['canonical_completion_s'] for row in rows),
              'families': families, 'rows': rows,
              'limits': ['No independent-seed learning CI or positive target-time claim.',
                         'Norm/clip summaries are post hoc, at different parameters and progress; no causal inference.',
                         'Block completion gives coarse time bins, not individual update timestamps.',
                         'All protected validation/test losses and further learning stages were not run.']}
    result['analysis_wall_s_before_publication'] = time.perf_counter()-started
    write_new(HERE/'recorded-grid-analysis.json', result)
    fields = ['id', 'family', 'batch', 'rho', 'lr', 'full_development_nll',
              'confirmed_target_time_s', 'target_censored', 'endpoint_guard_pass',
              'logical_updates', 'valid_targets', 'physical_targets', 'canonical_completion_s']
    with (HERE/'all-grid-outcomes.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    with (HERE/'development-monitor-curves.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'family', 'kind', 'completed_s', 'nll', 'valid_targets'])
        writer.writeheader();writer.writerows(curves)
    print(json.dumps({'status': result['status'], 'trials': len(rows), 'q_star': target,
                      'endpoint_quality_limit': quality_limit,
                      'families': {name: {'trials': value['trials'],
                          'best_nll': value['best_endpoint']['full_development_nll'],
                          'censored': value['censored_target_trials']} for name,value in families.items()},
                      'analysis_wall_s': time.perf_counter()-started}, indent=2))


if __name__ == '__main__':
    main()
