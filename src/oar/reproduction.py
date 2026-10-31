"""Predeclared numerical/categorical comparison for the bounded synthetic study.

A same-seed repeat tests reproducibility, never independent scientific confirmation.
Timing, paths and regenerated tensor-byte hashes are deliberately not numerical keys.
"""
import math

NUMERIC_FIELDS = (
    'complement_bound', 'complement_norm', 'cosine', 'dense_development_delta',
    'dense_displacement_norm', 'dense_same_batch_delta', 'development_delta',
    'displacement_norm', 'energy', 'gradient_norm', 'lost_progress', 'ordinary_bound',
    'orthogonality', 'probe_complement_bound', 'probe_residual_bound', 'projected_norm',
    'projection_defect', 'proposal_complement_bound', 'proposal_ordinary_bound',
    'proposal_probe_complement_bound', 'proposal_raw_alignment', 'proposal_relative_residual',
    'proposal_true_alignment', 'raw_alignment', 'relative_residual', 'residual_norm',
    'same_batch_delta', 'true_alignment',
)
CATEGORICAL_FIELDS = (
    'actual_rank', 'basis_age', 'basis_kind', 'basis_source_id', 'basis_source_step',
    'condition', 'count_unit', 'development_id', 'layer', 'loss_unit', 'missing_reason',
    'numerical_allowance', 'numerical_reason', 'observations', 'online_eligible',
    'proposal_id', 'query_id', 'query_step', 'rank', 'runtime_evidence_reason',
    'schema_version', 'seed', 'split',
)
POLICY = dict(atol=2e-6, rtol=2e-4, taus=[.25, .5, .75], relative_thresholds=[.1, .25, .5, 1., 2.],
              row_key=['proposal_id', 'layer'], numeric_fields=list(NUMERIC_FIELDS),
              categorical_fields=list(CATEGORICAL_FIELDS),
              curve_key=['condition', 'seed', 'step', 'target_phase'], curve_field='development_mse',
              sign_fields=['proposal_true_alignment', 'same_batch_delta', 'development_delta', 'lost_progress',
                           'dense_same_batch_delta', 'dense_development_delta'],
              excluded='wall times, resource snapshots, source revision, output paths, regenerated tensor/basis hashes',
              interpretation='same-seed numerical reproducibility only; not independent replication or confirmation')


def validate_clone(original, clone):
    """Allow only the two independent output locations, preserving all science."""
    original = {k: v for k, v in original.items() if k not in ('output', 'checkpoints_dir')}
    clone = {k: v for k, v in clone.items() if k not in ('output', 'checkpoints_dir')}
    if original != clone:
        raise ValueError('reproduction changes scientific configuration')


def compare_rows(original, repeated):
    """Compare complete keyed rows; retain every failure instead of a chosen prefix."""
    failures, maxima = [], {key: 0. for key in NUMERIC_FIELDS}
    def keyed(rows, label):
        result = {}
        for row in rows:
            try:
                key = tuple(row[k] for k in POLICY['row_key'])
                if key in result:
                    failures.append(f'{label}: duplicate row {key}')
                result[key] = row
            except KeyError as exc:
                failures.append(f'{label}: missing row identity {exc}')
        return result
    left, right = keyed(original, 'reference'), keyed(repeated, 'repeat')
    if left.keys() != right.keys():
        failures.append(f'row coverage differs: reference={len(left)} repeat={len(right)}')
    def finite(value):
        return type(value) in (float, int) and math.isfinite(value)
    for key in sorted(left.keys() & right.keys()):
        a, b = left[key], right[key]
        for field in CATEGORICAL_FIELDS:
            if field not in a or field not in b or a[field] != b[field]:
                failures.append(f'{key}: categorical {field} differs/missing')
        for field in NUMERIC_FIELDS:
            x, y = a.get(field), b.get(field)
            if field not in a or field not in b:
                failures.append(f'{key}: missing numeric {field}')
            elif x is None and y is None:
                continue
            elif not finite(x) or not finite(y):
                failures.append(f'{key}: nonfinite/missing {field}')
            else:
                maxima[field] = max(maxima[field], abs(x-y))
                if not math.isclose(x, y, abs_tol=POLICY['atol'], rel_tol=POLICY['rtol']):
                    failures.append(f'{key}: numerical {field} differs: {x} vs {y}')
        for field in POLICY['sign_fields']:
            x, y = a.get(field), b.get(field)
            if finite(x) and finite(y) and (x > 0, x < 0) != (y > 0, y < 0):
                failures.append(f'{key}: sign {field} differs')
        for field in ('proposal_ordinary_bound', 'proposal_complement_bound', 'proposal_probe_complement_bound'):
            for tau in POLICY['taus']:
                values = [r.get(k) for r in (a, b) for k in (field, 'proposal_raw_alignment')]
                if all(finite(v) for v in values):
                    decisions = [r['proposal_raw_alignment'] > 0 and r[field] <= tau*r['proposal_raw_alignment'] for r in (a, b)]
                    if decisions[0] != decisions[1]:
                        failures.append(f'{key}: acceptance {field}/tau{tau} differs')
        for threshold in POLICY['relative_thresholds']:
            values = [r.get('proposal_relative_residual') for r in (a, b)]
            if all(finite(v) for v in values) and (values[0] <= threshold) != (values[1] <= threshold):
                failures.append(f'{key}: relative acceptance/{threshold} differs')
    return dict(status='failed' if failures else 'reproduced', failures=failures,
                rows_compared=len(left.keys() & right.keys()), max_absolute_difference=maxima, policy=POLICY)


def compare_curves(original, repeated):
    """Map predeclared logical `step` to the existing `committed_updates` schema.

    This is a field-name correction only: keys, observations and tolerances
    retain their intended meaning. The original declaration remains unchanged.
    """
    failures = []
    def keyed(rows, label):
        result = {}
        for row in rows:
            try:
                key = tuple(row['committed_updates' if k == 'step' else k] for k in POLICY['curve_key'])
                if key in result:
                    failures.append(f'{label}: duplicate curve {key}')
                result[key] = row[POLICY['curve_field']]
            except KeyError as exc:
                failures.append(f'{label}: missing curve field {exc}')
        return result
    a, b = keyed(original, 'reference'), keyed(repeated, 'repeat')
    if a.keys() != b.keys():
        failures.append('curve coverage/identity differs')
    differences = []
    for key in a.keys() & b.keys():
        x, y = a[key], b[key]
        if (type(x) not in (int, float) or type(y) not in (int, float) or
                not math.isfinite(x) or not math.isfinite(y)):
            failures.append(f'curve {key} is nonfinite')
            continue
        differences.append(abs(x-y))
        if not math.isclose(x, y, abs_tol=POLICY['atol'], rel_tol=POLICY['rtol']):
            failures.append(f'curve {key} differs')
    return dict(failures=failures, curves_compared=len(a.keys() & b.keys()),
                curve_max_absolute_difference=max(differences, default=None),
                curve_schema_mapping='predeclared logical step = existing committed_updates; no observation/tolerance change')
