"""Prepare, then execute one full path-only repeat of the locked synthetic study.

Preparation durably writes the exact config, comparison policy and amendment.
Commit these files before executing --run. This helper never relaxes the oracle
permission check; it temporarily appends the predeclared path-only binding and
restores the previous latest amendment in a finally block. Run exclusively.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from oar.oracle import load_oracle_config, require_structural_oracle, sha, write_json
from oar.reproduction import POLICY, compare_curves, compare_rows, validate_clone


def prepare(out):
    root = Path('.').resolve()
    out = out.resolve()
    if not out.is_relative_to(root/'analysis/reproduction') or out == root/'analysis/reproduction':
        raise ValueError('use a new directory inside analysis/reproduction')
    if out.exists():
        raise ValueError('output already exists; never overwrite a repeat')
    original_path = Path('configs/structural_oracle.yaml')
    original = load_oracle_config(original_path)
    amendment = require_structural_oracle(original, original_path)
    config = yaml.safe_load(original_path.read_text())
    config.update(output=str(out.relative_to(root)), checkpoints_dir=f'checkpoints/reproduction/{out.name}')
    validate_clone(yaml.safe_load(original_path.read_text()), config)
    if Path(config['checkpoints_dir']).exists():
        raise ValueError('independent checkpoint location already exists')
    out.mkdir(parents=True)
    clone = out/'config.yaml'
    clone.write_text(yaml.safe_dump(config, sort_keys=False))
    amendment = dict(amendment, config_sha256=sha(clone))
    write_json(out/'amendment.json', amendment)
    plan = dict(schema_version=1, kind='full_path_only_same_seed_reproduction', comparison=POLICY,
                config_path=str(clone.relative_to(root)), config_sha256=sha(clone),
                original_config_path=str(original_path), original_config_sha256=sha(original_path),
                amendment_sha256=sha(out/'amendment.json'),
                reference={name:sha(name) for name in ('analysis/structural_oracle_summary.json','analysis/structural_oracle_audit.jsonl')},
                source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                sources={name:sha(name) for name in ('src/oar/oracle.py','src/oar/reproduction.py','analysis/reproduce_structural_oracle.py')},
                scope_amendment='Only output/checkpoint paths change. Same full shape, seeds, histories, queries, ranks, ages, LR, grid and 600s ceiling. No benchmark, corpus, trainer/controller or confirmation. Append amendment.json before execution; restore preceding permission afterwards.',
                original_research_revision='2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07')
    write_json(out/'plan.json', plan)
    print(f'Prepared {out.relative_to(root)}/plan.json; commit declaration before --run')


def run(plan_path):
    plan_path = plan_path.resolve()
    out = plan_path.parent
    root = Path('.').resolve()
    if not out.is_relative_to(root/'analysis/reproduction') or (out/'comparison.json').exists():
        raise ValueError('require a new prepared reproduction within this repository')
    plan = json.loads(plan_path.read_text())
    if plan['schema_version'] != 1 or plan['comparison'] != POLICY or plan['kind'] != 'full_path_only_same_seed_reproduction':
        raise ValueError('unsupported or changed predeclared comparison policy')
    for name, digest in {**plan['reference'], **plan['sources'], plan['config_path']:plan['config_sha256'],
                         plan['original_config_path']:plan['original_config_sha256']}.items():
        if sha(name) != digest:
            raise ValueError(f'prepared input/source changed: {name}')
    if sha(out/'amendment.json') != plan['amendment_sha256']:
        raise ValueError('prepared amendment changed')
    original = load_oracle_config(Path(plan['original_config_path']))
    require_structural_oracle(original, Path(plan['original_config_path']))
    clone = yaml.safe_load(Path(plan['config_path']).read_text())
    validate_clone(yaml.safe_load(Path(plan['original_config_path']).read_text()), clone)
    if Path(clone['checkpoints_dir']).exists() or (out/'structural_oracle_summary.json').exists():
        raise ValueError('refuse to overwrite previous raw repeat/checkpoints')
    ledger = Path('decisions/amendments.jsonl')
    prior = ledger.read_text().splitlines()[-1]
    amendment = (out/'amendment.json').read_text()
    def append(value):
        with ledger.open('a') as stream:
            stream.write(json.dumps(json.loads(value), sort_keys=True)+'\n'); stream.flush(); os.fsync(stream.fileno())
    result = dict(schema_version=1, status='failed', failures=[], plan_sha256=sha(plan_path),
                  executed_config_path=plan['config_path'], executed_config_sha256=plan['config_sha256'],
                  original_reference_config_sha256=plan['original_config_sha256'],
                  original_research_revision=plan['original_research_revision'],
                  reproduction_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  interpretation=POLICY['interpretation'])
    started = time.perf_counter()
    try:
        append(amendment)
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK='0')
        command = [sys.executable, '-m', 'oar.cli', 'audit-checkpoints', '--config', plan['config_path']]
        with (out/'invocation.log').open('w') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        result['invocation_wall_s'] = time.perf_counter()-started
        result['command'] = command; result['exit_code'] = process.returncode
        if process.returncode:
            raise RuntimeError(f'fresh MPS invocation failed: exit {process.returncode}; see invocation.log')
        summary_path = out/'structural_oracle_summary.json'
        summary = json.loads(summary_path.read_text())
        baseline = json.loads(Path('analysis/structural_oracle_summary.json').read_text())
        rows = [json.loads(line) for line in (out/'structural_oracle_audit.jsonl').read_text().splitlines()]
        reference = [json.loads(line) for line in Path('analysis/structural_oracle_audit.jsonl').read_text().splitlines()]
        result.update(compare_rows(reference, rows))
        failures = result['failures']
        if not summary['coverage_complete'] or summary['row_count'] != len(rows) or summary['outcome'] != baseline['outcome']:
            failures.append('incomplete/different study outcome')
        if summary['fallback_environment'] != '0' or summary['config']['device'] != 'mps' or summary['config'] != clone:
            failures.append('fresh actual-MPS/config provenance differs')
        curve_result = compare_curves(baseline['curves'], summary['curves'])
        failures.extend(curve_result.pop('failures'))
        result.update(curve_result)
        attempts = lambda s: {(r['condition'],r['seed']):r['learnable'] for r in s['attempts']}
        if attempts(summary) != attempts(baseline): failures.append('learnability/attempt coverage differs')
        if len(summary['artifact_index']) != len(baseline['artifact_index']): failures.append('checkpoint coverage differs')
        result['checkpoint_hashes_validated'] = 0
        for artifact in summary['artifact_index']:
            if sha(artifact['path']) != artifact['sha256']:
                failures.append(f'checkpoint byte hash differs: {artifact["path"]}')
            else: result['checkpoint_hashes_validated'] += 1
        result['worker_wall_s'] = summary['total_wall_s']
        result['endpoint_rss_bytes'] = summary['rss_at_finish_bytes']
        result['endpoint_mps_driver_bytes'] = summary['mps_driver_at_finish_bytes']
        result['status'] = 'failed' if failures else 'reproduced'
    except Exception as exc:
        result['status'] = 'failed'; result['failures'].append(f'{type(exc).__name__}: {exc}')
    finally:
        append(prior)
        result['total_helper_wall_s'] = time.perf_counter()-started
        result['artifacts'] = {str(p.relative_to(root)):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'comparison.json'}
        write_json(out/'comparison.json', result)
    print(json.dumps({k:result[k] for k in ('status','failures','total_helper_wall_s')}, sort_keys=True))
    return 0 if result['status'] == 'reproduced' else 1


def recover_comparison(plan_path):
    """Reprocess retained bytes after the documented curve-field schema error.

    Original failure, declaration and raw evidence are immutable inputs. This
    path launches no experiment and changes no gate or amendment.
    """
    out = plan_path.parent
    target = out/'comparison_recovered.json'
    if target.exists():
        raise ValueError('recovered comparison exists; preserve previous outcome')
    plan = json.loads(plan_path.read_text())
    failed = json.loads((out/'comparison.json').read_text())
    if (plan['comparison'] != POLICY or failed['status'] != 'failed' or
            failed['failures'] != ["KeyError: 'step'"] or sha(plan_path) != failed['plan_sha256']):
        raise ValueError('recovery applies only to the retained curve-schema failure')
    for name, digest in {**plan['reference'], **failed['artifacts']}.items():
        if sha(name) != digest:
            raise ValueError(f'retained comparison input changed: {name}')
    baseline = json.loads(Path('analysis/structural_oracle_summary.json').read_text())
    summary = json.loads((out/'structural_oracle_summary.json').read_text())
    rows = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
    result = compare_rows(rows(Path('analysis/structural_oracle_audit.jsonl')), rows(out/'structural_oracle_audit.jsonl'))
    curves = compare_curves(baseline['curves'], summary['curves'])
    result['failures'].extend(curves.pop('failures')); result.update(curves)
    clone = yaml.safe_load(Path(plan['config_path']).read_text())
    validate_clone(yaml.safe_load(Path(plan['original_config_path']).read_text()), clone)
    if (sha(plan['config_path']) != plan['config_sha256'] or summary['config'] != clone or
            not summary['coverage_complete'] or summary['row_count'] != len(rows(out/'structural_oracle_audit.jsonl')) or
            summary['outcome'] != baseline['outcome'] or summary['fallback_environment'] != '0' or clone['device'] != 'mps'):
        result['failures'].append('coverage/backend/config provenance differs')
    attempts = lambda s: {(r['condition'],r['seed']):r['learnable'] for r in s['attempts']}
    if attempts(summary) != attempts(baseline): result['failures'].append('learnability/attempt coverage differs')
    result['checkpoint_hashes_validated'] = 0
    original_keys = {(r['condition'],r['seed'],r['step']) for r in baseline['artifact_index']}
    repeated_keys = {(r['condition'],r['seed'],r['step']) for r in summary['artifact_index']}
    if original_keys != repeated_keys or len(repeated_keys) != len(summary['artifact_index']):
        result['failures'].append('checkpoint coverage/identity differs')
    for artifact in summary['artifact_index']:
        if sha(artifact['path']) != artifact['sha256']:
            result['failures'].append('checkpoint hash differs: '+artifact['path'])
        else: result['checkpoint_hashes_validated'] += 1
    result.update(schema_version=1, status='failed' if result['failures'] else 'reproduced',
                  original_failure_path=str(out/'comparison.json'), original_failure_sha256=sha(out/'comparison.json'),
                  plan_sha256=sha(plan_path), recovery_sources={p:sha(p) for p in ('src/oar/reproduction.py','analysis/reproduce_structural_oracle.py')},
                  original_research_revision=plan['original_research_revision'], reproduction_revision=summary['code_revision'],
                  interpretation=POLICY['interpretation'],
                  resource_accounting=dict(worker_wall_s=summary['total_wall_s'], invocation_wall_s=failed['invocation_wall_s'],
                      endpoint_rss_bytes=summary['rss_at_finish_bytes'], endpoint_mps_driver_bytes=summary['mps_driver_at_finish_bytes']),
                  timing_qualification='A report-only CPU pytest invocation (1.27s) may have overlapped this repeat. No MPS test/collector/benchmark overlap. Wall/resource values are accounting only, never a timing replication or new speed estimate.',
                  postprocessing_correction='Original helper compared all792 rows successfully then failed on nonexistent curve key step. Recovered comparison maps logical step to committed_updates on all30 retained curves, with unchanged numeric tolerance and exact categorical checks. No research rerun.')
    write_json(target, result)
    print(json.dumps({k:result[k] for k in ('status','failures','rows_compared','curves_compared','checkpoint_hashes_validated')},sort_keys=True))
    return 0 if result['status'] == 'reproduced' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path, metavar='NEW_OUTPUT_DIRECTORY')
    group.add_argument('--run', type=Path, metavar='PREDECLARED_PLAN_JSON')
    group.add_argument('--recover-comparison', type=Path, metavar='FAILED_PLAN_JSON')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare)
    elif args.run:
        raise SystemExit(run(args.run))
    else:
        raise SystemExit(recover_comparison(args.recover_comparison))
