# SPDX-License-Identifier: AGPL-3.0-only
"""Replay immutable public metadata analysis into a new directory, without Torch."""
import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    original = Path(__file__).resolve().parent
    source = original/'analyze-recorded-grid.py'
    reference = json.loads((original/'recorded-grid-analysis.json').read_text())
    assert sha(source) == reference['analysis_code_sha256']
    output = args.out.resolve()
    assert not output.is_relative_to(original.parents[1]/'pilot')
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('sealed_grid_metadata_replay', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only the publication directory moves; frozen input roots/source stay intact.
    module.HERE = output
    module.main()
    reproduced = json.loads((output/'recorded-grid-analysis.json').read_text())
    for value in (reference, reproduced):
        value.pop('analysis_wall_s_before_publication')
    assert reference == reproduced
    names = ('all-grid-outcomes.csv', 'development-monitor-curves.csv')
    for name in names:
        assert sha(original/name) == sha(output/name), name
    receipt = {'schema_version': 1, 'status': 'pass', 'kind': 'public_metadata_replay_only',
               'scientific_state': reproduced['status'], 'training_repeated': False,
               'protected_losses_accessed': False, 'python': platform.python_version(),
               'original_analysis_code_sha256': sha(source), 'replay_code_sha256': sha(Path(__file__)),
               'record_sha256': reproduced['record_sha256'],
               'all_scientific_fields_identical': True,
               'excluded_comparison_field': 'analysis_wall_s_before_publication',
               'csv_sha256': {name: sha(output/name) for name in names},
               'replay_wall_s_before_receipt': time.perf_counter()-started}
    with (output/'reproduction-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'replay_status': 'pass', 'out': str(output), 'training_repeated': False}))


if __name__ == '__main__':
    main()
