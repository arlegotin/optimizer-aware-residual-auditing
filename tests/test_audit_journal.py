"""Real durable storage, fresh streams, stale writers and torn-write recovery."""

from dataclasses import replace
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from oar.audit_journal import AuditJournal, JournalError
from oar.records import AuditCandidate, from_payload, to_payload


def test_crash_does_not_refund_observed_test(tmp_path):
    path = tmp_path / 'audit.jsonl'
    first = AuditJournal(path, .01, 0).reserve(('mlp.weight',), 1)[0]
    retry = AuditJournal(path, .01, 0).reserve(('mlp.weight',), 1)[0]
    assert retry.index == first.index + 1
    assert retry.probe_id != first.probe_id
    assert retry.seed != first.seed
    assert retry.delta < first.delta
    assert first.delta == pytest.approx(.006079271018540267)
    assert retry.delta == pytest.approx(.0015198177546350667)
    assert from_payload(to_payload(first)) == first


def test_reserve_each_layer_including_zero_and_rejected_tests(tmp_path):
    journal = AuditJournal(tmp_path / 'audit.jsonl', .01, 0)
    allocated = journal.reserve(('zero', 'rejected', 'accepted'), 2)
    assert [r.index for r in allocated] == [1, 2, 3]
    assert [r.layer_id for r in allocated] == ['zero', 'rejected', 'accepted']
    assert sum(r.delta for r in allocated) < .01
    assert journal.counter == 3


def test_checkpoint_and_journal_take_durable_monotone_max(tmp_path):
    path = tmp_path / 'audit.jsonl'
    old = AuditJournal(path, .01, 0)
    old.reserve(('w',), 1)
    ahead = AuditJournal(path, .01, 50)
    assert ahead.counter == 50
    assert AuditJournal(path, .01, 0).counter == 50
    assert old.reserve(('w',), 2)[0].index == 51
    assert AuditJournal(path, .01, 2).counter == 51


@pytest.mark.parametrize('contents', [b'', b'{', b'{}\n', b'{"schema_version":99}\n'])
def test_malformed_or_empty_existing_journal_never_silently_resets(tmp_path, contents):
    path = tmp_path / 'audit.jsonl'
    path.write_bytes(contents)
    with pytest.raises(JournalError):
        AuditJournal(path, .01, 0)
    assert path.read_bytes() == contents


@pytest.mark.parametrize('damage', ['truncate', 'unknown_field', 'nan', 'duplicate_row', 'flip_seed'])
def test_corrupt_final_evidence_fails_closed(tmp_path, damage):
    path = tmp_path / 'audit.jsonl'
    AuditJournal(path, .01, 0).reserve(('w',), 1)
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if damage == 'truncate':
        broken = raw[:-7]
    elif damage == 'duplicate_row':
        broken = raw + lines[-1]
    else:
        row = json.loads(lines[-1])
        if damage == 'unknown_field':
            row['unrecognized'] = 1
        elif damage == 'nan':
            row['reservation']['delta'] = math.nan
        else:
            row['reservation']['seed'] += 1
        broken = b''.join(lines[:-1]) + json.dumps(row).encode() + b'\n'
    path.write_bytes(broken)
    with pytest.raises(JournalError):
        AuditJournal(path, .01, 100)
    assert path.read_bytes() == broken


def test_durable_fsync_failure_exposes_no_probe_and_retry_spends_fresh(tmp_path, monkeypatch):
    path = tmp_path / 'audit.jsonl'
    journal = AuditJournal(path, .01, 0)
    real_fsync = os.fsync
    def interrupted(_fd):
        raise OSError('simulated interruption after write before durable return')
    monkeypatch.setattr(os, 'fsync', interrupted)
    with pytest.raises(OSError):
        journal.reserve(('w',), 1)
    monkeypatch.setattr(os, 'fsync', real_fsync)
    assert AuditJournal(path, .01, 0).reserve(('w',), 1)[0].index == 2


def test_stale_instances_and_parallel_reservations_cannot_reuse_indices(tmp_path):
    path = tmp_path / 'audit.jsonl'
    journals = [AuditJournal(path, .01, 0) for _ in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        allocated = list(pool.map(lambda j: j.reserve(('w',), 1)[0], journals))
    assert sorted(r.index for r in allocated) == list(range(1, 13))
    assert len({r.probe_id for r in allocated}) == 12
    assert len({r.seed for r in allocated}) == 12


def test_probe_stream_is_durable_domain_separated_and_model_rng_unchanged(tmp_path):
    path = tmp_path / 'audit.jsonl'
    journal = AuditJournal(path, .01, 0, audit_seed=123)
    first = journal.reserve(('w',), 3)[0]
    candidate = AuditCandidate('w', 3, 2, 1, 1)
    before = torch.get_rng_state().clone()
    z = journal.draw_probes(first, candidate, 5, 8, dtype=torch.float64)
    assert torch.equal(before, torch.get_rng_state())
    reopened = AuditJournal(path, .01, 0, audit_seed=123)
    reproduced = torch.from_numpy(np.random.Generator(np.random.PCG64(first.seed)).standard_normal((5, 8)))
    torch.testing.assert_close(z, reproduced, rtol=0, atol=0)
    with pytest.raises(JournalError):
        reopened.draw_probes(first, candidate, 5, 8)
    with pytest.raises(JournalError):
        journal.draw_probes(first, candidate, 5, 8)
    raw_construction = torch.randn(5, 8, generator=torch.Generator().manual_seed(123), dtype=torch.float64)
    assert not torch.equal(z, raw_construction)
    second = reopened.reserve(('w',), 3)[0]
    assert not torch.equal(z, reopened.draw_probes(second, candidate, 5, 8, dtype=torch.float64))
    for forged in (replace(first, seed=first.seed + 1), replace(first, delta=.1)):
        with pytest.raises(JournalError):
            reopened.draw_probes(forged, candidate, 5, 8)
    fresh = reopened.reserve(('w',), 3)[0]
    with pytest.raises(ValueError):
        reopened.draw_probes(fresh, replace(candidate, construction_stream='audit'), 5, 8)


@pytest.mark.parametrize('delta,counter,seed', [(0., 0, 0), (1., 0, 0), (math.nan, 0, 0), (.01, -1, 0), (.01, True, 0), (.01, 0, -1)])
def test_invalid_journal_configuration_cannot_allocate(tmp_path, delta, counter, seed):
    with pytest.raises(ValueError):
        AuditJournal(tmp_path / 'audit.jsonl', delta, counter, audit_seed=seed)


def test_run_identity_and_budget_cannot_change_on_resume(tmp_path):
    path = tmp_path / 'audit.jsonl'
    AuditJournal(path, .01, 0, audit_seed=12).reserve(('w',), 1)
    for delta, seed in [(.02, 12), (.01, 13)]:
        with pytest.raises(JournalError):
            AuditJournal(path, delta, 0, audit_seed=seed)


def test_invalid_reservation_and_underflow_never_expose_a_test(tmp_path):
    journal = AuditJournal(tmp_path / 'audit.jsonl', .01, 0)
    for layers, step in [(('w', 'w'), 1), (('',), 1), (('w',), -1), (('w',), True)]:
        with pytest.raises(ValueError):
            journal.reserve(layers, step)
    assert journal.counter == 0
    tiny = AuditJournal(tmp_path / 'tiny.jsonl', math.ulp(0.), 0)
    with pytest.raises(JournalError, match='underflow|represent'):
        tiny.reserve(('w',), 1)


def test_process_interruption_after_write_does_not_reuse_a_test(tmp_path):
    import subprocess
    import sys
    path = tmp_path / 'audit.jsonl'
    AuditJournal(path, .01, 0)
    # Hard process exit during fsync: no Python cleanup and no reservation
    # reaches the caller. Complete appended evidence still consumes index 1.
    program = '''
import os
import sys
from pathlib import Path
from oar.audit_journal import AuditJournal
j = AuditJournal(Path(sys.argv[1]), .01, 0)
os.fsync = lambda fd: os._exit(23)
j.reserve(('w',), 1)
'''
    result = subprocess.run([sys.executable, '-c', program, str(path)], capture_output=True, text=True)
    assert result.returncode == 23, result.stderr
    assert AuditJournal(path, .01, 0).reserve(('w',), 1)[0].index == 2


def test_removing_journal_from_live_instance_cannot_create_a_replacement(tmp_path):
    path = tmp_path / 'audit.jsonl'
    journal = AuditJournal(path, .01, 0)
    journal.reserve(('w',), 1)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        journal.reserve(('w',), 1)
    assert not path.exists()


def test_probe_generation_uses_full_seed_bits_after_large_checkpoint_jump(tmp_path):
    path = tmp_path / 'audit.jsonl'
    journal = AuditJournal(path, .01, 0, audit_seed=17)
    first = journal.reserve(('w',), 3)[0]
    candidate = AuditCandidate('w', 3, 1, 1, 1)
    first_z = journal.draw_probes(first, candidate, 5, 8, dtype=torch.float64)
    resumed = AuditJournal(path, .01, 2 ** 32, audit_seed=17)
    later = resumed.reserve(('w',), 3)[0]
    assert (later.seed - first.seed) % (2 ** 63) == 2 ** 32
    later_z = resumed.draw_probes(later, candidate, 5, 8, dtype=torch.float64)
    assert not torch.equal(first_z, later_z)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='requires actual visible MPS')
def test_gaussian_transfer_preserves_both_model_rng_streams(tmp_path):
    journal = AuditJournal(tmp_path / 'audit.jsonl', .01, 0)
    reservation = journal.reserve(('w',), 3)[0]
    candidate = AuditCandidate('w', 3, 1, 1, 1)
    cpu_before = torch.get_rng_state().clone()
    mps_before = torch.mps.get_rng_state().clone()
    z = journal.draw_probes(reservation, candidate, 5, 8, device='mps')
    expected = np.random.Generator(np.random.PCG64(reservation.seed)).standard_normal((5, 8), dtype=np.float32)
    assert z.device.type == 'mps'
    torch.testing.assert_close(z.cpu(), torch.from_numpy(expected), rtol=0, atol=0)
    assert torch.equal(cpu_before, torch.get_rng_state())
    assert torch.equal(mps_before, torch.mps.get_rng_state())


@pytest.mark.parametrize('field,value', [('schema_version', 2), ('numpy_version', 'unqualified'), ('unexpected', 0)])
def test_unsupported_header_rejected_even_with_consistent_content_hash(tmp_path, field, value):
    import hashlib
    path = tmp_path / 'audit.jsonl'
    AuditJournal(path, .01, 0)
    row = json.loads(path.read_text())
    row.pop('sha256')
    row[field] = value
    row['sha256'] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(JournalError):
        AuditJournal(path, .01, 0)


def test_other_current_layer_probe_cannot_enter_candidate_construction(tmp_path):
    journal = AuditJournal(tmp_path / 'audit.jsonl', .01, 0)
    first, other = journal.reserve(('w', 'other'), 3)
    candidate = AuditCandidate('w', 3, 1, 1, 1, probe_ids_used=(other.probe_id,))
    with pytest.raises(ValueError):
        journal.draw_probes(first, candidate, 5, 8)


@pytest.mark.parametrize('counter,require_existing', [(0, True), (5, False), (5, True)])
def test_missing_resume_journal_cannot_restart_from_stale_checkpoint(tmp_path, counter, require_existing):
    path = tmp_path / 'lost.jsonl'
    with pytest.raises(JournalError, match='missing|resume'):
        AuditJournal(path, .01, counter, require_existing=require_existing)
    assert not path.exists()


def test_explicit_zero_counter_resume_accepts_existing_ledger(tmp_path):
    path = tmp_path / 'audit.jsonl'
    first = AuditJournal(path, .01, 0).reserve(('w',), 1)[0]
    resumed = AuditJournal(path, .01, 0, require_existing=True)
    assert resumed.reserve(('w',), 1)[0].index == first.index + 1
