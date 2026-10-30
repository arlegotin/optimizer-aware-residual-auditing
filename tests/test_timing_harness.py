import hashlib

import pytest

from oar.timing import ActiveClock, _measure_blocks, measure_blocks, paired_blocks, warmup


class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def test_synchronization_work_is_charged_and_prior_queue_is_excluded():
    clock = Clock()
    durations = _measure_blocks(lambda: clock.advance(3), lambda: clock.advance(2), 2, clock)
    assert durations == [5., 5.]


def test_active_clock_only_excludes_explicit_external_pauses(monkeypatch):
    clock = Clock()
    monkeypatch.setattr('oar.timing.time.perf_counter', clock)
    active = ActiveClock(7.)
    clock.advance(2)
    with active.external_pause():
        clock.advance(100)
        with active.external_pause():
            clock.advance(50)
    clock.advance(3)
    assert active.elapsed_s() == 12.


def test_external_pause_exception_restores_clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr('oar.timing.time.perf_counter', clock)
    active = ActiveClock()
    with pytest.raises(RuntimeError), active.external_pause():
        clock.advance(100)
        raise RuntimeError()
    clock.advance(2)
    assert active.elapsed_s() == 2.


def test_real_deliberately_slower_cpu_control():
    fast = measure_blocks(lambda: hashlib.pbkdf2_hmac('sha256', b'control', b'salt', 1000), lambda: None, 3)
    slow = measure_blocks(lambda: hashlib.pbkdf2_hmac('sha256', b'control', b'salt', 100000), lambda: None, 3)
    assert min(slow) > max(fast)


def test_randomized_pairs_preserve_pair_identity_and_both_orders():
    clock = Clock()
    rows = paired_blocks({'a': lambda: clock.advance(2), 'b': lambda: clock.advance(5)},
                         lambda: None, 12, 15, clock=clock)
    assert {tuple(r['order']) for r in rows} == {('a', 'b'), ('b', 'a')}
    assert all(r['seconds'] == {'a': 2., 'b': 5.} for r in rows)


def test_pair_context_retains_ledger_age_without_changing_block_boundaries():
    clock = Clock()
    count = [0]
    def operation():
        count[0] += 1
        clock.advance(2)
    def context():
        clock.advance(100)
        return {'reservations': count[0]}
    rows = paired_blocks({'dense': lambda: clock.advance(3), 'attempt': operation},
                         lambda: None, 2, 1, clock=clock, context=context)
    assert [r['context_before']['reservations'] for r in rows] == [0, 1]
    assert [r['context_after']['reservations'] for r in rows] == [1, 2]
    assert all(r['seconds'] == {'dense': 3., 'attempt': 2.} for r in rows)


def test_warmup_adapts_and_records_unstable_limit():
    clock = Clock()
    durations = iter([9, 8, 3, 3, 3, 3])
    result = warmup(lambda: clock.advance(next(durations)), lambda: None,
                    minimum=3, maximum=6, window=3, relative_range=.01, clock=clock)
    assert result['stable'] and len(result['raw_s']) == 5
    durations = iter([1, 2, 1, 2])
    result = warmup(lambda: clock.advance(next(durations)), lambda: None,
                    minimum=3, maximum=4, window=3, relative_range=.01, clock=clock)
    assert not result['stable'] and len(result['raw_s']) == 4


@pytest.mark.parametrize('blocks', [0, -1, True, 1.5])
def test_invalid_block_counts_fail(blocks):
    with pytest.raises(ValueError):
        measure_blocks(lambda: None, lambda: None, blocks)
