"""Inclusive wall clocks and synchronized blocks; no profiling sum is a step."""

from contextlib import contextmanager
import math
import random
import time


class ActiveClock:
    """Top-level clock. Only explicitly declared external pauses are excluded."""

    def __init__(self, elapsed_s: float = 0.0):
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError('elapsed time must be finite and nonnegative')
        self._carried = elapsed_s
        self._start = time.perf_counter()
        self._excluded = 0.
        self._pause_start = None
        self._depth = 0

    def elapsed_s(self) -> float:
        end = self._pause_start if self._depth else time.perf_counter()
        return self._carried + end - self._start - self._excluded

    @contextmanager
    def external_pause(self):
        if not self._depth:
            self._pause_start = time.perf_counter()
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if not self._depth:
                self._excluded += time.perf_counter() - self._pause_start
                self._pause_start = None


def _measure_blocks(operation, synchronize, blocks, clock):
    if type(blocks) is not int or blocks <= 0:
        raise ValueError('blocks must be a positive integer')
    result = []
    for _ in range(blocks):
        synchronize()  # drain prior work, outside the new block
        start = clock()
        operation()
        synchronize()  # all submitted work completes inside this block
        result.append(clock() - start)
    return result


def measure_blocks(operation, synchronize, blocks: int) -> list[float]:
    return _measure_blocks(operation, synchronize, blocks, time.perf_counter)


def paired_blocks(operations, synchronize, blocks, seed, *, clock=time.perf_counter, context=None):
    """Randomized within-pair order, retained with the raw measurements."""
    rng = random.Random(seed)
    rows = []
    for index in range(blocks):
        order = list(operations)
        rng.shuffle(order)
        before = context() if context else None
        seconds = {name: _measure_blocks(operations[name], synchronize, 1, clock)[0] for name in order}
        row = dict(pair=index, order=order, seconds=seconds)
        if context:
            row.update(context_before=before, context_after=context())
        rows.append(row)
    return rows


def warmup(operation, synchronize, *, minimum, maximum, window, relative_range,
           clock=time.perf_counter):
    raw = []
    for _ in range(maximum):
        raw.extend(_measure_blocks(operation, synchronize, 1, clock))
        if len(raw) >= max(minimum, window):
            tail = raw[-window:]
            center = sum(tail) / window
            if center > 0 and (max(tail) - min(tail)) / center <= relative_range:
                return dict(stable=True, raw_s=raw)
    return dict(stable=False, raw_s=raw)
