"""Append-only, fsync-before-visibility Gaussian test spending.

Local POSIX filesystems are the supported durability boundary. Keep this journal
outside rollback of training checkpoints. Advisory locks serialize cooperating
writers; a SHA-256 chain detects malformed/partial/edited evidence, not malicious
whole-file rollback. An interrupted append is either a consumed valid record or
an unrecoverable journal: corruption never resets the counter.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from .audit import validate_candidate
from .records import AuditCandidate, AuditReservation, from_payload, to_payload


_STREAM = 'oar.audit.pcg64.gaussian.v1'
_SEED_SPACE = 2 ** 63


class JournalError(ValueError):
    """Confidence evidence cannot safely authorize further tests."""


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JournalError('duplicate JSON field')
        result[key] = value
    return result


def _reject_constant(value):
    raise JournalError(f'nonfinite journal evidence: {value}')


class AuditJournal:
    """Allocate tests with delta_run*6/(pi^2*j^2), j beginning at one.

    ``audit_seed`` defaults to zero for the planned three-argument interface.
    Resume callers must set ``require_existing=True``, including checkpoint
    counter zero: an absent ledger may contain spending newer than a stale
    checkpoint. Positive checkpoint counters always require an existing ledger.
    The persisted UUID and domain label separate this stream from a construction
    generator even if numeric user seeds coincide. Index-derived seeds are
    unique within the journal's 63-bit allocation space. Do not roll back or
    copy a live journal to fork training: a fork needs a new run identity and
    its own declared risk budget.

    ``reserve`` returns only after fsync. ``draw_probes`` may expose each
    reservation once, only on the instance that reserved it. On restart or any
    possibly observed interruption reserve again; never regenerate old probes
    for a new candidate. Stored seeds permit offline reproduction of evidence,
    which does not authorize another online decision.
    """

    def __init__(self, path: Path, delta_run: float, checkpoint_counter: int, *,
                 audit_seed: int = 0, require_existing: bool = False):
        if type(delta_run) not in (int, float) or not 0 < delta_run < 1:
            raise ValueError('require finite 0 < delta_run < 1')
        if (type(checkpoint_counter) is not int or not 0 <= checkpoint_counter < _SEED_SPACE
                or type(audit_seed) is not int or not 0 <= audit_seed < _SEED_SPACE
                or type(require_existing) is not bool):
            raise ValueError('checkpoint counter and audit_seed must be nonnegative 63-bit integers')
        self.path = Path(path).absolute()
        self.delta_run = float(delta_run)
        self.audit_seed = audit_seed
        self._available: dict[str, AuditReservation] = {}
        self._run_id: str | None = None
        created = False
        if require_existing or checkpoint_counter > 0:
            try:
                fd = os.open(self.path, os.O_RDWR | os.O_APPEND)
            except FileNotFoundError as exc:
                raise JournalError('missing audit journal on resume; checkpoint cannot recover newer spending') from exc
        else:
            try:
                fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
            except FileExistsError:
                fd = os.open(self.path, os.O_RDWR | os.O_APPEND)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if created:
                header = dict(schema_version=1, kind='header', delta_run=self.delta_run,
                              audit_seed=audit_seed, run_id=uuid.uuid4().hex, stream=_STREAM,
                              numpy_version=np.__version__)
                self._append(fd, [header], '')
            header, counter, chain, _ = self._read(fd)
            self._run_id = header['run_id']
            # Also repair an earlier initialization interrupted before directory
            # fsync. No reservation is exposed until the name is durable.
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            if checkpoint_counter > counter:
                self._append(fd, [dict(schema_version=1, kind='checkpoint_floor',
                                       index=checkpoint_counter)], chain)
        finally:
            os.close(fd)

    @contextmanager
    def _locked(self):
        # Never recreate an absent journal on an already initialized instance.
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            os.close(fd)

    def _reservation(self, header: dict, index: int, layer_id: str, logical_step: int) -> AuditReservation:
        if not 1 <= index < _SEED_SPACE:
            raise JournalError('audit seed/index space exhausted')
        coefficient = math.nextafter(6 / math.pi ** 2, 0.)
        delta = math.nextafter(self.delta_run * coefficient, 0.)
        delta = math.nextafter(delta / index, 0.)
        delta = math.nextafter(delta / index, 0.)
        if not math.isfinite(delta) or delta <= 0:
            raise JournalError('allocation underflow; positive delta is not representable')
        key = hashlib.sha256(_canonical([_STREAM, header['run_id'], self.audit_seed])).digest()
        offset = int.from_bytes(key[:8], 'big') % _SEED_SPACE
        return AuditReservation(index, layer_id, logical_step, delta,
                                f"{header['run_id']}:audit:{index}", (offset + index) % _SEED_SPACE)

    def _read(self, fd):
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), 'rb') as reader:
            raw = reader.read()
        if not raw or not raw.endswith(b'\n'):
            raise JournalError('empty or truncated journal; refusing confidence reset')
        counter, chain, header, reservations = 0, '', None, {}
        try:
            for line_number, line in enumerate(raw.splitlines()):
                row = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                if not isinstance(row, dict):
                    raise JournalError('journal row must be an object')
                digest = row.pop('sha256')
                if row.pop('previous_sha256') != chain:
                    raise JournalError('journal hash chain mismatch')
                checked = {**row, 'previous_sha256': chain}
                if digest != hashlib.sha256(_canonical(checked)).hexdigest():
                    raise JournalError('journal content hash mismatch')
                chain = digest
                if type(row.get('schema_version')) is not int or row['schema_version'] != 1:
                    raise JournalError('unsupported journal schema')
                if line_number == 0:
                    if (set(row) != {'schema_version', 'kind', 'delta_run', 'audit_seed', 'run_id', 'stream', 'numpy_version'}
                            or row['kind'] != 'header' or row['stream'] != _STREAM
                            or row['numpy_version'] != np.__version__
                            or type(row['delta_run']) is not float or row['delta_run'] != self.delta_run
                            or type(row['audit_seed']) is not int or row['audit_seed'] != self.audit_seed
                            or type(row['run_id']) is not str or len(row['run_id']) != 32
                            or any(c not in '0123456789abcdef' for c in row['run_id'])
                            or self._run_id is not None and self._run_id != row['run_id']):
                        raise JournalError('journal identity/budget mismatch or malformed header')
                    header = row
                elif row['kind'] == 'checkpoint_floor':
                    if (set(row) != {'schema_version', 'kind', 'index'} or type(row['index']) is not int
                            or not counter < row['index'] < _SEED_SPACE):
                        raise JournalError('invalid or nonmonotone checkpoint floor')
                    counter = row['index']
                elif row['kind'] == 'reservation':
                    if set(row) != {'schema_version', 'kind', 'reservation'}:
                        raise JournalError('unknown reservation fields')
                    reservation = from_payload(row['reservation'])
                    if (type(reservation) is not AuditReservation or reservation.index <= counter
                            or not reservation.layer_id or reservation.logical_step < 0
                            or reservation != self._reservation(header, reservation.index,
                                                               reservation.layer_id, reservation.logical_step)):
                        raise JournalError('malformed or nonmonotone reservation')
                    counter = reservation.index
                    reservations[reservation.probe_id] = reservation
                else:
                    raise JournalError('unknown journal row kind')
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            if isinstance(exc, JournalError):
                raise
            raise JournalError('malformed journal evidence; refusing confidence reset') from exc
        return header, counter, chain, reservations

    @staticmethod
    def _append(fd, rows, chain):
        output = bytearray()
        for row in rows:
            checked = {**row, 'previous_sha256': chain}
            chain = hashlib.sha256(_canonical(checked)).hexdigest()
            output.extend(_canonical({**checked, 'sha256': chain}) + b'\n')
        view = memoryview(output)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError('journal append made no progress')
            view = view[written:]
        os.fsync(fd)

    @property
    def counter(self) -> int:
        """Fresh durable maximum, suitable for the next training checkpoint."""
        with self._locked() as fd:
            return self._read(fd)[1]

    def reserve(self, layer_ids: tuple[str, ...], logical_step: int) -> tuple[AuditReservation, ...]:
        if (type(layer_ids) is not tuple or any(type(layer) is not str or not layer for layer in layer_ids)
                or len(set(layer_ids)) != len(layer_ids)
                or type(logical_step) is not int or logical_step < 0):
            raise ValueError('require unique layer IDs and nonnegative integer logical_step')
        if not layer_ids:
            return ()
        with self._locked() as fd:
            header, counter, chain, _ = self._read(fd)
            allocated = tuple(self._reservation(header, counter + i, layer, logical_step)
                              for i, layer in enumerate(layer_ids, start=1))
            rows = [dict(schema_version=1, kind='reservation', reservation=to_payload(r)) for r in allocated]
            self._append(fd, rows, chain)
            # Publication occurs strictly after all reservations are durable.
            self._available.update((r.probe_id, r) for r in allocated)
            return allocated

    def draw_probes(self, reservation: AuditReservation, candidate: AuditCandidate,
                    input_width: int, q: int, *, dtype: torch.dtype = torch.float32,
                    device: str | torch.device = 'cpu') -> torch.Tensor:
        """Generate independent CPU Gaussians then explicitly transfer once.

        Reuse the returned tensor across the fixed effective batch. This call
        and any transfer/synchronization belong inside charged active time.
        Failure after draw begins consumes the reservation; retries reserve anew.
        """
        validate_candidate(candidate, reservation, current_rank=candidate.rank,
                           current_basis_version=candidate.basis_version)
        if (type(input_width) is not int or input_width <= 0 or type(q) is not int or q <= 0
                or dtype not in (torch.float32, torch.float64)):
            raise ValueError('require positive integer widths and FP32/FP64 probes')
        with self._locked() as fd:
            _, _, _, reservations = self._read(fd)
            if reservations.get(reservation.probe_id) != reservation:
                raise JournalError('probes require an authentic durable reservation')
            validate_candidate(candidate, reservation, current_rank=candidate.rank,
                               current_basis_version=candidate.basis_version,
                               current_probe_ids=tuple(r.probe_id for r in reservations.values()
                                                       if r.logical_step == candidate.logical_step))
            if self._available.pop(reservation.probe_id, None) != reservation:
                raise JournalError('reservation already exposed or from another journal instance; reserve fresh')
        # Torch's CPU normal stream ignores high seed bits in this runtime.
        # PCG64 consumes the full persisted seed, including after large restart
        # counter jumps. Neither NumPy's global RNG nor Torch's model RNG moves.
        generator = np.random.Generator(np.random.PCG64(reservation.seed))
        array = generator.standard_normal((input_width, q), dtype=np.float64 if dtype == torch.float64 else np.float32)
        return torch.from_numpy(array).to(device)
