<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# R1 implementation task — execute only after root releases the E4 freeze

This brief is inert preparation while E4 runs. Root will dispatch a fresh
implementer AFTER the parent training process exits and the attempt is sealed.
The task fixes the accepted R1 backend/cost review finding; it does not reopen
scientific selection or authorize another grid/algorithm/precision/data path.

Requirements: read `recovery-preflight-accounting-issue.md`, then the R1 section
of `backend-cost-recovery-final-review.md`, then the unexecuted
`recovery-preflight-test-report.md` and `test_recovery_preflight_pending.py` in
this directory. Those are one task's requirements and preparation, not claimed
passing tests. Current source is Git89c5bd8 / artifact launchHEAD43c2cec; root
will supply the later actual BASE commit containing the sealed E4 public data.

Own only `analysis/run_dw_sampling.py`, `tests/test_dw_sampling_protocol.py`,
and your ignored task report. Root owns prospective docs/config/amendment/Git,
all scientific artifact retention/analysis and actual hardware validation. Do
not change sampled_linear.py, sampling_lm.py, sampling_data.py, the model,
optimizer, scheduler, sampler, historical files, old gates/amendments/configs,
public result records or protected-data permissions. No new dependency/unit,
framework, automatic retry, new sampler, native optimization change or kernel.
No child agents, network, corpus/checkpoint body reads from actual runs,
training, hardware benchmarks or MPS work. Tiny invented test checkpoints are
permitted AFTER freeze release. Do not use the undercharged real resume path.

Follow systematic debugging and TDD. Integrate/adapt the pending candidate
fragment into the existing protocol module. Run focused tests against unchanged
production source first and record genuine RED accounting/durability assertions.
Fix fixture/collection/import/schema errors before calling a test RED. Keep real
parent execute->launcher->ledger/hash/decode boundaries and real tiny stdlib
child lifecycle; no mock-only implementation-mirroring tests. Record commands,
return codes and focused output paths. Only then implement the smallest fix.

Required behavior:
- Capture and durably identify active recovery preparation BEFORE the earlier
  execute_learning_trial ledger/CP hashes as well as launcher deep validation.
  Moving only the launcher's final timer is insufficient. A separately explicit
  selected environmental resume can route directly to the paid launcher before
  its current read-only reuse path. Preserve fresh no-resume timing/behavior.
- Charge every active preparation/hash/decode/import/request/spawn/load/teardown
  interval to the same cumulative H/LR/target/budget clock. Retain prior wasted
  uncommitted work and original checkpoint t0; exclude external downtime.
- A parent preflight OSError/MemoryError/known environmental interruption before
  a child exists still leaves a durable known active interval. The next
  separately explicit eligible recovery charges it exactly once. No automatic
  continuation. A start without a proved terminal (including os._exit) remains
  unknown and blocks favorable continuation; never invent a death time.
- When prior integrity cannot be authenticated, retain failure evidence but do
  not invent a trusted cumulative base or grant training. A numerical,
  ownership, selection or integrity failure cannot become a retryable bad draw.
  Do not weaken source/config/receipt chains, deep payload integrity, original
  numeric-over-signal precedence, optimizer commitment or checkpoint pruning.
- Keep strict fresh-child Torch-free policy preflight and the existing
  configure_torch guard/order. New receipt metadata and ledger reconstruction
  must not import Torch. Deep CPU proof stays after child policy. Avoid a
  framework/general registry; extend only the small new-branch ledger schema.
- Preserve all already sealed v9 identities/results; the repair has no exposure
  to the actual uninterrupted E4 measurements. Root separately freezes the new
  source and documents affected checks before any future recovery.

Run RED then GREEN focused `-k recovery_preflight`, retained strict child-policy
regressions, affected protocol/operator coverage and full CPU suite once at the
end (`PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'`). All tests must
be meaningful and deterministic; no timing statistical p-value. Do not loosen
predeclared numerical tolerances. Root will perform serial actual-MPS/full-suite
and current-driver stages after your release. No Git mutations/commits by you;
return control after code/tests/self-review/report for independent review.

Report `.superpowers/dw_sampling_reopening/recovery-preflight-implementation-report.md`
(CC-BY-4.0) with status; exact files/functions/schema/control-flow; RED/GREEN/full
CPU commands/counts/codes; which new test proves R1 versus retained guards;
failed/unknown preparation handling; exact no-resume continuity; concerns and
no hardware/performance/source-freeze claims. Software stays AGPL-3.0-only.
Return a short status, test summary and concerns, not the full diff/report.
