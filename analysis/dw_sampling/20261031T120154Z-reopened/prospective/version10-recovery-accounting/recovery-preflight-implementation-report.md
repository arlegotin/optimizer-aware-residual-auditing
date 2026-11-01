<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# R1 recovery preparation accounting — implementation report

2026-11-01. Status: scoped implementation and self-review complete; ready for
root's independent review. Root released the E4 freeze and supplied sealed public
BASE `a3e3229768dd7186f628419917ed746d9e0dcf7b`. I did not change or reassess the
sealed E4 results, read actual corpus/checkpoint bodies, or perform hardware work.
No source-freeze, MPS correctness, throughput or performance claim is made here.

Owned writes: `analysis/run_dw_sampling.py`,
`tests/test_dw_sampling_protocol.py`, and this ignored report only. Software
retains AGPL-3.0-only; this report is CC-BY-4.0. No Git mutations, child agents,
network, new dependencies, actual training or benchmarks were used. Root owns
prospective metadata, source freeze, hardware validation and independent review.

## Root cause and control flow

On unchanged production source, `execute_learning_trial` authenticated prior
checkpoint bytes before entering the launcher, and the launcher authenticated
and deeply decoded them before its start timestamp. An interrupted parent before
launch also had no durable active interval. The pending candidate fragment was
integrated into the existing protocol module and executed against that unchanged
source before any production edit: three genuine accounting/durability failures,
not collection/schema errors.

The scoped repair changes these functions:

- `execute_learning_trial`: an explicitly supplied resume selection now routes
  directly to the paid launcher before its read-only ledger/reuse path.
- New `learning_recovery_preparation`: under existing exclusive ownership, records
  the recovery start and durable identity before run-directory allocation or any prior receipt/checkpoint
  body authentication. Allocation inspects invocation names only, preserving
  monotone attempt numbering across stage invocations. It catches pre-launch
  failures, including a failure immediately after publishing the start receipt.
- `launch_learning_worker`: excludes its own unfinished preparation from prior
  ledger reconstruction, retains the authenticated prior ledger when available,
  then performs the unchanged deep resume and worker-request checks. A successful
  launch uses the preparation's original start through child startup, load,
  result/prefix hashing and teardown; no preparation charge is added a second
  time. Current-stage sealed-record proof, previously done in execute, runs inside
  paid recovery preparation instead. The existing child lifecycle/classification
  code and native execution remain unchanged.
- `read_trial_ledger`: reconstructs ended preparation-only intervals as well as
  existing launch/exit intervals; authenticates preparation-bound launches; never
  infers a missing terminal/death time. Existing payload integrity, ownership,
  pruning, receipt prefixes, prior chain and checkpoint count checks remain.
- `previous_learning_trial_dirs`: also discovers preparation-only histories, so
  failed preparation cannot vanish on a later stage invocation. Source/config
  identity and sealed-record authentication remain required.

## Small recovery-only receipt schema

Each selected recovery first publishes
`trials/<trial-id>/attempt-NNNN.preparation.json` in the already-owned trial
directory, before creating its child run directory. It contains
bindings, attempt ID, `launch_perf_s`, explicit path/SHA selection, clock
implementation, host and parent PID. Its successful `launch.json` additionally
binds `preparation_sha256` and must use exactly the preparation start. Existing
old launch/exit files require no new fields and are not rewritten.

A failure before `launch.json` produces the sibling marker
`attempt-NNNN.preparation-exit.json`, binding the start
receipt hash, exact observed end and active duration, failure exception/reason,
classification, and authenticated prior elapsed/previous-terminal hash if known.
When prior authentication did not finish, both prior fields are null: this does
not invent a zero or trusted cumulative base. A later explicit eligible recovery
must authenticate all preceding history and retained checkpoint bytes anew before
adding that observed interval once. Its next launch hashes this preparation
terminal as the previous terminal in the existing chain.

An environmental preparation failure is usable only after a proved preceding
environmental interruption and a matching own live nonfinal checkpoint receipt.
OSError, MemoryError, KeyboardInterrupt and the existing timeout classes qualify;
other failures use the existing fail-closed `numerical_failure` classification.
Selection/integrity/frozen-state rejection therefore cannot become a retryable
scientific draw. If retained bytes remain unauthenticated, ledger construction
fails rather than granting training or returning a trusted base. Repairing bytes
or correcting a selected SHA does not erase a recorded nonenvironmental rejection.
No automatic continuation is added: the failure is raised; another explicit
eligible resume invocation is necessary.

A durable start without a proved terminal, including real `os._exit(77)` during
the first selected-checkpoint hash or before run-directory allocation, remains `unknown_active_interval` and blocks
continuation. Failed/partial receipt publication that cannot itself be durably
completed must remain unusable; no death timestamp is fabricated. Receipt
publication continues to use the existing exclusive fsynced JSON writer.

## Scientific and fresh-path continuity

Root explicitly ruled that the common pre-admission metadata/ownership prelude
remains at the same boundary shared with fresh launches: trial-directory setup,
binding construction and exclusive-lock acquisition precede the per-worker
trajectory clock and remain in whole-stage cost. The repair creates no new
ownership framework. From recovery admission onward, name allocation, start
publication, run-directory allocation (including a failure that creates no run
directory), authentication and later work are charged. A marker alone is now a
ledger entry even when run.mkdir created no directory; an unknown admitted
interval cannot support favorable continuation. This is not a claim that every
filesystem failure can publish a durable receipt. Unpublishable initial metadata
is not authenticated admission evidence; whole-stage failure/unknown-evidence
review and the explicit continuation rules still apply. No pre-admission setup
failure is claimed to be a durably measured trajectory interval.

No model, optimizer, scheduler, sampler, grid, target, numerical tolerance,
protected permission or checkpoint-pruning implementation changed. The worker
receives the same prior base plus the complete active recovery interval through
its existing `ChargedClock`, H/LR/target/budget policy and original checkpoint t0.
Wasted uncommitted work stays in prior ended intervals; external downtime is
excluded. Known numerical child/prefix failure still takes precedence over a
later signal; retained regressions exercise that original code.

With no selected resume, the new context yields an empty preparation without a
clock read or new receipt. Existing execute reuse, ledger/deep-check order, fresh
start immediately before request construction, request serialization, child
lifecycle and canonical endpoint remain in their original positions. The fresh
regression confirms zero prior charge and the same launch/request clock boundary.
The only new ledger probes on old/fresh histories check for absent preparation
metadata; they introduce no checkpoint-body decode or framework import.

`configure_torch` and its guard/order were not changed. New receipt reconstruction
uses only the standard library. A new fresh subprocess reconstructs a real failed
preparation history and performs shallow selection without importing Torch;
the retained fresh main-path tests independently verify Torch-free prepolicy and
`POLICY -> DEEP -> WORKER`, including rejection before policy for a bad hash.

## Test evidence

All commands used `.venv/bin/python` with `PYTHONPATH=src`; all child processes
were tiny invented CPU/stdlib fixtures. Each listed pytest failure returned 1;
passing commands returned 0. Output paths below are local `/tmp` evidence, not
public scientific artifacts.

1. Original genuine RED, unchanged production source:
   `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k recovery_preflight`
   — **3 failed, 4 passed, 371 deselected**, 2.06 s, exit 1.
   `/tmp/recovery-preflight-red.log` retains the original output. Successful
   recovery charged 23 rather than observed 32 seconds; failed decode retained
   only 5 rather than observed 14 seconds; real abrupt parent death did not block
   subsequent dispatch. The integrity, fresh-boundary and controller guards passed.
2. Initial minimal GREEN, same focused command — **7 passed, 371 deselected**,
   1.67 s, exit 0. Later focused output supersedes this temporary GREEN log.
3. Expanded coverage exposed the cross-stage attempt-ID reset — **1 failed,
   16 passed, 371 deselected**, 1.84 s, exit 1 after fixture corrections.
   `/tmp/recovery-preflight-expanded-red.log`. Name-only allocation now preserves
   the prior numbering across stage invocations.
4. Publication-boundary RED, same focused command — **1 failed, 19 passed,
   371 deselected**, 2.04 s, exit 1.
   `/tmp/recovery-preflight-publication-red.log`. An OSError immediately after
   durable start publication left an unknown rather than a known interval.
   Moving that publication into the existing recovery exception handler fixed it.
5. Final focused GREEN, same focused command — **22 passed, 371 deselected**,
   2.10 s, exit 0. `/tmp/recovery-preflight-green.log`.
6. Retained strict child-policy command:
   `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k 'learning_worker_preflight_torch_free_in_fresh_process or learning_resume_deep_checkpoint_proof_follows_policy_before_worker or learning_resume_prepolicy_rejects_unreceipted_hash_without_torch'`
   — **6 passed, 387 deselected**, 2.63 s, exit 0.
   `/tmp/recovery-preflight-policy.log`; rerun after the allocation repair.
7. Final affected coverage:
   `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps' tests/test_dw_sampling_protocol.py tests/test_sampled_linear.py`
   — **487 passed, 7 deselected**, 15.14 s, exit 0.
   `/tmp/recovery-preflight-protocol-operator.log`.
8. Final full CPU suite, after the allocation repair:
   `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'`
   — **842 passed, 16 deselected**, 17.92 s, exit 0.
   `/tmp/recovery-preflight-full-cpu.log`.
9. `git diff --check -- analysis/run_dw_sampling.py tests/test_dw_sampling_protocol.py`
   — exit 0, no whitespace errors. This was read-only; no Git mutation occurred.

The new parent-path paid-work, failed-decode and real-parent-death tests prove
R1 directly, retaining real execute -> launcher -> ledger/hash/decode boundaries
and actual tiny child spawn/wait/exit. Costs are injected beneath real operations;
expected cumulative charge is original 5 seconds plus independently observed
active boundary costs, never a production accounting helper. Additional cases
cover interrupted ledger hashing before prior authentication, MemoryError decode,
KeyboardInterrupt request setup, start/request publication failure, run.mkdir failure before directory creation, exact-once
reconstruction, external gaps, monotone clocks, cross-stage history, tampering,
missing terminals, persistent rejection and Torch-free reconstruction. Original
t0 and scheduler guards exercise real controller restore separately.

Fixture-only corrections were kept separate from production RED evidence:
expanded tests initially tried to overwrite immutable invented checkpoint/receipt
files through exclusive writers; the fixtures now deliberately replace/tamper
those invented bytes explicitly. A later Torch-free test initially pinned an old
redundant hash count; its expectation now comes from independent observed injected
work. An affected-suite run already loaded that old test before correction and
reported **1 failed, 482 passed, 7 deselected**, 15.25 s, exit 1; retained at
`/tmp/recovery-preflight-protocol-operator-fixture-failure.log`. This failure was
fixture-only and is resolved in the final focused, affected and full CPU runs.
No numerical tolerance was changed.

## Remaining review/validation boundaries

The first full CPU run passed **840 tests, 16 deselected**, 17.80 s, exit 0.
Root then identified the recovery run.mkdir prelude gap. Its deterministic
focused RED produced **1 failed, 20 passed, 371 deselected**, 2.26 s, exit 1:
`/tmp/recovery-preflight-allocation-red.log`. Only the original 5 seconds were
retained after an observed 3-second failed allocation. Publishing the external
per-attempt marker before mkdir closes that disappearance. Final coverage also
uses real os._exit before mkdir, proving unknown rather than invented duration
when no run directory exists. The extra final broad verification below is
justified by this new source change; it is not an unreported repeat of a passing
unchanged implementation.

No known scoped CPU failure remains. Root should independently review the new
recovery receipt/chain handling and perform its separately owned actual-MPS,
current-driver and source-freeze stages before any actual environmental recovery.
This implementation does not authorize that recovery, a new scientific run, a
protected confirmation, or any change to sealed v9 identities/results. Durable
storage failure can still prevent a terminal receipt; such evidence must remain
blocked rather than acquire a favorable inferred duration. No actual checkpoint
or scientific path was exercised to make this report.
