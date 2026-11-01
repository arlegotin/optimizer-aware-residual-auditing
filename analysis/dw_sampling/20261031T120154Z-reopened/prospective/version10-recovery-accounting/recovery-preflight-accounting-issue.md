<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Recovery preflight accounting — pending repair after frozen E4

Discovered by the independent `sampling_backend_final_review` read-only context
while v9 pilot attempt-0004 was running. Root traced and confirmed the source
ordering. No test or measured-source change has occurred for this finding yet.

## Root cause and current impact

`execute_learning_trial` reconstructs/authenticates its prior ledger before
calling `launch_learning_worker`. The launcher repeats ledger verification and
calls `validate_resume_selection(..., deep=True)` before its canonical start
timestamp. Ledger verification hashes retained checkpoint bodies; deep selection
hashes and deserializes a complete CPU checkpoint through `load_checkpoint`.
Consequently recovery-specific parent verification/loading is outside the
trajectory clock, although worker-side verification/loading is inside it.
Outer stage wall time retains this work but does not feed per-run H/T_budget or
target timestamps, so it does not repair the undercharge.

Relevant v9 lines: driver 2671, 2888-2896, 2450-2454, 2472, 2480-2483;
sampling_lm.py 704-708. The source order is the evidence, not an executed timing
measurement of the missing cost.

All first five completed E4 trials used attempt-0001 with no prior matching
ledger, no interruption and no resume. This checkpoint preflight branch was not
executed by their trajectories. Fresh launch request preparation/validation,
child startup, model/data/sampler setup, evaluation, checkpoints and teardown are
inside the existing clock. The currently running sixth trial is also fresh.
Current uninterrupted E4 can continue unchanged. **Do not use the known
undercharged environmental-resume path if E4 is interrupted.** Preserve and seal
the interrupted evidence, then repair/freeze before further recovery.

## Required later work

After E4 seals (or an interruption forces it to stop), use systematic debugging
and TDD. First make a deterministic fake-clock regression that reproduces the
excluded parent CP verification/decode interval, including the earlier
execute_learning_trial ledger work. Do not fix only the launcher's final timer
placement. Keep native model/optimizer/data/sampler execution and no-resume
timing unchanged where possible.

The final repair must durably charge every active recovery preparation interval,
including failed preflight attempts, hashing/deserialization, new startup/loading
and teardown, while excluding external downtime. Preserve original t0/H/budgets,
all previous wasted work, integrity/source/selection authority and numerical
failure precedence. An unproved parent-death interval must remain unknown and
block favorable timing, rather than receive an invented duration. No automatic
retries or newly sampled draws after a numerical failure.

Tests must cover successful recovery, failed parent preflight then an explicitly
authorized recovery, unknown termination, prior-checkpoint integrity rejection,
no duplicate charge, monotone cumulative clock and preserved t0, plus unchanged
fresh-launch behavior and genuinely Torch-free child preflight. Run no tests
while E4 hardware work remains active. Review the exact repair, freeze/hash its
new source version and rerun affected checks before any actual recovery or
protected confirmation.

Historical uninterrupted measurements retain v9 identity. Do not rewrite them,
retroactively claim v9 recovery was fully charged, or attribute old measurements
to the repaired source. If E stops scientifically, this scoped repair is still
required for a usable delivered recovery implementation; no new scientific
training algorithm or grid is authorized by it.

## Root design notes while E4 remains frozen (unimplemented)

2026-11-01: new test-only preparer `/root/recovery_preflight_tests` reads
`recovery-preflight-test-brief.md`, writes only ignored candidate test/report
text, and executes nothing. Root verified actual Git HEAD remains 43c2cec.
Do not infer RED/GREEN or a fix from these preparation files.

A minimal candidate repair should route an explicitly selected resume from
`execute_learning_trial` to the launcher before its current ledger work. The
resume launch must start and durably identify its active preparation interval
before ledger/checkpoint hashing/deep decode. Merely moving the timestamp below
`execute_learning_trial` is insufficient. Preserve the current fresh branch and
avoid extra clock calls there.

Preparation can fail before a child/request exists. Its known active time needs
a terminal receipt that is reconstructed exactly once on a later explicit
eligible environmental resume. A new attempt directory with a durable start but
no terminal is an unknown interval and must prevent favorable continuation.
When prior ledger integrity cannot itself be established, retain the observed
failure/active duration but never fabricate a trusted cumulative base. A failed
integrity/config/numerical validation must not become a numerical retry route.
All successful recovery work, wasted uncommitted work and subsequent child
startup/load/teardown remain charged; no wall-clock external downtime enters H.
Restore the original t0 from the selected committed checkpoint. Keep immutable
old receipts/source identities, deep payload checks, strict Torch-free child
preflight and native training execution unchanged. Exact schema/control-flow
choices remain to be decided after actual candidate RED tests; this note is not
an implementation patch or permission to alter active measured files.
