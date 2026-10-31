<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Independent recovery-primitives review

Verdict: **fix required before recovery sign-off**. Two malformed-state validation
gaps are reproducible. No implementation, committed test, configuration, hardware,
download, dependency, or worker-loop changes were made by this reviewer.

## Scope

Reviewed `src/oar/sampling_lm.py` BatchStream, ChargedClock, TrainingState,
snapshot/restore, and save/load checkpoint code through line 640; read
`recovery-report.md`, the 17 scratch recovery fixtures, and their folded copies in
`tests/test_dw_sampling_protocol.py`. Model/evaluation internals had a separate
review. Parent-owned synchronized commit boundaries, active-attempt time ledger,
worker scheduling, and worker-loop integration are outside this sign-off; their
absence is not a finding against these primitives. Only tiny invented CPU data
and the pinned virtual environment were used.

## Required fix 1: validate native AdamW per-parameter state before any mutation

Evidence: `sampling_lm.py:519-536` validates the outer optimizer dictionary,
parameter-group IDs/order, static metadata, and LR, but never validates entries
inside `optimizer_state['state']`. At line 537 restore becomes mutating;
line 538 loads the model and line 540 delegates unchecked moments to native
`optimizer.load_state_dict`. Native loading neither establishes the full AdamW
state schema nor validates moment shape/finiteness or consistency with the saved
logical update count.

Starting from the supplied dense tiny fixture after one successful update, change
only the first native parameter-state entry in its otherwise valid snapshot:

| Single mutation | Observed result |
| --- | --- |
| Delete the entire parameter-state entry | Restore succeeds; next update silently initializes fresh moments for that parameter. |
| Replace `exp_avg` with `torch.zeros(1)` | Restore and snapshot succeed; next step fails with `output with shape [1] doesn't match the broadcast shape [259, 8]`. |
| Fill `exp_avg` with NaN | Restore and snapshot succeed; next step also succeeds, advances counters, and leaves nonfinite model parameters. |
| Set native `step` to 99 while `logical_updates` is 1 | Restore and next step succeed, with incorrect Adam bias correction. |
| Delete native `step` | Restore raises raw `KeyError: 'step'` from native Adam, after the target model weights have already changed. The target becomes unusable until another restore. |

The last case separately confirms the promised rejection-before-mutation property
does not hold: comparing the target embedding with its pre-restore clone yields
`model_changed_before_rejection=True`, `_failed=True`.

For this fixed, fully trainable ByteLM, every unique parameter receives a gradient
tensor on every successful update, including zero-gradient updates. Validate the
expected parameter-state ID set and native field set, CPU tensor shapes/dtypes,
finite moments, and valid native scalar steps consistent with logical updates
before line 537. Initial update 0 snapshots legitimately have empty native state.
Keep native AdamW responsible for application/device casting. No general freezing
or fine-tuning framework is needed.

## Required fix 2: reject impossible batch cursor boundaries

Evidence: `sampling_lm.py:359-360` checks only `0 <= cursor <= len(order)`.
`next_batch` at lines340-343 advances by whole batches and the actual final
partial batch, so for the supplied five-block, batch-size 2 fixture the only
post-draw cursors are 2, 4, 5. Changing a valid cursor 2 snapshot to cursor 1 is
accepted by `BatchStream.load_state_dict`; subsequent draws resume halfway
through a batch and skip/repartition data relative to any real helper history.

Require an empty initial order to have cursor 0, and a drawn epoch to have a
positive cursor at a whole-batch boundary or exactly the final block count.
Reject before replacing the live order, cursor, epoch, or generator state.

## Verification and passing behavior

Fresh scoped commands:

```
PYTHONPATH=src .venv/bin/python -m pytest -q .superpowers/dw_sampling_reopening/test_byte_recovery_pending.py -m 'not mps'
17 passed in 1.50s

PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k test_byte_recovery -m 'not mps'
17 passed, 207 deselected in 1.20s

PYTHONPATH=src .venv/bin/python -m pytest -q .superpowers/dw_sampling_reopening/test_byte_recovery_review_pending.py -m 'not mps'
6 failed in 1.14s
```

The six new scratch probes are intentional red regressions: the first four
optimizer cases and cursor case fail with `DID NOT RAISE ValueError`; missing
native step fails with the raw KeyError from line 540. They are retained for the
implementer's repair verification.

Within the reviewed scope the existing fixtures support: ongoing dedicated data
randperm and final actual batch size; exact fixed-recorded-LR interrupted resume
for dense/uniform/importance arms including model, native moments, sampling draws,
data cursor/RNG and counters; owned CPU snapshots; unequal-valid-target summed
CE divided by actual valid targets; native clipping/AdamW; zero-gradient success;
failure fencing and no counter commit on thrown loss/gradient/native-step
failures; no post-optimizer scalar reads; cumulative prior-plus-active-launch
clock with restart downtime excluded; original outer `t0_s` and scheduler rule
preservation; hash-before-weights-only loading and exact bindings.

Filesystem semantics are appropriate for the documented unique-destination,
single-controller contract. Write, file-fsync, and rename failures preserve the
prior checkpoint and clean owned scratch paths. An additional injected failure
on the second fsync (directory fsync, after rename) raised OSError and left the
prior file byte-identical plus a complete new file, with no temp/lock residue.
Independent hashing and load confirmed the new file was complete. This matches
the report's documented unreceipted-publication case: the caller must not advance
its committed manifest without the save receipt. No filesystem fix is requested.

Re-review the two malformed-state regressions after repair. MPS qualification and
integrated worker recovery remain parent-owned later checks.

## Scoped re-review after native-state repair

Reviewed the latest recovery-report section and source changes through restore;
no unfinished worker functions were reviewed. The new fully trainable ByteLM and
fresh native optimizer construction contract is appropriately narrow. The
standalone sampled operator's separate frozen-parameter behavior is unaffected.

Required fix 1 is resolved in the inspected implementation. Before the first
mutating operation at `sampling_lm.py:594`, lines 563-593 require the exact
native parameter-state ID set after a successful update, exact field sets,
finite CPU FP32 moment tensors of matching parameter shapes, nonnegative second
moments, and native scalar CPU FP32/FP64 step tensors (FP32 for fused) whose
finite integer value equals logical_updates. Zero-update snapshots require empty
native state. Model tensors are also checked for finiteness before application.
The constructor, step, snapshot and restore enforce the all-trainable contract;
constructor state must be empty and resume uses restore. Native load still owns
device placement and casting.

Fresh verification with the original six reviewer probes unchanged:

```
PYTHONPATH=src .venv/bin/python -m pytest -q .superpowers/dw_sampling_reopening/test_byte_recovery_review_pending.py .superpowers/dw_sampling_reopening/test_byte_recovery_pending.py -m 'not mps'
28 passed in 1.32s
```

This includes the implementer's added malformed native fields/steps, nonfinite
model tensors, zero-update state, native default/single/foreach/fused exact CPU
resume, and nonfused native FP64 step compatibility fixtures. No source or tests
were edited during re-review.

Required fix 2 is only partially resolved at this checkpoint. The new modulo
check at lines 361-362 rejects cursor 1 for batch size 2, so the original red
probe passes. A direct unchanged-fixture probe still accepts an impossible drawn
epoch with cursor 0: draw the first batch, set the saved cursor to 0, then call
load_state_dict. Observed `epoch=1, cursor=0, order_len=5`, with no exception.
No public next_batch return exposes a nonempty order with cursor 0; the initial
cursor 0 is valid only with the empty order/epoch 0. Keep this remaining cursor
case open until it is rejected before state application. No other new finding
was identified in this scoped re-review.

## Final cursor correction and disposition

Inspected the final narrow correction at `sampling_lm.py:361-362`: a nonempty
drawn order with cursor 0 now raises ValueError inside validation, before the
live stream mutation at lines 372-374. Empty initial order/epoch 0/cursor 0 still
passes this check, and the existing modulo/final-partial-batch validation remains.

Ran only the requested targeted CPU regression:

```
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py::test_byte_recovery_drawn_epoch_rejects_zero_cursor_before_mutation -m 'not mps'
1 passed in 1.05s
```

The regression verifies rejection and equality of the full stream state before
and after the rejected load. Required fix 2 is resolved. Together with the prior
28 passing recovery checks and inspected native-state validation, both findings
from this review are closed. **Scoped recovery-primitives review: pass.** No
source/test changes were made by this reviewer during re-review. This disposition
does not extend to the parent-owned worker integration, active-attempt ledger,
or MPS qualification.
