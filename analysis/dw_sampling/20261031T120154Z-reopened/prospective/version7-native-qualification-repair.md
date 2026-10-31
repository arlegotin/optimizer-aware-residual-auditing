# Native optimizer qualification repair

Read the retained native-only diagnostic in
`analysis/dw_sampling/20261031T120154Z-reopened/native_qualification_diagnostic/attempt-0001`.
Its independent scalar/scalar trajectories already produced different
gradients. Independent fused/scalar trajectories then exceeded the fixed
parameter tolerance at one coordinate, while identical-gradient fused/scalar
updates had zero mismatches across all five steps. This identifies a
qualification confound from independent nonlinear backward trajectories,
not evidence requiring weaker native optimizer arithmetic checks.

## Exact repair

Only existing driver definition `model_native_variants` changed. Added the
small helper `qualify_model_native_gradients(actual, reference)`:

- Fresh actual/reference initialization, seed streams, scalar reference,
  native optimizer parameter groups, epsilon, decay and tolerances stay as
  before. No snapshot is transferred across native variants.
- Actual fitting training still computes each current full-model gradient
  and performs its existing native clipping/validity checks.
- During exactly five numerical checks, a temporary actual optimizer entry
  clones every actual parameter's detached, already clipped gradient into
  the corresponding scalar reference parameter. A missing gradient is fatal.
- Invoke both captured **original native AdamW** implementations, then call
  the unchanged `assert_native_equivalent` on parameters, moments and exact
  step-counter values. The reference model does no forward/backward work;
  its training-engine counters and snapshots are unused.
- A `finally` block restores the exact original instance `step` attribute if
  one existed, or deletes the temporary instance attribute to restore native
  class-descriptor lookup if it did not. This happens on success, numerical
  disagreement and optimizer exceptions, before warmup/pilots can begin.
- Raw variant records disclose
  `qualification_mode: identical current full-model gradients after native clipping`.
  Successful qualification records also state `qualification_updates: 5`.

Qualification work remains outside measured blocks and charged to the whole
invocation. Warmup and pilot calls use the original native optimizer entry,
with no qualification callback or copied-reference work left in timing.
No tolerance, model, sampler, config, old native/D function, primitive,
optimizer arithmetic or performance/statistic gate changed. No hardware,
stage invocation, network, dependency operation, Git mutation or corpus access
occurred. Prior v6 attempt1 and all diagnostic artifacts remain untouched.

## Test-first evidence

Added two protocol-test definitions, one parametrized over four cleanup cases.
They use the existing actual tiny CPU model fixture and real native optimizers.

RED command:

```
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps' -k 'shares_identical_clipped or shared_gradient_qualification_restores'
```

`native-qualification-repair-red.log`: **5 failed, 272 deselected in 1.85s**.
The old code called the forbidden reference training step; the helper and
restoration contract were also absent.

GREEN command:

```
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps' -k 'shares_identical_clipped or shared_gradient_qualification_restores or actual_model_native_variants or actual_native_numerical_failure or model_native_numerical_and_device'
```

`native-qualification-repair-green.log`: **10 passed, 267 deselected in 2.93s**.

The real-model regression forbids scalar-reference model forward and training
step calls; checks bitwise-equal, separately allocated detached gradients for
every parameter at all five steps of every variant; verifies native clipping,
epsilon/betas, nonzero moments, optimizer step counts and unchanged reference
engine counters; and inspects exact native `step` lookup before **every** warmup
and timing operation. It checks persisted qualification-mode metadata.

Cleanup cases cover success and injected numerical failure, with both absent
and preexisting optimizer instance `step` attributes. After cleanup another
actual model step advances only the actual optimizer, proving no reference
qualification callback remains installed. Existing numerical/device-failure
fatality and fresh-reference/no-snapshot regressions also pass.

`git diff --check -- analysis/run_dw_sampling.py tests/test_dw_sampling_protocol.py`
passed with no whitespace errors. No broad suite was run, as requested.

## Source scope and handoff

`native-qualification-repair-diff.json` compares top-level AST definitions
against original head `b7506e537e1bb5e118745caf135e3200d8623d3a` and confirms:

- Driver: modified only `model_native_variants`; added only
  `qualify_model_native_gradients`; removed none.
- Protocol tests: no prior definition modified/removed; added only
  `test_model_native_qualification_shares_identical_clipped_gradients_and_leaves_pilots_native`
  and
  `test_model_shared_gradient_qualification_restores_exact_optimizer_step_on_success_and_failure`.

Independent scoped review remains the parent's next required check, followed
by prospective v7 archival/freeze/amendment and actual A/C before any pilot
attempt2. The CPU tests establish the revised qualification mechanics; they
are not an actual-MPS qualification or sampled cost/quality result.

Final SHA-256:

```
ccd8490f6048905ffa5f48864143ab1df6e12ed77af146a7877b47ab2230d5bd  analysis/run_dw_sampling.py
b271586886f1798b06a30c91834c4ceb470a4b3e5133de4f05a953932391f035  tests/test_dw_sampling_protocol.py
```

Driver/protocol ownership released after this report; no edits or tests remain
running.
