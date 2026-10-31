<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Native qualification repair: independent review

2026-10-31. **PASS — safe to freeze version 7, rerun A/C, and, after their
successful current gates, execute a fresh pilot attempt 2. No required fix
identified in this repair.** This approves the exact source/test/scope hashes
below and their full diff for renewed B/D continuity. It is not actual-MPS
qualification of the repair or permission to reuse partial variant timings.

Reviewed `native-qualification-repair.md`, the full driver/test diff against
`b7506e537e1bb5e118745caf135e3200d8623d3a`, the new v7 scope section, and the
retained native-only diagnostic's prospective declaration and results. Only
`model_native_variants` changed among existing driver definitions; only
`qualify_model_native_gradients` was added. No existing test definition changed
or disappeared; the two new definitions cover five parametrized cases.

The diagnostic supports this narrow isolation: scalar/scalar independent
backwards already differ, independent fused/scalar reaches one parameter
tolerance mismatch at steps3–5, while identical-gradient fused/scalar reports
zero parameter mismatches and zero gradient differences for all five steps.
It does not erase the independently observed differences or establish general
backend determinism. Those observations and the failed pilot remain evidence.

## Correctness and timing boundary

- `run_dw_sampling.py:1555-1584` captures both original native optimizer
  entries. Actual training supplies each current complete-model gradient
  through its ordinary forward, backward, finite checks and native clipping.
  The callback clones detached gradients before either optimizer step. It
  clears the reference gradients first and treats a missing actual gradient
  as fatal. Both optimizers therefore receive identical, independently owned
  inputs; no full reference forward/backward is mixed into this arithmetic
  comparison.
- Both models and native optimizers are freshly constructed. Identical
  initialization is checked before replay; matching parameter names, groups,
  LR, epsilon, betas, decay and step history remain supplied by the existing
  fixture. No cross-variant snapshot is loaded. Reference engine counters are
  intentionally unused; the reference native optimizer's parameters, moments
  and actual step counters are compared after each of five updates.
- `assert_native_equivalent` is unchanged: parameters/moments retain
  rtol1e-4/atol1e-6, and optimizer step values compare exactly. Neither a
  tolerance relaxation nor an unsupported-variant exemption hides disagreement.
  Numerical disagreement remains fatal under the existing classifier.
- The `finally` restores the exact prior instance `step` object when present;
  otherwise it deletes the temporary instance attribute, restoring the native
  class descriptor. This applies to success, numerical assertion failures and
  exceptions from either original optimizer. Warmup and pilot timing occur
  only after the helper returns and the reference is released. No hook,
  gradient-copy callback or reference optimizer work remains on the timed path.
  Qualification overhead remains part of the whole invocation.
- Raw native variant records disclose the identical-clipped-gradient mode;
  passing variants record five qualification updates. The v7 scope requires
  fresh complete native pilots, preserves the failed attempt, and repeats
  current A/C. It correctly retains original B/D attribution and protected
  official-validation-loss/test boundaries after the already-authorized body
  acquisition in pilot1.

## Independent verification

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py \
  -m 'not mps' \
  -k 'shares_identical_clipped or shared_gradient_qualification_restores or actual_model_native_variants or actual_native_numerical_failure or model_native_numerical_and_device'
10 passed, 267 deselected in 2.86s

git diff --check -- analysis/run_dw_sampling.py tests/test_dw_sampling_protocol.py
exit 0, no output
```

Two additional tiny invented CPU-model probes injected an exception into,
respectively, the captured actual optimizer entry and the reference optimizer
entry. Both propagated the original exception, restored exact actual `step`
lookup and left the training state fenced against reuse. These probes did not
edit source/tests or perform hardware measurements.

Independent AST checks confirm no protected original B/D measurement,
statistical or gate definition changed. Native configuration comparison still
passes with the already-reviewed exclusions. Read and verified complete B/D
artifact seals:

```text
B 716ec10cfe1827b8e85f03deb5229484ff7c20b80d2c7178b1395c4b5acda506
D d52f2062995810f4541022d02ed21e8f33ad72257675f66f8bed773c34fa9b18
```

The full diff adds no module initialization change or new call from an old
native B/D consumer, so the selected AST check is supplemented by direct
transitive-path inspection. All 184 original historical file hashes still
match. The retained pilot1 record and all its sealed artifacts also validate:
`5cf30c43d64f3cb52100537299d0aba909c70007635dd5b10fbf0635b84eeb00`,
status `INCONCLUSIVE_MEASUREMENT`; it is not upgraded by this approval.

Exact reviewed hashes, checked again after the v7 scope addition:

```text
ccd8490f6048905ffa5f48864143ab1df6e12ed77af146a7877b47ab2230d5bd  analysis/run_dw_sampling.py
b271586886f1798b06a30c91834c4ceb470a4b3e5133de4f05a953932391f035  tests/test_dw_sampling_protocol.py
e5f17839a1e3696793b9adee485c873484f2b91a8ab4880032d0a7d67122e018  decisions/dw_sampling_reopening.md
```

All other source/config restrictions from the preceding qualification review
remain in force. Final amendment generation must bind these exact bytes and
retain the archived v6 identities. No hardware, network, corpus-body request,
Git mutation or implementation edit was performed by this reviewer.
