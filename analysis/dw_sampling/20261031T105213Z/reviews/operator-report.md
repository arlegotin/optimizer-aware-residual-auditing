<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Task 1 operator report

Status: DONE for standalone sampler and deterministic CPU correctness. No Git
mutations, hardware tests, downloads, training, or benchmark measurements were
performed. Only `src/oar/sampled_linear.py`, `tests/test_sampled_linear.py`, and
requested `.superpowers/dw_sampling/operator-*` records were written.

## Implementation and interface

- `SamplingState(base_seed: int, layer_id: str, sampler='importance', rho=.25)`
  owns a dedicated CPU Torch generator and resolved named `seed`; snapshot and
  restore use `state_dict()` / `load_state_dict()` with metadata validation and
  independent generator-state storage. Ratios supported: .125, .25, .5, 1.
- `named_seed(base_seed, domain, *ids)` uses SHA256 of compact ASCII JSON,
  first eight bytes big endian modulo `2**63-1`.
- `sampled_linear(x, weight, bias, state, *, k=None)` uses native `F.linear`
  forward. Production k is `max(1, ceil(rho*n))`; k>=n dispatches directly to
  native `F.linear`, without custom autograd or RNG draws. Sampled backward
  computes exact input and bias derivatives and actual dense sampled dW.
- `importance_probabilities(x2, d2)` returns CPU FP64 probabilities, or None
  only after proving a whole unscaled operand zero. Separate maxabs scaling
  precedes device FP32 row norms, preserving CPU FP64 reference scales before
  casting. Only the n-element score vector moves to CPU. Invalid/nonfinite
  scales/scores, zero score sums for nonzero operands, invalid probabilities,
  or nonfinite produced gradients raise errors; no fallback/resampling occurs.
- `gather_weight_gradient(x2, d2, indices_cpu, probabilities_cpu, k)` is the
  actual importance runtime gather, inverse-square-root split scaling, and
  small dense GEMM path. Duplicates remain present. Enumeration independently
  computes reference FP64 probabilities and supplies them to this actual path.
- Uniform uses direct dedicated-generator CPU `randint`, checks finite scales
  and proved zeros, evaluates no score vector, and applies its single n/k
  factor once to the narrower gathered operand before its small GEMM.
- `validate_sampled_parameters(model, eligible_module_names,
  optimizer_parameters)` accepts an iterable of actual optimizer Parameters
  (flatten parameter groups at the call site). Every unique model Parameter
  must appear exactly once; sampled modules/weights cannot be reused, aliased,
  or share storage with another distinct model Parameter/view. Exact native
  embedding/output ties remain legal when excluded from eligible names.
- A bounded global identity/weakref registry records one last backward task ID
  per live sampled weight. It detects repeated sampled use in one backward
  task even across distinct SamplingState objects, allows independent
  microbatch backwards, and releases dead weights. It requires
  `torch._C._current_graph_task_id`; absence/unsupported execution fails
  visibly. This primitive remains part of actual-backend qualification.

Sampled execution supports FP32 CPU/MPS and explicit CPU FP64 reference inputs;
AMP, other dtypes, empty rows, malformed budgets, and sampled higher-order
`create_graph` requests are rejected. Native exact dispatch intentionally keeps
native F.linear's differentiation policy. The caller must check native and
unsampled model gradients before optimizer.step; custom sampled backwards
validate their produced dW/dX/db. Model updates must await the entire effective
batch backward. Caller loss weighting carries valid-token fractions.

## Deterministic test coverage

Finite enumeration of (n,k)=(2,1),(3,1),(3,2), for importance and uniform, checks
unbiasedness, independent analytic variance, and retained duplicate draws.
Reference enumeration probabilities are FP64 row-norm mixtures independent of
runtime device-FP32 scores; a separate hand-computed fixture checks the runtime
mixture. Exact dispatch for both samplers and n=1, budget boundaries, native
forward equality, frozen inputs/weights, bias presence, noncontiguous leading
axes, and deep nonlinear residual root/bias/unsampled gradients are covered.

Numerical cases include concentration and score underflow through n=8192,
actual gathered FP32 inverse-scale rounding bounds, rank deficiency,
cancellation with a prospectively fixed scale-aware absolute bound,
wide/tiny/large reciprocal scales, CPU FP64 1e-150 and 1e150 scales,
nonfinite operands, reciprocal-scale score-underflow aborts, gradient overflow,
proved zeros, exact-derivative overflow despite zero dW, and exactly one draw
consumed on failed sampled-gradient production. RNG tests cover global-stream
isolation, distinct named layer/sampler streams, deterministic draw consumption,
restore, independent unequal-size valid-token-weighted microbatch backwards,
state alias protection, and dynamic shared-weight/registry lifecycle behavior.

Multistep native AdamW is compared with independently precomputed sampled outer
products using identical draws, clipping, nonzero moments, nondefault epsilon,
parameter groups, weight decay, and zero gradients. Separate tests demonstrate
zero versus None optimizer effects and the unbiased [-1,3] gradient's biased
Adam displacement. Conditioned FP64 comparisons use rtol1e-10/atol1e-12;
FP32 comparisons use at most rtol1e-4/atol1e-6. No Monte Carlo acceptance gates
or timing claims are present.

## Commands and retained results

All commands executed from the repository root with the pinned `.venv` Python.
Each pytest command redirected stdout/stderr to the indicated retained log.

1. `PYTHONPATH=src .venv/bin/python -m pytest tests/test_sampled_linear.py -q`
   -> `operator-red.log`: exit 1, 1 passed, 47 setup errors. Every setup error
   is the explicit assertion that the requested standalone sampler is absent;
   the independent mathematical Adam counterexample passes without the module.
   This RED was recorded before writing any sampler implementation.
2. Same command -> `operator-green-initial.log`: exit 0, 48 passed.
3. Same command after broader boundary tests ->
   `operator-green-expanded.log`: exit 1, 86 passed, 1 failed. Failure was a test
   fixture accidentally creating an integer zero weight with requires_grad;
   corrected to floating zero without changing tolerances or expectations.
4. Same command -> `operator-green-expanded-corrected.log`: exit 0, 87 passed.
5. Same command -> `operator-green-final.log`: exit 0, 92 passed.
6. `PYTHONPATH=src .venv/bin/python -m pytest -m 'not mps' -k 'not test_mps_import_order_uses_fresh_fallback_disabled_worker' -q`
   -> `operator-cpu-suite.log`: exit 1, 446 passed, 7 failed, 10 deselected.
   The nine existing marked MPS tests and the unmarked fresh-MPS-worker test
   were excluded to honor the no-hardware instruction. All seven failures
   were new parent-owned protocol tests run before `analysis/run_dw_sampling.py`
   existed; six raised FileNotFoundError importing it, and the --help subprocess
   failed for the same absent file. The parent subsequently reported its seven
   protocol tests passing after creating the driver. The retained failed tests:
   - test_configuration_rejects_changed_protected_protocol_and_unknown_fields
   - test_attempts_append_and_sealed_records_detect_tampering
   - test_missing_failed_and_source_changed_prerequisites_cannot_launch
   - test_geometric_paired_estimator_and_noise_threshold
   - test_zero_dw_control_keeps_adjoint_bias_and_native_moment_decay
   - test_summary_handles_incomplete_run_without_promoting_it
   - test_help_has_no_output_directory_or_training_side_effect
7. After review tightened FP64 adjoint tolerances, independent enumeration
   probabilities, and both-sampler exact-dispatch coverage, same focused command
   -> `operator-green-reviewed.log`: exit 0, **100 passed in 1.97s**.

Testing has stopped per parent instruction; parent owns serial comprehensive
qualification and hardware scheduling. No sampler test failure remains.

## Source identities and remaining boundaries

`shasum -a 256 src/oar/sampled_linear.py tests/test_sampled_linear.py` returned:

- sampler: `26a3c880e9bc6e1093a6f1487b20e7df05cf0a76e663da44011fd0914cf7d228`
- tests: `e4f131100065a836281030db98880530316086fce0f0be4e00b99fa63c838443`

No MPS correctness/performance claim is made. Private backward-task primitive
availability and actual MPS operations require parent qualification. Exact
native dispatch needs caller gradient-finiteness checks and retains native
higher-order semantics. Registry ownership is designed for the scoped serial
backward/microbatch lifecycle; concurrent backwards on one weight are outside
the scoped driver lifecycle. No trainer or historical mechanism was modified.
