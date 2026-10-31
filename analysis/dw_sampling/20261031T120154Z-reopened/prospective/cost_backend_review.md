<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Backend, optimizer, and charged-cost review

Reviewed HEAD `802e1f5b16ac915fcce5204aa96c3a64bacd6154` on 2026-10-31.
Independent source review; no source changes, tests, benchmarks, Torch/NumPy
computations, profiling, telemetry collection, or downloads were performed.
The active exclusive M3 Max/MPS benchmark was left undisturbed. Only this new
review note was written; historical files were not edited.

Disposition: **no blocking implementation finding in the reviewed cost path**.
Current D is incomplete. Provisional first-session ratios, including the
reported middle4096 importance-0.125 and uniform-0.125 results, are not a D pass
or evidence of language learning.

## Finding: preparation retains an artificial deadline

**Nonblocking follow-up before a future prepare rerun.**
`analysis/run_dw_sampling.py:584-590` unconditionally creates a deadline at
twice `qualification_forecast_s`; the configured value is 120 seconds
(`configs/dw_sampling.yaml:98`). That 240-second deadline is checked and passed
to every preparation command (`analysis/run_dw_sampling.py:594-627`), regardless
of `maximum_stage_s: null` (`configs/dw_sampling.yaml:100`). Thus a sufficiently
slow, otherwise valid qualification still stops for the fixed preparation cap.
This differs from the reopened removal of arbitrary whole-stage duration
ceilings (`decisions/dw_sampling_reopening.md:48-58`).

The already-passed preparation is unaffected; this is not grounds to discard
or interrupt current D. Before another prepare run, make its deadline obey the
nullable runtime policy and freeze the resulting source/metadata continuity.
Keep real numerical, memory, allocator, disk, and exclusivity safeguards.

## Cost boundary and baseline checks

- `Fixture.step` executes complete residual forward/loss, effective-batch
  backward, native total-norm clipping with `error_if_nonfinite=True`, and
  native AdamW (`analysis/run_dw_sampling.py:486-522`). Both arms retain root
  input derivatives and bias gradients. The baseline calls `F.linear` directly
  and creates no sampler state (`analysis/run_dw_sampling.py:478-483,492-498`).
- Candidate probability/gradient finite checks and their scalar conversions
  execute inside backward. Operand reductions, FP32 score construction, score
  transfer to CPU, FP64 normalization/validation, categorical RNG, inverse
  weights, integer/weight transfers, gathering, smaller GEMM, and exact dX/db
  are inside the timed step (`src/oar/sampled_linear.py:64-135,169-201`). Uniform
  uses CPU `randint` and its constant factor without importance-score work.
  These checks can synchronize the device; those waits are charged.
- `_measure_blocks` drains previous work before starting, then runs the entire
  operation and final device synchronization before stopping
  (`src/oar/timing.py:39-49`). No component-profile sum determines a step time.
- The two per-layer CPU generator constructions are timed at fixture setup
  (`analysis/run_dw_sampling.py:478-483`), then added once across the 30 timed
  candidate blocks (`analysis/run_dw_sampling.py:760-768`). The charged
  `statistics`, rather than `steady_state_statistics`, drive the gate.
- Common parameters, nonzero optimizer history, data-generator states, and
  sampling streams are restored outside each measured block, with reset wall
  costs recorded (`analysis/run_dw_sampling.py:524-542,724-768,847-868`). This
  exclusion is expressly specified for the replay experiment
  (`decisions/dw_sampling_scope.md:109-110`); it does not describe a training
  checkpoint/reset cost saving. Repeated sample streams are timing replays,
  not independent learning runs.

## Native variants and backend qualification

`native_variant_qualification` executes default/single/foreach/fused against
five scalar-native updates, checking parameters, moments, and exact scalar step
values (`analysis/run_dw_sampling.py:655-721`). CPU versus MPS storage of the
native step counter is handled by comparing its CPU values, preserving zero
tolerance. Baseline-only median pilots select the implementation; unstable
warmup does not exclude it. Fatal gradient/device/allocation failures propagate
instead of silently selecting a weaker implementation. D carries the selected
B implementation and recalibrates block duration using dense work only
(`analysis/run_dw_sampling.py:824-855`).

The retained B middle4096 native-variant JSON files for both sessions list all
four variants as passing, including fused; this review found no missing variant
or fused-counter disqualification at that relevant shape. Selection remains
the fastest measured supported pilot, not proof of an absolute hardware optimum.

Fresh workers enforce fallback/fast-math/prefer-metal values before Torch
import and require actual available MPS (`analysis/run_dw_sampling.py:318-320,
442-452`). An environment string alone does not prove device execution. The
preparation path additionally requires doctor's actual-MPS qualification,
device/full-suite commands, and actual-shape sampler backward checks
(`analysis/run_dw_sampling.py:545-577,594-630`). This review inspected those
paths and retained native-variant evidence; it did not rerun device checks or
infer absence of all internal CPU work. The sampler's explicit CPU work is
intentional and charged.

## Test coverage and claim boundary

Inspected regression coverage includes fatal native failures versus numerical
disqualification (`tests/test_dw_sampling_protocol.py:439-472`), setup charging
despite unstable warmup (`:504-528`), matched arm counts and dense-only duration
calibration (`:589-646`), native counter device differences (`:720-736`), actual
MPS transfer/backward boundaries (`tests/test_sampled_linear.py:545-581`), and
final-synchronization charging (`tests/test_timing_harness.py:19-22`). These are
source observations, not a claim that tests were run during this review.

D requires the charged upper 95% paired ratio below 1 for the same arm and
middle2048/middle4096 shape in both completed sessions, after all declared
shapes and the separately paid diagnostics (`analysis/run_dw_sampling.py:
1050-1089`). The stricter `1-delta` screen remains descriptive. Zero-dW is a
nonlearning diagnostic. Neither a cost pass nor a copied one-step diagnostic
establishes target recovery or endpoint language quality. The gated future LM
trainer/recovery work is not implemented and is not a pre-cost review defect.
