<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Mathematical/statistical review of 802e1f5

Read-only source review on 2026-10-31; the only write is this report. No tests,
Torch/NumPy calculations, profiling, downloads, or benchmark work were run.
The cost collection is incomplete; this review neither evaluates its raw
prefix nor declares a cost or learning benefit.

**Disposition: no blocking mathematical/statistical implementation defect
found in the inspected scope. Continue the frozen exploratory collection;
withhold a gate decision until both complete sessions and the paid diagnostic
are retained.** The limitations below must accompany any positive report.

## Estimator and optimizer

- `src/oar/sampled_linear.py:85-103,122-135,180-186` implements the 0.9 norm
  product / 0.1 uniform mixture using globally normalized FP32 score operands,
  CPU FP64 probability normalization, and the same probability tensor for
  replacement draws and inverse weights. Scaling both gathered operands by
  `(k*p)^(-1/2)` produces the required sum of outer products divided by `k*p`;
  duplicate indices remain separate contributions. FP32 score/weight rounding
  means numerical, rather than literal real-arithmetic, agreement with the
  ideal norm formula. Positive mixture support preserves the ideal estimator
  identity even when score rounding changes its variance.
- `sampled_linear.py:85-96,174-200` returns a zero tensor only after proving an
  unscaled complete operand is zero, rejects a zero normalized score sum for
  nonzero operands, and computes/checks raw dense dX and db separately. The
  reciprocal-scale FP32 counterexample is explicitly tested to abort before
  drawing (`tests/test_sampled_linear.py:145-154`). Native exact dispatch
  consumes no RNG (`sampled_linear.py:231-236`).
- The expectation/entrywise-variance enumeration is substantive: independently
  computed FP64 probabilities and all ordered replacement tuples are compared
  against both the exact dense gradient and the analytical second moment
  (`test_sampled_linear.py:34-59`). It tests the runtime gather independently
  of production score rounding; hand-calculated mixture and MPS references
  separately exercise production probabilities (`387-391,545-581`). It does
  not establish machine-exact unbiasedness for every FP32 input.
- Native AdamW receives the sampled dense gradient after native clipping
  (`analysis/run_dw_sampling.py:473-477,515-521`). Tests compare multiple
  optimizer steps and moments against independently injected outer products,
  demonstrate unequal expected Adam displacement despite an unbiased gradient,
  and distinguish zero gradients from absent gradients
  (`test_sampled_linear.py:324-364,492-501`). No unbiased optimizer-update or
  learning claim follows from gradient unbiasedness.

## Timing and decision rule

`run_dw_sampling.py:243-252` correctly computes the geometric mean of paired
candidate/dense ratios and a 10,000-replicate percentile bootstrap of paired
log ratios with seed 62001. `724-768` retains randomized within-pair order,
synchronized full steps, resets, and setup-inclusive and steady-state results.
Baseline-only block calibration precedes each shape's candidates (`829-855`).
The driver launches two fresh workers (`1035-1055`) and requires the same arm
and same middle2048/middle4096 shape to have upper endpoint strictly below 1
in both sessions (`1074-1088`). The old `1-delta` rule remains separately
reported sensitivity. Delta is written before candidate workers (`1028-1033`).
The current amendment records version 5 and a freeze at 12:59:41 UTC; HEAD
802e1f5 records its commit at 12:59:43 UTC. The reopening explicitly discloses
the post-native, pre-candidate policy change (`decisions/dw_sampling_reopening.md:
101-118`). Source/metadata inspection supports that intended chronology;
it cannot independently prove the absence of unrecorded prior outcomes.

## Material caveats, not blockers for this exploratory screen

1. **Serial dependence and conditional replay.** The bootstrap resamples whole
   pairs independently (`249-250`), preserving within-pair dependence but not
   serial host/thermal/allocator drift. Randomized order helps with order bias
   and does not prove exchangeability. Every pair restores the same sampler
   state (`530-538,729-753,865`), so its candidate trajectory is replayed;
   30 blocks are timing replicates, not 30 independent sampling/training seeds.
   Two fresh processes also do not establish independent hardware environments.
   Disclose order/time trends and unstable warmup flags; a borderline interval
   is conditional evidence, not a robust device-wide speed guarantee.
2. **Multiplicity and estimand.** Six arms across two eligible shapes give
   twelve possible successes; requiring two sessions does not supply a
   familywise error guarantee. All controls and failures must stay visible.
   The estimand is `exp(mean(log(candidate/dense)))`, not the arithmetic
   ratio of total training wall times. This is already disclosed as
   exploratory (`dw_sampling_reopening.md:38-54,68-73`).
3. **Diagnostic limits.** Offline variance correctly uses
   `(sum(||D_i||^2||X_i||^2/p_i)-||D.T X||_F^2)/k`
   (`run_dw_sampling.py:944-958`), conditional on fixed native operands.
   The 32 copied-state one-step draws at three states (`975-1005`) cannot
   establish endpoint quality, time to target, or accumulated AdamW behavior.
   The uniform diagnostic always scales X (`990-991`), whereas production
   scales the narrower gathered operand (`sampled_linear.py:194-197`): these
   are algebraically equal but not a bitwise FP32 replay, especially for w2.
   Treat those one-step numbers as diagnostic approximations.
4. **Zero-state diagnostic robustness.** The current random fixture is not
   shown to hit it, but a future zero/cancelling state can give `p=None`,
   `a.sum()==0`, or `exact_norm2==0` in offline diagnostics (`948-956,986-992`).
   Those paths are not all guarded, despite `normalized_variance` guarding its
   own denominator. This is a follow-up robustness gap, not evidence of a
   failure in this incomplete fixed-fixture run. Preserve/report any actual
   diagnostic failure rather than promoting timing alone to a completed gate.

Declined to judge: corpus use authorization, backend performance, artifact
seal/transitive continuity auditing, and original historical OAR behavior;
those require separate documentary/backend/reproducibility review and no
historical method changes were requested here.
