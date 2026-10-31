<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Backend, optimizer, cost and recovery review

Reviewed 2026-10-31, before measurements. This verdict applies to the fixed
`.superpowers/dw_sampling/review-package.txt`, including driver SHA256
`8598cd51451f1c53d2fd94b8c91f3850d8a22688de3c05031d6ec7e2e5dbfb1f`,
config `e94af731fffb69bcb8168bf66d17f4b173ee78c57e9518c99b8d5c3f51913dbf`,
sampler `26a3c880e9bc6e1093a6f1487b20e7df05cf0a76e663da44011fd0914cf7d228`
and the other package hashes. Source baseline is
`9d8ad069c51b4d0e6c13e9c6e465579e9b44a339`. Line references below are
file-local lines in that package. Live driver/config changes observed during
review are outside this verdict and require affected re-review. No source
was edited, suite rerun, hardware test or benchmark performed. The reported
100 operator plus 7 protocol CPU passes are supplied evidence, not an
independent result of this review.

**Spec verdict: changes required for A/B/D measurement readiness.**
**Quality verdict: standalone sampler mathematics and dense native AdamW
integration are sound on inspection; protocol driver is not ready to produce
fully qualified cost evidence.** None of these findings supplies a performance
or learning result. E/F trainer/checkpoint infrastructure is intentionally
unreached and is not a missing deliverable here.

## Findings

1. **P1 — Cost stage cannot launch under the frozen resource cap.**
   `analysis/run_dw_sampling.py:592-607,708-713` and
   `configs/dw_sampling.yaml:48`.
   For cost, `arms=6`, so `blocks=6*(60+64)=744`. The block-time floor is
   0.2 seconds, giving each shape a forecast strictly above
   `3*.2*744+15=461.4` seconds. Five shapes and two sessions, plus startup and
   the diagnostic allowance, give a forecast above
   `2*5*461.4+60+300=4974` seconds and deadline above 9948 seconds. The frozen
   cap is 7200, so every possible calibration returns INCOMPLETE_RESOURCE
   before launching D. This is a fail-closed configuration conflict, not a
   false speed claim or evidence of a hardware limitation. Reconcile the
   prospective capacity/forecast without weakening the prescribed sample
   counts or silently shrinking charged work before attempting D.

2. **P1 — A does not enforce the pinned runtime it claims to qualify.**
   `analysis/run_dw_sampling.py:230-252,397-400,413-427`.
   `dependency_inventory()` verifies installed files against each installed
   distribution's own RECORD, but never checks that required distributions
   are present at the versions in `requirements.lock.txt`, or that the
   interpreter is CPython 3.13. A different, internally consistent Torch
   installation passes the hash gate; a missing pinned package is not
   necessarily detected by this inventory. The inherited doctor's GPU gate
   checks native arm64, MPS and operations, not the pinned Python/package
   versions. Recording the lock/wheel manifest hashes does not bind the
   installed runtime to those archives. Assert the pinned runtime and package
   set, and distinguish RECORD integrity from verified wheel provenance.

3. **P1 — Exclusive measurement sessions are neither established nor
   evidenced.** `analysis/run_dw_sampling.py:511-589,714-731`.
   Fresh workers are serial within one invocation, but there is no shared
   measurement lock, competing-work check, external exclusivity attestation
   or per-session overlap invalidation. Two independent driver invocations
   can both run and seal passing measurements. The protected config lists
   `overlapping_process` as invalid, but nothing consumes this condition.
   A stable interfering workload can survive A/A and warmup checks. Establish
   and record exclusive-session ownership (including an explicit policy for
   external GPU work), and retain/invalidate overlaps rather than accepting
   them as qualified B/D evidence.

4. **P1 — Preparation can exceed its inclusive 240-second deadline while
   continuing device work.**
   `analysis/run_dw_sampling.py:199-205,390-425`.
   Each serial command receives a fresh default 240-second timeout, and
   dependency hashing has no remaining-time check. The aggregate deadline is
   checked only after doctor, CPU tests, report, MPS tests, full suite and
   primitive qualification. Even worse, the hardware-blocked return at
   line 414 bypasses that final check. This can spend multiples of the
   authorized stage deadline and misclassify the stop. Pass one common
   remaining deadline through all preparation work and check it before each
   launch and before every return. Disk reserve/output limits are also only
   recorded/configured, not enforced; bind these guards before launching
   hardware qualification.

5. **P2 — Native selection accepts unstable baseline pilots.**
   `analysis/run_dw_sampling.py:451-459,735-739`.
   A native variant is appended to `supported` even when its own baseline
   `warmup()['stable']` is false, and it can be selected as fastest using the
   next three blocks. The later instability gate scans only the comparison
   warmups, excluding `row['variants']`. Thus B may claim a qualified fastest
   native implementation whose selection timings never met the stability
   criterion. Keep the invalid native pilot evidence and prevent selection
   from unstable pilots; any replacement must remain within the permitted
   replacement policy.

6. **P2 — The selected dense implementation never verifies its minimum
   block duration.** `analysis/run_dw_sampling.py:537-554`.
   The pilot uses `Fixture`'s default `single` implementation, makes at most
   one duration adjustment, then selects a potentially faster native variant.
   It neither measures the adjusted block nor verifies the selected dense
   variant reaches 0.2 seconds when below 100 updates. D then reuses that B
   block length. Freeze block length from the selected dense implementation
   and retain its validation timing, allowing a shorter block only when the
   prescribed 100-update cap is reached.

7. **P2 — Recorded endpoint memory does not describe candidate endpoints.**
   `analysis/run_dw_sampling.py:531,560-577`.
   The shape's `resources_before` is sampled before fixture allocation and
   `resources_after` after every candidate is deleted, while dense/common
   remain live. No arm or block has a candidate endpoint memory snapshot.
   These separate RSS/Metal/driver fields are correctly not added, but they
   cannot substantiate the required endpoint memory of each comparison.
   Capture synchronized endpoint snapshots while the relevant arm remains
   live, outside timing, and label cache/shared-fixture context explicitly.

8. **P2 — Summary terminal state can remain an obsolete failed attempt.**
   `analysis/run_dw_sampling.py:775-785`.
   Records are grouped by stage and the last non-pass anywhere is chosen;
   newer passing attempts do not supersede an older failure of the same
   stage. A valid later prepare pass after BLOCKED_HARDWARE can therefore
   leave the final report claiming BLOCKED_HARDWARE. Preserve all attempt
   history, but derive current status from authoritative latest attempts and
   the dependency chain, consistently with `require_stage()`.

## Math and optimizer assessment

The importance gather at `src/oar/sampled_linear.py:105-133` uses the same
CPU FP64 probabilities for categorical sampling and inverse-square-root
weights. Scaling both gathered matrices produces
`sum_j D[i_j]^T X[i_j]/(k*p[i_j])`, retains duplicates and introduces no
extra division by n. For any valid positive p its conditional expectation
is `D.T@X`; its Frobenius variance is
`(sum_i ||D_i||^2||X_i||^2/p_i - ||D.T@X||_F^2)/k`. The offline formula at
driver line 645 agrees. Device FP32 score rounding changes the sampling
distribution and variance but does not create a missing inverse-probability
factor. FP32 inverse-weight/GEMM rounding remains a numerical approximation
covered by the stated tolerances, rather than an exact-real arithmetic claim.

Separate maximum-absolute scaling, finiteness checks and the positive score
sum check correctly distinguish proved unscaled-zero operands from reciprocal
scale score underflow (`sampled_linear.py:69-102`). Uniform uses CPU randint
and the single n/k factor. Native exact dispatch occurs before custom autograd
when k>=n. Input/bias adjoints remain algebraically exact; all sampled output
gradients are checked. Higher derivative/AMP rejection and weak-reference
graph-task shared-use detection are explicit. Static ownership/storage and
optimizer uniqueness validation has a sensible caller contract.

The residual fixture performs both linears, exact GELU, root adjoint, complete
microbatch backward, clipping and native dense AdamW before changing any
parameter (`run_dw_sampling.py:296-332`). Four equal microbatches use correct
1/4 loss weights. Zero-dW returns an allocated dense zero gradient and still
updates moment histories/decay. The common snapshot restores nonzero native
moments, parameters and sampling streams outside timed blocks. Both arms
use the selected optimizer variant, so loading the common optimizer state
does not introduce a variant-flag mismatch. The injected-sample multistep
test and biased-Adam-displacement counterexample test the relevant distinction:
unbiased dW does not imply unbiased Adam displacement or faster learning.

## Permitted limitations and recommendation

No genuine MPS qualification means BLOCKED_HARDWARE and CPU correctness/dry
runs only. That outcome, a correctly established STOP_COST, or a genuine
resource stop completes the bounded evaluation without a positive claim.
Unresolved documentary corpus restrictions correctly block all body use and
trainer expansion. No E/F restart checkpoint implementation is required
before those gates. B/D fixture snapshots are controlled benchmark resets,
not training recovery claims. Missing prospective amendment material before
the planned freeze is a preparation prerequisite, not a sampler defect.

The driver already preserves failed prefixes, seals artifact hashes, rejects
failed latest prerequisites, avoids summing separate profiles into step time
and makes no learning claim. Preserve these properties while correcting the
findings. **Recommendation: continue CPU-only preparation and affected code
corrections; do not accept B/D measurements until the readiness findings are
resolved and the corrected source/config are frozen and re-reviewed.**
