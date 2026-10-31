<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Owner-authorized reopening: pursue a useful positive result

2026-10-31. The owner explicitly instructed: “rewrite the rules: the primary
 goal is to achieve positive results, do not put stupid artificial caps; fix
 it and continue working.” This supersedes the prior timing stop and brittle
measurement limits. Implementation starts from local commit
40fd56fa71ec80bb0d28aa2e49ab8e5b2a13fef4 on research/dw-sampling.
The original review release remains 9d8ad069c51b4d0e6c13e9c6e465579e9b44a339.

The primary engineering objective is to obtain a useful, repeatable measured
benefit from the specified sampler, then pursue the contextual learning claim
if its prerequisites can be established. We will fix ordinary engineering
problems and use profiles to improve the implementation. Positive results
must be real: no discarded slow valid observations, hidden costs, weaker
baselines, changed correctness tolerances, selective reporting or invented
learning outcomes. These are accuracy requirements, not arbitrary runtime caps.

## What changes prospectively

- Remove the 100-update ceiling. Baseline-only calibration increases integer
  updates per measurement until the complete synchronized dense block lasts
  at least 0.5 seconds. No fixed iteration/update ceiling prevents that target.
  Record calibration observations and freeze that block length before matched
  candidate measurements. Longer measured work is charged.
- Last-eight warmup ranges remain diagnostics. Shader/allocator warmup uses
  the existing 8-to-32-block observation harness, but an unstable flag does not
  reject a native variant, shape, session, or the entire experiment. Actual
  paired observations and their uncertainty determine the cost conclusion.
  This is not retrying a bad sample or deleting slow blocks.
- Start with the relevant n4096/n2048 shapes, retain the accumulation/wide/tiny
  controls, and complete the declared shapes in both fresh sessions. Tiny is
  an adverse diagnostic, with no authority to veto a relevant shape's result.
- Qualify all native AdamW variants numerically, comparing step-counter values
  despite legitimate CPU/device storage differences. Select the fastest
  supported native implementation from baseline-only pilots, retaining all
  observations and warmup flags; no native optimization is disabled.
- Replace the fixed 5% minimum gain with measured uncertainty. Freeze
  delta=2*linear_quantile95(abs(TA/TB-1)) using A/A pairs at n2048/n4096 across
  the two sessions before inspecting sampler timing. There is no 15% delta
  rejection cap. The paired candidate upper interval must beat 1-delta at a
  relevant shape in both sessions. This remains an exploratory screening rule,
  not a familywise guarantee or a learning result. All shape results remain
  visible, including noisy or negative controls.
- Remove the one-replacement-session rule and arbitrary whole-stage duration
  and output-size ceilings. The initial plan still collects two fresh sessions
  and 30 matched pairs per arm; it does not keep drawing until a positive
  interval appears. Identified implementation or environmental failures can
  be repaired and followed by a new versioned attempt, with every prior
  attempt retained and affected evidence invalidated. Any additional timing
  session gets a prospective reason and its complete data are reported.
- Resource forecasts remain advisory and complete. Real available-memory,
  allocator, disk-reserve, numerical-validity and exclusive-session safeguards
  stay active. Do not kill applications, disable allocator safeguards, buy
  external compute or consume unavailable disk to force a result.

The owner has authorized continuing this exploratory engineering work; no
repeat confirmation is needed for these local fixes, tests, measurements or
local commits. Corpus licensing/use restrictions, untouched official test,
algorithm identity, exact forward/dX/db, native AdamW and fair baselines remain
in force. Generic removal of artificial caps is not a documentary resolution
of the WikiText notice conflict; investigate that independently before body
acquisition if cost permits language work. No old online OAR work is reopened.

The new run root is analysis/dw_sampling/20261031T120154Z-reopened. Its initial
inventory and prospective directory preserve the previous config/amendment.
Earlier stopped results and all historical scientific evidence remain intact.
This new scope is exploratory after prior timing exposure, not a retroactive
repair of the old gate or a frozen confirmation. Source/config hashes and a
new amendment must be committed before new comparative measurements.

## Implementation plan

1. Test uncapped block calibration, diagnostic warmup handling, measured-noise
   delta without an arbitrary floor, and nullable resource limits. Keep actual
   bad gradients/probabilities fatal; never resample after a numerical failure.
2. Update only the new sampling driver/config/protocol tests and prospective
   bindings; keep the already-correct sampler unchanged until profiling gives
   a concrete reason to improve it. Preserve old snapshots and direct hashes.
3. Re-run qualified environment/correctness checks; execute reopened B and D
   sequentially with complete native optimizer work, common cloned states,
   exact adjoints, all declared shapes/rhos/controls, inclusive setup charges,
   raw observations and separate component diagnostics. No overlapping tests
   or collectors during timings.
4. If costs identify a concrete removable implementation overhead, improve it
   without changing the algorithm, primitive reliability or baseline, freeze
   the new source and repeat affected comparisons. Do not substitute a new
   algorithm or claim a component win establishes learning.
5. Pursue the specified language workload only after cost and asset gates;
   otherwise report the actual unresolved boundary and useful measured result.
   Run mathematical/statistical, backend/optimizer/cost, and autonomous
   reproducibility reviews of the reached implementation and evidence.

Software AGPL-3.0-only; original writing/results CC-BY-4.0. Prior-art attribution
and all original method/data/quality safeguards not explicitly revised above
remain as recorded in dw_sampling_scope.md and the owner's original task.
