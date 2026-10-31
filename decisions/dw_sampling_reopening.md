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
- Remove a fixed minimum required speedup. Preserve
  delta=2*linear_quantile95(abs(TA/TB-1)) as a native A/A jitter diagnostic,
  computed from n2048/n4096 across both sessions before sampler timing. There
  is no 15% delta rejection cap. For sampled cost, require the paired candidate
  95% upper time-ratio interval to be below 1 at the same relevant shape in
  BOTH sessions. Also report the old stricter 1-delta screen descriptively.
  Individual-block jitter is not itself a minimum worthwhile mean improvement;
  the paired interval quantifies uncertainty in the measured mean log ratio.
  These are exploratory screens, not familywise guarantees or learning results.
  All shape results remain visible, including noisy or negative controls.
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

## Cost-policy refinement before any sampled timing

After B completed and before D/candidate timing, the native A/A diagnostic was
0.2085046466766194. Requiring that much minimum gain in addition to a paired
confidence interval would reject even a repeatable smaller benefit. Under the
owners explicit removal of artificial limits, use upper95%<1 for D and retain
the 1-delta comparison as a disclosed sensitivity screen. No sampler timing or
learning outcome was inspected to make this change. B's native work, raw data,
original delta and passed headroom result remain exactly as recorded at d778286.

Carry over unaffected A/B prerequisites only through a new hash-bound record
continuity declaration. It must match the exact prior sealed record, old/new
measurement maps, Git source identities, unchanged measured function ASTs,
unchanged sampler/hardware/timing source, and unchanged native configuration
(after removing only the new cost-gate field). Current correctness tests run
again. This avoids remeasuring 24 minutes of unchanged native work merely
because an unused D decision branch or metadata validator changed. It never
turns a failed gate into a pass or attributes old times to new measured code.

The A/A jitter diagnostic likewise does not become a minimum required learning
speedup. If learning gates are reached, freeze the held-out target and unchanged
common-budget quality guard before protected outcomes, and require the seed-pair
95% upper target-time ratio below 1 plus the existing endpoint-quality bound.
Monitor resolution, interval sensitivity and simpler-competitor ties are still
reported. Choose a fixed confirmation run count from independent development
variability and real available resources; a historical arbitrary maximum of16
is not a reason to discard an otherwise feasible adequately powered study.
Do not add seeds after inspecting a confirmation p-value, retune protected
thresholds, or call an unresolved/negative result positive. These clarifications
are prospective: no sampled cost or language-learning outcome has been observed.

Continuity declarations approve only the exact source hashes and complete diff
reviewed for that declaration. The selected AST comparison is a consistency
check, not a general proof about arbitrary future module code or transitive
helpers. Every refreshed declaration needs a renewed exact-diff review that
also checks unchanged helper/initialization/statistical code; an old declaration
cannot approve a new driver hash automatically.

## Post-cost implementation and qualification, version 6

The complete D record at cost/attempt-0001 in the reopened run is PASS, SHA256
`d52f2062995810f4541022d02ed21e8f33ad72257675f66f8bed773c34fa9b18`.
Both sessions, all five fixtures and the paid 576-draw diagnostic are retained.
The old stricter 1-delta screen passed no arm; v5's prospectively frozen
upper95%<1 rule is the basis for continuation. This is an exploratory operator
result, with no language quality outcome. The original 184 tracked files remain
unchanged; the raw D evidence was committed separately at b4982f3.

Implement the specified byte-data/model/runner units now. First freeze and run
`learning.action: qualification`: acquire and verify the conditionally permitted
train/validation bytes, freeze lossless article/chunk/window manifests, and run
full-model native qualification, two-session A/A and candidate cost, and cold
baseline calibration. This action returns `not_run` for the overall pilot when
qualification succeeds: it cannot pass Stage E or unlock confirmation before
tuning and independent development. It records STOP_COST if the full-model
cost gate actually fails. A subsequent frozen implementation runs the remaining
pilot work. This staged engineering does not stop the authorized work or turn
an unfinished pilot into successful learning evidence.

Use one pooled strongest native AdamW variant, chosen only from the two native
variant pilots and supported in both sessions, in both final candidate sessions
and subsequent matched arms. Native A/A for that fixed variant precedes candidate
timing. Candidate fractions are only D survivors at 4096: importance1/8 and1/4,
uniform1/8,1/4 and1/2. No model growth or extra sampler. All steps include actual
causal attention, data collation, padding, bias/embedding/norm/tied-output gradients,
native clipping and native AdamW. All input/bias adjoints are exact before global
clipping; the native global clipping factor depends on the sampled gradient norm,
so applied unsampled parameter updates need not equal dense training.

Schema 3 adds model/data metadata; its version number is excluded from native
B/D configuration continuity only for the reviewed supported schema 2→3 mapping.
Actual native shapes, dtype, flags, seeds, optimizer and timing are unchanged.
The fixed D cost gate remains1 in that comparison. New model/data files are
bound directly for preparation/correctness/pilot/confirm, while unrelated file
addition does not invalidate old native work. Renew exact-diff approval at the
final new source hashes; selected AST equality alone is insufficient. Repeat
current preparation and correctness after freezing the new code. Removing the
remaining preparation/default command timeout does not alter the already-passed
native timing observations. End-of-stage source/config/amendment drift now
invalidates an attempt rather than silently attributing old execution to new code.

The conditional local-use asset decision and documentary sources in the run's
prospective directory permit only local nonredistributed copying, labeled
preprocessing and private research at the pinned WikiText revision. Preserve the
CC-BY-SA3.0/GFDL versus4.0 conflict and all original notices; do not claim an
authoritative version choice or erase the historical blocked manifest. This new
configuration enables only train/validation body acquisition after the sealed
cost gate and exact asset-decision hash. PyArrow 26.0.0 is the sole added reader,
with its official native wheel SHA256 recorded; original dependency pins remain
unchanged. Test body/parsing stays sealed. Official-validation loss outcomes
remain sealed until the complete confirmation protocol is frozen.

Learning calibration retains a 4096-native-update-equivalent horizon and the
baseline-only cadence/overhead measurements. A/A jitter is diagnostic and no
longer vetoes a calibration merely for exceeding 0.15. The common cutoff overshoot
limit is 0.01*H, rather than multiplying H by a removed required-effect threshold.
Report r_time=Delta+Bmax and its fraction of each reached target time. Initial
full and monitor evaluations precede updates; an initial committed checkpoint
at the first block's original t0 is charged within that zero-LR first block and
included in conservative startup calibration. Native loss-finiteness validation
occurs after backward/native clipping and before AdamW, avoiding an unnecessary
separate forward/backward host barrier. Scalar step statistics are transferred
only after the synchronized block.

If learning gates are reached, plan fixed N from the three independent development
seed pairs: N_time=ceil((2.8*s_logtime/(-mean_logtime))^2), requiring a negative
mean, and retain the original quality-margin formula. Use N=max(8,N_time,N_quality),
without an arbitrary16 ceiling; actual device-hours/storage must cover the plan.
The observed pilot effect is uncertain and can overstate planning power. Freeze
a vetted t critical value for N before protected outcomes; use the supplied
df7..15 values when applicable. Do not add seeds after observing confirmation.
Primary upper time-ratio CI<1 and the unchanged endpoint quality bound must both
pass and agree with bootstrap sensitivity. Keep the original target construction,
twice-observed crossing, simpler-control ties, censor treatment and independent
repeat rules. No favorable unobserved target crossing is inferred.

Checkpoint files have a prospective retention policy: two latest live resume
files plus final endpoint, with every creation/pruning receipt and SHA retained.
Only files created by this new run under its ignored private checkpoint cache
are eligible for this managed pruning; no prior-run or user files are removed.
A failed write cannot replace a prior good file, and an unreceipted orphan is not
a committed checkpoint. A restart explicitly selects a retained hash-validated
file; the parent ledger includes all previous active work, wasted uncommitted
work, new imports/loading and I/O, excludes external downtime, and preserves
original t0. Native moments, their scalar steps, model/data/RNG states and
compatibility metadata are validated before restore mutates live state. The
fixed LM engine supports its fully trainable model; standalone sampled_linear
continues to support frozen operands independently.

Pooling native variant pilots uses seconds per update with each session's actual block count, retaining raw seconds/counts. Horizon calibration's U and Bmax use all qualified observations for the fixed native implementation, including selected-variant pilots, both native A/A arms, every candidate-comparison native block, and the three cold controller blocks. No slow valid observation is deleted. Parent and cold worker record and compare monotonic-clock implementations.

While a run is active, retain at most two live resume files. After a durable final checkpoint receipt, retire that run's remaining ephemeral resume files and retain the final endpoint alone. Every receipt, hash and pruning event stays in the public ledger. Interrupted blocks retain their LR, host-completed update/token counts, failure record and last committed checkpoint; unconfirmed MPS completion is explicitly unknown. These are prospective storage/recovery rules before any real training run.

## Native optimizer qualification repair, version 7

Version 6 preparation and correctness passed on the actual M3 Max. Pilot
attempt 1 then stopped during native optimizer qualification, before any
sampled full-model timing or language-loss outcome. Independent fused/scalar
training trajectories exceeded the unchanged parameter tolerance at one
coordinate. A separately recorded native-only diagnostic reproduced this
comparison and found small backward differences even between scalar/scalar
trajectories. Giving the native fused and scalar optimizers identical gradients
passed all five checks. The failed attempt and all diagnostic observations
remain retained; no tolerance is relaxed and no native variant is excluded
because of this result.

For version 7, qualify native optimizer arithmetic with the actual current
full-model gradients after native clipping. Clone those gradients into the
fresh scalar reference and invoke both original native AdamW implementations.
Compare parameters, moments and step counters after each of five steps with
the existing fixed tolerances. The reference has no independent forward or
backward trajectory. Remove the temporary qualification entry in a `finally`
block, restoring exact native method lookup before any warmup or timing.
Repeat all native variant pilots in fresh workers; do not reuse the partial
variant timings from the failed attempt. This is an engineering repair after
native-only outcome exposure, not a new sampled method or a learning result.

Freeze the repaired source and renewed exact-diff review, then repeat current
preparation and correctness before pilot attempt 2. Unchanged operator B/D
results keep their original measured revisions and sealed records under the
renewed exact source-continuity declaration. Train and validation bytes were
verified and parsed during attempt 1 under the conditional local-use asset
decision; reuse only hash-verified private caches and record new preprocessing
costs. Official-validation loss outcomes and the official-test body remain
sealed. All prospective quality, target, simpler-control and confirmation
protections continue to apply.

## Learning pilot implementation, version 8 (prospective)

The sealed v7 model qualification at `pilot/attempt-0002` has record SHA256
`d01c854daf8eb7f7fbc8505b0879b40bc869bfc469a141fa727ae086757c4209`.
Its nested qualification is PASS; its overall pilot remains `not_run`.
All five declared arms passed both full-model cost sessions against the pooled
native fused AdamW baseline. Preserve that distinction and the original measured
revision. The append-only model-cost interim report v2 names the exact formal
429-block calibration ledger and its excluded, retained sizing/warmup records;
no calibration number changes after seeing candidate outcomes.

The frozen learning cutoff is H=1461.4585008954164 seconds, reserve
R=29.399479834130034, T_budget=1490.8579807295464, monitoring interval
Delta=29.229170017908327, r_time=31.627891642889008, block size8,
overshoot limit14.614585008954164, and epsilon_eval=1e-6. Reuse the exact sealed
calibration, source/config/data identity and initial evaluation protocol rather
than recalibrating to an exposed learning outcome. The current eligible grid has
12 native trials, six importance trials and nine uniform trials, all seed3101.
Learning/selection/independent development remain unmeasured at this specification.

Complete the existing pilot action through the original endpoint/target
selection, simpler controls, independent seeds3102/3103/3104 and fixed confirmation
planning. A separate exact qualification consumer may read only the specifically
sealed successful nested qualification. It must preserve the overall `not_run`
status, compare full old/current source/config maps and obtain renewed complete
exact-diff review of every changed executed helper/initializer. It must not make
`require_stage(..., 'pilot', ...)` accept an unfinished or failed pilot. Adding
unconsumed training/orchestration paths does not relabel the old timing code as
a newly measured revision. Repeat current preparation/correctness after the new
source/config/amendment freeze; remeasure affected costs if a consumed path changes.

Select the native batch16 secondary control by its best finite full-development
endpoint grid LR, with epsilon_eval ties preferring lower LR. Select uniform by
its own fastest endpoint/target-eligible grid result, with r_time ties preferring
larger rho then lower LR. If none is eligible, retain the best finite endpoint
uniform arm as a disclosed adverse mandatory control; a censored control is not
evidence of favorable dominance. If every allowed uniform arm is numerically
invalid, do not claim that the required control qualified. Freeze these settings
before independent seeds. Exact role aliases reuse the same run without funding
additional alternatives. Preserve the specified uniform simplification and the
native smaller-batch/qualified-precision dominance rules.

The native precision control uses an isolated implementation inside the existing
model module, keeping the measured FP32 step/model/evaluation helpers unchanged.
Qualify actual native BF16 autocast first; otherwise qualify native FP16 with a
supported native MPS GradScaler. Preserve FP32 model/master/moment tensors,
native clipping/AdamW and FP32 evaluation. Missing implementation is not backend
unsupported evidence. Numerical/scaler-skipped steps cannot become successful
logical updates. The one screen and independent reproductions have the same
frozen budget and selected native hyperparameters; no precision grid or sampler
precision port is authorized.

The canonical launcher ledger includes every invocation's active time through
result serialization and worker exit, as well as previously wasted work and
restart/import/loading costs. Preserve worker observations separately and enforce
final budget on actual canonical invocation completion. Resume explicitly selects
a receipted hash-validated committed checkpoint; numerical failure never triggers
retry-until-acceptable. Parent failure with an unprovable active termination time
is incomplete evidence, not invented elapsed telemetry. Extend the existing own-run
two-live-checkpoint/final-only policy across invocation directories and retain all
creation/pruning hashes. Never prune a user or different-run file.

Prospectively capture logical update1024 on every independent-development arm as
a descriptive equal-step secondary checkpoint, if reached. Its capture/I/O is
charged; evaluate it separately as scientific work after the trajectory. Retain
an explicit unavailable status if not reached. This secondary does not change H,
primary monitoring, hyperparameter selection or the common-budget endpoint guard.
Freeze and charge the original bounded synthetic moment-history/distribution-shift,
context128 cost and component diagnostics separately before executing them; they
cannot select a model, new sampler or protected threshold.

The inclusive current forecast is52+2*N full-budget equivalents, or about28.16
M3 Max device-hours at provisional minimum N=8, before deduplicating identical
controls. Add and disclose diagnostics, engineering, shared assets and separate
scientific evaluations. This forecast is not completed training or a fixed choice
of N; refresh actual time/storage with the independently planned N before protected
confirmation. Current local storage can cover the minimum-N endpoint/secondary
checkpoint forecast with the existing reserve. Check actual memory/storage before
each hardware stage, preserve native safeguards and stop for real resource failure.
No external compute purchase, arbitrary whole-stage timeout or outcome-based grid
expansion is authorized.

Before any native-precision outcome, apply the existing FP32 MPS replay tolerance
(rtol1e-4/atol1e-6) narrowly to replayed floating model/master tensors and native
Adam first/second(/maximum-second) moments. The conditioned CPU replay remains
exact. RNG/data/sampler bytes, metadata, optimizer step counters, native scaler
state, hyperparameters and logical/token counts remain exact on every backend.
Persist the comparison policy and maximum floating deviations. This prospectively
avoids an invented bitwise-MPS gate: the v7 native-only diagnostic already recorded
small independent-backward variation. It does not authorize loosening tolerances
after seeing an AMP failure or broadly comparing arbitrary checkpoint fields with
allclose. Retain every precision-mode attempt, including unsupported-mode evidence
preceding a later fatal numerical error. Known numerical failure takes precedence
over a subsequent signal or cleanup/I/O failure and forbids a numerical retry.
Validate sealed later-stage records before using their action labels to decide
whether they supersede qualification. A raw worker pass followed by a failed
process exit cannot become valid training evidence.

Clarify the original independent-development gate before outcomes: only primary
candidate/native pairs must have observed uncensored target times. Every required
secondary control must still complete with finite, budget-valid evidence. A valid
secondary control that misses the target retains its censoring and an explicit
`dominance_not_evaluable` result, without H substitution, a favorable subset
comparison or a claim of superiority over that control. Its target censoring
alone does not add a new veto on otherwise valid primary confirmation planning.
Missing, failed, numerical or budget-invalid controls still block. An alias of a
primary arm retains the primary uncensored requirement; uniform can replace
importance only after its own complete independent comparison passes. This follows
the original primary-pair gate and the prospectively declared adverse uniform
control; no training outcome has been used to change a protected threshold.
