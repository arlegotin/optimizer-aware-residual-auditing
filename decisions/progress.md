# OAR implementation progress

## 2026-10-30 — Task 1: configuration and evidence scaffolding

Implemented schema-v1 strict exploratory configuration, sorted SHA-256 hashing,
shared metadata/tensor transport records, append-only gate evidence with
fail-closed prerequisite checks, and a local package/empty command shell.
CPU configurations resolve to the `correctness_structural` scope. Confirmatory
configuration is blocked until its protocol essentials have an owning schema.

Validation used native arm64 CPython 3.13.5 in `.venv`, PyTorch 2.15.0,
NumPy 2.5.4, pytest 9.2.0, PyYAML 6.0.3 and psutil 7.2.2. Initial tests
failed on missing contracts; subsequent regression cycles covered detached
audit tensors, transitive stopped prerequisites, immutable replay descriptors,
and exact integer resource counts. Final `python -m pytest`: **41 passed in
0.64s**. CLI help exits successfully and advertises no unimplemented commands;
`pip check` reports no broken requirements. Local disk footprint at validation:
802 MiB for `.venv`, 151 MiB for the disposable pip cache. No experiment
runtime, memory forecast, training outcome or speedup was measured.

The network-restricted dependency download was retried with the authorized
tool escalation and succeeded. Source notices and patches remain accessible;
environment/cache/checkpoint artifacts and plan scratch are ignored.

**Scientific status:** WP00 remains unresolved. Scaffolding tests do not satisfy
environment provenance, wheel hashes, asset identities or MPS capability gates.
There is no scientific pass record in `decisions/gates.jsonl`. Next permitted
action is Task 2 discovery: freeze wheel/source identities, inspect native
CPU/MPS capability and record actual WP00 evidence. Later owners must explicitly
extend configuration fields and gate permissions for any evidence-qualified
structural-oracle branch; the initial gate graph is deliberately conservative.

Detailed local implementation/TDD evidence is in the ignored task report:
`.superpowers/sdd/2026-10-30-oar-implementation/task-1-report.md`.

## 2026-10-30 — Task 1 review completion

Task 1 implementation `45a229e` and correction `2115d04` have been independently
reviewed. The initial nonfinite gate-evidence issue was fixed and re-reviewed
clean; the reviewed baseline has **45 passing tests**. No scientific gate was
passed by those scaffolding tests.

## 2026-10-30 — Task 2: actual hardware, operations and provenance (WP00)

Implemented allowlisted hardware redaction, tiny capability checks, `doctor`, a
separate strict discovery configuration and content-bound local asset/notice
inspection. The shell-output fixture exercises native parsing while actual CPU
and MPS checks exercise numerical outputs, optimizer mutation, independent
CPU-generated Gaussian streams and model-RNG save/restore. Changed asset or
notice bytes invalidate supplied approval; missing, conflicting or unresolved
license evidence remains blocked.

Actual discovery was rerun outside the sandbox with
`PYTORCH_ENABLE_MPS_FALLBACK=0`. Sandbox MPS unavailability and denied `sysctl`
were process restrictions: the actual machine is native arm64, Apple M3 Max,
Mac15,11, macOS 26.7.1, with **38,654,705,664 physical bytes** and about **75.9 GB
free disk** at discovery, on AC power (`pmset` mode code 0; the separate legacy
low-power flag was unavailable). Tiny FP32 GEMM/noncontiguous input, exact native
linear derivatives, AdamW, layer norm/GELU/masked-loss and attention backward,
CPU probe transfer and model-RNG replay all passed. Tiny native QR/SVD and
explicit CPU FP64 QR/SVD also passed. Combined MPS-to-CPU FP64 conversion raised
`RuntimeError: Undefined type Double`; discovery now transfers to CPU first and
then casts. Sampling initially changed the MPS model RNG; a real-device
regression verifies that the entire suite now restores it.

Froze **19 actual runtime/tooling wheels (158,389,709 bytes)** with SHA-256 and
hash-enforced requirements. Eighteen archives came from the existing pip cache;
only the previously uncached pip 25.1.1 wheel needed a small download. Archived
package/bundled notices and selected upstream source/notices are under
`third_party/discovery/`. Whole-repository revisions pin INSTANT, VCAS, DropBP,
Fashion-MNIST and WikiText; historical excerpt blob IDs are not used as commit
pins. The exact installed PyTorch revision's fallback implementation confirms
that an absent variable or `0` disables general fallback. Its unconditional
`embedding_renorm_` CPU fallback is outside this critical-operation suite and
must not enter a timed critical path without explicit accounting.

**Gate interpretation:** WP00 passes for CPU correctness and the next operator
qualification/headroom work on actual MPS. This rests on actual operations,
frozen package/source identities and at least one available, permissible core
dataset: Fashion-MNIST has consistent MIT evidence at its pinned official
revision and both pinned training-file HEAD requests returned 200 with their
expected sizes. Dataset bodies have not been downloaded; their local SHA-256
fields remain null, upstream checksums are separately labeled and
`workload_ready` remains false pending byte verification/preprocessing.
WikiText's pinned official card still conflicts (CC BY-SA 3.0/GFDL metadata
versus CC BY-SA 4.0 prose); **the language workload remains blocked** until that
evidence is reconciled. Vision cannot substitute for a language-pretraining
claim. Official validation/test bodies were not accessed.

Final verification: **73 passed in 1.51s** outside the sandbox with actual MPS
and fallback disabled; the sandbox run had **72 passed, 1 skipped**. The
hash-enforced offline lock dry run resolved all 19 wheels, `pip check` found no
broken requirements, and all gate/source/wheel/notice hashes and environment
redaction checks passed. Exact reproduction evidence is recorded in
`.superpowers/sdd/2026-10-30-oar-implementation/task-2-report.md`. Discovery is
limited to tiny shapes; no custom OAR operator, checkpoint/replay equivalence,
shape-specific speed, thermal budget, training, learning or speedup claim has
been established. WP01 numerical/transaction verification is the next action;
WP02 headroom still depends on WP01. `doctor` reports actual capability scope
and never writes a gate pass itself.

### Task 2 review correction — fix round 1

CLI discovery configuration errors now return status 2 with a concise diagnostic
and no traceback. Effective PyTorch fallback policy is registered at import;
changing the environment later cannot prove that policy. MPS checks therefore
run in a fresh worker whose environment is set to `0` before interpreter startup
and whose pre-import state is verified. GPU eligibility requires worker policy
evidence. The manifest marks caller environment text as non-authoritative and
leaves effective fallback unknown when it has not been verified.

The import-order regression first imports Torch with fallback 1, changes the
variable to 0, and forbids MPS RNG access in that process; the real worker then
executes independently. Verification: **76 passed in 5.93s** on actual MPS,
**75 passed, 1 skipped in 3.01s** in the sandbox. A module CLI smoke test confirmed
exit 2 for fallback-enabled configuration. Actual `doctor` again classified
`exploratory_gpu`, with 11 CPU and 13 isolated MPS-context rows passing; the
fresh-process overhead is included in discovery wall time. Refreshed the
environment manifest and appended WP00 evidence with updated hashes. Dataset
limits and the WikiText block remain unchanged.

## 2026-10-30 — Task 2 review completion

Task 2 implementation `36b7dcb` and correction `d3f8fcb` have completed
independent review. The correction's actual-device baseline is **76 passing
tests**. WP00 remains qualified only for its documented tiny operator and
next-stage discovery scope; WikiText is still blocked and dataset bodies remain
unverified. Task 3 starts from the reviewed correction.

## 2026-10-30 — Task 3: projected derivatives and past-only construction

Implemented `projected_linear` with native forward, exact input/bias adjoints,
projected dense weight gradients and detached streaming `H`/`R`. One
`LinearCapture` spans an effective batch, with fixed basis/probe storage and
parameter identity/version checks. Higher-order differentiation fails before
capture mutation. No token normalization is applied inside the operator, and no
activation/adjoint archive is retained after ordinary backward releases its
saved tensors.

Implemented ordered randomized QR/small-SVD construction from a supplied exact
pre-update gradient, using an independent CPU generator and explicit CPU FP64
decomposition. Numerical null directions are dropped using the source
precision. `BasisState.for_step` and `LinearCapture.from_basis` require a source
step strictly before the consuming step. Source and returned-vector transfers,
MPS synchronization and inclusive construction time are recorded. The trainer
will own refresh-gradient lifetime and include complete construction time in
active time; the basis state retains no gradient archive.

TDD began with four expected missing-module errors. A subsequent red/green
regression found and fixed the missing parameter-freeze check between
microbatches. Tests compare against independently materialized native CPU FP64
autograd, including rectangular/noncontiguous inputs, rank zero/full/deficient
bases, zero adjoints, bias/no-bias cases, four residual blocks, an injected wrong
input derivative, higher-order rejection and seven unequal masked microbatches.
The MPS marker now selects both new device checks and the existing discovery
device check: **3 selected, 125 deselected**.

Final full verification outside the sandbox with
`PYTORCH_ENABLE_MPS_FALLBACK=0`: **128 passed in 6.06s** on actual MPS/CPU.
Committed [operator correctness evidence](../analysis/operator_correctness.json)
contains 12 CPU/MPS cases, six deficient-basis constructions, accumulation and
deep-adjoint errors, seeds, source hashes, shapes and tolerances. Maximum
absolute operator error over those fixtures was **9.04e-15 CPU FP64** and
**2.39e-6 MPS FP32**. These are tiny correctness results; construction times are
inclusive diagnostics, with no performance or speedup claim and no calibrated
production audit guardband. Detailed commands, TDD evidence, interfaces and
self-review are in `.superpowers/sdd/2026-10-30-oar-implementation/task-3-report.md`.

**Scientific status:** Task 3 is complete; WP01 remains open for Tasks 4 and 5
(confidence/audit allocation and functional optimizer preview). No WP01 pass or
dependent experiment authorization was written. Existing dataset and workload
blocks remain in force. Standard `.DS_Store` exclusion was added while
preserving the local Finder file's bytes.

## 2026-10-30 — Task 4: independent Gaussian audits and durable spending

Implemented CPU FP64 log-domain confidence roots, scaled residual reductions and
margin decisions that preserve the raw numerical allowance in inner-product
units. Invalid/unrepresentable roots, malformed sketches, nonfinite totals and
unresolved/mismatched calibration fail closed. `NumericalPolicy` remains
**unresolved by default**: tiny Task 3 errors do not determine a shape-specific
allowance, and no production coefficient was invented.

`AuditJournal` durably appends and fsyncs per-layer allocations before exposing
reservations/probes, persists checkpoint floors, takes the monotone maximum on
resume and never refunds rejected, zero or interrupted tests. Locked writers,
strict versioned readers and a SHA-256 record chain reject malformed/truncated
evidence. Reopened instances cannot draw old reservations; retry requires a new
allocation. Resume callers use `require_existing=True` even at checkpoint zero;
a missing ledger cannot be recreated from a stale checkpoint. Positive
checkpoint counters always require an existing journal. Candidate metadata checks enforce past-only construction, no use of
any current-step probes, frozen rank/version and next-step-only changes. Real
online lifecycle integration remains Task 12.

Actual Torch 2.15 CPU normal generators repeated probes for seeds separated by
`2**32`. The audit therefore uses the full persisted seed with a versioned
**NumPy PCG64** stream, domain-separated by the durable run identity and audit
label, and checks the NumPy version on resume. FP32/FP64 arrays are created on
CPU before explicit device transfer; model RNG streams remain untouched.

Verification included observed TDD failures for missing modules, subnormal
allocation rounding, invalid calibration contexts, high-bit seed reuse and
cross-layer current-probe dependence. A real subprocess exits during reservation
fsync; recovery consumes its complete appended test. The final fresh-process
full suite with `PYTORCH_ENABLE_MPS_FALLBACK=0` passed **211 tests in 6.03s** on
actual CPU/MPS. [Numerical evidence](../analysis/audit_confidence.json) records
scalar residuals, Task 3 error envelopes, high-bit seed behavior and 20,000
replicates per rank-one/flat/decaying spectrum (`q=8`, `delta=0.1`, seed 8401),
with failure fractions 0.02425 / 0 / 0.0024 and Wilson intervals. Those finite
simulations make no tiny-probability certification claim.

A cancellation stress case on both CPU and MPS produced a zero FP32 residual
while the CPU FP64 reference on stored inputs was nonzero, including absolute
errors up to `1e8` under the tested scaling. This is diagnostic evidence for
keeping calibration unresolved, not a fitted universal allowance. The journal
assumes a durable local POSIX filesystem and independent preservation from
training rollback; it detects partial/edited records, not malicious whole-file
rollback. Full chain validation is currently linear in ledger size and must be
charged in later timing.

**Scientific status:** Task 4 implementation is complete; **WP01 remains open
for Task 5** (functional optimizer preview/commit). Shape-specific numerical
calibration remains Tasks 6/14. No WP01 pass, training, performance or online
learning claim was written. Detailed TDD commands/output, interfaces and
self-review: `.superpowers/sdd/2026-10-30-oar-implementation/task-4-report.md`.

## 2026-10-30 — Task 5: native-equivalent transactional AdamW

Implemented eager, scalar-hyperparameter, non-AMSGrad/non-differentiable,
non-fused/non-foreach AdamW preview for dense FP32/FP64 tensors. Detached copies
receive native L2 global clipping, including its `1e-6` stabilizer. The native
operation order applies decoupled decay before the adaptive update. Missing and
`None` gradients skip decay, initialization and counters; zero gradients
participate. `AdamWParamState.step` is the committed pre-preview count; pending
state contains the native updated count, including native default scalar-counter
rounding at the FP32 `2**24` boundary. Custom persisted counter dtypes are outside
the integer-only transport contract.

`PendingUpdate` owns candidate parameters, actual stored `theta - candidate`
displacements and pending moments. `raw_alignment` reduces original unclipped
gradients against those displacements explicitly on CPU FP64; transfers and
synchronization belong in active time. Preview preserves live parameters,
`.grad`, raw gradient tensors, moments, counters and RNG streams. Commit checks
both logical versions, exact source ownership and ordinary tensor mutation
versions before any writes; it consumes the update once and transfers pending
moment ownership to live state. A transported/checkpoint-decoded pending record
has no process-local authority and must be previewed again. Callers must serialize
commits, advance returned versions and avoid `.data`/raw-storage mutation;
concurrent mutation or device failure during writes is not rollback-protected.
Aliased parameter storage is rejected in favor of one canonical parameter.

TDD observed the expected missing-module error, then independent regressions for
native counter precision and ambiguous aliased parameter storage. Tests compare
four-step trajectories with independent `torch.optim.AdamW` instances, mixed
groups, missing/initialized moments, epsilon, decay, clipping, zero/None values,
version failures, foreign owners, modified sources/candidates and double commits.
The blueprint Appendix's bias-correction `t=100` is an updated count, so native
pre-step state 99 reproduces its displacement; committed count 100 correctly
advances to 101. Both conventions are tested without altering optimizer rules.

The original Adam counterexample gives cosine **0.999200319872**, true alignment
**-0.018967380478**, and approximate alignment **0.019584821032**. An actual
orthogonal projection onto normalized `(1,-.02)` gives the same high cosine,
true alignment **-0.018967251241** and approximate alignment **0.019568536946**.
These constructed states refute raw-cosine descent inference without claiming
frequency in training.

Final fresh-process verification with `PYTORCH_ENABLE_MPS_FALLBACK=0` passed
**254 tests in 6.92s** on actual CPU/MPS; the focused actual-device run passed
**39 tests in 1.88s** before the final four counter/alias/missing-gradient cases
were added and included in the full run. The reproducible
[collector](../analysis/collect_optimizer_correctness.py) command is
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python analysis/collect_optimizer_correctness.py`.
Its [numerical evidence](../analysis/optimizer_correctness.json) retains 18
four-step CPU FP64/CPU FP32/MPS FP32 reference cases with zero observed maximum
absolute discrepancy in parameters, moments, displacements and raw alignment.
Declared tolerances are FP64 `rtol=1e-10, atol=1e-12` and FP32
`rtol=2e-6, atol=2e-7`; these tiny checks do not calibrate production audit error.

A full-participation FP32 tensor payload count measures **16P bytes of pending
storage** above the dense **16P**, hence **32P retained bytes**, exceeding the
blueprint's approximate 28P because both candidates and actual displacements
are retained. This excludes activations, clipping copies, denominators, CPU FP64
reduction temporaries and other workspaces, and is not a measured peak/RSS or
performance result. No lazy storage optimization was introduced.

**Scientific status:** WP01 passes for the tested Tasks 3–5 numerical,
mathematical and optimizer-transaction scope, with hashes in the append-only
gate ledger. This permits the next WP02 headroom work. Shape-specific numerical
policy remains unresolved; online acceptance, full trainer/replay/checkpoint
integration, training, learning and speedup claims remain unqualified. Existing
asset/license blocks still apply. Detailed exact commands, red/green evidence,
interfaces and self-review are in
`.superpowers/sdd/2026-10-30-oar-implementation/task-5-report.md`.


## 2026-10-30 — Task 6: full-cost headroom and numerical diagnostics (WP02)

Added inclusive `ActiveClock`, synchronized raw blocks, adaptive warmup/noise,
randomized shared-round paired comparisons, strict benchmark configuration and
configuration hashing, `CostTable` and `bench --config`. One fresh fallback0
worker handles each complete study. Input guards reject duplicate tuple keys
before a hardware worker can silently replace accumulation/layout results.
Native single/foreach/fused/default variants were numerically checked and timed
in a common randomized window; every paired control uses the selected native
variant and complete forward/exact-input-backward/weight-backward/clipping/
optimizer boundary. Fine component and resource profiles are separate from
headline timing. Full attempts include reconstruction, probes, generation,
fsynced spending, preview, full CPU FP64 raw alignment, explicit complement,
decision and commit pricing; rejection and exact replay are measured directly.
Refresh includes explicit CPU FP64 QR/SVD and both transfers.

Actual exclusive MPS studies on the Apple M3 Max found stable candidate-only
headroom: at n2048/di512/do2048/r16/q8, candidate6.040ms versus native7.127ms,
paired95% saving[.868,1.328]ms; at n4096,9.941ms versus12.459ms,
[2.023,2.931]ms. A safely screened wider kernel (one block, not a larger model)
at n8192/di768/do3072/r16/q16 gave28.315ms versus37.364ms,
[8.384,9.792]ms. Complete p0 attempts cost26.557/33.390/55.251ms,
respectively, versus their paired native controls7.112/12.151/36.709ms.
The wide forced-reject total is90.965ms and direct exact replay36.973ms.
All complete paths are slower even before refresh. Local break-even rejection
probabilities are outside[0,1]; wide illustrative K16 p*=-.521,
bootstrap95%[-.567,-.473]. K/p forecasts are arithmetic sensitivities,
not selected policies or observed full-cycle training throughput.

The small n128 candidate is slower; n512/noncontiguous is inconclusive.
The four-microbatch/r8 positive interval fails dense warmup/noise qualification
and is excluded from qualified candidate regimes. Raw timings, all failures,
uncertainty, warmup limits, selected native variants, comparison order and ledger
ages are in [main](../analysis/timing_calibration.json) and
[wide](../analysis/timing_calibration_wide.json) artifacts; the
[combined CSV](../analysis/headroom_summary.csv) states boundaries and native
variants. Final main/wide research costs were32.056s/12.740s before final JSON
serialization, including initialization, warmup, profiling and refresh.

Named reference costs preserve the distinction from unavoidable audit work.
At n2048/r16, preview alone profiles14.223ms including4.857ms input validation;
one expansion raw-alignment transfer/reduction costs1.165ms. Journal/probe/
capture cost grows3.347 ->9.110 ->25.919ms at0/64/256 prior reservations,
from repeated full-chain reparsing. These profiles are never summed into a
headline step. No journal/optimizer semantics were weakened or speculative
online redesign added. The reference retains32P bytes before temporaries,
not the blueprint's rough28P. Wider-kernel retained32P is151,117,824 bytes.

Physical memory38,654,705,664 bytes, recommended Metal30,150,672,384,
blueprint engineering screen24,120,537,907. Wider allocation screening
1,712,013,312 bytes passed against current available7,978,123,264.
Observed study profile peaks include505,757,696 RSS and2,168,143,872 Metal
driver bytes, with five fixtures resident; overlapping RSS/Metal readings are
not added or presented as one-model training/checkpoint peaks. Pre-existing
swap, OS-wide swap activity, unknown thermal limits and post-study pressure
queries remain explicit in raw metadata and
[resource context](../analysis/resource_context.json). No apps/OS settings or
allocation safeguards were changed. Block intervals qualify only these studies
under their recorded system conditions, not independent training runs.

[Shape/scale diagnostics](../analysis/guardband_calibration.json) record120
actual MPS FP32 versus CPU FP64 cases over both matrix directions, main shapes,
three scales, generic/near-subspace spectra and development/held-out seeds.
Maximum numerical margin discrepancy is1.773 development and1.910 held-out;
no validated allowance formula follows, and wide has timing evidence only.
`NumericalPolicy` stays unresolved. Conditional commits price p=0 by deliberately
bypassing its rejection and are not online acceptance or learning evidence.
Task5 extreme-derived-overflow decisions remain fail-closed.

Committed durable Task3/4 collectors and actually regenerated their bounded
numerical evidence before using refreshed hashes. Original operator/confidence
results reproduce; Task5 collector/artifact are reused unchanged. Runnable
commands and collector/artifact SHA256 values are in the
[collector manifest](../analysis/collector_manifest.json) and
[reproduction guide](../analysis/README.md). Normalized upstream
`repository_tree_sha` fields that actually held commit IDs to
`repository_commit_sha`, preserving archived source/license bytes. The
[historical index](../analysis/evidence_history.json) keeps exact previous gate
artifacts, a contaminated overlapping pilot, and a clean earlier sequential
pilot whose control drift/replay-subtraction limits prompted interleaving.
The initial serialization-failed pilot has console diagnostics only and is
excluded. Final measured-source bytes are archived solely as execution evidence;
the later four-line pre-worker duplicate-key guard changed no timed kernel.

Final verification:26 focused tests passed in1.77s; actual fallback0 CPU/MPS
full suite **280 passed in6.84s**. No tests/collectors/profilers overlapped final
headline timing. All requested red/green commands, pilot failures, raw resource
facts and interfaces are in
`.superpowers/sdd/2026-10-30-oar-implementation/task-6-report.md`.

**Scientific status:** WP02 stops the current online branch because complete
attempts have no measured headroom, while stable candidate-only headroom and
identified reference overhead preserve the scientific opportunity. Next action
is preparing a bounded synthetic structural-oracle information study and an
explicit narrow branch/protocol amendment before execution; normal WP03/WP04
prerequisites remain blocked. No full trainer/controller, data ingestion,
online acceptance, larger-model/MLX expansion, production guardband or measured
learning claim is authorized. Dataset bodies remain unverified; WikiText remains
blocked. Historical gate records are preserved and refreshed evidence appends.


### Task 6 final evidence-layout correction

Finalization found that the numerical collector's configured noncontiguous flag
had been carried into metadata while its actual inputs were contiguous. The
same120-case study was regenerated with slicing after MPS transfer, explicit
observed stride fields and assertions:24 actual stride-2 cases now match their
configured flags. The numerical maxima and unresolved-policy conclusion remain
unchanged. Final collector cost9.575s. Exact superseded artifact/index/source
bytes are archived; refreshed gate records append without rewriting history.

The benchmark already sliced device storage after transfer. Extracted that
identical initialization expression into a directly tested helper; actual CPU
and MPS checks of the named n512/di256 case passed with stride(512,2), leaf
input gradients and exact stored values. The final measured-source delta is
pre-worker input guards plus this behavior-equivalent initialization extraction;
timed method operations and cost artifacts are unchanged. The source snapshot
is archival evidence only, not a second maintained runtime.

Final actual-device verification after these layout checks:
**282 passed in7.74s**, no failures/skips; the named layout check separately
passed2 tests in1.90s. The original280-test result above is superseded. WP02
still stops online engineering and permits only bounded-oracle preparation
with an explicit narrow branch/protocol amendment before execution.


### Task 6 review fix1 — exact mixture threshold and separate RNG streams

Corrected scheduled-mixture break-even to `(D-A-C/(K-1))/R` for K>1;
K1 explicitly has no approximate segment or rejection threshold. Bootstrap K16
uses15 approximate steps. Hand-derived equality and constant-bootstrap tests
reproduce and cover the arithmetic error.

Worker requests now preserve resolved construction/audit/data/model seeds.
Independent data/model generators make initial weights invariant to input batch
size/layout and data-seed consumption; construction and audit retain their own
resolved streams. Metadata records61001/61002/61003/61004 and the separation
policy. Prior timing/derived/source bytes are archived unchanged and explicitly
labeled coupled initialization; they were not retroactively relabeled.

Reran only the same bounded main/wide configurations exclusively and serially.
Qualified n2048/r16 candidate5.314ms vs6.285ms native (saving95%[.648,1.299]ms),
n4096 candidate8.138ms vs10.239ms ([1.772,2.426]ms), and wide candidate28.875ms
vs36.882ms ([7.395,8.605]ms). Full attempts24.965/27.854/54.406ms remain slower
than paired native6.087/10.320/36.738ms. Wide forced-reject total91.401ms,
direct replay37.366ms. Corrected wide K16 p*=-.492802,
bootstrap95%[-.535047,-.447323], outside feasible[0,1]. Raw current files and
regenerated summaries retain uncertainty, seed policy and resource facts.
No new shapes/policy/optimization/calibration study was added.

TDD red5 failed,15 passed,1 skipped; actual-device covering tests21 passed
in1.51s. Final fallback0 actual CPU/MPS suite **286 passed in6.08s**.
WP02 stop/oracle-preparation outcome is unchanged; NumericalPolicy remains
unresolved and no online/training/learning claim follows. Exact commands,
current data and historical limitations append to the same Task6 report.

## 2026-10-31 — Task 11 bounded structural-oracle preparation

Added strict synthetic exploratory config, hashed narrow scope amendment and
oracle-only permission check bound to genuine WP02 stop/candidate-potential
and WP01 correctness evidence. Ordinary WP03/WP04 stay blocked. Durable
`audit-checkpoints --config` uses the measured residual block and copied native
AdamW history; no general trainer/controller or corpus use. Fourteen new oracle
correctness tests include independent original-gradient/native trajectory
references, leakage/RNG isolation, guard/tamper and unresolved rare-event claims.
Initial actual fallback0 CPU/MPS full suite: **300 passed in6.93s**. Next permitted
action is the bounded development-only learning-rate pilot before query study;
current full online cost stays negative and NumericalPolicy unresolved.

### Task 11 pilot and pre-query lock

Committed first code/scope as a100d5c before real pilot. Fresh MPS pilot26.046s:
full-rank seed1101 MSE.870242 ->.108646 after128 updates at selected.003;
all four rates/curves retained in `analysis/structural_oracle_pilot.json`.
Boundary/untuned limitation explicit. Corrected root dX to match fixture;
pilot learning valid but wall is not an equivalent-boundary headline. Locked
study config/second hash amendment before query outcomes. Shift moved64 ->120
without adding cases so q128 age16 straddles shift; native moments continuous.
First-query measured forecast controls conservative600s engineering ceiling;
partial coverage becomes incomplete/inconclusive. Actual fallback0 full suite:
**303 passed in6.85s**. No online gate reopens; next is bounded query collection.

### Task 11 bounded result — online stop retained

The committed study source2e89e74 ran fresh/exclusive actual MPS fallback0 and
completed all792 layer rows,18 checkpoints and six learnable synthetic histories
in60.959s on its documented worker boundary (startup/final summary tail outside;
not an invocation throughput benchmark). First-checkpoint forecast95.30s fit the
conservative600s ceiling.30 curve points and every attempt retained; checkpoint
index binds~485MB of ignored regenerable tensors. All loss units MSE/output
element, all counts synthetic observations; no language or real-task claim.

Past proposals carry some signal, but all144 evaluation proposals lose
independent-development progress relative to native dense. Exact ordinary and
complement bounds accept0/144 at *every* tau.25/.5/.75, independent of fitting.
Complement median B/Bordinary=.854 but minimum B/a=.991, still outside grid.
Nontrivial correlated score AUCs remain descriptive (same-batch.8805 complement
vs.8750 ordinary); no claim that geometric information is absent. Paid exact
current-SVD/r64 has just two harmless tau.75 accepts, both still worse than
dense. Fitted complement/ordinary/relative reject all, matching a constant exact
fallback; fixed rank64/age16 accepts12 and is slightly worse. Full unfitted
acceptance envelope and condition/seed/stage/rank/age strata preserved.
Lost-progress AUC unresolved because every label is positive. Dense itself harms
stage8 queries; early full/shift stages duplicate history, so proposal counts
are not independent runs or prevalence estimates.

No ordinary WP03/WP04 pass, controller/trainer/data ingestion, NumericalPolicy
resolution or online reopening. WP02 stop unchanged; broad WP04 inconclusive.
Next permitted action after Task11 review: **WP09 synthesis**. No new cost remedy,
shape/grid/optimization or collection authorized by these synthetic results.
Full protocol/TDD/provenance/timing/interpretation/self-review is in
[durable Task11 report](reports/structural_oracle.md);
durable public and postprocessor commands are in `analysis/README.md`.

Final Task11 verification: actual fallback0 CPU/MPS full suite **305 passed
in6.71s**, no failures/skips;19 bounded oracle/derived-analysis tests.
Independent raw-row rank-sum AUC checks reproduce stored pairwise AUCs;
all18 raw-checkpoint byte hashes and792 finite unit/temporal-labeled rows pass
independent checks. Final derived-only postprocessor invocation.0584s.
WP04 append is inconclusive for broad unmet requirements; WP02 remains stop.

Task11 final control refinement: flat and rotating constructed SGD fixtures now
use genuinely flat and60-degree-rotated spectra, with hand-derived energy/
residual checks that fail against prior fixtures. No empirical results or study
code changed. Latest full actual fallback0 suite **305 passed in7.24s**, zero
failures/skips; refreshed WP04 inconclusive hashes append. Results commitdc56170
remains the separate study artifact commit; WP02 stop and WP09 next action hold.

### Task11 review fix1 — partial failure evidence and durable report

Registered native condition/seed/LR attempts before bootstrap and retained
partial curves/checkpoint paths/hashes, last confirmed update/source ordinal,
last selected checkpoint, failure reason/status and elapsed native cost when
budget or numerical errors occur inside bootstrap. Tiny CPU regressions perform
three real updates before each failure; saved checkpoint2/moment counters remain
verifiable alongside confirmed update3, without claiming an unsaved resume state.
Full actual fallback0 CPU/MPS suite: **307 passed in7.84s**, no failures/skips.
No research rerun; measured source remains2e89e74 and original artifacts unchanged.
Complete durable [Task11 report](reports/structural_oracle.md) now owns full
protocol/TDD/provenance/timing/results and fix evidence. Scratch report untracked
with its ignored working copy preserved; historical Git/raw evidence retained.
Fresh WP04 hashes append for durable report and later fix code/tests. WP04 stays
inconclusive, WP02 online stop unchanged, next WP09 synthesis after review.
