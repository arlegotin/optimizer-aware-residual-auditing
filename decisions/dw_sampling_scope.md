<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Prospective scope: fixed-budget norm-importance dW sampling

Authorized by the owner's “FINAL CODING-AGENT INSTRUCTION: Bounded evaluation
of importance-sampled weight gradients with exact input adjoints”, received
2026-10-31. Recorded before new tests, device qualification or measurements.
Reviewed and initial actual HEAD: `9d8ad069c51b4d0e6c13e9c6e465579e9b44a339`
(`v0.1.0`); the checkout was clean on `main`. Initial inventory and hashes of
every historical tracked file are in
`analysis/dw_sampling/20261031T105213Z/initial_inspection.json`.

This is a separate prospective branch, not permission to resume online OAR.
Historical WP02 remains stopped, WP03 unfulfilled, WP04 inconclusive and
NumericalPolicy unresolved. Historical gates, amendments, evidence, configs,
index, report and restrictions will not be modified. The old measurements
retain their recorded source identities, including structural queries at
`2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07` and unremeasured maintenance at
`7628be7792030e3cfbb17dba82d3517642c866ca`.

## Question and authorized implementation

Test whether current-row sampling of selected FFN weight gradients can reduce
charged time to a fixed held-out target on the specified FP32 MPS byte language
model, with a common-budget endpoint quality guard. A blocked/negative result
with correct code and evidence completes this scope. No broader accelerator,
optimizer, model, memory-saving or mathematical acceptance claim is permitted.

Implement independently with native Torch. For flattened X,D, n>0,
G=D.T@X, a_i=norm(D_i)*norm(X_i), p_i=.9*a_i/sum(a)+.1/n. Draw k iid rows
with replacement, retaining duplicates and using the same indices in X and D.
Return sum(D_i.T X_i/(k p_i)); do not divide by n again. Production
k=max(1,ceil(rho*n)), rho in {1/8,1/4,1/2}, fixed across layers/steps.
k>=n dispatches directly to native F.linear outside custom autograd.
Forward, input and bias derivatives are exact. No parameters change before
the entire effective-batch backward completes. Native AdamW consumes actual
dense gradient tensors, including zeros; no preview, audit, rejection or replay.

Compute detached device FP32 scores after separately scaling X and D by their
maximum absolute values. Verify finite scales. Only a proved zero *unscaled*
entire X or D allows zero dW without sampling. Nonzero operands with zero,
invalid or nonfinite score sum abort. In particular reciprocal-scale rows
X=(1e30,1e-30), D=(1e-30,1e30) must abort, not return zero. Transfer only the
n-element scores to CPU FP64, normalize/validate the mixture there, use native
CPU torch.multinomial with the layer generator, and derive inverse-square-root
weights from the SAME FP64 probabilities. Transfer indices and cast weights;
gather, split-scale, and use a smaller dense GEMM. No full-n masked GEMM,
n-by-n sketch, per-row outer product stack, custom CDF or device kernel.
Uniform uses direct CPU randint and constant n/k, with no score work.

Support FP32 CPU/MPS and explicit CPU FP64 reference fixtures; reject AMP,
other dtypes, higher derivatives and sampled shared/tied weights. Each sampled
weight has one module owner; every unique optimizer parameter occurs once.
The native embedding/output tie remains exact. Accumulation weights each
microbatch loss by its valid-target fraction, with independent local draws.

Named seeds are SHA256 of ASCII compact JSON [base_seed,domain,*ids], first
eight bytes big endian modulo 2**63-1. Domains: init,data,sampling,order.
Each layer/sampler has a dedicated CPU generator; IDs block0.ffn_in/out through
block3. Exact and proved-zero paths consume no draws. Record seeds and states.

## Gates and resource boundaries

The driver exposes prepare, feasibility, correctness, cost, pilot, confirm,
summarize, with required --stage/--config/--out. Fresh attempt directories,
hash-validated prerequisites, no import-time execution, downloads or training.
Preparation and --help never download corpus bodies. Summaries work when
blocked/incomplete. A measured source/config change invalidates affected
downstream evidence; direct source hashes accompany Git identity. New stage
records are separate from the old schema/ledgers. Freeze an immutable dated
manifest before execution; commit the prospective specification if permitted.

A: use the existing pinned native arm64 Python3.13 environment, verify package
and dependency hashes, CPU suite and new-path historical report rendering.
Start fresh interpreters with fallback=0, fast-math=0, prefer-metal held fixed.
Doctor's active fallback check, nine existing MPS tests, full suite and actual
sampler primitive checks qualify hardware. Record OS/device/power/memory,
versions, flags, skips, unavailable telemetry and command exits. No genuine
MPS access => CPU correctness/dry runs only, BLOCKED_HARDWARE; no CPU cost claim.
Inspect documentary WikiText status immediately, without corpus bodies.

B: before trainer work, complete two-linear GELU residual-block steps including
root dX, bias, loss, clipping and native AdamW. Qualify default/single/foreach/
fused against multistep scalar native, select fastest with baseline-only blocks.
Shapes: (128,128,512), (2048,512,2048), (4096,512,2048), (8192,768,3072), plus
separate n2048 four-microbatch stride-2 control. Zero-dW returns dense zeros,
keeps exact dX/db and all optimizer work; it is a nonlearning diagnostic.
Native A/A freezes delta=max(.05,2*linear_quantile95(abs(TA/TB-1))) pooled
across both sessions, before sampler timing inspection. delta>.15 or unstable
warmup permits one replacement exclusive session; retained invalid evidence.
Require upper paired-bootstrap zero/native ratio <1-delta at a middle shape
in BOTH sessions; wide-only success cannot authorize model growth.

C: implement only sampled_linear and meaningful tests, even if B is hardware
blocked. Exact enumeration n,k=(2,1),(3,1),(3,2), actual gather/weight path,
expectation/variance/duplicates, exact dispatch, exact network adjoints,
accumulation, needs_input_grad, layouts, streams/restore, shared exclusion,
numerical extremes and the biased Adam displacement counterexample. Injected
identical samples must give identical multistep native AdamW with clipping,
groups, decay and nonzero moments. FP64 tolerance <=1e-10/1e-12; FP32
<=1e-4/1e-6 except prospectively justified scale-aware cancellation checks.
CPU probabilities: sum tolerance 5e-15, min >=(.1/n)*(1-1e-12).
FP32 cast weights: abs(k*p*w*w-1)<=8*FP32 eps. No flaky distribution gate.

D: dense/importance/uniform, all three rho values, all five shape/layout
fixtures. Charge every reduction/check/synchronization/transfer/gather,
allocation, optimizer and setup. Separate profiles cannot be summed into
observed step times. In each of two fresh exclusive sessions: warmup min8,
max32, last8 relative range<=.10; 30 randomized pairs; dense fixes block length
>=.2s capped100 updates. Restore common parameters, nonzero native moments,
data and named streams outside timing, recording reset costs. Record every
raw block/order and endpoint memory (RSS/Metal/driver separately). Ratio is
exp(mean(log(candidate/dense))); 10,000 paired percentile bootstrap resamples,
seed62001, linear percentiles. Require upper95%<1-delta in BOTH sessions at
n2048/4096. If only uniform passes, retain only that contingency. Neither
passes => STOP_COST, no trainer. Bounded offline variance/information work:
32 draws at native states0/32/128, charged separately, never runtime input.
No slow valid block may be discarded. No kernel/controller project.

Before each hardware stage bind an inclusive conservative forecast and a
deadline twice that forecast, including startup/warmup/reset/serialization.
Initial qualification forecast: 120s (deadline240s), <=256MB output. Later
forecasts must use measured conservative complete costs, not old timings.
No implicit external compute or storage purchase; unavailable resources stop.

E/F are conditional, not authorized through a failed A/B/D. WikiText must have
a new documentary/owner asset decision resolving the recorded contradictory
notices before any corpus body use. No alternative corpus. The current
restriction is retained, not interpreted away by this scope. If still blocked
after useful earlier work, BLOCKED_ASSET with no trainer/data expansion.

If reached, E is exactly the owner's byte LM: four pre-LN width512/head8/
FFN2048 blocks, context256, exact GELU, dropout0, biased QKV/attention/FFNs,
LN eps1e-5, tied exact 259-token output, N(0,.02^2) unique weights in declared
order. Explicit Boolean causal/key-valid SDPA mask; no embedding/residual
scales. Candidate B16; dense B4/8/16/32 without accumulation back to16.
Full-model native variants and two-session cost gate must pass again. Data
are pinned revision b08601e04326c79dfdd32d625aee71d232d685c3 with manifest
LFS hashes; pyarrow only if necessary. Preserve rows/line endings/articles,
single-level title regex, nonempty preambles; train-hash modulo10 development;
exact cross-split evaluation dedup; no near-duplicate selection. UTF-8 bytes,
BOS256/EOS257/PAD258; all targets including EOS once. Training chunks256;
evaluation stride128 with up to256 context, at most64 hash-selected monitors,
batch16, CPU FP64 accumulation of per-target losses. Official test sealed.

E uses native AdamW (.9,.999), eps1e-8, matrix decay.01, other decay0,
clip1 error_if_nonfinite. Canonical launch clock includes imports/setup/data/
checks/monitors/checkpoints/restarts. Eight-update synchronized blocks (or the
single prescribed all-arm fallback to1). H/Delta/R/T_budget/r_time frozen by
three baseline cost measurements and j=100,50,25 rule, with monitor overhead
<=2%, resolution delta<=.15, overshoot bound and >=8 periodic monitors.
Time-budget LR has run-specific preserved t0, 5% warmup then cosine to.1peak.
No catch-up monitors. Numerical evaluation epsilon from three identical-weight
repeats; >1e-3 stops. Tuning seed3101, LR1e-4/3e-4/1e-3, at most12 dense,
9 importance,9 uniform trials; independent development3102/3103/3104.
Freeze q_star at R0's first monitor at/after.6H, epsilon_q=min(log1.01,.1gain),
require meaningful baseline gain and two later crossings. Choose fastest
target-time eligible endpoint arms, with declared simplicity tie rules.
Uniform and smaller-dense competitors may defeat the candidate. One native
correct-autocast precision screen and three-seed reproduction of dominance;
no model.half shortcut. Moment-history shift/context128 are diagnostics only.

F freezes all choices/hashes before official-validation losses. Fixed N8..16
from the specified three-seed time/quality variability formula; invalid or
N>16 => INCONCLUSIVE_PRECISION. Seeds5001..5000+N, sequential randomized
AB/BA fresh processes, twice-consecutive observed monitor crossing, no
censored-time substitution. Paired seed-level log-time t CI and arithmetic
quality t CI (given df7..15 quantiles), 10,000 bootstrap sensitivity seed62002;
upper time<1-delta AND upper quality<=epsilon_q, disagreements inconclusive.
Endpoint/clock/reserve failures preclude a positive claim. Frozen final models
score test once after analysis freeze; material regression withholds broad
learning conclusion. Independent fresh-checkout/context first-three-pair
repeats are corroboration, not additional seeds. Maximum55+2N full-budget
equivalents (87 at16), deduplicate without spending saved slots. Bind total
hours/storage and twice-inclusive-stage deadline before any language trials.

Only synchronized committed checkpoints may resume: atomic integrity-checked
write, model/native optimizer/scheduler/counters/data cursor/shuffle/sampler/
model RNG/time/t0/source/config/assets/schema. Preserve last valid checkpoint
on errors; no partial optimizer checkpoint, resampling or retrying bad draws.
Test fixed-clock replay and separately charged real-clock restart if reached.

## Implementation plan and reviews

1. Preserve inventory; documentary asset query; scope/config/amendment freeze.
2. Test-first standalone sampler and protocol/prerequisite/immutability checks.
3. Prepare/A; CPU correctness may precede blocked B. Execute B and D only
   after qualified actual MPS, valid immutable prerequisites and resources.
4. Stop at failed prerequisite; E/F infrastructure is implemented only if
   reached. Summarize every attempt, hashes, costs, skips and precise stop.
5. Explicit mathematical/statistical, backend/optimizer/cost/recovery, and
   clean-process autonomous reproduction reviews; correct substantive defects
   and repeat only affected checks, retaining earlier failures.

Terminal states: STOP_COST, STOP_QUALITY, STOP_DOMINATED, BLOCKED_HARDWARE,
BLOCKED_ASSET, INCONCLUSIVE_MEASUREMENT, INCONCLUSIVE_PRECISION,
INCOMPLETE_RESOURCE, CONFIRMED_SCOPED_BENEFIT. No positive claim from a blocked
or synthetic-only branch. Final report includes actual commands/counts/device,
source/artifact hashes, omissions, adverse cases, review corrections and one
stop/continue recommendation.

Software is AGPL-3.0-only; original writing/results CC-BY-4.0. Established
column-row sampling: [Adelman et al., NeurIPS2021](https://arxiv.org/pdf/1805.08079).
[Wang et al., ICLR2024](https://arxiv.org/pdf/2402.17227) additionally adapts
ratios and samples activation gradients. This implementation is **fixed-budget
norm-importance dW sampling**, not a faithful VCAS reproduction; their CUDA
timings are not evidence here. No third-party sampler code is copied.
