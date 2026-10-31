<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Fixed-budget norm-importance dW sampling: bounded evaluation

**Terminal state: INCONCLUSIVE_MEASUREMENT. Recommendation: stop this run.**
The native timing pilots did not stabilize at the adverse tiny control in the
initial process or the one permitted fresh replacement. No paired feasibility
comparison qualified. The protocol therefore stopped before sampler cost,
language-model infrastructure, tuning, or confirmation. This is neither a
cost defeat of sampling nor evidence for a training speedup. WikiText use also
remains independently **BLOCKED_ASSET**.

## Revisions, authorization, and preservation

The reviewed release is `9d8ad069c51b4d0e6c13e9c6e465579e9b44a339`, the peeled
`v0.1.0` commit. Initial checkout: clean `main`, full non-shallow history.
Work proceeded on local `research/dw-sampling`; nothing was pushed or published.
The owner's bounded instruction is recorded in the new
[prospective scope](../../../decisions/dw_sampling_scope.md). Local commits:

- `afa51cb`: prospective specification, standalone sampler, strict configuration,
  separate driver/records, tests and initial inventory, committed before the
  staged experiment.
- `3500387`: explicit score-transfer/CPU-FP64 and inverse-weight-cast/device
  boundaries, with a reproduced MPS failure and regression. Stage A passed and
  the attempted B sessions used this source version.
- `cf6481741933130f2e86b440e750909686b81eb3`: current implemented source. Corrects
  only qualification of native fused AdamW scalar step counters and adds its
  regression. It was made after B exposure; **no performance session was
  reopened**. The native default/single/foreach warmup observations retain
  their v2 source identity. No new confirmation exists.

All **184 historical tracked files** retain their initial bytes; the diff of
those paths against the reviewed commit is empty. See
[initial inventory](initial_inspection.json) and
[preservation verification](historical_preservation.json). Old gates/amendments,
NumericalPolicy, configs, raw evidence, index and report were not edited. The
stopped online OAR branch remains stopped. Historical Apple timings are not
measurements of the new sampler; structural queries remain attributed to
`2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07`, with later maintenance unremeasured.

## Implementation and mathematical boundary

[src/oar/sampled_linear.py](../../../src/oar/sampled_linear.py) independently
implements fixed-budget norm-importance dW sampling and direct-randint uniform
sampling with native Torch. Forward is native `F.linear`; the custom backward
computes exact raw input and bias derivatives and samples only dense dW.
Flattened row count includes all physical rows. Loss weighting is already in
D; there is no extra token division. k is ceil(rho*n), at least one; k>=n
uses native dispatch outside custom autograd and consumes no sampling draws.

Importance scores use detached, separately max-scaled operands and FP32 device
norms. Only the score vector transfers to CPU, then casts to FP64. The fixed
mixture p=.9*a/sum(a)+.1/n is normalized/validated on CPU. Native CPU
multinomial draws with replacement; the same CPU probabilities determine
inverse-square-root weights, cast on CPU before transfer. Gathered operands
share indices and preserve duplicates. Uniform uses CPU randint and one n/k
scaling on the narrower gathered operand, without score formation. Every layer
has an isolated named CPU generator, with checkpointable stream state.

For A_i=outer(D_i,X_i), G=sum A_i and iid draws, ideal conditional expectation
is G and Frobenius variance is `(sum_i ||A_i||_F^2/p_i-||G||_F^2)/k`.
Unregularized norm probabilities minimize this variance in the iid estimator
family; the fixed mixture gives support and robustness while giving up that
optimum. With uniform p the numerator is `n*sum_i ||A_i||_F^2-||G||_F^2`.
FP32 score, scaling and GEMM rounding do not constitute exact arithmetic or a
formal unbiased-machine-arithmetic guarantee. Bias and unsampled gradients
remain exact **before global clipping**; random clipping factors and native
AdamW can change their eventual displacement. Unbiased raw dW does not imply
unbiased Adam moments, descent, generalization, or dense-trajectory equivalence.

The cost question remains unmeasured: score reductions/checks, CPU
normalization/draws, transfers, gathers/scaling, smaller GEMM and sampler setup
must together beat replaceable dense dW while retaining exact dX/db, clipping
and full native optimizer work. No activation-memory saving is claimed.
The zero-dW arm is explicitly nonlearning and retains dense zero gradients,
exact dX/db, moments and decay; it is only an optimistic headroom diagnostic,
not a universal hardware lower bound. It was not reached in qualified pairs.

Reference tests enumerate all ordered draws for (2,1),(3,1),(3,2), including
duplicates, using independent FP64 probabilities and the actual gather path.
They cover exact dispatch/network adjoints, unequal valid-token accumulation,
independent partitions, noncontiguous/leading axes, frozen operands, optional
bias, stream isolation/restore, ownership/optimizer uniqueness, higher-order
and AMP rejection, nonfinite/overflow conditions, native AdamW with identical
injected gradients, groups/clipping/nonzero moments, zero-versus-None, and the
unbiased-gradient/biased-Adam counterexample. No flaky frequency gate was used.
The reciprocal-scale X=(1e30,1e-30), D=(1e-30,1e30) case aborts as required;
it never silently substitutes zero or resamples a failed draw. Supported
numerics are FP32 CPU/MPS and CPU FP64 reference fixtures.

This is established column-row sampling, not a faithful VCAS reproduction:
[Adelman et al., NeurIPS 2021](https://arxiv.org/pdf/1805.08079) and
[Wang et al., ICLR 2024](https://arxiv.org/pdf/2402.17227). VCAS additionally
adapts ratios and samples activation gradients. Its CUDA timings do not
support a claim about this implementation. No upstream sampler code was copied.

## Actual environment and execution

Actual accelerator: **Apple M3 Max**, Mac15,11, 38,654,705,664 bytes RAM,
macOS 26.7.1, native arm64 CPython 3.13.5, Torch 2.15.0. The default shell
Python3.11 was not used. Existing `.venv` pins were retained; no dependency
was added or changed. Installed SHA256 RECORD entries, versions, original
lock/wheel hashes and exact manifest hashes are in
[dependencies.json](prepare/attempt-0003/dependencies.json). RECORD consistency
is distinct from independently re-downloading and hashing wheel archives.

Sandbox doctor hid MPS; the host-visible fresh-process doctor actively verified
fallback policy and actual MPS execution. See [initial sandbox query](initial-doctor.json),
[host query](host-doctor.json), and [qualified environment](prepare/attempt-0003/environment.json).
Fallback, fast math and prefer-metal were all `0` before Torch import;
matmul precision was `highest`. AC power and mode code0 were reported;
low-power status was unavailable. No allocator safeguard or application was
changed. Temperature/continuous thermal telemetry was not collected; no values
are fabricated. Resource counters are endpoint observations, not peaks.

The native tiny B processes started with about 9.75–9.78GB available RAM and
35.68GB pre-existing OS-wide swap use. Swap-in counters increased; swap use and
swap-out counter were unchanged over the reported endpoints. This context may
matter, but the data do not identify a cause for timing jitter. RSS around
459MB and Metal-driver allocation around9.13MB at process completion overlap
other memory measures and must not be added. Other external applications were
left running; repository driver ownership used an advisory exclusive lock and
repo test/collector/benchmark process screening. No agent tests or collectors
overlapped the B sessions. This does not establish control over unrelated
external GPU activity.

## Reached gates and failures

| Stage / attempt | Result | Evidence |
| --- | --- | --- |
| Prepare1 | inventory-only dry run; no tests/device work/download/training | [record](prepare/attempt-0001/record.json) |
| Prepare2, v1 | BLOCKED_HARDWARE label for unsupported combined transfer/cast on existing MPS | [failure](prepare/attempt-0002/command-006.stderr.txt) |
| Prepare3, v2 | PASS: CPU/MPS/full suites, historical renderer, actual-shape sampler operations, pinned environment | [record](prepare/attempt-0003/record.json), [operations](prepare/attempt-0003/qualification.json) |
| Feasibility1, v2 | INCONCLUSIVE_MEASUREMENT, initial and one replacement both unstable | [record](feasibility/attempt-0001/record.json) |
| Correctness1, v3 | PASS, 117 operator/protocol tests including two new MPS regressions | [record](correctness/attempt-0001/record.json) |
| Cost / pilot / confirm | not launched: B prerequisite failed; corpus gate additionally unresolved | no downstream scientific evidence |
| Summarize1 | PASS, reports incomplete/blocked prefix without upgrading it | [summary](summarize/attempt-0001/summary.json) |

Prepare2's `Undefined type Double` came from combined `scores.to(cpu,
float64)`. Separate transfer then CPU cast, plus CPU weight cast before device
transfer, reproduced RED→GREEN on actual MPS. The failed record remains intact;
its label is not a claim that this machine lacks an accelerator. Prior and
current amendment/source bindings and the repair rationale are retained in
[frozen_versions](frozen_versions/).

B stopped at (n,di,do)=(128,128,512). It used 100 updates per warmup block, the
prescribed cap; these launch-bound blocks lasted below0.2s. Each variant's
warmup used max32 blocks and required the last8 range/mean<=.10. These are the
observed **last8 relative ranges**, not confidence intervals:

| Native variant | Initial process, seed61001 | One replacement, seed61003 |
| --- | ---: | ---: |
| default | .2300 | .1614 |
| single | .1535 | .2639 |
| foreach | .2866 | .3016 |

All six warmups failed, with every raw block retained. No slow block was
discarded. See [initial variants](feasibility/attempt-0001/session-invocation-1-tiny-native-variants.json)
and [replacement variants](feasibility/attempt-0001/session-invocation-2-tiny-native-variants.json).
These are unqualified pilot observations, not a throughput comparison. No
strongest native variant was selected. No A/A or zero-dW paired block series,
frozen delta, bootstrap ratio/CI, f estimate, or sampler timing exists. The
middle, wide, and accumulation fixtures have Stage-A operation checks only;
they were not measured for comparative cost after the stop.

The same B source incorrectly labeled fused unsupported because its native
scalar step counter is on MPS while scalar-reference AdamW's counter is on
CPU. The current checker compares exact counter values on CPU and retains
normal parameter/moment checks. A five-step tiny fused/scalar correctness
regression passes. This does not qualify fused timing or retroactively choose
it as fastest. The other three variants' warmup failures are still retained;
there was **no second replacement or post-repair B rerun**. A future study
would require fresh full-shape native qualification.

## Commands, tests, and charged work

The stage command form, using the existing environment, was:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_MPS_PREFER_METAL=0 \
PYTHONPATH=src .venv/bin/python analysis/run_dw_sampling.py \
  --stage <prepare|feasibility|correctness|summarize> \
  --config configs/dw_sampling.yaml --out analysis/dw_sampling/20261031T105213Z
```

The first prepare additionally used `--dry-run`. CLI help was tested for no
output-directory side effects. Each child command retains argv, environment,
exit code, stdout, stderr and invocation duration in its attempt directory.
Tests/rendering used exactly:

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'
PYTHONPATH=src .venv/bin/python -m pytest -q -m mps
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_sampled_linear.py tests/test_dw_sampling_protocol.py
PYTHONPATH=src .venv/bin/python -m oar.cli report --gates decisions/gates.jsonl --run-index analysis/evidence_index.json --out <new-attempt>/historical-report-check.md
```

All host-visible test processes also had the three MPS flags above set before
import. Recorded counts:

| Check | Pass / fail / excluded | Source and scope |
| --- | --- | --- |
| Initial CPU preflight | 468 / 0 / 9 deselected | earlier development snapshot |
| v1 final preflight | 479 / 0 / 0 | before MPS boundary regression |
| Prepare2 CPU / MPS / full | 470 /0/9; 9/0/470; 479/0/0 | v1, unchanged original355 CPU +9 MPS included |
| Prepare3 CPU / MPS / full | 470/0/10; 10/0/470; 480/0/0 | v2, includes score-transfer MPS regression |
| Correctness1 | 117 / 0 / 0 | current v3, operator101 + protocol16 |
| Final full suite | **481 / 0 / 0**, 8.61s | v3, includes original364 and two new MPS regressions |

Final output: [validation/final-full.txt](validation/final-full.txt).
Historical evidence rendering succeeded in both hardware preparations, into
new files. Expected test-first failures and the earlier seven driver-absent
integration failures are preserved in [development_checks](development_checks/);
they are development failures, not hidden scientific trials. Both actual MPS
regressions retain specific failing and passing outputs. No tolerance was
loosened to proceed.

Internal stage invocation times were prepare2 29.303s, prepare3 29.476s,
feasibility22.612s, correctness2.143s (plus initial inventory/dry run,
preflights, debugging, review and final verification). B workers consumed
10.320s and10.331s inclusively. The prospective B conservative forecast was
4621.663s, hard deadline9243.326s, including up to three possible invocations,
warmup, all arms, reset and serialization; it stopped much earlier. Resource
forecasts and actual command times are preserved separately. Stage internal
clocks exclude interpreter time before the driver starts, while parent
subprocess records include child startup. These are diagnostic/research costs,
not canonical language-training clocks or a complete human/agent development
labor estimate. No failed attempt or restart is reclassified as free training.

**Language-model tuning/selection cost: zero trials, zero language-model
updates, no corpus preprocessing/download, no checkpoints.** The planned grid
and maximum87 full-budget equivalents were not allocated or spent. No quality
target, target-time ratio, endpoint NLL/perplexity, censoring analysis,
confirmation CI or independent training-seed replication is available. No
unreached run is represented as a censored training observation.

## Asset status, reviews, and limits

The [new asset decision](asset_decision.json) records documentary inspection
only. The [pinned primary card](https://huggingface.co/datasets/Salesforce/wikitext/raw/b08601e04326c79dfdd32d625aee71d232d685c3/README.md)
still has CC-BY-SA3.0/GFDL metadata and CC-BY-SA4.0 prose. Neither documentary
resolution nor an explicit owner determination was obtained. No corpus body,
substitute corpus, loader dependency, tokenizer, protected evaluation, or
language trainer was introduced. The historical blocked manifest stays intact.

Three explicit review tracks cover mathematical/statistical validity,
backend/optimizer/cost/recovery, and autonomous clean-checkout reproducibility.
The first two source reviews and all dispositions are in [reviews](reviews/).
The backend review found and closed premeasurement resource-cap, pin,
exclusivity/deadline, unstable-pilot selection, block-length, endpoint-memory
and stale-summary issues. The later transfer and counter defects have covering
regressions and preserved failures. Final empirical-report review and the
clean-checkout execution are recorded in the completion addendum below.

Mathematical review limits remain binding: block bootstrap would be a
conditional timing screen with possible serial dependence, no simultaneous
coverage over the eligibility search, and repeated restored sampler
trajectories rather than independent learning seeds. A/A delta would be a
frozen engineering margin, not a confidence bound. Neither bootstrap nor delta
was reached here. Clipping/AdamW nonlinearities, cancellation, score underflow,
CPU sampling overhead, retained activations/dense optimizer state, unknown
long-horizon quality, and uncontrolled external device activity limit claims.

**Stop recommendation:** preserve this CPU/MPS-correct operator and the complete
measurement-inconclusive evidence. Reopening requires explicit prospective
permission for a new timing attempt, a defensible stable/exclusive session
without relaxing the frozen rule after these outcomes, corrected full-shape
native qualification, then successful B/D gates. Language work additionally
requires a documented WikiText restriction resolution. No new backend,
algorithm, controller, horizon, model or corpus is recommended by these data.

## Artifact identities

The latest [amendment](../../../decisions/dw_sampling_amendment.json) binds all
historical files and new source/config bytes. Each [stage record](summarize/attempt-0001/summary.json)
links its immutable artifact hashes and source manifest. Key SHA256 values:

| Artifact | SHA256 |
| --- | --- |
| sampled_linear.py | `11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3` |
| run_dw_sampling.py, current v3 | `227f6977d4713e84d01cf2ebeb61d390d769f391da6fe14e65704daac47df663` |
| config | `afbff17b101ffc45963b0bb8132574c405ba821f196704ff254e721e9e7ad4c3` |
| scope | `d91d9350dc117f83e39c648e48d435de75ab9ea5a3b6bc197b216c4b73c11181` |
| current amendment | `7c269df87b3c726def0dccebf245e978da00f1835def79482ecaaf425d613c27` |
| initial historical inventory | `10fc155f316905ba509ee1f76b4b8bd8d6a4c438537b153e026100d346e666fc` |

A separate final hash inventory excludes itself to avoid a hash cycle. Original
software remains AGPL-3.0-only; writing/results remain CC-BY-4.0; upstream rights
are unchanged. No corpus excerpts or text-bearing tensors are distributed.

## Completion addendum

Final empirical-report reviews and fresh-checkout execution are pending at this
report snapshot. Their actual results will be appended; no independent training
replication or clean-checkout pass is claimed before that execution.
