# Optimizer-aware residual auditing: implementation specification

**Date:** 2026-10-30

**Status:** Proposed implementation contract; no software implementation or new experimental evidence.

**Source:** [Research blueprint](../../blueprint.md), especially §§4, 6–13 and Appendix A.

**Companion:** [Implementation plan](../plans/2026-10-30-oar-implementation.md).

## 1. Intent and evidence boundaries

Build a local, reproducible proof of concept that asks whether approximating selected linear-layer weight gradients improves learning per charged wall time on an actual M3 Max. OAR proposes a projected gradient, previews the actual AdamW displacement, and audits how the missing gradient could interact with the displacement outside the retained subspace. Correctness, useful gradient reuse, audit information value, hardware headroom and learning value are separate questions.

The beneficiaries are researchers training small models from scratch under limited accelerator time. The deliverable is an executable study with an honest positive, negative or inconclusive conclusion. A fast matrix multiplication alone is insufficient. An online controller is justified only after both timing and future-batch oracle tests identify a useful operating region.

The blueprint's historical CPU checks and source inspections remain attributed to that document. They have not been rerun for this specification. Its bibliography is an evidence ledger, not a package lock or an assertion of current upstream behavior. Execution must pin and inspect actual packages, assets and upstream revisions. No methodological novelty, formal floating-point certificate, M3 speedup or 10×–100× training improvement is established here.

## 2. Goals, non-goals and constraints

Goals:

- Preserve forward computation and exact input/bias derivatives while replacing only eligible weight derivatives.
- Evaluate H1 reusable information, H2 optimizer-aware discrimination, H3 inclusive cost, H4 held-out learning value and H5 calibration from blueprint §2.3.
- Compare native dense training, simpler approximations, information-matched controls and faithfully executable prior art.
- Make every decision reproducible from versioned configuration, manifests, gate records, logs and committed checkpoints.
- Preserve stopped branches, failed trials and scope limitations as useful outcomes.

Non-goals:

- Approximate incoming adjoints, shared/tied weights, attention projections or whole backward propagation in the primary method.
- Build an autodiff engine, compiler, predictor, gradient dictionary or new architecture.
- Automatically implement pivots B/C/D, mixed precision, MLX, new corpora or larger models when an earlier gate fails.
- Establish distributed, frontier-model, rare real-world capability, patentability or general convergence claims.

Project-wide constraints, repeated verbatim in the plan:

- Work in the existing repository on the current branch; no nested repository or automatic push.
- Primary runtime: native arm64 CPython and PyTorch MPS; CPU FP64 references; primary training FP32.
- Use NumPy, pytest, PyYAML and psutil initially; add dataset/tokenizer/Parquet tooling only after asset and backend gates.
- Resolve actual compatible package versions and wheel hashes at discovery; do not infer versions from the blueprint's documentation links.
- No remote GPU, paid API, CUDA dependency, hosted tracking service or automatic OS upgrade.
- Disable silent CPU fallback in timed critical paths; explicit CPU decomposition and transfers are allowed only when logged and charged.
- Approximate only untied MLP weight matrices initially; keep input gradients, biases, norms, embeddings and shared/output weights exact.
- Freeze parameters, bases, ranks and probes within an effective optimizer batch; adapt ranks only for the next logical update.
- Keep construction, audit, data and model randomness separate; never use current audit probes to construct the candidate.
- Preserve raw gradients through clipping; preview the full declared AdamW rule without mutating live state.
- Rejection replays the same effective batch and model randomness; commit optimizer, scheduler and data cursor exactly once.
- Charge initialization, refresh, preview, audit, transfer, replay, monitoring, routine checkpoints and restart overhead to active wall time.
- Keep official evaluation outcomes out of tuning and policy code; freeze the confirmatory protocol before protected evaluation.
- Publish no training or speedup claim beyond the actual backend, workload, baselines, uncertainty and guardrails tested.

## 3. Approaches and scope decision

The supplied blueprint already selects the first approach. This document carries that choice forward; it does not record a new design approval.

| Approach | Benefit | Cost or limitation | Role |
|---|---|---|---|
| Projected weight gradients, exact adjoints, complement-aware AdamW audit | Tests whether residual information matters for the actual update without changing upstream credit propagation | Dense optimizer states, probing, preview and replay can consume all savings | Selected primary method, conditional on gates |
| Fixed projection or a development-frozen schedule | Same proposer with less runtime work | Cannot react to current update risk | Mandatory simpler control and preferred outcome if auditing adds no value |
| Importance-sampled weight gradients; complete published BP approximations | Tests a different information/cost trade-off | Sampling overhead or approximate adjoints; portability and calibration obligations | Competitive baselines, not unannounced replacements for OAR |

Implement one connected research pipeline with bounded stages, rather than several independent feature projects. Separate plans are required if a pivot changes the objective or architecture. Hardware, protocol values and engineering details that the blueprint intentionally leaves empirical remain gate outputs, not invented constants.

## 4. Mathematical and numerical contract

### 4.1 Linear operator and basis

Flatten batch/token axes: `X[n, di]`, `W[do, di]`, `D[n, do]`. Forward is `X @ W.T + bias`. Dense derivatives are `G = D.T @ X`, `dX = D @ W`, and `db = D.sum(0)`. Loss normalization is already present in `D`; never divide these gradients by token count again.

For a fixed orthonormal left basis `U[do, r]`:

```text
H = (D @ U).T @ X
G_hat = U @ H
E = G - G_hat
R = D.T @ (X @ Z) - U @ (H @ Z) = E @ Z
s = ||R||_F² / q
```

`Z[di, q]` contains independent standard Gaussian entries, independent of the current candidate conditional on history. Primary AdamW receives a dense reconstructed `G_hat`; OAR does not claim optimizer-state memory savings. Keep only detached running `H` and `R` beyond each backward; do not retain all adjoints.

Start with a native dense step. Its saved pre-update exact gradient creates a basis for the next step only. At scheduled refreshes use `Q = qr(G @ Omega)`, small SVD of `Q.T @ G`, and ordered left singular vectors lifted by `Q`. Construction width is `min(di, r_max + p)`, with initial development `p = 8`, no power iterations. Shrink for rank deficiency and measure orthogonality. Current exact SVD is an offline oracle, charged as research cost, never an implicit online shortcut.

Rank zero is a correctness fixture; `dense` is a separate runtime state. A full left basis must recover the dense derivative. Production approximate ranks are below `min(di, do)` and must have measured headroom. Shared parameters are excluded by parameter identity even if reached through two module names. Higher-order differentiation fails explicitly.

### 4.2 Confidence and numerical qualification

Solve in CPU FP64 for `0 < c < 1`:

```text
c - 1 - log(c) = 2 log(1/delta) / q
b_math = sqrt(s / c)
```

The real-arithmetic model gives `Pr(||E||_F > b_math) <= delta`; blueprint §4.4 contains the derivation. Validate `q > 0`, `0 < delta < 1`, finite nonnegative statistics and a representable positive root. Numerical underflow, nonfinite values or an unresolved guardband must fail closed.

Allocate residual-test indices before generating/observing probes. Default indefinite spending is `delta_j = delta_run * 6 / (pi² * j²)`, with `j` beginning at 1. Rejected, zero-residual and possibly observed interrupted tests all consume allocations. A finite equal allocation is permitted only when a frozen maximum test count is enforced. Journal reservations durably; after a crash use the maximum spent index from checkpoint and journal. Restore historical training/model state, but never roll back the spent audit ledger or reuse the consumed probe identifier.

FP32 projection defects, reduction errors and cancellation prevent a formal machine-arithmetic guarantee. Calibrate an empirical guardband on development stress fixtures against CPU FP64, log its distribution and orthogonality defect, and reject uncertain margins. The resolved guardband configuration must define a nonnegative total margin allowance `u_margin` in gradient–displacement inner-product units, including residual and orthogonality uncertainty. Accept only when `a > 0` and `B + u_margin <= tau * a`; reject if the numerical policy cannot produce a finite allowance. The allowance formula/coefficients are empirical outputs of operator development and must be frozen before online learning claims. This is an explicitly qualified empirical audit with a real-arithmetic confidence model. Gaussian probes cannot be replaced with another distribution under the same theorem. Monte Carlo tests check calibration at resolvable probabilities; no finite small simulation proves a one-in-a-million guarantee.

### 4.3 Accumulation, displacement and decision

For microbatch valid-token means `L_m`, backpropagate `w_m * L_m`, where `w_m = valid_m / valid_effective`. Ignore zero-valid microbatches without changing the effective objective; reject an entirely zero-valid batch before state mutation. Freeze the same `U`, rank and `Z` for all microbatches. Accumulate `H = sum(H_m)` and `R = sum(R_m)`, then test the aggregate residual once per layer. Never sum microbatch squared residual norms.

Preview AdamW using globally clipped input `h = gamma * g_hat`, existing per-parameter moments/step counts, native clipping stabilization, per-group learning rates, epsilon, bias correction and declared decoupled decay. Initial scope excludes AMSGrad, fused semantics that have not passed equivalence tests, differentiable optimizers and mixed precision. Missing gradients preserve native skip behavior; a zero gradient tensor remains a participating gradient and may change moments/decay. Raw `g_hat` remains unchanged.

Use the actual parameter decrement `d = theta - theta_candidate`, including decay. Compute across all participating parameters:

```text
a = <g_hat_raw, d>
d_perp_l = d_l - U_l @ (U_l.T @ d_l)
B = sum_l b_l * ||d_perp_l||_F
```

An algebraic acceptance requires finite values, `a > 0`, `0 < tau < 1` and `B <= tau * a`. Numerical acceptance additionally requires the conservative guardband above. Use explicit complements initially; subtracting squared norms is a separately validated optimization. A positive margin is first-order and minibatch-local, not a finite-step loss or generalization guarantee. Projected SGD with no momentum/decay is the automatic-descent negative control.

### 4.4 Transaction and next-step controller

Each pending update is tagged with live parameter and optimizer versions, has no external side effects, and may be committed once. Version mismatch or double commit raises an error. On rejection discard pending moments, gradients and sketches; restore the initial model RNG; replay identical descriptors with native exact backward and one native optimizer update. Do not restore audit randomness/counters. A dense fallback is executed unchanged even when its own instantaneous data-loss alignment is negative; log `exact_fallback_no_descent_claim`.

Checkpoint only after optimizer commit, future-use scheduled basis construction, controller update and single scheduler/data-cursor advance. Count valid training tokens once and processed/replayed exposures separately. Nonzero dropout, checkpoint recomputation and stateful forward buffers require explicit restore/once-only-side-effect tests. Primary models have no BatchNorm.

The controller implements blueprint §4.9 exactly: measured rank candidates (powers of two from 4 plus efficient measured ranks), next-step-only changes, stable layer-ID ties, nonpositive-cost increases before positive-cost ratios, one-layer changes, low-slack shrink streaks and fixed dense refresh schedule. Development explores `tau` in `{0.25, 0.5, 0.75}` and streak `h` in `{8, 32, 128}`; rank sets, probe count, refresh `K`, oversampling and all thresholds freeze later. For `a <= 0`, schedule one dense step and retain prior rank configuration. Rejection does not itself refresh bases. Dense layers re-enter only at scheduled refresh. If no approximate rank can help, continue/report dense mode without an online-advantage claim.

## 5. Architecture and durable interfaces

Use the existing repository root for `pyproject.toml`, `src/oar/`, `tests/`, `configs/`, `manifests/`, `decisions/`, `runs/` and `analysis/`; blueprint §10.1's `oar-poc/` is illustrative, not another repository. Do not install software or scaffold the runtime as part of this documentation task.

| Files | Responsibility |
|---|---|
| `config.py`, `records.py`, `cli.py` | Strict resolved configs, schema-versioned domain records and commands |
| `hardware.py`, `provenance.py` | Redacted environment, asset/source/wheel identities and operation capabilities |
| `linear.py`, `basis.py` | Exact/projected autograd branch, streaming sketches, paid past-only construction |
| `audit.py`, `audit_journal.py` | Scalar confidence model, numerical decision and durable spending ledger |
| `optimizer_preview.py`, `controller.py` | Functional pending AdamW state and pure next-step rank decisions |
| `data.py`, `models.py`, `synthetic.py` | Protected split access, exact target masks and shared architectures |
| `timing.py`, `checkpoints.py` | Inclusive clocks, separate profiling and atomic committed snapshots |
| `methods.py`, `train.py` | Native fast path and replayable logical-step orchestration |
| `baselines/` | Separate sampling, audit controls and faithful named ports |
| `oracle.py`, `protocol.py`, `analysis.py` | Counterfactual study, frozen experiment selection and run-level inference |

These are proposed interfaces, not existing APIs. Use dataclasses and strict parsers rather than adding a configuration framework. `records.py` defines:

- `ParameterId`: canonical model parameter name, identity-deduplicated; `LayerId` aliases the eligible weight's `ParameterId`.
- `ReplayableBatch`: immutable descriptor with source IDs, ordered microbatches, target masks, valid counts and pre/post data cursor; it can rematerialize identical tensors without advancing an iterator.
- `StepContext`: logical step, parameter/state versions, valid-token normalization, RNG snapshots, schedule position, descriptors and timing boundary.
- `LayerAudit`: layer ID, shapes, rank/basis version, probe ID/count, allocation, detached sketch/statistic, bound, orthogonality, guardband and optional profiling values.
- `PendingUpdate`: source versions, candidate parameters/decrements and pending per-parameter AdamW state, plus a consumed marker.
- `Decision`: enum `accept_approx`, `scheduled_dense`, `reject_replay`, `numerical_reject`, `stop_invalid`; reason codes, `a`, `B`, allowance and next-step changes.
- `StepRecord`: versions, decision, logical/valid/processed counters, active time and committed checkpoint reference; all optional measurements are null, never fabricated zero.
- `GateRecord`: package, status `pass|stop|inconclusive|blocked`, artifact hashes, measured/forecast resources, reason and next permitted action.

Preserve the method lifecycle from blueprint §10.2: `begin_step`, `backward_microbatch`, `propose`, `decide`, `commit_or_replay`, `state_dict`, `load_state_dict`. The native dense path bypasses preview/probes/complement and avoids an OAR-only overhead tax. Configuration validation rejects unknown fields, invalid fractions, unsupported method/parameter eligibility and confirmatory essentials that remain unresolved. CPU-only runs are explicitly classified as correctness/structural work.

Checkpoint payloads include all model buffers, moments, scheduler/counters, ranks/bases/refresh/streak state, RNGs, data cursor/replay descriptor, config/data/tokenizer/code hashes, active time and audit index. Write a checksummed temporary payload, flush, atomically rename and retain a prior valid checkpoint. Refuse incompatible resume; restart overhead remains charged. Audit reservations survive even when tentative training state is discarded.

## 6. Workloads, comparators and protected information

The language study uses pinned WikiText-2 raw assets first, a training-article fitting/development split initially 90/10, fitting-only byte BPE with vocabulary target 8192 and explicit BOS/EOS/PAD. Preserve case, normalize line endings only, reconstruct article boundaries and record duplicate/excluded article hashes. Official validation is reserved for the frozen monitoring protocol; official test stays sealed until declared final scoring. Use blueprint §7.1's chunk construction: score EOS, exclude BOS/PAD, preserve preceding-token continuity, prevent cross-article attention. Evaluation stride is `floor(S/2)` with each target scored once and input length at most `S`.

The common scratch decoder is pre-LN, causal attention, GELU MLP expansion 4, tied embedding/output and learned positions. Dropout starts at zero; nonzero dropout remains a replay test. MLP weights alone are eligible. Tiers A/B/C/D are `(L, d) = (6,256), (8,384), (8,512), (12,768)`; start timing at context candidates 128/256/512 and choose scale from evidence, not presumed capacity. Effective batch is matched across methods; changing it creates a new workload and retuning obligation.

Secondary Fashion-MNIST uses a stratified training development split, [0,1] pixels and a residual MLP initially depth 4/width 512. Synthetic associative recall resamples per-sequence mappings, uses disjoint key/value ranges, a visible first/last-occurrence rule marker, minority training probabilities 0.01 and 0.001 in separate conditions, balanced independent evaluation and a majority-to-mixture shift. Dense BP must first learn both rules. Synthetic or vision success cannot replace language evidence.

Implement B0 native AdamW, B1 checkpointed dense when memory matters, B2 eligible-matrix Muon when faithfully supported, B3 fixed projection, B4 development-frozen schedules, both B5 audits, B6 weight-only importance sampling, and attempt B7 INSTANT/B8 VCAS/B9 DropBP before broad competitiveness claims. Named ports require whole-revision pins, notices, reviewed patches, source-equivalence tests, complete calibration and charged data exposures. B6 is not VCAS. INSTANT's literal helper behavior and paper-intent variant remain separately labeled exactly as blueprint §6.2 specifies. Blocked ports narrow comparison scope; they are never reported as defeated.

Relative B5 uses the blueprint's aggregate residual norm criterion, all-zero handling, squared slack/contributions and its own tuned tolerance; it can use native AdamW after acceptance without an unnecessary preview. Ordinary B5 substitutes full displacement norms. Both preserve the same probes, spending, refresh information and replay rules. B6 samples with replacement using row-norm-product scores plus a positive uniform floor, executes a smaller gathered GEMM, and handles all-zero scores exactly.

Diagnostic controls D0 native/custom exact, D1 paid current-subspace/residual oracles, D2 plain/projected SGD and D3 information-matched dense/refresh schedules isolate operator overhead, stale bases, guaranteed projection descent and extra information. A schedule extracted from confirmatory OAR after the fact is a diagnostic, not a deployable control.

## 7. Evaluation and resource contract

`active_wall_s` begins at model/optimizer construction, includes prescribed monitoring and serialization, and accumulates restart/recompile/load cost. Asset download/shared one-time tokenization/external pauses are separately recorded, with a cold-start total. Method-induced stalls/swapping count. At horizon `H`, select the last fully committed saved evaluation checkpoint whose cost including serialization is `<= H`; exclude later states. Record unused budget and cadence. Score the protected endpoint once, recording its separate cost.

Training runs paired by initialization/data-order seed are the independent units. The primary endpoint is held-out NLL difference against both the strongest development-selected executable baseline and a matched no-information projection control at fixed active time. Freeze sample size from pilot variance, useful absolute NLL resolution, precision/power and measured affordability. Use the prespecified paired t analysis or a justified preselected paired randomization/bootstrap alternative; adjust the two primary comparisons with Holm at familywise 0.05. Report individual pairs, 95% intervals, censoring and all failures. No checkpoint/token/block pseudoreplication, adaptive significance stopping or test-driven method selection.

Freeze quality targets and rare-slice noninferiority margins from development; an unsupported margin cannot justify a no-degradation claim. Time-to-target requires confirmation at the next scheduled evaluation; report nonattainment/censoring. Negative or imprecise results remain valid deliverables. See blueprint §§6 and 8 for the statistical assumptions and exact boundary rules.

Use synchronized measured blocks on MPS, adaptive warmup, null-block overhead, randomized AB/BA ordering and complete-step headline measurements. Keep fine-grained profiling separate. Record power/thermal context, invalidation rules, tensor/driver/RSS measurements separately and memory pressure. Initial memory ceiling is the smaller of approximately 70% of physical memory and 80% of a reported recommended Metal working set, with an OS reserve; never disable safeguards. Preview adds roughly 12P bytes beyond dense AdamW's 16P bytes before activations/workspaces. Measure full-step, refresh, replay and checkpoint peaks.

Do not infer whole-step speed from FLOPs. The necessary local break-even check is `T_A + p*T_R + C/K < T_D`, qualified by actual scheduled-dense mixture accounting; full-cycle measured means are authoritative. Choose `H`, ranks, `q`, `K`, microbatch, confirmatory run count and storage from measured development evidence. No fixed laptop-hours allocation or percentage-win threshold is invented by this spec.

## 8. Gates, artifacts and acceptance

Every package appends `decisions/gates.jsonl` and updates `decisions/progress.md` with what ran, resources, artifact hashes, interpretation and next action. A failed gate stops dependent work while still permitting synthesis. Essential amendments append to `decisions/amendments.jsonl` and create a new protocol hash before dependent experiments.

| Package | Required evidence and artifacts | Exit / stop rule |
|---|---|---|
| WP00 discovery | Redacted environment, wheel/source/asset manifests, tiny CPU/MPS capability results | Unsupported GPU narrows to correctness; unresolved assets block that workload |
| WP01 correctness | Projection, exact adjoints, Gaussian bound, accumulation, native-preview and transaction tests | Any mathematical/state defect blocks dependent experiments |
| WP02 headroom | Raw shape/block timings, full-block timing, memory and break-even map | No candidate-only headroom stops online engineering; audit-only cost failure permits a cheap structural oracle study |
| WP03 harness/baselines | Learnable dense curves, leakage/timing/A/A/resume tests, native and port feasibility results | Invalid harness blocks study; unavailable baselines narrow claims |
| WP04 oracle value | Past-only, disjoint-future-batch update diagnostics and same/copied-state counterfactual loss changes | WP05 requires useful proposals, audit discrimination and plausible positive inclusive cost jointly |
| WP05 online OAR | Complete logged accept/reject/refresh/replay controller with matched audit controls | Incorrect replay/versioning blocks; correct but slower is a result, not permission to expand scope |
| WP06 pilot/freeze | All development trials, tuning fairness, precision/runtime forecasts, hashed complete protocol | Unresolved essentials or unaffordable precision block confirmation |
| WP07 confirmation | Fixed paired runs, saved endpoint records, protected scoring and prespecified analyses | Positive claim requires both declared comparisons and guardrails; wide intervals are inconclusive |
| WP08 robustness | Fresh seeds, replay/dropout/rare-rule/shift/secondary-task evidence and clean-process repeat | Narrow or withdraw claims that fail replication/guardrails |
| WP09 synthesis | Report, raw index, limitations, stopped/failed runs and local reproduction recipe | Complete even after an earlier stop if conclusion and reproducibility status are explicit |

The first decisive outcome is a cost-qualified oracle comparison, not a polished online trainer. WP03 is deliberately minimal until WP04 answers that question. Heavy port tuning and full controller integration can be deferred until this gate, but feasibility attempts and fair comparisons must be completed before protocol freeze.

## 9. Verification and builder discretion

Required tests cover projection/full basis/zero/rank deficiency, exact multi-block adjoints, corrupted adjoint detection, invalid probe dependence, noncontiguous inputs, unequal masks/accumulation, skipped versus zero gradients, clipping/decay/mixed groups, transaction rollback, dropout replay, checkpoint recomputation, journal crash recovery, shared-parameter exclusion, protected split access, endpoint overrun and deliberately slower timing controls. Full-function finite-difference gradcheck applies only to exact/full-basis operators, not the deliberately projected weight derivative.

The builder may choose local filenames beneath the mapped modules, internal helper implementations, CPU-calibration mechanics, cache locations and ordinary refactoring while preserving the contracts. Numeric guardband coefficients, hardware-efficient ranks, runtime versions and experiment budgets must be measured/frozen at their gates. Changes to objective, hypothesis, evaluation population, information access, baseline scope or numerical guarantee are amendments, not routine implementation choices. Pivots from blueprint §14 require their own spec and plan.

For this documentation task, acceptance means a source-traceable spec, an actionable gate-ordered plan, a builder review, internally consistent interfaces/links and meaningful commits on the original branch. Runtime tests and scientific gates remain future work; writing their commands does not mean they have passed.

**Preparation review:** A fresh numerical-ML builder reviewed this document against the blueprint on 2026-10-30 and reported “nothing important left” when asked for consequential unanswered planning questions. This checks plan readiness; it is not user approval, an implementation review or experimental validation. Engineering clarifications made here include the explicit numerical margin allowance, canonical parameter identity, durable audit spending on crash recovery and the existing-root layout.
