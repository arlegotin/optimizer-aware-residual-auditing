# OAR reached-stage evidence and execution outcome

**No training speedup was established.** All supported measurements are exploratory or component/reference evidence. Training quality, time-to-quality, complete training memory and independent confirmation are unavailable unless explicitly measured; this schema supplies no confirmation validator.

Evidence binds the first 16 gate lines, SHA256 `95909fde5d222f7ff6366a93cc6f1ba3a8ea72682db7c6e6e8786e5a17ee1b95`. Later WP09 records do not enter this scientific snapshot. Artifact annotations are authored interpretations; hashes establish attribution, not scientific validation.

Candidate reconstruction is28.875ms versus paired native36.882ms; the complete conditional audited attempt is54.406ms versus its paired native36.738ms. No absolute training-quality gain or full-training-memory trade-off was measured. Wide retained parameters/optimizer/preview total151,117,824 bytes; study sampled RSS/Metal readings overlap and are not single-model training peaks. Scope: Apple M3 Max, Mac15,11,38,654,705,664 physical bytes; macOS26.7.1; native arm64 CPython3.13.5/PyTorch2.15.0 MPS FP32 with fallback0. One two-linear GELU residual block, n8192/di768/do3072/rank16/q16, not a language model. Fastest validated fused native AdamW selected; all compared variants validated against three single-tensor updates. Uncertainty: Exploratory randomized shared-round paired block bootstrap, not independent training runs. Recorded memory pressure/swap and unknown thermal limits qualify all times; candidate95% saving7.395–8.605ms and full-cost saving−19.052 to−15.941ms. p0 commit bypasses unresolved rejection only for pricing, never for online acceptance. Source: [analysis/timing_calibration_wide.json](timing_calibration_wide.json).

Every complete p0 attempt in the six measured configurations is slower. Forced rejection is91.401ms; direct exact replay37.366ms and paired extra refresh11.191ms in the wide block. Named preview including CPU alignment19.667ms (input validation6.484ms, raw CPU FP64 alignment/transfer2.071ms) and generation/journal/transfer/capture3.257ms are reference costs, not irreducible lower bounds. Journal reparse grows3.121/9.438/27.934ms at0/64/256 reservations. Do not subtract independently aged conditional/rejected paths to infer replay. Scope: Same isolated MPS operator study; main/wide research clocks25.779/12.577s include initialization, warmup and profiling before final JSON serialization. K/p grid is arithmetic sensitivity, not an observed training cycle. Uncertainty: Correct threshold p*=(D−A−C/(K−1))/R for K>1; K1 has no approximate segment. Wide K16 p*=−.492802,95%[−.535047,−.447323], no feasible probability. Four-microbatch candidate failed warmup/noise qualification despite positive raw timing. No time-to-quality claim. Source: [analysis/headroom_decision.json](headroom_decision.json).

Past evaluation proposals144:36 negative true alignments,43 same-batch harms,72 independent synthetic-development harms; all144 lose development progress versus matched dense. Ordinary/exact-complement/q8-probe-complement accept0/144 at every declared tau.25/.5/.75 before policy fitting. Relative thresholds.1/.25/.5/1/2 accept0/0/8/56/104, including development harms0/0/8/37/54. Paid current-SVD complement accepts2/54 only at tau.75/rank64; both still lose development progress versus dense. Scores have nontrivial descriptive AUCs; the complement-versus-ordinary difference is unconfirmed and stage-confounded. Scope: Synthetic native teacher-regression histories, n2048/di512/do2048, six condition-seed histories/two initialization-data seeds;792 layer rows/396 proposals,18 checkpoints and disjoint tuning/evaluation/development descriptors. MSE per output element; counts are synthetic observations. Study measured at2e89e74, results retained atdc56170. Uncertainty: Correlated projections/checkpoints and duplicated early full-rank/shift stages are not independent run replication. One query per split/checkpoint. Dense itself harms early queries; projection is not the sole explanation. No real-corpus NLL, approximate learner trajectory, fixed-time quality, generalization or H4 confirmation. Source: [analysis/structural_oracle_aggregate.json](structural_oracle_aggregate.json).

Native development MSE per output element initial→128: low_rank/seed1101: 0.846770→0.101799; low_rank/seed2202: 0.864220→0.101958; full_rank/seed1101: 0.870242→0.108646; full_rank/seed2202: 0.849622→0.107520; direction_shift/seed1101: 0.870242→0.286898; direction_shift/seed2202: 0.849622→0.283887. All six histories learn; shift phase120→128 improves. Pilot at a100d5c selected LR.003, the upper bracket boundary, before queries. Study worker60.959s,18 ignored local checkpoints485,101,734 bytes; endpoint RSS597,868,544/Metal driver1,126,924,288 bytes overlap. Current SVD/CPU decompositions/transfers, preview, query scoring and serialization are paid offline costs; repeated layer/shared row clocks must not be summed. Scope: Original locked full synthetic study at2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07. Forecast95.30s under predeclared600s engineering ceiling. All18 checkpoints, six histories and792 rows completed; no actual study attempt was dropped or failed. Uncertainty: Eight post-shift updates do not establish convergence/recovery. Native scaffold is learnability calibration, not competitive baseline tuning. Original worker clock excludes fresh-process startup and final-summary tail; endpoint resources miss peaks and cannot establish training memory. No tensor/corpus/weights redistribution. Source: [analysis/structural_oracle_summary.json](structural_oracle_summary.json).

At priced rank16,0/48 held-out past proposals have positive exact lower margin a−B; among42 with positive a, min(B/a)=1.1028868. This diagnoses failure of a necessary condition, not a new selected threshold. Rank64 has2/48 tiny positive margins only beyond the declared grid. Do not infer no possible tau anywhere or no information. Rank4/64 at this structural shape lack matching measured runtimes; no rank-cost frontier or deployable paid-current-SVD pipeline exists. Scope: Deduplicated simultaneous past evaluation proposals, using exact complement bound and raw alignment in retained raw rows. Only rank16 matches the previously measured structural-study operator shape. Uncertainty: Derived descriptive check only; actual numerical allowance remains unresolved. The original unfitted-grid statement is primary. Source: [analysis/structural_oracle_audit.jsonl](structural_oracle_audit.jsonl).

## Evidence classes

algebra/reference: 2

### Tested mathematical and optimizer contracts

Status: measured. Exact incoming adjoints and streaming accumulation, independent Gaussian bounds, crash-safe spending and native-equivalent full AdamW preview/commit are implemented.18 CPU FP64/FP32/MPS FP32 multistep reference cases had zero observed discrepancy. Constructed high-cosine harmful AdamW, flat-spectrum/rotating/rare-coordinate controls are retained. Correctness in these cases is not a formal finite-precision certificate.

Scope: Reached Tasks1–5 component references only, no full trainer replay. Original optimizer collector artifact reused unchanged by Task6. Uncertainty: Real-arithmetic identities and tested implementation behavior; no claim of universal floating-point closure, online safety or empirical control frequency.


Source: [analysis/optimizer_correctness.json](optimizer_correctness.json).

### Recovered same-seed numerical reproducibility

Status: measured. Authorized field-name mapping logical step→committed_updates recovered all30 learning-curve comparisons. All792 keyed rows matched exactly across28 numeric and23 categorical fields, exact harm signs and all declared acceptance decisions; all30 MSE values have zero difference.18 local checkpoint hashes validate. Original failed comparison and predeclared tolerance2e-6 absolute/2e-4 relative remain unchanged; no research rerun.

Scope: Full same-seed bounded synthetic repeat in a fresh actual-MPS fallback0 worker, original study source2e89e74 versus reproduction source7c0706e. Actual clone path/hash is separately bound from original hardcoded worker config provenance. Uncertainty: Numerical reproducibility is not independent scientific confirmation. Repeat61.229s worker/62.914s invocation include separately identified overhead; they do not repair original startup/tail accounting. CPU report-test overlap precludes timing replication; no benchmark/MPS-test/collector overlap.

- checkpoint_hashes_validated: 18 (local checkpoints)
- curves_compared: 30 (curve observations)
- max_curve_difference: 0.0 (MSE per output element)
- rows_compared: 792 (layer rows)

Source: [analysis/reproduction/wp09/comparison_recovered.json](reproduction/wp09/comparison_recovered.json).

backend timing: 2

### Component headroom versus complete-cost deficit

Status: measured. Candidate reconstruction is28.875ms versus paired native36.882ms; the complete conditional audited attempt is54.406ms versus its paired native36.738ms. No absolute training-quality gain or full-training-memory trade-off was measured. Wide retained parameters/optimizer/preview total151,117,824 bytes; study sampled RSS/Metal readings overlap and are not single-model training peaks.

Scope: Apple M3 Max, Mac15,11,38,654,705,664 physical bytes; macOS26.7.1; native arm64 CPython3.13.5/PyTorch2.15.0 MPS FP32 with fallback0. One two-linear GELU residual block, n8192/di768/do3072/rank16/q16, not a language model. Fastest validated fused native AdamW selected; all compared variants validated against three single-tensor updates. Uncertainty: Exploratory randomized shared-round paired block bootstrap, not independent training runs. Recorded memory pressure/swap and unknown thermal limits qualify all times; candidate95% saving7.395–8.605ms and full-cost saving−19.052 to−15.941ms. p0 commit bypasses unresolved rejection only for pricing, never for online acceptance.

- candidate: 28.87526389935778 (ms)
- candidate_paired_native: 36.88247204344305 (ms)
- candidate_saving: 8.007208144085276 (ms); interval [7.3949336433239905, 8.604643594784042]
- complete_paired_native: 36.73832876504295 (ms)
- complete_saving: -17.667898015740015 (ms); interval [-19.052433659413108, -15.940849807359177]
- conditional_complete_attempt: 54.406226780782966 (ms)
- full_training_peak_memory: unavailable (bytes)
- retained_parameters_optimizer_preview: 151117824 (bytes)
- training_quality_gain: unavailable (not measured)

Source: [analysis/timing_calibration_wide.json](timing_calibration_wide.json).

### Replay, refresh, journal aging and cold-start costs

Status: measured. Every complete p0 attempt in the six measured configurations is slower. Forced rejection is91.401ms; direct exact replay37.366ms and paired extra refresh11.191ms in the wide block. Named preview including CPU alignment19.667ms (input validation6.484ms, raw CPU FP64 alignment/transfer2.071ms) and generation/journal/transfer/capture3.257ms are reference costs, not irreducible lower bounds. Journal reparse grows3.121/9.438/27.934ms at0/64/256 reservations. Do not subtract independently aged conditional/rejected paths to infer replay.

Scope: Same isolated MPS operator study; main/wide research clocks25.779/12.577s include initialization, warmup and profiling before final JSON serialization. K/p grid is arithmetic sensitivity, not an observed training cycle. Uncertainty: Correct threshold p*=(D−A−C/(K−1))/R for K>1; K1 has no approximate segment. Wide K16 p*=−.492802,95%[−.535047,−.447323], no feasible probability. Four-microbatch candidate failed warmup/noise qualification despite positive raw timing. No time-to-quality claim.

- K16_break_even_p: -0.4928020451125202 (probability); interval [-0.5350468991297499, -0.44732254684257744]
- direct_exact_replay: 37.36582401001619 (ms)
- extra_refresh: 11.190847105657063 (ms)
- forced_reject: 91.40139334421191 (ms)

Source: [analysis/headroom_decision.json](headroom_decision.json).

exploratory learning: 3

### Bounded oracle retains information but no actionable declared past-margin region

Status: measured. Past evaluation proposals144:36 negative true alignments,43 same-batch harms,72 independent synthetic-development harms; all144 lose development progress versus matched dense. Ordinary/exact-complement/q8-probe-complement accept0/144 at every declared tau.25/.5/.75 before policy fitting. Relative thresholds.1/.25/.5/1/2 accept0/0/8/56/104, including development harms0/0/8/37/54. Paid current-SVD complement accepts2/54 only at tau.75/rank64; both still lose development progress versus dense. Scores have nontrivial descriptive AUCs; the complement-versus-ordinary difference is unconfirmed and stage-confounded.

Scope: Synthetic native teacher-regression histories, n2048/di512/do2048, six condition-seed histories/two initialization-data seeds;792 layer rows/396 proposals,18 checkpoints and disjoint tuning/evaluation/development descriptors. MSE per output element; counts are synthetic observations. Study measured at2e89e74, results retained atdc56170. Uncertainty: Correlated projections/checkpoints and duplicated early full-rank/shift stages are not independent run replication. One query per split/checkpoint. Dense itself harms early queries; projection is not the sole explanation. No real-corpus NLL, approximate learner trajectory, fixed-time quality, generalization or H4 confirmation.

- development_harm: 72 (proposals)
- lost_progress_versus_dense: 144 (proposals)
- negative_true_alignment: 36 (proposals)
- past_evaluation_proposals: 144 (correlated proposals)
- same_batch_harm: 43 (proposals)

Source: [analysis/structural_oracle_aggregate.json](structural_oracle_aggregate.json).

### Absolute synthetic quality and paid study resources

Status: measured. Native development MSE per output element initial→128: low_rank/seed1101: 0.846770→0.101799; low_rank/seed2202: 0.864220→0.101958; full_rank/seed1101: 0.870242→0.108646; full_rank/seed2202: 0.849622→0.107520; direction_shift/seed1101: 0.870242→0.286898; direction_shift/seed2202: 0.849622→0.283887. All six histories learn; shift phase120→128 improves. Pilot at a100d5c selected LR.003, the upper bracket boundary, before queries. Study worker60.959s,18 ignored local checkpoints485,101,734 bytes; endpoint RSS597,868,544/Metal driver1,126,924,288 bytes overlap. Current SVD/CPU decompositions/transfers, preview, query scoring and serialization are paid offline costs; repeated layer/shared row clocks must not be summed.

Scope: Original locked full synthetic study at2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07. Forecast95.30s under predeclared600s engineering ceiling. All18 checkpoints, six histories and792 rows completed; no actual study attempt was dropped or failed. Uncertainty: Eight post-shift updates do not establish convergence/recovery. Native scaffold is learnability calibration, not competitive baseline tuning. Original worker clock excludes fresh-process startup and final-summary tail; endpoint resources miss peaks and cannot establish training memory. No tensor/corpus/weights redistribution.

- endpoint_Metal_driver: 1126924288 (bytes)
- endpoint_RSS: 597868544 (bytes)
- worker_research_wall: 60.959142667008564 (s)

Source: [analysis/structural_oracle_summary.json](structural_oracle_summary.json).

### Derived necessary-condition check at priced rank16

Status: measured. At priced rank16,0/48 held-out past proposals have positive exact lower margin a−B; among42 with positive a, min(B/a)=1.1028868. This diagnoses failure of a necessary condition, not a new selected threshold. Rank64 has2/48 tiny positive margins only beyond the declared grid. Do not infer no possible tau anywhere or no information. Rank4/64 at this structural shape lack matching measured runtimes; no rank-cost frontier or deployable paid-current-SVD pipeline exists.

Scope: Deduplicated simultaneous past evaluation proposals, using exact complement bound and raw alignment in retained raw rows. Only rank16 matches the previously measured structural-study operator shape. Uncertainty: Derived descriptive check only; actual numerical allowance remains unresolved. The original unfitted-grid statement is primary.

- rank16_min_B_over_positive_a: 1.1028868063716888 (ratio)
- rank16_positive_a: 42 (proposals)
- rank16_positive_lower_margin: 0 (of48 proposals)
- rank64_positive_lower_margin: 2 (of48 diagnostic proposals)

Source: [analysis/structural_oracle_audit.jsonl](structural_oracle_audit.jsonl).

confirmed learning: 0

unresolved: 5

### Numerical allowance and full execution certificate

Status: unresolved. NumericalPolicy remains unresolved.120 calibration/stress cases, including corrected observed stride metadata, do not provide formal machine or run-wide execution certification. Actual audit decisions reject unresolved/nonfinite margins. Extreme finite FP32 endpoints may still yield nonfinite stored displacement/alignment; fail-closed rejection is retained.

Scope: CPU/reference and bounded MPS component numerical evidence; wide block has timing evidence only. Uncertainty: No online acceptance or validated production allowance. Tiny near-zero finite MSE signs remain exploratory.


Source: [analysis/guardband_calibration.json](guardband_calibration.json).

### Unavailable native alternatives and nearest-art baselines

Status: blocked. Native default/single/foreach/fused AdamW controls were validated and the fastest used by timing pairs. Muon, activation-checkpointed full BP, INSTANT, DropBP and VCAS training ports were not executed. Upstream code/notices were inspected and archived; source pins do not establish implementation fidelity, equal tuning or competitiveness. INSTANT energy rescaling, calibration exposure and literal helper indexing remain port requirements. WikiText license/version conflict and dataset bodies remain unverified; no corpus ingestion occurred.

Scope: Discovery provenance and isolated AdamW operator controls only. The conditional Task7 licensing ruling was never used. Uncertainty: No superiority against unexecuted alternatives, language workload, strongest local training baseline or real-task frontier.


Source: [manifests/upstream_sources.json](../manifests/upstream_sources.json).

### Failed, contaminated and superseded timing trials retained

Status: failed. Initial serialization-failed pilot lost temporary raw blocks and supplies no gate evidence; its aggregate console history is unavailable as raw proof. An overlapping-collector pilot is retained and excluded. Earlier sequential/control-drift/replay-identification and coupled-initialization studies remain archived and superseded. The incorrect layout-label numerical artifact and intermediate correction are archived; final120-case calibration remains unresolved. Current seed-separated shared-round studies alone support latest WP02.

Scope: All historical exploratory operator/calibration branches; no favorable-prefix selection. Uncertainty: No missing raw pilot values are reconstructed or substituted. Resource telemetry is sampled and OS-wide pressure is not attributed to a process.


Source: [analysis/evidence_history.json](evidence_history.json).

### Deferred implementation and confirmation

Status: blocked. Tasks7–10 and12–16 remain skipped/deferred by the gate stop. Ordinary WP03 harness/baseline prerequisites are unfulfilled and full WP04 is inconclusive; WP05–WP08 are unexecuted. No trainer, data preparation, controller, online policy, frozen protocol, dropout/RNG-resume lifecycle, primary H4 test or independent confirmation was implemented. Only synthesis can complete on the stopped branch.

Scope: Original blueprint preserved unchanged; reached Tasks1–6 and bounded Task11 amendment are narrower than its full study plan. Uncertainty: Missing data remain unavailable; passing WP09 means documented synthesis, never scientific success.


Source: [docs/blueprint.md](../docs/blueprint.md).

### Initial reproduction postprocessing failure

Status: failed. Fresh actual-MPS full repeat succeeded, all792 row comparisons were exact, then the helper failed KeyError:step at curve postprocessing. This failed processing outcome and original plan are retained. The actual curve field is committed_updates; no failed scientific comparison is relabeled as success.

Scope: Repeat measured at committed7c0706e with independent output/checkpoint paths and exact clone hashb86fb64b..., original configa387445b.... Uncertainty: Postprocessing recovery is separately reported. A1.27s report-only CPU pytest may have overlapped; wall times are accounting only, not timing replication.


Source: [analysis/reproduction/wp09/comparison.json](reproduction/wp09/comparison.json).

## All gate outcomes, including superseded stops

### Gate 1: WP00 — pass (superseded record retained)

2026-10-30: actual native arm64 CPU/MPS tiny operations pass with general fallback disabled; exact wheels/source pins retained; pinned Fashion-MNIST MIT notices reviewed and training assets HEAD-available. WikiText license conflict blocks language; no corpus bytes downloaded or workload-ready qualification.

Next action: WP01 CPU FP64/numerical and transaction verification, then conditional MPS operator/headroom work. Verify dataset bytes before workload use; reconcile WikiText notices before language use. No training or speedup claim.

### Gate 2: WP00 — pass (superseded record retained)

2026-10-30 fix round 1: actual MPS checks reverified in a fresh worker with fallback=0 before any Torch import; caller environment text is not accepted as effective-policy evidence. CPU/MPS basics, frozen provenance and available pinned MIT Fashion-MNIST metadata remain qualified; WikiText remains blocked and corpus bytes undownloaded.

Next action: WP01 CPU FP64/numerical and transaction verification, then conditional MPS operator/headroom work. Verify dataset bytes before workload use; reconcile WikiText notices before language use. No training or speedup claim.

### Gate 3: WP01 — pass (superseded record retained)

2026-10-30: Tasks 3–5 tiny CPU FP64 / actual MPS FP32 projection, exact-adjoint, accumulated-sketch, independent-confidence and native AdamW preview/single-use commit reference tests pass; fresh fallback=0 full suite254 passed. Numerical policy remains unresolved for production shapes. This gate covers tested mathematics and optimizer transactions, not unimplemented trainer/full-batch replay or performance.

Next action: WP02 operator/headroom measurements and explicit shape-specific numerical calibration. Preserve existing dataset/workload blocks; qualify trainer replay separately in Tasks9/12. No training, online-acceptance, learning or speedup claim.

### Gate 4: WP00 — pass (superseded record retained)

2026-10-30 Task6 provenance maintenance only: normalize mislabeled repository_tree_sha commit IDs to repository_commit_sha, retaining upstream/source/license bytes and exact prior manifest snapshot. Existing actual fallback0 WP00 operation evidence and asset limitations are reused unchanged; no new discovery measurement or workload readiness claimed.

Next action: Continue only already-qualified operator/reference work; verify corpus bytes before workload use and reconcile WikiText notices before language use. WP02 governs subsequent cost-qualified branch permission.

### Gate 5: WP01 — pass (superseded record retained)

2026-10-30 Task6 maintenance: durable Task3/4 collectors actually regenerated bounded CPU FP64/MPS FP32 evidence (12 operator/6 basis cases and original confidence/cancellation diagnostics). Prior optimizer evidence reused unchanged. Actual fallback0 full suite280 passed in6.84s. Tested numerical/transaction scope only; NumericalPolicy remains unresolved and no trainer replay or learning claim is added.

Next action: Use only bounded operator cost/calibration evidence and WP02 decision for subsequent work. Production numerical acceptance and trainer/checkpoint lifecycle remain unqualified.

### Gate 6: WP02 — stop (superseded record retained)

2026-10-30 actual MPS fallback0 interleaved whole-block studies: stable candidate-only savings at n2048/512/2048 r16q8, n4096/512/2048 r16q8, and n8192/768/3072 r16q16 against common-window fastest validated native AdamW. Every complete conditional p0 attempt is slower; wide55.251ms vs36.709ms native, candidate28.315ms vs37.364ms. Explicit direct replay/refresh and aged journal costs retained. Preview/finite validation/CPU FP64 alignment and full-chain journal reparsing are named reference overhead, not established method lower bounds.120 stress cases do not resolve NumericalPolicy. This stops current online engineering, not the scientific idea. Actual280-test suite passes; system swap/pressure and unknown thermal limits remain explicit.

Next action: Prepare only a bounded synthetic structural-oracle information study with copied optimizer states and a predeclared cost/scope. Record an explicit narrow branch/protocol amendment before execution: normal WP03/WP04 prerequisites remain blocked by this stop. No full trainer/controller, data ingestion, online acceptance, larger-model or MLX retry without new evidence and amendment.

### Gate 7: WP00 — pass (superseded record retained)

2026-10-30 Task6 provenance maintenance only: normalize mislabeled repository_tree_sha commit IDs to repository_commit_sha, retaining upstream/source/license bytes and exact prior manifest snapshot. Existing actual fallback0 WP00 operation evidence and asset limitations are reused unchanged; no new discovery measurement or workload readiness claimed. Metadata/index hashes refreshed after explicitly archived numerical-layout correction; actual WP00 operation/asset evidence is unchanged.

Next action: Continue only already-qualified operator/reference work; verify corpus bytes before workload use and reconcile WikiText notices before language use. WP02 governs subsequent cost-qualified branch permission.

### Gate 8: WP01 — pass (superseded record retained)

2026-10-30 Task6 maintenance: durable Task3/4 collectors actually regenerated bounded CPU FP64/MPS FP32 evidence (12 operator/6 basis cases and original confidence/cancellation diagnostics). Prior optimizer evidence reused unchanged. Actual fallback0 full suite280 passed in6.84s. Tested numerical/transaction scope only; NumericalPolicy remains unresolved and no trainer replay or learning claim is added. Final layout correction: same120 numerical cases regenerated with24 observed/asserted stride-2 MPS inputs. Benchmark input initialization is behavior-equivalent and directly CPU/MPS tested; timed cost operations unchanged. Superseded source/artifact/index bytes are archived. Final actual-device suite282 passed in7.74s. NumericalPolicy remains unresolved.

Next action: Use only bounded operator cost/calibration evidence and WP02 decision for subsequent work. Production numerical acceptance and trainer/checkpoint lifecycle remain unqualified.

### Gate 9: WP02 — stop (superseded record retained)

2026-10-30 actual MPS fallback0 interleaved whole-block studies: stable candidate-only savings at n2048/512/2048 r16q8, n4096/512/2048 r16q8, and n8192/768/3072 r16q16 against common-window fastest validated native AdamW. Every complete conditional p0 attempt is slower; wide55.251ms vs36.709ms native, candidate28.315ms vs37.364ms. Explicit direct replay/refresh and aged journal costs retained. Preview/finite validation/CPU FP64 alignment and full-chain journal reparsing are named reference overhead, not established method lower bounds.120 stress cases do not resolve NumericalPolicy. This stops current online engineering, not the scientific idea. Actual280-test suite passes; system swap/pressure and unknown thermal limits remain explicit. Final layout correction: same120 numerical cases regenerated with24 observed/asserted stride-2 MPS inputs. Benchmark input initialization is behavior-equivalent and directly CPU/MPS tested; timed cost operations unchanged. Superseded source/artifact/index bytes are archived. Final actual-device suite282 passed in7.74s. NumericalPolicy remains unresolved.

Next action: Prepare only a bounded synthetic structural-oracle information study with copied optimizer states and a predeclared cost/scope. Record an explicit narrow branch/protocol amendment before execution: normal WP03/WP04 prerequisites remain blocked by this stop. No full trainer/controller, data ingestion, online acceptance, larger-model or MLX retry without new evidence and amendment.

### Gate 10: WP00 — pass (latest in snapshot)

2026-10-30 Task6 review fix1 provenance maintenance: existing WP00 operation/asset qualification unchanged; evidence index now preserves superseded coupled-init timing/source/derived bytes. No discovery rerun, asset ingestion or expanded permission.

Next action: Continue only already-qualified operator/reference work; verify corpus bytes before workload use and reconcile WikiText notices before language use. WP02 governs subsequent cost-qualified branch permission.

### Gate 11: WP01 — pass (latest in snapshot)

2026-10-30 Task6 review fix1: bounded Task3/4/5 numerical evidence reused unchanged, not recalibrated. Corrected timing-cycle arithmetic and transported separate benchmark construction/audit/data/model seeds; actual-device full suite286 passed in6.08s. NumericalPolicy remains unresolved; no trainer/learning qualification added.

Next action: Use only bounded operator cost/calibration evidence and WP02 decision for subsequent work. Production numerical acceptance and trainer/checkpoint lifecycle remain unqualified.

### Gate 12: WP02 — stop (latest in snapshot)

2026-10-30 Task6 review fix1: exact cycle threshold is(D-A-C/(K-1))/R forK>1; K1 has no approximate segment/threshold and K16 bootstrap uses15 approximate steps. Fresh fallback0 MPS workers reran only the same five main and one wide configurations with transported resolved construction61001/audit61002/data61003/model61004 streams. Initial model weights are independent of input shape/layout/data-stream consumption. Prior coupled-init raw/derived/source evidence is archived unchanged and explicitly labeled. Qualified candidate-only regimes remain n2048r16,n4096r16,wide n8192r16q16. Wide candidate28.875ms vsnative36.882ms; complete p0 attempt54.406ms vs36.738ms; forced reject91.401ms; direct replay37.366ms. Corrected K16 p*=-.492802,95%[-.535047,-.447323], no feasible rejection probability. Named reference validation/preview/CPU alignment/journal costs remain distinct from method lower bounds; NumericalPolicy unresolved. No new shapes/policy/optimization/calibration study.286 actual-device tests pass in6.08s; recorded swap/pressure/thermal limits qualify these within-study results.

Next action: Prepare only a bounded synthetic structural-oracle information study with copied optimizer states and a predeclared cost/scope. Record an explicit narrow branch/protocol amendment before execution: normal WP03/WP04 prerequisites remain blocked by this stop. No full trainer/controller, data ingestion, online acceptance, larger-model or MLX retry without new evidence and amendment.

### Gate 13: WP04 — inconclusive (superseded record retained)

Bounded synthetic-only amendment completed at committed measured source2e89e74:6 learnable native histories,792 layer rows,18 checkpoints. Ordinary WP03/full WP04 requirements unfulfilled. Past exact ordinary/complement margins accept0/144 held-out proposals at every declared tau; paid current-SVD/r64 complement permits2/54 at tau.75, both still less development progress than dense. Scores contain nontrivial correlated descriptive information, but fitted complement equals constant exact fallback and all past proposals lose development progress relative to dense. No population/no-information/language failure claim. Current WP02 full-cost stop and unresolved NumericalPolicy remain; no online permission.305 actual fallback0 CPU/MPS tests pass.

Next action: After Task11 review proceed to WP09 synthesis, retaining full acceptance envelope, per-seed/stage/rank/age results, limits and costs. No new experiment/grid/shape/controller/trainer/cost-remedy or online branch without separately authorized evidence and amendment.

### Gate 14: WP04 — inconclusive (superseded record retained)

Bounded synthetic-only amendment completed at committed measured source2e89e74:6 learnable native histories,792 layer rows,18 checkpoints. Ordinary WP03/full WP04 requirements unfulfilled. Past exact ordinary/complement margins accept0/144 held-out proposals at every declared tau; paid current-SVD/r64 complement permits2/54 at tau.75, both still less development progress than dense. Scores contain nontrivial correlated descriptive information, but fitted complement equals constant exact fallback and all past proposals lose development progress relative to dense. No population/no-information/language failure claim. Current WP02 full-cost stop and unresolved NumericalPolicy remain; no online permission.305 actual fallback0 CPU/MPS tests pass. Final constructed-control refinement: actual identity spectrum and60-degree query rotation now independently asserted; old fixtures fail strengthened checks.305 actual fallback0 tests pass in7.24s. No empirical collection/source change; original results source2e89e74 and artifact commitdc56170 preserved.

Next action: After Task11 review proceed to WP09 synthesis, retaining full acceptance envelope, per-seed/stage/rank/age results, limits and costs. No new experiment/grid/shape/controller/trainer/cost-remedy or online branch without separately authorized evidence and amendment.

### Gate 15: WP04 — inconclusive (superseded record retained)

Bounded synthetic-only amendment completed at committed measured source2e89e74:6 learnable native histories,792 layer rows,18 checkpoints. Ordinary WP03/full WP04 requirements unfulfilled. Past exact ordinary/complement margins accept0/144 held-out proposals at every declared tau; paid current-SVD/r64 complement permits2/54 at tau.75, both still less development progress than dense. Scores contain nontrivial correlated descriptive information, but fitted complement equals constant exact fallback and all past proposals lose development progress relative to dense. No population/no-information/language failure claim. Current WP02 full-cost stop and unresolved NumericalPolicy remain; no online permission.305 actual fallback0 CPU/MPS tests pass. Final constructed-control refinement: actual identity spectrum and60-degree query rotation now independently asserted; old fixtures fail strengthened checks.305 actual fallback0 tests pass in7.24s. No empirical collection/source change; original results source2e89e74 and artifact commitdc56170 preserved. Report-only header correction states current completed bounded-study/inconclusive-online-stop status; no implementation or empirical change.

Next action: After Task11 review proceed to WP09 synthesis, retaining full acceptance envelope, per-seed/stage/rank/age results, limits and costs. No new experiment/grid/shape/controller/trainer/cost-remedy or online branch without separately authorized evidence and amendment.

### Gate 16: WP04 — inconclusive (latest in snapshot)

Bounded synthetic-only amendment completed at committed measured source2e89e74:6 learnable native histories,792 layer rows,18 checkpoints. Ordinary WP03/full WP04 requirements unfulfilled. Past exact ordinary/complement margins accept0/144 held-out proposals at every declared tau; paid current-SVD/r64 complement permits2/54 at tau.75, both still less development progress than dense. Scores contain nontrivial correlated descriptive information, but fitted complement equals constant exact fallback and all past proposals lose development progress relative to dense. No population/no-information/language failure claim. Current WP02 full-cost stop and unresolved NumericalPolicy remain; no online permission.305 actual fallback0 CPU/MPS tests pass. Final constructed-control refinement: actual identity spectrum and60-degree query rotation now independently asserted; old fixtures fail strengthened checks.305 actual fallback0 tests pass in7.24s. No empirical collection/source change; original results source2e89e74 and artifact commitdc56170 preserved. Report-only header correction states current completed bounded-study/inconclusive-online-stop status; no implementation or empirical change. Task11 fix1: register attempts before native bootstrap and preserve partial curves/checkpoint hashes, last confirmed committed/source counters, failure reason/status and elapsed native cost on internal budget/numerical failures. Two real three-update tiny CPU regressions pass. Complete report moved to decisions/reports/structural_oracle.md; scratch untracked but ignored working copy retained.307 actual fallback0 CPU/MPS tests pass in7.84s. Later fix code/tests/docs are not measured research; study source2e89e74 and raw artifacts unchanged.

Next action: After Task11 review proceed to WP09 synthesis, retaining full acceptance envelope, per-seed/stage/rank/age results, limits and costs. No new experiment/grid/shape/controller/trainer/cost-remedy or online branch without separately authorized evidence and amendment.

No recorded gate outcome: WP03, WP05, WP06, WP07, WP08. Absence never implies completion.

## Outcome decision and challenge audits

Blueprint §14: stop the current online branch where a prerequisite is stopped or inconclusive; narrow conclusions to reached measurements. A component benefit alone does not establish learning efficiency. Wide intervals are inconclusive, not equivalence. Continue/revise/pivot requires the applicable evidence and a new specification; synthesis grants no experimental permission.

### Scientific audit

Blueprint§15 challenge repeated: originality is not certified; prior subspace/compression mechanisms overlap known work. Tested actual AdamW displacement can be harmful despite high cosine; ordinary/complement bounds retain information but fail actionable declared past-grid acceptance. Past/current-SVD oracles remain distinct; current exact gradients/SVD are paid diagnostics. Exact incoming adjoints constrain headroom. First-order minibatch usefulness, finite MSE updates and generalization are distinct. Rare-coordinate controls demonstrate a possible omitted direction, not rare-pattern population prevalence. Ordinary SGD can make auditing redundant; no broad optimizer superiority or universal past-reuse failure is claimed.

Sources: [docs/blueprint.md](../docs/blueprint.md), [analysis/optimizer_correctness.json](optimizer_correctness.json), [decisions/reports/structural_oracle.md](../decisions/reports/structural_oracle.md).

### Experimental audit

Blueprint§15 challenge repeated: missing/sandbox-hidden MPS is explicitly distinguished from actual device evidence; hardware-only marker suite was run. Low memory is screened and sampled, not a full-training OOM proof. Missing corpus blocks language claims. All approximate audit attempts reject under unresolved NumericalPolicy; the exact permitted past-grid rejects all, and forced conditional commit is pricing only. Independent RNG domains, tuning/evaluation/development query IDs and pre-query scope/LR lock are checked, but only two seeds with duplicated early stages and correlated proposals limit inference. Nonzero dropout, real rare-pattern generalization, ordinary-SGD training, Muon beating both AdamW methods, positive-synthetic/negative-real transfer, and resumed full-run RNG/counter consistency are unexecuted. No fixed-time censoring or paired independent-run statistics were invented.

Sources: [docs/blueprint.md](../docs/blueprint.md), [analysis/README.md](README.md), [analysis/structural_oracle_schema.json](structural_oracle_schema.json), [tests/test_oracle.py](../tests/test_oracle.py).

### Implementation audit

Blueprint§15 challenge repeated: exact adjoints/streaming accumulation, zero-versus-missing gradients with weight decay, duplicate/shared-parameter exclusions, native preview, mutation invalidation and journal crashes have component tests. Actual hardware shapes and explicit CPU decomposition/transfer costs were measured. Preview retains32P rather than approximate28P. Timed reference validation/alignment/journal bottlenecks are not irreducible lower bounds. Full scheduler/data cursor replay, checkpoint hooks, dropout RNG restoration, whole-model interruption/resume and full-training memory remain gated, not passed. Extreme finite FP32 endpoint overflow remains a disclosed fail-closed limitation. No finite-precision certificate is claimed.

Sources: [tests/test_optimizer_preview.py](../tests/test_optimizer_preview.py), [tests/test_audit_journal.py](../tests/test_audit_journal.py), [tests/test_accumulation.py](../tests/test_accumulation.py), [tests/test_exact_adjoints.py](../tests/test_exact_adjoints.py), [analysis/headroom_decision.json](headroom_decision.json), [analysis/wp09_verification.json](wp09_verification.json).

### Decision audit

Blueprint§14 outcome: stop the present online branch and narrow the finding to component headroom plus complete-cost failure and an inconclusive/negative actionable policy region in this bounded synthetic grid. Correct mathematics alone and positive component timing do not establish practical learning efficiency. Current-SVD diagnostic improvement is paid and cannot reopen online work. Wide intervals, absent guardrails, unavailable baselines and lack of independent confirmation are unresolved, not equivalence or universal failure. No performance redesign, alternate basis, architecture pivot, MLX retry or corpus branch is authorized; any pivot requires a new specification and evidence. Rulings retain superseded/unused decisions and their practical rework costs; final whole-branch review remains with the controller. The hash-bound review_summary.md and outcome.json are pre-final review snapshots. README links to the durable final review/rulings addendum, deliberately outside this scientific index and WP09 artifact hashes; scoped re-review is pending there.

Sources: [docs/blueprint.md](../docs/blueprint.md), [decisions/rulings.md](../decisions/rulings.md), [decisions/review_summary.md](../decisions/review_summary.md), [analysis/headroom_decision.json](headroom_decision.json), [decisions/outcome.json](../decisions/outcome.json).

### Reproduction audit

Local commands, pinned arm64 environment, source/hash resolution and independent-path reproduction are documented in README and synthesis_schema. Original artifacts were not overwritten. Prepare writes the full path-only amendment/policy before execution; run validates hashes and appends/restores the previous exact permission. Initial comparison failure is retained; postprocessing-only step→committed_updates recovery uses identical observations/tolerances and compares all792 rows/30 curves, not a favorable subset. No benchmark was repeated because no timed kernel changed. Large trusted local checkpoints are ignored and regenerable; no corpus/pretrained weights are required or redistributed. Original study, historical source revisions and later review fixes retain separate attribution.

Sources: [README.md](../README.md), [decisions/reports/synthesis_schema.md](../decisions/reports/synthesis_schema.md), [analysis/reproduction/wp09/plan.json](reproduction/wp09/plan.json), [analysis/reproduction/wp09/comparison.json](reproduction/wp09/comparison.json), [analysis/reproduction/wp09/comparison_recovered.json](reproduction/wp09/comparison_recovered.json).

### Implementation audit

Final review fixes at7628be7: unsupported internally overlapping writable parameter layouts reject before pending publication or any commit write; supported transposes and gapped slices retain native-equivalent updates. One absolute oracle study deadline covers later native histories, persisted audit rows, summary completion and final synchronization, retaining actual partial progress as incomplete/inconclusive. Normal/recovery checkpoint reproduction shares identity/duplicate/cardinality/hash validation. Public preview API documents finite-endpoint/nonfinite-derived metadata and fail-closed consumer requirement. Focused115 tests passed; single full actual-MPS fallback0 suite364 passed in7.80s with no skips/warnings. No benchmark/research rerun: original2e89e74 study and timing source hashes remain authoritative for measured bytes, not changed validation overhead. Scoped re-review was pending at that maintenance point; the README-linked coordination addendum now records acceptance on 2026-10-31 (durable acceptance committed at f001b2b) and remains outside scientific evidence hashes.

Sources: [src/oar/optimizer_preview.py](../src/oar/optimizer_preview.py), [src/oar/oracle.py](../src/oar/oracle.py), [analysis/reproduce_structural_oracle.py](reproduce_structural_oracle.py), [tests/test_optimizer_preview.py](../tests/test_optimizer_preview.py), [tests/test_oracle.py](../tests/test_oracle.py), [tests/test_reproduction.py](../tests/test_reproduction.py), [analysis/wp09_verification.json](wp09_verification.json).

### Reproduction audit

Publication maintenance only: concise README, detailed reproduction instructions, original-material license/citation scope and package metadata. Historical measured bytes remain attributed to their recorded revisions; added publication bindings do not establish new research or remeasure later validation changes. Full Git history is required for historical artifact verification.

Sources: [README.md](../README.md), [REPRODUCE.md](../REPRODUCE.md), [pyproject.toml](../pyproject.toml), [LICENSE](../LICENSE), [licenses/CC-BY-4.0.txt](../licenses/CC-BY-4.0.txt), [NOTICE](../NOTICE), [LICENSING.md](../LICENSING.md), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), [CITATION.cff](../CITATION.cff), [docs/releases/v0.1.0.md](../docs/releases/v0.1.0.md), [third_party/discovery/pytorch/LICENSE](../third_party/discovery/pytorch/LICENSE).

## Raw artifact and historical hash index

Current artifacts are checked from disk; historical revisions are checked using `git show REV:PATH`. Explicitly unavailable historical artifacts stay unavailable. No index/report/WP09 hash cycle is used.

- `.superpowers/sdd/2026-10-30-oar-implementation/task-11-report.md`: `62994a68534d278c80f078d913488a242ee913d01d93f91c978eef8c02fd0fc5` (unavailable: Historical disposable workflow report; excluded from current scientific support. Durable Task11 report is decisions/reports/structural_oracle.md.)
- `.superpowers/sdd/2026-10-30-oar-implementation/task-11-report.md`: `cafffbb259431a34f5e3433f06c2e94aa025efd5e19d0ca3f15415765bf03b33` (unavailable: Historical disposable workflow report; excluded from current scientific support. Durable Task11 report is decisions/reports/structural_oracle.md.)
- `.superpowers/sdd/2026-10-30-oar-implementation/task-11-report.md`: `d29597eb70f68b1dda3515dada02ce29b6d4d9f28c2775591e5f20b8aee15d0e` (unavailable: Historical disposable workflow report; excluded from current scientific support. Durable Task11 report is decisions/reports/structural_oracle.md.)
- [CITATION.cff](../CITATION.cff): `50b85511a02c17f184fda76c163e48beffc6360735ca5f18880620654178ef3c` (verified current bytes)
- [LICENSE](../LICENSE): `0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0` (verified current bytes)
- [LICENSING.md](../LICENSING.md): `1978932693122396b386f3f10beacba838da8dcb19e9b95cc9a7fb35e47939eb` (verified current bytes)
- [NOTICE](../NOTICE): `607e70a1d5190a926360e62f004eeeb7754fc58c1752aa124b1e4081aea0095f` (verified current bytes)
- [README.md](../README.md): `981aaa19db4a093bd3d607039281eadcf0fa84e09c57727299abd4a5ccc43544` (verified current bytes)
- [REPRODUCE.md](../REPRODUCE.md): `5b159cad2c5b93e4c768d73120edffe4ac327ad3b2ddb22667f299aba32c9ba8` (verified current bytes)
- [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md): `560068932adb815b2712287d7de8008c463eebfe1b6e21e35e26d8f1f983ed64` (verified current bytes)
- `analysis/README.md`: `0ab213582ff1b123bed16f12ba2f84bb97caadd1bd0383012ab265a08b1ef348` (verified historical revision dc561706752f656706c4fac5ae9b9f1f9c53b601)
- [analysis/README.md](README.md): `130eae9c807339feaf0ab121c3aad5e92bba07df259491711adb0b2311ef2b5f` (verified current bytes)
- `analysis/README.md`: `2e6b5100d97868a8e59dda5d5e1822d505b1246a76ac95117b0e1ecafa628f3d` (verified archive analysis/history/fix1-README-2e6b5100d978.md)
- `analysis/README.md`: `95de31ab2d977d53dbebd38a02b59d0542904490245e129a1ccf76dd7600826c` (verified archive analysis/history/README-before-layout-95de31ab2d97.md)
- `analysis/README.md`: `def28f72eb5fa1b87964c5804416ba10ceb40bfb6e971ab58bb186bd30147510` (verified historical revision 6f04c6eea4d4aab0f1df31dfe6e1361275c91666)
- `analysis/audit_confidence.json`: `e1c6e2252a0d5ad1b9f3098ebdc61883a4355a4f9c800cdd47ef1a69858052f7` (verified archive analysis/history/audit_confidence-e1c6e2252a0d.json)
- [analysis/audit_confidence.json](audit_confidence.json): `fa96b975a2d0128067fb9a623c31191033124a5cc6b6d2e48b1739a72aa84a95` (verified current bytes)
- [analysis/collect_audit_confidence.py](collect_audit_confidence.py): `95e83e9cee959f3c18036f55a8d4cc830cd92f1c9159db20fdac5879b418f3e5` (verified current bytes)
- [analysis/collect_numerical_calibration.py](collect_numerical_calibration.py): `797515951845645f1b12e01d788b6ce76b2ebcb95916ffe3e275b46618b3f3f0` (verified current bytes)
- `analysis/collect_numerical_calibration.py`: `9d52f632e1aad74e73a22595d838e2995698df3b9020e03d5da9e48db4575e82` (verified archive analysis/history/collect_numerical_calibration-before-layout-9d52f632e1aa.py)
- [analysis/collect_operator_correctness.py](collect_operator_correctness.py): `e69ba75c227c808226d0e3d619ec90bbf25b87e22bf020ef66d8bf59db405cf5` (verified current bytes)
- [analysis/collect_optimizer_correctness.py](collect_optimizer_correctness.py): `acae516d3300760f4097ea9c3a55a12f98f7df0e53c18c3918ff944f6bbda93a` (verified current bytes)
- [analysis/collect_synthesis.py](collect_synthesis.py): `4a4d3ffb8e06947fea58b3dba7a9f6086688078d1697e70f383981e73a24090c` (verified current bytes)
- [analysis/collector_manifest.json](collector_manifest.json): `6720db8392db71ecfedf3db1ee2ddd41fe90efa96e04bb033ed5712427a35baf` (verified current bytes)
- `analysis/collector_manifest.json`: `82c5b13aa1a2ca0c612f9e2782c516fb78b252f1489ec9252169357eaf2a26dd` (verified archive analysis/history/fix1-collector_manifest-82c5b13aa1a2.json)
- `analysis/collector_manifest.json`: `c94e3d7189aecd1762898fd8ed21c0490ca2bf42bc08ff0c9ef66b82040356f0` (verified archive analysis/history/collector_manifest-before-layout-c94e3d7189ae.json)
- `analysis/evidence_history.json`: `32dcdd7a2a01688f4d10c95c5f3590356af96b26503f3b32333b941372aa6712` (verified archive analysis/history/fix1-evidence_history-32dcdd7a2a01.json)
- [analysis/evidence_history.json](evidence_history.json): `3c1f20b20f93682330ba0992e930e2165815ef442b92de30d11f01c4944f7894` (verified current bytes)
- `analysis/evidence_history.json`: `53c7cfcab0a5edfa66f841c2073e7d2d5b9b93593603d4147008e75ea5e0a860` (verified archive analysis/history/evidence-index-before-layout-53c7cfcab0a5.json)
- [analysis/guardband_calibration.json](guardband_calibration.json): `24d5455bb39376610c2c0767a2ceb0744b741347dae37a5d399e12e6556ff387` (verified current bytes)
- `analysis/guardband_calibration.json`: `c8e621abe6f856a36f4f902e1f4130063136b8ee09e1a983f9c4cbbadcae333f` (verified archive analysis/history/guardband-before-layout-correction-c8e621abe6f8.json)
- [analysis/headroom_decision.json](headroom_decision.json): `490688ee41700aa87adb1d627ecb6e1ff526a66b8a3fe797b22755d11555a059` (verified current bytes)
- `analysis/headroom_decision.json`: `f1c74cd93b48bccbeff5026b064d21e7838317aca90fd9b7d4464304597f5af4` (verified archive analysis/history/fix1-headroom_decision-f1c74cd93b48.json)
- [analysis/headroom_summary.csv](headroom_summary.csv): `3311f81cb50f4d1763227af08b1d937b5da40d31dd687999b92e221acef90f61` (verified current bytes)
- `analysis/headroom_summary.csv`: `a9c559b0f18b547c44d62f20ae9856641549af52341d80981ef8d294dec24dc7` (verified archive analysis/history/fix1-headroom_summary-a9c559b0f18b.csv)
- [analysis/history/README-before-layout-95de31ab2d97.md](history/README-before-layout-95de31ab2d97.md): `95de31ab2d977d53dbebd38a02b59d0542904490245e129a1ccf76dd7600826c` (verified current bytes)
- [analysis/history/audit_confidence-e1c6e2252a0d.json](history/audit_confidence-e1c6e2252a0d.json): `e1c6e2252a0d5ad1b9f3098ebdc61883a4355a4f9c800cdd47ef1a69858052f7` (verified current bytes)
- [analysis/history/bench-before-layout-f2d2016eee1b.py](history/bench-before-layout-f2d2016eee1b.py): `f2d2016eee1b67f8147b96755d4fe534311b4c57729888f46bed839dfccefb22` (verified current bytes)
- [analysis/history/bench-measured-213397e0c7d3.py](history/bench-measured-213397e0c7d3.py): `213397e0c7d3c23b9446bc809453d558c73f0aa16a3dc1c125eb508b53a52ec8` (verified current bytes)
- [analysis/history/collect_numerical_calibration-before-layout-9d52f632e1aa.py](history/collect_numerical_calibration-before-layout-9d52f632e1aa.py): `9d52f632e1aad74e73a22595d838e2995698df3b9020e03d5da9e48db4575e82` (verified current bytes)
- [analysis/history/collector_manifest-before-layout-c94e3d7189ae.json](history/collector_manifest-before-layout-c94e3d7189ae.json): `c94e3d7189aecd1762898fd8ed21c0490ca2bf42bc08ff0c9ef66b82040356f0` (verified current bytes)
- [analysis/history/evidence-index-before-layout-53c7cfcab0a5.json](history/evidence-index-before-layout-53c7cfcab0a5.json): `53c7cfcab0a5edfa66f841c2073e7d2d5b9b93593603d4147008e75ea5e0a860` (verified current bytes)
- [analysis/history/fix1-README-2e6b5100d978.md](history/fix1-README-2e6b5100d978.md): `2e6b5100d97868a8e59dda5d5e1822d505b1246a76ac95117b0e1ecafa628f3d` (verified current bytes)
- [analysis/history/fix1-bench-4504126115b6.py](history/fix1-bench-4504126115b6.py): `4504126115b6bd0515a7c47e6882ce724c185a69dea0ec1f3785577fd8bed43f` (verified current bytes)
- [analysis/history/fix1-collector_manifest-82c5b13aa1a2.json](history/fix1-collector_manifest-82c5b13aa1a2.json): `82c5b13aa1a2ca0c612f9e2782c516fb78b252f1489ec9252169357eaf2a26dd` (verified current bytes)
- [analysis/history/fix1-evidence_history-32dcdd7a2a01.json](history/fix1-evidence_history-32dcdd7a2a01.json): `32dcdd7a2a01688f4d10c95c5f3590356af96b26503f3b32333b941372aa6712` (verified current bytes)
- [analysis/history/fix1-headroom_decision-f1c74cd93b48.json](history/fix1-headroom_decision-f1c74cd93b48.json): `f1c74cd93b48bccbeff5026b064d21e7838317aca90fd9b7d4464304597f5af4` (verified current bytes)
- [analysis/history/fix1-headroom_summary-a9c559b0f18b.csv](history/fix1-headroom_summary-a9c559b0f18b.csv): `a9c559b0f18b547c44d62f20ae9856641549af52341d80981ef8d294dec24dc7` (verified current bytes)
- [analysis/history/fix1-summarize_headroom-28af5349c413.py](history/fix1-summarize_headroom-28af5349c413.py): `28af5349c41379aaf7b897117354a3fd0a867f9643b8413e2155c513650ee69c` (verified current bytes)
- [analysis/history/fix1-test_bench_accounting-83c112ad2c89.py](history/fix1-test_bench_accounting-83c112ad2c89.py): `83c112ad2c890ea4ab8df8cbc46d9b8c62fab757f56415a1d7350ebeaa274ef8` (verified current bytes)
- [analysis/history/fix1-timing_calibration-7da3fd8be160.csv](history/fix1-timing_calibration-7da3fd8be160.csv): `7da3fd8be160593603008c480fd92b599bf19fa63ee11abd8607cfcba848fe39` (verified current bytes)
- [analysis/history/fix1-timing_calibration-fbdc4962d5ec.json](history/fix1-timing_calibration-fbdc4962d5ec.json): `fbdc4962d5ec208db855f61ca1bf5ef06f2e5d8e2fe7e8082267beb0a09c195b` (verified current bytes)
- [analysis/history/fix1-timing_calibration_wide-9200e6ff9944.csv](history/fix1-timing_calibration_wide-9200e6ff9944.csv): `9200e6ff994445fb951ee3419ad3e711fa9f787986560b697e019cf4fb398ca3` (verified current bytes)
- [analysis/history/fix1-timing_calibration_wide-95ca861c643a.json](history/fix1-timing_calibration_wide-95ca861c643a.json): `95ca861c643a70408140f7fefafc263a75958ced729947fb239e3e94bdba630e` (verified current bytes)
- [analysis/history/guardband-before-layout-correction-c8e621abe6f8.json](history/guardband-before-layout-correction-c8e621abe6f8.json): `c8e621abe6f856a36f4f902e1f4130063136b8ee09e1a983f9c4cbbadcae333f` (verified current bytes)
- [analysis/history/guardband-layout-before-stride-record-715c8a8e8bcf.json](history/guardband-layout-before-stride-record-715c8a8e8bcf.json): `715c8a8e8bcf4ba5939b99bc6628a20b884809678c2cc10585c41cef1ce0b276` (verified current bytes)
- [analysis/history/operator_correctness-d25265d9d4ff.json](history/operator_correctness-d25265d9d4ff.json): `d25265d9d4ff55f9a5a957f2ddd92a21b2ea96263466dfcb4eede3ef72f621fe` (verified current bytes)
- [analysis/history/test_bench_accounting-before-layout-16cf18886bd4.py](history/test_bench_accounting-before-layout-16cf18886bd4.py): `16cf18886bd425055771b309cd720cc60b5909494eff7719c72bf11b4fe66d83` (verified current bytes)
- [analysis/history/timing-clean-before-replay-refinement-24a9da72ee37.json](history/timing-clean-before-replay-refinement-24a9da72ee37.json): `24a9da72ee371072f757b1717906c038d0ea3a3d16d25e15505990307e725700` (verified current bytes)
- [analysis/history/timing-pilot-overlap-6c71818fb989.json](history/timing-pilot-overlap-6c71818fb989.json): `6c71818fb9897e3d1b61411a92f851d84a5a01a0eb39ffafe0791dc8ef3722fb` (verified current bytes)
- [analysis/history/upstream_sources-4d6c586c9863.json](history/upstream_sources-4d6c586c9863.json): `4d6c586c986334bb0bca23f08cc4a796d0d69f985b636484cfceac228e56f3ce` (verified current bytes)
- [analysis/history/wp09-initial-evidence_index-0d54bb4ce15e.json](history/wp09-initial-evidence_index-0d54bb4ce15e.json): `0d54bb4ce15eff245d71de9c5d42ac8450473f519db80930710b76207d2ab768` (verified current bytes)
- [analysis/history/wp09-initial-report-8d531c846e95.md](history/wp09-initial-report-8d531c846e95.md): `8d531c846e95c593f3854b5f7fe49c2265b800ae042e9ee5af2fa89ccd180f6d` (verified current bytes)
- [analysis/history/wp09-initial-synthesis_annotations-44407ae27dbf.json](history/wp09-initial-synthesis_annotations-44407ae27dbf.json): `44407ae27dbfa37bb1b63f2d6996ea5d270c3289f73a5481d121d1825e1186ba` (verified current bytes)
- [analysis/history/wp09-initial-synthesis_schema-11d452f6a029.md](history/wp09-initial-synthesis_schema-11d452f6a029.md): `11d452f6a02980c03b45ce9c1a96625fb67b380e72c9dd00136be29bc68a7e76` (verified current bytes)
- [analysis/history/wp09-initial-wp09_verification-677d37a41f09.json](history/wp09-initial-wp09_verification-677d37a41f09.json): `677d37a41f09da512947c7df6a40ab596103dee8cc74a2400ae1b608a1e8f791` (verified current bytes)
- [analysis/operator_correctness.json](operator_correctness.json): `ae092c3e4c66ca168f8e5e29b2982ee9c489565443c7142238a5087f30f7f044` (verified current bytes)
- `analysis/operator_correctness.json`: `d25265d9d4ff55f9a5a957f2ddd92a21b2ea96263466dfcb4eede3ef72f621fe` (verified archive analysis/history/operator_correctness-d25265d9d4ff.json)
- [analysis/optimizer_correctness.json](optimizer_correctness.json): `2f378ad2355b64f380e385ef6641df17bc01234862dff6ced9167b5da69c56f4` (verified current bytes)
- [analysis/reproduce_structural_oracle.py](reproduce_structural_oracle.py): `961d7dcd81f2a81abff72aa385d9962af733d82cf84a8970ed38ec79248f83fa` (verified current bytes)
- [analysis/reproduction/wp09/amendment.json](reproduction/wp09/amendment.json): `ccc6e9ddfa270fc7133f12824635409ee01f98b5d5bc6fad1affa78077b45894` (verified current bytes)
- [analysis/reproduction/wp09/comparison.json](reproduction/wp09/comparison.json): `d1633354a42e6985a8cdffd3ce0def70e3716dcff3d574e75a5a6e7306f1fe11` (verified current bytes)
- [analysis/reproduction/wp09/comparison_recovered.json](reproduction/wp09/comparison_recovered.json): `bc4290cb79b809f73787a04bd1e4d7103f43837108e4ca463dab1f5307e22cff` (verified current bytes)
- [analysis/reproduction/wp09/config.yaml](reproduction/wp09/config.yaml): `b86fb64b3e7f26d96548d1c2e7cdad0d912126060c99c70e2e9842d409ebad5d` (verified current bytes)
- [analysis/reproduction/wp09/plan.json](reproduction/wp09/plan.json): `d278cba861a40835f72a4c7697cbb3a65d863ebf8b97e6c53b64471dff077c92` (verified current bytes)
- [analysis/reproduction/wp09/structural_oracle_audit.jsonl](reproduction/wp09/structural_oracle_audit.jsonl): `7f8398a2699c7951fc9978b46db3927a36658187cfd004eb8850b862bc43a075` (verified current bytes)
- [analysis/reproduction/wp09/structural_oracle_schema.json](reproduction/wp09/structural_oracle_schema.json): `b702c98a720f0932954f575f1893769b3860b474921b872e0e87ecc8405af524` (verified current bytes)
- [analysis/reproduction/wp09/structural_oracle_summary.json](reproduction/wp09/structural_oracle_summary.json): `f9f2fac192c011f2f2c9944d49cccda47deaa35a6d10a90c6ab4e06cb3ab2b56` (verified current bytes)
- [analysis/resource_context.json](resource_context.json): `e389ac6bad68c2252c2a6ba27de9dd75143c4e0d9cbefd4a7352482566bcd5d5` (verified current bytes)
- [analysis/scientific_gate_snapshot.json](scientific_gate_snapshot.json): `62c3af64f199cb398c6e24e1cc6655f2c9b8b9e90a3bad2aa684bd60d8a6cf32` (verified current bytes)
- [analysis/structural_oracle_aggregate.json](structural_oracle_aggregate.json): `90c4bcba1138a07a01c2bd6295de8846cafa02f678e9f8ecd17cf886a91d54b8` (verified current bytes)
- [analysis/structural_oracle_audit.jsonl](structural_oracle_audit.jsonl): `2c7c4374a58037a27d2ae4b76ee799089328690b07e7c2194ca3317e8f70c518` (verified current bytes)
- [analysis/structural_oracle_checkpoint_index.json](structural_oracle_checkpoint_index.json): `f1ac6266d158869acbd7db78e71a42b1f974d0a0169fefbd5d847bf59a353226` (verified current bytes)
- [analysis/structural_oracle_learning.csv](structural_oracle_learning.csv): `b3a37a6c4be6a23d5377da98125c867a5cabe4b332b085f8e5cc196768bd7925` (verified current bytes)
- [analysis/structural_oracle_pilot.json](structural_oracle_pilot.json): `6e727d64b54a52f560936a9cc1b64f704ff94669d8889cbe5b748734386efc18` (verified current bytes)
- [analysis/structural_oracle_schema.json](structural_oracle_schema.json): `b702c98a720f0932954f575f1893769b3860b474921b872e0e87ecc8405af524` (verified current bytes)
- [analysis/structural_oracle_summary.json](structural_oracle_summary.json): `df45a00eb921a4a55594a18de155b24eb774a27f10fb0f81d2fbd5257d5f7558` (verified current bytes)
- `analysis/summarize_headroom.py`: `28af5349c41379aaf7b897117354a3fd0a867f9643b8413e2155c513650ee69c` (verified archive analysis/history/fix1-summarize_headroom-28af5349c413.py)
- [analysis/summarize_headroom.py](summarize_headroom.py): `5cf53dbeef8772dbc3520de926d0f2f583d16561a9610d93fd5ca41d9b88aa4a` (verified current bytes)
- [analysis/summarize_structural_oracle.py](summarize_structural_oracle.py): `fcf77c34f1fcf4f9dcd7e0d5d0dc874e2e77091c54f2d4a6ace73f083fd376e9` (verified current bytes)
- [analysis/timing_calibration.csv](timing_calibration.csv): `2bfa5633fddf952888e21d43f914b9acbb5de78b02bd8392955c18d4d4ae04e6` (verified current bytes)
- `analysis/timing_calibration.csv`: `7da3fd8be160593603008c480fd92b599bf19fa63ee11abd8607cfcba848fe39` (verified archive analysis/history/fix1-timing_calibration-7da3fd8be160.csv)
- [analysis/timing_calibration.json](timing_calibration.json): `66f698a25c2ca414bb6eca13e541a70dc8ec6ede70cc77ae2831cc5e05f9c0fa` (verified current bytes)
- `analysis/timing_calibration.json`: `fbdc4962d5ec208db855f61ca1bf5ef06f2e5d8e2fe7e8082267beb0a09c195b` (verified archive analysis/history/fix1-timing_calibration-fbdc4962d5ec.json)
- [analysis/timing_calibration_wide.csv](timing_calibration_wide.csv): `2c368ba42588829939b9fa461855963cf7ef3c7426565fd2e666fdc909b13d10` (verified current bytes)
- `analysis/timing_calibration_wide.csv`: `9200e6ff994445fb951ee3419ad3e711fa9f787986560b697e019cf4fb398ca3` (verified archive analysis/history/fix1-timing_calibration_wide-9200e6ff9944.csv)
- [analysis/timing_calibration_wide.json](timing_calibration_wide.json): `04a0157a7c5c0b4821a1ebabdb157237e185e21a154199f3d52dad38def2b0e6` (verified current bytes)
- `analysis/timing_calibration_wide.json`: `95ca861c643a70408140f7fefafc263a75958ced729947fb239e3e94bdba630e` (verified archive analysis/history/fix1-timing_calibration_wide-95ca861c643a.json)
- [analysis/wp09_verification.json](wp09_verification.json): `697516d857645b8aaf407f2ba88615619f32def41801289e162f381f01afe31d` (verified current bytes)
- [configs/discovery.yaml](../configs/discovery.yaml): `0fa914fb68e6b48eaf2e19a5f3b1aefcbf5189e5860d28f2349969994177d99d` (verified current bytes)
- [configs/operator.yaml](../configs/operator.yaml): `fc6caa711980c09d315bc205cf4cb121e662bc02659bc916702cc62bf7b4ffcb` (verified current bytes)
- [configs/operator_wide.yaml](../configs/operator_wide.yaml): `1ea9e3ca5ffadd83316e9d1240896c0bab2e7897b9af4bafc4c61159dfe9f400` (verified current bytes)
- [configs/structural_oracle.yaml](../configs/structural_oracle.yaml): `a387445beda9eeb81f4d2107cfa3c457fb89b3399c5684035d6c5fb5990c169b` (verified current bytes)
- `decisions/amendments.jsonl`: `6637c152543e95724340301e413bdc533371b2cc35451965bdf9d22aee91454e` (verified historical revision 2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07)
- [decisions/amendments.jsonl](../decisions/amendments.jsonl): `78ae5c0c6d1daa0e819e77da889553efa72547693451b8cfbf9627c6f2e23bda` (verified current bytes)
- [decisions/outcome.json](../decisions/outcome.json): `d06b25e1b475d1527f9f8a4e9a618bae20f82ad3c48abcd131e3847f37fb4aad` (verified current bytes)
- [decisions/reports/structural_oracle.md](../decisions/reports/structural_oracle.md): `09abcc2e5b84aab8fea0e7bbb6a1eda5388ad3e2d3fa64b05516c04b4793ff65` (verified current bytes)
- [decisions/reports/synthesis_annotations.json](../decisions/reports/synthesis_annotations.json): `d0b4510349e5b15e6b09fd16173c2466bb367d8793ba61fa838f72701afec86f` (verified current bytes)
- [decisions/reports/synthesis_schema.md](../decisions/reports/synthesis_schema.md): `657573fd3122d3b66e08fb09a67b699b370dc041585bc7757f393d3e4a3c3753` (verified current bytes)
- [decisions/review_summary.md](../decisions/review_summary.md): `ffef9af52cbad21b27d51780ead392039921e5953599790264a2b688610d5c06` (verified current bytes)
- [decisions/rulings.md](../decisions/rulings.md): `70c7e8b55f5deb93d86b5e3b40f2d2098319d10a9a8ee5efb5c02c4b6efa3c08` (verified current bytes)
- [decisions/structural_oracle_scope.md](../decisions/structural_oracle_scope.md): `44f73b644ec3f9119a5ba8e9aa5abfebff28cdab487e186c4d3e13a24ae796a1` (verified current bytes)
- [docs/blueprint.md](../docs/blueprint.md): `cea178cc359653f80f460b53809f5ff912639d57392a44adbe0ab16d56b18ae1` (verified current bytes)
- [docs/releases/v0.1.0.md](../docs/releases/v0.1.0.md): `94fe295df26ca3b8de2fb7a7d5758d177ddb60bdacf390ed684f9aaf1ff8af63` (verified current bytes)
- [licenses/CC-BY-4.0.txt](../licenses/CC-BY-4.0.txt): `9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411` (verified current bytes)
- [manifests/assets.json](../manifests/assets.json): `92e7319d4a0c89810161fb718b682b7bcd3dbc665eaea2c97848ce92e58d9bdf` (verified current bytes)
- [manifests/environment.json](../manifests/environment.json): `8c1f736f911c1b080ba7ecfced2eb9bfed3f866539d83d6c213646158e789459` (verified current bytes)
- `manifests/environment.json`: `c411b683e5d68c7f30446a9e57fc02ebf4edfa9e048f2d30643702540043baa4` (verified historical revision 36b7dcb1f410c4c0a28c691000163237cc06a496)
- `manifests/upstream_sources.json`: `4d6c586c986334bb0bca23f08cc4a796d0d69f985b636484cfceac228e56f3ce` (verified archive analysis/history/upstream_sources-4d6c586c9863.json)
- [manifests/upstream_sources.json](../manifests/upstream_sources.json): `50752ce03efd68b1eb91c97586c1a67f4d6e214d743a6eab06b0d093bab6d21e` (verified current bytes)
- [manifests/wheels.sha256](../manifests/wheels.sha256): `23ab092a85a03a4e6116b4613bf654b21b250c3a45b891002dcc4e2d20931599` (verified current bytes)
- [pyproject.toml](../pyproject.toml): `abf0b04cf2cc7a304199f939829b70d875842f9e0441dc8c55a3b9a52136b3e8` (verified current bytes)
- [requirements.lock.txt](../requirements.lock.txt): `5393bd3b88fa9b1d427447b35f12e4566fbfca6f91a2af6f04674a5104d2349e` (verified current bytes)
- [src/oar/__init__.py](../src/oar/__init__.py): `ad2a02f6606d83b3c5c083e1c72ff55c9aea0621fcb54281304863a1f327c5b1` (verified current bytes)
- [src/oar/audit.py](../src/oar/audit.py): `7d386972050adacb66c01e66c0e66e93f3e8c0aac682db83dd5cd47880a1dd35` (verified current bytes)
- [src/oar/audit_journal.py](../src/oar/audit_journal.py): `8043cfae563d80e720fba0d935fb8c951c3cc03bd4f48b89923e82463e697a75` (verified current bytes)
- [src/oar/basis.py](../src/oar/basis.py): `0d39eb74b7f6eb1d1dab30f3ce03ae5adfe252ea08fe199828ee15ac80290b4c` (verified current bytes)
- `src/oar/bench.py`: `4504126115b6bd0515a7c47e6882ce724c185a69dea0ec1f3785577fd8bed43f` (verified archive analysis/history/fix1-bench-4504126115b6.py)
- [src/oar/bench.py](../src/oar/bench.py): `e06035ce04cfceb56074a9b000ceb27230e21d557d88cc80483a6be1c2ddbf39` (verified current bytes)
- `src/oar/bench.py`: `f2d2016eee1b67f8147b96755d4fe534311b4c57729888f46bed839dfccefb22` (verified archive analysis/history/bench-before-layout-f2d2016eee1b.py)
- [src/oar/cli.py](../src/oar/cli.py): `0b8130539bf733e0beb11377391fc2173f151aef305ceb454c055a0ed7445707` (verified current bytes)
- `src/oar/cli.py`: `65af2908f8fa2505c7d705f372664b4361067624649b828b39f310eac6c5e585` (verified historical revision a100d5c2aa031420d50ae56dcb45fc186e52a45f)
- `src/oar/cli.py`: `7a8bf194dfaa24d0663599d7628d2f9afdc09b6d61db9d60fd5343ab368f39ea` (verified historical revision 4dae4331ab338dcd6f5fd0c2ad8e541cfb2fcf21)
- [src/oar/config.py](../src/oar/config.py): `042dadf66c6d328e635accadc8d28e6ee2752a6ded80733f6185da1f362ba23a` (verified current bytes)
- [src/oar/gates.py](../src/oar/gates.py): `bde06fe48e416d5f0f2517fafd782617493002fcd3983d9ae0332b3926985c8f` (verified current bytes)
- [src/oar/hardware.py](../src/oar/hardware.py): `4ba118d816eec9d255cfb3d7925416373119fe181f82529fb58a96347d8e5048` (verified current bytes)
- [src/oar/linear.py](../src/oar/linear.py): `e0e0ee246c43e1e71b4a98ba1e97b9a1a8774a13c8271fb063a4020524119b61` (verified current bytes)
- [src/oar/optimizer_preview.py](../src/oar/optimizer_preview.py): `37935501f1f2988fc30ba941dda75062a0afffccfbbc0edf637c15179063805e` (verified current bytes)
- `src/oar/optimizer_preview.py`: `a0f965d69874e71d3b0e91663713293884e1270d08442fe07531f4b057a45888` (verified historical revision a042839e315cd60db8d9ff699c5c6a6c2940c011)
- `src/oar/oracle.py`: `917ac86d911c624f318d921b0ef9bffecc859efef039e4b98beb7bd1a49b609d` (verified historical revision dc561706752f656706c4fac5ae9b9f1f9c53b601)
- [src/oar/oracle.py](../src/oar/oracle.py): `a9d2ed4728053d184835ff1f9727c9fa22d08b8a86a56b1e48c658a80594acd1` (verified current bytes)
- `src/oar/oracle.py`: `f07f6f5c6f025601640b2db8e405e34e64223ad7db123bc12a21cd0c8445dbb9` (verified historical revision 2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07)
- `src/oar/oracle.py`: `febb3517407260eb05c4992b0541d28e0e47290e085f27c1ca4f0a7d9ffee889` (verified historical revision 3d7b57c663ba104f89d92de62977efb8211e4028)
- [src/oar/provenance.py](../src/oar/provenance.py): `ec15cbd992a51dbd7acd7cd566f5fdc84df3e9c2bbdf73b0e3ca8e3a0cd96aa0` (verified current bytes)
- [src/oar/records.py](../src/oar/records.py): `2f06d6ee5ce7debbc3bfad99fd160fe79a125335a15fdd69f46242a9924bd300` (verified current bytes)
- [src/oar/report.py](../src/oar/report.py): `176a5409f4f669849189d58aa45d7c755c793baa401e3a0835449cab787a22c3` (verified current bytes)
- [src/oar/reproduction.py](../src/oar/reproduction.py): `222ccef3c5da1540f006907c2d6a85f4205621640665e1792cac552b92c66c3c` (verified current bytes)
- [src/oar/timing.py](../src/oar/timing.py): `25351cd20abc14385dd1cb3e482c67f73619b1ead30900ca0150fe57072458e5` (verified current bytes)
- [tests/fixtures/config_minimal.yaml](../tests/fixtures/config_minimal.yaml): `db5c32cab889f61e28315bb3741497fd546f8c393a8ef251eda469f4e057d75e` (verified current bytes)
- [tests/fixtures/config_reordered.yaml](../tests/fixtures/config_reordered.yaml): `7005c9fe144fd0bab730f96125d7cae4e53196da51f0d15031a83f631cb191c4` (verified current bytes)
- [tests/test_accumulation.py](../tests/test_accumulation.py): `b036abf408b31b143933773fa33436bbbd02100fed5a1397cd8cab9081cb35f7` (verified current bytes)
- [tests/test_algebra.py](../tests/test_algebra.py): `57bcf5ba0141a246c11dca1680338f27c8656a497e4ab0c65ab7fc09e41f6756` (verified current bytes)
- [tests/test_audit_decision.py](../tests/test_audit_decision.py): `3435eca89bf8fc154f1157b02ae7956d35c03eaed8beb0ac051795683bf9f346` (verified current bytes)
- [tests/test_audit_journal.py](../tests/test_audit_journal.py): `92439ec67a4039e9a4b9daf3d948d4ddc7a944e1d03efc16d14b9b09171401d2` (verified current bytes)
- [tests/test_basis.py](../tests/test_basis.py): `97bd57733140469a7a23bf95a7bd29a2142935651909cbc921bfb5edd0e6ac01` (verified current bytes)
- `tests/test_bench_accounting.py`: `16cf18886bd425055771b309cd720cc60b5909494eff7719c72bf11b4fe66d83` (verified archive analysis/history/test_bench_accounting-before-layout-16cf18886bd4.py)
- `tests/test_bench_accounting.py`: `83c112ad2c890ea4ab8df8cbc46d9b8c62fab757f56415a1d7350ebeaa274ef8` (verified archive analysis/history/fix1-test_bench_accounting-83c112ad2c89.py)
- [tests/test_bench_accounting.py](../tests/test_bench_accounting.py): `f600b9d92a5c4c3570d8c705f269dec8574ab01ca8dc8fa0c33b90d18dfda13a` (verified current bytes)
- [tests/test_confidence.py](../tests/test_confidence.py): `7f6f16f8bac0ec4cdec5dfec10ca6b122a9ce5702d8f2dc0cb2b18be4c2ca690` (verified current bytes)
- [tests/test_config_records.py](../tests/test_config_records.py): `bb50cd9aea2f98f928324ab2f13e4dee29612804cb06f1efef78e86d92dcdd23` (verified current bytes)
- [tests/test_discovery.py](../tests/test_discovery.py): `337f2ed69cc17225e2198532ce23d957cec87688d25fd744d96b1605456685b2` (verified current bytes)
- [tests/test_exact_adjoints.py](../tests/test_exact_adjoints.py): `7276d899195b667615e19ee75a8ab955dfcff24bd25c9f6259ae13173cc229ba` (verified current bytes)
- [tests/test_optimizer_preview.py](../tests/test_optimizer_preview.py): `178c6cd2f3275abaf5614c81fea49e6aef665db32d1f269d29c48fa603b75ada` (verified current bytes)
- `tests/test_optimizer_preview.py`: `9e77fa8756b159ad7bcb2c300f64551460f97a827cfd8bf33c9d2229b58c1c79` (verified historical revision a042839e315cd60db8d9ff699c5c6a6c2940c011)
- [tests/test_oracle.py](../tests/test_oracle.py): `0687bb463152a59768e97f7bb81adc7704bbdfc06160e9e87adc4c59007ed58f` (verified current bytes)
- `tests/test_oracle.py`: `41dff73e0e1791a34fbca384f4937f0ca1a284f7c8fd6e017185bf38c95fcc3b` (verified historical revision f1efd557d36312e4f88869e83355b941ad6cd021)
- `tests/test_oracle.py`: `581e0ab4b9ec3c62a270b39ffe113a06f53628f62217742ddc249efd3b12eb8d` (verified historical revision 3d7b57c663ba104f89d92de62977efb8211e4028)
- `tests/test_oracle.py`: `6a40b3651a9b68b64e63e9d9345b94d0549e144f9ad3f22ee771a3eac8cf1120` (verified historical revision dc561706752f656706c4fac5ae9b9f1f9c53b601)
- [tests/test_report.py](../tests/test_report.py): `e5c2d01e361f43d49ef3f57787f685c6c9d6a2393b4dcbae6fd65825bb31a284` (verified current bytes)
- [tests/test_reproduction.py](../tests/test_reproduction.py): `60d1e54b0e55359ade742b79033bbcc6e93a9afb70695fbd754fdae2dad03540` (verified current bytes)
- [tests/test_synthesis_collection.py](../tests/test_synthesis_collection.py): `dc9cc7cba68250e139861e5b984c8c06ef5569cf717e5827a1f736dadd414dc0` (verified current bytes)
- [tests/test_timing_harness.py](../tests/test_timing_harness.py): `123cdf5261ff458693c7386e64b7f0d05758a21311aa95bc050fa4059da3de90` (verified current bytes)
- [third_party/discovery/pytorch/LICENSE](../third_party/discovery/pytorch/LICENSE): `bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43` (verified current bytes)
