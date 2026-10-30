# Operator evidence reproduction

Run from the repository root using the frozen `.venv`. Hardware work requires actual MPS access; a sandbox GPU skip cannot qualify a gate. All commands below launch fresh interpreters with fallback disabled before Torch import. Run collectors, timing studies and tests sequentially, with no overlapping GPU or CPU profiling workload.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python analysis/collect_operator_correctness.py
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python analysis/collect_audit_confidence.py
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python analysis/collect_numerical_calibration.py
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli bench --config configs/operator.yaml
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli bench --config configs/operator_wide.yaml
.venv/bin/python analysis/summarize_headroom.py
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q
```

The existing Task 5 evidence is reproduced independently with `PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python analysis/collect_optimizer_correctness.py`; Task 6 reused its unchanged artifact. [Collector manifest](collector_manifest.json) records exact script hashes, commands and artifact hashes. Regeneration changes timestamps and inclusive diagnostic times; record new artifact hashes and append gate records before using regenerated evidence for a gate.

[Main raw blocks](timing_calibration.json) and [wide raw blocks](timing_calibration_wide.json) retain native variant selection, randomized orders, warmup/noise flags, per-pair ledger ages, block uncertainty, separate component profiles, explicit CPU decomposition/transfers, rejection/replay, resource samples and source hashes. [Combined CSV](headroom_summary.csv) states each selected native variant and complete-step boundary. The CLI also emits one compact CSV per configuration. Cost-table keys must be unique within a configuration; use separate artifacts for layout/accumulation variants at an otherwise identical key. Both YAML and direct API inputs reject duplicates before launching hardware. `ActiveClock` study totals include initialization, warmup, profiling and refresh but stop before final JSON serialization; they are research cost, not training throughput.

Native variants are validated against three single-tensor AdamW updates, then measured together in randomized common-window blocks. The selected executable variant is used by every paired native control. Candidate/conditional/reject/replay comparisons run in shared randomized rounds with AB/BA ordering inside each pair. Comparisons can remain noisy at the declared measurement cap: no samples are discarded to obtain a positive result. Warmup and noise tolerances control measurement, not a required speedup or production guardband. Intervals describe blocks within a study under the recorded system conditions, not independent training runs.

Each headline operation is one complete deterministic residual MLP forward, GELU, residual loss, exact input/bias backward, dense or reconstructed weight backward, clipping and optimizer update. The complete approximate attempt additionally reserves/fsyncs confidence spending, generates/transfers independent probes, constructs validated captures, computes sketches, previews all parameters and moments, transfers/reduces raw alignment in CPU FP64, forms explicit complements, evaluates the unresolved numerical decision and prices commit. Conditional commit deliberately bypasses that rejection solely to measure `p=0`; it cannot authorize online acceptance. Rejection discards pending state and executes the same tensor batch exactly with one native optimizer update. No dropout, scheduler, data cursor, trainer or checkpoint lifecycle is claimed here.

The six kernel configurations vary batch, rectangular width, rank, probe count, layout and accumulation. The wider kernel is one block, not a larger model. Rank/probe/batch changes are partly confounded and do not optimize a rank/q policy. The four-microbatch positive candidate interval has failed dense warmup/noise targets and is excluded from qualified candidate regimes. Tiny fixtures and the 120 shape/scale numerical stress cases do not calibrate a production numerical allowance; the wide shape has timing evidence only. [Numerical metadata](guardband_calibration.json) keeps `NumericalPolicy` unresolved, including held-out stress discrepancies.

[Decision](headroom_decision.json) uses measured complete exact replay blocks and extra paired refresh work. It never infers replay cost by subtracting independently aged conditional/reject journals. Its K/p grid is arithmetic sensitivity, not selected policy or observed full-cycle training. No feasible rejection probability can repair a complete attempt already slower at `p=0`.

Resource readings separate physical/available memory, RSS, Metal allocations, driver allocations and OS-wide swapping. Five fixtures reside in a timing worker; observed 5ms resource maxima describe the study working set and can miss short peaks. They do not establish single-model training or checkpoint peaks. RSS and Metal readings overlap and are never added. Pre-existing swap, swap activity and unavailable thermal limits remain explicit. [Post-study pressure](resource_context.json) records query-only OS observations, not continuous block telemetry. No apps were killed or OS safeguards changed.

[Historical index](evidence_history.json) preserves exact prior gate artifacts and completed exploratory runs. The overlapping pilot is excluded from gate evidence. The clean earlier sequential study is retained with its control-drift/replay-identification limits; only the current seed-separated interleaved studies support the latest WP02 record. An initial serialization-failed pilot has aggregate console results in the local Task 6 report, but its temporary raw blocks were lost and are not used for a gate.

The WP02 result stops the current online implementation branch. Candidate-only headroom supports preparing a bounded synthetic structural-oracle information study and a narrow gate amendment before execution. Normal WP03/WP04 prerequisites remain blocked. No full trainer/controller, data ingestion, larger-model or MLX retry is authorized by these artifacts. Dataset bodies remain unverified and WikiText remains blocked.

For the original pre-review measurements, final interface review added only pre-worker duplicate-key validation after measurement. That exact older benchmark source is preserved in the historical index; then-current changes added those input guards and extracted the identical initialization input expression into a directly CPU/MPS-tested helper. Both paths slice device storage after transfer. Calibration records/asserts observed strides, including24 stride-2 cases. Timed method operations are unchanged; the snapshot is archival, not a second maintained implementation, and no cost results were relabeled as measured from changed kernels.

The first numerical artifact mislabeled the actual layout for the configured noncontiguous case. It and the intermediate correction are archived explicitly; the final same120-case regeneration records observed post-transfer strides and leaves NumericalPolicy unresolved. Superseded Task6 gate bytes remain recoverable through the historical index.


Task6 review fix1 supersedes the earlier headline timing/forecast data. Current
main/wide files were rerun exclusively at the same six configurations with
resolved construction/audit/data/model seeds transported into the fresh worker.
Data and model initialization have separate CPU generators; changing batch
shape/layout or the data seed cannot advance the model initialization stream.
Construction uses its own generator and durable audit allocation uses its own
resolved audit seed. Raw metadata records all four seeds and the policy tag;
measurement ordering/bootstrap retain the separately declared benchmark seed.
The earlier files/sources are archived unchanged and explicitly labeled coupled
initialization in the historical index; they are not retroactively relabeled.

For `[D+C+(K-1)(A+pR)]/K`, the exact scheduled-mixture threshold is
`p* = (D-A-C/(K-1))/R` for K>1. K=1 has no approximate segment and returns no
rejection threshold. The K16 bootstrap uses15 approximate steps. Covering tests
check the hand-derived4/33 threshold, the equality cycle at that threshold, K1,
worker seed transport and data/model initialization independence. The refreshed
derived artifacts use this corrected arithmetic. Numerical calibration remains
unresolved and was not rerun by this review fix.

Current fix1 main/wide source hashes match the current measured implementation. Older source-snapshot/input-only delta notes above describe the superseded pre-review studies, not the rerun with separated seed streams.

## Bounded synthetic structural oracle (Task 11)

The public command is:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli audit-checkpoints --config configs/structural_oracle.yaml
.venv/bin/python analysis/summarize_structural_oracle.py
```

Run MPS work exclusively; the command launches a fresh fallback0 worker and
validates the latest hashed synthetic-only amendment plus genuine WP02 stop /
qualified candidate-potential artifacts and passed WP00/WP01 numerical evidence.
Ordinary WP03/WP04 remain blocked; there is no general training command.
The pilot was run at commit `a100d5c` with that revision's stage-pilot config.
The full query study ran at `2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07` with
locked LR.003 and shift120 before inspecting query outcomes. The later YAML
syntax-error normalization changes no valid-config experiment operation.

`structural_oracle_audit.jsonl` retains all792 layer rows: paired weight rows
share one simultaneous proposal. `structural_oracle_summary.json` contains the
30 native-learning curve observations, all six condition/seed attempts,
18-checkpoint raw artifact index, tuning-only policies and per-seed/stage
summaries. Postprocessing produces `structural_oracle_learning.csv`, a separate
small `structural_oracle_checkpoint_index.json`, and
`structural_oracle_aggregate.json` with the full unfitted acceptance envelope,
per-condition/seed/stage/rank/age strata and exact algebraic margin ranges.
`structural_oracle_schema.json` declares fields/units. JSONL replaces broad
Task11's Parquet within this explicit narrow amendment.

All losses/deltas are **mean squared error per output element**. Counts are
synthetic observations. Model is the qualified two-linear GELU residual block,
n2048/di512/do2048. Gaussian inputs and independent teacher/student initializers
come from deterministic CPU generators; domain seeds derive by SHA256 of
`[seed, domain, *ids]`, with the first8 bytes modulo `2**63-1`. See
`domain_seed` and `replay_synthetic` in `src/oar/oracle.py` for complete descriptor
reconstruction, including teacher low-rank and query/development IDs. Stored
`rng_domains` are base-domain anchors; per-step/index/name seeds derive from
the same durable rule. No learner stream advances during diagnostics.

Past bases use actual pre-update, pre-clipping source gradients; current exact
SVD is separately labeled paid oracle. Native default AdamW parameters/moments
continue across the teacher shift. Tensor checkpoints (~485MB total) are ignored
and regenerate from the command; the small versioned index retains byte hashes
and exact source steps. Counterfactuals never alter the native history.

The score uses `a = sum(g_hat * actual_displacement)` across all parameters.
`ordinary_bound = sum(||G-G_hat|| * ||d||)` and
`complement_bound = sum(||G-G_hat|| * ||d-U(U.T d)||)` over two weight matrices.
An algebraic margin at tau is `tau*a-bound`, with positive `a` required.
The relative score is `sqrt(sum ||G-G_hat||^2 / sum ||G_hat||^2)` over weights.
Probe q8/delta.01 scores are offline diagnostics, with no run-wide allocation or
validated numerical allowance. NumericalPolicy remains unresolved.

`oracle_cost_s` repeats on two layer rows; shared query and exact-SVD times also
repeat across proposals. Do not sum these fields as a total. The worker's
60.959s clock includes initialization, native histories, basis construction,
explicit CPU FP64 SVD/transfers, previews, scoring, row/schema output and policy
summaries up to its final summary write. Process import/startup and the last
small summary serialization tail are outside that clock; it is not a fully
inclusive invocation benchmark. The pilot clock is26.046s on its documented
older root-input boundary. Postprocessing is separate CPU work. Endpoint RSS /
Metal snapshots overlap and are not peaks. Existing matched-rank16 costs are
historical benchmark references; no runtime evidence is invented for ranks4/64
or for the current-SVD construction oracle, and none yields deployable savings.

All144 past evaluation proposals lose development progress relative to dense;
ordinary/complement bounds accept none at every declared tau. Fitted policies
reject all and match a constant exact-fallback control. Scores have nontrivial
but correlated descriptive AUCs; this is not proof of no geometric information.
Current-SVD/r64 complement has only two algebraic accepts at tau.75, both still
worse than dense. Lost-progress AUC is unresolved because its label is constant.
Early full-rank/shift stages share inputs and trajectories, and dense updates
also harm the six stage8 evaluation queries. No inference from proposal counts
to independent-run prevalence, real-task quality or online utility follows.
Keep WP02's online stop; the next permitted action after review is WP09 synthesis.
