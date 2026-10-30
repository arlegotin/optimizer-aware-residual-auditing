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
