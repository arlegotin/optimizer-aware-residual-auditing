# Optimizer-aware residual auditing

A stopped exploratory research implementation on an Apple M3 Max, native arm64 CPython3.13 and PyTorch MPS. **No training speedup was established.** Candidate weight-gradient reconstruction was faster at some isolated residual-block shapes; complete audited attempts were slower even at conditional zero rejection. The bounded synthetic oracle found no accepting past-only exact-margin region in its declared tau grid. It retains descriptive information, not a competitive training result.

Read the [findings and execution outcome](analysis/report.md), [original blueprint](docs/blueprint.md), [complete synthetic-oracle report](decisions/reports/structural_oracle.md), [rulings](decisions/rulings.md), [pre-final review snapshot](decisions/review_summary.md) and [final review and rulings addendum](decisions/reports/final_review.md). WP02 stops normal online work; ordinary WP03 remains unfulfilled and WP04 inconclusive. Tasks1–6, the narrow Task11 amendment, and Task17 synthesis were reached. NumericalPolicy remains unresolved. No corpus, full trainer/controller, protected evaluation or confirmatory result exists.

## Local environment

The measured environment is pinned in [requirements.lock.txt](requirements.lock.txt), [environment discovery](manifests/environment.json) and [wheel hashes](manifests/wheels.sha256). Use a native arm64 CPython3.13 interpreter with its compatible macOS wheels; these pins describe the measured runtime, not a portability promise. From the repository root:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli doctor --config configs/discovery.yaml --out /tmp/oar-environment.json
```

`doctor` is a capability/provenance query. A sandbox may hide MPS: run hardware checks with actual device access. CPU checks and sandbox skips do not establish MPS eligibility. Disable silent fallback before importing Torch.

## Supported commands and verification

```sh
.venv/bin/python -m oar.cli --help
.venv/bin/python -m pytest -q -m 'not mps'
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q -m mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q
.venv/bin/python -m oar.cli report --gates decisions/gates.jsonl --run-index analysis/evidence_index.json --out analysis/report.md
```

The MPS commands require real hardware access. Run hardware work sequentially with no overlapping benchmarks, collectors or tests. Hardware skips are explicit missing evidence. The report also accepts an early stop without an index:

```sh
.venv/bin/python -m oar.cli report --gates decisions/gates.jsonl --out /tmp/oar-gates-only.md
```

The gates-only report supplies gate reasons and marks measurements unavailable. The indexed report verifies the frozen scientific gate prefix and all current/historical artifact hashes before rendering; later WP09 synthesis records are excluded from that prefix to avoid self-hashing. New scientific gate records require a deliberately updated index. Schema1 is an annotated stopped-study index; confirmation claims are classified unresolved. See [schema and generation instructions](decisions/reports/synthesis_schema.md).

For explicit operator-cost regeneration only, the supported benchmark commands are:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli bench --config configs/operator.yaml
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli bench --config configs/operator_wide.yaml
.venv/bin/python analysis/summarize_headroom.py
```

These overwrite the named timing artifacts. Preserve original bytes and source attribution before regeneration and update evidence bindings before using new measurements. Task17 did not repeat this benchmark because no measured kernel or cost implementation changed. The [analysis instructions](analysis/README.md) list component collectors, complete accounting boundaries, timing exclusions, failed/superseded trials and provenance.

## Small bounded research reproduction

The guarded public research command is:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli audit-checkpoints --config configs/structural_oracle.yaml
.venv/bin/python analysis/summarize_structural_oracle.py
```

It regenerates the original artifact paths and roughly485MB of ignored local tensor checkpoints. It requires the latest exact hash-bound synthetic amendment, passed component gates, and the actual WP02 stop. This permission never authorizes normal WP03/WP04, online acceptance, data ingestion or a general training command. Source archives do not redistribute corpus bodies or pretrained weights; neither is needed here.

To preserve original results, use a fresh directory for an independent full same-seed repeat:

```sh
.venv/bin/python analysis/reproduce_structural_oracle.py --prepare analysis/reproduction/local-repeat
# Review and commit the generated plan.json, config.yaml and amendment.json before execution.
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python analysis/reproduce_structural_oracle.py --run analysis/reproduction/local-repeat/plan.json
```

Preparation refuses existing output/checkpoint paths and preserves every scientific config field. The plan predeclares28 numeric fields,23 categorical fields, exact sign/acceptance decisions and30 curve observations, with absolute tolerance2e-6/relative tolerance2e-4. Execution validates reference/source/config hashes, temporarily appends only the declared output-path amendment, invokes the ordinary guarded fresh-MPS command, and restores the preceding amendment by append. Preserve the resulting amendment ledger. A changed code hash requires a new preparation; do not loosen tolerances after outcomes. Interrupted execution may leave the temporary amendment latest: inspect retained artifacts and explicitly restore the preceding permission before another command.

The committed [Task17 plan](analysis/reproduction/wp09/plan.json), [initial failed comparison](analysis/reproduction/wp09/comparison.json) and [recovered comparison](analysis/reproduction/wp09/comparison_recovered.json) and separate raw outputs preserve the actual local reproduction. Same-seed agreement is numerical reproducibility, not independent replication or confirmation. Times and endpoint resources are accounting observations, not a benchmark replication. All failed comparisons remain failures; large tensors remain local and ignored.

The first comparison helper failed after the row checks because the curve schema names the logical step `committed_updates`. The separately retained recovery maps that field without changing observations or tolerances, using `.venv/bin/python analysis/reproduce_structural_oracle.py --recover-comparison analysis/reproduction/wp09/plan.json`. That command preserves an existing recovered result by refusing to overwrite it. A 1.27s report-only CPU test may have overlapped the original repeat; those wall times support resource accounting only.

Final review maintenance at `7628be7` rejects unsupported internally overlapping parameter layouts before publication/commit, shares one oracle deadline through native histories and completion, and shares normal/recovery checkpoint identity validation. Its component verification is separate from the original research and timing sources. No benchmark or research study was repeated; existing timings do not measure these later validation changes. The [final addendum](decisions/reports/final_review.md) retains the complete review, disposition and subsequent rulings; scoped re-review is pending. The hash-bound review/outcome files retain their earlier snapshot status.
