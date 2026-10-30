# Optimizer-aware Residual Auditing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, gate-controlled local study of whether optimizer-aware audited weight-gradient projection improves small-model learning per inclusive wall time.

**Architecture:** Native dense PyTorch training remains the reference fast path. A separate projected-linear branch collects detached effective-batch residual sketches, previews AdamW transactionally and commits or replays exactly once; an oracle and timing study must justify integrating this online method. Versioned manifests, spending journals, committed checkpoints and a frozen evaluation protocol make both stopped and completed studies reportable.

**Tech Stack:** Native arm64 CPython, PyTorch/MPS, CPU FP64 NumPy references, pytest, PyYAML, psutil, local JSONL/CSV; gated dataset/tokenizer/Parquet packages.

**Spec:** [2026-10-30-oar-design.md](../specs/2026-10-30-oar-design.md). Read it and the cited [blueprint](../../blueprint.md) sections before each task. These commands, modules and tests are proposed work, not existing capabilities or recorded results.

## Global Constraints

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

## Review Focus

1. Crash after audit probes may have been observed: spend the reservation permanently and use a new probe identity on retry. Owned by Task 4's crash-journal test and Task 12's replay test.
2. Unequal token masks, empty microbatches and accumulation: match the token-weighted effective objective and one aggregate residual; reject an all-empty batch before mutation. Owned by Tasks 3, 7 and 12.
3. Tied parameters, nonzero dropout and checkpoint recomputation: exclude shared weights, restore model randomness and count sketch side effects once. Owned by Tasks 8, 9 and 12.
4. Missing gradients, zero gradients, clipping and decay: match native AdamW state and actual displacement while retaining the raw gradient for alignment. Owned by Task 5.
5. Checkpoint serialization crossing the time horizon: select only an earlier fully saved committed endpoint and charge restart cost. Owned by Tasks 9 and 15.

## Execution boundaries and task map

This is a plan to review, not authorization to begin training. This documentation request ends after committing the spec and plan. Before a later implementation run, obtain design/plan review and choose the execution method; preserve the current-branch instruction. Do not create a worktree unless the user changes that instruction.

Run tasks in order through the gate that permits them. A failed scientific gate skips dependent implementation and goes to Task 17 with the artifacts already collected. Hardware or asset blocks can leave a CPU correctness or structural result. Do not turn a negative result into automatic scope expansion.

| Blueprint package | Tasks | Gate artifact |
|---|---|---|
| WP00 | 1–2 | Discovery/backend/assets status |
| WP01 | 3–5 | Algebra, confidence, preview/transaction correctness |
| WP02 | 6 | Measured headroom and full-block cost map |
| WP03 | 7–10 | Valid minimal dense harness and baseline feasibility |
| WP04 | 11 | Past-only oracle information/cost decision |
| WP05 | 12–13 | Complete online method and attribution controls |
| WP06 | 14 | Frozen development-selected protocol |
| WP07 | 15 | Fixed paired results and protected endpoint analysis |
| WP08 | 16 | Robustness and clean-process confirmation |
| WP09 | 17 | Scope-limited report and reproduction status |

Every task's final check writes/updates `decisions/progress.md`; package exits append a schema-versioned `GateRecord` to `decisions/gates.jsonl`. Each record names its evidence, actual cost, interpretation, next action and unavailable comparisons. A scientific gate cannot be satisfied by a mocked unit test. Test doubles verify mechanisms; actual artifacts establish empirical exits.

## Proposed file structure and common types

All paths below are relative to the existing root. Create only files needed by the current task; postpone optional integrations until their gate.

| Files | Responsibility / owner |
|---|---|
| `pyproject.toml`, `.gitignore`, `src/oar/__init__.py` | Installable local package and exclusions; Task 1 |
| `src/oar/config.py`, `records.py`, `cli.py`, `gates.py` | Strict typed configuration, record schemas, command dispatch, gate eligibility; Task 1, extended by owning tasks |
| `hardware.py`, `provenance.py`, `configs/discovery.yaml` | Redacted discovery and provenance; Task 2 |
| `linear.py`, `basis.py` | Projected branch and past-only basis; Task 3 |
| `audit.py`, `audit_journal.py` | Confidence, conservative decisions and durable allocations; Task 4 |
| `optimizer_preview.py` | Functional AdamW and single-use pending states; Task 5 |
| `timing.py`, `bench.py`, `configs/operator.yaml` | Inclusive clock and separate microbenchmark/profile paths; Task 6 |
| `data.py`, `configs/lm_dev.yaml`, `vision_dev.yaml` | Protected assets, tokenizer, chunks/masks; Task 7 |
| `models.py`, `synthetic.py` | Shared decoder, residual MLP, marked recall task; Task 8 |
| `methods.py`, `train.py`, `checkpoints.py` | Native trainer, logical-step lifecycle and atomic snapshots; Tasks 9/12 |
| `baselines/{projection,sampling,instant,vcas,dropbp,auditors}.py` | Distinct labeled comparison methods; Tasks 10/13 |
| `oracle.py`, `controller.py`, `protocol.py`, `analysis.py`, `report.py` | Oracle gate, online policy, experiment freeze, inference and synthesis; Tasks 11–17 |
| `tests/`, `tests/fixtures/` | Small deterministic inputs and meaningful independent-reference tests; owned per task |
| `manifests/`, `decisions/`, `analysis/` | Small versioned evidence manifests, decisions and aggregate artifacts |
| `third_party/`, `data/`, `runs/`, `.venv/` | Local sources/caches/runtime outputs; ignore raw data, wheels and large checkpoints by default; track notices/patches/indices deliberately |

Use dataclasses from `records.py`; do not add a general workflow engine. `Tensor` below means `torch.Tensor`; `Path` means `pathlib.Path`. `ParameterId`/`LayerId` are canonical string names. `RunConfig`, `Protocol`, `BasisState`, `ControllerState`, `CostTable`, `CheckpointState` and result dataclasses belong to their owning modules listed below. All durable records have `schema_version = 1`; strict readers reject unknown schemas. Extend schemas explicitly as each owner adds fields.

`records.py` owns the spec's `ReplayableBatch`, `StepContext`, `LayerAudit`, `PendingUpdate`, `Decision`, `StepRecord`, `GateRecord` and these optimizer transport types: `AdamWParamState(step: int, exp_avg: Tensor, exp_avg_sq: Tensor)` and `AdamWGroup(parameter_ids: tuple[str, ...], lr: float, betas: tuple[float, float], eps: float, weight_decay: float)`. Tensor-bearing runtime objects serialize through explicit checkpoint encoders, never JSON stringification. Frozen metadata records do not claim to make their underlying tensors immutable; version checks enforce that boundary.

CLI commands return 0 for successful operations, 2 for invalid config/protocol, and 3 for blocked gates. Empirically valid negative/inconclusive study results are successful operations with explicit outcome fields. MPS-only tests use an `mps` pytest marker: skipping it on CPU is allowed for references and must never satisfy a hardware gate.

### Task 1: Strict configuration, evidence records and local package (WP00)

**Files:** Create `pyproject.toml`, `.gitignore`, `src/oar/{__init__,config,records,gates,cli}.py`, `tests/test_config_records.py`, `tests/fixtures/config_minimal.yaml`.

**Interfaces:**

- `load_config(path: Path) -> RunConfig`, `resolve_config(config: RunConfig) -> dict[str, object]`, `config_sha256(config: RunConfig) -> str` in `config.py`. Initially cover schema, phase, method, device, seed streams and paths; later tasks extend strict nested fields.
- `record_gate(record: GateRecord, out: Path) -> None`, `require_gate(package: str, ledger: Path) -> None` in `gates.py`; unknown/stopped prerequisite raises `GateBlocked`.
- `main(argv: list[str] | None = None) -> int` in `cli.py`; owning tasks register subcommands.
- Produce the shared records above, including explicit `None` optional measurements and versioned payload encoders. No unneeded training infrastructure yet.

- [ ] **Step 1: Write the failing config/record tests.** Create two YAML fixtures differing only in mapping order; test unknown fields, missing measurements and a stopped prerequisite.

```python
def test_canonical_config_and_stopped_gate(config_pair, stopped_ledger):
    left, right = map(load_config, config_pair)
    assert config_sha256(left) == config_sha256(right)
    with pytest.raises(GateBlocked):
        require_gate("WP02", stopped_ledger)

def test_unknown_field_rejected(bad_config):
    with pytest.raises(ValueError, match="unknown"):
        load_config(bad_config)
```

- [ ] **Step 2: Bootstrap the smallest compatible native environment and observe red.** Create `.venv` using a supported arm64 interpreter; install an actual compatible PyTorch/NumPy/pytest/PyYAML/psutil combination and `pip install -e .` after minimal package metadata. Run `python -m pytest tests/test_config_records.py -q`; expect missing contracts/test failures, not a dependency-resolution failure. Freeze versions/wheel identities in Task 2.
- [ ] **Step 3: Implement the interfaces.** Canonical sorted configuration hashing, strict enum/range/schema validation, append-only gate records and explicit JSON nulls. Ignore data caches, virtual environments, wheel archives and checkpoints; keep source-controlled notices/patches accessible. Do not introduce arbitrary experiment durations.
- [ ] **Step 4: Verify green.** Run `python -m pytest tests/test_config_records.py -q` and `python -m oar.cli --help`; expect all tests pass and no advertised unimplemented commands.
- [ ] **Step 5: Commit.** Stage only these package/config/tests files; commit `feat: establish OAR configuration and evidence records`.

### Task 2: Redacted hardware, operation and asset discovery (WP00)

**Files:** Create `src/oar/{hardware,provenance}.py`, `configs/discovery.yaml`, `tests/test_discovery.py`; modify `config.py`, `cli.py`; produce `manifests/{environment,assets,upstream_sources}.json`, `manifests/wheels.sha256`, `requirements.lock.txt`.

**Interfaces:**

- `discover_environment() -> dict[str, object]`, `check_operations(device: str) -> list[dict[str, object]]` in `hardware.py`.
- `inspect_asset(uri: str, revision: str) -> dict[str, object]`, `hash_file(path: Path) -> str` in `provenance.py`. Manifest fields: source/revision/hash, size, notice/license evidence, local status and restrictions; unresolved evidence cannot be marked approved.
- CLI `doctor --out PATH`; later bench expands shape-specific capabilities. CPU-probe generation and model-RNG restoration must be tested, not assumed.

- [ ] **Step 1: Write failing tests using recorded shell-output fixtures and a tiny local asset.** Cover absent MPS, unsupported critical ops, arch mismatch, redaction, file hash and conflicting/missing asset notices.

```python
def test_discovery_never_publishes_identifiers(mac_fixture):
    report = redact_hardware(mac_fixture)
    assert "serial_number" not in json.dumps(report).lower()
    assert report["physical_memory_bytes"] > 0

def test_no_gpu_is_classified_not_faked(cpu_environment):
    assert cpu_environment["gpu_claim_eligible"] is False
    assert cpu_environment["scope"] == "correctness_only"
```

- [ ] **Step 2: Run `python -m pytest tests/test_discovery.py -q`; expect missing discovery functions.** `redact_hardware(raw: dict) -> dict` is a public helper in `hardware.py` solely because redaction has an independent contract.
- [ ] **Step 3: Implement discovery and manifests.** Use blueprint §9.1's native commands, omit identifiers, record memory/disk/power/macOS/package/accelerator/fallback state, and execute tiny GEMM/autograd/optimizer/RNG tests. Inspect small dataset metadata/notices before bulk downloads. Freeze actual wheel/source hashes and source revisions; file-blob IDs from the blueprint are not repository pins. Select explicit CPU decomposition or record a blocked capability, never hidden fallback.
- [ ] **Step 4: Run the tests and `python -m oar.cli doctor --out manifests/environment.json`.** Tests pass; inspect actual output and write WP00 evidence. Missing assets block that workload; missing GPU permits CPU correctness only. Keep asset approval distinct from loader success.
- [ ] **Step 5: Commit small discovery code, tests, resolved config/lock and redacted manifests.** Commit `feat: discover local backend and asset provenance`; exclude downloaded assets and binaries.

### Task 3: Projected linear derivative, accumulation and past-only basis (WP01)

**Files:** Create `src/oar/{linear,basis}.py`, `tests/{test_algebra,test_accumulation,test_exact_adjoints,test_basis}.py`; extend records with `LinearCapture` and `BasisState` in the owning modules.

**Interfaces:**

- `projected_linear(x: Tensor, weight: Tensor, bias: Tensor | None, capture: LinearCapture) -> Tensor`; `LinearCapture` owns fixed `basis`, `probes`, `basis_version`, running `h_sum`/`r_sum`, and a backward contribution counter. It never retains a stack of `X`/`D`.
- `build_basis(gradient: Tensor, max_rank: int, oversampling: int, generator: torch.Generator, source_step: int) -> BasisState` in `basis.py`; returns ordered vectors, numerical rank, source step and orthogonality diagnostic. Basis use requires `source_step < current_step`.

- [ ] **Step 1: Write deterministic CPU FP64 tests against materialized dense autograd and blueprint Appendix A identities.** Include rectangular/noncontiguous inputs, rank zero/full basis/deficiency, zero adjoints, bias/no bias, deep residual blocks and an injected wrong input derivative. For accumulation use seven unequal microbatches with unequal valid counts and the same probes.

```python
def test_streamed_residual_is_aggregate(accumulated_case):
    case = accumulated_case  # fixture runs weighted microbatches and dense reference
    torch.testing.assert_close(case.capture.r_sum, case.exact_residual @ case.z,
                               rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(case.input_grad, case.dense_input_grad,
                               rtol=1e-10, atol=1e-12)
    assert case.capture.backward_contributions == 7
```

- [ ] **Step 2: Run `python -m pytest tests/test_algebra.py tests/test_accumulation.py tests/test_exact_adjoints.py tests/test_basis.py -q`; expect missing operators.** Define fixtures within these test modules using seeded small tensors and native autograd; no production helper computes the expected gradient.
- [ ] **Step 3: Implement the custom branch and paid randomized construction.** `dX = D @ W`, exact `db`, projected `dW`, detached sketches; no extra loss scaling. Oversampling initially 8, no power iteration. Drop deficient directions, avoid pretending QR completion adds useful modes, and reject higher-order requests. Retain exact pre-update refresh gradient for next-step basis use only.
- [ ] **Step 4: Run the suite; expect pass at declared scale-aware FP64 tolerances.** Full/native operators may use gradcheck; projected derivatives are compared with their specified formula. Run supported FP32 MPS cases explicitly and record errors separately. Do not inflate tolerances to hide a defect.
- [ ] **Step 5: Commit `feat: add projected weight gradients with exact input adjoints`.**

### Task 4: Gaussian confidence, numerical decisions and crash-safe spending (WP01)

**Files:** Create `src/oar/{audit,audit_journal}.py`, `tests/{test_confidence,test_audit_journal,test_audit_decision}.py`; modify `records.py`.

**Interfaces:**

- `lower_fraction(q: int, delta: float) -> float`, `residual_bound(sketch: Tensor, q: int, delta: float) -> float` in `audit.py`; root via CPU FP64 log-domain bisection.
- `decide_margin(a: float, contributions: dict[str, float], tau: float, u_margin: float) -> Decision`; nonfinite/unresolved uncertainty is rejection.
- `AuditJournal.reserve(layer_ids: tuple[str, ...], logical_step: int) -> tuple[AuditReservation, ...]`; constructor `AuditJournal(path: Path, delta_run: float, checkpoint_counter: int)`. Reservations carry monotone `index`, `delta`, unique `probe_id`/seed. Durable reservation precedes probe visibility; recovery takes max(checkpoint, journal).
- `NumericalPolicy` in `audit.py` supplies the measured nonnegative `u_margin` under a hashed calibration configuration; unresolved calibration forbids online acceptance, but does not block algebra/headroom tests.

- [ ] **Step 1: Write bound, decision and crash tests.** Cover invalid q/delta, all-zero residual, rank-one/flat/decaying spectra at resolvable probability, underflow, NaNs, orthogonality/cancellation uncertainty and fresh probe streams after interruption.

```python
def test_crash_does_not_refund_observed_test(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditJournal(path, 0.01, 0).reserve(("mlp.weight",), 1)[0]
    retry = AuditJournal(path, 0.01, 0).reserve(("mlp.weight",), 1)[0]
    assert retry.index == first.index + 1
    assert retry.probe_id != first.probe_id
    assert retry.delta < first.delta

def test_uncertainty_blocks_boundary_acceptance():
    assert decide_margin(1.0, {"w": 0.49}, 0.5, 0.02).kind == "numerical_reject"
```

- [ ] **Step 2: Run `python -m pytest tests/test_confidence.py tests/test_audit_journal.py tests/test_audit_decision.py -q`; expect missing bound/journal behavior.** Corrupt/truncate the final journal line in an additional fixture; a possibly consumed reservation must not be silently reused (fail closed if it cannot be reconstructed safely).
- [ ] **Step 3: Implement the confidence and decision interfaces.** Spend `delta_run*6/(pi²*j²)` from `j=1`, including zero/rejected tests. Keep model RNG separate from construction/audit streams. Check candidate metadata independence and forbid current-rank mutation after probe observation. Guardband calibration uses Task 3 FP64-vs-FP32 stress errors and is finalized with actual shapes in Task 6/14, not presented as a proof.
- [ ] **Step 4: Run the tests; expect pass.** Record Monte Carlo seed/replicates/uncertainty, scalar root residuals and dtype/scale error envelopes; do not infer tiny failure probabilities from zero observed failures.
- [ ] **Step 5: Commit `feat: add independent residual audits and durable confidence spending`.**

### Task 5: Native-equivalent transactional AdamW preview (WP01)

**Files:** Create `src/oar/optimizer_preview.py`, `tests/test_optimizer_preview.py`; extend `PendingUpdate` ownership/version support in `records.py`.

**Interfaces:**

- `preview_adamw(parameters: dict[str, Tensor], raw_gradients: dict[str, Tensor | None], state: dict[str, AdamWParamState], groups: tuple[AdamWGroup, ...], parameter_version: int, state_version: int, max_grad_norm: float | None) -> PendingUpdate`.
- `commit_pending(pending: PendingUpdate, parameters: dict[str, Tensor], state: dict[str, AdamWParamState], parameter_version: int, state_version: int) -> tuple[int, int]`.
- `PendingUpdate.displacements` uses actual `theta - candidate`; missing parameters have no displacement/state transition. `raw_alignment` uses original unclipped gradients. Native missing-state initialization and per-parameter counts are preserved.

- [ ] **Step 1: Write one-/multi-step comparisons with native `torch.optim.AdamW`.** Parameterize mixed groups, initialized/missing moments, epsilon, zero versus None gradients, clipping stabilization, decay, no decay, stale momentum, extreme second moments, version mismatch and double commit.

```python
def test_preview_is_transactional_and_native_equivalent(preview_case):
    case = preview_case  # fixture makes two independent native-identical states
    pending = preview_adamw(**case.arguments)
    assert case.live_state_equals_initial()
    assert case.raw_gradients_equal_initial()
    case.commit(pending)
    case.run_native_reference()
    case.assert_parameters_and_moments_close(rtol=1e-10, atol=1e-12)
    with pytest.raises(RuntimeError, match="consumed"):
        case.commit(pending)
```

- [ ] **Step 2: Run `python -m pytest tests/test_optimizer_preview.py -q`; expect missing preview.** Add blueprint §5.2's Adam counterexample as an independent oracle: cosine `> 0.999` and true alignment `< 0` while approximate alignment `> 0`.
- [ ] **Step 3: Implement functional preview and guarded commit.** Reuse native clipping semantics on detached copies, preserve raw inputs, return candidate moments/parameters without touching live state or RNG. Initial scope is non-AMSGrad, non-differentiable AdamW; native variants receive their own equivalence tests before timing. Inspect API behavior at the actual resolved version.
- [ ] **Step 4: Run the suite and supported MPS FP32 comparisons; expect pass at declared tolerances.** Write WP01 gate evidence from Tasks 3–5. CPU correctness does not mark MPS compatibility passed.
- [ ] **Step 5: Commit `feat: preview and commit AdamW updates transactionally`.**

### Task 6: Full-cost kernel headroom and numerical calibration (WP02)

**Files:** Create `src/oar/{timing,bench}.py`, `configs/operator.yaml`, `tests/{test_timing_harness,test_bench_accounting}.py`; modify `cli.py`, `hardware.py`; produce `analysis/timing_calibration.csv` and guardband calibration metadata.

**Interfaces:**

- `ActiveClock` in `timing.py`: constructor `(elapsed_s: float = 0.0)`, `elapsed_s() -> float`, explicit external-pause context; only the top-level run boundary excludes declared external pauses.
- `measure_blocks(operation: Callable[[], None], synchronize: Callable[[], None], blocks: int) -> list[float]` in `timing.py`; configurable warmup policy and null-block calibration live in the benchmark config.
- `benchmark(config: RunConfig) -> CostTable` in `bench.py`; `CostTable` maps `(backend, dtype, n, di, do, rank, q)` to raw timing summary, uncertainty, memory and full-block cost. CLI `bench --config PATH`.

- [ ] **Step 1: Write deterministic harness/accounting tests using an injected fake clock/synchronizer, plus one real deliberately-slower CPU control.** Fake clocks establish boundaries, not GPU speed. Include preview, reconstruction, complement, transfers, refresh and replay in accounting.

```python
def test_complete_attempt_cannot_hide_preview(cost_fixture):
    result = cost_fixture.run_with_preview_cost(2.0)
    assert result.active_wall_s == pytest.approx(7.0)
    assert result.candidate_only_s == pytest.approx(3.0)
    assert result.audit_and_preview_s == pytest.approx(4.0)
```

- [ ] **Step 2: Run `python -m pytest tests/test_timing_harness.py tests/test_bench_accounting.py -q`; expect missing timing interfaces.** Expose timer injection only in testable helpers; production clock uses monotonic high-resolution time.
- [ ] **Step 3: Implement and execute shape-adaptive sweeps.** Compare native dense, projected/reconstructed, probes, explicit complement, preview, CPU QR/SVD transfers, sampling primitives and complete small residual blocks. Cover tall MLP matrices, noncontiguous inputs and accumulation. Separate fine profiling from headline timing, use stabilized warmup/randomized AB/BA blocks, record physical/Metal/RSS/pressure separately and validate fallback is disabled. Refine `NumericalPolicy` from actual shape/scale stress data.
- [ ] **Step 4: Run tests and `python -m oar.cli bench --config configs/operator.yaml`.** Inspect raw timings and full-cycle forecasts. WP02 records candidate-only and complete-attempt headroom with uncertainty, feasible ranks, memory ceilings and break-even rejection probabilities. No candidate headroom stops online engineering; audit-cost failure allows only a bounded structural oracle study. Larger models/MLX require evidence and an amendment, not an automatic retry.
- [ ] **Step 5: Commit `perf: measure projected-gradient headroom with full audit costs`.** Track compact raw tables/metadata, not profiler binaries.

### Task 7: Article-safe data preparation and protected evaluation access (WP03)

**Files:** Create `src/oar/data.py`, `configs/{lm_dev,vision_dev}.yaml`, `tests/{test_data_leakage,test_token_windows}.py`, `tests/fixtures/articles.txt`; extend `config.py`, `cli.py`; produce asset/tokenizer/split manifests.

**Interfaces:**

- `prepare_data(config: RunConfig) -> dict[str, object]`; CLI `prepare-data --config PATH`.
- `iter_effective_batches(config: RunConfig, cursor: dict) -> Iterator[ReplayableBatch]`.
- `evaluation_windows(tokens: list[int], context: int) -> list[dict[str, object]]` records input IDs, exact original target positions and score masks.
- `open_population(name: str, phase: str, protocol_sha256: str | None) -> object`; development cannot open official validation/test. Frozen monitoring and final-test access require explicit protocol roles.

- [ ] **Step 1: Write tiny article/token fixtures for titles/subheadings/punctuation/empty text, duplicate groups, chunk continuity and evaluation windows.** Include unequal valid masks, zero-valid microbatch and all-empty effective batch before cursor mutation.

```python
def test_evaluation_scores_every_target_once(token_article):
    windows = evaluation_windows(token_article, context=8)
    scored = [p for w in windows for p in w["scored_target_positions"]]
    assert sorted(scored) == list(range(1, len(token_article)))
    assert len(scored) == len(set(scored))
    assert max(len(w["input_ids"]) for w in windows) <= 8
```

- [ ] **Step 2: Run `python -m pytest tests/test_data_leakage.py tests/test_token_windows.py -q`; expect missing data functions.** Fixture sequence contains BOS at position 0 and EOS last; assertions therefore exclude BOS and include EOS.
- [ ] **Step 3: Implement blueprint §7.1's exact preprocessing/access rules.** Pin approved WikiText-2 revision, line-ending-only normalization, whole-article 90/10 hash split, fitting-only target-8192 byte BPE with BOS/EOS/PAD, duplicate/exclusion manifests, same-article chunks and stride `floor(S/2)` target windows. Add verified Fashion-MNIST only when needed. Add actual locked/hashes of dataset/tokenizer dependencies now, not during protected runs.
- [ ] **Step 4: Run the tests and `python -m oar.cli prepare-data --config configs/lm_dev.yaml` on approved assets.** Inspect a small article-boundary sample, target counts, split/source/tokenizer hashes and notice record. Asset failure blocks language claims; do not silently substitute Fashion-MNIST or another corpus.
- [ ] **Step 5: Commit `feat: prepare leakage-controlled articles and exact token scoring`.** Commit manifests/config/tests, not corpus excerpts or caches.

### Task 8: Shared models, marked recall and eligibility (WP03)

**Files:** Create `src/oar/{models,synthetic}.py`, `tests/{test_models,test_synthetic,test_parameter_eligibility}.py`; extend model config.

**Interfaces:**

- `build_model(config: RunConfig) -> torch.nn.Module`, `eligible_weights(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]` in `models.py`.
- `make_recall_examples(seed: int, count: int, minority_probability: float, distractors: int) -> list[dict[str, object]]` in `synthetic.py`; labels refer to first/last occurrence according to a visible regime marker.

- [ ] **Step 1: Write causal-mask, shared-parameter and label tests.** Construct an accidental two-module shared MLP parameter; both aliases must be excluded from approximation, not merely deduplicated into an eligible entry.

```python
def test_tied_output_and_shared_hidden_weights_are_exact(shared_model):
    selected = eligible_weights(shared_model)
    ids = {id(p) for p in selected.values()}
    assert id(shared_model.embedding.weight) not in ids
    assert id(shared_model.shared_hidden_weight) not in ids

def test_recall_regime_marker_removes_ambiguity(repeated_key_pair):
    assert repeated_key_pair.first_rule.target == repeated_key_pair.first_value
    assert repeated_key_pair.last_rule.target == repeated_key_pair.last_value
    assert repeated_key_pair.first_value != repeated_key_pair.last_value
```

- [ ] **Step 2: Run `python -m pytest tests/test_models.py tests/test_synthetic.py tests/test_parameter_eligibility.py -q`; expect missing builders.**
- [ ] **Step 3: Implement the shared pre-LN decoder and residual MLP from spec §6.** Decoder expansion 4, tied embedding/output, learned positions, primary dropout 0, tier A/B/C/D dimensions exactly as spec. Use validated common causal attention. Synthetic training uses separate 0.01/0.001 minority conditions, independent balanced evaluation, resampled mappings, equal lengths and marked rules; expose a majority-to-mixture shift configuration. No BatchNorm.
- [ ] **Step 4: Run the tests; expect pass.** Task 9 establishes dense learnability before these synthetic tasks become approximate-method guardrails. Test nonzero dropout paths even though primary experiments start at zero.
- [ ] **Step 5: Commit `feat: add shared scratch models and marked recall guardrails`.**

### Task 9: Minimal native trainer, inclusive clock and atomic resume (WP03)

**Files:** Create `src/oar/{methods,train,checkpoints}.py`, `tests/{test_replay_resume,test_budget_endpoints,test_dense_training}.py`; modify `timing.py`, `cli.py`.

**Interfaces:**

- `TrainingMethod` lifecycle: `begin_step(context: StepContext, replayable_batch: ReplayableBatch) -> None`, `backward_microbatch(batch: dict, loss_weight: float) -> None`, `propose(model, optimizer_state) -> PendingUpdate`, `decide(pending_update: PendingUpdate) -> Decision`, `commit_or_replay(decision: Decision, replayable_batch: ReplayableBatch) -> StepRecord`, `state_dict() -> dict`, `load_state_dict(state: dict) -> None`. Dense execution bypasses OAR-only stages internally.
- `run_training(config: RunConfig, method: str, resume: Path | None = None) -> Path` in `train.py`; CLI `train --method NAME --config PATH [--resume PATH]`, initially only `dense_adamw`.
- `save_checkpoint(state: CheckpointState, directory: Path, clock: ActiveClock) -> dict[str, object]`, `load_checkpoint(path: Path, expected_hashes: dict[str, str]) -> CheckpointState`, `select_endpoint(checkpoint_records: list[dict], horizon_s: float) -> dict | None` in `checkpoints.py`.

- [ ] **Step 1: Write tiny native A/A, save/resume and time-budget tests.** Cover RNG with dropout, unequal-token loss aggregation, buffer snapshots, corrupt temporary checkpoint, insufficient disk, two-checkpoint retention and mismatch refusal. Use deterministic clock costs to place serialization beyond `H`.

```python
def test_serialization_overrun_retains_earlier_checkpoint():
    records = [
        {"id": "a", "status": "committed", "saved_active_wall_s": 9.0},
        {"id": "b", "status": "committed", "saved_active_wall_s": 10.2},
    ]
    assert select_endpoint(records, 10.0)["id"] == "a"

def test_restart_cost_is_charged(resume_case):
    assert resume_case.active_time_after_load == pytest.approx(13.0)
    assert resume_case.checkpoint_active_time == pytest.approx(10.0)
```

- [ ] **Step 2: Run `python -m pytest tests/test_replay_resume.py tests/test_budget_endpoints.py tests/test_dense_training.py -q`; expect absent trainer/checkpoint behavior.** `resume_case` uses fake loading cost 3 seconds; no machine-specific timing assertion.
- [ ] **Step 3: Implement native training, append-only events and atomic checkpointing.** Begin timing before model/optimizer construction; charge monitoring/checkpoints/restart, mask/token weight correctly, preserve all spec checkpoint fields. Store accumulated elapsed time at save start in the payload; after flush/rename, record completion-inclusive eligibility time in the durable checkpoint index, charging index work too. A checkpoint without a valid completion record cannot be an endpoint. A crash loses uncommitted training state but never audit reservations. Profiling stays separate.
- [ ] **Step 4: Run tests and short development `train --method dense_adamw` runs on the selected feasible scale.** Verify exact A/A numerical behavior, known-slower detection, useful dense learning and both synthetic regimes' learnability. These are measured gate prerequisites, not hardcoded learning-rate/loss tests. Record dense curves and forecasts with actual memory. Stop if the harness fails or the task is unlearnable.
- [ ] **Step 5: Commit `feat: add native training with charged endpoints and atomic recovery`.**

### Task 10: Simple competitors and faithful-port feasibility (WP03)

**Files:** Create `src/oar/baselines/{__init__,projection,sampling,instant,vcas,dropbp}.py`, `tests/{test_simple_baselines,test_baseline_ports}.py`; modify method registry/config; create upstream notices/patches and `manifests/baseline_feasibility.json`.

**Interfaces:**

- `build_method(name: str, config: RunConfig, model, optimizer) -> TrainingMethod` in `methods.py`; names `dense_adamw`, `dense_checkpointed`, `dense_muon`, `fixed_projection`, `scheduled_projection`, `weight_sampling`, `instant_literal`, `instant_paper_intent`, `vcas`, `dropbp`. Unsupported/unvalidated names raise `MethodUnavailable` with evidence reference.
- `sample_weight_gradient(x: Tensor, d: Tensor, samples: int, uniform_floor: float, generator: torch.Generator) -> Tensor` in `baselines/sampling.py`.
- Named-port adapters must expose the same lifecycle but preserve original algebra/controller/RNG, not force exact adjoints onto methods that approximate them.

- [ ] **Step 1: Write competitor tests.** Tiny finite-population expectations for sampling (including duplicates, all-zero scores and with-replacement `k=n`), exact input branch, fixed schedule independence from probes, native checkpoint recomputation equivalence and eligible-only Muon groups. Test source-helper spectra/boundary/zero-energy behavior for both INSTANT labels and full controller/sampling paths for any named VCAS claim.

```python
def test_sampling_zero_scores_is_exact_zero():
    x = torch.zeros(3, 2, dtype=torch.float64)
    d = torch.ones(3, 4, dtype=torch.float64)
    result = sample_weight_gradient(x, d, 2, 0.1, torch.Generator().manual_seed(7))
    assert torch.equal(result, torch.zeros(4, 2, dtype=torch.float64))
```

- [ ] **Step 2: Run `python -m pytest tests/test_simple_baselines.py tests/test_baseline_ports.py -q`; expect missing methods.** Use pinned local source fixtures or archived equivalence outputs for port tests, without requiring network during tests. Unavailable ports have explicit evidence/exclusion tests, not falsely passing equivalence tests.
- [ ] **Step 3: Implement B0–B4/B6 and bounded B7/B8/B9 feasibility attempts.** Preserve blueprint §6.2's full INSTANT contractions/calibration and literal/paper-intent helper distinction; charge extra no-update calibration passes. Inspect full VCAS sampling/controller dependencies before declaring fidelity. DropBP must detach actual branches and preserve Python RNG. Attempt Muon with installed official API semantics and separate tuning; no assumption that an API name proves MPS support. Use smaller gathered sampling GEMMs rather than zero-masked dense GEMMs.
- [ ] **Step 4: Run tests and short development comparisons; write WP03 gate.** Establish strong native and closest-art paths or record precisely blocked scope. Defer expensive port optimization/tuning until WP04 if necessary; complete feasibility attempts and fair selected-baseline tuning before freeze. Native dense remains the headline reference; custom exact/full basis is D0 overhead control only.
- [ ] **Step 5: Commit `feat: add simple competitors and auditable baseline ports`.** Record each source revision/patch/notice; do not commit upstream binaries or silently incomplete named algorithms.

### Task 11: Future-batch oracle discrimination before online integration (WP04)

**Files:** Create `src/oar/oracle.py`, `tests/test_oracle.py`; modify `cli.py`; produce `analysis/oracle_audit.parquet` plus schema/aggregate summaries.

**Interfaces:**

- `audit_checkpoint(checkpoint: Path, past_batches: list[ReplayableBatch], query_batches: list[ReplayableBatch], ranks: tuple[int, ...], costs: CostTable) -> list[dict[str, object]]`.
- CLI `audit-checkpoints --config PATH`; config names initialization/early/mid/late development checkpoints, disjoint past/future batch IDs, layer selection and coverage/uncertainty target. Extend strict config accordingly.
- Each row contains checkpoint/query/layer/rank/basis-age IDs, energy/cosine, exact residual, ordinary/complement/probe margins, same-batch and independent-development finite loss deltas, actual proposal/dense displacement and charged oracle cost.

- [ ] **Step 1: Write constructed-state tests for temporal leakage, copied-state isolation, stale-vs-current basis labels and displacement counterfactuals.** Include projected SGD, flat/rotating spectra, rare low-energy coordinates and true negative-alignment Adam states; fixture frequency is not claimed as empirical prevalence.

```python
def test_query_never_enters_past_basis(oracle_case):
    rows = audit_checkpoint(**oracle_case.arguments)
    assert all(row["basis_source_step"] < row["query_step"] for row in rows)
    assert oracle_case.live_state_unchanged()
    assert all(row["oracle_cost_s"] >= 0 for row in rows)
```

- [ ] **Step 2: Run `python -m pytest tests/test_oracle.py -q`; expect missing oracle pipeline.**
- [ ] **Step 3: Implement streamed diagnostics and copied-state counterfactuals.** Use actual optimizer history and independent development batches. Keep exact-current-subspace/residual oracles labeled and charged separately. Compare no-information fixed schedules, relative error, ordinary norm and complement margins at matched proposer/information/cost. Archive scalars/sketches and selected matrices, not every dense gradient. Add actual locked Parquet dependency only if needed for the output.
- [ ] **Step 4: Run tests and `python -m oar.cli audit-checkpoints --config configs/operator.yaml`.** WP04 passes only when useful past-only proposals, additional audit discrimination and plausible inclusive headroom coexist with adequate precision. No exact-oracle advantage means stop larger auditor work; absent cheap probes despite oracle value is an information–cost gap. Inconclusive prevalence is reported with uncertainty. Go directly to Task 17 if stopped.
- [ ] **Step 5: Commit `feat: test past-only gradient reuse and oracle audit value`.**

### Task 12: Minimum online OAR controller and exact fallback (WP05)

**Files:** Create `src/oar/controller.py`, `tests/{test_controller,test_online_oar}.py`; modify `methods.py`, `train.py`, `audit.py`, `checkpoints.py`, `test_replay_resume.py`.

**Interfaces:**

- `next_controller_state(state: ControllerState, decision: Decision, layer_audits: tuple[LayerAudit, ...], costs: CostTable) -> ControllerState` in `controller.py`; pure and deterministic.
- `ControllerState`: ranks (`int` or `dense`), refresh `K`, success streak, shrink `h`, `tau`, temporary one-dense-next-step flag, prior ranks and stable layer IDs. `CostTable` uses the actual shape-specific entries from Task 6.
- `OARMethod` in `methods.py` implements the Task 9 lifecycle using Tasks 3–5; register name `oar`. `StepRecord` logs raw alignment, residual margin/allowance, reasons, valid/processed counters and state versions.

- [ ] **Step 1: Write forced-rejection, all-accept/all-dense and controller fixtures.** Cover cost-nonmonotonic rank choices, stable ties, one-layer changes, `a<=0`, shrink streak reset, schedule-only refresh/re-entry, version errors, NaN/uncertain margins and deliberate checkpoint interruptions. Include nonzero dropout, shared exclusions, unequal/empty microbatches and activation-checkpoint recomputation.

```python
def test_rejection_matches_exact_with_dropout(rejection_case):
    case = rejection_case  # same initial state, batch, model RNG; forced rejection
    case.run_oar_and_native()
    case.assert_committed_states_close()
    assert case.oar_record.valid_tokens_total == case.native_record.valid_tokens_total
    assert case.oar_record.processed_tokens_total == 2 * case.batch_valid_tokens
    assert case.oar_scheduler_steps == case.native_scheduler_steps == 1
    assert case.audit_counter_after > case.audit_counter_before
```

- [ ] **Step 2: Run `python -m pytest tests/test_controller.py tests/test_online_oar.py tests/test_replay_resume.py -q`; expect absent integrated OAR behavior.** Test all-empty batch refusal before live mutation or cursor movement. Probe allocations already durably made remain spent if an error occurs later.
- [ ] **Step 3: Integrate the full logical transaction from blueprint §10.4.** Use one fixed capture/probe set per eligible layer/effective batch, raw full-gradient alignment, explicit complements and frozen empirical uncertainty policy. Commit accepted pending state once; on reject restore only appropriate training/model state, replay exact and advance logical state once. Preserve audit ledger and discarded-attempt costs. Snapshot forward buffers if supported; prevent duplicate sketches during checkpoint recomputation. No late rank changes or hidden exact `dW` on accepted approximate steps.
- [ ] **Step 4: Run tests, a forced-rejection tiny-model experiment and a complete measured WP04-region cycle.** Compare uninterrupted/resumed/native states, refresh/replay/checkpoint peaks and full-path costs. Write WP05 evidence. A correct slow method may proceed to bounded attribution/failure analysis, but does not earn scale-up.
- [ ] **Step 5: Commit `feat: integrate audited OAR updates with exact transactional replay`.**

### Task 13: Audit ablations and no-information schedule controls (WP05)

**Files:** Create `src/oar/baselines/auditors.py`, `tests/test_audit_controls.py`; modify registry/config/controller adapters.

**Interfaces:**

- Register `relative_audit`, `ordinary_audit` and `matched_schedule` in `build_method`.
- `decide_relative(bounds: dict[str, float], projected_norms: dict[str, float], epsilon_f: float) -> Decision`; relative slack is squared and contributions are `b_l²`.
- `fit_schedule(development_records: list[StepRecord], max_steps: int) -> dict[int, dict[str, int | str]]` in `baselines/projection.py`. Produced schedule is hashed/frozen before confirmation; it never reads current probes or protected outcomes.

- [ ] **Step 1: Write all-zero/nonzero residual relative-denominator tests, ordinary/complement geometry tests and information-access tests.** Spy on preview/complement calls: relative audit may use native AdamW after acceptance and must avoid OAR-only preview work.

```python
def test_relative_zero_denominator():
    assert decide_relative({"w": 0.0}, {"w": 0.0}, 0.1).kind == "accept_approx"
    assert decide_relative({"w": 1.0}, {"w": 0.0}, 0.1).kind == "reject_replay"
```

- [ ] **Step 2: Run `python -m pytest tests/test_audit_controls.py -q`; expect missing control policies.**
- [ ] **Step 3: Implement B5 and D3 per blueprint §§6.2/12.1.** Same candidates, probe spending, uncertainty treatment, replay and scheduled refresh information. Relative uses rejection/shrink squared slack thresholds 1 and 1/2; ordinary uses full displacement norm. Each threshold is tuned separately under equal charged development compute. Matched refresh/fallback schedules are trained on development only. Label any retrospective confirmatory schedule replay diagnostic-only.
- [ ] **Step 4: Run tests and short attribution comparisons.** Plot/record quality–time and quality–rank/refresh frontiers; distinguish proposer, policy, extra exact information, optimizer/schedule and kernel improvements. Higher acceptance alone is not a learning result. Finish WP05 attribution evidence.
- [ ] **Step 5: Commit `feat: add fair audit ablations and information-matched schedules`.**

### Task 14: Fair pilot selection, precision planning and protocol freeze (WP06)

**Files:** Create `src/oar/protocol.py`, `tests/test_protocol.py`; modify config/CLI; produce `configs/protocol_frozen.yaml`, development trial index and forecast artifacts.

**Interfaces:**

- `run_pilot(config: RunConfig) -> dict[str, object]`, CLI `pilot --config PATH`.
- `freeze_protocol(input_path: Path, out: Path) -> str` returns canonical SHA-256; CLI `freeze-protocol --input PATH --out PATH`.
- `load_protocol(path: Path) -> Protocol` verifies schema/hash, prerequisite evidence and required selections. `Protocol` includes all blueprint §8.1 fields, method/baseline identities, audit/guardband policy, fresh paired seeds/fixed sample size, H/cadence, targets, guardrails, invalidations, statistical assumptions and exact protected-scoring role/population.

- [ ] **Step 1: Write invalid/missing essential-field, tamper and protected-selection tests.** Include unresolved guardband, missing closest-art attempt, unsupported comparison claim, inadequate endpoint cadence, missing paired seed count and changed primary metric/data/method identity.

```python
def test_freeze_refuses_unresolved_horizon(unresolved_protocol, tmp_path):
    with pytest.raises(ValueError, match="horizon"):
        freeze_protocol(unresolved_protocol, tmp_path / "frozen.yaml")

def test_amendment_changes_protocol_identity(protocol_pair):
    assert protocol_pair.original_sha256 != protocol_pair.amended_sha256
```

- [ ] **Step 2: Run `python -m pytest tests/test_protocol.py -q`; expect missing freeze/validation.**
- [ ] **Step 3: Implement development selection and freeze.** Start AdamW brackets at rates `{1e-4,3e-4,1e-3,3e-3}` and decay `{0,0.1}`, expanding boundary winners; tune other optimizers on their own conventions. OAR explores spec `tau/h`, measured ranks/q/drift-based K and numerics. Equalize charged development compute, retain every trial/failure, and finish required port feasibility/fidelity obligations. Select one architecture/OAR configuration and strongest executable comparator plus matched control. Derive H/cadence/quality targets/noninferiority margins/sample size from actual dense curves, paired pilot variance and attainable absolute NLL precision. Freeze one justified inferential method and all resource projections.
- [ ] **Step 4: Run tests, `pilot`, then `freeze-protocol` on actual development evidence.** A complete schema is necessary but insufficient: inspect affordability/identifiability and common optimizations, record all calls and WP06 decision. Unaffordable precision or nonpositive tuned frontier is a stop/inconclusive result, not permission to weaken comparators or leak protected data.
- [ ] **Step 5: Commit `feat: freeze fair development-selected confirmatory protocols`.** The frozen file is created only after real pilots; never commit a placeholder as an executed freeze.

### Task 15: Fixed paired execution and honest endpoint inference (WP07)

**Files:** Create `src/oar/analysis.py`, `tests/{test_confirmation,test_statistical_analysis}.py`; modify `protocol.py`, CLI/trainer; produce run-level protected result index and `analysis/confirmatory_results.csv`.

**Interfaces:**

- `run_confirmation(protocol: Protocol) -> Path` in `protocol.py`; CLI `confirm --protocol PATH`.
- `analyze_confirmation(protocol: Protocol, results: Path) -> dict[str, object]` in `analysis.py`; CLI `analyze --protocol PATH` reads the protocol's fixed result index.
- `paired_summary(differences: list[float], confidence: float = 0.95) -> dict[str, float]`, `holm_adjust(p_values: list[float]) -> list[float]`; implement the preselected alternative when required by the frozen pilot. Add and lock a statistical dependency only if needed for that analysis.

- [ ] **Step 1: Write fixed-seed/order, saved-endpoint, tamper and analytic-fixture tests.** Use published mathematical distributions or hand-computable small samples for statistics; resample whole paired runs. Include checkpoint overrun, missing endpoint, failed run, never-hit target, restart load cost, nonattainment and minority-margin failure.

```python
def test_holm_primary_family():
    assert holm_adjust([0.01, 0.04]) == pytest.approx([0.02, 0.04])

def test_failed_seed_not_silently_dropped(failed_pair_results):
    summary = analyze_confirmation(**failed_pair_results.arguments)
    assert summary["status"] != "confirmed_improvement"
    assert summary["failed_runs"] == 1
```

- [ ] **Step 2: Run `python -m pytest tests/test_confirmation.py tests/test_statistical_analysis.py tests/test_budget_endpoints.py -q`; expect missing analysis/runner.**
- [ ] **Step 3: Implement exact frozen execution/analysis.** Interleave/randomize paired complete-run order as frozen, preserve all seeds and attempts, enforce last fully saved endpoint `<= H`, forbid mid-confirmation policy changes and seal primary files before follow-up. Score only declared protected populations/models. Report NLL differences, means, 95% intervals, two-comparison Holm familywise 0.05, guardrails and failures. Require correct effect direction plus both comparisons/guardrails for a positive claim. Targets need next-evaluation confirmation; censored/clipped times are not uncensored speedups. Validation/evaluation labels stay outside policy interfaces.
- [ ] **Step 4: Run tests; only with WP06 passed run `confirm` and `analyze`.** Inspect individual paired rows, seed count, censoring, cold/active costs, endpoint unused budget and protocol identity. Record WP07 as positive, negative or inconclusive under the frozen analysis. Few or failed pairs remain visible; never stop when a p-value becomes favorable.
- [ ] **Step 5: Commit `feat: execute fixed paired comparisons and guarded statistical analysis`.** Commit aggregate/index artifacts without model tensors or restricted text.

### Task 16: Focused robustness and clean-process reproduction (WP08)

**Files:** Create `tests/test_robustness_protocol.py`, `configs/robustness.yaml`; modify `protocol.py`, CLI; produce separate robustness/reproduction summaries.

**Interfaces:**

- `run_robustness(protocol: Protocol, config: RunConfig) -> Path`; CLI `robustness --protocol PATH --config PATH`.
- `verify_reproduction(original: Path, repeated: Path, tolerances: dict[str, float]) -> dict[str, object]` in `analysis.py`; compares hashes, environment, trajectories, endpoint outcomes and declared numerical rather than assumed bitwise equivalence.

- [ ] **Step 1: Write tests for independent labels/seeds, missing agenda, incompatible amendment and unsuccessful reproduction.** A negative primary result selects targeted failure diagnosis rather than an automatic exhaustive grid.

```python
def test_reproduction_failure_narrows_claim(reproduction_pair):
    result = verify_reproduction(**reproduction_pair.arguments)
    assert result["reproduced"] is False
    assert result["claim_status"] == "unstable"
```

- [ ] **Step 2: Run `python -m pytest tests/test_robustness_protocol.py -q`; expect missing robustness functions.**
- [ ] **Step 3: Implement separate, prewritten robustness agendas.** Prioritize dropout/replay, batch/rank/basis-age, optimizer/clipping/decay, rare-rule/shift, secondary Fashion-MNIST and a clean-process main repeat. New precision/width/corpus/objective claims require fresh local protocol/equivalence gates and measured reasons. Do not change the original primary results; new claims need fresh seeds/protected evaluation where selection leaked.
- [ ] **Step 4: Run tests, then only justified robustness configurations and clean-process reproduction.** Record WP08 failures, numerical drift, scope limitations and resource costs. A failing replication or rare-slice guardrail defeats an unconditional claim; synthetic-only success remains synthetic evidence.
- [ ] **Step 5: Commit `feat: validate robustness and independent local reproduction`.**

### Task 17: Reproducible synthesis for completed or stopped studies (WP09)

**Files:** Create `src/oar/report.py`, `tests/test_report.py`, `README.md`; modify CLI; produce `analysis/report.md`, raw artifact index and final decisions.

**Interfaces:**

- `build_report(gates: Path, run_index: Path | None, out: Path) -> Path`; CLI `report --gates PATH [--run-index PATH] --out PATH`.
- Report accepts an early stop with no run index; it never fabricates absent training measurements. Evidence classes: algebra/reference, backend timing, exploratory learning, confirmed learning and unresolved.

- [ ] **Step 1: Write report tests for a CPU-only stop, unhelpful oracle, correct-but-slower OAR, blocked named baseline, wide intervals, rare regression and failed reproduction.** Check exact evidence classification and absence of unsupported success language/data.

```python
def test_early_stop_is_reportable_without_speedup(stopped_study, tmp_path):
    path = build_report(stopped_study.gates, None, tmp_path / "report.md")
    text = path.read_text()
    assert "No training speedup was established" in text
    assert stopped_study.reason in text
```

- [ ] **Step 2: Run `python -m pytest tests/test_report.py -q`; expect missing synthesis.**
- [ ] **Step 3: Implement source-traceable report and local reproduction instructions.** Opening states what changed, versus what, workload/backend, absolute quality/time/memory trade-off, uncertainty and exploratory/confirmed status. Include every stopped/blocked gate, failed trial, unavailable baseline, cold-start/replay cost and blueprint §14 outcome decision. Repeat the scientific/experimental/implementation challenge audits in blueprint §15. Offer only a measured continue/revise/narrow/stop decision; pivots require new specs.
- [ ] **Step 4: Verify actual final artifacts.** Run `python -m pytest -q` for implemented CPU suites, the explicit MPS marker suite when eligible, and the documented small reproduction in a clean process. Run `python -m oar.cli report --gates decisions/gates.jsonl --out analysis/report.md` with the run index when present. Inspect hashes, failed/stopped branches and README commands; record inability to reproduce as an outcome rather than omitting it. Write WP09 evidence.
- [ ] **Step 5: Commit `docs: publish reproducible OAR findings and execution outcome`.** Stage report, code/tests, decisions and small artifact indices deliberately; no automatic push.

## Plan self-review and handoff

- **Coverage:** Tasks 1–6 implement spec §§2/4/5's contracts and discovery/headroom gates; Tasks 7–10 implement §6 and the minimal valid harness; Task 11 enforces the decisive oracle gate; Tasks 12–13 implement online state and attribution; Tasks 14–16 implement §7's frozen evaluation/robustness; Task 17 implements §8's honest stopped/completed outcome. The task map covers WP00–WP09 without implementing pivots.
- **Interfaces:** One `TrainingMethod` lifecycle, canonical parameter IDs and shared transport records are used throughout. `PendingUpdate`, audit reservations, CostTable and Protocol have single owning tasks; no later task silently changes optimizer signs, normalization or current-rank dependence.
- **Review focus:** All five failure classes above have named owner tests, including crash spending, all-empty batches, dropout/recomputation, zero-versus-missing gradients and saved-endpoint overrun.
- **Proportion:** Bodies are left to the implementer. Code snippets pin assertions; actual hardware settings, horizons, precision targets and controller parameters remain measured gate outputs. Focused test fixtures are created with their owning task rather than a speculative universal fixture library.
- **Completion evidence:** This document describes future red/green cycles and experiments. Documentation review/link checks are the only checks performed while preparing the spec and plan. The runtime suite and empirical gates must be executed before any corresponding success claim.

After reviewing these documents, choose an execution method before implementation. Native execution is a reasonable default for this tightly coupled numerical pipeline; subagent-driven execution adds per-task independent review. Either method must honor stop gates, current-branch work and the existing scope of authorization.
