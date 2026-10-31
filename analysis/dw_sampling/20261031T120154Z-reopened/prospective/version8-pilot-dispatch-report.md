<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Stage E pilot dispatcher implementation handoff

Implementation ownership is released to the parent after the final verification recorded below. This report does not claim an actual pilot PASS or learning success. No real-data training, protected loss, hardware benchmark, network acquisition, Git mutation, new source module, or historical-184 mutation was performed by this implementation task. Native MPS qualification and the actual learning pilot remain the parent's post-review/freeze/current-A/C actions. Protected confirmation dispatch is the separate next implementation task.

## Owned files and implementation

- `analysis/run_dw_sampling.py`: new pure `build_learning_manifest`, `build_learning_grid`, `learning_forecast`, qualification-only continuity verifier (`require_model_qualification`, separate `qualification_git_blob`), durable receipts/ledger/recovery, strict learning-worker guard and launcher, selection/control/development dispatch, secondary evaluation/retirement, and guarded diagnostic paths. Original definitions changed only: `main` and `validate_worker_request`. `PROTOCOL` is regenerated for the exact learning configuration. Original `load_config`, qualification, native/model cost/timing/statistics, and selection/statistical helpers remain unchanged.
- `src/oar/sampling_lm.py`: original `train_budget` receives incoming hash-bound checkpoint history, cross-invocation retention, controller/resume validation, secondary capture/history and durable logging/result serialization. Original `run_training_worker` remains unchanged. New isolated `NativePrecisionState`, `NativePrecisionUnsupported`, `precision_backend_unsupported`, `validate_native_precision_replay`, `qualify_native_precision`, and `run_precision_worker` provide baseline-only native autocast/scaler support. No change to `TrainingState` (including step/snapshot/restore), `ModelFixture`, `ByteLM`, optimizer construction, evaluation, sampler, parser, original checkpoint primitives or timing functions.
- `configs/dw_sampling.yaml`: exact `learning.action: learning`, qualification record path/SHA, fixed `learning_manifest_path`; no manifest SHA inside config (outer amendment binds it). Existing model, seeds, allowed grid, calibration, scheduler, statistical rules and resource guards remain fixed.
- `tests/test_dw_sampling_protocol.py`: invented metadata/tiny CPU integration tests and two new explicitly marked native MPS precision tests for later host execution. Old qualification-only fixtures explicitly select their original qualification action; production learning guards stay strict.

## Qualification and manifest contract

Base checkout: `ce4a09ea64b0c2dc1f3663f51e6e0db17e858de9`.
Measured full-model revision: `110be7739067eec0aaddef40e21763818691fea8`.
Qualification record: `analysis/dw_sampling/20261031T120154Z-reopened/pilot/attempt-0002/record.json`, SHA256 `d01c854daf8eb7f7fbc8505b0879b40bc869bfc469a141fa727ae086757c4209`.
Source-manifest SHA256: `9b6c136a8da76e19ddffef57874ad9063fa0e7231e5628754fa258be6ff586e8`.
Enclosing status remains `not_run`, nested qualification `pass`, `learning_success: false`.

The verifier checks the complete sealed qualification and its separate `model_qualification.json`, prepared data manifest/cache references, original committed bytes for all 11 measured sources (including both LEARNING files), exact current maps and config exclusions, and explicit renewed complete-diff/transitive/initializer review. It verifies every existing sealed later attempt before trusting/excluding its action; source/action/seal inconsistency fails closed. An unsealed later learning attempt is excluded only with the exact current action/config/source/amendment/bound-file provenance. Genuine later failed/incomplete qualification cannot be bypassed.

Parent amendment field `model_qualification_continuity` must contain:

```text
status: approved
record_path: <ROOT-relative exact qualification record>
record_sha256: <exact original record SHA>
source_manifest_sha256: <exact original sealed source manifest SHA>
measured_head: <original full 40-character execution HEAD>
previous_measurement_hashes: <complete original 11-file map>
current_measurement_hashes: <complete current 11-file pilot map>
changed_files: <sorted exact changed-file list>
executed_paths_unchanged: true
initializers_and_transitive_callers_reviewed: true
reviews: [{path: <ROOT-relative reviewed artifact>, sha256: <exact bytes>}]
```

Each current source is also required in outer amendment `bound_files`. Changed sampler/data/timing/hardware sources are forbidden. The AST checks corroborate the explicit complete review; they do not grant automatic broad carryover.

The pure API is:

```python
build_learning_manifest(cfg, qualification_record, *, qualification_reference,
                        measurement_hashes, config_sha256) -> dict
```

Manifest top-level schema is exactly: `schema_version`, `action`, `qualification`, `measured_source_hashes`, `measurement_hashes`, `config_sha256`, `native_variant`, `eligible`, `calibration`, `data`, `data_manifest`, `grid`, `development_seeds`, `permitted_worker_kinds`, `checkpoint_policy`, `diagnostics`, `forecast`, `protected_losses`. `data` includes only fitting-development; validation/test worker fields are rejected. `calibration` copies the entire sealed calibration verbatim. Every arm has exactly `sampler/rho/batch/lr/precision`; grid rows include deterministic ID, development phase, seed, roles and named order stream/index.

`qualification_reference` generated/expected by `require_model_qualification` contains six fields: `record_path`, `record_sha256`, `source_manifest_sha256`, `measured_head`, absolute `path`, and `sha256`. The parent can generate the prospective manifest before updating the amendment (no circular self-hash) with this metadata-only command, after final source review:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import importlib.util, json
from pathlib import Path
root = Path.cwd()
spec = importlib.util.spec_from_file_location('dw', root/'analysis/run_dw_sampling.py')
d = importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
config = root/'configs/dw_sampling.yaml'; cfg = d.load_config(config)
path = root/cfg['learning']['qualification_record_path']
record = d.read_record(path)
source = path.parent/'source_manifest.json'
identity = json.loads(source.read_text())
reference = {'record_path': str(path.relative_to(root)), 'record_sha256': d.digest(path),
             'source_manifest_sha256': d.digest(source), 'measured_head': identity['actual_head'],
             'path': str(path.resolve()), 'sha256': d.digest(path)}
manifest = d.build_learning_manifest(cfg, record, qualification_reference=reference,
    measurement_hashes=d.source_hashes('pilot'), config_sha256=d.digest(config))
d.durable_json(root/cfg['learning']['learning_manifest_path'], manifest)
PY
```

The parent then binds the manifest's exact bytes in outer amendment `bound_files`. This command reads sealed metadata/artifacts and code/config, not corpus bodies. The actual `load_learning_freeze` requires both that outer hash binding and exact equality with a freshly built manifest. Merely selecting config action learning grants no production launch permission.

Frozen values independently checked against sealed metadata: fused native; importance rho .125/.25, uniform .125/.25/.5; exact 27 slots (12 dense + 6 importance + 9 uniform), tuning seed3101. H1461.4585008954164, Delta29.229170017908327, R29.399479834130034, T_budget1490.8579807295464, r_time31.627891642889008, B_max2.3987216249806806, block_updates8, overshoot_limit_s14.614585008954164, epsilon_eval1e-6, required_time_ratio1, diagnostic_delta.2085046466766194. No recalibration or replacement slots.

## Worker, recovery and retention behavior

Every real trajectory uses one serial fresh child. The parent monotonic timestamp precedes request serialization/spawn/imports. Exclusive ownership, immutable launch/request/PID/exit evidence, a chained cumulative interval ledger and terminal hashes of worker prefixes retain every ended invocation, including wasted/uncommitted work. Canonical completion follows worker result serialization/fsync, process exit and parent result/prefix reads/hashes; it overrides the worker's earlier completion_s for T_budget. Raw worker result files are preserved. A pass-shaped raw file plus a failed terminal never yields a pass aggregate.

Known numerical evidence in either the structured worker sidecar or the training failure prefix wins over a later signal, parent interruption or filesystem error. No numerical trajectory automatically retries or resumes. Environmental recovery requires explicit checkpoint path+SHA selection (`--learning-resume` JSON mapping trial IDs to those exact pairs), full source/data/arm/seed/clock/controller compatibility and retained own receipts. Across outer pilot attempts, ended results are reused without spending new slots; histories/active charge remain cumulative. Every existing enclosing stage seal is checked before reuse. Scientific aggregate fields are reconstructed by the same transformation used at launch, from hash-bound raw worker results, terminal status/clock and frozen trial/request; a differing stored aggregate fails closed rather than being overwritten or trusted. Parent crashes without provable terminal intervals fail closed, even if the child later disappears; downtime is not invented as active time and an orphan is not silently resumed.

Two own live ephemeral resumes survive across invocations; a durable final receipt retires them all. Creation/pruning evidence is retained, other runs/user paths and frozen final checkpoints are protected, and post-terminal receipt insertion cannot adopt an orphan because worker prefix hashes are bound by terminal evidence. Incoming checkpoint/controller history is checked before live restore. Nonzero/scaler-skipped updates cannot be counted as completed valid evidence.

Each development arm prospectively captures update1024 (or records unavailable-if-not-reached), with capture/IO charged to its trajectory. The extra checkpoint is separate from the two live resumes. A resume from an older boundary retires an abandoned future secondary. Full secondary evaluation runs after the trajectory in its own charged worker; its result is durably recorded before deletion, directory fsync and a retirement receipt. Primary ledger reopening validates the evaluation/retirement against that separate worker's terminal prefix. A failed retirement/receipt or missing unknown state cannot silently produce a positive completed pilot.

## Selection, controls and diagnostics

The original `select_learning_arms`, `confirmed_target_time`, `simpler_dominates` and `plan_confirmation` implement the actual decisions; their formulas were not copied/changed. All declared grid outcomes are retained. Selection/control settings freeze before3102..3104. B16 uses its own finite endpoint optimum and epsilon_eval/lower-LR ties; uniform uses its own eligible fastest arm with r_time/rho/LR ties, or a disclosed mandatory finite adverse endpoint fallback. All-invalid uniform blocks a qualified control claim. Exact phase/seed/settings roles deduplicate without new alternatives. Independent development requires complete finite budget-valid triples for all mandatory roles; only primary candidate/dense pairs must also have uncensored target times. Matching initial parameters, strict geometric mean primary time improvement and prescribed primary mean/maximum endpoint guards remain mandatory. A complete finite secondary control that misses the target is recorded as NOT_EVALUABLE for dominance, with dominates null, superiority_claim false and its three actual target times including nulls. It adds no veto to otherwise valid primary planning, no H substitution or subset ratio is formed, and no superiority over that censored control is claimed. A censored uniform cannot be promoted; a secondary alias of either primary inherits that primary uncensored requirement. Missing, failed, numerical or budget-invalid mandatory controls still block planning. Dominating smaller dense/nativeB16/qualified precision stops; independently validated uniform can replace importance, otherwise uniform domination stops.

Precision control is isolated from original FP32 step/evaluate. Native BF16 autocast is preferred; disabled autocast is explicitly unsupported, not silently relabeled FP32 success. FP16 uses only installed enabled native MPS GradScaler, native AdamW and FP32 masters/moments; no AMP sampler, model.half or substitute scaler. Mode-attempt prefixes (`precision_attempts.jsonl`) durably retain unsupported BF16 even if FP16 then fails numerically/replay. Numerical failures remain fatal. The parent's prospectively accepted replay policy applies existing native MPS rtol1e-4/atol1e-6 only to floating model and Adam moment tensors. CPU execution is exact. RNG/data/sampler, counters, optimizer step, scaler, metadata and hyperparameters remain exact. Qualification receipts retain policy and maximum model/moment absolute deviations. This was decided before actual AMP outcomes.

The fixed diagnostic manifest declares seed62003, eight native nonzero-moment history updates, sign-flipped teacher regression, 32 shifted updates per dense/importance/uniform .125 arm; existing unbiased Adam counterexample is `tests/test_sampled_linear.py::test_unbiased_gradient_does_not_imply_unbiased_adam_displacement`. Context128 cost uses existing baseline block calibration/history/30-pair machinery in two sessions, with inclusive target coverage and a diagnostic-only slice of the fixed256 padding. Fresh native history follows calibration. Separate component profiles cover native and eligible sampled arms in two sessions; observed complete-step cost is separately measured and asynchronous components are never summed into a step estimate. All diagnostics dispatch through the same guarded pilot/config manifest, without protected losses or primary H changes.

Inclusive forecast is 27 grid + <=15 dev + precision1 +2N confirmation + secondaryB16 3 + repeats6 =52+2N trajectories (68 at provisional N8), with trajectory/setup/diagnostic/evaluation science allowances, endpoint/secondary/active storage and actual current memory/disk guards. T_budget is per trajectory, never a stage timeout. Actual N/t-critical/protected confirmation freeze remains the next task.

## Verification and review fixes

Observed RED→GREEN regressions cover missing dispatcher APIs/frozen-final/secondary capture, malformed worker JSON with durable terminal charge, forged restart charge and duplicate pairs, cross-attempt ownership/retention/native optimizer+data/sampler history, abandoned secondary capture, missing/smaller-dense controls, inclusive context128 target/padding/native history, disabled autocast, post-terminal orphan adoption, nonzero exit after raw pass, separately durable secondary retirement, forged sealed-later action, numerical-prefix precedence over SIGTERM, precision support-prefix retention, and the narrow prospective replay tolerance.

Representative commands actually executed (with invented fixtures only):

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k 'learning_grid_dispatch or learning_dispatch_manifest or development_dispatch_controls or learning_ledger or learning_resume_frozen or secondary_checkpoint or native_precision_cpu or native_precision_rejects'
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k 'malformed_worker or unreceipted_restart or duplicate_pair or real_child_exit'
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k 'uncommitted_future or mandatory_role_missing'
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k 'real_tiny_cpu_paths or qualification_prefers_bf16 or controller_metadata_rejected'
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k 'real_child_exit or secondary_retires or learning_ledger or cross_attempt_retention'
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -k 'qualification_continuity'
PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps' tests/test_dw_sampling_protocol.py -k 'replay_allows_only or replay_rejects_other or known_numeric_failure or native_precision'
PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'
git diff --check
```

Targeted post-fix counts: initial nine contract tests 9 pass; first malformed/charge/duplicate/real-child set 4 pass then the three reproduced defects fixed; later guarded/legacy targeted set15 pass; cross-attempt/foreign/reuse coverage3 pass (a fixture assertion was corrected to distinguish periodic from initial evaluations); tiny diagnostics/precision/controller set4 pass; launcher/secondary/retention set12 pass; qualification continuity14 pass including sealed-later regression; final precision/replay/numeric-precedence set17 pass; aggregate authentication/real-child/reuse/numeric-precedence set12 pass. Two newly marked native precision MPS tests are present for parent's later actual host execution; no successful MPS support evidence is inferred from CPU fixtures or unavailable-backend skips.

The invented end-to-end pilot exercised all27 grid rows,12 deduplicated independent runs,12 unavailable-if-not-reached secondary records, five diagnostic results and a fixed prospective N8 plan, with learning_success false. Actual tiny CPU diagnostics exercised native nonzero moment history, 30 paired blocks and separately observed component steps. Actual tiny CPU BF16 used native optimizer, FP32 master/moment tensors, complete snapshot/replay and exact FP32 evaluation.

Parent-requested review findings addressed in this same fix round: (1) verify sealed later provenance before action exclusion, (2) failed terminal overrides raw pass, (3) numerical evidence overrides later interruption, (4) persist unsupported/fatal precision prefixes, (5) narrowly retain existing native MPS floating replay tolerance, and (6) explicit durable secondary evaluation/retirement lifetime, and (7) enclosing-stage seal verification plus authenticated aggregate reconstruction on restart/reuse (sealed and unsealed positive/negative fixtures). Covering RED/green results are recorded above. The parent still owns independent broad exact-diff review and freeze.

## Remaining limits and release

No known required pilot path is intentionally stubbed. Actual native MPS precision support/replay, context128/component costs, learning effects and hardware recovery are unmeasured in this implementation session; their guarded future workers may legitimately return unsupported, stop or inconclusive. Missing parent terminal intervals remain deliberately non-recoverable without independent durable proof. Any failed, missing, numerically invalid or budget-invalid mandatory diagnostic/control, or failed retirement/evaluation, prevents actual pilot PASS. Target censoring of a complete finite secondary control alone does not veto the primary plan; its dominance remains NOT_EVALUABLE and supports no superiority claim. Full protected confirmation/test/reproduction dispatch and N>16 t-critical provenance are separate future work and do not authorize loss exposure now.

The current amendment is intentionally stale. The parent must review the exact final diff, regenerate/bind the manifest and continuity evidence/amendment, commit/freeze and rerun current preparation/correctness before any real learning launch. All four owned files are released; no further implementation edits are planned after the final hashes below.

## Final exact-byte proof and hashes

All11 original measured hashes matched committed110be blobs. AST comparison against that measured revision:

```json
{
  "analysis/run_dw_sampling.py": {
    "original_definitions": 80,
    "unchanged_original_definitions": 78,
    "changed_original_definitions": [
      "main",
      "validate_worker_request"
    ],
    "module_initializers_imports_unchanged_except_driver_docstring_PROTOCOL": true
  },
  "src/oar/sampling_lm.py": {
    "original_definitions": 25,
    "unchanged_original_definitions": 24,
    "changed_original_definitions": [
      "train_budget"
    ],
    "module_initializers_imports_unchanged_except_driver_docstring_PROTOCOL": true
  }
}
```

Current complete pilot source map (including parent-owned scope at handoff):

```json
{
  "analysis/run_dw_sampling.py": "07923e3415fe457274fec60eff42af7ff7191fb066f0d776a0787a47cd079c54",
  "configs/dw_sampling.yaml": "1f1d4bf1c718a942a2e76d9649ab56c10194741ee9a5939a99321aab84e5ac47",
  "decisions/dw_sampling_scope.md": "d91d9350dc117f83e39c648e48d435de75ab9ea5a3b6bc197b216c4b73c11181",
  "decisions/dw_sampling_reopening.md": "a7f442d2f526f09a3fed89a7cabc1e8c41a9c1ea4ad84c26780208794abd5e9e",
  "src/oar/timing.py": "25351cd20abc14385dd1cb3e482c67f73619b1ead30900ca0150fe57072458e5",
  "src/oar/hardware.py": "4ba118d816eec9d255cfb3d7925416373119fe181f82529fb58a96347d8e5048",
  "src/oar/sampled_linear.py": "11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3",
  "tests/test_sampled_linear.py": "3ed4adbf6f62599172b0322374e84c7552f7fc4b46761f80057053469822eaa9",
  "tests/test_dw_sampling_protocol.py": "7df50efb82f5a841f0d2f3d7da25fd131853a870b1ff046b00904369fa015d4e",
  "src/oar/sampling_lm.py": "ccf487079280d385a9fed1b3eb7069921a34a15fd78cb6b33f7778644a405d3c",
  "src/oar/sampling_data.py": "cfcf4b782eb9a444ce70f29df72e063518f794108020e51a5d12238118f06583"
}
```

Final full CPU verification after all review fixes and aggregate-reuse authentication: `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'` → **793 passed, 16 deselected, 17.60s**. `git diff --check` → exit0, no findings. The 16 deselected tests include the two new native precision backend tests; they remain for parent execution after freeze. Source ownership is now released. Actual Stage E remains not run; learning_success remains false.

## Prospective secondary-control censoring reconciliation

Before v2 freeze and before any learning outcomes, the parent/reviewer reconciled the original owner requirement: target observability is mandatory for primary candidate/dense pairs, while a mandatory finite adverse secondary control may miss the target. Only `finish_development` and invented protocol fixtures changed; config, LM and qualified FP32 paths stayed unchanged. Six RED cases (uniform/nativeB16/precision, one or all three seeds censored) now allow the unchanged valid primary N8 plan and retain NOT_EVALUABLE dominance/null target evidence. Seven negative cases preserve blocking for numerical, missing, budget-invalid, nonfinite-endpoint/clock, primary-censored and aliased-primary-censored evidence. Scoped `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps' tests/test_dw_sampling_protocol.py -k 'development_dispatch or grid_dispatch_to_independent'` →18 passed,339 deselected. No new science, loss exposure, time substitution or outcome-based retuning occurred.

Final verification after that reconciliation: `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'` → **806 passed,16 deselected,17.02s**. `git diff --check` is clean. The current source map above is refreshed for this release (driver `03aba1bc93bec1de7a3ae083f7f38a498dd76b160181b6581e5b8e497aee6174`, protocol tests `1c3d28a3ccd55395ae6cacfad5eb90b9c6e6d74ade44f11e6272d0c6288abf91`). LM/config hashes are unchanged from the prior release; original-qualified AST comparison still changes only driver main/worker routing and unconsumed train_budget. Ownership is released again for the parent's v2 review/freeze. Actual learning remains not run.

## Review round2: terminal eligibility before budget annotation

The sole remaining v2 review finding was reproduced RED: a pass-shaped finite uniform endpoint with a known numerical terminal and121/120 seconds was relabeled INCONCLUSIVE by the unconditional budget conversion, making it eligible as an adverse mandatory control. The narrow fix changes only new `aggregate_learning_result`: over-budget trajectories retain `budget_exceeded: true` and `budget_excess_s`; only a COMPLETE terminal converts to INCONCLUSIVE_MEASUREMENT. A numerical terminal remains NUMERICAL_FAILURE, and an environmental terminal remains INCOMPLETE_RESOURCE. This preserves terminal eligibility, raw result bytes and canonical clock evidence. The regression confirms an all-numerically-invalid uniform set remains STOP_CONTROL even with finite raw endpoints and budget exceedance. No other pipeline/helper/config/LM change was made.

Observed RED3 failures → GREEN20 passed,340 deselected in1.21s with `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps' tests/test_dw_sampling_protocol.py -k 'aggregate_budget_failure_priority or learning_ledger or reuse_authenticates or development_dispatch_controls'`. The prior full CPU run remains806 passed/16 deselected; per parent instruction no unchanged broad run was repeated for this small reviewed delta. The parent's post-freeze A/current correctness stage will run broad verification. No hardware, network, corpus body or Git mutation occurred.

Fresh released driver SHA256: `07923e3415fe457274fec60eff42af7ff7191fb066f0d776a0787a47cd079c54`; protocol tests SHA256: `7df50efb82f5a841f0d2f3d7da25fd131853a870b1ff046b00904369fa015d4e`. The complete source map above is refreshed. LM/config hashes remain unchanged. Source ownership is released for v3 delta review/freeze; actual learning remains not run.
