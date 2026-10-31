<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Stage E pilot dispatcher — final independent review

## Disposition

**APPROVED: safe to freeze the exact source hashes below, execute current preparation/correctness and the required guarded native checks, then launch the declared pilot only if those prerequisites pass.** All seven initial findings, the scoped v2 terminal/budget interaction, and the separately agreed secondary-control censoring reconciliation are **ADDRESSED**. No required correction remains from this review.

This approval includes exact renewed continuity for the unchanged B/native and D/operator execution paths and the sealed full-model FP32 qualification paths. It does not authorize a stale amendment, bypass a failed prerequisite, grant protected validation/test access, or declare an actual pilot/learning success. The parent must bind the exact manifest, source maps, continuity references and review artifacts in the prospective amendment and complete its normal commit/freeze before execution. Actual confirmation/test/reproduction dispatch remains separately scoped future work.

## Review identity and scope

Base/current checkout at implementation review: `ce4a09ea64b0c2dc1f3663f51e6e0db17e858de9`.

The complete initial code/spec/transitive review is retained in `pilot-dispatch-review.md`, SHA256 `7b2244c03a27680515d479ff91dd38075efaafc7649abfc38d3b1aac6e57f0a6`. Its historical v1 rejection remains valid for those old bytes; this final approval supersedes that release disposition only for the corrected hashes below.

The v2 inspection covered the seven corrective changes and prospective censoring reconciliation, using `pilot-dispatch-review-package-v2.txt` SHA256 `81f7a6158d8c1da9ee3f9629529b966fcc7841606a5e5e832316335cff23621a` and `pilot-dispatch-corrective-delta-v2.diff` SHA256 `3fd59c0543e880b1156cac15d515a8bd72bc887a3309e550b36562585a8342d1`. It found one residual interaction: a budget overrun could overwrite numerical failure status and permit an invalid adverse-control selection. The v3 review covered only that small corrective delta and its tests, retaining the completed initial/v2 inspections.

Final reviewed artifacts:

- `pilot-dispatch-review-package-v3.txt`: `84797f8daf13ff78322570def7d94fed33b79819e648686045585a3a64d64eaa`
- `pilot-dispatch-corrective-delta-v3.diff`: `362c202e346344e244c008cd3d23bc521c8761a11dd479cb5887553711bc8d50`
- `pilot-dispatch-report.md`: `49db0338222fff3fb38853740d92b0fd245842529269b4a70084d35dcd799dde`

The final implementation report was read, including its manifest builder/schema, recovery/clock authority, native precision policy, verification history, prospective censoring reconciliation and final terminal-budget repair. The report distinguishes CPU implementation evidence from unperformed hardware/learning outcomes.

## Exact approved source map

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

## Findings disposition

| Initial finding | Final disposition and evidence |
| --- | --- |
| P1: raw pass survives abnormal child exit | **ADDRESSED.** Aggregate construction gives terminal failure precedence and preserves raw bytes. The final delta records budget exceedance separately and changes status for budget alone only when terminal classification is complete. A numerical overrun stays numerical and cannot qualify the adverse uniform control. |
| P1: altered later action hides sealed qualification failure | **ADDRESSED.** Existing later records are seal-validated before classification, with source/action/map binding checks. Unfinished learning exclusion requires the exact current configuration/source/amendment provenance. Legitimate sealed/unsealed learning remains distinct from later qualification. |
| P1: numerical failure becomes resumable after signal | **ADDRESSED.** Known numerical sidecar or training-prefix evidence overrides later interruption. Ledger classification forbids numerical resume; budget annotation no longer obscures numerical eligibility. |
| P1: restart reuses corrupted scientific aggregate | **ADDRESSED.** Existing parent seals are checked; request/raw hashes and frozen trial bindings are checked; stored aggregate must exactly match deterministic reconstruction from raw evidence and terminal. Sealed and unsealed endpoint-corruption cases are covered. |
| P2: secondary lifetime unbounded | **ADDRESSED.** Own secondary deletion follows durable evaluation, directory fsync and retirement evidence. Primary-ledger reopening recognizes the separate terminal-bound retirement; unexplained deletion/tampering fails closed. The final primary checkpoint is retained. |
| P2: preceding precision support failure lost | **ADDRESSED.** Durable mode-attempt prefix retains BF16 unsupported evidence and a subsequent fatal FP16 error separately; numerical failure remains fatal. |
| P2: unjustified bitwise MPS floating replay | **ADDRESSED.** Existing rtol1e-4/atol1e-6 applies only to MPS floating model/Adam-moment tensors, with recorded policy/deviations. CPU replay and all RNG/data/sampler/scaler/metadata/counter fields remain exact. |

The secondary-control interpretation is also **ADDRESSED**. Primary candidate/native pairs remain complete, finite, budget-valid and uncensored. A complete finite secondary with an unobserved target retains all three target observations/nulls and `NOT_EVALUABLE`, `dominates: null`, `superiority_claim: false`; no H substitution, subset comparison or uniform promotion occurs. Missing/failed/numerical/budget-invalid controls still block. A primary role alias retains the primary uncensored requirement. This was a prospective alignment with the binding primary-pair/adverse-control rule, before learning outcomes, rather than a changed statistical threshold.

## Exact continuity approval

The full initial inspection and scoped corrective deltas establish that the old B/native and D/operator measured implementations, timing/statistical functions, native optimizer/reference procedures, sampler, hardware and data helpers are unchanged. The retained original B record is SHA256 `716ec10cfe1827b8e85f03deb5229484ff7c20b80d2c7178b1395c4b5acda506`; D is `d52f2062995810f4541022d02ed21e8f33ad72257675f66f8bed773c34fa9b18`. Renewed carryover to the exact current consumer map above is approved; their original measured attribution and prior sealed records must be preserved.

Full-model qualification continuity is separately approved for record SHA256 `d01c854daf8eb7f7fbc8505b0879b40bc869bfc469a141fa727ae086757c4209`, measured revision `110be7739067eec0aaddef40e21763818691fea8`, sealed source-manifest SHA256 `9b6c136a8da76e19ddffef57874ad9063fa0e7231e5628754fa258be6ff586e8`. The complete original 11-file source map was independently checked against that Git revision during initial review. Enclosing status remains `not_run`, nested qualification `pass`, learning_success false.

At final hashes, the only changed old driver definitions are `main` and `validate_worker_request`; the only changed old LM definition is previously unconsumed `train_budget`. `run_training_worker`, original FP32 classes/initialization/TrainingState/ModelFixture/evaluate/checkpoint helpers and all old native/model qualification/calibration/statistical functions remain unchanged. Imports/global initialization remain identical except reviewed protocol digest/docstring. Added classes/helpers perform no import-time device work. The changes to routing and strict config were inspected in full, including transitive callers; AST checks corroborate this review rather than replacing it. Config equality outside the four explicit learning action/reference keys was verified. The v3 delta changes only the new aggregate helper and tests, so it cannot alter those qualified paths.

The original fused variant, five eligible sampler/rho arms, 27-slot grid, H1461.4585008954164, R29.399479834130034, T_budget1490.8579807295464, Delta29.229170017908327, r_time31.627891642889008, B_max2.3987216249806806, block_updates8, overshoot_limit14.614585008954164 and epsilon_eval1e-6 remain the original sealed quantities. No recalibration or reattribution is approved. The original 184 tracked files were independently byte-compared during initial review and again at v2: zero changes; v3 touches only new driver/test files.

## Verification and execution boundary

Independent reviewer verification:

- v2 corrective regressions: `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps' tests/test_dw_sampling_protocol.py -k 'real_child_exit or secondary_retires_only or cannot_hide_sealed_later or known_numeric_failure or retains_bf16_unsupported or replay_allows_only or replay_rejects_other or reuse_authenticates or complete_censored_secondary or censor_reconciliation'` — **37 passed, 320 deselected, 2.16s**.
- Additional invented v2 ledger probe: valid terminal-bound secondary retirement permits primary-ledger reopening; altered retirement hash is rejected.
- The v2 numerical-terminal/budget/adverse-control interaction was independently reproduced before repair.
- Final v3 targeted regression: `PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps' tests/test_dw_sampling_protocol.py -k 'aggregate_budget_failure_priority'` — **3 passed, 357 deselected, 0.06s**.
- Final direct source/artifact hashes verified; final old-definition AST comparison reconfirmed; `git diff --check` clean.

The implementer additionally reports v2 full CPU **806 passed / 16 MPS deselected**, and final v3 scoped **20 passed / 340 deselected**. This review does not present those as independently repeated full-suite results. No unchanged broad suite was rerun; the parent's current frozen A/C checks remain required.

No hardware, network, real corpus-body access, protected loss, Git mutation, source edit or subagent was performed by this reviewer. Actual native precision/backend support, model learning effects and diagnostics remain outcomes for guarded execution. Scientific success cannot be inferred from source approval or CPU fixtures. Subject to the exact prospective metadata freeze and passing current required gates, the implementation is approved to proceed to its declared development pilot.
