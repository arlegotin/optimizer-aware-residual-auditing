<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Stage E pilot dispatcher independent review

## Initial review disposition — v1, corrections required

**Not safe to freeze or run this v1 implementation.** The full initial specification/code-quality review is complete. The findings below are the stable correction list; the subsequent review will cover their fixes, the exact corrective delta and any regressions introduced by that delta. This is not an approval of future source hashes or a claim that the pilot passed.

Binding brief: `.superpowers/dw_sampling_reopening/pilot-dispatch-brief.md`, SHA256 `fc612e1ac6749411e6001b7886562e09c4b09f3a336b4763d9a437b2f19ca933`. Also read the original learning plan and remaining-dispatch plan, including global constraints and Tasks 1–4, and the v8 scope addition. Protected confirmation dispatch remains a separate task.

Initial exact package: `.superpowers/dw_sampling_reopening/pilot-dispatch-review-package.txt`, SHA256 `33a8afdcde57fe772b2cff4e7fda5b8883417d371f88662cb11709b365e3aaad`, recorded 2026-10-31 17:46:37 UTC. Base/current Git HEAD: `ce4a09ea64b0c2dc1f3663f51e6e0db17e858de9`.

| Reviewed file | v1 SHA256 |
| --- | --- |
| `analysis/run_dw_sampling.py` | `ac5b7ac25de854f9657e400917489ba08eb360ea5dc8f6a9e4460f407d0394ab` |
| `src/oar/sampling_lm.py` | `8e5d3395f3984d39de7140c8eb9de834d46615523b43467ac26c271a14412abb` |
| `tests/test_dw_sampling_protocol.py` | `f8b956e9352e081219c5f0bebc53044fe4968f00700cc51ed489d50db8ebde7a` |
| `configs/dw_sampling.yaml` | `1f1d4bf1c718a942a2e76d9649ab56c10194741ee9a5939a99321aab84e5ac47` |
| `decisions/dw_sampling_reopening.md` | `c7e68d46dd2213f6d1f923009e93ebaee3fb75d1cebed4969b38eaf1c7181190` |

The package was reconstructed in `/tmp/pilot-review-v1` from its declared Git base and unified diff; every reconstructed file matched the hashes above. Invented probes used that frozen copy, so subsequent working-tree changes do not affect these findings. The implementation report was not yet present during this initial review and must be read before final disposition.

## Stable required corrections

1. **P1 — abnormal worker exit can retain a successful aggregate status.** In v1 `launch_learning_worker` (lines 2733–2824), terminal classification can be numerical/environmental failure, but an existing raw result is copied into the aggregate without overriding its `status`. A worker that serializes a pass and subsequently exits nonzero can therefore enter selection as successful. Canonical budget enforcement alone does not repair this. The implementer independently identified the same defect. Require terminal failure precedence, retain the raw bytes unchanged, and test successful serialization followed by positive nonzero exit and signal termination.

2. **P1 — an unvalidated later source marker can hide a sealed qualification failure.** In `require_model_qualification` (v1 lines 2219–2233), `source_manifest.learning_action == 'learning'` skips the directory before validating an existing record seal. Independent invented-fixture probe: a sealed later `STOP_COST` qualification correctly superseded the pinned pass; changing only its sealed source-manifest action to `learning` made the old pass accepted. Validate every existing later sealed record and its source identity/action consistency before excluding a learning attempt. Incomplete legitimate learning must remain distinguishable without weakening sealed-record checks.

3. **P1 — a known numerical failure can become resumable after a later signal.** The v1 launcher honors environmental worker sidecars but not numerical ones. Independent real tiny subprocess probe: write `worker-failure.json` with `classification='numerical_failure'` and reason `nonfinite gradient`, then terminate with SIGTERM. The resulting aggregate was `INCOMPLETE_RESOURCE` and terminal classification `environmental_interruption`. A retained checkpoint could then satisfy environmental-resume authorization, retrying a known numerical failure. Durable numerical failure evidence must take precedence over a subsequent interrupt/signal. Test that resume is rejected while genuine environmental interruption remains resumable.

4. **P1 — restarted trials can reuse altered scientific aggregates despite intact contradictory raw evidence.** `execute_learning_trial` validates raw-result SHA and a few metadata/clock fields, but neither authenticates aggregate scientific fields nor checks an existing enclosing seal. Independent invented subprocess probe: raw endpoint NLL 2.5; seal the outer learning attempt; change only aggregate endpoint NLL to 1.0; `read_record` correctly rejects the prior seal, but a new outer attempt's `execute_learning_trial` returns NLL 1.0/status pass. Validate existing sealed parent evidence and reconstruct or authenticate reused aggregates against raw worker evidence plus the validated terminal. Test altered endpoint, monitors, status and completion evidence; unsealed interrupted parents must still support only justified durable continuation.

5. **P2 — secondary checkpoint lifetime is not bounded through completed evaluation.** V1 captures the extra update-1024 checkpoint outside the two-resume set, but never retires it after its separate evaluation. This is the implementer's independently identified pending correction. Require an own-checkpoint retirement after a durable evaluation receipt, preserving the creation/evaluation/pruning hashes and final primary checkpoint. Subsequent ledger validation and restart must recognize the legitimate retirement rather than treating it as unexplained missing evidence. An interrupted or failed secondary evaluation must not authorize premature pruning.

6. **P2 — precision fallback loses preceding support-failure evidence on a later fatal error.** `qualify_native_precision` keeps `attempts` only in memory. If BF16 is unsupported and FP16 then fails numerically/replay validation, the exception escapes and only the final error reaches the worker sidecar. Independent mocked-capability probe against exact v1 LM: two attempted modes, final FP16 nonfinite exception, no attempts metadata and no durable qualification artifact; the BF16 unsupported reason was lost. Persist per-mode qualification evidence or retain it in a structured fatal receipt, while keeping numerical failure fatal and distinct from unsupported. Preserve costs and skipped-update evidence where available.

7. **P2 — MPS precision replay imposes an unjustified bitwise floating-point gate.** V1 recursively applies `torch.equal` to all snapshot tensors, including MPS-computed floating model parameters and Adam moments. The existing native-equivalence and MPS recovery checks already use the prospectively declared `rtol=1e-4, atol=1e-6` for those tensors; counters remain exact. This stronger new condition can falsely halt native precision qualification after the full grid, despite already known native reduction nondeterminism. Before any AMP outcome, use the existing tolerance only for MPS floating model/Adam-moment tensors; preserve exact CPU replay, RNG/data/sampler bytes, step/logical/token counters, metadata and scaler state. Record the fixed comparison rule and observed deviations. Do not broaden tolerance to arbitrary snapshot fields or adjust it after observing an outcome. This is a source/spec inconsistency assessment, not a claim about an unrun actual AMP result.

All seven items were communicated to the parent. No source fixes were made by the reviewer. No additional broad audit is required for the corrective release; review the stable fixes and their exact delta.

## Qualification/source continuity inspected

Independently verified the pinned record SHA256 `d01c854daf8eb7f7fbc8505b0879b40bc869bfc469a141fa727ae086757c4209` from bytes and its seal. The enclosing record is `not_run`, nested qualification is `pass`, and no learning-success upgrade is authorized. Its source-manifest hash matches both record fields and artifact seal; `model_qualification.json` equals the sealed nested qualification. All 11 measured source hashes match Git revision `110be7739067eec0aaddef40e21763818691fea8`.

Verified frozen fused native selection and exactly five eligible arms: importance .125/.25 and uniform .125/.25/.5. Verified the sealed horizon values: H `1461.4585008954164`, R `29.399479834130034`, T_budget `1490.8579807295464`, Delta `29.229170017908327`, r_time `31.627891642889008`, B_max `2.3987216249806806`, block_updates 8, overshoot limit `14.614585008954164`, epsilon_eval `1e-6`, required_time_ratio `1.0`. No recalibration or outcome inspection occurred.

The complete v1 diff and transitive paths were inspected, not only selected ASTs:

- Old driver definitions changed only in `validate_worker_request` and `main`. Old model definitions changed only in the previously unconsumed `train_budget`; `run_training_worker` remains unchanged.
- All original FP32 model classes, `TrainingState`, initialization, optimizer construction, `ModelFixture`, data loading, evaluation, native/cost/calibration routines, statistics and helpers retain their old ASTs. Imports/module initialization are identical except the reviewed strict `PROTOCOL` digest and driver docstring. Added precision classes define methods without importing a backend or running device work at module initialization.
- Main/worker routing adds the explicit learning action while preserving the existing qualification path. New training/precision/diagnostic calls do not enter the old full-model qualification execution chain. Existing native/data modules are unchanged.
- Configuration is semantically identical to measured qualification after removing only `learning.action`, `learning_manifest_path`, `qualification_record_path`, and `qualification_record_sha256`. Those changes select the prospective route and pinned evidence; model, optimizer, assets, calibration and timing settings remain identical.
- Independently compared every file tracked at original revision `9d8ad069c51b4d0e6c13e9c6e465579e9b44a339`: **184 files, zero byte changes**.

Subject to the correction delta and final freeze review, this supports precise continuity of the original measured qualification, not attribution of its measurements to new source. The intentionally stale outer amendment/manifest is a release prerequisite owned by the parent, not a v1 implementation defect or current authority to run.

## Other reviewed obligations and verification limits

Inspected exact 27-trial grid/seed declaration, R0 and target helper reuse, own-LR B16 control, own uniform eligible/adverse selection, exact role deduplication, independent triples and simpler-control replacement/dominance, fixed-N planning, development-only cache authorization, protected validation/test exclusion, prospective forecast, context128 target coverage/cost isolation, nonzero-history synthetic shift, separate component observations, charged secondary capture, explicit resume ownership, checkpoint history and stream restoration, no automatic numerical retry, all-ended-interval charging and final canonical completion budget. No additional blocker was found in these v1 paths beyond the listed corrections.

The parent reported 773 CPU tests passed / 14 MPS deselected in 17.70 s for v1. This review did **not** rerun that unchanged full suite or claim an independent full-suite pass. It ran only the targeted invented provenance/terminal/aggregate/precision-prefix probes described above and independent source/config/historical hash checks. No hardware operation, real corpus-body read, protected loss evaluation, networking, Git mutation or source edit occurred. No subagents were spawned.

Final release requires the implementation report, correction regressions, exact replacement package/hashes, and final review update below. Actual backend qualification, current preparation/correctness, prospective metadata freeze and real scientific outcomes remain separate parent-owned execution obligations. No whole-pilot PASS is justified by this review.
