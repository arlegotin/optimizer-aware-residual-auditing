<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Independent review — v9 Torch-free worker startup repair

## Disposition

**APPROVED at the exact hashes below: safe to freeze v9, run current preparation/correctness, then the affected Stage E pilot only if the required gates pass.** No required correction was found in this narrow repair. Renewed B/native, D/operator and v7 full-model FP32 qualification continuity is approved for this exact consumer map, retaining original measured source identities and calibration.

This review does not claim repaired actual-MPS startup or learning success. The parent must first bind the new source/config/manifest/review/continuity metadata in the v9 amendment and complete the prospective commit/freeze. A stale amendment is not launch authority. Protected official-validation losses and official-test bodies remain outside this task.

## Reviewed identity

Repair base, actual v8 execution: `fafbcfeeef67c40c4da2a4d7d1de73f7dc104ce1`. Review-package current HEAD: `09a8686bc0c6e383d5cf55068d09d41f29729a90`.

- `torch-free-worker-repair.md`: SHA256 `3c4ff4c292e2fcd0cc00611f0841853db54f7e75dd8841edbde805d7737d23ba`.
- `torch-free-worker-review-package.txt`: SHA256 `159a6e8fbe16ba043e2230296e5fd92b9fb38ecf9f0d4f11638c9b939eaa6f34`.
- Prior complete pilot review and corrective-delta disposition remain in `pilot-dispatch-review-v3.md`, SHA256 `16e1926eb56ffff93d1ba9607fde52cb5304f58ebdebc57d5ce70ff54f7666e8`. This review extends that inspection only across the scoped v9 repair; it does not repeat or replace the broad pilot audit.

Exact approved measurement map:

```json
{
  "analysis/run_dw_sampling.py": "bc004d8716b1ab6a6ff8be4df7eb3328b69e0f7bec4f35ee1aebe1ca695266d3",
  "configs/dw_sampling.yaml": "1bd8b99be2ddb27bbf1a0555287137733a4496cda8e0dad66e0274f302dde101",
  "decisions/dw_sampling_scope.md": "d91d9350dc117f83e39c648e48d435de75ab9ea5a3b6bc197b216c4b73c11181",
  "decisions/dw_sampling_reopening.md": "7d1cffdb861f8436cd0629b27bb4e9dea97f0f6ab1146ce22a0e9eefc5294e55",
  "src/oar/timing.py": "25351cd20abc14385dd1cb3e482c67f73619b1ead30900ca0150fe57072458e5",
  "src/oar/hardware.py": "4ba118d816eec9d255cfb3d7925416373119fe181f82529fb58a96347d8e5048",
  "src/oar/sampled_linear.py": "11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3",
  "tests/test_sampled_linear.py": "3ed4adbf6f62599172b0322374e84c7552f7fc4b46761f80057053469822eaa9",
  "tests/test_dw_sampling_protocol.py": "9965812491973defa2cf0fbd5c197fd8790e4563c66cf51b5441d8c899d6638c",
  "src/oar/sampling_lm.py": "ccf487079280d385a9fed1b3eb7069921a34a15fd78cb6b33f7778644a405d3c",
  "src/oar/sampling_data.py": "cfcf4b782eb9a444ce70f29df72e063518f794108020e51a5d12238118f06583"
}
```

## Failure evidence and interpretation

Independently read and seal-validated `pilot/attempt-0003/record.json`, SHA256 `5e2718ebcdaea047a92d58a3a6271f733ab49ba85c471c9fd0e18fa29093f8f7`. Its source-manifest hash and measurement map agree with its sealed identity. All 27 retained stderr files contain the strict pre-import-policy failure, and there are zero worker `events.jsonl` files. The 27 terminal intervals sum to `41.98803141526878` seconds; the enclosing recorded invocation is `44.16540000005625` seconds. These distinct costs remain charged and retained.

The raw enclosing status remains `STOP_QUALITY`. Given the common pre-policy traceback, this is a startup/dispatcher failure, not observed model quality, a sampled-gradient failure, or adverse AMP evidence. This repair does not erase the old outcomes/costs or justify retrying a bad draw: those launches never reached the model worker. The new prospective manifest/source identity creates a fresh affected E attempt without reusing or relabeling the v8 failed attempts.

## Scoped correctness findings

**Seed import repair — addressed.** The two pre-policy builders now call a stdlib-only helper implementing the unchanged compact-ASCII-JSON/SHA256 named-seed formula and argument checks. No operator import occurs merely to derive order metadata. The parity test covers 672 combinations. An independent fresh-interpreter metadata audit regenerated all 27 actual grid rows and compared the complete dictionaries against the immutable v8 manifest: exact equality, including IDs, arms, seeds, order seeds and order indices, with Torch still absent.

**Resume validation order — addressed.** Before policy entry, structural validation still requires environmental interruption, an explicit own live initial/periodic receipt, matching SHA/path, ownership/symlink guards, ledger bindings and a finite receipt completion bounded by the cumulative charged interval. It no longer deserializes a Torch payload. The actual `main` unconditionally repeats deep request/resume validation after the unchanged `configure_torch()` for every resume and before the learning worker; no request key or CLI switch can disable this. Original weights-only checkpoint validation and the unchanged complete training-state/controller restore remain in force. Parent-launcher validation keeps its deep default. Frozen checkpoints and unreceipted hashes remain rejected.

**Historical learning identity — addressed without relaxing current authority.** Existing later sealed records are validated before their action is trusted. The new historical classifier checks the recorded complete 11-file map against its own committed revision, exact committed config hash and learning action, and its original committed amendment hash/bound-file map. This prevents a valid old sealed learning attempt from turning into a qualification failure solely because the current freeze changes. The unsealed branch still requires exact current configuration/source/amendment identity; the new historical path is not available there. Genuine later qualification records, old qualification actions, mismatched committed bytes/amendments and edited sealed source markers still fail closed. The reviewer independently verified the real v8 attempt's historical identity using its committed bytes, with no Torch import.

**Current protocol/model scope — unchanged.** Parsed v8/v9 configs are exactly equal after removing only `learning.learning_manifest_path`; it changes to `learning-v9.json`, and the strict driver protocol digest is updated accordingly. No model, optimizer, precision rule, sampler, seed, grid, data, H/reserve/cadence, target or statistical threshold changes. The v8 manifest is retained. The prospective scope allows only a transient macOS `caffeinate -i -w <owned-driver-PID>` assertion with recorded PID/lifetime and equal application to all arms; it authorizes no permanent power preference or unrelated process change. This is an operational setup permission, not evidence that any assertion or hardware run has already occurred.

## Continuity approval

Read the complete repair diff and its affected pre-policy call chain, including config loading, prerequisite checks, manifest/grid/development construction, ledger checks, resume validation and the new historical-source reads. Those authority paths use stdlib and the existing Torch-free YAML/config loader before `configure_torch`. Original native/model qualification branches do not enter the new deep-resume or learning-seed code.

Independent comparison against the v8 execution revision found exactly these changed old driver definitions: `validate_worker_request`, `build_learning_grid`, `require_model_qualification`, `validate_resume_selection`, `build_development_trials`, `validate_learning_worker_request`, and `main`. The three additions are `learning_named_seed`, `historical_learning_git_blob`, and `historical_learning_identity`. `configure_torch` is AST-identical and remains before Torch payload loading/model work. Module initialization is identical except the reviewed `PROTOCOL` digest. All LM, operator, sampling-data, timing and hardware file bytes equal the v8 execution revision. The full-diff/transitive inspection, not AST equality alone, supports this approval.

Together with the prior complete review, this renews exact carryover of B record SHA256 `716ec10cfe1827b8e85f03deb5229484ff7c20b80d2c7178b1395c4b5acda506` and D record SHA256 `d52f2062995810f4541022d02ed21e8f33ad72257675f66f8bed773c34fa9b18` to the current map above. Their original measured implementation, configuration dependencies, statistics and attribution remain unchanged.

Separately, v7 full-model qualification record SHA256 `d01c854daf8eb7f7fbc8505b0879b40bc869bfc469a141fa727ae086757c4209`, measured revision `110be7739067eec0aaddef40e21763818691fea8`, retains its approved unchanged FP32 model/native/calibration paths. No LM or previously measured native function is altered in this repair. Preserve the nested PASS/enclosing `not_run` distinction, fused variant, five eligible arms and exact frozen H/calibration; no remeasurement or recalibration is justified merely by this Torch-free metadata repair. Current A/C and affected E still require new execution. Earlier original-184 integrity findings remain supported because this narrow delta touches none of those historical files.

## Independent verification and limits

Executed:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps' tests/test_dw_sampling_protocol.py -k 'worker_preflight_torch_free or seed_metadata_preserves or resume_deep_checkpoint_proof or resume_prepolicy_rejects or historical_sealed_learning or cannot_hide_sealed_later'
12 passed, 359 deselected in 3.32s
```

The fresh subprocess tests exercise actual main/request/manifest/declaration/ledger ordering with invented authority fixtures; prerequisite/qualification authority is mocked so they cannot authorize hardware or real corpus access. Valid resume observes `POLICY -> DEEP -> WORKER`; frozen resume stops after `POLICY -> DEEP`; a forged checkpoint hash is rejected before policy with Torch absent. Separately, the reviewer ran real retained-metadata/committed-history/grid checks in a fresh interpreter, confirming genuine historical identity, all-27 grid equality and no Torch import. Final direct source/report/package hashes were checked; `git diff --check` is clean.

The implementer reports a full CPU run of 820 passed / 16 MPS deselected and targeted 25 passed. The reviewer did not repeat that unchanged broad suite or perform actual MPS execution. No network, corpus-body read, protected loss, Git mutation, source edit or subagent was performed. Approval is for the exact prospective repair/freeze and required next guarded execution, not for an unobserved scientific result.
