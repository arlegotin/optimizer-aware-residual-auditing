<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Torch-free worker preflight repair

Ownership is released for parent review, v9 source/config/amendment freeze and current preparation/correctness. No hardware run, new stage, real corpus read, network action or Git mutation was performed by this repair task. Only the driver, protocol tests and explicitly authorized manifest-path config field changed; this report is the requested handoff artifact.

## Observed failure and retained evidence

Base execution revision: `fafbcfeeef67c40c4da2a4d7d1de73f7dc104ce1` (driver SHA `07923e3415fe457274fec60eff42af7ff7191fb066f0d776a0787a47cd079c54`). Actual sealed v8 startup attempt: `analysis/dw_sampling/20261031T120154Z-reopened/pilot/attempt-0003/record.json`, SHA `5e2718ebcdaea047a92d58a3a6271f733ab49ba85c471c9fd0e18fa29093f8f7`.

The retained stderr at `trials/grid-4312224b176ec538/attempt-0001/stderr.txt` shows `main -> configure_torch -> RuntimeError: fresh hardware worker must configure policy before Torch import`. All27 child attempts failed at setup, with zero worker event files/model-update evidence. The stage's recorded invocation wall time is44.16540000005625s; its27 recorded child active intervals sum to41.98803141526878s. Both remain retained, with their distinct accounting meanings. The original generic `STOP_QUALITY` / `no complete finite native grid arm` status is unchanged and is not actual language-quality evidence. No bad gradient was drawn, so prospective v9 fresh execution is repair of startup, not retry of a numerically invalid draw.

Root cause: before policy entry, `validate_worker_request -> load_learning_freeze -> build_learning_manifest -> build_learning_grid` imported `oar.sampled_linear.named_seed`; that module imports Torch at module scope. A development request also reaches `build_development_trials`, which repeated that import. An explicit resume additionally reached `validate_resume_selection -> sampling_lm.load_checkpoint`, importing Torch during preflight. Existing in-process CPU tests already had Torch loaded and did not expose this order-of-import failure.

## Exact scoped changes

`analysis/run_dw_sampling.py`:

- New `learning_named_seed` uses only already imported stdlib JSON/SHA256/int operations and the exact existing formula: compact ASCII JSON `[base_seed, domain, *ids]`, SHA256 first8 bytes, unsigned big endian, modulo`2**63-1`, with the same argument validation.
- `build_learning_grid` and `build_development_trials` use that helper instead of importing the Torch-backed operator solely for seed derivation. Actual declared rows/IDs/phase/seeds/order and all initialization/data/sampling streams are unchanged.
- `validate_worker_request` defaults to Torch-free resume preflight. `validate_learning_worker_request` forwards the explicit deep-validation choice. `validate_resume_selection(deep=False)` retains explicit path/SHA selection, environmental-only eligibility, source/arm/seed binding through the ledger, own live receipt/kind/ownership/symlink checks, actual checkpoint byte hashing, and receipt clock <= cumulative charge; it does not deserialize Torch state.
- After the unchanged strict `configure_torch()` returns, `main` repeats the full resume guard with `deep_resume=True` before calling any learning worker. This executes the original weights-only CPU checkpoint schema/binding/frozen/clock validation. The unchanged LM/controller still restores and validates the complete model/native optimizer/data/sampler/scaler state before any optimizer update/commit. Parent-side launcher deep checks retain their original default. No request field or CLI flag can opt a resumed worker out of the post-policy proof.
- New narrow `historical_learning_git_blob` / `historical_learning_identity` allow a verified sealed prior learning record to keep its original action across v9. After the existing seal/source/action checks, a historical learning identity must match its own committed complete11-file source map, committed config SHA with action learning, original committed amendment SHA and exact old bound_files including every measured source. It is not compared to a different current freeze. Unsealed attempts still need the existing strict current source/config/amendment provenance. Tampered sealed identities and actual later failed qualification remain rejected. No general version registry or relaxed current config parser was introduced.
- `PROTOCOL` changes only to bind the authorized config bytes below.

`configs/dw_sampling.yaml`: only `learning.learning_manifest_path` changes from `analysis/dw_sampling/20261031T120154Z-reopened/prospective/learning-v8.json` to `analysis/dw_sampling/20261031T120154Z-reopened/prospective/learning-v9.json`. The strict canonical digest is now `88532f55d93bec6a682568fa3a7e8b4d0c79be94cd5d07d5cce5048d0fca3775`. Parsed old/new configs are exactly equal after removing that single path field. Action learning, qualification reference, all model/algorithm/precision/grid/LR/rho/seeds/data/schedule/H/target settings are unchanged. The immutable v8 manifest was not overwritten; v9 metadata remains parent-owned.

`tests/test_dw_sampling_protocol.py`: adds invented metadata/tiny CPU checkpoint fixtures and fresh-process import-order/deep-validation/history tests. No actual MPS backend execution or real data is used.

## RED -> GREEN and coverage

The first fresh-process reproducer exercised actual `main -> validate_worker_request -> load_learning_freeze/build_learning_manifest -> validate_learning_worker_request` before a policy-entry sentinel. It mocked only prerequisite/qualification authority using invented metadata and maps; manifest bytes/hash checking, deterministic builders, selection/development derivation, request declarations, path/ledger/checkpoint receipt checks and main ordering ran their real code. The child began with no Torch module, so pytest's previously imported Torch could not hide a leak. Normal, development and explicit-resume requests all initially failed the sentinel assertion. Seed-helper parity initially failed because the helper was absent.

After only the stdlib seed fix, normal/development/parity passed and explicit resume still failed: this isolated the remaining checkpoint-loader import. After separating structural from deep validation, all three paths reached policy entry with Torch absent. Two further fresh CPU subprocesses assert exact event order `POLICY -> DEEP -> WORKER` for a valid resume and `POLICY -> DEEP` (no worker) for a frozen checkpoint. Those subprocesses deliberately import Torch for CPU validation only after their policy sentinel; they never configure or execute MPS. A forged checkpoint SHA is rejected before policy with Torch still absent.

Parity covers672 combinations: eight base seeds, all four `init/data/sampling/order` domains, all eight FFN IDs with both samplers, order-stream IDs, empty IDs and ASCII escaping/Unicode. Each result equals the unchanged operator `named_seed`. A separate metadata-only audit confirms all27 actual declared grid rows, IDs, seeds and order equal the immutable v8 manifest exactly, without importing Torch.

Historical-freeze tests initially reproduced v9 rejecting a legitimate sealed old learning record. The fix preserves that record after current scope/source/amendment changes, and rejects mismatched committed source bytes, mismatched old amendment hash, or an old committed qualification action. Existing sealed-action-tampering and later failed/incomplete qualification tests remain green. A read-only fresh-process audit also verified the real attempt3 source identity against its own committed revision and amendment SHA `65098139b7a433aba28a6030f50bb0b75c5862e105715e17a29e24b510fd9fbf`, with Torch absent and raw status unchanged.

Commands/results:

```text
pytest protocol -k 'worker_preflight_torch_free or seed_metadata_preserves'
  RED4 failures; seed-only repair3pass/1remaining resume failure
pytest protocol -k 'resume_deep_checkpoint_proof or resume_prepolicy_rejects'
  RED2 failures/1pass, then GREEN with full repair
pytest protocol -k 'historical_sealed_learning'
  RED1 failure/3pass, then GREEN
PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps' tests/test_dw_sampling_protocol.py -k 'qualification_continuity or worker_preflight_torch_free or seed_metadata_preserves or resume_deep_checkpoint_proof or resume_prepolicy_rejects'
  25 passed,346 deselected in4.48s
PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'
  820 passed,16 deselected in19.67s
```

Read-only exact-diff AST comparison against fafbcfee reports only these existing driver definitions changed: `build_development_trials`, `build_learning_grid`, `main`, `require_model_qualification`, `validate_learning_worker_request`, `validate_resume_selection`, `validate_worker_request`. Added definitions are the three helpers listed above. `configure_torch` AST is exactly unchanged; it is neither weakened nor bypassed or moved after a Torch import. Original qualified FP32/model/operator/native/timing/statistical functions remain unchanged. LM and operator files match committed fafbcfee bytes exactly. Parent owns renewed exact-diff continuity review; AST equality is corroboration, not automatic broad approval.

## Final hashes and handoff

- Driver: `bc004d8716b1ab6a6ff8be4df7eb3328b69e0f7bec4f35ee1aebe1ca695266d3`.
- Protocol tests: `9965812491973defa2cf0fbd5c197fd8790e4563c66cf51b5441d8c899d6638c`.
- Config: `1bd8b99be2ddb27bbf1a0555287137733a4496cda8e0dad66e0274f302dde101`.
- Unchanged LM: `ccf487079280d385a9fed1b3eb7069921a34a15fd78cb6b33f7778644a405d3c`.
- Unchanged operator: `11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3`.

`git diff --check` is clean. No known startup-import path remains before worker policy entry in the audited normal/development/resume guards. Actual repaired MPS startup/learning is not claimed; it requires the parent's v9 review/freeze/current A/C and a fresh append-only E attempt. Existing startup failures, all costs, v8 source attribution and qualification H remain preserved. Ownership of the driver, protocol tests and authorized config-path change is released now.
