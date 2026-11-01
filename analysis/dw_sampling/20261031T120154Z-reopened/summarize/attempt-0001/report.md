<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Bounded dW sampling report

Terminal state: **STOP_QUALITY**.

Fixed-budget norm-importance dW sampling and the direct-randint uniform control retain native forward, exact input/bias adjoints and native AdamW. Historical OAR results are unchanged.

Qualification and component costs do not establish time to a held-out language target. Asset acquisition, parser manifests and any reached model qualification are recorded in their pilot attempt; the official test remains protected until its final reporting gate.

| Stage / attempt | State | Reason |
| --- | --- | --- |
| prepare / attempt-0001 | not_run | prepare-only dry run: validated immutable config/source/history; no device work/tests/training/downloads |
| prepare / attempt-0002 | pass | existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified |
| prepare / attempt-0003 | pass | existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified |
| prepare / attempt-0004 | pass | existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified |
| prepare / attempt-0005 | pass | existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified |
| prepare / attempt-0006 | pass | existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified |
| prepare / attempt-0007 | pass | existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified |
| feasibility / attempt-0001 | pass | upper95% paired ratio must be below 1-delta in both sessions at a planned-model middle shape |
| correctness / attempt-0001 | pass | CPU operator/protocol reference tests; actual-device qualification separately in prepare |
| correctness / attempt-0002 | pass | CPU operator/protocol reference tests; actual-device qualification separately in prepare |
| correctness / attempt-0003 | pass | CPU operator/protocol reference tests; actual-device qualification separately in prepare |
| correctness / attempt-0004 | pass | CPU operator/protocol reference tests; actual-device qualification separately in prepare |
| correctness / attempt-0005 | pass | CPU operator/protocol reference tests; actual-device qualification separately in prepare |
| correctness / attempt-0006 | pass | CPU operator/protocol reference tests; actual-device qualification separately in prepare |
| cost / attempt-0001 | pass | upper95% paired ratio must be below 1.0 in both sessions at a planned-model middle shape |
| pilot / attempt-0001 | INCONCLUSIVE_MEASUREMENT | {"status": "INCONCLUSIVE_MEASUREMENT", "reason": "full-model worker did not complete; raw prefix and command failure retained", "request_path": "/Users/artemlegotin/Documents/git_alter/optimizer-aware-residual-auditing/analysis/dw_sampling/20261031T120154Z-reopened/pilot/attempt-0001/model-block8-variants-0-request.json", "exit_code": 1} |
| pilot / attempt-0002 | not_run | full-model qualification passed; Stage E tuning/selection awaits full learning dispatcher and future freeze |
| pilot / attempt-0003 | STOP_QUALITY | no complete finite native grid arm |
| pilot / attempt-0004 | STOP_QUALITY | no sampler and native pair meets target plus endpoint guard |

Raw command exits/stdout/stderr, resource forecasts, source identities, stage records and their hashes accompany this report. Timing block intervals describe blocks, never independent training seeds.

Retain the recorded boundary and evidence; any repair must freeze its changed sources before repeating affected work.
