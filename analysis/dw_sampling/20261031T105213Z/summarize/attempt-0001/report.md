<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Bounded dW sampling report

Terminal state: **INCONCLUSIVE_MEASUREMENT**.

Fixed-budget norm-importance dW sampling and the direct-randint uniform control retain native forward, exact input/bias adjoints and native AdamW. Historical OAR results are unchanged.

No time-to-held-out-target or endpoint language quality result exists. No corpus bodies were ingested; no trainer was added before its prerequisites.

| Stage / attempt | State | Reason |
| --- | --- | --- |
| prepare / attempt-0001 | not_run | prepare-only dry run: validated immutable config/source/history; no device work/tests/training/downloads |
| prepare / attempt-0002 | BLOCKED_HARDWARE | essential actual-shape primitive failed; retained worker failure, no cost qualification |
| prepare / attempt-0003 | pass | existing suite/report, pinned dependencies and actual-shape MPS sampler operations qualified |
| feasibility / attempt-0001 | INCONCLUSIVE_MEASUREMENT | native variant selection warmup unstable |
| correctness / attempt-0001 | pass | CPU operator/protocol reference tests; actual-device qualification separately in prepare |

Raw command exits/stdout/stderr, resource forecasts, source identities, stage records and their hashes accompany this report. Timing block intervals describe blocks, never independent training seeds.

Stop at the recorded boundary; reopening requires the unmet prerequisite and a new hash-bound prospective version.
