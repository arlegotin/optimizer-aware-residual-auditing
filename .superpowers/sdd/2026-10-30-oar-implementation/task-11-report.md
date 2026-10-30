# Task 11 — bounded synthetic structural oracle

Status: implementation and exploratory pilot preparation, no real study yet.
Authority: task-11-brief.md narrowed by structural-oracle-amendment-brief.md;
reviewed base6f04c6eea4d4aab0f1df31dfe6e1361275c91666. Read global constraints,
common interfaces, spec§8 and blueprint§§1/2,7.3,11. No whole-plan execution.
No subagents/reviewers, no corpus ingestion or general trainer/controller.

## Scope and protocol rulings

`decisions/structural_oracle_scope.md` and strict
`configs/structural_oracle.yaml` are the authoritative bounded exploratory
scope. `decisions/amendments.jsonl` binds their byte hashes, canonical latest
WP02 stop and actual numerical/cost artifacts. `require_structural_oracle`
validates typed gate records, same config object/file, latest WP00/WP01 pass,
latest WP02 stop, all artifact hashes and actual qualified starting-shape
candidate headroom/audit-cost failure. Only `audit-checkpoints` consumes this
exception; normal `require_gate` and WP03/WP04 prerequisites stay blocked.

Initial stage is development-only native128-update pilot at the four blueprint
starting learning rates. Selection minimizes final fixed independent-development
MSE before any query outcomes. The bracket is a learnability pilot, not a claim
of an optimized strong baseline. A boundary winner is reported without an
unbounded search. No wider shape authorized. Total600s cap, no large grid.

Same cost-fixture residual network `x + W2 GELU(W1 x+b1)+b2`, n2048/di512/do2048.
Student and teacher independently initialized. Low-rank W2 teacher, full-rank
teacher, declared output-row roll at update64 are distinct conditions. Input
Gaussian observations. All losses/deltas: MSE per output element; no NLL/token
or real-task quality claims. Low-rank teacher ranks and seed domains persisted.

Model initialization is stateless per domain and batch RNG derives from
SHA256(seed,domain,step,index), with student/data/teacher/construction/tuning/
evaluation/development/probe separated. No learner stream consumed by queries.
Source gradients captured before clipping at their actual native pre-update
weights. `build_basis` supplies past modes; selected bases plus committed native
parameters/moments are retained, not blanket dense gradient history. Query
checkpoints follow8/32/128 committed updates; ages1/4/16 where available, ranks
4/16/64. `BasisState.for_step` enforces strict past-only use. Current exact SVD
is separate paid oracle with explicit label, source step equal query step.

`audit_checkpoint` retains required research interface. It reads checkpoints
into private model/moment storage, obtains exact current gradients, uses existing
`projected_linear`/`LinearCapture` for exact-input-adjoint projected proposals,
previews full qualified AdamW with clipping and decay, and compares copied
candidate/dense models on same and independently replayed development batches.
No diagnostic model feeds back to native trajectory. Actual displacement norms,
alignments, complement/probe bounds, defects, sources, hashes and charged times
are archived. Probe residual score uses fixed q8/delta.01 only as an offline
score, not a run-wide probability guarantee. NumericalPolicy unresolved.

One tuning and one evaluation query per checkpoint; evaluation IDs and their
development IDs differ. Tuning-only objectives fit relative/ordinary/complement
thresholds and fixed rank/age schedules. Tau grid.25/.5/.75; relative thresholds
.1/.25/.5/1/2. AUC and harm sensitivity unresolved with fewer than five examples
in requisite classes, never fabricated perfect values. First-order alignment,
same-batch finite harm, development finite harm and lost progress are separate.

JSONL plus schema/summary explicitly replaces Parquet for bounded synthetic
branch, avoiding a data dependency. Large tensor checkpoints ignored, regenerated
via durable CLI. Existing `CostTable` supplies only matching rank16 runtime;
ranks4/64 remain structural diagnostics. No real exact-oracle work counts as
online savings. Shared query/proposal costs repeated in row metadata; total wall
is authoritative; never sum duplicated layer rows. Endpoint RSS/Metal readings
are overlapping resource snapshots, not measured peaks.

## TDD and validation before real runs

- Initial `.venv/bin/python -m pytest tests/test_oracle.py -q`:10 failed, expected
  missing oracle module. Then8 passed/2 failed on AdamWGroup init-only schema
  reconstruction; explicit checkpoint encoder fixed schema handling and encoded
  BasisState tensors/metadata rather than serializing mutation-version authority.
- Focused10 passed. Added independent original-gradient/native-history reference,
  positive/tampered permission bindings and CLI ownership;12 passed/1 failed
  on missing CLI command. Implemented durable public command;13 passed.
- Added rare-event test:13 passed/1 failed because AUC1.0 from one harmful event.
  Implemented minimal support rule;14 passed in1.23s.
- Independent NumPy references check residual/complement/true alignment, actual
  orthogonal projection high-cosine harmful Adam state and projected SGD descent.
- Real native4-step CPU reference independently reconstructs old gradient SVD
  projectors, saved parameters/moments and exact finite-query displacement loss.
- Descriptor replay/global RNG equality, temporal descriptor rejection,
  source/query provenance, checkpoint-byte isolation, strict config failures,
  changed evaluation labels leaving fitted tuning policy unchanged, absent and
  rare harm, tiny CPU full worker pipeline, guard hash tampering all exercised.

Full actual-device suite and commit/run evidence follows below.

Initial full fresh actual-device verification:
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q`:
**300 passed in6.93s**, no failures/skips. Approval escalation used for real MPS;
no experiment/timing overlapped this suite. Scope/config and tested code commit
precedes all real synthetic pilot/query experiments.

## Initial committed MPS pilot and pre-query correction

Initial implementation/scope/config commit **a100d5c**. Command:
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m oar.cli audit-checkpoints --config configs/structural_oracle.yaml`
with config stage pilot in that commit. Fresh real MPS fallback0 process,
no overlapping GPU work. Durable artifact `analysis/structural_oracle_pilot.json`
contains code revision, resolved config/domains, all curves and wall/resource
snapshots. Total26.0455868s; individual rates:

| LR |128-update wall s |initial/final fixed-development MSE |
|---|---:|---|
|.0001 |7.313630 |.870242 -> .309506 |
|.0003 |6.278665 |.870242 -> .166266 |
|.001 |6.242725 |.870242 -> .121952 |
|.003 |6.197695 |.870242 -> .108646 |

Declared rule chooses.003. Boundary winner retained, no unbounded expansion,
no claim of tuned competitive baseline. Endpoint RSS329,154,560 and Metal
1,118,535,680 bytes overlap and are not peaks. Six native histories forecast
~37–45s before paid decompositions/diagnostics, total engineering ceiling600s.
Early learning/cost milestone sent to controller before query execution.

Self-review found replay inputs did not require root input gradients, unlike
the measured fixture. Pilot learning gradients/values remain valid; its wall
is not a timing-equivalent headline. Added root-input-gradient test:14 passed/
1 failed as expected. Enabled root dX for native/projected study boundaries.
Added measured first-query resource forecast assertion (expected missing key
failure), implemented conservative forecast and resource ceiling. A tiny-ceiling
test initially failed on operational RuntimeError; changed bounded stops to
preserved `incomplete_inconclusive` results with incomplete coverage, never a
favorable prefix. Added explicit constant/unconditional projection control test
(expected missing-control failure), then implementation.17 focused tests pass.

Pre-query shift coverage ruling, disclosed to controller:64 ->120 so query128
age16 source112 is before shift and ages4/1 source124/127 after it. Original64
proposal had no basis straddling shift at requested checkpoints. No extra cases,
no query outcomes, full-rank-only pilot unaffected. Both initial and120→128
post-shift development learning must decrease before that condition's inference.
Early full-rank/shift stages share exact input/init history and are not extra
independent seeds. New stage-study config locks LR.003, shift120 and hashes.
Second amendment appends, preserving initial exploratory authorization history.

Pre-query full actual-device verification after correction:
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q`:
**303 passed in6.85s**, no failures/skips. Pilot artifact SHA256:
`6e727d64b54a52f560936a9cc1b64f704ff94669d8889cbe5b748734386efc18`.
Native moments remain continuous across teacher shift; no state restart.
