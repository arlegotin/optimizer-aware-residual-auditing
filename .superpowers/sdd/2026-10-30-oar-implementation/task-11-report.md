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

## Executed bounded query study

Committed correction/scope **2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07** before
query outcomes. Ran the same public command with committed stage-study config
SHA256 `a387445beda9eeb81f4d2107cfa3c457fb89b3399c5684035d6c5fb5990c169b`.
Fresh actual MPS process with fallback0, approved escalation, exclusive GPU;
no tests/collectors/profilers overlapped. No external data or remote services.

All six declared condition/seed histories completed, all passed learnability;
no real attempt failure, unlearnable case, omitted condition or wider shape.
792 layer rows =396 simultaneous proposals (288 past +108 current-SVD), with
198 evaluation proposals (144 past +54 current).18 committed model/moment
checkpoints,30 development curve observations. Past sources exactly q-age;
current oracle labeled source=q/age0. At q128, source112 is pre-shift and
124/127 post-shift. No current query updates return to the native trajectory.

The measured first history6.0968305s and first-checkpoint audit forecast total
95.300s before continuing. Observed worker clock **60.9591427s**; ceiling600s
not reached. Forecast intentionally scales shared query overhead per row and
is not a confidence bound. Native history/basis/checkpoint blocks cost
6.0968/5.1780/4.9713/5.5800/5.1307/5.3173s in declared condition/seed order.
This is synthetic scaffolding cost, not isolated benchmark step throughput.
Endpoint RSS597,868,544 and Metal driver1,126,924,288 bytes overlap and are not
peaks. Ignored tensor checkpoint bytes total485,101,734 (~485MB); all18 hashes
validated. Construction/SVD and diagnostic replay/preview/scoring are charged
paid research, never counted as deployable savings. Individual proposal clocks
span.03230–1.09205s; the slow initial case is retained, no warmup exclusion.

Clock qualification: worker60.959s includes native initialization/histories,
old bases, CPU FP64 current SVD/transfers, candidate/dense previews, scoring,
raw-row/schema serialization and policy summaries up to final small summary
write. It excludes fresh-process import/startup and final summary serialization
tail; therefore this is not a fully inclusive invocation benchmark. Postprocess
work is separate. Repeated layer/shared/SVD row costs must not be summed as
independent times. No speed or throughput claim uses these clocks. This explicit
bookkeeping limitation is retained rather than fabricating zero overhead.

## Dense learnability (MSE per output element)

| Condition |Seed |MSE initial ->32 ->128 |post-shift120 ->128 |
|---|---:|---|---|
|low_rank |1101 |.846770 ->.122533 ->.101799 |n/a |
|low_rank |2202 |.864220 ->.121965 ->.101958 |n/a |
|full_rank |1101 |.870242 ->.137931 ->.108646 |n/a |
|full_rank |2202 |.849622 ->.135824 ->.107520 |n/a |
|direction_shift |1101 |.870242 ->.137931 ->.286898 |.732676 ->.286898 |
|direction_shift |2202 |.849622 ->.135824 ->.283887 |.729109 ->.283887 |

Fixed independent Gaussian development inputs; changed teacher target at120.
Both shift phases learn, but eight post-shift updates are not a convergence or
complete recovery claim. Early full-rank/shift trajectories intentionally
share weights, inputs and targets before120. Their repeated stages do not add
independent information. LR.003 remains an untuned upper-bracket winner.

## Descriptive structural results and separate labels

Evaluation counts below collapse paired layer rows; proposals sharing a query
and checkpoint are correlated, not independent runs or empirical population
prevalence estimates.Counts are synthetic proposal/update metadata, never language tokens.

| Condition/stage |Past proposals |negative true alignment |same-batch harm |development harm |
|---|---:|---:|---:|---:|
|low_rank8 |12 |12 |12 |12 |
|low_rank32 |18 |0 |0 |0 |
|low_rank128 |18 |0 |0 |18 |
|full_rank8 |12 |12 |12 |12 |
|full_rank32 |18 |0 |0 |0 |
|full_rank128 |18 |0 |7 |18 |
|direction_shift8 |12 |12 |12 |12 |
|direction_shift32 |18 |0 |0 |0 |
|direction_shift128 |18 |0 |0 |0 |

Totals36/144 first-order harm,43/144 same-batch harm and72/144 independent
synthetic development harm. All144 past and54 current-SVD evaluation proposals
lose development progress relative to dense (positive `development_delta -
dense_development_delta`). Lost-progress AUC is correctly null/one-label-class;
it is not perfect discrimination. Raw finite MSE signs, especially tiny late
changes, are not certified by a resolved production NumericalPolicy.

Dense itself harms the six stage8 evaluation queries: mean development deltas
+.047581 low_rank and+.049470 full_rank/shift. Approximate stage8 deltas are
+.079240/+.082736. Thus those harmful labels are not all failures caused solely
by projection. Stage32 proposals improve development but lag dense; shifted128
proposals also improve development but lag dense. This temporal structure and
early duplicated stages can explain score separability; no causal/population
claim follows from pooled AUCs.

| Rank |Past per-weight mean energy |Current-SVD mean energy |past dev harm/proposals |current dev harm/proposals |
|---|---:|---:|---:|---:|
|4 |.202758 |.427559 |24/48 |10/18 |
|16 |.327301 |.568534 |24/48 |9/18 |
|64 |.399819 |.725554 |24/48 |6/18 |

Energy means are unweighted means over both weight-layer rows, not a pooled
full-model energy. Exact current SVD improves reconstruction ceilings, but
never beats dense development progress in this one-step study. Past bases
preserve some genuine useful update information; no blanket failure of learning
from past subspaces follows. Constructed flat/rotating/rare-coordinate SGD and
high-cosine harmful-Adam controls remain mathematical controls only, never
frequency estimates from these histories.

## Scores, acceptance envelope and controls

Unfitted scores remain archived. Descriptive past-evaluation AUCs:

| Label |relative |ordinary |complement |
|---|---:|---:|---:|
|negative true alignment |.158179 |.828704 |.834619 |
|same-batch harm |.303477 |.874971 |.880497 |
|development harm |.352431 |.729745 |.735532 |
|lost progress |unresolved |unresolved |unresolved |

These demonstrate nontrivial geometric information in this narrow correlated
sample; the small complement-vs-ordinary differences are not confirmed added
value. No posthoc score inversion, new threshold or favorable rank selected.

Distinguish fitting from the bounds themselves. Across *every* allowed tau
.25/.5/.75, past ordinary, exact complement and q8 probe-complement criteria
allow **zero/144 evaluation proposals**. This is true before threshold fitting.
Past relative thresholds.1/.25/.5/1/2 allow0/0/8/56/104 proposals, including
0/0/8/37/54 development-harm proposals respectively. Thus fitted rejection is
not alone evidence about margin conservatism; the exact margin envelope itself
has no accepting past region on this grid.

Paid current-SVD oracle: ordinary allows zero/54 at all tau; exact complement
allows0/0/**2** at tau.25/.5/.75. Both are low_rank, stage128, rank64 (one per
seed), harmless same/development finite steps but still less progress than dense.
They require exact current gradient/SVD and have no priced deployable pipeline.
Probe complement allows zero at all tau. Current relative grid allows
0/18/28/42/50, with0/18/18/19/21 development-harm proposals. Full unfitted
accepted IDs/counts and per-seed/stage/rank/age margin ranges are durable in
`analysis/structural_oracle_aggregate.json`. No larger grid or experiment added.

Fitting minimizes tuning-only mean development delta with exact fallback over
shared proposal cases. It naturally favors dense when all projections lose
progress. The held-out comparison, therefore, is descriptive selection behavior
rather than evidence that geometric information is absent:

| Policy |tuning choice |accepted/144 |mean evaluation development delta with exact fallback |
|---|---|---:|---:|
|constant/no information |always exact |0 |+.00773760 |
|unconditional projection |always project |144 |+.01654391 |
|fixed schedule |rank64/age16 |12 |+.00780859 |
|relative |.1 |0 |+.00773760 |
|ordinary |tau.25 |0 |+.00773760 |
|complement |tau.25 |0 |+.00773760 |

These averages weight proposal cases, including repeated dense outcomes across
rank/age alternatives. A fixed schedule here gates cases matching its rank/age
and uses exact otherwise; it is not a fully trained fixed-rank baseline. Scores
share the same proposer/state and full offline diagnostics; no control is
charged an invented online implementation time. Per-condition/seed/stage fitted
results and IDs are in the original summary; only tuning labels fit policies.

## Source-data provenance, postprocessing and independent validation

Original summary records measured code revision2e89e74, exact code/config/scope/
cost artifact SHA256 values, seed-domain anchors and every curve/attempt. Actual
per-step/index/name generator seeds derive from SHA256(seed,domain,*ids), whose
durable function and batch descriptors specify exact reconstruction. No learner
model/data RNG consumed by diagnostics; probes and construction are separate.
All source tensors were originally pre-update/pre-clipping; only selected bases
and native states retained. No blanket dense matrix history or data download.

Original artifacts: `analysis/structural_oracle_pilot.json`,
`structural_oracle_audit.jsonl`, `structural_oracle_schema.json` and
`structural_oracle_summary.json`. Durable stdlib-only postprocessor command:
`.venv/bin/python analysis/summarize_structural_oracle.py`, producing aggregate
JSON with input hashes, native-learning CSV and small raw-checkpoint index.
No postprocessor changes training, fitting, thresholds or experimental coverage.

Postprocessor test initially failed on missing script, then checked layer-row
deduplication, retained dense harm and independently hand-derived margins.
Adding the unfitted envelope assertion failed on its missing field; implemented
only the predeclared-grid descriptive envelope. Additional config-syntax test
failed on unnormalized PyYAML ParserError; loader now returns CLI invalid-config
code2. This post-run parser correction changes no valid-config timed operation;
measured source remains the committed2e89e74, not an invented later revision.
19 focused tests now pass; full final suite follows below.

Independent retained-artifact checks validated all792 scalar rows finite and
unit labeled,18 checkpoint byte hashes,30 curves, disjoint18 tuning vs18
evaluation query IDs and development IDs, actual past-vs-current source steps,
rank16-only measured runtime references and all observed real-arithmetic
lower-bound inequalities within the declared diagnostic tolerance. Max recorded
projection defect1.89512e-7; max orthogonality defect1.08770e-7. These observed
values do not resolve a general numerical allowance.

## Self-review, limitations and exit

No WP03/full harness/baseline task completed; no corpus, tokenizer, language
model, protected evaluation, controller, online acceptance, numerical-policy
bypass, new backend, performance optimization or competitive training run.
Only six synthetic native histories/two seeds; one tuning/evaluation query per
checkpoint, duplicated early full/shift stages and many correlated projections.
FP32 finite changes and raw label signs are exploratory, especially near zero.
The distinct same-batch/development/dense-relative labels remain separate;
absent/rare harm and constant labels produce explicit unresolved outputs.

Large actual checkpoints are local ignored regeneration artifacts (~485MB),
not a general checkpoint lifecycle/resume implementation. Trusted local payloads
are explicitly encoded dictionaries of tensors/metadata; no live pending-update
authority is transported. No full-model resume/replay robustness claim.
Resource snapshots miss peaks; start/serialization clock tails are unmeasured.
Rank4/64 costs and current-SVD construction have no deployable headroom evidence.
Native LR.003 is selected for learnability, not optimized baseline fairness.

Bounded outcome: past subspaces retain some update signal and unfitted geometry
scores carry information, but this grid has no accepting exact past-margin
region, only two paid/current-r64 oracle accepts, no projected development
progress advantage, and no fitted complement-policy advantage over ordinary or
constant fallback. This does not establish a universal absence of information
or language-task failure. It does not justify an online controller or a cost
remedy by itself. Current full measured online costs remain negative and
NumericalPolicy unresolved. Record ordinary WP04 **inconclusive**, never passed;
WP03 stays unfulfilled and WP02 stop unchanged. After Task11 review, the exact
next permitted action is **WP09 synthesis**, retaining this narrow negative/
inconclusive structural result. No new grid/optimization/experiment launched.

Additional retained-row geometry distribution, positive-a evaluation proposals
only (global simultaneous-proposal bounds, layer rows deduplicated):

| Basis |n |Bcomp/Bordinary min /Q25 /median /Q75 /max |Bcomp/a min /Q25 /median /Q75 /max |
|---|---:|---|---|
|past |126 |.5696 /.7881 /.8543 /.9173 /.9931 |.9907 /2.5676 /6.1414 /10.9428 /128.5988 |
|current SVD |36 |.5285 /.7957 /.8832 /.9273 /.9874 |.5783 /1.4634 /2.7288 /4.8137 /38.6822 |

Complement geometry often reduces the bound (past median reduction14.6%),
but the smallest positive-a past Bcomp/a remains.9907, above every allowed tau.
The paid current subspace improves that ceiling sufficiently for two.75-grid
accepts. This is a measured structural/information gap for this proposer and
grid, not absence of geometric information. Quantiles use linear interpolation
at `(n-1)*p`; durable aggregate contains exact values. Added test assertion first
failed on missing distribution field, then passed with this derived-only code.
No new collection, fitting, optimization or scope expansion.

## Final verification and durable exit evidence

Final fresh actual-device command:
`PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q`:
**305 passed in6.71s**, zero failures/skips. This includes19 oracle/derived
postprocessing tests plus the286 existing real CPU/MPS tests. No GPU experiment
was running. Focused19 passed in1.46s after final derived distribution changes.
Independent rank-sum AUC calculation from raw rows reproduced ordinary.874971218,
complement.880497352 and relative.303476859 same-batch AUCs, including tied scores,
without using the production pairwise AUC helper. Final documented postprocessor
invocation took **.0584246s** inclusive of its process startup/output writes.
All scalar/temporal/unit/source/hash validations described above passed.

Append WP04 inconclusive evidence without altering WP02 stop or creating a
WP03 pass. The numerical policy stays unresolved. The normal WP04/WP05 gate
checks must remain blocked, while the narrow already-amended synthetic command
continues to validate its unchanged WP02/entry evidence for reproduction.
Results commit is separate from pre-experiment implementation/scope commits;
no push, dataset body, remote service or general trainer was used.

### Final constructed-control refinement

Results committed separately as **dc56170** after implementation a100d5c and
pre-query lock2e89e74. A final test self-review found the initial three-case
SGD parametrization's 'flat' spectrum included a tiny coordinate, and its
'rotation' was within a repeated singular-value space. Tightened controls to
identity spectrum (retained energy1/3, residual sqrt2) and a genuinely60-degree
rotated rank-one query relative to fixed past e1 (energy1/4, residual sqrt3/2),
keeping the distinct rare-coordinate control. New independent assertions fail
on old fixtures:2 failed/1 passed, as expected. Restored corrected fixtures;
focused19 passed in2.15s. Final actual fallback0 CPU/MPS suite:
**305 passed in7.24s**, no failures/skips. This supersedes the6.71s verification
clock above, with unchanged test count and unchanged study source/results.
No research experiment, fitting, threshold or coverage changed. Refreshed
WP04 inconclusive evidence hashes append without rewriting previous records.
