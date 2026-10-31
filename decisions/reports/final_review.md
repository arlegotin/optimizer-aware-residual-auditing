# Final review and controller ruling addendum

Current status: **Accepted after scoped independent re-review** on 2026-10-31. All four findings are addressed, with no new breakage. The exact reviewed revision and final verification appear below.

Publication status at10069d2 was scoped re-review pending; the earlier review sections preserve that context.

Original review BASE: `af715ac2bf6e6c490ac5178fe18725414bfcbe34`.
Original reviewed HEAD / fix-wave BASE: `e388b43148899e242800d4df348876470565ead1`.
Corrected implementation HEAD: `7628be7792030e3cfbb17dba82d3517642c866ca`.
The publication commit containing this addendum follows that implementation commit on current `main`; its identity is discoverable with `git log -- decisions/reports/final_review.md`. Subsequent acceptance entries must name the exact re-reviewed HEAD.

This coordination addendum is linked from [README](../../README.md) and intentionally excluded from the scientific evidence index and WP09 artifact hashes. It may receive a final acceptance-status append without creating a hash cycle. [Review summary](../review_summary.md) and [outcome](../outcome.json) preserve their complete hash-bound pre-final-review bytes/status. The original20 [rulings](../rulings.md) are unchanged; all16 later controller Ruling lines are copied below, including D1–D14, M disposition and addendum policy. The scientific report explicitly distinguishes the older snapshots from this addendum.

## Findings and implemented disposition

| Finding | Disposition and coverage | Practical limitation / cost if wrong |
|---|---|---|
| I1 — overlapping destination partial commit | Fixed before pending publication: a sorted stride/span proof rejects unsupported internally overlapping writable parameters; all commit destinations are revalidated before the first copy. Expanded and overlapping `as_strided` second parameters reject without mutating values, gradients, moments or counters. A layout mutation after preview rejects without consuming the pending update. Existing transpose/native checks and a gapped-slice/native commit test pass. | The proof is deliberately sufficient rather than a complete overlap solver: exotic uncertain layouts may reject. A wrongly admitted layout would require corrected validation and transaction retesting; no partial rollback is attempted. Serialized callers and the existing raw-storage/concurrent-mutation exclusions still apply. |
| I2 — reset/missed study ceiling | Fixed with one absolute worker deadline passed through every native history; check at native update/checkpoint boundaries, after persisted audit rows, before successful summary completion, and after final synchronization. Selected completed checkpoints and actual progress remain available. No scientific config field is replaced with a remaining budget. Four injected-clock regressions cover a later history with5s remaining and final audit/summary/sync reaching the exact ceiling. | Kernel/checkpoint/audit operations are indivisible; time can cross inside one operation and is classified at its boundary. Final JSON serialization remains outside the reported worker clock, as before; historical timings are unchanged. A wrong boundary would require control-flow repair and truthful incomplete status, not evidence deletion or relabeling. |
| M1 — normal checkpoint identity gap | Fixed with shared normal/recovery identity, duplicate, cardinality and byte-hash validation. Fixture integration exercises both real helper paths with real files for duplicate valid entry, missing entry, corrupt hash and valid complete indexes. | Local trusted checkpoint bytes only; not secure ingestion. A wrong identity rule would require failed comparison retention and helper repair; published recovered evidence already had strong identity checks and is unchanged. |
| M2 — nonfinite derived metadata | Documented in the public `preview_adamw` API: finite endpoints/moments do not guarantee finite displacement/alignment, and consumers must fail closed (as `decide_margin` does). Existing numerical rejection remains intact. | No new timed arithmetic checks or universal finite-input closure claim. A consumer requiring finite derived metadata needs an explicit guard and fresh cost qualification. |

No experiment, benchmark, dataset, dependency, trainer/controller, criterion, shape/backend pivot or confirmation work was performed. Original scientific gates1–16, raw outputs/configs, measured study source `2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07` and timing source hashes remain unchanged. Current validation overhead has not been remeasured. Prior publication bytes resolve through Git; the current index/report and appended WP09 maintenance evidence bind the current supported source/README bytes.

## Actual verification evidence

These are implementer executions of the changed components, not new research or reviewer reruns. RED was observed before each corresponding code fix.

- I1 reproduction on original HEAD: normal first parameter changed from1 to `0.9000000357627869`; expanded second parameter stayed `[1.0,1.0]`; `copy_` raised unsupported overlapping-write RuntimeError, state remained `{}` and `pending.consumed` stayed false.
- I1 RED: `.venv/bin/python -m pytest -q tests/test_optimizer_preview.py -k 'internally_overlapping or changed_second or gapped_noncontiguous'` → `2 failed, 2 passed, 43 deselected in 1.56s`. Both overlap cases failed because preview did not raise.
- I1 GREEN: `.venv/bin/python -m pytest -q tests/test_optimizer_preview.py -m 'not mps'` → `44 passed, 3 deselected in 1.16s`.
- I2 RED: `.venv/bin/python -m pytest -q tests/test_oracle.py -k 'remaining_study_deadline or final_completion_deadline'` → `4 failed, 21 deselected in 2.25s`. Later history wrongly marked completed; final audit/summary/sync wrongly marked full coverage.
- I2 GREEN: `.venv/bin/python -m pytest -q tests/test_oracle.py` → `25 passed in 2.45s`.
- M1 RED after correcting a fixture-only missing subprocess.STDOUT: `.venv/bin/python -m pytest -q tests/test_reproduction.py -k checkpoint_identity` → `1 failed, 7 passed, 10 deselected in 0.83s`. Duplicate-valid normal path returned0/reproduced; recovery already rejected it.
- M1 GREEN: `.venv/bin/python -m pytest -q tests/test_reproduction.py` → `18 passed in 0.86s`.
- Focused integration: `.venv/bin/python -m pytest -q -m 'not mps' tests/test_optimizer_preview.py tests/test_oracle.py tests/test_reproduction.py tests/test_report.py` → `115 passed, 3 deselected in 2.03s`.
- **Single final actual-device suite** at corrected implementation HEAD: `PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q` → **`364 passed in 7.80s`**, no skips/warnings. Executed sequentially with actual MPS access; no research/benchmark/collector overlap. Hardware access used the authorized local escalation.

Exact named regressions:

- `test_internally_overlapping_second_parameter_rejected_without_mutation[expanded/overlapping_strides]`
- `test_commit_rejects_changed_second_parameter_layout_before_any_write`
- `test_gapped_noncontiguous_parameter_commit_matches_native`
- `test_later_native_history_uses_remaining_study_deadline`
- `test_final_completion_deadline_retains_completed_evidence[final_audit/final_summary/final_sync]`
- `test_reproduction_helper_validates_checkpoint_identity_count_and_bytes[duplicate/missing/hash/valid-normal/recovery]`

Publication-only checks and the superseding WP09 record follow the tested source commit. The final completion entry below records their exact results. Existing scientific uncertainty, same-seed-repeat limitation and original failed processing/recovery/CPU-overlap disclosures remain in force.

## Controller rulings since the immutable original20

The following are copied verbatim from the controller ledger; each includes its own practical rework cost if wrong. All are retained scope/disposition decisions. D1–D13 do not certify omitted behavior; D14 preserves the local-only integration boundary. The last two rulings govern this exact M disposition and publication graph.

### Ruling 21

> - Ruling: Final review D1: Keep general trainer replay/dropout/resume unreviewed and unimplemented at the WP02 stop — Task9/12 not reached — if wrong, reopen the trainer/state-machine gates before claiming that lifecycle.

### Ruling 22

> - Ruling: Final review D2: Keep corpus/protected-language evaluation outside acceptance — synthetic-only amendment — if wrong, reopen asset, tokenization and protected-evaluation gates before any language claim.

### Ruling 23

> - Ruling: Final review D3: Do not certify model-level eligibility integration — Task8 not reached — if wrong, implement tied/shared/attention eligibility checks before a general model integration.

### Ruling 24

> - Ruling: Final review D4: Do not call named baseline ports reproduced or defeated — archives establish provenance only and Task10 not reached — if wrong, complete faithful ports and matched tuning before a competitive claim.

### Ruling 25

> - Ruling: Final review D5: Do not accept an online controller or deployment utility claim — Tasks12/13 not reached — if wrong, implement/version/replay-test the online path under a newly passed gate.

### Ruling 26

> - Ruling: Final review D6: Do not accept confirmation or a fixed-time quality frontier — Tasks14–16 not reached — if wrong, freeze and execute an adequately powered protected protocol before such a claim.

### Ruling 27

> - Ruling: Final review D7: Keep the formal machine-arithmetic certificate unresolved and online policy closed — current NumericalPolicy is explicitly unresolved — if wrong, derive/validate a numerical allowance before a production audit guarantee.

### Ruling 28

> - Ruling: Final review D8: Keep unsupported optimizer variants outside the preview contract — only qualified scalar eager AdamW semantics were implemented — if wrong, add independent native-equivalence qualification for any new variant.

### Ruling 29

> - Ruling: Final review D9: Do not begin a performance redesign, new shapes or backend pivot — negative cost/oracle gate permits synthesis only — if wrong, a separate specified and measured opportunity is required before reopening research.

### Ruling 30

> - Ruling: Final review D10: Limit compatibility claims to the measured runtime and implemented CPU references — no unrelated hardware/future-wheel qualification — if wrong, repeat discovery and relevant numerical/backend checks for the new platform.

### Ruling 31

> - Ruling: Final review D11: Retain the documented cooperative local/serialized mutation contract — hostile rollback/raw-storage writes/concurrent mutation are excluded explicitly — if wrong, a stronger adversarial/durability design is needed if those become requirements.

### Ruling 32

> - Ruling: Final review D12: Treat checkpoints as locally generated regenerable research inputs only — untrusted checkpoint loading/general crash-resumable trainer was not implemented — if wrong, add safe ingestion and full trainer-checkpoint contracts before accepting external checkpoints.

### Ruling 33

> - Ruling: Final review D13: Issue no new legal or novelty conclusion — the conditional licensing ruling was unused and no corpus was ingested — if wrong, reopen primary-source asset/prior-art review before those claims.

### Ruling 34

> - Ruling: Final review D14: Keep all work on current main and perform no push/merge; final acceptance waits on scoped fixes and verification — user authorized current-branch commits and no external publication — if wrong, correct the local acceptance record if verification fails; external integration needs user authorization.

### Ruling 35

> - Ruling: address M1 by sharing checkpoint identity/hash validation; address M2 by documenting derived nonfinite metadata and existing fail-closed consumer, without new timed numerical checks — both fit the current component contract and preserve source-attributed measurements — if wrong, a caller requiring finite metadata must gain an explicit guard and fresh cost qualification.

### Ruling 36

> - Ruling: preserve hash-bound pre-final review snapshots and publish an appendable final coordination/ruling addendum outside the scientific evidence index, linked from README — avoids self-referential review-status hashing while keeping final decisions discoverable — if wrong, regenerate coordination links/status records without relabeling or rerunning scientific data.

## Complete original whole-branch review (verbatim)

The original reviewed revision, line references, recommendations and verdict below are historical review content. They are preserved exactly and are not a verdict on the corrected HEAD.

# Final whole-branch review

Reviewed range: `af715ac2bf6e6c490ac5178fe18725414bfcbe34` through `e388b43148899e242800d4df348876470565ead1`.

Requirements: `docs/superpowers/specs/2026-10-30-oar-design.md`, the gate-ordered implementation plan, and the pre-query `decisions/structural_oracle_scope.md` amendment. The reached branch is Tasks 1–6, bounded Task 11, and Task 17. Skipped dependent tasks are intentional scientific stops, not missing authorized features.

Method: reviewed all live `src/oar` modules, configs/dependency setup, collectors and postprocessors, the report/reproduction helpers, corresponding test contracts and selected test bodies, README/report/outcome/rulings, and source-attribution machinery in passes. The full artifact inventory and raw/historical evidence remained available; repetitive raw rows and upstream verbatim archives were not reread wholesale. Existing full-suite and actual-MPS evidence was used rather than rerunning it. Root additionally reported fresh 348-pass verification at this HEAD, deterministic report regeneration, all 21 latest WP09 bindings valid, and unchanged blueprint bytes. Those are attributed controller checks, not reviewer reruns.

Two focused CPU reproductions were run only after source inspection exposed unanswered edge cases. They used no MPS, no benchmark, no research rerun, no repository data changes, and temporary files outside the checkout. The only checkout write by this reviewer is this requested report. No subagents were used.

## Strengths

- The implementation has a coherent numerical boundary: native forward/exact incoming adjoints, streaming aggregate residuals, past-only bases, preserved raw gradients, and actual stored AdamW displacements. Tests use dense/native/NumPy references rather than merely repeating implementation formulas (`src/oar/linear.py:75`, `src/oar/basis.py:35`, `tests/test_exact_adjoints.py:38`, `tests/test_optimizer_preview.py:79`).
- Confidence accounting is unusually explicit. Reservations become visible only after fsync, corrupt/truncated journals fail closed, resumed counters never move backwards, current-batch cross-layer probe reuse is checked, and the documented missing-journal resume case cannot silently refund spending (`src/oar/audit_journal.py:71`, `src/oar/audit_journal.py:214`, `tests/test_audit_journal.py:233`). The unresolved `NumericalPolicy` correctly blocks online acceptance (`src/oar/audit.py:147`).
- The scientific stop is respected through integration. The bounded exception checks the genuine latest WP02 stop and hash-bound evidence without changing the ordinary gate graph (`src/oar/oracle.py:110`, `src/oar/gates.py:16`). The public command surface does not advertise a trainer or controller that was never implemented (`src/oar/cli.py:9`).
- The cost interpretation uses complete blocks and the scheduled-dense mixture. `C/(K-1)` and the K=1 boundary are correct, direct replay is measured separately, and profiling subtotals are not presented as whole-step time (`src/oar/bench.py:114`, `analysis/summarize_headroom.py:29`). Randomized shared rounds and explicit journal aging qualify the measured reference costs.
- The final scientific wording is appropriately narrow. It distinguishes 792 layer rows from 396 proposals and two initialization/data seeds; current SVD from past-only proposals; descriptive discrimination from actionable acceptance; and same-seed reproducibility from independent confirmation (`analysis/report.md:13`, `README.md:3`, `README.md:69`). The priced rank-16 no-positive-margin diagnostic does not erase the two unpriced rank-64 margins.
- The synthesis verifies current/historical bytes, protects input artifacts from output overwrite, binds an explicit gate prefix, and refuses to promote an authored confirmation assertion into confirmed learning (`src/oar/report.py:58`, `src/oar/report.py:140`). The failed initial comparison and unchanged-tolerance recovery remain visible.

## Issues

### Critical

None found.

### Important — should fix before acceptance

#### I1. An internally overlapping parameter can cause a partially applied commit

- **Locations:** `src/oar/optimizer_preview.py:51`, `src/oar/optimizer_preview.py:163`.
- `_validate` checks aliasing between parameter storages, but accepts a parameter whose own elements overlap. A zero-stride expanded tensor is finite, dense and `torch.strided`, so preview succeeds and clones it into a writable candidate. `commit_pending` then writes parameters sequentially. A later expanded destination fails `copy_` after earlier parameters have already changed.
- **Focused observed reproduction:** with `p = {'a': torch.tensor([1.]), 'b': torch.tensor([1.]).expand(2)}`, all-one gradients, empty state, and `AdamWGroup(tuple(p), .1, (.9, .999), 1e-8, 0.)`, preview succeeds. Commit raises `RuntimeError: more than one element ... refers to a single memory location`. Afterwards `a == [0.9000000357627869]`, `b == [1.0, 1.0]`, optimizer state is still `{}`, and `pending.consumed is False`.
- **Impact:** a routine unsupported tensor layout breaks the advertised transaction boundary and leaves model/state versions inconsistent. A reasonable caller of the eager FP32/FP64 tensor API expects unsupported input to be rejected before any live write. This is not the documented `.data`/concurrent-mutation escape hatch. It does not invalidate current experiments, which use ordinary parameter layouts.
- **Fix:** reject unsupported internally overlapping writable parameter layouts during validation, before returning a transaction or performing any commit write. Preserve supported transposes and ordinary noncontiguous layouts. Do not attempt a risky partial rollback after the first write.
- **Verification:** add a regression with a valid parameter preceding an expanded/overlapping parameter; assert rejection and unchanged parameters, moments, counters and consumed state. Retain the existing noncontiguous/native-equivalence checks.

#### I2. The study-wide time ceiling is not enforced at all checkpoint/completion boundaries

- **Locations:** `src/oar/oracle.py:227`, `src/oar/oracle.py:284`, `src/oar/oracle.py:492`, `src/oar/oracle.py:528`, `src/oar/oracle.py:561`; contract: `decisions/structural_oracle_scope.md:85`.
- Native histories start a fresh local timer and receive the original full `max_wall_s`, rather than the remaining study budget. A history begun near the global deadline can pass subsequent native checkpoint boundaries while its local timer remains below 600s. In addition, the worker checks the global deadline before auditing a checkpoint but not after the last audit; it can then set `coverage_complete=True` and a completed outcome even though the final checkpoint crossed the engineering ceiling.
- **Focused observed reproduction:** used the existing tiny CPU config with one full-rank seed and checkpoints 2/4. Wrapped `audit_checkpoint` to advance an injected monotonic clock by 1200s immediately after the second/final real tiny audit. The worker retained 48 rows and returned `coverage_complete=True`, `outcome='synthetic_only_online_still_blocked'`, `total_wall_s=1200.7901123339543`, with `max_wall_s=600`. No real delay was introduced. The first forecast was below the cap, so it did not mask this final-boundary defect.
- **Impact:** the public bounded reproduction can exceed its declared study budget and call that capped execution complete. The explicit amendment requires partial/inconclusive reporting when actual time reaches the ceiling at a checkpoint boundary. This matters even though the original 60.959s run and retained repeat are safely below the ceiling and their conclusions remain valid.
- **Fix:** carry one absolute study deadline (or an equivalent shared budget checker) through native history boundaries and audit boundaries. Check after a completed audit and before marking the study complete. Preserve completed rows, curves, checkpoint index and the last committed state as incomplete evidence; do not discard the overrun prefix or change scientific configuration stored in checkpoints. No need to interrupt an indivisible kernel mid-operation.
- **Verification:** use injected time for (a) a later native history that starts with little study budget left and (b) the final audit crossing the deadline. Both must stop at the next allowed boundary, retain real progress, and report `incomplete_inconclusive` with `coverage_complete=False`. Keep the below-budget normal path unchanged. A hardware/research rerun is unnecessary for this control-flow fix.

### Minor — optional hardening

#### M1. Normal reproduction validates checkpoint count and bytes but not checkpoint identity

- **Location:** `analysis/reproduce_structural_oracle.py:117`; stronger existing implementation: `analysis/reproduce_structural_oracle.py:172`.
- The normal `--run` path compares index lengths and each entry's byte hash. An index with a duplicate valid checkpoint replacing a different `(condition, seed, step)` can satisfy those checks. `--recover-comparison` already checks the identity set and duplicate cardinality.
- **Impact/calibration:** raw row/curve coverage is independently compared, so this does not let a changed scientific result pass unnoticed. The published recovered repeat uses the stronger identity checks. It is a small consistency gap in what future normal runs call complete checkpoint reproduction.
- **Recommendation:** share checkpoint identity/count/hash validation between normal and recovery paths; test a repeated valid entry replacing one expected checkpoint. No experiment rerun is needed.

#### M2. Extreme finite endpoints can still produce nonfinite derived preview metadata

- **Location:** `src/oar/optimizer_preview.py:132`; fail-closed consumer: `src/oar/audit.py:94`; existing disclosure: `decisions/review_summary.md:11`.
- Candidate parameters and moments are checked for finiteness, but the subtraction forming FP32 displacement and the final alignment are not. Opposite-sign finite endpoints near FP32 maximum can overflow the displacement, producing nonfinite `raw_alignment` despite finite inputs/candidates. For example, a sufficiently large finite decoupled-decay factor can flip a parameter near `3e38` to a finite negative value while its difference overflows.
- **Impact/calibration:** `decide_margin` rejects nonfinite alignment and serializes safe null metadata, so this is not an accepting audit or an observed scientific defect. Native candidate/state values can remain valid. The retained component limitation is appropriately disclosed; arbitrary finite-input arithmetic closure is not established.
- **Recommendation:** either explicitly document the derived-value limitation in the preview API or reject unrepresentable displacement/alignment before returning a pending update, with a focused boundary test. If adding numerical checks changes timed implementation work, preserve the existing measured-source attribution; do not relabel old timing as a measurement of changed code. This minor alone does not block the stopped-study deliverable.

## Integration and evidence assessment

No additional finding was identified in the reached operator algebra, confidence allocation, exact-versus-zero-gradient semantics, cost equation, gate exception, descriptive oracle comparisons or published stopped-study conclusion. This is a review conclusion within the inspected contracts, not a proof of all numerical states.

The source truthfully records paid current SVD and CPU decompositions, unpriced ranks, unresolved guardband, repeated shared query costs, finite-step harm versus alignment, and the lack of language or fixed-time training evidence. The scoped result is useful negative/inconclusive evidence even though online OAR was never built. The two Important findings concern component transactional behavior and bounded-run control flow; neither supplies a reason to rerun the existing benchmark or research study.

The review-status statements at `decisions/review_summary.md:3`, `decisions/review_summary.md:14` and `decisions/outcome.json:32` are accurate as pre-final-review snapshot text at this HEAD. After disposition, preserve that hash-bound snapshot and add a durable, discoverable final review/ruling addendum naming the reviewed and corrected revisions. A README link can distinguish the final disposition from the older snapshot without silently rewriting historical evidence. This is a completion recommendation, not an already-existing false-status finding.

## Declined to judge

Each line below is a behavior considered and set aside as outside the reached scope. The controller must explicitly rule on each; none is silently treated as implemented or passed.

- **D1 — General trainer replay, scheduler/data-cursor once-only updates, dropout and checkpoint-recomputation lifecycle:** `docs/superpowers/plans/2026-10-30-oar-implementation.md:313`; Task 9/12 work is gated off by WP02. Component transaction behavior remains in scope and produced I1.
- **D2 — Corpus acquisition, tokenization, article masks/protected splits and language-model NLL:** `decisions/structural_oracle_scope.md:7`; the amendment explicitly permits synthetic MSE scaffolding only. No dataset or language-quality claim is being accepted.
- **D3 — Full model eligibility integration for tied/shared weights, attention projections and model architecture:** `docs/superpowers/plans/2026-10-30-oar-implementation.md:284`; Task 8 was not reached. The existing tensor API's canonical-storage validation was reviewed, but there is no model integrator to certify.
- **D4 — Faithful INSTANT/VCAS/DropBP/Muon ports, competitive baseline tuning and nearest-art superiority:** `decisions/review_summary.md:16`; archives establish attribution, not executed ports, and Task 10 remains gated. The report explicitly does not call them defeated.
- **D5 — Online controller/rank adaptation, matched replaying audit ablations and deployment utility:** `docs/superpowers/plans/2026-10-30-oar-implementation.md:393`; Tasks 12/13 were not authorized by the stopped graph. Offline proposal comparisons do not imply an online implementation.
- **D6 — Confirmatory sample size, Holm tests, protected endpoints, noninferiority, independent robustness and a fixed-time quality frontier:** `docs/superpowers/plans/2026-10-30-oar-implementation.md:444`; Tasks 14–16 were not reached. The two-seed same-seed-repeat evidence is expressly exploratory/reproducibility evidence.
- **D7 — A formal FP32 or run-wide machine-arithmetic certificate and a production NumericalPolicy:** `docs/superpowers/specs/2026-10-30-oar-design.md:100`; empirical calibration remains unresolved and fails closed. The real-arithmetic Gaussian model and its implemented spending mechanism were reviewed; absence of a stronger certificate is not an implementation pass claim.
- **D8 — AMSGrad, differentiable/capturable optimizers, mixed precision and unqualified optimizer variants:** `docs/superpowers/specs/2026-10-30-oar-design.md:106`; the preview intentionally qualifies scalar eager non-AMSGrad/non-fused/non-foreach semantics. Benchmark native variants have separate bounded equivalence checks.
- **D9 — New backends/shapes, broader rank-price frontiers, faster journal/preview designs and long-run throughput:** `decisions/structural_oracle_scope.md:23`, `src/oar/bench.py:504`; no optimization pivot is authorized. The observed journal aging and reference costs are disclosed; they are not claimed irreducible lower bounds.
- **D10 — Portability to unrelated hardware, future library versions or a different wheel platform:** `README.md:9`; the lock and claims identify the measured native arm64/macOS runtime. Existing CPU references do not promise full cross-platform execution.
- **D11 — Hostile journal rollback, cross-host/network-filesystem durability and malicious `.data`/concurrent mutation:** `src/oar/audit_journal.py:3`, `src/oar/optimizer_preview.py:148`; these are explicitly outside the local cooperative POSIX/serialized-caller contract. Ordinary malformed evidence and supported-API mutations remain in scope and were reviewed.
- **D12 — Secure deserialization of untrusted externally supplied checkpoints or a general crash-resumable research trainer:** `src/oar/oracle.py:332`, `src/oar/oracle.py:316`; bounded oracle checkpoints are locally generated, ignored, regenerable research inputs. Full atomic trainer resume belongs to unreached Task 9; the audit journal's separate durable-spending contract was reviewed.
- **D13 — A current legal conclusion on corpus licensing or fresh upstream novelty claims:** `decisions/rulings.md:105`, `decisions/review_summary.md:16`; the conditional asset ruling was unused and no corpus was ingested. This review assesses recorded provenance and stopped-study truthfulness, not a new external legal/novelty investigation.
- **D14 — Final integration approval, push/merge or resolution of this review before controller action:** `decisions/review_summary.md:3`; this review provides findings for one consolidated fix wave and scoped re-review. It does not predeclare its own findings resolved or silently supersede the immutable publication snapshot.

## Recommendations and assessment

Address I1 and I2 in the single consolidated fix wave with focused regression checks. M1 is a small useful consistency fix; M2 can remain a documented fail-closed limitation if explicitly ruled on. Preserve all measured source identities and the current raw scientific results. Record each declined item and the final disposition in durable review records before disposable workflow cleanup.

**Ready to merge? With fixes.** The reached scientific scope and published conclusion are soundly bounded, but unsupported overlapping storage can partially commit an update and the declared global time cap can be bypassed at completion. Correct those two control boundaries and verify them without reopening gated scientific work.

## Publication maintenance verification

- `.venv/bin/python -m pytest -q tests/test_report.py tests/test_synthesis_collection.py tests/test_reproduction.py` → `49 passed in 1.60s` after publication edits.
- `.venv/bin/python analysis/collect_synthesis.py` → `164 artifact/hash bindings; 12 evidence annotations; 16 scientific gate lines`.
- `.venv/bin/python -m oar.cli report --gates decisions/gates.jsonl --run-index analysis/evidence_index.json --out analysis/report.md` → `report: analysis/report.md`. Unchanged-input regeneration is byte-identical, including after the appended WP09 record.
- Index SHA256 `3d1014fa516685f1cdc79b6601faed32a6ffff47129f7660fe19fb41469ec4fd`; report SHA256 `c931afc51793dcf9e2585f60f80284309ddc7c4cc831b8acc6f39c6977fc4716`.
- `.venv/bin/python -m oar.cli --help` lists only `doctor,bench,audit-checkpoints,report`.
- Direct checks verify202 current publication local links; complete verbatim original review; all16 subsequent rulings; unchanged original20 rulings, pre-final review/outcome, blueprint, scientific first16 lines and all raw research/reproduction/config bytes; distinct current/historical source bindings; and exclusion of this addendum from scientific/WP09 hashes.
- `git diff --check` passes. A new WP09 maintenance record appends current tested implementation/report bindings and explicitly leaves independent scoped re-review pending. Prior gate lines are untouched. No additional archive copies or measured-source relabeling were introduced.

## Scoped re-review and final controller disposition

Accepted reached-stage implementation/publication HEAD: `10069d22780e8cc67493aca5da46e494ed9cc7ff`. The subsequent commit changes this coordination record only. The independent scoped reviewer verified the complete fix range `e388b43148899e242800d4df348876470565ead1..10069d22780e8cc67493aca5da46e494ed9cc7ff`: **all findings addressed, no new Critical/Important/Minor breakage, no open findings**.

I1 and I2 are fixed; M1 shares the stronger checkpoint validation; M2 is resolved by explicit API documentation and the existing fail-closed consumer contract. Nonfinite derived metadata remains a stated limitation, not a finite-precision certificate. D1–D14 retain their recorded scope dispositions.

Final controller verification on the reviewed code:

- `PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q` → **364 passed in 8.58s**, no skips or warnings.
- `PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest -q -m mps` → **9 passed, 355 deselected in 2.92s**.
- Independent checks: all 16 latest WP09 artifact hashes match; scientific first16-line prefix is unchanged; report and index regenerate byte-identically; all 36 controller rulings are durable and ordered; 202 publication links resolve; original blueprint and 21 locked/raw research/reproduction files retain their bytes.

The reached study is complete: Tasks 1–6, bounded Task 11 amendment and Task 17. Tasks 7–10 and 12–16 remain gated/not run. WP02 stays stopped and ordinary WP04 inconclusive. **No training speedup was established.** The same-seed fresh-process repetition reproduced 792 diagnostic rows and 30 development-loss values exactly; it is reproducibility evidence, not independent scientific confirmation.

Work remains on current `main` with local meaningful commits. No push, merge, new experiment, backend pivot or controller work was performed. The disposable plan workspace may be removed after its identity, durable review/rulings and empty tracked-scratch checks. Scientific evidence and locally regenerated checkpoints remain available.

### Independent scoped re-review, verbatim

- **I1 — Internally overlapping parameter can partially commit** — **ADDRESSED**. `src/oar/optimizer_preview.py:32` proves supported layouts nonoverlapping from sorted strides and occupied spans, rejecting expanded and overlapping destinations during whole-input validation at `src/oar/optimizer_preview.py:67`, before candidates or a pending update are published. Commit revalidates every destination before the first copy at `src/oar/optimizer_preview.py:184`. The proof admits transposes and ordinary gapped slices; conservative rejection of uncertain exotic layouts is documented. Regression coverage at `tests/test_optimizer_preview.py:320`, `tests/test_optimizer_preview.py:333` and `tests/test_optimizer_preview.py:347` checks unchanged values/versions/gradients/moments/counters, unconsumed pending state after supported-API mutation, and native-equivalent gapped-slice commit.
- **I2 — Study-wide deadline misses later histories and completion boundaries** — **ADDRESSED**. One absolute worker deadline originates at `src/oar/oracle.py:489` and is passed into every native history at `src/oar/oracle.py:507`; `src/oar/oracle.py:295` checks it before another update, after saving any completed selected checkpoint. Native final checkpoint/synchronization precede the completion check at `src/oar/oracle.py:308`. Completed audit rows are flushed and retained before checking at `src/oar/oracle.py:549`. Summary completion and final synchronization are checked at `src/oar/oracle.py:573` and `src/oar/oracle.py:586`; reached ceilings produce incomplete/inconclusive status. The config saved in checkpoints is unchanged. Injected-clock regressions at `tests/test_oracle.py:358` and `tests/test_oracle.py:399` cover the later-history budget and final audit/summary/sync boundaries while retaining real progress and evidence.
- **M1 — Normal reproduction lacks checkpoint identity validation** — **ADDRESSED**. The shared validator at `analysis/reproduce_structural_oracle.py:57` checks identity sets, duplicates on both sides, cardinality and retained byte hashes. Both normal and recovery paths use it at `analysis/reproduce_structural_oracle.py:134` and `analysis/reproduce_structural_oracle.py:183`. `tests/test_reproduction.py:91` exercises both entry paths with duplicate-valid replacement, missing entry, bad hash and valid coverage.
- **M2 — Finite endpoints can yield nonfinite derived metadata** — **ADDRESSED under the binding documented-limitation disposition**. The public API explicitly discloses displacement/alignment overflow and requires fail-closed consumers at `src/oar/optimizer_preview.py:111`. The diff adds no timed arithmetic checks and leaves numerical rejection intact. This addresses the selected disclosure remedy; it does not claim finite-input arithmetic closure.

## New Breakage in the Fix Diff

- **Critical: None. Important: None. Minor: None.**

## Out-of-Scope Observations

- **None.**

## Checks

- **Scope checked:** the one packaged consolidated diff `review-e388b43..10069d2.diff`, reviewed in passes, covers BASE `e388b43148899e242800d4df348876470565ead1` through HEAD `10069d22780e8cc67493aca5da46e494ed9cc7ff`. No fresh whole-branch review, subagent dispatch, Git/index/HEAD mutation, research run or benchmark was performed. The only checkout write is this requested report.
- **Test evidence checked against code:** the fix report names covering tests and includes actual RED/GREEN command output. I1 records two overlap regressions failing before the fix; I2 records four injected-clock failures; M1 separately preserves the initial fixture authoring failure and the corrected real duplicate-normal RED. Reported subsequent results are 44 optimizer tests, 25 oracle tests, 18 reproduction tests, 115 focused integration passes, 364 full actual-MPS fallback0 passes and 49 publication-helper passes. These are implementer executions, not reviewer reruns. No unanswered code doubt required a focused test or suite rerun.
- **Durable record checked:** a read-only Python assertion check confirms the complete original review appears verbatim in `decisions/reports/final_review.md:120`, all 16 subsequent controller Ruling lines are copied, and scoped re-review remains pending for the controller's actual append. README links the addendum at `README.md:5`; `decisions/reports/final_review.md:10` and the synthesis distinguish immutable pre-final snapshots from this appendable coordination record.
- **Preservation checked:** the complete 14-file fix package changes neither original rulings, pre-final review/outcome snapshots, blueprint, configs nor raw research/reproduction artifacts. Its gate change is one appended WP09 maintenance record. A read-only hash assertion verifies the unchanged scientific first-16-line prefix against its retained snapshot. No new research/backend/controller is introduced.
- **Evidence graph checked:** read-only assertions verify all 125 current and 22 archived artifact hashes, all 16 latest WP09 artifact hashes, and exclusion of the final addendum from the scientific index and every WP09 hash map. The 14 historical revision bindings remain separately attributed in the diff; they were not rehashed by this reviewer. Changed optimizer/oracle sources have distinct current and historical entries, and the measured study remains attributed to `2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07`. The fix is explicitly unremeasured. All 202 current publication local links resolve. Deterministic regeneration and historical-Git-byte validation are retained implementer checks, not reviewer reruns.

## Verdict

- **Fix round: All findings addressed, no new Critical/Important breakage.** No open findings. The stopped scientific scope and original measurement attribution remain intact; final controller acceptance and its durable append remain the controller's action.
