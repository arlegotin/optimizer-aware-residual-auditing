# Stopped-study synthesis schema and reproduction

`build_report(gates: Path, run_index: Path | None, out: Path) -> Path` renders a stopped/reached-stage report; the CLI is `report --gates PATH [--run-index PATH] --out PATH`. It never appends gates. With no index, measurements stay unavailable. Invalid evidence fails before writing the report; input evidence cannot be the output path.

Schema1 supports only an explicitly annotated artifact index, not arbitrary experiment logs or a confirmation protocol. Required fields are `schema_version:1`, `root` (relative to the index), `gate_snapshot:{lines,sha256}`, `artifacts`, `evidence` and `audits`. JSON duplicate keys, unsupported fields, nonfinite metrics, missing scopes/units and malformed intervals are rejected. Artifact paths stay within the root; symlink escapes are rejected.

An artifact has `path` and SHA256, with optional full `revision` for historical `git show REV:PATH` verification `archived_path` for exact bytes preserved at a different local path, or `unavailable_reason` for an explicitly unavailable historical reference. These options cannot be combined. Current evidence/audits require a verified current artifact. Every gate artifact/hash pair in the snapshot must appear. Historical snapshots are verified from Git objects rather than silently checked against evolved current files. Disposable historical workflow report references are retained as unavailable and excluded from scientific support; the complete durable oracle report remains current.

An evidence annotation requires `evidence_class`, `title`, `artifact`, `status`, `scope`, `summary`, `uncertainty` and `metrics`. Classes are exactly `algebra/reference`, `backend timing`, `exploratory learning`, `confirmed learning` and `unresolved`. Status is `measured`, `blocked`, `failed` or `unresolved`. Nonmeasured entries classify unresolved. Schema1 cannot establish valid confirmation: any `confirmed learning` claim is reclassified unresolved and its asserted title/summary/measurements are omitted. Metrics have a numeric/null `value`, a nonempty `unit`, and optional finite ordered two-element `interval`. Null stays unavailable. Audits have `kind` (scientific/experimental/implementation/decision/reproduction), `summary` and verified `artifacts`.

Annotations are authored interpretations; byte integrity does not validate a scientific conclusion. Their scope and uncertainty are displayed alongside their values. No training measurements, statistical tests or successful learning results are inferred from absent data or a passed synthesis gate. The current supported use is an honestly stopped study, rather than a future complete-protocol report engine.

## Stable scientific snapshot

The first16 lines of [gates](../gates.jsonl) are explicitly bound in [snapshot](../../analysis/scientific_gate_snapshot.json). Report/index output and future WP09 records are outside that prefix. The report allows only WP09 records after it; a later scientific gate requires deliberate snapshot/index revision. The [annotations](synthesis_annotations.json) carry source-linked interpretations. The [collector](../../analysis/collect_synthesis.py) rebuilds current/historical bindings without research execution:

```sh
.venv/bin/python analysis/collect_synthesis.py
.venv/bin/python -m oar.cli report --gates decisions/gates.jsonl --run-index analysis/evidence_index.json --out analysis/report.md
```

Neither report nor index hashes itself. WP09 is appended only after checking report output and reproduction. Rerunning both commands with unchanged inputs must regenerate identical bytes even after appended WP09 records. New annotations/source bytes deliberately change the index/report and require new synthesis evidence, not fabricated empirical provenance.

## Declared research repeat and schema correction

The committed [plan](../../analysis/reproduction/wp09/plan.json) binds original study bytes/source attribution, the actual path-only config clone, its scope amendment,28 scalar fields,23 categorical fields, all exact harm signs/acceptance decisions, and all30 learning curves. Numeric tolerances are absolute2e-6/relative2e-4, fixed before execution. The [helper](../../analysis/reproduce_structural_oracle.py) validates hashes, invokes the existing guarded fresh-MPS command with fallback0, and restores the preceding amendment by append. Its new repeat may change only output/checkpoint paths. Both actual clone and original config hashes are distinct in the manifest; the worker's historical hardcoded original config provenance is preserved, not relabeled.

The first processing attempt compared all792 rows exactly, then failed on the nonexistent curve key `step`. Its [failure](../../analysis/reproduction/wp09/comparison.json), original declaration and raw outputs are preserved. An explicitly authorized postprocessing correction maps predeclared logical `step` to existing `committed_updates`; it changes no observations, tolerances, categories or selection. The [recovery](../../analysis/reproduction/wp09/comparison_recovered.json) compares all30 curves and validates all18 local checkpoint hashes without launching research. Both row/curve differences are zero. The original status remains failed; recovered numerical reproducibility is separately established. No independent seed confirmation follows.

The repeat's61.229s worker and62.914s invocation are distinct accounting clocks. A 1.27s report-only CPU test may have overlapped; no MPS tests, benchmarks or collectors overlapped. These times are not timing replication or speed estimates. Original60.959s worker excludes startup/final-summary tail and is never retroactively repaired using this repeat.

The first WP09 synthesis output/index are preserved under `analysis/history/wp09-initial-*`. Final staging found an extra trailing blank line; newline normalization changes no scientific content. A superseding WP09 record binds the normalized deterministic output. The scientific16-line snapshot and all original research remain unchanged.

## Final review coordination

The [review summary](../review_summary.md) and [outcome](../outcome.json) preserve the hash-bound pre-final snapshot. The [final review addendum](final_review.md), discoverable from README, preserves the complete whole-branch review, all subsequent controller rulings and component-fix verification; the scoped independent re-review was accepted on 2026-10-31, with the durable acceptance committed at `f001b2b`. It is deliberately outside the scientific evidence index and WP09 artifact hashes so a later acceptance-status append cannot introduce a hash cycle. Current source/README/index/report bindings are refreshed with an appended WP09 maintenance record; prior published bytes resolve through Git. Original measured source identities and raw study/timing artifacts are unchanged. Validation changes at `7628be7` were tested, not remeasured.
