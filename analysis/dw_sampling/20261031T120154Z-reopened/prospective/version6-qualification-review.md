<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Whole reached-branch qualification review

2026-10-31. Initial disposition: **fix required before qualification freeze**.
This is review of the new uncommitted source, not an assertion that current HEAD
contains the model implementation. No corpus body, model MPS run, benchmark,
network request, dependency change, or implementation edit was made by this
reviewer. The stale amendment is a known pending freeze, not a measurement claim.

## Required fixes

1. **Important: pool equal-work native observations.** In the initially reviewed
   `analysis/run_dw_sampling.py:1557-1558`, each fresh variants worker calibrates
   its own updates-per-block. Lines 1584-1585 retain that count, but
   `pooled_model_native_variant` at lines 1600-1604 discards it and pools raw
   block seconds. This can select a slower native baseline, contrary to the
   version-6 strongest pooled native requirement.

   A pure arithmetic call to the real helper reproduces it. Session 0 has one
   update per block: single=.60 seconds, fused=.50 seconds, each repeated three
   times. Session 1 has three updates per block: single=1.20 seconds,
   fused=1.32 seconds, each repeated three times. Current helper selects single
   (pooled raw medians .90 versus .91); equal-work per-update pooled medians are
   .50 versus .47, selecting fused. Normalize each observation by its own
   validated positive integer count, retain raw seconds/counts/provenance, and
   add this unequal-count regression before freezing. Parent accepted the fix.

2. **Integration: update stale schema-exclusion assertions and bind the model
   qualification regressions.** The scoped CPU command below has two failures
   at `tests/test_dw_sampling_protocol.py:151` and `:1074`: assertions omit the
   deliberately documented `schema_version` exclusion. The implementation's
   explicit supported schema2/3 check is appropriate; repair the stale test
   expectations. At initial review the nine-worker sequencing, whole-round
   fallback and cost-stop regressions exist only in the ignored
   `test_model_cost_pending.py`; fold them into the bound protocol suite before
   final A/C. Parent confirmed this folding is planned.

The parent independently identified a related conservative calibration gap at
`aggregate_model_calibration`, initially lines 1789-1794: U/Bmax use only the
three cold blocks, omitting qualified fixed-native timing observations. The
parent is adding a baseline-only provenance ledger of selected native pilots,
both A/A sides, and candidate comparisons' dense sides, retaining all valid
observations. This change still needs its scoped re-review before final approval.

## Scope and inspected behavior

Read the original scope, final v6 reopening, model-cost/model/recovery reports,
data/model/recovery/mathematical reviews, the complete new model/data sources,
the full driver/config changes, targeted protocol fixtures and the scratch
qualification fixtures. Checked changed existing ASTs against both exact B/D
source commits as a supplement to reading their complete driver diffs.

- All 184 original tracked file hashes match the initial inspection. Historical
  OAR records/configuration, the stopped negative result and asset manifest are
  untouched. No historical scientific state is upgraded.
- Independently ran `read_record` against B and D, which verified their complete
  artifact seals. B SHA is
  `716ec10cfe1827b8e85f03deb5229484ff7c20b80d2c7178b1395c4b5acda506`;
  D SHA is
  `d52f2062995810f4541022d02ed21e8f33ad72257675f66f8bed773c34fa9b18`.
  Every original measurement-map file matches its original committed Git blob.
  D retains the five declared surviving arms at4096. The v5 upper95%<1 rule
  and unsuccessful old 1-delta sensitivity remain disclosed.
- Existing native fixture, sampled operator, device configuration, measured
  timing/statistical functions and D gate are unchanged. Added model modules
  are lazily loaded by new paths. Changed existing driver functions concern
  prerequisite/source validation, uncapped command/preparation policy, worker
  routing and main orchestration. The earlier B-to-D `hardware_stage` change
  keeps B's gate semantics and changes the prospectively approved D threshold.
  No new transitive alteration of the old native measurement path was found.
- Current complete dependency maps and exact sealed B/D carryover must be
  pinned by the new amendment. A/C must rerun for the added model/data source.
  Selected AST checks alone are not universal approval. End-of-stage drift
  invalidates the result while retaining original source identity.
- Pilot qualification checks all four prerequisites before body preparation.
  The exact local-use decision gates pinned train/validation SHA/size checks;
  official test has no acquisition/parser path. Processed validation tokens are
  separate from the only development artifact accepted by model workers.
  No qualification path computes official-validation losses. PyArrow version
  is checked and the new reader/wheel evidence must join the final bindings.
- The production model remains fixed FP32, four blocks, width512, heads8,
  FFN2048, context256, B16, with complete causal attention, padding, exact tied
  output and native AdamW/clipping. Baseline FFNs retain native Linear forward
  and no sampling state/hooks. All runtime candidate checks and sampler work
  are inside the timed steps. Native common history includes nonzero moments;
  fixture-only restoration adapts only the candidate sampler identity/initial
  streams, while production restoration preserves strict bindings.
- The nine-worker sequence freezes one pooled native variant, then A/A, then
  jitter before two all-survivor candidate sessions; cold calibration follows
  only an actual two-session model cost pass. One complete common-block fallback
  is retained. Success returns overall pilot `not_run` with no learning success;
  actual cost failure returns STOP_COST and cannot unlock confirmation.
- The cold worker launch timestamp is taken in the parent before subprocess
  launch/Torch import, with data/source validation and initialization charged.
  Native finite-loss checking occurs after backward/clipping and before AdamW;
  scalar logging follows synchronized blocks. Initial checkpoints/evaluations,
  trained checkpoint IO and full endpoint costs are measured. The finite-loss,
  native-state and cursor corrections identified by earlier scoped reviewers
  are present.

The future `train_budget`/training-worker primitives were read as integration
context. They do not establish a complete authorized learning dispatcher,
parent interruption ledger, tuning/independent-development selection, protected
confirmation enforcement or independent reproduction. Qualification approval
must remain explicitly scoped; those future items are not fabricated successes
and are not reasons to add an unrelated framework now.

## Verification

The independent scoped tiny-CPU command was:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q \
  .superpowers/dw_sampling_reopening/test_model_cost_pending.py \
  tests/test_dw_sampling_protocol.py \
  -k 'model or budget or language or end_source_stability or native_pass_continuity or cost_continuity' \
  -m 'not mps'
2 failed, 56 passed, 209 deselected in 7.78s
```

Both failures are the explicit schema exclusion assertions above. The native
pool defect was separately reproduced with the real helper and literal timing
arrays, without Torch, timing, or hardware work.

Initial inspected source hashes (superseded by fixes; no final approval yet):

```text
96ce86183922284eeb57ffa17917ed81b27b3a1ab8d2ce9b70266edc1b171884  analysis/run_dw_sampling.py
c02e51d440566f74e7a2852c79d72a42aaa65a150e5a6ce4ca4c9f98cd031c91  src/oar/sampling_lm.py
cfcf4b782eb9a444ce70f29df72e063518f794108020e51a5d12238118f06583  src/oar/sampling_data.py
7411e08b4430d06152c4c93a0c588660eedb9f6a27577ab002721b28284fe368  configs/dw_sampling.yaml
2ea082adb4b3e8f99b61fd60b0c72f7ac9e486b9736018790b38f120e3bcb182  tests/test_dw_sampling_protocol.py
9ce20da782a0122fa3a622ef305689b90407dd301bc787cf5d5eb12fae964fbe  decisions/dw_sampling_reopening.md
```

The known old summary prose must be corrected before publishing a post-body
qualification report. It has not been treated as a new finding because the
parent already owns that pending change. Final exact source approval will need
the completed fixes, regression folding, renderer change and final freeze map;
this initial review does not authorize carrying its approval to unseen bytes.

## Final source re-review — 2026-10-31

**PASS for the qualification source freeze at the exact hashes below. No
remaining qualification-blocking finding.** This supersedes the initial fix
disposition above. It approves the reviewed full diff for exact B/D continuity
into this qualification-only consumer. It does not assert that the pending
amendment has been frozen, A/C have rerun, corpus bytes are verified, MPS model
qualification has passed, or any language learning result exists.

Verified all six hashes supplied for re-review before proceeding. Re-read the
changed pooling, calibration, pair-order seed, clock, renderer, interruption
and checkpoint-retention code, the folded qualification tests and new
regressions, and the corresponding prospective scope additions.

- Required pooling fix is closed at `run_dw_sampling.py:1600-1622`: raw
  observations are divided by each session's validated positive integer update
  count. Raw seconds/counts/session provenance and the per-update unit remain
  explicit. The exact unequal-count reviewer case now selects fused, with
  normalized observations .50 and .44 instead of selecting single from raw
  block lengths.
- Both stale schema-exclusion assertions are corrected. The controller and
  actual tiny CPU worker tests are now in the bound protocol file, including
  all-survivor sequencing, native support intersection, fatal numerical
  failures, nine-worker ordering, cost failure and the single whole-round
  fallback. They are no longer dependent on ignored scratch tests for A/C.
- The parent's conservative-calibration correction is closed at
  `run_dw_sampling.py:1798-1845`. The controller passes the selected variant's
  pilots, both A/A sides and every candidate comparison's native side into a
  retained ledger; the three cold controller blocks join it. U is the maximum
  seconds/update and Bmax the maximum complete block duration. Counts/durations
  are validated; slow valid rows are retained. The 1.2s/four-update regression
  dominates the .8s/eight-update cold rows as intended. Candidate elapsed
  durations do not determine these native budget bounds.
- Model comparison pair ordering now uses the named `order` domain and fixed
  session/arm/boundary identifiers. The parent records its monotonic clock
  implementation before launch, and the cold worker rejects a differing
  implementation before its model/data work. Torch import remains inside the
  parent's launched interval.
- Summary rendering no longer incorrectly denies reached body ingestion or
  recommends stopping merely because qualification is an unfinished pilot.
  It still makes no learning claim.
- The dormant learning loop records the selected LR and planned block before
  execution; an interrupted partial block records host counters, last committed
  checkpoint and failure attribution without claiming confirmed MPS completion.
  It fences the live state against reuse. The final checkpoint receipt is
  fsynced before this invocation's ephemeral checkpoints are pruned. No old
  user/prior-run files are selected for pruning.

One future-dispatcher integration limitation remains explicit: the loop's
`checkpoints` list begins empty on resume, so this helper's retention covers
files created by the current invocation. The future parent run-wide ledger
must enforce the two-live/final-alone policy across attempts, as well as full
interrupted active time. The current qualification route never invokes this
loop and does not claim that dispatcher exists; this is not a qualification
blocker or an approval of whole-run recovery enforcement.

Independent final verification:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps'
268 passed, 4 deselected in 9.92s
git diff --check
exit 0, no output
```

Revalidated both complete B/D artifact seals and every original measurement-map
Git blob. All protected native timing/statistical definitions remain unchanged,
and native configuration equality passes with the documented exclusions (D
retains its exact cost gate). Reviewed all changed existing driver definitions
and added model paths for transitive effects: source/prerequisite/command/
preparation/routing/summary changes do not alter the already-recorded native
measurement semantics. All 184 historical file hashes still match. Strict
configuration loading resolves to `learning.action == qualification`.

Exact approved complete pilot-consumer source map:

```text
9944561dff1909e168549eea97f727a2fdbb5e9254d790a32a0403796235277f  analysis/run_dw_sampling.py
7411e08b4430d06152c4c93a0c588660eedb9f6a27577ab002721b28284fe368  configs/dw_sampling.yaml
d91d9350dc117f83e39c648e48d435de75ab9ea5a3b6bc197b216c4b73c11181  decisions/dw_sampling_scope.md
f86a1ba2a28c5dd7e70c2ac685fe4a6ec62824d95f429bc935ef4adfedfec54f  decisions/dw_sampling_reopening.md
25351cd20abc14385dd1cb3e482c67f73619b1ead30900ca0150fe57072458e5  src/oar/timing.py
4ba118d816eec9d255cfb3d7925416373119fe181f82529fb58a96347d8e5048  src/oar/hardware.py
11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3  src/oar/sampled_linear.py
3ed4adbf6f62599172b0322374e84c7552f7fc4b46761f80057053469822eaa9  tests/test_sampled_linear.py
4e419fc5a493be734792a0d2e333366f7a318b7c7ca5cf6c895d0f522ea84d6e  tests/test_dw_sampling_protocol.py
b6a0ec4e1b66ccdf542e8d19c6830b897aec9fa4d4d4a77b13265dedb67bb21c  src/oar/sampling_lm.py
cfcf4b782eb9a444ce70f29df72e063518f794108020e51a5d12238118f06583  src/oar/sampling_data.py
```

Proceed with the exact new amendment/source/asset freeze and current A/C;
only their successful gates permit the already-scoped qualification action.
No corpus request, actual MPS test, benchmark or hardware qualification was
performed during this re-review. No implementation/test/config/scope bytes
were edited by this reviewer.
