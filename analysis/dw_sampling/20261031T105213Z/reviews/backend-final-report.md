<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Final backend, optimizer, cost and recovery report review

Reviewed 2026-10-31. Source commit:
`cf6481741933130f2e86b440e750909686b81eb3`.
Scope: the native fused/scalar counter-comparison repair since `3500387`,
its regression, and the stopped empirical report's source attribution,
resource/exclusivity and recovery claims. No tests, hardware work, comparative
timing, source edits or subagents were used by this reviewer. Read-only
artifact/hash checks and arithmetic on already-recorded timings were used.

## Reviewed identities

| Item | SHA256 |
| --- | --- |
| final_report.md, snapshot including pending completion addendum | `770417e82136ad1f4529cdc5b32c3d50783aa23b4f528530481699ca0d91c80c` |
| analysis/run_dw_sampling.py, v3 | `227f6977d4713e84d01cf2ebeb61d390d769f391da6fe14e65704daac47df663` |
| src/oar/sampled_linear.py | `11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3` |
| tests/test_dw_sampling_protocol.py | `9772893f8002de46004ff334a39209d9df9438fc6bbf51b9dac91cf004dd1c3f` |
| tests/test_sampled_linear.py | `3ed4adbf6f62599172b0322374e84c7552f7fc4b46761f80057053469822eaa9` |
| decisions/dw_sampling_amendment.json | `7c269df87b3c726def0dccebf245e978da00f1835def79482ecaaf425d613c27` |

The report's completion addendum explicitly marks the final reviews and
fresh-checkout execution pending at this snapshot. This review supplies the
backend report disposition only; subsequent addendum results must retain their
own actual evidence and final report hash.

## Findings and verdict

**No material defect or unsupported backend/cost/recovery conclusion found
within this scope. Spec verdict: pass for the reported stopped outcome.
Quality verdict: pass for the counter repair and the scoped final report.**
The experiment remains **INCONCLUSIVE_MEASUREMENT**, with no qualified sampler
cost result and no learning result. This approval does not turn the stopped
run into a complete or passing B qualification.

The original B checker compared a fused MPS scalar step counter directly with
a CPU scalar-reference counter. The retained RED identifies only a device
attribute mismatch. `analysis/run_dw_sampling.py:564-576` now compares detached
counter values on CPU at zero relative and absolute tolerance. Parameter and
first/second moment checks retain their previous device requirements and
FP32 tolerances. Moving the counter comparison does not alter either native
optimizer's state or update. Its synchronization belongs to qualification,
outside subsequent timed blocks. No optimizer implementation, clipping,
decay, sampling or benchmark rule was changed by this repair.

`tests/test_dw_sampling_protocol.py:206-223` exercises the same production
checker over five native fused/scalar steps with matched, varying gradients.
It verifies the differing counter devices explicitly. It is a meaningful
regression for the observed false rejection; its tiny fixture is correctly
not presented as full-shape fused timing qualification. The supplied original
RED and 117-pass GREEN logs support this repair. The saved final full-suite
output reports 481 passes in 8.61 seconds. These are inspected execution
records, not runs performed during this review.

## Evidence and source mapping

Record sidecar hashes and every listed artifact hash were checked for
Prepare3, Feasibility1, Correctness1 and Summarize1; all matched.
Their source manifests map Prepare3/B to
`3500387ec42f0559299ab919d6e3e6af26c3f069` and C/summary to
`cf6481741933130f2e86b440e750909686b81eb3`. The report attributes them accordingly.
The unchanged v2 warmup observations are not relabeled as measurements of
the corrected v3 qualification path. The v1 transfer failure is retained and
correctly explained as an unsupported combined operation on existing actual
MPS, rather than hardware absence.

The B record contains two worker invocations and exactly one replacement.
Both retained native-variant files contain 32 warmup blocks for each of
default/single/foreach, with `stable=false`. Recomputing the prescribed last8
range/mean reproduces the report's values:

| Variant | Initial | Replacement |
| --- | ---: | ---: |
| default | .2300 | .1614 |
| single | .1535 | .2639 |
| foreach | .2866 | .3016 |

These six failures support the actual conservative stop under the frozen
selection rule. Fused was wrongly rejected before its timing warmup; the
report discloses that defect and does not claim fused instability, timing
qualification, strongest-native selection or a counterfactual speed result.
No post-repair B rerun or second replacement is represented. No A/A margin,
zero-dW pairs, sampler cost pairs, ratio/CI, or middle/wide comparative result
is inferred from incomplete native pilots or Stage-A operation checks.

## Resource, exclusivity and recovery claims

The B resource forecast and deadline were bound before the sessions; the
workers stopped far inside them. The retained start record reports no
screened repository blockers and zero unavailable process records, with an
advisory repository lock and an explicit external-application limitation.
The report preserves that limit instead of asserting global device idleness.
RAM/swap observations are context, not a diagnosed cause of instability;
separate RSS/Metal/driver endpoint measures are not added or called peaks.
Research invocation/startup costs are not promoted to a canonical training
clock or silently omitted from a claimed time-to-target result.

The report states zero language trials, updates, corpus body processing and
checkpoints. Sampler stream save/restore and fixture resets therefore do not
become claims of E/F committed-checkpoint recovery, restart-clock equivalence
or recovery-tested language training. C correctness on the current source
does not bypass failed B or the independently unresolved asset gate.

**Recommendation: stop this run, preserve its evidence and the corrected
operator, and accept the report's limited conclusion.** Any future timing
attempt needs new prospective authorization and valid hash-bound affected
qualification; current correctness does not authorize reopening this B
attempt. Completion reviews/fresh-checkout evidence may be appended with
their actual scope, without upgrading the absent cost or learning outcomes.
