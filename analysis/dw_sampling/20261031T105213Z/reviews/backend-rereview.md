<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Scoped backend readiness re-review

Reviewed 2026-10-31 against `.superpowers/dw_sampling/review-package-fixed.txt`:
driver SHA256 `da4f72ed243325ff877102a9fe933b94abab23369d429cb63fe98902e1717e2f`,
config `afbff17b101ffc45963b0bb8132574c405ba821f196704ff254e721e9e7ad4c3`,
protocol tests `030df25def24da60f35092c35a92372ae630af95dd20de98554e88aa50052c9d`.
Sampler and operator tests retain their previous package hashes. File-local
line numbers below refer to this fixed snapshot. This review covers the eight
previous findings and directly introduced behavior, not new scope. No source
edits, test runs, device checks or benchmarks were performed. The reported
113 CPU passes and RED/GREEN logs are supplied regression evidence.

**Spec verdict: two targeted corrections remain before calling the driver
fully measurement-ready. Quality verdict: the substantive timing/optimizer
fixes and bounded replacement logic look correct on inspection; no new
sampler mathematics or native AdamW defect found.**

## Remaining findings

1. **P2 — Repository overlap screening misses ordinary relative/module
   invocations.** `analysis/run_dw_sampling.py:278-292,1010-1016`.
   `competing_processes()` requires the repository's absolute path to appear
   in the command line. A process launched from this repository as
   `/external/venv/bin/python -m pytest` or `python analysis/collect_foo.py`
   has repository work in progress without ROOT in argv. It is not a
   cooperating driver, so the new advisory lock does not protect against it,
   and the stage can record an empty blocker list and accept overlapping
   measurements. This is a remaining part of original finding 3, distinct
   from the explicitly documented limit on unknown external applications.
   Use process cwd (with access failures recorded) and relevant resolved
   script paths in addition to argv. Screen at the session boundaries and
   preserve the stated no-concurrent-agent-work policy. A synthetic process
   fixture with external Python, `-m pytest` and cwd=ROOT would verify the
   missed case without hardware.

2. **P2 — Shared-deadline timeouts still produce incorrect scientific stop
   reasons.** `analysis/run_dw_sampling.py:207-222,503-520`.
   `command()` returns 124 for an actual timeout. Preparation treats 124
   from CPU/MPS suites as INCONCLUSIVE_MEASUREMENT and, more significantly,
   from the actual sampler primitive worker as BLOCKED_HARDWARE at lines
   519-520. A primitive can be valid on genuine MPS and simply run out of the
   remaining A budget. The final elapsed-time check is bypassed by that
   earlier return. This no longer restarts the timeout allowance or permits
   another expensive launch, so the main overspending defect in original
   finding 4 is fixed, but the required precise stop boundary remains wrong.
   Handle code 124 as INCOMPLETE_RESOURCE before operation/test failure
   classification. Keep the command's timeout and partial logs as evidence;
   do not infer missing hardware from budget exhaustion.

## Resolution of original findings

| Original | Assessment of fixed snapshot |
| --- | --- |
| 1: impossible cost cap | Closed. Cap is now 21600 seconds prospectively. Including the new possible replacement, the floor deadline is about 14562 seconds, so cost is no longer rejected for every possible calibration. Actual larger measured forecasts still correctly stop. |
| 2: missing runtime pins | Closed for the reported defect. Installed required package versions are compared against the lock; Python 3.13 and arm64 are checked before performance qualification. RECORD integrity and separately retained wheel identities remain explicitly distinguished. |
| 3: exclusive sessions | Partially closed. Cooperating invocations share a repository lock, retained policy and start screen. The cwd-only competing workload gap above remains. Unknown non-repository applications are an explicit observation limitation, not a claimed enforcement guarantee. |
| 4: inclusive A deadline/storage | Main allowance defect closed: commands use the same remaining deadline, hashing is followed by a deadline check, and storage guards run before/after the stage. Stop classification still needs the targeted correction above. |
| 5: unstable native pilots | Closed. Any supported variant with unstable selection warmup prevents selection and invalidates the session; invalid native evidence is written. |
| 6: selected block duration | Closed. Selected dense blocks are measured until duration or cap; D revalidates the frozen B count and fails closed if insufficient. |
| 7: endpoint memory | Closed for the required arm endpoint. Synchronized candidate-live snapshots retain separate RSS/Metal/driver readings and describe common fixtures/cache context without claiming peaks. |
| 8: obsolete failed attempt | Closed for the reported same-stage regression. Latest attempts supersede earlier failures while full history remains in the summary. |

## Replacement and introduced behavior

The stage budget includes a maximum of three session invocations. The local
replacement counter permits at most one replacement for failed warmup; B can
instead replace a predefined first slot after pooled native-only A/A delta
exceeds .15. It does not choose which session to discard by zero-dW or sampler
speed. Session 2 is fresh, has separately named seed/order streams, and uses
the appropriate B slot when D needs a replacement. Only the retained two
sessions supply the threshold/statistics. Failed prefixes, raw paired blocks,
completed warmups and separate invocation artifacts remain available for
sealing. Unsupported/device/resource errors do not receive a warmup retry.
No unlimited replacement or candidate-speed selection defect was found.

The separately labeled setup amortization charges the measured layer-generator
setup over the declared timed update count, while retaining steady-state
statistics and raw block times. The dense dW group profile stays separate
from observed full-step measurements. These additions do not change native
optimizer semantics or the sampler's conditional estimator. Trainer,
corpus-body access and E/F recovery remain correctly gated and unreached.

Recommendation: correct the two small remaining protocol issues, freeze the
affected source/test hashes, and continue through A. Genuine hardware,
stability, memory, storage or measured forecast failures must retain their
permitted blocked/incomplete outcomes; this review does not authorize a
performance claim.
