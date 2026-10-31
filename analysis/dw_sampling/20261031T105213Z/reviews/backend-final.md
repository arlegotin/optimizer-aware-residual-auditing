<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Final scoped backend review

Reviewed 2026-10-31 against `.superpowers/dw_sampling/review-package-final.txt`.
Driver SHA256:
`812b4095dcfa275acb18e384335b839fb31d8dac55913151fcd43bfa2276b1a3`.
Protocol test SHA256:
`5de609f6870bbd6dd7e56ba75869751f83cc5ee5927bde717fb0625273480ada`.
Config SHA256 remains
`afbff17b101ffc45963b0bb8132574c405ba821f196704ff254e721e9e7ad4c3`.
Scope is the two residual findings in `backend-rereview.md` and defects
introduced by those fixes. This is a source review: no source edits, test
runs, hardware checks or comparative measurements were performed here.

**Spec verdict: pass for this scoped readiness review. Quality verdict:
pass; no remaining material defect found in the two fixes or their call
sites. Both residual findings are closed.** Together with the earlier
re-review, the eight original backend findings are addressed to the reviewed
scope. The earlier sampler mathematics/native AdamW assessment is unchanged.

1. **Repository cwd detection: closed.**
   `analysis/run_dw_sampling.py:284-303`.
   `is_competing_work()` accepts repository identity from the resolved process
   working directory as well as argv. An external Python executable running
   `-m pytest` or a relative collector from this repository is now recognized.
   `competing_processes()` requests cwd and calls that predicate before
   measurement authorization. The new regression exercises both previously
   missed command forms and a non-repository negative case. The advisory
   driver lock, no-concurrent-agent-work policy and documented limit on
   unrelated external applications remain consistent with the prior review.

2. **Deadline stop classification: closed.**
   `analysis/run_dw_sampling.py:249-259,503-535`.
   `check_command_deadline()` raises TimeoutError for code 124 and also
   checks elapsed expiry after a nominally successful command. Preparation
   invokes it immediately after doctor, CPU/report, MPS/full-suite and sampler
   primitive commands, before their status branches. These exceptions reach
   the existing INCOMPLETE_RESOURCE classification; a spent budget can no
   longer be reported as unavailable hardware by the primitive branch. The
   regression covers timeout exit, elapsed expiry and a valid remaining
   interval.

The supplied RED/GREEN and combined 115-pass result support these regressions.
The supplied host-visible 479-pass result, including nine MPS tests, is
additional execution evidence, not a test run performed by this reviewer or
a comparative cost result. Frozen source/config/amendment and the validated
prepare-only dry run are preparation evidence; the staged qualification and
measurement gates still determine whether execution may proceed.

**Recommendation: proceed through A and, only on its valid recorded pass,
the gated B/D stages.** No remaining backend review finding blocks that
sequence. Resource, hardware, stability or cost failures retain their
specified stop outcomes. No performance or learning benefit is established
by this approval.
