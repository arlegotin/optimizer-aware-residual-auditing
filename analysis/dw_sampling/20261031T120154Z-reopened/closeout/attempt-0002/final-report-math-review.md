<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Final v2 mathematical/statistical report review

2026-11-01. Reviewed `analysis/dw_sampling/20261031T120154Z-reopened/final_report-v2.md`,
SHA256 `d814ef894662ef72cac4eb70b53d2e500de02d01af10b0022b5b808e3baffaf2`.

**Approved in this mathematical/statistical scope. No Critical or Important
finding.** The prior sealed-outcome reconstruction and draft-consistency review
remain applicable; no broad algebra/backend audit or test suite was repeated.

The final preserves the actual STOP_QUALITY, all 15 sampled endpoint failures and
target censoring, R0 native B8/LR3e-4, observed target time 907.1045402500313s,
endpoint ceiling 1.241502616051292, family table and observed-time simplicity tie.
It retains the distinction between later-only target validation and R0's
full-trajectory twice-confirmed time. No H substitution, finite sampled time
ratio, independent-seed learning interval or favorable unseen outcome is claimed.
The single tuning seed, unexecuted protected/development/repeat stages, and limits
of the timing-block bootstrap are explicit. Cost savings remain separate from
learning benefit.

The conditional variance formula is the iid common-row squared Frobenius-error
identity. Raw adjoints are distinguished from globally clipped applied gradients,
and finite PRNG/score/weight/GEMM rounding from ideal arithmetic. The clipping
percentages and throughput values agree with the earlier independent raw-metadata
calculation; their post-hoc, different-trajectory and noncausal status is preserved.

One Minor wording precision was sent to the author: “the 0.1 uniform mixture does
not [minimize variance]” should read “is not generally the unconstrained optimum.”
When all row norm products have the same positive value, both norm probabilities and their
uniform mixture are uniform and attain the same optimum. This does not affect
the implementation, reported arithmetic, measured outcomes or approval.

Independently verified the added accounting against all 26 unique ledger records:
each recorded hash/status/stage/wall charge agrees, run subtotals sum correctly,
and total known outer wall is 47673.64810620784s = 13.242680029502177h. The wording
correctly includes nested attempt-0002 preprocessing once, separates outside-stage
costs, and avoids calling the subtotal GPU-busy time or complete project expenditure.
No estimates were changed.

Authenticated `cost/attempt-0001/diagnostic.json` through its sealed cost-record
hash (diagnostic SHA256
`6ca37886ebb2f14f36fa4c079727af0064538f0fe3afd918a2e9750d0619c836`). Recomputed all
six dense-GEMM and score/probability/check medians from the original three samples
per state/layer: every displayed millisecond value agrees at six decimals.
The diagnostic's 15.263710667029954s rounds to the reported 15.263711s. The final
correctly fences these observations as perturbed component measurements that
cannot be added to infer an asynchronous complete step or runtime learning win.

Current-source test/backend/recovery/fresh-checkout claims remain the authority of
their separate artifacts/review contexts; this mathematical addendum does not
claim to have rerun them. The final evidence manifest is intentionally still
pending assembly and is not a premature missing-evidence finding.

Used only public metadata, source/document reads and standard-library arithmetic.
No project imports, tests, hardware, network, corpus/checkpoint bodies or source
edits. The sole write for this final check is this ignored report. Approval is
bound to the report SHA above; a minor “not generally” correction preserves it
substantively, while material mathematical/scientific wording changes need a new
narrow consistency check.

## Exact final v3 approval — 2026-11-01

Reviewed `final_report-v3.md`, verified SHA256
`f73627316aa6d60e12e0e00cccb2c78ebee0311f6b24603a51b03533815ab690`, and inspected
its complete diff from reviewed v2. **Approved; no Critical, Important or remaining
Minor mathematical/statistical finding.**

Besides its version title, v3 makes only the stated corrections: the mixture is
“not generally the unconstrained optimum”; update1024 captures were not requested
or created in the grid; and the private-checkpoint verification claim refers only
to final endpoints. Narrow public-metadata checks confirm all 27 grid requests
have no secondary update and empty secondary history, and all 27 result rows have
no secondary checkpoint. Private endpoint-byte verification remains the separate
backend/integrity review's authority; no private bytes were read here.

No arithmetic, source identity, censoring, selection, outcome or inferential limit
changed. Prior substantive approval now applies to exact v3. Only this ignored
review was appended; no tests, project imports, hardware or broader audit occurred.
The final evidence manifest remains intentionally pending the approval addenda.
