<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Scoped MPS transfer-boundary disposition

Reviewed 2026-10-31, commit `3500387`, limited to
`git diff afa51cb..3500387 -- src/oar/sampled_linear.py tests/test_sampled_linear.py`.
Reviewed sampler SHA256:
`11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3`;
test SHA256:
`3ed4adbf6f62599172b0322374e84c7552f7fc4b46761f80057053469822eaa9`.
Also inspected `mps-boundary-report.md` and the retained v1 preparation
attempt-0002 record. No tests, device operations, benchmarks or source edits
were performed during this review; A v2 remains under the parent's control.

**Spec verdict: pass. Quality verdict: pass. No material defect found in the
two transfer changes or the added regression.**

At `src/oar/sampled_linear.py:92-93`, device FP32 scores now move to CPU
before the CPU FP64 cast and normalization. At lines 125-127, inverse-square-root
weights are still computed from the same CPU FP64 probabilities, then cast
to the training dtype on CPU before transfer. Both orderings avoid presenting
an FP64 tensor to MPS. They preserve the estimator, score normalization,
mixture, duplicate samples, inverse-probability weights, production budget,
RNG consumption and native optimizer semantics. CPU FP64 reference behavior
also remains valid. The runtime timing path naturally charges these actual
operations when comparative work is eventually reached.

The MPS regression at `tests/test_sampled_linear.py:545-581` exercises both
boundaries rather than checking source text. Its independently computed
probabilities are correct: normalized scores .5,1,.5,1,0 sum to 3, giving
the .9/.1 mixture [.17,.32,.17,.32,.02]. The duplicate-row gather reference
has the correct factor 1/p for two identical draws at k=2. The sampled
backward reference checks the real dense dW, exact dX/db, forward result,
device/dtype/finiteness and dedicated CPU RNG state, at unchanged tolerances.
The supplied actual-MPS RED/GREEN and focused CPU results support this
specific correction; this reviewer did not rerun them.

The retained v1 status `BLOCKED_HARDWARE` denotes the failed qualification of
that implementation's combined transfer/cast call. Actual MPS hardware was
present; the failure does not establish hardware absence or impossibility
of the prescribed sampler. The correction is an ordinary implementation fix
within the same prospective experiment. Archiving v1, freezing v2 before any
comparative timing and repeating affected A qualification preserve source
attribution and gate validity. The old failed attempt must remain retained.

**Disposition: accept the scoped fix and continue the already-running A v2
qualification.** Subsequent B/D work still requires its valid hash-bound
pass and all other gates. This review establishes no speed or learning claim.
