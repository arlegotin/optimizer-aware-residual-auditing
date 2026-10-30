# OAR implementation progress

## 2026-10-30 — Task 1: configuration and evidence scaffolding

Implemented schema-v1 strict exploratory configuration, sorted SHA-256 hashing,
shared metadata/tensor transport records, append-only gate evidence with
fail-closed prerequisite checks, and a local package/empty command shell.
CPU configurations resolve to the `correctness_structural` scope. Confirmatory
configuration is blocked until its protocol essentials have an owning schema.

Validation used native arm64 CPython 3.13.5 in `.venv`, PyTorch 2.15.0,
NumPy 2.5.4, pytest 9.2.0, PyYAML 6.0.3 and psutil 7.2.2. Initial tests
failed on missing contracts; subsequent regression cycles covered detached
audit tensors, transitive stopped prerequisites, immutable replay descriptors,
and exact integer resource counts. Final `python -m pytest`: **41 passed in
0.64s**. CLI help exits successfully and advertises no unimplemented commands;
`pip check` reports no broken requirements. Local disk footprint at validation:
802 MiB for `.venv`, 151 MiB for the disposable pip cache. No experiment
runtime, memory forecast, training outcome or speedup was measured.

The network-restricted dependency download was retried with the authorized
tool escalation and succeeded. Source notices and patches remain accessible;
environment/cache/checkpoint artifacts and plan scratch are ignored.

**Scientific status:** WP00 remains unresolved. Scaffolding tests do not satisfy
environment provenance, wheel hashes, asset identities or MPS capability gates.
There is no scientific pass record in `decisions/gates.jsonl`. Next permitted
action is Task 2 discovery: freeze wheel/source identities, inspect native
CPU/MPS capability and record actual WP00 evidence. Later owners must explicitly
extend configuration fields and gate permissions for any evidence-qualified
structural-oracle branch; the initial gate graph is deliberately conservative.

Detailed local implementation/TDD evidence is in the ignored task report:
`.superpowers/sdd/2026-10-30-oar-implementation/task-1-report.md`.

## 2026-10-30 — Task 1 review completion

Task 1 implementation `45a229e` and correction `2115d04` have been independently
reviewed. The initial nonfinite gate-evidence issue was fixed and re-reviewed
clean; the reviewed baseline has **45 passing tests**. No scientific gate was
passed by those scaffolding tests.

## 2026-10-30 — Task 2: actual hardware, operations and provenance (WP00)

Implemented allowlisted hardware redaction, tiny capability checks, `doctor`, a
separate strict discovery configuration and content-bound local asset/notice
inspection. The shell-output fixture exercises native parsing while actual CPU
and MPS checks exercise numerical outputs, optimizer mutation, independent
CPU-generated Gaussian streams and model-RNG save/restore. Changed asset or
notice bytes invalidate supplied approval; missing, conflicting or unresolved
license evidence remains blocked.

Actual discovery was rerun outside the sandbox with
`PYTORCH_ENABLE_MPS_FALLBACK=0`. Sandbox MPS unavailability and denied `sysctl`
were process restrictions: the actual machine is native arm64, Apple M3 Max,
Mac15,11, macOS 26.7.1, with **38,654,705,664 physical bytes** and about **75.9 GB
free disk** at discovery, on AC power (`pmset` mode code 0; the separate legacy
low-power flag was unavailable). Tiny FP32 GEMM/noncontiguous input, exact native
linear derivatives, AdamW, layer norm/GELU/masked-loss and attention backward,
CPU probe transfer and model-RNG replay all passed. Tiny native QR/SVD and
explicit CPU FP64 QR/SVD also passed. Combined MPS-to-CPU FP64 conversion raised
`RuntimeError: Undefined type Double`; discovery now transfers to CPU first and
then casts. Sampling initially changed the MPS model RNG; a real-device
regression verifies that the entire suite now restores it.

Froze **19 actual runtime/tooling wheels (158,389,709 bytes)** with SHA-256 and
hash-enforced requirements. Eighteen archives came from the existing pip cache;
only the previously uncached pip 25.1.1 wheel needed a small download. Archived
package/bundled notices and selected upstream source/notices are under
`third_party/discovery/`. Whole-repository revisions pin INSTANT, VCAS, DropBP,
Fashion-MNIST and WikiText; historical excerpt blob IDs are not used as commit
pins. The exact installed PyTorch revision's fallback implementation confirms
that an absent variable or `0` disables general fallback. Its unconditional
`embedding_renorm_` CPU fallback is outside this critical-operation suite and
must not enter a timed critical path without explicit accounting.

**Gate interpretation:** WP00 passes for CPU correctness and the next operator
qualification/headroom work on actual MPS. This rests on actual operations,
frozen package/source identities and at least one available, permissible core
dataset: Fashion-MNIST has consistent MIT evidence at its pinned official
revision and both pinned training-file HEAD requests returned 200 with their
expected sizes. Dataset bodies have not been downloaded; their local SHA-256
fields remain null, upstream checksums are separately labeled and
`workload_ready` remains false pending byte verification/preprocessing.
WikiText's pinned official card still conflicts (CC BY-SA 3.0/GFDL metadata
versus CC BY-SA 4.0 prose); **the language workload remains blocked** until that
evidence is reconciled. Vision cannot substitute for a language-pretraining
claim. Official validation/test bodies were not accessed.

Final verification: **73 passed in 1.51s** outside the sandbox with actual MPS
and fallback disabled; the sandbox run had **72 passed, 1 skipped**. The
hash-enforced offline lock dry run resolved all 19 wheels, `pip check` found no
broken requirements, and all gate/source/wheel/notice hashes and environment
redaction checks passed. Exact reproduction evidence is recorded in
`.superpowers/sdd/2026-10-30-oar-implementation/task-2-report.md`. Discovery is
limited to tiny shapes; no custom OAR operator, checkpoint/replay equivalence,
shape-specific speed, thermal budget, training, learning or speedup claim has
been established. WP01 numerical/transaction verification is the next action;
WP02 headroom still depends on WP01. `doctor` reports actual capability scope
and never writes a gate pass itself.

### Task 2 review correction — fix round 1

CLI discovery configuration errors now return status 2 with a concise diagnostic
and no traceback. Effective PyTorch fallback policy is registered at import;
changing the environment later cannot prove that policy. MPS checks therefore
run in a fresh worker whose environment is set to `0` before interpreter startup
and whose pre-import state is verified. GPU eligibility requires worker policy
evidence. The manifest marks caller environment text as non-authoritative and
leaves effective fallback unknown when it has not been verified.

The import-order regression first imports Torch with fallback 1, changes the
variable to 0, and forbids MPS RNG access in that process; the real worker then
executes independently. Verification: **76 passed in 5.93s** on actual MPS,
**75 passed, 1 skipped in 3.01s** in the sandbox. A module CLI smoke test confirmed
exit 2 for fallback-enabled configuration. Actual `doctor` again classified
`exploratory_gpu`, with 11 CPU and 13 isolated MPS-context rows passing; the
fresh-process overhead is included in discovery wall time. Refreshed the
environment manifest and appended WP00 evidence with updated hashes. Dataset
limits and the WikiText block remain unchanged.

## 2026-10-30 — Task 2 review completion

Task 2 implementation `36b7dcb` and correction `d3f8fcb` have completed
independent review. The correction's actual-device baseline is **76 passing
tests**. WP00 remains qualified only for its documented tiny operator and
next-stage discovery scope; WikiText is still blocked and dataset bodies remain
unverified. Task 3 starts from the reviewed correction.

## 2026-10-30 — Task 3: projected derivatives and past-only construction

Implemented `projected_linear` with native forward, exact input/bias adjoints,
projected dense weight gradients and detached streaming `H`/`R`. One
`LinearCapture` spans an effective batch, with fixed basis/probe storage and
parameter identity/version checks. Higher-order differentiation fails before
capture mutation. No token normalization is applied inside the operator, and no
activation/adjoint archive is retained after ordinary backward releases its
saved tensors.

Implemented ordered randomized QR/small-SVD construction from a supplied exact
pre-update gradient, using an independent CPU generator and explicit CPU FP64
decomposition. Numerical null directions are dropped using the source
precision. `BasisState.for_step` and `LinearCapture.from_basis` require a source
step strictly before the consuming step. Source and returned-vector transfers,
MPS synchronization and inclusive construction time are recorded. The trainer
will own refresh-gradient lifetime and include complete construction time in
active time; the basis state retains no gradient archive.

TDD began with four expected missing-module errors. A subsequent red/green
regression found and fixed the missing parameter-freeze check between
microbatches. Tests compare against independently materialized native CPU FP64
autograd, including rectangular/noncontiguous inputs, rank zero/full/deficient
bases, zero adjoints, bias/no-bias cases, four residual blocks, an injected wrong
input derivative, higher-order rejection and seven unequal masked microbatches.
The MPS marker now selects both new device checks and the existing discovery
device check: **3 selected, 125 deselected**.

Final full verification outside the sandbox with
`PYTORCH_ENABLE_MPS_FALLBACK=0`: **128 passed in 6.06s** on actual MPS/CPU.
Committed [operator correctness evidence](../analysis/operator_correctness.json)
contains 12 CPU/MPS cases, six deficient-basis constructions, accumulation and
deep-adjoint errors, seeds, source hashes, shapes and tolerances. Maximum
absolute operator error over those fixtures was **9.04e-15 CPU FP64** and
**2.39e-6 MPS FP32**. These are tiny correctness results; construction times are
inclusive diagnostics, with no performance or speedup claim and no calibrated
production audit guardband. Detailed commands, TDD evidence, interfaces and
self-review are in `.superpowers/sdd/2026-10-30-oar-implementation/task-3-report.md`.

**Scientific status:** Task 3 is complete; WP01 remains open for Tasks 4 and 5
(confidence/audit allocation and functional optimizer preview). No WP01 pass or
dependent experiment authorization was written. Existing dataset and workload
blocks remain in force. Standard `.DS_Store` exclusion was added while
preserving the local Finder file's bytes.
