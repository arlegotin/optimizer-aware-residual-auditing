<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Pre-D cost-policy and prerequisite continuity review

2026-10-31. Scope: the fixed cost-policy package against
`d7782862c025da6ee27dc9f7692b75fe926cb070`, actual current sources, original
sealed A/B evidence and the revised reopening scope. Source inspection,
read-only Git/AST comparisons and artifact hashing only; no tests, hardware
work, source edits, corpus use or subagents.

**Verdict: ready to freeze this exact reviewed revision and proceed to current
correctness qualification and exploratory D. No blocking finding in the actual
patch or original prerequisite evidence.** This approval is specific to the
hashes below. Parent must pin exact carryover entries and current hashes in
the new amendment and commit them before candidate measurements. The stale
pre-freeze amendment is expected and is not an implementation finding.

## Statistical policy

The new D rule requires the setup-inclusive paired candidate/dense interval's
95% upper endpoint strictly below 1 for the same arm and relevant shape in
both fresh sessions. Equality at 1 or a failing session does not pass. This
is an honest exploratory cost screen when prospectively frozen before any D
timings or candidate outcomes, with the policy change and its timing disclosed.
It is a weaker criterion than the previous `1-delta` screen and must not be
described as passing that earlier criterion unless the separate sensitivity
result also passes.

The observed A/A diagnostic `delta = 0.2085046466766194` measures individual
block discrepancies. It is not a minimum worthwhile geometric-mean speedup
or the standard error of the paired mean log ratio. Removing its role as a
required 20.85% gain does not by itself discard observations, weaken the
native implementation, or manufacture a measured benefit. The paired
bootstrap estimator and actual measurement boundary are unchanged. The
result explicitly reports `gate_upper_ratio`, `noise_guard_upper_ratio`,
`eligible` and `noise_guard_eligible`; the inherited delta artifact still
precedes candidate workers.

This is a policy refinement after native timing exposure, not an untouched
preregistration. The record must preserve that fact. Neither two sessions
nor the block bootstrap establishes a familywise guarantee across arms and
shapes, independent training-seed evidence, or immunity to correlated host
timing drift. The scope correctly describes an exploratory screen. Report
all intervals, effect sizes, controls and the old `1-delta` sensitivity;
do not promote a small or inconsistent effect into a useful learning claim.
The future language/confirmation paragraphs do not themselves supply a
learning implementation, asset permission, seed-count design or protected
confirmation qualification; those remain separately frozen future work.

## Continuity and evidence

The original B record remains `pass` at both middle2048 and middle4096 under
its original `upper95% < 1-delta` rule. Both sessions contain all five declared
shapes. Original delta, eligibility, source hashes and record bytes remain
attributed to d778286. At inspection there were no cost-stage records. The
pre-candidate timing declaration is supplied by the parent/scope; filesystem
absence alone would not prove that no unrecorded timing ever occurred.

Independent read-only checks verified the A record seal plus its 28 artifacts
and the B record seal plus its 75 artifacts, with no integrity errors. A and B
both have sealed PASS status. A comparison of all existing top-level driver
function/class ASTs found changes only to `require_stage` and
`hardware_stage`; the only added definitions are `git_blob` and
`prerequisite_carryover`. The actual diff leaves the estimator, delta
calculation, all measured bodies and other runtime helpers unchanged.

The carryover path requires the latest prerequisite to have the requested
stage and PASS before consulting a declaration. Its exact record digest,
old measurement map and current consumer map must match an amendment entry.
It checks complete original dependency keys, actual current file hashes,
the sealed original source manifest and the original Git commit's blobs.
Sampler, sampler tests, hardware, timing and the original scope must remain
byte-identical to their original Git versions. Only driver, config, reopening
scope and protocol tests may differ; omitted changed protocol tests still
need an explicit current amendment pin. Native configuration equality keeps
shapes, rhos, optimizer options, flags and timing settings intact, excluding
only the new D decision field and A/B-unused learning/asset dictionaries.
The actual config patch adds only the D cutoff, apart from a comment.

Consequently this exact patch neither masks a changed sampler/timing/native
configuration nor upgrades a failed prerequisite. Original record fields and
source identity are retained, with continuity metadata added to the returned
view. Any later failed attempt still supersedes an older pass. Carryover
cannot authorize correctness, cost or learning stages. Current correctness
qualification must still run under the final binding.

## Material limitation of the general AST argument

**Scoped approval condition:** the 14-definition AST equality check is not a
complete transitive behavioral proof for arbitrary future driver edits.
Helpers such as `paired_ratio`, `freeze_delta`, `command`,
`remaining_timeout`, and `storage_guard`, plus module-level code, are outside
that comparison. For example, changing `paired_ratio` could alter inference
while all 14 listed definitions remain equal. An entry with refreshed exact
maps is therefore necessary evidence of explicit source approval, not a
substitute for reviewing the entire driver diff. This does not block the
present patch: independent full-definition comparison and direct diff review
confirm those omitted dependencies are unchanged here. Do not reuse this
review for changed driver hashes; renew the exact-diff review or broaden the
mechanical continuity proof first. Every refreshed continuity declaration
requires renewed exact-diff review, including the remaining helpers and
module-level code; the 14 AST checks are a scoped consistency check.

## Retained verification evidence

Inspected implementer output reports **186 passed, 2 MPS tests deselected in
4.08 seconds**; the parent's retained full-suite output reports **552 passed
in 10.87 seconds**. These commands were not rerun by this reviewer. Source
regressions cover exact approved carryover, bad identities, missing original
maps, current maps differing from actual files, protected source/native
config mutations, unsealed manifests, failed and later-failed prerequisites,
native-only protocol-test pinning, same-shape two-session strict cutoff,
and separate sensitivity eligibility. The retained unmocked Git continuity
audit confirms both original heads, original mapped blob hashes, 14 measured
ASTs and native configurations.

## SHA-256 identities

| Artifact | SHA-256 |
| --- | --- |
| `analysis/run_dw_sampling.py` | `ef2ac4894f1217f6fbf3ef844febf360aaea6448f6d4028ce9a0e42b11604059` |
| `configs/dw_sampling.yaml` | `fdfcadb9298c6a57057c0bb6d646c2b997f7298efbfdd301bd9016471238fe34` |
| `decisions/dw_sampling_reopening.md` | `14eefa9244952b59dcdad70b29365cec2d237ff9f3e15f09e3fef2aaf37fec32` |
| `tests/test_dw_sampling_protocol.py` | `9306b07f009ab8c83e82c278675867c012fabc3671db60a052a7fb6e3643532b` |
| `src/oar/sampled_linear.py` | `11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3` |
| `src/oar/timing.py` | `25351cd20abc14385dd1cb3e482c67f73619b1ead30900ca0150fe57072458e5` |
| `src/oar/hardware.py` | `4ba118d816eec9d255cfb3d7925416373119fe181f82529fb58a96347d8e5048` |
| `.superpowers/dw_sampling_reopening/cost-policy-package.txt` | `48328a41c12f68c809666466a30d02fc5f9430885a0eb3dbd33bc9487f758129` |
| `.superpowers/dw_sampling_reopening/cost-policy-real-continuity.json` | `205902a9863d83374984934bd10ab31404657ca2e8ba8825a572e224f0ca5021` |
| `.superpowers/dw_sampling_reopening/cost-policy-report.md` | `34a423207d9c1cfaf22be3c17b67fa154ed7fbf201ec5851dea03460777080a4` |
| `.superpowers/dw_sampling_reopening/cost-policy-green.log` | `cf8a8a5c4c5bacdcbe8a6dd7a2377e514218a4d7eefb8414516696335b47dadc` |
| This run's `prospective/full-tests-cost-policy.txt` | `476e33270b38d0c5c8412746064457288f9c00484b80d606b25300e5abb427ed` |
| This run's `prepare/attempt-0002/record.json` | `15ce3e2a98ab2a4275cc6a251539b76c3ff274f51d2d5cf3166600438d2cf961` |
| This run's `feasibility/attempt-0001/record.json` | `716ec10cfe1827b8e85f03deb5229484ff7c20b80d2c7178b1395c4b5acda506` |

Canonical parsed config identity enforced by `PROTOCOL`:
`8d0ce1920bdc06d0c02f94a0ca14a7a7e9e976060225820270bf8c1affbeca67`.
Run-relative paths above use `analysis/dw_sampling/20261031T120154Z-reopened`.
