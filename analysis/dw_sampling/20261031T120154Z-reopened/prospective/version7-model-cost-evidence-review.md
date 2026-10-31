<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Independent full-model cost evidence review

2026-10-31. Scope: completed qualification evidence and its interim report,
not the currently changing, unconsumed learning dispatcher. Inspected measured
source from immutable Git `110be7739067eec0aaddef40e21763818691fea8`; did not
attribute current working-tree source to those observations. No hardware,
network, rebenchmark, corpus loading, Git mutation, or source edits. Small CPU
recomputations used the measured mathematical helper definitions and retained
scalar timing records. Checkpoints were hashed as opaque bytes.

**Disposition: evidence and exploratory cost gate PASS. V1's one wording
finding is resolved by append-only v2, bound below. No learning benefit or
completed pilot is established.** The enclosing `not_run`, nested qualification `pass`, and
`learning_success: false` states are consistent and correctly disclosed.

## Identities and integrity

| Reviewed object | SHA256 |
| --- | --- |
| Interim report Markdown | `4f2bc00296d97fd2dfc060ab2ca830c3e3ef5c8637d14ffe908f6bddfccc1824` |
| Interim JSON manifest | `805ef7a413c0261fe20393af300968de7f09cc8247bae488d3a2b5bb8bea5997` |
| Pilot attempt0002 record | `d01c854daf8eb7f7fbc8505b0879b40bc869bfc469a141fa727ae086757c4209` |
| Attempt0002 source manifest | `9b6c136a8da76e19ddffef57874ad9063fa0e7231e5628754fa258be6ff586e8` |
| Measured driver | `ccd8490f6048905ffa5f48864143ab1df6e12ed77af146a7877b47ab2230d5bd` |

The interim manifest correctly binds the report. Attempt0002's record seal and
all 114 artifact hashes pass. Its eleven measurement-source hashes each match
their Git110be77 blobs. The seals and every mapped artifact also pass across
all eleven retained prepare/feasibility/correctness/cost/pilot records,
including failed pilot0001. This verifies retained bytes, not an independent
rerun or proof that no unrecorded work occurred. Original-184 continuity and
the eventual new dispatcher's full diff remain outside this task.

## Recomputed results

All ten retained raw JSONL timing lists exactly match their sealed aggregate
pair lists. Recomputed setup-inclusive geometric ratios and 10,000-replicate
paired percentile intervals with seed62001 equal the stored statistics and
the report's rounded table. Every arm has thirty pairs in each of two fresh
sessions; all five arms pass the same-arm upper95%<1 requirement twice. Fused
is independently reproduced as the pooled native-pilot choice; all eight
variant/session qualification entries report five identical-current-gradient
checks passed. No slower arm was dropped from the table.

Full-model A/A delta recomputes as .1854212253056062. The frozen maximum with
prior D delta is .2085046466766194, and both candidate requests bind that exact
frozen artifact. Thus the old threshold is .7914953533233806; no full-model
arm passes it. The report correctly attributes continuation to the prospective
reopened <1 gate, rather than claiming the earlier sensitivity gate passed.

The 429-row calibration ledger reproduces exactly: six fixed-fused native
variant pilot blocks, 120 A/A sides, 300 native sides of candidate comparisons,
and three cold controller blocks. Its largest included per-update value is
.698062084033154/2=.349031042016577, from session2's uniform1/2 comparison's
native side; Bmax=2.3987216249806806 is cold worker3's controller block.
Taking the recorded cold maxima reproduces H=1461.4585008954164,
R=29.399479834130034, T_budget=1490.8579807295464, j=50,
Delta=29.229170017908327, r_time=31.627891642889008 and epsilon_eval=1e-6.
The j=100 trial fails Bmax<=Delta/10; j=50 passes overhead and block resolution.
The 27-run grid and 52+2N forecast reproduce: N=8 gives 68 equivalents and
28.1606507471 nominal device-hours, before separately charged work/deduplication.

## One report clarification

**Minor, fix before presenting the report as final:** line25 says the native
update bound uses “every qualified fixed-native observation.” The actual
prospectively implemented ledger uses the four categories above and excludes
retained warmup and block-calibration pilot observations. For example,
`model-block8-AA-1.json` contains a selected-dense pilot raw observation of
.4587729999329895 seconds for one update, above the quoted .349031042 bound.
That observation is an early block-length sizing trial, rather than a warmed
formal timing block; its exclusion follows the measured function's frozen
category rule, not a new classification based on its observed value.
Name the actual ledger categories, rather than implying a maximum over every
recorded native duration. This wording issue does not change the recorded
ledger, H, raw cost observations, or exploratory gate. Whether a future horizon
policy should include other categories is a separate prospective decision;
do not silently recalculate the frozen H after learning outcomes.

## Interpretation limits

The report explicitly withholds a language-learning claim and distinguishes
timing blocks from independent training seeds. This is appropriate. Uniform1/8
has the lowest measured full-step ratios in both sessions; endpoint quality,
time to a fixed target and conventional competitors remain untested here.
Paired resampling preserves each pair but assumes exchangeable pairs; it does
not protect against serial host/thermal/allocator effects. As one illustration,
session2 importance1/4's raw log-ratio lag1 correlation is about .318, estimated
from only thirty pairs. It is not proof of invalidity or a correction factor,
but reinforces the report's serial-dependence caution. Exploratory comparisons,
fixed trajectory replay, one host, and geometric-ratio rather than total-time
inference limit generalization. The measured gains justify the specified next
study, not a guarantee of useful learning speedup.

## Append-only v2 report binding

The parent preserved the reviewed v1 bytes and supplied append-only v2. A
prose-only diff/hash check verified:

| Final interim object | SHA256 |
| --- | --- |
| `model_cost_interim_report-v2.md` | `4b18793f698b448f113cf951f360f734902b9c87ccbd000bf1ad75dd7500c8f5` |
| `model_cost_interim_manifest-v2.json` | `01657034ebbe673435e4e3b004fc2af531563b875a395a1daae5ecd40851ea03` |

V2 names the prespecified 429-row categories, distinguishes the retained and
charged early sizing/warmup measurements, explicitly reports no old-threshold
survivor, and strengthens the exchangeability/serial-dependence disclosure.
Its source record, results and H are unchanged; the v2 manifest binds the
correct report hash. **The minor wording finding is resolved; v2 is ready as
an interim exploratory cost report.** No gate or raw-data recomputation was
needed for this prose-only revision. All earlier learning-claim limitations
and deferred implementation/continuity reviews remain in force.
