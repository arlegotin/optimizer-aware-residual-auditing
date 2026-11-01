<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Independent sealed-outcome mathematics/statistics review

2026-11-01. **The sealed E4 STOP_QUALITY is reproduced. No critical or important
outcome/selection defect identified.** This narrow addendum preserves the earlier
pre-outcome algebra review. It does not authorize new training or protected losses.

## Evidence identity and method

Immutable review BASE: `a3e3229768dd7186f628419917ed746d9e0dcf7b`.
Public attempt: `analysis/dw_sampling/20261031T120154Z-reopened/pilot/attempt-0004`.
Record SHA256: `3e15fe5e4526ae001b5112a896b69d43669a2044aa60f4febc7c5ee05125f7bd`.
Source-manifest SHA256: `dca2ab2acbf5a07ee4392001a93eb46a41bb305802a572f9a6c19c1c57201e11`.

Using independent standard-library calculations, I authenticated the record/seal,
all 329 sealed public metadata artifacts, and exact BASE copies of the record,
source manifest and grid results. All 11 measured-file hashes agree at BASE,
measured source `89c5bd8f459ec245de3ea83e71a7a5971b50c028`, and recorded launch HEAD
`43c2cec0b3d2fb2593966c1a497de92b810a36d7`. The v9 frozen manifest, configuration,
amendment and 23 bound review/continuity/source-freeze proof hashes also agree
with immutable Git bytes. Hash authentication does not newly prove every earlier
review's reasoning; the previous scoped reviews remain its authority.

For all 27 declared grid entries, I matched frozen ID/arm/seed/roles, raw worker
result, aggregate and per-row metadata, request/launch/terminal hash links,
canonical active charge and actual periodic-monitor events. Every arm has one
fresh attempt-0001, no resume, terminal classification COMPLETE/exit 0, raw worker
status pass, finite endpoint and valid budget/overshoot. All share seed 3101 and one initial
parameter hash. The single missed initial deadline is retained; no fabricated
catch-up monitor enters selection. Root's separate 184-file historical integrity
check was not repeated here.

## Independent reconstruction

The frozen rules are `BASE:analysis/run_dw_sampling.py:1370-1433`; the early stop
before independent development/precision is at `3149-3162`.

- H = 1461.4585008954164s; T_budget = 1490.8579807295464s;
  r_time = 31.627891642889008s; epsilon_eval = 1e-6.
- R0 is `grid-ce2496e36ca2e84c`, native FP32 B8/LR3e-4, the lowest full-development
  endpoint: 1.231552285198124 NLL. Its initial NLL is 5.7851592537702325, so
  gain = 4.553606968572108 and epsilon_q = min(log(1.01), .1*gain)
  = 0.009950330853168092. The endpoint ceiling is 1.241502616051292.
- R0's first monitor at/after .6H is deadline 30, completed at 877.2205836670473s,
  update 7128, with q_star = 1.3082633545763616. Two *later* consecutive crossings
  complete at 907.1045402500313s and 935.8225742500508s, validating target construction.
  R0's target time is 907.1045402500313s: the calibration monitor itself and the
  next monitor are the first two consecutive crossings in its full trajectory.
  The later-only target-validation time and full-trajectory target time serve
  different rules and correctly differ.
- Only native B4/LR3e-4 and B8/LR3e-4 meet both endpoint and observed-target
  requirements. B8 is faster by 87.229036s, exceeding r_time, and is selected.
  No sampled arm meets the endpoint bound, and no sampled monitor even crosses
  q_star once before H. Thus all 15 sampled target times are censored, the eligible
  candidate list is empty, candidate is null, and STOP_QUALITY follows directly.

Each row below is the lowest endpoint within its declared batch/fraction grid,
for descriptive coverage of every family. Full results for all 27 trials, all LR
values, endpoint differences, censoring and available time ratios are retained
in `math-sealed-outcome-check.json` alongside this review.

| Arm | Best endpoint LR | Endpoint NLL | Perplexity / R0 | Observed target time (s) |
| --- | --- | --- | --- | --- |
| native B4 | 0.0003 | 1.233644900 | 1.002095 | 994.333577 |
| native B8 | 0.0003 | 1.231552285 | 1.000000 | 907.104540 |
| native B16 | 0.0003 | 1.255988721 | 1.024737 | 1052.879617 |
| native B32 | 0.001 | 1.255889577 | 1.024636 | 1083.631996 |
| importance rho=0.125 | 0.0003 | 1.378494670 | 1.158287 | censored |
| importance rho=0.25 | 0.0003 | 1.344515038 | 1.119590 | censored |
| uniform rho=0.125 | 0.0003 | 1.407013047 | 1.191795 | censored |
| uniform rho=0.25 | 0.0003 | 1.377663540 | 1.157325 | censored |
| uniform rho=0.5 | 0.0003 | 1.319570568 | 1.092008 | censored |

The best importance and uniform endpoints are respectively 0.112962753 and
0.088018283 NLL worse than R0 (perplexity ratios 1.119590 and 1.092008). Relative to
the best native B16 endpoint, their perplexity ratios are 1.092563 and 1.065647.
These comparisons are descriptive selected-grid observations. Five native arms
have observed target times; seven native and all 15 sampled arms are censored.
Only two of those five native arms also meet the endpoint guard. No finite
sampled/native target-time ratio, H substitution, subset ratio or seed-level
confidence interval is justified.

## Actual cost and inference limits

The 27 canonical trial charges sum to 39865.108167834s (11.073641h). The enclosing E4
invocation charges 39871.41611858306s (11.075393h), including 6.307950749s additional
parent orchestration. All 27 trials are charged: 12 native, six importance and
nine uniform. They total 214584 optimizer updates and 643264263 valid training
targets. Trial charges range from 1474.662605–1480.793650s; each remains within T_budget.

The four recorded pilot invocations total 41000.67397879122s (11.389076h), including
the 70.989846s earlier qualification failure, 1014.102614s qualification, and 44.165400s
startup-only v8 failure. That prior generic STOP_QUALITY is not a learning outcome.
The 2.3640637500211596s preprocessing recorded within pilot/attempt-0002 is already
included in its 1014.1026140000904s outer invocation and in this pilot subtotal;
it is a nested charge and must not be added again. The subtotal excludes A/B/D,
tests, engineering diagnostics, and asset/documentary/review/setup work performed
outside these four invocation clocks; it must not be presented as total project
expenditure.

The 27 alternatives share one tuning seed; they are not 27 independent training
seeds. Tuning-selected family minima and monitor measurements do not support
paired seed inference, broad superiority/inferiority claims, or estimates of
confirmation power. The supported finding is that no declared sampled grid arm
qualified on seed 3101 at this frozen budget. Earlier repeated step-cost savings
remain cost evidence and do not establish a learning benefit. Independent seeds
3102–3104, precision/simpler-control reproduction, fixed-N confirmation,
official-validation losses, final-test losses and corroborating repeats were not
reached by this stopped path. No protected inference is silently inferred.

## Assessment and remaining boundary

No source correction or scientific retuning is requested. Preserve STOP_QUALITY
and every adverse/censored trial. A final report-consistency addendum remains
pending because the final report has not yet been written.

Executed only standard-library public-metadata analysis and immutable Git reads;
no project module imports, Torch, tests, hardware work, collectors, network,
corpus bodies or checkpoint tensor bodies. The only writes were this ignored
report and its permitted JSON facts file. Broader algebra/backend/source reviews
were not repeated, and no new learning claim is made.

## Draft report-consistency addendum — 2026-11-01

Reviewed `scientific_report-draft-v1.md`, SHA256
`9b83aaaabd3a80f118dcb187f0e9bcba9dd93a1bf1e768660859f4b76c8716c4`.
**Scientific wording and checked numbers agree; no critical or important
report-consistency finding.** Its STOP_QUALITY, all 15 sampled guard failures and
censored targets, R0/target/quality thresholds, nine-row family table, native
selection, actual pilot charges and one-seed inference limits match the sealed
reconstruction. The stated time resolution rounds correctly to 3.49% of R0's
observed target time. The draft keeps cost savings separate from learning benefit,
discloses unreached stages/protected outcomes, and labels software closeout pending.

Compared all 27 rows of the separate post-hoc metadata analysis with this review's
independent facts. Recomputed the displayed update/token throughputs and clipping
fractions directly from public raw step metadata. They agree at the stated
rounding. The clipping summaries count recorded preclip norm >1; the final-quarter
bin uses block completion >=.75H. The draft appropriately treats these summaries
as exploratory observations at differing parameters/data progress, without causal
attribution or permission to retune. The analysis code hash agrees with its recorded
output (`recorded-grid-analysis.json` SHA256
`a0b6a66bbb34dc34c5bd05821ae16382d8406de50066e62fd1ba5716e4919e7a`).

Read the metadata replay and analysis source without importing/executing either.
The replay compares scientific JSON fields after excluding its measured publication
time and compares both CSV byte hashes; it explicitly disclaims training repetition.
This review does not claim to have run that replay or independently verified corpus
interpretation, private checkpoints or pending software checks.

Only this ignored report was changed for the addendum. No project imports,
training, hardware, network, corpus/checkpoint bodies or source edits were used.
The eventual final report still needs a narrow final-diff check if its mathematical
or scientific wording changes materially; pending software/version/test/review
details are outside this draft's scientific approval.

Accounting wording correction before archive: the earlier broad exclusion of
“separately charged preprocessing” was imprecise. Immutable BASE calls
`prepare_language_data` inside `pilot_qualification` (`run_dw_sampling.py:2032`),
after the outer invocation clock starts (`3264`) and before its wall charge is
recorded (`3315`). The measured qualification source at `110be773` has the same
inclusion. Attempt-0002 records the nested 2.3640637500211596s explicitly. All
estimates remain unchanged; the final report should distinguish this included
nested work from genuinely additional work outside stage clocks. This narrowly
qualifies the draft addendum's accounting approval; it does not change STOP_QUALITY
or any scientific inference.
