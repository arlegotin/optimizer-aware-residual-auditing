<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Prospective timing repair review

Review date: 2026-10-31. Reviewed the fixed package against
`40fd56fa71ec80bb0d28aa2e49ab8e5b2a13fef4`, the owner-authorized reopening
scope, actual driver/config/protocol tests, and the unchanged sampler, timing
and hardware boundaries. This review performed source reads and read-only
hash checks only: no tests, benchmarks, corpus use, source edits or subagents.

**Verdict: ready to freeze and attempt fresh exploratory hardware qualification
and measurements. No blocking material finding.** Mathematical and backend
structure satisfy the revised timing scope. This is not a performance result,
an actual-device qualification by this reviewer, a confirmation, or permission
to bypass the separate corpus gate. Parent must freeze and commit the new
amendment/source/config before comparative measurements. The currently stale
amendment is expected before that step, not a repair defect.

## Material checks

- **Uncapped baseline calibration:** `choose_block_length` validates positive
  integer counts and finite positive durations, increases counts without an
  iteration/update ceiling, and stops only at the configured 0.5-second
  minimum. B calibrates the initial dense fixture and selected native variant;
  D starts from B's count and can increase it using only its own dense pilot.
  All three calibration timings at each visited count are retained on normal
  completion. The selected count is fixed before any candidate timing for the
  shape, and every arm uses that same count. Calibration does not inspect
  candidate speed or remove slow paired observations.
- **Warmup:** the existing 8–32-block harness and raw observations remain.
  Neither native selection nor paired collection nor stage eligibility
  branches on its stable flag. Instability is reported as a diagnostic and
  cannot trigger a replacement or silently exclude a shape.
- **Native baseline:** all four declared default/single/foreach/fused variants
  undergo multistep parameter/moment comparison with scalar native AdamW.
  Step counters compare exact values on CPU, permitting legitimate storage
  differences. Selection uses the lowest median of three baseline-only pilot
  times among qualified variants. Unsupported or numerically disagreeing
  implementations are recorded; actual gradient/device/allocation failures
  propagate rather than select a slower baseline. Seeded pilot order and
  observations remain visible. D inherits the per-session selected variant.
  This is the strongest observed supported pilot selection, not proof of a
  population-wide speed optimum from three observations.
- **Complete work and shapes:** both sessions retain middle4096, middle2048,
  the four-microbatch stride-2 control, wide and tiny in relevant-first order.
  A control cannot veto an otherwise eligible relevant shape. All six sampled
  arms and both feasibility arms remain. The fixture retains exact native
  forward, root input/bias adjoints, GELU, loss, full clipping and native AdamW
  moments/decay. No optimizer shortcut or dense sampling instrumentation was
  introduced. Sampler checks, score transfer, categorical draws, gather,
  allocations and synchronization remain within complete step measurements;
  sampler generator setup is included through explicit amortization in the
  eligibility statistic. Reset/cloning costs and separate component profiles
  remain recorded with their declared experimental boundaries.
- **Noise threshold and decision math:** B computes
  `delta = 2 * linear_quantile95(abs(TA/TB - 1))` from exactly the two declared
  relevant shapes across both sessions (120 A/A pairs). The configured floor
  is zero and there is no delta cap. Other shapes' A/A results remain visible.
  D inherits this value and writes its frozen artifact before launching its
  first candidate worker. Each arm's statistic is the geometric paired
  candidate/dense ratio, with setup charged and 10,000 seeded paired bootstrap
  resamples. Eligibility requires the upper interval strictly below
  `1 - delta` for the same relevant shape in both sessions. A delta above one
  naturally makes this criterion unattainable; the implementation does not
  truncate it or redraw. These intervals are exploratory block uncertainty,
  without familywise or independent-training-seed claims.
- **Resources and failed observations:** null stage deadlines reach workers
  and subprocesses as `None`; null output ceilings do not disable the actual
  disk reserve. Memory screening, native allocator safeguards, finite-gradient
  checks, ownership lock and competing-process screening remain. Forecasts
  are advisory under the frozen null stage ceiling. A retains its separate
  functional qualification watchdog. Exactly two fresh timing workers are
  requested on success; any failure returns the retained prefix and stops
  that attempt. There is no automatic replacement, retry-after-bad-draw,
  outcome-dependent collection extension or slow-block filter. A repaired
  follow-up attempt still needs a prospective reason and retained predecessor.
- **Bindings and history:** reopening scope is included in `CORE`; strict
  configuration hashing, worker source validation, latest-prerequisite
  checks, immutable attempt allocation and artifact seals remain. Read-only
  comparison against the reopening's 372-file initial inventory found no
  missing files and changes only to `analysis/run_dw_sampling.py`,
  `configs/dw_sampling.yaml`, and `tests/test_dw_sampling_protocol.py`.
  Historical stopped evidence and the sampled implementation were unchanged.

## Verification evidence and limits

The retained parent full-suite output reports **512 passed in 9.19 seconds**.
The retained implementer CPU output reports **146 passed, 2 deselected in
3.17 seconds**. I inspected those logs and the meaningful source regressions:
counts beyond 100 and 120 calibration iterations; unstable warmup with pairs
and setup still charged; fastest native median despite warmup diagnostics;
fatal gradient/device failures; both session indices, all shapes and arms;
target-only noise pooling; delta below .05 and above .15; pre-worker delta
freeze; null resources/deadlines; and exact CPU/device native counters.
I did not rerun these commands. New actual-shape MPS qualification and B/D raw
evidence remain to be produced under the final source binding. External GPU
activity and unavailable thermal telemetry remain declared host limitations.

No change to the sampler's conditional-unbiasedness/variance mathematics was
made by this repair. Repeated paired blocks reset to the declared common
state and stream; they are timing blocks, not independent learning runs.
Longer blocks and diagnostic-only warmup improve the measurement opportunity
without making any learning or positive-speed claim by themselves.

## Inspected SHA-256 identities

| File | SHA-256 |
| --- | --- |
| `.superpowers/dw_sampling_reopening/review_package.txt` | `b6e25240ef59bf46d72099ad79087d5041c3e4518a9d69c0898f8867fd011601` |
| `.superpowers/dw_sampling_reopening/driver-report.md` | `0ebef75cea9c5daddbc5a41cb13b3aa18902276c7843e90b29878ce811378250` |
| `analysis/run_dw_sampling.py` | `95fcdc418b000afd5a405bfb42076b76ef86b31ffe34c94abced68cbee69e22d` |
| `configs/dw_sampling.yaml` | `f3ca918c8feb299ccc595169c15b1e7dc8800be105c1dbda154f8ac8752f0c9f` |
| `tests/test_dw_sampling_protocol.py` | `202d7ee25cc06f44e8d3c48895d17719b99bc4802ab34e50a40d6cd9f6081ab6` |
| `decisions/dw_sampling_reopening.md` | `e494ab971dfdba55b8673242cc149e7b8180d86bb1b8452037c71652999c1618` |
| `decisions/dw_sampling_scope.md` | `d91d9350dc117f83e39c648e48d435de75ab9ea5a3b6bc197b216c4b73c11181` |
| `src/oar/sampled_linear.py` | `11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3` |
| `src/oar/timing.py` | `25351cd20abc14385dd1cb3e482c67f73619b1ead30900ca0150fe57072458e5` |
| `src/oar/hardware.py` | `4ba118d816eec9d255cfb3d7925416373119fe181f82529fb58a96347d8e5048` |
| `prospective/full-tests-before-freeze.txt` (this run root) | `34518420c6b39d1fac917f312b3c84477c5c56c3eae62214673ae86946143c93` |
| `.superpowers/dw_sampling_reopening/driver-green.log` | `56b27802803293d62c65ec97e4a2842173800a9cd6a40c63ab37aaee314771e9` |
| `initial_inspection.json` (this run root) | `1958a0004115bd34164d297c38e6b911a11001d208ca48322adcfc1c0e6cf90c` |

The canonical configuration identity enforced by `PROTOCOL` is
`f407dd2d636c078bd7e2f79112ea31b665636d42cf818d3f6f835b4a39ab2a80`;
it hashes canonical parsed configuration, distinct from the YAML byte hash.
