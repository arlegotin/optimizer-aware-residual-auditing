# Timing repair implementation report

Implemented only `analysis/run_dw_sampling.py` and
`tests/test_dw_sampling_protocol.py`, plus this report and CPU verification
logs in the assigned directory. Read the owner reopening and driver brief.
No sampler/historical source edits, Git mutation, subagents, corpus acquisition,
or hardware execution occurred.

## Driver changes and interfaces

- `choose_block_length(measure_for_count, initial, *, minimum_s=.5)` validates
  a positive integer initial count, finite positive target and durations. It
  increases counts without an update or iteration ceiling until an observed
  complete dense block reaches the target, returning every calibration
  observation. Tests require more than 100 updates and 121 observations.
- B initializes counts from `timing.minimum_block_s`, calibrates the initial
  dense fixture and selected native implementation against the same target,
  and retains all three raw timings at each count in `dense_pilot` and
  `selected_dense_pilot`. D starts at the corresponding B count and may
  increase it using only current dense timings. Rows record the original
  count, target, calibration observations and recalibration flag. Each arm
  then uses the same frozen count; candidate times never calibrate counts.
- `select_native_variant(results)` chooses the supported implementation with
  the smallest median of baseline-only raw pilot times. Unsupported numerical
  variants remain excluded. Invalid pilot times and actual numerical/device
  execution failures remain fatal. Native qualification retains raw pilot and
  warmup measurements, medians/means, declared order seed and executed order.
  Pilot order uses the declared named order stream and reverses odd sessions.
  Native variants are also persisted separately before pair work. The
  existing exact CPU/device step-counter comparison is unchanged.
- Warmup uses the existing 8–32-block harness. Its raw measurements and stable
  flags remain in pair files and records, while instability cannot prevent
  pairs, select a slower native implementation, replace a session or veto a
  shape. Both sessions follow the relevant-first configured shape order.
  There are exactly two fresh session worker invocations, no automatic
  replacement and no retry-until-positive collection.
- `freeze_delta(absolute_aa_errors, *, floor=0.0)` uses twice the linear 95th
  percentile, subject only to the declared nonnegative finite floor. There is
  no 5% floor or 15% rejection cap. B pools only the declared two middle
  shapes across both sessions while preserving A/A results for every shape.
  D persists the inherited frozen delta artifact before launching its first
  candidate worker. Qualification still requires the paired upper interval
  below `1-delta` for the same relevant shape in both sessions.
- `remaining_timeout(None)` returns `None`. Hardware stage workers and the
  separate offline diagnostic accept nullable deadlines; subprocess calls
  receive `timeout=None` under the revised configuration. Explicit finite
  deadline utility behavior and A's existing functional watchdog remain.
  `storage_guard` accepts an absent output ceiling while retaining the real
  free-disk reserve and explicit finite output-ceiling behavior.
- Forecasts use uncapped update counts, the configured block target and two
  sessions, retain candidate/component allowances and return JSON-safe
  `deadline_s=None` for an absent stage ceiling. They do not reject the
  reopened hardware stages. A's forecast now records the configured nullable
  output ceiling accurately.
- Added `decisions/dw_sampling_reopening.md` to `CORE`. Updated strict
  `PROTOCOL` to the canonical revised-config hash
  `f407dd2d636c078bd7e2f79112ea31b665636d42cf818d3f6f835b4a39ab2a80`.
  Protected configuration changes, unknown fields, duplicate keys and schema
  downgrade remain rejected. The algorithm, dtypes, numerical tolerances,
  raw pair/warmup persistence, setup charges, reset/stream evidence, native
  optimizer work, actual-shape primitive checks, sealing and asset gates are
  preserved.

## CPU tests and retained RED/GREEN

All commands used `.venv/bin/python` with `PYTHONPATH=src`. Device operations
were excluded using `-m 'not mps'`; CPU fixtures replace hardware allocation,
device synchronization and elapsed measurement only where necessary.

1. `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps'`
   — `driver-red.log`: **22 failed, 15 passed, 1 deselected** against the old
   implementation/config binding. Expected failures cover caps, invalid
   initial/target validation, diagnostic warmup, native selection, measured
   delta, nullable resources/deadlines and protected revised configuration.
2. Same command after the initial repair — `driver-green-initial.log`:
   **37 passed, 1 deselected**.
3. `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps' -k 'native_qualification'`
   — `driver-red-native.log`: **3 failed, 1 passed, 38 deselected**. This
   exposed broad unsupported-error matching hiding fatal gradient/device
   errors and coincident pilot orders in the two declared streams. Narrowed
   capability rejection and explicit odd-session reversal repaired these.
4. `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps' -k 'stage_uses or offline_diagnostics or native_qualification'`
   — `driver-red-freeze.log`: **2 failed, 7 passed, 36 deselected**. The
   assertion required the D frozen-delta artifact before worker execution;
   artifact creation now occurs before either candidate session.
5. `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps' -k 'cost_session_recalibrates'`
   — `driver-red-raw-calibration.log`: **4 failed, 43 deselected**, requiring
   all raw selected-dense calibration timings rather than only their minima.
6. Final verification:
   `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py tests/test_sampled_linear.py -m 'not mps'`
   — `driver-green.log`: **146 passed, 2 deselected in 3.17s**, exit 0.
7. `git diff --check -- analysis/run_dw_sampling.py tests/test_dw_sampling_protocol.py`
   — exit 0, no whitespace errors (read-only Git inspection).

The integration regressions exercise B and D, both session indices, all five
shapes, all arms and 30 pairs per arm. They check raw files, matched counts,
setup charging, diagnostic flags, retained noisy controls, deltas below .05
and above .15, exactly two worker requests, frozen thresholds before D and
the nullable-deadline offline diagnostic with real CPU native optimizer and
sampler operations. The exact native MPS counter regression remains marked
`mps` and unchanged.

## Handoff limits

No hardware qualification or performance claim follows from these CPU tests.
The full original suite was deliberately not run, per the assigned brief's
serial measurement schedule; parent owns that later check. Parent still owns
amendment/source freeze, review and all later actual-device execution. The
existing amendment has not been rewritten by this task and must be rebound
before a driver stage can use the edited sources. Language work remains
asset-gated.

## SHA-256 at handoff

```
95fcdc418b000afd5a405bfb42076b76ef86b31ffe34c94abced68cbee69e22d  analysis/run_dw_sampling.py
202d7ee25cc06f44e8d3c48895d17719b99bc4802ab34e50a40d6cd9f6081ab6  tests/test_dw_sampling_protocol.py
f3ca918c8feb299ccc595169c15b1e7dc8800be105c1dbda154f8ac8752f0c9f  configs/dw_sampling.yaml
e494ab971dfdba55b8673242cc149e7b8180d86bb1b8452037c71652999c1618  decisions/dw_sampling_reopening.md
11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3  src/oar/sampled_linear.py
25351cd20abc14385dd1cb3e482c67f73619b1ead30900ca0150fe57072458e5  src/oar/timing.py
```
