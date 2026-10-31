# Native prerequisite continuity and sampled cost-policy report

Implemented only the assigned driver and protocol-test edits, plus verification
logs and this report. No measured function/class was edited. No hardware,
candidate timing, corpus work, Git mutation or subagents were used.

## Cost gate

`hardware_stage` keeps B feasibility's `upper95% < 1-delta` requirement in both
sessions. D sampled cost instead uses the newly declared
`timing.cost_gate_upper_ratio` (1.0), again requiring the same shape/arm to pass
in both sessions. The result reports:

- `eligible`: shape/arm combinations satisfying the stage's actual gate.
- `gate_upper_ratio`: that gate's cutoff.
- `noise_guard_upper_ratio`: the prior `1-delta` cutoff.
- `noise_guard_eligible`: combinations satisfying that prior stricter screen,
  retained descriptively for D.

The A/A delta and its original source data remain intact. D still writes the
inherited frozen-delta artifact before either candidate worker starts. No
warmup, block-count, pairing, native-optimizer or measurement code changed.
The current strict canonical configuration hash is
`8d0ce1920bdc06d0c02f94a0ca14a7a7e9e976060225820270bf8c1affbeca67`.

## Explicit amendment entries

The new optional amendment field is a list of exact declarations:

```json
{
  "prerequisite_carryover": [
    {
      "stage": "prepare",
      "record_sha256": "SHA-256 of the exact original sealed record.json",
      "previous_measurement_hashes": {"entire original record map": "unchanged"},
      "current_measurement_hashes": {"entire actual consumer map": "current hashes"}
    }
  ]
}
```

An additional entry with `stage: feasibility` pins B. Separate entries can pin
different complete consumer maps for one prerequisite. `source_hashes('cost')`
and `source_hashes('correctness')` currently produce the same full map; the
native-only feasibility map excludes sampled-source/test dependencies.

`require_stage` retains the latest-attempt rule and requires original PASS
before consulting any carryover declaration. With a source/config mismatch it
calls `prerequisite_carryover`; absent an exact matching entry, the mismatch
still rejects. No older pass can hide a newer failed attempt. Carryover is
allowed only for `prepare` and `feasibility`, never a prior correctness/cost or
learning stage.

The helper verifies:

1. Exact stage, sealed record SHA-256, complete previous measurement map and
   complete current consumer map. The supplied current map must equal actual
   files, even if the caller and declaration agree on a forged hash.
2. A sealed original source manifest, unchanged original measurement map and
   full original Git head. `git_blob` reads bytes using actual `git show` at
   that head, verifies every original mapped source hash, and checks all
   current known dependencies against the original Git versions.
3. Changed source keys are limited to driver, config, reopening scope and
   protocol tests. Sampler, sampler tests, hardware, timing and original scope
   must match the original Git blobs. If a native-only consumer omits the
   changed protocol test file from its own map, its current hash must still
   be explicitly pinned by `amendment.bound_files`.
4. AST equality for exactly `configure_torch`, `Fixture`, `zero_dw_linear`,
   `native_variant_qualification`, `timed_pairs`, `state_streams`,
   `cost_session`, `select_native_variant`, `assert_native_equivalent`,
   `choose_block_length`, `primitive_qualification`, `environment`,
   `dependency_inventory` and `prepare`. Metadata validators and the D gate
   are outside these measured definitions.
5. Canonical native configuration equality after removing only
   `timing.cost_gate_upper_ratio`, `learning` and `asset`. The two dictionaries
   are a fixed exclusion because native A/B do not consume them. This neither
   licenses data/model work nor qualifies a learning result; those still need
   their own manifest, freeze and gates. Every shape, rho, optimizer setting,
   device flag and other timing/config field must remain identical. The actual
   current v4-to-v5 config changes only the cost-gate field.

The returned value preserves every original record field and its original
`measurement_hashes` and manifest hash, adding a `carryover` object containing
the original record path/hash/head, old/current maps, verified definition names
and native config exclusions. Original records, seals, artifacts and source
identities are never rewritten or relabeled.

## Test-first and verification evidence

All test runs used `.venv/bin/python`, `PYTHONPATH=src`, and excluded MPS.

- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps'`
  — `cost-policy-red.log`: **19 failed, 56 passed, 1 deselected** before the
  policy/continuity implementation. Failures covered the newly protected
  config, exact approved carryover, revised D policy and sensitivity outputs.
- `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py -m 'not mps' -k 'unconsumed_model or complete_original or native_consumer_carryover'`
  — `cost-policy-red-dependencies.log`: **3 failed, 2 passed, 82 deselected**
  with model/asset exclusion behavior removed and original-map completeness
  not yet implemented. These regressions require the narrow unconsumed
  exclusions and reject missing original driver/config bindings.
- Initial combined CPU verification, before the last completeness cases:
  `cost-policy-green-initial.log`: **182 passed, 2 deselected**.
- Final command:
  `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dw_sampling_protocol.py tests/test_sampled_linear.py -m 'not mps'`
  — `cost-policy-green.log`: **186 passed, 2 deselected in 4.08s**, exit 0.
- `git diff --check -- analysis/run_dw_sampling.py tests/test_dw_sampling_protocol.py`
  — exit 0, no whitespace errors; read-only inspection.

New cases exercise approved A/B carryover while retaining original attribution;
missing/wrong declarations, record/stage/old-map/current-map mismatches;
protected sampler/hardware/timing/original-scope changes; changed measured
function ASTs; changed native config; original Git bytes inconsistent with
the sealed map; failed/latest failed prerequisites; source-manifest tampering;
incomplete original maps; protocol-test pinning for a native-only consumer;
and changed shape/rho/optimizer/flags/timing settings even when a declaration
pins their new hashes. D cases include a gain below 1 but above `1-delta`,
one failed session, and an interval exactly at 1, preserving strict inequality.

`cost-policy-real-continuity.json` additionally records a successful read-only
check using **unmocked actual Git blobs and actual sealed A/B artifacts**:

| Prerequisite | Record SHA-256 |
| --- | --- |
| prepare/attempt-0002 | `15ce3e2a98ab2a4275cc6a251539b76c3ff274f51d2d5cf3166600438d2cf961` |
| feasibility/attempt-0001 | `716ec10cfe1827b8e85f03deb5229484ff7c20b80d2c7178b1395c4b5acda506` |

Both original heads are `d7782862c025da6ee27dc9f7692b75fe926cb070`. All original
Git hashes match their sealed measurement maps; all 14 measured ASTs and native
configs are equal. The only changed keys are exactly the four allowed source
paths. This check does not create an approval entry; parent must still pin
the exact declaration and current source hashes in the new amendment.

## Handoff

The full original suite and actual hardware were deliberately left to parent
for serial scheduling. Existing MPS counter tests remain marked and unchanged.
Parent owns amendment entry creation, the new source freeze, fresh current
correctness checks and any later D execution. No sampled performance result is
claimed by this task.

Final file SHA-256 values:

```
ef2ac4894f1217f6fbf3ef844febf360aaea6448f6d4028ce9a0e42b11604059  analysis/run_dw_sampling.py
9306b07f009ab8c83e82c278675867c012fabc3671db60a052a7fb6e3643532b  tests/test_dw_sampling_protocol.py
fdfcadb9298c6a57057c0bb6d646c2b997f7298efbfdd301bd9016471238fe34  configs/dw_sampling.yaml
14eefa9244952b59dcdad70b29365cec2d237ff9f3e15f09e3fef2aaf37fec32  decisions/dw_sampling_reopening.md
```
