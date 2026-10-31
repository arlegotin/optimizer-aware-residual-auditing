<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# MPS transfer boundary correction

Status: DONE. The authorized host-visible MPS regression reproduced the Stage A
failure before the fix and passed afterwards. The focused CPU sampler suite
also passed. No test runners remain active; parent owns source freezing,
amendment v2, and repeating affected Stage A qualification. No cost work,
training, corpus access, Git mutation, or unrelated code change was performed.

## Evidence and root cause

The committed source identified by parent as `afa51cb` failed at
`importance_probabilities` during Stage A, preserved at
`analysis/dw_sampling/20261031T105213Z/prepare/attempt-0002/command-006.stderr.txt`.
The traceback identifies the combined device transfer/FP64 cast
`scores.to(device='cpu', dtype=torch.float64)` and `RuntimeError: Undefined type
Double`. The implementation combined two operations that the prospective
specification required to be separate; CPU-only tests did not expose this MPS
boundary defect. Existing native paths in `basis.py`, `audit.py`, `oracle.py`,
and `hardware.py` already use separate transfer/cast ordering.

A new marked regression was added first, then run on the actual visible MPS
device with fallback disabled. It failed at the same source line with the same
`Undefined type Double` exception, without being skipped. After this RED, only
the two device/cast boundaries were changed:

- Scores transfer in FP32 to CPU, then cast there to FP64:
  `scores.to(device='cpu').to(dtype=torch.float64)`.
- Inverse-square-root weights cast on CPU to training dtype, then transfer:
  `weights_cpu.to(dtype=x2.dtype).to(device=x2.device)`.

No sampling algorithm, probability calculation, normalization, budget, RNG,
gradient formula, or tolerance was changed. Production operands remain on the
training device; only scores, indices, and small inverse weights cross the
existing boundaries.

## Regression

`test_mps_score_transfer_fp64_normalization_weights_and_sampled_backward` is
marked `mps` and skips only when actual MPS is unavailable. It uses FP32 operands
with exactly representable normalized scores and independently hand-computed
probabilities `[.17,.32,.17,.32,.02]`. It verifies CPU FP64 normalized
probabilities and their existing bounds, the real gathered gradient with a
repeated index and CPU FP64 inverse weights, native forward, actual sampled
backward with exact dX/db and independently computed sampled dW, MPS FP32
outputs, finite gradients, and identical dedicated CPU RNG consumption.
Existing FP64 probability tolerance 1e-10/1e-12 and FP32 output tolerance
1e-4/1e-6 are unchanged.

## Commands and results

All commands ran from the repository root in fresh interpreters. Host-visible
MPS execution used the authorized `require_escalated` tool path. MPS and CPU
runs were sequential, with other parent tests/hardware paused.

RED, before implementation correction:

```
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_MPS_PREFER_METAL=0 .venv/bin/python -m pytest tests/test_sampled_linear.py -m mps -q -rA > .superpowers/dw_sampling/mps-boundary-red.log 2>&1
```

Exit 1: **1 failed, 100 deselected**, 1.58s pytest elapsed. Failure:
`RuntimeError: Undefined type Double` at the original combined scores transfer.

GREEN, after the two boundary corrections, same host-visible command with log
`.superpowers/dw_sampling/mps-boundary-green.log`:

Exit 0: **1 passed, 100 deselected**, 1.93s pytest elapsed. The MPS test ran;
there were no skips.

Focused CPU regression:

```
PYTHONPATH=src PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_MPS_PREFER_METAL=0 .venv/bin/python -m pytest tests/test_sampled_linear.py -m 'not mps' -q > .superpowers/dw_sampling/mps-boundary-cpu-green.log 2>&1
```

Exit 0: **100 passed, 1 deselected**, 1.91s pytest elapsed. Elapsed values are
verification records, not benchmark/performance evidence.

A read-only scoped diff confirmed only the two sampler boundary lines (plus
explanatory comments) and the new marked test changed: sampler 6 changed lines,
tests 39 added lines. No Git writes were made.

Final SHA256 from `shasum -a 256 src/oar/sampled_linear.py tests/test_sampled_linear.py`:

- sampler: `11398f9340a48b0dcabdc733c74dfa52b53663ea2005989f2ad7b53079827ca3`
- tests: `3ed4adbf6f62599172b0322374e84c7552f7fc4b46761f80057053469822eaa9`

The isolated regression establishes this boundary correction only. Prior Stage
A failure remains retained; parent must freeze the corrected sources and repeat
affected qualification before proceeding. No broader hardware qualification or
performance claim is made here.
