<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Independent task2a model-core review

Recommendation: **fix the finite-evaluation gap before recording quality results**.
Otherwise the reviewed model, initialization, tying, optimizer construction, and
evaluation reduction match the model/evaluation sections of `model-brief.md`.

Scope: `src/oar/sampling_lm.py` lines 1–233 as initially supplied, the brief,
`model-report.md`, `test_byte_model_pending.py`, and the reviewed
`sampling_data.py` collation interface. Runner, recovery, scheduling, checkpoint
commit logic, production configuration enforcement, and hardware qualification
are outside this review. No source, test, or configuration files were changed.

## Finding

**Important — reject nonfinite scored losses before returning quality metrics.**
`src/oar/sampling_lm.py:220`–231 computes per-target cross-entropy, accumulates its
masked CPU FP64 values, then returns NLL/perplexity without a finite check. A tiny
CPU model with `final_ln.weight` filled with NaN returned both `nll=nan` and
`perplexity=nan` as an otherwise normal result (`valid_targets=8`). This can let a
caller record invalid quality, compare against a best checkpoint incorrectly, or
serialize nonstandard JSON values instead of aborting clearly.

Prefer checking finiteness of the masked CPU losses before accumulation, and
checking the final metric conversion for a clear finite-result failure. The
losses already reside on CPU, so this adds no device scalar synchronization.
Raise a descriptive exception and retain the existing `finally` mode restoration.
Add a tiny fixture demonstrating rejection and mode/sampler-state preservation.
A caller-owned check can satisfy the integration requirement only if every
quality-recording/checkpoint-selection call site validates finite NLL and
perplexity before using or writing them and that failure path is tested; this
core currently neither enforces nor documents that obligation. Very large finite
NLL also currently exposes the raw `math.exp` overflow rather than a descriptive
evaluation failure.

## Checks that passed

- Defaults are four pre-LN blocks, width512/heads8/FFN2048/context256/vocab259,
  exact GELU, zero dropout, affine LayerNorm eps1e-5, final LayerNorm, learned
  unscaled token-plus-position input, biased internal linears and bias-free head.
- Native SDPA receives the Boolean `[B,1,Q,K]` key-valid AND causal allow mask,
  dropout0 and `is_causal=False`; native default attention scaling is retained.
  Right-padded query rows retain valid prefix keys through the data interface.
- Dense FFNs retain native `nn.Linear.forward`; candidate subclasses wrap only
  FFN weights, with independent stable sampler IDs. No-grad evaluation bypasses
  the sampled operator. Dense/candidate tensors and forward results match, and
  unsampled gradients, including the tied embedding/head, match in the fixtures.
- Construction is explicitly CPU FP32. No-init constructors and constant
  LayerNorm/bias initialization consume no global random draws. One isolated
  `named_seed(base_seed, 'init')` stream fills token, position, then per-block
  QKV/attention-out/FFN-in/FFN-out weights with normal(0,.02). The head is tied
  without another draw. The exact Parameter tie is restored after conversion
  and state loading, including `assign=True`.
- Native AdamW receives each unique Parameter once. Matrices including the tied
  embedding use decay.01; bias/LayerNorm use zero. Betas(.9,.999), eps1e-8 and
  native variant flags match the brief; sampled aliases are rejected. Variant
  construction alone is correctly not claimed to establish hardware support.
- Evaluation uses eval/no-grad/no autocast, supplied source order, B16 by
  default, collation right-padding to256, and actual final partial B. Per-target
  FP32 CE moves to CPU before FP64 conversion and masking; one global target
  denominator avoids unequal-batch averaging. Prior root training mode is
  restored on both success and exceptions. Sampler states do not advance.

Tiny architecture overrides and evaluation batch overrides are deliberately
fixture-capable APIs. Production callers must enforce the frozen architecture
and B16; this review does not treat those documented fixture hooks as defects.

## Verification evidence

```
PYTHONPATH=src .venv/bin/python -m pytest -q .superpowers/dw_sampling_reopening/test_byte_model_pending.py -m 'not mps'
10 passed in 1.04s
```

One additional inline, tiny CPU-only reproduction constructed a one-block
width8/heads2/FFN16 candidate, generated one synthetic article through the data
interface, set `final_ln.weight` to NaN under no-grad, and called `evaluate`:

```
{'returned_nonfinite_nll': True, 'returned_nonfinite_perplexity': True,
 'training_mode_restored': True, 'valid_targets': 8}
```

No downloads, new dependencies, hardware measurements, large model construction,
training, or protected-corpus access were performed.

## Scoped fix re-review disposition

**PASS — the Important finite-evaluation finding is addressed.** Inspected the
updated `evaluate()` at `src/oar/sampling_lm.py:231`–249. It now rejects nonfinite
masked CPU FP64 scored losses before accumulation, rejects nonfinite aggregate
NLL, and converts overflowing or nonfinite perplexity into a descriptive
`RuntimeError`. The existing mode-restoration `finally` still runs for loss
failures; mode is already restored before the final-metric checks. There is no
new device scalar synchronization and the reduction/count/order logic is
unchanged.

The two new tiny CPU regressions verify NaN loss rejection with training-mode
and sampler-state preservation, and a finite NLL near1000 whose perplexity
overflows with a clear failure. Re-ran only the requested model scratch suite:

```
PYTHONPATH=src .venv/bin/python -m pytest -q .superpowers/dw_sampling_reopening/test_byte_model_pending.py -m 'not mps'
12 passed in 1.11s
```

No remaining defect was found within this fix scope. This is a scratch review
disposition, not a sealed final report; recovery functions appended below the
model/evaluation core remain outside this re-review. No source or tests were
edited by the reviewer.
