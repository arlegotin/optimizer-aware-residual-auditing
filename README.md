# Optimizer-aware residual auditing

**Research into cheaper weight gradients while keeping the full forward context.**

Training a Transformer linear layer involves three large products: the forward
pass, the input gradient, and the weight gradient. This repository tries to make
the third cheaper. Every token still contributes to context; raw input and bias
gradients stay exact.

It contains the PyTorch operators, native AdamW comparisons, a small byte-language
model, and the complete evidence from testing the idea. **No training speedup has
been demonstrated yet.** The implementation works; the tested approximations
haven't beaten ordinary training.

## How it works

**Original OAR:** build a low-rank weight gradient, preview the AdamW update, then
use fresh residual probes to audit its alignment. AdamW rescales coordinates using
its moment history, so a gradient that looks close can still produce a poor update.
The audit tests the optimizer's step rather than trusting gradient similarity.

**Fixed-budget sampling:** choose paired rows of activations and incoming
gradients, reweight their contributions, and compute a smaller weight-gradient
product. Importance sampling favors rows with larger norm products. A uniform
control chooses rows directly. Both use native AdamW; neither approximates the
input gradient or discards forward context.

The sampler uses established column-row sampling ideas from
[Adelman et al.](https://arxiv.org/abs/1805.08079) and
[Wang et al.](https://arxiv.org/abs/2402.17227). It keeps fixed ratios and exact
input adjoints; it doesn't reproduce VCAS's adaptive method.

## Results

On an **M3 Max, FP32 MPS**, some sampled arms cut complete-step benchmark time
by roughly **7–17%**. That saving didn't turn into a training win.

The language study ran **27 configurations** of a four-block byte Transformer on
WikiText-2, with one tuning seed and the same roughly 24-minute optimization
budget. Lower development NLL is better; these are the best endpoints per family:

| Method | Development NLL ↓ |
| --- | ---: |
| Native dense AdamW | **1.2316** |
| Uniform weight-gradient sampling | 1.3196 |
| Importance weight-gradient sampling | 1.3445 |

Dense training reached the frozen monitor target in about **15 minutes**. None
of the **15 sampled configurations** reached it before cutoff. Independent
confirmation and official-test evaluation were therefore not run.
[Full study, uncertainty and costs](analysis/dw_sampling/20261031T120154Z-reopened/final_report-v3.md).

The earlier OAR audit also cost more than native training: all six complete
measured configurations were slower, and its synthetic evaluation accepted
**0/144** past-only proposals. Online OAR remains stopped; finite-precision
acceptance remains unresolved. [Original findings](analysis/report.md).

## Check and reuse

With Python 3.13, verify the published language-study analysis without Torch,
corpus files or Apple hardware. Choose a new output directory:

```sh
python3.13 analysis/verify_dw_sampling.py --out /tmp/oar-public-replay
```

Reuse the [sampled linear operator](src/oar/sampled_linear.py), inspect the
[model and recovery implementation](src/oar/sampling_lm.py), or follow
[setup, tests and reproduction](REPRODUCE.md). The checked implementation passed
**858 tests, including 16 MPS tests**. Public analysis was reproduced from a fresh
checkout; that is separate from independently repeating training.

By **Artem Legotin**. **AGPL-3.0-only** software; **CC BY 4.0** writing and original
results. [Licensing](LICENSING.md) · [Upstream notices](THIRD_PARTY_NOTICES.md) ·
[Citation](CITATION.cff). Development used AI assistance; README editing used
[write-like-me](https://github.com/arlegotin/write-like-me).
