# Reproduce and inspect OAR

Version 0.2.0 includes two completed research paths: the original low-rank OAR
audit and fixed-budget weight-gradient sampling. Neither established a training
speedup. The [sampling report](analysis/dw_sampling/20261031T120154Z-reopened/final_report-v3.md)
records all 27 development configurations; the [original findings](analysis/report.md)
retain the earlier audited-block and synthetic results.

Use a **full Git clone** or the release Git bundle. Historical validation reads
exact committed blobs; a source ZIP alone lacks the required Git objects.
No command below trains a model or downloads a corpus.

## Check the published language-study results

With Python 3.13, no third-party packages or accelerator needed:

```sh
python3.13 analysis/verify_dw_sampling.py --out /tmp/oar-public-replay
```

Use a new output directory each time. The check verifies the sealed public
artifacts, replays all 27 outcomes/monitor curves, and compares every scientific
JSON field and both CSV byte hashes. Only the separately recorded analysis wall
time may differ. It reproduces the analysis, not the original training runs.

The frozen cutoff was H = 1461.458501 s, with a 29.399480 s endpoint reserve. Native
batch 8 / LR 0.0003 reached the monitor target in 907.104540 s. Every sampled arm was
censored and failed the endpoint guard. A censored time is not H, infinity or a
measured speed ratio. Independent development, protected confirmation, official
validation loss and official-test bodies/losses were not reached.

## Install and run tests

The recorded stack is native arm64 CPython 3.13.5 on macOS, Torch 2.15.0 and
NumPy 2.5.4. Existing dependency pins and hashes remain unchanged. On that stack:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

The sampling data tests also use the permitted PyArrow 26.0.0 reader. Its measured
macOS CPython 3.13 wheel has SHA256
`a6ca849f90cf73fe361f08a5762c783ead9671e4548c1f558cc637b54c9103f2`.
Install that reader on the recorded platform with its hash pin:

```sh
.venv/bin/python -m pip install --only-binary=:all: --no-deps --require-hashes -r /dev/stdin <<'EOF'
pyarrow==26.0.0 --hash=sha256:a6ca849f90cf73fe361f08a5762c783ead9671e4548c1f558cc637b54c9103f2
EOF
```

[Installed files, wheel inventory and dependency hashes](analysis/dw_sampling/20261031T120154Z-reopened/prepare/attempt-0007/dependencies.json)
record that addition. Install a compatible reader for another platform rather
than forcing macOS wheels. Corpus loaders/tokenizers are not dependencies.

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q -m 'not mps'
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_MPS_PREFER_METAL=0 PYTHONPATH=src .venv/bin/python -m pytest -q -m mps
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 PYTORCH_MPS_PREFER_METAL=0 PYTHONPATH=src .venv/bin/python -m pytest -q
```

On the recorded host, the checked implementation passed 858 tests including 16 MPS tests;
an independent source checkout passed 842 CPU tests / 16 deselected. The source and
exact command outputs are retained in the [final evidence manifest](analysis/dw_sampling/20261031T120154Z-reopened/final_evidence_manifest-v1.json).
Release packaging checks have separate [v0.2.0 records](docs/releases/v0.2.0.md).
CPU checks and MPS skips are not accelerator performance evidence. A sandbox can
hide MPS; verify real device access. Run actual hardware work sequentially, with
fallback and fast math disabled before Torch imports. Thermal values that the
host did not expose remain unavailable.

## Reuse an operator

The sampled operator has its own state and API; it does not retrofit the older
projected-linear interface:

```python
import torch
from oar.sampled_linear import SamplingState, sampled_linear

x = torch.randn(32, 8, requires_grad=True)
w = torch.randn(16, 8, requires_grad=True)
state = SamplingState(61001, "example.ffn_in", "importance", 0.25)
y = sampled_linear(x, w, None, state)
y.square().mean().backward()
```

Forward and raw input/bias gradients are exact at the same current parameters;
only dW is sampled. Native clipping and AdamW then act on that estimate. Sampling
is with replacement; duplicate rows and inverse-probability weights matter.
FP32 is the model path; CPU FP64 is a reference/probability path. Unsupported
AMP/higher-order/shared-sampled-weight use fails explicitly. This is not an
activation-memory optimization or certified finite-step descent mechanism.
See [operator tests](tests/test_sampled_linear.py) and [model/protocol tests](tests/test_dw_sampling_protocol.py).

## Regenerate the original OAR report

The original scientific index remains unchanged. This release's copy explicitly
resolves edited publication documents through their actual historical Git bytes:

```sh
PYTHONPATH=src .venv/bin/python -m oar.cli report --gates decisions/gates.jsonl --run-index analysis/releases/v0.2.0/legacy_evidence_index.json --out /tmp/oar-original-report.md
```

Use a fresh report path. No old observation, gate, raw report or scientific index
is overwritten. The [v0.1.0 reproduction guide](https://github.com/arlegotin/optimizer-aware-residual-auditing/blob/v0.1.0/REPRODUCE.md)
retains the original synthetic-repeat commands and permissions. Same-seed numeric
agreement is not independent training confirmation. Normal online OAR remains
stopped; NumericalPolicy remains unresolved.

## Training replay and data boundaries

The released study is closed at STOP_QUALITY. Current code checks and public
metadata replay do not authorize another learning grid or protected evaluation.
The [prospective sampling scope](decisions/dw_sampling_scope.md),
[reopening history](decisions/dw_sampling_reopening.md) and
[publication authority](decisions/releases/v0.2.0-publication.md) keep those
permissions separate.

Training used pinned WikiText revision b08601e04326c79dfdd32d625aee71d232d685c3
under a new [local-only permission decision](analysis/dw_sampling/20261031T120154Z-reopened/prospective/asset_local_use_decision.json).
The original card's license-version discrepancy is preserved, not erased.
Corpus bodies, processed token tensors and checkpoints are not distributed.
Official validation was prepared but no protected loss was observed; test bodies
remain unopened. All reported learning losses are development byte/EOS NLL,
not standard word perplexity.

Historical qualification/cache/checkpoint receipts contain original absolute
paths. Future training elsewhere needs declared, byte-verified private-asset
hydration and a new prospective source/config/asset/path/resource freeze. Do not
silently rewrite sealed receipts or rely on an original cache. Source/config or
numerical/timing changes invalidate affected downstream evidence; unchanged old
measurements keep their original source identity. The current sampled-learning
manifest is archival and does not authorize changed-source training.
