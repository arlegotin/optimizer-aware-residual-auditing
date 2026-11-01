<!-- SPDX-License-Identifier: CC-BY-4.0 -->
The new writing, original numerical results and metadata in this run are licensed
CC-BY-4.0. Software, including the standalone analysis/replay scripts, remains
AGPL-3.0-only. Historical material retains its original notices and attribution.

This repository does not distribute the private WikiText corpus bodies, derived
token tensors or model/optimizer checkpoints. Their original source notices and
the narrow local-use decision remain separate from the license on original
measurement results. The asset decision does not settle the historical card's
license-version discrepancy or authorize external redistribution.

Fixed-budget norm-importance dW sampling independently implements elementary
column-row sampling with native Torch primitives. References:

- Adelman et al., NeurIPS 2021: https://arxiv.org/pdf/1805.08079
- Wang et al., ICLR 2024: https://arxiv.org/pdf/2402.17227

This implementation does not reproduce VCAS's adaptive ratios or activation-
gradient sampling. Prior CUDA performance claims are not measurements of this
FP32 MPS implementation.
