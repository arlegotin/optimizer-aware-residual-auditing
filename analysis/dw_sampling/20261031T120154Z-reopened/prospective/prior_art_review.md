<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Primary-source check, 2026-10-31

[Adelman, Levy, Hakimi and Silberstein, NeurIPS 2021, *Faster Neural Network Training with Approximate Tensor Operations*, arXiv v3](https://arxiv.org/pdf/1805.08079): section2.2 equations1–4 describe weighted column-row sampling and norm-product probabilities. Their backward approximation discussion and SGD results do not establish this study's finite-time native-AdamW or held-out-quality claim. Their reported training speed is from their implementation/workloads, not new MPS evidence.

[Wang, Chen and Zhu, ICLR 2024, *Efficient Backpropagation with Variance-Controlled Adaptive Sampling*](https://arxiv.org/pdf/2402.17227): VCAS samples activation gradients as well as token contributions to weight gradients and adapts sampling to control variance. This branch fixes a global fraction, retains exact input adjoints and uses the specified norm-product/uniform-mixture probabilities. It is not a faithful VCAS implementation, and its results must stand on the charged local measurements.

Only primary paper text/abstracts were read. No third-party implementation was copied or imported. The elementary estimator, numerical support checks and native Torch gather/GEMM implementation were independently written and independently enumerated in the repository tests. These sources support prior-art attribution, not a positive result for this experiment.
