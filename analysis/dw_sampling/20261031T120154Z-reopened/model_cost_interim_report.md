<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Full-model cost and calibration: interim evidence

Recorded 2026-10-31T17:16:15.472061+00:00. Full-model qualification passed; enclosing pilot status remains `not_run`, because tuning and independent development have not run. No language-learning benefit is established. Official-validation loss and official-test body remain unopened.

Measured actual HEAD: `110be7739067eec0aaddef40e21763818691fea8` (full identity in source_manifest.json). Direct driver SHA256: `ccd8490f6048905ffa5f48864143ab1df6e12ed77af146a7877b47ab2230d5bd`. Native FP32 MPS on Apple M3 Max with fallback/fast-math/prefer-metal disabled. Strongest baseline-only pooled native AdamW variant: **fused**. Five identical-current-clipped-gradient numerical checks passed for all four native variants in each fresh session; the previous failed independent-trajectory qualification is retained separately.

Whole steps include fixed four-block byte model, causal attention, embeddings/tied output, actual data collation/padding, backward, native clipping and AdamW. Sampling approximates only eight FFN weight gradients. Raw input/bias adjoints remain exact at the same parameters; native global clipping can change applied updates of other parameters.

| Session | Arm/fraction | Candidate/native ratio | 95% paired-block bootstrap interval |
| --- | --- | --- | --- |
| 1 | importance-0.125 | 0.903179 | [0.890101, 0.917626] |
| 1 | importance-0.25 | 0.931107 | [0.925404, 0.937022] |
| 1 | uniform-0.125 | 0.843164 | [0.837049, 0.849290] |
| 1 | uniform-0.25 | 0.877647 | [0.871657, 0.883805] |
| 1 | uniform-0.5 | 0.949557 | [0.943740, 0.955381] |
| 2 | importance-0.125 | 0.896137 | [0.883343, 0.908546] |
| 2 | importance-0.25 | 0.927363 | [0.918220, 0.936142] |
| 2 | uniform-0.125 | 0.833434 | [0.819180, 0.848309] |
| 2 | uniform-0.25 | 0.883582 | [0.872128, 0.895041] |
| 2 | uniform-0.5 | 0.949515 | [0.923894, 0.975706] |

Each interval describes 30 timing pairs, not independent training seeds. Both sessions are retained; random order, resets, warmup and raw observations are in the sealed attempt. Serial dependence, exploratory shape/fraction comparisons and geometric rather than arithmetic ratios limit interpretation. The prospectively reopened gate is upper95% ratio<1 in both sessions, not the earlier 1-delta threshold. No original OAR result or old gate is changed.

Frozen baseline calibration: H=1461.458501s; reserve R=29.399480s; nominal T_budget=1490.857981s; Delta=29.229170s; r_time=31.627892s; block updates=8; epsilon_eval=1e-06. Native conservative update bound is 0.349031042s, using every qualified fixed-native observation. All three cold initial/full/monitor and endpoint/checkpoint measurements are retained. No outcome-based recalibration is authorized.

Qualification invocation: 1014.102614s. Shared verified preprocessing this attempt: 2.364064s, separately disclosed. Maximum currently eligible grid: 27 runs. Whole remaining plan before deduplication: 52+2*N full-budget equivalents, approximately 28.16 device-hours at provisional minimum N=8, plus separately charged engineering/diagnostic costs. N will be fixed from independent development, not confirmation. These are forecasts, not completed trials.

Recommendation: continue to the already-specified learning study. Uniform1/8 is the strongest speed contender; quality and conventional simpler competitors may defeat it. Stop or report inconclusive if the declared quality, dominance, censoring or confirmation gates fail.

Raw attempt: `pilot/attempt-0002/`; record SHA256 `d01c854daf8eb7f7fbc8505b0879b40bc869bfc469a141fa727ae086757c4209`. Its source/config/amendment identity and all artifact hashes are sealed there. Original historical 184 tracked files remain unchanged. No corpus excerpts or tensors are distributed in this report.
