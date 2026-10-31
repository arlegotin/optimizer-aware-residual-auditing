<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Reopened operator cost result — interim, no learning outcome

Analyzed release: 9d8ad069c51b4d0e6c13e9c6e465579e9b44a339. B/native measurements: d7782862c025da6ee27dc9f7692b75fe926cb070. D measurements: 802e1f5b16ac915fcce5204aa96c3a64bacd6154, with direct source hashes in cost/attempt-0001/source_manifest.json. All 184 original tracked files remain byte-for-byte unchanged.

**Stage D passed the owner-reopened exploratory cost screen. There is no language-model target-time or endpoint-quality result.** The old online OAR study and its stopped gates remain unchanged. This is fixed-budget norm-importance dW sampling, with a direct-randint uniform comparator, exact native forward/input/bias adjoints and native AdamW.

Actual hardware: Apple M3 Max, native arm64 Python3.13.5/Torch2.15.0, FP32 MPS; fallback=0 actively qualified, fast-math=0, prefer-metal=0. AC power. Dependencies and RECORD hashes are retained in prepare/attempt-0002. CPU timings are not used as accelerator evidence.

Current frozen-version full suite: **552 passed in10.87s** (prospective/full-tests-cost-policy.txt). C/operator+protocol: **188 passed**, record SHA 6f02398a2985f684538f900672defc7987ee49a18e37cee3854e88af0c0c0595. Tests currently being added for the next stage are not part of those counts.

## Complete residual-block costs

Each cell is the geometric mean candidate/dense time ratio with its paired95% percentile-bootstrap interval, not a ratio of total invocation times. Thirty randomized pairs per arm/shape/session, synchronized complete blocks, fixed dense-calibrated counts, identical cloned starting tensors/nonzero native moments/data/sampling streams; repeated blocks are timing replicates, not independent training seeds. All candidate checks, CPU sampling/probabilities, transfers, gathers, smaller GEMMs, exact adjoints, clipping and native optimizer work are charged. Resets/cloning are outside steady-state timing and separately recorded. One-time sampler setup is amortized over the declared blocks. Native variant selection used baseline-only pilots among supported default/single/foreach/fused implementations.

| Shape | Arm | Session1 ratio [95% interval] | Session2 ratio [95% interval] | Pass at this planned-model shape |
| --- | --- | --- | --- | --- |
| middle4096 | importance-0.125 | 0.8607 [0.8410, 0.8805] | 0.8635 [0.8483, 0.8779] | yes |
| middle4096 | importance-0.25 | 0.8936 [0.8785, 0.9094] | 0.9059 [0.8966, 0.9161] | yes |
| middle4096 | importance-0.5 | 1.0963 [1.0852, 1.1074] | 1.0187 [1.0067, 1.0309] | no |
| middle4096 | uniform-0.125 | 0.7533 [0.7423, 0.7653] | 0.8146 [0.7765, 0.8602] | yes |
| middle4096 | uniform-0.25 | 0.8364 [0.8220, 0.8516] | 0.8527 [0.8452, 0.8600] | yes |
| middle4096 | uniform-0.5 | 0.9241 [0.9064, 0.9410] | 0.9376 [0.9092, 0.9679] | yes |
| middle2048 | importance-0.125 | 0.9294 [0.9099, 0.9511] | 1.0597 [1.0182, 1.1156] | no |
| middle2048 | importance-0.25 | 0.9863 [0.9606, 1.0119] | 1.0992 [1.0277, 1.1816] | no |
| middle2048 | importance-0.5 | 1.1121 [1.0970, 1.1276] | 1.1321 [1.1124, 1.1523] | no |
| middle2048 | uniform-0.125 | 0.8334 [0.8164, 0.8498] | 0.8292 [0.8184, 0.8395] | yes |
| middle2048 | uniform-0.25 | 0.8814 [0.8719, 0.8916] | 0.8902 [0.8770, 0.9029] | yes |
| middle2048 | uniform-0.5 | 0.9881 [0.9753, 1.0022] | 1.0056 [0.9948, 1.0165] | no |
| middle2048_accum4_stride2 | importance-0.125 | 1.6625 [1.6261, 1.6988] | 1.7702 [1.7444, 1.7976] | no |
| middle2048_accum4_stride2 | importance-0.25 | 1.7552 [1.7094, 1.8025] | 1.8152 [1.7645, 1.8811] | no |
| middle2048_accum4_stride2 | importance-0.5 | 1.9065 [1.8773, 1.9363] | 1.8809 [1.8531, 1.9099] | no |
| middle2048_accum4_stride2 | uniform-0.125 | 1.3609 [1.3325, 1.3891] | 1.4301 [1.3596, 1.5281] | no |
| middle2048_accum4_stride2 | uniform-0.25 | 1.4168 [1.3866, 1.4482] | 1.5757 [1.4613, 1.7216] | no |
| middle2048_accum4_stride2 | uniform-0.5 | 1.5530 [1.5309, 1.5772] | 1.5900 [1.5151, 1.6933] | no |
| wide | importance-0.125 | 0.6675 [0.6555, 0.6794] | 0.7189 [0.7074, 0.7312] | no |
| wide | importance-0.25 | 0.7614 [0.7368, 0.7894] | 0.7781 [0.7707, 0.7857] | no |
| wide | importance-0.5 | 0.8635 [0.8434, 0.8837] | 0.9058 [0.8852, 0.9288] | no |
| wide | uniform-0.125 | 0.6447 [0.6352, 0.6542] | 0.6799 [0.6727, 0.6870] | no |
| wide | uniform-0.25 | 0.7049 [0.6854, 0.7237] | 0.7223 [0.7104, 0.7345] | no |
| wide | uniform-0.5 | 0.8354 [0.8172, 0.8533] | 0.8483 [0.8141, 0.8832] | no |
| tiny | importance-0.125 | 6.4189 [6.2146, 6.6118] | 6.1194 [6.0718, 6.1700] | no |
| tiny | importance-0.25 | 6.0540 [4.8523, 7.5002] | 6.3930 [6.1604, 6.7377] | no |
| tiny | importance-0.5 | 6.2964 [6.2228, 6.3859] | 6.8087 [6.1625, 7.5876] | no |
| tiny | uniform-0.125 | 4.5736 [4.5086, 4.6386] | 4.6354 [4.5483, 4.7298] | no |
| tiny | uniform-0.25 | 4.6236 [3.8401, 5.5382] | 4.5948 [4.1781, 5.0303] | no |
| tiny | uniform-0.5 | 4.6640 [4.5745, 4.7574] | 4.5389 [4.3102, 4.7238] | no |

At4096 rows, importance1/8 took approximately14% less step time in each session; uniform1/8 took25% and19% less time respectively. Corresponding throughput gains are approximately16% for importance and33%/23% for uniform. Importance1/2 was slower in both sessions. All accumulation/stride2 and tiny controls were slower. Wide-only results do not authorize changing model size.

**Prospective policy change disclosed:** the pooled native A/A diagnostic delta was 0.2085046466766194. Before any sampled timing, v5 replaced the old minimum-gain gate with upper95% ratio<1 at the same middle shape in both sessions. No arm passed the old upper<1-delta screen in both sessions. The current pass depends on that authorized, pre-candidate policy; it is not an old-protocol pass. The bootstrap assumes independent/exchangeable whole pairs and does not establish a familywise guarantee over12 eligible arm/shape opportunities. Fixed arm order, drift, warmup flags and every observation remain visible.

## Information and adverse controls

The paid offline study retained576 independent draws:32 draws x6 arms x3 native states. Conditional variance is calculated from CPUFP64 full operands and the actual CPU probabilities; these full CPU copies occur only in this separately charged diagnostic. Row-score heterogeneity was approximately1.002–1.007, so norm importance reduced this synthetic fixture variance by only0.2–0.7% versus uniform. This does not predict score dispersion in language training.

At state128, normalized conditional variance at rho1/8 reached approximately54.6 for the input FFN weight and13.3 for its output weight. The exact gradient shrinks relative to individual row contributions, so sampling noise can be large near a fit. Unbiased gradients do not imply unbiased AdamW displacement or preserved quality. All one-step loss/displacement and per-layer gradient errors remain in diagnostic-state-*.json; they are diagnostics, not language results or runtime acceptance tests. Uniform offline scaling is algebraically equivalent but differs in FP32 rounding from the narrower-operand runtime path, as the mathematical review notes.

## Cost, evidence and review

D invocation wall: 4733.915 seconds, including both sessions/warmups/reset/serialization and the separately paid 15.264-second offline diagnostic. This is1800 paired comparisons/3600 timed complete blocks. B/native feasibility additionally cost1465.270 seconds; preparation and the earlier stopped run have their own retained ledgers. No language tuning cost has been incurred. Full-model timing must choose its own strongest qualified native eager implementation before learning; a smaller update batch or correct native precision control may still win.

Raw records: cost/attempt-0001/record.json (SHA256 d52f2062995810f4541022d02ed21e8f33ad72257675f66f8bed773c34fa9b18), session-invocation-{1,2}-*-pairs.jsonl, complete shape/session JSON, warmup/calibration/reset/stream/resource records, diagnostic.json, and commands.log. The record seal hashes every stage artifact. Source start/end maps were independently checked equal to the then-current files and amendment. RSS/Metal/driver endpoint observations overlap and are not added or called exact peaks. Global swap/available-memory conditions are recorded, with no causal attribution and no memory-saving claim.

Reviews: prospective/cost_math_review.md and cost_backend_review.md found no blocking issue. They disclose serial dependence, multiple exploratory comparisons, replayed streams, geometric estimand and diagnostic limitations. The backend review found a remaining fixed prepare240s timeout; it did not affect the passed qualification, and will be corrected before any future prepare rerun. A fresh-checkout review of reached model/runner code is still pending. Prior-art scope is in prospective/prior_art_review.md.

**Continue recommendation:** implement and test the specified fixed byte model, freeze its source/config/data bindings, acquire only conditionally approved local-use assets with pinned byte hashes, and run the full-model cost gate. No primary positive-learning claim is supported yet. Official test remains unopened; official-validation losses remain protected.
