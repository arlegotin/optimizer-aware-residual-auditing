# Reached-task review record

Task review status below is transcribed from the controller's accepted task records. It is not a new independent audit of unimplemented work. Final whole-branch review is pending with the controller.

| Task | Actual commit range | Accepted findings and fixes | Remaining scope limits |
|---|---|---|---|
| 1, records/configs | af715ac..2115d04 | Nonfinite evidence read path rejected; displacement record aligned before consumers. Review clean after fix. | Owning-task schema extensions only when reached. |
| 2, discovery/provenance | 2115d04..d3f8fcb | CLI config exit2 and fallback-before-import classification fixed.44 archived source hashes checked; actual MPS validated. | Dataset availability is not ingestion; WikiText remains blocked. |
| 3, projected operator | d3f8fcb..6393209 | Exact adjoints and reference numerical tests accepted; MPS marker added. | Durable collector minor subsequently fixed in Task6; no trainer integration. |
| 4, audits/journal | 6393209..0cbaa96 | Independent streams, high-bit seed handling and fail-closed journal resume accepted. | NumericalPolicy unresolved; full-model resume was never reached. |
| 5, optimizer preview | 0cbaa96..a042839 | Native-equivalent updates and constructed adaptive-optimizer counterexample accepted. | Extreme finite FP32 endpoints can produce nonfinite displacement/alignment; audit decision rejects it. No claim of arbitrary finite-input preview closure. |
| 6, operator costs | a042839..6f04c6e | Corrected C/(K-1), K1 boundary, transported separated RNG streams; same-shape actual-MPS study repeated.54 thresholds and24 paired means independently checked. Review clean. | Recorded swap/pressure, within-study intervals and reference implementation costs; no training speedup. Tree-SHA labeling and durable collector minors addressed here. |
| 11, bounded amendment only | 6f04c6e..3d7b57c | Partial failed native histories/checkpoints retained; full report moved to durable path. Independent control assertions strengthened. Review clean after fix. | Measured source2e89e74; later fixes did not retroactively remeasure. Ordinary WP03 unfulfilled/WP04 inconclusive; correlated synthetic counts are not independent runs. |
| 17, stopped synthesis | 3d7b57c..8e6b5b2 (implementation/reproduction); following publication commit | Report/comparison TDD, immutable failed processing plus recovered same-seed evidence, archived hash resolution and nonoverwrite guards verified;348 actual-device tests pass. Review remains pending with controller. | No Tasks7–10 or12–16 were implemented; no online, corpus or confirmation branch. |

The original Task11 research ran at `2e89e742131eb5eebaeb7a028d8e8f2e4fc50e07`; results were retained in `dc56170`. Its pilot ran at `a100d5c`. Timing was refreshed at `6f04c6e`, with raw source hashes attached. Source pins and archives do not validate faithful ports of INSTANT, DropBP or VCAS; those baselines were not executed.

Remaining practical limitations are explicit: no formal finite-precision/run-wide execution certificate, real-corpus quality, full trainer replay, dropout/resume lifecycle, Muon/nearest-art competition, independently seeded confirmation, or full training time/memory frontier. These are omitted-stage limitations, not passing outcomes. See [rulings](rulings.md), [oracle report](reports/structural_oracle.md) and [synthesis](../analysis/report.md).
