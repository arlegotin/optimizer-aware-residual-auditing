<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Version 6 qualification integration

The operator cost record remains source-attributed to 802e1f5 and native B to d778286. This version adds the fixed byte model/data/recovery engine and qualification-only pilot, after the complete operator cost pass. It has no language time-to-target result.

Independent reviews corrected nonfinite evaluation handling, malformed Adam state/cursors before restore, and unequal-work native optimizer pooling. Native pool observations are normalized by their own update count, preserving raw observations. U/Bmax retain every qualifying fixed-native block including slow observations. Actual training defers loss/norm logging until the block fence; required loss validation remains before AdamW without an unnecessary forward/backward fence. An interrupted block records its LR and host counters; unconfirmed MPS completion is unknown.

Checkpoint retention is two live files during a run and final alone after its durable receipt, retaining all hashes/pruning receipts. The future full-learning launcher must manage this policy across resumed attempts as well as within one worker and must account for all prior interrupted active work. Current qualification does not invoke the training loop. Official-validation losses and official-test bodies remain protected.

The three actual MPS model smoke tests await prepare; CPU reference checks do not qualify accelerator performance. Current CPU suite: 723 passed, 14 MPS deselected. Software AGPL-3.0-only; writing/results CC-BY-4.0. No historical evidence was changed, no corpus text or checkpoints are distributed.
