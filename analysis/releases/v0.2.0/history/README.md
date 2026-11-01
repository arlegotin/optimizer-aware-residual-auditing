# Optimizer-aware residual auditing

Optimizer-aware residual auditing (OAR) constructs low-rank weight gradients, then checks the AdamW step they would produce. A projected gradient can look close to the dense gradient and still produce a harmful AdamW step. OAR previews that step and audits its alignment with fresh residual probes. Input gradients stay exact, so the approximation stays local to the weight gradient.

The repository includes the parts needed to inspect that mechanism: low-rank linear operators, streaming microbatch accumulation, transactional AdamW preview/commit, a durable confidence-spending journal, synchronized MPS cost accounting, and a bounded synthetic oracle. The oracle separates past-only proposals from paid current-gradient SVD diagnostics. Reports bind observations to configuration and source hashes, including historical source bytes when the implementation has changed.

You can reuse the component code and its reference tests, examine where the audit spends its time, or regenerate the findings without running an experiment. The [blueprint](docs/blueprint.md) explains the mathematical argument and the questions the study was meant to answer.

## What the measurements say

On the recorded Apple M3 Max environment, one wide residual block gave these paired timings:

| Recorded boundary | OAR | Native AdamW |
| --- | ---: | ---: |
| Candidate reconstruction | 28.875 ms | 36.882 ms |
| Complete audited attempt, conditional zero rejection | 54.406 ms | 36.738 ms |

That component saving didn't survive the full bill. Every complete attempt in the six measured configurations was slower. These are isolated block measurements with recorded memory pressure and timing uncertainty; the conditional commit is pricing only. Later validation fixes were tested, not remeasured.

The synthetic oracle accepted **0 of 144 past-only evaluation proposals** throughout its declared exact-margin threshold grid. A fresh same-seed repeat reproduced all 792 layer rows and 30 learning-curve observations exactly after a separately recorded comparison-schema correction. That's numerical reproducibility, with two initialization/data seeds and correlated proposals. It establishes no independent confirmation or training speedup.

Normal online work remains stopped, finite-precision acceptance is unresolved, and there are no language-model or end-to-end training results. The [findings](analysis/report.md) retain the costs, failed comparisons and scope limits.

## Run and reuse

After the pinned setup in [REPRODUCE.md](REPRODUCE.md), regenerate the report from a full Git clone:

```sh
.venv/bin/python -m oar.cli report --gates decisions/gates.jsonl --run-index analysis/evidence_index.json --out analysis/report.md
```

That page also covers tests, actual MPS access and the guarded synthetic repeat. The [final review](decisions/reports/final_review.md) records accepted implementation fixes.

Original software is AGPL-3.0-only; original writing and result presentation are CC-BY-4.0. See [licensing](LICENSING.md) for scope and upstream rights; [CITATION.cff](CITATION.cff) records the citation. Development and reporting used AI assistance under Artem Legotin's direction; README editing used [write-like-me](https://github.com/arlegotin/write-like-me).
