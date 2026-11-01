<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Independent fresh-checkout reproducibility and final code-quality review

2026-11-01. **Spec/public-reproducibility verdict: PASS within the stated scope. Code-quality verdict: no Critical or Important correction identified.** This approves current-source correctness and public-metadata reproducibility, not independent MPS learning/performance reproduction or completion of the still-unwritten final report. E4 remains **STOP_QUALITY**.

Reviewed immutable HEAD `6d85a26dafe3e3351979577d09ea90d6949d05cc`, whose 11 frozen implementation/config/scope/test files equal source commit `5361bb59aedccf12af66314f15391c5af60ad595`. The full-history local `git clone --no-hardlinks` is detached at that HEAD in `/private/tmp/dw-sampling-final-review-5fvh08cu/checkout`. It has no object alternates/hardlinks or private data cache. The original checkout supplied only the explicitly shared `.venv/bin/python`; `PYTHONPATH=src` resolves `oar` inside the clone. No original source/cache fallback was used.

## Executed evidence

Exact argv, cwd, environment, exit codes, stdout/stderr and outer wall charges are retained in `01-*` through `09-*` beside this public report. All substantive verification commands passed; the one reviewer-wrapper publication failure is disclosed below.

| Check | Actual result |
| --- | --- |
| Full applicable CPU suite: `python -m pytest -q -m 'not mps'` | **842 passed, 0 failed, 16 deselected; 24.54s**, outer command 24.958076709s; exit 0 |
| Historical reviewed object and files | Full `9d8ad069c51b4d0e6c13e9c6e465579e9b44a339` exists; **184/184** files byte-identical to that commit and frozen hashes |
| Current bindings | **11/11** source hashes match clone/HEAD/source commit; **359/359** amendment bound-file hashes match; no tracked changes |
| Public E4 seal | Record hash and **329/329** sealed public artifact hashes match; no private checkpoint bodies inspected |
| Driver `--help` | Exit 0 |
| `--stage prepare --config configs/dw_sampling.yaml --out <fresh-root> --dry-run` | Exit 0, `not_run`; current amendment validates wholly in clone |
| Public recorded-grid replay into fresh output | Exit 0; all scientific JSON fields match; both CSVs byte-identical |

The dry-run used the unmodified driver under a stdlib audit sentinel: no Torch/body-loader import, corpus/checkpoint open, network event, training subprocess or original-source read occurred. Only the expected three read-only Git subprocesses ran. The first wrapper invocation completed the production dry-run successfully but then rejected its own receipt publication into the original checkout. Its exit-1 log and first dry-run artifacts are retained. A separately saved v2 wrapper changes only the receipt destination to private temp, preserving every guard; its fresh-output invocation passed. No production file or guard changed, and no passed CPU suite was repeated.

Replay reproduces the 27 declared trajectories' recorded analysis, including all 15 sampled arms' censored targets and failed endpoint guards. It runs **zero new training trajectories**. Only `analysis_wall_s_before_publication` is excluded from scientific JSON comparison; both source/reference and reproduced CSV hashes and the replay receipt are retained. The draft SHA256 is `9b83aaaabd3a80f118dcb187f0e9bcba9dd93a1bf1e768660859f4b76c8716c4`; all reviewed report/source hashes are recorded in `binding-verification.json`.

## Source and report assessment

Inspected the actual sampler, article/block construction, model/optimizer/evaluation/controller/checkpoint implementation, source/amendment gates, replay source, recovery admission/ledger/worker policy, and relevant regression tests; read the v10 recovery and sealed-outcome reviews and prospective bindings. Same-probability inverse weighting preserves iid sampling duplicates; native forward/raw dX/db remain distinct from globally clipped/Adam updates. Data boundaries, padding masks, token-weighted evaluation, exact tied output and independently named streams agree with the scoped protocol. The draft appropriately separates complete-step cost evidence from failed learning quality and avoids positive time-to-quality or seed-level inference.

R1's repaired recovery path begins its charged interval before run allocation, prior checkpoint authentication/decode and child preparation. Sibling admission markers preserve unknown deaths without a run directory; known failed preparation contributes once; unauthenticated prior state is null; prior numerical/integrity rejection does not become retryable. Regression tests use real serialization and subprocess death with independently injected costs. Fresh-launch timing and strict Torch-free worker preflight remain intact. Static directory/binding/ownership admission stays explicitly in whole-stage cost. No actual recovery, new timing observation or E4 attribution to repaired code is claimed: all 27 sealed E4 trials were fresh and uninterrupted.

## F1 and limits

**None of the commands executed in this review needs private corpus/checkpoint bodies.** CPU tests use invented fixtures, including tiny test-created checkpoints. Help/dry-run and public replay use source/public metadata only. Current native preparation/correctness are likewise body-independent, but their actual-device portions were not executed here.

An independently authorized future model-qualification run would acquire/parse pinned train and official-validation source bodies and create private processed `.pt` caches. An authorized learning/precision/secondary worker would require the hash-bound fitting/development cache; environmental recovery and ledger reuse also hash retained own checkpoints, and recovery decodes its explicit selection. A future protected confirmation implementation would additionally require separately authorized sealed-validation bytes. The frozen qualification metadata and learning manifest retain original absolute metadata/cache paths, while receipt/checkpoint ownership binds original locations. Relocation therefore requires explicit byte-verified hydration and a new prospective source/config/asset/path freeze; copying public Git files alone does not reproduce training. Current v10 does not authorize another grid, and the confirmation dispatcher remains gated/unimplemented for this stopped study. No official-validation loss or official-test body was opened here.

Shared Python **3.13.5**, Torch **2.15.0**, NumPy **2.5.4**, PyYAML **6.0.3**, psutil **7.2.2**, pytest **9.2.0**, PyArrow **26.0.0** were read from the existing environment; no dependencies were installed or independently rebuilt. This is an independent reviewer/fresh source checkout on the same host/environment/day, not independent hardware, timing, operator/day or training-seed evidence. Root's supplied **858 passed including 16 MPS** and **494 passed** actual-host results remain separately attributed existing evidence, not this reviewer's executions. Software/report license notices are preserved.

No new scientific run, network fetch, corpus hydration, real corpus/checkpoint body read, source/config edit, original Git mutation or child agent was used. **A narrow final-report-diff consistency check remains required once the final report is assembled**, especially accounting, version/test attribution and any changed scientific wording. This review does not declare whole engineering completion.
