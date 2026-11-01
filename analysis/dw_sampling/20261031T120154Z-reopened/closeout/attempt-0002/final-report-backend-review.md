<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Final report v2 — backend, recovery and cost consistency review

2026-11-01. Reviewed `analysis/dw_sampling/20261031T120154Z-reopened/final_report-v2.md`, SHA256 **`d814ef894662ef72cac4eb70b53d2e500de02d01af10b0022b5b808e3baffaf2`**. **Verdict: report correction required before final approval.** One Important report-only attribution error was found; it requires no source fix, new measurement, scientific rerun or change to STOP_QUALITY.

## Important: secondary captures are claimed although that stage was not reached

At line51, v2 says “stored update1024 captures are retained”; line57 says “final and secondary private checkpoint bytes were separately verified.” E4 created no update1024 secondary checkpoints. Every one of the27 trial entries in the retained `closeout/attempt-0001/sealed-pilot-verification.json` has `secondary_history: []`. The inspected E4 grid request has `secondary_update: null`; the driver sets update1024 capture on independent-development trials (`analysis/run_dw_sampling.py:2586`), which were never reached after the failed grid gate. Runtime capture is conditional on that request field (`src/oar/sampling_lm.py:843-845,981-982`).

Correct line51 to say the update1024 capture/scoring stage was not reached and no such secondary checkpoints were created. Correct line57 to attribute private verification to final endpoint checkpoints only. Preserve the existing explicit not-run status of independent development, precision, context128/adverse-shift learning diagnostics and protected outcomes. This finding concerns the report's claim of created/verified artifacts; the negative learning conclusion and approved recovery implementation are unchanged. Root was notified with exact evidence before this report was written.

## Otherwise consistent in the reviewed scope

- Source attribution separates v9 measured implementation89c5bd8 / launch43c2cec / sealed-evidence a3e3229 from repaired source5361bb59, prospective freeze f7ce82f and host-evidence commit6d85a26. No old performance is relabeled as v10 recovery performance.
- Existing preparation/correctness stdout confirms **842 CPU passed/16 deselected**, **16 MPS passed/842 deselected**, **858 full passed**, and **494 correctness passed**, with the stated durations. These are existing root executions, not new reviewer executions; the report properly avoids adding overlapping test subsets. The separate clone review records842 CPU checks, metadata replay, shared original read-only environment and same-host/day limits, without claiming independent training/MPS timing.
- The report accurately states repaired admitted-recovery timing, known failed preparation charged once, null unauthenticated prior, unknown missing terminal, nonretryable numerical/integrity rejection, original t0/wasted-work preservation and downtime exclusion. Its static directory/binding/ownership pre-admission boundary matches the approved v10 decision. It does not promise universal first-write durability or actual MPS recovery evidence.
- The D component table agrees with the retained three-observation profile groups and15.263710667029954s diagnostic wall record. Its separate fencing/non-additivity, teacher-model setting, uniform-versus-importance cost distinction and absence of a new performance measurement are explicit. Full-model decomposition and later learning diagnostics are correctly labeled not reached.
- The retained unique outer-stage ledger records47673.64810620784s and explicitly excludes double-counting nested work, GPU-busy interpretation and invented unrecorded engineering durations. The final prose preserves those limits, distinguishes canonical trial/E4/pilot/whole-branch allocations, and states the nested preprocessing inclusion.
- STOP_QUALITY,27 finite fresh/no-resume trajectories, censored sampled times, absent positive time-to-quality claim, separate steady-state cost evidence, unaccessed protected losses/test bodies, local-use/private-body limits, guarded/unreached F, and future private hydration requirements are consistently stated. Current public metadata reproduction is not presented as self-contained relocated training.

## Review limits

This was a narrow report/evidence read. I read existing public report/review text, selected small JSON fields, current test stdout, E4 request/verification metadata and D diagnostic profile metadata, and hashed the report text itself. I ran no tests, hardware/network work, training, real corpus/checkpoint-body read/hash, collector, Git mutation, source/config/record edit or subagent. I did not repeat old source/seal audits or independently rederive all scientific calculations. The final evidence-manifest link is intentionally awaiting the review addenda and is not flagged as a broken-artifact defect here.

The sole write is this ignored CC-BY-4.0 report. A narrowly checked corrected immutable report revision can resolve the Important finding; no broader verification is justified by these two prose corrections.

## Scoped v3 correction addendum — ADDRESSED / APPROVED

2026-11-01. Independently read the complete v2→v3 text diff and hashed `final_report-v3.md`: **`f73627316aa6d60e12e0e00cccb2c78ebee0311f6b24603a51b03533815ab690`**. The only changes are the title, the two secondary-checkpoint sentences identified above, and the mathematical review's narrow mixture-optimum wording correction. V2 remains retained.

**The Important attribution finding is ADDRESSED. Final report v3 is APPROVED within this backend/recovery/cost consistency scope.** V3 correctly says grid update1024 captures were not requested/created because independent development was not reached, and attributes private verification to final endpoint checkpoints only. This agrees with the previously inspected27 empty secondary histories. The additional mixture wording introduces no backend/cost conflict. STOP_QUALITY, measured/current-source attribution, existing host/clone test evidence, admitted-recovery boundary, cost-accounting limits and protected/not-run claims are otherwise unchanged.

No Critical or Important issue remains in this reviewed scope. No broader audit/test, hardware/network action, real corpus/checkpoint-body access, source/config/scientific-record edit or new scientific run was performed or authorized. This addendum records the exact corrected report; final evidence-manifest publication remains root-owned bookkeeping.
