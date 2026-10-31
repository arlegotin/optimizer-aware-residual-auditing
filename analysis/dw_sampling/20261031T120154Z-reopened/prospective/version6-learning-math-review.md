<!-- SPDX-License-Identifier: CC-BY-4.0 -->
# Prospective learning mathematics/statistics review

2026-10-31. Reviewed the current version-6 reopening, rule reconciliation,
`analysis/run_dw_sampling.py:1321-1489`, and
`src/oar/sampling_lm.py:715-720,796-964` before language outcomes. Source reads
only; no tests, hardware work, downloads, or numerical programs were run.
The driver is concurrently receiving qualification orchestration, so this is
approval of the listed mathematical helpers/rules, not a final-byte hash audit.

**PASS for the qualification source freeze. No substantive mathematical fix
identified in this scope.** Qualification must continue to return `not_run`
for the overall pilot and cannot unlock confirmation; the later tuning and
confirmation dispatcher still requires prospective review and binding.

The rules correctly distinguish individual-block A/A jitter from uncertainty
in a mean effect. Removing the inherited `.15` veto and using upper time-ratio
CI strictly below 1 are consistent with v5/v6; neither requires a fictitious
zero minimum effect in an overshoot or sample-size denominator.

- `calibrate_horizon` retains startup plus 4096 native-equivalent updates,
  j=100/50/25 in order, monitor/checkpoint overhead at most 2%, and Bmax at
  most Delta/10. Reserve, resolution and `.01*H` overshoot are explicit and
  independent of jitter (`1327-1350`). The overshoot subcondition is already
  implied by Bmax<=H/(10*j) for these j values; its redundancy is harmless.
- `time_budget_lr` implements preserved t0, 5% linear warmup and cosine to
  0.1 peak. The loop charges initial evaluation/checkpoint work, deliberately
  holds the first block at zero LR, preserves t0 on resume, skips missed
  monitor deadlines, stops updates at a completed block boundary, and marks
  excessive cutoff/endpoint charges inconclusive (`sampling_lm.py:866-948`).
  Monitors/checkpoints remain paid training-budget work; endpoint evaluation
  cannot perform further optimization.
- Target construction uses R0's first monitor at/after .6H, meaningful gain,
  and two later observed crossings. Target time is the second consecutive
  completed crossing strictly before H, with no interpolation or censored
  substitution. Endpoint eligibility and declared simpler ties precede
  candidate/native selection (`1353-1416`).
- Confirmation planning uses three independent development pairs, sample SD,
  the observed negative mean log time ratio, the unchanged conservative
  quality margin `epsilon_q-max(0,mean_q)`, and N=max(8,N_time,N_quality), with
  no arbitrary 16 ceiling (`1419-1435`). This is a normal-approximation planning
  formula, not a guarantee of 80% achieved power: three seeds estimate variance
  poorly, selection can inflate the pilot effect, and the actual t criterion
  is more demanding at small N. Disclose these uncertainties.
- Confirmation inference uses paired log-time t intervals and arithmetic
  paired quality t intervals, correct supplied df7..15 critical values, and
  paired 10,000-draw percentile sensitivity with seed62002. Benefit requires
  both time and quality to pass, with any t/bootstrap decision disagreement
  inconclusive (`1438-1472`). Requiring both claims is an intersection rule;
  it does not justify claims about every tuned alternative. Fixed independent
  protected seeds are essential to avoid carrying tuning selection bias into
  those intervals. Small-N skew/outlier sensitivity remains a disclosure.
- `simpler_dominates` uses the three paired seeds, geometric time-ratio
  comparison with additive frozen resolution expressed relative to candidate
  geometric mean time, and arithmetic quality differences (`1475-1489`).
  It is a prescribed descriptive dominance rule, not a powered equivalence
  test. Nonfinite/censored target inputs fail closed.

Before any future protected confirmation, enforce the planned N, seed IDs,
target, endpoint margin, source/data identities and the vetted df-specific
critical value in the frozen manifest/dispatcher. `analyze_confirmation` is
an analysis helper: it checks only N>=8 and positivity of a supplied critical
value, so it cannot itself prove fixed-N adherence or validate the provenance
of an externally supplied quantile. For N>16, retain the specified installed
tool/table provenance and supplied-value cross-check before outcomes. This
future enforcement requirement does not block the current qualification-only
action.

Raw dX/db exactness is before global clipping. The native clipping factor
depends on sampled dW and therefore legitimately changes applied unsampled
parameter updates; no exact dense-update or unbiased AdamW-displacement claim
is warranted. Full-model cost, development target-time behavior, endpoint
quality, official-validation confirmation and independent reproduction remain
separate evidential requirements.

Declined to judge: corpus authorization, model/backend implementation,
transitive source continuity, qualification orchestration still being edited,
and unseen outcome/manifest integrity. No language result is asserted here.
