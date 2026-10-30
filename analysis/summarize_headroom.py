"""Derive bounded cost decisions from measured whole blocks, never profile sums."""

import hashlib
import json
import csv
from pathlib import Path

import numpy as np

from oar.bench import cycle_forecast


sources = [Path('analysis/timing_calibration.json'), Path('analysis/timing_calibration_wide.json')]
tables = [json.loads(source.read_text()) for source in sources]
measured_rows = []
for source, table in zip(sources, tables):
    for measured in table['rows']:
        measured_rows.append((source, measured))
rows = []
for source, measured in measured_rows:
    h = measured['headline']
    candidate_ci = h['candidate']['dense_minus_other']['mean_bootstrap_95_s']
    complete_ci = h['conditional']['dense_minus_other']['mean_bootstrap_95_s']
    candidate_qualified = h['candidate']['noise_target_met'] and all(v['stable'] for v in h['candidate']['warmup'].values())
    td = h['conditional']['dense']['mean_s']
    ta = h['conditional']['other']['mean_s']
    reject_total = h['reject']['other']['mean_s']
    replay_extra = h['replay_only']['other']['mean_s']
    refresh = measured['refresh']
    refresh_extra = refresh['other']['mean_s']-refresh['dense']['mean_s']
    forecasts = []
    # Sensitivity grid only; these values are neither recommended nor selected
    # online refresh/acceptance policies. Actual full-cycle training is absent.
    if replay_extra > 0 and refresh_extra >= 0:
        for interval in (4, 16, 64):
            for rejection_probability in (0., .1, 1.):
                forecasts.append(dict(interval=interval, rejection_probability=rejection_probability,
                                      **cycle_forecast(td, ta, replay_extra, refresh_extra, interval, rejection_probability)))
    rng = np.random.default_rng(63001)
    # Independent block bootstrap across the two separate comparisons; retain
    # drift and pairing limitations rather than implying training-run inference.
    attempt_raw = np.array(h['conditional']['other']['raw_s'])
    dense_raw = np.array(h['conditional']['dense']['raw_s'])
    replay_raw = np.array(h['replay_only']['other']['raw_s'])
    refresh_pairs = refresh['pairs']
    refresh_diff = np.array([r['seconds']['other']-r['seconds']['dense'] for r in refresh_pairs])
    bootstrap = []
    for _ in range(2000):
        idx = rng.integers(len(attempt_raw), size=len(attempt_raw))
        aa, dd = attempt_raw[idx].mean(), dense_raw[idx].mean()
        rr = rng.choice(replay_raw, len(replay_raw)).mean()
        cc = rng.choice(refresh_diff, len(refresh_diff)).mean()
        if rr > 0:
            bootstrap.append((dd-aa-cc/15)/rr)
    rows.append(dict(key=measured['key'], shape=measured['shape'], source=str(source), fastest_native=measured['fastest_native'],
        candidate_headroom=('observed_positive' if candidate_qualified else 'positive_but_unstable_or_noisy') if candidate_ci[0] > 0 else 'observed_negative' if candidate_ci[1] < 0 else 'unresolved',
        candidate_measurement_qualified=candidate_qualified,
        candidate_saving_95_s=candidate_ci, conditional_full_saving_95_s=complete_ci,
        conditional_full_headroom='observed_positive' if complete_ci[0] > 0 else 'observed_negative' if complete_ci[1] < 0 else 'unresolved',
        dense_s=td, conditional_attempt_s=ta, forced_reject_total_s=reject_total,
        direct_exact_replay_s=replay_extra, paired_refresh_extra_s=refresh_extra,
        rejected_minus_conditional_s_not_replay_cost=reject_total-ta,
        local_break_even_p_K16_bootstrap_95=np.quantile(bootstrap,[.025,.975]).tolist() if bootstrap else None,
        forecasts=forecasts,
        warmup_stable={name:{k:v['stable'] for k,v in comparison['warmup'].items()} for name,comparison in h.items()},
        noise_target_met={name:c['noise_target_met'] for name,c in h.items()},
        named_reference_costs_ms={name:float(np.mean(measured['profile_raw_s'][name])*1000) for name in
            ('preview_including_raw_alignment','preview_input_validation','raw_alignment_cpu_fp64_transfer','generation_journal_transfer_capture')},
        journal_age_ms=[dict(prior_reservations=a['prior_reservations'], mean_ms=float(np.mean(a['raw_s'])*1000)) for a in measured['journal_age']]))

candidate_exists = any(r['candidate_headroom']=='observed_positive' for r in rows)
complete_exists = any(r['conditional_full_headroom']=='observed_positive' for r in rows)
decision = ('audit_cost_failure_candidate_potential' if candidate_exists and not complete_exists else
            'no_resolved_candidate_headroom' if not candidate_exists else 'conditional_cost_headroom_only')
output = dict(schema_version=1, decision=decision, rows=rows,
    source_sha256={**{str(source):hashlib.sha256(source.read_bytes()).hexdigest() for source in sources},
                   'analysis/summarize_headroom.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    next_permitted_action=('Prepare a bounded structural oracle information study using synthetic inputs and copied optimizer states; record its narrow branch/scope amendment before execution because normal WP03/WP04 prerequisites remain blocked. Do not build a full trainer/controller or ingest datasets. No online branch until useful proposals/discrimination, numerical policy and plausible inclusive cost jointly pass an explicit amended gate.' if candidate_exists else
                           'Stop online engineering for measured regimes; synthesize the negative/inconclusive headroom result. Any larger-model/backend retry requires evidence and an amendment.'),
    limits='p=0 is conditional pricing, not measured acceptance. NumericalPolicy remains unresolved. No learning, training throughput, power, finite-step loss, or generalization claim. Forecasts use direct complete exact replay blocks and paired extra refresh work; they are local arithmetic sensitivity estimates, not observed cycles. Rejected-minus-conditional means cannot identify replay cost because independent journals age differently; raw per-pair reservation counts are retained. Bootstrap intervals describe within-study blocks under recorded system pressure, not independent runs. Fresh journals understate long-run reference costs; aged measurements are retained without extrapolating beyond 256 reservations.')
Path('analysis/headroom_decision.json').write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False)+'\n')
with Path('analysis/headroom_summary.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['backend','dtype','n','di','do','rank','q','accumulation','native_variant','comparison',
                     'boundary','dense_s','other_s','saving_low_s','saving_high_s'])
    for source, measured in measured_rows:
        for name,c in measured['headline'].items():
            writer.writerow([*measured['key'], measured['shape']['accumulation'], measured['fastest_native'], name,
                'complete residual MLP forward + exact dX/bias + dense or reconstructed dW + clipping + optimizer; conditional adds generation/journal/probes/preview/raw alignment/complement/decision/commit; rejection adds exact replay',
                c['dense']['mean_s'], c['other']['mean_s'], *c['dense_minus_other']['mean_bootstrap_95_s']])
print(json.dumps(dict(decision=decision, candidate_regimes=[r['shape'] for r in rows if r['candidate_headroom']=='observed_positive'],
                      full_path_positive_regimes=[r['shape'] for r in rows if r['conditional_full_headroom']=='observed_positive'])))
