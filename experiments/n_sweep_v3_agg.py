"""
n_sweep_v3_agg.py -- Extended N-sweep for Direction C (aggregation-
aware v3, agg_savings=0.9) vs LEACH/HEED.

Resumable: each (N, strategy) result saved to disk immediately.
Safe to interrupt and rerun -- completed cells are skipped.
"""
import numpy as np
import json
import time
import os
from scipy import stats
from flow_simulate_agg import run_episode_v3_agg
from protocols import PROTOCOL_FUNCS

OUTDIR = 'n_sweep_v3_agg_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, R_c = 1.0, 35.0
MAX_ROUNDS = 3000
N_VALUES = [30, 60, 100, 200, 500]
N_SEEDS = 15
AGG_SAVINGS = 0.9
STRATEGIES = ['v3_agg', 'LEACH', 'HEED']


def log(msg):
    print('[' + time.strftime('%H:%M:%S') + '] ' + msg, flush=True)


def run_one(strat, N, seed):
    if strat == 'v3_agg':
        return run_episode_v3_agg(N=N, E0=E0, R_c=R_c, agg_savings=AGG_SAVINGS,
                                   max_rounds=MAX_ROUNDS, seed=seed,
                                   record_series=False)
    else:
        return PROTOCOL_FUNCS[strat](N=N, E0=E0, max_rounds=MAX_ROUNDS,
                                      seed=seed, beta=1.0)


log('=== PROFILING ===')
profile = {}
for N in N_VALUES:
    for strat in STRATEGIES:
        t0 = time.time()
        res = run_one(strat, N, seed=0)
        elapsed = time.time() - t0
        profile[str(N) + '_' + strat] = elapsed
        log('  N=' + str(N) + ' ' + strat + ': ' + str(round(elapsed, 2)) + 's  HND=' + str(res['HND']))

total_est = sum(profile.values()) * N_SEEDS
log('')
log('Estimated total sweep time: ' + str(round(total_est / 60, 1)) + ' minutes')
with open(OUTDIR + '/profile.json', 'w') as f:
    json.dump(profile, f, indent=2)

if total_est > 5 * 3600:
    log('WARNING: estimate exceeds 5 hours. Continuing in 30s (Ctrl+C to abort).')
    time.sleep(30)

log('')
log('=== FULL SWEEP ===')
results = {}
for N in N_VALUES:
    results[N] = {}
    for strat in STRATEGIES:
        cache_path = OUTDIR + '/N' + str(N) + '_' + strat + '.json'
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                hnds = json.load(f)
            log('  N=' + str(N) + ' ' + strat + ': [cached] mean=' + str(round(np.mean(hnds), 1)) + '  n=' + str(len(hnds)))
            results[N][strat] = hnds
            continue

        t0 = time.time()
        hnds = []
        for seed in range(N_SEEDS):
            res = run_one(strat, N, seed)
            hnds.append(res['HND'])
        elapsed = time.time() - t0
        with open(cache_path, 'w') as f:
            json.dump(hnds, f)
        results[N][strat] = hnds
        log('  N=' + str(N) + ' ' + strat + ': mean=' + str(round(np.mean(hnds), 1)) +
            '  std=' + str(round(np.std(hnds), 1)) + '  n=' + str(N_SEEDS) +
            '  time=' + str(round(elapsed, 0)) + 's')

log('')
log('=== SUMMARY TABLE ===')
log('N     v3_agg     LEACH     HEED      v3/LEACH  v3/HEED')
summary_rows = []
for N in N_VALUES:
    v3_mean = np.mean(results[N]['v3_agg'])
    leach_mean = np.mean(results[N]['LEACH'])
    heed_mean = np.mean(results[N]['HEED'])
    ratio_leach = v3_mean / leach_mean if leach_mean > 0 else float('inf')
    ratio_heed = v3_mean / heed_mean if heed_mean > 0 else float('inf')
    summary_rows.append((N, v3_mean, ratio_leach, ratio_heed))
    log('%-5d %-10.1f %-9.1f %-9.1f %.3fx    %.3fx' % (
        N, v3_mean, leach_mean, heed_mean, ratio_leach, ratio_heed))

log('')
log('=== PAIRWISE SIGNIFICANCE, per N ===')
for N in N_VALUES:
    v3 = np.array(results[N]['v3_agg'])
    for strat in ['LEACH', 'HEED']:
        other = np.array(results[N][strat])
        u, p = stats.mannwhitneyu(v3, other, alternative='two-sided')
        pooled_std = np.sqrt((v3.var(ddof=1) + other.var(ddof=1)) / 2)
        d = (v3.mean() - other.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        sig = 'SIG' if p < 0.05 else 'n.s.'
        result = 'v3 WINS' if v3.mean() > other.mean() else 'v3 loses'
        log('  N=' + str(N) + ' v3_agg vs ' + strat + ': ' + result +
            '  p=' + str(round(p, 5)) + '  d=' + str(round(d, 3)) + '  ' + sig)

log('')
log('=== TREND TEST: does the v3/LEACH and v3/HEED advantage GROW with N? ===')
Ns = [r[0] for r in summary_rows]
ratios_leach = [r[2] for r in summary_rows]
ratios_heed = [r[3] for r in summary_rows]

rho_l, p_l = stats.spearmanr(Ns, ratios_leach)
log('vs LEACH -- Spearman rho: ' + str(round(rho_l, 3)) + '  p=' + str(round(p_l, 5)))
if p_l < 0.05 and rho_l > 0:
    log('  SIGNIFICANT increasing trend vs LEACH.')
elif p_l < 0.05 and rho_l < 0:
    log('  SIGNIFICANT decreasing trend vs LEACH -- advantage shrinks with N.')
else:
    log('  No significant monotonic trend vs LEACH at this sample size.')

rho_h, p_h = stats.spearmanr(Ns, ratios_heed)
log('vs HEED  -- Spearman rho: ' + str(round(rho_h, 3)) + '  p=' + str(round(p_h, 5)))
if p_h < 0.05 and rho_h > 0:
    log('  SIGNIFICANT increasing trend vs HEED.')
elif p_h < 0.05 and rho_h < 0:
    log('  SIGNIFICANT decreasing trend vs HEED -- advantage shrinks with N.')
else:
    log('  No significant monotonic trend vs HEED at this sample size.')

with open(OUTDIR + '/final_summary.json', 'w') as f:
    json.dump(dict(summary_rows=summary_rows, spearman_leach=(rho_l, p_l),
                    spearman_heed=(rho_h, p_h)), f, indent=2)

log('')
log('=== DONE ===')
