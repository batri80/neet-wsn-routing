"""
final_strategy_sweep.py -- completes the full five-point N-sweep
(N=30,60,100,200,500) for the paper's final comparison strategy:
mincost-v3 vs static-CT (primary, correct peer) and vs LEACH/HEED
(secondary, cross-paradigm). Reuses already-computed results from
mincost_validation_results/ (mincost-v3 at N=100,500) and
n_sweep_v3_vs_protocols_results/ (LEACH/HEED at all 5 N) where
available -- only computes genuinely missing cells.

Resumable: every (N, strategy) result cached to final_strategy_results/
immediately. Safe to interrupt and rerun.
"""
import numpy as np
import json
import time
import os
from scipy import stats
from flow_simulate_mincost import run_episode_v3_mincost
from flow_simulate_static_ct import run_episode_static_ct
from protocols import PROTOCOL_FUNCS

OUTDIR = 'final_strategy_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, R_c = 1.0, 35.0
MAX_ROUNDS = 3000
N_VALUES = [30, 60, 100, 200, 500]
N_SEEDS = 15
STRATEGIES = ['mincost_v3', 'static_CT', 'LEACH', 'HEED']

# Pre-existing result locations to check before computing fresh.
PRIOR_SOURCES = {
    'mincost_v3': 'mincost_validation_results/N{N}_v3_mincost.json',
    'LEACH': 'n_sweep_v3_vs_protocols_results/N{N}_LEACH.json',
    'HEED': 'n_sweep_v3_vs_protocols_results/N{N}_HEED.json',
}


def log(msg):
    print('[' + time.strftime('%H:%M:%S') + '] ' + msg, flush=True)


def run_one(strat, N, seed):
    if strat == 'mincost_v3':
        return run_episode_v3_mincost(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                                       max_rounds=MAX_ROUNDS, seed=seed,
                                       record_series=False)['HND']
    elif strat == 'static_CT':
        return run_episode_static_ct(N=N, E0=E0, R_c=R_c, max_neighbors=15,
                                      max_rounds=MAX_ROUNDS, seed=seed,
                                      record_series=False)['HND']
    else:
        return PROTOCOL_FUNCS[strat](N=N, E0=E0, max_rounds=MAX_ROUNDS,
                                      seed=seed, beta=1.0)['HND']


def get_results(N, strat):
    cache_path = OUTDIR + '/N' + str(N) + '_' + strat + '.json'
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            hnds = json.load(f)
        log('  N=' + str(N) + ' ' + strat + ': [cached in final_strategy_results] mean=' +
            str(round(np.mean(hnds), 1)) + '  n=' + str(len(hnds)))
        return hnds

    if strat in PRIOR_SOURCES:
        prior_path = PRIOR_SOURCES[strat].format(N=N)
        if os.path.exists(prior_path):
            with open(prior_path) as f:
                hnds = json.load(f)
            with open(cache_path, 'w') as f:
                json.dump(hnds, f)
            log('  N=' + str(N) + ' ' + strat + ': [reused from ' + prior_path + '] mean=' +
                str(round(np.mean(hnds), 1)) + '  n=' + str(len(hnds)))
            return hnds

    t0 = time.time()
    hnds = [run_one(strat, N, s) for s in range(N_SEEDS)]
    elapsed = time.time() - t0
    with open(cache_path, 'w') as f:
        json.dump(hnds, f)
    log('  N=' + str(N) + ' ' + strat + ': [computed fresh] mean=' + str(round(np.mean(hnds), 1)) +
        '  std=' + str(round(np.std(hnds), 1)) + '  n=' + str(N_SEEDS) + '  time=' + str(round(elapsed, 0)) + 's')
    return hnds


log('=== FILLING GAPS (reusing prior results where possible) ===')
results = {}
for N in N_VALUES:
    results[N] = {}
    for strat in STRATEGIES:
        results[N][strat] = get_results(N, strat)

log('')
log('=== SUMMARY TABLE ===')
log('N     mincost-v3  static-CT  LEACH     HEED      v3/CT     v3/LEACH  v3/HEED')
summary_rows = []
for N in N_VALUES:
    mc = np.mean(results[N]['mincost_v3'])
    ct = np.mean(results[N]['static_CT'])
    leach = np.mean(results[N]['LEACH'])
    heed = np.mean(results[N]['HEED'])
    r_ct = mc / ct if ct > 0 else float('inf')
    r_leach = mc / leach if leach > 0 else float('inf')
    r_heed = mc / heed if heed > 0 else float('inf')
    summary_rows.append((N, mc, r_ct, r_leach, r_heed))
    log('%-5d %-11.1f %-10.1f %-9.1f %-9.1f %.3fx    %.3fx    %.3fx' % (
        N, mc, ct, leach, heed, r_ct, r_leach, r_heed))

log('')
log('=== PAIRWISE SIGNIFICANCE, per N ===')
for N in N_VALUES:
    mc = np.array(results[N]['mincost_v3'])
    for strat in ['static_CT', 'LEACH', 'HEED']:
        other = np.array(results[N][strat])
        u, p = stats.mannwhitneyu(mc, other, alternative='two-sided')
        pooled_std = np.sqrt((mc.var(ddof=1) + other.var(ddof=1)) / 2)
        d = (mc.mean() - other.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        sig = 'SIG' if p < 0.05 else 'n.s.'
        result = 'v3 WINS' if mc.mean() > other.mean() else 'v3 loses'
        log('  N=' + str(N) + ' mincost_v3 vs ' + strat + ': ' + result +
            '  p=' + str(round(p, 5)) + '  d=' + str(round(d, 3)) + '  ' + sig)

log('')
log('=== TREND TESTS ===')
Ns = [r[0] for r in summary_rows]
for label, idx in [('vs static_CT', 2), ('vs LEACH', 3), ('vs HEED', 4)]:
    ratios = [r[idx] for r in summary_rows]
    rho, p = stats.spearmanr(Ns, ratios)
    direction = 'INCREASING' if rho > 0 else 'DECREASING'
    sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
    trailing = direction if p < 0.05 else ''
    log('  ' + label + ': Spearman rho=' + str(round(rho,3)) + '  p=' + str(round(p,5)) + '  ' + sig + ' ' + trailing)

with open(OUTDIR + '/final_summary.json', 'w') as f:
    json.dump(dict(summary_rows=summary_rows), f, indent=2)

log('')
log('=== DONE ===')
