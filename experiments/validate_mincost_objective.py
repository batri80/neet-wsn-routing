"""
validate_mincost_objective.py -- proper n=15 validation of the pure
total-cost-minimizing v3 objective (root-cause fix) against LEACH and
HEED, at N=100 and N=500. Follows a promising n=5 scouting result at
N=500: mean=778.6 vs LEACH=845.8, HEED=778.4 (essentially tied) --
a 49.7% improvement over the min-max fairness objective's 520.2.

Resumable: each (N, strategy) result saved to disk immediately.
Safe to interrupt and rerun -- completed cells are skipped.
"""
import numpy as np
import json
import time
import os
from scipy import stats
from flow_simulate_mincost import run_episode_v3_mincost
from protocols import PROTOCOL_FUNCS

OUTDIR = 'mincost_validation_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, R_c = 1.0, 35.0
MAX_ROUNDS = 3000
N_VALUES = [100, 500]
N_SEEDS = 15
STRATEGIES = ['v3_mincost', 'LEACH', 'HEED']


def log(msg):
    print('[' + time.strftime('%H:%M:%S') + '] ' + msg, flush=True)


def run_one(strat, N, seed):
    if strat == 'v3_mincost':
        return run_episode_v3_mincost(N=N, E0=E0, R_c=R_c, max_neighbors=15,
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
log('Estimated total time: ' + str(round(total_est / 60, 1)) + ' minutes')
with open(OUTDIR + '/profile.json', 'w') as f:
    json.dump(profile, f, indent=2)

log('')
log('=== FULL VALIDATION ===')
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
log('=== SIGNIFICANCE TESTS ===')
for N in N_VALUES:
    v3 = np.array(results[N]['v3_mincost'])
    for strat in ['LEACH', 'HEED']:
        other = np.array(results[N][strat])
        u, p = stats.mannwhitneyu(v3, other, alternative='two-sided')
        pooled_std = np.sqrt((v3.var(ddof=1) + other.var(ddof=1)) / 2)
        d = (v3.mean() - other.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        sig = 'SIG' if p < 0.05 else 'n.s.'
        result = 'v3 WINS' if v3.mean() > other.mean() else 'v3 loses'
        log('  N=' + str(N) + ' v3_mincost vs ' + strat + ': ' + result +
            '  p=' + str(round(p, 5)) + '  d=' + str(round(d, 3)) + '  ' + sig)

log('')
log('=== DONE ===')
