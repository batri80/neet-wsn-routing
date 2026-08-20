"""
Extended N-sweep, N in {30, 60, 100, 200, 500}: v3_flow vs static,
mincost, eqopt. Directly addresses Reviewer 2's original criticism
(only 3 E0 levels tested, unsupported trend claims) by using 5 N
points with a formal Spearman trend test on the advantage ratio,
rather than an eyeballed 2-point comparison.

Resumable: each (N, strategy) result is saved to disk immediately.
Safe to interrupt (Ctrl+C) and rerun -- completed cells are skipped.
"""
import numpy as np
import json
import time
import os
from scipy import stats
from flow_simulate import run_episode_v3
from simulate import run_episode

OUTDIR = 'n_sweep_v3_results'
os.makedirs(OUTDIR, exist_ok=True)

E0, R_c = 1.0, 35.0
MAX_ROUNDS = 3000
N_VALUES = [30, 60, 100, 200, 500]
N_SEEDS = 15
STRATEGIES = ['v3_flow', 'static', 'mincost', 'eqopt']


def log(msg):
    print('[' + time.strftime('%H:%M:%S') + '] ' + msg, flush=True)


def run_one(strat, N, seed):
    if strat == 'v3_flow':
        return run_episode_v3(N=N, E0=E0, R_c=R_c, max_rounds=MAX_ROUNDS,
                               seed=seed, record_series=False)
    else:
        return run_episode(N=N, E0=E0, R_c=R_c, strategy=strat,
                            max_rounds=MAX_ROUNDS, seed=seed, record_series=False)


# --- Step 0: profile one episode per (N, strategy) before committing ---
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

if total_est > 4 * 3600:
    log('WARNING: estimate exceeds 4 hours. Continuing in 30s (Ctrl+C to abort).')
    time.sleep(30)

# --- Step 1: full sweep, resumable ---
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

# --- Step 2: summary table + pairwise significance at each N ---
log('')
log('=== SUMMARY TABLE ===')
log('N     v3_flow    static    mincost   eqopt     v3/best_baseline')
summary_rows = []
for N in N_VALUES:
    v3_mean = np.mean(results[N]['v3_flow'])
    baseline_means = [np.mean(results[N][s]) for s in ['static', 'mincost', 'eqopt']]
    best_baseline = max(baseline_means)
    ratio = v3_mean / best_baseline if best_baseline > 0 else float('inf')
    summary_rows.append((N, v3_mean, ratio))
    log('%-5d %-10.1f %-9.1f %-9.1f %-9.1f %.2fx' % (
        N, v3_mean, np.mean(results[N]['static']),
        np.mean(results[N]['mincost']), np.mean(results[N]['eqopt']), ratio))

log('')
log('=== PAIRWISE SIGNIFICANCE, per N ===')
for N in N_VALUES:
    v3 = np.array(results[N]['v3_flow'])
    for strat in ['static', 'mincost', 'eqopt']:
        other = np.array(results[N][strat])
        u, p = stats.mannwhitneyu(v3, other, alternative='two-sided')
        pooled_std = np.sqrt((v3.var(ddof=1) + other.var(ddof=1)) / 2)
        d = (v3.mean() - other.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        sig = 'SIG' if p < 0.05 else 'n.s.'
        log('  N=' + str(N) + ' v3 vs ' + strat + ': p=' + str(round(p, 5)) +
            '  d=' + str(round(d, 2)) + '  ' + sig)

# --- Step 3: formal trend test (addresses Reviewer 2's original criticism) ---
log('')
log('=== TREND TEST: does the advantage GROW with N? ===')
Ns = [r[0] for r in summary_rows]
ratios = [r[2] for r in summary_rows]
rho, p_trend = stats.spearmanr(Ns, ratios)
log('Spearman rho (N vs v3/best_baseline ratio): ' + str(round(rho, 3)) +
    '  p=' + str(round(p_trend, 5)))
if p_trend < 0.05 and rho > 0:
    log('SIGNIFICANT increasing trend: the v3 advantage grows with N.')
elif p_trend < 0.05 and rho < 0:
    log('SIGNIFICANT decreasing trend: the v3 advantage shrinks with N -- would match the v2 failure pattern.')
else:
    log('No significant monotonic trend detected at this sample size.')

with open(OUTDIR + '/final_summary.json', 'w') as f:
    json.dump(dict(summary_rows=summary_rows, spearman_rho=rho, spearman_p=p_trend), f, indent=2)

log('')
log('=== DONE ===')
