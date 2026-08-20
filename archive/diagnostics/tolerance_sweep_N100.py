"""
Lexicographic EqOpt-H tolerance sweep at N=100. cost_tolerance=0 should
behave like mincost; large tolerance should approach one-step eqopt.
Testing values in between to find a genuine hybrid sweet spot.
"""
from simulate import run_episode
from scipy import stats
import numpy as np
import time

N, E0, R_c = 100, 1.0, 35.0
MAX_ROUNDS = 3000
N_SEEDS = 20

results = {}
for label, kwargs in [('static', dict(strategy='static')),
                       ('mincost', dict(strategy='mincost')),
                       ('eqopt', dict(strategy='eqopt')),
                       ('eqopt_h_0.05', dict(strategy='eqopt_h', cost_tolerance=0.05)),
                       ('eqopt_h_0.15', dict(strategy='eqopt_h', cost_tolerance=0.15)),
                       ('eqopt_h_0.30', dict(strategy='eqopt_h', cost_tolerance=0.30)),
                       ('eqopt_h_0.50', dict(strategy='eqopt_h', cost_tolerance=0.50))]:
    t0 = time.time()
    hnds = []
    for seed in range(N_SEEDS):
        res = run_episode(N=N, E0=E0, R_c=R_c, max_rounds=MAX_ROUNDS,
                           seed=seed, record_series=False, **kwargs)
        hnds.append(res['HND'])
    results[label] = np.array(hnds)
    print(f"{label:14s}: mean={np.mean(hnds):.1f}  std={np.std(hnds):.1f}  "
          f"n={N_SEEDS}  time={time.time()-t0:.0f}s", flush=True)

print("\n=== best eqopt_h variant vs baselines ===")
h_labels = ['eqopt_h_0.05', 'eqopt_h_0.15', 'eqopt_h_0.30', 'eqopt_h_0.50']
best_label = max(h_labels, key=lambda l: results[l].mean())
print(f"best tolerance: {best_label}")
eh = results[best_label]
for strat in ['static', 'mincost', 'eqopt']:
    other = results[strat]
    u, p = stats.mannwhitneyu(eh, other, alternative='two-sided')
    pooled_std = np.sqrt((eh.var(ddof=1) + other.var(ddof=1)) / 2)
    d = (eh.mean() - other.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
    result = 'WINS' if eh.mean() > other.mean() else 'loses'
    sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
    print(f"{best_label} vs {strat:10s}: {result}  p={p:.4f}  d={d:.3f}  {sig}")
