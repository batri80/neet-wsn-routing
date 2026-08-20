"""
Corrected K-sweep: R_c=35.0 explicit throughout, matching the established
baseline conditions (static=0.06183, one-step eqopt=0.06500 at this R_c).
This supersedes calibrate_K.py, which silently used the R_c=30.0 default.
"""
from simulate import run_episode
import numpy as np
import time

WINDOW = 362
N, E0, R_c = 100, 1.0, 35.0
STATIC_BASELINE = 0.06183
EQOPT_BASELINE = 0.06500

for K in [5, 10, 20]:
    t0 = time.time()
    vals = []
    for seed in range(10):
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy='eqopt_k', K=K,
                           max_rounds=WINDOW, seed=seed, record_series=True)
        psi = res['series']['Psi']
        if len(psi) >= WINDOW:
            vals.append(psi[:WINDOW].max())
    elapsed = time.time() - t0
    mean_psi = np.mean(vals) if vals else float('nan')
    std_psi = np.std(vals) if vals else float('nan')
    vs_static = 'BEATS static' if mean_psi < STATIC_BASELINE else 'worse than static'
    vs_eqopt = 'BEATS one-step' if mean_psi < EQOPT_BASELINE else 'worse than one-step'
    print(f"K={K:3d}: mean Psi_max={mean_psi:.5f}  std={std_psi:.5f}  n={len(vals)}/10  "
          f"time={elapsed:.1f}s  [{vs_static}]  [{vs_eqopt}]")
