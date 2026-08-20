"""
Fine-grained tolerance sweep in the unexplored 0.0-0.05 band, where the
coarse sweep suggests the entire interesting transition may be
compressed.
"""
from simulate import run_episode
import numpy as np
import time

N, E0, R_c = 100, 1.0, 35.0
MAX_ROUNDS = 3000
N_SEEDS = 20

for tol in [0.0, 0.01, 0.02, 0.03, 0.04]:
    t0 = time.time()
    hnds = []
    for seed in range(N_SEEDS):
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy='eqopt_h',
                           cost_tolerance=tol, max_rounds=MAX_ROUNDS,
                           seed=seed, record_series=False)
        hnds.append(res['HND'])
    print(f"tolerance={tol:.2f}: mean={np.mean(hnds):.1f}  std={np.std(hnds):.1f}  "
          f"n={N_SEEDS}  time={time.time()-t0:.0f}s", flush=True)
print("\nreference: static=239.6  mincost=239.7  eqopt=226.4")
