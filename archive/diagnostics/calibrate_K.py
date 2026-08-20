from simulate import run_episode
import numpy as np
import time

WINDOW = 362  # matches the N=100 matched window already established
STATIC_BASELINE = 0.06183

for K in [5, 10, 20, 40]:
    t0 = time.time()
    vals = []
    for seed in range(10):
        res = run_episode(N=100, E0=1.0, strategy='eqopt_k', K=K,
                           max_rounds=WINDOW, seed=seed, record_series=True)
        psi = res['series']['Psi']
        if len(psi) >= WINDOW:
            vals.append(psi[:WINDOW].max())
    elapsed = time.time() - t0
    mean_psi = np.mean(vals) if vals else float('nan')
    print(f"K={K:3d}: mean Psi_max={mean_psi:.5f}  n={len(vals)}/10  "
          f"time={elapsed:.1f}s  {'BEATS static' if mean_psi < STATIC_BASELINE else 'still worse'}")
