from simulate import run_episode
import numpy as np

for N in [30, 100, 500]:
    rows = [run_episode(N=N, E0=1.0, strategy='eqopt', max_rounds=1500, seed=s, record_series=False)
            for s in range(5)]
    n_global = np.mean([r['n_global'] for r in rows])
    psi_max = np.mean([r['Psi_max'] for r in rows])
    print(f"N={N:4d}: mean n_global reconfigs={n_global:.1f}  "
          f"top-K/N covered per trigger={5/N:.3f}  mean Psi_max={psi_max:.4f}")
