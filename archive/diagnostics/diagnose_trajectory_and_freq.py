"""
Two checks:
1. Single-episode Psi(t) trajectory shape, eqopt vs static, with
   reconfiguration event markers -- looking for jagged/wandering drift
   vs a single smooth arc.
2. Psi_c sensitivity sweep on matched-window Psi_max -- does a HIGHER
   threshold (fewer, less frequent reconfigurations) reduce or eliminate
   the eqopt-worse-than-static effect?
"""
import numpy as np
from simulate import run_episode

print("=== 1. Trajectory shape, N=100, window=362 ===")
for strat in ['static', 'eqopt']:
    res = run_episode(N=100, E0=1.0, strategy=strat, max_rounds=362, seed=0, record_series=True)
    psi = res['series']['Psi']
    checkpoints = [50, 100, 150, 200, 250, 300, 361]
    vals = [round(psi[min(c, len(psi)-1)], 5) for c in checkpoints]
    print(f"{strat:8s}: Psi at rounds {checkpoints} = {vals}  (n_reconfigs={res['n_reconfigs']})")

print("\n=== 2. Psi_c sensitivity on matched-window Psi_max, N=100 ===")
for psi_c in [0.065, 0.15, 0.30, 0.50, 999]:
    vals_eqopt, vals_static = [], []
    for seed in range(10):
        r_e = run_episode(N=100, E0=1.0, strategy='eqopt', Psi_c=psi_c,
                           max_rounds=362, seed=seed, record_series=True)
        r_s = run_episode(N=100, E0=1.0, strategy='static',
                           max_rounds=362, seed=seed, record_series=True)
        pe, ps = r_e['series']['Psi'], r_s['series']['Psi']
        if len(pe) >= 362: vals_eqopt.append(pe[:362].max())
        if len(ps) >= 362: vals_static.append(ps[:362].max())
    print(f"Psi_c={psi_c:6.3f}: eqopt mean Psi_max={np.mean(vals_eqopt):.5f}  "
          f"static mean Psi_max={np.mean(vals_static):.5f}  "
          f"eqopt {'WINS' if np.mean(vals_eqopt) < np.mean(vals_static) else 'loses'}")
