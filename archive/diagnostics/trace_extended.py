"""
No new hypothesis -- just extend the trajectory comparison further and
watch the gap open, using the same trustworthy method as before (direct
simulate.py calls, not a hand-rolled reimplementation).
"""
from simulate import run_episode
import numpy as np

res_s = run_episode(N=100, E0=1.0, strategy='static', max_rounds=362, seed=2, record_series=True)
res_e = run_episode(N=100, E0=1.0, strategy='eqopt', max_rounds=362, seed=2, record_series=True)
psi_s, psi_e = res_s['series']['Psi'], res_e['series']['Psi']

print(f"{'round':>6} {'static_Psi':>12} {'eqopt_Psi':>12} {'gap':>10}")
for r in range(0, 362, 20):
    if r < len(psi_s) and r < len(psi_e):
        gap = psi_e[r] - psi_s[r]
        print(f"{r:6d} {psi_s[r]:12.6f} {psi_e[r]:12.6f} {gap:10.6f}")
