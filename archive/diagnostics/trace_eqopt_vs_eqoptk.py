from simulate import run_episode
import numpy as np

res_e = run_episode(N=100, E0=1.0, strategy='eqopt', max_rounds=362, seed=2, record_series=True)
res_ek = run_episode(N=100, E0=1.0, strategy='eqopt_k', K=10, max_rounds=362, seed=2, record_series=True)
psi_e, psi_ek = res_e['series']['Psi'], res_ek['series']['Psi']

print(f"{'round':>6} {'eqopt_Psi':>12} {'eqopt_k_Psi':>12} {'gap':>10}")
for r in range(0, 362, 20):
    if r < len(psi_e) and r < len(psi_ek):
        print(f"{r:6d} {psi_e[r]:12.6f} {psi_ek[r]:12.6f} {psi_ek[r]-psi_e[r]:10.6f}")
