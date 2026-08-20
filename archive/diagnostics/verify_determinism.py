"""
Most basic possible check: does calling the REAL run_episode() twice with
the identical seed produce identical output? If not, we have a much more
fundamental problem than a hand-rolled script being wrong.
"""
from simulate import run_episode
import numpy as np

res1 = run_episode(N=100, E0=1.0, strategy='static', max_rounds=362, seed=0, record_series=True)
res2 = run_episode(N=100, E0=1.0, strategy='static', max_rounds=362, seed=0, record_series=True)

psi1, psi2 = res1['series']['Psi'], res2['series']['Psi']
print(f"run 1 Psi_max: {psi1.max():.6f}   run 2 Psi_max: {psi2.max():.6f}")
print(f"identical? {np.array_equal(psi1, psi2)}")
if not np.array_equal(psi1, psi2):
    diverge_at = np.argmax(psi1 != psi2) if len(psi1) == len(psi2) else min(len(psi1), len(psi2))
    print(f"first round of divergence (if same length): {diverge_at}")
    print(f"lengths: {len(psi1)} vs {len(psi2)}")
