import numpy as np
from simulate import run_episode

print("=== partition events per strategy ===")
for strat in ['random', 'eqopt']:
    parts, reconfigs = [], []
    for seed in range(8):
        res = run_episode(N=60, E0=1.0, strategy=strat, max_rounds=1500,
                           seed=seed, record_series=False)
        parts.append(res['partition_events'])
        reconfigs.append(res['n_reconfigs'])
    print(f"{strat:8s}: mean partition_events={np.mean(parts):.1f}  mean n_reconfigs={np.mean(reconfigs):.1f}")

print("\n=== Psi(t) trajectory shape, single episode each ===")
for strat in ['random', 'eqopt']:
    res = run_episode(N=60, E0=1.0, strategy=strat, max_rounds=1500,
                       seed=2, record_series=True)
    psi = res['series']['Psi']
    checkpoints = [100, 300, 600, 900, 1200]
    vals = [round(psi[min(c, len(psi)-1)], 5) for c in checkpoints]
    print(f"{strat:8s}: Psi at rounds {checkpoints} = {vals}")
