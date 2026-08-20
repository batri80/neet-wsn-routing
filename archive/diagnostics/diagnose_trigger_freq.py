from simulate import run_episode
import numpy as np

for psi_c in [0.065, 0.15, 0.30]:
    psis_at_1200, reconfigs, partitions = [], [], []
    for seed in range(5):
        res = run_episode(N=60, E0=1.0, strategy='eqopt', Psi_c=psi_c,
                           max_rounds=1500, seed=seed, record_series=True)
        psi = res['series']['Psi']
        psis_at_1200.append(psi[min(1200, len(psi)-1)])
        reconfigs.append(res['n_reconfigs'])
        partitions.append(res['partition_events'])
    print(f"Psi_c={psi_c}: mean Psi@1200={np.mean(psis_at_1200):.5f}  "
          f"mean reconfigs={np.mean(reconfigs):.1f}  mean partitions={np.mean(partitions):.1f}")
